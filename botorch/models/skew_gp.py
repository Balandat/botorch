#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Skew Gaussian processes models and accompanying methods.
For details, see [benavoli2020unified]_ and [benavoli2020skew]_.

.. [benavoli2020unified]
    A. Benavoli and D. Azzimonti and D. Piga. A unified framework for closed-form
    nonparametric regression, classification, preference and mixed problems with
    Skew Gaussian Processes. arXiv Preprint, 2020.

.. [benavoli2020skew]
    A. Benavoli and  D. Azzimonti and D. Piga. Skew Gaussian processes for
    classification. Machine Learning, 2020.
"""


from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.models.likelihoods.affine_probit import AffineProbitLikelihood
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.utils import (
    check_min_max_scaling,
    check_no_nans,
    gpt_posterior_settings,
)
from botorch.posteriors.torch import TorchPosterior
from botorch.settings import _Flag, validate_input_scaling
from botorch.utils.probability import TruncatedMultivariateNormal, UnifiedSkewNormal
from gpytorch import settings as gpt_settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from gpytorch.models import GP
from gpytorch.module import Module
from gpytorch.priors.torch_priors import GammaPrior
from linear_operator.operators import IdentityLinearOperator, LinearOperator, to_dense
from torch import BoolTensor, Tensor
from torch.distributions import Bernoulli, Distribution
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad


NoneType = type(None)


class ExactMarginalLogLikelihood_v2(ExactMarginalLogLikelihood):
    r"""
    Same as `ExactMarginalLogLikelihood` but calls `likelihood.log_marginal`
    directly while passing in target values from `forward`.

    TODO: Incorporate these changes directly into `ExactMarginalLogLikelihood`.
    """

    def forward(
        self, function_dist: Distribution, target: Tensor, *params: Tensor
    ) -> MarginalLogLikelihood:
        log_prob = self.likelihood.log_marginal(target, function_dist)
        log_prob = self._add_other_terms(log_prob, params)
        num_data = function_dist.event_shape.numel()
        return log_prob.div_(num_data)


class SkewGPPredictionStrategy:
    def __init__(
        self,
        train_prior: MultivariateNormal,
        likelihood: AffineProbitLikelihood,
    ):
        self.train_prior = train_prior
        self.likelihood = likelihood

    def __call__(
        self,
        test_prior: MultivariateNormal,
        train_test_covar: Union[LinearOperator, Tensor],
        predict_noise: Union[bool, Tensor] = False,
    ) -> UnifiedSkewNormal:
        r"""Returns the conditional distribution of test variables.

        Args:
            test_prior: The prior distribution of test variables.
            train_test_covar: The prior cross-covariance of training and test variables.
            predict_noise: If True, use the likelihood's noise module to noise
                the posterior distribution of latent random variables. If a Tensor,
                add it directly to the latent random variables covariance matrix.

        Returns:
            The conditional distribution of test variables.
        """
        if predict_noise is True and self.likelihood.noise_covar is not None:
            predict_noise = self.likelihood.noise_covar(shape=test_prior.mean.shape)

        if isinstance(predict_noise, (Tensor, LinearOperator)):
            test_prior = MultivariateNormal(
                mean=test_prior.mean,
                covariance_matrix=test_prior.covariance_matrix + predict_noise,
            )

        A, _ = self.affine_transform

        # ensuring that trunc and gauss have the same batch_shape
        trunc = self.train_prior_predictive
        gauss = test_prior
        batch_t = trunc.batch_shape
        batch_g = gauss.batch_shape
        if batch_t != batch_g:
            batch_shape = torch.broadcast_shapes(batch_t, batch_g)
            trunc = trunc.expand(batch_shape)
            gauss = gauss.expand(batch_shape)

        return UnifiedSkewNormal(
            trunc=trunc,
            gauss=gauss,
            cross_covariance_matrix=(A @ train_test_covar),
        )

    @lazy_property
    def affine_transform(self) -> Tuple[Tensor, Optional[Tensor]]:
        return self.likelihood.get_affine_transform()

    @property
    def num_train(self):
        return self.train_shape.numel()

    @lazy_property
    def train_shape(self):
        return self.train_prior.event_shape

    @lazy_property
    def train_prior_predictive(self) -> TruncatedMultivariateNormal:
        return self.likelihood.marginal(self.train_prior)


class SkewGP(Model, GP):
    def __init__(
        self,
        likelihood: AffineProbitLikelihood,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        mean_module: Optional[Module] = None,
        covar_module: Optional[Module] = None,
        input_transform: Optional[InputTransform] = None,
        **kwargs: Any,
    ) -> None:
        r"""Infinite-dimensional analogue of a unified skew normal distribution.

        Args:
            likelihood: An AffineProbitLikelihood that defines a set of affine constraints.
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x 1` boolean tensor of training observations.
            mean_module: The module used to compute prior means.
            covar_module: The module used to compute prior covariances.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        super().__init__()
        self._validate_inputs(train_X=train_X, train_Y=train_Y)
        with torch.no_grad():
            X = train_X if input_transform is None else input_transform(train_X)

        if mean_module is None:
            mean_module = ConstantMean()

        if covar_module is None:
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=X.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            #  Initialize to modest values to avoid potential numerical issues
            covar_module.base_kernel.lengthscale = 0.25 * (X.shape[-1] ** 0.5)

        self._train_inputs = None
        self._train_targets = None
        self._prediction_strategy = None
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
        if input_transform is not None:
            self.input_transform = input_transform

        self.set_train_data(
            inputs=(train_X,),
            targets=None if train_Y is None else train_Y.squeeze(-1),
        )
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *inputs, **kwargs):
        # Training and prior modes
        if self.training or gpt_settings.prior_mode.on() or self.train_inputs is None:
            if self.training:
                if self.train_inputs is None:
                    raise RuntimeError("train_inputs cannot be None in training mode.")
                if gpt_settings.debug.on() and len(inputs):
                    if len(inputs) != len(self.train_inputs) or not all(
                        A.equal(B) for A, B in zip(self.train_inputs, inputs)
                    ):
                        raise RuntimeError("You must train on the training inputs!")

            return self.forward(*self.train_inputs)

        test_prior = self.forward(*inputs)
        train_test_covar = self.covar_module(*self.train_inputs, *inputs)
        with gpt_settings.cg_tolerance(gpt_settings.eval_cg_tolerance.value()):
            return self.prediction_strategy(
                test_prior=test_prior, train_test_covar=train_test_covar, **kwargs
            )

    def posterior(self, X: Tensor) -> TorchPosterior:
        r"""Computes the posterior distribution of process values at test locations `X`.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.

        Returns:
            A `TorchPosterior` object with a `UnifiedSkewNormal` distribution,
            representing a batch of `b` joint distributions over `q` points.
        """
        self.eval()
        with gpt_posterior_settings():
            usn = self(self.transform_inputs(X))
        return TorchPosterior(distribution=usn)

    def posterior_predictive(
        self, X: Tensor, predict_noise: Union[bool, Tensor] = True
    ) -> TorchPosterior:
        r"""Computes the posterior distribution of observables at test locations `X`.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            predict_noise: If True, use the likelihood's noise module to noise
                the posterior distribution of latent random variables. If a Tensor,
                add it directly to the latent random variables covariance matrix.

        Returns:
            A `TorchPosterior` object with a `UnifiedSkewNormal` distribution,
            representing a batch of `b` joint distributions over `q` points.
            Includes observation noise if specified.
        """
        self.eval()
        with gpt_posterior_settings():
            usn = self(self.transform_inputs(X), predict_noise=predict_noise)
        return TorchPosterior(distribution=usn)

    def set_train_data(
        self,
        inputs: Optional[Tuple[Tensor, ...]] = None,
        targets: Optional[Tensor] = None,
        strict: Optional[bool] = None,
    ) -> None:
        r"""Set training data (does not re-fit model hyper-parameters).

        Args:
            inputs: The new training inputs.
            targets: The new training targets.
            strict: If `True`, the new inputs and targets must have
                the same shape, dtype, and device as the current inputs and targets.
                Otherwise, any shape/dtype/device are allowed. Default to `False` if
                no training inputs/targets have not yet been defined; otherwise, `True`.
        """
        if inputs is not None:
            _strict = strict or (strict is None and self.train_inputs is not None)
            if _strict and len(inputs) != len(self.train_inputs):
                raise RuntimeError("Cannot modify number of input tensors.")

            if torch.is_tensor(inputs):
                inputs = (inputs.unsqueeze(-1) if inputs.ndimension() == 1 else inputs,)
            else:
                inputs = (X.unsqueeze(-1) if X.ndimension() == 1 else X for X in inputs)

            if _strict:
                for input_, t_input in zip(inputs, self.train_inputs):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(
                                attr=attr, e_attr=expected_attr, f_attr=found_attr
                            )
                            raise RuntimeError(msg)
                self.train_inputs = inputs
            else:
                self.train_inputs = tuple(inputs)

        if targets is not None:
            if strict or (strict is None and self.train_targets is not None):
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(
                            attr=attr, e_attr=expected_attr, f_attr=found_attr
                        )
                        raise RuntimeError(msg)
            self.train_targets = targets
        self._prediction_strategy = None

    @property
    def train_inputs(self) -> Tuple[Tensor, ...]:
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, train_inputs: Tuple[Tensor]) -> None:
        self._train_inputs = train_inputs

    @property
    def train_targets(self) -> None:
        return None

    @train_targets.setter
    def train_targets(self, train_targets: NoneType) -> None:
        if train_targets is not None:
            raise UnsupportedError("SkewGP does not have 'train_targets'.")

    @property
    def prediction_strategy(self) -> SkewGPPredictionStrategy:
        if self._prediction_strategy is None:
            self._prediction_strategy = SkewGPPredictionStrategy(
                train_prior=self.forward(*self.train_inputs),
                likelihood=self.likelihood,
            )
        return self._prediction_strategy

    @classmethod
    def _validate_inputs(
        cls,
        train_X: Tensor,
        train_Y: Optional[Tensor],
        raise_on_fail: bool = False,
        ignore_X_dims: Optional[List[int]] = None,
    ) -> None:
        if train_Y is not None:
            if train_Y.dtype != torch.bool:
                raise ValueError("`train_Y` must be a boolean tensor")
            if train_Y.shape[-1] != 1:
                raise ValueError("Only a single output is supported for `train_Y`.")

        if validate_input_scaling.off():
            return

        check_no_nans(train_X)
        check_min_max_scaling(
            X=train_X, raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
        )


class SkewGPClassifierPredictionStrategy(SkewGPPredictionStrategy):
    def __init__(
        self,
        train_prior: MultivariateNormal,
        likelihood: AffineProbitLikelihood,
        train_targets: BoolTensor,
    ):
        super().__init__(train_prior=train_prior, likelihood=likelihood)
        self.train_targets = train_targets
        self.predict_latents = type("PredictLatentsFlag", (_Flag,), {})

    def __call__(self, *args, **kwargs) -> Union[Bernoulli, UnifiedSkewNormal]:
        if self.predict_latents.on():
            return super().__call__(*args, **kwargs)

        return self.forward(*args, **kwargs)

    def forward(
        self,
        test_prior: MultivariateNormal,
        train_test_covar: Union[Tensor, LinearOperator],
        test_targets: Optional[BoolTensor] = None,
        predict_noise: Union[bool, Tensor] = True,
    ) -> Bernoulli:
        r"""Returns the conditional distribution of the Boolean random variable defined
        as whether or not the test targets are correct.

        Args:
            test_prior: The prior distribution of test variables.
            train_test_covar: The prior cross-covariance of training and test variables.
            test_targets: Target values for the test set, defaults to positive.
            predict_noise: If True, use the likelihood's noise module to noise
                the posterior distribution of latent random variables. If a Tensor,
                add it directly to the latent random variables covariance matrix.

        Returns:
           The conditional distribution of the Boolean random variable defined as
           whether or not the test targets are correct.
        """
        test_loc = test_prior.loc
        test_covar = test_prior.covariance_matrix
        if predict_noise is True and self.likelihood.noise_covar is not None:
            predict_noise = self.likelihood.noise_covar(shape=test_loc.shape)

        if isinstance(predict_noise, (Tensor, LinearOperator)):
            test_covar = test_covar + predict_noise

        if test_targets is not None:
            signs = 2 * test_targets.to(test_prior.loc.dtype) - 1
            test_loc = signs * test_prior.loc
            test_covar = signs.unsqueeze(-1) * test_covar * signs.unsqueeze(-2)
            train_test_covar = train_test_covar * signs.unsqueeze(-2)

        A, _ = self.affine_transform
        train_solver = self.train_prior_predictive.solver
        cross_covar = to_dense((A @ train_test_covar).transpose(-1, -2))
        train_perm = train_solver.perm.expand(
            *cross_covar.shape[:-1], train_solver.perm.shape[-1]
        )
        joint_solver = train_solver.expand(*test_covar.shape[:-2]).augment(
            covariance_matrix=to_dense(test_covar),
            cross_covariance_matrix=cross_covar.gather(-1, train_perm),
            bounds=pad(-test_loc.unsqueeze(-1), (0, 1), value=float("inf")),
        )
        log_probs = joint_solver.solve() - self.train_prior_predictive.log_partition
        return Bernoulli(probs=log_probs.exp())

    @lazy_property
    def affine_transform(self) -> Tuple[Tensor, Optional[Tensor]]:
        signed_train_targets = -1 + 2 * self.train_targets.to(
            dtype=self.train_prior.loc.dtype, device=self.train_prior.loc.device
        )
        return self.likelihood.get_affine_transform(diag=signed_train_targets)

    @lazy_property
    def train_prior_predictive(self) -> TruncatedMultivariateNormal:
        return self.likelihood.marginal(self.train_prior, self.train_targets)


class SkewGPClassifier(SkewGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: BoolTensor,
        likelihood: Optional[AffineProbitLikelihood] = None,
        mean_module: Optional[Module] = None,
        covar_module: Optional[Module] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: NoneType = None,
        **kwargs: Any,
    ) -> None:
        r"""Infinite-dimensional analogue of pushing a (truncated) multivariate normal
        random vector through a Heaviside step function, see [benavoli2020skew].

        Let `f ~ GP(m, k)` be a Gaussian process and define observations as binary
        random variables `y_i = 1_{f(x_i) + e(x_i) > 0}`, where `e(.)` is a centered
        Gaussian noise model. Then, `P(Y) = P(Af(X) + e(X) > 0)`, where `A` is a
        diagonal matrix such that `A_{ii} = 2y_i - 1`.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of binary training labels.
            likelihood: A likelihood. If omitted, use an AffineProbitLikelihood
                with homoskedastic Gaussian noise.
            covar_module: The module used to compute prior covariances.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        assert outcome_transform is None, UnsupportedError(
            "SkewGPClassifier does not use an outcome transform."
        )

        if likelihood is None:
            weight = IdentityLinearOperator(
                train_X.shape[:1], device=train_X.device, dtype=train_X.dtype
            )
            noise_covar = HomoskedasticNoise(noise_prior=GammaPrior(0.9, 10.0))
            likelihood = AffineProbitLikelihood(weight=weight, noise_covar=noise_covar)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            **kwargs,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self, X: Tensor) -> TorchPosterior:
        r"""Computes the posterior distribution of process values at test locations `X`.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.

        Returns:
            A `TorchPosterior` object with a `UnifiedSkewNormal` distribution,
            representing a batch of `b` joint distributions over `q` points.
        """
        self.eval()
        with self.prediction_strategy.predict_latents(True), gpt_posterior_settings():
            usn = self(self.transform_inputs(X))
        return TorchPosterior(distribution=usn)

    def posterior_predictive(
        self,
        X: Tensor,
        predict_noise: Union[bool, Tensor] = True,
        test_targets: Optional[BoolTensor] = None,
        **kwargs: Any,
    ) -> TorchPosterior:
        r"""Computes the posterior distribution of the Boolean random variable defined
        as whether or not the test targets are correct.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            predict_noise: If True, use the likelihood's noise module to add
                noise to the covariance of the latent random variables. If a Tensor,
                add it directly to the aforementioned covariancce
                (must be of shape `(batch_shape) x q`).
            test_targets: Target values for the test set, assumed positive by default.

        Returns:
            A `TorchPosterior` object with a `Bernoulli` distribution, representing a
            batch of `b` joint distributions over `q` points.
        """
        self.eval()
        with self.prediction_strategy.predict_latents(False), gpt_posterior_settings():
            bernoulli = self(
                self.transform_inputs(X),
                predict_noise=predict_noise,
                test_targets=test_targets,
            )
        return TorchPosterior(distribution=bernoulli)

    @property
    def train_targets(self) -> Tensor:
        return self._train_targets

    @train_targets.setter
    def train_targets(self, train_targets: Tensor) -> None:
        self._train_targets = train_targets

    @property
    def prediction_strategy(self) -> SkewGPClassifierPredictionStrategy:
        if self._prediction_strategy is None:
            self._prediction_strategy = SkewGPClassifierPredictionStrategy(
                train_prior=self.forward(*self.train_inputs),
                likelihood=self.likelihood,
                train_targets=self.train_targets,
            )
        return self._prediction_strategy

    @classmethod
    def _validate_inputs(
        cls,
        train_X: Tensor,
        train_Y: BoolTensor,
        train_Yvar: Optional[Tensor] = None,
        raise_on_fail: bool = False,
        ignore_X_dims: Optional[List[int]] = None,
    ) -> None:
        if validate_input_scaling.off():
            return

        check_no_nans(train_X)
        check_no_nans(train_Y)
        if train_Yvar is not None:
            check_no_nans(train_Yvar)
            if torch.any(train_Yvar < 0):
                raise InputDataError("Input data contains negative variances.")

        check_min_max_scaling(
            X=train_X, raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
        )
