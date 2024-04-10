#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import suppress

from itertools import product
from typing import Iterator, NamedTuple, Optional, Tuple, TypeVar, Union
from unittest.mock import patch
from warnings import catch_warnings, filterwarnings

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.affine_probit import AffineProbitLikelihood
from botorch.models.skew_gp import (
    ExactMarginalLogLikelihood_v2,
    SkewGP,
    SkewGPClassifier,
)
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.probability.unified_skew_normal import UnifiedSkewNormal
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings as gpytorch_settings
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior
from torch import Size, Tensor
from torch.distributions.bernoulli import Bernoulli

T = TypeVar("T")
Choices = type("Choices", (tuple,), {})
Option = Union[T, Tuple[T, ...]]


class TestConfigSkewGP(NamedTuple):
    seed: Option[int] = Choices([0])
    dtype: Option[torch.dtype] = Choices([torch.float32, torch.float64])
    device: Option[torch.device] = Choices([torch.device("cpu")])
    input_dim: Option[int] = Choices([2])
    num_train: Option[int] = Choices([3])
    num_constraints: Option[int] = Choices([2])
    test_shapes: Option[Tuple[Size, ...]] = Choices([(Size([4]), Size([2, 4]))])
    transform_inputs: Option[bool] = Choices([False, True])


class TestCacheSkewGP(NamedTuple):
    model: SkewGP
    train_X: Tensor
    test_Xs: Tuple[Tensor, ...]
    likelihood: AffineProbitLikelihood
    input_transform: Optional[InputTransform] = None


class TestSkewGP(BotorchTestCase):
    def setUp(self):
        self.choices = TestConfigSkewGP()

    def gen_config(self) -> Iterator[NamedTuple]:
        for config in map(
            lambda tpl: type(self.choices)(*tpl),
            product(*(x if isinstance(x, Choices) else (x,) for x in self.choices)),
        ):
            yield config

    @property
    def cases(self) -> Iterator[Tuple[TestConfigSkewGP, TestCacheSkewGP]]:
        for config in self.gen_config():
            yield config, self.get_case(config)

    def get_case(self, config: TestConfigSkewGP) -> TestCacheSkewGP:
        assert config.num_train > 1

        tkwargs = {"device": config.device, "dtype": config.dtype}
        with torch.random.fork_rng():
            torch.random.manual_seed(config.seed)
            train_X = torch.rand(config.num_train, config.input_dim, **tkwargs)
            test_Xs = tuple(
                torch.rand(test_shape + (config.input_dim,), **tkwargs)
                for test_shape in config.test_shapes
            )

            weight = torch.rand(config.num_constraints, config.num_train, **tkwargs)
            bias = torch.rand(config.num_constraints, **tkwargs)

        noise_covar = HomoskedasticNoise(noise_prior=GammaPrior(0.9, 10.0))
        likelihood = AffineProbitLikelihood(
            weight=weight, bias=bias, noise_covar=noise_covar
        )

        if config.transform_inputs:
            warping = Normalize(
                d=config.input_dim,
                transform_on_train=True,
                bounds=torch.tensor(
                    [config.input_dim * [-1.0], config.input_dim * [1.0]], **tkwargs
                ),
            )
        else:
            warping = None

        model = SkewGP(train_X=train_X, likelihood=likelihood, input_transform=warping)

        return TestCacheSkewGP(
            model=model.to(**tkwargs),
            likelihood=likelihood,
            train_X=train_X,
            test_Xs=test_Xs,
            input_transform=warping,
        )

    def test_init(self):
        for _, cache in self.cases:
            model = cache.model
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            base_kernel = model.covar_module.base_kernel
            self.assertIsInstance(base_kernel, MaternKernel)
            self.assertIsInstance(base_kernel.lengthscale_prior, GammaPrior)
            if cache.input_transform is not None:
                self.assertEqual(model.input_transform, cache.input_transform)

    def test_call(self):
        for _, cache in self.cases:
            model = cache.model
            model.train()
            with patch.object(model, "_train_inputs", None):
                with self.assertRaisesRegex(RuntimeError, "train_inputs cannot be Non"):
                    model(cache.train_X)

            with gpytorch_settings.debug(True):
                with self.assertRaisesRegex(RuntimeError, "must train on the training"):
                    model(cache.test_Xs[0])

    def test_set_train_data(self):
        config, cache = next(self.cases)

        X = cache.test_Xs[0]
        while X.ndim > 2:
            X = X[0]

        with self.assertRaisesRegex(RuntimeError, "Cannot modify number of"):
            cache.model.set_train_data(inputs=(), strict=True)

        with self.assertRaisesRegex(RuntimeError, "Cannot modify shape of"):
            cache.model.set_train_data(inputs=(X[:1],), strict=True)

        with self.assertRaisesRegex(UnsupportedError, "SkewGP does not have"):
            cache.model.set_train_data(targets=X)  # fails regardless of strict

    def test_fit(self):
        for _, cache in self.cases:
            mll = ExactMarginalLogLikelihood_v2(cache.model.likelihood, cache.model)
            with catch_warnings():
                filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

    def test_eval(self):
        for config, cache in self.cases:
            model = cache.model
            model.eval()
            self.assertEqual(model.prediction_strategy.num_train, config.num_train)
            self.assertEqual(
                tuple(model.prediction_strategy.train_shape), (config.num_train,)
            )
            if config.transform_inputs:
                transformed_X = cache.input_transform(cache.train_X)
                self.assertTrue(model.train_inputs[0].equal(transformed_X))

    def test_posterior(self):
        for _, cache in self.cases:
            for test_X in cache.test_Xs:
                posterior = cache.model.posterior(test_X)
                self.assertIsInstance(posterior, TorchPosterior)
                self.assertIsInstance(posterior.distribution, UnifiedSkewNormal)
                self.assertEqual(posterior._extended_shape(), test_X.shape[:-1])

                posterior_pred = cache.model.posterior_predictive(test_X)
                self.assertIsInstance(posterior_pred, TorchPosterior)
                self.assertIsInstance(posterior_pred.distribution, UnifiedSkewNormal)
                self.assertEqual(posterior_pred._extended_shape(), test_X.shape[:-1])


class TestConfigSkewGPClassifier(NamedTuple):
    seed: Option[int] = Choices([0])
    dtype: Option[torch.dtype] = Choices([torch.float32, torch.float64])
    device: Option[torch.device] = Choices([torch.device("cpu")])
    input_dim: Option[int] = Choices([2])
    num_train: Option[int] = Choices([3])
    test_shapes: Option[Tuple[Size, ...]] = Choices([(Size([4]), Size([2, 4]))])
    transform_inputs: Option[bool] = Choices([False, True])


class TestCacheSkewGPClassifier(NamedTuple):
    model: SkewGP
    train_X: Tensor
    train_Y: Tensor
    test_Xs: Tuple[Tensor, ...]
    input_transform: Optional[InputTransform] = None


class TestSkewGPClassifier(TestSkewGP):
    def setUp(self):
        self.choices = TestConfigSkewGPClassifier()

    def get_case(self, config: TestConfigSkewGPClassifier) -> TestCacheSkewGPClassifier:
        assert config.num_train > 1

        tkwargs = {"device": config.device, "dtype": config.dtype}
        with torch.random.fork_rng():
            torch.random.manual_seed(config.seed)
            train_X = torch.rand(config.num_train, config.input_dim, **tkwargs)
            train_Y = torch.rand(config.num_train) > 0.5
            test_Xs = tuple(
                torch.rand(test_shape + (config.input_dim,), **tkwargs)
                for test_shape in config.test_shapes
            )

        if config.transform_inputs:
            warping = Normalize(
                d=config.input_dim,
                transform_on_train=True,
                bounds=torch.tensor(
                    [config.input_dim * [-1.0], config.input_dim * [1.0]], **tkwargs
                ),
            )
        else:
            warping = None

        model = SkewGPClassifier(
            train_X=train_X, train_Y=train_Y, input_transform=warping
        )

        return TestCacheSkewGPClassifier(
            model=model.to(**tkwargs),
            train_X=train_X,
            train_Y=train_Y,
            test_Xs=test_Xs,
            input_transform=warping,
        )

    def test_set_train_data(self):
        config, cache = next(self.cases)

        X = cache.test_Xs[0]
        while X.ndim > 2:
            X = X[0]

        with self.assertRaisesRegex(RuntimeError, "Cannot modify number of"):
            cache.model.set_train_data(inputs=(), strict=True)

        with self.assertRaisesRegex(RuntimeError, "Cannot modify shape of"):
            cache.model.set_train_data(inputs=(X[:1],), strict=True)

        with self.assertRaisesRegex(RuntimeError, "Cannot modify"):
            cache.model.set_train_data(targets=X, strict=True)

    def test_posterior(self):
        for _, cache in self.cases:
            for test_X in cache.test_Xs:
                # TODO: Batch-mode gradients currently disabled due to pivoting
                with (torch.no_grad if test_X.ndim > 2 else suppress)():
                    posterior = cache.model.posterior(test_X)
                    self.assertIsInstance(posterior, TorchPosterior)
                    self.assertIsInstance(posterior.distribution, UnifiedSkewNormal)
                    self.assertEqual(posterior._extended_shape(), test_X.shape[:-1])

                    posterior_pred = cache.model.posterior_predictive(test_X)
                    self.assertIsInstance(posterior_pred, TorchPosterior)
                    self.assertIsInstance(posterior_pred.distribution, Bernoulli)
                    self.assertEqual(
                        posterior_pred._extended_shape(), test_X.shape[:-2]
                    )
                self.assertEqual(posterior_pred.event_shape, test_X.shape[:-1])
