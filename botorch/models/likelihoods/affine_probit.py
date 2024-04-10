#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from botorch.utils.probability import TruncatedMultivariateNormal, UnifiedSkewNormal
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import Noise
from linear_operator.operators import DiagLinearOperator, LinearOperator
from torch import BoolTensor, Tensor
from torch.nn.functional import pad


class AffineProbitLikelihood(_GaussianLikelihoodBase, Likelihood):
    def __init__(
        self,
        weight: Union[LinearOperator, Tensor],
        bias: Optional[Union[LinearOperator, Tensor]] = None,
        noise_covar: Optional[Noise] = None,
    ):
        """Affine probit likelihood `P(f + e > 0)`, where `f = Ax + b` is as an affine
        transformation of an `n`-dimensional Gaussian random vector `x` and `e ~ Noise`
        is an `m`-dimensional centered, Gaussian noise vector.

        Args:
            weight: Matrix `A` with shape (... x n x m).
            bias: Vector `b` with shape (... x m).
            noise_covar: Noise covariance matrix with shape (... x m x m).
        """
        Likelihood.__init__(self)
        self.weight = weight
        self.bias = bias
        self.noise_covar = noise_covar

    def get_affine_transform(
        self, diag: Optional[Tensor] = None
    ) -> Tuple[Union[Tensor, LinearOperator], Optional[Union[Tensor, LinearOperator]]]:
        """Returns the base affine transform with sign flips for negative labels.

        Args:
            diag: Scaling factors `d` for the affine transform such that (DA, Db) is
            returned, where `D = diag(d)`.

        Returns:
            Tensor representation of the affine transform (A, b).

        """
        if diag is None:
            return self.weight, self.bias

        D = DiagLinearOperator(diag)
        return D @ self.weight, None if (self.bias is None) else D @ self.bias

    def marginal(
        self,
        function_dist: MultivariateNormal,
        observations: Optional[BoolTensor] = None,
    ) -> TruncatedMultivariateNormal:
        """Returns the truncated multivariate normal distribution of `h | h > 0`, where
        `x` is a Gaussian random vector, `h = (Ax + b) + e`, and `e ~ Noise`."""
        gauss_loc = function_dist.loc
        gauss_cov = function_dist.covariance_matrix
        signed_labels = (
            None
            if observations is None
            else 2 * observations.to(dtype=gauss_loc.dtype, device=gauss_loc.device) - 1
        )

        A, b = self.get_affine_transform(diag=signed_labels)
        trunc_loc = A @ gauss_loc if (b is None) else A @ gauss_loc + b
        trunc_cov = A @ gauss_cov @ A.transpose(-1, -2)
        if self.noise_covar is not None:
            noise_diag = self.noise_covar(shape=trunc_cov.shape[:-1])
            trunc_cov = (trunc_cov + noise_diag).to_dense()

        return TruncatedMultivariateNormal(
            loc=trunc_loc,
            covariance_matrix=trunc_cov,
            bounds=pad(torch.full_like(trunc_loc, float("inf")).unsqueeze(-1), (1, 0)),
            validate_args=False,
        )

    def log_marginal(
        self,
        observations: BoolTensor,
        function_dist: MultivariateNormal,
    ) -> Tensor:
        """Returns the log marginal likelihood `ln p(y) = ln P([2y - 1](f + e) > 0)`,
        where `f = Ax + b` and `e ~ Noise`."""
        return self.marginal(function_dist, observations=observations).log_partition

    def latent_marginal(
        self,
        function_dist: MultivariateNormal,
        observations: Optional[BoolTensor] = None,
    ) -> UnifiedSkewNormal:
        """Returns the UnifiedSkewNormal distribution of `x | f + e > 0`, where
        `x` is a Gaussian random vector, `f = Ax + b`, and `e ~ Noise`."""
        gauss_loc = function_dist.loc
        gauss_cov = function_dist.covariance_matrix
        signed_labels = (
            None
            if observations is None
            else 2 * observations.to(dtype=gauss_loc.dtype, device=gauss_loc.device) - 1
        )

        A, b = self.get_affine_transform(diag=signed_labels)
        trunc_loc = A @ gauss_loc if (b is None) else A @ gauss_loc + b
        cross_cov = A @ gauss_cov
        trunc_cov = cross_cov @ A.transpose(-1, -2)
        if self.noise_covar is not None:
            noise_diag = self.noise_covar(shape=trunc_cov.shape[:-1])
            trunc_cov = (trunc_cov + noise_diag).to_dense()

        trunc = TruncatedMultivariateNormal(
            loc=trunc_loc,
            covariance_matrix=trunc_cov,
            bounds=pad(torch.full_like(trunc_loc, float("inf")).unsqueeze(-1), (1, 0)),
            validate_args=False,
        )

        return UnifiedSkewNormal(
            trunc=trunc,
            gauss=function_dist,
            cross_covariance_matrix=cross_cov,
        )
