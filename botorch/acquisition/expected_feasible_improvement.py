#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.skew_gp import ExactMarginalLogLikelihood_v2, SkewGPClassifier
from botorch.utils.transforms import t_batch_mode_transform


class SkewGPClassifierMixin:
    def __init__(
        self,
        X_pending: Optional[torch.Tensor] = None,
        classifier: Optional[SkewGPClassifier] = None,
    ) -> None:
        self.set_X_pending(X_pending)
        self._classifier = classifier

        # Fit classifier in advance to avoid it done inside the query optimizer's closure
        self.classifier  # TODO: Improve me

    def set_X_pending(self, *args, **kwargs) -> None:
        AcquisitionFunction.set_X_pending(self, *args, **kwargs)
        self._classifier = None

    @property
    def classifier(self) -> Optional[SkewGPClassifier]:
        if self._classifier is None and self.X_pending is not None:
            X_succ = self.model.train_inputs[0]
            X_fail = self.X_pending
            # deal with multi-output SingleTaskGP models (which have an additional batch dim)
            if X_succ.ndim > X_fail.ndim:
                if not all((X_ == X_succ[0]).all() for X_ in X_succ[1:]):
                    # if we don't have a block design things are ambiguous - give up
                    raise UnsupportedError("Only block design models are supported")
                X_succ = X_succ[0]
            X = torch.cat([X_succ, X_fail], dim=0)
            Y = torch.cat(
                [
                    torch.full(X_succ.shape[:-1], True),
                    torch.full(X_fail.shape[:-1], False),
                ],
                dim=0,
            )
            model = self._classifier = SkewGPClassifier(train_X=X, train_Y=Y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood_v2(model.likelihood, model))
        return self._classifier


class ExpectedFeasibleImprovement(SkewGPClassifierMixin, ExpectedImprovement):
    def __init__(
        self,
        *args,
        X_pending: Optional[torch.Tensor] = None,
        classifier: Optional[SkewGPClassifier] = None,
        **kwargs,
    ):
        ExpectedImprovement.__init__(self, *args, **kwargs)
        SkewGPClassifierMixin.__init__(self, X_pending=X_pending, classifier=classifier)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ei = super().forward(X)
        if self.classifier is None:
            return ei

        p_feas = self.classifier.posterior_predictive(X)
        return p_feas.mean.view(ei.shape) * ei


class qExpectedFeasibleImprovement(SkewGPClassifierMixin, qExpectedImprovement):
    def __init__(
        self,
        *args,
        X_pending: Optional[torch.Tensor] = None,
        classifier: Optional[SkewGPClassifier] = None,
        **kwargs,
    ):
        qExpectedImprovement.__init__(self, *args, **kwargs)
        SkewGPClassifierMixin.__init__(self, X_pending=X_pending, classifier=classifier)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ei = super().forward(X)
        if self.classifier is None:
            return ei

        p_feas = self.classifier.posterior_predictive(X)
        return p_feas.mean.view(ei.shape) * ei
