"""Multinomial logistic regression solved with full-batch L-BFGS in torch (GPU if present).

Drop-in for sklearn.linear_model.LogisticRegression(C, class_weight="balanced")
on the standardised fused features: the same objective,

    0.5 * ||W||^2 + C * sum_i w_{y_i} * CE(x_i, y_i)      (bias not penalised)

but ~5x faster (GPU) on a few thousand 1-3k-dim rows, which makes grouped CV over
backbones x C grid practical. Exposes coef_, intercept_ and decision_function.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, C: float = 1.0, max_iter: int = 500, class_weight: str | None = "balanced",
                 tol: float = 1e-6, device: str | None = None):
        self.C, self.max_iter, self.class_weight, self.tol = C, max_iter, class_weight, tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchLogisticRegression":
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        yi = np.searchsorted(self.classes_, y)
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(yi, device=self.device)
        if self.class_weight == "balanced":
            cw = len(yi) / (k * np.bincount(yi, minlength=k))
        else:
            cw = np.ones(k)
        sw = torch.as_tensor(cw, dtype=torch.float64, device=self.device)[y_t]
        W = torch.zeros(X.shape[1], k, dtype=torch.float64, device=self.device, requires_grad=True)
        b = torch.zeros(k, dtype=torch.float64, device=self.device, requires_grad=True)
        # Divide the objective by C * N so its scale (and L-BFGS tolerances) do not depend on C or N.
        scale = 1.0 / (self.C * len(yi))
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(X_t @ W + b, y_t, reduction="none")
            loss = (sw * ce).sum() / len(yi) + 0.5 * scale * (W * W).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.coef_ = W.detach().T.cpu().numpy()
        self.intercept_ = b.detach().cpu().numpy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(1)]
