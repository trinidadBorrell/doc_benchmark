"""Provide scikit-learn estimator for kernel ridge classification."""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score


class KernelRidgeClassifier(KernelRidge):
    """Scikit-learn estimator for kernel ridge classification.

    Fits a KernelRidge Regressor to the binary labels 0 and 1 and
    then uses 0.5 as a threshold to binarise the predictions.
    """

    def fit(self, X, y):
        """Implement fit."""
        y_arr = np.asarray(y)
        self._is_binary_target = y_arr.ndim == 1 and set(np.unique(y_arr)).issubset(
            {0, 1}
        )

        # Preserve strict binary-classifier semantics for 1-D labels while
        # remaining compatible with existing multiclass one-hot usage.
        if y_arr.ndim == 1 and not self._is_binary_target:
            raise AssertionError("Target values should be 0 or 1!")

        super().fit(X, y)
        return self

    def decision_function(self, X):
        """Return raw Kernel Ridge outputs before thresholding."""
        return super().predict(X)

    def predict(self, X):
        """Implement predict."""
        raw = self.decision_function(X)
        if getattr(self, "_is_binary_target", False):
            return np.where(raw < 0.5, 0, 1)
        return raw

    def score(self, X, y, sample_weight=None):
        """Implement score."""
        if getattr(self, "_is_binary_target", False):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        return super().score(X, y, sample_weight=sample_weight)