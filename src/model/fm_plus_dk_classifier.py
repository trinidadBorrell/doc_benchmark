"""Run nested CV classification on pooled embeddings concatenated with DK markers.

This script combines per-session foundation model pooled embeddings (dimension D)
with domain-knowledge markers from baseline scalars CSV (all available markers by
default, or evoked-only subset via ``--evoked-only``) to obtain D+M features per
subject/session, then reuses the same classification pipeline as
``fm_embedding_classifier.py`` (MLP, Random Forest, Kernel Ridge).

Common session pool
-------------------
Before any model is run, the intersection of sessions across all FM models,
DK markers, and the labels file is computed.  All models are then evaluated on
this common pool using **identical** nested CV folds (StratifiedGroupKFold outer,
StratifiedGroupKFold inner) so that performance comparisons are fair.

Shifted (permutation) variants
--------------------------------
Three additional "null" conditions permute the FM and/or DK feature blocks
**within each fold** (train and test independently, with different seeds) to
break the feature-label association for the permuted modality:

- ``shift_fm``   : FM block permuted, DK block intact
- ``shift_dk``   : DK block permuted, FM block intact
- ``shift_both`` : both blocks permuted (with independent permutations)

Output layout per foundation model:
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED/
        {target}/nested_cv/{classifier_name}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_FM_SHIFTED/
        {target}/nested_cv/{classifier_name}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_DK_SHIFTED/
        {target}/...
    {results_root}/{FM_model}/doc_patients/EMBEDDING_DK_COMBINED_BOTH_SHIFTED/
        {target}/...
"""

import argparse
from datetime import datetime
import joblib
import json
import os
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP
    from .cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
        load_cv_splits,
    )
    from .kernel_ridge_classifier import KernelRidgeClassifier
except ImportError:
    from mlp_embedding_classifier import EmbeddingClassifier, REDUCTION_MAP
    from cv_utils import (
        build_common_session_pool,
        generate_nested_cv_folds,
        save_cv_splits,
        load_cv_splits,
    )
    from kernel_ridge_classifier import KernelRidgeClassifier


DEFAULT_RESULTS_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
DEFAULT_MARKER_CSV = (
    "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
)
DEFAULT_PATIENT_LABELS = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)
DEFAULT_PATIENT_LABELS_FULL = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)

DEFAULT_POOLED_SUBPATH = op.join("doc_patients", "MLP_EMBEDDING", "pooled_embeddings")
TARGET_CHOICES = ["crs", "etiology", "cs_6m", "cs_1y", "cs_2y"]

# Evoked markers (TimeLocked, WindowDecoding, CNV) + SymbolicMutualInformation
EVOKED_MARKER_PREFIXES = (
    "nice/marker/TimeLocked",
    "nice/marker/WindowDecoding",
    "nice/marker/ContingentNegativeVariation",
    "nice/marker/SymbolicMutualInformation",
    "nice/marker/PermutationEntropy",
    "nice/marker/KolmogorovComplexity",
    "nice/marker/PowerSpectralDensity",
)

SHIFT_MODES = ("shift_fm", "shift_dk", "shift_both")


def discover_foundation_models(results_root, pooled_subpath):
    """Return model directory names that contain a pooled embeddings folder."""
    if not op.isdir(results_root):
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    model_names = []
    for name in sorted(os.listdir(results_root)):
        candidate = op.join(results_root, name, pooled_subpath)
        if op.isdir(candidate):
            model_names.append(name)
    return model_names


class DKCombinedEmbeddingClassifier(EmbeddingClassifier):
    """Embedding classifier with marker concatenation: X = [FM || DK].

    After ``load_embeddings`` is called, the instance attributes
    ``n_fm_features`` and ``n_dk_features`` record the block sizes so that
    downstream permutation utilities can split the combined feature matrix.
    """

    def __init__(
        self,
        *args,
        marker_csv,
        marker_reduction,
        expected_marker_dim=None,
        evoked_only=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pooled_embeddings_dir = self.data_dir
        self.marker_csv = marker_csv
        self.marker_reduction = marker_reduction
        self.expected_marker_dim = expected_marker_dim
        self.evoked_only = evoked_only
        self._markers_dict = None
        self._marker_names = None
        self.n_fm_features: int | None = None
        self.n_dk_features: int | None = None

    def _load_markers_once(self):
        if self._markers_dict is None or self._marker_names is None:
            markers_dict, marker_names = self.load_markers_from_csv(
                self.marker_csv, self.marker_reduction
            )

            if self.evoked_only:
                keep_idx = [
                    i
                    for i, name in enumerate(marker_names)
                    if name.startswith(EVOKED_MARKER_PREFIXES)
                ]
                marker_names = [marker_names[i] for i in keep_idx]
                markers_dict = {key: vec[keep_idx] for key, vec in markers_dict.items()}
                print(
                    f"   Filtered to {len(marker_names)} evoked markers",
                    flush=True,
                )
            else:
                print(
                    f"   Using all {len(marker_names)} DK markers",
                    flush=True,
                )

            self._markers_dict = markers_dict
            self._marker_names = marker_names

            if (
                self.expected_marker_dim is not None
                and len(self._marker_names) != self.expected_marker_dim
            ):
                print(
                    "Warning: expected "
                    f"{self.expected_marker_dim} marker dimensions, got "
                    f"{len(self._marker_names)}.",
                    flush=True,
                )
        return self._markers_dict, self._marker_names

    def load_embeddings(self, embedding_suffix="_embedding.npy"):
        """Load pooled embeddings and append marker vectors per session key.

        Sets ``self.n_fm_features`` and ``self.n_dk_features`` for use by
        permutation utilities after this method is called.
        """
        base_embeddings = super().load_embeddings(embedding_suffix=embedding_suffix)
        markers_dict, marker_names = self._load_markers_once()

        combined = {}
        n_missing_markers = 0
        n_nan_rows = 0

        fm_dim: int | None = None
        dk_dim: int | None = None

        for key, emb in sorted(base_embeddings.items()):
            marker_vec = markers_dict.get(key)
            if marker_vec is None:
                n_missing_markers += 1
                continue

            emb_vec = np.asarray(emb, dtype=float).reshape(-1)
            marker_vec = np.asarray(marker_vec, dtype=float).reshape(-1)

            if np.isnan(emb_vec).any() or np.isnan(marker_vec).any():
                n_nan_rows += 1
                continue

            if fm_dim is None:
                fm_dim = int(emb_vec.shape[0])
                dk_dim = int(marker_vec.shape[0])

            combined[key] = np.concatenate([emb_vec, marker_vec], axis=0)

        if not combined:
            raise ValueError(
                "No combined embeddings available after intersecting pooled "
                "embeddings with marker CSV."
            )

        self.n_fm_features = fm_dim
        self.n_dk_features = dk_dim

        first_key = next(iter(combined))
        total_dim = int(combined[first_key].shape[0])
        print(
            "   Combined embeddings ready: "
            f"{len(combined)} sessions, fm_dim={fm_dim}, marker_dim={len(marker_names)}, "
            f"total_dim={total_dim}, missing_markers={n_missing_markers}, "
            f"nan_rows={n_nan_rows}",
            flush=True,
        )
        return combined


# ===========================================================================
# Shifted (permutation) CV utilities
# ===========================================================================


def _permute_block(X, n_fm_features, shift_mode, rng):
    """Return a copy of X with FM and/or DK feature blocks permuted row-wise.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Combined feature matrix [FM || DK].
    n_fm_features : int
        Number of FM features (columns 0 … n_fm_features-1).
    shift_mode : str
        One of ``shift_fm``, ``shift_dk``, ``shift_both``.
    rng : np.random.RandomState
        Source of randomness.  Caller is responsible for seeding it in a
        way that is independent between train and test permutations.

    Returns
    -------
    X_perm : ndarray, same shape as X (copy).
    """
    X_out = X.copy()
    n = X.shape[0]

    if shift_mode in ("shift_fm", "shift_both"):
        perm = rng.permutation(n)
        X_out[:, :n_fm_features] = X[perm, :n_fm_features]

    if shift_mode in ("shift_dk", "shift_both"):
        perm = rng.permutation(n)
        X_out[:, n_fm_features:] = X[perm, n_fm_features:]

    return X_out


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _get_classifier_and_grid(classifier_name, random_state):
    """Return (estimator, param_grid) for MLP, RF, or KR."""
    if classifier_name == "mlp":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=500, early_stopping=True, random_state=random_state
                    ),
                ),
            ]
        )
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 1e-2],
        }
    elif classifier_name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=1
        )
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
    elif classifier_name == "kernel_ridge":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kr", KernelRidgeClassifier(kernel="rbf")),
            ]
        )
        param_grid = {
            "kr__alpha": [0.01, 0.1, 1.0, 10.0],
            "kr__gamma": [0.001, 0.01, 0.1, 1.0],
        }
    elif classifier_name == "svm":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        param_grid = {
            "svc__C": [0.01, 0.1, 1.0, 10.0],
            "svc__kernel": ["rbf", "linear"],
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return estimator, param_grid


def _predict(estimator, X_test, classifier_name):
    """Return (y_pred, y_score) for any of the three classifiers."""
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
        y_score = proba[:, 1] if proba.ndim == 2 else np.asarray(proba).reshape(-1)
    elif classifier_name == "kernel_ridge":
        if isinstance(estimator, Pipeline):
            raw = estimator.named_steps["kr"].decision_function(
                estimator.named_steps["scaler"].transform(X_test)
            )
        else:
            raw = estimator.decision_function(X_test)
        y_score = np.clip(np.asarray(raw).reshape(-1), 0.0, 1.0)
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X_test)).reshape(-1)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        y_score = np.asarray(y_pred).astype(float)
    return np.asarray(y_pred).astype(int), np.asarray(y_score).astype(float)


def run_shifted_nested_cv(
    X,
    y,
    groups,
    n_fm_features,
    shift_mode,
    cv_folds,
    random_state,
    classifiers=("mlp", "random_forest", "kernel_ridge", "svm"),
    output_dir=None,
    session_keys=None,
):
    """Run nested CV with within-fold permutation of FM and/or DK blocks.

    For each outer fold:
      1. Permute the specified block(s) of X_train and X_test **independently**
         using seeds derived from ``random_state + fold_idx * 100 ± block_offset``.
      2. Train on permuted X_train, evaluate on permuted X_test.

    Parameters
    ----------
    X : ndarray, shape (n_sessions, n_features)
        Combined [FM || DK] feature matrix ordered to match *cv_folds*.
    y, groups : ndarrays of shape (n_sessions,)
    n_fm_features : int
        Column split point between FM and DK blocks.
    shift_mode : str
        One of ``shift_fm``, ``shift_dk``, ``shift_both``.
    cv_folds : list of fold dicts
        From :func:`cv_utils.generate_nested_cv_folds`.
    random_state : int
    classifiers : sequence of str
    output_dir : str or None
        If provided, fitted models are saved as
        ``{output_dir}/nested_cv/{clf_name}/fold_{N:02d}_model.joblib``.
    session_keys : list of str or None
        Ordered session keys aligned with rows of *X* (i.e. ``common_sessions``).
        Used to populate the ``subjects`` accumulator for downstream plotting.

    Returns
    -------
    tuple (fold_results, accum_preds)
        fold_results : dict mapping classifier_name → list of fold-metric dicts.
        accum_preds  : dict mapping classifier_name → dict with keys
                       ``y_true``, ``y_pred``, ``y_proba``, ``subjects``
                       (lists of per-fold arrays / session key lists).
    """
    results = {clf: [] for clf in classifiers}
    accum_preds = {
        clf: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
        for clf in classifiers
    }

    for fold_idx, fold in enumerate(cv_folds):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold.get("inner_splits")

        # Independent RNGs for train/test permutations.
        rng_train = np.random.RandomState(random_state + fold_idx * 100)
        rng_test = np.random.RandomState(random_state + fold_idx * 100 + 1)

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        X_train = _permute_block(X_train_raw, n_fm_features, shift_mode, rng_train)
        X_test = _permute_block(X_test_raw, n_fm_features, shift_mode, rng_test)

        for clf_name in classifiers:
            estimator, param_grid = _get_classifier_and_grid(clf_name, random_state)
            cv_for_gs = inner_splits if inner_splits is not None else 3

            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=cv_for_gs,
                n_jobs=1,
                refit=True,
            )
            gs.fit(X_train, y_train)

            y_pred, y_score = _predict(gs.best_estimator_, X_test, clf_name)
            auc = _safe_auc(y_test, y_score)

            # Save fitted model if output_dir provided.
            if output_dir is not None:
                clf_dir = op.join(output_dir, "nested_cv", clf_name)
                os.makedirs(clf_dir, exist_ok=True)
                joblib.dump(
                    gs.best_estimator_,
                    op.join(clf_dir, f"fold_{fold_idx:02d}_model.joblib"),
                )

            # Accumulate predictions for downstream micro/macro metrics and plots.
            accum_preds[clf_name]["y_true"].append(y_test)
            accum_preds[clf_name]["y_pred"].append(y_pred)
            accum_preds[clf_name]["y_proba"].append(y_score)
            if session_keys is not None:
                accum_preds[clf_name]["subjects"].extend(
                    [session_keys[i] for i in test_idx]
                )

            results[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "auc": auc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(g_train))),
                    "best_params": gs.best_params_,
                    "shift_mode": shift_mode,
                }
            )

    return results, accum_preds


def run_pca_nested_cv(
    X,
    y,
    groups,
    n_fm_features,
    cv_folds,
    random_state,
    classifiers=("mlp", "random_forest", "kernel_ridge", "svm"),
    output_dir_combined=None,
    output_dir_fm_only=None,
    session_keys=None,
):
    """Run nested CV with per-fold PCA on the FM feature block.

    For each outer fold:

    1. Fit PCA on the FM columns of X_train with
       ``n_components = min(n_dk_features, n_fm_features, n_train - 1)``.
    2. Transform FM columns of X_train **and** X_test with the *same* fitted
       PCA (no data leakage).
    3. Train / evaluate classifiers on two conditions using the **identical**
       PCA projection for that fold:

       - ``combined``  : [FM_pca || DK] — FM_pca has the same dimensionality
                         as the DK block so the two blocks are balanced.
       - ``fm_only``   : [FM_pca]       — FM embeddings after PCA alone.

    Parameters
    ----------
    X : ndarray, shape (n_sessions, n_fm_features + n_dk_features)
        Combined [FM || DK] feature matrix ordered to match *cv_folds*.
    y, groups : ndarrays of shape (n_sessions,)
    n_fm_features : int
        Number of FM columns (columns 0 … n_fm_features-1).
    cv_folds : list of fold dicts
        From :func:`cv_utils.generate_nested_cv_folds`.
    random_state : int
    classifiers : sequence of str
    output_dir_combined : str or None
        If provided, fitted models for the combined condition are saved as
        ``{output_dir_combined}/nested_cv/{clf_name}/fold_{N:02d}_model.joblib``.
    output_dir_fm_only : str or None
        If provided, fitted models for the FM-only condition are saved
        analogously under *output_dir_fm_only*.
    session_keys : list of str or None
        Ordered session keys aligned with rows of *X*.

    Returns
    -------
    results_combined : dict  clf_name → list of fold-metric dicts
    accum_combined   : dict  clf_name → {y_true, y_pred, y_proba, subjects}
    results_fm_only  : dict  clf_name → list of fold-metric dicts
    accum_fm_only    : dict  clf_name → {y_true, y_pred, y_proba, subjects}
    """
    n_dk_features = X.shape[1] - n_fm_features

    results_combined = {clf: [] for clf in classifiers}
    results_fm_only = {clf: [] for clf in classifiers}
    accum_combined = {
        clf: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
        for clf in classifiers
    }
    accum_fm_only = {
        clf: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
        for clf in classifiers
    }

    for fold_idx, fold in enumerate(cv_folds):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold.get("inner_splits")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        # Fit PCA on FM block of training data only — no leakage.
        n_components = min(n_dk_features, n_fm_features, len(train_idx) - 1)
        if n_components != n_dk_features:
            print(
                f"   [PCA fold {fold_idx}] Requested n_components={n_dk_features} "
                f"reduced to {n_components} due to data/feature constraints.",
                flush=True,
            )
        pca = PCA(n_components=n_components)
        X_fm_train_pca = pca.fit_transform(X_train_raw[:, :n_fm_features])
        X_fm_test_pca = pca.transform(X_test_raw[:, :n_fm_features])
        pca_var_exp = float(pca.explained_variance_ratio_.sum())

        X_dk_train = X_train_raw[:, n_fm_features:]
        X_dk_test = X_test_raw[:, n_fm_features:]

        X_train_combined = np.concatenate([X_fm_train_pca, X_dk_train], axis=1)
        X_test_combined = np.concatenate([X_fm_test_pca, X_dk_test], axis=1)

        cv_for_gs = inner_splits if inner_splits is not None else 3

        for clf_name in classifiers:
            # ---- Combined [FM_pca || DK] ----
            est_c, grid_c = _get_classifier_and_grid(clf_name, random_state)
            gs_c = GridSearchCV(
                estimator=est_c,
                param_grid=grid_c,
                scoring="balanced_accuracy",
                cv=cv_for_gs,
                n_jobs=1,
                refit=True,
            )
            gs_c.fit(X_train_combined, y_train)
            y_pred_c, y_score_c = _predict(gs_c.best_estimator_, X_test_combined, clf_name)
            auc_c = _safe_auc(y_test, y_score_c)

            if output_dir_combined is not None:
                clf_dir = op.join(output_dir_combined, "nested_cv", clf_name)
                os.makedirs(clf_dir, exist_ok=True)
                joblib.dump(
                    gs_c.best_estimator_,
                    op.join(clf_dir, f"fold_{fold_idx:02d}_model.joblib"),
                )

            accum_combined[clf_name]["y_true"].append(y_test)
            accum_combined[clf_name]["y_pred"].append(y_pred_c)
            accum_combined[clf_name]["y_proba"].append(y_score_c)
            if session_keys is not None:
                accum_combined[clf_name]["subjects"].extend(
                    [session_keys[i] for i in test_idx]
                )

            results_combined[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, y_pred_c)
                    ),
                    "auc": auc_c,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(g_train))),
                    "best_params": gs_c.best_params_,
                    "n_pca_components": n_components,
                    "pca_explained_variance": pca_var_exp,
                    "n_dk_features": n_dk_features,
                }
            )

            # ---- FM-PCA only ----
            est_f, grid_f = _get_classifier_and_grid(clf_name, random_state)
            gs_f = GridSearchCV(
                estimator=est_f,
                param_grid=grid_f,
                scoring="balanced_accuracy",
                cv=cv_for_gs,
                n_jobs=1,
                refit=True,
            )
            gs_f.fit(X_fm_train_pca, y_train)
            y_pred_f, y_score_f = _predict(gs_f.best_estimator_, X_fm_test_pca, clf_name)
            auc_f = _safe_auc(y_test, y_score_f)

            if output_dir_fm_only is not None:
                clf_dir = op.join(output_dir_fm_only, "nested_cv", clf_name)
                os.makedirs(clf_dir, exist_ok=True)
                joblib.dump(
                    gs_f.best_estimator_,
                    op.join(clf_dir, f"fold_{fold_idx:02d}_model.joblib"),
                )

            accum_fm_only[clf_name]["y_true"].append(y_test)
            accum_fm_only[clf_name]["y_pred"].append(y_pred_f)
            accum_fm_only[clf_name]["y_proba"].append(y_score_f)
            if session_keys is not None:
                accum_fm_only[clf_name]["subjects"].extend(
                    [session_keys[i] for i in test_idx]
                )

            results_fm_only[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, y_pred_f)
                    ),
                    "auc": auc_f,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(g_train))),
                    "best_params": gs_f.best_params_,
                    "n_pca_components": n_components,
                    "pca_explained_variance": pca_var_exp,
                }
            )

    return results_combined, accum_combined, results_fm_only, accum_fm_only


def run_pca_shifted_nested_cv(
    X,
    y,
    groups,
    n_fm_features,
    shift_mode,
    cv_folds,
    random_state,
    classifiers=("mlp", "random_forest", "kernel_ridge", "svm"),
    output_dir=None,
    session_keys=None,
):
    """Run nested CV with per-fold PCA on FM block followed by within-fold permutation.

    Per outer fold:

    1. Fit PCA on X_train FM columns (n_components = min(dk_dim, fm_dim, n_train-1)).
    2. Transform FM columns of X_train **and** X_test with the *same* fitted PCA.
    3. Permute the PCA-reduced FM and/or raw DK block using independent RNGs for
       train and test (same seeding logic as :func:`run_shifted_nested_cv`).
    4. Train/evaluate classifiers on the permuted [FM_pca || DK] matrix.

    This gives the null-distribution baseline for ``EMBEDDING_DK_COMBINED_FM_PCA``.

    Parameters
    ----------
    X : ndarray, shape (n_sessions, n_fm_features + n_dk_features)
    y, groups : ndarrays of shape (n_sessions,)
    n_fm_features : int
    shift_mode : str — one of ``shift_fm``, ``shift_dk``, ``shift_both``
    cv_folds, random_state, classifiers, output_dir, session_keys :
        Same semantics as :func:`run_shifted_nested_cv`.

    Returns
    -------
    fold_results : dict  clf_name → list of fold-metric dicts
    accum_preds  : dict  clf_name → {y_true, y_pred, y_proba, subjects}
    """
    n_dk_features = X.shape[1] - n_fm_features

    results = {clf: [] for clf in classifiers}
    accum_preds = {
        clf: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
        for clf in classifiers
    }

    for fold_idx, fold in enumerate(cv_folds):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold.get("inner_splits")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        # Fit PCA on training FM block only.
        n_components = min(n_dk_features, n_fm_features, len(train_idx) - 1)
        pca = PCA(n_components=n_components)
        X_fm_train_pca = pca.fit_transform(X_train_raw[:, :n_fm_features])
        X_fm_test_pca = pca.transform(X_test_raw[:, :n_fm_features])
        pca_var_exp = float(pca.explained_variance_ratio_.sum())

        # Assemble [FM_pca || DK] before permutation.
        X_train_combined = np.concatenate(
            [X_fm_train_pca, X_train_raw[:, n_fm_features:]], axis=1
        )
        X_test_combined = np.concatenate(
            [X_fm_test_pca, X_test_raw[:, n_fm_features:]], axis=1
        )

        # Permute using n_components as the FM split point (PCA-reduced).
        rng_train = np.random.RandomState(random_state + fold_idx * 100)
        rng_test = np.random.RandomState(random_state + fold_idx * 100 + 1)
        X_train = _permute_block(X_train_combined, n_components, shift_mode, rng_train)
        X_test = _permute_block(X_test_combined, n_components, shift_mode, rng_test)

        cv_for_gs = inner_splits if inner_splits is not None else 3

        for clf_name in classifiers:
            estimator, param_grid = _get_classifier_and_grid(clf_name, random_state)
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=cv_for_gs,
                n_jobs=1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            y_pred, y_score = _predict(gs.best_estimator_, X_test, clf_name)
            auc = _safe_auc(y_test, y_score)

            if output_dir is not None:
                clf_dir = op.join(output_dir, "nested_cv", clf_name)
                os.makedirs(clf_dir, exist_ok=True)
                joblib.dump(
                    gs.best_estimator_,
                    op.join(clf_dir, f"fold_{fold_idx:02d}_model.joblib"),
                )

            accum_preds[clf_name]["y_true"].append(y_test)
            accum_preds[clf_name]["y_pred"].append(y_pred)
            accum_preds[clf_name]["y_proba"].append(y_score)
            if session_keys is not None:
                accum_preds[clf_name]["subjects"].extend(
                    [session_keys[i] for i in test_idx]
                )

            results[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "auc": auc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(g_train))),
                    "best_params": gs.best_params_,
                    "n_pca_components": n_components,
                    "pca_explained_variance": pca_var_exp,
                    "shift_mode": shift_mode,
                }
            )

    return results, accum_preds


def _summarize_fold_metrics(fold_rows):
    """Aggregate fold-level metric dicts into mean ± std summary."""
    auc_vals = [r["auc"] for r in fold_rows if r["auc"] is not None]
    bal_vals = [r["balanced_accuracy"] for r in fold_rows]
    return {
        "n_folds": len(fold_rows),
        "balanced_accuracy_mean": float(np.mean(bal_vals)),
        "balanced_accuracy_std": float(np.std(bal_vals)),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
        "auc_std": float(np.std(auc_vals)) if auc_vals else None,
    }


# ===========================================================================
# Model runners
# ===========================================================================


def _validate_requested_models(requested_models, discovered_models):
    if requested_models is None:
        return discovered_models

    discovered_set = set(discovered_models)
    missing = [m for m in requested_models if m not in discovered_set]
    if missing:
        raise ValueError(
            "Requested FM model(s) not found with pooled embeddings path: "
            f"{missing}. Available: {discovered_models}"
        )
    return requested_models


def _run_target_for_model(
    model_name,
    pooled_dir,
    args,
    target,
    precomputed_splits,
    common_sessions,
):
    """Run original (non-shifted) DK+FM classification for one model/target."""
    out_base = op.join(
        args.results_root,
        model_name,
        "doc_patients",
        "EMBEDDING_DK_COMBINED",
        target,
    )
    os.makedirs(out_base, exist_ok=True)

    classifier = DKCombinedEmbeddingClassifier(
        data_dir=pooled_dir,
        patient_labels_file=args.patient_labels,
        output_dir=out_base,
        random_state=args.random_state,
        full_cv=True,
        n_cv_folds=args.n_cv_folds,
        classifiers=tuple(args.classifiers),
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        expected_marker_dim=args.expected_marker_dim,
        evoked_only=args.evoked_only,
    )

    print("=" * 80, flush=True)
    print(f"FM model: {model_name} | target: {target} | mode: original", flush=True)
    print(f"Input pooled embeddings: {pooled_dir}", flush=True)
    print(f"Output directory: {out_base}", flush=True)
    print("=" * 80, flush=True)

    cv_kwargs = dict(
        precomputed_splits=precomputed_splits,
        common_sessions=common_sessions,
    )

    if target == "crs":
        classifier.run_full_cv(target="crs", **cv_kwargs)
    else:
        classifier.run_full_cv(
            target=target,
            labels_file=args.patient_labels_full,
            binary_outcome=args.binary_outcome,
            death_binary=args.death_binary,
            **cv_kwargs,
        )

    # Return n_fm_features for the shifted runs.
    return classifier.n_fm_features


def _run_shifted_target_for_model(
    model_name,
    pooled_dir,
    args,
    target,
    shift_mode,
    precomputed_splits,
    common_sessions,
    n_fm_features,
):
    """Run one shifted variant for one model/target using precomputed splits."""
    suffix_map = {
        "shift_fm": "EMBEDDING_DK_COMBINED_FM_SHIFTED",
        "shift_dk": "EMBEDDING_DK_COMBINED_DK_SHIFTED",
        "shift_both": "EMBEDDING_DK_COMBINED_BOTH_SHIFTED",
    }
    out_base = op.join(
        args.results_root,
        model_name,
        "doc_patients",
        suffix_map[shift_mode],
        target,
    )
    os.makedirs(out_base, exist_ok=True)

    print("=" * 80, flush=True)
    print(
        f"FM model: {model_name} | target: {target} | mode: {shift_mode}",
        flush=True,
    )
    print(f"Output directory: {out_base}", flush=True)
    print("=" * 80, flush=True)

    # Build combined feature matrix for common_sessions.
    classifier = DKCombinedEmbeddingClassifier(
        data_dir=pooled_dir,
        patient_labels_file=args.patient_labels,
        output_dir=out_base,
        random_state=args.random_state,
        full_cv=True,
        n_cv_folds=args.n_cv_folds,
        classifiers=tuple(args.classifiers),
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        expected_marker_dim=args.expected_marker_dim,
        evoked_only=args.evoked_only,
    )

    # collect_data returns (X, y, subjects); use common_sessions ordering.
    X_all, y_all, subjects_all = classifier.collect_data(target="crs")

    session_to_idx = {s: i for i, s in enumerate(subjects_all)}
    missing = [s for s in common_sessions if s not in session_to_idx]
    if missing:
        raise ValueError(
            f"[{model_name}/{shift_mode}] {len(missing)} common sessions missing: "
            f"{missing[:3]}"
        )
    keep = [session_to_idx[s] for s in common_sessions]
    X = X_all[keep]
    y = y_all[keep]
    groups = np.array([common_sessions[i].split("_ses-")[0] for i in range(len(keep))])

    if n_fm_features is None:
        # Fall back: load embeddings to determine block size.
        _ = classifier.load_embeddings()
        n_fm_features = classifier.n_fm_features

    fold_results, accum_preds = run_shifted_nested_cv(
        X=X,
        y=y,
        groups=groups,
        n_fm_features=n_fm_features,
        shift_mode=shift_mode,
        cv_folds=precomputed_splits,
        random_state=args.random_state,
        classifiers=tuple(args.classifiers),
        output_dir=out_base,
        session_keys=common_sessions,
    )

    _clf_display = {
        "mlp": "MLP",
        "random_forest": "Random Forest",
        "kernel_ridge": "Kernel Ridge",
        "svm": "SVM",
    }

    # Save per-classifier results: CSV + JSON + same plots as original variant.
    for clf_name, fold_rows in fold_results.items():
        clf_dir = op.join(out_base, "nested_cv", clf_name)
        os.makedirs(clf_dir, exist_ok=True)

        for row in fold_rows:
            row["model"] = model_name
            row["shift_mode"] = shift_mode
            row["classifier"] = clf_name
            row["target"] = target
            row["best_params"] = json.dumps(row["best_params"], sort_keys=True)

        pd.DataFrame(fold_rows).to_csv(
            op.join(clf_dir, "fold_metrics.csv"), index=False
        )
        summary = _summarize_fold_metrics(fold_rows)
        summary.update(
            {
                "model": model_name,
                "shift_mode": shift_mode,
                "classifier": clf_name,
                "target": target,
            }
        )
        with open(op.join(clf_dir, "summary_metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        auc_str = (
            f"{summary['auc_mean']:.3f}" if summary["auc_mean"] is not None else "N/A"
        )
        print(
            f"  [{model_name}|{shift_mode}|{clf_name}] "
            f"bal_acc={summary['balanced_accuracy_mean']:.3f} "
            f"auc={auc_str}",
            flush=True,
        )

        # Generate plots using accumulated predictions (mirrors original variant).
        clf_accum = accum_preds[clf_name]
        if clf_accum["y_true"]:
            y_true_all = np.concatenate(clf_accum["y_true"])
            y_pred_all = np.concatenate(clf_accum["y_pred"])
            y_proba_all = np.concatenate(clf_accum["y_proba"])
            subjects_all = clf_accum["subjects"]

            plot_results = classifier._compute_test_metrics(
                y_true_all, y_pred_all, y_proba_all, subjects_all
            )
            plot_results.update(
                {
                    "n_samples": int(len(y_true_all)),
                    "n_features": int(X.shape[1]),
                    "n_subjects": int(
                        len(np.unique([s.split("_ses-")[0] for s in subjects_all]))
                        if subjects_all
                        else 0
                    ),
                    "full_cv": True,
                    "n_folds": len(fold_rows),
                    "shift_mode": shift_mode,
                }
            )
            macro_results = classifier._compute_macro_average(
                clf_accum["y_true"],
                clf_accum["y_pred"],
                clf_accum["y_proba"],
            )
            plot_results["macro_average"] = macro_results

            display_name = f"{_clf_display.get(clf_name, clf_name)} [{shift_mode}]"
            classifier._save_results(
                plot_results, clf_dir, model=None, model_type=clf_name
            )
            classifier._plot_results(plot_results, display_name, clf_dir)
            classifier._plot_macro_vs_micro(plot_results, display_name, clf_dir)

    return fold_results


_CLF_DISPLAY = {
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "kernel_ridge": "Kernel Ridge",
    "svm": "SVM",
}

_OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}
_ETIOLOGY_TARGETS = {"etiology", "etiology_code"}
_ALL_TARGETS = ["crs", "etiology", "etiology_code", "cs_6m", "cs_1y", "cs_2y"]

# (mode_name, binary_outcome, death_binary, vs_to_mcs, mcs_to_conscious,
#  binary_improvement, baseline_filter)
_OUTCOME_MODES = [
    ("multiclass", False, False, False, False, False, None),
    ("binary", True, False, False, False, False, None),
    ("binary_death", False, True, False, False, False, None),
    ("binary_vs_to_mcs", False, False, True, False, False, None),
    ("binary_mcs_to_conscious", False, False, False, True, False, None),
    ("binary_improvement", False, False, False, False, True, None),
]
# (mode_name, baseline_filter)
_ETIOLOGY_MODES = [("", None), ("vs_only", "VS"), ("mcs_only", "MCS")]


def _save_pca_condition_results(
    fold_results,
    accum,
    out_base,
    X_shape,
    classifier,
    model_name,
    condition_tag,
    target,
):
    """Save fold CSV, summary JSON, and plots for one PCA condition."""
    for clf_name, fold_rows in fold_results.items():
        clf_dir = op.join(out_base, "nested_cv", clf_name)
        os.makedirs(clf_dir, exist_ok=True)

        for row in fold_rows:
            row.setdefault("model", model_name)
            row.setdefault("condition", condition_tag)
            row.setdefault("classifier", clf_name)
            row.setdefault("target", target)
            if "best_params" in row and not isinstance(row["best_params"], str):
                row["best_params"] = json.dumps(row["best_params"], sort_keys=True)

        pd.DataFrame(fold_rows).to_csv(op.join(clf_dir, "fold_metrics.csv"), index=False)
        summary = _summarize_fold_metrics(fold_rows)
        summary.update(
            {
                "model": model_name,
                "condition": condition_tag,
                "classifier": clf_name,
                "target": target,
            }
        )
        with open(op.join(clf_dir, "summary_metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        auc_str = (
            f"{summary['auc_mean']:.3f}" if summary["auc_mean"] is not None else "N/A"
        )
        print(
            f"  [{model_name}|{condition_tag}|{clf_name}] "
            f"bal_acc={summary['balanced_accuracy_mean']:.3f} auc={auc_str}",
            flush=True,
        )

        clf_accum = accum[clf_name]
        if clf_accum["y_true"]:
            y_true_all = np.concatenate(clf_accum["y_true"])
            y_pred_all = np.concatenate(clf_accum["y_pred"])
            y_proba_all = np.concatenate(clf_accum["y_proba"])
            subjects_plot = clf_accum["subjects"]

            plot_results = classifier._compute_test_metrics(
                y_true_all, y_pred_all, y_proba_all, subjects_plot
            )
            plot_results.update(
                {
                    "n_samples": int(len(y_true_all)),
                    "n_features": int(X_shape[1]),
                    "n_subjects": int(
                        len(np.unique([s.split("_ses-")[0] for s in subjects_plot]))
                        if subjects_plot
                        else 0
                    ),
                    "full_cv": True,
                    "n_folds": len(fold_rows),
                    "condition": condition_tag,
                }
            )
            macro_results = classifier._compute_macro_average(
                clf_accum["y_true"],
                clf_accum["y_pred"],
                clf_accum["y_proba"],
            )
            plot_results["macro_average"] = macro_results
            display_name = f"{_CLF_DISPLAY.get(clf_name, clf_name)} [{condition_tag}]"
            classifier._save_results(plot_results, clf_dir, model=None, model_type=clf_name)
            classifier._plot_results(plot_results, display_name, clf_dir)
            classifier._plot_macro_vs_micro(plot_results, display_name, clf_dir)


def _make_dk_classifier(pooled_dir, args, out_dir):
    """Construct a DKCombinedEmbeddingClassifier for the given output directory."""
    return DKCombinedEmbeddingClassifier(
        data_dir=pooled_dir,
        patient_labels_file=args.patient_labels,
        output_dir=out_dir,
        random_state=args.random_state,
        full_cv=True,
        n_cv_folds=args.n_cv_folds,
        classifiers=tuple(args.classifiers),
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        expected_marker_dim=args.expected_marker_dim,
        evoked_only=args.evoked_only,
    )


def _run_target_for_model_pca(
    model_name,
    pooled_dir,
    args,
    target,
    precomputed_splits,
    common_sessions,
    shift_modes=(),
):
    """Run PCA-on-FM combined condition for one model/target (+ optional shifted).

    Produces:
    - ``EMBEDDING_DK_COMBINED_FM_PCA/{target}/`` : [FM_pca || DK]
    - ``EMBEDDING_DK_COMBINED_FM_PCA_{SHIFT}/{target}/`` : shifted null baselines

    FM-PCA-only results for all targets are handled separately by
    :func:`_run_pca_fm_only_all_targets`.
    """
    out_combined = op.join(
        args.results_root, model_name, "doc_patients",
        "EMBEDDING_DK_COMBINED_FM_PCA", target,
    )
    os.makedirs(out_combined, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"FM model: {model_name} | target: {target} | mode: pca_combined", flush=True)
    print(f"Output: {out_combined}", flush=True)
    print("=" * 80, flush=True)

    classifier = _make_dk_classifier(pooled_dir, args, out_combined)
    X_all, y_all, subjects_all = classifier.collect_data(target="crs")

    session_to_idx = {s: i for i, s in enumerate(subjects_all)}
    missing = [s for s in common_sessions if s not in session_to_idx]
    if missing:
        raise ValueError(
            f"[{model_name}/pca] {len(missing)} common sessions missing: {missing[:3]}"
        )
    keep = [session_to_idx[s] for s in common_sessions]
    X = X_all[keep]
    y = y_all[keep]
    groups = np.array([common_sessions[i].split("_ses-")[0] for i in range(len(keep))])
    n_fm_features = classifier.n_fm_features

    print(
        f"   n_fm_features={n_fm_features}, n_dk_features={classifier.n_dk_features}, "
        f"target_pca_components={classifier.n_dk_features}",
        flush=True,
    )

    # ---- Combined [FM_pca || DK] ----
    (fold_results_combined, accum_combined, _, _) = run_pca_nested_cv(
        X=X, y=y, groups=groups, n_fm_features=n_fm_features,
        cv_folds=precomputed_splits, random_state=args.random_state,
        classifiers=tuple(args.classifiers),
        output_dir_combined=out_combined, output_dir_fm_only=None,
        session_keys=common_sessions,
    )
    _save_pca_condition_results(
        fold_results_combined, accum_combined, out_combined,
        X.shape, classifier, model_name, "pca_combined", target,
    )

    # ---- Shifted null baselines for PCA-combined ----
    _PCA_SHIFT_SUFFIX = {
        "shift_fm": "EMBEDDING_DK_COMBINED_FM_PCA_FM_SHIFTED",
        "shift_dk": "EMBEDDING_DK_COMBINED_FM_PCA_DK_SHIFTED",
        "shift_both": "EMBEDDING_DK_COMBINED_FM_PCA_BOTH_SHIFTED",
    }
    for shift_mode in shift_modes:
        out_shift = op.join(
            args.results_root, model_name, "doc_patients",
            _PCA_SHIFT_SUFFIX[shift_mode], target,
        )
        os.makedirs(out_shift, exist_ok=True)
        print("=" * 80, flush=True)
        print(
            f"FM model: {model_name} | target: {target} | mode: pca+{shift_mode}",
            flush=True,
        )
        print(f"Output: {out_shift}", flush=True)
        print("=" * 80, flush=True)

        fold_results_shift, accum_shift = run_pca_shifted_nested_cv(
            X=X, y=y, groups=groups, n_fm_features=n_fm_features,
            shift_mode=shift_mode, cv_folds=precomputed_splits,
            random_state=args.random_state,
            classifiers=tuple(args.classifiers),
            output_dir=out_shift,
            session_keys=common_sessions,
        )
        condition_tag = f"pca_{shift_mode}"
        _save_pca_condition_results(
            fold_results_shift, accum_shift, out_shift,
            X.shape, classifier, model_name, condition_tag, target,
        )

    return fold_results_combined


def _run_pca_fm_only_all_targets(
    model_name,
    pooled_dir,
    args,
    precomputed_splits,
    common_sessions,
):
    """Run FM-PCA-only classification across all clinical targets.

    Mirrors the full target suite of ``mlp_embedding_classifier`` / ``baseline``:

    - ``crs``               — single binary run (uses precomputed_splits)
    - ``etiology``, ``etiology_code`` — all / vs_only / mcs_only
    - ``cs_6m``, ``cs_1y``, ``cs_2y`` — multiclass / binary / binary_death /
                              binary_vs_to_mcs / binary_mcs_to_conscious /
                              binary_improvement

    PCA is fitted per outer fold on the FM block of X_train only (no leakage).
    n_components = min(dk_dim, fm_dim, n_train-1), targeting dk_dim.

    For ``crs`` the shared ``precomputed_splits`` are used.
    For all other targets fresh StratifiedGroupKFold folds are generated for
    the sessions that have labels for that target (session pool differs from crs).
    """
    base_out = op.join(
        args.results_root, model_name, "doc_patients", "EMBEDDING_FM_PCA_ONLY"
    )

    # Load embeddings once to get n_fm_features / n_dk_features.
    _tmp = _make_dk_classifier(pooled_dir, args, base_out)
    _tmp.collect_data(target="crs")  # populates n_fm_features / n_dk_features
    n_fm_features = _tmp.n_fm_features
    n_dk_features = _tmp.n_dk_features
    print(
        f"\n[{model_name}] FM-PCA-only full target suite | "
        f"n_fm_features={n_fm_features}, n_dk_features={n_dk_features}",
        flush=True,
    )

    def _run_one(out_dir, target, collect_kwargs, use_splits, use_sessions):
        os.makedirs(out_dir, exist_ok=True)
        clf = _make_dk_classifier(pooled_dir, args, out_dir)

        X_all, y_all, subjects_all = clf.collect_data(target=target, **collect_kwargs)

        if use_sessions is not None:
            # Reorder to match precomputed splits (crs pool).
            s2i = {s: i for i, s in enumerate(subjects_all)}
            missing = [s for s in use_sessions if s not in s2i]
            if missing:
                raise ValueError(
                    f"[{model_name}/pca_fm_only/{target}] "
                    f"{len(missing)} common sessions missing"
                )
            keep = [s2i[s] for s in use_sessions]
            X_fold = X_all[keep]
            y_fold = y_all[keep]
            groups_fold = np.array(
                [use_sessions[i].split("_ses-")[0] for i in range(len(keep))]
            )
            session_keys = use_sessions
            folds = use_splits
        else:
            X_fold = X_all
            y_fold = y_all
            groups_fold = np.array([s.split("_ses-")[0] for s in subjects_all])
            session_keys = subjects_all
            # Build a simple labels dict for fold generation.
            _labels_for_folds = {subjects_all[i]: int(y_all[i]) for i in range(len(y_all))}
            try:
                folds = generate_nested_cv_folds(
                    common_sessions=subjects_all,
                    labels=_labels_for_folds,
                    n_outer=args.n_cv_folds,
                    random_state=args.random_state,
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not generate folds for {target}: {exc}"
                ) from exc

        if len(X_fold) == 0:
            raise ValueError(f"No sessions for target={target}")

        # Run FM-PCA-only (output_dir_combined=None).
        (_, _, fold_results_fm, accum_fm) = run_pca_nested_cv(
            X=X_fold, y=y_fold, groups=groups_fold, n_fm_features=n_fm_features,
            cv_folds=folds, random_state=args.random_state,
            classifiers=tuple(args.classifiers),
            output_dir_combined=None, output_dir_fm_only=out_dir,
            session_keys=session_keys,
        )
        _save_pca_condition_results(
            fold_results_fm, accum_fm, out_dir,
            X_fold.shape, clf, model_name, "pca_fm_only", target,
        )

    for target in _ALL_TARGETS:
        print(f"\n{'─' * 60}", flush=True)
        print(f"  [FM-PCA-only] target: {target}", flush=True)
        print(f"{'─' * 60}", flush=True)

        if target == "crs":
            try:
                _run_one(
                    out_dir=op.join(base_out, "crs"),
                    target="crs",
                    collect_kwargs={},
                    use_splits=precomputed_splits,
                    use_sessions=common_sessions,
                )
            except Exception as exc:
                print(f"   [SKIP] crs: {exc}", flush=True)

        elif target in _ETIOLOGY_TARGETS:
            for mode_name, bl_filter in _ETIOLOGY_MODES:
                subdir = (
                    op.join(base_out, target, mode_name) if mode_name
                    else op.join(base_out, target)
                )
                label = f" / {mode_name}" if mode_name else ""
                print(f"   {target}{label}", flush=True)
                try:
                    _run_one(
                        out_dir=subdir,
                        target=target,
                        collect_kwargs=dict(
                            labels_file=args.patient_labels_full,
                            baseline_filter=bl_filter,
                        ),
                        use_splits=None,
                        use_sessions=None,
                    )
                except Exception as exc:
                    print(f"   [SKIP] {target}{label}: {exc}", flush=True)

        else:  # outcome targets
            for (
                mode_name, binary_outcome, death_binary, vs_to_mcs,
                mcs_to_conscious, binary_improvement, baseline_filter,
            ) in _OUTCOME_MODES:
                subdir = op.join(base_out, target, mode_name)
                print(f"   {target} / {mode_name}", flush=True)
                try:
                    _run_one(
                        out_dir=subdir,
                        target=target,
                        collect_kwargs=dict(
                            labels_file=args.patient_labels_full,
                            binary_outcome=binary_outcome,
                            death_binary=death_binary,
                            vs_to_mcs=vs_to_mcs,
                            mcs_to_conscious=mcs_to_conscious,
                            binary_improvement=binary_improvement,
                            baseline_filter=baseline_filter,
                        ),
                        use_splits=None,
                        use_sessions=None,
                    )
                except Exception as exc:
                    print(f"   [SKIP] {target}/{mode_name}: {exc}", flush=True)


# ===========================================================================
# Comparison plot
# ===========================================================================


def plot_shifted_comparison(all_results, output_dir):
    """Bar plot comparing original vs shifted performance.

    Parameters
    ----------
    all_results : dict
        ``{model_name: {condition: {clf_name: summary_dict}}}``
        where *condition* is one of ``original``, ``shift_fm``, ``shift_dk``,
        ``shift_both``.
    output_dir : str
        Directory where PNG files are saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = ["original", "shift_fm", "shift_dk", "shift_both"]
    classifiers = ["mlp", "random_forest", "kernel_ridge", "svm"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.dpi"] = 120

    for clf_name in classifiers:
        models = sorted(all_results.keys())
        n_models = len(models)
        n_cond = len(conditions)
        x = np.arange(n_models)
        width = 0.8 / n_cond
        offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

        fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))
        for ci, cond in enumerate(conditions):
            auc_vals = []
            for model in models:
                cond_data = all_results.get(model, {}).get(cond, {})
                clf_data = cond_data.get(clf_name, {})
                auc_vals.append(clf_data.get("auc_mean"))

            y_plot = [v if v is not None else 0.0 for v in auc_vals]
            ax.bar(x + offsets[ci], y_plot, width, label=cond)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("AUC (mean across folds)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
        ax.set_title(f"Original vs Shifted — {clf_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_file = op.join(output_dir, f"shifted_comparison_{clf_name}.png")
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate pooled FM embeddings with domain-knowledge marker "
            "embeddings and run nested CV classification (MLP/RF/KR).  "
            "All models share a common session pool and identical fold "
            "assignments for fair comparison.  Optional shifted (permuted) "
            "variants test the contribution of each modality."
        )
    )

    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing {FM_model}/doc_patients/MLP_EMBEDDING/pooled_embeddings",
    )
    parser.add_argument(
        "--pooled-subpath",
        default=DEFAULT_POOLED_SUBPATH,
        help="Relative path under each model directory where pooled embeddings live",
    )
    parser.add_argument(
        "--fm-models",
        nargs="+",
        default=None,
        help="Optional list of FM models to run (default: auto-discover all)",
    )
    parser.add_argument(
        "--feature-predicted",
        nargs="+",
        default=["crs", "etiology", "etiology_code", "cs_6m", "cs_1y", "cs_2y"],
        choices=TARGET_CHOICES,
        help="Prediction target(s). Results are written under this folder name.",
    )

    parser.add_argument("--marker-csv", default=DEFAULT_MARKER_CSV)
    parser.add_argument(
        "--marker-reduction",
        choices=sorted(REDUCTION_MAP.keys()),
        default="A",
        help="Marker reduction letter from baseline CSV",
    )
    parser.add_argument(
        "--expected-marker-dim",
        type=int,
        default=None,
        help="Expected number of marker dimensions (warns if different)",
    )
    parser.add_argument(
        "--evoked-only",
        action="store_true",
        default=False,
        help="Only use evoked markers (TimeLocked, WindowDecoding, CNV, etc.)",
    )

    parser.add_argument("--patient-labels", default=DEFAULT_PATIENT_LABELS)
    parser.add_argument("--patient-labels-full", default=DEFAULT_PATIENT_LABELS_FULL)

    parser.add_argument(
        "--binary-outcome",
        action="store_true",
        default=False,
        help="For cs_* targets: map to binary VS vs MCS",
    )
    parser.add_argument(
        "--death-binary",
        action="store_true",
        default=False,
        help="For cs_* targets: map to DEATH vs NON_DEATH",
    )

    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--shift-modes",
        nargs="+",
        default=list(SHIFT_MODES),
        choices=list(SHIFT_MODES),
        help=(
            "Shifted (permutation) variants to run. "
            "Default: all three (shift_fm shift_dk shift_both). "
            "Pass an empty list to skip shifted variants."
        ),
    )
    parser.add_argument(
        "--no-shifts",
        action="store_true",
        default=False,
        help="Skip all shifted variants (equivalent to --shift-modes with no args).",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Path to a pre-computed common_cv_splits.json. "
            "When provided, the stored splits are used instead of generating new ones."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help="Path to save the generated common CV splits JSON.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["mlp", "random_forest", "kernel_ridge", "svm"],
        choices=["mlp", "random_forest", "kernel_ridge", "svm"],
        help=(
            "Subset of classifiers to train. Default: all four. "
            "Use e.g. --classifiers random_forest to restrict to RF."
        ),
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        default=False,
        help=(
            "Apply per-fold PCA to the FM embedding block before classification. "
            "PCA is fitted on the training fold only (no leakage) and the number "
            "of components is set to the DK marker dimensionality so both blocks "
            "have equal size. Produces two additional output conditions: "
            "EMBEDDING_DK_COMBINED_FM_PCA (FM_pca concatenated with DK) and "
            "EMBEDDING_FM_PCA_ONLY (FM_pca alone). Folds are shared with the "
            "main run for fair comparison."
        ),
    )
    parser.add_argument(
        "--skip-common-pool",
        action="store_true",
        default=False,
        help="Skip shared common pool generation and let the model build its own splits.",
    )

    args = parser.parse_args()

    if args.binary_outcome and args.death_binary:
        raise ValueError("Choose at most one of --binary-outcome or --death-binary.")

    shift_modes_to_run = [] if args.no_shifts else args.shift_modes

    discovered_models = discover_foundation_models(
        args.results_root, args.pooled_subpath
    )
    selected_models = _validate_requested_models(args.fm_models, discovered_models)
    if not selected_models:
        raise ValueError(
            "No foundation models found. Check --results-root and --pooled-subpath."
        )

    print(f"Discovered FM models: {discovered_models}", flush=True)
    print(f"Selected FM models: {selected_models}", flush=True)
    print(f"Targets: {args.feature_predicted}", flush=True)
    print(f"Shift modes: {shift_modes_to_run}", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Build common session pool (intersection across ALL FM models
    # + DK markers + labels).
    # ------------------------------------------------------------------
    print("\nBuilding common session pool ...", flush=True)

    # Load CRS labels as a simple dict for cv_utils.
    _ldf = pd.read_csv(args.patient_labels)
    labels_dict = {}
    for _, row in _ldf.iterrows():
        state = row.get("diagnostic_crs_final", "")
        if pd.isna(state) or state == "n/a":
            continue
        if state == "UWS":
            lbl = 0
        elif state in ("MCS+", "MCS-"):
            lbl = 1
        else:
            continue
        try:
            subj = str(row["subject"])
            ses = f"ses-{int(row['session']):02d}"
        except Exception:
            continue
        labels_dict[f"{subj}_{ses}"] = lbl

    def _make_combined_loader(
        pooled_dir,
        marker_csv,
        marker_reduction,
        evoked_only,
        expected_marker_dim,
        random_state,
        patient_labels,
    ):
        def _loader():
            tmp = DKCombinedEmbeddingClassifier(
                data_dir=pooled_dir,
                patient_labels_file=patient_labels,
                output_dir="/tmp",
                random_state=random_state,
                full_cv=True,
                marker_csv=marker_csv,
                marker_reduction=marker_reduction,
                expected_marker_dim=expected_marker_dim,
                evoked_only=evoked_only,
            )
            return tmp.load_embeddings()

        return _loader

    source_loaders = {
        model_name: _make_combined_loader(
            pooled_dir=op.join(args.results_root, model_name, args.pooled_subpath),
            marker_csv=args.marker_csv,
            marker_reduction=args.marker_reduction,
            evoked_only=args.evoked_only,
            expected_marker_dim=args.expected_marker_dim,
            random_state=args.random_state,
            patient_labels=args.patient_labels,
        )
        for model_name in selected_models
    }

    if args.skip_common_pool:
        common_sessions = None
        precomputed_splits = None
        print("Skipping common session pool building (--skip-common-pool).", flush=True)
    else:
        common_sessions = build_common_session_pool(source_loaders, labels_dict)
        print(
            f"Common pool: {len(common_sessions)} sessions, "
            f"{len({k.split('_ses-')[0] for k in common_sessions})} subjects\n",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Step 2: Generate or load nested CV splits from the common pool.
        # ------------------------------------------------------------------
        if args.splits_file and op.isfile(args.splits_file):
            print(f"Loading pre-computed splits from {args.splits_file}", flush=True)
            precomputed_splits, common_sessions, labels_dict = load_cv_splits(
                args.splits_file
            )
        else:
            print("Generating nested CV splits from common pool ...", flush=True)
            precomputed_splits = generate_nested_cv_folds(
                common_sessions=common_sessions,
                labels=labels_dict,
                n_outer=args.n_cv_folds,
                random_state=args.random_state,
            )
            if args.save_splits_to:
                save_cv_splits(
                    folds=precomputed_splits,
                    common_sessions=common_sessions,
                    labels=labels_dict,
                    path=args.save_splits_to,
                )
                print(f"Splits saved to {args.save_splits_to}", flush=True)

        if not (args.splits_file and op.isfile(args.splits_file)):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_tag = "_".join(sorted(args.feature_predicted))
            splits_dir = op.join(args.results_root, "cv_splits")
            os.makedirs(splits_dir, exist_ok=True)
            default_splits_path = op.join(splits_dir, f"dk_combined_{target_tag}_{ts}.json")
            save_cv_splits(
                folds=precomputed_splits,
                common_sessions=common_sessions,
                labels=labels_dict,
                path=default_splits_path,
            )
            print(f"Common splits saved to: {default_splits_path}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Run original + shifted variants for each model × target.
    # ------------------------------------------------------------------
    run_log = {
        "results_root": args.results_root,
        "pooled_subpath": args.pooled_subpath,
        "marker_csv": args.marker_csv,
        "marker_reduction": args.marker_reduction,
        "n_common_sessions": (
            len(common_sessions) if common_sessions is not None else None
        ),
        "n_outer_folds": (
            len(precomputed_splits) if precomputed_splits is not None else None
        ),
        "models": {},
    }

    # Accumulate results for comparison plots.
    all_plot_results: dict[str, dict] = {}

    for model_name in selected_models:
        pooled_dir = op.join(args.results_root, model_name, args.pooled_subpath)
        run_log["models"][model_name] = {}
        all_plot_results[model_name] = {}

        for target in args.feature_predicted:
            # ---- Original ----
            try:
                n_fm_features = _run_target_for_model(
                    model_name=model_name,
                    pooled_dir=pooled_dir,
                    args=args,
                    target=target,
                    precomputed_splits=precomputed_splits,
                    common_sessions=common_sessions,
                )
                run_log["models"][model_name][target] = {"original": "ok"}
            except Exception as exc:
                print(
                    f"[FAILED] original model={model_name}, target={target}: {exc}",
                    flush=True,
                )
                run_log["models"][model_name][target] = {
                    "original": {"status": "failed", "error": str(exc)}
                }
                n_fm_features = None

            # ---- PCA combined + PCA shifted variants ----
            if args.pca:
                try:
                    _run_target_for_model_pca(
                        model_name=model_name,
                        pooled_dir=pooled_dir,
                        args=args,
                        target=target,
                        precomputed_splits=precomputed_splits,
                        common_sessions=common_sessions,
                        shift_modes=shift_modes_to_run,
                    )
                    run_log["models"][model_name][target]["pca"] = "ok"
                except Exception as exc:
                    print(
                        f"[FAILED] pca model={model_name}, target={target}: {exc}",
                        flush=True,
                    )
                    run_log["models"][model_name][target]["pca"] = {
                        "status": "failed",
                        "error": str(exc),
                    }

            # ---- Shifted variants ----
            for shift_mode in shift_modes_to_run:
                try:
                    fold_results = _run_shifted_target_for_model(
                        model_name=model_name,
                        pooled_dir=pooled_dir,
                        args=args,
                        target=target,
                        shift_mode=shift_mode,
                        precomputed_splits=precomputed_splits,
                        common_sessions=common_sessions,
                        n_fm_features=n_fm_features,
                    )
                    # Summarise for plots.
                    if model_name not in all_plot_results:
                        all_plot_results[model_name] = {}
                    all_plot_results[model_name][shift_mode] = {
                        clf: _summarize_fold_metrics(rows)
                        for clf, rows in fold_results.items()
                    }
                    run_log["models"][model_name][target][shift_mode] = "ok"
                except Exception as exc:
                    print(
                        f"[FAILED] {shift_mode} model={model_name}, target={target}: {exc}",
                        flush=True,
                    )
                    run_log["models"][model_name][target][shift_mode] = {
                        "status": "failed",
                        "error": str(exc),
                    }

        # ---- FM-PCA-only across all targets (once per model) ----
        if args.pca:
            try:
                _run_pca_fm_only_all_targets(
                    model_name=model_name,
                    pooled_dir=pooled_dir,
                    args=args,
                    precomputed_splits=precomputed_splits,
                    common_sessions=common_sessions,
                )
                run_log["models"][model_name]["pca_fm_only_all_targets"] = "ok"
            except Exception as exc:
                print(
                    f"[FAILED] pca_fm_only_all_targets model={model_name}: {exc}",
                    flush=True,
                )
                run_log["models"][model_name]["pca_fm_only_all_targets"] = {
                    "status": "failed",
                    "error": str(exc),
                }

    # ------------------------------------------------------------------
    # Step 4: Comparison plots.
    # ------------------------------------------------------------------
    if all_plot_results:
        plot_dir = op.join(args.results_root, "dk_combined_shift_comparison")
        plot_shifted_comparison(all_plot_results, plot_dir)

    summary_file = op.join(args.results_root, "dk_embedding_combined_run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run summary written to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
