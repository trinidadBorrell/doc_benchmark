"""Dimension-wise marker residualization of FM embeddings and VS/MCS classification.

=================================================================
Dimension Residualization + Classification Pipeline
=================================================================

For each EEG foundation model (LaBram, CBraMod, NeuroLM, TOTEM), this
script solves the *inverse* problem of ``residualization_embeddings.py``:

Instead of probing from embeddings → markers, here we fit a Ridge
regression from markers → each embedding dimension independently
(a multi-output Ridge), then subtract the marker-predicted value from
each dimension:

    d_i = b0_i + b1_i * m1 + ... + bN_i * mN      (fit on training fold)
    d_i_res = d_i - (b0_i + b1_i * m1 + ... + bN_i * mN)

This removes the variance in each embedding dimension that is linearly
explained by the neurophysiological markers, leaving the residual
embedding that cannot be predicted from markers alone.

Pipeline per FM model
---------------------
1. **Load embeddings**: last-layer pooled embeddings from
   ``MLP_EMBEDDING/pooled_embeddings``.

2. **Load markers**: scalar markers from a CSV (same format as the
   other interp scripts).

3. **Nested CV** (StratifiedGroupKFold, subject-level groups):

   For each outer fold:

   a. Fit ``DimensionResidualizer`` on training subjects — this trains a
      multi-output Ridge regression: ``markers_train → X_train`` (all
      embedding dims at once).
   b. Residualize both train and test embeddings using their own markers:
      ``X_res = X - ridge.predict(Y_scaled)``.
   c. Inner CV hyperparameter search for the classifier (same fold
      structure as ``residualization_embeddings.py``).
   d. Evaluate AUC on the residualized test embeddings.

4. **Save results**: ``results.json``, ``results.csv``, and
   ``auc_comparison.png``.

The same ``--random-state`` seed governs both the residualizer inner CV
and the classifier inner/outer folds, and the same ``--splits-file``
format is used so folds can be shared across interp scripts for direct
comparisons.

Usage
-----
::

    python residualization_dim.py \\
        --results-root /data/project/eeg_foundation/data/benchmark_results/new_results \\
        --marker-csv /data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv \\
        --patient-labels /data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv \\
        --output-dir /path/to/output

    # Single model
    python residualization_dim.py --model LaBram ...

    # Use pre-computed splits from residualization_embeddings.py for fair comparison
    python residualization_dim.py --splits-file /path/to/residualization_crs_*.json ...

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import os
import os.path as op
import sys
from datetime import datetime
from glob import glob
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Import KernelRidgeClassifier with fallback for direct execution
try:
    from ..model.kernel_ridge_classifier import KernelRidgeClassifier
except ImportError:
    _model_dir = op.abspath(op.join(op.dirname(__file__), "..", "model"))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    from kernel_ridge_classifier import KernelRidgeClassifier

# Import CV utilities (shared with model scripts for consistent fold assignments)
try:
    from ..model.cv_utils import (
        check_no_subject_leakage,
        generate_nested_cv_folds,
        load_cv_splits,
        save_cv_splits,
    )
except ImportError:
    _model_dir = op.abspath(op.join(op.dirname(__file__), "..", "model"))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    from cv_utils import (
        check_no_subject_leakage,
        generate_nested_cv_folds,
        load_cv_splits,
        save_cv_splits,
    )

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"

# ── Constants ────────────────────────────────────────────────────────────────

FM_MODELS = ["LaBram", "CBraMod", "NeuroLM", "TOTEM"]

DEFAULT_RESULTS_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
DEFAULT_MARKER_CSV = (
    "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
)
DEFAULT_PATIENT_LABELS = (
    "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
)

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

_MODEL_COLORS = {
    "CBraMod": "#2ca02c",
    "CbraMod": "#2ca02c",
    "NeuroLM": "#1f77b4",
    "TOTEM": "#ff7f0e",
    "LaBram": "#d62728",
}

_CLASSIFIER_MARKERS = {
    "kernel_ridge": "o",
    "svm": "s",
    "mlp": "^",
    "random_forest": "D",
}

# ── Data loading ─────────────────────────────────────────────────────────────


def load_embeddings(results_root, model_name):
    """Load last-layer pooled embeddings for a foundation model.

    Returns
    -------
    embeddings : dict
        ``{subject_session_key: np.ndarray of shape (emb_dim,)}``
    """
    emb_root = op.join(
        results_root,
        model_name,
        "doc_patients",
        "MLP_EMBEDDING",
        "pooled_embeddings",
    )
    embeddings = {}
    for npz_path in sorted(glob(op.join(emb_root, "sub-*", "ses-*", "embedding.npz"))):
        parts = npz_path.split(os.sep)
        ses = [p for p in parts if p.startswith("ses-")][-1]
        sub = [p for p in parts if p.startswith("sub-")][-1]
        subject_id = sub.replace("sub-", "")
        key = f"{subject_id}_{ses}"
        data = np.load(npz_path)
        arr = data[list(data.keys())[0]]
        embeddings[key] = arr.astype(np.float64).ravel()
    print(
        f"   [{model_name}] Loaded {len(embeddings)} embeddings"
        + (
            f" (dim={next(iter(embeddings.values())).shape[0]})"
            if embeddings
            else " (none)"
        ),
        flush=True,
    )
    return embeddings


def load_markers(marker_csv, reduction="A"):
    """Load baseline scalar markers from CSV.

    Returns
    -------
    markers_dict : dict
        ``{subject_session_key: np.ndarray of shape (n_markers,)}``
    marker_names : list of str
    """
    reduction_str = REDUCTION_MAP[reduction]

    with open(marker_csv) as fh:
        header_line = fh.readline()
    sep = "," if "Reduction" in header_line.split(",") else ";"

    df = pd.read_csv(marker_csv, sep=sep)
    df_filtered = df[df["Reduction"] == reduction_str].copy()

    meta_cols = {"Subject", "Reduction", "Label", "Subject_ID", "Date", "id"}
    marker_names = [
        c
        for c in df_filtered.columns
        if c not in meta_cols
        and not str(c).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df_filtered[c])
    ]

    markers_dict = {}
    for _, row in df_filtered.iterrows():
        raw_subject = str(row["Subject"])
        parts = raw_subject.rsplit("_", 1)
        if len(parts) != 2:
            continue
        subject_id, sesnum = parts
        try:
            key = f"{subject_id}_ses-{int(sesnum):02d}"
        except ValueError:
            continue
        markers_dict[key] = np.array([row[m] for m in marker_names], dtype=float)

    print(
        f"   Loaded markers for {len(markers_dict)} subjects "
        f"({len(marker_names)} markers, reduction={reduction_str})",
        flush=True,
    )
    return markers_dict, marker_names


def load_patient_labels(labels_file):
    """Load CRS labels (VS vs MCS).

    Returns
    -------
    labels_dict : dict
        ``{subject_session_key: label_str}`` where label is ``"VS"`` or ``"MCS"``.
    """
    df = pd.read_csv(labels_file)
    df = df.dropna(subset=["subject", "session"])

    labels_dict = {}
    for _, row in df.iterrows():
        subject = row["subject"]
        session = f"ses-{int(row['session']):02d}"
        state = row["diagnostic_crs_final"]
        if pd.isna(state) or str(state).strip().lower() in ("n/a", ""):
            continue
        if state == "UWS":
            state = "VS"
        elif state in ["MCS+", "MCS-"]:
            state = "MCS"
        else:
            continue
        key = f"{subject}_{session}"
        labels_dict[key] = state
    return labels_dict


# ── DimensionResidualizer ────────────────────────────────────────────────────


class DimensionResidualizer(BaseEstimator, TransformerMixin):
    """Remove marker-predictable variance from each embedding dimension.

    Fits a multi-output Ridge regression from markers to embeddings::

        X ≈ Y_scaled @ W.T + b          (shape: (n, emb_dim))

    During ``transform``, the marker-predicted component is subtracted
    from each embedding dimension independently::

        X_res = X - ridge.predict(Y_scaled)

    This is the *inverse* of the direction used in
    ``residualization_embeddings.py``: here markers predict embeddings
    rather than embeddings predicting markers.

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength (used as default when inner CV is
        skipped or only one inner fold is available).
    alpha_grid : list of float
        Grid of alpha values for inner-CV model selection.
    n_inner_folds : int
        Number of inner GroupKFold splits for alpha selection.
    """

    def __init__(self, alpha=1.0, alpha_grid=None, n_inner_folds=3):
        self.alpha = alpha
        self.alpha_grid = alpha_grid or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.n_inner_folds = n_inner_folds

    def fit(self, Y, X, groups=None):
        """Fit multi-output Ridge: markers → all embedding dimensions.

        Parameters
        ----------
        Y : ndarray, shape (n, n_markers)
            Marker scalars for training subjects (NaN allowed — will be
            imputed column-wise with the training-set median).
        X : ndarray, shape (n, emb_dim)
            Embeddings for training subjects.
        groups : ndarray, shape (n,) or None
            Subject-level group IDs for inner GroupKFold.  If None, inner
            CV is skipped and ``self.alpha`` is used directly.
        """
        # ── Impute NaNs in markers (column median on training data) ──────
        self._col_medians = np.nanmedian(Y, axis=0)
        Y_imp = self._impute(Y)

        # ── Scale markers ────────────────────────────────────────────────
        self._scaler = StandardScaler()
        Y_scaled = self._scaler.fit_transform(Y_imp)

        if groups is None:
            groups = np.arange(len(Y))

        # ── Inner CV to select alpha ──────────────────────────────────────
        n_unique = len(np.unique(groups))
        n_inner = min(self.n_inner_folds, n_unique)

        best_alpha = self.alpha
        if n_inner >= 2:
            inner_cv = GroupKFold(n_splits=n_inner)
            best_score = -np.inf
            for a in self.alpha_grid:
                fold_scores = []
                for tr, va in inner_cv.split(Y_scaled, groups=groups):
                    ridge = Ridge(alpha=a)
                    ridge.fit(Y_scaled[tr], X[tr])
                    # Mean R² across all embedding dimensions
                    fold_scores.append(ridge.score(Y_scaled[va], X[va]))
                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = a

        # ── Refit on all training data ────────────────────────────────────
        self._ridge = Ridge(alpha=best_alpha)
        self._ridge.fit(Y_scaled, X)

        print(
            f"      [DimensionResidualizer] Fitted multi-output Ridge "
            f"(alpha={best_alpha}, markers→emb_dim={X.shape[1]})",
            flush=True,
        )
        return self

    def transform(self, Y, X):
        """Subtract marker-predicted component from embeddings.

        Parameters
        ----------
        Y : ndarray, shape (n, n_markers)
            Marker scalars (NaN allowed — imputed with training medians).
        X : ndarray, shape (n, emb_dim)
            Raw embeddings.

        Returns
        -------
        X_res : ndarray, shape (n, emb_dim)
            Residualized embeddings (marker-predicted variance removed).
        """
        Y_imp = self._impute(Y)
        Y_scaled = self._scaler.transform(Y_imp)
        X_pred = self._ridge.predict(Y_scaled)  # (n, emb_dim)
        return X - X_pred

    def _impute(self, Y):
        """Replace NaNs with column medians from training data."""
        Y_imp = Y.copy()
        for j in range(Y_imp.shape[1]):
            mask = np.isnan(Y_imp[:, j])
            if mask.any():
                Y_imp[mask, j] = self._col_medians[j]
        return Y_imp


# ── Classifier construction ───────────────────────────────────────────────────


def _safe_auc(y_true, y_score, y_pred):
    """AUC with balanced_accuracy fallback for single-class folds."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return balanced_accuracy_score(y_true, y_pred)


def _decision_scores(clf, clf_name, X):
    """Return a 1-D score array suitable for roc_auc_score."""
    if clf_name in ("mlp", "random_forest"):
        return clf.predict_proba(X)[:, 1]
    if clf_name == "svm":
        raw = clf.decision_function(X)
        if raw.ndim > 1:
            raw = raw[:, 1]
        return raw
    # kernel_ridge
    return clf.decision_function(X)


def _inner_cv_classifier(
    clf_name, X_train, y_train, groups_train, random_state, inner_splits=None
):
    """Run inner CV grid search and return the best fitted classifier.

    Uses pre-computed ``inner_splits`` (list of (train_idx, val_idx) relative
    to X_train) when provided, otherwise falls back to 3-fold GroupKFold.

    Returns the classifier fitted on the full training set with the best
    hyperparameters found during the inner CV.
    """
    if inner_splits is not None:
        cv_iter = inner_splits
    else:
        n_unique = len(np.unique(groups_train))
        n_inner = min(3, n_unique)
        inner_cv = GroupKFold(n_splits=n_inner)
        cv_iter = list(inner_cv.split(X_train, y_train, groups=groups_train))

    if clf_name == "kernel_ridge":
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha, best_score = 1.0, -np.inf
        for a in alphas:
            scores = []
            for tr, va in cv_iter:
                clf = KernelRidgeClassifier(alpha=a, kernel="rbf")
                clf.fit(X_train[tr], y_train[tr])
                raw = clf.decision_function(X_train[va])
                scores.append(_safe_auc(y_train[va], raw, clf.predict(X_train[va])))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_alpha = a
        final = KernelRidgeClassifier(alpha=best_alpha, kernel="rbf")
        final.fit(X_train, y_train)
        return final

    if clf_name == "svm":
        param_grid = list(product([0.01, 0.1, 1.0, 10.0], ["rbf", "linear"]))
        best_params, best_score = {"C": 1.0, "kernel": "rbf"}, -np.inf
        for C, kernel in param_grid:
            scores = []
            for tr, va in cv_iter:
                clf = SVC(
                    C=C,
                    kernel=kernel,
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                    max_iter=5000,
                )
                clf.fit(X_train[tr], y_train[tr])
                raw = clf.decision_function(X_train[va])
                scores.append(_safe_auc(y_train[va], raw, clf.predict(X_train[va])))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = {"C": C, "kernel": kernel}
        final = SVC(
            **best_params,
            class_weight="balanced",
            probability=True,
            random_state=random_state,
            max_iter=5000,
        )
        final.fit(X_train, y_train)
        return final

    if clf_name == "mlp":
        configs = [(128,), (256,), (256, 128), (512, 256)]
        best_cfg, best_score = (256, 128), -np.inf
        for hidden in configs:
            scores = []
            for tr, va in cv_iter:
                clf = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=300,
                    random_state=random_state,
                    early_stopping=True,
                    n_iter_no_change=20,
                )
                clf.fit(X_train[tr], y_train[tr])
                proba = clf.predict_proba(X_train[va])[:, 1]
                scores.append(_safe_auc(y_train[va], proba, clf.predict(X_train[va])))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_cfg = hidden
        final = MLPClassifier(
            hidden_layer_sizes=best_cfg,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=20,
        )
        final.fit(X_train, y_train)
        return final

    if clf_name == "random_forest":
        param_grid = list(product([100, 200, 500], [None, 5, 10]))
        best_params, best_score = {"n_estimators": 200, "max_depth": None}, -np.inf
        for n_est, max_d in param_grid:
            scores = []
            for tr, va in cv_iter:
                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    class_weight="balanced",
                    random_state=random_state,
                )
                clf.fit(X_train[tr], y_train[tr])
                proba = clf.predict_proba(X_train[va])[:, 1]
                scores.append(_safe_auc(y_train[va], proba, clf.predict(X_train[va])))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = {"n_estimators": n_est, "max_depth": max_d}
        final = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            random_state=random_state,
        )
        final.fit(X_train, y_train)
        return final

    raise ValueError(f"Unknown classifier: {clf_name!r}")


# ── R² per embedding dimension ───────────────────────────────────────────────


def compute_r2_per_dim(Y, X, groups=None):
    """Fit DimensionResidualizer on all data and return per-dimension R².

    This is a global fit (not per-fold) intended purely for visualization:
    it shows which embedding dimensions are linearly predictable from markers.

    Parameters
    ----------
    Y : ndarray, shape (n, n_markers)
    X : ndarray, shape (n, emb_dim)
    groups : ndarray, shape (n,) or None

    Returns
    -------
    r2_values : ndarray, shape (emb_dim,)
        R² for each embedding dimension.
    """
    residualizer = DimensionResidualizer()
    residualizer.fit(Y, X, groups=groups)
    Y_imp = residualizer._impute(Y)
    Y_scaled = residualizer._scaler.transform(Y_imp)
    X_pred = residualizer._ridge.predict(Y_scaled)  # (n, emb_dim)
    r2_values = np.array(
        [r2_score(X[:, i], X_pred[:, i]) for i in range(X.shape[1])]
    )
    return r2_values


def plot_r2_per_dim(r2_values, model_name, output_path):
    """Plot R² per embedding dimension (one curve per FM model plot).

    Parameters
    ----------
    r2_values : ndarray, shape (emb_dim,)
    model_name : str
    output_path : str
        Full path for the output PNG.
    """
    emb_dim = len(r2_values)
    x = np.arange(emb_dim)

    # Sort dimensions by R² to reveal structure
    sort_idx = np.argsort(r2_values)
    r2_sorted = r2_values[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    color = _MODEL_COLORS.get(model_name, "#1f77b4")

    # Left: by index order
    axes[0].plot(x, r2_values, linewidth=0.8, color=color, alpha=0.85)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("Embedding dimension index", fontsize=12)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("By dimension index", fontsize=11)
    axes[0].grid(True, alpha=0.25)

    # Right: sorted ascending by R²
    axes[1].plot(np.arange(emb_dim), r2_sorted, linewidth=0.8, color=color, alpha=0.85)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Dimension rank (sorted by R²)", fontsize=12)
    axes[1].set_ylabel("R²", fontsize=12)
    axes[1].set_title("Sorted ascending", fontsize=11)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"{model_name} — R² per embedding dimension\n"
        f"(Ridge regression: markers → embedding, {emb_dim} dims)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


# ── Nested CV pipeline ───────────────────────────────────────────────────────


def run_nested_cv(
    X,
    Y,
    y_cls,
    subjects,
    marker_names,
    classifiers,
    n_folds,
    random_state,
    precomputed_splits=None,
):
    """Nested CV with dimension-wise residualization.

    Outer loop: pre-computed StratifiedGroupKFold folds (from
    ``precomputed_splits``) or freshly generated StratifiedGroupKFold when
    ``precomputed_splits`` is None.

    For each outer fold:

    1. Fit ``DimensionResidualizer`` on ``(Y_train, X_train)`` — learns
       how to predict each embedding dimension from markers.
    2. Residualize: ``X_train_res = X_train - ridge.predict(Y_train_scaled)``,
       ``X_test_res = X_test - ridge.predict(Y_test_scaled)``.
       Both train and test use their *own* markers, not the training set's
       predicted values — the scaler/Ridge parameters come from training only.
    3. Inner loop: hyperparameter search for classifier using pre-computed
       inner splits when available, else 3-fold GroupKFold.
    4. Evaluate AUC on ``X_test_res``.

    Parameters
    ----------
    X : ndarray (n, emb_dim)
        Raw embeddings.
    Y : ndarray (n, n_markers)
        Marker scalars (NaN allowed).
    y_cls : ndarray (n,)
        Integer class labels {0, 1}.
    subjects : list of str
        Subject-session keys (used for group extraction).
    marker_names : list of str
    classifiers : list of str
        Classifier names to evaluate.
    n_folds : int
        Number of outer CV folds (used when ``precomputed_splits`` is None).
    random_state : int
    precomputed_splits : list of fold dicts or None
        Pre-computed folds from :func:`generate_nested_cv_folds` /
        :func:`load_cv_splits`.  When provided, ``n_folds`` and
        ``random_state`` are ignored for the outer loop.

    Returns
    -------
    results : dict
        ``{clf_name: {"auc_per_fold": [...], "mean_auc": float,
                       "std_auc": float, "per_fold_details": [...]}}``
    """
    groups = np.array([s.split("_ses-")[0] for s in subjects])

    # ── Build fold iterator ───────────────────────────────────────────────────
    if precomputed_splits is not None:
        folds_iter = precomputed_splits
        n_folds_actual = len(folds_iter)
    else:
        n_unique_groups = len(np.unique(groups))
        effective_folds = min(n_folds, n_unique_groups)
        outer_cv = StratifiedGroupKFold(
            n_splits=effective_folds, shuffle=True, random_state=random_state
        )
        folds_iter = [
            {"train_idx": tr, "test_idx": te, "inner_splits": None}
            for tr, te in outer_cv.split(X, y_cls, groups=groups)
        ]
        n_folds_actual = effective_folds

    # ── Leakage sanity checks ─────────────────────────────────────────────────
    for fold_idx, fold in enumerate(folds_iter):
        check_no_subject_leakage(
            groups, fold["train_idx"], fold["test_idx"], label=f"outer fold {fold_idx}"
        )

    fold_aucs = {clf_name: [] for clf_name in classifiers}
    fold_details = {clf_name: [] for clf_name in classifiers}

    for fold_idx, fold in enumerate(folds_iter):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        inner_splits = fold.get("inner_splits")

        print(
            f"   Outer fold {fold_idx + 1}/{n_folds_actual}: "
            f"train={len(train_idx)}, test={len(test_idx)}",
            flush=True,
        )

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        y_train, y_test = y_cls[train_idx], y_cls[test_idx]
        groups_train = groups[train_idx]

        # ── Step 1: fit residualizer on training data ──────────────────────
        residualizer = DimensionResidualizer()
        residualizer.fit(Y_train, X_train_raw, groups=groups_train)

        # ── Step 2: residualize (each split uses its own markers) ──────────
        X_train_res = residualizer.transform(Y_train, X_train_raw)
        X_test_res = residualizer.transform(Y_test, X_test_raw)

        # ── Step 3 & 4: fit and evaluate classifiers ───────────────────────
        for clf_name in classifiers:
            try:
                clf = _inner_cv_classifier(
                    clf_name,
                    X_train_res,
                    y_train,
                    groups_train,
                    random_state,
                    inner_splits=inner_splits,
                )
                scores = _decision_scores(clf, clf_name, X_test_res)
                preds = clf.predict(X_test_res)
                auc_val = _safe_auc(y_test, scores, preds)
            except Exception as exc:
                print(
                    f"      [WARN] {clf_name} fold {fold_idx + 1} failed: {exc}",
                    flush=True,
                )
                auc_val = float("nan")
                preds = np.zeros(len(y_test), dtype=int)

            fold_aucs[clf_name].append(float(auc_val))
            fold_details[clf_name].append(
                {
                    "fold": fold_idx + 1,
                    "auc": float(auc_val),
                    "train_subjects": [subjects[i] for i in train_idx],
                    "test_subjects": [subjects[i] for i in test_idx],
                    "n_train_VS": int(np.sum(y_train == 0)),
                    "n_train_MCS": int(np.sum(y_train == 1)),
                    "n_test_VS": int(np.sum(y_test == 0)),
                    "n_test_MCS": int(np.sum(y_test == 1)),
                }
            )
            print(
                f"      [{clf_name}] AUC = {auc_val:.3f}",
                flush=True,
            )

    results = {}
    for clf_name in classifiers:
        valid_aucs = [a for a in fold_aucs[clf_name] if not np.isnan(a)]
        results[clf_name] = {
            "auc_per_fold": fold_aucs[clf_name],
            "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
            "std_auc": float(np.std(valid_aucs)) if len(valid_aucs) > 1 else 0.0,
            "n_valid_folds": len(valid_aucs),
            "per_fold_details": fold_details[clf_name],
        }

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_auc_comparison(all_results, output_path, classifiers):
    """Plot mean AUC ± std per FM model.

    Parameters
    ----------
    all_results : dict
        ``{model_name: {clf_name: {"mean_auc": float, "std_auc": float}}}``
    output_path : str
        Full path for the output PNG.
    classifiers : list of str
    """
    model_names = [m for m in FM_MODELS if m in all_results]
    if not model_names:
        model_names = sorted(all_results.keys())

    n_clf = len(classifiers)
    x = np.arange(len(model_names))
    width = 0.7 / max(n_clf, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.8), 5))

    for ci, clf_name in enumerate(classifiers):
        means = []
        stds = []
        for model_name in model_names:
            clf_res = all_results.get(model_name, {}).get(clf_name, {})
            means.append(clf_res.get("mean_auc", float("nan")))
            stds.append(clf_res.get("std_auc", 0.0))

        offset = (ci - (n_clf - 1) / 2) * width
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt=_CLASSIFIER_MARKERS.get(clf_name, "o"),
            label=clf_name,
            capsize=4,
            markersize=8,
            linewidth=1.5,
        )

    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC (mean ± std)", fontsize=12)
    ax.set_xlabel("Foundation Model", fontsize=12)
    ax.set_title(
        "VS vs MCS classification after dimension-wise marker residualization\n"
        "(nested CV, last-layer embeddings)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_model(
    model_name,
    results_root,
    marker_csv,
    patient_labels_file,
    output_dir,
    reduction,
    classifiers,
    n_folds,
    random_state,
    precomputed_splits=None,
    common_sessions_ref=None,
):
    """Run the full dimension residualization + classification pipeline for one FM model.

    Parameters
    ----------
    precomputed_splits : list of fold dicts or None
        Pre-computed CV splits to use.  When provided together with
        ``common_sessions_ref``, data is aligned to that session list and
        splits indices are used directly (no new folds are generated).
    common_sessions_ref : list of str or None
        Ordered session list from the splits file.  Acts as the reference
        for X/Y/y_cls array alignment so fold indices are valid.

    Returns
    -------
    model_results : dict or None
    subjects_info : dict or None
    splits_used : list of fold dicts
        The folds actually used (pre-computed or freshly generated).
    common_keys_used : list of str
        Session list corresponding to the returned splits.
    labels_for_splits : dict
        ``{session_key: int_label}`` for ``common_keys_used``.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Foundation model: {model_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("  Loading embeddings ...", flush=True)
    embeddings = load_embeddings(results_root, model_name)
    if not embeddings:
        print(f"  [SKIP] No embeddings found for {model_name}.", flush=True)
        return None, None, None, None, None

    print("  Loading markers ...", flush=True)
    markers_dict, marker_names = load_markers(marker_csv, reduction)

    print("  Loading patient labels ...", flush=True)
    labels_dict = load_patient_labels(patient_labels_file)

    # ── Align subjects ────────────────────────────────────────────────────
    if precomputed_splits is not None and common_sessions_ref is not None:
        available = set(embeddings) & set(markers_dict) & set(labels_dict)
        missing = [s for s in common_sessions_ref if s not in available]
        if missing:
            print(
                f"  [WARN] {len(missing)} sessions from splits not available "
                f"for {model_name} — regenerating splits for this model.",
                flush=True,
            )
            precomputed_splits = None
            common_keys = sorted(available)
        else:
            common_keys = list(common_sessions_ref)
    else:
        common_keys = sorted(set(embeddings) & set(markers_dict) & set(labels_dict))

    if len(common_keys) < 10:
        print(
            f"  [SKIP] Only {len(common_keys)} aligned subjects — need at least 10.",
            flush=True,
        )
        return None, None, None, None, None

    class_counts = {}
    for k in common_keys:
        lbl = labels_dict[k]
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    if len(class_counts) < 2:
        print(
            f"  [SKIP] Only one class present: {class_counts}",
            flush=True,
        )
        return None, None, None, None, None

    X = np.stack([embeddings[k] for k in common_keys])       # (n, emb_dim)
    Y = np.stack([markers_dict[k] for k in common_keys])     # (n, n_markers)
    label_strings = [labels_dict[k] for k in common_keys]

    le = LabelEncoder()
    y_cls = le.fit_transform(label_strings)
    class_names = list(le.classes_)

    print(
        f"  {len(common_keys)} subjects, classes: "
        f"{dict(zip(class_names, np.bincount(y_cls)))}",
        flush=True,
    )

    subjects_info = {
        "n_subjects": len(common_keys),
        "subject_keys": common_keys,
        "class_names": class_names,
        "class_counts": {c: int(n) for c, n in zip(class_names, np.bincount(y_cls))},
    }

    # ── Generate splits if not provided ───────────────────────────────────
    labels_for_splits = {s: int(y_cls[i]) for i, s in enumerate(common_keys)}
    if precomputed_splits is None:
        precomputed_splits = generate_nested_cv_folds(
            common_sessions=common_keys,
            labels=labels_for_splits,
            n_outer=n_folds,
            random_state=random_state,
        )
        print(
            f"  Generated {len(precomputed_splits)} outer folds "
            f"(StratifiedGroupKFold, seed={random_state})",
            flush=True,
        )

    # ── Nested CV ─────────────────────────────────────────────────────────
    print("  Running nested CV ...", flush=True)
    model_results = run_nested_cv(
        X=X,
        Y=Y,
        y_cls=y_cls,
        subjects=common_keys,
        marker_names=marker_names,
        classifiers=classifiers,
        n_folds=n_folds,
        random_state=random_state,
        precomputed_splits=precomputed_splits,
    )

    for clf_name, res in model_results.items():
        print(
            f"  [{clf_name}] AUC = {res['mean_auc']:.3f} ± {res['std_auc']:.3f}"
            f" ({res['n_valid_folds']} folds)",
            flush=True,
        )

    # ── R² per embedding dimension (global fit for visualization) ─────────
    print("  Computing R² per embedding dimension ...", flush=True)
    groups_all = np.array([s.split("_ses-")[0] for s in common_keys])
    try:
        r2_values = compute_r2_per_dim(Y, X, groups=groups_all)
        r2_plot_path = op.join(output_dir, f"r2_per_dim_{model_name}.png")
        os.makedirs(output_dir, exist_ok=True)
        plot_r2_per_dim(r2_values, model_name, r2_plot_path)
        # Also save the R² values as npz for later inspection
        np.savez(
            op.join(output_dir, f"r2_per_dim_{model_name}.npz"),
            r2_per_dim=r2_values,
        )
    except Exception as exc:
        print(f"  [WARN] R² per dim plot failed: {exc}", flush=True)

    return model_results, subjects_info, precomputed_splits, common_keys, labels_for_splits


def save_results(all_results, subjects_per_model, output_dir, reduction, n_folds):
    """Save JSON and CSV summaries of all results.

    Parameters
    ----------
    all_results : dict
        ``{model_name: {clf_name: {"mean_auc", "std_auc", ...}}}``
    subjects_per_model : dict
        ``{model_name: subjects_info}``
    output_dir : str
    reduction : str
    n_folds : int
    """
    os.makedirs(output_dir, exist_ok=True)

    json_payload = {
        "method": "dimension_residualization",
        "reduction": reduction,
        "n_folds": n_folds,
        "results": {},
    }
    for model_name, clf_results in all_results.items():
        json_payload["results"][model_name] = {
            "subjects_info": subjects_per_model.get(model_name, {}),
            "classifiers": {},
        }
        for clf_name, res in clf_results.items():
            json_payload["results"][model_name]["classifiers"][clf_name] = {
                "mean_auc": res["mean_auc"],
                "std_auc": res["std_auc"],
                "n_valid_folds": res["n_valid_folds"],
                "auc_per_fold": res["auc_per_fold"],
                "per_fold_details": res["per_fold_details"],
            }

    json_path = op.join(output_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"   Saved: {json_path}", flush=True)

    rows = []
    for model_name, clf_results in all_results.items():
        sub_info = subjects_per_model.get(model_name, {})
        n_subjects = sub_info.get("n_subjects", "?")
        class_counts = sub_info.get("class_counts", {})
        for clf_name, res in clf_results.items():
            rows.append(
                {
                    "model": model_name,
                    "classifier": clf_name,
                    "mean_auc": round(res["mean_auc"], 4),
                    "std_auc": round(res["std_auc"], 4),
                    "n_folds_valid": res["n_valid_folds"],
                    "n_subjects": n_subjects,
                    "n_VS": class_counts.get("VS", "?"),
                    "n_MCS": class_counts.get("MCS", "?"),
                    "reduction": reduction,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = op.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Residualize FM embeddings dimension-wise w.r.t. EEG markers "
            "(markers → each embedding dim), then classify VS/MCS."
        )
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory with per-model benchmark results.",
    )
    parser.add_argument(
        "--marker-csv",
        default=DEFAULT_MARKER_CSV,
        help="Path to nice_scalars_all.csv (or similar) with marker scalars.",
    )
    parser.add_argument(
        "--patient-labels",
        default=DEFAULT_PATIENT_LABELS,
        help="Path to metadata_patient_labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/residualization_dim",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--model",
        choices=FM_MODELS + ["all"],
        default="all",
        help="Foundation model to process (default: all).",
    )
    parser.add_argument(
        "--reduction",
        choices=list(REDUCTION_MAP.keys()),
        default="A",
        help="Marker reduction variant (default: A).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=["kernel_ridge", "svm", "mlp", "random_forest"],
        default=["kernel_ridge", "svm", "mlp", "random_forest"],
        help="Classifiers to evaluate (default: all four).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (same across all FM models for comparability).",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Path to a pre-computed CV splits JSON (from generate_nested_cv_folds). "
            "When provided, the same outer/inner folds are used for all models — "
            "enabling direct comparison with residualization_embeddings.py results."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help=(
            "Path to save the CV splits JSON when generating new splits. "
            "If omitted, splits are auto-saved to "
            "{output_dir}/cv_splits/residualization_dim_crs_{timestamp}.json."
        ),
    )
    args = parser.parse_args()

    models_to_run = FM_MODELS if args.model == "all" else [args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load or prepare splits ────────────────────────────────────────────────
    precomputed_splits = None
    common_sessions_ref = None
    labels_ref = None

    if args.splits_file and op.isfile(args.splits_file):
        precomputed_splits, common_sessions_ref, labels_ref = load_cv_splits(
            args.splits_file
        )
        print(
            f"Loaded {len(precomputed_splits)} pre-computed outer folds "
            f"({len(common_sessions_ref)} sessions) from {args.splits_file}",
            flush=True,
        )

    all_results = {}
    subjects_per_model = {}
    _splits_saved = precomputed_splits is not None

    for model_name in models_to_run:
        result = run_model(
            model_name=model_name,
            results_root=args.results_root,
            marker_csv=args.marker_csv,
            patient_labels_file=args.patient_labels,
            output_dir=op.join(args.output_dir, model_name),
            reduction=args.reduction,
            classifiers=args.classifiers,
            n_folds=args.n_folds,
            random_state=args.random_state,
            precomputed_splits=precomputed_splits,
            common_sessions_ref=common_sessions_ref,
        )
        model_results, subjects_info, splits_used, common_used, labels_used = result
        if model_results is not None:
            all_results[model_name] = model_results
            subjects_per_model[model_name] = subjects_info or {}

            if not _splits_saved and splits_used is not None:
                precomputed_splits = splits_used
                common_sessions_ref = common_used
                labels_ref = labels_used
                save_path = args.save_splits_to
                if not save_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    splits_dir = op.join(args.output_dir, "cv_splits")
                    save_path = op.join(
                        splits_dir, f"residualization_dim_crs_{ts}.json"
                    )
                save_cv_splits(splits_used, common_used, labels_used, save_path)
                print(f"  CV splits saved to {save_path}", flush=True)
                _splits_saved = True

    if not all_results:
        print("\nNo results to save — all models were skipped.", flush=True)
        return

    print("\nSaving results ...", flush=True)
    save_results(
        all_results=all_results,
        subjects_per_model=subjects_per_model,
        output_dir=args.output_dir,
        reduction=args.reduction,
        n_folds=args.n_folds,
    )

    print("Generating AUC comparison plot ...", flush=True)
    plot_path = op.join(args.output_dir, "auc_comparison.png")
    plot_auc_comparison(
        all_results=all_results,
        output_path=plot_path,
        classifiers=args.classifiers,
    )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
