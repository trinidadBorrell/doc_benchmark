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

try:
    from xgboost import XGBClassifier as _XGBClassifier
except ImportError:
    _XGBClassifier = None

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
        filter_excluded_markers,
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
        filter_excluded_markers,
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

FM_MODELS = ["LaBram", "CBraMod", "NeuroLM", "TOTEM", "BIOT", "EEGPT"]

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

# All (target, label_mode, subset, output_subdir, title_suffix) combinations.
# subset is 'VS', 'MCS', or None. label_mode is only relevant for cs_* targets.
TARGET_CONFIGS = [
    # ── CRS ──────────────────────────────────────────────────────────────
    ("crs",           None,                      None,  "crs",                            "VS vs MCS"),
    # ── Etiology ─────────────────────────────────────────────────────────
    ("etiology",      None,                      None,  "etiology/all",                   "Etiology (all)"),
    ("etiology",      None,                      "VS",  "etiology/vs_only",               "Etiology (VS baseline)"),
    ("etiology",      None,                      "MCS", "etiology/mcs_only",              "Etiology (MCS baseline)"),
    ("etiology_code", None,                      None,  "etiology_code/all",              "Etiology Code (all)"),
    ("etiology_code", None,                      "VS",  "etiology_code/vs_only",          "Etiology Code (VS baseline)"),
    ("etiology_code", None,                      "MCS", "etiology_code/mcs_only",         "Etiology Code (MCS baseline)"),
    # ── cs_6m ────────────────────────────────────────────────────────────
    ("cs_6m", "multiclass",              None, "cs_6m/multiclass",               "6m Outcome (4-class)"),
    ("cs_6m", "binary",                  None, "cs_6m/binary",                   "6m Outcome (VS vs MCS)"),
    ("cs_6m", "binary_death",            None, "cs_6m/binary_death",             "6m Outcome (DEATH vs non)"),
    ("cs_6m", "binary_vs_to_mcs",        None, "cs_6m/binary_vs_to_mcs",         "6m Outcome (VS→MCS)"),
    ("cs_6m", "binary_mcs_to_conscious", None, "cs_6m/binary_mcs_to_conscious",  "6m Outcome (MCS→CONSCIOUS)"),
    ("cs_6m", "binary_improvement",      None, "cs_6m/binary_improvement",       "6m Outcome (IMPROVED)"),
    # ── cs_1y ────────────────────────────────────────────────────────────
    ("cs_1y", "multiclass",              None, "cs_1y/multiclass",               "1y Outcome (4-class)"),
    ("cs_1y", "binary",                  None, "cs_1y/binary",                   "1y Outcome (VS vs MCS)"),
    ("cs_1y", "binary_death",            None, "cs_1y/binary_death",             "1y Outcome (DEATH vs non)"),
    ("cs_1y", "binary_vs_to_mcs",        None, "cs_1y/binary_vs_to_mcs",         "1y Outcome (VS→MCS)"),
    ("cs_1y", "binary_mcs_to_conscious", None, "cs_1y/binary_mcs_to_conscious",  "1y Outcome (MCS→CONSCIOUS)"),
    ("cs_1y", "binary_improvement",      None, "cs_1y/binary_improvement",       "1y Outcome (IMPROVED)"),
    # ── cs_2y ────────────────────────────────────────────────────────────
    ("cs_2y", "multiclass",              None, "cs_2y/multiclass",               "2y Outcome (4-class)"),
    ("cs_2y", "binary",                  None, "cs_2y/binary",                   "2y Outcome (VS vs MCS)"),
    ("cs_2y", "binary_death",            None, "cs_2y/binary_death",             "2y Outcome (DEATH vs non)"),
    ("cs_2y", "binary_vs_to_mcs",        None, "cs_2y/binary_vs_to_mcs",         "2y Outcome (VS→MCS)"),
    ("cs_2y", "binary_mcs_to_conscious", None, "cs_2y/binary_mcs_to_conscious",  "2y Outcome (MCS→CONSCIOUS)"),
    ("cs_2y", "binary_improvement",      None, "cs_2y/binary_improvement",       "2y Outcome (IMPROVED)"),
]

_MODEL_COLORS = {
    "CBraMod": "#2ca02c",
    "CbraMod": "#2ca02c",
    "NeuroLM": "#1f77b4",
    "TOTEM": "#ff7f0e",
    "LaBram": "#d62728",
    "BIOT": "#9467bd",
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

    marker_names, markers_dict, n_dropped = filter_excluded_markers(
        marker_names, markers_dict
    )
    if n_dropped:
        print(
            f"   Excluded {n_dropped} paradigm-locked markers "
            f"(TimeLockedContrast/WindowDecoding) → {len(marker_names)} markers remain",
            flush=True,
        )

    print(
        f"   Loaded markers for {len(markers_dict)} subjects "
        f"({len(marker_names)} markers, reduction={reduction_str})",
        flush=True,
    )
    return markers_dict, marker_names


def load_labels_for_target(labels_file, target, label_mode=None, subset=None):
    """Load classification labels for the specified target and mode.

    Parameters
    ----------
    labels_file : str
        Path to metadata_patient_labels.csv.
    target : str
        One of: ``'crs'``, ``'etiology'``, ``'etiology_code'``,
        ``'cs_6m'``, ``'cs_1y'``, ``'cs_2y'``.
    label_mode : str or None
        For ``cs_*`` targets: ``'multiclass'``, ``'binary'``,
        ``'binary_death'``, ``'binary_vs_to_mcs'``,
        ``'binary_mcs_to_conscious'``, ``'binary_improvement'``.
        Ignored for other targets.
    subset : str or None
        For ``etiology`` / ``etiology_code``: ``None`` (all subjects),
        ``'VS'`` (UWS-baseline only), or ``'MCS'`` (MCS+/MCS- baseline only).

    Returns
    -------
    labels_dict : dict
        ``{subject_session_key: label_str}``
    is_multiclass : bool
        True only when ``label_mode == 'multiclass'``.
    """
    _BL_VS = {"UWS"}
    _BL_MCS = {"MCS+", "MCS-"}
    _VS_STATES = {"VS", "VS/MCS"}
    _MCS_STATES = {"MCS+", "MCS-", "MCS"}

    df = pd.read_csv(labels_file)
    df = df.dropna(subset=["subject", "session"])

    labels_dict = {}
    is_multiclass = label_mode == "multiclass"

    for _, row in df.iterrows():
        subject = row["subject"]
        session = f"ses-{int(row['session']):02d}"
        key = f"{subject}_{session}"

        if target == "crs":
            state = row["diagnostic_crs_final"]
            if pd.isna(state) or str(state).strip().lower() in ("n/a", ""):
                continue
            if state == "UWS":
                state = "VS"
            elif state in ("MCS+", "MCS-"):
                state = "MCS"
            else:
                continue
            labels_dict[key] = state

        elif target in ("etiology", "etiology_code"):
            col = (
                "etiology_medical_condition" if target == "etiology" else "etiology_code"
            )
            val = row[col]
            if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                continue
            if subset is not None:
                bl = row.get("diagnostic_crs_final", None)
                if subset == "VS" and (pd.isna(bl) or str(bl).strip() not in _BL_VS):
                    continue
                if subset == "MCS" and (pd.isna(bl) or str(bl).strip() not in _BL_MCS):
                    continue
            if target == "etiology":
                label = str(val).strip().lower()
            else:
                label = str(val).strip().lower()
                if label == "anoxia":
                    label = "ANOXIA"
                elif label == "tbi":
                    label = "TBI"
                else:
                    continue
            labels_dict[key] = label

        elif target in ("cs_6m", "cs_1y", "cs_2y"):
            val = row[target]
            if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                continue
            label = str(val).strip()
            bl_raw = row.get("diagnostic_crs_final", None)
            bl_str = str(bl_raw).strip() if not pd.isna(bl_raw) else ""

            if label_mode == "multiclass":
                if label in _VS_STATES:
                    label = "VS"
                elif label in _MCS_STATES:
                    label = "MCS"
                elif label in ("CONSCIOUS", "DEATH"):
                    pass
                else:
                    continue

            elif label_mode == "binary":
                if label in _VS_STATES:
                    label = "VS"
                elif label in _MCS_STATES:
                    label = "MCS"
                else:
                    continue

            elif label_mode == "binary_death":
                if label == "DEATH":
                    label = "DEATH"
                elif label in _VS_STATES or label in _MCS_STATES or label == "CONSCIOUS":
                    label = "NON_DEATH"
                else:
                    continue

            elif label_mode == "binary_vs_to_mcs":
                if bl_str != "UWS":
                    continue
                if label in _MCS_STATES or label == "CONSCIOUS":
                    label = "IMPROVED"
                else:
                    label = "OTHER"

            elif label_mode == "binary_mcs_to_conscious":
                if bl_str not in ("MCS+", "MCS-"):
                    continue
                if label == "CONSCIOUS":
                    label = "IMPROVED"
                else:
                    label = "OTHER"

            elif label_mode == "binary_improvement":
                if bl_str == "UWS":
                    label = "IMPROVED" if label in _MCS_STATES or label == "CONSCIOUS" else "NON_IMPROVED"
                elif bl_str in ("MCS+", "MCS-"):
                    label = "IMPROVED" if label == "CONSCIOUS" else "NON_IMPROVED"
                else:
                    continue
            else:
                continue

            labels_dict[key] = label

    mode_str = f", mode={label_mode!r}" if label_mode else ""
    sub_str = f", subset={subset!r}" if subset else ""
    print(
        f"   Loaded {len(labels_dict)} sessions "
        f"(target={target!r}{mode_str}{sub_str})",
        flush=True,
    )
    return labels_dict, is_multiclass


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


def _safe_auc(y_true, y_score, y_pred, is_multiclass=False):
    """AUC with balanced_accuracy fallback.

    For multiclass targets uses balanced_accuracy directly (avoids
    the complexity of multiclass AUC scoring).
    """
    if is_multiclass:
        return balanced_accuracy_score(y_true, y_pred)
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return balanced_accuracy_score(y_true, y_pred)


def _decision_scores(clf, clf_name, X):
    """Return a 1-D score array suitable for roc_auc_score.

    For multiclass targets the caller uses balanced_accuracy_score and ignores
    this score, but we still return something to avoid exceptions.
    """
    if clf_name in ("mlp", "random_forest", "xgboost"):
        proba = clf.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]  # ignored for multiclass; caller uses preds
    if clf_name == "svm":
        raw = clf.decision_function(X)
        if raw.ndim > 1:
            return raw[:, 0]  # ignored for multiclass
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

    if clf_name == "xgboost":
        if _XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost>=2.0")
        param_grid = list(product([100, 300], [3, 6], [0.05, 0.1]))
        best_params, best_score = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}, -np.inf
        for n_est, max_d, lr in param_grid:
            scores = []
            for tr, va in cv_iter:
                clf = _XGBClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                )
                clf.fit(X_train[tr], y_train[tr])
                proba = clf.predict_proba(X_train[va])[:, 1]
                scores.append(_safe_auc(y_train[va], proba, clf.predict(X_train[va])))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}
        final = _XGBClassifier(
            **best_params,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
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
    is_multiclass=False,
    pca=False,
    pca_components=27,
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
    4. Evaluate AUC on ``X_test_res`` (or balanced accuracy for multiclass).

    Parameters
    ----------
    X : ndarray (n, emb_dim)
        Raw embeddings.
    Y : ndarray (n, n_markers)
        Marker scalars (NaN allowed).
    y_cls : ndarray (n,)
        Integer class labels.
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
    is_multiclass : bool
        When True, use balanced_accuracy_score as the metric instead of AUC.

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

        # ── Optional: per-fold train-only PCA on residualized embeddings ──
        pca_n_comp = None
        pca_var_exp = None
        if pca:
            from sklearn.decomposition import PCA as _PCA
            n_train = X_train_res.shape[0]
            n_feat = X_train_res.shape[1]
            pca_n_comp = int(min(pca_components, max(n_train - 1, 1), n_feat))
            _pca = _PCA(n_components=pca_n_comp, random_state=random_state)
            X_train_res = _pca.fit_transform(X_train_res)
            X_test_res = _pca.transform(X_test_res)
            pca_var_exp = float(_pca.explained_variance_ratio_.sum())

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
                auc_val = _safe_auc(y_test, scores, preds, is_multiclass=is_multiclass)
            except Exception as exc:
                print(
                    f"      [WARN] {clf_name} fold {fold_idx + 1} failed: {exc}",
                    flush=True,
                )
                auc_val = float("nan")
                preds = np.zeros(len(y_test), dtype=int)

            metric_name = "Bal.Acc" if is_multiclass else "AUC"
            fold_aucs[clf_name].append(float(auc_val))
            fold_detail = {
                "fold": fold_idx + 1,
                "auc": float(auc_val),
                "train_subjects": [subjects[i] for i in train_idx],
                "test_subjects": [subjects[i] for i in test_idx],
                "class_counts_train": {
                    str(c): int(np.sum(y_train == c))
                    for c in np.unique(y_train)
                },
                "class_counts_test": {
                    str(c): int(np.sum(y_test == c))
                    for c in np.unique(y_test)
                },
            }
            if pca:
                fold_detail["n_pca_components"] = pca_n_comp
                fold_detail["pca_explained_variance"] = pca_var_exp
            fold_details[clf_name].append(fold_detail)
            print(
                f"      [{clf_name}] {metric_name} = {auc_val:.3f}",
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


def plot_auc_comparison(
    all_results, output_path, classifiers, title_suffix="Classification", is_multiclass=False
):
    """Plot mean AUC ± std per FM model.

    Parameters
    ----------
    all_results : dict
        ``{model_name: {clf_name: {"mean_auc": float, "std_auc": float}}}``
    output_path : str
        Full path for the output PNG.
    classifiers : list of str
    title_suffix : str
        Appended to the plot title to identify the target/mode.
    is_multiclass : bool
        If True, label the y-axis as balanced accuracy and set chance at 1/n_classes.
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

    chance = 0.25 if is_multiclass else 0.5
    ax.axhline(
        chance, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    metric_label = "Balanced Accuracy (mean ± std)" if is_multiclass else "AUC (mean ± std)"
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Foundation Model", fontsize=12)
    ax.set_title(
        f"{title_suffix} — dimension-wise marker residualization\n"
        "(nested CV, last-layer embeddings)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


def plot_summary_heatmap(summary_data, output_path, best_clf="svm"):
    """Plot a heatmap of best-classifier mean AUC across all targets × FM models.

    Parameters
    ----------
    summary_data : list of dict
        Each dict: ``{"subdir": str, "title": str, "model": str,
        "clf": str, "mean_auc": float}``.
    output_path : str
    best_clf : str
        Which classifier column to use for the heatmap (default: "svm").
    """
    # Collect all targets and models in display order
    targets_seen = []
    models_seen = []
    data_map = {}  # (title, model) -> mean_auc
    for row in summary_data:
        title = row["title"]
        model = row["model"]
        if title not in targets_seen:
            targets_seen.append(title)
        if model not in models_seen and model in FM_MODELS:
            models_seen.append(model)
        data_map[(title, model)] = max(
            data_map.get((title, model), float("nan")), row["mean_auc"]
        )

    # Respect FM_MODELS ordering
    models_seen = [m for m in FM_MODELS if m in models_seen]

    if not targets_seen or not models_seen:
        return

    matrix = np.full((len(targets_seen), len(models_seen)), np.nan)
    for ti, title in enumerate(targets_seen):
        for mi, model in enumerate(models_seen):
            val = data_map.get((title, model), np.nan)
            matrix[ti, mi] = val

    fig_h = max(6, len(targets_seen) * 0.45)
    fig_w = max(5, len(models_seen) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Mean AUC / Bal.Acc")

    ax.set_xticks(np.arange(len(models_seen)))
    ax.set_xticklabels(models_seen, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(targets_seen)))
    ax.set_yticklabels(targets_seen, fontsize=8)

    for ti in range(len(targets_seen)):
        for mi in range(len(models_seen)):
            val = matrix[ti, mi]
            if not np.isnan(val):
                ax.text(mi, ti, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.45 < val < 0.9 else "white")

    ax.set_title(
        "Dim-wise residualization — best-classifier mean score\n(all targets × all FM models)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_model(
    model_name,
    results_root,
    marker_csv,
    labels_dict,
    is_multiclass,
    output_dir,
    reduction,
    classifiers,
    n_folds,
    random_state,
    precomputed_splits=None,
    common_sessions_ref=None,
    compute_r2=True,
    pca=False,
    pca_components=27,
):
    """Run dimension residualization + classification for one FM model and one target.

    Parameters
    ----------
    labels_dict : dict
        ``{subject_session_key: label_str}`` — pre-loaded for the current target.
    is_multiclass : bool
        When True, uses balanced_accuracy_score instead of AUC.
    precomputed_splits : list of fold dicts or None
        Pre-computed CV splits.  When provided with ``common_sessions_ref``,
        data is aligned to that session list.
    common_sessions_ref : list of str or None
        Reference session order so fold indices remain valid across models.
    compute_r2 : bool
        If True, compute and save R² per embedding dimension.

    Returns
    -------
    model_results : dict or None
    subjects_info : dict or None
    splits_used : list of fold dicts
    common_keys_used : list of str
    labels_for_splits : dict  ``{session_key: int_label}``
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Foundation model: {model_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ── Load embeddings & markers ─────────────────────────────────────────
    print("  Loading embeddings ...", flush=True)
    embeddings = load_embeddings(results_root, model_name)
    if not embeddings:
        print(f"  [SKIP] No embeddings found for {model_name}.", flush=True)
        return None, None, None, None, None

    print("  Loading markers ...", flush=True)
    markers_dict, marker_names = load_markers(marker_csv, reduction)

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

    X = np.stack([embeddings[k] for k in common_keys])    # (n, emb_dim)
    Y = np.stack([markers_dict[k] for k in common_keys])  # (n, n_markers)
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
    metric_name = "Bal.Acc" if is_multiclass else "AUC"
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
        is_multiclass=is_multiclass,
        pca=pca,
        pca_components=pca_components,
    )

    for clf_name, res in model_results.items():
        print(
            f"  [{clf_name}] {metric_name} = {res['mean_auc']:.3f} ± {res['std_auc']:.3f}"
            f" ({res['n_valid_folds']} folds)",
            flush=True,
        )

    # ── R² per embedding dimension (global fit, only for CRS to save time) ──
    if compute_r2:
        print("  Computing R² per embedding dimension ...", flush=True)
        groups_all = np.array([s.split("_ses-")[0] for s in common_keys])
        try:
            r2_values = compute_r2_per_dim(Y, X, groups=groups_all)
            os.makedirs(output_dir, exist_ok=True)
            r2_plot_path = op.join(output_dir, f"r2_per_dim_{model_name}.png")
            plot_r2_per_dim(r2_values, model_name, r2_plot_path)
            np.savez(
                op.join(output_dir, f"r2_per_dim_{model_name}.npz"),
                r2_per_dim=r2_values,
            )
        except Exception as exc:
            print(f"  [WARN] R² per dim plot failed: {exc}", flush=True)

    return model_results, subjects_info, precomputed_splits, common_keys, labels_for_splits


def save_results(
    all_results, subjects_per_model, output_dir, reduction, n_folds,
    target="crs", label_mode=None, subset=None, is_multiclass=False,
):
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
    target, label_mode, subset, is_multiclass
        Target configuration for metadata embedding.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_payload = {
        "method": "dimension_residualization",
        "reduction": reduction,
        "n_folds": n_folds,
        "target": target,
        "label_mode": label_mode,
        "subset": subset,
        "is_multiclass": is_multiclass,
        "metric": "balanced_accuracy" if is_multiclass else "roc_auc",
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
                    "n_classes": len(class_counts),
                    "classes": str(list(class_counts.keys())),
                    "reduction": reduction,
                    "target": target,
                    "label_mode": label_mode or "",
                    "subset": subset or "",
                    "is_multiclass": is_multiclass,
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
            "(markers → each embedding dim), then classify. "
            "Supports all clinical targets from baseline.py."
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
        choices=["kernel_ridge", "svm", "mlp", "random_forest", "xgboost"],
        default=["kernel_ridge", "svm", "mlp", "random_forest", "xgboost"],
        help="Classifiers to evaluate (default: all five).",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help=(
            "Apply per-fold train-only PCA to residualized FM embeddings before "
            "classification. Components fit on the training split only to avoid leakage."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=27,
        help="Number of PCA components when --pca is set (default: 27).",
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
            "If omitted, splits are auto-saved inside {output_dir}/cv_splits/."
        ),
    )
    # ── Target selection ──────────────────────────────────────────────────────
    _target_choices = ["crs", "etiology", "etiology_code", "cs_6m", "cs_1y", "cs_2y"]
    parser.add_argument(
        "--target",
        choices=_target_choices,
        default="crs",
        help="Prediction target (default: crs). Ignored when --all-targets is set.",
    )
    parser.add_argument(
        "--label-mode",
        default=None,
        choices=[
            "multiclass", "binary", "binary_death",
            "binary_vs_to_mcs", "binary_mcs_to_conscious", "binary_improvement",
        ],
        help="Label mode for cs_* targets (default: binary).",
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=["VS", "MCS"],
        help="Restrict etiology/etiology_code to VS or MCS baseline subjects.",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "Run all 25 target × mode configurations sequentially and produce "
            "a summary heatmap across all targets and FM models."
        ),
    )
    args = parser.parse_args()

    models_to_run = FM_MODELS if args.model == "all" else [args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build list of target configs to run ───────────────────────────────────
    if args.all_targets:
        configs_to_run = TARGET_CONFIGS
    else:
        # Default label_mode for cs_* if not specified
        lm = args.label_mode
        if lm is None and args.target in ("cs_6m", "cs_1y", "cs_2y"):
            lm = "binary"
        # Find matching config entry for title_suffix; fall back to inline
        title_suffix = args.target
        for tgt, lmode, sub, subdir, tsuffix in TARGET_CONFIGS:
            if tgt == args.target and lmode == lm and sub == args.subset:
                title_suffix = tsuffix
                break
        configs_to_run = [(args.target, lm, args.subset, None, title_suffix)]

    # ── Main loop over target configs ─────────────────────────────────────────
    summary_rows = []  # for cross-target heatmap

    for cfg_target, cfg_mode, cfg_subset, cfg_subdir, cfg_title in configs_to_run:
        if cfg_subdir is not None:
            target_output_dir = op.join(args.output_dir, cfg_subdir)
        else:
            target_output_dir = args.output_dir
        os.makedirs(target_output_dir, exist_ok=True)

        print(
            f"\n{'#' * 65}\n"
            f"  TARGET: {cfg_title!r}\n"
            f"  output: {target_output_dir}\n"
            f"{'#' * 65}",
            flush=True,
        )

        # Load labels for this target
        print("Loading patient labels ...", flush=True)
        labels_dict, is_multiclass = load_labels_for_target(
            args.patient_labels, cfg_target, cfg_mode, cfg_subset
        )
        if len(labels_dict) < 10:
            print(f"  [SKIP] Too few labelled sessions ({len(labels_dict)}) — skip.", flush=True)
            continue

        # ── Load or prepare per-target splits ─────────────────────────────
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
        # Only compute R² for the CRS target to keep runtimes manageable
        _compute_r2 = (cfg_target == "crs")

        for model_name in models_to_run:
            result = run_model(
                model_name=model_name,
                results_root=args.results_root,
                marker_csv=args.marker_csv,
                labels_dict=labels_dict,
                is_multiclass=is_multiclass,
                output_dir=op.join(target_output_dir, model_name),
                reduction=args.reduction,
                classifiers=args.classifiers,
                n_folds=args.n_folds,
                random_state=args.random_state,
                precomputed_splits=precomputed_splits,
                common_sessions_ref=common_sessions_ref,
                compute_r2=_compute_r2,
                pca=args.pca,
                pca_components=args.pca_components,
            )
            model_results, subjects_info, splits_used, common_used, labels_used = result
            if model_results is None:
                continue

            all_results[model_name] = model_results
            subjects_per_model[model_name] = subjects_info or {}

            # Collect best-clf score for summary heatmap
            for clf_name, res in model_results.items():
                summary_rows.append({
                    "title": cfg_title,
                    "subdir": cfg_subdir or "",
                    "model": model_name,
                    "clf": clf_name,
                    "mean_auc": res["mean_auc"],
                })

            # Lock in splits on first successful run per target
            if not _splits_saved and splits_used is not None:
                precomputed_splits = splits_used
                common_sessions_ref = common_used
                labels_ref = labels_used
                save_path = args.save_splits_to
                if not save_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    splits_dir = op.join(target_output_dir, "cv_splits")
                    tag = (cfg_subdir or cfg_target).replace("/", "_")
                    save_path = op.join(splits_dir, f"dim_splits_{tag}_{ts}.json")
                save_cv_splits(splits_used, common_used, labels_used, save_path)
                print(f"  CV splits saved to {save_path}", flush=True)
                _splits_saved = True

        if not all_results:
            print("  No results — all models were skipped for this target.", flush=True)
            continue

        print("\nSaving results ...", flush=True)
        save_results(
            all_results=all_results,
            subjects_per_model=subjects_per_model,
            output_dir=target_output_dir,
            reduction=args.reduction,
            n_folds=args.n_folds,
            target=cfg_target,
            label_mode=cfg_mode,
            subset=cfg_subset,
            is_multiclass=is_multiclass,
        )

        print("Generating AUC comparison plot ...", flush=True)
        plot_auc_comparison(
            all_results=all_results,
            output_path=op.join(target_output_dir, "auc_comparison.png"),
            classifiers=args.classifiers,
            title_suffix=cfg_title,
            is_multiclass=is_multiclass,
        )

    # ── Summary heatmap (only meaningful when --all-targets) ──────────────────
    if args.all_targets and summary_rows:
        print("\nGenerating summary heatmap across all targets ...", flush=True)
        plot_summary_heatmap(
            summary_data=summary_rows,
            output_path=op.join(args.output_dir, "summary_heatmap.png"),
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
