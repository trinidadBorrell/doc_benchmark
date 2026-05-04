"""No-leakage dimension-wise marker residualization of FM embeddings.

=================================================================
Leakage-Free Residualization + Classification Pipeline
=================================================================

Same logic as ``residualization_dim.py`` with one critical fix:

**FM embeddings (X) are now standardized fold-internally** (StandardScaler
fit on training subjects only, applied to both train and test).  The original
script omitted this step, causing Ridge regularization to be applied in the
raw embedding scale and distance-based classifiers (SVM, KernelRidge) to
operate in a space whose geometry was not computed fold-internally.

Rigorous per-fold protocol
--------------------------
1. **Split** — StratifiedGroupKFold (5 outer folds, subject-level groups).
2. **Standardize Y** — NaN-impute with training-set column medians, then
   StandardScaler fit on training markers only; applied to test with those
   training statistics.  (Done inside DimensionResidualizer — unchanged.)
3. **Standardize X** [NEW FIX] — StandardScaler fit on training embeddings
   only; applied to test with those training statistics.
4. **Residualize** — Multi-output Ridge (inner GroupKFold for alpha selection)
   fit on (Y_train_scaled → X_train_scaled).  Residuals:
   ``R = X_scaled - Ridge.predict(Y_scaled)`` for train and test separately.
5. **Optional PCA** — fit on R_train, transform R_test.
6. **Classify** — inner CV for hyperparameter selection on R_train only;
   final fit on full R_train; AUC on R_test.

This script always runs **both** PCA modes (no_pca and pca27) in one job,
saving results to ``{output_dir}/no_pca/`` and ``{output_dir}/pca27/``.

A live-updating ``progress.txt`` table is written after every
(FM, target, PCA-mode) combination to keep you informed without tailing logs.

Usage
-----
::

    python fold_internal_residualisation.py \\
        --results-root /data/project/eeg_foundation/data/benchmark_results/new_results \\
        --marker-csv /data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv \\
        --patient-labels /data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv \\
        --output-dir /data/project/eeg_foundation/data/benchmark_results/new_results/RES_NO_LEAKAGE

    # Single model
    python fold_internal_residualisation.py --model LaBram ...

    # Use pre-computed splits for fair comparison with other scripts
    python fold_internal_residualisation.py --splits-file /path/to/splits.json ...
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

# Import KernelRidgeClassifier with fallback for direct execution.
# This script lives one directory deeper than residualization_dim.py, so the
# model dir is two levels up (res_no_leakage/ → interp/ → src/ → model/).
try:
    from ...model.kernel_ridge_classifier import KernelRidgeClassifier
except ImportError:
    _model_dir = op.abspath(op.join(op.dirname(__file__), "..", "..", "model"))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    from kernel_ridge_classifier import KernelRidgeClassifier

try:
    from ...model.cv_utils import (
        check_no_subject_leakage,
        filter_excluded_markers,
        generate_nested_cv_folds,
        load_cv_splits,
        save_cv_splits,
    )
except ImportError:
    _model_dir = op.abspath(op.join(op.dirname(__file__), "..", "..", "model"))
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

# ── Constants ─────────────────────────────────────────────────────────────────

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
    "EEGPT": "#8c564b",
}

_CLASSIFIER_MARKERS = {
    "kernel_ridge": "o",
    "svm": "s",
    "mlp": "^",
    "random_forest": "D",
    "xgboost": "P",
}

_ALL_CLASSIFIERS = ["kernel_ridge", "svm", "mlp", "random_forest", "xgboost"]
_PCA_MODES = [("no_pca", False), ("pca27", True)]


# ── Data loading ──────────────────────────────────────────────────────────────


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


# ── DimensionResidualizer ─────────────────────────────────────────────────────


class DimensionResidualizer(BaseEstimator, TransformerMixin):
    """Remove marker-predictable variance from each embedding dimension.

    Fits a multi-output Ridge regression from markers to embeddings::

        X ≈ Y_scaled @ W.T + b          (shape: (n, emb_dim))

    During ``transform``, the marker-predicted component is subtracted
    from each embedding dimension independently::

        X_res = X - ridge.predict(Y_scaled)

    Y is imputed and scaled using training-set statistics only.
    X should be pre-scaled fold-internally before calling fit/transform.

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength.
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
            Marker scalars for training subjects (NaN allowed — imputed with
            the training-set median).
        X : ndarray, shape (n, emb_dim)
            Embeddings for training subjects.  Should be pre-standardized
            fold-internally (via the x_scaler in run_nested_cv_no_leakage).
        groups : ndarray, shape (n,) or None
            Subject-level group IDs for inner GroupKFold.
        """
        # ── Impute NaNs in markers (column median on training data) ──────
        self._col_medians = np.nanmedian(Y, axis=0)
        Y_imp = self._impute(Y)

        # ── Scale markers (fit on training data only) ─────────────────────
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
            Pre-standardized embeddings (same x_scaler as in fit).

        Returns
        -------
        X_res : ndarray, shape (n, emb_dim)
            Residualized embeddings (marker-predicted variance removed).
        """
        Y_imp = self._impute(Y)
        Y_scaled = self._scaler.transform(Y_imp)
        X_pred = self._ridge.predict(Y_scaled)
        return X - X_pred

    def _impute(self, Y):
        """Replace NaNs with column medians from training data."""
        Y_imp = Y.copy()
        for j in range(Y_imp.shape[1]):
            mask = np.isnan(Y_imp[:, j])
            if mask.any():
                Y_imp[mask, j] = self._col_medians[j]
        return Y_imp


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_auc(y_true, y_score, y_pred, is_multiclass=False):
    """AUC with balanced_accuracy fallback."""
    if is_multiclass:
        return balanced_accuracy_score(y_true, y_pred)
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return balanced_accuracy_score(y_true, y_pred)


def _decision_scores(clf, clf_name, X):
    """Return a 1-D score array suitable for roc_auc_score."""
    if clf_name in ("mlp", "random_forest", "xgboost"):
        proba = clf.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]
    if clf_name == "svm":
        raw = clf.decision_function(X)
        if raw.ndim > 1:
            return raw[:, 0]  # ignored for multiclass; balanced_accuracy uses preds
        return raw
    # kernel_ridge (binary) or OneVsRestClassifier(KernelRidgeClassifier) (multiclass)
    raw = clf.decision_function(X)
    if raw.ndim > 1:
        return raw[:, 0]  # ignored for multiclass; balanced_accuracy uses preds
    return raw


def _inner_cv_classifier(
    clf_name, X_train, y_train, groups_train, random_state, inner_splits=None
):
    """Run inner CV grid search and return (best_fitted_clf, best_inner_score)."""
    if inner_splits is not None:
        cv_iter = inner_splits
    else:
        n_unique = len(np.unique(groups_train))
        n_inner = min(3, n_unique)
        inner_cv = GroupKFold(n_splits=n_inner)
        cv_iter = list(inner_cv.split(X_train, y_train, groups=groups_train))

    if clf_name == "kernel_ridge":
        from sklearn.multiclass import OneVsRestClassifier as _OvR
        _is_mc = len(np.unique(y_train)) > 2

        def _make_kr(alpha):
            kr = KernelRidgeClassifier(alpha=alpha, kernel="rbf")
            return _OvR(kr) if _is_mc else kr

        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha, best_score = 1.0, -np.inf
        for a in alphas:
            scores = []
            for tr, va in cv_iter:
                clf = _make_kr(a)
                clf.fit(X_train[tr], y_train[tr])
                preds = clf.predict(X_train[va])
                raw = clf.decision_function(X_train[va])
                if hasattr(raw, "ndim") and raw.ndim > 1:
                    raw = raw[:, 0]  # balanced_accuracy uses preds; raw ignored
                scores.append(_safe_auc(y_train[va], raw, preds, is_multiclass=_is_mc))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_alpha = a
        final = _make_kr(best_alpha)
        final.fit(X_train, y_train)
        return final, best_score

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
        return final, best_score

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
        return final, best_score

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
                    n_jobs=4,
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
            n_jobs=4,
        )
        final.fit(X_train, y_train)
        return final, best_score

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
                    n_jobs=4,
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
            n_jobs=4,
        )
        final.fit(X_train, y_train)
        return final, best_score

    raise ValueError(f"Unknown classifier: {clf_name!r}")


# ── R² per embedding dimension ────────────────────────────────────────────────


def compute_r2_per_dim(Y, X, groups=None):
    """Fit DimensionResidualizer on all data and return per-dimension R².

    Global fit for visualization only — not fold-internal.
    """
    residualizer = DimensionResidualizer()
    residualizer.fit(Y, X, groups=groups)
    Y_imp = residualizer._impute(Y)
    Y_scaled = residualizer._scaler.transform(Y_imp)
    X_pred = residualizer._ridge.predict(Y_scaled)
    r2_values = np.array(
        [r2_score(X[:, i], X_pred[:, i]) for i in range(X.shape[1])]
    )
    return r2_values


def plot_r2_per_dim(r2_values, model_name, output_path):
    """Plot R² per embedding dimension."""
    emb_dim = len(r2_values)
    x = np.arange(emb_dim)
    sort_idx = np.argsort(r2_values)
    r2_sorted = r2_values[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    color = _MODEL_COLORS.get(model_name, "#1f77b4")

    axes[0].plot(x, r2_values, linewidth=0.8, color=color, alpha=0.85)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("Embedding dimension index", fontsize=12)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("By dimension index", fontsize=11)
    axes[0].grid(True, alpha=0.25)

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


# ── ProgressTracker ───────────────────────────────────────────────────────────


class ProgressTracker:
    """Write a live-updating ASCII table to progress.txt.

    Tracks (FM, target_title, pca_label) → status and per-classifier AUC.
    The table is rewritten after every mark_running / mark_done / mark_failed
    call so you can tail progress.txt or open it mid-run.
    """

    _STATUS_PENDING = "PENDING"
    _STATUS_RUNNING = "RUNNING"
    _STATUS_DONE    = "DONE"
    _STATUS_FAILED  = "FAILED"
    _STATUS_SKIPPED = "SKIPPED"

    def __init__(self, output_dir, fms, target_configs, pca_labels, classifiers):
        # One file per FM when running a single model (parallel jobs); shared
        # file only when all models are processed in one process.
        fname = f"progress_{fms[0]}.txt" if len(fms) == 1 else "progress.txt"
        self.output_path = op.join(output_dir, fname)
        self.classifiers = classifiers
        self.total = len(fms) * len(target_configs) * len(pca_labels)
        self._rows = {}  # key (fm, title, pca_label) -> dict

        for pca_label in pca_labels:
            for fm in fms:
                for _, _, _, _, title in target_configs:
                    key = (fm, title, pca_label)
                    self._rows[key] = {
                        "fm": fm,
                        "title": title,
                        "pca": pca_label,
                        "status": self._STATUS_PENDING,
                        "aucs": {},
                    }
        self._write()

    def mark_running(self, fm, title, pca_label):
        key = (fm, title, pca_label)
        if key in self._rows:
            self._rows[key]["status"] = self._STATUS_RUNNING
            self._write()

    def mark_done(self, fm, title, pca_label, results_by_clf):
        """results_by_clf: {clf_name: {"mean_auc": float}} or None."""
        key = (fm, title, pca_label)
        if key not in self._rows:
            return
        if results_by_clf is None:
            self._rows[key]["status"] = self._STATUS_SKIPPED
        else:
            self._rows[key]["status"] = self._STATUS_DONE
            self._rows[key]["aucs"] = {
                clf: res.get("mean_auc", float("nan"))
                for clf, res in results_by_clf.items()
            }
        self._write()

    def mark_failed(self, fm, title, pca_label):
        key = (fm, title, pca_label)
        if key in self._rows:
            self._rows[key]["status"] = self._STATUS_FAILED
            self._write()

    def _write(self):
        done_count = sum(
            1 for r in self._rows.values()
            if r["status"] in (self._STATUS_DONE, self._STATUS_SKIPPED, self._STATUS_FAILED)
        )

        clf_abbrev = {
            "kernel_ridge": "KR",
            "svm": "SVM",
            "mlp": "MLP",
            "random_forest": "RF",
            "xgboost": "XGB",
        }
        clf_headers = [clf_abbrev.get(c, c[:5]) for c in self.classifiers]

        # Column widths
        W_FM     = 9
        W_TITLE  = 26
        W_PCA    = 7
        W_CLF    = 6
        W_STATUS = 8

        sep = (
            "-" * W_FM + "-+-"
            + "-" * W_TITLE + "-+-"
            + "-" * W_PCA + "-+-"
            + "-+-".join(["-" * W_CLF] * len(self.classifiers))
            + "-+-"
            + "-" * W_STATUS
        )

        lines = [
            "=== RES_NO_LEAKAGE Progress (leakage-free residualization) ===",
            f"Updated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Progress: {done_count} / {self.total} combinations completed",
            "",
            " | ".join([
                f"{'FM':<{W_FM}}",
                f"{'Target':<{W_TITLE}}",
                f"{'PCA':<{W_PCA}}",
            ] + [f"{h:>{W_CLF}}" for h in clf_headers] + [f"{'Status':<{W_STATUS}}"]),
            sep,
        ]

        for row in self._rows.values():
            aucs_str = []
            for clf in self.classifiers:
                v = row["aucs"].get(clf, float("nan"))
                if row["status"] in (self._STATUS_PENDING, self._STATUS_RUNNING):
                    aucs_str.append(f"{'—':>{W_CLF}}")
                elif np.isnan(v):
                    aucs_str.append(f"{'NaN':>{W_CLF}}")
                else:
                    aucs_str.append(f"{v:>{W_CLF}.3f}")
            lines.append(
                " | ".join([
                    f"{row['fm']:<{W_FM}}",
                    f"{row['title']:<{W_TITLE}}",
                    f"{row['pca']:<{W_PCA}}",
                ] + aucs_str + [f"{row['status']:<{W_STATUS}}"])
            )

        lines.append("")
        try:
            with open(self.output_path, "w") as fh:
                fh.write("\n".join(lines))
        except OSError:
            pass


# ── Core CV loop (no-leakage version) ────────────────────────────────────────


def run_nested_cv_no_leakage(
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
    n_repeats=1,
    force_clf_per_repeat_fold=None,
    precomputed_repeats=None,
):
    """Nested CV with fold-internal X standardization + dimension residualization.

    Leakage-free protocol per outer fold k
    ----------------------------------------
    1. Split into train (≈80%) and test (≈20%) using subject-level groups.
    2. Standardize X (FM embeddings): fit StandardScaler on X_train,
       apply to both X_train and X_test.  **This is the fix absent in
       residualization_dim.py.**
    3. Fit DimensionResidualizer on (Y_train, X_train_scaled):
       - Impute Y NaNs with training-set column medians.
       - Scale Y with StandardScaler fit on Y_train.
       - Select Ridge alpha via inner GroupKFold on training fold.
       - Refit Ridge on full training fold.
    4. Residualize: R_train = X_train_scaled - Ridge.predict(Y_train_scaled)
                    R_test  = X_test_scaled  - Ridge.predict(Y_test_scaled)
       (same Ridge coefficients; no test information in those coefficients)
    5. Optional PCA: fit on R_train, transform R_test.
    6. Fit classifier on R_train (inner CV for hyperparameters), evaluate on R_test.

    Parameters
    ----------
    X : ndarray (n, emb_dim)       Raw embeddings.
    Y : ndarray (n, n_markers)     Marker scalars (NaN allowed).
    y_cls : ndarray (n,)           Integer class labels.
    subjects : list of str         Subject-session keys.
    marker_names : list of str
    classifiers : list of str
    n_folds : int
    random_state : int
    precomputed_splits : list of fold dicts or None
    is_multiclass : bool
    pca : bool
    pca_components : int

    Returns
    -------
    results : dict
        ``{clf_name: {"auc_per_fold": [...], "mean_auc": float, ...}}``
    """
    groups = np.array([s.split("_ses-")[0] for s in subjects])

    # ── Build repeat list ────────────────────────────────────────────────────
    # Priority order:
    #   1. precomputed_repeats (manifest-driven, multiple repeats with their own folds)
    #   2. precomputed_splits  (single set of folds → single repeat)
    #   3. auto-generate per-repeat folds with seed = random_state + r
    if precomputed_repeats is not None:
        _repeats_iter = list(precomputed_repeats)
        n_repeats = len(_repeats_iter)
        print(
            f"  Using {n_repeats} pre-computed repeats from manifest "
            f"(seeds: {[s for _, _, s in _repeats_iter[:3]]}...)",
            flush=True,
        )
    elif precomputed_splits is not None:
        # Single externally provided split → single repeat
        if n_repeats > 1:
            print(
                "  [WARN] n_repeats > 1 ignored when precomputed_splits "
                "provided without precomputed_repeats.",
                flush=True,
            )
            n_repeats = 1
        _repeats_iter = [(0, precomputed_splits, random_state)]
    else:
        _repeats_iter = []
        for _r in range(n_repeats):
            _seed_r = random_state + _r
            _n_uniq = len(np.unique(groups))
            _eff = min(n_folds, _n_uniq)
            _outer_cv = StratifiedGroupKFold(
                n_splits=_eff, shuffle=True, random_state=_seed_r
            )
            _folds_r = [
                {"train_idx": tr, "test_idx": te, "inner_splits": None}
                for tr, te in _outer_cv.split(X, y_cls, groups=groups)
            ]
            _repeats_iter.append((_r, _folds_r, _seed_r))

    # Accumulators: per-clf across all repeats x folds.
    fold_aucs    = {clf_name: [] for clf_name in classifiers}
    fold_details = {clf_name: [] for clf_name in classifiers}
    # Best-clf accumulator across all repeats x folds.
    best_clf_aucs    = []
    best_clf_details = []   # {"fold", "repeat", "clf", "auc", ...}
    manifest_repeats = []

    for _repeat_idx, folds_iter, _repeat_seed in _repeats_iter:
        n_folds_actual = len(folds_iter)

        # ── Subject-level leakage sanity checks ──────────────────────────────
        for fold_idx, fold in enumerate(folds_iter):
            check_no_subject_leakage(
                groups, fold["train_idx"], fold["test_idx"],
                label=f"outer fold {fold_idx}"
            )

        _repeat_best_clf_per_fold = []

        for fold_idx, fold in enumerate(folds_iter):
            train_idx = fold["train_idx"]
            test_idx  = fold["test_idx"]
            inner_splits = fold.get("inner_splits")

            print(
                f"   Outer fold {fold_idx + 1}/{n_folds_actual}: "
                f"train={len(train_idx)}, test={len(test_idx)}",
                flush=True,
            )

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            Y_train,     Y_test     = Y[train_idx], Y[test_idx]
            y_train,     y_test     = y_cls[train_idx], y_cls[test_idx]
            groups_train            = groups[train_idx]

            # ── LEAKAGE FIX: standardize FM embeddings fold-internally ────
            x_scaler = StandardScaler()
            X_train_raw = x_scaler.fit_transform(X_train_raw)
            X_test_raw  = x_scaler.transform(X_test_raw)

            # ── Fit residualizer on training data ─────────────────────────
            residualizer = DimensionResidualizer()
            residualizer.fit(Y_train, X_train_raw, groups=groups_train)

            # ── Residualize ───────────────────────────────────────────────
            X_train_res = residualizer.transform(Y_train, X_train_raw)
            X_test_res  = residualizer.transform(Y_test,  X_test_raw)

            # ── Optional per-fold train-only PCA ──────────────────────────
            pca_n_comp  = None
            pca_var_exp = None
            if pca:
                from sklearn.decomposition import PCA as _PCA
                n_train = X_train_res.shape[0]
                n_feat  = X_train_res.shape[1]
                pca_n_comp = int(min(pca_components, max(n_train - 1, 1), n_feat))
                _pca = _PCA(n_components=pca_n_comp, random_state=random_state)
                X_train_res = _pca.fit_transform(X_train_res)
                X_test_res  = _pca.transform(X_test_res)
                pca_var_exp = float(_pca.explained_variance_ratio_.sum())

            # ── Fit and evaluate all classifiers ──────────────────────────
            _fold_inner_scores  = {}
            _fold_clf_objects   = {}
            _fold_clf_auc_vals  = {}
            _fold_clf_pred_vals = {}
            _fold_clf_score_vals = {}

            for clf_name in classifiers:
                try:
                    clf, inner_score = _inner_cv_classifier(
                        clf_name,
                        X_train_res,
                        y_train,
                        groups_train,
                        random_state,
                        inner_splits=inner_splits,
                    )
                    scores_arr = _decision_scores(clf, clf_name, X_test_res)
                    preds      = clf.predict(X_test_res)
                    auc_val    = _safe_auc(y_test, scores_arr, preds,
                                          is_multiclass=is_multiclass)
                except Exception as exc:
                    print(
                        f"      [WARN] {clf_name} fold {fold_idx + 1} failed: {exc}",
                        flush=True,
                    )
                    inner_score = float("nan")
                    auc_val     = float("nan")
                    preds       = np.zeros(len(y_test), dtype=int)
                    clf         = None

                _fold_inner_scores[clf_name]  = float(inner_score)
                _fold_clf_objects[clf_name]   = clf
                _fold_clf_auc_vals[clf_name]  = float(auc_val)
                _fold_clf_pred_vals[clf_name] = preds

                metric_name = "Bal.Acc" if is_multiclass else "AUC"
                fold_aucs[clf_name].append(float(auc_val))
                fold_detail = {
                    "fold": fold_idx + 1,
                    "repeat": _repeat_idx,
                    "auc": float(auc_val),
                    "train_subjects": [subjects[i] for i in train_idx],
                    "test_subjects":  [subjects[i] for i in test_idx],
                    "class_counts_train": {
                        str(c): int(np.sum(y_train == c)) for c in np.unique(y_train)
                    },
                    "class_counts_test": {
                        str(c): int(np.sum(y_test == c)) for c in np.unique(y_test)
                    },
                }
                if pca:
                    fold_detail["n_pca_components"]      = pca_n_comp
                    fold_detail["pca_explained_variance"] = pca_var_exp
                fold_details[clf_name].append(fold_detail)
                print(f"      [{clf_name}] {metric_name} = {auc_val:.3f}", flush=True)

            # ── Per-fold best-classifier selection ────────────────────────
            valid_inner = {
                k: v for k, v in _fold_inner_scores.items() if not np.isnan(v)
            }
            if valid_inner:
                # Check for forced selection (cv_nested_matched mode).
                if (
                    force_clf_per_repeat_fold is not None
                    and _repeat_idx < len(force_clf_per_repeat_fold)
                    and fold_idx < len(force_clf_per_repeat_fold[_repeat_idx])
                    and force_clf_per_repeat_fold[_repeat_idx][fold_idx] in valid_inner
                ):
                    best_clf_name = force_clf_per_repeat_fold[_repeat_idx][fold_idx]
                else:
                    best_clf_name = max(valid_inner, key=valid_inner.get)

                best_auc = _fold_clf_auc_vals.get(best_clf_name, float("nan"))
                print(
                    f"   [best-clf] {best_clf_name} "
                    f"(inner={valid_inner.get(best_clf_name, float('nan')):.3f}, "
                    f"AUC={best_auc:.3f})",
                    flush=True,
                )
                best_clf_aucs.append(best_auc)
                best_clf_details.append({
                    "fold": fold_idx + 1,
                    "repeat": _repeat_idx,
                    "clf": best_clf_name,
                    "inner_score": valid_inner.get(best_clf_name),
                    "all_inner_scores": dict(valid_inner),
                    "auc": best_auc,
                })
                _repeat_best_clf_per_fold.append({
                    "fold": fold_idx,
                    "clf": best_clf_name,
                    "inner_score": valid_inner.get(best_clf_name),
                    "all_inner_scores": dict(valid_inner),
                })

        # Manifest entry for this repeat.
        manifest_repeats.append({
            "repeat_idx": _repeat_idx,
            "random_state": _repeat_seed,
            "folds": [
                {
                    "train_idx": f["train_idx"].tolist(),
                    "test_idx":  f["test_idx"].tolist(),
                }
                for f in folds_iter
            ],
            "best_clf_per_fold": _repeat_best_clf_per_fold,
        })

    # ── Aggregate per-classifier results ─────────────────────────────────────
    results = {}
    for clf_name in classifiers:
        valid_aucs = [a for a in fold_aucs[clf_name] if not np.isnan(a)]
        results[clf_name] = {
            "auc_per_fold":     fold_aucs[clf_name],
            "mean_auc":         float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
            "std_auc":          float(np.std(valid_aucs)) if len(valid_aucs) > 1 else 0.0,
            "n_valid_folds":    len(valid_aucs),
            "per_fold_details": fold_details[clf_name],
        }

    # ── Best-clf aggregate ────────────────────────────────────────────────────
    valid_bc = [a for a in best_clf_aucs if not np.isnan(a)]
    from collections import Counter as _Counter
    clf_sel_counts = _Counter(d["clf"] for d in best_clf_details)
    results["best_clf"] = {
        "auc_per_fold":              best_clf_aucs,
        "mean_auc":                  float(np.mean(valid_bc)) if valid_bc else float("nan"),
        "std_auc":                   float(np.std(valid_bc)) if len(valid_bc) > 1 else 0.0,
        "n_valid_folds":             len(valid_bc),
        "n_repeats":                 n_repeats,
        "per_fold_details":          best_clf_details,
        "classifier_selection_counts": dict(clf_sel_counts),
        "manifest_repeats":          manifest_repeats,
        # per_fold_metrics in macro_average format for plot scripts:
        "macro_average": {
            "auc_score": {
                "mean": float(np.mean(valid_bc)) if valid_bc else float("nan"),
                "std":  float(np.std(valid_bc)) if len(valid_bc) > 1 else 0.0,
            },
            "per_fold_metrics": [
                {
                    "fold": d["fold"],
                    "auc_score": d["auc"],
                    "classifier": d["clf"],
                    "repeat": d.get("repeat", 0),
                }
                for d in best_clf_details
            ],
        },
    }

    return results


# ── Manifest loading for cv_nested_matched mode ───────────────────────────────


def load_fm_manifest(manifest_root, model_name, target, target_subdir=None):
    """Load FM-embedding cv_split_manifest.json for the given target/model.

    `manifest_root` is expected to point at the per-model MLP_EMBEDDING dir
    (e.g. ``paper_results/BIOT/doc_patients/MLP_EMBEDDING``) — same convention
    as the dk_embeddings_classification submit files.  If the path includes
    ``{model_name}/doc_patients/MLP_EMBEDDING`` we use it as-is; if not, we
    auto-append the per-model suffix (results-root convention).

    Manifest structure (written by fm_embedding_classifier.py):
        {
          "common_sessions": [<subject_session_keys>],
          "repeats": [
            {"repeat_idx": int, "random_state": int,
             "folds": [{"train_idx": [...], "test_idx": [...]}, ...],
             "best_clf_per_fold": [<clf_name>, ...]},
            ...
          ]
        }

    Returns (common_sessions, precomputed_repeats, force_clf_per_repeat_fold)
    where precomputed_repeats is a list of (repeat_idx, folds, seed) tuples
    and force_clf_per_repeat_fold is a list of lists of classifier names.

    Returns (None, None, None) if the manifest file does not exist.
    """
    target_path = target if target_subdir is None else op.join(target, target_subdir)
    # Auto-detect whether manifest_root already includes the per-model suffix.
    # The submit files pass per-model paths like
    #   .../paper_results/BIOT/doc_patients/MLP_EMBEDDING
    # which already ends with "MLP_EMBEDDING".
    if op.basename(manifest_root.rstrip("/")) == "MLP_EMBEDDING":
        per_model_root = manifest_root
    else:
        per_model_root = op.join(
            manifest_root, model_name, "doc_patients", "MLP_EMBEDDING"
        )
    manifest_path = op.join(
        per_model_root,
        target_path,
        "nested_cv_repeated",
        "cv_split_manifest.json",
    )
    if not op.isfile(manifest_path):
        print(
            f"  [matched-mode] No manifest at {manifest_path} — "
            f"falling back to cv_nested (independent folds).",
            flush=True,
        )
        return None, None, None

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    common_sessions = manifest.get("common_sessions") or []
    repeats = manifest.get("repeats") or []
    if not common_sessions or not repeats:
        return None, None, None

    precomputed_repeats = []
    force_clf_per_repeat_fold = []
    for r in repeats:
        folds = []
        for fold in r["folds"]:
            folds.append({
                "train_idx": np.asarray(fold["train_idx"], dtype=np.int64),
                "test_idx":  np.asarray(fold["test_idx"],  dtype=np.int64),
                "inner_splits": fold.get("inner_splits"),
            })
        precomputed_repeats.append(
            (int(r["repeat_idx"]), folds, int(r["random_state"]))
        )
        force_clf_per_repeat_fold.append(list(r["best_clf_per_fold"]))

    print(
        f"  Loaded manifest from {manifest_path} "
        f"({len(common_sessions)} sessions, {len(repeats)} repeats)",
        flush=True,
    )
    return common_sessions, precomputed_repeats, force_clf_per_repeat_fold


# ── Per-model runner ──────────────────────────────────────────────────────────


def run_model_no_leakage(
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
    n_repeats=1,
    force_clf_per_repeat_fold=None,
    cv_output_label=None,
    precomputed_repeats=None,
    target_subdir=None,
):
    """Run leakage-free residualization + classification for one FM and one target.

    Returns
    -------
    (model_results, subjects_info, splits_used, common_keys_used, labels_for_splits)
    Any element may be None on skip.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Foundation model : {model_name}", flush=True)
    print(f"  PCA              : {'pca27' if pca else 'no_pca'}", flush=True)
    print(f"{'=' * 60}", flush=True)

    print("  Loading embeddings ...", flush=True)
    embeddings = load_embeddings(results_root, model_name)
    if not embeddings:
        print(f"  [SKIP] No embeddings found for {model_name}.", flush=True)
        return None, None, None, None, None

    print("  Loading markers ...", flush=True)
    markers_dict, marker_names = load_markers(marker_csv, reduction)

    # ── Align subjects ────────────────────────────────────────────────────
    if (
        (precomputed_splits is not None or precomputed_repeats is not None)
        and common_sessions_ref is not None
    ):
        available = set(embeddings) & set(markers_dict) & set(labels_dict)
        missing   = [s for s in common_sessions_ref if s not in available]
        if missing:
            if precomputed_repeats is not None:
                # Matched mode requires identical sessions; can't safely remap
                # arbitrary fold indices when sessions differ. Skip this model.
                print(
                    f"  [SKIP] {len(missing)} sessions from manifest not available "
                    f"for {model_name} — cannot run cv_nested_matched.",
                    flush=True,
                )
                return None, None, None, None, None
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
        print(f"  [SKIP] Only one class present: {class_counts}", flush=True)
        return None, None, None, None, None

    X = np.stack([embeddings[k]    for k in common_keys])
    Y = np.stack([markers_dict[k]  for k in common_keys])
    label_strings = [labels_dict[k] for k in common_keys]

    le     = LabelEncoder()
    y_cls  = le.fit_transform(label_strings)
    class_names = list(le.classes_)

    print(
        f"  {len(common_keys)} subjects, classes: "
        f"{dict(zip(class_names, np.bincount(y_cls)))}",
        flush=True,
    )

    subjects_info = {
        "n_subjects":   len(common_keys),
        "subject_keys": common_keys,
        "class_names":  class_names,
        "class_counts": {c: int(n) for c, n in zip(class_names, np.bincount(y_cls))},
    }

    labels_for_splits = {s: int(y_cls[i]) for i, s in enumerate(common_keys)}
    if precomputed_splits is None and n_repeats == 1:
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
    # When n_repeats > 1 and no manifest is loaded, leave precomputed_splits=None
    # so run_nested_cv_no_leakage's per-repeat loop generates fresh folds per
    # repeat with seed = random_state + repeat_idx (fairness across models is
    # preserved because they all use the same seed schedule).

    metric_name = "Bal.Acc" if is_multiclass else "AUC"
    print("  Running leakage-free nested CV ...", flush=True)
    model_results = run_nested_cv_no_leakage(
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
        n_repeats=n_repeats,
        force_clf_per_repeat_fold=force_clf_per_repeat_fold,
        precomputed_repeats=precomputed_repeats,
    )

    for clf_name, res in model_results.items():
        print(
            f"  [{clf_name}] {metric_name} = {res['mean_auc']:.3f}"
            f" ({res['n_valid_folds']} folds)",
            flush=True,
        )

    # Save per-model best-clf results to a separate path for plot scripts.
    _bc = model_results.get("best_clf", {})
    if _bc and not (isinstance(_bc.get("mean_auc"), float) and
                    np.isnan(_bc["mean_auc"])):
        _pca_label = "pca27" if pca else "no_pca"
        _bc_label = cv_output_label or ("cv_nested_matched"
                                        if force_clf_per_repeat_fold is not None
                                        else "cv_nested")
        _bc_parts = [output_dir, _bc_label, _pca_label, "best_clf", model_name]
        if target_subdir:
            _bc_parts.append(target_subdir)
        _bc_dir = op.join(*_bc_parts)
        os.makedirs(_bc_dir, exist_ok=True)
        import json as _json
        with open(op.join(_bc_dir, "results.json"), "w") as _fh:
            _json.dump(_bc, _fh, indent=2, default=lambda o: (
                float(o) if isinstance(o, np.floating) else
                int(o) if isinstance(o, np.integer) else
                o.tolist() if isinstance(o, np.ndarray) else str(o)
            ))
        print(
            f"  [best-clf] saved to {_bc_dir}/results.json "
            f"(AUC {_bc['mean_auc']:.3f}+/-{_bc['std_auc']:.3f})",
            flush=True,
        )

    if compute_r2:
        print("  Computing R² per embedding dimension (global fit) ...", flush=True)
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


# ── Save results ──────────────────────────────────────────────────────────────


def save_results(
    all_results, subjects_per_model, output_dir, reduction, n_folds,
    target="crs", label_mode=None, subset=None, is_multiclass=False, pca_label="no_pca",
):
    """Save JSON and CSV summaries of nested CV results.

    Uses a per-directory file lock so that parallel FM jobs (one per model)
    can all write to the same results.json / results.csv safely by merging
    their entries rather than overwriting each other.
    """
    import fcntl

    os.makedirs(output_dir, exist_ok=True)

    json_path = op.join(output_dir, "results.json")
    csv_path  = op.join(output_dir, "results.csv")
    lock_path = op.join(output_dir, ".results.lock")

    # Build the new model entries to add / update.
    new_model_entries = {}
    for model_name, clf_results in all_results.items():
        new_model_entries[model_name] = {
            "subjects_info": subjects_per_model.get(model_name, {}),
            "classifiers": {
                clf_name: {
                    "mean_auc":         res["mean_auc"],
                    "std_auc":          res["std_auc"],
                    "n_valid_folds":    res["n_valid_folds"],
                    "auc_per_fold":     res["auc_per_fold"],
                    "per_fold_details": res["per_fold_details"],
                }
                for clf_name, res in clf_results.items()
            },
        }

    # Acquire an exclusive lock so parallel FM jobs don't clobber each other.
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            # ── JSON: read → merge → write ────────────────────────────────
            if op.isfile(json_path):
                with open(json_path) as fh:
                    payload = json.load(fh)
            else:
                payload = {
                    "method":       "dimension_residualization_no_leakage",
                    "pca_mode":     pca_label,
                    "reduction":    reduction,
                    "n_folds":      n_folds,
                    "target":       target,
                    "label_mode":   label_mode,
                    "subset":       subset,
                    "is_multiclass": is_multiclass,
                    "metric":       "balanced_accuracy" if is_multiclass else "roc_auc",
                    "leakage_fix":  "fold_internal_x_standardization",
                    "results":      {},
                }
            payload["results"].update(new_model_entries)

            with open(json_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"   Saved: {json_path}", flush=True)

            # ── CSV: read → drop stale rows → append → write ──────────────
            new_rows = []
            for model_name, clf_results in all_results.items():
                sub_info     = subjects_per_model.get(model_name, {})
                n_subjects   = sub_info.get("n_subjects", "?")
                class_counts = sub_info.get("class_counts", {})
                for clf_name, res in clf_results.items():
                    new_rows.append({
                        "model":         model_name,
                        "classifier":    clf_name,
                        "pca_mode":      pca_label,
                        "mean_auc":      round(res["mean_auc"], 4),
                        "std_auc":       round(res["std_auc"],  4),
                        "n_folds_valid": res["n_valid_folds"],
                        "n_subjects":    n_subjects,
                        "n_classes":     len(class_counts),
                        "classes":       str(list(class_counts.keys())),
                        "reduction":     reduction,
                        "target":        target,
                        "label_mode":    label_mode or "",
                        "subset":        subset or "",
                        "is_multiclass": is_multiclass,
                    })

            df_new = pd.DataFrame(new_rows)
            if op.isfile(csv_path) and op.getsize(csv_path) > 0:
                df_old = pd.read_csv(csv_path)
                # Drop any rows belonging to the models we are updating.
                models_updating = set(all_results.keys())
                df_old = df_old[~df_old["model"].isin(models_updating)]
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new

            df.to_csv(csv_path, index=False)
            print(f"   Saved: {csv_path}", flush=True)

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_auc_comparison(
    all_results, output_path, classifiers, title_suffix="Classification", is_multiclass=False
):
    """Plot mean AUC ± std per FM model."""
    model_names = [m for m in FM_MODELS if m in all_results]
    if not model_names:
        model_names = sorted(all_results.keys())

    n_clf = len(classifiers)
    x     = np.arange(len(model_names))
    width = 0.7 / max(n_clf, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.8), 5))

    for ci, clf_name in enumerate(classifiers):
        means = []
        stds  = []
        for model_name in model_names:
            clf_res = all_results.get(model_name, {}).get(clf_name, {})
            means.append(clf_res.get("mean_auc", float("nan")))
            stds.append(clf_res.get("std_auc",  0.0))

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
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    metric_label = "Balanced Accuracy (mean ± std)" if is_multiclass else "AUC (mean ± std)"
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Foundation Model", fontsize=12)
    ax.set_title(
        f"{title_suffix} — leakage-free dim-wise residualization\n"
        "(nested CV, fold-internal X standardization, last-layer embeddings)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


def plot_summary_heatmap(summary_data, output_path):
    """Plot a heatmap of best-classifier mean AUC across all targets × FM models."""
    targets_seen = []
    models_seen  = []
    data_map     = {}

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

    models_seen = [m for m in FM_MODELS if m in models_seen]

    if not targets_seen or not models_seen:
        return

    matrix = np.full((len(targets_seen), len(models_seen)), np.nan)
    for ti, title in enumerate(targets_seen):
        for mi, model in enumerate(models_seen):
            matrix[ti, mi] = data_map.get((title, model), np.nan)

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
        "Leakage-free dim-wise residualization — best-classifier mean score\n"
        "(all targets × all FM models)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free dimension-wise marker residualization of FM embeddings. "
            "Runs both no_pca and pca27 modes in a single job. "
            "Writes progress.txt with a live-updating summary table."
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
        help="Path to nice_scalars_all.csv.",
    )
    parser.add_argument(
        "--patient-labels",
        default=DEFAULT_PATIENT_LABELS,
        help="Path to metadata_patient_labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/res_no_leakage",
        help=(
            "Root output directory.  Results are written to "
            "{output_dir}/no_pca/ and {output_dir}/pca27/."
        ),
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
        choices=_ALL_CLASSIFIERS,
        default=_ALL_CLASSIFIERS,
        help="Classifiers to evaluate (default: all five).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=27,
        help="Number of PCA components for the pca mode (default: 27).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help=(
            "Number of repeated nested CV runs (default: 1). "
            "Results land in cv_nested/{no_pca,pca27}/best_clf/{model}/results.json."
        ),
    )
    parser.add_argument(
        "--cv-output-label",
        default=None,
        help=(
            "Override the cv output subdirectory name (default: 'cv_nested' "
            "for independent mode, 'cv_nested_matched' when --manifest-root "
            "loads successfully). Used by shard submits to write each repeat "
            "to its own dir, e.g. 'cv_nested_shard/repeat_007'."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Root directory for FM-embedding manifests. "
            "When set, also runs cv_nested_matched using the manifest at "
            "{manifest_root}/{model}/doc_patients/MLP_EMBEDDING/"
            "nested_cv_repeated/cv_split_manifest.json."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help="Path to a pre-computed CV splits JSON to share folds across scripts.",
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help="Path to save generated CV splits JSON.",
    )
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
        help="Label mode for cs_* targets.",
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
        help="Run all 25 target × mode configurations.",
    )
    args = parser.parse_args()

    models_to_run = FM_MODELS if args.model == "all" else [args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_targets:
        configs_to_run = TARGET_CONFIGS
    else:
        lm = args.label_mode
        if lm is None and args.target in ("cs_6m", "cs_1y", "cs_2y"):
            lm = "binary"
        title_suffix = args.target
        for tgt, lmode, sub, subdir, tsuffix in TARGET_CONFIGS:
            if tgt == args.target and lmode == lm and sub == args.subset:
                title_suffix = tsuffix
                break
        configs_to_run = [(args.target, lm, args.subset, None, title_suffix)]

    # ── Initialize progress tracker ───────────────────────────────────────────
    pca_labels = [label for label, _ in _PCA_MODES]
    tracker = ProgressTracker(
        output_dir=args.output_dir,
        fms=models_to_run,
        target_configs=configs_to_run,
        pca_labels=pca_labels,
        classifiers=args.classifiers,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Outer loop: target → load labels & generate splits once per target.
    # Inner loops: pca_mode, then fm — both share the same splits.
    all_summary_rows = []

    for cfg_target, cfg_mode, cfg_subset, cfg_subdir, cfg_title in configs_to_run:
        print(
            f"\n{'#' * 65}\n"
            f"  TARGET: {cfg_title!r}\n"
            f"{'#' * 65}",
            flush=True,
        )

        print("Loading patient labels ...", flush=True)
        labels_dict, is_multiclass = load_labels_for_target(
            args.patient_labels, cfg_target, cfg_mode, cfg_subset
        )
        if len(labels_dict) < 10:
            print(
                f"  [SKIP] Too few labelled sessions ({len(labels_dict)}) — skip.",
                flush=True,
            )
            for pca_label in pca_labels:
                for fm in models_to_run:
                    tracker.mark_done(fm, cfg_title, pca_label, None)
            continue

        # Load external splits if provided (shared across all PCA modes)
        precomputed_splits_base = None
        common_sessions_ref_base = None

        if args.splits_file and op.isfile(args.splits_file):
            precomputed_splits_base, common_sessions_ref_base, _ = load_cv_splits(
                args.splits_file
            )
            print(
                f"Loaded {len(precomputed_splits_base)} pre-computed folds "
                f"from {args.splits_file}",
                flush=True,
            )

        # Run both PCA modes; splits are generated once from the first FM's
        # data and then reused across all other FMs and both PCA modes.
        for pca_label, pca_flag in _PCA_MODES:
            if cfg_subdir is not None:
                target_output_dir = op.join(args.output_dir, pca_label, cfg_subdir)
            else:
                target_output_dir = op.join(args.output_dir, pca_label)
            os.makedirs(target_output_dir, exist_ok=True)

            print(
                f"\n  --- PCA mode: {pca_label} ---\n"
                f"  output: {target_output_dir}",
                flush=True,
            )

            # Splits are shared across FMs within a target.
            # They are generated on the first FM that has enough data,
            # then locked in for the rest (same outer folds).
            precomputed_splits   = precomputed_splits_base
            common_sessions_ref  = common_sessions_ref_base
            _splits_saved        = precomputed_splits is not None
            _compute_r2          = (cfg_target == "crs" and pca_label == "no_pca")

            all_results      = {}
            subjects_per_model = {}

            for model_name in models_to_run:
                tracker.mark_running(model_name, cfg_title, pca_label)

                # Load manifest for cv_nested_matched mode (per model + target).
                _manifest_sessions = None
                _precomputed_repeats = None
                _force_clf_per_repeat_fold = None
                # CLI override (e.g. for shard runs) takes priority over the default.
                _cv_output_label = getattr(args, "cv_output_label", None) or "cv_nested"
                if getattr(args, "manifest_root", None):
                    (
                        _manifest_sessions,
                        _precomputed_repeats,
                        _force_clf_per_repeat_fold,
                    ) = load_fm_manifest(
                        args.manifest_root,
                        model_name,
                        cfg_target,
                        target_subdir=cfg_subdir,
                    )
                    if (
                        _precomputed_repeats is not None
                        and not getattr(args, "cv_output_label", None)
                    ):
                        _cv_output_label = "cv_nested_matched"

                try:
                    result = run_model_no_leakage(
                        model_name=model_name,
                        results_root=args.results_root,
                        marker_csv=args.marker_csv,
                        labels_dict=labels_dict,
                        is_multiclass=is_multiclass,
                        output_dir=args.output_dir,
                        reduction=args.reduction,
                        classifiers=args.classifiers,
                        n_folds=args.n_folds,
                        random_state=args.random_state,
                        precomputed_splits=(
                            None if _precomputed_repeats is not None
                            else precomputed_splits
                        ),
                        common_sessions_ref=(
                            _manifest_sessions if _manifest_sessions is not None
                            else common_sessions_ref
                        ),
                        compute_r2=_compute_r2,
                        pca=pca_flag,
                        pca_components=args.pca_components,
                        n_repeats=getattr(args, "n_repeats", 1),
                        force_clf_per_repeat_fold=_force_clf_per_repeat_fold,
                        cv_output_label=_cv_output_label,
                        precomputed_repeats=_precomputed_repeats,
                        target_subdir=cfg_subdir,
                    )
                    model_results, subjects_info, splits_used, common_used, labels_used = result
                except Exception as exc:
                    print(f"  [ERROR] {model_name} failed: {exc}", flush=True)
                    tracker.mark_failed(model_name, cfg_title, pca_label)
                    continue

                if model_results is None:
                    tracker.mark_done(model_name, cfg_title, pca_label, None)
                    continue

                all_results[model_name]      = model_results
                subjects_per_model[model_name] = subjects_info or {}
                tracker.mark_done(model_name, cfg_title, pca_label, model_results)

                for clf_name, res in model_results.items():
                    all_summary_rows.append({
                        "title":    cfg_title,
                        "subdir":   cfg_subdir or "",
                        "model":    model_name,
                        "clf":      clf_name,
                        "pca_mode": pca_label,
                        "mean_auc": res["mean_auc"],
                    })

                # Lock in splits on first successful model for this target.
                # Skipped when n_repeats > 1 (fresh per-repeat folds are
                # generated inside run_nested_cv_no_leakage; locking a single
                # set here would silently override n_repeats to 1).
                if (
                    not _splits_saved
                    and splits_used is not None
                    and getattr(args, "n_repeats", 1) == 1
                ):
                    precomputed_splits  = splits_used
                    common_sessions_ref = common_used
                    save_path = args.save_splits_to
                    if not save_path:
                        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                        splits_dir = op.join(args.output_dir, "cv_splits")
                        os.makedirs(splits_dir, exist_ok=True)
                        tag       = (cfg_subdir or cfg_target).replace("/", "_")
                        save_path = op.join(splits_dir, f"noleakage_splits_{tag}_{ts}.json")
                    save_cv_splits(splits_used, common_used, labels_used, save_path)
                    print(f"  CV splits saved to {save_path}", flush=True)
                    _splits_saved = True

            if not all_results:
                print("  No results — all models were skipped.", flush=True)
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
                pca_label=pca_label,
            )

            print("Generating AUC comparison plot ...", flush=True)
            plot_auc_comparison(
                all_results=all_results,
                output_path=op.join(target_output_dir, "auc_comparison.png"),
                classifiers=args.classifiers,
                title_suffix=f"{cfg_title} [{pca_label}]",
                is_multiclass=is_multiclass,
            )

    # ── Summary heatmaps (one per PCA mode) ──────────────────────────────────
    if args.all_targets and all_summary_rows:
        print("\nGenerating summary heatmaps ...", flush=True)
        for pca_label in pca_labels:
            rows_this_pca = [r for r in all_summary_rows if r["pca_mode"] == pca_label]
            if rows_this_pca:
                plot_summary_heatmap(
                    summary_data=rows_this_pca,
                    output_path=op.join(args.output_dir, f"summary_heatmap_{pca_label}.png"),
                )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
