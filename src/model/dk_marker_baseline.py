"""Baseline classification on pre-computed neurophysiological scalar markers.

==================================================
Marker Baseline Classifier for DoC Prediction
==================================================

This script trains 4 classifiers (SVM, MLP, Random Forest, Kernel Ridge) directly
on the pre-computed scalar markers from
``baseline_stable_20210128_scalars.csv``.  It supports two operating modes:

1. **Classic split** (default):
   Single GroupShuffleSplit outer split (train/test), with an inner
   GroupKFold grid search per model.

2. **Nested CV** (``--full-cv``):
   StratifiedGroupKFold / GroupKFold outer loop (5 folds by default),
   with inner 3-fold GroupKFold grid search per model per fold.

Prediction targets
------------------
All six clinical targets are run in sequence:

- ``crs``            — binary VS vs MCS  (UWS→VS, MCS+/MCS-→MCS)
- ``etiology``       — binary acute vs chronic (+ vs_only / mcs_only subsets)
- ``etiology_code``  — binary ANOXIA vs TBI (+ vs_only / mcs_only subsets)
- ``cs_6m``          — outcome score at 6 months (6 label modes)
- ``cs_1y``          — outcome score at 1 year (6 label modes)
- ``cs_2y``          — outcome score at 2 years (6 label modes)

Outcome label modes (cs_6m / cs_1y / cs_2y)
--------------------------------------------
- ``multiclass``              — VS / MCS / CONSCIOUS / DEATH
- ``binary``                  — VS vs MCS (others excluded)
- ``binary_death``            — DEATH vs NON_DEATH
- ``binary_vs_to_mcs``        — IMPROVED vs OTHER (UWS-baseline subjects only)
- ``binary_mcs_to_conscious`` — IMPROVED vs OTHER (MCS-baseline subjects only)
- ``binary_improvement``      — IMPROVED vs NON_IMPROVED (all subjects; VS
                                baseline improves to MCS/CONSCIOUS; MCS baseline
                                improves to CONSCIOUS)

Group-based CV prevents subject-level data leakage (all sessions of the
same patient stay on the same side of every split).

Reduction map (``--marker-reduction``)
---------------------------------------
- A: ``icm/lg/egi256/trim_mean80``  (default)
- B: ``icm/lg/egi256/std``
- C: ``icm/lg/egi256gfp/trim_mean80``
- D: ``icm/lg/egi256gfp/std``

Output structure
----------------
::

    {main_path}/MARKER_BASELINE/
    ├── crs/
    │   ├── classic_split/{svm,mlp,random_forest,kernel_ridge}/
    │   └── nested_cv/{svm,mlp,random_forest,kernel_ridge}/
    ├── {etiology,etiology_code}/
    │   ├── classic_split/   (all subjects)
    │   ├── nested_cv/
    │   ├── vs_only/classic_split/   (UWS baseline only)
    │   ├── vs_only/nested_cv/
    │   ├── mcs_only/classic_split/  (MCS+/MCS- baseline only)
    │   └── mcs_only/nested_cv/
    └── {cs_6m,cs_1y,cs_2y}/
        ├── multiclass/
        ├── binary/
        ├── binary_death/
        ├── binary_vs_to_mcs/
        ├── binary_mcs_to_conscious/
        └── binary_improvement/
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import os
import os.path as op
import json
import warnings
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GroupKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

try:
    from xgboost import XGBClassifier as _XGBClassifier
except ImportError:
    _XGBClassifier = None


def _json_serialise_baseline(obj):
    """JSON default serialiser for numpy types (dk_marker_baseline.py)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _safe_auc_score(y_true, y_score, fallback_preds):
    """AUC with balanced_accuracy fallback for single-class folds."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return balanced_accuracy_score(y_true, fallback_preds)


try:
    from .kernel_ridge_classifier import KernelRidgeClassifier
    from .cv_utils import generate_nested_cv_folds, load_cv_splits, save_cv_splits
except ImportError:
    from kernel_ridge_classifier import KernelRidgeClassifier
    from cv_utils import generate_nested_cv_folds, load_cv_splits, save_cv_splits

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


# ======================================================================
# Constants
# ======================================================================

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

TARGETS = ["crs", "etiology", "etiology_code", "cs_6m", "cs_1y", "cs_2y"]

MODEL_CONFIGS = {
    "svm": {"name": "SVM"},
    "mlp": {"name": "MLP"},
    "random_forest": {"name": "Random Forest"},
    "kernel_ridge": {"name": "Kernel Ridge"},
    "xgboost": {"name": "XGBoost"},
}


# ======================================================================
# Classifier
# ======================================================================


class BaselineClassifier:
    """Classify DoC patients using pre-computed neurophysiological markers.

    Trains SVM, Random Forest, and Kernel Ridge on scalar markers loaded
    from the baseline CSV.  Group-based CV prevents subject-level leakage.

    Parameters
    ----------
    scalars_csv : str
        Path to semicolon-delimited baseline_stable scalars CSV.
    patient_labels_file : str
        Path to patient_labels.csv (must contain target columns).
    output_dir : str
        Root output directory.
    reduction_letter : str
        One of A/B/C/D — selects the Reduction row from the CSV.
    random_state : int
        Random seed for reproducibility.
    full_cv : bool
        If True, run StratifiedGroupKFold / GroupKFold outer loop.
    n_cv_folds : int
        Number of outer CV folds (default 5).
    """

    def __init__(
        self,
        scalars_csv,
        patient_labels_file,
        output_dir,
        reduction_letter="A",
        random_state=42,
        full_cv=False,
        n_cv_folds=5,
        classifiers=None,
    ):
        if reduction_letter not in REDUCTION_MAP:
            raise ValueError(
                f"Unknown reduction letter {reduction_letter!r}. "
                f"Choose from {list(REDUCTION_MAP)}."
            )
        self.scalars_csv = scalars_csv
        self.patient_labels_file = patient_labels_file
        self.output_dir = output_dir
        self.reduction_letter = reduction_letter
        self.reduction_str = REDUCTION_MAP[reduction_letter]
        self.random_state = random_state
        self.full_cv = full_cv
        self.n_cv_folds = n_cv_folds
        self.classifiers = set(classifiers) if classifiers else set(MODEL_CONFIGS.keys())

        os.makedirs(self.output_dir, exist_ok=True)

        # Set by collect_data
        self.X = None
        self.y_encoded = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.is_binary = True
        self.feature_names = []
        self._baseline_states_dict = None  # populated during binary_improvement loading
        self.baseline_states = None  # aligned to subjects after collect_data

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_scalars(self):
        """Load scalar features from baseline CSV for the chosen reduction.

        Returns
        -------
        features_dict : dict
            Mapping ``subject_session_key -> np.ndarray`` of shape (n_markers,).
        feature_names : list of str
            Marker column names.
        groups_dict : dict
            Mapping ``subject_session_key -> subject_id`` (for group-based CV).
        """
        print(
            f"Loading scalars from: {self.scalars_csv} "
            f"(reduction={self.reduction_str})",
            flush=True,
        )
        df = pd.read_csv(self.scalars_csv, sep=",")

        df_filt = df[df["Reduction"] == self.reduction_str].copy()
        if df_filt.empty:
            raise ValueError(
                f"No rows found for Reduction={self.reduction_str!r} "
                f"in {self.scalars_csv}"
            )

        # Infer marker columns: numeric, not metadata, not unnamed index
        meta_cols = {"Reduction", "Subject", "Subject_ID", "Date", "Label"}
        feature_names = [
            c
            for c in df_filt.columns
            if c not in meta_cols
            and not str(c).startswith("Unnamed")
            and pd.api.types.is_numeric_dtype(df_filt[c])
        ]

        features_dict = {}
        groups_dict = {}

        for _, row in df_filt.iterrows():
            raw_subject = str(row["Subject"])
            # Format: "CC036_1" → subject_id="CC036", session=1
            parts = raw_subject.rsplit("_", 1)
            if len(parts) != 2:
                continue
            subject_id, sesnum = parts
            try:
                key = f"{subject_id}_ses-{int(sesnum):02d}"
            except ValueError:
                continue

            vals = np.array([row[m] for m in feature_names], dtype=float)
            features_dict[key] = vals
            groups_dict[key] = subject_id

        print(
            f"   Loaded scalars for {len(features_dict)} subject/sessions "
            f"({len(feature_names)} markers)",
            flush=True,
        )
        return features_dict, feature_names, groups_dict

    def load_labels_for_target(
        self,
        target,
        binary_outcome=False,
        death_binary=False,
        vs_to_mcs=False,
        mcs_to_conscious=False,
        binary_improvement=False,
        baseline_filter=None,
    ):
        """Load labels for a given prediction target from patient_labels.csv.

        Parameters
        ----------
        target : str
            One of 'crs', 'etiology', 'etiology_code', 'cs_6m', 'cs_1y', 'cs_2y'.
        binary_outcome : bool
            If True and target is an outcome score (cs_6m/cs_1y/cs_2y),
            collapse classes to binary VS vs MCS:
            VS / VS/MCS → VS; MCS+ / MCS- / MCS / CONSCIOUS → MCS;
            DEATH and all others are skipped.
        death_binary : bool
            If True and target is an outcome score, collapse to binary
            DEATH vs NON_DEATH (all subjects included).
        binary_improvement : bool
            If True and target is an outcome score, assign IMPROVED/NON_IMPROVED
            across all subjects: VS baseline improves to MCS/CONSCIOUS; MCS
            baseline improves only to CONSCIOUS. Populates _baseline_states_dict.
        baseline_filter : str or None
            If 'VS', restrict etiology/etiology_code to UWS-baseline subjects only.
            If 'MCS', restrict to MCS+/MCS- baseline subjects only.

        Returns
        -------
        labels_dict : dict
            Mapping ``subject_session_key -> label_str``.
        class_names : list of str
        is_binary : bool
        """
        print(
            f"Loading labels for target '{target}' from: {self.patient_labels_file}",
            flush=True,
        )
        df = pd.read_csv(self.patient_labels_file)
        # Drop rows where subject or session is missing
        df = df.dropna(subset=["subject", "session"])

        labels_dict = {}
        available_states = set()

        if target == "crs":
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
                available_states.add(state)
            is_binary = True

        elif target == "etiology":
            col = "etiology_medical_condition"
            _BL_VS = {"UWS"}
            _BL_MCS = {"MCS+", "MCS-"}
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                val = row[col]
                if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                    continue
                if baseline_filter is not None:
                    bl = row.get("diagnostic_crs_final", None)
                    if baseline_filter == "VS" and (
                        pd.isna(bl) or str(bl).strip() not in _BL_VS
                    ):
                        continue
                    if baseline_filter == "MCS" and (
                        pd.isna(bl) or str(bl).strip() not in _BL_MCS
                    ):
                        continue
                label = str(val).strip().lower()
                key = f"{subject}_{session}"
                labels_dict[key] = label
                available_states.add(label)
            is_binary = True

        elif target == "etiology_code":
            col = "etiology_code"
            _BL_VS = {"UWS"}
            _BL_MCS = {"MCS+", "MCS-"}
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                val = row[col]
                if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                    continue
                if baseline_filter is not None:
                    bl = row.get("diagnostic_crs_final", None)
                    if baseline_filter == "VS" and (
                        pd.isna(bl) or str(bl).strip() not in _BL_VS
                    ):
                        continue
                    if baseline_filter == "MCS" and (
                        pd.isna(bl) or str(bl).strip() not in _BL_MCS
                    ):
                        continue
                label = str(val).strip().lower()
                if label in ("anoxia",):
                    label = "ANOXIA"
                elif label == "tbi":
                    label = "TBI"
                else:
                    continue  # skip all other etiology codes
                key = f"{subject}_{session}"
                labels_dict[key] = label
                available_states.add(label)
            is_binary = True

        elif target in ("cs_6m", "cs_1y", "cs_2y"):
            # Multiclass (4 classes): VS/VS/MCS → VS, MCS+/MCS-/MCS → MCS,
            #   CONSCIOUS → CONSCIOUS, DEATH → DEATH.
            # Binary: keep only VS and MCS (exclude CONSCIOUS and DEATH).
            _VS_STATES = {"VS", "VS/MCS"}
            _MCS_STATES = {"MCS+", "MCS-", "MCS"}
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                val = row[target]
                if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                    continue
                label = str(val).strip()
                if binary_improvement:
                    baseline = row.get("diagnostic_crs_final", None)
                    if pd.isna(baseline):
                        continue
                    bl_str = str(baseline).strip()
                    if bl_str == "UWS":
                        baseline_state = "VS"
                        if label in _MCS_STATES or label == "CONSCIOUS":
                            label = "IMPROVED"
                        else:
                            label = "NON_IMPROVED"
                    elif bl_str in ("MCS+", "MCS-"):
                        baseline_state = "MCS"
                        if label == "CONSCIOUS":
                            label = "IMPROVED"
                        else:
                            label = "NON_IMPROVED"
                    else:
                        continue  # unknown baseline state, skip
                elif vs_to_mcs:
                    baseline_state = row.get("diagnostic_crs_final", None)
                    if pd.isna(baseline_state) or str(baseline_state).strip() != "UWS":
                        continue  # only baseline VS subjects
                    if label in _MCS_STATES or label == "CONSCIOUS":
                        label = "IMPROVED"
                    else:
                        label = "OTHER"
                elif mcs_to_conscious:
                    baseline_state = row.get("diagnostic_crs_final", None)
                    if pd.isna(baseline_state) or str(baseline_state).strip() not in (
                        "MCS+",
                        "MCS-",
                    ):
                        continue  # only baseline MCS subjects
                    if label == "CONSCIOUS":
                        label = "IMPROVED"
                    else:
                        label = "OTHER"
                elif death_binary:
                    if label == "DEATH":
                        label = "DEATH"
                    elif (
                        label in _VS_STATES
                        or label in _MCS_STATES
                        or label == "CONSCIOUS"
                    ):
                        label = "NON_DEATH"
                    else:
                        continue  # skip anything unrecognised
                else:
                    if label in _VS_STATES:
                        label = "VS"
                    elif label in _MCS_STATES:
                        label = "MCS"
                    elif label in ("CONSCIOUS", "DEATH"):
                        if binary_outcome:
                            continue  # exclude from binary run
                    else:
                        continue  # skip anything unrecognised
                key = f"{subject}_{session}"
                labels_dict[key] = label
                available_states.add(label)
                if binary_improvement:
                    if self._baseline_states_dict is None:
                        self._baseline_states_dict = {}
                    self._baseline_states_dict[key] = baseline_state
            is_binary = (
                binary_outcome
                or death_binary
                or vs_to_mcs
                or mcs_to_conscious
                or binary_improvement
            )

        else:
            raise ValueError(f"Unknown target: {target!r}")

        class_names = sorted(available_states)
        print(
            f"   Loaded {len(labels_dict)} subject/sessions, classes: {class_names}",
            flush=True,
        )
        return labels_dict, class_names, is_binary

    def collect_data(
        self,
        target,
        binary_outcome=False,
        death_binary=False,
        vs_to_mcs=False,
        mcs_to_conscious=False,
        binary_improvement=False,
        baseline_filter=None,
    ):
        """Build X, y, subjects arrays for a given prediction target.

        Parameters
        ----------
        target : str
        binary_outcome : bool
            Passed through to ``load_labels_for_target``; collapses outcome
            classes to binary VS/MCS when True.
        death_binary : bool
            Passed through to ``load_labels_for_target``; collapses outcome
            classes to binary DEATH/NON_DEATH when True.
        binary_improvement : bool
            Passed through to ``load_labels_for_target``; builds IMPROVED/NON_IMPROVED
            labels across all baseline states.
        baseline_filter : str or None
            Passed through to ``load_labels_for_target``; restricts etiology samples
            to 'VS' or 'MCS' baseline subjects.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y_encoded : np.ndarray, shape (n_samples,)
        subjects : list of str
        groups : np.ndarray — subject-level group IDs for GroupKFold
        """
        print("Collecting data ...", flush=True)

        self._baseline_states_dict = None  # reset before each collect_data call

        features_dict, feature_names, groups_dict = self.load_scalars()
        labels_dict, class_names, is_binary = self.load_labels_for_target(
            target,
            binary_outcome=binary_outcome,
            death_binary=death_binary,
            vs_to_mcs=vs_to_mcs,
            mcs_to_conscious=mcs_to_conscious,
            binary_improvement=binary_improvement,
            baseline_filter=baseline_filter,
        )

        self.feature_names = feature_names
        self.is_binary = is_binary

        common_keys = sorted(set(features_dict) & set(labels_dict))
        print(
            f"   Scalars: {len(features_dict)} sessions | "
            f"Labels ({target}): {len(labels_dict)} sessions | "
            f"Intersection: {len(common_keys)} sessions",
            flush=True,
        )
        if not common_keys:
            raise ValueError(
                f"No subjects matched between scalars CSV and labels for "
                f"target '{target}'!"
            )

        X_list, y_list, subjects_list, groups_list = [], [], [], []
        for key in common_keys:
            vals = features_dict[key]
            if np.any(np.isnan(vals)):
                continue  # skip rows with NaN markers
            X_list.append(vals)
            y_list.append(labels_dict[key])
            subjects_list.append(key)
            groups_list.append(groups_dict[key])

        if not X_list:
            raise ValueError(
                f"No valid samples remain after NaN filtering for target '{target}'!"
            )

        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.subjects = subjects_list

        # Build baseline_states array for binary_improvement CV stratification
        if self._baseline_states_dict is not None:
            self.baseline_states = [
                self._baseline_states_dict.get(s, "UNKNOWN") for s in subjects_list
            ]
        else:
            self.baseline_states = None

        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_

        groups = np.array(groups_list)

        print(
            f"   Dataset: {self.X.shape[0]} samples x {self.X.shape[1]} features",
            flush=True,
        )
        print(f"   Classes: {list(self.class_names)}", flush=True)
        unique, counts = np.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"      {cls}: {cnt}", flush=True)

        return self.X, self.y_encoded, self.subjects, groups

    # ------------------------------------------------------------------
    # Grid searches (inner 3-fold GroupKFold)
    # ------------------------------------------------------------------

    def _build_inner_cv_splits(self, y_train, groups_train, X_train, precomputed=None):
        """Return a list of (train_idx, val_idx) inner CV splits.

        Uses ``StratifiedGroupKFold`` for binary targets and ``GroupKFold``
        for multiclass.  Inner fold count is ``n_cv_folds - 1`` (capped by
        the number of subjects and smallest class count).

        If *precomputed* is provided those splits are returned directly,
        ensuring all models evaluated on the same outer folds also share
        identical inner splits.
        """
        if precomputed is not None:
            return precomputed

        class_counts = np.bincount(y_train, minlength=2)
        n_groups = len(np.unique(groups_train))
        n_inner = int(
            min(
                self.n_cv_folds - 1,
                n_groups,
                int(np.min(class_counts)),
            )
        )
        n_inner = max(2, n_inner)

        if self.is_binary:
            inner_cv = StratifiedGroupKFold(
                n_splits=n_inner,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            inner_cv = GroupKFold(n_splits=n_inner)

        return list(inner_cv.split(X_train, y_train, groups=groups_train))

    def _grid_search_svm(
        self, X_train, y_train, groups_train, precomputed_inner_splits=None
    ):
        """Inner StratifiedGroupKFold grid search for SVM (with StandardScaler).

        Returns
        -------
        model : Pipeline (StandardScaler + SVC)
        best_params : dict
        """
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        }
        # gamma only relevant for rbf
        gamma_options = {"rbf": ["scale", 0.001, 0.01, 0.1], "linear": [None]}

        inner_splits = self._build_inner_cv_splits(
            y_train, groups_train, X_train, precomputed=precomputed_inner_splits
        )
        best_score = -1.0
        best_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

        for C, kernel in product(param_grid["C"], param_grid["kernel"]):
            for gamma in gamma_options[kernel]:
                scores = []
                for tr_idx, va_idx in inner_splits:
                    svc_kw = dict(
                        C=C,
                        kernel=kernel,
                        class_weight="balanced",
                        probability=True,
                        random_state=self.random_state,
                        max_iter=5000,
                    )
                    if kernel == "rbf":
                        svc_kw["gamma"] = gamma
                    pipe = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("svc", SVC(**svc_kw)),
                        ]
                    )
                    pipe.fit(X_train[tr_idx], y_train[tr_idx])
                    decision_vals = pipe.decision_function(X_train[va_idx])
                    preds = pipe.predict(X_train[va_idx])
                    scores.append(
                        _safe_auc_score(y_train[va_idx], decision_vals, preds)
                    )

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"C": C, "kernel": kernel, "gamma": gamma}

        print(
            f"      SVM best params: {best_params} (auc={best_score:.3f})",
            flush=True,
        )

        svc_kw = dict(
            C=best_params["C"],
            kernel=best_params["kernel"],
            class_weight="balanced",
            probability=True,
            random_state=self.random_state,
            max_iter=5000,
        )
        if best_params["kernel"] == "rbf" and best_params["gamma"] is not None:
            svc_kw["gamma"] = best_params["gamma"]

        best_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(**svc_kw)),
            ]
        )
        best_model.fit(X_train, y_train)
        return best_model, best_params, best_score

    def _grid_search_rf(
        self, X_train, y_train, groups_train, precomputed_inner_splits=None
    ):
        """Inner StratifiedGroupKFold grid search for Random Forest.

        Returns
        -------
        model : RandomForestClassifier
        best_params : dict
        """
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
        inner_splits = self._build_inner_cv_splits(
            y_train, groups_train, X_train, precomputed=precomputed_inner_splits
        )

        best_score = -1.0
        best_params = {"n_estimators": 100, "max_depth": None}

        for n_est, max_d in product(
            param_grid["n_estimators"], param_grid["max_depth"]
        ):
            scores = []
            for tr_idx, va_idx in inner_splits:
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train[tr_idx], y_train[tr_idx])
                proba = rf.predict_proba(X_train[va_idx])[:, 1]
                preds = rf.predict(X_train[va_idx])
                scores.append(_safe_auc_score(y_train[va_idx], proba, preds))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"n_estimators": n_est, "max_depth": max_d}

        print(
            f"      RF best params: {best_params} (auc={best_score:.3f})",
            flush=True,
        )

        best_model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)
        return best_model, best_params, best_score

    def _grid_search_kr(
        self, X_train, y_train, groups_train, precomputed_inner_splits=None
    ):
        """Inner StratifiedGroupKFold grid search for Kernel Ridge (RBF).

        Binary: predictions clipped to [0,1] and thresholded at 0.5.
        Multiclass: one-hot targets, argmax for scoring.

        Returns
        -------
        model : KernelRidgeClassifier
        best_params : dict
        """
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "gamma": [0.001, 0.01, 0.1, 1.0],
        }
        inner_splits = self._build_inner_cv_splits(
            y_train, groups_train, X_train, precomputed=precomputed_inner_splits
        )

        if not self.is_binary:
            n_cls = len(np.unique(y_train))
            y_onehot = np.eye(n_cls)[y_train]
        else:
            y_onehot = None

        best_score = -1.0
        best_params = {"alpha": 1.0, "gamma": 0.01}

        for alpha, gamma in product(param_grid["alpha"], param_grid["gamma"]):
            scores = []
            for tr_idx, va_idx in inner_splits:
                kr = KernelRidgeClassifier(kernel="rbf", alpha=alpha, gamma=gamma)
                if self.is_binary:
                    kr.fit(X_train[tr_idx], y_train[tr_idx])
                    raw = kr.decision_function(X_train[va_idx])
                    pseudo_proba = np.clip(raw, 0, 1)
                    preds = (pseudo_proba >= 0.5).astype(int)
                    scores.append(_safe_auc_score(y_train[va_idx], pseudo_proba, preds))
                else:
                    kr.fit(X_train[tr_idx], y_onehot[tr_idx])
                    raw = kr.predict(X_train[va_idx])
                    preds = np.argmax(raw, axis=1)
                    scores.append(balanced_accuracy_score(y_train[va_idx], preds))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"alpha": alpha, "gamma": gamma}

        print(
            f"      KR best params: {best_params} (auc={best_score:.3f})",
            flush=True,
        )

        best_model = KernelRidgeClassifier(
            kernel="rbf",
            alpha=best_params["alpha"],
            gamma=best_params["gamma"],
        )
        if self.is_binary:
            best_model.fit(X_train, y_train)
        else:
            n_cls = len(np.unique(y_train))
            best_model.fit(X_train, np.eye(n_cls)[y_train])
        return best_model, best_params, best_score

    def _grid_search_mlp(
        self, X_train, y_train, groups_train, precomputed_inner_splits=None
    ):
        """Inner StratifiedGroupKFold grid search for MLP (StandardScaler + MLPClassifier).

        Returns
        -------
        model : Pipeline (StandardScaler + MLPClassifier)
        best_params : dict
        """
        inner_splits = self._build_inner_cv_splits(
            y_train, groups_train, X_train, precomputed=precomputed_inner_splits
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=500,
                        early_stopping=True,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        param_grid = {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 1e-2],
        }
        gs = GridSearchCV(
            pipe,
            param_grid,
            cv=inner_splits,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        print(
            f"      MLP best params: {gs.best_params_} (bal_acc={gs.best_score_:.3f})",
            flush=True,
        )
        return gs.best_estimator_, gs.best_params_, gs.best_score_

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_svm(self, model, X):
        """Return (probabilities, predicted_labels) for SVM pipeline."""
        probs = model.predict_proba(X)
        if self.is_binary:
            probs = probs[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
        return probs, preds

    def _predict_rf(self, model, X):
        """Return (probabilities, predicted_labels) for Random Forest."""
        probs = model.predict_proba(X)
        if self.is_binary:
            probs = probs[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
        return probs, preds

    def _predict_kr(self, model, X):
        """Return (probabilities, predicted_labels) for Kernel Ridge."""
        raw = (
            model.decision_function(X)
            if hasattr(model, "decision_function")
            else model.predict(X)
        )
        if self.is_binary:
            probs = np.clip(raw, 0, 1)
            preds = (probs >= 0.5).astype(int)
        else:
            exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
            probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
        return probs, preds

    def _predict_mlp(self, model, X):
        """Return (probabilities, predicted_labels) for MLP pipeline."""
        probs = model.predict_proba(X)
        if self.is_binary:
            probs = probs[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
        return probs, preds

    def _grid_search_xgb(
        self, X_train, y_train, groups_train, precomputed_inner_splits=None
    ):
        """Inner grid search for XGBoost (StandardScaler + XGBClassifier).

        Returns
        -------
        model : Pipeline (StandardScaler + XGBClassifier)
        best_params : dict
        """
        if _XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        inner_splits = self._build_inner_cv_splits(
            y_train, groups_train, X_train, precomputed=precomputed_inner_splits
        )
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }
        best_score, best_params = -1.0, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
        for n_est, max_d, lr in product(
            param_grid["n_estimators"], param_grid["max_depth"], param_grid["learning_rate"]
        ):
            scores = []
            for tr_idx, va_idx in inner_splits:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("xgb", _XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        eval_metric="logloss", random_state=self.random_state,
                        n_jobs=-1,
                    )),
                ])
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                proba = pipe.predict_proba(X_train[va_idx])[:, 1]
                preds = pipe.predict(X_train[va_idx])
                scores.append(_safe_auc_score(y_train[va_idx], proba, preds))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}
        print(f"      XGB best params: {best_params} (auc={best_score:.3f})", flush=True)
        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", _XGBClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                learning_rate=best_params["learning_rate"],
                eval_metric="logloss", random_state=self.random_state,
                n_jobs=-1,
            )),
        ])
        best_model.fit(X_train, y_train)
        return best_model, best_params, best_score

    def _predict_xgb(self, model, X):
        """Return (probabilities, predicted_labels) for XGBoost pipeline."""
        probs = model.predict_proba(X)
        if self.is_binary:
            probs = probs[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
        return probs, preds

    # ------------------------------------------------------------------
    # Feature importance (permutation, R² scorer)
    # ------------------------------------------------------------------

    def _compute_permutation_importance(
        self, model, model_type, X_test, y_test, n_repeats=10
    ):
        """Compute permutation feature importance using R² as the scorer.

        For binary targets the scorer is R²(y_true, predicted_probability).
        For multiclass the scorer falls back to balanced accuracy.

        Parameters
        ----------
        model : fitted estimator
        model_type : str
            One of 'svm', 'random_forest', 'kernel_ridge'.
        X_test : np.ndarray
        y_test : np.ndarray
        n_repeats : int

        Returns
        -------
        dict
            Mapping ``marker_full_path -> {"mean": float, "std": float,
            "short_name": str}``.
        """
        is_binary = self.is_binary

        if model_type in ("svm", "mlp", "random_forest"):

            def _scorer(m, X, y):
                proba = m.predict_proba(X)
                if is_binary:
                    return r2_score(y.astype(float), proba[:, 1])
                else:
                    return balanced_accuracy_score(y, np.argmax(proba, axis=1))
        else:  # kernel_ridge

            def _scorer(m, X, y):
                raw = m.predict(X)
                if is_binary:
                    proba = np.clip(raw, 0, 1)
                    return r2_score(y.astype(float), proba)
                else:
                    preds = np.argmax(raw, axis=1)
                    return balanced_accuracy_score(y, preds)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring=_scorer,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        importance = {}
        for i, fname in enumerate(self.feature_names):
            short_name = "_".join(fname.split("/")[-2:])
            importance[fname] = {
                "mean": float(result.importances_mean[i]),
                "std": float(result.importances_std[i]),
                "short_name": short_name,
            }
        return importance

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_test_metrics(self, y_test, test_preds, test_probs, subjects_test):
        """Compute standard classification metrics on a test set.

        Handles both binary (1-D probs) and multiclass (2-D probs).
        """
        test_bal_acc = balanced_accuracy_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds)

        test_auc = None
        n_classes_present = len(np.unique(y_test))
        if self.is_binary:
            if n_classes_present == 2:
                try:
                    test_auc = roc_auc_score(y_test, test_probs)
                except Exception:
                    pass
        else:
            if n_classes_present >= 2 and np.ndim(test_probs) == 2:
                try:
                    test_auc = roc_auc_score(
                        y_test,
                        test_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                except Exception:
                    pass

        all_labels = list(range(len(self.class_names)))
        test_precision, test_recall, test_f1, test_support = (
            precision_recall_fscore_support(
                y_test,
                test_preds,
                labels=all_labels,
                average=None,
                zero_division=0,
            )
        )
        test_conf = confusion_matrix(y_test, test_preds, labels=all_labels)
        test_report = classification_report(
            y_test,
            test_preds,
            labels=all_labels,
            target_names=list(self.class_names),
            output_dict=True,
            zero_division=0,
        )

        return {
            "test_accuracy": float(test_acc),
            "test_balanced_accuracy": float(test_bal_acc),
            "test_auc_score": (float(test_auc) if test_auc is not None else None),
            "test_precision": test_precision.tolist(),
            "test_recall": test_recall.tolist(),
            "test_f1_score": test_f1.tolist(),
            "test_support": test_support.tolist(),
            "test_confusion_matrix": test_conf.tolist(),
            "test_classification_report": test_report,
            "class_names": list(self.class_names),
            "subjects_test": subjects_test,
            "y_test_true": y_test.tolist(),
            "y_test_pred": test_preds.tolist(),
            "y_test_probs": (
                test_probs.tolist()
                if isinstance(test_probs, np.ndarray)
                else list(test_probs)
            ),
        }

    def _compute_macro_average(self, accum_y_true, accum_y_pred, accum_y_proba):
        """Compute per-fold metrics and return mean +/- std across folds.

        Parameters
        ----------
        accum_y_true : list of np.ndarray
            One array per fold.
        accum_y_pred : list of np.ndarray
            One array per fold.
        accum_y_proba : list of np.ndarray
            One array per fold.

        Returns
        -------
        dict
            Macro-average metrics with mean, std, and per-fold details.
        """
        n_folds = len(accum_y_true)
        per_fold = []
        accuracies = []
        balanced_accuracies = []
        auc_scores = []
        n_classes = len(self.class_names)
        precisions = np.zeros((n_folds, n_classes))
        recalls = np.zeros((n_folds, n_classes))
        f1_scores = np.zeros((n_folds, n_classes))

        for i in range(n_folds):
            y_true_i = accum_y_true[i]
            y_pred_i = accum_y_pred[i]
            y_proba_i = accum_y_proba[i]
            # Reconstruct per-fold subjects (not needed for metrics)
            dummy_subjects = [f"fold{i}_s{j}" for j in range(len(y_true_i))]

            fold_metrics = self._compute_test_metrics(
                y_true_i, y_pred_i, y_proba_i, dummy_subjects
            )

            accuracies.append(fold_metrics["test_accuracy"])
            balanced_accuracies.append(fold_metrics["test_balanced_accuracy"])
            auc_scores.append(fold_metrics["test_auc_score"])
            precisions[i] = fold_metrics["test_precision"]
            recalls[i] = fold_metrics["test_recall"]
            f1_scores[i] = fold_metrics["test_f1_score"]

            per_fold.append(
                {
                    "fold": i,
                    "accuracy": fold_metrics["test_accuracy"],
                    "balanced_accuracy": fold_metrics["test_balanced_accuracy"],
                    "auc_score": fold_metrics["test_auc_score"],
                    "precision": fold_metrics["test_precision"],
                    "recall": fold_metrics["test_recall"],
                    "f1_score": fold_metrics["test_f1_score"],
                }
            )

        # Filter None AUC values
        valid_aucs = [a for a in auc_scores if a is not None]

        return {
            "accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
            },
            "balanced_accuracy": {
                "mean": float(np.mean(balanced_accuracies)),
                "std": float(np.std(balanced_accuracies)),
            },
            "auc_score": {
                "mean": float(np.mean(valid_aucs)) if valid_aucs else None,
                "std": float(np.std(valid_aucs)) if valid_aucs else None,
            },
            "precision": {
                "mean": precisions.mean(axis=0).tolist(),
                "std": precisions.std(axis=0).tolist(),
            },
            "recall": {
                "mean": recalls.mean(axis=0).tolist(),
                "std": recalls.std(axis=0).tolist(),
            },
            "f1_score": {
                "mean": f1_scores.mean(axis=0).tolist(),
                "std": f1_scores.std(axis=0).tolist(),
            },
            "per_fold_metrics": per_fold,
        }

    def _save_best_clf_results(
        self,
        bc_accum,
        out_dir,
        n_folds,
        n_repeats_block,
        n_samples,
        n_subjects,
    ):
        """Save best-clf-per-fold results to classification_results.json."""
        os.makedirs(out_dir, exist_ok=True)

        y_true_all = np.concatenate(bc_accum["y_true"])
        y_pred_all = np.concatenate(bc_accum["y_pred"])
        y_proba_all = np.concatenate(bc_accum["y_proba"])

        micro = self._compute_test_metrics(
            y_true_all, y_pred_all, y_proba_all, bc_accum["subjects"]
        )
        macro = self._compute_macro_average(
            bc_accum["y_true"], bc_accum["y_pred"], bc_accum["y_proba"]
        )

        fold_info = bc_accum.get("fold_info", [])
        for i, pf in enumerate(macro.get("per_fold_metrics", [])):
            if i < len(fold_info):
                pf["classifier"] = fold_info[i].get("clf", "unknown")
                pf["inner_score"] = fold_info[i].get("inner_score")
                if "repeat" in fold_info[i]:
                    pf["repeat"] = fold_info[i]["repeat"]

        from collections import Counter
        clf_counts = Counter(fi.get("clf", "unknown") for fi in fold_info)

        results = {
            "auc_mean": macro["auc_score"]["mean"],
            "auc_std": macro["auc_score"]["std"],
            "n_repeats": n_repeats_block,
            "n_folds": n_folds,
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "full_cv": True,
            "best_clf_selection": True,
            "classifier_selection_counts": dict(clf_counts),
            "macro_average": macro,
        }
        results.update(micro)

        out_path = op.join(out_dir, "classification_results.json")
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_serialise_baseline)
        print(
            f"   [best-clf] saved to {out_path} "
            f"(macro AUC {macro['auc_score']['mean']:.3f}±{macro['auc_score']['std']:.3f})",
            flush=True,
        )

    def _plot_macro_vs_micro(self, results, model_display_name, output_dir):
        """Generate a separate PNG comparing micro vs macro evaluation."""
        macro = results.get("macro_average")
        if macro is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        cv_tag = ""
        if results.get("full_cv"):
            cv_tag = f" ({results['n_folds']}-fold CV)"
        fig.suptitle(
            f"{model_display_name} — Micro vs Macro Evaluation{cv_tag}",
            fontsize=14,
        )

        # --- Panel 1: Scalar metrics comparison ---
        ax = axes[0]
        metric_names = ["Accuracy", "Bal. Accuracy", "AUC"]
        micro_vals = [
            results["test_accuracy"],
            results["test_balanced_accuracy"],
            results["test_auc_score"] if results["test_auc_score"] is not None else 0,
        ]
        macro_means = [
            macro["accuracy"]["mean"],
            macro["balanced_accuracy"]["mean"],
            macro["auc_score"]["mean"] if macro["auc_score"]["mean"] is not None else 0,
        ]
        macro_stds = [
            macro["accuracy"]["std"],
            macro["balanced_accuracy"]["std"],
            macro["auc_score"]["std"] if macro["auc_score"]["std"] is not None else 0,
        ]

        x = np.arange(len(metric_names))
        width = 0.35
        bars1 = ax.bar(
            x - width / 2, micro_vals, width, label="Micro", color="steelblue"
        )
        bars2 = ax.bar(
            x + width / 2,
            macro_means,
            width,
            yerr=macro_stds,
            label="Macro (mean\u00b1std)",
            color="darkorange",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Scalar Metrics")
        ax.legend()
        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar, std in zip(bars2, macro_stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + std + 0.01,
                f"{h:.3f}\u00b1{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        # --- Panel 2: Per-class metrics ---
        ax = axes[1]
        class_names = results.get("class_names", self.class_names)
        n_classes = len(class_names)
        metric_labels = ["Precision", "Recall", "F1"]
        micro_class = [
            results["test_precision"],
            results["test_recall"],
            results["test_f1_score"],
        ]
        macro_class_mean = [
            macro["precision"]["mean"],
            macro["recall"]["mean"],
            macro["f1_score"]["mean"],
        ]
        macro_class_std = [
            macro["precision"]["std"],
            macro["recall"]["std"],
            macro["f1_score"]["std"],
        ]

        x = np.arange(n_classes)
        total_width = 0.8
        bar_width = total_width / (len(metric_labels) * 2)
        colors_micro = ["#1f77b4", "#2ca02c", "#9467bd"]
        colors_macro = ["#ff7f0e", "#d62728", "#8c564b"]

        for mi, mlabel in enumerate(metric_labels):
            offset_micro = (mi * 2) * bar_width - total_width / 2
            offset_macro = (mi * 2 + 1) * bar_width - total_width / 2
            ax.bar(
                x + offset_micro,
                micro_class[mi],
                bar_width,
                label=f"{mlabel} (micro)" if mi == 0 or True else "",
                color=colors_micro[mi],
                alpha=0.8,
            )
            ax.bar(
                x + offset_macro,
                macro_class_mean[mi],
                bar_width,
                yerr=macro_class_std[mi],
                capsize=3,
                label=f"{mlabel} (macro)" if mi == 0 or True else "",
                color=colors_macro[mi],
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.legend(fontsize=7, ncol=2)

        # --- Panel 3: Per-fold distribution ---
        ax = axes[2]
        per_fold = macro["per_fold_metrics"]
        fold_bal_accs = [f["balanced_accuracy"] for f in per_fold]
        fold_aucs = [f["auc_score"] for f in per_fold if f["auc_score"] is not None]

        positions = []
        data = []
        labels = []
        if fold_bal_accs:
            positions.append(1)
            data.append(fold_bal_accs)
            labels.append("Bal. Acc")
        if fold_aucs:
            positions.append(2)
            data.append(fold_aucs)
            labels.append("AUC")

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True)
            colors_box = ["lightblue", "lightsalmon"]
            for patch, color in zip(bp["boxes"], colors_box[: len(data)]):
                patch.set_facecolor(color)
            # Overlay individual fold points
            for pos, d in zip(positions, data):
                ax.scatter([pos] * len(d), d, color="black", zorder=3, s=30, alpha=0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            # Add micro reference lines
            ax.axhline(
                results["test_balanced_accuracy"],
                color="steelblue",
                ls="--",
                alpha=0.7,
                label="Micro bal. acc",
            )
            if results["test_auc_score"] is not None and fold_aucs:
                ax.axhline(
                    results["test_auc_score"],
                    color="darkorange",
                    ls="--",
                    alpha=0.7,
                    label="Micro AUC",
                )
            ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Per-Fold Distribution")

        plt.tight_layout()
        plot_file = op.join(output_dir, "macro_vs_micro_comparison.png")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Macro vs micro plot saved to: {plot_file}", flush=True)

    # ------------------------------------------------------------------
    # Single-split classification
    # ------------------------------------------------------------------

    def run_classification(
        self,
        target,
        test_size=0.2,
        binary_outcome=False,
        death_binary=False,
        vs_to_mcs=False,
        mcs_to_conscious=False,
        binary_improvement=False,
        baseline_filter=None,
    ):
        """Classic single-split: GroupShuffleSplit outer, inner grid search.

        Parameters
        ----------
        target : str
        test_size : float
        binary_outcome : bool
            If True, collapses outcome classes to binary VS/MCS.
        death_binary : bool
            If True, collapses outcome classes to binary DEATH/NON_DEATH.
        binary_improvement : bool
            If True, builds IMPROVED/NON_IMPROVED labels across all baseline states.
        baseline_filter : str or None
            If 'VS' or 'MCS', restrict etiology samples to that baseline group.
        """
        if binary_improvement:
            mode_tag = " [binary_improvement]"
        elif death_binary:
            mode_tag = " [binary_death]"
        elif binary_outcome:
            mode_tag = " [binary]"
        elif vs_to_mcs:
            mode_tag = " [binary_vs_to_mcs]"
        elif mcs_to_conscious:
            mode_tag = " [binary_mcs_to_conscious]"
        else:
            mode_tag = ""
        if baseline_filter:
            mode_tag += f" [{baseline_filter.lower()}_only]"
        print("=" * 80, flush=True)
        print(
            f"MARKER BASELINE — Classic Split | target={target}{mode_tag} | "
            f"reduction={self.reduction_letter}",
            flush=True,
        )
        print("=" * 80, flush=True)

        np.random.seed(self.random_state)

        X, y, subjects, groups = self.collect_data(
            target,
            binary_outcome=binary_outcome,
            death_binary=death_binary,
            vs_to_mcs=vs_to_mcs,
            mcs_to_conscious=mcs_to_conscious,
            binary_improvement=binary_improvement,
            baseline_filter=baseline_filter,
        )

        unique_groups = np.unique(groups)
        print(
            f"   {len(unique_groups)} unique subjects across {len(subjects)} sessions",
            flush=True,
        )

        # Outer split
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_state
        )
        trainval_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_trainval = X[trainval_idx]
        X_test = X[test_idx]
        y_trainval = y[trainval_idx]
        y_test = y[test_idx]
        groups_trainval = groups[trainval_idx]
        subjects_test = [subjects[i] for i in test_idx]
        subjects_train = [subjects[i] for i in trainval_idx]

        train_groups_set = set(groups_trainval)
        test_groups_set = set(groups[test_idx])
        if train_groups_set & test_groups_set:
            raise ValueError(
                f"Train/test leakage: {train_groups_set & test_groups_set}"
            )

        print(
            f"   Train: {len(X_trainval)} sessions ({len(train_groups_set)} subjects)",
            flush=True,
        )
        print(
            f"   Test:  {len(X_test)} sessions ({len(test_groups_set)} subjects)",
            flush=True,
        )
        for split_name, y_split in [("Train", y_trainval), ("Test", y_test)]:
            u, c = np.unique(y_split, return_counts=True)
            dist = ", ".join(f"{self.class_names[ci]}: {n}" for ci, n in zip(u, c))
            print(f"   {split_name} class distribution: {dist}", flush=True)

        split_dir = op.join(self.output_dir, "classic_split")
        all_results = {}

        # ---- SVM ----
        print("\n   Training SVM (grid search) ...", flush=True)
        svm_model, svm_params = self._grid_search_svm(
            X_trainval, y_trainval, groups_trainval
        )
        svm_probs, svm_preds = self._predict_svm(svm_model, X_test)
        svm_results = self._compute_test_metrics(
            y_test, svm_preds, svm_probs, subjects_test
        )
        svm_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "feature_names": list(self.feature_names),
                "best_params": svm_params,
                "subjects_train": subjects_train,
                "reduction": self.reduction_str,
                "target": target,
            }
        )
        print("   Computing SVM permutation importance ...", flush=True)
        svm_fi = self._compute_permutation_importance(svm_model, "svm", X_test, y_test)
        svm_dir = op.join(split_dir, "svm")
        os.makedirs(svm_dir, exist_ok=True)
        self._save_results(
            svm_results,
            svm_dir,
            model=svm_model,
            model_type="svm",
            feature_importance=svm_fi,
        )
        self._plot_results(
            svm_results,
            "SVM",
            svm_dir,
            best_params=svm_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["svm"] = svm_results

        # ---- MLP ----
        print("\n   Training MLP (grid search) ...", flush=True)
        mlp_model, mlp_params = self._grid_search_mlp(
            X_trainval, y_trainval, groups_trainval
        )
        mlp_probs, mlp_preds = self._predict_mlp(mlp_model, X_test)
        mlp_results = self._compute_test_metrics(
            y_test, mlp_preds, mlp_probs, subjects_test
        )
        mlp_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "feature_names": list(self.feature_names),
                "best_params": {k: str(v) for k, v in mlp_params.items()},
                "subjects_train": subjects_train,
                "reduction": self.reduction_str,
                "target": target,
            }
        )
        print("   Computing MLP permutation importance ...", flush=True)
        mlp_fi = self._compute_permutation_importance(mlp_model, "mlp", X_test, y_test)
        mlp_dir = op.join(split_dir, "mlp")
        os.makedirs(mlp_dir, exist_ok=True)
        self._save_results(
            mlp_results,
            mlp_dir,
            model=mlp_model,
            model_type="mlp",
            feature_importance=mlp_fi,
        )
        self._plot_results(
            mlp_results,
            "MLP",
            mlp_dir,
            best_params=mlp_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["mlp"] = mlp_results

        # ---- Random Forest ----
        print("\n   Training Random Forest (grid search) ...", flush=True)
        rf_model, rf_params = self._grid_search_rf(
            X_trainval, y_trainval, groups_trainval
        )
        rf_probs, rf_preds = self._predict_rf(rf_model, X_test)
        rf_results = self._compute_test_metrics(
            y_test, rf_preds, rf_probs, subjects_test
        )
        rf_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "feature_names": list(self.feature_names),
                "best_params": rf_params,
                "subjects_train": subjects_train,
                "reduction": self.reduction_str,
                "target": target,
            }
        )
        print("   Computing RF permutation importance ...", flush=True)
        rf_fi = self._compute_permutation_importance(
            rf_model, "random_forest", X_test, y_test
        )
        rf_dir = op.join(split_dir, "random_forest")
        os.makedirs(rf_dir, exist_ok=True)
        self._save_results(
            rf_results,
            rf_dir,
            model=rf_model,
            model_type="random_forest",
            feature_importance=rf_fi,
        )
        self._plot_results(
            rf_results,
            "Random Forest",
            rf_dir,
            best_params=rf_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["random_forest"] = rf_results

        # ---- Kernel Ridge ----
        print("\n   Training Kernel Ridge (grid search) ...", flush=True)
        kr_model, kr_params = self._grid_search_kr(
            X_trainval, y_trainval, groups_trainval
        )
        kr_probs, kr_preds = self._predict_kr(kr_model, X_test)
        kr_results = self._compute_test_metrics(
            y_test, kr_preds, kr_probs, subjects_test
        )
        kr_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "feature_names": list(self.feature_names),
                "best_params": kr_params,
                "subjects_train": subjects_train,
                "reduction": self.reduction_str,
                "target": target,
            }
        )
        print("   Computing KR permutation importance ...", flush=True)
        kr_fi = self._compute_permutation_importance(
            kr_model, "kernel_ridge", X_test, y_test
        )
        kr_dir = op.join(split_dir, "kernel_ridge")
        os.makedirs(kr_dir, exist_ok=True)
        self._save_results(
            kr_results,
            kr_dir,
            model=kr_model,
            model_type="kernel_ridge",
            feature_importance=kr_fi,
        )
        self._plot_results(
            kr_results,
            "Kernel Ridge",
            kr_dir,
            best_params=kr_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["kernel_ridge"] = kr_results

        # Summary
        print("\n" + "=" * 80, flush=True)
        print(
            f"MARKER BASELINE SUMMARY — classic_split | target={target}",
            flush=True,
        )
        print("=" * 80, flush=True)
        for mk, cfg in MODEL_CONFIGS.items():
            r = all_results[mk]
            auc_str = (
                f"{r['test_auc_score']:.3f}"
                if r["test_auc_score"] is not None
                else "N/A"
            )
            print(
                f"   {cfg['name']:15s}  "
                f"bal_acc={r['test_balanced_accuracy']:.3f}  "
                f"AUC={auc_str}",
                flush=True,
            )
        print(f"   Results saved to: {split_dir}", flush=True)
        print("=" * 80, flush=True)

        return all_results

    # ------------------------------------------------------------------
    # Nested CV classification
    # ------------------------------------------------------------------

    def run_full_cv(
        self,
        target,
        binary_outcome=False,
        death_binary=False,
        vs_to_mcs=False,
        mcs_to_conscious=False,
        binary_improvement=False,
        baseline_filter=None,
        precomputed_splits=None,
        common_sessions=None,
        n_repeats=1,
        cv_output_subdir=None,
    ):
        """Nested CV: StratifiedGroupKFold / GroupKFold outer loop.

        Parameters
        ----------
        target : str
        binary_outcome : bool
            If True, collapses outcome classes to binary VS/MCS.
        death_binary : bool
            If True, collapses outcome classes to binary DEATH/NON_DEATH.
        binary_improvement : bool
            If True, builds IMPROVED/NON_IMPROVED labels; CV is stratified by
            baseline state (VS/MCS) to ensure balanced folds.
        baseline_filter : str or None
            If 'VS' or 'MCS', restrict etiology samples to that baseline group.
        precomputed_splits:
            Optional list of fold dicts from :func:`cv_utils.generate_nested_cv_folds`.
            When provided (with *common_sessions*), all models use identical fold
            assignments so results are directly comparable across embedding types.
        common_sessions:
            Ordered session keys matching the rows expected by *precomputed_splits*.
            Required when *precomputed_splits* is not None.
        """
        n_folds = self.n_cv_folds
        if binary_improvement:
            mode_tag = " [binary_improvement]"
        elif death_binary:
            mode_tag = " [binary_death]"
        elif binary_outcome:
            mode_tag = " [binary]"
        elif vs_to_mcs:
            mode_tag = " [binary_vs_to_mcs]"
        elif mcs_to_conscious:
            mode_tag = " [binary_mcs_to_conscious]"
        else:
            mode_tag = ""
        if baseline_filter:
            mode_tag += f" [{baseline_filter.lower()}_only]"

        print("=" * 80, flush=True)
        print(
            f"MARKER BASELINE — {n_folds}-Fold Nested CV | target={target}{mode_tag} | "
            f"reduction={self.reduction_letter}",
            flush=True,
        )
        print("=" * 80, flush=True)

        np.random.seed(self.random_state)

        X, y, subjects, groups = self.collect_data(
            target,
            binary_outcome=binary_outcome,
            death_binary=death_binary,
            vs_to_mcs=vs_to_mcs,
            mcs_to_conscious=mcs_to_conscious,
            binary_improvement=binary_improvement,
            baseline_filter=baseline_filter,
        )

        # Filter and reorder to common session pool when precomputed splits provided.
        # Outcome targets have fewer sessions than the CRS pool used to build the
        # splits — silently use the intersection and remap fold indices rather than
        # raising an error.
        if common_sessions is not None:
            session_to_idx = {s: i for i, s in enumerate(subjects)}
            missing_count = sum(1 for s in common_sessions if s not in session_to_idx)
            if missing_count:
                print(
                    f"   Note: {missing_count} of {len(common_sessions)} common sessions "
                    f"not available for target '{target}' — using intersection only.",
                    flush=True,
                )
            filtered_common = [s for s in common_sessions if s in session_to_idx]
            keep = [session_to_idx[s] for s in filtered_common]
            X = X[keep]
            y = y[keep]
            if self.baseline_states is not None:
                self.baseline_states = [self.baseline_states[i] for i in keep]
            subjects = [subjects[i] for i in keep]
            groups = groups[keep]

            # Remap precomputed fold indices from the full common_sessions ordering
            # to the filtered ordering so they remain valid for this target.
            if precomputed_splits is not None:
                old_pos = {s: i for i, s in enumerate(common_sessions)}
                old_to_new = {
                    old_pos[s]: new_i for new_i, s in enumerate(filtered_common)
                }
                remapped = []
                for fold in precomputed_splits:
                    new_train = np.array(
                        [old_to_new[i] for i in fold["train_idx"] if i in old_to_new],
                        dtype=int,
                    )
                    new_test = np.array(
                        [old_to_new[i] for i in fold["test_idx"] if i in old_to_new],
                        dtype=int,
                    )
                    remapped.append(
                        {
                            "train_idx": new_train,
                            "test_idx": new_test,
                            "inner_splits": None,  # recomputed per fold
                        }
                    )
                precomputed_splits = remapped

        unique_groups = np.unique(groups)
        n_unique = len(unique_groups)
        print(
            f"   {n_unique} unique subjects across {len(subjects)} sessions",
            flush=True,
        )

        # Build session list for manifest.
        _sessions_for_cv = list(subjects)
        _labels_for_cv = {s: int(y[i]) for i, s in enumerate(_sessions_for_cv)}

        # Precomputed splits force single repeat.
        if precomputed_splits is not None and n_repeats > 1:
            print(
                "   [WARN] n_repeats > 1 ignored when precomputed_splits are provided.",
                flush=True,
            )
            n_repeats = 1

        # Output base directory.
        if cv_output_subdir is not None:
            _cv_base_name = cv_output_subdir
        elif n_repeats > 1:
            _cv_base_name = "nested_cv_repeated"
        else:
            _cv_base_name = "nested_cv"
        _cv_base_dir = op.join(self.output_dir, _cv_base_name)
        os.makedirs(_cv_base_dir, exist_ok=True)

        # Build repeat list.
        if precomputed_splits is not None:
            if common_sessions is None:
                raise ValueError(
                    "common_sessions must be provided together with precomputed_splits"
                )
            print(
                f"   Using {len(precomputed_splits)} pre-computed folds (shared across models)",
                flush=True,
            )
            _repeats_data = [(0, precomputed_splits, self.random_state)]
        else:
            _repeats_data = []
            for _r in range(n_repeats):
                _seed_r = self.random_state + _r
                _eff_r = min(n_folds, n_unique)
                if _eff_r < 2:
                    raise ValueError(
                        f"Only {n_unique} unique subjects — cannot perform "
                        f"{n_folds}-fold CV"
                    )
                if self.is_binary:
                    _strat_y = y
                    if binary_improvement and self.baseline_states is not None:
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _strat_y = _LE().fit_transform(self.baseline_states)
                    _sgkf = StratifiedGroupKFold(
                        n_splits=_eff_r, shuffle=True, random_state=_seed_r
                    )
                    _folds_r = [
                        {"train_idx": tr, "test_idx": te, "inner_splits": None}
                        for tr, te in _sgkf.split(X, _strat_y, groups=groups)
                    ]
                else:
                    _gkf = GroupKFold(n_splits=_eff_r)
                    _folds_r = [
                        {"train_idx": tr, "test_idx": te, "inner_splits": None}
                        for tr, te in _gkf.split(X, y, groups=groups)
                    ]
                _repeats_data.append((_r, _folds_r, _seed_r))

        # Global best-clf accumulator.
        _global_bc: dict = {
            "y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_info": []
        }
        _manifest_repeats: list = []

        # ======================================================================
        # Repeat loop
        # ======================================================================
        for _repeat_idx, folds, _repeat_seed in _repeats_data:
            effective_folds = len(folds)

            if n_repeats > 1:
                print(f"\n{'=' * 60}", flush=True)
                print(
                    f"  REPEAT {_repeat_idx + 1}/{n_repeats}  (seed={_repeat_seed})",
                    flush=True,
                )
                print(f"{'=' * 60}", flush=True)

            _repeat_dir = (
                _cv_base_dir if n_repeats == 1
                else op.join(_cv_base_dir, f"repeat_{_repeat_idx:03d}")
            )

            accum = {
                mk: {
                    "y_true": [],
                    "y_pred": [],
                    "y_proba": [],
                    "subjects": [],
                    "fi_means": [],
                }
                for mk in MODEL_CONFIGS
                if mk in self.classifiers
            }

            _repeat_bc: dict = {
                "y_true": [], "y_pred": [], "y_proba": [], "subjects": [], "fold_info": []
            }
            _best_clf_per_fold: list = []

            # ------------------------------------------------------------------
            # Fold loop
            # ------------------------------------------------------------------
            for fold_idx, fold in enumerate(folds):
                train_idx = fold["train_idx"]
                test_idx = fold["test_idx"]
                inner_splits_for_fold = fold.get("inner_splits")

                print(f"\n{'─' * 60}", flush=True)
                print(f"  FOLD {fold_idx + 1}/{effective_folds}", flush=True)
                print(f"{'─' * 60}", flush=True)

                X_train_fold = X[train_idx]
                X_test_fold = X[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                groups_train_fold = groups[train_idx]
                subjects_test_fold = [subjects[i] for i in test_idx]

                train_subj = set(groups[train_idx])
                test_subj = set(groups[test_idx])
                if train_subj & test_subj:
                    raise ValueError(
                        f"Fold {fold_idx}: subject leakage: {train_subj & test_subj}"
                    )
                # Save fold split for SHAP reproducibility.
                _splits_dir = op.join(_repeat_dir, "fold_splits")
                os.makedirs(_splits_dir, exist_ok=True)
                with open(op.join(_splits_dir, f"fold_{fold_idx:02d}.json"), "w") as _f:
                    json.dump(
                        {
                            "train_subjects": [subjects[i] for i in train_idx],
                            "test_subjects": subjects_test_fold,
                        },
                        _f,
                        indent=2,
                    )

                print(
                    f"   Train: {len(train_subj)} subjects ({len(train_idx)} sessions)",
                    flush=True,
                )
                print(
                    f"   Test:  {len(test_subj)} subjects ({len(test_idx)} sessions)",
                    flush=True,
                )

                _fold_inner_scores: dict = {}
                _fold_probas: dict = {}
                _fold_preds_map: dict = {}

                # ---- SVM ----
                if "svm" in self.classifiers:
                    print("   Training SVM ...", flush=True)
                    svm_model, _, svm_score = self._grid_search_svm(
                        X_train_fold, y_train_fold, groups_train_fold,
                        precomputed_inner_splits=inner_splits_for_fold,
                    )
                    _fold_inner_scores["svm"] = float(svm_score)
                    _fold_dir = op.join(_repeat_dir, "svm")
                    os.makedirs(_fold_dir, exist_ok=True)
                    joblib.dump(svm_model, op.join(_fold_dir, f"fold_{fold_idx:02d}_model.joblib"))
                    svm_probs, svm_preds = self._predict_svm(svm_model, X_test_fold)
                    _fold_probas["svm"] = svm_probs
                    _fold_preds_map["svm"] = svm_preds
                    accum["svm"]["y_true"].append(y_test_fold)
                    accum["svm"]["y_pred"].append(svm_preds)
                    accum["svm"]["y_proba"].append(svm_probs)
                    accum["svm"]["subjects"].extend(subjects_test_fold)
                    svm_fi_fold = self._compute_permutation_importance(
                        svm_model, "svm", X_test_fold, y_test_fold, n_repeats=5
                    )
                    accum["svm"]["fi_means"].append(
                        [svm_fi_fold[f]["mean"] for f in self.feature_names]
                    )
                    print(f"   SVM fold bal_acc: "
                          f"{balanced_accuracy_score(y_test_fold, svm_preds):.3f}", flush=True)

                # ---- MLP ----
                if "mlp" in self.classifiers:
                    print("   Training MLP ...", flush=True)
                    mlp_model, _, mlp_score = self._grid_search_mlp(
                        X_train_fold, y_train_fold, groups_train_fold,
                        precomputed_inner_splits=inner_splits_for_fold,
                    )
                    _fold_inner_scores["mlp"] = float(mlp_score)
                    _fold_dir = op.join(_repeat_dir, "mlp")
                    os.makedirs(_fold_dir, exist_ok=True)
                    joblib.dump(mlp_model, op.join(_fold_dir, f"fold_{fold_idx:02d}_model.joblib"))
                    mlp_probs, mlp_preds = self._predict_mlp(mlp_model, X_test_fold)
                    _fold_probas["mlp"] = mlp_probs
                    _fold_preds_map["mlp"] = mlp_preds
                    accum["mlp"]["y_true"].append(y_test_fold)
                    accum["mlp"]["y_pred"].append(mlp_preds)
                    accum["mlp"]["y_proba"].append(mlp_probs)
                    accum["mlp"]["subjects"].extend(subjects_test_fold)
                    mlp_fi_fold = self._compute_permutation_importance(
                        mlp_model, "mlp", X_test_fold, y_test_fold, n_repeats=5
                    )
                    accum["mlp"]["fi_means"].append(
                        [mlp_fi_fold[f]["mean"] for f in self.feature_names]
                    )
                    print(f"   MLP fold bal_acc: "
                          f"{balanced_accuracy_score(y_test_fold, mlp_preds):.3f}", flush=True)

                # ---- Random Forest ----
                if "random_forest" in self.classifiers:
                    print("   Training Random Forest ...", flush=True)
                    rf_model, _, rf_score = self._grid_search_rf(
                        X_train_fold, y_train_fold, groups_train_fold,
                        precomputed_inner_splits=inner_splits_for_fold,
                    )
                    _fold_inner_scores["random_forest"] = float(rf_score)
                    _fold_dir = op.join(_repeat_dir, "random_forest")
                    os.makedirs(_fold_dir, exist_ok=True)
                    joblib.dump(rf_model, op.join(_fold_dir, f"fold_{fold_idx:02d}_model.joblib"))
                    rf_probs, rf_preds = self._predict_rf(rf_model, X_test_fold)
                    _fold_probas["random_forest"] = rf_probs
                    _fold_preds_map["random_forest"] = rf_preds
                    accum["random_forest"]["y_true"].append(y_test_fold)
                    accum["random_forest"]["y_pred"].append(rf_preds)
                    accum["random_forest"]["y_proba"].append(rf_probs)
                    accum["random_forest"]["subjects"].extend(subjects_test_fold)
                    rf_fi_fold = self._compute_permutation_importance(
                        rf_model, "random_forest", X_test_fold, y_test_fold, n_repeats=5
                    )
                    accum["random_forest"]["fi_means"].append(
                        [rf_fi_fold[f]["mean"] for f in self.feature_names]
                    )
                    print(f"   RF fold bal_acc: "
                          f"{balanced_accuracy_score(y_test_fold, rf_preds):.3f}", flush=True)

                # ---- Kernel Ridge ----
                if "kernel_ridge" in self.classifiers:
                    print("   Training Kernel Ridge ...", flush=True)
                    kr_model, _, kr_score = self._grid_search_kr(
                        X_train_fold, y_train_fold, groups_train_fold,
                        precomputed_inner_splits=inner_splits_for_fold,
                    )
                    _fold_inner_scores["kernel_ridge"] = float(kr_score)
                    _fold_dir = op.join(_repeat_dir, "kernel_ridge")
                    os.makedirs(_fold_dir, exist_ok=True)
                    joblib.dump(kr_model, op.join(_fold_dir, f"fold_{fold_idx:02d}_model.joblib"))
                    kr_probs, kr_preds = self._predict_kr(kr_model, X_test_fold)
                    _fold_probas["kernel_ridge"] = kr_probs
                    _fold_preds_map["kernel_ridge"] = kr_preds
                    accum["kernel_ridge"]["y_true"].append(y_test_fold)
                    accum["kernel_ridge"]["y_pred"].append(kr_preds)
                    accum["kernel_ridge"]["y_proba"].append(kr_probs)
                    accum["kernel_ridge"]["subjects"].extend(subjects_test_fold)
                    kr_fi_fold = self._compute_permutation_importance(
                        kr_model, "kernel_ridge", X_test_fold, y_test_fold, n_repeats=5
                    )
                    accum["kernel_ridge"]["fi_means"].append(
                        [kr_fi_fold[f]["mean"] for f in self.feature_names]
                    )
                    print(f"   KR fold bal_acc: "
                          f"{balanced_accuracy_score(y_test_fold, kr_preds):.3f}", flush=True)

                # ---- XGBoost ----
                if "xgboost" in self.classifiers and _XGBClassifier is not None:
                    print("   Training XGBoost ...", flush=True)
                    xgb_model, _, xgb_score = self._grid_search_xgb(
                        X_train_fold, y_train_fold, groups_train_fold,
                        precomputed_inner_splits=inner_splits_for_fold,
                    )
                    _fold_inner_scores["xgboost"] = float(xgb_score)
                    _fold_dir = op.join(_repeat_dir, "xgboost")
                    os.makedirs(_fold_dir, exist_ok=True)
                    joblib.dump(xgb_model, op.join(_fold_dir, f"fold_{fold_idx:02d}_model.joblib"))
                    xgb_probs, xgb_preds = self._predict_xgb(xgb_model, X_test_fold)
                    _fold_probas["xgboost"] = xgb_probs
                    _fold_preds_map["xgboost"] = xgb_preds
                    accum["xgboost"]["y_true"].append(y_test_fold)
                    accum["xgboost"]["y_pred"].append(xgb_preds)
                    accum["xgboost"]["y_proba"].append(xgb_probs)
                    accum["xgboost"]["subjects"].extend(subjects_test_fold)
                    accum["xgboost"]["fi_means"].append([0.0] * len(self.feature_names))
                    print(f"   XGB fold bal_acc: "
                          f"{balanced_accuracy_score(y_test_fold, xgb_preds):.3f}", flush=True)

                # -- Per-fold best-classifier selection -------------------------
                if _fold_inner_scores:
                    _best_clf_name = max(_fold_inner_scores, key=_fold_inner_scores.get)
                    print(
                        f"   [best-clf] {_best_clf_name} selected "
                        f"(inner score={_fold_inner_scores[_best_clf_name]:.3f})",
                        flush=True,
                    )
                    _sel_info = {
                        "fold": fold_idx,
                        "clf": _best_clf_name,
                        "inner_score": _fold_inner_scores[_best_clf_name],
                        "all_inner_scores": dict(_fold_inner_scores),
                    }
                    _repeat_bc["y_true"].append(y_test_fold)
                    _repeat_bc["y_pred"].append(_fold_preds_map[_best_clf_name])
                    _repeat_bc["y_proba"].append(_fold_probas[_best_clf_name])
                    _repeat_bc["subjects"].extend(subjects_test_fold)
                    _repeat_bc["fold_info"].append(_sel_info)
                    _best_clf_per_fold.append(_sel_info)

            # ------------------------------------------------------------------
            # Aggregate per-classifier results
            # ------------------------------------------------------------------
            print(f"\n{'=' * 60}", flush=True)
            print("AGGREGATING RESULTS ACROSS ALL FOLDS", flush=True)
            print(f"{'=' * 60}", flush=True)

            for mk, cfg in MODEL_CONFIGS.items():
                if mk not in self.classifiers:
                    continue
                y_true_all = np.concatenate(accum[mk]["y_true"])
                y_pred_all = np.concatenate(accum[mk]["y_pred"])
                y_proba_all = np.concatenate(accum[mk]["y_proba"])
                subjects_all = accum[mk]["subjects"]

                results = self._compute_test_metrics(
                    y_true_all, y_pred_all, y_proba_all, subjects_all
                )
                results.update(
                    {
                        "n_samples": len(X),
                        "n_features": int(X.shape[1]),
                        "feature_names": list(self.feature_names),
                        "n_subjects": int(n_unique),
                        "n_folds": effective_folds,
                        "full_cv": True,
                        "reduction": self.reduction_str,
                        "target": target,
                    }
                )

                fi_matrix = np.array(accum[mk]["fi_means"])
                fi_mean = fi_matrix.mean(axis=0)
                fi_std = fi_matrix.std(axis=0)
                cv_fi = {
                    fname: {
                        "mean": float(fi_mean[i]),
                        "std": float(fi_std[i]),
                        "short_name": "_".join(fname.split("/")[-2:]),
                    }
                    for i, fname in enumerate(self.feature_names)
                }

                macro_results = self._compute_macro_average(
                    accum[mk]["y_true"],
                    accum[mk]["y_pred"],
                    accum[mk]["y_proba"],
                )
                results["macro_average"] = macro_results

                model_dir = op.join(_repeat_dir, mk)
                os.makedirs(model_dir, exist_ok=True)
                self._save_results(
                    results, model_dir, model=None, model_type=mk,
                    feature_importance=cv_fi,
                )
                self._plot_results(results, cfg["name"], model_dir)
                self._plot_macro_vs_micro(results, cfg["name"], model_dir)

                micro_auc_str = (
                    f"{results['test_auc_score']:.3f}"
                    if results["test_auc_score"] is not None
                    else "N/A"
                )
                macro_ba = macro_results["balanced_accuracy"]
                macro_auc = macro_results["auc_score"]
                macro_auc_str = (
                    f"{macro_auc['mean']:.3f}\u00b1{macro_auc['std']:.3f}"
                    if macro_auc["mean"] is not None
                    else "N/A"
                )
                print(
                    f"   {cfg['name']:15s}  "
                    f"micro bal_acc={results['test_balanced_accuracy']:.3f}  "
                    f"macro bal_acc={macro_ba['mean']:.3f}\u00b1{macro_ba['std']:.3f}  "
                    f"micro AUC={micro_auc_str}  "
                    f"macro AUC={macro_auc_str}",
                    flush=True,
                )

            # -- Save best-clf results for this repeat -------------------------
            if _repeat_bc["y_true"]:
                self._save_best_clf_results(
                    _repeat_bc,
                    op.join(_repeat_dir, "best_clf"),
                    n_folds=effective_folds,
                    n_repeats_block=1,
                    n_samples=len(X),
                    n_subjects=int(n_unique),
                )

            # Accumulate into global store.
            _global_bc["y_true"].extend(_repeat_bc["y_true"])
            _global_bc["y_pred"].extend(_repeat_bc["y_pred"])
            _global_bc["y_proba"].extend(_repeat_bc["y_proba"])
            _global_bc["subjects"].extend(_repeat_bc["subjects"])
            _global_bc["fold_info"].extend(
                [{**fi, "repeat": _repeat_idx} for fi in _repeat_bc["fold_info"]]
            )

            # Manifest entry.
            _manifest_repeats.append(
                {
                    "repeat_idx": _repeat_idx,
                    "random_state": _repeat_seed,
                    "folds": [
                        {
                            "train_idx": f["train_idx"].tolist(),
                            "test_idx": f["test_idx"].tolist(),
                            "train_subjects": f.get("train_subjects") or [],
                            "test_subjects": f.get("test_subjects") or [],
                        }
                        for f in folds
                    ],
                    "best_clf_per_fold": _best_clf_per_fold,
                }
            )

        # ======================================================================
        # Global aggregate (only for n_repeats > 1)
        # ======================================================================
        if n_repeats > 1 and _global_bc["y_true"]:
            self._save_best_clf_results(
                _global_bc,
                op.join(_cv_base_dir, "best_clf"),
                n_folds=n_folds,
                n_repeats_block=n_repeats,
                n_samples=len(X),
                n_subjects=int(n_unique),
            )

        # Save manifest.
        _manifest = {
            "n_repeats": n_repeats,
            "base_random_state": self.random_state,
            "n_folds": n_folds,
            "common_sessions": _sessions_for_cv,
            "repeats": _manifest_repeats,
        }
        with open(op.join(_cv_base_dir, "cv_split_manifest.json"), "w") as fh:
            json.dump(_manifest, fh, indent=2)

        print(f"\n   Results saved to: {_cv_base_dir}", flush=True)
        print("=" * 80, flush=True)
    # ------------------------------------------------------------------
    # Multi-target runner
    # ------------------------------------------------------------------

    def run_all_targets(
        self,
        test_size=0.2,
        precomputed_splits=None,
        common_sessions=None,
        n_repeats=1,
        cv_output_subdir=None,
    ):
        """Run classification for all targets.

        For outcome targets (cs_6m, cs_1y, cs_2y) six runs are performed:
          - ``{target}/multiclass/``              — full multi-class
          - ``{target}/binary/``                  — VS vs MCS binary collapse
          - ``{target}/binary_death/``            — DEATH vs NON_DEATH
          - ``{target}/binary_vs_to_mcs/``        — VS→MCS transition only
          - ``{target}/binary_mcs_to_conscious/`` — MCS→CONSCIOUS transition only
          - ``{target}/binary_improvement/``      — IMPROVED vs NON_IMPROVED (all)

        For etiology targets (etiology, etiology_code) three runs are performed:
          - ``{target}/``          — all subjects
          - ``{target}/vs_only/``  — UWS baseline subjects only
          - ``{target}/mcs_only/`` — MCS+/MCS- baseline subjects only

        For the crs target a single run is used.

        Parameters
        ----------
        precomputed_splits, common_sessions :
            When provided, all ``run_full_cv`` calls use the same fold
            assignments (CRS-based, shared across targets).
        """
        _OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}
        _ETIOLOGY_TARGETS = {"etiology", "etiology_code"}
        base_output_dir = self.output_dir

        for target in TARGETS:
            print(f"\n{'=' * 80}", flush=True)
            print(f"TARGET: {target}", flush=True)
            print(f"{'=' * 80}", flush=True)

            if target in _OUTCOME_TARGETS:
                modes = [
                    # (mode_name, binary_outcome, death_binary, vs_to_mcs,
                    #  mcs_to_conscious, binary_improvement, baseline_filter)
                    ("multiclass", False, False, False, False, False, None),
                    ("binary", True, False, False, False, False, None),
                    ("binary_death", False, True, False, False, False, None),
                    ("binary_vs_to_mcs", False, False, True, False, False, None),
                    ("binary_mcs_to_conscious", False, False, False, True, False, None),
                    ("binary_improvement", False, False, False, False, True, None),
                ]
                for (
                    mode,
                    binary_outcome,
                    death_binary,
                    vs_to_mcs,
                    mcs_to_conscious,
                    binary_improvement,
                    baseline_filter,
                ) in modes:
                    print(f"\n--- {target} / {mode} ---", flush=True)
                    self.output_dir = op.join(base_output_dir, target, mode)
                    os.makedirs(self.output_dir, exist_ok=True)
                    try:
                        if self.full_cv:
                            self.run_full_cv(
                                target,
                                binary_outcome=binary_outcome,
                                death_binary=death_binary,
                                vs_to_mcs=vs_to_mcs,
                                mcs_to_conscious=mcs_to_conscious,
                                binary_improvement=binary_improvement,
                                baseline_filter=baseline_filter,
                                precomputed_splits=precomputed_splits,
                                common_sessions=common_sessions,
                                n_repeats=n_repeats,
                                cv_output_subdir=cv_output_subdir,
                            )
                        else:
                            self.run_classification(
                                target,
                                test_size,
                                binary_outcome=binary_outcome,
                                death_binary=death_binary,
                                vs_to_mcs=vs_to_mcs,
                                mcs_to_conscious=mcs_to_conscious,
                                binary_improvement=binary_improvement,
                                baseline_filter=baseline_filter,
                            )
                    except ValueError as exc:
                        print(
                            f"   Skipping target '{target}' ({mode}): {exc}",
                            flush=True,
                        )

            elif target in _ETIOLOGY_TARGETS:
                etiology_modes = [
                    # (mode_name, baseline_filter)
                    ("", None),
                    ("vs_only", "VS"),
                    ("mcs_only", "MCS"),
                ]
                for mode_name, bl_filter in etiology_modes:
                    subdir = (
                        op.join(base_output_dir, target, mode_name)
                        if mode_name
                        else op.join(base_output_dir, target)
                    )
                    label = f" / {mode_name}" if mode_name else ""
                    print(f"\n--- {target}{label} ---", flush=True)
                    self.output_dir = subdir
                    os.makedirs(self.output_dir, exist_ok=True)
                    try:
                        if self.full_cv:
                            self.run_full_cv(
                                target,
                                baseline_filter=bl_filter,
                                precomputed_splits=precomputed_splits,
                                common_sessions=common_sessions,
                                n_repeats=n_repeats,
                                cv_output_subdir=cv_output_subdir,
                            )
                        else:
                            self.run_classification(
                                target,
                                test_size,
                                baseline_filter=bl_filter,
                            )
                    except ValueError as exc:
                        print(f"   Skipping target '{target}'{label}: {exc}", flush=True)

            else:
                self.output_dir = op.join(base_output_dir, target)
                os.makedirs(self.output_dir, exist_ok=True)
                try:
                    if self.full_cv:
                        self.run_full_cv(
                            target,
                            precomputed_splits=precomputed_splits,
                            common_sessions=common_sessions,
                            n_repeats=n_repeats,
                            cv_output_subdir=cv_output_subdir,
                        )
                    else:
                        self.run_classification(target, test_size)
                except ValueError as exc:
                    print(f"   Skipping target '{target}': {exc}", flush=True)

        self.output_dir = base_output_dir

    # ------------------------------------------------------------------
    # Saving & plotting
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results,
        output_dir,
        model=None,
        model_type="svm",
        feature_importance=None,
    ):
        """Save JSON results, trained model, predictions CSV, and feature importance."""
        results_file = op.join(output_dir, "classification_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}", flush=True)

        # Save feature importance
        if feature_importance is not None:
            fi_file = op.join(output_dir, "feature_importance.json")
            with open(fi_file, "w") as f:
                json.dump(feature_importance, f, indent=2)
            print(f"   Feature importance saved to: {fi_file}", flush=True)

        # Save model (single split only)
        if model is not None:
            model_file = op.join(output_dir, "trained_model.joblib")
            joblib.dump(model, model_file)

        # Subject predictions CSV
        pred_labels = [self.class_names[p] for p in results["y_test_pred"]]
        true_labels = [self.class_names[t] for t in results["y_test_true"]]
        probs = results["y_test_probs"]

        df = pd.DataFrame(
            {
                "subject_session": results["subjects_test"],
                "true_state": true_labels,
                "predicted_state": pred_labels,
                "correct": [t == p for t, p in zip(true_labels, pred_labels)],
            }
        )

        # For binary add a single prob column; for multiclass one per class
        if self.is_binary:
            df["prob_positive"] = probs
        else:
            probs_arr = np.array(probs)
            if probs_arr.ndim == 2:
                for ci, cls in enumerate(self.class_names):
                    df[f"prob_{cls}"] = probs_arr[:, ci]

        csv_file = op.join(output_dir, "subject_predictions.csv")
        df.to_csv(csv_file, index=False)
        print(f"   Predictions CSV saved to: {csv_file}", flush=True)

    def _plot_results(
        self,
        results,
        model_display_name,
        output_dir,
        best_params=None,
        n_samples=None,
        n_features=None,
        class_balance=None,
    ):
        """Generate a 2×2 figure: model info, confusion matrix, dataset info,
        ROC curve."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        n_test = len(results["y_test_true"])
        n_train = results.get("n_train", results.get("n_samples", "?"))
        cv_tag = ""
        if results.get("full_cv"):
            cv_tag = f" ({results['n_folds']}-fold nested CV)"

        target = results.get("target", "")
        reduction = results.get("reduction", "")
        fig.suptitle(
            f"{model_display_name} — Marker Baseline{cv_tag}\n"
            f"target={target}  |  reduction={reduction}  |  "
            f"train={n_train}  test={n_test}",
            fontsize=14,
        )

        # --- (0, 0) Model info ---
        ax = axes[0, 0]
        ax.axis("off")
        info_lines = [f"{model_display_name} Model Info"]
        if best_params:
            info_lines.append("")
            info_lines.append("Best hyperparameters:")
            for k, v in best_params.items():
                info_lines.append(f"  {k} = {v}")
        ax.text(
            0.5,
            0.5,
            "\n".join(info_lines),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        ax.set_title("Model Configuration")

        # --- (0, 1) Confusion matrix ---
        ax = axes[0, 1]
        conf = np.array(results["test_confusion_matrix"])
        im = ax.imshow(conf, cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        thresh = conf.max() / 2.0
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(
                    j,
                    i,
                    str(conf[i, j]),
                    ha="center",
                    va="center",
                    color="white" if conf[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Test Set)")

        # --- (1, 0) Dataset summary ---
        ax = axes[1, 0]
        ax.axis("off")
        info_lines = ["Dataset Summary"]
        if n_samples is not None:
            info_lines.append(f"  Training samples: {n_samples}")
        if n_features is not None:
            info_lines.append(f"  Features: {n_features}")
        bal_acc = results.get("test_balanced_accuracy", None)
        if bal_acc is not None:
            info_lines.append(f"  Test bal. accuracy: {bal_acc:.3f}")
        acc = results.get("test_accuracy", None)
        if acc is not None:
            info_lines.append(f"  Test accuracy:      {acc:.3f}")
        if class_balance:
            info_lines.append("  Class balance (train):")
            for cls_idx, cnt in class_balance.items():
                cls_name = (
                    self.class_names[cls_idx]
                    if cls_idx < len(self.class_names)
                    else str(cls_idx)
                )
                info_lines.append(f"    {cls_name}: {cnt}")
        ax.text(
            0.5,
            0.5,
            "\n".join(info_lines),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        )
        ax.set_title("Dataset Info")

        # --- (1, 1) ROC curve ---
        ax = axes[1, 1]
        if results["test_auc_score"] is not None:
            y_true = np.array(results["y_test_true"])
            y_probs = np.array(results["y_test_probs"])
            if self.is_binary:
                fpr, tpr, _ = roc_curve(y_true, y_probs)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC (AUC = {roc_auc_val:.3f})",
                )
                ax.plot([0, 1], [0, 1], "navy", lw=2, ls="--")
                ax.legend(loc="lower right")
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
                for ci, (cls_name, color) in enumerate(zip(self.class_names, colors)):
                    try:
                        fpr, tpr, _ = roc_curve(
                            (y_true == ci).astype(int), y_probs[:, ci]
                        )
                        roc_auc_val = auc(fpr, tpr)
                        ax.plot(
                            fpr,
                            tpr,
                            color=color,
                            lw=1.5,
                            label=f"{cls_name} (AUC={roc_auc_val:.2f})",
                        )
                    except Exception:
                        pass
                ax.plot([0, 1], [0, 1], "navy", lw=1, ls="--")
                ax.legend(loc="lower right", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "AUC not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Test Set)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = op.join(output_dir, "classification_results.png")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved to: {plot_file}", flush=True)


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline classification on pre-computed neurophysiological "
            "scalar markers.  Trains SVM, Random Forest, and Kernel Ridge "
            "for five clinical targets using group-based CV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Classic split, all 5 targets, reduction A (default)
python dk_marker_baseline.py \\
    --original-metadata /data/project/eeg_foundation/data/original_DoC/baseline_stable_20210128_scalars.csv \\
    --patient-labels /data/project/eeg_foundation/data/metadata/patient_labels.csv \\
    --main-path /data/project/eeg_foundation/data/benchmark_results/new_results

# 5-fold nested CV, reduction B
python dk_marker_baseline.py \\
    --original-metadata /path/to/scalars.csv \\
    --patient-labels /path/to/patient_labels.csv \\
    --main-path /path/to/results \\
    --full-cv --n-cv-folds 5 --marker-reduction B

Reduction map:
  A = icm/lg/egi256/trim_mean80  (default)
  B = icm/lg/egi256/std
  C = icm/lg/egi256gfp/trim_mean80
  D = icm/lg/egi256gfp/std
        """,
    )

    parser.add_argument(
        "--original-metadata",
        default=("/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"),
        help="Path to baseline_stable scalars CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--patient-labels",
        default=(
            "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
        ),
        help="Path to patient_labels.csv with target columns.",
    )
    parser.add_argument(
        "--main-path",
        default=("/data/project/eeg_foundation/data/benchmark_results/new_results"),
        help="Root results directory. Outputs land in {main-path}/MARKER_BASELINE/.",
    )
    parser.add_argument(
        "--marker-reduction",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Reduction variant to use from marker CSV (default: A).",
    )
    parser.add_argument(
        "--full-cv",
        action="store_true",
        default=False,
        help="Run nested GroupKFold CV instead of single split.",
    )
    parser.add_argument(
        "--n-cv-folds",
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
            "Results land in nested_cv_repeated/ with a cv_split_manifest.json."
        ),
    )
    parser.add_argument(
        "--cv-output-subdir",
        default=None,
        help=(
            "Override the cv output subdirectory name. Used by shard submits "
            "to write each repeat to its own dir (e.g. nested_cv_shard/repeat_007). "
            "When None, default is nested_cv/ (n_repeats=1) or nested_cv_repeated/ (n_repeats>1)."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of subjects for test set in classic-split mode (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--target",
        choices=TARGETS + ["all"],
        default="all",
        help=(
            "Which target to predict.  'all' (default) runs all 5 targets in sequence."
        ),
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Path to a pre-computed common_cv_splits.json (from cv_utils). "
            "When provided, the stored session pool and fold assignments are "
            "used so the baseline is evaluated on the same folds as FM models."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help=(
            "Path where the generated CV splits should be saved as JSON. "
            "Only used when --splits-file is not provided. Splits are built "
            "from the DK session pool for the 'crs' target and reused across "
            "all targets so results are internally comparable."
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help="Classifiers to run (default: all). Use to run a subset, e.g. --classifiers xgboost.",
    )

    args = parser.parse_args()

    output_dir = op.join(args.main_path, "MARKER_BASELINE")
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-computed shared splits if provided.
    precomputed_splits = None
    common_sessions = None
    if args.splits_file and op.isfile(args.splits_file):
        print(f"Loading pre-computed splits from {args.splits_file}", flush=True)
        precomputed_splits, common_sessions, _ = load_cv_splits(args.splits_file)
        print(
            f"  {len(common_sessions)} common sessions, "
            f"{len(precomputed_splits)} outer folds",
            flush=True,
        )

    classifier = BaselineClassifier(
        scalars_csv=args.original_metadata,
        patient_labels_file=args.patient_labels,
        output_dir=output_dir,
        reduction_letter=args.marker_reduction,
        random_state=args.random_state,
        full_cv=args.full_cv,
        n_cv_folds=args.n_cv_folds,
        classifiers=args.classifiers,
    )

    # Generate splits from the marker session pool (crs target) when no
    # pre-computed splits file was provided.  The CRS-based folds are reused
    # for all subsequent run_full_cv calls so every target is evaluated on
    # the same fold assignments (internally comparable, and compatible with
    # FM-model splits generated from the same data).
    #
    # Skipped when n_repeats > 1: run_full_cv silently drops n_repeats when
    # precomputed_splits is provided, so for repeated CV we let it generate
    # its own per-repeat folds (seed = random_state + repeat_idx → identical
    # folds across targets within each repeat, same fairness guarantee).
    if precomputed_splits is None and args.full_cv and args.n_repeats == 1:
        print(
            "\nGenerating CV splits from marker session pool (crs target) ...",
            flush=True,
        )
        X_tmp, y_tmp, subjects_tmp, _ = classifier.collect_data("crs")
        labels_for_splits = {s: int(y_tmp[i]) for i, s in enumerate(subjects_tmp)}
        common_sessions = list(subjects_tmp)
        precomputed_splits = generate_nested_cv_folds(
            common_sessions,
            labels_for_splits,
            n_outer=args.n_cv_folds,
            random_state=args.random_state,
        )
        # Always auto-save with timestamp so different runs don't overwrite.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        splits_dir = op.join(output_dir, "cv_splits")
        os.makedirs(splits_dir, exist_ok=True)
        auto_path = op.join(splits_dir, f"baseline_crs_{ts}.json")
        save_cv_splits(
            precomputed_splits, common_sessions, labels_for_splits, path=auto_path
        )
        print(
            f"  Saved {len(precomputed_splits)} folds over "
            f"{len(common_sessions)} sessions to {auto_path}",
            flush=True,
        )
        if args.save_splits_to:
            save_cv_splits(
                precomputed_splits,
                common_sessions,
                labels_for_splits,
                path=args.save_splits_to,
            )
            print(f"  Also saved to explicit path: {args.save_splits_to}", flush=True)

    _OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}

    if args.target == "all":
        classifier.run_all_targets(
            test_size=args.test_size,
            precomputed_splits=precomputed_splits,
            common_sessions=common_sessions,
            n_repeats=args.n_repeats,
            cv_output_subdir=args.cv_output_subdir,
        )
    elif args.target in _OUTCOME_TARGETS:
        modes = [
            ("multiclass", False, False),
            ("binary", True, False),
            ("binary_death", False, True),
        ]
        for mode, binary_outcome, death_binary in modes:
            classifier.output_dir = op.join(output_dir, args.target, mode)
            os.makedirs(classifier.output_dir, exist_ok=True)
            if args.full_cv:
                classifier.run_full_cv(
                    args.target,
                    binary_outcome=binary_outcome,
                    death_binary=death_binary,
                    precomputed_splits=precomputed_splits,
                    common_sessions=common_sessions,
                    n_repeats=args.n_repeats,
                    cv_output_subdir=args.cv_output_subdir,
                )
            else:
                classifier.run_classification(
                    args.target,
                    args.test_size,
                    binary_outcome=binary_outcome,
                    death_binary=death_binary,
                )
    else:
        classifier.output_dir = op.join(output_dir, args.target)
        os.makedirs(classifier.output_dir, exist_ok=True)
        if args.full_cv:
            classifier.run_full_cv(
                args.target,
                precomputed_splits=precomputed_splits,
                common_sessions=common_sessions,
                n_repeats=args.n_repeats,
                cv_output_subdir=args.cv_output_subdir,
            )
        else:
            classifier.run_classification(args.target, args.test_size)


if __name__ == "__main__":
    main()
