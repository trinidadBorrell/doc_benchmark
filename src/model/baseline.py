"""Baseline classification on pre-computed neurophysiological scalar markers.

==================================================
Marker Baseline Classifier for DoC Prediction
==================================================

This script trains 3 classifiers (SVM, Random Forest, Kernel Ridge) directly
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
All five clinical targets are run in sequence:

- ``crs``       — binary VS vs MCS  (UWS→VS, MCS+/MCS-→MCS)
- ``etiology``  — binary acute vs chronic
- ``cs_6m``     — multiclass outcome score at 6 months
- ``cs_1y``     — multiclass outcome score at 1 year
- ``cs_2y``     — multiclass outcome score at 2 years

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
    └── {crs,etiology,cs_6m,cs_1y,cs_2y}/
        ├── classic_split/
        │   ├── svm/
        │   │   ├── classification_results.json
        │   │   ├── classification_results.png
        │   │   └── subject_predictions.csv
        │   ├── random_forest/
        │   └── kernel_ridge/
        └── nested_cv/
            ├── svm/
            ├── random_forest/
            └── kernel_ridge/

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import numpy as np
import pandas as pd
import argparse
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

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

TARGETS = ["crs", "etiology", "cs_6m", "cs_1y", "cs_2y"]

MODEL_CONFIGS = {
    "svm": {"name": "SVM"},
    "random_forest": {"name": "Random Forest"},
    "kernel_ridge": {"name": "Kernel Ridge"},
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

        os.makedirs(self.output_dir, exist_ok=True)

        # Set by collect_data
        self.X = None
        self.y_encoded = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.is_binary = True
        self.feature_names = []

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
        df = pd.read_csv(self.scalars_csv, sep=";")

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

    def load_labels_for_target(self, target, binary_outcome=False):
        """Load labels for a given prediction target from patient_labels.csv.

        Parameters
        ----------
        target : str
            One of 'crs', 'etiology', 'cs_6m', 'cs_1y', 'cs_2y'.
        binary_outcome : bool
            If True and target is an outcome score (cs_6m/cs_1y/cs_2y),
            collapse classes to binary VS vs MCS:
            VS / VS/MCS → VS; MCS+ / MCS- / MCS / CONSCIOUS → MCS;
            DEATH and all others are skipped.

        Returns
        -------
        labels_dict : dict
            Mapping ``subject_session_key -> label_str``.
        class_names : list of str
        is_binary : bool
        """
        print(
            f"Loading labels for target '{target}' from: "
            f"{self.patient_labels_file}",
            flush=True,
        )
        df = pd.read_csv(self.patient_labels_file)

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
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                val = row[col]
                if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                    continue
                label = str(val).strip().lower()
                key = f"{subject}_{session}"
                labels_dict[key] = label
                available_states.add(label)
            is_binary = True

        elif target in ("cs_6m", "cs_1y", "cs_2y"):
            _VS_STATES = {"VS", "VS/MCS"}
            _MCS_STATES = {"MCS+", "MCS-", "MCS", "CONSCIOUS"}
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                val = row[target]
                if pd.isna(val) or str(val).strip().lower() in ("n/a", ""):
                    continue
                label = str(val).strip()
                if binary_outcome:
                    if label in _VS_STATES:
                        label = "VS"
                    elif label in _MCS_STATES:
                        label = "MCS"
                    else:
                        continue  # skip DEATH and other unresolvable states
                key = f"{subject}_{session}"
                labels_dict[key] = label
                available_states.add(label)
            is_binary = binary_outcome

        else:
            raise ValueError(f"Unknown target: {target!r}")

        class_names = sorted(available_states)
        print(
            f"   Loaded {len(labels_dict)} subject/sessions, "
            f"classes: {class_names}",
            flush=True,
        )
        return labels_dict, class_names, is_binary

    def collect_data(self, target, binary_outcome=False):
        """Build X, y, subjects arrays for a given prediction target.

        Parameters
        ----------
        target : str
        binary_outcome : bool
            Passed through to ``load_labels_for_target``; collapses outcome
            classes to binary VS/MCS when True.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y_encoded : np.ndarray, shape (n_samples,)
        subjects : list of str
        groups : np.ndarray — subject-level group IDs for GroupKFold
        """
        print("Collecting data ...", flush=True)

        features_dict, feature_names, groups_dict = self.load_scalars()
        labels_dict, class_names, is_binary = self.load_labels_for_target(
            target, binary_outcome=binary_outcome
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
                f"No valid samples remain after NaN filtering for target "
                f"'{target}'!"
            )

        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.subjects = subjects_list

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

    def _grid_search_svm(self, X_train, y_train, groups_train):
        """Inner 3-fold GroupKFold grid search for SVM (with StandardScaler).

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

        inner_cv = GroupKFold(n_splits=3)
        best_score = -1.0
        best_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

        for C, kernel in product(param_grid["C"], param_grid["kernel"]):
            for gamma in gamma_options[kernel]:
                scores = []
                for tr_idx, va_idx in inner_cv.split(
                    X_train, y_train, groups=groups_train
                ):
                    svc_kw = dict(C=C, kernel=kernel, class_weight="balanced",
                                  probability=True,
                                  random_state=self.random_state,
                                  max_iter=5000)
                    if kernel == "rbf":
                        svc_kw["gamma"] = gamma
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", SVC(**svc_kw)),
                    ])
                    pipe.fit(X_train[tr_idx], y_train[tr_idx])
                    preds = pipe.predict(X_train[va_idx])
                    scores.append(
                        balanced_accuracy_score(y_train[va_idx], preds)
                    )

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"C": C, "kernel": kernel, "gamma": gamma}

        print(
            f"      SVM best params: {best_params} "
            f"(bal_acc={best_score:.3f})",
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

        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(**svc_kw)),
        ])
        best_model.fit(X_train, y_train)
        return best_model, best_params

    def _grid_search_rf(self, X_train, y_train, groups_train):
        """Inner 3-fold GroupKFold grid search for Random Forest.

        Returns
        -------
        model : RandomForestClassifier
        best_params : dict
        """
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
        inner_cv = GroupKFold(n_splits=3)

        best_score = -1.0
        best_params = {}

        for n_est, max_d in product(
            param_grid["n_estimators"], param_grid["max_depth"]
        ):
            scores = []
            for tr_idx, va_idx in inner_cv.split(
                X_train, y_train, groups=groups_train
            ):
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train[tr_idx], y_train[tr_idx])
                preds = rf.predict(X_train[va_idx])
                scores.append(
                    balanced_accuracy_score(y_train[va_idx], preds)
                )

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"n_estimators": n_est, "max_depth": max_d}

        print(
            f"      RF best params: {best_params} (bal_acc={best_score:.3f})",
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
        return best_model, best_params

    def _grid_search_kr(self, X_train, y_train, groups_train):
        """Inner 3-fold GroupKFold grid search for Kernel Ridge (RBF).

        Binary: predictions clipped to [0,1] and thresholded at 0.5.
        Multiclass: one-hot targets, argmax for scoring.

        Returns
        -------
        model : KernelRidge
        best_params : dict
        """
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "gamma": [0.001, 0.01, 0.1, 1.0],
        }
        inner_cv = GroupKFold(n_splits=3)

        if not self.is_binary:
            n_cls = len(np.unique(y_train))
            y_onehot = np.eye(n_cls)[y_train]
        else:
            y_onehot = None

        best_score = -1.0
        best_params = {}

        for alpha, gamma in product(param_grid["alpha"], param_grid["gamma"]):
            scores = []
            for tr_idx, va_idx in inner_cv.split(
                X_train, y_train, groups=groups_train
            ):
                kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
                if self.is_binary:
                    kr.fit(X_train[tr_idx], y_train[tr_idx])
                    raw = kr.predict(X_train[va_idx])
                    preds = (np.clip(raw, 0, 1) >= 0.5).astype(int)
                else:
                    kr.fit(X_train[tr_idx], y_onehot[tr_idx])
                    raw = kr.predict(X_train[va_idx])
                    preds = np.argmax(raw, axis=1)
                scores.append(
                    balanced_accuracy_score(y_train[va_idx], preds)
                )

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"alpha": alpha, "gamma": gamma}

        print(
            f"      KR best params: {best_params} (bal_acc={best_score:.3f})",
            flush=True,
        )

        best_model = KernelRidge(
            kernel="rbf",
            alpha=best_params["alpha"],
            gamma=best_params["gamma"],
        )
        if self.is_binary:
            best_model.fit(X_train, y_train)
        else:
            n_cls = len(np.unique(y_train))
            best_model.fit(X_train, np.eye(n_cls)[y_train])
        return best_model, best_params

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
        raw = model.predict(X)
        if self.is_binary:
            probs = np.clip(raw, 0, 1)
            preds = (probs >= 0.5).astype(int)
        else:
            exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
            probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
        return probs, preds

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_test_metrics(
        self, y_test, test_preds, test_probs, subjects_test
    ):
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
                y_test, test_preds,
                labels=all_labels,
                average=None, zero_division=0,
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
            "test_auc_score": (
                float(test_auc) if test_auc is not None else None
            ),
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

    # ------------------------------------------------------------------
    # Single-split classification
    # ------------------------------------------------------------------

    def run_classification(self, target, test_size=0.2, binary_outcome=False):
        """Classic single-split: GroupShuffleSplit outer, inner grid search.

        Parameters
        ----------
        target : str
        test_size : float
        binary_outcome : bool
            If True, collapses outcome classes to binary VS/MCS.
        """
        mode_tag = " [binary]" if binary_outcome else ""
        print("=" * 80, flush=True)
        print(
            f"MARKER BASELINE — Classic Split | target={target}{mode_tag} | "
            f"reduction={self.reduction_letter}",
            flush=True,
        )
        print("=" * 80, flush=True)

        np.random.seed(self.random_state)

        X, y, subjects, groups = self.collect_data(target, binary_outcome=binary_outcome)

        unique_groups = np.unique(groups)
        print(
            f"   {len(unique_groups)} unique subjects across "
            f"{len(subjects)} sessions",
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
            f"   Train: {len(X_trainval)} sessions "
            f"({len(train_groups_set)} subjects)",
            flush=True,
        )
        print(
            f"   Test:  {len(X_test)} sessions "
            f"({len(test_groups_set)} subjects)",
            flush=True,
        )
        for split_name, y_split in [
            ("Train", y_trainval), ("Test", y_test)
        ]:
            u, c = np.unique(y_split, return_counts=True)
            dist = ", ".join(
                f"{self.class_names[ci]}: {n}" for ci, n in zip(u, c)
            )
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
        svm_results.update({
            "n_train": len(X_trainval),
            "n_test": len(X_test),
            "n_features": int(X.shape[1]),
            "feature_names": list(self.feature_names),
            "best_params": svm_params,
            "subjects_train": subjects_train,
            "reduction": self.reduction_str,
            "target": target,
        })
        svm_dir = op.join(split_dir, "svm")
        os.makedirs(svm_dir, exist_ok=True)
        self._save_results(svm_results, svm_dir, model=svm_model,
                           model_type="svm")
        self._plot_results(svm_results, "SVM", svm_dir,
                           best_params=svm_params,
                           n_samples=len(X_trainval),
                           n_features=X.shape[1],
                           class_balance=dict(
                               zip(*np.unique(y_trainval, return_counts=True))
                           ))
        all_results["svm"] = svm_results

        # ---- Random Forest ----
        print("\n   Training Random Forest (grid search) ...", flush=True)
        rf_model, rf_params = self._grid_search_rf(
            X_trainval, y_trainval, groups_trainval
        )
        rf_probs, rf_preds = self._predict_rf(rf_model, X_test)
        rf_results = self._compute_test_metrics(
            y_test, rf_preds, rf_probs, subjects_test
        )
        rf_results.update({
            "n_train": len(X_trainval),
            "n_test": len(X_test),
            "n_features": int(X.shape[1]),
            "feature_names": list(self.feature_names),
            "best_params": rf_params,
            "subjects_train": subjects_train,
            "reduction": self.reduction_str,
            "target": target,
        })
        rf_dir = op.join(split_dir, "random_forest")
        os.makedirs(rf_dir, exist_ok=True)
        self._save_results(rf_results, rf_dir, model=rf_model,
                           model_type="random_forest")
        self._plot_results(rf_results, "Random Forest", rf_dir,
                           best_params=rf_params,
                           n_samples=len(X_trainval),
                           n_features=X.shape[1],
                           class_balance=dict(
                               zip(*np.unique(y_trainval, return_counts=True))
                           ))
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
        kr_results.update({
            "n_train": len(X_trainval),
            "n_test": len(X_test),
            "n_features": int(X.shape[1]),
            "feature_names": list(self.feature_names),
            "best_params": kr_params,
            "subjects_train": subjects_train,
            "reduction": self.reduction_str,
            "target": target,
        })
        kr_dir = op.join(split_dir, "kernel_ridge")
        os.makedirs(kr_dir, exist_ok=True)
        self._save_results(kr_results, kr_dir, model=kr_model,
                           model_type="kernel_ridge")
        self._plot_results(kr_results, "Kernel Ridge", kr_dir,
                           best_params=kr_params,
                           n_samples=len(X_trainval),
                           n_features=X.shape[1],
                           class_balance=dict(
                               zip(*np.unique(y_trainval, return_counts=True))
                           ))
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

    def run_full_cv(self, target, binary_outcome=False):
        """Nested CV: StratifiedGroupKFold / GroupKFold outer loop.

        Parameters
        ----------
        target : str
        binary_outcome : bool
            If True, collapses outcome classes to binary VS/MCS.
        """
        n_folds = self.n_cv_folds
        mode_tag = " [binary]" if binary_outcome else ""

        print("=" * 80, flush=True)
        print(
            f"MARKER BASELINE — {n_folds}-Fold Nested CV | target={target}{mode_tag} | "
            f"reduction={self.reduction_letter}",
            flush=True,
        )
        print("=" * 80, flush=True)

        np.random.seed(self.random_state)

        X, y, subjects, groups = self.collect_data(target, binary_outcome=binary_outcome)

        unique_groups = np.unique(groups)
        n_unique = len(unique_groups)
        print(
            f"   {n_unique} unique subjects across {len(subjects)} sessions",
            flush=True,
        )

        effective_folds = min(n_folds, n_unique)
        if effective_folds < 2:
            raise ValueError(
                f"Only {n_unique} unique subjects — cannot perform "
                f"{n_folds}-fold CV"
            )

        if self.is_binary:
            sgkf = StratifiedGroupKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            sgkf = GroupKFold(n_splits=effective_folds)

        accum = {
            mk: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
            for mk in MODEL_CONFIGS
        }

        for fold_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(X, y, groups=groups)
        ):
            print(f"\n{'─' * 60}", flush=True)
            print(f"  FOLD {fold_idx + 1}/{effective_folds}", flush=True)
            print(f"{'─' * 60}", flush=True)

            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]
            groups_train_fold = groups[train_idx]
            subjects_test_fold = [subjects[i] for i in test_idx]

            # Leakage check
            train_subj = set(groups[train_idx])
            test_subj = set(groups[test_idx])
            if train_subj & test_subj:
                raise ValueError(
                    f"Fold {fold_idx}: subject leakage: "
                    f"{train_subj & test_subj}"
                )
            print(
                f"   Train: {len(train_subj)} subjects "
                f"({len(train_idx)} sessions)",
                flush=True,
            )
            print(
                f"   Test:  {len(test_subj)} subjects "
                f"({len(test_idx)} sessions)",
                flush=True,
            )

            # ---- SVM ----
            print("   Training SVM ...", flush=True)
            svm_model, _ = self._grid_search_svm(
                X_train_fold, y_train_fold, groups_train_fold
            )
            svm_probs, svm_preds = self._predict_svm(svm_model, X_test_fold)
            accum["svm"]["y_true"].append(y_test_fold)
            accum["svm"]["y_pred"].append(svm_preds)
            accum["svm"]["y_proba"].append(svm_probs)
            accum["svm"]["subjects"].extend(subjects_test_fold)
            print(
                f"   SVM fold bal_acc: "
                f"{balanced_accuracy_score(y_test_fold, svm_preds):.3f}",
                flush=True,
            )

            # ---- Random Forest ----
            print("   Training Random Forest ...", flush=True)
            rf_model, _ = self._grid_search_rf(
                X_train_fold, y_train_fold, groups_train_fold
            )
            rf_probs, rf_preds = self._predict_rf(rf_model, X_test_fold)
            accum["random_forest"]["y_true"].append(y_test_fold)
            accum["random_forest"]["y_pred"].append(rf_preds)
            accum["random_forest"]["y_proba"].append(rf_probs)
            accum["random_forest"]["subjects"].extend(subjects_test_fold)
            print(
                f"   RF fold bal_acc: "
                f"{balanced_accuracy_score(y_test_fold, rf_preds):.3f}",
                flush=True,
            )

            # ---- Kernel Ridge ----
            print("   Training Kernel Ridge ...", flush=True)
            kr_model, _ = self._grid_search_kr(
                X_train_fold, y_train_fold, groups_train_fold
            )
            kr_probs, kr_preds = self._predict_kr(kr_model, X_test_fold)
            accum["kernel_ridge"]["y_true"].append(y_test_fold)
            accum["kernel_ridge"]["y_pred"].append(kr_preds)
            accum["kernel_ridge"]["y_proba"].append(kr_probs)
            accum["kernel_ridge"]["subjects"].extend(subjects_test_fold)
            print(
                f"   KR fold bal_acc: "
                f"{balanced_accuracy_score(y_test_fold, kr_preds):.3f}",
                flush=True,
            )

        # Aggregate across folds
        print(f"\n{'=' * 60}", flush=True)
        print("AGGREGATING RESULTS ACROSS ALL FOLDS", flush=True)
        print(f"{'=' * 60}", flush=True)

        cv_dir = op.join(self.output_dir, "nested_cv")

        for mk, cfg in MODEL_CONFIGS.items():
            y_true_all = np.concatenate(accum[mk]["y_true"])
            y_pred_all = np.concatenate(accum[mk]["y_pred"])
            y_proba_all = np.concatenate(accum[mk]["y_proba"])
            subjects_all = accum[mk]["subjects"]

            results = self._compute_test_metrics(
                y_true_all, y_pred_all, y_proba_all, subjects_all
            )
            results.update({
                "n_samples": len(X),
                "n_features": int(X.shape[1]),
                "feature_names": list(self.feature_names),
                "n_subjects": int(n_unique),
                "n_folds": effective_folds,
                "full_cv": True,
                "reduction": self.reduction_str,
                "target": target,
            })

            model_dir = op.join(cv_dir, mk)
            os.makedirs(model_dir, exist_ok=True)
            self._save_results(results, model_dir, model=None, model_type=mk)
            self._plot_results(results, cfg["name"], model_dir)

            auc_str = (
                f"{results['test_auc_score']:.3f}"
                if results["test_auc_score"] is not None
                else "N/A"
            )
            print(
                f"   {cfg['name']:15s}  "
                f"bal_acc={results['test_balanced_accuracy']:.3f}  "
                f"AUC={auc_str}",
                flush=True,
            )

        print(f"\n   Results saved to: {cv_dir}", flush=True)
        print("=" * 80, flush=True)

    # ------------------------------------------------------------------
    # Multi-target runner
    # ------------------------------------------------------------------

    def run_all_targets(self, test_size=0.2):
        """Run classification for all 5 targets.

        For binary targets (crs, etiology) the existing single run is used.
        For outcome targets (cs_6m, cs_1y, cs_2y) two runs are performed:
          - ``{target}/multiclass/`` — full multi-class classification
          - ``{target}/binary/``     — VS vs MCS binary collapse
        """
        _OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}
        base_output_dir = self.output_dir

        for target in TARGETS:
            print(f"\n{'=' * 80}", flush=True)
            print(f"TARGET: {target}", flush=True)
            print(f"{'=' * 80}", flush=True)

            if target in _OUTCOME_TARGETS:
                for mode, binary_outcome in [("multiclass", False), ("binary", True)]:
                    print(f"\n--- {target} / {mode} ---", flush=True)
                    self.output_dir = op.join(base_output_dir, target, mode)
                    os.makedirs(self.output_dir, exist_ok=True)
                    try:
                        if self.full_cv:
                            self.run_full_cv(target, binary_outcome=binary_outcome)
                        else:
                            self.run_classification(
                                target, test_size, binary_outcome=binary_outcome
                            )
                    except ValueError as exc:
                        print(
                            f"   Skipping target '{target}' ({mode}): {exc}",
                            flush=True,
                        )
            else:
                self.output_dir = op.join(base_output_dir, target)
                os.makedirs(self.output_dir, exist_ok=True)
                try:
                    if self.full_cv:
                        self.run_full_cv(target)
                    else:
                        self.run_classification(target, test_size)
                except ValueError as exc:
                    print(f"   Skipping target '{target}': {exc}", flush=True)

        self.output_dir = base_output_dir

    # ------------------------------------------------------------------
    # Saving & plotting
    # ------------------------------------------------------------------

    def _save_results(self, results, output_dir, model=None, model_type="svm"):
        """Save JSON results, trained model, and predictions CSV."""
        results_file = op.join(output_dir, "classification_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}", flush=True)

        # Save model (single split only)
        if model is not None:
            model_file = op.join(output_dir, "trained_model.joblib")
            joblib.dump(model, model_file)

        # Subject predictions CSV
        pred_labels = [self.class_names[p] for p in results["y_test_pred"]]
        true_labels = [self.class_names[t] for t in results["y_test_true"]]
        probs = results["y_test_probs"]

        df = pd.DataFrame({
            "subject_session": results["subjects_test"],
            "true_state": true_labels,
            "predicted_state": pred_labels,
            "correct": [t == p for t, p in zip(true_labels, pred_labels)],
        })

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
            0.5, 0.5,
            "\n".join(info_lines),
            ha="center", va="center",
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
                    j, i, str(conf[i, j]),
                    ha="center", va="center",
                    color="white" if conf[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold",
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
            0.5, 0.5,
            "\n".join(info_lines),
            ha="center", va="center",
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
                    fpr, tpr,
                    color="darkorange", lw=2,
                    label=f"ROC (AUC = {roc_auc_val:.3f})",
                )
                ax.plot([0, 1], [0, 1], "navy", lw=2, ls="--")
                ax.legend(loc="lower right")
            else:
                colors = plt.cm.tab10(
                    np.linspace(0, 1, len(self.class_names))
                )
                for ci, (cls_name, color) in enumerate(
                    zip(self.class_names, colors)
                ):
                    try:
                        fpr, tpr, _ = roc_curve(
                            (y_true == ci).astype(int), y_probs[:, ci]
                        )
                        roc_auc_val = auc(fpr, tpr)
                        ax.plot(
                            fpr, tpr, color=color, lw=1.5,
                            label=f"{cls_name} (AUC={roc_auc_val:.2f})",
                        )
                    except Exception:
                        pass
                ax.plot([0, 1], [0, 1], "navy", lw=1, ls="--")
                ax.legend(loc="lower right", fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "AUC not available",
                ha="center", va="center", transform=ax.transAxes,
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
python baseline.py \\
    --original-metadata /data/project/eeg_foundation/data/original_DoC/baseline_stable_20210128_scalars.csv \\
    --patient-labels /data/project/eeg_foundation/data/metadata/patient_labels.csv \\
    --main-path /data/project/eeg_foundation/data/benchmark_results/new_results

# 5-fold nested CV, reduction B
python baseline.py \\
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
        default=(
            "/data/project/eeg_foundation/data/original_DoC/"
            "baseline_stable_20210128_scalars.csv"
        ),
        help="Path to baseline_stable scalars CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--patient-labels",
        default=(
            "/data/project/eeg_foundation/data/metadata/patient_labels.csv"
        ),
        help="Path to patient_labels.csv with target columns.",
    )
    parser.add_argument(
        "--main-path",
        default=(
            "/data/project/eeg_foundation/data/benchmark_results/new_results"
        ),
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
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of subjects for test set in classic-split mode "
             "(default: 0.2).",
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
            "Which target to predict.  'all' (default) runs all 5 targets "
            "in sequence."
        ),
    )

    args = parser.parse_args()

    output_dir = op.join(args.main_path, "MARKER_BASELINE")
    os.makedirs(output_dir, exist_ok=True)

    classifier = BaselineClassifier(
        scalars_csv=args.original_metadata,
        patient_labels_file=args.patient_labels,
        output_dir=output_dir,
        reduction_letter=args.marker_reduction,
        random_state=args.random_state,
        full_cv=args.full_cv,
        n_cv_folds=args.n_cv_folds,
    )

    _OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}

    if args.target == "all":
        classifier.run_all_targets(test_size=args.test_size)
    elif args.target in _OUTCOME_TARGETS:
        for mode, binary_outcome in [("multiclass", False), ("binary", True)]:
            classifier.output_dir = op.join(output_dir, args.target, mode)
            os.makedirs(classifier.output_dir, exist_ok=True)
            if args.full_cv:
                classifier.run_full_cv(args.target, binary_outcome=binary_outcome)
            else:
                classifier.run_classification(
                    args.target, args.test_size, binary_outcome=binary_outcome
                )
    else:
        classifier.output_dir = op.join(output_dir, args.target)
        os.makedirs(classifier.output_dir, exist_ok=True)
        if args.full_cv:
            classifier.run_full_cv(args.target)
        else:
            classifier.run_classification(args.target, args.test_size)


if __name__ == "__main__":
    main()
