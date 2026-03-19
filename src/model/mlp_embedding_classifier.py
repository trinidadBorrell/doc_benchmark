"""Multi-model classification on pre-computed foundation model embeddings.

==================================================
Embedding Classifier for EEG Foundation Models
==================================================

This script trains 3 classifiers (MLP, Random Forest, Kernel Ridge) on
pre-computed embeddings from EEG foundation models (CBraMod, TOTEM, LaBram,
NeuroLM).  It supports three operating modes (controlled by CLI flags):

1. **Standard classification** (default):
   Binary VS vs MCS classification (UWS → VS, MCS+/MCS- → MCS).

2. **Marker regression** (``--marker-regression``):
   Predict each pre-computed scalar neurophysiological marker from the
   embeddings using Ridge regression with inner GroupKFold grid search.
   Outputs R², Pearson r, MSE, MAE per marker plus a ranked bar chart.

3. **Full metric prediction** (``--full-metric-prediction``):
   Run the same MLP + RF + KR classification pipeline for five clinical
   targets: ``crs`` (binary VS/MCS), ``etiology`` (binary acute/chronic),
   ``cs_6m``, ``cs_1y``, ``cs_2y`` (multiclass outcome scores).
   Results land in ``{output_dir}/{target}/``.

Common features across all modes
---------------------------------
- Embedding-size agnostic: input dimension determined at runtime
- Cross-subject splitting with GROUP-BASED CV (no session leakage)
- Optional nested StratifiedGroupKFold (``--full-cv``) or single split
- Optional subject intersection across multiple foundation models

CLI quick reference
-------------------
::

    # Standard binary classification (single split)
    python mlp_embedding_classifier.py \\
        --data-dir /path/to/embeddings \\
        --patient-labels /path/to/patient_labels_with_controls.csv \\
        --output-dir /path/to/out

    # 5-fold nested CV
    python mlp_embedding_classifier.py ... --full-cv --n-cv-folds 5

    # Marker regression (single outer split, reduction A)
    python mlp_embedding_classifier.py ... \\
        --marker-regression \\
        --marker-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \\
        --marker-reduction A

    # Full metric prediction (5 targets, single split)
    python mlp_embedding_classifier.py ... \\
        --full-metric-prediction \\
        --patient-labels-full /data/metadata/patient_labels.csv

    # Subject intersection across two foundation models
    python mlp_embedding_classifier.py ... \\
        --use-subject-intersection \\
        --embedding-dirs TOTEM=/path/to/totem CBraMod=/path/to/cbramod

Reduction map (``--marker-reduction``)
---------------------------------------
- A: ``icm/lg/egi256/trim_mean80``
- B: ``icm/lg/egi256/std``
- C: ``icm/lg/egi256gfp/trim_mean80``
- D: ``icm/lg/egi256gfp/std``

Output structure
----------------
::

    {output_dir}/
    ├── classic_split/          # default single-split mode
    │   ├── mlp/classification_results.{json,png}
    │   ├── random_forest/classification_results.{json,png}
    │   └── kernel_ridge/classification_results.{json,png}
    ├── nested_cv/              # --full-cv mode
    │   └── {mlp,random_forest,kernel_ridge}/classification_results.{json,png}
    ├── regressor_results/      # --marker-regression
    │   ├── summary.json
    │   ├── marker_performance.png
    │   └── per_marker/{marker_name}/predictions.csv
    └── {crs,etiology,cs_6m,cs_1y,cs_2y}/  # --full-metric-prediction
        └── {classic_split,nested_cv}/...

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Prevent MKL/OpenBLAS segfaults when running as a subprocess
torch.set_num_threads(1)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
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
import joblib

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


# ======================================================================
# Model configurations
# ======================================================================

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

MODEL_CONFIGS = {
    "mlp": {"name": "MLP", "needs_val": True, "has_curves": True},
    "random_forest": {"name": "Random Forest", "needs_val": False, "has_curves": False},
    "kernel_ridge": {"name": "Kernel Ridge", "needs_val": False, "has_curves": False},
}


# ======================================================================
# Neural network model
# ======================================================================


class EmbeddingMLP(nn.Module):
    """Simple MLP for binary or multiclass classification on embeddings.

    Architecture is embedding-size agnostic: input_dim is determined
    at runtime from the loaded data.
    """

    def __init__(self, input_dim: int, n_classes: int = 1):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input embeddings.
        n_classes : int
            Number of output classes. 1 = binary (BCEWithLogitsLoss),
            >1 = multiclass (CrossEntropyLoss).
        """
        super().__init__()
        self.n_classes = n_classes
        out_dim = max(n_classes, 1)
        self.net = nn.Sequential(
            #   nn.Linear(input_dim, 128),
            #   nn.ReLU(),
            #   nn.BatchNorm1d(128),
            #   nn.Dropout(0.3),
            #   nn.Linear(128, 64),
            #   nn.ReLU(),
            #   nn.BatchNorm1d(64),
            #   nn.Dropout(0.3),
            #   nn.Linear(input_dim, out_dim)
            nn.Linear(input_dim, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.n_classes == 1 else out


# ======================================================================
# Classifier
# ======================================================================


class EmbeddingClassifier:
    """Cross-subject multi-model classifier on foundation model embeddings.

    Trains MLP, Random Forest, and Kernel Ridge on the same data splits.
    Uses two nested GroupShuffleSplits (single-split) or StratifiedGroupKFold /
    GroupKFold (full-CV) to create train / val / test sets with no subject
    overlap.

    Three operating modes (selected at call time, not construction):

    classify
        Binary VS vs MCS (default).  Call ``run_classification()`` or
        ``run_full_cv()``.

    marker regression
        Ridge regression: embeddings → neurophysiological scalar markers.
        Call ``run_marker_regression(marker_csv, reduction_letter)``.
        Requires a semicolon-delimited baseline scalars CSV; use
        ``--marker-reduction A/B/C/D`` to select the aggregation variant.

    full metric prediction
        MLP+RF+KR for five clinical targets (crs, etiology, cs_6m, cs_1y,
        cs_2y).  Call ``run_full_metric_prediction(patient_labels_full)``.
        Binary targets (crs, etiology) use BCEWithLogitsLoss; multiclass
        targets (cs_6m/cs_1y/cs_2y) use CrossEntropyLoss and GroupKFold.

    All modes respect ``self.allowed_subjects`` for subject intersection
    filtering.
    """

    def __init__(
        self,
        data_dir,
        patient_labels_file,
        output_dir=None,
        random_state=42,
        n_epochs=500,
        lr=1e-3,
        batch_size=32,
        weight_decay=1e-4,
        patience=100,
        full_cv=False,
        n_cv_folds=5,
        use_subject_intersection=False,
        embedding_dirs=None,
        allowed_subjects=None,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing sub-{ID}/ses-{NUM}/*_embedding.npy
        patient_labels_file : str
            Path to CSV with patient labels (must have diagnostic_crs_final
            column)
        output_dir : str
            Output directory for results
        random_state : int
            Random state for reproducibility
        n_epochs : int
            Maximum training epochs for MLP
        lr : float
            Learning rate for MLP
        batch_size : int
            Training batch size for MLP
        weight_decay : float
            L2 regularisation strength for MLP
        patience : int
            Early stopping patience (epochs without val loss improvement)
        full_cv : bool
            If True, run 5-fold StratifiedGroupKFold cross-validation
        n_cv_folds : int
            Number of CV folds (default: 5)
        use_subject_intersection : bool
            If True, filter subjects to intersection across embedding_dirs
        embedding_dirs : dict
            Mapping {model_name: path} for subject intersection computation
        allowed_subjects : list
            Pre-computed list of allowed subject keys (from intersection)
        """
        self.data_dir = data_dir
        self.patient_labels_file = patient_labels_file
        self.output_dir = output_dir or "results/mlp_embedding"
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.full_cv = full_cv
        self.n_cv_folds = n_cv_folds
        self.use_subject_intersection = use_subject_intersection
        self.embedding_dirs = embedding_dirs
        self.allowed_subjects = allowed_subjects
        # Pooled embeddings are always cached under {output_dir}/pooled_embeddings/
        # so they are co-located with the model results and never shared across runs.
        self.pooled_embeddings_dir = op.join(self.output_dir, "pooled_embeddings")

        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data containers
        self.X = None
        self.y = None
        self.subjects = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.is_binary = True  # updated by collect_data for multiclass targets

    # ------------------------------------------------------------------
    # Subject intersection
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_subjects(data_dir):
        """Walk sub-{ID}/ses-{NUM}/ looking for embedding files.

        Returns
        -------
        set of str
            Subject-session keys like "subject_id_ses-NUM".
        """
        found = set()
        data_dir = str(data_dir)
        if not os.path.isdir(data_dir):
            return found

        for subject_dir in sorted(os.listdir(data_dir)):
            if not subject_dir.startswith("sub-"):
                continue
            subject_path = op.join(data_dir, subject_dir)
            if not op.isdir(subject_path):
                continue
            subject_id = subject_dir.replace("sub-", "")

            for session_dir in sorted(os.listdir(subject_path)):
                if not session_dir.startswith("ses-"):
                    continue
                session_path = op.join(subject_path, session_dir)
                if not op.isdir(session_path):
                    continue

                has_emb = any(
                    f.endswith(
                        (
                            "_embedding.npy",
                            "_embeddings.npy",
                            "_embedding.npz",
                            "_embeddings.npz",
                        )
                    )
                    and not f.endswith("_metadata.npy")
                    for f in os.listdir(session_path)
                )
                if has_emb:
                    found.add(f"{subject_id}_{session_dir}")

        return found

    @staticmethod
    def compute_subject_intersection(embedding_dirs):
        """Compute intersection of subjects across all embedding directories.

        Parameters
        ----------
        embedding_dirs : dict
            Mapping {model_name: path} for each foundation model.

        Returns
        -------
        list of str
            Sorted list of subject-session keys present in ALL directories.
        """
        print(
            "Computing subject intersection across embedding directories ...",
            flush=True,
        )

        all_sets = {}
        for name, path in embedding_dirs.items():
            subjects = EmbeddingClassifier._scan_subjects(path)
            all_sets[name] = subjects
            print(f"   {name}: {len(subjects)} subjects", flush=True)

        if not all_sets:
            return []

        intersection = set.intersection(*all_sets.values())
        print(f"   Intersection: {len(intersection)} subjects", flush=True)
        return sorted(intersection)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_patient_labels(self):
        """Load patient labels for binary VS vs MCS classification.

        Uses the ``diagnostic_crs_final`` column, mapping UWS -> VS and
        MCS+/MCS- -> MCS.  All other states are skipped.

        Returns
        -------
        labels_dict : dict
            Mapping ``subject_session_key -> label``
        available_states : list
            Sorted list of unique states found
        """
        print(f"Loading patient labels from: {self.patient_labels_file}", flush=True)

        df = pd.read_csv(self.patient_labels_file)

        labels_dict = {}
        available_states = set()

        for _, row in df.iterrows():
            subject = row["subject"]
            session = f"ses-{row['session']:02d}"
            state = row["diagnostic_crs_final"]

            if pd.isna(state) or state == "n/a":
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

        print(f"   Loaded labels for {len(labels_dict)} subject/sessions", flush=True)
        print(f"   Available states: {sorted(available_states)}", flush=True)
        return labels_dict, sorted(available_states)

    def load_patient_labels_for_target(self, labels_file, target, binary_outcome=False):
        """Load patient labels for a given prediction target.

        Parameters
        ----------
        labels_file : str
            Path to patient_labels.csv (must have subject, session columns
            plus target-specific columns).
        target : str
            One of 'crs', 'etiology', 'cs_6m', 'cs_1y', 'cs_2y'.
        binary_outcome : bool
            If True and target is an outcome score (cs_6m/cs_1y/cs_2y),
            collapse to binary VS vs MCS (excludes DEATH and CONSCIOUS).

        Returns
        -------
        labels_dict : dict
            Mapping subject_session_key -> label (str)
        class_names : list of str
            Sorted unique class names
        is_binary : bool
            True for binary targets (crs, etiology), False for multiclass
        """
        print(f"Loading labels for target '{target}' from: {labels_file}", flush=True)
        df = pd.read_csv(labels_file)

        labels_dict = {}
        available_states = set()

        if target == "crs":
            for _, row in df.iterrows():
                subject = row["subject"]
                session = f"ses-{int(row['session']):02d}"
                state = row["diagnostic_crs_final"]
                if pd.isna(state) or state == "n/a":
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

    def load_embeddings(self, embedding_suffix="_embedding.npy"):
        """Discover and load embedding files, mean-pooling across windows.

        Parameters
        ----------
        embedding_suffix : str
            File suffix to match (default: ``_embedding.npy``).

        Returns
        -------
        dict
            Mapping ``subject_session_key -> (embedding_dim,)`` numpy array.
        """
        print(f"Loading embeddings from: {self.data_dir}", flush=True)
        embeddings = {}

        subject_dirs = sorted(
            [
                d
                for d in os.listdir(self.data_dir)
                if d.startswith("sub-") and op.isdir(op.join(self.data_dir, d))
            ]
        )

        for subject_dir in subject_dirs:
            subject_id = subject_dir.replace("sub-", "")
            subject_path = op.join(self.data_dir, subject_dir)

            session_dirs = sorted(
                [
                    d
                    for d in os.listdir(subject_path)
                    if d.startswith("ses-") and op.isdir(op.join(subject_path, d))
                ]
            )

            for session_dir in session_dirs:
                session_path = op.join(subject_path, session_dir)
                key = f"{subject_id}_{session_dir}"

                # ── (A) Check pooled cache ────────────────────────────────
                if self.pooled_embeddings_dir is not None:
                    cache_path = op.join(
                        self.pooled_embeddings_dir,
                        f"sub-{subject_id}",
                        session_dir,
                        "embedding.npz",
                    )
                    if op.isfile(cache_path):
                        try:
                            npz = np.load(cache_path)
                            data = npz["embedding"]
                            embeddings[key] = data
                            continue
                        except Exception as e:
                            print(
                                f"   Warning: cache load failed for {cache_path}: {e}."
                                " Re-pooling from raw.",
                                flush=True,
                            )

                # ── (B) Load raw embeddings and pool ─────────────────────
                emb_files = sorted(
                    [
                        f
                        for f in os.listdir(session_path)
                        if (
                            f.endswith(embedding_suffix)
                            or f.endswith("_embeddings.npy")
                            or f.endswith("_embedding.npz")
                            or f.endswith("_embeddings.npz")
                        )
                        and not f.endswith("_metadata.npy")
                    ]
                )

                if not emb_files:
                    continue

                emb_path = op.join(session_path, emb_files[0])
                try:
                    if emb_path.endswith(".npz"):
                        npz = np.load(emb_path)
                        if "embedding" in npz:
                            data = npz["embedding"]
                        elif "embeddings" in npz:
                            data = npz["embeddings"]
                        else:
                            data = npz[list(npz.keys())[0]]
                    else:
                        data = np.load(emb_path)
                    original_shape = data.shape

                    if data.ndim == 1:
                        pass
                    elif data.ndim == 2:
                        data = data.mean(axis=0)
                    elif data.ndim >= 3:
                        axes_to_average = tuple(range(data.ndim - 1))
                        data = data.mean(axis=axes_to_average)
                        print(
                            f"   Pooled {original_shape} -> {data.shape} "
                            f"for {op.basename(emb_path)}",
                            flush=True,
                        )
                    else:
                        print(
                            f"   Unexpected shape {data.shape} in {emb_path}, skipping",
                            flush=True,
                        )
                        continue

                    embeddings[key] = data

                    # ── (B2) Save to cache ────────────────────────────────
                    if self.pooled_embeddings_dir is not None:
                        cache_path = op.join(
                            self.pooled_embeddings_dir,
                            f"sub-{subject_id}",
                            session_dir,
                            "embedding.npz",
                        )
                        os.makedirs(op.dirname(cache_path), exist_ok=True)
                        np.savez_compressed(cache_path, embedding=data)

                except Exception as e:
                    print(f"   Error loading {emb_path}: {e}", flush=True)

        print(
            f"   Loaded embeddings for {len(embeddings)} subject/sessions", flush=True
        )
        return embeddings

    def collect_data(
        self,
        embedding_suffix="_embedding.npy",
        target="crs",
        labels_file=None,
        binary_outcome=False,
    ):
        """Match embeddings with labels and build feature / label arrays.

        If ``self.allowed_subjects`` is set, filters to only those keys.

        Parameters
        ----------
        embedding_suffix : str
            Embedding file suffix (default: ``_embedding.npy``).
        target : str
            Label target. Default ``'crs'`` uses existing binary VS/MCS logic
            via ``load_patient_labels()``. Other targets use
            ``load_patient_labels_for_target()``.
        labels_file : str or None
            Override labels CSV path. If None and target != 'crs', uses
            ``self.patient_labels_file``.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y_encoded : np.ndarray, shape (n_samples,)
        subjects : list of str
        """
        print("Collecting data ...", flush=True)

        if target == "crs" and labels_file is None:
            labels_dict, available_states = self.load_patient_labels()
            self.is_binary = True
        else:
            lf = labels_file or self.patient_labels_file
            labels_dict, available_states, self.is_binary = (
                self.load_patient_labels_for_target(lf, target, binary_outcome=binary_outcome)
            )

        embeddings = self.load_embeddings(embedding_suffix)

        intersection_keys = set(embeddings) & set(labels_dict)
        print(
            f"   Embeddings: {len(embeddings)} sessions | "
            f"Labels ({target}): {len(labels_dict)} sessions | "
            f"Intersection: {len(intersection_keys)} sessions",
            flush=True,
        )

        X_list, y_list, subjects_list = [], [], []

        for key, emb in sorted(embeddings.items()):
            if key not in labels_dict:
                continue

            if X_list and emb.shape != X_list[0].shape:
                print(
                    f"   Shape mismatch for {key}: expected "
                    f"{X_list[0].shape}, got {emb.shape}",
                    flush=True,
                )
                continue

            X_list.append(emb)
            y_list.append(labels_dict[key])
            subjects_list.append(key)

        if not X_list:
            raise ValueError("No subjects matched between embeddings and labels!")

        # Filter by allowed subjects if set
        if self.allowed_subjects is not None:
            allowed_set = set(self.allowed_subjects)
            n_before = len(X_list)
            filtered = [
                (x, y, s)
                for x, y, s in zip(X_list, y_list, subjects_list)
                if s in allowed_set
            ]
            if not filtered:
                raise ValueError("No subjects remain after intersection filtering!")
            X_list, y_list, subjects_list = zip(*filtered)
            X_list, y_list, subjects_list = (
                list(X_list),
                list(y_list),
                list(subjects_list),
            )
            print(
                f"   Subject intersection filter: {n_before} -> {len(X_list)} samples",
                flush=True,
            )

        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.subjects = subjects_list

        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = self.label_encoder.classes_

        print(
            f"   Dataset: {self.X.shape[0]} samples x {self.X.shape[1]} features",
            flush=True,
        )
        print(f"   Classes: {list(self.class_names)}", flush=True)
        unique, counts = np.unique(self.y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"      {cls}: {cnt}", flush=True)

        return self.X, self.y_encoded, self.subjects

    # ------------------------------------------------------------------
    # Marker data loading (for --marker-regression)
    # ------------------------------------------------------------------

    def load_markers_from_csv(self, marker_csv, reduction_letter):
        """Load baseline scalar markers from CSV for a given reduction.

        Parameters
        ----------
        marker_csv : str
            Path to semicolon-delimited CSV with Subject, Reduction, Label,
            and marker columns.
        reduction_letter : str
            One of A/B/C/D, mapped via REDUCTION_MAP.

        Returns
        -------
        markers_dict : dict
            Mapping subject_session_key -> np.ndarray of shape (n_markers,)
        marker_names : list of str
            Ordered marker column names
        """
        if reduction_letter not in REDUCTION_MAP:
            raise ValueError(
                f"Unknown reduction letter {reduction_letter!r}. "
                f"Choose from {list(REDUCTION_MAP)}."
            )
        reduction_str = REDUCTION_MAP[reduction_letter]

        df = pd.read_csv(marker_csv, sep=";")

        # Filter to requested reduction
        df_filtered = df[df["Reduction"] == reduction_str].copy()
        if df_filtered.empty:
            raise ValueError(
                f"No rows found for Reduction={reduction_str!r} in {marker_csv}"
            )

        # Infer marker columns: keep only numeric-dtype columns, excluding
        # known metadata names and any unnamed index columns (Unnamed: N).
        meta_cols = {"Subject", "Reduction", "Label"}
        marker_names = [
            c for c in df_filtered.columns
            if c not in meta_cols
            and not str(c).startswith("Unnamed")
            and pd.api.types.is_numeric_dtype(df_filtered[c])
        ]

        markers_dict = {}
        for _, row in df_filtered.iterrows():
            raw_subject = str(row["Subject"])
            # "AA079_2" -> "AA079_ses-02"
            parts = raw_subject.rsplit("_", 1)
            if len(parts) != 2:
                continue
            subject_id, sesnum = parts
            try:
                key = f"{subject_id}_ses-{int(sesnum):02d}"
            except ValueError:
                continue
            markers_dict[key] = np.array(
                [row[m] for m in marker_names], dtype=float
            )

        print(
            f"   Loaded markers for {len(markers_dict)} subject/sessions "
            f"({len(marker_names)} markers, reduction={reduction_str})",
            flush=True,
        )
        return markers_dict, marker_names

    def collect_data_for_regression(self, marker_csv, reduction_letter):
        """Build (X, Y, subjects, marker_names) for marker regression.

        Intersects embedding keys with marker keys, applying
        ``self.allowed_subjects`` filter if set.

        Returns
        -------
        X : np.ndarray, shape (n, emb_dim)
        Y : np.ndarray, shape (n, n_markers)
        subjects : list of str
        marker_names : list of str
        """
        print("Collecting data for marker regression ...", flush=True)
        markers_dict, marker_names = self.load_markers_from_csv(
            marker_csv, reduction_letter
        )
        embeddings = self.load_embeddings()

        common_keys = sorted(set(embeddings) & set(markers_dict))
        if self.allowed_subjects is not None:
            allowed_set = set(self.allowed_subjects)
            common_keys = [k for k in common_keys if k in allowed_set]

        if not common_keys:
            raise ValueError(
                "No subjects matched between embeddings and marker CSV!"
            )

        X_list, Y_list, subjects_list = [], [], []
        for key in common_keys:
            emb = embeddings[key]
            if X_list and emb.shape != X_list[0].shape:
                print(
                    f"   Shape mismatch for {key}: skipping", flush=True
                )
                continue
            X_list.append(emb)
            Y_list.append(markers_dict[key])
            subjects_list.append(key)

        X = np.array(X_list)
        Y = np.array(Y_list)
        print(
            f"   Regression dataset: {X.shape[0]} samples x {X.shape[1]} features, "
            f"{Y.shape[1]} markers",
            flush=True,
        )
        return X, Y, subjects_list, marker_names

    # ------------------------------------------------------------------
    # MLP training
    # ------------------------------------------------------------------

    def _make_loader(self, X, y, shuffle=True, label_dtype=None):
        if label_dtype is None:
            label_dtype = torch.float32 if self.is_binary else torch.long
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=label_dtype),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_model(self, X_train, y_train, X_val, y_val, pos_weight, n_classes=1):
        """Train an MLP with early stopping on validation loss.

        Additionally tracks balanced error rate (1 - balanced accuracy)
        on both train and val sets every epoch.

        Returns
        -------
        model : EmbeddingMLP
        train_losses : list of float
        val_losses : list of float
        train_errors : list of float
        val_errors : list of float
        """
        input_dim = X_train.shape[1]
        model = EmbeddingMLP(input_dim, n_classes=n_classes).to(self.device)

        if n_classes == 1:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=self.device)
            )
            train_loader = self._make_loader(
                X_train, y_train, shuffle=True, label_dtype=torch.float32
            )
            val_loader = self._make_loader(
                X_val, y_val, shuffle=False, label_dtype=torch.float32
            )
        else:
            classes = np.arange(n_classes)
            cw = compute_class_weight("balanced", classes=classes, y=y_train)
            cw_tensor = torch.tensor(cw, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=cw_tensor)
            train_loader = self._make_loader(
                X_train, y_train, shuffle=True, label_dtype=torch.long
            )
            val_loader = self._make_loader(
                X_val, y_val, shuffle=False, label_dtype=torch.long
            )

        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        train_losses, val_losses = [], []
        train_errors, val_errors = [], []

        for epoch in range(self.n_epochs):
            # --- train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            all_train_preds, all_train_labels = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

                if n_classes == 1:
                    preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                else:
                    preds = logits.argmax(dim=-1).cpu().numpy()
                all_train_preds.append(preds)
                all_train_labels.append(yb.cpu().numpy().astype(int))

            train_losses.append(epoch_loss / max(n_batches, 1))
            train_preds = np.concatenate(all_train_preds)
            train_labels = np.concatenate(all_train_labels)
            train_errors.append(
                1.0 - balanced_accuracy_score(train_labels, train_preds)
            )

            # --- validate ---
            model.eval()
            val_loss = 0.0
            n_val = 0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    n_val += 1

                    if n_classes == 1:
                        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                    else:
                        preds = logits.argmax(dim=-1).cpu().numpy()
                    all_val_preds.append(preds)
                    all_val_labels.append(yb.cpu().numpy().astype(int))

            val_losses.append(val_loss / max(n_val, 1))
            val_preds = np.concatenate(all_val_preds)
            val_labels = np.concatenate(all_val_labels)
            val_errors.append(1.0 - balanced_accuracy_score(val_labels, val_preds))

            # --- epoch log ---
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"      Epoch {epoch + 1:>3d}/{self.n_epochs}  "
                    f"train_loss={train_losses[-1]:.4f}  "
                    f"val_loss={val_losses[-1]:.4f}  "
                    f"train_err={train_errors[-1]:.4f}  "
                    f"val_err={val_errors[-1]:.4f}",
                    flush=True,
                )

            # --- early stopping ---
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, train_losses, val_losses, train_errors, val_errors

    @torch.no_grad()
    def _predict(self, model, X):
        """Return (probabilities, predicted_labels) for MLP model.

        For binary (n_classes==1): returns 1-D prob array (sigmoid).
        For multiclass: returns 2-D prob array (softmax) and argmax preds.
        """
        model.eval()
        loader = self._make_loader(
            X, np.zeros(len(X)), shuffle=False, label_dtype=torch.float32
        )
        all_probs = []
        for xb, _ in loader:
            xb = xb.to(self.device)
            logits = model(xb)
            if model.n_classes == 1:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        probs = np.concatenate(all_probs)
        if model.n_classes == 1:
            preds = (probs >= 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
        return probs, preds

    # ------------------------------------------------------------------
    # Sklearn model training (RF, KR)
    # ------------------------------------------------------------------

    def _grid_search_rf(self, X_train, y_train, groups_train):
        """Inner 3-fold GroupKFold grid search for Random Forest.

        Returns
        -------
        model : RandomForestClassifier
            Best model refitted on full X_train.
        best_params : dict
            Best hyperparameters found.
        """
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
        inner_cv = GroupKFold(n_splits=3)

        best_score = -1
        best_params = {}

        for n_est, max_d in product(
            param_grid["n_estimators"], param_grid["max_depth"]
        ):
            scores = []
            for tr_idx, va_idx in inner_cv.split(X_train, y_train, groups=groups_train):
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train[tr_idx], y_train[tr_idx])
                preds = rf.predict(X_train[va_idx])
                scores.append(balanced_accuracy_score(y_train[va_idx], preds))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"n_estimators": n_est, "max_depth": max_d}

        print(
            f"      RF best params: {best_params} (bal_acc={best_score:.3f})",
            flush=True,
        )

        # Refit on full training set with best params
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

        Binary: predictions are clipped to [0,1] and thresholded at 0.5.
        Multiclass: one-hot encode targets, inner loop uses argmax for scoring.

        Returns
        -------
        model : KernelRidge
            Best model refitted on full X_train.
        best_params : dict
            Best hyperparameters found.
        """
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "gamma": [0.001, 0.01, 0.1, 1.0],
        }
        inner_cv = GroupKFold(n_splits=3)

        # Prepare targets: binary uses y_train directly; multiclass uses one-hot
        if not self.is_binary:
            n_cls = len(np.unique(y_train))
            y_onehot = np.eye(n_cls)[y_train]
        else:
            y_onehot = None

        best_score = -1
        best_params = {}

        for alpha, gamma in product(param_grid["alpha"], param_grid["gamma"]):
            scores = []
            for tr_idx, va_idx in inner_cv.split(X_train, y_train, groups=groups_train):
                kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
                if self.is_binary:
                    kr.fit(X_train[tr_idx], y_train[tr_idx])
                    raw = kr.predict(X_train[va_idx])
                    preds = (np.clip(raw, 0, 1) >= 0.5).astype(int)
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
            f"      KR best params: {best_params} (bal_acc={best_score:.3f})",
            flush=True,
        )

        # Refit on full training set with best params
        best_model = KernelRidge(
            kernel="rbf", alpha=best_params["alpha"], gamma=best_params["gamma"]
        )
        if self.is_binary:
            best_model.fit(X_train, y_train)
        else:
            n_cls = len(np.unique(y_train))
            y_onehot_full = np.eye(n_cls)[y_train]
            best_model.fit(X_train, y_onehot_full)
        return best_model, best_params

    def _predict_sklearn(self, model, X):
        """Return (probabilities, predicted_labels) for sklearn models.

        Binary: RF uses predict_proba[:,1]; KR clips to [0,1].
        Multiclass: RF returns (n, n_classes) proba natively;
                    KR returns raw regression, softmax for pseudo-probs.
        """
        if isinstance(model, RandomForestClassifier):
            probs = model.predict_proba(X)
            if self.is_binary:
                probs = probs[:, 1]
                preds = (probs >= 0.5).astype(int)
            else:
                preds = np.argmax(probs, axis=1)
        elif isinstance(model, KernelRidge):
            raw = model.predict(X)
            if self.is_binary:
                probs = np.clip(raw, 0, 1)
                preds = (probs >= 0.5).astype(int)
            else:
                # Softmax over raw regression outputs for pseudo-probs
                exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
                probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
                preds = np.argmax(probs, axis=1)
        else:
            raise ValueError(f"Unknown sklearn model type: {type(model)}")
        return probs, preds

    def _train_single_regressor(self, X_train, y_train, groups_train):
        """Inner 3-fold GroupKFold grid search for Ridge regression.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
        y_train : np.ndarray, shape (n_train,)  — 1-D marker values
        groups_train : array-like — subject groups (no leakage)

        Returns
        -------
        model : Ridge
            Best model refitted on full X_train.
        best_val_r2 : float
        best_info : dict
        """
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        n_unique_groups = len(np.unique(groups_train))

        # Skip inner CV if zero-variance or too few groups
        if np.std(y_train) == 0 or n_unique_groups < 3:
            best_alpha = 1.0
            best_val_r2 = float("nan")
        else:
            inner_cv = GroupKFold(n_splits=3)
            best_score = -float("inf")
            best_alpha = 1.0

            for alpha in alphas:
                scores = []
                for tr_idx, va_idx in inner_cv.split(
                    X_train, y_train, groups=groups_train
                ):
                    m = Ridge(alpha=alpha)
                    m.fit(X_train[tr_idx], y_train[tr_idx])
                    pred = m.predict(X_train[va_idx])
                    scores.append(r2_score(y_train[va_idx], pred))

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            best_val_r2 = best_score

        # Refit on full training set
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        return model, best_val_r2, {"alpha": best_alpha}

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_test_metrics(self, y_test, test_preds, test_probs, subjects_test):
        """Compute standard classification metrics on a test set.

        Handles both binary (1-D probs) and multiclass (2-D probs) cases.
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
            # Multiclass OvR macro AUC; guard for single-class test sets
            if n_classes_present >= 2 and test_probs.ndim == 2:
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

    # ------------------------------------------------------------------
    # Single-split classification (all 3 models)
    # ------------------------------------------------------------------

    def run_classification(
        self,
        test_size=0.2,
        val_size=0.2,
        target="crs",
        labels_file=None,
        binary_outcome=False,
    ):
        """Run classification with all 3 models on the same train/test split.

        Steps
        -----
        1. Collect data (embeddings + labels)
        2. Outer split: GroupShuffleSplit -> trainval / test (shared)
        3. Inner split: GroupShuffleSplit -> train / val (shared)
        4. Train all 3 models on the same splits
        5. Evaluate each on held-out test set
        6. Save results + plots per model
        """
        mode_tag = f" [{target}{'  binary' if binary_outcome else ''}]"
        print("=" * 80, flush=True)
        print(f"EMBEDDING CLASSIFICATION{mode_tag} — 3 MODELS", flush=True)
        print("=" * 80, flush=True)
        print(f"Data directory: {self.data_dir}", flush=True)
        print(f"Labels file: {labels_file or self.patient_labels_file}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(
            f"MLP epochs: {self.n_epochs}, LR: {self.lr}, Batch: {self.batch_size}",
            flush=True,
        )
        print(flush=True)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 1. Collect data
        X, y, subjects = self.collect_data(
            target=target, labels_file=labels_file, binary_outcome=binary_outcome
        )

        # 2. Subject groups (prevent leakage)
        groups = np.array([s.split("_ses-")[0] for s in subjects])
        unique_groups = np.unique(groups)
        print(
            f"   {len(unique_groups)} unique subjects across {len(subjects)} sessions",
            flush=True,
        )

        # 3. Outer split: trainval / test (shared across all models)
        gss_outer = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_state
        )
        trainval_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        groups_trainval = groups[trainval_idx]
        subjects_test = [subjects[i] for i in test_idx]

        # 4. Inner split: train / val (shared across all models)
        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=self.random_state
        )
        train_idx_inner, val_idx_inner = next(
            gss_inner.split(X_trainval, y_trainval, groups=groups_trainval)
        )

        X_train = X_trainval[train_idx_inner]
        y_train = y_trainval[train_idx_inner]
        X_val = X_trainval[val_idx_inner]
        y_val = y_trainval[val_idx_inner]
        groups_train = groups_trainval[train_idx_inner]
        groups_val = groups_trainval[val_idx_inner]

        subjects_train = [subjects[trainval_idx[i]] for i in train_idx_inner]
        subjects_val = [subjects[trainval_idx[i]] for i in val_idx_inner]

        # Verify no subject leakage across any pair of splits
        train_groups_set = set(groups_train)
        val_groups_set = set(groups_val)
        test_groups_set = set(groups[test_idx])

        if train_groups_set & val_groups_set:
            raise ValueError(f"Train/val leakage: {train_groups_set & val_groups_set}")
        if train_groups_set & test_groups_set:
            raise ValueError(
                f"Train/test leakage: {train_groups_set & test_groups_set}"
            )
        if val_groups_set & test_groups_set:
            raise ValueError(f"Val/test leakage: {val_groups_set & test_groups_set}")

        print(
            f"   Train: {len(X_train)} sessions ({len(train_groups_set)} subjects)",
            flush=True,
        )
        print(
            f"   Val:   {len(X_val)} sessions ({len(val_groups_set)} subjects)",
            flush=True,
        )
        print(
            f"   Test:  {len(X_test)} sessions ({len(test_groups_set)} subjects)",
            flush=True,
        )

        # Class balance info
        for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            u, c = np.unique(y_split, return_counts=True)
            dist = ", ".join(f"{self.class_names[ci]}: {n}" for ci, n in zip(u, c))
            print(f"   {name} class distribution: {dist}", flush=True)

        # pos_weight for BCEWithLogitsLoss (binary only)
        if self.is_binary:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            pos_weight = n_neg / max(n_pos, 1)
            print(f"   pos_weight (class balance): {pos_weight:.3f}", flush=True)
        else:
            pos_weight = 1.0  # ignored for multiclass (CrossEntropyLoss uses weights)
        n_classes_mlp = 1 if self.is_binary else len(self.class_names)

        all_results = {}

        # Output goes under classic_split/
        split_dir = op.join(self.output_dir, "classic_split")
        os.makedirs(split_dir, exist_ok=True)

        # ---- MLP ----
        print(
            f"\n   Training MLP (max {self.n_epochs} epochs, "
            f"patience {self.patience}) ...",
            flush=True,
        )
        mlp_model, train_losses, val_losses, train_errors, val_errors = (
            self._train_model(
                X_train, y_train, X_val, y_val, pos_weight, n_classes=n_classes_mlp
            )
        )
        n_trained_epochs = len(train_losses)
        print(f"   MLP training stopped after {n_trained_epochs} epochs", flush=True)

        val_probs, val_preds = self._predict(mlp_model, X_val)
        val_bal_acc = balanced_accuracy_score(y_val, val_preds)
        print(f"   MLP val balanced accuracy: {val_bal_acc:.3f}", flush=True)

        test_probs, test_preds = self._predict(mlp_model, X_test)
        mlp_results = self._compute_test_metrics(
            y_test, test_preds, test_probs, subjects_test
        )
        mlp_results.update(
            {
                "val_balanced_accuracy": float(val_bal_acc),
                "val_accuracy": float(accuracy_score(y_val, val_preds)),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "training_epochs": n_trained_epochs,
                "subjects_train": subjects_train,
                "subjects_val": subjects_val,
            }
        )

        mlp_dir = op.join(split_dir, "mlp")
        os.makedirs(mlp_dir, exist_ok=True)
        self._save_results(mlp_results, mlp_dir, model=mlp_model, model_type="mlp")
        self._plot_results(
            mlp_results,
            "MLP",
            mlp_dir,
            train_losses=train_losses,
            val_losses=val_losses,
            train_errors=train_errors,
            val_errors=val_errors,
        )
        all_results["mlp"] = mlp_results

        # ---- Random Forest ----
        print("\n   Training Random Forest (grid search) ...", flush=True)
        rf_model, rf_best_params = self._grid_search_rf(
            X_trainval, y_trainval, groups_trainval
        )
        rf_probs, rf_preds = self._predict_sklearn(rf_model, X_test)
        rf_results = self._compute_test_metrics(
            y_test, rf_preds, rf_probs, subjects_test
        )
        rf_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "best_params": rf_best_params,
                "subjects_train": subjects_train + subjects_val,
            }
        )

        rf_dir = op.join(split_dir, "random_forest")
        os.makedirs(rf_dir, exist_ok=True)
        self._save_results(
            rf_results, rf_dir, model=rf_model, model_type="random_forest"
        )
        self._plot_results(
            rf_results,
            "Random Forest",
            rf_dir,
            best_params=rf_best_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["random_forest"] = rf_results

        # ---- Kernel Ridge ----
        print("\n   Training Kernel Ridge (grid search) ...", flush=True)
        kr_model, kr_best_params = self._grid_search_kr(
            X_trainval, y_trainval, groups_trainval
        )
        kr_probs, kr_preds = self._predict_sklearn(kr_model, X_test)
        kr_results = self._compute_test_metrics(
            y_test, kr_preds, kr_probs, subjects_test
        )
        kr_results.update(
            {
                "n_train": len(X_trainval),
                "n_test": len(X_test),
                "n_features": int(X.shape[1]),
                "best_params": kr_best_params,
                "subjects_train": subjects_train + subjects_val,
            }
        )

        kr_dir = op.join(split_dir, "kernel_ridge")
        os.makedirs(kr_dir, exist_ok=True)
        self._save_results(
            kr_results, kr_dir, model=kr_model, model_type="kernel_ridge"
        )
        self._plot_results(
            kr_results,
            "Kernel Ridge",
            kr_dir,
            best_params=kr_best_params,
            n_samples=len(X_trainval),
            n_features=X.shape[1],
            class_balance=dict(zip(*np.unique(y_trainval, return_counts=True))),
        )
        all_results["kernel_ridge"] = kr_results

        # Save subject intersection info if used
        if self.allowed_subjects is not None:
            intersection_file = op.join(split_dir, "subject_intersection.json")
            with open(intersection_file, "w") as f:
                json.dump(
                    {
                        "allowed_subjects": self.allowed_subjects,
                        "n_subjects": len(self.allowed_subjects),
                    },
                    f,
                    indent=2,
                )

        # Summary
        print("\n" + "=" * 80, flush=True)
        print("EMBEDDING CLASSIFICATION SUMMARY (classic_split)", flush=True)
        print("=" * 80, flush=True)
        for mk, cfg in MODEL_CONFIGS.items():
            r = all_results[mk]
            auc_str = (
                f"{r['test_auc_score']:.3f}"
                if r["test_auc_score"] is not None
                else "N/A"
            )
            print(
                f"   {cfg['name']:15s}  bal_acc={r['test_balanced_accuracy']:.3f}  "
                f"AUC={auc_str}",
                flush=True,
            )
        print(f"   Results saved to: {self.output_dir}", flush=True)
        print("=" * 80, flush=True)

        return all_results

    # ------------------------------------------------------------------
    # Full cross-validation (all 3 models)
    # ------------------------------------------------------------------

    def run_full_cv(
        self,
        target="crs",
        labels_file=None,
        binary_outcome=False,
    ):
        """Run 5-fold StratifiedGroupKFold CV with all 3 models.

        For each fold:
        - MLP: inner GroupShuffleSplit for train/val, early stopping.
          Per-fold training curves (loss + balanced error) are collected
          and overlaid on the final MLP plot.
        - RF: inner 3-fold GroupKFold grid search, refit on full train fold
        - KR: inner 3-fold GroupKFold grid search, refit on full train fold
        All predict on the same held-out fold.
        """
        n_folds = self.n_cv_folds
        mode_tag = f" [{target}{'  binary' if binary_outcome else ''}]"

        print("=" * 80, flush=True)
        print(
            f"FULL {n_folds}-FOLD CV EMBEDDING CLASSIFICATION{mode_tag} — 3 MODELS",
            flush=True,
        )
        print("=" * 80, flush=True)
        print(f"Data directory: {self.data_dir}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)
        print(flush=True)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 1. Collect data
        X, y, subjects = self.collect_data(
            target=target, labels_file=labels_file, binary_outcome=binary_outcome
        )

        # Subject groups
        groups = np.array([s.split("_ses-")[0] for s in subjects])
        n_unique = len(np.unique(groups))
        print(
            f"   {n_unique} unique subjects across {len(subjects)} sessions", flush=True
        )

        effective_folds = min(n_folds, n_unique)
        if effective_folds < 2:
            raise ValueError(
                f"Only {n_unique} unique subjects — cannot perform {n_folds}-fold CV"
            )

        if self.is_binary:
            sgkf = StratifiedGroupKFold(
                n_splits=effective_folds, shuffle=True, random_state=self.random_state
            )
        else:
            sgkf = GroupKFold(n_splits=effective_folds)

        # Accumulators per model
        accum = {
            mk: {"y_true": [], "y_pred": [], "y_proba": [], "subjects": []}
            for mk in MODEL_CONFIGS
        }

        # Per-fold MLP training curves for overlay plot
        mlp_fold_curves = []  # list of (train_losses, val_losses, train_errors, val_errors)

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

            # Verify no leakage
            train_subj = set(groups[train_idx])
            test_subj = set(groups[test_idx])
            if train_subj & test_subj:
                raise ValueError(
                    f"Fold {fold_idx}: subject leakage: {train_subj & test_subj}"
                )
            print(
                f"   Train: {len(train_subj)} subjects ({len(train_idx)} sessions)",
                flush=True,
            )
            print(
                f"   Test:  {len(test_subj)} subjects ({len(test_idx)} sessions)",
                flush=True,
            )

            # ---- MLP: inner split for val ----
            print("   Training MLP ...", flush=True)
            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=self.random_state + fold_idx
            )
            tr_inner, va_inner = next(
                gss_val.split(X_train_fold, y_train_fold, groups=groups_train_fold)
            )

            X_tr = X_train_fold[tr_inner]
            y_tr = y_train_fold[tr_inner]
            X_va = X_train_fold[va_inner]
            y_va = y_train_fold[va_inner]

            if self.is_binary:
                n_pos = (y_tr == 1).sum()
                n_neg = (y_tr == 0).sum()
                pos_weight = n_neg / max(n_pos, 1)
            else:
                pos_weight = 1.0
            n_classes_mlp = 1 if self.is_binary else len(self.class_names)

            mlp_model, train_losses, val_losses, train_errors, val_errors = (
                self._train_model(
                    X_tr, y_tr, X_va, y_va, pos_weight, n_classes=n_classes_mlp
                )
            )
            mlp_fold_curves.append((train_losses, val_losses, train_errors, val_errors))
            # Save per-fold MLP model for later SHAP analysis
            mlp_fold_dir = op.join(self.output_dir, "nested_cv", "mlp")
            os.makedirs(mlp_fold_dir, exist_ok=True)
            torch.save(
                mlp_model.state_dict(),
                op.join(mlp_fold_dir, f"fold_{fold_idx:02d}_model.pt"),
            )
            with open(
                op.join(mlp_fold_dir, f"fold_{fold_idx:02d}_config.json"), "w"
            ) as f:
                json.dump(
                    {
                        "input_dim": int(X.shape[1]),
                        "n_classes": int(n_classes_mlp),
                    },
                    f,
                )
            mlp_probs, mlp_preds = self._predict(mlp_model, X_test_fold)
            accum["mlp"]["y_true"].append(y_test_fold)
            accum["mlp"]["y_pred"].append(mlp_preds)
            accum["mlp"]["y_proba"].append(mlp_probs)
            accum["mlp"]["subjects"].extend(subjects_test_fold)

            ba_mlp = balanced_accuracy_score(y_test_fold, mlp_preds)
            print(f"   MLP fold bal_acc: {ba_mlp:.3f}", flush=True)

            # ---- Random Forest ----
            print("   Training Random Forest ...", flush=True)
            rf_model, _ = self._grid_search_rf(
                X_train_fold, y_train_fold, groups_train_fold
            )
            # Save per-fold RF model
            rf_fold_dir = op.join(self.output_dir, "nested_cv", "random_forest")
            os.makedirs(rf_fold_dir, exist_ok=True)
            joblib.dump(
                rf_model,
                op.join(rf_fold_dir, f"fold_{fold_idx:02d}_model.joblib"),
            )
            rf_probs, rf_preds = self._predict_sklearn(rf_model, X_test_fold)
            accum["random_forest"]["y_true"].append(y_test_fold)
            accum["random_forest"]["y_pred"].append(rf_preds)
            accum["random_forest"]["y_proba"].append(rf_probs)
            accum["random_forest"]["subjects"].extend(subjects_test_fold)

            ba_rf = balanced_accuracy_score(y_test_fold, rf_preds)
            print(f"   RF fold bal_acc: {ba_rf:.3f}", flush=True)

            # ---- Kernel Ridge ----
            print("   Training Kernel Ridge ...", flush=True)
            kr_model, _ = self._grid_search_kr(
                X_train_fold, y_train_fold, groups_train_fold
            )
            # Save per-fold KR model
            kr_fold_dir = op.join(self.output_dir, "nested_cv", "kernel_ridge")
            os.makedirs(kr_fold_dir, exist_ok=True)
            joblib.dump(
                kr_model,
                op.join(kr_fold_dir, f"fold_{fold_idx:02d}_model.joblib"),
            )
            kr_probs, kr_preds = self._predict_sklearn(kr_model, X_test_fold)
            accum["kernel_ridge"]["y_true"].append(y_test_fold)
            accum["kernel_ridge"]["y_pred"].append(kr_preds)
            accum["kernel_ridge"]["y_proba"].append(kr_probs)
            accum["kernel_ridge"]["subjects"].extend(subjects_test_fold)

            ba_kr = balanced_accuracy_score(y_test_fold, kr_preds)
            print(f"   KR fold bal_acc: {ba_kr:.3f}", flush=True)

        # Aggregate across folds
        print(f"\n{'=' * 60}", flush=True)
        print("AGGREGATING RESULTS ACROSS ALL FOLDS", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Output goes under nested_cv/
        cv_dir = op.join(self.output_dir, "nested_cv")
        os.makedirs(cv_dir, exist_ok=True)

        for model_key, cfg in MODEL_CONFIGS.items():
            y_true_all = np.concatenate(accum[model_key]["y_true"])
            y_pred_all = np.concatenate(accum[model_key]["y_pred"])
            y_proba_all = np.concatenate(accum[model_key]["y_proba"])
            subjects_all = accum[model_key]["subjects"]

            results = self._compute_test_metrics(
                y_true_all, y_pred_all, y_proba_all, subjects_all
            )
            results.update(
                {
                    "n_samples": len(X),
                    "n_features": int(X.shape[1]),
                    "n_subjects": int(n_unique),
                    "full_cv": True,
                    "n_folds": effective_folds,
                }
            )

            model_dir = op.join(cv_dir, model_key)
            os.makedirs(model_dir, exist_ok=True)
            self._save_results(results, model_dir, model=None, model_type=model_key)

            # MLP gets per-fold training curves overlay
            if model_key == "mlp":
                self._plot_results(
                    results,
                    cfg["name"],
                    model_dir,
                    fold_curves=mlp_fold_curves,
                )
            else:
                self._plot_results(results, cfg["name"], model_dir)

            auc_str = (
                f"{results['test_auc_score']:.3f}"
                if results["test_auc_score"] is not None
                else "N/A"
            )
            print(
                f"   {cfg['name']:15s}  bal_acc="
                f"{results['test_balanced_accuracy']:.3f}  "
                f"AUC={auc_str}",
                flush=True,
            )

        # Save subject intersection info if used
        if self.allowed_subjects is not None:
            intersection_file = op.join(cv_dir, "subject_intersection.json")
            with open(intersection_file, "w") as f:
                json.dump(
                    {
                        "allowed_subjects": self.allowed_subjects,
                        "n_subjects": len(self.allowed_subjects),
                    },
                    f,
                    indent=2,
                )

        print(f"\n   Results saved to: {cv_dir}", flush=True)
        print("=" * 80, flush=True)

    # ------------------------------------------------------------------
    # Marker regression
    # ------------------------------------------------------------------

    def run_marker_regression(self, marker_csv, reduction_letter, test_size=0.2):
        """Predict each scalar marker from embeddings using Ridge regression.

        Uses group-based CV to prevent subject-level leakage.

        Parameters
        ----------
        marker_csv : str
            Path to baseline scalars CSV (semicolon-delimited).
        reduction_letter : str
            One of A/B/C/D; selects Reduction row in CSV.
        test_size : float
            Fraction of subjects for outer test split.
        """
        print("=" * 80, flush=True)
        print("MARKER REGRESSION (embeddings -> scalar markers)", flush=True)
        print("=" * 80, flush=True)

        X, Y, subjects, marker_names = self.collect_data_for_regression(
            marker_csv, reduction_letter
        )
        groups = np.array([s.split("_ses-")[0] for s in subjects])

        output_dir = op.join(self.output_dir, "regressor_results")
        os.makedirs(output_dir, exist_ok=True)

        metrics_per_marker = {}
        all_preds_dict = {}

        if not self.full_cv:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.random_state
            )
            splits = list(gss.split(X, groups=groups))
            train_idx, test_idx = splits[0]

            for i, marker in enumerate(marker_names):
                y_col = Y[:, i]

                # NaN mask
                valid_train = train_idx[~np.isnan(y_col[train_idx])]
                valid_test = test_idx[~np.isnan(y_col[test_idx])]

                if len(valid_train) < 10:
                    print(
                        f"   Skipping {marker}: only {len(valid_train)} valid train samples",
                        flush=True,
                    )
                    metrics_per_marker[marker] = {"skipped": True}
                    continue

                model, best_val_r2, best_info = self._train_single_regressor(
                    X[valid_train], y_col[valid_train], groups[valid_train]
                )

                y_pred = model.predict(X[valid_test])
                y_true = y_col[valid_test]
                test_subjects = [subjects[j] for j in valid_test]

                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                try:
                    pearson_r, pearson_p = pearsonr(y_true, y_pred)
                except Exception:
                    pearson_r, pearson_p = float("nan"), float("nan")

                metrics_per_marker[marker] = {
                    "r2": float(r2),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "mse": float(mse),
                    "mae": float(mae),
                    "best_alpha": best_info["alpha"],
                    "n_train": len(valid_train),
                    "n_test": len(valid_test),
                    "skipped": False,
                }
                all_preds_dict[marker] = {
                    "subjects": test_subjects,
                    "true": y_true.tolist(),
                    "pred": y_pred.tolist(),
                }

        else:
            # Full CV: GroupKFold, accumulate predictions per fold
            n_unique = len(np.unique(groups))
            effective_folds = min(self.n_cv_folds, n_unique)
            gkf = GroupKFold(n_splits=effective_folds)

            # Accumulators: {marker_idx: {true: [], pred: [], subjects: []}}
            accum = {
                i: {"true": [], "pred": [], "subjects": []}
                for i in range(Y.shape[1])
            }

            for fold_idx, (train_idx, test_idx) in enumerate(
                gkf.split(X, groups=groups)
            ):
                print(f"   Fold {fold_idx + 1}/{effective_folds}", flush=True)
                for i in range(Y.shape[1]):
                    y_col = Y[:, i]
                    valid_train = train_idx[~np.isnan(y_col[train_idx])]
                    valid_test = test_idx[~np.isnan(y_col[test_idx])]

                    if len(valid_train) < 10:
                        continue

                    model, _, _ = self._train_single_regressor(
                        X[valid_train], y_col[valid_train], groups[valid_train]
                    )
                    y_pred = model.predict(X[valid_test])
                    accum[i]["true"].extend(y_col[valid_test].tolist())
                    accum[i]["pred"].extend(y_pred.tolist())
                    accum[i]["subjects"].extend(
                        [subjects[j] for j in valid_test]
                    )

            for i, marker in enumerate(marker_names):
                if not accum[i]["true"]:
                    metrics_per_marker[marker] = {"skipped": True}
                    continue
                y_true = np.array(accum[i]["true"])
                y_pred = np.array(accum[i]["pred"])
                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                try:
                    pearson_r, pearson_p = pearsonr(y_true, y_pred)
                except Exception:
                    pearson_r, pearson_p = float("nan"), float("nan")

                metrics_per_marker[marker] = {
                    "r2": float(r2),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "mse": float(mse),
                    "mae": float(mae),
                    "n_train": int(len(X) - len(X) // effective_folds),
                    "n_test": len(y_true),
                    "skipped": False,
                }
                all_preds_dict[marker] = {
                    "subjects": accum[i]["subjects"],
                    "true": y_true.tolist(),
                    "pred": y_pred.tolist(),
                }

        self._save_regression_results(
            marker_names, metrics_per_marker, all_preds_dict, output_dir
        )
        print(f"\n   Regression results saved to: {output_dir}", flush=True)
        print("=" * 80, flush=True)
        return metrics_per_marker

    def _save_regression_results(
        self, marker_names, metrics_per_marker, all_preds_dict, output_dir
    ):
        """Save regression summary JSON, per-marker prediction CSVs, and bar chart."""
        # Summary JSON
        summary_file = op.join(output_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(metrics_per_marker, f, indent=2)
        print(f"   Summary saved to: {summary_file}", flush=True)

        # Per-marker prediction CSVs
        per_marker_dir = op.join(output_dir, "per_marker")
        for marker, preds in all_preds_dict.items():
            safe_name = marker.replace("/", "__")
            mdir = op.join(per_marker_dir, safe_name)
            os.makedirs(mdir, exist_ok=True)
            df = pd.DataFrame(
                {
                    "subject_session": preds["subjects"],
                    "true_value": preds["true"],
                    "predicted_value": preds["pred"],
                    "squared_error": [
                        (t - p) ** 2
                        for t, p in zip(preds["true"], preds["pred"])
                    ],
                }
            )
            df.to_csv(op.join(mdir, "predictions.csv"), index=False)

        # Bar chart: R² per marker (sorted descending)
        valid_metrics = {
            m: v for m, v in metrics_per_marker.items() if not v.get("skipped", False)
        }
        if valid_metrics:
            sorted_markers = sorted(
                valid_metrics, key=lambda m: valid_metrics[m]["r2"], reverse=True
            )
            r2_vals = [valid_metrics[m]["r2"] for m in sorted_markers]

            fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_markers) * 0.35)))
            y_pos = np.arange(len(sorted_markers))
            ax.barh(y_pos, r2_vals, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_markers, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8, ls="--")
            ax.set_xlabel("R²")
            ax.set_title("Marker Regression Performance (R²)")
            ax.invert_yaxis()
            plt.tight_layout()
            chart_file = op.join(output_dir, "marker_performance.png")
            plt.savefig(chart_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   Chart saved to: {chart_file}", flush=True)

    # ------------------------------------------------------------------
    # Full metric prediction (5 targets)
    # ------------------------------------------------------------------

    def run_full_metric_prediction(
        self, patient_labels_full, test_size=0.2, val_size=0.2
    ):
        """Run MLP+RF+KR classification for 5 prediction targets.

        Targets: crs, etiology, cs_6m, cs_1y, cs_2y.

        For crs and etiology results land under {output_dir}/{target}/.
        For outcome targets (cs_6m, cs_1y, cs_2y) two runs are performed:
          - {output_dir}/{target}/multiclass/  — 4-class (VS, MCS, CONSCIOUS, DEATH)
          - {output_dir}/{target}/binary/      — VS vs MCS only

        Parameters
        ----------
        patient_labels_full : str
            Path to patient_labels.csv with all target columns.
        test_size : float
        val_size : float
        """
        _OUTCOME_TARGETS = {"cs_6m", "cs_1y", "cs_2y"}
        targets = ["crs", "etiology", "cs_6m", "cs_1y", "cs_2y"]
        base_output_dir = self.output_dir

        for target in targets:
            print(f"\n{'=' * 80}", flush=True)
            print(f"FULL METRIC PREDICTION: target={target}", flush=True)
            print(f"{'=' * 80}", flush=True)

            if target in _OUTCOME_TARGETS:
                modes = [("multiclass", False), ("binary", True)]
            else:
                modes = [("", False)]  # single run, no subfolder

            for mode, binary_outcome in modes:
                if mode:
                    print(f"\n--- {target} / {mode} ---", flush=True)
                    self.output_dir = op.join(base_output_dir, target, mode)
                else:
                    self.output_dir = op.join(base_output_dir, target)
                os.makedirs(self.output_dir, exist_ok=True)

                try:
                    if self.full_cv:
                        self.run_full_cv(
                            target=target,
                            labels_file=patient_labels_full,
                            binary_outcome=binary_outcome,
                        )
                    else:
                        self.run_classification(
                            test_size,
                            val_size,
                            target=target,
                            labels_file=patient_labels_full,
                            binary_outcome=binary_outcome,
                        )
                except ValueError as e:
                    label = f"'{target}'" + (f" ({mode})" if mode else "")
                    print(f"   Skipping {label}: {e}", flush=True)

        self.output_dir = base_output_dir

    # ------------------------------------------------------------------
    # Saving & plotting
    # ------------------------------------------------------------------

    def _save_results(self, results, output_dir, model=None, model_type="mlp"):
        """Save JSON results, trained model, and predictions CSV."""
        results_file = op.join(output_dir, "classification_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}", flush=True)

        # Save model (single-split only, not in full CV)
        if model is not None:
            if model_type == "mlp":
                model_file = op.join(output_dir, "trained_model.pt")
                torch.save(model.state_dict(), model_file)
                # Save architecture config for later reconstruction (e.g. SHAP loading)
                config_file = op.join(output_dir, "model_config.json")
                with open(config_file, "w") as f:
                    json.dump(
                        {
                            "input_dim": int(model.net[0].in_features),
                            "n_classes": int(model.n_classes),
                        },
                        f,
                    )
            else:
                model_file = op.join(output_dir, "trained_model.joblib")
                joblib.dump(model, model_file)

        # Subject predictions CSV
        pred_labels = [self.class_names[p] for p in results["y_test_pred"]]
        true_labels = [self.class_names[t] for t in results["y_test_true"]]
        df = pd.DataFrame(
            {
                "subject_session": results["subjects_test"],
                "true_state": true_labels,
                "predicted_state": pred_labels,
                "correct": [t == p for t, p in zip(true_labels, pred_labels)],
                "prob_positive": results["y_test_probs"],
            }
        )
        csv_file = op.join(output_dir, "subject_predictions.csv")
        df.to_csv(csv_file, index=False)

    def _plot_results(
        self,
        results,
        model_display_name,
        output_dir,
        train_losses=None,
        val_losses=None,
        train_errors=None,
        val_errors=None,
        fold_curves=None,
        best_params=None,
        n_samples=None,
        n_features=None,
        class_balance=None,
    ):
        """Generate a 2x2 figure: loss/info, confusion matrix, error/info, ROC.

        Parameters
        ----------
        fold_curves : list of tuples, optional
            Per-fold MLP curves for nested CV overlay. Each element is
            ``(train_losses, val_losses, train_errors, val_errors)``.
            When provided, all folds are overlaid on the loss and error
            axes with low opacity (one color per role: blue=train,
            orange=val), so fold-to-fold variability is visible.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        n_test = len(results["y_test_true"])
        n_train = results.get("n_train", results.get("n_samples", "?"))
        n_val = results.get("n_val", "")
        subtitle_parts = [f"Train: {n_train}"]
        if n_val:
            subtitle_parts.append(f"Val: {n_val}")
        subtitle_parts.append(f"Test: {n_test}")

        cv_tag = ""
        if results.get("full_cv"):
            cv_tag = f" ({results['n_folds']}-fold nested CV)"

        fig.suptitle(
            f"{model_display_name} Embedding Classification Results{cv_tag}\n"
            + ", ".join(subtitle_parts),
            fontsize=16,
        )

        # Determine which curve mode we're in
        has_single_curves = train_losses is not None and val_losses is not None
        has_fold_curves = fold_curves is not None and len(fold_curves) > 0

        loss_label = "BCEWithLogitsLoss" if self.is_binary else "CrossEntropyLoss"

        # --- (0, 0) Training & validation loss ---
        ax = axes[0, 0]
        if has_fold_curves:
            n_fc = len(fold_curves)
            for fi, (tl, vl, _, _) in enumerate(fold_curves):
                epochs_x = np.arange(len(tl))
                ax.plot(
                    epochs_x,
                    tl,
                    color="tab:blue",
                    lw=1.2,
                    alpha=0.35,
                    label="Train" if fi == 0 else None,
                )
                ax.plot(
                    epochs_x,
                    vl,
                    color="tab:orange",
                    lw=1.2,
                    alpha=0.35,
                    label="Val" if fi == 0 else None,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Loss ({loss_label})")
            ax.set_title(f"Training & Validation Loss ({n_fc} folds overlaid)")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3)
        elif has_single_curves:
            epochs_x = np.arange(len(train_losses))
            ax.plot(epochs_x, train_losses, color="tab:blue", lw=2, label="Train")
            ax.plot(epochs_x, val_losses, color="tab:orange", lw=2, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Loss ({loss_label})")
            ax.set_title("Training & Validation Loss")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3)
        else:
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
                fontsize=13,
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

        # --- (1, 0) Balanced error rate ---
        ax = axes[1, 0]
        if has_fold_curves:
            n_fc = len(fold_curves)
            for fi, (_, _, te, ve) in enumerate(fold_curves):
                epochs_x = np.arange(len(te))
                ax.plot(
                    epochs_x,
                    te,
                    color="tab:blue",
                    lw=1.2,
                    alpha=0.35,
                    label="Train" if fi == 0 else None,
                )
                ax.plot(
                    epochs_x,
                    ve,
                    color="tab:orange",
                    lw=1.2,
                    alpha=0.35,
                    label="Val" if fi == 0 else None,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Balanced Error Rate (1 - Bal. Acc.)")
            ax.set_title(f"Training & Validation Error ({n_fc} folds overlaid)")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3)
        elif has_single_curves:
            epochs_x = np.arange(len(train_errors))
            ax.plot(epochs_x, train_errors, color="tab:blue", lw=2, label="Train")
            ax.plot(epochs_x, val_errors, color="tab:orange", lw=2, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Balanced Error Rate (1 - Bal. Acc.)")
            ax.set_title("Training & Validation Error")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")
            info_lines = ["Dataset Summary"]
            if n_samples is not None:
                info_lines.append(f"  Training samples: {n_samples}")
            if n_features is not None:
                info_lines.append(f"  Features: {n_features}")
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
                fontsize=13,
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
                # One OvR curve per class
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
                for ci, (cls_name, color) in enumerate(
                    zip(self.class_names, colors)
                ):
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


# Backward compatibility alias
EmbeddingMLPClassifier = EmbeddingClassifier


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-model classification on pre-computed EEG foundation model "
            "embeddings.  Supports binary VS/MCS classification (default), "
            "marker regression (--marker-regression), and multi-target "
            "clinical prediction (--full-metric-prediction)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Default: binary VS vs MCS, single train/test split
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out

# 5-fold nested cross-validation
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out --full-cv --n-cv-folds 5

# Marker regression: predict scalar markers from embeddings (reduction A)
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out \\
    --marker-regression \\
    --marker-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \\
    --marker-reduction A

# Marker regression with nested CV (reduction B)
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out --full-cv \\
    --marker-regression \\
    --marker-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \\
    --marker-reduction B

# Full metric prediction (5 targets: crs, etiology, cs_6m, cs_1y, cs_2y)
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out \\
    --full-metric-prediction \\
    --patient-labels-full /data/metadata/patient_labels.csv

# Full metric prediction with 5-fold CV
python mlp_embedding_classifier.py \\
    --data-dir /path/to/embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out --full-cv \\
    --full-metric-prediction \\
    --patient-labels-full /data/metadata/patient_labels.csv

# Subject intersection across two foundation models
python mlp_embedding_classifier.py \\
    --data-dir /path/to/totem_embeddings \\
    --patient-labels /path/to/patient_labels_with_controls.csv \\
    --output-dir /path/to/out \\
    --use-subject-intersection \\
    --embedding-dirs TOTEM=/path/to/totem CBraMod=/path/to/cbramod

Reduction map for --marker-reduction:
  A = icm/lg/egi256/trim_mean80   (default)
  B = icm/lg/egi256/std
  C = icm/lg/egi256gfp/trim_mean80
  D = icm/lg/egi256gfp/std
        """,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to directory with sub-{ID}/ses-{NUM}/*_embedding.npy",
    )
    parser.add_argument(
        "--patient-labels",
        required=True,
        help="Path to patient_labels_with_controls.csv",
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of subjects for test set"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of remaining subjects for validation set",
    )
    parser.add_argument("--random-state", type=int, default=42)

    # New: full CV
    parser.add_argument(
        "--full-cv",
        action="store_true",
        default=False,
        help="Run 5-fold StratifiedGroupKFold CV",
    )
    parser.add_argument(
        "--n-cv-folds", type=int, default=5, help="Number of CV folds (default: 5)"
    )

    # New: subject intersection
    parser.add_argument(
        "--use-subject-intersection",
        action="store_true",
        default=False,
        help="Restrict subjects to intersection across all foundation models",
    )
    parser.add_argument(
        "--embedding-dirs",
        nargs="+",
        metavar="NAME=PATH",
        help="Embedding dirs for all models (e.g., TOTEM=/path CBraMod=/path)",
    )

    # New: marker regression
    parser.add_argument(
        "--marker-regression",
        action="store_true",
        default=False,
        help="Predict scalar markers from embeddings (Ridge regression per marker)",
    )
    parser.add_argument(
        "--marker-csv",
        default=None,
        help="Path to baseline scalars CSV (required with --marker-regression)",
    )
    parser.add_argument(
        "--marker-reduction",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Reduction variant to use from marker CSV (A/B/C/D, default: A)",
    )

    # New: full metric prediction
    parser.add_argument(
        "--full-metric-prediction",
        action="store_true",
        default=False,
        help="Run classification for 5 targets: crs, etiology, cs_6m, cs_1y, cs_2y",
    )
    parser.add_argument(
        "--patient-labels-full",
        default=None,
        help="Path to patient_labels.csv (required with --full-metric-prediction)",
    )

    args = parser.parse_args()

    # Parse embedding dirs into dict
    allowed_subjects = None
    embedding_dirs_dict = None
    if args.use_subject_intersection and args.embedding_dirs:
        embedding_dirs_dict = {}
        for entry in args.embedding_dirs:
            if "=" not in entry:
                parser.error(f"--embedding-dirs entry must be NAME=PATH, got: {entry}")
            name, path = entry.split("=", 1)
            embedding_dirs_dict[name] = path

        allowed_subjects = EmbeddingClassifier.compute_subject_intersection(
            embedding_dirs_dict
        )

        # Save intersection
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            intersection_file = op.join(args.output_dir, "subject_intersection.json")
            with open(intersection_file, "w") as f:
                json.dump(
                    {
                        "allowed_subjects": allowed_subjects,
                        "n_subjects": len(allowed_subjects),
                        "embedding_dirs": embedding_dirs_dict,
                    },
                    f,
                    indent=2,
                )
            print(f"   Intersection saved to: {intersection_file}", flush=True)

    classifier = EmbeddingClassifier(
        data_dir=args.data_dir,
        patient_labels_file=args.patient_labels,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
        full_cv=args.full_cv,
        n_cv_folds=args.n_cv_folds,
        use_subject_intersection=args.use_subject_intersection,
        embedding_dirs=embedding_dirs_dict,
        allowed_subjects=allowed_subjects,
    )

    ran_special = False
    if args.marker_regression:
        if not args.marker_csv:
            parser.error("--marker-regression requires --marker-csv")
        classifier.run_marker_regression(
            args.marker_csv, args.marker_reduction, args.test_size
        )
        ran_special = True
    if args.full_metric_prediction:
        if not args.patient_labels_full:
            parser.error("--full-metric-prediction requires --patient-labels-full")
        classifier.run_full_metric_prediction(
            args.patient_labels_full, args.test_size, args.val_size
        )
        ran_special = True
    if not ran_special:
        if args.full_cv:
            classifier.run_full_cv()
        else:
            classifier.run_classification(
                test_size=args.test_size,
                val_size=args.val_size,
            )


if __name__ == "__main__":
    main()
