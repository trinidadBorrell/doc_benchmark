"""Non-linear probing on multi-layer foundation model embeddings.

================================================================
Non-Linear Probing — RandomForest Regressor & Nested CV Classification
================================================================

This script is the non-linear analogue of ``linear_probing.py``.  It probes
multi-layer embeddings from foundation models (CbraMod, NeuroLM) using
non-linear models:

1. **RandomForest Regression** (per layer):
   For each marker, train a RandomForestRegressor that predicts the marker
   scalar from the mean-pooled layer embedding.  Output: R² per marker per FM,
   saved as ``summary.json`` and plotted as ``feature_importance_raw.png``.

2. **Nested CV Classification** (per layer):
   Binary VS-vs-MCS classification using three classifiers (SVM,
   KernelRidgeClassifier, RandomForest) with StratifiedGroupKFold outer
   and GroupKFold inner CV.  Output: AUC (mean +/- std) per classifier per
   FM per layer.

The regression probe uses ``RandomForestRegressor`` (non-linear) instead of
``Ridge`` (linear), allowing it to capture non-linear relationships between
embeddings and neurophysiological markers.  The classification probe is
identical to ``linear_probing.py`` (already uses non-linear classifiers).

Pooling strategy (two-stage)
----------------------------
Reuses the same ``pooled_chan_layers.npz`` cache produced by
``linear_probing.py`` (or ``run_pool.sh``).  If the cache does not exist,
this script pools the raw multi-layer ``.npz`` files identically.

Usage
-----
::

    python non_linear_probing.py \\
        --data-dir /data/project/eeg_foundation/data \\
        --output-dir /path/to/results \\
        --patient-labels /path/to/patient_labels_with_controls.csv \\
        --marker-csv /path/to/baseline_stable_20210128_scalars.csv
"""

import argparse
import json
import os
import os.path as op
import sys
from datetime import datetime
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
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
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"

# -- Model configuration ---------------------------------------------------

# Default root for benchmark results (used in last_layer mode)
_BENCHMARK_RESULTS_DEFAULT = (
    "/data/project/eeg_foundation/data/benchmark_results/new_results"
)

# Maps script model name -> benchmark results subdirectory name.
# Used in last_layer mode to locate pooled_embeddings.
LAST_LAYER_RESULT_DIRS = {
    "CbraMod": "CBraMod",
    "NeuroLM": "NeuroLM",
    "TOTEM": "TOTEM",
    "LaBram": "LaBram",
    "BIOT": "BIOT",
}

# Models valid for each mode
MULTILAYER_MODELS = ["CbraMod", "NeuroLM", "BIOT"]
LAST_LAYER_MODELS = ["CbraMod", "NeuroLM", "TOTEM", "LaBram", "BIOT"]

MODEL_LAYER_KEYS = {
    "CbraMod": {
        "data_subdir": "CbraMod/multi_layer_embeddings",
        "layers": [
            "patch_emb",
            "layer_0",
            "layer_3",
            "layer_6",
            "layer_9",
            "layer_11",
        ],
        "disk_pool_axes": (0, 2),
        "n_channels": 256,
        "emb_dim": 200,
    },
    "NeuroLM": {
        "data_subdir": "NeuroLM/multi_layer_embeddings",
        "layers": ["vq_emb", "gpt_0", "gpt_3", "gpt_6", "gpt_9", "gpt_11"],
        "disk_pool_axes": (0,),
        "n_channels": 256,
        "emb_dim": 1024,
        "emb_dim_override": {"vq_emb": 768},
    },
    "BIOT": {
        # Raw data lives outside results_root — specify absolute path directly.
        "abs_data_dir": "/data/project/eeg_foundation/data/BIOT",
        "data_subdir": "BIOT",  # used only for pooled cache path
        # BIOT sessions contain both *_layers.npz and *_embeddings.npz.
        # We must load the layers file to get all intermediate representations.
        "npz_suffix": "_layers.npz",
        "layers": [
            "pre_transformer",
            "transformer_0",
            "transformer_1",
            "transformer_2",
            "transformer_3",
            "final_emb",
        ],
        # Transformer layers: (n_epochs, 36, 256) → mean over epochs → (36, 256)
        # final_emb:          (n_epochs, 256)     → mean over epochs → (256,)
        "disk_pool_axes": (0,),
        "n_channels": 36,
        "emb_dim": 256,
        # final_emb has no channel dimension after disk pooling → skip RAM channel pool.
        "layer_no_chan_pool": ["final_emb"],
    },
}

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

_MODEL_COLORS = {
    "CbraMod": "#2ca02c",
    "NeuroLM": "#1f77b4",
    "TOTEM": "#ff7f0e",
    "LaBram": "#d62728",
    "BIOT": "#9467bd",
}

FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_COLORS = {
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",
    "wsmi": "#8b4513",
    "evoked": "#c55a11",
}


# -- Helpers ----------------------------------------------------------------


def _safe_auc_score(y_true, y_score, fallback_preds):
    """AUC with balanced_accuracy fallback for single-class folds."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return balanced_accuracy_score(y_true, fallback_preds)


def _short_name(full_path: str) -> str:
    """``nice/marker/PowerSpectralDensity/theta`` -> ``PowerSpectralDensity_theta``."""
    parts = full_path.split("/")
    return "_".join(parts[-2:])


def _marker_family(marker_name: str) -> str:
    """Map marker name to family bucket for grouped x-axis ordering."""
    if "KolmogorovComplexity" in marker_name or "PermutationEntropy" in marker_name:
        return "information_theory"
    if "PowerSpectralDensity" in marker_name:
        return "power_spectral_density"
    if "SymbolicMutualInformation" in marker_name:
        return "wsmi"
    return "evoked"


def _ordered_markers_by_family_and_mean(r2_per_model: dict) -> list:
    """Order markers by family, then ascending mean raw R2 across models."""
    all_markers = set()
    for dct in r2_per_model.values():
        all_markers.update(dct.keys())

    def _mean_raw(marker):
        vals = [r2_per_model[m].get(marker, np.nan) for m in r2_per_model]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    families = {fam: [] for fam in FAMILY_ORDER}
    for marker in all_markers:
        fam = _marker_family(marker)
        families.setdefault(fam, []).append(marker)

    ordered = []
    for fam in FAMILY_ORDER:
        fam_markers = families.get(fam, [])
        fam_markers = sorted(fam_markers, key=lambda m: (_mean_raw(m), _short_name(m)))
        ordered.extend(fam_markers)
    return ordered


def _color_xticks_by_family(ax, marker_order: list) -> None:
    """Color x tick labels according to marker family."""
    for tick, marker in zip(ax.get_xticklabels(), marker_order):
        fam = _marker_family(marker)
        tick.set_color(FAMILY_COLORS.get(fam, "black"))


# -- NonLinearProber class --------------------------------------------------


class NonLinearProber:
    """Probe multi-layer foundation model embeddings via RandomForest and classifiers.

    Non-linear analogue of ``LinearProber`` from ``linear_probing.py``.
    The regression probe uses ``RandomForestRegressor`` instead of ``Ridge``.
    The classification probe is identical (already uses non-linear classifiers).

    The pooled embedding cache (``pooled_chan_layers.npz``) is shared with
    ``LinearProber`` — both scripts read from the same cache directory.

    Parameters
    ----------
    data_dir : str
        Root data directory containing model embedding subdirectories.
    output_dir : str
        Directory where all results (JSON, plots) are saved.  Should be
        different from the linear probing output dir (e.g. NON_LINEAR_PROBING).
    patient_labels_file : str
        Path to ``patient_labels_with_controls.csv``.
    marker_csv : str or None
        Path to ``baseline_stable_20210128_scalars.csv``.
    reduction : str
        One of A/B/C/D.
    models : list of str
    layers : list of str or ``"all"``
    n_cv_folds : int
    random_state : int
    full_cv : bool
    """

    def __init__(
        self,
        data_dir,
        output_dir,
        patient_labels_file,
        marker_csv=None,
        reduction="A",
        models=None,
        layers="all",
        n_cv_folds=5,
        random_state=42,
        full_cv=True,
        precomputed_splits=None,
        common_sessions=None,
        mode="multilayer",
        benchmark_results_dir=_BENCHMARK_RESULTS_DEFAULT,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patient_labels_file = patient_labels_file
        self.marker_csv = marker_csv
        self.reduction = reduction
        self.mode = mode  # "multilayer" or "last_layer"
        self.benchmark_results_dir = benchmark_results_dir
        if models is None:
            models = LAST_LAYER_MODELS if mode == "last_layer" else MULTILAYER_MODELS
        self.models = models
        self.layers_filter = layers
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.full_cv = full_cv
        self.precomputed_splits = precomputed_splits
        self.common_sessions = common_sessions

        os.makedirs(output_dir, exist_ok=True)

        self._markers_cache = None
        self._marker_names_cache = None
        self._labels_cache = None

    def _get_layers(self, model_name):
        """Return the list of layer keys to probe for a given model."""
        all_layers = MODEL_LAYER_KEYS[model_name]["layers"]
        if self.layers_filter == "all":
            return all_layers
        requested = [lyr.strip() for lyr in self.layers_filter.split(",")]
        return [lyr for lyr in requested if lyr in all_layers]

    def _get_layer_keys_for_mode(self, model_name):
        """Return layer keys to iterate, dispatching on self.mode."""
        if self.mode == "last_layer":
            return ["last_layer"]
        return self._get_layers(model_name)

    def _load_last_layer_embeddings(self, model_name):
        """Load pre-pooled final-layer embeddings from the MLP_EMBEDDING cache.

        Reads ``embedding.npz`` files saved by ``mlp_embedding_classifier.py``
        at ``{benchmark_results_dir}/{result_dir}/doc_patients/MLP_EMBEDDING/
        pooled_embeddings/sub-*/ses-*/embedding.npz``.

        Each file contains a 1-D vector of shape ``(emb_dim,)`` — no further
        pooling is needed.

        Returns
        -------
        dict[str, np.ndarray]
            ``"SubjectID_ses-NN"`` -> 1-D embedding.
        """
        result_dir = LAST_LAYER_RESULT_DIRS.get(model_name, model_name)
        pooled_root = op.join(
            self.benchmark_results_dir,
            result_dir,
            "doc_patients",
            "MLP_EMBEDDING",
            "pooled_embeddings",
        )
        embeddings = {}

        if not op.isdir(pooled_root):
            print(
                f"   [WARN] Pooled embeddings dir not found: {pooled_root}",
                flush=True,
            )
            return embeddings

        for sub_dir in sorted(os.listdir(pooled_root)):
            sub_path = op.join(pooled_root, sub_dir)
            if not op.isdir(sub_path) or not sub_dir.startswith("sub-"):
                continue
            subject_id = sub_dir.replace("sub-", "")
            for ses_dir in sorted(os.listdir(sub_path)):
                ses_path = op.join(sub_path, ses_dir)
                if not op.isdir(ses_path) or not ses_dir.startswith("ses-"):
                    continue
                emb_path = op.join(ses_path, "embedding.npz")
                if not op.isfile(emb_path):
                    continue
                try:
                    npz = np.load(emb_path)
                    if "embedding" in npz.files:
                        data = npz["embedding"]
                    elif "embeddings" in npz.files:
                        data = npz["embeddings"]
                    else:
                        data = npz[npz.files[0]]
                    key = f"{subject_id}_{ses_dir}"
                    embeddings[key] = data.flatten()
                except Exception as e:
                    print(f"   [WARN] Failed to load {emb_path}: {e}", flush=True)

        print(
            f"   [{model_name}/last_layer] Loaded {len(embeddings)} subjects "
            f"from {pooled_root}",
            flush=True,
        )
        return embeddings

    def _get_embeddings(self, model_name, layer_key):
        """Dispatch embedding loading based on mode."""
        if self.mode == "last_layer":
            return self._load_last_layer_embeddings(model_name)
        return self._load_pooled_embeddings(model_name, layer_key)

    # ------------------------------------------------------------------
    # Pooling  (multilayer mode only — reuses the same cache as LinearProber)
    # ------------------------------------------------------------------

    def _pool_single_subject(self, npz_path, model_name):
        """Mean-pool each layer keeping the channel dimension."""
        config = MODEL_LAYER_KEYS[model_name]
        result = {}
        npz = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        try:
            for layer_key in config["layers"]:
                if layer_key not in npz.files:
                    continue
                arr = npz[layer_key]
                raw_shape = arr.shape
                # Apply disk pool for all layers, including layer_no_chan_pool ones.
                # layer_no_chan_pool only skips the *RAM* channel-mean step later.
                pooled = arr.mean(axis=config["disk_pool_axes"])
                pooled = np.asarray(pooled, dtype=np.float32)
                print(
                    f"      [POOL] {layer_key}: {raw_shape} "
                    f"--mean(axis={config['disk_pool_axes']})--> {pooled.shape}",
                    flush=True,
                )
                result[layer_key] = pooled
        finally:
            if hasattr(npz, "close"):
                npz.close()
        return result

    def pool_and_cache_all(self, model_name, subject_ids=None):
        """Pool all subjects' multi-layer embeddings and cache to disk.

        Writes to the same cache directory as LinearProber so both scripts
        share the same pooled files.
        """
        config = MODEL_LAYER_KEYS[model_name]
        emb_dir = config.get("abs_data_dir", op.join(self.data_dir, config["data_subdir"]))
        cache_root = op.join(self.output_dir, "pooled_embeddings", model_name)

        if not op.isdir(emb_dir):
            print(f"   [WARN] Embedding dir not found: {emb_dir}", flush=True)
            return

        _dim_note = (
            f"{config['emb_dim']} (default)"
            if config.get("emb_dim_override")
            else str(config["emb_dim"])
        )
        print(
            f"   [{model_name}] Pooling strategy: "
            f"disk_pool_axes={config['disk_pool_axes']} -> "
            f"saved shape per layer = ({config['n_channels']}, emb_dim={_dim_note}), "
            f"channel pooling deferred to RAM",
            flush=True,
        )

        n_cached = 0
        n_pooled = 0
        n_skipped = 0

        all_sub_dirs = sorted(os.listdir(emb_dir))
        if subject_ids is not None:
            subject_set = set(subject_ids)
            all_sub_dirs = [d for d in all_sub_dirs if d in subject_set]
            print(
                f"   [{model_name}] Filtering to {len(all_sub_dirs)} subjects",
                flush=True,
            )

        for sub_dir in all_sub_dirs:
            sub_path = op.join(emb_dir, sub_dir)
            if not op.isdir(sub_path) or not sub_dir.startswith("sub-"):
                continue
            for ses_dir in sorted(os.listdir(sub_path)):
                ses_path = op.join(sub_path, ses_dir)
                if not op.isdir(ses_path) or not ses_dir.startswith("ses-"):
                    continue

                cache_path = op.join(
                    cache_root, sub_dir, ses_dir, "pooled_chan_layers.npz"
                )
                if op.isfile(cache_path):
                    n_cached += 1
                    continue

                # If config specifies npz_suffix (e.g. "_layers.npz"), restrict to
                # that suffix to avoid accidentally loading a different file type.
                _npz_suffix = config.get("npz_suffix", ".npz")
                npz_files = [
                    f
                    for f in os.listdir(ses_path)
                    if f.endswith(_npz_suffix) and "original" not in f.lower()
                ]
                if not npz_files:
                    npz_files = [
                        f for f in os.listdir(ses_path) if f.endswith(_npz_suffix)
                    ]
                if not npz_files:
                    n_skipped += 1
                    continue

                npz_path = op.join(ses_path, npz_files[0])
                print(
                    f"   [{model_name}] Pooling {sub_dir}/{ses_dir} ...",
                    flush=True,
                )
                try:
                    pooled = self._pool_single_subject(npz_path, model_name)
                except Exception as e:
                    print(
                        f"   [WARN] Failed to pool {npz_path}: {e}",
                        flush=True,
                    )
                    n_skipped += 1
                    continue

                if not pooled:
                    n_skipped += 1
                    continue

                os.makedirs(op.dirname(cache_path), exist_ok=True)
                np.savez_compressed(cache_path, **pooled)
                n_pooled += 1

                if (n_pooled + n_cached) % 20 == 0:
                    print(
                        f"   [{model_name}] Pooled {n_pooled}, "
                        f"cached {n_cached}, skipped {n_skipped}",
                        flush=True,
                    )

        print(
            f"   [{model_name}] Pooling done: "
            f"{n_pooled} new, {n_cached} from cache, {n_skipped} skipped",
            flush=True,
        )

    def _load_pooled_embeddings(self, model_name, layer_key):
        """Load channel-level embeddings and mean-pool channels in RAM."""
        config = MODEL_LAYER_KEYS[model_name]
        cache_root = op.join(self.output_dir, "pooled_embeddings", model_name)
        embeddings = {}

        if not op.isdir(cache_root):
            print(
                f"   [WARN] No cached embeddings at {cache_root}. "
                f"Run pool_and_cache_all first.",
                flush=True,
            )
            return embeddings

        _no_chan_pool_layers = config.get("layer_no_chan_pool", [])
        no_pooling = layer_key in _no_chan_pool_layers
        layer_emb_dim = config.get("emb_dim_override", {}).get(layer_key, config["emb_dim"])
        if no_pooling:
            expected_disk_shape = (layer_emb_dim,)
        else:
            expected_disk_shape = (config["n_channels"], layer_emb_dim)
        n_loaded = 0

        for sub_dir in sorted(os.listdir(cache_root)):
            sub_path = op.join(cache_root, sub_dir)
            if not op.isdir(sub_path) or not sub_dir.startswith("sub-"):
                continue
            subject_id = sub_dir.replace("sub-", "")
            for ses_dir in sorted(os.listdir(sub_path)):
                ses_path = op.join(sub_path, ses_dir)
                if not op.isdir(ses_path) or not ses_dir.startswith("ses-"):
                    continue
                cache_file = op.join(ses_path, "pooled_chan_layers.npz")
                if not op.isfile(cache_file):
                    continue
                try:
                    npz = np.load(cache_file)
                    if layer_key not in npz.files:
                        continue
                    arr = npz[layer_key]
                    if arr.shape != expected_disk_shape:
                        print(
                            f"   [WARN] Shape mismatch for {sub_dir}/{ses_dir} "
                            f"layer {layer_key}: {arr.shape} "
                            f"vs {expected_disk_shape}",
                            flush=True,
                        )
                        continue
                    vec = arr if no_pooling else arr.mean(axis=0)
                    key = f"{subject_id}_{ses_dir}"
                    embeddings[key] = vec
                    n_loaded += 1
                except Exception as e:
                    print(
                        f"   [WARN] Failed to load {cache_file}: {e}",
                        flush=True,
                    )

        if no_pooling:
            print(
                f"   [{model_name}/{layer_key}] Loaded {n_loaded} subjects: "
                f"shape {expected_disk_shape} (no channel pooling)",
                flush=True,
            )
        else:
            print(
                f"   [{model_name}/{layer_key}] Loaded {n_loaded} subjects: "
                f"disk {expected_disk_shape} --mean(axis=0)--> ({layer_emb_dim},)",
                flush=True,
            )
        return embeddings

    # ------------------------------------------------------------------
    # Data loading  (identical to LinearProber)
    # ------------------------------------------------------------------

    def load_markers_from_csv(self):
        """Load baseline scalar markers from CSV for the configured reduction."""
        if self._markers_cache is not None:
            return self._markers_cache, self._marker_names_cache

        if self.reduction not in REDUCTION_MAP:
            raise ValueError(
                f"Unknown reduction {self.reduction!r}. "
                f"Choose from {list(REDUCTION_MAP)}."
            )
        reduction_str = REDUCTION_MAP[self.reduction]

        with open(self.marker_csv) as f:
            header_line = f.readline()
        if "Reduction" in header_line.split(","):
            sep = ","
        else:
            sep = ";"

        df = pd.read_csv(self.marker_csv, sep=sep)
        df_filtered = df[df["Reduction"] == reduction_str].copy()
        if df_filtered.empty:
            raise ValueError(
                f"No rows for Reduction={reduction_str!r} in {self.marker_csv}"
            )

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
            f"   Loaded markers for {len(markers_dict)} subject/sessions "
            f"({len(marker_names)} markers, reduction={reduction_str}, "
            f"sep={sep!r})",
            flush=True,
        )
        self._markers_cache = markers_dict
        self._marker_names_cache = marker_names
        return markers_dict, marker_names

    def load_patient_labels(self):
        """Load CRS labels (VS vs MCS) from patient_labels CSV."""
        if self._labels_cache is not None:
            return self._labels_cache

        df = pd.read_csv(self.patient_labels_file)
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

        print(
            f"   Loaded CRS labels for {len(labels_dict)} subject/sessions",
            flush=True,
        )
        self._labels_cache = labels_dict
        return labels_dict

    def collect_regression_data(self, model_name, layer_key):
        """Build aligned (X, Y, subjects, marker_names) for marker regression."""
        embeddings = self._get_embeddings(model_name, layer_key)
        markers_dict, marker_names = self.load_markers_from_csv()

        common = sorted(set(embeddings) & set(markers_dict))
        if not common:
            raise ValueError(
                f"No subjects in common between {model_name}/{layer_key} "
                f"embeddings and markers."
            )

        X_list, Y_list, subjects = [], [], []
        for key in common:
            X_list.append(embeddings[key])
            Y_list.append(markers_dict[key])
            subjects.append(key)

        X = np.stack(X_list)
        Y = np.stack(Y_list)
        print(
            f"   [{model_name}/{layer_key}] Regression data: "
            f"X={X.shape}, Y={Y.shape}, {len(subjects)} subjects",
            flush=True,
        )
        return X, Y, subjects, marker_names

    def collect_classification_data(self, model_name, layer_key):
        """Build aligned (X, y_encoded, subjects, groups) for classification."""
        embeddings = self._get_embeddings(model_name, layer_key)
        labels_dict = self.load_patient_labels()

        common = sorted(set(embeddings) & set(labels_dict))
        if not common:
            raise ValueError(
                f"No subjects in common between {model_name}/{layer_key} "
                f"embeddings and CRS labels."
            )

        X_list, y_list, subjects = [], [], []
        for key in common:
            X_list.append(embeddings[key])
            y_list.append(labels_dict[key])
            subjects.append(key)

        X = np.stack(X_list)
        le = LabelEncoder()
        y = le.fit_transform(y_list)
        self._class_names = list(le.classes_)
        groups = np.array([s.split("_ses-")[0] for s in subjects])

        print(
            f"   [{model_name}/{layer_key}] Classification data: "
            f"X={X.shape}, classes={dict(zip(le.classes_, np.bincount(y)))}, "
            f"{len(np.unique(groups))} unique subjects",
            flush=True,
        )
        return X, y, subjects, groups

    # ------------------------------------------------------------------
    # RandomForest Regression  (non-linear probe)
    # ------------------------------------------------------------------

    def _train_single_regressor(self, X_train, y_train, groups_train):
        """Fit a fixed-hyperparameter RandomForest regressor.

        No inner CV grid search — fixed at n_estimators=200, max_depth=None.
        For a probing study the exact RF tuning matters little; avoiding the
        inner grid search reduces compute from O(markers × folds × grid × inner)
        to O(markers × folds) and is fast enough to run in minutes.

        Returns
        -------
        model : RandomForestRegressor
        best_val_r2 : float  (nan — not computed without inner CV)
        best_params : dict
        """
        params = {"n_estimators": 200, "max_depth": None}
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, float("nan"), params

    def run_regression_for_layer(
        self, model_name, layer_key, X, Y, subjects, marker_names
    ):
        """Run RandomForest regression for all markers on a single layer.

        Returns
        -------
        metrics_per_marker : dict
            ``{marker_name: {r2, pearson_r, pearson_p, mse, mae, ...}}``
        """
        groups = np.array([s.split("_ses-")[0] for s in subjects])
        metrics_per_marker = {}

        if self.full_cv:
            if self.precomputed_splits is not None and self.common_sessions is not None:
                available_set = set(subjects)
                missing = [s for s in self.common_sessions if s not in available_set]
                if not missing:
                    subj_to_row = {s: i for i, s in enumerate(subjects)}
                    folds_iter = [
                        {
                            "train_idx": np.array([
                                subj_to_row[self.common_sessions[i]]
                                for i in fold["train_idx"]
                                if self.common_sessions[i] in subj_to_row
                            ]),
                            "test_idx": np.array([
                                subj_to_row[self.common_sessions[i]]
                                for i in fold["test_idx"]
                                if self.common_sessions[i] in subj_to_row
                            ]),
                        }
                        for fold in self.precomputed_splits
                    ]
                else:
                    folds_iter = None
            else:
                folds_iter = None

            if folds_iter is None:
                n_unique = len(np.unique(groups))
                effective_folds = min(self.n_cv_folds, n_unique)
                gkf = GroupKFold(n_splits=effective_folds)
                folds_iter = [
                    {"train_idx": tr, "test_idx": te}
                    for tr, te in gkf.split(X, groups=groups)
                ]

            effective_folds = len(folds_iter)

            for fold_idx, fold in enumerate(folds_iter):
                check_no_subject_leakage(
                    groups, fold["train_idx"], fold["test_idx"],
                    label=f"regression fold {fold_idx}",
                )

            accum = {
                i: {"true": [], "pred": [], "subjects": []} for i in range(Y.shape[1])
            }

            for fold_idx, fold in enumerate(folds_iter):
                train_idx = fold["train_idx"]
                test_idx = fold["test_idx"]
                print(
                    f"      Fold {fold_idx + 1}/{effective_folds}",
                    flush=True,
                )
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
                    accum[i]["subjects"].extend([subjects[j] for j in valid_test])

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
                    pr, pp = pearsonr(y_true, y_pred)
                except Exception:
                    pr, pp = float("nan"), float("nan")

                metrics_per_marker[marker] = {
                    "r2": float(r2),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "mse": float(mse),
                    "mae": float(mae),
                    "n_samples": len(y_true),
                    "skipped": False,
                }
        else:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=self.random_state
            )
            train_idx, test_idx = next(gss.split(X, groups=groups))

            for i, marker in enumerate(marker_names):
                y_col = Y[:, i]
                valid_train = train_idx[~np.isnan(y_col[train_idx])]
                valid_test = test_idx[~np.isnan(y_col[test_idx])]
                if len(valid_train) < 10:
                    metrics_per_marker[marker] = {"skipped": True}
                    continue

                model, _, best_info = self._train_single_regressor(
                    X[valid_train], y_col[valid_train], groups[valid_train]
                )
                y_pred = model.predict(X[valid_test])
                y_true = y_col[valid_test]

                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                try:
                    pr, pp = pearsonr(y_true, y_pred)
                except Exception:
                    pr, pp = float("nan"), float("nan")

                metrics_per_marker[marker] = {
                    "r2": float(r2),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "mse": float(mse),
                    "mae": float(mae),
                    "best_params": best_info,
                    "n_train": len(valid_train),
                    "n_test": len(valid_test),
                    "skipped": False,
                }

        return metrics_per_marker

    def run_regression(self):
        """Run RandomForest regression for all models and layers."""
        print("=" * 70, flush=True)
        print("RANDOMFOREST REGRESSION (embedding -> marker)", flush=True)
        print("=" * 70, flush=True)

        layer_r2_across_models = {}

        for model_name in self.models:
            print(f"\n-- {model_name} --", flush=True)
            if self.mode == "multilayer":
                self.pool_and_cache_all(model_name)

            layers = self._get_layer_keys_for_mode(model_name)
            for layer_key in layers:
                print(f"\n   Layer: {layer_key}", flush=True)
                try:
                    X, Y, subjects, marker_names = self.collect_regression_data(
                        model_name, layer_key
                    )
                except ValueError as e:
                    print(f"   [SKIP] {e}", flush=True)
                    continue

                metrics = self.run_regression_for_layer(
                    model_name, layer_key, X, Y, subjects, marker_names
                )

                layer_dir = op.join(
                    self.output_dir, "regression", layer_key, model_name
                )
                os.makedirs(layer_dir, exist_ok=True)
                summary_path = op.join(layer_dir, "summary.json")
                with open(summary_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"   Saved: {summary_path}", flush=True)

                r2_dict = {
                    m: v["r2"]
                    for m, v in metrics.items()
                    if not v.get("skipped", False)
                }
                if layer_key not in layer_r2_across_models:
                    layer_r2_across_models[layer_key] = {}
                layer_r2_across_models[layer_key][model_name] = r2_dict

        for layer_key, r2_per_model in layer_r2_across_models.items():
            if len(r2_per_model) < 1:
                continue
            self._plot_feature_importance_raw(layer_key, r2_per_model)

        agg_summary = {}
        for layer_key, r2_per_model in layer_r2_across_models.items():
            agg_summary[layer_key] = {}
            for model_name, r2_dict in r2_per_model.items():
                vals = list(r2_dict.values())
                agg_summary[layer_key][model_name] = {
                    "mean_r2": float(np.nanmean(vals)) if vals else None,
                    "n_markers": len(r2_dict),
                }
        agg_path = op.join(self.output_dir, "regression", "regression_summary.json")
        os.makedirs(op.dirname(agg_path), exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(agg_summary, f, indent=2)
        print(f"\n   Aggregated summary: {agg_path}", flush=True)

    # ------------------------------------------------------------------
    # Classification  (identical to LinearProber)
    # ------------------------------------------------------------------

    def _get_inner_cv_iter(self, X_train, y_train, groups_train, inner_splits):
        if inner_splits is not None:
            return list(inner_splits)
        n_unique = len(np.unique(groups_train))
        n_inner = min(3, n_unique)
        inner_cv = GroupKFold(n_splits=n_inner)
        return list(inner_cv.split(X_train, y_train, groups=groups_train))

    def _grid_search_svm(self, X_train, y_train, groups_train, inner_splits=None):
        """Inner CV grid search for SVM."""
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        }
        gamma_options = {
            "rbf": ["scale", 0.001, 0.01, 0.1],
            "linear": [None],
        }

        cv_iter = self._get_inner_cv_iter(X_train, y_train, groups_train, inner_splits)
        best_score = -1.0
        best_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}

        for C, kernel in product(param_grid["C"], param_grid["kernel"]):
            for gamma in gamma_options[kernel]:
                scores = []
                for tr_idx, va_idx in cv_iter:
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
            [("scaler", StandardScaler()), ("svc", SVC(**svc_kw))]
        )
        best_model.fit(X_train, y_train)
        return best_model, best_params

    def _grid_search_rf(self, X_train, y_train, groups_train, inner_splits=None):
        """Inner CV grid search for Random Forest."""
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
        }
        cv_iter = self._get_inner_cv_iter(X_train, y_train, groups_train, inner_splits)

        best_score = -1.0
        best_params = {"n_estimators": 100, "max_depth": None}

        for n_est, max_d in product(
            param_grid["n_estimators"], param_grid["max_depth"]
        ):
            scores = []
            for tr_idx, va_idx in cv_iter:
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

        best_model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)
        return best_model, best_params

    def _grid_search_kr(self, X_train, y_train, groups_train, inner_splits=None):
        """Inner CV grid search for Kernel Ridge (binary)."""
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "gamma": [0.001, 0.01, 0.1, 1.0],
        }
        cv_iter = self._get_inner_cv_iter(X_train, y_train, groups_train, inner_splits)

        best_score = -1.0
        best_params = {"alpha": 1.0, "gamma": 0.01}

        for alpha, gamma in product(param_grid["alpha"], param_grid["gamma"]):
            scores = []
            for tr_idx, va_idx in cv_iter:
                kr = KernelRidgeClassifier(kernel="rbf", alpha=alpha, gamma=gamma)
                kr.fit(X_train[tr_idx], y_train[tr_idx])
                raw = kr.decision_function(X_train[va_idx])
                pseudo_proba = np.clip(raw, 0, 1)
                preds = (pseudo_proba >= 0.5).astype(int)
                scores.append(_safe_auc_score(y_train[va_idx], pseudo_proba, preds))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"alpha": alpha, "gamma": gamma}

        best_model = KernelRidgeClassifier(
            kernel="rbf",
            alpha=best_params["alpha"],
            gamma=best_params["gamma"],
        )
        best_model.fit(X_train, y_train)
        return best_model, best_params

    def _predict_model(self, model, X):
        if isinstance(model, Pipeline):
            probs = model.decision_function(X)
            preds = model.predict(X)
        elif isinstance(model, RandomForestClassifier):
            proba = model.predict_proba(X)
            probs = proba[:, 1]
            preds = (probs >= 0.5).astype(int)
        elif isinstance(model, KernelRidgeClassifier):
            raw = model.decision_function(X)
            probs = np.clip(raw, 0, 1)
            preds = (probs >= 0.5).astype(int)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        return probs, preds

    def run_classification_for_layer(
        self, model_name, layer_key, X, y, subjects, groups
    ):
        """Nested CV classification for a single model/layer."""
        classifiers = {
            "svm": self._grid_search_svm,
            "kernel_ridge": self._grid_search_kr,
            "random_forest": self._grid_search_rf,
        }

        if self.precomputed_splits is not None and self.common_sessions is not None:
            available_set = set(subjects)
            missing = [s for s in self.common_sessions if s not in available_set]
            if missing:
                print(
                    f"   [WARN] {len(missing)} sessions from splits not in layer "
                    f"{layer_key} — generating fresh folds.",
                    flush=True,
                )
                folds_iter = None
            else:
                subj_to_row = {s: i for i, s in enumerate(subjects)}
                folds_iter = []
                for fold in self.precomputed_splits:
                    tr = np.array([subj_to_row[self.common_sessions[i]]
                                   for i in fold["train_idx"]
                                   if self.common_sessions[i] in subj_to_row])
                    te = np.array([subj_to_row[self.common_sessions[i]]
                                   for i in fold["test_idx"]
                                   if self.common_sessions[i] in subj_to_row])
                    folds_iter.append({
                        "train_idx": tr,
                        "test_idx": te,
                        "inner_splits": fold.get("inner_splits"),
                    })
        else:
            folds_iter = None

        if folds_iter is None:
            n_unique = len(np.unique(groups))
            effective_folds = min(self.n_cv_folds, n_unique)
            if effective_folds < 2:
                print(
                    f"   [SKIP] Only {n_unique} unique subjects, need >= 2",
                    flush=True,
                )
                return {}
            sgkf = StratifiedGroupKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            folds_iter = [
                {"train_idx": tr, "test_idx": te, "inner_splits": None}
                for tr, te in sgkf.split(X, y, groups=groups)
            ]

        effective_folds = len(folds_iter)

        for fold_idx, fold in enumerate(folds_iter):
            check_no_subject_leakage(
                groups, fold["train_idx"], fold["test_idx"],
                label=f"layer {layer_key} fold {fold_idx}",
            )

        accum = {
            name: {"y_true": [], "y_pred": [], "y_proba": []} for name in classifiers
        }

        for fold_idx, fold in enumerate(folds_iter):
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]
            inner_splits = fold.get("inner_splits")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            print(
                f"      Fold {fold_idx + 1}/{effective_folds} "
                f"(train={len(train_idx)}, test={len(test_idx)})",
                flush=True,
            )

            for clf_name, grid_fn in classifiers.items():
                try:
                    model, _ = grid_fn(X_train, y_train, groups_train,
                                       inner_splits=inner_splits)
                    probs, preds = self._predict_model(model, X_test)
                    accum[clf_name]["y_true"].append(y_test)
                    accum[clf_name]["y_pred"].append(preds)
                    accum[clf_name]["y_proba"].append(probs)
                except Exception as e:
                    print(
                        f"      [WARN] {clf_name} fold {fold_idx}: {e}",
                        flush=True,
                    )

        results = {}
        for clf_name in classifiers:
            if not accum[clf_name]["y_true"]:
                continue
            n_folds = len(accum[clf_name]["y_true"])
            auc_scores = []
            bal_acc_scores = []

            for i in range(n_folds):
                yt = accum[clf_name]["y_true"][i]
                yp = accum[clf_name]["y_pred"][i]
                ypr = accum[clf_name]["y_proba"][i]

                bal_acc_scores.append(balanced_accuracy_score(yt, yp))

                n_classes = len(np.unique(yt))
                if n_classes == 2:
                    try:
                        auc_scores.append(roc_auc_score(yt, ypr))
                    except Exception:
                        pass
                else:
                    auc_scores.append(balanced_accuracy_score(yt, yp))

            results[clf_name] = {
                "auc_mean": (float(np.mean(auc_scores)) if auc_scores else None),
                "auc_std": (float(np.std(auc_scores)) if auc_scores else None),
                "balanced_accuracy_mean": float(np.mean(bal_acc_scores)),
                "balanced_accuracy_std": float(np.std(bal_acc_scores)),
                "n_folds": n_folds,
            }

        return results

    def _maybe_generate_and_save_splits(
        self, subjects, y, groups, save_splits_to=None
    ):
        labels_for_splits = {s: int(y[i]) for i, s in enumerate(subjects)}
        self.common_sessions = list(subjects)
        self.precomputed_splits = generate_nested_cv_folds(
            common_sessions=self.common_sessions,
            labels=labels_for_splits,
            n_outer=self.n_cv_folds,
            random_state=self.random_state,
        )
        print(
            f"   Generated {len(self.precomputed_splits)} outer folds "
            f"(StratifiedGroupKFold, seed={self.random_state})",
            flush=True,
        )
        save_path = save_splits_to
        if not save_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            splits_dir = op.join(self.output_dir, "cv_splits")
            save_path = op.join(splits_dir, f"non_linear_probing_crs_{ts}.json")
        save_cv_splits(
            self.precomputed_splits, self.common_sessions, labels_for_splits, save_path
        )
        print(f"   CV splits saved to {save_path}", flush=True)

    def run_classification(self, save_splits_to=None):
        """Run nested CV classification for all models and layers."""
        print("\n" + "=" * 70, flush=True)
        print("NESTED CV CLASSIFICATION (embedding -> CRS)", flush=True)
        print("=" * 70, flush=True)

        all_results = {}

        for model_name in self.models:
            print(f"\n-- {model_name} --", flush=True)
            if self.mode == "multilayer":
                self.pool_and_cache_all(model_name)
            all_results[model_name] = {}

            layers = self._get_layer_keys_for_mode(model_name)
            for layer_key in layers:
                print(f"\n   Layer: {layer_key}", flush=True)
                try:
                    X, y, subjects, groups = self.collect_classification_data(
                        model_name, layer_key
                    )
                except ValueError as e:
                    print(f"   [SKIP] {e}", flush=True)
                    continue

                if self.full_cv and self.precomputed_splits is None:
                    self._maybe_generate_and_save_splits(
                        subjects, y, groups, save_splits_to=save_splits_to
                    )

                results = self.run_classification_for_layer(
                    model_name, layer_key, X, y, subjects, groups
                )
                all_results[model_name][layer_key] = results

                layer_dir = op.join(self.output_dir, "classification", layer_key)
                os.makedirs(layer_dir, exist_ok=True)
                for clf_name, clf_results in results.items():
                    clf_path = op.join(
                        layer_dir,
                        f"{clf_name}_{model_name}_results.json",
                    )
                    with open(clf_path, "w") as f:
                        json.dump(clf_results, f, indent=2)

                parts = []
                for c, r in results.items():
                    auc = r.get("auc_mean")
                    if auc is not None:
                        parts.append(f"{c}: AUC={auc:.3f}")
                    else:
                        parts.append(f"{c}: AUC=N/A")
                print(f"   Results: {', '.join(parts)}", flush=True)

        summary_path = op.join(
            self.output_dir, "classification", "classification_summary.json"
        )
        os.makedirs(op.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n   Classification summary: {summary_path}", flush=True)

        self._plot_classification_summary(all_results)

    # ------------------------------------------------------------------
    # Plotting  (identical to LinearProber)
    # ------------------------------------------------------------------

    def _plot_feature_importance_raw(self, layer_key, r2_per_model):
        """Plot raw R2 comparison across FM models for a single layer."""
        marker_order = _ordered_markers_by_family_and_mean(r2_per_model)
        short_names = [_short_name(m) for m in marker_order]
        x = np.arange(len(marker_order))

        fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 7))

        for mi, (em, r2_dict) in enumerate(r2_per_model.items()):
            y_raw = [r2_dict.get(m, np.nan) for m in marker_order]
            color = _MODEL_COLORS.get(em, f"C{mi}")
            ax.plot(
                x,
                y_raw,
                label=em,
                color=color,
                marker="o",
                linestyle="-",
                linewidth=1.6,
                markersize=5,
                alpha=0.9,
            )

        ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xlabel("Markers", fontsize=17)
        ax.set_ylabel("R\u00b2", fontsize=17)
        ax.set_title(
            f"Raw R\u00b2 (RandomForest regression: embedding \u2192 marker)"
            f" \u2014 {layer_key}",
            fontsize=18,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=12)
        _color_xticks_by_family(ax, marker_order)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(loc="upper left", fontsize=12)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out_dir = op.join(self.output_dir, "regression", layer_key)
        os.makedirs(out_dir, exist_ok=True)
        out_path = op.join(out_dir, "feature_importance_raw.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved: {out_path}", flush=True)

    def _plot_classification_summary(self, all_results):
        """Plot AUC per layer per classifier per model as grouped bars."""
        if not all_results:
            return

        clf_names = ["svm", "kernel_ridge", "random_forest"]
        clf_labels = {
            "svm": "SVM",
            "kernel_ridge": "KernelRidge",
            "random_forest": "RandomForest",
        }
        clf_colors = {
            "svm": "#1f77b4",
            "kernel_ridge": "#ff7f0e",
            "random_forest": "#2ca02c",
        }

        for model_name, layer_results in all_results.items():
            if not layer_results:
                continue

            layers = list(layer_results.keys())
            n_layers = len(layers)
            n_clf = len(clf_names)
            bar_width = 0.8 / n_clf
            x = np.arange(n_layers)

            fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.5), 6))

            for ci, clf_name in enumerate(clf_names):
                means = []
                stds = []
                for layer_key in layers:
                    r = layer_results[layer_key].get(clf_name, {})
                    means.append(r.get("auc_mean", 0) or 0)
                    stds.append(r.get("auc_std", 0) or 0)

                offset = (ci - n_clf / 2 + 0.5) * bar_width
                ax.bar(
                    x + offset,
                    means,
                    bar_width,
                    yerr=stds,
                    label=clf_labels.get(clf_name, clf_name),
                    color=clf_colors.get(clf_name, f"C{ci}"),
                    alpha=0.85,
                    capsize=3,
                )

            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("AUC", fontsize=12)
            ax.set_title(
                f"{model_name} \u2014 Classification AUC per Layer",
                fontsize=14,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(layers, rotation=30, ha="right")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.25, axis="y")
            ax.set_ylim(0, 1.05)

            plt.tight_layout()
            out_path = op.join(
                self.output_dir,
                "classification",
                f"classification_summary_{model_name}.png",
            )
            os.makedirs(op.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   Classification plot saved: {out_path}", flush=True)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self):
        """Re-read per-layer JSON results and produce summaries + plots."""
        print("=" * 70, flush=True)
        print("AGGREGATE MODE — reading per-layer results", flush=True)
        print("=" * 70, flush=True)

        reg_root = op.join(self.output_dir, "regression")
        layer_r2_across_models = {}

        if op.isdir(reg_root):
            for model_name in self.models:
                layers = self._get_layer_keys_for_mode(model_name)
                for layer_key in layers:
                    summary_path = op.join(
                        reg_root, layer_key, model_name, "summary.json"
                    )
                    if not op.isfile(summary_path):
                        continue
                    with open(summary_path) as f:
                        metrics = json.load(f)
                    r2_dict = {
                        m: v["r2"]
                        for m, v in metrics.items()
                        if not v.get("skipped", False)
                    }
                    if layer_key not in layer_r2_across_models:
                        layer_r2_across_models[layer_key] = {}
                    layer_r2_across_models[layer_key][model_name] = r2_dict
                    print(
                        f"   [regression] {model_name}/{layer_key}: "
                        f"{len(r2_dict)} markers",
                        flush=True,
                    )

            for layer_key, r2_per_model in layer_r2_across_models.items():
                if r2_per_model:
                    self._plot_feature_importance_raw(layer_key, r2_per_model)

            agg_summary = {}
            for layer_key, r2_per_model in layer_r2_across_models.items():
                agg_summary[layer_key] = {}
                for model_name, r2_dict in r2_per_model.items():
                    vals = list(r2_dict.values())
                    agg_summary[layer_key][model_name] = {
                        "mean_r2": float(np.nanmean(vals)) if vals else None,
                        "n_markers": len(r2_dict),
                    }
            agg_path = op.join(reg_root, "regression_summary.json")
            os.makedirs(op.dirname(agg_path), exist_ok=True)
            with open(agg_path, "w") as f:
                json.dump(agg_summary, f, indent=2)
            print(f"   Regression summary: {agg_path}", flush=True)

        clf_root = op.join(self.output_dir, "classification")
        all_results = {}

        if op.isdir(clf_root):
            clf_names = ["svm", "kernel_ridge", "random_forest"]
            for model_name in self.models:
                all_results[model_name] = {}
                layers = self._get_layer_keys_for_mode(model_name)
                for layer_key in layers:
                    layer_dir = op.join(clf_root, layer_key)
                    if not op.isdir(layer_dir):
                        continue
                    layer_results = {}
                    for clf_name in clf_names:
                        clf_path = op.join(
                            layer_dir,
                            f"{clf_name}_{model_name}_results.json",
                        )
                        if not op.isfile(clf_path):
                            continue
                        with open(clf_path) as f:
                            layer_results[clf_name] = json.load(f)
                    if layer_results:
                        all_results[model_name][layer_key] = layer_results
                        parts = []
                        for c, r in layer_results.items():
                            auc = r.get("auc_mean")
                            if auc is not None:
                                parts.append(f"{c}: AUC={auc:.3f}")
                            else:
                                parts.append(f"{c}: AUC=N/A")
                        print(
                            f"   [classification] {model_name}/{layer_key}: "
                            f"{', '.join(parts)}",
                            flush=True,
                        )

            summary_path = op.join(clf_root, "classification_summary.json")
            os.makedirs(op.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"   Classification summary: {summary_path}", flush=True)
            self._plot_classification_summary(all_results)

        print("Aggregation complete.", flush=True)

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------

    def run(self, skip_regression=False, skip_classification=False, save_splits_to=None):
        """Run the full non-linear probing pipeline."""
        print("=" * 70, flush=True)
        print("NON-LINEAR PROBING -- Embedding Analysis", flush=True)
        print(f"  mode          : {self.mode}", flush=True)
        print(f"  data_dir      : {self.data_dir}", flush=True)
        print(f"  output_dir    : {self.output_dir}", flush=True)
        print(f"  models        : {self.models}", flush=True)
        if self.mode == "multilayer":
            print(f"  layers        : {self.layers_filter}", flush=True)
        print(f"  n_cv_folds    : {self.n_cv_folds}", flush=True)
        print(f"  full_cv       : {self.full_cv}", flush=True)
        if self.precomputed_splits is not None:
            print(
                f"  splits        : {len(self.precomputed_splits)} pre-computed folds",
                flush=True,
            )
        print("=" * 70, flush=True)

        if not skip_regression:
            if self.marker_csv is None:
                print(
                    "[WARN] No --marker-csv provided; skipping regression.",
                    flush=True,
                )
            else:
                self.run_regression()

        if not skip_classification:
            self.run_classification(save_splits_to=save_splits_to)

        summary = {
            "models": self.models,
            "layers_filter": self.layers_filter,
            "n_cv_folds": self.n_cv_folds,
            "reduction": self.reduction,
            "full_cv": self.full_cv,
            "probe_type": "non_linear",
        }
        summary_path = op.join(self.output_dir, "non_linear_probing_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70, flush=True)
        print("NON-LINEAR PROBING COMPLETE", flush=True)
        print(f"  Results at: {self.output_dir}", flush=True)
        print("=" * 70, flush=True)


# -- CLI --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Non-linear probing on multi-layer foundation model embeddings. "
            "RandomForest regression for marker prediction and nested CV "
            "classification for CRS (VS vs MCS)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
# Full run:
python non_linear_probing.py \\
    --data-dir /data/project/eeg_foundation/data \\
    --output-dir /data/project/eeg_foundation/data/benchmark_results/new_results/NON_LINEAR_PROBING \\
    --patient-labels /data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv \\
    --marker-csv /data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv

# Regression only, single model:
python non_linear_probing.py \\
    --data-dir /data/project/eeg_foundation/data \\
    --output-dir /tmp/nlp_test \\
    --patient-labels /data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv \\
    --marker-csv /data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv \\
    --models CbraMod --layers layer_0 --skip-classification
        """,
    )
    parser.add_argument(
        "--data-dir",
        default="/data/project/eeg_foundation/data",
        help="Root data directory containing model embedding subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/project/eeg_foundation/data/benchmark_results"
            "/new_results/NON_LINEAR_PROBING"
        ),
        help="Directory where all results are saved.",
    )
    parser.add_argument(
        "--patient-labels",
        default=(
            "/data/project/eeg_foundation/data/metadata/metadata_patient_labels.csv"
        ),
        help="Path to patient_labels_with_controls.csv.",
    )
    parser.add_argument(
        "--marker-csv",
        default=(
            "/data/project/eeg_foundation/data/original_DoC/nice_scalars_all.csv"
        ),
        help="Path to baseline scalar markers CSV.",
    )
    parser.add_argument(
        "--reduction",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Marker aggregation variant (default: A = trim_mean80).",
    )
    parser.add_argument(
        "--n-cv-folds",
        type=int,
        default=5,
        help="Number of outer CV folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--mode",
        choices=["multilayer", "last_layer"],
        default="multilayer",
        help=(
            "Probing mode. 'multilayer': probe each intermediate layer from "
            "multi_layer_embeddings (CbraMod/NeuroLM only). 'last_layer': use "
            "the pre-pooled final-layer embeddings from the MLP_EMBEDDING "
            "pipeline (all models: CbraMod, NeuroLM, TOTEM, LaBram)."
        ),
    )
    parser.add_argument(
        "--benchmark-results-dir",
        default=_BENCHMARK_RESULTS_DEFAULT,
        help=(
            "Root of benchmark results (used in last_layer mode to locate "
            "pooled_embeddings). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Foundation models to probe. Defaults: multilayer mode → CbraMod "
            "NeuroLM; last_layer mode → CbraMod NeuroLM TOTEM LaBram."
        ),
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer keys or 'all'. Only used in multilayer mode.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip RandomForest regression (marker prediction).",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip nested CV classification.",
    )
    parser.add_argument(
        "--single-split",
        action="store_true",
        help="Use single train/test split instead of full CV for regression.",
    )
    parser.add_argument(
        "--pool-only",
        action="store_true",
        help=(
            "Only run the pooling stage (save channel-level embeddings to "
            "disk) then exit.  Useful for parallelising pooling across "
            "HTCondor jobs before running regression/classification."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Only aggregate per-layer JSON results into summaries and "
            "plots.  Run this after all per-layer analysis jobs finish."
        ),
    )
    parser.add_argument(
        "--subjects",
        help=(
            "Comma-separated list of sub-* directory names to process "
            "(e.g. 'sub-001,sub-002').  If omitted, all subjects are "
            "processed."
        ),
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Pre-computed CV splits JSON (from generate_nested_cv_folds). "
            "When provided, all layers use identical outer/inner fold assignments."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help=(
            "Path to save generated CV splits JSON for reuse. "
            "If omitted and no --splits-file given, splits are auto-saved to "
            "{output_dir}/cv_splits/non_linear_probing_crs_{timestamp}.json."
        ),
    )

    args = parser.parse_args()

    # Validate models against the chosen mode
    valid_models = LAST_LAYER_MODELS if args.mode == "last_layer" else MULTILAYER_MODELS
    requested_models = args.models or valid_models
    for m in requested_models:
        if m not in valid_models:
            parser.error(
                f"Model {m!r} not valid for --mode {args.mode}. "
                f"Valid: {valid_models}"
            )

    subject_ids = None
    if args.subjects:
        subject_ids = [s.strip() for s in args.subjects.split(",")]

    precomputed_splits = None
    common_sessions = None

    if args.splits_file and op.isfile(args.splits_file):
        precomputed_splits, common_sessions, labels_ref = load_cv_splits(args.splits_file)
        print(
            f"Loaded {len(precomputed_splits)} pre-computed outer folds "
            f"({len(common_sessions)} sessions) from {args.splits_file}",
            flush=True,
        )

    prober = NonLinearProber(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patient_labels_file=args.patient_labels,
        marker_csv=args.marker_csv,
        reduction=args.reduction,
        models=requested_models,
        layers=args.layers,
        n_cv_folds=args.n_cv_folds,
        random_state=args.random_state,
        full_cv=not args.single_split,
        precomputed_splits=precomputed_splits,
        common_sessions=common_sessions,
        mode=args.mode,
        benchmark_results_dir=args.benchmark_results_dir,
    )

    if args.pool_only:
        if args.mode == "last_layer":
            parser.error("--pool-only is only valid in multilayer mode.")
        print("=" * 70, flush=True)
        print("POOL-ONLY MODE — saving (n_channels, emb_dim) to disk", flush=True)
        print("=" * 70, flush=True)
        for model_name in prober.models:
            prober.pool_and_cache_all(model_name, subject_ids=subject_ids)
        print("Pooling complete.", flush=True)
    elif args.aggregate_only:
        prober.aggregate_results()
    else:
        prober.run(
            skip_regression=args.skip_regression,
            skip_classification=args.skip_classification,
            save_splits_to=args.save_splits_to,
        )


if __name__ == "__main__":
    main()
