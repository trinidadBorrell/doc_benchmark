"""Marker-guided embedding steering on FM embeddings.

==============================================================
Embedding Steering — Probe-Direction Interventions
==============================================================

For each EEG foundation model (LaBraM, CBraMod, NeuroLM, TOTEM), this script:

1. **Train linear probes** (per outer CV fold): Ridge regression predicting
   each classical EEG marker from the last-layer pooled embeddings
   (``MLP_EMBEDDING/pooled_embeddings/…/embedding.npz``).  Probes are fit
   **only on the fold's training split** to avoid data leakage.

2. **Extract probe direction**: The probe weight vector *w* defines a
   direction in embedding space along which the marker varies.  Normalize
   to a unit vector *w_hat*.

3. **Compute intervention magnitude**: For each marker, project VS and MCS
   embeddings (test split only) onto the probe direction and compute
   *delta = mean(proj_MCS) − mean(proj_VS)*.

4. **Steer embeddings**: Patch every VS embedding by shifting it along the
   probe direction by *delta*, leaving other dimensions untouched.

5. **Measure causal effect**: Feed the patched embedding to the pre-trained
   VS/MCS classifier and measure how the prediction changes.
   ``causal_effect = mean(P_patched(MCS) − P_original(MCS))``.
   Results are aggregated across outer CV folds (mean ± std).

Note
----
This method is **embedding steering** / **representation engineering**
(cf. Zou et al., 2023), not classical activation patching (Meng et al., 2022).
Activation patching operates layer-by-layer on intermediate transformer
activations; this script works on final pooled embeddings only.

Usage
-----
::

    python embedding_steering.py --model LaBram
    python embedding_steering.py --model all          # run all four models
    python embedding_steering.py --model all --compare # also produce cross-model plot
    python embedding_steering.py --model all \\
        --splits-file /path/to/dk_combined_crs_20260331_170511.json

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import os
import os.path as op
import sys
from datetime import datetime
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Import KernelRidgeClassifier with fallback for direct execution
try:
    from ..model.kernel_ridge_classifier import KernelRidgeClassifier
except ImportError:
    _model_dir = op.abspath(op.join(op.dirname(__file__), "..", "model"))
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)

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

# ── Constants ────────────────────────────────────────────────────────────

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
    "CbraMod": "#2ca02c",
    "CBraMod": "#2ca02c",
    "NeuroLM": "#1f77b4",
    "TOTEM": "#ff7f0e",
    "LaBram": "#d62728",
}

FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_COLORS = {
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",
    "wsmi": "#8b4513",
    "evoked": "#c55a11",
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _short_name(full_path: str) -> str:
    """``nice/marker/PowerSpectralDensity/theta`` -> ``PSD_theta``."""
    parts = full_path.split("/")
    family = parts[-2] if len(parts) >= 2 else parts[-1]
    # Abbreviate long family names
    abbr = {
        "PowerSpectralDensity": "PSD",
        "PowerSpectralDensitySummary": "PSDsum",
        "PermutationEntropy": "PE",
        "SymbolicMutualInformation": "wSMI",
        "KolmogorovComplexity": "KC",
        "ContingentNegativeVariation": "CNV",
        "TimeLockedTopography": "TLT",
        "TimeLockedContrast": "TLC",
        "WindowDecoding": "WD",
    }
    return f"{abbr.get(family, family)}_{parts[-1]}"


def _marker_family(marker_name: str) -> str:
    """Map marker name to family bucket."""
    if "KolmogorovComplexity" in marker_name or "PermutationEntropy" in marker_name:
        return "information_theory"
    if "PowerSpectralDensity" in marker_name:
        return "power_spectral_density"
    if "SymbolicMutualInformation" in marker_name:
        return "wsmi"
    return "evoked"


def _color_xticks_by_family(ax, marker_order: list) -> None:
    """Color x tick labels according to marker family."""
    for tick, marker in zip(ax.get_xticklabels(), marker_order):
        fam = _marker_family(marker)
        tick.set_color(FAMILY_COLORS.get(fam, "black"))


# ── Data loading ─────────────────────────────────────────────────────────


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
        f"   [{model_name}] Loaded {len(embeddings)} embeddings "
        f"(dim={next(iter(embeddings.values())).shape[0]})",
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

    with open(marker_csv) as f:
        header_line = f.readline()
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
        ``{subject_session_key: label_str}`` where label is VS or MCS.
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


def load_classifier(results_root, model_name, classifier_type="kernel_ridge"):
    """Load the first fold of the pre-trained VS/MCS classifier.

    We load fold_00 as the representative classifier.  This avoids
    information leakage from ensembling across folds.

    Returns
    -------
    clf : fitted classifier
    results : dict with classification_results.json contents
    """
    import joblib

    clf_dir = op.join(
        results_root,
        model_name,
        "doc_patients",
        "MLP_EMBEDDING",
        "crs",
        "nested_cv",
        classifier_type,
    )
    # Load the classification results to get scaler/training info
    results_path = op.join(clf_dir, "classification_results.json")
    with open(results_path) as f:
        results = json.load(f)

    # Load fold models — we use all folds and average their predictions
    fold_models = []
    for i in range(results.get("n_folds", 5)):
        model_path = op.join(clf_dir, f"fold_{i:02d}_model.joblib")
        if op.isfile(model_path):
            fold_models.append(joblib.load(model_path))

    if not fold_models:
        raise FileNotFoundError(f"No fold models found in {clf_dir}")

    print(
        f"   [{model_name}] Loaded {len(fold_models)} fold classifiers "
        f"from {classifier_type}",
        flush=True,
    )
    return fold_models, results


# ── Core causal tracing logic ────────────────────────────────────────────


def train_probes(X, Y, marker_names, groups, n_inner_splits=3, random_state=42):
    """Train one Ridge regression probe per marker.

    Uses GroupKFold cross-validation on the provided data to select the best
    alpha, then refits on the full dataset to get the probe direction.

    Parameters
    ----------
    X : np.ndarray, shape (n, emb_dim)
        Training embeddings (outer fold's train split).
    Y : np.ndarray, shape (n, n_markers)
        Marker scalars for training subjects (NaN allowed).
    marker_names : list of str
    groups : np.ndarray, shape (n,)
        Subject-level group IDs for inner GroupKFold.
    n_inner_splits : int
        Number of inner CV splits for alpha selection (default: 3).
    random_state : int
        Unused (kept for API consistency with generate_nested_cv_folds).

    Returns
    -------
    probes : dict
        ``{marker_name: {"weight": w, "weight_hat": w_hat, "r2": r2, ...}}``
        where *w_hat* is the unit-normalised probe direction.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    probes = {}

    for i, marker in enumerate(marker_names):
        y = Y[:, i]

        # Skip markers with no variance or all NaN
        valid_mask = np.isfinite(y)
        if valid_mask.sum() < 10 or np.std(y[valid_mask]) < 1e-12:
            continue

        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask]
        g_valid = groups[valid_mask]

        # Inner GroupKFold to pick best alpha (regression, so no stratification needed)
        n_splits = min(n_inner_splits, len(np.unique(g_valid)))
        if n_splits < 2:
            continue

        best_alpha, best_score = alphas[0], -np.inf
        gkf = GroupKFold(n_splits=n_splits)
        for a in alphas:
            scores = []
            for tr_idx, te_idx in gkf.split(X_valid, groups=g_valid):
                ridge = Ridge(alpha=a)
                ridge.fit(X_valid[tr_idx], y_valid[tr_idx])
                scores.append(ridge.score(X_valid[te_idx], y_valid[te_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = a

        # Refit on all training data with best alpha
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_valid, y_valid)

        w = ridge.coef_  # shape: (emb_dim,)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w_hat = w / w_norm

        # R² on training data (for probe quality reporting only)
        r2 = ridge.score(X_valid, y_valid)

        probes[marker] = {
            "weight": w,
            "weight_hat": w_hat,
            "r2": float(r2),
            "best_alpha": float(best_alpha),
            "scaler": scaler,
        }

    print(f"   Trained {len(probes)} marker probes", flush=True)
    return probes


def compute_causal_effects(
    X_test,
    test_keys,
    labels_dict,
    probes,
    marker_names,
    fold_clf,
    class_names,
):
    """Compute causal effect of each marker on VS→MCS classification (single fold).

    Parameters
    ----------
    X_test : np.ndarray, shape (n_test, emb_dim)
        Embeddings for the test split of one outer fold.
    test_keys : list of str
        Session keys corresponding to rows of X_test.
    labels_dict : dict {key: "VS" or "MCS"}
    probes : dict from ``train_probes`` (trained on the fold's train split)
    marker_names : list of str
    fold_clf : fitted classifier
        Single fold classifier (trained on the corresponding train split).
    class_names : list of str, e.g. ["MCS", "VS"]

    Returns
    -------
    results : dict
        Per-marker causal effect + per-subject details.
    """
    # Identify which class index is MCS
    mcs_idx = class_names.index("MCS") if "MCS" in class_names else 0

    # Split test set into VS and MCS
    vs_keys = [k for k in test_keys if labels_dict.get(k) == "VS"]
    mcs_keys = [k for k in test_keys if labels_dict.get(k) == "MCS"]

    if not vs_keys or not mcs_keys:
        raise ValueError(
            f"Need both VS and MCS in test split. Got {len(vs_keys)} VS, "
            f"{len(mcs_keys)} MCS."
        )

    key_to_idx = {k: i for i, k in enumerate(test_keys)}
    h_VS = X_test[[key_to_idx[k] for k in vs_keys]]   # (n_vs, emb_dim)
    h_MCS = X_test[[key_to_idx[k] for k in mcs_keys]]  # (n_mcs, emb_dim)

    print(
        f"   Test split — VS: {len(vs_keys)}, MCS: {len(mcs_keys)}",
        flush=True,
    )

    def _predict_proba_mcs(X):
        """Return P(MCS) from the fold classifier."""
        raw = fold_clf.decision_function(X)
        if mcs_idx == 0:
            return 1.0 - raw
        return raw

    results = {}
    for marker in marker_names:
        if marker not in probes:
            continue

        probe = probes[marker]
        w_hat = probe["weight_hat"]
        scaler = probe["scaler"]

        # Scale embeddings the same way the probe was trained
        h_VS_scaled = scaler.transform(h_VS)
        h_MCS_scaled = scaler.transform(h_MCS)

        # Project onto probe direction
        proj_VS = h_VS_scaled @ w_hat  # (n_vs,)
        proj_MCS = h_MCS_scaled @ w_hat  # (n_mcs,)
        delta = proj_MCS.mean() - proj_VS.mean()

        # ── VS→MCS direction: patch VS embeddings toward MCS ──────────
        h_VS_patched_scaled = h_VS_scaled + delta * w_hat[np.newaxis, :]
        h_VS_patched = scaler.inverse_transform(h_VS_patched_scaled)

        p_original_vs = _predict_proba_mcs(h_VS)
        p_patched_vs = _predict_proba_mcs(h_VS_patched)
        causal_effect_vs = float(np.mean(p_patched_vs - p_original_vs))

        # ── MCS→VS direction: patch MCS embeddings toward VS ──────────
        h_MCS_patched_scaled = h_MCS_scaled + (-delta) * w_hat[np.newaxis, :]
        h_MCS_patched = scaler.inverse_transform(h_MCS_patched_scaled)

        p_original_mcs = _predict_proba_mcs(h_MCS)
        p_patched_mcs = _predict_proba_mcs(h_MCS_patched)
        causal_effect_mcs = float(np.mean(p_patched_mcs - p_original_mcs))

        results[marker] = {
            "causal_effect": causal_effect_vs,
            "causal_effect_mcs": causal_effect_mcs,
            "delta": float(delta),
            "probe_r2": probe["r2"],
            "probe_alpha": probe["best_alpha"],
            "mean_proj_VS": float(proj_VS.mean()),
            "mean_proj_MCS": float(proj_MCS.mean()),
            "per_subject": {
                "vs_keys": vs_keys,
                "p_original": p_original_vs.tolist(),
                "p_patched": p_patched_vs.tolist(),
            },
            "per_subject_mcs": {
                "mcs_keys": mcs_keys,
                "p_original": p_original_mcs.tolist(),
                "p_patched": p_patched_mcs.tolist(),
            },
        }

    print(f"   Computed causal effects for {len(results)} markers", flush=True)
    return results


# ── Plotting ─────────────────────────────────────────────────────────────


def _plot_causal_bar(
    results, marker_names, model_name, out_dir, effect_key, title_suffix, filename
):
    """Bar plot: causal effect per marker for one model (single direction)."""
    markers_with_data = [
        m for m in marker_names if m in results and effect_key in results[m]
    ]
    if not markers_with_data:
        return

    markers_with_data.sort(
        key=lambda m: abs(results[m][effect_key]),
        reverse=True,
    )

    effects = [results[m][effect_key] for m in markers_with_data]
    short_names = [_short_name(m) for m in markers_with_data]
    colors = [
        FAMILY_COLORS.get(_marker_family(m), "steelblue") for m in markers_with_data
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(effects)), effects, color=colors, edgecolor="none")
    ax.set_xticks(range(len(effects)))
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Causal Effect  Δ P(MCS)")
    ax.set_title(f"Causal Tracing — {model_name}{title_suffix}")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    for tick, marker in zip(ax.get_xticklabels(), markers_with_data):
        fam = _marker_family(marker)
        tick.set_color(FAMILY_COLORS.get(fam, "black"))

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[f], label=f.replace("_", " "))
        for f in FAMILY_ORDER
        if f in FAMILY_COLORS
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(op.join(out_dir, filename), dpi=150)
    plt.close(fig)
    print(f"   Saved {filename}", flush=True)


def plot_causal_effects(results, marker_names, model_name, out_dir):
    """Bar plots: causal effect per marker for one model (both directions)."""
    # VS→MCS direction
    _plot_causal_bar(
        results,
        marker_names,
        model_name,
        out_dir,
        effect_key="causal_effect",
        title_suffix=" (VS→MCS)",
        filename="causal_effects.png",
    )
    # MCS→VS direction
    _plot_causal_bar(
        results,
        marker_names,
        model_name,
        out_dir,
        effect_key="causal_effect_mcs",
        title_suffix=" (MCS→VS)",
        filename="causal_effects_mcs.png",
    )


def _get_common_markers(all_results):
    """Return set of markers common to all models in *all_results*."""
    common = None
    for results in all_results.values():
        markers_in_model = set(results.keys())
        common = markers_in_model if common is None else (common & markers_in_model)
    return common or set()


def _order_by_family_and_effect(common_markers, all_results):
    """Order markers by family group, ascending mean |causal effect|."""

    def _mean_abs(m):
        vals = [
            abs(all_results[model][m]["causal_effect"])
            for model in all_results
            if m in all_results[model]
        ]
        return float(np.mean(vals)) if vals else 0.0

    families = {fam: [] for fam in FAMILY_ORDER}
    for marker in common_markers:
        fam = _marker_family(marker)
        families.setdefault(fam, []).append(marker)

    ordered = []
    for fam in FAMILY_ORDER:
        fam_markers = sorted(
            families.get(fam, []), key=lambda m: (_mean_abs(m), _short_name(m))
        )
        ordered.extend(fam_markers)
    return ordered


def _load_r2_marker_order(results_root):
    """Load R² from embedding regressors and return family+R²-ordered list."""
    embed_raw = {}
    for model in FM_MODELS:
        path = op.join(
            results_root,
            model,
            "doc_patients",
            "MLP_EMBEDDING",
            "regressor_results",
            "summary.json",
        )
        if not op.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        embed_raw[model] = {
            k: v["r2"] for k, v in data.items() if not v.get("skipped", False)
        }

    if not embed_raw:
        return None

    all_markers = set()
    for dct in embed_raw.values():
        all_markers.update(dct.keys())

    def _mean_raw(marker):
        vals = [embed_raw[em].get(marker, np.nan) for em in embed_raw]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    families = {fam: [] for fam in FAMILY_ORDER}
    for marker in all_markers:
        fam = _marker_family(marker)
        families.setdefault(fam, []).append(marker)

    ordered = []
    for fam in FAMILY_ORDER:
        fam_markers = sorted(
            families.get(fam, []), key=lambda m: (_mean_raw(m), _short_name(m))
        )
        ordered.extend(fam_markers)
    return ordered


def _plot_causal_line(marker_order, all_results, ax, effect_key="causal_effect"):
    """Draw one line per FM model on *ax* for the given *marker_order*."""
    x = np.arange(len(marker_order))
    for model in FM_MODELS:
        if model not in all_results:
            continue
        effects = [
            all_results[model].get(m, {}).get(effect_key, np.nan) for m in marker_order
        ]
        ax.plot(
            x,
            effects,
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=model,
            color=_MODEL_COLORS.get(model, None),
            alpha=0.85,
        )

    short_names = [_short_name(m) for m in marker_order]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Causal Effect  Δ P(MCS)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _color_xticks_by_family(ax, marker_order)


def _save_comparison_fig(
    marker_order, all_results, title, out_path, effect_key="causal_effect"
):
    """Helper: create and save a single comparison line plot."""
    fig, ax = plt.subplots(figsize=(16, 5))
    _plot_causal_line(marker_order, all_results, ax, effect_key=effect_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    fname = op.basename(out_path)
    print(f"   Saved {fname}", flush=True)


def plot_comparison(all_results, marker_names, out_dir):
    """Line plots: causal effect per marker, one curve per model.

    Markers are ordered by family group (evoked → wSMI → information_theory
    → power_spectral_density), ascending mean |causal effect| within each.
    Produces both VS→MCS and MCS→VS direction plots.
    """
    common_markers = _get_common_markers(all_results)
    if not common_markers:
        print("   No common markers across models for comparison plot.")
        return

    marker_order = _order_by_family_and_effect(common_markers, all_results)

    # VS→MCS
    _save_comparison_fig(
        marker_order,
        all_results,
        "Causal Tracing — Model Comparison (VS→MCS)",
        op.join(out_dir, "causal_effects_comparison.png"),
        effect_key="causal_effect",
    )
    # MCS→VS
    _save_comparison_fig(
        marker_order,
        all_results,
        "Causal Tracing — Model Comparison (MCS→VS)",
        op.join(out_dir, "causal_effects_comparison_mcs.png"),
        effect_key="causal_effect_mcs",
    )


def plot_comparison_r2_order(all_results, out_dir, results_root):
    """Same as causal_effects_comparison but with markers ordered by R².

    Uses the family + ascending mean R² ordering from the embedding
    regressors (matching feature_importance_raw.png).
    Produces both VS→MCS and MCS→VS direction plots.
    """
    r2_order = _load_r2_marker_order(results_root)
    if r2_order is None:
        print("   Could not load R² data for R²-ordered comparison plot.")
        return

    common_markers = _get_common_markers(all_results)
    if not common_markers:
        print("   No common markers across models for R²-ordered plot.")
        return

    marker_order = [m for m in r2_order if m in common_markers]
    if not marker_order:
        print("   No overlapping markers between causal and R² results.")
        return

    # VS→MCS
    _save_comparison_fig(
        marker_order,
        all_results,
        "Causal Tracing — Model Comparison (R² order, VS→MCS)",
        op.join(out_dir, "causal_effects_comparison_r2_order.png"),
        effect_key="causal_effect",
    )
    # MCS→VS
    _save_comparison_fig(
        marker_order,
        all_results,
        "Causal Tracing — Model Comparison (R² order, MCS→VS)",
        op.join(out_dir, "causal_effects_comparison_r2_order_mcs.png"),
        effect_key="causal_effect_mcs",
    )


# ── Save results ─────────────────────────────────────────────────────────


def save_results(results, marker_names, model_name, out_dir):
    """Save JSON (aggregated fold results) and CSV summary."""
    os.makedirs(out_dir, exist_ok=True)

    # ── JSON ──
    json_data = {
        "model": model_name,
        "method": "embedding_steering",
        "markers": {},
    }
    for marker, data in results.items():
        json_data["markers"][marker] = {
            "causal_effect_vs_to_mcs_mean": data["causal_effect"],
            "causal_effect_vs_to_mcs_std": data.get("causal_effect_std", 0.0),
            "causal_effect_mcs_to_vs_mean": data["causal_effect_mcs"],
            "causal_effect_mcs_to_vs_std": data.get("causal_effect_mcs_std", 0.0),
            "n_folds": data.get("n_folds", 1),
            "causal_effect_per_fold": data.get("causal_effect_per_fold", [data["causal_effect"]]),
            "causal_effect_mcs_per_fold": data.get("causal_effect_mcs_per_fold", [data["causal_effect_mcs"]]),
        }

    json_path = op.join(out_dir, "embedding_steering_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"   Saved {json_path}", flush=True)

    # ── Summary CSV: one row per marker (mean ± std across folds) ──
    summary_rows = []
    for marker, data in results.items():
        summary_rows.append(
            {
                "marker": _short_name(marker),
                "marker_full": marker,
                "causal_effect_vs_to_mcs_mean": data["causal_effect"],
                "causal_effect_vs_to_mcs_std": data.get("causal_effect_std", 0.0),
                "causal_effect_mcs_to_vs_mean": data["causal_effect_mcs"],
                "causal_effect_mcs_to_vs_std": data.get("causal_effect_mcs_std", 0.0),
                "n_folds": data.get("n_folds", 1),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(
        "causal_effect_vs_to_mcs_mean",
        ascending=False,
    )
    summary_csv_path = op.join(out_dir, "embedding_steering_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"   Saved {summary_csv_path} ({len(df_summary)} rows)", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────


def run_model(
    model_name,
    results_root,
    marker_csv,
    patient_labels,
    reduction,
    output_root,
    classifier_type,
    precomputed_splits=None,
    common_sessions_ref=None,
    n_cv_folds=5,
    random_state=42,
):
    """Run the fold-wise embedding steering pipeline for a single FM model.

    Probe training happens **per outer fold** (fit on train split, steer on test
    split) to avoid data leakage.  Results are aggregated across folds.

    Parameters
    ----------
    precomputed_splits : list of fold dicts or None
        Pre-computed CV folds.  When provided, fold assignments match the
        MLP_EMBEDDING classifier splits for a fair comparison.
    common_sessions_ref : list of str or None
        Session list corresponding to the splits file.
    n_cv_folds : int
        Outer fold count when generating new splits (default: 5).
    random_state : int
        Seed for fold generation when no splits file is provided.

    Returns
    -------
    results_agg : dict
        Per-marker aggregated causal effects (mean/std across folds).
    precomputed_splits : list of fold dicts
        Folds used (pre-computed or newly generated).
    common_keys : list of str
        Session list used for the fold indices.
    labels_for_splits : dict
        ``{session_key: int_label}`` for ``common_keys``.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f" Embedding Steering: {model_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # 1. Load data
    print("\n[1/5] Loading embeddings...", flush=True)
    embeddings = load_embeddings(results_root, model_name)
    if not embeddings:
        print(f"  [SKIP] No embeddings found for {model_name}.", flush=True)
        return None, None, None, None

    print("[1/5] Loading markers...", flush=True)
    markers_dict, marker_names = load_markers(marker_csv, reduction)

    print("[1/5] Loading patient labels...", flush=True)
    labels_dict = load_patient_labels(patient_labels)

    print("[1/5] Loading classifier...", flush=True)
    try:
        fold_models, clf_results = load_classifier(
            results_root, model_name, classifier_type
        )
    except FileNotFoundError as exc:
        print(f"  [SKIP] Classifier not found: {exc}", flush=True)
        return None, None, None, None
    class_names = clf_results["class_names"]

    # 2. Align sessions
    if precomputed_splits is not None and common_sessions_ref is not None:
        available = set(embeddings) & set(markers_dict) & set(labels_dict)
        missing = [s for s in common_sessions_ref if s not in available]
        if missing:
            print(
                f"  [WARN] {len(missing)} sessions from splits not available — "
                "regenerating splits for this model.",
                flush=True,
            )
            precomputed_splits = None
            common_keys = sorted(available)
        else:
            common_keys = list(common_sessions_ref)
    else:
        common_keys = sorted(
            set(embeddings) & set(markers_dict) & set(labels_dict)
        )

    if len(common_keys) < 10:
        print(
            f"  [SKIP] Only {len(common_keys)} aligned sessions — need >= 10.",
            flush=True,
        )
        return None, None, None, None

    # Binary integer labels for splits (0 = VS, 1 = MCS)
    int_label = {"VS": 0, "MCS": 1}
    labels_for_splits = {k: int_label[labels_dict[k]] for k in common_keys
                         if labels_dict[k] in int_label}
    common_keys = [k for k in common_keys if k in labels_for_splits]

    X_all = np.stack([embeddings[k] for k in common_keys])   # (n, emb_dim)
    Y_all = np.stack([markers_dict[k] for k in common_keys])  # (n, n_markers)

    # 3. Generate splits if not provided
    if precomputed_splits is None:
        precomputed_splits = generate_nested_cv_folds(
            common_sessions=common_keys,
            labels=labels_for_splits,
            n_outer=n_cv_folds,
            random_state=random_state,
        )
        print(
            f"\n[2/5] Generated {len(precomputed_splits)} outer folds "
            f"(StratifiedGroupKFold, seed={random_state})",
            flush=True,
        )
    else:
        print(
            f"\n[2/5] Using {len(precomputed_splits)} pre-computed outer folds.",
            flush=True,
        )

    groups_all = np.array([k.split("_ses-")[0] for k in common_keys])

    # Leakage sanity checks
    for fold_idx, fold in enumerate(precomputed_splits):
        check_no_subject_leakage(
            groups_all, fold["train_idx"], fold["test_idx"],
            label=f"outer fold {fold_idx}",
        )

    # 4. Fold-wise probe training + causal effect computation
    print("\n[3-4/5] Fold-wise probe training and causal effects...", flush=True)

    # Accumulate per-marker per-fold effects
    fold_effects: dict[str, list[float]] = {}  # marker -> [effect_per_fold]
    fold_effects_mcs: dict[str, list[float]] = {}

    for fold_idx, fold in enumerate(precomputed_splits):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        print(
            f"\n   Fold {fold_idx + 1}/{len(precomputed_splits)}: "
            f"train={len(train_idx)}, test={len(test_idx)}",
            flush=True,
        )

        X_train = X_all[train_idx]
        Y_train = Y_all[train_idx]
        X_test = X_all[test_idx]
        test_keys = [common_keys[i] for i in test_idx]
        groups_train = groups_all[train_idx]

        # Use the fold-specific classifier (avoid using test-fold classifiers)
        clf_idx = min(fold_idx, len(fold_models) - 1)
        fold_clf = fold_models[clf_idx]

        # Train probes on the training split only
        n_inner = len(fold.get("inner_splits") or []) or 3
        probes = train_probes(
            X_train, Y_train, marker_names, groups_train,
            n_inner_splits=n_inner, random_state=random_state,
        )

        # Compute causal effects on the test split
        try:
            fold_results = compute_causal_effects(
                X_test, test_keys, labels_dict, probes, marker_names,
                fold_clf, class_names,
            )
        except ValueError as exc:
            print(f"   [WARN] Fold {fold_idx + 1}: {exc}", flush=True)
            continue

        for marker, res in fold_results.items():
            fold_effects.setdefault(marker, []).append(res["causal_effect"])
            fold_effects_mcs.setdefault(marker, []).append(res["causal_effect_mcs"])

    # 5. Aggregate across folds
    results_agg = {}
    for marker in marker_names:
        effects = fold_effects.get(marker, [])
        effects_mcs = fold_effects_mcs.get(marker, [])
        if not effects:
            continue
        results_agg[marker] = {
            "causal_effect": float(np.mean(effects)),
            "causal_effect_std": float(np.std(effects)) if len(effects) > 1 else 0.0,
            "causal_effect_mcs": float(np.mean(effects_mcs)),
            "causal_effect_mcs_std": float(np.std(effects_mcs)) if len(effects_mcs) > 1 else 0.0,
            "n_folds": len(effects),
            "causal_effect_per_fold": effects,
            "causal_effect_mcs_per_fold": effects_mcs,
        }

    print(f"\n   Aggregated effects for {len(results_agg)} markers.", flush=True)

    # 6. Save and plot
    out_dir = op.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n[5/5] Saving results and plots...", flush=True)
    save_results(results_agg, marker_names, model_name, out_dir)
    plot_causal_effects(results_agg, marker_names, model_name, out_dir)

    return results_agg, precomputed_splits, common_keys, labels_for_splits


def main():
    parser = argparse.ArgumentParser(
        description="Embedding steering via marker-guided probe interventions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="all",
        help=(
            "Foundation model name (LaBram, CBraMod, NeuroLM, TOTEM) "
            "or 'all' for all four."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root of benchmark results (contains FM model folders).",
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
        "--reduction",
        default="A",
        choices=list(REDUCTION_MAP.keys()),
        help="Marker CSV reduction variant (default: A = trim_mean80).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to {results-root}/EMBEDDING_STEERING.",
    )
    parser.add_argument(
        "--classifier-type",
        default="kernel_ridge",
        choices=["kernel_ridge", "mlp", "random_forest"],
        help="Which pre-trained classifier to use (default: kernel_ridge).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Produce cross-model comparison plot (only with --model all).",
    )
    parser.add_argument(
        "--n-cv-folds",
        type=int,
        default=5,
        help="Number of outer CV folds when generating new splits (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for fold generation (default: 42).",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help=(
            "Pre-computed CV splits JSON (from generate_nested_cv_folds). "
            "Pass the same file used for the MLP_EMBEDDING classifier run to "
            "ensure fold assignments match."
        ),
    )
    parser.add_argument(
        "--save-splits-to",
        default=None,
        help=(
            "Path to save generated splits. If omitted, splits are auto-saved to "
            "{output_dir}/cv_splits/embedding_steering_crs_{timestamp}.json."
        ),
    )
    args = parser.parse_args()

    output_root = args.output_dir or op.join(args.results_root, "EMBEDDING_STEERING")
    os.makedirs(output_root, exist_ok=True)

    # ── Load or prepare splits ────────────────────────────────────────────────
    precomputed_splits = None
    common_sessions_ref = None

    if args.splits_file and op.isfile(args.splits_file):
        precomputed_splits, common_sessions_ref, _ = load_cv_splits(args.splits_file)
        print(
            f"Loaded {len(precomputed_splits)} pre-computed outer folds "
            f"({len(common_sessions_ref)} sessions) from {args.splits_file}",
            flush=True,
        )

    models = FM_MODELS if args.model == "all" else [args.model]

    all_results = {}
    all_marker_names = None
    _splits_saved = precomputed_splits is not None

    for model in models:
        try:
            result = run_model(
                model,
                args.results_root,
                args.marker_csv,
                args.patient_labels,
                args.reduction,
                output_root,
                args.classifier_type,
                precomputed_splits=precomputed_splits,
                common_sessions_ref=common_sessions_ref,
                n_cv_folds=args.n_cv_folds,
                random_state=args.random_state,
            )
            results_agg, splits_used, common_used, labels_used = result
            if results_agg is not None:
                all_results[model] = results_agg
                if all_marker_names is None:
                    _, all_marker_names = load_markers(args.marker_csv, args.reduction)

                # Lock in splits from first successful run
                if not _splits_saved and splits_used is not None:
                    precomputed_splits = splits_used
                    common_sessions_ref = common_used
                    save_path = args.save_splits_to
                    if not save_path:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        splits_dir = op.join(output_root, "cv_splits")
                        save_path = op.join(
                            splits_dir, f"embedding_steering_crs_{ts}.json"
                        )
                    save_cv_splits(splits_used, common_used, labels_used, save_path)
                    print(f"  CV splits saved to {save_path}", flush=True)
                    _splits_saved = True
        except Exception as e:
            print(f"\n   ERROR running {model}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    # Cross-model comparison plot
    if len(all_results) > 1 and (args.compare or args.model == "all"):
        print("\n" + "=" * 60, flush=True)
        print(" Cross-Model Comparison", flush=True)
        print("=" * 60, flush=True)
        if all_marker_names is not None:
            plot_comparison(all_results, all_marker_names, output_root)
            plot_comparison_r2_order(
                all_results,
                output_root,
                args.results_root,
            )

    print(f"\nDone. Results saved to {output_root}", flush=True)


if __name__ == "__main__":
    main()
