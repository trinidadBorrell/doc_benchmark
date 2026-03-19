"""Compare marker-level R² across foundation model embeddings with noise ceiling correction.

==================================================
Feature Importance / Marker Reconstruction Comparison
==================================================

This script produces a figure comparing how well each foundation model embedding
encodes each neurophysiological marker, corrected by the noise ceiling of each
marker (i.e. its split-half reliability).

* **Foundation model embeddings** (NeuroLM, TOTEM, CBraMod, LaBram):
  For each marker, the R² of a Ridge regression that predicts the marker
  value *from* the embedding (read from ``regressor_results/summary.json``).
  High R² → the embedding faithfully encodes that marker.

* **Noise ceiling** (split-half reliability, Spearman-Brown corrected):
  For each marker, the maximum R² achievable given the marker's
  test–retest reliability across subjects.

* **Noise-ceiling-normalised R²**:
  ``R²_norm = R²_raw / noise_ceiling`` (clipped to [0, 1]).
  Values near 1 → the embedding nearly perfectly captures the marker.

Usage
-----
::

    python feature_importance_comparison.py \\
        --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results \\
        --marker-csv /data/.../baseline_stable_20210128_scalars.csv \\
        --marker-reduction A \\
        --output-dir /data/.../combined_plots

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import logging
import os
import os.path as op
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"

# ── Foundation models whose embedding regressor summaries we compare ──────────
EMBEDDING_MODELS = ["NeuroLM", "TOTEM", "CBraMod", "LaBram"]

REDUCTION_MAP = {
    "A": "icm/lg/egi256/trim_mean80",
    "B": "icm/lg/egi256/std",
    "C": "icm/lg/egi256gfp/trim_mean80",
    "D": "icm/lg/egi256gfp/std",
}

# Colour palette: 4 embedding models
_EMBED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _short_name(full_path: str) -> str:
    """``nice/marker/PowerSpectralDensity/theta`` → ``PowerSpectralDensity_theta``."""
    parts = full_path.split("/")
    return "_".join(parts[-2:])


def _load_embedding_r2(results_dir: str, model_name: str) -> dict:
    """Load marker R² from MLP embedding regressor summary.

    Returns
    -------
    dict[str, float]  full_marker_path → r2
    """
    path = op.join(
        results_dir,
        model_name,
        "doc_patients",
        "MLP_EMBEDDING",
        "regressor_results",
        "summary.json",
    )
    if not op.isfile(path):
        print(f"   [WARN] Not found: {path}", flush=True)
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v["r2"] for k, v in data.items() if not v.get("skipped", False)}


def _compute_noise_ceiling(
    marker_csv: str, reduction_letter: str, n_splits: int = 50
) -> dict:
    """Compute Monte Carlo split-half noise ceiling for each marker.

    1. Load scalars CSV (semicolon-delimited), filter to the given reduction.
    2. Parse Subject column: "AA079_2" -> "AA079_ses-02"
    3. For each marker, repeat n_splits times:
         - randomly split subjects into two equal halves H1, H2
         - compute Pearson r between H1 and H2 (split-half reliability)
         - apply Spearman-Brown correction: r_sb = 2*r / (1 + |r|)
    4. Average r_sb across splits, then square: nc = mean(r_sb)^2 (clipped to [0,1]).
       Monte Carlo averaging removes dependence on any single random split.

    Parameters
    ----------
    marker_csv : str
        Path to semicolon-delimited scalars CSV.
    reduction_letter : str
        One of A/B/C/D; mapped to reduction string via REDUCTION_MAP.
    n_splits : int
        Number of random splits to average (default: 50).

    Returns
    -------
    dict[str, float]  marker_name -> noise_ceiling
    """
    if reduction_letter not in REDUCTION_MAP:
        raise ValueError(
            f"Unknown reduction {reduction_letter!r}. Choose from {list(REDUCTION_MAP)}."
        )
    reduction_str = REDUCTION_MAP[reduction_letter]

    df = pd.read_csv(marker_csv, sep=";")
    df_filt = df[df["Reduction"] == reduction_str].copy()
    if df_filt.empty:
        raise ValueError(
            f"No rows for Reduction={reduction_str!r} in {marker_csv}"
        )

    # Infer marker columns (numeric, not metadata/index)
    meta_cols = {"Subject", "Reduction", "Label"}
    marker_cols = [
        c
        for c in df_filt.columns
        if c not in meta_cols
        and not str(c).startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df_filt[c])
    ]

    # Parse subject keys: "AA079_2" -> "AA079_ses-02"
    parsed_subjects = []
    valid_rows = []
    for _, row in df_filt.iterrows():
        raw = str(row["Subject"])
        parts = raw.rsplit("_", 1)
        if len(parts) != 2:
            continue
        subj_id, sesnum = parts
        try:
            key = f"{subj_id}_ses-{int(sesnum):02d}"
        except ValueError:
            continue
        parsed_subjects.append(key)
        valid_rows.append(row[marker_cols].values.astype(float))

    if len(parsed_subjects) < 4:
        warnings.warn(
            f"Only {len(parsed_subjects)} subjects found — "
            "noise ceiling may be unreliable."
        )

    data_array = np.array(valid_rows)  # (n_subjects, n_markers)
    n_subj = len(parsed_subjects)

    half = n_subj // 2

    nc_dict = {}
    for mi, col in enumerate(marker_cols):
        r_sb_vals = []
        for seed in range(n_splits):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_subj)
            h1 = perm[:half]
            h2 = perm[half : 2 * half]

            v1 = data_array[h1, mi]
            v2 = data_array[h2, mi]

            # Remove NaN pairs
            valid = ~(np.isnan(v1) | np.isnan(v2))
            v1, v2 = v1[valid], v2[valid]

            if len(v1) < 3:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_half, _ = pearsonr(v1, v2)

            if np.isnan(r_half):
                continue

            # Spearman-Brown prophecy formula
            r_sb = (2.0 * r_half) / (1.0 + abs(r_half))
            r_sb_vals.append(r_sb)

        if not r_sb_vals:
            nc_dict[col] = 0.0
            continue

        # Average r_sb across splits, then square for noise ceiling
        mean_r_sb = float(np.mean(r_sb_vals))
        nc = float(np.clip(mean_r_sb ** 2, 0.0, 1.0))
        nc_dict[col] = nc

    print(
        f"   Computed noise ceilings for {len(nc_dict)} markers "
        f"(reduction={reduction_str}, n_subjects={n_subj})",
        flush=True,
    )
    return nc_dict


def _compute_r2_normalized(
    r2_dict: dict, nc_dict: dict
) -> tuple:
    """Normalise raw R² by noise ceiling.

    Parameters
    ----------
    r2_dict : dict[str, float]
        marker_name -> raw R² from embedding regressor
    nc_dict : dict[str, float]
        marker_name -> noise ceiling

    Returns
    -------
    r2_norm : dict[str, float]
    r2_raw  : dict[str, float]  (subset with noise ceiling available)
    nc      : dict[str, float]  (subset with noise ceiling available)
    """
    r2_norm = {}
    r2_raw_out = {}
    nc_out = {}

    for marker, r2 in r2_dict.items():
        # Try to match marker by the last two path components
        short = _short_name(marker)
        # Look for exact match or short-name match in nc_dict
        nc_val = nc_dict.get(marker) or nc_dict.get(short)
        if nc_val is None:
            # Try partial match: find nc key that ends with short
            for nc_key in nc_dict:
                if nc_key.endswith(short) or short.endswith(nc_key):
                    nc_val = nc_dict[nc_key]
                    break

        if nc_val is None:
            continue

        denom = max(nc_val, 0.01)
        norm = min(r2 / denom, 1.0)
        if r2 / denom > 1.0 + 1e-6:
            print(
                f"   [WARN] R²_norm > 1.0 for {short} "
                f"(r2={r2:.3f}, nc={nc_val:.3f}) — clipped to 1.0",
                flush=True,
            )
        r2_norm[marker] = float(norm)
        r2_raw_out[marker] = float(r2)
        nc_out[marker] = float(nc_val)

    return r2_norm, r2_raw_out, nc_out


def _compute_r2_max_normalized(
    embed_raw: dict,
) -> dict:
    """Max-normalize raw R²: for each marker, divide by max R² across models.

    Parameters
    ----------
    embed_raw : dict[model_name, dict[marker, float]]

    Returns
    -------
    dict[model_name, dict[marker, float]]
    """
    # Collect all markers and their max R² across models
    all_markers = set()
    for r2_dict in embed_raw.values():
        all_markers.update(r2_dict.keys())

    max_r2 = {}
    for marker in all_markers:
        vals = [d.get(marker, 0.0) for d in embed_raw.values()]
        max_r2[marker] = max(max(vals), 1e-8)

    embed_max_norm = {}
    for em, r2_dict in embed_raw.items():
        embed_max_norm[em] = {
            marker: min(r2 / max_r2[marker], 1.0)
            for marker, r2 in r2_dict.items()
        }
    return embed_max_norm


# ── Main plotting function ────────────────────────────────────────────────────


def plot_comparison(
    results_dir: str,
    marker_csv: str,
    marker_reduction: str,
    output_dir: str,
):
    """Generate noise-ceiling-corrected R² comparison plots.

    Parameters
    ----------
    results_dir : str
        Root results directory containing NeuroLM/, TOTEM/, etc.
    marker_csv : str
        Path to semicolon-delimited baseline scalars CSV.
    marker_reduction : str
        Reduction letter A/B/C/D.
    output_dir : str
        Directory where plots and summary CSV are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load embedding R² ─────────────────────────────────────────────────────
    embed_data = {}
    for em in EMBEDDING_MODELS:
        r2_dict = _load_embedding_r2(results_dir, em)
        if r2_dict:
            embed_data[em] = r2_dict
            print(
                f"   Loaded {len(r2_dict)} markers for {em} (embedding R²)",
                flush=True,
            )

    if not embed_data:
        print("No embedding data loaded — aborting.", flush=True)
        return

    # ── Compute noise ceilings ────────────────────────────────────────────────
    print("\n   Computing noise ceilings ...", flush=True)
    nc_dict = _compute_noise_ceiling(marker_csv, marker_reduction)

    # ── Normalise R² per model ────────────────────────────────────────────────
    embed_norm = {}
    embed_raw = {}
    nc_per_model = {}
    for em, r2_dict in embed_data.items():
        r2_norm, r2_raw, nc_used = _compute_r2_normalized(r2_dict, nc_dict)
        embed_norm[em] = r2_norm
        embed_raw[em] = r2_raw
        nc_per_model[em] = nc_used

    # ── Max-normalize R² ──────────────────────────────────────────────────────
    embed_max_norm = _compute_r2_max_normalized(embed_raw)

    # ── Determine common marker order ─────────────────────────────────────────
    all_markers = set()
    for d in embed_norm.values():
        all_markers.update(d.keys())
    # Sort by R² normalised (average across models) descending
    def _avg_norm(m):
        vals = [embed_norm[em].get(m, 0.0) for em in embed_norm]
        return -np.nanmean(vals)

    marker_order = sorted(all_markers, key=_avg_norm)
    short_names = [_short_name(m) for m in marker_order]
    x = np.arange(len(marker_order))

    # ── Plot: 3 subplots stacked ──────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(max(16, len(marker_order) * 0.65), 18), sharex=True
    )
    fig.suptitle(
        "Marker Reconstruction R² from Foundation Model Embeddings\n"
        f"(noise-ceiling correction, reduction={marker_reduction})",
        fontsize=13,
        y=0.98,
    )

    # ── Top: R²_normalised (noise-ceiling) ────────────────────────────────────
    ax_top = axes[0]
    for mi, (em, r2_norm) in enumerate(embed_norm.items()):
        y_norm = [r2_norm.get(m, np.nan) for m in marker_order]
        ax_top.plot(
            x, y_norm, label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o", markersize=4, linewidth=1.5, alpha=0.85,
        )

    # Noise ceiling line (shared property of marker, not model)
    # Use the first model's nc values as representative
    first_em = next(iter(nc_per_model))
    nc_vals_ordered = [nc_per_model[first_em].get(m, np.nan) for m in marker_order]
    ax_top.plot(
        x,
        nc_vals_ordered,
        color="black",
        linestyle="--",
        marker="D",
        markersize=4,
        label="Noise ceiling",
        alpha=0.7,
        zorder=5,
    )
    ax_top.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_top.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_ylabel("R²_normalised\n(R²_raw / noise_ceiling)", fontsize=10)
    ax_top.set_ylim(-0.05, 1.15)
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.grid(True, alpha=0.25)
    ax_top.set_title("Noise-ceiling-normalised R² (higher = embedding better encodes marker)")

    # ── Middle: R²_max_normalised ─────────────────────────────────────────────
    ax_mid = axes[1]
    for mi, em in enumerate(embed_max_norm):
        y_max = [embed_max_norm[em].get(m, np.nan) for m in marker_order]
        ax_mid.plot(
            x, y_max, label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o", markersize=4, linewidth=1.5, alpha=0.85,
        )
    ax_mid.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_mid.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_mid.set_ylabel("R²_max_norm\n(R²_raw / max across models)", fontsize=10)
    ax_mid.set_ylim(-0.05, 1.15)
    ax_mid.legend(loc="upper left", fontsize=8)
    ax_mid.grid(True, alpha=0.25)
    ax_mid.set_title("Max-normalised R² (relative encoding strength per marker)")

    # ── Bottom: raw R² ────────────────────────────────────────────────────────
    ax_bot = axes[2]
    for mi, (em, r2_raw) in enumerate(embed_raw.items()):
        y_raw = [r2_raw.get(m, np.nan) for m in marker_order]
        ax_bot.plot(
            x, y_raw, label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o", markersize=4, linewidth=1.5, alpha=0.85,
        )

    ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bot.set_ylabel("R² (raw)", fontsize=10)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_title("Raw R² (Ridge regression: embedding → marker)")

    plt.tight_layout()
    fname = f"feature_importance_comparison_reduction{marker_reduction}.png"
    out_path = op.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Plot saved to: {out_path}", flush=True)

    # ── Save per-model plots ──────────────────────────────────────────────────
    for em in embed_norm:
        _plot_single_model(
            em, embed_norm[em], embed_raw[em], nc_per_model[em],
            marker_order, short_names, marker_reduction, output_dir,
        )

    # ── Save noise ceiling summary CSV ────────────────────────────────────────
    rows = []
    for m in marker_order:
        short = _short_name(m)
        nc_val = nc_per_model.get(first_em, {}).get(m, np.nan)
        row = {"marker_short": short, "marker_full": m, "noise_ceiling": nc_val}
        for em in embed_norm:
            row[f"{em}_r2_raw"] = embed_raw[em].get(m, np.nan)
            row[f"{em}_r2_normalized"] = embed_norm[em].get(m, np.nan)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = op.join(output_dir, "noise_ceiling_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"   Summary CSV saved to: {csv_path}", flush=True)


def _plot_single_model(
    model_name, r2_norm, r2_raw, nc_vals,
    marker_order, short_names, reduction, output_dir,
):
    """Save a single-model comparison plot (normalised + raw + noise ceiling)."""
    x = np.arange(len(marker_order))
    y_norm = [r2_norm.get(m, np.nan) for m in marker_order]
    y_raw = [r2_raw.get(m, np.nan) for m in marker_order]
    nc_line = [nc_vals.get(m, np.nan) for m in marker_order]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(marker_order) * 0.6), 12), sharex=True)
    fig.suptitle(
        f"{model_name} — Marker R² (reduction={reduction})", fontsize=12, y=0.99
    )

    color = _EMBED_COLORS[EMBEDDING_MODELS.index(model_name) % len(_EMBED_COLORS)]

    ax0 = axes[0]
    ax0.plot(x, y_norm, color=color, marker="o", markersize=4, linewidth=1.5, alpha=0.8, label="R²_norm")
    ax0.plot(x, nc_line, color="black", linestyle="--", marker="D", markersize=4, label="NC", alpha=0.7, zorder=5)
    ax0.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax0.axhline(1, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax0.set_ylim(-0.05, 1.15)
    ax0.set_ylabel("R²_normalised")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Noise-ceiling-normalised R²")

    ax1 = axes[1]
    ax1.plot(x, y_raw, color=color, marker="o", markersize=4, linewidth=1.5, alpha=0.8, label="R²_raw")
    ax1.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax1.set_ylabel("R² (raw)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Raw R²")

    plt.tight_layout()
    out = op.join(output_dir, f"feature_importance_comparison_{model_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Per-model plot saved: {out}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-marker R² from foundation-model embedding regressors, "
            "normalised by split-half noise ceiling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
python feature_importance_comparison.py \\
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results \\
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \\
    --marker-reduction A \\
    --output-dir /tmp/plots
        """,
    )
    parser.add_argument(
        "--results-dir",
        default=(
            "/data/project/eeg_foundation/data/benchmark_results/new_results"
        ),
        help="Root results directory containing NeuroLM/, TOTEM/, CBraMod/, LaBram/.",
    )
    parser.add_argument(
        "--marker-csv",
        required=True,
        help="Path to semicolon-delimited baseline scalars CSV.",
    )
    parser.add_argument(
        "--marker-reduction",
        default="A",
        choices=list(REDUCTION_MAP.keys()),
        help="Reduction letter A/B/C/D (default: A = trim_mean80).",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/project/eeg_foundation/data/benchmark_results/new_results"
            "/combined_plots"
        ),
        help="Directory to save plots and CSV.",
    )

    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("FEATURE IMPORTANCE COMPARISON (noise-ceiling-corrected)", flush=True)
    print(f"  results_dir      : {args.results_dir}", flush=True)
    print(f"  marker_csv       : {args.marker_csv}", flush=True)
    print(f"  marker_reduction : {args.marker_reduction}", flush=True)
    print(f"  output_dir       : {args.output_dir}", flush=True)
    print("=" * 70, flush=True)

    plot_comparison(
        results_dir=args.results_dir,
        marker_csv=args.marker_csv,
        marker_reduction=args.marker_reduction,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
