"""Compare marker-level R² across foundation model embeddings.

==================================================
Feature Importance / Marker Reconstruction Comparison
==================================================

This script produces a figure comparing how well each foundation model embedding
encodes each neurophysiological marker.

* **Foundation model embeddings** (NeuroLM, TOTEM, CBraMod, LaBram):
  For each marker, the R² of a Ridge regression that predicts the marker
  value *from* the embedding (read from ``regressor_results/summary.json``).
  High R² → the embedding faithfully encodes that marker.

* **Max-normalised R²**:
  ``R²_max_norm = R²_raw / max(R²_across_models)`` per marker.
  Highlights relative encoding strength across models.

Usage
-----
::

    python feature_importance_comparison.py \\
        --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results \\
        --output-dir /data/.../combined_plots

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import os
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_COLORS = {
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",  # violet
    "wsmi": "#8b4513",  # brown
    "evoked": "#c55a11",  # dark orange
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _short_name(full_path: str) -> str:
    """``nice/marker/PowerSpectralDensity/theta`` → ``PowerSpectralDensity_theta``."""
    parts = full_path.split("/")
    return "_".join(parts[-2:])


def _marker_family(marker_name: str) -> str:
    """Map marker name to family bucket used for grouped x-axis ordering."""
    if "KolmogorovComplexity" in marker_name or "PermutationEntropy" in marker_name:
        return "information_theory"
    if (
        "PowerSpectralDensity" in marker_name
        or "PowerSpectralDensitySummary" in marker_name
    ):
        return "power_spectral_density"
    if "SymbolicMutualInformation" in marker_name:
        return "wsmi"
    return "evoked"


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
            marker: min(r2 / max_r2[marker], 1.0) for marker, r2 in r2_dict.items()
        }
    return embed_max_norm


def _ordered_markers_by_family_and_mean(embed_raw: dict) -> list:
    """Order markers by family, then ascending mean raw R² across models."""
    all_markers = set()
    for dct in embed_raw.values():
        all_markers.update(dct.keys())

    def _mean_raw(marker: str) -> float:
        vals = [embed_raw[em].get(marker, np.nan) for em in embed_raw]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    families = {fam: [] for fam in FAMILY_ORDER}
    for marker in all_markers:
        fam = _marker_family(marker)
        families.setdefault(fam, []).append(marker)

    ordered = []
    for fam in FAMILY_ORDER:
        fam_markers = families.get(fam, [])
        fam_markers = sorted(
            fam_markers,
            key=lambda m: (_mean_raw(m), _short_name(m)),
        )
        ordered.extend(fam_markers)
    return ordered


def _color_xticks_by_family(ax, marker_order: list) -> None:
    """Color x tick labels according to marker family."""
    for tick, marker in zip(ax.get_xticklabels(), marker_order):
        fam = _marker_family(marker)
        tick.set_color(FAMILY_COLORS.get(fam, "black"))


def _save_raw_only_plot(
    embed_raw: dict,
    marker_order: list,
    short_names: list,
    output_dir: str,
) -> None:
    """Save a standalone raw R² plot (equivalent to the bottom combined subplot)."""
    x = np.arange(len(marker_order))

    fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 7))

    for mi, (em, r2_raw) in enumerate(embed_raw.items()):
        y_raw = [r2_raw.get(m, np.nan) for m in marker_order]
        ax.plot(
            x,
            y_raw,
            label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            alpha=0.9,
        )

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Markers", fontsize=17)
    ax.set_ylabel("R²", fontsize=17)
    ax.set_title("Raw R² (Ridge regression: embedding → marker)", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=12)
    _color_xticks_by_family(ax, marker_order)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = op.join(output_dir, "feature_importance_raw.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Raw-only plot saved: {out_path}", flush=True)


# ── Main plotting function ────────────────────────────────────────────────────


def plot_comparison(
    results_dir: str,
    output_dir: str,
):
    """Generate R² comparison plots (max-normalised and raw).

    Parameters
    ----------
    results_dir : str
        Root results directory containing NeuroLM/, TOTEM/, etc.
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

    # Use raw R² directly from embedding regressors
    embed_raw = embed_data

    # ── Max-normalize R² ──────────────────────────────────────────────────────
    embed_max_norm = _compute_r2_max_normalized(embed_raw)

    # ── Determine marker order: family groups, then ascending mean across models ──
    marker_order = _ordered_markers_by_family_and_mean(embed_raw)
    short_names = [_short_name(m) for m in marker_order]
    x = np.arange(len(marker_order))

    # ── Plot: 2 subplots stacked ──────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(max(16, len(marker_order) * 0.65), 13), sharex=True
    )
    fig.suptitle(
        "Marker Reconstruction R² from Foundation Model Embeddings",
        fontsize=13,
        y=0.98,
    )

    # ── Top: R²_max_normalised ────────────────────────────────────────────────
    ax_top = axes[0]
    for mi, em in enumerate(embed_max_norm):
        y_max = [embed_max_norm[em].get(m, np.nan) for m in marker_order]
        ax_top.plot(
            x,
            y_max,
            label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            alpha=0.85,
        )
    ax_top.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_top.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_top.set_ylabel("R²_max_norm\n(R²_raw / max across models)", fontsize=10)
    ax_top.set_ylim(-0.05, 1.15)
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.grid(True, alpha=0.25)
    ax_top.set_title("Max-normalised R² (relative encoding strength per marker)")

    # ── Bottom: raw R² ────────────────────────────────────────────────────────
    ax_bot = axes[1]
    for mi, (em, r2_raw) in enumerate(embed_raw.items()):
        y_raw = [r2_raw.get(m, np.nan) for m in marker_order]
        ax_bot.plot(
            x,
            y_raw,
            label=em,
            color=_EMBED_COLORS[mi % len(_EMBED_COLORS)],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            alpha=0.85,
        )

    ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bot.set_ylabel("R²", fontsize=10)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    _color_xticks_by_family(ax_bot, marker_order)
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_title("Raw R² (Ridge regression: embedding → marker)")

    plt.tight_layout()
    fname = "feature_importance_comparison.png"
    out_path = op.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Plot saved to: {out_path}", flush=True)

    # ── Save per-model plots ──────────────────────────────────────────────────
    for em in embed_raw:
        _plot_single_model(
            em,
            embed_raw[em],
            marker_order,
            short_names,
            output_dir,
        )

    # ── Save summary CSV ─────────────────────────────────────────────────────
    import pandas as pd

    rows = []
    for m in marker_order:
        short = _short_name(m)
        row = {"marker_short": short, "marker_full": m}
        for em in embed_raw:
            row[f"{em}_r2_raw"] = embed_raw[em].get(m, np.nan)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = op.join(output_dir, "r2_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"   Summary CSV saved to: {csv_path}", flush=True)

    # ── Save dedicated raw-only plot (bottom subplot style) ─────────────────
    _save_raw_only_plot(embed_raw, marker_order, short_names, output_dir)


def _plot_single_model(
    model_name,
    r2_raw,
    marker_order,
    short_names,
    output_dir,
):
    """Save a single-model raw R² plot."""
    x = np.arange(len(marker_order))
    y_raw = [r2_raw.get(m, np.nan) for m in marker_order]

    color = _EMBED_COLORS[EMBEDDING_MODELS.index(model_name) % len(_EMBED_COLORS)]

    fig, ax = plt.subplots(figsize=(max(14, len(marker_order) * 0.6), 7))
    fig.suptitle(f"{model_name} — Marker R²", fontsize=12, y=0.99)

    ax.plot(
        x,
        y_raw,
        color=color,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
        alpha=0.8,
        label="R²_raw",
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("R²")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    _color_xticks_by_family(ax, marker_order)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title("Raw R²")

    plt.tight_layout()
    out = op.join(output_dir, f"feature_importance_comparison_{model_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Per-model plot saved: {out}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-marker R² from foundation-model embedding regressors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
python feature_importance_comparison.py \\
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results \\
    --output-dir /tmp/plots
        """,
    )
    parser.add_argument(
        "--results-dir",
        default=("/data/project/eeg_foundation/data/benchmark_results/new_results"),
        help="Root results directory containing NeuroLM/, TOTEM/, CBraMod/, LaBram/.",
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
    print("FEATURE IMPORTANCE COMPARISON", flush=True)
    print(f"  results_dir : {args.results_dir}", flush=True)
    print(f"  output_dir  : {args.output_dir}", flush=True)
    print("=" * 70, flush=True)

    plot_comparison(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
