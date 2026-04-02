#!/usr/bin/env python3
"""AUC dot-plots: per-target and per-FM-variant.

For CRS
-------
  1. crs_main_models.png         — x: Baseline, TOTEM, CBraMod, LaBram, NeuroLM (MLP_EMBEDDING)
  2. crs_TOTEM_variants.png      — x: TOTEM, TOTEM (Combined), TOTEM (DK Shifted),
                                       TOTEM (FM Shifted), TOTEM (All Shifted)
  3. crs_CBraMod_variants.png
  4. crs_LaBram_variants.png
  5. crs_NeuroLM_variants.png

For every other target (cs_1y, cs_2y, cs_6m, etiology, etiology_code,
and their binary sub-variants)
----------------------------------------------------------------------
  {target_slug}_main_models.png  — x: Baseline, TOTEM, CBraMod, LaBram, NeuroLM (MLP_EMBEDDING)

Each plot shows 4 classifiers as distinct markers with AUC ± std error bars.

Usage
-----
    python auc_plots.py
    python auc_plots.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"

# ── Constants ──────────────────────────────────────────────────────────────────

BASE = Path("/data/project/eeg_foundation/data/benchmark_results/new_results/")

FM_MODELS = ["TOTEM", "CBraMod", "LaBram", "NeuroLM"]
ALL_MODELS = ["MARKER_BASELINE"] + FM_MODELS

CLASSIFIERS = ["svm", "mlp", "kernel_ridge", "random_forest"]

_CLASSIFIER_MARKERS = {
    "svm":          ("s", "#ff7f0e"),
    "mlp":          ("^", "#2ca02c"),
    "kernel_ridge": ("o", "#1f77b4"),
    "random_forest":("D", "#d62728"),
}

# Display labels
_MODEL_LABELS = {
    "MARKER_BASELINE": "Baseline",
    "TOTEM":   "TOTEM",
    "CBraMod": "CBraMod",
    "LaBram":  "LaBram",
    "NeuroLM": "NeuroLM",
}

# CRS shifted variants: (embedding_kind, display_suffix)
_SHIFT_VARIANTS = [
    ("MLP_EMBEDDING",                    ""),           # base model — no suffix
    ("EMBEDDING_DK_COMBINED",            " (Combined)"),
    ("EMBEDDING_DK_COMBINED_DK_SHIFTED", " (DK Shifted)"),
    ("EMBEDDING_DK_COMBINED_FM_SHIFTED", " (FM Shifted)"),
    ("EMBEDDING_DK_COMBINED_BOTH_SHIFTED"," (All Shifted)"),
]

# Non-CRS targets to produce overview plots for
_OTHER_TASKS = [
    {"metric": "cs_1y",                        "metric_path": "cs_1y"},
    {"metric": "cs_2y",                        "metric_path": "cs_2y"},
    {"metric": "cs_6m",                        "metric_path": "cs_6m"},
    {"metric": "etiology",                     "metric_path": "etiology"},
    {"metric": "etiology_code",                "metric_path": "etiology_code"},
    {"metric": "cs_1y/binary_vs_to_mcs",       "metric_path": "cs_1y/binary_vs_to_mcs"},
    {"metric": "cs_2y/binary_vs_to_mcs",       "metric_path": "cs_2y/binary_vs_to_mcs"},
    {"metric": "cs_6m/binary_vs_to_mcs",       "metric_path": "cs_6m/binary_vs_to_mcs"},
    {"metric": "cs_1y/binary_mcs_to_conscious","metric_path": "cs_1y/binary_mcs_to_conscious"},
    {"metric": "cs_2y/binary_mcs_to_conscious","metric_path": "cs_2y/binary_mcs_to_conscious"},
    {"metric": "cs_6m/binary_mcs_to_conscious","metric_path": "cs_6m/binary_mcs_to_conscious"},
]

_METRIC_TITLE = {
    "crs":                            "CRS",
    "cs_1y":                          "CS 1Y",
    "cs_2y":                          "CS 2Y",
    "cs_6m":                          "CS 6M",
    "etiology":                       "Etiology",
    "etiology_code":                  "Etiology Code",
    "cs_1y/binary_vs_to_mcs":        "CS 1Y — VS to MCS",
    "cs_2y/binary_vs_to_mcs":        "CS 2Y — VS to MCS",
    "cs_6m/binary_vs_to_mcs":        "CS 6M — VS to MCS",
    "cs_1y/binary_mcs_to_conscious": "CS 1Y — MCS to Conscious",
    "cs_2y/binary_mcs_to_conscious": "CS 2Y — MCS to Conscious",
    "cs_6m/binary_mcs_to_conscious": "CS 6M — MCS to Conscious",
}

# ── Score reading ──────────────────────────────────────────────────────────────


def _safe_get(dct, *keys):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_scores(path: Path) -> dict[str, float | None]:
    """Read AUC mean/std from classification_results.json or summary_metrics.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # classification_results.json layout
    auc_mean = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "mean"))
    auc_std  = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "std"))

    # summary_metrics.json layout (shifted variants)
    if auc_mean is None:
        auc_mean = _to_float_or_none(data.get("auc_mean"))
    if auc_std is None:
        auc_std  = _to_float_or_none(data.get("auc_std"))

    # last fallback
    if auc_mean is None:
        auc_mean = _to_float_or_none(data.get("test_auc_score"))

    return {"mean": auc_mean, "std": auc_std if auc_std is not None else 0.0}


def _candidate_paths_main(model: str, metric_path: str, embedding_kind: str,
                           classifier: str) -> list[Path]:
    """Return candidate JSON paths for main-model results (priority order)."""
    metric_parts = Path(metric_path).parts

    if model == "MARKER_BASELINE":
        model_root = BASE / model
    else:
        model_root = BASE / model / "doc_patients" / embedding_kind

    candidates = [
        model_root / metric_path / "nested_cv" / classifier / "classification_results.json"
    ]

    # cs_* targets may live under a multiclass/ or binary/ subdirectory
    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        mn = metric_parts[0]
        candidates += [
            model_root / mn / "multiclass" / "nested_cv" / classifier / "classification_results.json",
            model_root / mn / "binary"     / "nested_cv" / classifier / "classification_results.json",
        ]

    if len(metric_parts) == 1:
        candidates.append(
            model_root / metric_parts[0] / "binary" / "nested_cv" / classifier / "classification_results.json"
        )

    # deduplicate while preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in candidates:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _candidate_paths_shifted(model: str, embedding_kind: str,
                              classifier: str) -> list[Path]:
    """Return candidate JSON paths for CRS shifted variants."""
    nested_cv = BASE / model / "doc_patients" / embedding_kind / "crs" / "nested_cv" / classifier
    return [
        nested_cv / "summary_metrics.json",
        nested_cv / "classification_results.json",
    ]


def _resolve_scores(candidates: list[Path]) -> dict[str, float | None] | None:
    for path in candidates:
        if path.is_file():
            return _read_json_scores(path)
    return None


# ── Generic data collector ─────────────────────────────────────────────────────


def collect_main_model_data(metric_path: str) -> dict[str, dict[str, dict]]:
    """Return {model_label: {clf: {mean, std}}} for Baseline + 4 FM models."""
    result: dict[str, dict[str, dict]] = {}
    for model in ALL_MODELS:
        label = _MODEL_LABELS[model]
        result[label] = {}
        for clf in CLASSIFIERS:
            candidates = _candidate_paths_main(model, metric_path, "MLP_EMBEDDING", clf)
            scores = _resolve_scores(candidates)
            if scores is not None:
                result[label][clf] = scores
    return result


def collect_crs_variant_data(fm_model: str) -> dict[str, dict[str, dict]]:
    """Return {variant_label: {clf: {mean, std}}} for all CRS variants of one FM model."""
    base_label = _MODEL_LABELS[fm_model]
    result: dict[str, dict[str, dict]] = {}
    for emb_kind, suffix in _SHIFT_VARIANTS:
        label = base_label + suffix
        result[label] = {}
        for clf in CLASSIFIERS:
            if emb_kind == "MLP_EMBEDDING":
                candidates = _candidate_paths_main(fm_model, "crs", "MLP_EMBEDDING", clf)
            else:
                candidates = _candidate_paths_shifted(fm_model, emb_kind, clf)
            scores = _resolve_scores(candidates)
            if scores is not None:
                result[label][clf] = scores
    return result


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_auc_dots(
    data: dict[str, dict[str, dict]],
    x_order: list[str],
    title: str,
    output_path: Path,
    band_groups: list[tuple[int, int]] | None = None,
) -> None:
    """Dot-plot with error bars: classifiers as different markers/colours.

    Parameters
    ----------
    data       : {model_label: {clf_name: {"mean": float, "std": float}}}
    x_order    : explicit x-axis order (labels present in *data*)
    title      : plot title
    output_path: where to save
    band_groups: optional list of (start_idx, end_idx) for background shading
    """
    x_labels = [m for m in x_order if m in data]
    if not x_labels:
        print(f"   Skipping (no data): {output_path.name}")
        return

    n_clf = len(CLASSIFIERS)
    x = np.arange(len(x_labels))
    width = 0.7 / max(n_clf, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.8), 5))

    if band_groups:
        for i, (start, end) in enumerate(band_groups):
            if i % 2 == 0:
                ax.axvspan(start - 0.5, end + 0.5, color="gray", alpha=0.07, zorder=0)

    handles = []
    for ci, clf_name in enumerate(CLASSIFIERS):
        marker, color = _CLASSIFIER_MARKERS.get(clf_name, ("o", None))
        means, stds = [], []
        for label in x_labels:
            entry = data.get(label, {}).get(clf_name, {})
            means.append(entry.get("mean", float("nan")))
            stds.append(entry.get("std", 0.0))

        offset = (ci - (n_clf - 1) / 2) * width
        (line,) = ax.plot([], [], marker=marker, color=color,
                          label=clf_name, linestyle="none")
        handles.append(line)
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt=marker,
            color=color,
            capsize=4,
            markersize=8,
            linewidth=1.5,
        )

    chance = ax.axhline(0.5, color="gray", linestyle="--",
                        linewidth=1, alpha=0.7, label="chance")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC (mean ± std)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(
        handles=handles + [chance],
        labels=list(CLASSIFIERS) + ["chance"],
        loc="lower right",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PNGs (default: new_results/PLOTS/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else BASE / "PLOTS"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CRS: overview plot (Baseline + 4 FM models, MLP_EMBEDDING) ───────────
    print("CRS main-models plot...")
    crs_main = collect_main_model_data("crs")
    x_order_main = ["Baseline", "TOTEM", "CBraMod", "LaBram", "NeuroLM"]
    plot_auc_dots(
        crs_main,
        x_order=x_order_main,
        title="CRS — AUC by model (MLP embedding, nested CV)",
        output_path=output_dir / "crs_main_models.png",
    )

    # ── CRS: per-FM variant plots ─────────────────────────────────────────────
    for fm_model in FM_MODELS:
        print(f"CRS {fm_model} variants plot...")
        variant_data = collect_crs_variant_data(fm_model)
        base_label = _MODEL_LABELS[fm_model]
        x_order_variants = [
            base_label,
            base_label + " (Combined)",
            base_label + " (DK Shifted)",
            base_label + " (FM Shifted)",
            base_label + " (All Shifted)",
        ]
        plot_auc_dots(
            variant_data,
            x_order=x_order_variants,
            title=f"CRS — {base_label}: raw vs. shifted variants (nested CV)",
            output_path=output_dir / f"crs_{fm_model}_variants.png",
        )

    # ── Other targets: overview plots ─────────────────────────────────────────
    for task in _OTHER_TASKS:
        metric = task["metric"]
        metric_path = task["metric_path"]
        title_str = _METRIC_TITLE.get(metric, metric)
        slug = metric.replace("/", "__")

        print(f"{title_str} main-models plot...")
        task_data = collect_main_model_data(metric_path)
        plot_auc_dots(
            task_data,
            x_order=x_order_main,
            title=f"{title_str} — AUC by model (MLP embedding, nested CV)",
            output_path=output_dir / f"{slug}_main_models.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
