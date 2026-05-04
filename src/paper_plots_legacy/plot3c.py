#!/usr/bin/env python3
"""DK-FM embedding alignment (MKNN k=20) per Foundation Model.

Compact version of ``EMBEDDING_COMPARISON/plots/alignment_overview.png``:
one error-bar point per FM, using the same colors and x-axis order as
``plot3_combined.png``.

Reads ``EMBEDDING_COMPARISON/component5_mknn/mknn_summary.json`` and writes
``PLOTS/plot3_geo.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"


BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
# L2-normalised MKNN results (preferred — matches PHR paper cosine similarity).
MKNN_DIR_L2 = BASE / "EMBEDDING_COMPARISON" / "component5_mknn_l2"
# Legacy raw-Euclidean results (fallback when L2 run not yet available).
MKNN_DIR = BASE / "EMBEDDING_COMPARISON" / "component5_mknn"
MKNN_SUMMARY = MKNN_DIR / "mknn_summary.json"
OUT = BASE / "PLOTS"

# Default column order — matches plot3_combined.png.
MODEL_ORDER = ["BIOT", "TOTEM", "LaBram", "NeuroLM", "CbraMod"]
# EEGPT variant: TOTEM is dropped, EEGPT inserted in its slot.
MODEL_ORDER_EEGPT = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CbraMod"]
MODEL_DISPLAY = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBraM",
    "EEGPT": "EEGPT",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
}
FM_COLORS = {
    "BIOT": "#1f77b4",
    "TOTEM": "#d62728",
    "LaBram": "#2ca02c",
    "EEGPT": "#8c564b",
    "NeuroLM": "#9467bd",
    "CbraMod": "#ff7f0e",
}

# embedding_comparison.py writes JSON with "CBraMod" (capital B);
# plot3 convention uses "CbraMod". All other names match.
_JSON_KEY = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBram",
    "EEGPT": "EEGPT",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
}

K_VALUE = "20"


def _extract_k_entry(entry: dict) -> tuple[float, float, float, float] | None:
    """Return ``(mean, std, ci_lo, ci_hi)`` for a single k-entry, or None."""
    mean = entry.get("mean")
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return None
    lo = entry.get("ci_lower", float("nan"))
    hi = entry.get("ci_upper", float("nan"))
    std = entry.get("boot_std")
    if std is None or (isinstance(std, float) and np.isnan(std)):
        std = (hi - lo) / 3.92 if not (np.isnan(lo) or np.isnan(hi)) else 0.0
    return float(mean), float(max(0.0, std)), float(lo), float(hi)


def _load_mknn_per_model(model: str) -> tuple[float, float, float, float] | None:
    """Load MKNN k=K_VALUE result for one model.

    Priority: L2-normalised results (component5_mknn_l2/) → per-model legacy
    JSON → aggregated legacy summary.
    """
    # 1. L2-normalised run (preferred)
    per_model_l2 = MKNN_DIR_L2 / f"mknn_results_{_JSON_KEY[model]}.json"
    if per_model_l2.is_file():
        with per_model_l2.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return _extract_k_entry(d.get("k_results", {}).get(K_VALUE, {}))

    # 2. Legacy per-model file
    per_model = MKNN_DIR / f"mknn_results_{_JSON_KEY[model]}.json"
    if per_model.is_file():
        with per_model.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return _extract_k_entry(d.get("k_results", {}).get(K_VALUE, {}))

    # 3. Legacy aggregated summary
    if MKNN_SUMMARY.is_file():
        with MKNN_SUMMARY.open("r", encoding="utf-8") as f:
            d = json.load(f)
        entry = d.get(_JSON_KEY[model], {}).get("k_results", {}).get(K_VALUE, {})
        return _extract_k_entry(entry)

    return None


def _load_mknn(
    model_order: list[str] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Return ``{model: (mean, std, ci_lo, ci_hi)}`` for models in ``model_order``."""
    order = model_order if model_order is not None else MODEL_ORDER
    out: dict[str, tuple[float, float, float, float]] = {}
    for m in order:
        res = _load_mknn_per_model(m)
        if res is not None:
            out[m] = res
        else:
            print(f"[plot3_geo]   {m}: no MKNN data — skipping")
    return out


def _load_mknn_baseline_per_model(model: str) -> tuple[float, float, float, float] | None:
    """Load shuffled-DK baseline (k=K_VALUE) for one model, or None if file missing.

    Prefers L2-normalised baseline; falls back to legacy.
    """
    for search_dir in (MKNN_DIR_L2, MKNN_DIR):
        path = search_dir / f"mknn_baseline_{_JSON_KEY[model]}.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            return _extract_k_entry(d.get("k_results", {}).get(K_VALUE, {}))
    return None


def _load_mknn_baselines(
    model_order: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    """Return {model: (mean, std, ci_lo, ci_hi)} for the shuffled-DK baseline."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for m in model_order:
        res = _load_mknn_baseline_per_model(m)
        if res is not None:
            out[m] = res
    return out


def _write_table_csv(
    scores: dict[str, tuple[float, float, float, float]],
    model_order: list[str],
    out_path: Path,
) -> None:
    """Write a CSV table with exact MKNN results for ``model_order``."""
    lines = ["model,k,mean,std,ci_lower,ci_upper"]
    for m in model_order:
        if m not in scores:
            continue
        mean, std, lo, hi = scores[m]
        lines.append(
            f"{MODEL_DISPLAY[m]},{K_VALUE},{mean:.6f},{std:.6f},{lo:.6f},{hi:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[plot3_geo] wrote {out_path}")


def plot_geo(
    scores: dict[str, tuple[float, float, float, float]],
    out_path: Path,
    model_order: list[str] | None = None,
    with_table: bool = False,
    baselines: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    order = model_order if model_order is not None else MODEL_ORDER
    models_present = [m for m in order if m in scores]
    n = len(models_present)
    if n == 0:
        print("[plot3_geo] no models with MKNN data — skipping")
        return

    xticks = [MODEL_DISPLAY[m] for m in models_present]
    x_pos = np.arange(n)

    if with_table:
        fig, (ax, ax_tbl) = plt.subplots(
            2, 1,
            figsize=(max(4.5, 0.8 * n + 1.5), 5.2),
            gridspec_kw=dict(height_ratios=[3.2, 1.6]),
        )
    else:
        fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * n + 1.5), 3.2))
        ax_tbl = None

    for xi, m in enumerate(models_present):
        mean, std, _lo, _hi = scores[m]
        color = FM_COLORS[m]
        ax.errorbar(
            [x_pos[xi]], [mean], yerr=[std],
            fmt="s", color=color, ecolor=color,
            markersize=7, elinewidth=1.8, capsize=5, capthick=1.6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )
        if baselines and m in baselines:
            b_mean, b_std, _blo, _bhi = baselines[m]
            ax.errorbar(
                [x_pos[xi] + 0.25], [b_mean], yerr=[b_std],
                fmt="o", color="gray", ecolor="gray",
                markersize=6, elinewidth=1.4, capsize=4, capthick=1.2,
                markerfacecolor="none", markeredgewidth=1.4, zorder=2,
            )

    if baselines:
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
                       markersize=7, linestyle="none", label="Observed"),
                Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
                       markersize=6, markeredgewidth=1.4, linestyle="none",
                       label="Shuffled-DK baseline"),
            ],
            loc="upper right",
            fontsize=9,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticks, fontsize=11)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Foundation Model", fontsize=11)
    ax.set_ylabel("Alignment score (mean ± std)", fontsize=11)
    ax.set_title(f"DK–FM embedding alignment (MKNN k={K_VALUE})", fontsize=11, pad=6)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)

    if ax_tbl is not None:
        ax_tbl.axis("off")
        has_bl = bool(baselines)
        col_labels = ["Model", "mean", "± std", "95% CI"]
        if has_bl:
            col_labels += ["baseline mean", "± std"]
        cell_text = []
        for m in models_present:
            mean, std, lo, hi = scores[m]
            row = [MODEL_DISPLAY[m], f"{mean:.3f}", f"{std:.3f}", f"[{lo:.3f}, {hi:.3f}]"]
            if has_bl and m in baselines:
                bm, bs, _blo, _bhi = baselines[m]
                row += [f"{bm:.3f}", f"{bs:.3f}"]
            elif has_bl:
                row += ["—", "—"]
            cell_text.append(row)
        tbl = ax_tbl.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.3)
        # Color the Model cell with its FM color, header row bolded.
        for c in range(len(col_labels)):
            tbl[(0, c)].set_text_props(weight="bold")
        for r, m in enumerate(models_present, start=1):
            tbl[(r, 0)].set_text_props(color=FM_COLORS[m], weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3_geo] wrote {out_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Union of both orders so we load EEGPT too when available.
    all_models = list(dict.fromkeys(MODEL_ORDER + MODEL_ORDER_EEGPT))
    scores = _load_mknn(all_models)
    baselines = _load_mknn_baselines(all_models)
    for m, (mean, std, lo, hi) in scores.items():
        print(
            f"[plot3_geo]   {m}: MKNN k={K_VALUE} = {mean:.3f} ± {std:.3f} "
            f"[{lo:.3f}, {hi:.3f}]"
        )
    for m, (bm, bs, blo, bhi) in baselines.items():
        print(
            f"[plot3_geo]   {m}: baseline k={K_VALUE} = {bm:.3f} ± {bs:.3f} "
            f"[{blo:.3f}, {bhi:.3f}]"
        )
    if not baselines:
        print("[plot3_geo] no baseline files found — run embedding_comparison.py to generate them")

    # Original figure (unchanged column set).
    plot_geo(scores, OUT / "plot3_geo.png", model_order=MODEL_ORDER, baselines=baselines)

    # EEGPT variant — figure with inline table + separate CSV.
    plot_geo(
        scores, OUT / "plot3_geo_eegpt.png",
        model_order=MODEL_ORDER_EEGPT, with_table=True, baselines=baselines,
    )
    _write_table_csv(scores, MODEL_ORDER_EEGPT, OUT / "plot3_geo_eegpt_table.csv")


if __name__ == "__main__":
    main()
