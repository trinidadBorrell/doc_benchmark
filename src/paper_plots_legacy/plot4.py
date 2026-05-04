#!/usr/bin/env python3
"""Random-Forest CRS overview: Baseline + FM + FM (Combined) + FM (res).

One-panel version of ``crs_combined_overview.png`` restricted to the
random-forest classifier and extended with Lao's dimension-wise
residualization as a third per-FM variant labelled ``(res)``.

Color grouping replaces the alternating gray background: each FM's
three ticks (raw, Combined, res) share the FM color from
``plot3_auc_layers_boxes_points.py``. Baseline is drawn in gray at the left.

Data sources (all for ``crs`` target, classifier = ``random_forest``):

  - Baseline
        MARKER_BASELINE/crs/nested_cv/random_forest/classification_results.json
        -> macro_average.auc_score.{mean,std}
  - {FM} (raw)
        {FM}/doc_patients/MLP_EMBEDDING/crs/nested_cv/random_forest/
        classification_results.json
        (BIOT: doubly-nested MLP_EMBEDDING/BIOT/doc_patients/MLP_EMBEDDING/...)
  - {FM} (Combined)
        {FM}/doc_patients/EMBEDDING_DK_COMBINED/crs/nested_cv/random_forest/
        classification_results.json
  - {FM} (res)  -- Lao's dimension-wise residualization
        RESIDUALIZATION_DIM/crs/results.json
        -> results.{FM}.classifiers.random_forest.{mean_auc, std_auc}

Outputs:
  - data/benchmark_results/new_results/PLOTS/plot4_crs_rf_overview.png
        FM set: BIOT, TOTEM, LaBram, NeuroLM, CBraMod
  - data/benchmark_results/new_results/PLOTS/plot4_crs_rf_overview_eegpt.png
        Same layout, but TOTEM is replaced by EEGPT.
        Note: EEGPT has no ``MLP_EMBEDDING/crs/...`` result; its "raw"
        variant is read from ``EMBEDDING_FM_PCA_ONLY/crs/...`` instead
        (the FM-only embedding path used for EEGPT).
  - data/benchmark_results/new_results/PLOTS/plot4_crs_rf_overview_eegpt_pca.png
        FM set: BIOT, EEGPT, LaBram, NeuroLM, CBraMod, but every per-FM
        variant uses the PCA-reduced embedding:
            (FM-PCA)            EMBEDDING_FM_PCA_ONLY/...
            (Combined FM-PCA)   EMBEDDING_DK_COMBINED_FM_PCA/...
            (res FM-PCA)        RESIDUALIZATION_DIM/pca27/crs/results.json
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
OUT = BASE / "PLOTS" / "plot4_crs_rf_overview.png"
OUT_EEGPT = BASE / "PLOTS" / "plot4_crs_rf_overview_eegpt.png"
OUT_EEGPT_PCA = BASE / "PLOTS" / "plot4_crs_rf_overview_eegpt_pca.png"

CLASSIFIER = "random_forest"
FM_MODELS = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CBraMod"]
FM_MODELS_EEGPT = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CBraMod"]

DK_COLOR = "#555555"
FM_COLORS = {
    "TOTEM":   "#d62728",
    "CBraMod": "#ff7f0e",
    "LaBram":  "#2ca02c",
    "NeuroLM": "#9467bd",
    "BIOT":    "#1f77b4",
    "EEGPT":   "#8c564b",
}


def _macro_auc(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    auc = d.get("macro_average", {}).get("auc_score", {})
    m = auc.get("mean")
    s = auc.get("std")
    if m is None:
        return None
    return float(m), float(s) if s is not None else 0.0


def load_baseline() -> tuple[float, float] | None:
    path = (
        BASE / "MARKER_BASELINE" / "crs" / "nested_cv"
        / CLASSIFIER / "classification_results.json"
    )
    return _macro_auc(path)


def load_fm_embedding(model: str, embedding_kind: str) -> tuple[float, float] | None:
    # EEGPT's dedicated non-PCA RF run (jobs/mlp_classifier_eegpt_no_pca.submit)
    # lives under MLP-CLASSIFIER/EEGPT/doc_patients/MLP_EMBEDDING/. Before that
    # run existed we fell back to EMBEDDING_FM_PCA_ONLY; prefer the real
    # non-PCA numbers now so this stays in sync with plot4_targets.py.
    if model == "EEGPT" and embedding_kind == "MLP_EMBEDDING":
        eegpt_primary = (
            BASE / "MLP-CLASSIFIER" / "EEGPT" / "doc_patients" / "MLP_EMBEDDING"
            / "crs" / "nested_cv" / CLASSIFIER / "classification_results.json"
        )
        result = _macro_auc(eegpt_primary)
        if result is not None:
            return result
        embedding_kind = "EMBEDDING_FM_PCA_ONLY"
    primary = (
        BASE / model / "doc_patients" / embedding_kind / "crs"
        / "nested_cv" / CLASSIFIER / "classification_results.json"
    )
    result = _macro_auc(primary)
    if result is not None:
        return result
    # BIOT MLP_EMBEDDING lives at a doubly-nested path.
    fallback = (
        BASE / model / "doc_patients" / embedding_kind / model
        / "doc_patients" / embedding_kind / "crs"
        / "nested_cv" / CLASSIFIER / "classification_results.json"
    )
    return _macro_auc(fallback)


def load_fm_res(model: str) -> tuple[float, float] | None:
    return _load_res(BASE / "RESIDUALIZATION_DIM" / "crs" / "results.json", model)


def load_fm_res_pca(model: str) -> tuple[float, float] | None:
    return _load_res(
        BASE / "RESIDUALIZATION_DIM" / "pca27" / "crs" / "results.json",
        model,
    )


def _load_res(path: Path, model: str) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    entry = d.get("results", {}).get(model, {}).get("classifiers", {}).get(CLASSIFIER)
    if not entry:
        return None
    m = entry.get("mean_auc")
    s = entry.get("std_auc")
    if m is None:
        return None
    return float(m), float(s) if s is not None else 0.0


def build_rows_pca(fm_models: list[str]) -> list[tuple[str, float, float, str]]:
    """PCA variant of build_rows: every per-FM tick uses PCA paths."""
    rows: list[tuple[str, float, float, str]] = []

    baseline = load_baseline()
    if baseline is not None:
        rows.append(("DK", baseline[0], baseline[1], DK_COLOR))
    else:
        print("[plot4] WARNING: Baseline not found — skipping")

    for model in fm_models:
        color = FM_COLORS[model]
        variants = [
            (f"{model} (FM-PCA)",
                load_fm_embedding(model, "EMBEDDING_FM_PCA_ONLY")),
            (f"{model} (Combined FM-PCA)",
                load_fm_embedding(model, "EMBEDDING_DK_COMBINED_FM_PCA")),
            (f"{model} (res FM-PCA)",
                load_fm_res_pca(model)),
        ]
        for label, res in variants:
            if res is None:
                print(f"[plot4] WARNING: missing {label}")
                continue
            rows.append((label, res[0], res[1], color))

    return rows


def build_rows(fm_models: list[str] = FM_MODELS) -> list[tuple[str, float, float, str]]:
    """Return [(x_label, mean, std, color), ...] in plot order."""
    rows: list[tuple[str, float, float, str]] = []

    baseline = load_baseline()
    if baseline is not None:
        rows.append(("DK", baseline[0], baseline[1], DK_COLOR))
    else:
        print("[plot4] WARNING: Baseline not found — skipping")

    for model in fm_models:
        color = FM_COLORS[model]
        variants = [
            (model,                load_fm_embedding(model, "MLP_EMBEDDING")),
            (f"{model} (Combined)", load_fm_embedding(model, "EMBEDDING_DK_COMBINED")),
            (f"{model} (res)",     load_fm_res(model)),
        ]
        for label, res in variants:
            if res is None:
                print(f"[plot4] WARNING: missing {label}")
                continue
            rows.append((label, res[0], res[1], color))

    return rows


def plot(
    rows: list[tuple[str, float, float, str]],
    out_path: Path,
    title: str = "CRS — Random Forest: Baseline vs. FM vs. FM (Combined) vs. FM (res)",
) -> None:
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(rows) + 2), 5.5))

    # Alternating gray/white bands per model group (group = contiguous run
    # of same color in the color sequence: Baseline alone, then each FM's
    # raw/Combined/res triple).
    gi = 0
    run_start = 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(run_start - 0.5, i - 0.5,
                           color="gray", alpha=0.08, zorder=0)
            gi += 1
            run_start = i

    chance = ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                        alpha=0.7, zorder=1, label="chance")

    # One errorbar call per point so per-tick color is easy.
    for xi, m, s, c in zip(x, means, stds, colors):
        ax.errorbar(
            [xi], [m], yerr=[s],
            fmt="D", color=c, capsize=4, markersize=8,
            linewidth=1.4, ecolor=c,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC (mean ± std, random forest)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    ax.legend(handles=[chance], loc="lower right", fontsize="small",
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4] wrote {out_path}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    default_title = (
        "CRS — Random Forest: Baseline vs. FM vs. FM (Combined) vs. FM (res)"
    )
    pca_title = (
        "CRS — Random Forest: Baseline vs. FM-PCA vs. Combined FM-PCA vs. res FM-PCA"
    )

    runs = [
        (FM_MODELS,       OUT,        build_rows,     default_title),
        (FM_MODELS_EEGPT, OUT_EEGPT,  build_rows,     default_title),
        (FM_MODELS_EEGPT, OUT_EEGPT_PCA, build_rows_pca, pca_title),
    ]
    for fm_models, out_path, builder, title in runs:
        print(f"[plot4] === {out_path.name} ({', '.join(fm_models)}) ===")
        rows = builder(fm_models)
        for label, m, s, _c in rows:
            print(f"[plot4]   {label:28s} AUC={m:.3f} ± {s:.3f}")
        plot(rows, out_path, title=title)


if __name__ == "__main__":
    main()
