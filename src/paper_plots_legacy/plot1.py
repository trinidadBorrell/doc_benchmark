#!/usr/bin/env python3
"""Best-classifier AUC summary across 7 models x 8 target conditions.

For each FM × target combination the classifier with the highest mean AUC
(across random_forest, svm, mlp, xgboost, kernel_ridge) is chosen
automatically. A markdown report is saved alongside the plots documenting
which classifier was selected for each combination.

Produces four figures in data/benchmark_results/new_results/PLOTS/:

  - plot1_stacked.png             (MLP_EMBEDDING: 2 rows x 1 col)
  - plot1_grid.png                (MLP_EMBEDDING: 2 rows x 4 cols)
  - plot1_stacked_pca.png         (EMBEDDING_FM_PCA_ONLY, 27-dim PCA)
  - plot1_grid_pca.png            (EMBEDDING_FM_PCA_ONLY, 27-dim PCA)
  - classifier_choices_plot1.md
  - classifier_choices_plot1_pca.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as _stats
import os


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"


BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
OUT = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]

MODEL_ORDER = [
    "DK",
    "BIOT",
    "LaBraM",
    "EEGPT",
    "NeuroLM",
    "CBraMod",
]

MODEL_PATHS_MLP: dict[str, tuple[str, str | None]] = {
    "DK": ("MARKER_BASELINE", None),
    "BIOT": ("BIOT", "doc_patients/MLP_EMBEDDING"),
    "LaBraM": ("LaBram", "doc_patients/MLP_EMBEDDING"),
    "EEGPT": ("EEGPT", "doc_patients/MLP_EMBEDDING"),
    "NeuroLM": ("NeuroLM", "doc_patients/MLP_EMBEDDING"),
    "CBraMod": ("CBraMod", "doc_patients/MLP_EMBEDDING"),
}

# EMBEDDING_FM_PCA_ONLY paths (27-dim PCA of FM embeddings).
MODEL_PATHS_PCA: dict[str, tuple[str, str | None]] = {
    "DK": ("MARKER_BASELINE", None),
    "BIOT": ("BIOT", "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "LaBraM": ("LaBram", "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "EEGPT": ("EEGPT", "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "NeuroLM": ("NeuroLM", "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "CBraMod": ("CBraMod", "doc_patients/EMBEDDING_FM_PCA_ONLY"),
}

ROW1 = [
    ("crs", "CRS Diagnostic"),
    ("cs_6m", "6m Improved"),
    ("cs_1y", "1y Improved"),
    ("cs_2y", "2y Improved"),
]

ROW2 = [
    ("etiology/vs_only", "Delay (VS)"),
    ("etiology/mcs_only", "Delay (MCS)"),
    ("etiology_code/vs_only", "Etiology (VS)"),
    ("etiology_code/mcs_only", "Etiology (MCS)"),
]

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


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
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    auc_mean = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "mean"))
    auc_std = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "std"))

    if auc_mean is None:
        auc_mean = _to_float_or_none(data.get("auc_mean"))
    if auc_std is None:
        auc_std = _to_float_or_none(data.get("auc_std"))
    if auc_mean is None:
        auc_mean = _to_float_or_none(data.get("test_auc_score"))

    return {"mean": auc_mean, "std": auc_std if auc_std is not None else 0.0}


def _candidate_paths(
    model_display: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
    classifier: str,
) -> list[Path]:
    on_disk, prefix = model_paths[model_display]
    model_root = BASE / on_disk if prefix is None else BASE / on_disk / prefix

    metric_parts = Path(metric_path).parts
    candidates: list[Path] = []

    # For cs_* targets, read the IMPROVED-vs-NON_IMPROVED transition run
    # (`binary_improvement/`). This matches plot4_targets.py and gives a
    # consistent task across all cs_{6m,1y,2y} columns.
    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        mn = metric_parts[0]
        candidates.append(
            model_root / mn / "binary_improvement" / "nested_cv" / classifier / "classification_results.json"
        )

    candidates.append(
        model_root / metric_path / "nested_cv" / classifier / "classification_results.json"
    )

    if len(metric_parts) == 1 and metric_parts[0] not in {"cs_1y", "cs_2y", "cs_6m"}:
        candidates.append(
            model_root / metric_parts[0] / "binary" / "nested_cv" / classifier / "classification_results.json"
        )

    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in candidates:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _load_repeated_best_clf(
    model_display: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> dict[str, float | None] | None:
    """Try to load from nested_cv_repeated/best_clf/ (in-fold classifier selection).

    Returns None if the file does not exist or contains no valid AUC.
    Falls back to the per-classifier post-hoc selection path.
    """
    on_disk, prefix = model_paths[model_display]
    model_root = BASE / on_disk if prefix is None else BASE / on_disk / prefix

    metric_parts = Path(metric_path).parts
    candidates: list[Path] = []

    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        mn = metric_parts[0]
        candidates.append(
            model_root / mn / "binary_improvement"
            / "nested_cv_repeated" / "best_clf" / "classification_results.json"
        )

    candidates.append(
        model_root / metric_path
        / "nested_cv_repeated" / "best_clf" / "classification_results.json"
    )

    for p in candidates:
        if p.is_file():
            scores = _read_json_scores(p)
            if scores.get("mean") is not None:
                return scores
    return None


def _select_best_classifier(
    model_display: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> tuple[str | None, dict | None]:
    """Return (classifier_name, scores) for the clf with the highest mean AUC.

    Prefers nested_cv_repeated/best_clf/ (in-fold selection) when available;
    falls back to post-hoc selection from per-classifier nested_cv/ directories.
    """
    # Priority: in-fold selection from repeated CV.
    repeated_scores = _load_repeated_best_clf(model_display, metric_path, model_paths)
    if repeated_scores is not None:
        return "best_clf_repeated", repeated_scores

    best_clf, best_scores = None, None
    for clf in CLASSIFIERS_TO_TRY:
        candidates = _candidate_paths(model_display, metric_path, model_paths, clf)
        scores = _resolve_scores(candidates)
        if scores and scores.get("mean") is not None:
            if best_scores is None or scores["mean"] > best_scores["mean"]:
                best_clf, best_scores = clf, scores
    return best_clf, best_scores


def _resolve_scores(candidates: list[Path]) -> dict[str, float | None] | None:
    # Some candidate layouts (notably `<target>/multiclass/...`) exist on disk
    # but carry `auc_score.mean = None` because multiclass AUC is undefined for
    # the fold configuration. Skip those and keep looking so we don't shadow a
    # valid sibling (e.g. `<target>/binary/...`).
    for path in candidates:
        if path.is_file():
            scores = _read_json_scores(path)
            if scores.get("mean") is not None:
                return scores
    return None


# ── Statistical annotation helpers ────────────────────────────────────────────

def _extract_fold_aucs(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception:
        return None
    pf = d.get("macro_average", {}).get("per_fold_metrics")
    if pf:
        aucs = [
            fold["auc_score"] for fold in pf
            if fold.get("auc_score") is not None
            and not (isinstance(fold["auc_score"], float) and np.isnan(fold["auc_score"]))
        ]
        if aucs:
            return aucs
    return None


def _load_fold_aucs(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
    classifier: str,
) -> list[float] | None:
    for path in _candidate_paths(model, metric_path, model_paths, classifier):
        result = _extract_fold_aucs(path)
        if result is not None:
            return result
    return None


def _p_fm_below_dk(fm_aucs: list[float], dk_aucs: list[float]) -> float:
    """One-tailed Welch p for H1: mean(FM) < mean(DK)."""
    if len(fm_aucs) < 2 or len(dk_aucs) < 2:
        return 1.0
    t, p_two = _stats.ttest_ind(fm_aucs, dk_aucs, equal_var=False)
    return float(p_two / 2 if t < 0 else 1.0 - p_two / 2)


def _below_dk_mask(
    models_present: list[str],
    conditions: list[tuple[str, str]],
    model_paths: dict[str, tuple[str, str | None]],
    clf_choices: dict[str, dict[str, str]],
    alpha: float = 0.05,
) -> np.ndarray:
    """Bool matrix (n_models, n_conditions): True where FM is sig. below DK."""
    n_m, n_c = len(models_present), len(conditions)
    mask = np.zeros((n_m, n_c), dtype=bool)
    dk_cache = {
        mp: _load_fold_aucs("DK", mp, model_paths, clf_choices.get("DK", {}).get(mp, "random_forest"))
        for mp, _ in conditions
    }
    for i, model in enumerate(models_present):
        if model == "DK":
            continue
        for j, (mp, _) in enumerate(conditions):
            clf = clf_choices.get(model, {}).get(mp, "random_forest")
            fm = _load_fold_aucs(model, mp, model_paths, clf)
            dk = dk_cache[mp]
            if fm and dk and _p_fm_below_dk(fm, dk) < alpha:
                mask[i, j] = True
    return mask


def load_all(
    conditions: list[tuple[str, str]],
    model_paths: dict[str, tuple[str, str | None]],
    tag: str,
) -> tuple[dict[str, dict[str, dict]], list[str], dict[str, dict[str, str]]]:
    """Return ({model: {metric_path: scores}}, models_present, clf_choices)."""
    scores: dict[str, dict[str, dict]] = {m: {} for m in MODEL_ORDER}
    clf_choices: dict[str, dict[str, str]] = {m: {} for m in MODEL_ORDER}
    for model in MODEL_ORDER:
        for metric_path, _label in conditions:
            clf, found = _select_best_classifier(model, metric_path, model_paths)
            if found is None or found.get("mean") is None:
                print(f"[plot1:{tag}] missing: {model} / {metric_path}")
                continue
            scores[model][metric_path] = found
            clf_choices[model][metric_path] = clf
    models_present = [m for m in MODEL_ORDER if scores[m]]
    dropped = [m for m in MODEL_ORDER if not scores[m]]
    if dropped:
        print(f"[plot1:{tag}] dropping models with no data: {dropped}")
    return scores, models_present, clf_choices


def build_matrix(
    models_present: list[str],
    conditions: list[tuple[str, str]],
    scores: dict[str, dict[str, dict]],
) -> tuple[np.ndarray, np.ndarray]:
    n_m, n_c = len(models_present), len(conditions)
    mean = np.full((n_m, n_c), np.nan)
    std = np.full((n_m, n_c), np.nan)
    for i, model in enumerate(models_present):
        for j, (metric_path, _label) in enumerate(conditions):
            entry = scores[model].get(metric_path)
            if entry and entry.get("mean") is not None:
                mean[i, j] = entry["mean"]
                std[i, j] = entry.get("std") or 0.0
    return mean, std


def _errorbar_row(ax, models, mean, std, conditions):
    n_c = len(conditions)
    x = np.arange(len(models))
    offsets = np.linspace(-0.22, 0.22, n_c) if n_c > 1 else np.array([0.0])
    for j, (_metric_path, label) in enumerate(conditions):
        ax.errorbar(
            x + offsets[j],
            mean[:, j],
            yerr=std[:, j],
            fmt="o",
            color=PALETTE[j % len(PALETTE)],
            capsize=4,
            markersize=9,
            linewidth=1.5,
            label=label,
        )
 #   ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC", fontsize = 15)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize="small", framealpha=0.9)


def plot_stacked(
    models: list[str],
    mean1: np.ndarray,
    std1: np.ndarray,
    mean2: np.ndarray,
    std2: np.ndarray,
    out_path: Path,
    title: str = "Classification comparison between DK and FM",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    _errorbar_row(axes[0], models, mean1, std1, ROW1)
    _errorbar_row(axes[1], models, mean2, std2, ROW2)
  #  axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot1] wrote {out_path}")


def plot_stacked_annotated(
    models: list[str],
    mean1: np.ndarray, std1: np.ndarray,
    mean2: np.ndarray, std2: np.ndarray,
    mask1: np.ndarray, mask2: np.ndarray,
    out_path: Path,
) -> None:
    """Like plot_stacked but adds a red * above FM dots that are sig. below DK."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    _errorbar_row(axes[0], models, mean1, std1, ROW1)
    _errorbar_row(axes[1], models, mean2, std2, ROW2)

    x = np.arange(len(models))
    for ax, mean, std, mask, conditions in [
        (axes[0], mean1, std1, mask1, ROW1),
        (axes[1], mean2, std2, mask2, ROW2),
    ]:
        n_c = len(conditions)
        offsets = np.linspace(-0.22, 0.22, n_c) if n_c > 1 else np.array([0.0])
        for i in range(len(models)):
            for j in range(n_c):
                if mask[i, j] and not np.isnan(mean[i, j]):
                    s = std[i, j] if not np.isnan(std[i, j]) else 0.0
                    ax.text(
                        x[i] + offsets[j], mean[i, j] + s + 0.03,
                        "*", color="red", ha="center", va="bottom",
                        fontsize=14, fontweight="bold",
                    )

    from matplotlib.lines import Line2D as _L2D
    fig.legend(
        handles=[_L2D([], [], marker="$*$", color="red", markersize=11, linestyle="",
                      label="FM < DK  (p < 0.05)")],
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01),
        bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot1] wrote {out_path}")


def plot_grid(
    models: list[str],
    mean1: np.ndarray,
    std1: np.ndarray,
    mean2: np.ndarray,
    std2: np.ndarray,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    x = np.arange(len(models))
    for j, (_metric_path, label) in enumerate(ROW1):
        ax = axes[0, j]
        ax.errorbar(x, mean1[:, j], yerr=std1[:, j], fmt="o",
                    color="#1f77b4", capsize=4, markersize=9, linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
      #  ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if j == 0:
            ax.set_ylabel("AUC", fontsize = 15)
    for j, (_metric_path, label) in enumerate(ROW2):
        ax = axes[1, j]
        ax.errorbar(x, mean2[:, j], yerr=std2[:, j], fmt="o",
                    color="#1f77b4", capsize=3, markersize=6, linewidth=1.2)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    #    ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if j == 0:
            ax.set_ylabel("AUC", fontsize = 15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower left", fontsize="medium",
                   bbox_to_anchor=(0.99, 0.99), framealpha=0.9)
 #   if suptitle:
 #       fig.suptitle(suptitle, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot1] wrote {out_path}")


def _save_classifier_choices_md(
    clf_choices: dict[str, dict[str, str]],
    scores: dict[str, dict[str, dict]],
    conditions: list[tuple[str, str]],
    pca: bool,
    out_path: Path,
    model_paths: dict[str, tuple[str, str | None]] | None = None,
) -> None:
    lines = [
        f"# Best Classifier per FM × Target (PCA={'yes' if pca else 'no'})\n",
        "| FM | Target | Best Classifier | Mean AUC | Found | Missing |",
        "|----|--------|-----------------|----------|-------|---------|",
    ]
    for model, target_map in clf_choices.items():
        for metric_path, clf in target_map.items():
            label = next((lbl for mp, lbl in conditions if mp == metric_path), metric_path)
            mean = scores.get(model, {}).get(metric_path, {}).get("mean")
            auc_str = f"{mean:.4f}" if mean is not None else "N/A"
            found, missing = [], []
            if model_paths is not None:
                for c in CLASSIFIERS_TO_TRY:
                    result = _resolve_scores(
                        _candidate_paths(model, metric_path, model_paths, c)
                    )
                    (found if result and result.get("mean") is not None else missing).append(c)
            found_str  = ", ".join(found)  or "—"
            missing_str = ", ".join(missing) or "—"
            lines.append(
                f"| {model} | {label} | {clf} | {auc_str} | {found_str} | {missing_str} |"
            )

    # Summary section: entries with no results at all
    if model_paths is not None:
        no_result_rows = []
        for model in MODEL_ORDER:
            for metric_path, label in conditions:
                if model not in clf_choices or metric_path not in clf_choices[model]:
                    no_result_rows.append((model, label))
        if no_result_rows:
            lines += [
                "",
                "## Completely missing (no classifier has results)",
                "| FM | Target |",
                "|----|--------|",
            ]
            for model, label in no_result_rows:
                lines.append(f"| {model} | {label} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[plot1] wrote {out_path}")


def _run_variant(
    tag: str,
    model_paths: dict[str, tuple[str, str | None]],
    stacked_out: Path,
    grid_out: Path,
    stacked_title: str,
    grid_suptitle: str | None = None,
    pca: bool = False,
) -> None:
    scores, models_present, clf_choices = load_all(ROW1 + ROW2, model_paths, tag)
    print(f"[plot1:{tag}] models present: {models_present}")

    mean1, std1 = build_matrix(models_present, ROW1, scores)
    mean2, std2 = build_matrix(models_present, ROW2, scores)

    plot_stacked(models_present, mean1, std1, mean2, std2, stacked_out, title=stacked_title)
    plot_grid(models_present, mean1, std1, mean2, std2, grid_out, suptitle=grid_suptitle)

    mask1 = _below_dk_mask(models_present, ROW1, model_paths, clf_choices)
    mask2 = _below_dk_mask(models_present, ROW2, model_paths, clf_choices)
    ann_out = stacked_out.with_name(stacked_out.stem + "_annotated.png")
    plot_stacked_annotated(
        models_present, mean1, std1, mean2, std2, mask1, mask2, ann_out
    )

    md_name = f"classifier_choices_plot1{'_pca' if pca else ''}.md"
    _save_classifier_choices_md(
        clf_choices, scores, ROW1 + ROW2, pca, OUT / md_name,
        model_paths=model_paths,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    _run_variant(
        tag="mlp",
        model_paths=MODEL_PATHS_MLP,
        stacked_out=OUT / "plot1_stacked.png",
        grid_out=OUT / "plot1_grid.png",
        stacked_title="Classification comparison between DK and FM",
        pca=False,
    )

    _run_variant(
        tag="pca",
        model_paths=MODEL_PATHS_PCA,
        stacked_out=OUT / "plot1_stacked_pca.png",
        grid_out=OUT / "plot1_grid_pca.png",
        stacked_title="Classification comparison between DK and FM (PCA 27-dim)",
        grid_suptitle="FM embeddings — PCA 27-dim",
        pca=True,
    )


if __name__ == "__main__":
    main()
