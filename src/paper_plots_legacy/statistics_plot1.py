#!/usr/bin/env python3
"""Statistical tests on nested-CV AUC scores, mirroring plot1_stacked.png.

For each model × target the best classifier (highest mean AUC across
random_forest, svm, mlp, xgboost, kernel_ridge) is selected | identical
logic to plot1.py | so each row corresponds to exactly one dot in the
stacked plot.

  Test 1 | above chance (0.5):
      One-sample one-tailed t-test on per-fold AUC values: H1: mean > 0.5
      Wilcoxon signed-rank test: H1: median > 0.5

  Test 2 | FM above DK (same condition):
      Independent two-sample one-tailed t-test: H1: FM_mean > DK_mean
      Mann-Whitney U test: H1: FM_median > DK_median
      (blank for DK rows)

  FDR correction (Benjamini-Hochberg) applied across all rows for each
  p-value family.

Results are saved as:
  - statistics_plot1.csv
  - statistics_plot1.html  (colour-coded by significance)

Four annotated plots (mirrors of plot1_stacked_annotated.png):
  - plot1_stacked_annotated_ttest.png
  - plot1_stacked_annotated_ttest_fdr.png
  - plot1_stacked_annotated_wilcoxon.png
  - plot1_stacked_annotated_wilcoxon_fdr.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corrected_ttest import (  # noqa: E402
    corrected_one_tailed,
    corrected_resampled_ttest,
    infer_n_train_test,
    paired_per_fold_aucs,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150

# ── Constants | mirror plot1.py exactly ───────────────────────────────────────

BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
OUT = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]

MODEL_ORDER = ["DK", "BIOT", "LaBraM", "EEGPT", "NeuroLM", "CBraMod"]

MODEL_PATHS_MLP: dict[str, tuple[str, str | None]] = {
    "DK":      ("MARKER_BASELINE", None),
    "BIOT":    ("BIOT",            "doc_patients/MLP_EMBEDDING"),
    "LaBraM":  ("LaBram",          "doc_patients/MLP_EMBEDDING"),
    "EEGPT":   ("EEGPT",           "doc_patients/MLP_EMBEDDING"),
    "NeuroLM": ("NeuroLM",         "doc_patients/MLP_EMBEDDING"),
    "CBraMod": ("CBraMod",         "doc_patients/MLP_EMBEDDING"),
}

MODEL_PATHS_PCA: dict[str, tuple[str, str | None]] = {
    "DK":      ("MARKER_BASELINE", None),
    "BIOT":    ("BIOT",            "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "LaBraM":  ("LaBram",          "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "EEGPT":   ("EEGPT",           "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "NeuroLM": ("NeuroLM",         "doc_patients/EMBEDDING_FM_PCA_ONLY"),
    "CBraMod": ("CBraMod",         "doc_patients/EMBEDDING_FM_PCA_ONLY"),
}

CONDITIONS: list[tuple[str, str]] = [
    ("crs",                    "CRS Diagnostic"),
    ("cs_6m",                  "6m Improved"),
    ("cs_1y",                  "1y Improved"),
    ("cs_2y",                  "2y Improved"),
    ("etiology/vs_only",       "Delay (UWS)"),
    ("etiology/mcs_only",      "Delay (MCS)"),
    ("etiology_code/vs_only",  "Etiology (UWS)"),
    ("etiology_code/mcs_only", "Etiology (MCS)"),
]

ROW1 = CONDITIONS[:4]   # top panel
ROW2 = CONDITIONS[4:]   # bottom panel

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

CHANCE = 0.5
ALPHA = 0.05


# ── Path resolution | mirrors plot1.py _candidate_paths() ─────────────────────

def _candidate_paths(
    model: str,
    metric_path: str,
    classifier: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> list[Path]:
    on_disk, prefix = model_paths[model]
    model_root = BASE / on_disk if prefix is None else BASE / on_disk / prefix

    metric_parts = Path(metric_path).parts
    candidates: list[Path] = []

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


def _select_best_clf(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> str:
    """Return the classifier name with the highest mean AUC for this model/condition.

    Mirrors plot1.py's _select_best_classifier / _resolve_scores: for each
    classifier, tries candidate paths in order and uses the first one where
    mean is not None (skipping files that exist but carry mean=None, e.g.
    multiclass runs).
    """
    best_clf, best_mean = "random_forest", -1.0
    for clf in CLASSIFIERS_TO_TRY:
        for path in _candidate_paths(model, metric_path, clf, model_paths):
            if path.is_file():
                try:
                    scores = _read_json_scores(path)
                    m = scores.get("mean")
                    if m is not None:
                        if m > best_mean:
                            best_clf, best_mean = clf, m
                        break  # found a valid file for this clf | stop trying candidates
                except Exception:
                    pass
    return best_clf


# ── Data loading ───────────────────────────────────────────────────────────────

def _extract_fold_aucs(path: Path) -> list[float] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    per_fold = (
        data.get("macro_average", {}).get("per_fold_metrics")
        or data.get("per_fold_metrics")
    )
    if per_fold:
        aucs = [
            fold["auc_score"] for fold in per_fold
            if fold.get("auc_score") is not None
            and not (isinstance(fold["auc_score"], float) and np.isnan(fold["auc_score"]))
        ]
        if aucs:
            return aucs

    for key in ("macro_average.auc_score.mean", "auc_mean", "test_auc_score"):
        parts = key.split(".")
        cur = data
        for p in parts:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is not None:
            return [float(cur)]
    return None


def load_fold_aucs(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> list[float] | None:
    # Priority: in-fold classifier selection from nested_cv_repeated/best_clf/.
    repeated_path = _classification_results_path(model, metric_path, model_paths)
    if repeated_path is not None:
        result = _extract_fold_aucs(repeated_path)
        if result is not None:
            return result

    clf = _select_best_clf(model, metric_path, model_paths)
    for path in _candidate_paths(model, metric_path, clf, model_paths):
        result = _extract_fold_aucs(path)
        if result is not None:
            return result
    return None


def _read_json_scores(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def _safe(*keys):
        cur = data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        try:
            return float(cur)
        except (TypeError, ValueError):
            return None

    mean = _safe("macro_average", "auc_score", "mean") or _safe("auc_mean") or _safe("test_auc_score")
    std  = _safe("macro_average", "auc_score", "std")  or _safe("auc_std") or 0.0
    return {"mean": mean, "std": std if std is not None else 0.0}


def load_mean_std(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> dict | None:
    repeated_path = _classification_results_path(model, metric_path, model_paths)
    if repeated_path is not None:
        scores = _read_json_scores(repeated_path)
        if scores.get("mean") is not None:
            return scores
    clf = _select_best_clf(model, metric_path, model_paths)
    for path in _candidate_paths(model, metric_path, clf, model_paths):
        if path.is_file():
            scores = _read_json_scores(path)
            if scores.get("mean") is not None:
                return scores
    return None


def build_plot_matrix(
    models_present: list[str],
    conditions: list[tuple[str, str]],
    model_paths: dict[str, tuple[str, str | None]],
) -> tuple[np.ndarray, np.ndarray]:
    n_m, n_c = len(models_present), len(conditions)
    mean = np.full((n_m, n_c), np.nan)
    std  = np.full((n_m, n_c), np.nan)
    for i, model in enumerate(models_present):
        for j, (mp, _) in enumerate(conditions):
            entry = load_mean_std(model, mp, model_paths)
            if entry and entry.get("mean") is not None:
                mean[i, j] = entry["mean"]
                std[i, j]  = entry.get("std") or 0.0
    return mean, std


# ── Statistical tests ──────────────────────────────────────────────────────────

def _stars(p: float | None) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    if p < 0.1:
        return "†"
    return "ns"


def test_above_chance(fold_aucs: list[float]) -> tuple[float, float]:
    """One-sample one-tailed t-test: H1: mean > 0.5."""
    if len(fold_aucs) < 2:
        return np.nan, np.nan
    t, p_two = stats.ttest_1samp(fold_aucs, CHANCE)
    p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
    return float(t), float(p_one)


def test_above_chance_wilcoxon(fold_aucs: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test: H1: median > 0.5."""
    if len(fold_aucs) < 2:
        return np.nan, np.nan
    diffs = np.array(fold_aucs) - CHANCE
    if np.all(diffs == 0):
        return np.nan, np.nan
    try:
        w, p = stats.wilcoxon(diffs, alternative="greater")
        return float(w), float(p)
    except ValueError:
        return np.nan, np.nan


def test_fm_vs_dk(
    fm_aucs: list[float], dk_aucs: list[float]
) -> tuple[float, float, float]:
    """Independent t-test returning t, p_above (FM>DK), p_below (FM<DK)."""
    if len(fm_aucs) < 2 or len(dk_aucs) < 2:
        return np.nan, np.nan, np.nan
    t, p_two = stats.ttest_ind(fm_aucs, dk_aucs, equal_var=False)
    p_above = p_two / 2 if t > 0 else 1.0 - p_two / 2
    p_below = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p_above), float(p_below)


def test_fm_vs_dk_wilcoxon(
    fm_aucs: list[float], dk_aucs: list[float]
) -> tuple[float, float, float]:
    """Mann-Whitney U test returning U, p_above (FM>DK), p_below (FM<DK)."""
    if len(fm_aucs) < 1 or len(dk_aucs) < 1:
        return np.nan, np.nan, np.nan
    try:
        u_above, p_above = stats.mannwhitneyu(fm_aucs, dk_aucs, alternative="greater")
        u_below, p_below = stats.mannwhitneyu(fm_aucs, dk_aucs, alternative="less")
        return float(u_above), float(p_above), float(p_below)
    except ValueError:
        return np.nan, np.nan, np.nan


def _p_fm_below_dk_ttest(fm_aucs: list[float], dk_aucs: list[float]) -> float:
    """One-tailed t-test p for H₁: mean(FM) < mean(DK)."""
    if len(fm_aucs) < 2 or len(dk_aucs) < 2:
        return 1.0
    t, p_two = stats.ttest_ind(fm_aucs, dk_aucs, equal_var=False)
    return float(p_two / 2 if t < 0 else 1.0 - p_two / 2)


def _p_fm_below_dk_wilcoxon(fm_aucs: list[float], dk_aucs: list[float]) -> float:
    """One-tailed Mann-Whitney U p for H₁: median(FM) < median(DK)."""
    if len(fm_aucs) < 1 or len(dk_aucs) < 1:
        return 1.0
    try:
        _, p = stats.mannwhitneyu(fm_aucs, dk_aucs, alternative="less")
        return float(p)
    except ValueError:
        return 1.0


# ── Corrected resampled t-test (Nadeau & Bengio 2003) ─────────────────────────

def _classification_results_path(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> Path | None:
    """Locate the best-clf nested_cv_repeated json for (model, target).

    Returns the first existing candidate or ``None``.
    """
    on_disk, prefix = model_paths[model]
    model_root = BASE / on_disk if prefix is None else BASE / on_disk / prefix
    metric_parts = Path(metric_path).parts

    candidates: list[Path] = []
    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        candidates.append(
            model_root / metric_parts[0] / "binary_improvement"
            / "nested_cv_repeated" / "best_clf" / "classification_results.json"
        )
    candidates.append(
        model_root / Path(metric_path)
        / "nested_cv_repeated" / "best_clf" / "classification_results.json"
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _manifest_path_for(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> Path | None:
    on_disk, prefix = model_paths[model]
    model_root = BASE / on_disk if prefix is None else BASE / on_disk / prefix
    metric_parts = Path(metric_path).parts

    candidates: list[Path] = []
    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        candidates.append(
            model_root / metric_parts[0] / "binary_improvement"
            / "nested_cv_repeated" / "cv_split_manifest.json"
        )
    candidates.append(
        model_root / Path(metric_path)
        / "nested_cv_repeated" / "cv_split_manifest.json"
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def test_fm_vs_dk_corrected(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> tuple[float, float, float, float, int, int, int] | None:
    """Paired Nadeau-Bengio corrected resampled t-test for FM vs DK.

    Pairs the per-fold AUCs by ``(repeat, fold)`` between the FM
    ``nested_cv_repeated/best_clf`` results and the DK
    ``MARKER_BASELINE/.../nested_cv_repeated/best_clf`` results.

    Returns ``(t, p_two, p_above, p_below, n_paired, n_train, n_test)`` or
    ``None`` if either side is unavailable / no folds overlap. ``p_two`` is the
    canonical Nadeau-Bengio two-sided p-value; ``p_above``/``p_below`` are the
    derived one-tailed variants kept for backwards compat.
    """
    if model == "DK":
        return None
    fm_path = _classification_results_path(model, metric_path, model_paths)
    dk_path = _classification_results_path("DK", metric_path, model_paths)
    if fm_path is None or dk_path is None:
        return None
    paired = paired_per_fold_aucs(fm_path, dk_path)
    if paired is None:
        return None
    fm_aucs, dk_aucs = paired
    diffs = np.asarray(fm_aucs) - np.asarray(dk_aucs)
    fm_manifest = _manifest_path_for(model, metric_path, model_paths)
    dk_manifest = _manifest_path_for("DK", metric_path, model_paths)
    n_train, n_test = infer_n_train_test(
        fm_path,
        manifest_path=fm_manifest if fm_manifest else dk_manifest,
    )
    t, p_two = corrected_resampled_ttest(diffs, n_train, n_test)
    if np.isnan(t):
        return None
    p_above = p_two / 2 if t > 0 else 1.0 - p_two / 2
    p_below = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p_two), float(p_above), float(p_below), len(diffs), n_train, n_test


def _p_fm_vs_dk_corrected_two_sided(
    model: str,
    metric_path: str,
    model_paths: dict[str, tuple[str, str | None]],
) -> float:
    """Canonical Nadeau-Bengio two-sided p-value for FM vs DK."""
    res = test_fm_vs_dk_corrected(model, metric_path, model_paths)
    return res[1] if res is not None else 1.0



# ── FDR helpers ────────────────────────────────────────────────────────────────

def _apply_fdr_to_series(p_series: pd.Series) -> pd.Series:
    """Apply BH FDR correction to a Series; NaN entries are passed through."""
    valid_idx = p_series.dropna().index
    if len(valid_idx) == 0:
        return p_series.copy()
    p_corrected = stats.false_discovery_control(p_series[valid_idx], method="bh")
    out = p_series.copy()
    out[valid_idx] = p_corrected
    return out


def _below_dk_masks(
    models_present: list[str],
    conditions: list[tuple[str, str]],
    model_paths: dict[str, tuple[str, str | None]],
    alpha: float = ALPHA,
) -> dict[str, np.ndarray]:
    """Compute 4 significance masks for FM significantly below DK (one-tailed).

    Returns dict with bool matrices (n_models × n_conditions):
      'ttest', 'ttest_fdr', 'wilcoxon', 'wilcoxon_fdr'
    """
    n_m, n_c = len(models_present), len(conditions)
    ttest_p    = np.ones((n_m, n_c))
    wilcoxon_p = np.ones((n_m, n_c))
    corrected_p = np.ones((n_m, n_c))
    valid           = np.zeros((n_m, n_c), dtype=bool)
    valid_corrected = np.zeros((n_m, n_c), dtype=bool)

    dk_cache = {mp: load_fold_aucs("DK", mp, model_paths) for mp, _ in conditions}

    for i, model in enumerate(models_present):
        if model == "DK":
            continue
        for j, (mp, _) in enumerate(conditions):
            fm = load_fold_aucs(model, mp, model_paths)
            dk = dk_cache[mp]
            if fm and dk:
                ttest_p[i, j]    = _p_fm_below_dk_ttest(fm, dk)
                wilcoxon_p[i, j] = _p_fm_below_dk_wilcoxon(fm, dk)
                valid[i, j] = True
            res = test_fm_vs_dk_corrected(model, mp, model_paths)
            if res is not None:
                _t, p_two, _pa, _pb, _n, _ntr, _nte = res
                corrected_p[i, j] = p_two
                valid_corrected[i, j] = True

    def _fdr_mask(p_matrix: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        indices = np.argwhere(valid_mask)
        if len(indices) == 0:
            return np.zeros((n_m, n_c), dtype=bool)
        p_vals = [p_matrix[r, c] for r, c in indices]
        p_adj = stats.false_discovery_control(p_vals, method="bh")
        mask = np.zeros((n_m, n_c), dtype=bool)
        for (r, c), p_a in zip(indices, p_adj):
            mask[r, c] = p_a < alpha
        return mask

    return {
        "ttest":         (ttest_p < alpha) & valid,
        "ttest_fdr":     _fdr_mask(ttest_p, valid),
        "wilcoxon":      (wilcoxon_p < alpha) & valid,
        "wilcoxon_fdr":  _fdr_mask(wilcoxon_p, valid),
        "corrected":     (corrected_p < alpha) & valid_corrected,
        "corrected_fdr": _fdr_mask(corrected_p, valid_corrected),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def _errorbar_row(ax, models, mean, std, conditions):
    n_c = len(conditions)
    x = np.arange(len(models))
    offsets = np.linspace(-0.22, 0.22, n_c) if n_c > 1 else np.array([0.0])
    for j, (_mp, label) in enumerate(conditions):
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
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.2, 1)
    ax.set_ylabel("AUC", fontsize=20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize="small", framealpha=0.9)


def _plot_stacked_annotated(
    models: list[str],
    mean1: np.ndarray, std1: np.ndarray,
    mean2: np.ndarray, std2: np.ndarray,
    mask1: np.ndarray, mask2: np.ndarray,
    out_path: Path,
    test_label: str,
) -> None:
    """Two-panel stacked plot with red * where FM is significantly below DK."""
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

    fig.legend(
        handles=[Line2D([], [], marker="$*$", color="red", markersize=11,
                        linestyle="", label=f"p < 0.05 | {test_label}")],
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
    print(f"[statistics_plot1] wrote {out_path}")


# ── Per-target boxplot (FM vs DK, one panel per target) ──────────────────────

# Same colours as plot4_targets.py so the same FM model is rendered identically
# across plots. "DK" is added with the DK_COLOR used in plot4_targets.
PER_TARGET_PALETTE: dict[str, str] = {
    "DK":      "#555555",
    "BIOT":    "#1f77b4",
    "LaBraM":  "#2ca02c",
    "EEGPT":   "#8c564b",
    "NeuroLM": "#9467bd",
    "CBraMod": "#ff7f0e",
}


def _build_per_target_box_data(
    model_paths: dict[str, tuple[str, str | None]],
) -> dict[str, dict[str, list[float]]]:
    """Return ``{metric_path: {model_display: [fold_aucs, ...]}}``.

    Cells with no data become an empty list so the panel still draws but the
    missing model contributes no box / swarm.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for metric_path, _label in CONDITIONS:
        per: dict[str, list[float]] = {}
        for model in MODEL_ORDER:
            aucs = load_fold_aucs(model, metric_path, model_paths)
            per[model] = list(aucs) if aucs else []
        out[metric_path] = per
    return out


def _per_target_pvalues(
    box_data: dict[str, dict[str, list[float]]],
    model_paths: dict[str, tuple[str, str | None]],
    *,
    kind: str,
) -> dict[tuple[str, str], tuple[float, str]]:
    """Compute two-sided FM-vs-DK p-values per (target, FM_model).

    Returns ``{(metric_path, model): (p_two_sided, direction)}`` where
    ``direction`` ∈ {"above", "below", "na"} is taken from ``mean(FM)`` vs
    ``mean(DK)``. Cells with insufficient data are omitted.

    ``kind`` selects between ``"welch"`` (independent two-sided t-test) and
    ``"corrected"`` (Nadeau-Bengio two-sided, paired by ``(repeat, fold)``).
    """
    out: dict[tuple[str, str], tuple[float, str]] = {}
    for metric_path, _label in CONDITIONS:
        per = box_data.get(metric_path, {})
        dk_aucs = per.get("DK", [])
        if not dk_aucs:
            continue
        for model in MODEL_ORDER:
            if model == "DK":
                continue
            fm_aucs = per.get(model, [])
            if not fm_aucs:
                continue
            if kind == "welch":
                if len(fm_aucs) < 2 or len(dk_aucs) < 2:
                    continue
                _t, p_two = stats.ttest_ind(fm_aucs, dk_aucs, equal_var=False)
                p = float(p_two)
            else:  # "corrected" — Nadeau-Bengio two-sided
                res = test_fm_vs_dk_corrected(model, metric_path, model_paths)
                if res is None:
                    continue
                _t, p_two, _pa, _pb, _n, _ntr, _nte = res
                p = float(p_two)
            if np.isnan(p):
                continue
            mean_fm = float(np.mean(fm_aucs))
            mean_dk = float(np.mean(dk_aucs))
            if mean_fm > mean_dk:
                direction = "above"
            elif mean_fm < mean_dk:
                direction = "below"
            else:
                direction = "na"
            out[(metric_path, model)] = (p, direction)
    return out


def _apply_fdr_flat(
    pmap: dict[tuple[str, str], tuple[float, str]],
) -> dict[tuple[str, str], tuple[float, str]]:
    """BH-FDR across all (target, model) cells; preserves the direction tag."""
    if not pmap:
        return {}
    keys = list(pmap.keys())
    pvals = [pmap[k][0] for k in keys]
    p_adj = stats.false_discovery_control(pvals, method="bh")
    return {k: (float(p_adj[i]), pmap[k][1]) for i, k in enumerate(keys)}


def _per_target_pvalues_directional(
    box_data: dict[str, dict[str, list[float]]],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute one-tailed FM-vs-DK p-values per (target, FM_model).

    Returns ``{(metric_path, model): (p_above, p_below)}`` from independent
    one-tailed t-tests:
      * ``p_above``: H₁ mean(FM) > mean(DK)
      * ``p_below``: H₁ mean(FM) < mean(DK)
    """
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for metric_path, _label in CONDITIONS:
        per = box_data.get(metric_path, {})
        dk_aucs = per.get("DK", [])
        if not dk_aucs or len(dk_aucs) < 2:
            continue
        for model in MODEL_ORDER:
            if model == "DK":
                continue
            fm_aucs = per.get(model, [])
            if not fm_aucs or len(fm_aucs) < 2:
                continue
            t, p_two = stats.ttest_ind(fm_aucs, dk_aucs, equal_var=False)
            if np.isnan(p_two):
                continue
            p_above = p_two / 2 if t > 0 else 1.0 - p_two / 2
            p_below = p_two / 2 if t < 0 else 1.0 - p_two / 2
            out[(metric_path, model)] = (float(p_above), float(p_below))
    return out


def _apply_fdr_directional(
    pmap: dict[tuple[str, str], tuple[float, float]],
) -> dict[tuple[str, str], tuple[float, float]]:
    """BH-FDR jointly across both directions (above/below) and all cells."""
    if not pmap:
        return {}
    keys = list(pmap.keys())
    pvals = [pmap[k][0] for k in keys] + [pmap[k][1] for k in keys]
    p_adj = stats.false_discovery_control(pvals, method="bh")
    n = len(keys)
    return {k: (float(p_adj[i]), float(p_adj[n + i])) for i, k in enumerate(keys)}


BOX_LINECOLOR = "#444444"
BOX_FILL_ALPHA = 0.65
SIG_STAR_COLOR = "red"


def _apply_box_alpha(ax, alpha: float = BOX_FILL_ALPHA) -> None:
    """Apply fill alpha to all box patches in an axis."""
    for patch in ax.patches:
        try:
            patch.set_alpha(alpha)
        except Exception:
            pass


def _plot_per_target_box(
    box_data: dict[str, dict[str, list[float]]],
    pmap: dict[tuple[str, str], tuple[float, str]],
    out_path: Path,
    test_label: str,
    alpha: float = ALPHA,
) -> None:
    """4×2 boxplot figure (one panel per target).

    Boxes show the per-fold AUC distribution for each model in
    ``MODEL_ORDER``. A red ``*`` marks any FM significantly different from
    DK (two-sided, ``p < alpha``) — direction is not encoded.
    """
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharey=True)

    for ax, (metric_path, label) in zip(axes.flat, CONDITIONS):
        per = box_data.get(metric_path, {})
        records = [
            {"model": model, "auc": auc}
            for model in MODEL_ORDER
            for auc in per.get(model, [])
        ]

        if records:
            df = pd.DataFrame(records)
            sns.boxplot(
                data=df, x="model", y="auc", order=MODEL_ORDER,
                hue="model", hue_order=MODEL_ORDER, palette=PER_TARGET_PALETTE,
                showfliers=False, width=0.4, legend=False,
                linecolor=BOX_LINECOLOR, linewidth=0.9, ax=ax,
            )
            _apply_box_alpha(ax)

        for j in range(0, len(MODEL_ORDER), 2):
            ax.axvspan(j - 0.5, j + 0.5, color="gray", alpha=0.1, zorder=0)

        ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                   alpha=0.7, zorder=1)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=15)
        ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        ax.set_ylim(0.2, 1.18)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label, fontsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        for j, model in enumerate(MODEL_ORDER):
            if model == "DK":
                continue
            entry = pmap.get((metric_path, model))
            if entry is None:
                continue
            p, _direction = entry
            if p >= alpha:
                continue
            fm_aucs = per.get(model, [])
            ymax = max(fm_aucs) if fm_aucs else 0.5
            ax.text(
                j, min(ymax + 0.02, 1), "*",
                color=SIG_STAR_COLOR, ha="center", va="bottom",
                fontsize=14, fontweight="bold",
            )

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=16)

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color=SIG_STAR_COLOR, markersize=11,
                   linestyle="", label=f"FM ≠ DK  (p < {alpha}, {test_label})"),
        ],
        loc="lower right", fontsize=14, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot1] wrote {out_path}")


def _plot_per_target_dot(
    box_data: dict[str, dict[str, list[float]]],
    pmap: dict[tuple[str, str], tuple[float, str]],
    out_path: Path,
    test_label: str,
    alpha: float = ALPHA,
) -> None:
    """4×2 dot figure (one panel per target) — same data and aesthetic as
    :func:`_plot_per_target_box`, but each model is shown as a coloured dot at
    the median with std error bars."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharey=True)

    for ax, (metric_path, label) in zip(axes.flat, CONDITIONS):
        per = box_data.get(metric_path, {})

        for j in range(0, len(MODEL_ORDER), 2):
            ax.axvspan(j - 0.5, j + 0.5, color="gray", alpha=0.1, zorder=0)

        ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                   alpha=0.7, zorder=1)

        for j, model in enumerate(MODEL_ORDER):
            aucs = per.get(model, [])
            if not aucs:
                continue
            med = float(np.median(aucs))
            sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            color = PER_TARGET_PALETTE.get(model, "#555555")
            ax.errorbar(
                [j], [med], yerr=[sd], fmt="o", color=color,
                capsize=3, markersize=8, linewidth=1.2, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            )

        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=15)
        ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        ax.set_ylim(0.2, 1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label, fontsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        for j, model in enumerate(MODEL_ORDER):
            if model == "DK":
                continue
            entry = pmap.get((metric_path, model))
            if entry is None:
                continue
            p, _direction = entry
            if p >= alpha:
                continue
            aucs = per.get(model, [])
            if not aucs:
                continue
            med = float(np.median(aucs))
            sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            ax.text(
                j, min(med + sd + 0.02, 1.1), "*",
                color=SIG_STAR_COLOR, ha="center", va="bottom",
                fontsize=14, fontweight="bold",
            )

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=18)

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color=SIG_STAR_COLOR, markersize=11,
                   linestyle="", label=f"FM ≠ DK  (p < {alpha}, {test_label})"),
        ],
        loc="lower right", fontsize=14, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot1] wrote {out_path}")


SIG_STAR_ABOVE_COLOR = "blue"
SIG_STAR_BELOW_COLOR = "red"


def _plot_per_target_dot_directional(
    box_data: dict[str, dict[str, list[float]]],
    pmap: dict[tuple[str, str], tuple[float, float]],
    out_path: Path,
    test_label: str,
    alpha: float = ALPHA,
) -> None:
    """4×2 dot figure (one panel per target) — same aesthetic as
    :func:`_plot_per_target_dot`, but plots two stars per cell:
      * blue ``*`` if FM > DK significantly (one-tailed, p < alpha)
      * red  ``*`` if FM < DK significantly (one-tailed, p < alpha)
    """
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharey=True)

    for ax, (metric_path, label) in zip(axes.flat, CONDITIONS):
        per = box_data.get(metric_path, {})

        for j in range(0, len(MODEL_ORDER), 2):
            ax.axvspan(j - 0.5, j + 0.5, color="gray", alpha=0.1, zorder=0)

        ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                   alpha=0.7, zorder=1)

        for j, model in enumerate(MODEL_ORDER):
            aucs = per.get(model, [])
            if not aucs:
                continue
            med = float(np.median(aucs))
            sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            color = PER_TARGET_PALETTE.get(model, "#555555")
            ax.errorbar(
                [j], [med], yerr=[sd], fmt="o", color=color,
                capsize=3, markersize=8, linewidth=1.2, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            )

        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=15)
        ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        ax.set_ylim(0.2, 1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label, fontsize=16)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        for j, model in enumerate(MODEL_ORDER):
            if model == "DK":
                continue
            entry = pmap.get((metric_path, model))
            if entry is None:
                continue
            p_above, p_below = entry
            aucs = per.get(model, [])
            if not aucs:
                continue
            med = float(np.median(aucs))
            sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            base_y = min(med + sd + 0.02, 0.98)
            stars = []
            if p_above < alpha:
                stars.append(SIG_STAR_ABOVE_COLOR)
            if p_below < alpha:
                stars.append(SIG_STAR_BELOW_COLOR)
            for k, c in enumerate(stars):
                ax.text(
                    j, min(base_y + 0.04 * k, 1.0), "*",
                    color=c, ha="center", va="bottom",
                    fontsize=14, fontweight="bold",
                )

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=18)

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color=SIG_STAR_ABOVE_COLOR,
                   markersize=11, linestyle="",
                   label=f"FM > DK  (p < {alpha}, {test_label})"),
            Line2D([], [], marker="$*$", color=SIG_STAR_BELOW_COLOR,
                   markersize=11, linestyle="",
                   label=f"FM < DK  (p < {alpha}, {test_label})"),
        ],
        loc="lower right", fontsize=14, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, -0.01), bbox_transform=fig.transFigure,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot1] wrote {out_path}")


def _write_per_target_box_counts_md(
    box_data: dict[str, dict[str, list[float]]],
    out_path: Path,
) -> None:
    """Write a markdown table of per-fold sample sizes for each box."""
    lines = [
        "# plot1 — Per-target boxplot sample sizes (non-PCA, MLP_EMBEDDING)",
        "",
        "Number of per-fold AUC values shown in each box of",
        "`plot1_per_target_box_*.png`. Empty cells mean no data was loaded.",
        "",
        "| Target | " + " | ".join(MODEL_ORDER) + " |",
        "|" + "|".join(["---"] * (len(MODEL_ORDER) + 1)) + "|",
    ]
    for metric_path, label in CONDITIONS:
        per = box_data.get(metric_path, {})
        cells = [str(len(per.get(m, []))) if per.get(m) else "" for m in MODEL_ORDER]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[statistics_plot1] wrote {out_path}")


# ── Per-variant runner ─────────────────────────────────────────────────────────

def _run_variant(
    model_paths: dict[str, tuple[str, str | None]],
    pca: bool,
) -> None:
    suffix = "_pca" if pca else ""
    tag = "PCA=yes" if pca else "PCA=no"

    # Pre-compute classifier choices so they're visible and consistent with
    # plot1.py's clf_choices dict (same _select_best_clf logic, same paths).
    clf_choices: dict[str, dict[str, str]] = {
        model: {
            metric_path: _select_best_clf(model, metric_path, model_paths)
            for metric_path, _ in CONDITIONS
        }
        for model in MODEL_ORDER
    }
    print(f"\n[statistics_plot1] Classifier choices ({tag}):")
    for model, cmap in clf_choices.items():
        for mp, clf in cmap.items():
            print(f"  {model:10s}  {mp:30s}  {clf}")

    dk_aucs: dict[str, list[float] | None] = {
        metric_path: load_fold_aucs("DK", metric_path, model_paths)
        for metric_path, _ in CONDITIONS
    }

    rows: list[dict] = []

    for model in MODEL_ORDER:
        for metric_path, condition_label in CONDITIONS:
            fold_aucs = load_fold_aucs(model, metric_path, model_paths)
            if fold_aucs is None:
                print(f"[skip] {model} / {condition_label} ({tag}) | no data")
                continue

            mean_auc = float(np.mean(fold_aucs))
            std_auc  = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else np.nan

            t_chance, p_chance   = test_above_chance(fold_aucs)
            w_chance, pw_chance  = test_above_chance_wilcoxon(fold_aucs)

            if model == "DK":
                t_dk = p_above_dk = p_below_dk = np.nan
                u_dk = pu_above_dk = pu_below_dk = np.nan
                t_dk_corr = p_two_dk_corr = p_above_dk_corr = p_below_dk_corr = np.nan
                n_paired_corr = np.nan
                n_train_corr = n_test_corr = np.nan
            else:
                dk = dk_aucs.get(metric_path)
                if dk:
                    t_dk, p_above_dk, p_below_dk    = test_fm_vs_dk(fold_aucs, dk)
                    u_dk, pu_above_dk, pu_below_dk  = test_fm_vs_dk_wilcoxon(fold_aucs, dk)
                else:
                    t_dk = p_above_dk = p_below_dk = np.nan
                    u_dk = pu_above_dk = pu_below_dk = np.nan
                corr_res = test_fm_vs_dk_corrected(model, metric_path, model_paths)
                if corr_res is not None:
                    (t_dk_corr, p_two_dk_corr, p_above_dk_corr, p_below_dk_corr,
                     n_paired_corr, n_train_corr, n_test_corr) = corr_res
                else:
                    t_dk_corr = p_two_dk_corr = p_above_dk_corr = p_below_dk_corr = np.nan
                    n_paired_corr = np.nan
                    n_train_corr = n_test_corr = np.nan

            def _r(v):
                return round(v, 4) if not np.isnan(v) else np.nan

            rows.append({
                "Model":           model,
                "Condition":       condition_label,
                "n_folds":         len(fold_aucs),
                "mean_AUC":        round(mean_auc, 4),
                "std_AUC":         _r(std_auc),
                # t-test above chance
                "t_vs_chance":     round(t_chance, 3) if not np.isnan(t_chance) else np.nan,
                "p_vs_chance":     _r(p_chance),
                "sig_vs_chance":   _stars(p_chance),
                # Wilcoxon above chance
                "w_vs_chance":     round(w_chance, 3) if not np.isnan(w_chance) else np.nan,
                "pw_vs_chance":    _r(pw_chance),
                "sigw_vs_chance":  _stars(pw_chance),
                # t-test FM vs DK
                "t_vs_DK":         round(t_dk, 3) if not np.isnan(t_dk) else np.nan,
                "p_above_DK":      _r(p_above_dk),
                "sig_above_DK":    _stars(p_above_dk),
                "p_below_DK":      _r(p_below_dk),
                "sig_below_DK":    _stars(p_below_dk),
                # Wilcoxon FM vs DK
                "u_vs_DK":         round(u_dk, 3) if not np.isnan(u_dk) else np.nan,
                "pu_above_DK":     _r(pu_above_dk),
                "sigu_above_DK":   _stars(pu_above_dk),
                "pu_below_DK":     _r(pu_below_dk),
                "sigu_below_DK":   _stars(pu_below_dk),
                # Corrected resampled t-test (Nadeau & Bengio 2003) FM vs DK
                "t_corr_vs_DK":     round(t_dk_corr, 3) if not np.isnan(t_dk_corr) else np.nan,
                "n_paired_corr":    int(n_paired_corr) if not (isinstance(n_paired_corr, float) and np.isnan(n_paired_corr)) else np.nan,
                "n_train_corr":     int(n_train_corr) if not (isinstance(n_train_corr, float) and np.isnan(n_train_corr)) else np.nan,
                "n_test_corr":      int(n_test_corr) if not (isinstance(n_test_corr, float) and np.isnan(n_test_corr)) else np.nan,
                "p_two_DK_corr":    _r(p_two_dk_corr),
                "sig_two_DK_corr":  _stars(p_two_dk_corr),
                "p_above_DK_corr":  _r(p_above_dk_corr),
                "sig_above_DK_corr": _stars(p_above_dk_corr),
                "p_below_DK_corr":  _r(p_below_dk_corr),
                "sig_below_DK_corr": _stars(p_below_dk_corr),
            })

    df = pd.DataFrame(rows)

    # ── FDR corrections on p-value columns ────────────────────────────────────
    for raw_col, fdr_col, sig_col in [
        ("p_vs_chance",      "p_vs_chance_fdr",      "sig_vs_chance_fdr"),
        ("pw_vs_chance",     "pw_vs_chance_fdr",     "sigw_vs_chance_fdr"),
        ("p_above_DK",       "p_above_DK_fdr",       "sig_above_DK_fdr"),
        ("p_below_DK",       "p_below_DK_fdr",       "sig_below_DK_fdr"),
        ("pu_above_DK",      "pu_above_DK_fdr",      "sigu_above_DK_fdr"),
        ("pu_below_DK",      "pu_below_DK_fdr",      "sigu_below_DK_fdr"),
        ("p_two_DK_corr",    "p_two_DK_corr_fdr",    "sig_two_DK_corr_fdr"),
        ("p_above_DK_corr",  "p_above_DK_corr_fdr",  "sig_above_DK_corr_fdr"),
        ("p_below_DK_corr",  "p_below_DK_corr_fdr",  "sig_below_DK_corr_fdr"),
    ]:
        df[fdr_col] = _apply_fdr_to_series(df[raw_col]).round(4)
        df[sig_col] = df[fdr_col].apply(_stars)

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = OUT / f"statistics_plot1{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[statistics_plot1] wrote {csv_path}")

    # ── HTML ───────────────────────────────────────────────────────────────────
    html_path = OUT / f"statistics_plot1{suffix}.html"

    def _color_sig(val: str) -> str:
        return {
            "***": "background-color:#1a7a1a; color:white",
            "**":  "background-color:#4caf50; color:white",
            "*":   "background-color:#a5d6a7",
            "†":   "background-color:#e8f5e9",
        }.get(str(val), "")

    sig_cols = [c for c in df.columns if c.startswith("sig")]
    float_fmt = {c: "{:.4f}" for c in df.select_dtypes("float").columns
                 if c not in sig_cols}
    stat_fmt  = {c: "{:.3f}" for c in ["t_vs_chance", "w_vs_chance",
                                         "t_vs_DK", "u_vs_DK"]}
    float_fmt.update(stat_fmt)

    caption = (
        f"Best-classifier nested-CV AUC ({tag}) | matches plot1_stacked{'_pca' if pca else ''}.png. "
        "T-test: one-tailed independent. Wilcoxon (vs chance): signed-rank. "
        "Wilcoxon (vs DK): Mann-Whitney U. FDR: Benjamini-Hochberg across all rows. "
        "Stars: *** p<0.001 · ** p<0.01 · * p<0.05 · † p<0.1 · ns not significant."
    )

    styled = (
        df.style
        .map(_color_sig, subset=sig_cols)
        .format(float_fmt, na_rep="")
        .set_caption(caption)
        .set_table_styles([
            {"selector": "th",
             "props": [("background", "#2c3e50"), ("color", "white"), ("padding", "6px 10px")]},
            {"selector": "td",
             "props": [("padding", "4px 8px"), ("border", "1px solid #ddd")]},
            {"selector": "tr:nth-child(even)",
             "props": [("background", "#f9f9f9")]},
            {"selector": "caption",
             "props": [("caption-side", "top"), ("font-style", "italic"),
                       ("font-size", "0.9em"), ("margin-bottom", "6px")]},
        ])
    )
    with html_path.open("w", encoding="utf-8") as f:
        f.write(styled.to_html())
    print(f"[statistics_plot1] wrote {html_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Full table (best classifier per FM×target, {tag}, matches plot1_stacked{'_pca' if pca else ''}.png):")
    print(df.to_string(index=False))

    for label, col in [
        ("Significant above chance | t-test (p < 0.05)",         "p_vs_chance"),
        ("Significant above chance | t-test FDR (p < 0.05)",     "p_vs_chance_fdr"),
        ("Significant above chance | Wilcoxon (p < 0.05)",       "pw_vs_chance"),
        ("Significant above chance | Wilcoxon FDR (p < 0.05)",   "pw_vs_chance_fdr"),
    ]:
        sub = df[df[col] < ALPHA]
        print(f"\n{'='*60}\n{label} [{tag}]:")
        print(sub[["Model", "Condition", "mean_AUC", col]].to_string(index=False))

    for label, col in [
        ("FM significantly above DK | t-test",                       "p_above_DK"),
        ("FM significantly above DK | t-test FDR",                   "p_above_DK_fdr"),
        ("FM significantly above DK | Wilcoxon",                     "pu_above_DK"),
        ("FM significantly above DK | Wilcoxon FDR",                 "pu_above_DK_fdr"),
        ("FM significantly above DK | Nadeau-Bengio corrected (one-tailed derived)",   "p_above_DK_corr"),
        ("FM significantly above DK | Nadeau-Bengio corrected (one-tailed) + FDR",     "p_above_DK_corr_fdr"),
        ("FM ≠ DK | Nadeau-Bengio corrected (two-sided)",                              "p_two_DK_corr"),
        ("FM ≠ DK | Nadeau-Bengio corrected (two-sided) + FDR",                        "p_two_DK_corr_fdr"),
    ]:
        sub = df[df[col] < ALPHA]
        print(f"\n{'='*60}\n{label} [{tag}]:")
        print(sub[["Model", "Condition", "mean_AUC", col]].to_string(index=False))

    # ── 4 annotated plots ─────────────────────────────────────────────────────
    models_present = [
        m for m in MODEL_ORDER
        if any(load_fold_aucs(m, mp, model_paths) is not None for mp, _ in CONDITIONS)
    ]

    mean1, std1 = build_plot_matrix(models_present, ROW1, model_paths)
    mean2, std2 = build_plot_matrix(models_present, ROW2, model_paths)

    masks = _below_dk_masks(models_present, CONDITIONS, model_paths)

    plot_variants = [
        ("ttest",         "T-test"),
        ("ttest_fdr",     "T-test + FDR correction"),
        ("wilcoxon",      "Wilcoxon (Mann-Whitney U)"),
        ("wilcoxon_fdr",  "Wilcoxon + FDR correction"),
        ("corrected",     "Nadeau-Bengio t-test (two-sided)"),
        ("corrected_fdr", "Nadeau-Bengio t-test (two-sided) + FDR correction"),
    ]
    n_row1 = len(ROW1)

    for key, label in plot_variants:
        full_mask = masks[key]
        mask1 = full_mask[:, :n_row1]
        mask2 = full_mask[:, n_row1:]
        # The "corrected" variant uses canonical (two-sided) Nadeau-Bengio;
        # the other variants are one-tailed FM<DK tests.
        hypothesis = "FM ≠ DK" if key.startswith("corrected") else "FM < DK"
        full_label = f"{hypothesis} ({label})"
        out_path = OUT / f"plot1_stacked_annotated_{key}{suffix}.png"
        _plot_stacked_annotated(
            models_present, mean1, std1, mean2, std2, mask1, mask2, out_path, full_label
        )

    # ── Per-target boxplot figures (non-PCA only) ────────────────────────────
    # Four variants of the FM-vs-DK two-sided test, one figure each:
    #   T-test, T-test+FDR, Nadeau-Bengio, Nadeau-Bengio+FDR.
    if not pca:
        box_data = _build_per_target_box_data(model_paths)
        for kind, do_fdr, fname_box, fname_dot, label in [
            ("welch",     False, "plot1_per_target_box_ttest.png",
             "plot1_per_target_dot_ttest.png", "T-test"),
            ("welch",     True,  "plot1_per_target_box_ttest_fdr.png",
             "plot1_per_target_dot_ttest_fdr.png", "T-test + FDR correction"),
            ("corrected", False, "plot1_per_target_box_corrected.png",
             "plot1_per_target_dot_corrected.png", "Nadeau-Bengio"),
            ("corrected", True,  "plot1_per_target_box_corrected_fdr.png",
             "plot1_per_target_dot_corrected_fdr.png",
             "Nadeau-Bengio + FDR correction"),
        ]:
            pmap = _per_target_pvalues(box_data, model_paths, kind=kind)
            if do_fdr:
                pmap = _apply_fdr_flat(pmap)
            _plot_per_target_box(box_data, pmap, OUT / fname_box, test_label=label)
            _plot_per_target_dot(box_data, pmap, OUT / fname_dot, test_label=label)

        # Two-direction t-test variant: blue * for FM > DK, red * for FM < DK.
        pmap_dir = _per_target_pvalues_directional(box_data)
        _plot_per_target_dot_directional(
            box_data, pmap_dir,
            OUT / "plot1_per_target_dot_ttest_directional.png",
            test_label="T-test",
        )
        pmap_dir_fdr = _apply_fdr_directional(pmap_dir)
        _plot_per_target_dot_directional(
            box_data, pmap_dir_fdr,
            OUT / "plot1_per_target_dot_ttest_fdr_directional.png",
            test_label="T-test + FDR correction",
        )

        _write_per_target_box_counts_md(
            box_data, BASE / "plot1_per_target_box_counts.md"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("\n[statistics_plot1] === Non-PCA (MLP_EMBEDDING) ===")
    _run_variant(MODEL_PATHS_MLP, pca=False)
    print("\n[statistics_plot1] === PCA 27-dim (EMBEDDING_FM_PCA_ONLY) ===")
    _run_variant(MODEL_PATHS_PCA, pca=True)


if __name__ == "__main__":
    main()
