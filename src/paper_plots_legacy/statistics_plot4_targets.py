#!/usr/bin/env python3
"""Statistical tests mirroring plot4_combined_pca.png and plot4_combined_nonpca.png.

For each (mode, FM model, condition) triple the figure shows three dots:
  PCA mode    — FM: EMBEDDING_FM_PCA_ONLY · FM+DK: EMBEDDING_DK_COMBINED_FM_PCA/pca27
                FM Res: RESIDUALIZATION_DIM/pca27
  non-PCA mode — FM: MLP_EMBEDDING        · FM+DK: EMBEDDING_DK_COMBINED
                FM Res: RESIDUALIZATION_DIM

Four test variants per row:
  t-test         — independent Welch one-tailed t-test
  t-test + FDR   — same with Benjamini-Hochberg correction
  Wilcoxon       — Mann-Whitney U one-tailed test
  Wilcoxon + FDR — same with Benjamini-Hochberg correction

  Test 1 — FM+DK above FM:   H1: mean(FM+DK) > mean(FM)
  Test 2 — FM Res below FM:  H1: mean(FM) > mean(FM_Res)

Results:
  - statistics_plot4_targets.csv   (Mode column: PCA / non-PCA)
  - statistics_plot4_targets.html

Eight annotated plots (TARGETS_ETIO × 2 modes × 4 tests):
  plot4_etiology_{nonpca|pca}_annotated_{ttest|ttest_fdr|wilcoxon|wilcoxon_fdr}.png
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
    corrected_resampled_ttest,
    infer_n_train_test,
    paired_per_fold_aucs,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150

# ── Constants — mirror plot4_targets.py exactly ───────────────────────────────

BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
OUT_DIR = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]
FM_MODELS = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CBraMod"]

FM_DISPLAY = {
    "BIOT":    "BIOT",
    "LaBram":  "LaBraM",
    "EEGPT":   "EEGPT",
    "NeuroLM": "NeuroLM",
    "CBraMod": "CBraMod",
}

TARGETS: list[tuple[str, list[str]]] = [
    ("CRS Diagnostic",            ["crs"]),
    ("6m",             ["cs_6m",         "binary_improvement"]),
    ("1y",             ["cs_1y",         "binary_improvement"]),
    ("2y",             ["cs_2y",         "binary_improvement"]),
    ("Delay (VS)",     ["etiology",      "vs_only"]),
    ("Delay (MCS)",    ["etiology",      "mcs_only"]),
    ("Etiology (VS)",  ["etiology_code", "vs_only"]),
    ("Etiology (MCS)", ["etiology_code", "mcs_only"]),
]

TARGETS_CS = [
    ("CRS Diagnostic",  ["crs"]),
    ("6m",   ["cs_6m",  "binary_improvement"]),
    ("1y",   ["cs_1y",  "binary_improvement"]),
    ("2y",   ["cs_2y",  "binary_improvement"]),
]
TARGETS_ETIO = [
    ("Delay (VS)",     ["etiology",      "vs_only"]),
    ("Delay (MCS)",    ["etiology",      "mcs_only"]),
    ("Etiology (VS)",  ["etiology_code", "vs_only"]),
    ("Etiology (MCS)", ["etiology_code", "mcs_only"]),
]

ALPHA = 0.05

DK_COLOR      = "#555555"
MISSING_COLOR = "#b0b0b0"
FM_COLORS = {
    "BIOT":    "#1f77b4",
    "EEGPT":   "#8c564b",
    "LaBram":  "#2ca02c",
    "NeuroLM": "#9467bd",
    "CBraMod": "#ff7f0e",
}


# ── Fold-AUC loaders (for statistical tests) ──────────────────────────────────

def _fold_aucs_from_classification_json(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    per_fold = d.get("macro_average", {}).get("per_fold_metrics")
    if not per_fold:
        return None
    aucs = [
        fold["auc_score"] for fold in per_fold
        if fold.get("auc_score") is not None
        and not (isinstance(fold["auc_score"], float) and np.isnan(fold["auc_score"]))
    ]
    return aucs if aucs else None


def load_fm_base(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    emb = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    rep = (BASE / model / "doc_patients" / emb / Path(*target_segs)
           / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    r = _fold_aucs_from_classification_json(rep)
    if r is not None:
        return r
    primary = (BASE / model / "doc_patients" / emb
               / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
    r = _fold_aucs_from_classification_json(primary)
    if r is not None:
        return r
    fallback = (BASE / model / "doc_patients" / emb / model / "doc_patients" / emb
                / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
    return _fold_aucs_from_classification_json(fallback)


def load_fm_combined(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    emb = "EMBEDDING_DK_COMBINED_FM_PCA" if pca else "EMBEDDING_DK_COMBINED"
    base = BASE / model / "doc_patients" / emb
    if pca:
        base = base / "pca27"
    for sub in ("cv_nested_matched", "cv_nested"):
        rep = base / Path(*target_segs) / sub / "best_clf" / "classification_results.json"
        r = _fold_aucs_from_classification_json(rep)
        if r is not None:
            return r
    rep = base / Path(*target_segs) / "nested_cv_repeated" / "best_clf" / "classification_results.json"
    r = _fold_aucs_from_classification_json(rep)
    if r is not None:
        return r
    primary = base / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json"
    r = _fold_aucs_from_classification_json(primary)
    if r is not None:
        return r
    fallback = (BASE / model / "doc_patients" / emb / model / "doc_patients" / emb
                / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
    return _fold_aucs_from_classification_json(fallback)


def load_fm_res(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    pca_label = "pca27" if pca else "no_pca"
    rep = (BASE / "RES_NO_LEAKAGE" / "cv_nested" / pca_label / "best_clf"
           / model / Path(*target_segs) / "results.json")
    if rep.is_file():
        r = _fold_aucs_from_classification_json(rep)
        if r is not None:
            return r
    path = BASE / "RES_NO_LEAKAGE" / pca_label / Path(*target_segs) / "results.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    clfs = d.get("results", {}).get(model, {}).get("classifiers", {})
    clf_data = clfs.get("best_clf") or clfs.get(classifier)
    if clf_data is None:
        return None
    aucs = clf_data.get("auc_per_fold")
    if not aucs:
        return None
    return [a for a in aucs if a is not None and not (isinstance(a, float) and np.isnan(a))]


# ── Mean/std loaders (for plotting) ───────────────────────────────────────────

def _macro_auc(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    auc = d.get("macro_average", {}).get("auc_score", {})
    m = auc.get("mean")
    s = auc.get("std")
    if m is None:
        return None
    return float(m), float(s) if s is not None else 0.0


def _load_ms_baseline(target_segs: list[str]) -> tuple[float, float] | None:
    rep = (BASE / "MARKER_BASELINE" / Path(*target_segs)
           / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    r = _macro_auc(rep)
    if r is not None:
        return r
    best: tuple[float, float] | None = None
    for clf in CLASSIFIERS_TO_TRY:
        r = _macro_auc(
            BASE / "MARKER_BASELINE" / Path(*target_segs)
            / "nested_cv" / clf / "classification_results.json"
        )
        if r is not None and (best is None or r[0] > best[0]):
            best = r
    return best


def _load_ms_fm_embedding(
    model: str, embedding_kind: str, target_segs: list[str], classifier: str
) -> tuple[float, float] | None:
    base = BASE / model / "doc_patients" / embedding_kind
    if embedding_kind == "EMBEDDING_DK_COMBINED_FM_PCA":
        base = base / "pca27"
    if embedding_kind in ("EMBEDDING_DK_COMBINED", "EMBEDDING_DK_COMBINED_FM_PCA"):
        for sub in ("cv_nested_matched", "cv_nested"):
            r = _macro_auc(base / Path(*target_segs) / sub / "best_clf" / "classification_results.json")
            if r is not None:
                return r
    r = _macro_auc(base / Path(*target_segs) / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    if r is not None:
        return r
    primary = base / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json"
    r = _macro_auc(primary)
    if r is not None:
        return r
    fallback = (BASE / model / "doc_patients" / embedding_kind / model
                / "doc_patients" / embedding_kind / Path(*target_segs)
                / "nested_cv" / classifier / "classification_results.json")
    return _macro_auc(fallback)


def _load_ms_fm_res(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> tuple[float, float] | None:
    pca_label = "pca27" if pca else "no_pca"
    rep = (BASE / "RES_NO_LEAKAGE" / "cv_nested" / pca_label / "best_clf"
           / model / Path(*target_segs) / "results.json")
    if rep.is_file():
        try:
            d = json.loads(rep.read_text())
            m = d.get("auc_mean")
            if m is None or (isinstance(m, float) and np.isnan(m)):
                m = d.get("macro_average", {}).get("auc_score", {}).get("mean")
            s = d.get("auc_std")
            if s is None or (isinstance(s, float) and np.isnan(s)):
                s = d.get("macro_average", {}).get("auc_score", {}).get("std", 0.0)
            if m is not None and not (isinstance(m, float) and np.isnan(m)):
                return float(m), float(s) if s is not None else 0.0
        except Exception:
            pass
    path = BASE / "RES_NO_LEAKAGE" / pca_label / Path(*target_segs) / "results.json"
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception:
        return None
    clfs = d.get("results", {}).get(model, {}).get("classifiers", {})
    entry = clfs.get("best_clf") or clfs.get(classifier)
    if not entry:
        return None
    m = entry.get("mean_auc")
    s = entry.get("std_auc")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    return float(m), float(s) if s is not None else 0.0


def _select_best_clf_for_fm(
    model: str, target_segs: list[str], pca: bool
) -> str:
    """Return the classifier with the highest FM-only mean AUC (mirrors plot1/plot4 logic)."""
    fm_kind = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    best_clf, best_mean = "random_forest", -1.0
    for clf in CLASSIFIERS_TO_TRY:
        r = _load_ms_fm_embedding(model, fm_kind, target_segs, clf)
        if r is not None and r[0] > best_mean:
            best_clf, best_mean = clf, r[0]
    return best_clf


def build_rows(
    target_segs: list[str], pca: bool
) -> list[tuple[str, float | None, float | None, str]]:
    rows: list[tuple[str, float | None, float | None, str]] = []
    b = _load_ms_baseline(target_segs)
    rows.append(("DK", b[0] if b else None, b[1] if b else None, DK_COLOR))

    variants = (
        [("", "EMBEDDING_FM_PCA_ONLY"), ("+ DK", "EMBEDDING_DK_COMBINED_FM_PCA"), ("Res", None)]
        if pca else
        [("", "MLP_EMBEDDING"),         ("+ DK", "EMBEDDING_DK_COMBINED"),         ("Res", None)]
    )
    for model in FM_MODELS:
        color = FM_COLORS[model]
        clf = _select_best_clf_for_fm(model, target_segs, pca)
        for suffix, embedding_kind in variants:
            label = f"{model} {suffix}".strip()
            if embedding_kind is None:
                r = _load_ms_fm_res(model, target_segs, pca=pca, classifier=clf)
            else:
                r = _load_ms_fm_embedding(model, embedding_kind, target_segs, clf)
            rows.append((label, r[0] if r else None, r[1] if r else None, color))
    return rows


# ── Statistical tests ──────────────────────────────────────────────────────────

def _stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
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


def _welch_one_tailed(a: list[float], b: list[float], h1: str) -> tuple[float, float]:
    """Welch t-test returning (t, p); h1='a>b' or 'a<b'."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    if h1 == "a>b":
        p = p_two / 2 if t > 0 else 1.0 - p_two / 2
    else:
        p = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p)


def _mwu_one_tailed(a: list[float], b: list[float], h1: str) -> tuple[float, float]:
    """Mann-Whitney U test returning (U, p); h1='a>b' or 'a<b'."""
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    alt = "greater" if h1 == "a>b" else "less"
    try:
        u, p = stats.mannwhitneyu(a, b, alternative=alt)
        return float(u), float(p)
    except ValueError:
        return np.nan, np.nan


def _welch_p(a: list[float], b: list[float], h1: str) -> float:
    """One-tailed Welch p-value only."""
    _, p = _welch_one_tailed(a, b, h1)
    return p if not np.isnan(p) else 1.0


def _mwu_p(a: list[float], b: list[float], h1: str) -> float:
    """One-tailed Mann-Whitney U p-value only."""
    _, p = _mwu_one_tailed(a, b, h1)
    return p if not np.isnan(p) else 1.0


# ── FDR correction ────────────────────────────────────────────────────────────

def _apply_fdr_to_series(p_series: pd.Series) -> pd.Series:
    """Apply BH FDR to a Series; NaN entries pass through unchanged."""
    valid_idx = p_series.dropna().index
    if len(valid_idx) == 0:
        return p_series.copy()
    p_adj = stats.false_discovery_control(p_series[valid_idx], method="bh")
    out = p_series.copy()
    out[valid_idx] = p_adj
    return out


# ── Annotation computation (4 test variants) ──────────────────────────────────

def _compute_annotations(
    target_list: list[tuple[str, list[str]]],
    pca: bool,
    test_key: str,
    alpha: float = ALPHA,
) -> dict[str, dict[int, str]]:
    """Compute per-panel annotation dicts for one of the 4 test variants.

    Panel row layout: DK=0, then for i=0..4: base=1+3i, combined=2+3i, res=3+3i.
    Blue * at combined(2+3i) if FM+DK > FM  (p < alpha).
    Red  * at res(3+3i)      if FM  > Res   (p < alpha).

    test_key: 'ttest' | 'ttest_fdr' | 'wilcoxon' | 'wilcoxon_fdr'
    """
    use_wilcoxon = "wilcoxon" in test_key
    use_fdr      = "fdr"      in test_key
    p_fn = _mwu_p if use_wilcoxon else _welch_p

    # Collect raw p-values indexed by (tname, model_index, comparison)
    p_comb: dict[tuple, float] = {}
    p_res:  dict[tuple, float] = {}

    for tname, tsegs in target_list:
        for i, model in enumerate(FM_MODELS):
            clf = _select_best_clf_for_fm(model, tsegs, pca)
            base = load_fm_base(model, tsegs, pca, clf)
            comb = load_fm_combined(model, tsegs, pca, clf)
            res  = load_fm_res(model, tsegs, pca, clf)
            if base and comb:
                p_comb[(tname, i)] = p_fn(comb, base, "a>b")
            if base and res:
                p_res[(tname, i)]  = p_fn(base, res,  "a>b")

    if use_fdr and (p_comb or p_res):
        comb_keys = list(p_comb.keys())
        res_keys  = list(p_res.keys())
        all_p     = [p_comb[k] for k in comb_keys] + [p_res[k] for k in res_keys]
        if all_p:
            p_adj = stats.false_discovery_control(all_p, method="bh")
            n_c = len(comb_keys)
            p_comb = {k: p_adj[j]     for j, k in enumerate(comb_keys)}
            p_res  = {k: p_adj[n_c+j] for j, k in enumerate(res_keys)}

    annots_by_target: dict[str, dict[int, str]] = {}
    for tname, _ in target_list:
        annots: dict[int, str] = {}
        for i in range(len(FM_MODELS)):
            if p_comb.get((tname, i), 1.0) < alpha:
                annots[2 + 3 * i] = "blue"
            if p_res.get((tname, i), 1.0) < alpha:
                annots[3 + 3 * i] = "red"
        annots_by_target[tname] = annots
    return annots_by_target


# ── Panel drawing (mirrored from plot4_targets.py) ────────────────────────────

def _plot_panel(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
) -> None:
    colors  = [r[3] for r in rows]
    x = np.arange(len(rows))

    gi, run_start = 0, 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(run_start - 0.5, i - 0.5, color="gray", alpha=0.08, zorder=0)
            gi += 1
            run_start = i

    ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

    for xi, (_lbl, m, s, c) in zip(x, rows):
        if m is None:
            ax.plot([xi], [0.5], marker="x", color=MISSING_COLOR,
                    markersize=10, markeredgewidth=1.6, zorder=2)
        else:
            ax.errorbar([xi], [m], yerr=[s or 0.0], fmt="D", color=c,
                        capsize=3, markersize=6, linewidth=1.2, ecolor=c,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_panel_annotated(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
    annots: dict[int, str],
) -> None:
    _plot_panel(ax, rows, title)
    for xi, color in annots.items():
        if xi < len(rows):
            _lbl, m, s, _c = rows[xi]
            if m is not None:
                ax.text(xi, m + (s or 0.0) + 0.04, "*",
                        color=color, ha="center", va="bottom",
                        fontsize=14, fontweight="bold")


def plot_2x2_annotated(
    target_list: list[tuple[str, list[str]]],
    pca: bool,
    annots_by_target: dict[str, dict[int, str]],
    out_path: Path,
    test_label: str,
) -> None:
    """Render a 2×2 annotated figure for the given target list."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, target_list):
        rows   = build_rows(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_annotated(ax, rows, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color="blue", markersize=11, linestyle="",
                   label=f"FM+DK > FM  (p < 0.05, {test_label})"),
            Line2D([], [], marker="$*$", color="red",  markersize=11, linestyle="",
                   label=f"FM > FM Res  (p < 0.05, {test_label})"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot4_targets] wrote {out_path}")


def plot_4x2_annotated(
    pca: bool,
    annots_by_target: dict[str, dict[int, str]],
    out_path: Path,
    test_label: str,
) -> None:
    """Render the full 4×2 combined figure (all 8 targets) with annotations."""
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows   = build_rows(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_annotated(ax, rows, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color="blue", markersize=11, linestyle="",
                   label=f"FM+DK > FM  (p < 0.05, {test_label})"),
            Line2D([], [], marker="$*$", color="red",  markersize=11, linestyle="",
                   label=f"FM > FM Res  (p < 0.05, {test_label})"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot4_targets] wrote {out_path}")


# ── Boxplot + swarmplot panels (per-fold AUC distributions) ───────────────────

def _load_dk_folds(target_segs: list[str]) -> list[float] | None:
    """DK baseline per-fold AUCs. Prefers in-fold-selected repeated CV; falls
    back to the per-classifier ``nested_cv`` directory with the highest mean
    (mirrors ``_load_ms_baseline`` selection)."""
    rep = (BASE / "MARKER_BASELINE" / Path(*target_segs)
           / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    r = _fold_aucs_from_classification_json(rep)
    if r is not None:
        return r
    best_aucs, best_mean = None, -1.0
    for clf in CLASSIFIERS_TO_TRY:
        p = (BASE / "MARKER_BASELINE" / Path(*target_segs)
             / "nested_cv" / clf / "classification_results.json")
        a = _fold_aucs_from_classification_json(p)
        if a is not None and len(a) > 0:
            m = float(np.mean(a))
            if m > best_mean:
                best_aucs, best_mean = a, m
    return best_aucs


def _build_rows_box(
    target_segs: list[str], pca: bool = False,
) -> list[tuple[str, list[float] | None, str]]:
    """Per-fold list version of :func:`build_rows`.

    Returns ``[(label, fold_aucs|None, color), ...]`` with the same column
    order as :func:`build_rows`: ``DK`` followed by, per FM, ``(FM, FM+DK,
    FM Res)``.
    """
    rows: list[tuple[str, list[float] | None, str]] = []
    rows.append(("DK", _load_dk_folds(target_segs), DK_COLOR))
    for model in FM_MODELS:
        color = FM_COLORS[model]
        clf = _select_best_clf_for_fm(model, target_segs, pca)
        rows.append((f"{model}",      load_fm_base(model, target_segs, pca, clf), color))
        rows.append((f"{model} + DK", load_fm_combined(model, target_segs, pca, clf), color))
        rows.append((f"{model} Res",  load_fm_res(model, target_segs, pca, clf), color))
    return rows


BOX_LINECOLOR = "#444444"
BOX_FILL_ALPHA = 0.65


def _apply_box_alpha(ax, alpha: float = BOX_FILL_ALPHA) -> None:
    for patch in ax.patches:
        try:
            patch.set_alpha(alpha)
        except Exception:
            pass


def _plot_panel_box(
    ax,
    rows_box: list[tuple[str, list[float] | None, str]],
    title: str,
) -> None:
    """Boxplot version of :func:`_plot_panel` (no swarm, soft outlines)."""
    n = len(rows_box)
    colors = [r[2] for r in rows_box]
    labels = [r[0] for r in rows_box]

    gi, run_start = 0, 0
    for i in range(1, n + 1):
        if i == n or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(run_start - 0.5, i - 0.5,
                           color="gray", alpha=0.08, zorder=0)
            gi += 1
            run_start = i

    ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=1)

    records: list[dict] = []
    for j, (_lbl, aucs, _c) in enumerate(rows_box):
        if aucs:
            for a in aucs:
                records.append({"col": j, "auc": a})

    if records:
        df = pd.DataFrame(records)
        order = list(range(n))
        palette = {j: colors[j] for j in order}
        sns.boxplot(
            data=df, x="col", y="auc", order=order,
            hue="col", hue_order=order, palette=palette,
            showfliers=False, width=0.4, legend=False,
            linecolor=BOX_LINECOLOR, linewidth=0.9, ax=ax,
        )
        _apply_box_alpha(ax)

    for j, (_lbl, aucs, _c) in enumerate(rows_box):
        if not aucs:
            ax.plot([j], [0.5], marker="x", color=MISSING_COLOR,
                    markersize=10, markeredgewidth=1.6, zorder=2)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, 1.3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_panel_box_annotated(
    ax,
    rows_box: list[tuple[str, list[float] | None, str]],
    title: str,
    annots: dict[int, str],
) -> None:
    _plot_panel_box(ax, rows_box, title)
    for xi, color in annots.items():
        if xi >= len(rows_box):
            continue
        _lbl, aucs, _c = rows_box[xi]
        if not aucs:
            continue
        ymax = max(aucs)
        ax.text(xi, min(ymax + 0.02, 1), "*",
                color=color, ha="center", va="bottom",
                fontsize=14, fontweight="bold")


def _write_box_counts_md(
    out_path: Path, *, pca: bool = False,
) -> None:
    """Markdown table of per-fold sample sizes for each box in plot4 figures."""
    all_targets = TARGETS_CS + TARGETS_ETIO
    template = _build_rows_box(all_targets[0][1], pca=pca)
    col_labels = [r[0] for r in template]

    lines = [
        f"# plot4 — Per-target boxplot sample sizes ({'PCA' if pca else 'non-PCA'})",
        "",
        "Number of per-fold AUC values shown in each box of",
        f"`plot4_combined_{'pca' if pca else 'nonpca'}_box_*.png`. "
        "Empty cells indicate no data on disk.",
        "",
        "| Target | " + " | ".join(col_labels) + " |",
        "|" + "|".join(["---"] * (len(col_labels) + 1)) + "|",
    ]
    for tname, tsegs in all_targets:
        rows = _build_rows_box(tsegs, pca=pca)
        cells = [str(len(r[1])) if r[1] else "" for r in rows]
        lines.append(f"| {tname} | " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[statistics_plot4_targets] wrote {out_path}")


def plot_4x2_box_annotated(
    annots_by_target: dict[str, dict[int, str]],
    out_path: Path,
    test_label: str,
    pca: bool = False,
) -> None:
    """4×2 boxplot+swarm figure across all 8 targets (TARGETS_CS+TARGETS_ETIO).

    Annotation columns follow the existing convention: index ``2 + 3*i``
    (FM+DK) gets a blue ``*`` and ``3 + 3*i`` (FM Res) gets a red ``*`` when
    the corresponding test is significant.
    """
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharey=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows_box = _build_rows_box(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_box_annotated(ax, rows_box, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=15)

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color="blue", markersize=11,
                   linestyle="",
                   label=f"FM+DK vs FM  (p < 0.05, {test_label})"),
            Line2D([], [], marker="$*$", color="red", markersize=11,
                   linestyle="",
                   label=f"FM vs FM Res  (p < 0.05, {test_label})"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[statistics_plot4_targets] wrote {out_path}")


# ── Nadeau-Bengio (two-sided) annotations: FM+DK vs FM, FM vs FM Res ──────────

def _fm_base_path(
    model: str, target_segs: list[str], pca: bool = False,
) -> Path | None:
    emb = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    p = (BASE / model / "doc_patients" / emb / Path(*target_segs)
         / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    return p if p.is_file() else None


def _fm_combined_path(
    model: str, target_segs: list[str], pca: bool = False,
) -> Path | None:
    emb = "EMBEDDING_DK_COMBINED_FM_PCA" if pca else "EMBEDDING_DK_COMBINED"
    base = BASE / model / "doc_patients" / emb
    if pca:
        base = base / "pca27"
    for sub in ("cv_nested_matched", "cv_nested", "nested_cv_repeated"):
        p = base / Path(*target_segs) / sub / "best_clf" / "classification_results.json"
        if p.is_file():
            return p
    return None


def _fm_res_path(
    model: str, target_segs: list[str], pca: bool = False,
) -> Path | None:
    pca_label = "pca27" if pca else "no_pca"
    p = (BASE / "RES_NO_LEAKAGE" / "cv_nested" / pca_label / "best_clf"
         / model / Path(*target_segs) / "results.json")
    return p if p.is_file() else None


def _fm_manifest_path(
    model: str, target_segs: list[str], pca: bool = False,
) -> Path | None:
    emb = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    p = (BASE / model / "doc_patients" / emb / Path(*target_segs)
         / "nested_cv_repeated" / "cv_split_manifest.json")
    return p if p.is_file() else None


def _nb_two_sided_p(
    path_a: Path, path_b: Path, manifest_path: Path | None,
) -> float:
    """Nadeau-Bengio corrected two-sided p for ``mean(a-b) ≠ 0``.

    Pairs the two files by ``(repeat, fold)``. Returns ``1.0`` if the test
    is impossible (missing pair, zero variance, fewer than two folds).
    """
    paired = paired_per_fold_aucs(path_a, path_b)
    if paired is None:
        return 1.0
    a, b = paired
    diffs = np.asarray(a) - np.asarray(b)
    n_train, n_test = infer_n_train_test(path_a, manifest_path=manifest_path)
    _t, p_two = corrected_resampled_ttest(diffs, n_train, n_test)
    return 1.0 if (np.isnan(p_two)) else float(p_two)


def _compute_annotations_corrected(
    target_list: list[tuple[str, list[str]]],
    *,
    pca: bool = False,
    use_fdr: bool = False,
    alpha: float = ALPHA,
) -> dict[str, dict[int, str]]:
    """Two NB-corrected two-sided tests per FM × target.

    Test 1 — FM+DK vs FM (pairs FM+DK and FM by ``(repeat, fold)``).
    Test 2 — FM    vs FM Res.

    Annotation columns mirror :func:`_compute_annotations`: blue ``*`` at
    column ``2 + 3*i`` (FM+DK) and red ``*`` at column ``3 + 3*i`` (FM Res).
    ``use_fdr`` applies BH-FDR across all tests in the figure.
    """
    p_comb: dict[tuple, float] = {}
    p_res:  dict[tuple, float] = {}

    for tname, tsegs in target_list:
        for i, model in enumerate(FM_MODELS):
            fm_path   = _fm_base_path(model, tsegs, pca)
            comb_path = _fm_combined_path(model, tsegs, pca)
            res_path  = _fm_res_path(model, tsegs, pca)
            mani_fm   = _fm_manifest_path(model, tsegs, pca)

            if fm_path is not None and comb_path is not None:
                p_comb[(tname, i)] = _nb_two_sided_p(comb_path, fm_path, mani_fm)
            if fm_path is not None and res_path is not None:
                p_res[(tname, i)]  = _nb_two_sided_p(fm_path, res_path, mani_fm)

    if use_fdr and (p_comb or p_res):
        comb_keys = list(p_comb.keys())
        res_keys  = list(p_res.keys())
        all_p = [p_comb[k] for k in comb_keys] + [p_res[k] for k in res_keys]
        if all_p:
            p_adj = stats.false_discovery_control(all_p, method="bh")
            n_c = len(comb_keys)
            p_comb = {k: float(p_adj[j])     for j, k in enumerate(comb_keys)}
            p_res  = {k: float(p_adj[n_c+j]) for j, k in enumerate(res_keys)}

    annots_by_target: dict[str, dict[int, str]] = {}
    for tname, _ in target_list:
        annots: dict[int, str] = {}
        for i in range(len(FM_MODELS)):
            if p_comb.get((tname, i), 1.0) < alpha:
                annots[2 + 3 * i] = "blue"
            if p_res.get((tname, i), 1.0) < alpha:
                annots[3 + 3 * i] = "red"
        annots_by_target[tname] = annots
    return annots_by_target


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    def _mean(aucs):
        return round(float(np.mean(aucs)), 4) if aucs else np.nan

    def _std(aucs):
        return round(float(np.std(aucs, ddof=1)), 4) if aucs and len(aucs) > 1 else np.nan

    def _r4(v):
        return round(v, 4) if not (isinstance(v, float) and np.isnan(v)) else np.nan

    def _r3(v):
        return round(v, 3) if not (isinstance(v, float) and np.isnan(v)) else np.nan

    for pca, mode_label in [(True, "PCA"), (False, "non-PCA")]:
        for model_key in FM_MODELS:
            display = FM_DISPLAY[model_key]
            for tname, tsegs in TARGETS:
                clf = _select_best_clf_for_fm(model_key, tsegs, pca)
                base_aucs     = load_fm_base(model_key, tsegs, pca, clf)
                combined_aucs = load_fm_combined(model_key, tsegs, pca, clf)
                res_aucs      = load_fm_res(model_key, tsegs, pca, clf)

                t1, p1  = _welch_one_tailed(combined_aucs or [], base_aucs or [], h1="a>b")
                t2, p2  = _welch_one_tailed(base_aucs or [], res_aucs or [], h1="a>b")
                u1, pu1 = _mwu_one_tailed(combined_aucs or [], base_aucs or [], h1="a>b")
                u2, pu2 = _mwu_one_tailed(base_aucs or [], res_aucs or [], h1="a>b")

                rows.append({
                    "Mode":               mode_label,
                    "Model":              display,
                    "Condition":          tname,
                    "n_folds_FM":         len(base_aucs)     if base_aucs     else np.nan,
                    "mean_FM":            _mean(base_aucs),
                    "std_FM":             _std(base_aucs),
                    "n_folds_FM+DK":      len(combined_aucs) if combined_aucs else np.nan,
                    "mean_FM+DK":         _mean(combined_aucs),
                    "std_FM+DK":          _std(combined_aucs),
                    "n_folds_Res":        len(res_aucs)      if res_aucs      else np.nan,
                    "mean_Res":           _mean(res_aucs),
                    "std_Res":            _std(res_aucs),
                    # t-test
                    "t_combined_vs_FM":   _r3(t1),
                    "p_combined_above":   _r4(p1),
                    "sig_combined_above": _stars(p1),
                    "t_FM_vs_Res":        _r3(t2),
                    "p_res_below":        _r4(p2),
                    "sig_res_below":      _stars(p2),
                    # Wilcoxon
                    "u_combined_vs_FM":   _r3(u1),
                    "pu_combined_above":  _r4(pu1),
                    "sigu_combined_above": _stars(pu1),
                    "u_FM_vs_Res":        _r3(u2),
                    "pu_res_below":       _r4(pu2),
                    "sigu_res_below":     _stars(pu2),
                })

    df = pd.DataFrame(rows)

    # ── FDR corrections ────────────────────────────────────────────────────────
    for raw_col, fdr_col, sig_col in [
        ("p_combined_above",  "p_combined_above_fdr",  "sig_combined_above_fdr"),
        ("p_res_below",       "p_res_below_fdr",       "sig_res_below_fdr"),
        ("pu_combined_above", "pu_combined_above_fdr", "sigu_combined_above_fdr"),
        ("pu_res_below",      "pu_res_below_fdr",      "sigu_res_below_fdr"),
    ]:
        df[fdr_col] = _apply_fdr_to_series(df[raw_col]).round(4)
        df[sig_col] = df[fdr_col].apply(_stars)

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "statistics_plot4_targets.csv"
    df.to_csv(csv_path, index=False)
    print(f"[statistics_plot4_targets] wrote {csv_path}")

    # ── HTML ───────────────────────────────────────────────────────────────────
    html_path = OUT_DIR / "statistics_plot4_targets.html"

    def _color_sig(val: str) -> str:
        return {
            "***": "background-color:#1a7a1a; color:white",
            "**":  "background-color:#4caf50; color:white",
            "*":   "background-color:#a5d6a7",
            "†":   "background-color:#e8f5e9",
        }.get(str(val), "")

    sig_cols = [c for c in df.columns if c.startswith("sig")]
    float_cols = {c: "{:.4f}" for c in df.select_dtypes("float").columns if c not in sig_cols}
    stat_cols  = {c: "{:.3f}" for c in ["t_combined_vs_FM", "t_FM_vs_Res",
                                          "u_combined_vs_FM", "u_FM_vs_Res"]}
    float_cols.update(stat_cols)

    styled = (
        df.style
        .map(_color_sig, subset=sig_cols)
        .format(float_cols, na_rep="—")
        .set_caption(
            "Random-forest nested-CV AUC — PCA and non-PCA modes. "
            "T-test: independent Welch one-tailed. Wilcoxon: Mann-Whitney U one-tailed. "
            "FDR: Benjamini-Hochberg across all rows per p-value family. "
            "Test 1: H₁ FM+DK > FM. Test 2: H₁ FM > FM_Res. "
            "Stars: *** p<0.001 · ** p<0.01 · * p<0.05 · † p<0.1 · ns."
        )
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
    print(f"[statistics_plot4_targets] wrote {html_path}")

    # ── Console summaries ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Full table:")
    print(df.to_string(index=False))

    for mode_label in ["PCA", "non-PCA"]:
        sub = df[df["Mode"] == mode_label]
        for label, col in [
            (f"[{mode_label}] FM+DK > FM — t-test",         "p_combined_above"),
            (f"[{mode_label}] FM+DK > FM — t-test FDR",     "p_combined_above_fdr"),
            (f"[{mode_label}] FM+DK > FM — Wilcoxon",       "pu_combined_above"),
            (f"[{mode_label}] FM+DK > FM — Wilcoxon FDR",   "pu_combined_above_fdr"),
            (f"[{mode_label}] FM > Res   — t-test",         "p_res_below"),
            (f"[{mode_label}] FM > Res   — t-test FDR",     "p_res_below_fdr"),
            (f"[{mode_label}] FM > Res   — Wilcoxon",       "pu_res_below"),
            (f"[{mode_label}] FM > Res   — Wilcoxon FDR",   "pu_res_below_fdr"),
        ]:
            sig = sub[sub[col] < ALPHA]
            print(f"\n{'='*65}\n{label} (p < 0.05):")
            print("  (none)" if sig.empty else
                  sig[["Model", "Condition", col]].to_string(index=False))

    # ── 16 annotated plots (2 layouts × 2 modes × 4 tests) ───────────────────
    test_variants = [
        ("ttest",        "T-test (Welch)"),
        ("ttest_fdr",    "T-test + FDR (BH)"),
        ("wilcoxon",     "Wilcoxon (Mann-Whitney U)"),
        ("wilcoxon_fdr", "Wilcoxon + FDR (BH)"),
    ]
    all_targets = TARGETS_CS + TARGETS_ETIO

    for pca, mode_tag in [(False, "nonpca"), (True, "pca")]:
        for test_key, test_label in test_variants:
            # 2×2 etiology plots
            annots_etio = _compute_annotations(TARGETS_ETIO, pca=pca,
                                               test_key=test_key, alpha=ALPHA)
            plot_2x2_annotated(TARGETS_ETIO, pca=pca,
                               annots_by_target=annots_etio,
                               out_path=OUT_DIR / f"plot4_etiology_{mode_tag}_annotated_{test_key}.png",
                               test_label=test_label)

            # 4×2 combined plots (all 8 targets)
            annots_all = _compute_annotations(all_targets, pca=pca,
                                              test_key=test_key, alpha=ALPHA)
            plot_4x2_annotated(pca=pca,
                               annots_by_target=annots_all,
                               out_path=OUT_DIR / f"plot4_combined_{mode_tag}_annotated_{test_key}.png",
                               test_label=test_label)

    # ── 3 boxplot+swarmplot 4×2 figures (non-PCA only) ────────────────────────
    # (1) Welch one-tailed + FDR (boxplot version of the existing
    #     plot4_combined_nonpca_annotated_ttest_fdr.png).
    annots_t_fdr = _compute_annotations(all_targets, pca=False,
                                        test_key="ttest_fdr", alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_t_fdr,
        OUT_DIR / "plot4_combined_nonpca_box_ttest_fdr.png",
        test_label="Welch one-tailed + FDR (BH)",
        pca=False,
    )

    # (2) Nadeau-Bengio two-sided, no FDR.
    annots_nb = _compute_annotations_corrected(all_targets, pca=False,
                                               use_fdr=False, alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_nb,
        OUT_DIR / "plot4_combined_nonpca_box_corrected.png",
        test_label="Nadeau-Bengio two-sided",
        pca=False,
    )

    # (3) Nadeau-Bengio two-sided + FDR (BH).
    annots_nb_fdr = _compute_annotations_corrected(all_targets, pca=False,
                                                   use_fdr=True, alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_nb_fdr,
        OUT_DIR / "plot4_combined_nonpca_box_corrected_fdr.png",
        test_label="Nadeau-Bengio two-sided + FDR (BH)",
        pca=False,
    )

    _write_box_counts_md(BASE / "plot4_combined_box_counts.md", pca=False)


if __name__ == "__main__":
    main()
