#!/usr/bin/env python3
"""Statistical tests for the leakage-free residualization (RES_NO_LEAKAGE).

Mirrors statistics_plot4_targets.py exactly, but the "Res" column reads from
RES_NO_LEAKAGE (fold-internal FM-embedding standardization) instead of
RESIDUALIZATION_DIM.

Four test variants per row:
  t-test / t-test + FDR correction / Wilcoxon (MWU) / Wilcoxon + FDR correction

  Test 1 | FM+DK above FM:  H1: mean(FM+DK) > mean(FM)
  Test 2 | FM Res below FM: H1: mean(FM)    > mean(FM_Res)

Outputs (data/benchmark_results/new_results/PLOTS/):
  statistics_plot4_noleak_targets.csv
  statistics_plot4_noleak_targets.html
  plot4_noleak_etiology_{nonpca|pca}_annotated_{ttest|ttest_fdr|wilcoxon|wilcoxon_fdr}.png
  plot4_noleak_combined_{nonpca|pca}_annotated_{ttest|ttest_fdr|wilcoxon|wilcoxon_fdr}.png
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

BASE    = Path(os.environ.get(
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
    ("CRS Diagnostic",  ["crs"]),
    ("6m Improved",  ["cs_6m",  "binary_improvement"]),
    ("1y Improved",  ["cs_1y",  "binary_improvement"]),
    ("2y Improved",  ["cs_2y",  "binary_improvement"]),
    ("Delay (UWS)",     ["etiology",      "vs_only"]),
    ("Delay (MCS)",    ["etiology",      "mcs_only"]),
    ("Etiology (UWS)",  ["etiology_code", "vs_only"]),
    ("Etiology (MCS)", ["etiology_code", "mcs_only"]),
]

TARGETS_CS = [
    ("CRS Diagnostic", ["crs"]),
    ("6m Improved",  ["cs_6m",  "binary_improvement"]),
    ("1y Improved",  ["cs_1y",  "binary_improvement"]),
    ("2y Improved",  ["cs_2y",  "binary_improvement"]),
]
TARGETS_ETIO = [
    ("Delay (UWS)",     ["etiology",      "vs_only"]),
    ("Delay (MCS)",    ["etiology",      "mcs_only"]),
    ("Etiology (UWS)",  ["etiology_code", "vs_only"]),
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


# ---------------------------------------------------------------------------
# Fold-AUC loaders
# ---------------------------------------------------------------------------


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
    """Read fold-level AUCs from RES_NO_LEAKAGE (leakage-free residualization).

    Prefers the merged ``cv_nested/{pca}/best_clf/{model}/{target}/results.json``
    (20-repeat aggregate) when available; otherwise falls back to the per-target
    multi-model file (one shard's worth of folds)."""
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


# ---------------------------------------------------------------------------
# Mean/std loaders (for plotting)
# ---------------------------------------------------------------------------


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
    """Read mean/std from RES_NO_LEAKAGE."""
    path = (BASE / "RES_NO_LEAKAGE" / "pca27" / Path(*target_segs) / "results.json"
            if pca else BASE / "RES_NO_LEAKAGE" / "no_pca" / Path(*target_segs) / "results.json")
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception:
        return None
    entry = d.get("results", {}).get(model, {}).get("classifiers", {}).get(classifier)
    if not entry:
        return None
    m = entry.get("mean_auc")
    s = entry.get("std_auc")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    return float(m), float(s) if s is not None else 0.0


def _select_best_clf_for_fm(model: str, target_segs: list[str], pca: bool) -> str:
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
        clf   = _select_best_clf_for_fm(model, target_segs, pca)
        for suffix, embedding_kind in variants:
            label = f"{model} {suffix}".strip()
            if embedding_kind is None:
                r = _load_ms_fm_res(model, target_segs, pca=pca, classifier=clf)
            else:
                r = _load_ms_fm_embedding(model, embedding_kind, target_segs, clf)
            rows.append((label, r[0] if r else None, r[1] if r else None, color))
    return rows


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < ALPHA: return "*"
    if p < 0.1:   return "†"
    return "ns"


def _welch_one_tailed(a: list[float], b: list[float], h1: str) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    if h1 == "a>b":
        p = p_two / 2 if t > 0 else 1.0 - p_two / 2
    else:
        p = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p)


def _mwu_one_tailed(a: list[float], b: list[float], h1: str) -> tuple[float, float]:
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    alt = "greater" if h1 == "a>b" else "less"
    try:
        u, p = stats.mannwhitneyu(a, b, alternative=alt)
        return float(u), float(p)
    except ValueError:
        return np.nan, np.nan


def _welch_p(a: list[float], b: list[float], h1: str) -> float:
    _, p = _welch_one_tailed(a, b, h1)
    return p if not np.isnan(p) else 1.0


def _mwu_p(a: list[float], b: list[float], h1: str) -> float:
    _, p = _mwu_one_tailed(a, b, h1)
    return p if not np.isnan(p) else 1.0


# ── Corrected resampled t-test (Nadeau & Bengio 2003) ───────────────────────


def _best_clf_path_fm(
    model: str, target_segs: list[str], pca: bool
) -> Path | None:
    """Path to FM-only ``nested_cv_repeated/best_clf/classification_results.json``."""
    emb = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    p = (BASE / model / "doc_patients" / emb / Path(*target_segs)
         / "nested_cv_repeated" / "best_clf" / "classification_results.json")
    return p if p.is_file() else None


def _best_clf_path_combined(
    model: str, target_segs: list[str], pca: bool
) -> Path | None:
    emb = "EMBEDDING_DK_COMBINED_FM_PCA" if pca else "EMBEDDING_DK_COMBINED"
    base = BASE / model / "doc_patients" / emb
    if pca:
        base_pca = base / "pca27"
        candidates = [
            base_pca / Path(*target_segs) / sub / "best_clf" / "classification_results.json"
            for sub in ("cv_nested_matched", "cv_nested", "nested_cv_repeated")
        ]
    else:
        candidates = [
            base / Path(*target_segs) / sub / "best_clf" / "classification_results.json"
            for sub in ("cv_nested_matched", "cv_nested", "nested_cv_repeated")
        ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _best_clf_path_res(
    model: str, target_segs: list[str], pca: bool
) -> Path | None:
    pca_label = "pca27" if pca else "no_pca"
    res_root = BASE / "RES_NO_LEAKAGE"
    candidates = [
        res_root / sub / pca_label / "best_clf" / model
        / Path(*target_segs) / "results.json"
        for sub in ("cv_nested_matched", "cv_nested")
    ] + [
        # Some res output layouts omit the per-target subdir.
        res_root / sub / pca_label / "best_clf" / model / "results.json"
        for sub in ("cv_nested_matched", "cv_nested")
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _manifest_path_fm(
    model: str, target_segs: list[str], pca: bool
) -> Path | None:
    emb = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    p = (BASE / model / "doc_patients" / emb / Path(*target_segs)
         / "nested_cv_repeated" / "cv_split_manifest.json")
    return p if p.is_file() else None


def _corrected_test_paired(
    path_a: Path | None,
    path_b: Path | None,
    manifest_path: Path | None,
    h1: str = "a>b",
    two_sided: bool = False,
) -> tuple[float, float, int, int, int] | None:
    """Run paired Nadeau-Bengio corrected t-test on two best_clf JSONs.

    Returns ``(t, p, n_paired, n_train, n_test)`` or ``None``. When
    ``two_sided=True`` the canonical two-sided p is returned and ``h1`` is
    ignored.
    """
    if path_a is None or path_b is None:
        return None
    paired = paired_per_fold_aucs(path_a, path_b)
    if paired is None:
        return None
    a, b = paired
    diffs = np.asarray(a) - np.asarray(b)
    n_train, n_test = infer_n_train_test(path_a, manifest_path=manifest_path)
    t, p_two = corrected_resampled_ttest(diffs, n_train, n_test)
    if np.isnan(t):
        return None
    if two_sided:
        p = p_two
    elif h1 == "a>b":
        p = p_two / 2 if t > 0 else 1.0 - p_two / 2
    else:
        p = p_two / 2 if t < 0 else 1.0 - p_two / 2
    return float(t), float(p), len(diffs), n_train, n_test


def corrected_p_combined_vs_fm_two_sided(
    model: str, target_segs: list[str], pca: bool
) -> float:
    res = _corrected_test_paired(
        _best_clf_path_combined(model, target_segs, pca),
        _best_clf_path_fm(model, target_segs, pca),
        _manifest_path_fm(model, target_segs, pca),
        two_sided=True,
    )
    return res[1] if res is not None else 1.0


def corrected_p_fm_vs_res_two_sided(
    model: str, target_segs: list[str], pca: bool
) -> float:
    res = _corrected_test_paired(
        _best_clf_path_fm(model, target_segs, pca),
        _best_clf_path_res(model, target_segs, pca),
        _manifest_path_fm(model, target_segs, pca),
        two_sided=True,
    )
    return res[1] if res is not None else 1.0


def corrected_full(
    model: str, target_segs: list[str], pca: bool, kind: str,
) -> tuple[float, float, int, int, int] | None:
    """``kind in {'comb_vs_fm', 'fm_vs_res'}`` | full result tuple.

    Uses the canonical Nadeau-Bengio two-sided p-value.
    """
    if kind == "comb_vs_fm":
        return _corrected_test_paired(
            _best_clf_path_combined(model, target_segs, pca),
            _best_clf_path_fm(model, target_segs, pca),
            _manifest_path_fm(model, target_segs, pca),
            two_sided=True,
        )
    elif kind == "fm_vs_res":
        return _corrected_test_paired(
            _best_clf_path_fm(model, target_segs, pca),
            _best_clf_path_res(model, target_segs, pca),
            _manifest_path_fm(model, target_segs, pca),
            two_sided=True,
        )
    raise ValueError(f"unknown kind: {kind}")


def _apply_fdr(p_series: pd.Series) -> pd.Series:
    valid_idx = p_series.dropna().index
    if len(valid_idx) == 0:
        return p_series.copy()
    p_adj = stats.false_discovery_control(p_series[valid_idx], method="bh")
    out = p_series.copy()
    out[valid_idx] = p_adj
    return out


# ---------------------------------------------------------------------------
# Annotation computation (4 test variants)
# ---------------------------------------------------------------------------


def _compute_annotations(
    target_list: list[tuple[str, list[str]]],
    pca: bool,
    test_key: str,
    alpha: float = ALPHA,
) -> dict[str, dict[int, str]]:
    use_wilcoxon  = "wilcoxon"  in test_key
    use_corrected = "corrected" in test_key
    use_fdr       = "fdr"       in test_key
    if use_corrected:
        p_fn = None  # handled separately below
    elif use_wilcoxon:
        p_fn = _mwu_p
    else:
        p_fn = _welch_p

    p_comb: dict[tuple, float] = {}
    p_res:  dict[tuple, float] = {}

    for tname, tsegs in target_list:
        for i, model in enumerate(FM_MODELS):
            if use_corrected:
                p_comb[(tname, i)] = corrected_p_combined_vs_fm_two_sided(model, tsegs, pca)
                p_res[(tname, i)]  = corrected_p_fm_vs_res_two_sided(model, tsegs, pca)
                continue
            clf  = _select_best_clf_for_fm(model, tsegs, pca)
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
            n_c   = len(comb_keys)
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


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------


def _plot_panel(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
) -> None:
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))

    gi, run_start = 0, 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(run_start - 0.5, i - 0.5, color="gray", alpha=0.1, zorder=0)
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
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=12)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=18)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _plot_panel_annotated(ax, rows, title, annots):
    _plot_panel(ax, rows, title)
    for xi, color in annots.items():
        if xi < len(rows):
            _lbl, m, s, _c = rows[xi]
            if m is not None:
                ax.text(xi, m + (s or 0.0) + 0.04, "*",
                        color=color, ha="center", va="bottom",
                        fontsize=14, fontweight="bold")


def _legend_handles(test_label: str, two_sided: bool = False):
    rel_comb = "FM+DK ≠ FM" if two_sided else "FM+DK > FM"
    rel_res  = "FM ≠ FM Res" if two_sided else "FM > FM Res"
    return [
        Line2D([], [], marker="$*$", color="blue", markersize=11, linestyle="",
               label=f"{rel_comb}  (p < 0.05, {test_label})"),
        Line2D([], [], marker="$*$", color="red",  markersize=11, linestyle="",
               label=f"{rel_res}  (p < 0.05, {test_label})"),
    ]


def plot_2x2_annotated(
    target_list, pca, annots_by_target, out_path: Path, test_label: str,
    two_sided: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharey=True, sharex=True)
    for ax, (tname, tsegs) in zip(axes.flat, target_list):
        rows   = build_rows(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_annotated(ax, rows, tname, annots)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=18)
    fig.legend(handles=_legend_handles(test_label, two_sided=two_sided),
               loc="lower right", fontsize=12, framealpha=0.9, handlelength=1.2,
               bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stats_plot4_noleak] wrote {out_path}")


def plot_4x2_annotated(
    pca, annots_by_target, out_path: Path, test_label: str,
    two_sided: bool = False,
) -> None:
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)
    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows   = build_rows(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_annotated(ax, rows, tname, annots)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=18)
    fig.legend(handles=_legend_handles(test_label, two_sided=two_sided),
               loc="lower right", fontsize=12, framealpha=0.9, handlelength=1.2,
               bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stats_plot4_noleak] wrote {out_path}")


# ---------------------------------------------------------------------------
# Boxplot panels (per-fold AUC distributions)
# ---------------------------------------------------------------------------


BOX_LINECOLOR = "#444444"
BOX_FILL_ALPHA = 0.65


def _apply_box_alpha(ax, alpha: float = BOX_FILL_ALPHA) -> None:
    for patch in ax.patches:
        try:
            patch.set_alpha(alpha)
        except Exception:
            pass


def _load_dk_folds(target_segs: list[str]) -> list[float] | None:
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
    """Per-fold list version of :func:`build_rows`. Same column order:
    DK, then per FM ``(FM, FM+DK, FM Res)`` — Res reads from RES_NO_LEAKAGE."""
    rows: list[tuple[str, list[float] | None, str]] = []
    rows.append(("DK", _load_dk_folds(target_segs), DK_COLOR))
    for model in FM_MODELS:
        color = FM_COLORS[model]
        clf = _select_best_clf_for_fm(model, target_segs, pca)
        rows.append((f"{model}",      load_fm_base(model, target_segs, pca, clf), color))
        rows.append((f"{model} + DK", load_fm_combined(model, target_segs, pca, clf), color))
        rows.append((f"{model} Res",  load_fm_res(model, target_segs, pca, clf), color))
    return rows


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
                           color="gainsboro", alpha=0.001, zorder=0)
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
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=18)
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


def plot_4x2_box_annotated(
    annots_by_target: dict[str, dict[int, str]],
    out_path: Path,
    test_label: str,
    pca: bool = False,
    two_sided: bool = False,
) -> None:
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharey=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows_box = _build_rows_box(tsegs, pca=pca)
        annots = annots_by_target.get(tname, {})
        _plot_panel_box_annotated(ax, rows_box, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=18)

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    rel_comb = "FM+DK ≠ FM" if two_sided else "FM+DK > FM"
    rel_res  = "FM ≠ FM Res" if two_sided else "FM > FM Res"
    fig.legend(
        handles=[
            Line2D([], [], marker="$*$", color="blue", markersize=11,
                   linestyle="",
                   label=f"{rel_comb}  (p < 0.05, {test_label})"),
            Line2D([], [], marker="$*$", color="red", markersize=11,
                   linestyle="",
                   label=f"{rel_res}  (p < 0.05, {test_label})"),
        ],
        loc="lower right", fontsize=12, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[stats_plot4_noleak] wrote {out_path}")


def _write_box_counts_md(
    out_path: Path, *, pca: bool = False,
) -> None:
    """Markdown table of per-fold sample sizes for each box in noleak figures."""
    all_targets = TARGETS_CS + TARGETS_ETIO
    template = _build_rows_box(all_targets[0][1], pca=pca)
    col_labels = [r[0] for r in template]

    lines = [
        f"# plot4 (no-leakage) — Per-target boxplot sample sizes "
        f"({'PCA' if pca else 'non-PCA'})",
        "",
        "Number of per-fold AUC values shown in each box of",
        f"`plot4_noleak_combined_{'pca' if pca else 'nonpca'}_box_*.png`. "
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
    print(f"[stats_plot4_noleak] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []

    def _mean(aucs): return round(float(np.mean(aucs)), 4) if aucs else np.nan
    def _std(aucs):  return round(float(np.std(aucs, ddof=1)), 4) if aucs and len(aucs) > 1 else np.nan
    def _r4(v): return round(v, 4) if not (isinstance(v, float) and np.isnan(v)) else np.nan
    def _r3(v): return round(v, 3) if not (isinstance(v, float) and np.isnan(v)) else np.nan

    for pca, mode_label in [(True, "PCA"), (False, "non-PCA")]:
        for model_key in FM_MODELS:
            display = FM_DISPLAY[model_key]
            for tname, tsegs in TARGETS:
                clf           = _select_best_clf_for_fm(model_key, tsegs, pca)
                base_aucs     = load_fm_base(model_key, tsegs, pca, clf)
                combined_aucs = load_fm_combined(model_key, tsegs, pca, clf)
                res_aucs      = load_fm_res(model_key, tsegs, pca, clf)

                t1, p1  = _welch_one_tailed(combined_aucs or [], base_aucs or [], h1="a>b")
                t2, p2  = _welch_one_tailed(base_aucs or [], res_aucs or [], h1="a>b")
                u1, pu1 = _mwu_one_tailed(combined_aucs or [], base_aucs or [], h1="a>b")
                u2, pu2 = _mwu_one_tailed(base_aucs or [], res_aucs or [], h1="a>b")

                # Nadeau-Bengio corrected resampled t-test (paired by repeat,fold)
                cf_comb = corrected_full(model_key, tsegs, pca, "comb_vs_fm")
                cf_res  = corrected_full(model_key, tsegs, pca, "fm_vs_res")
                if cf_comb is not None:
                    tc1, pc1, n_pair_c1, n_train_c, n_test_c = cf_comb
                else:
                    tc1 = pc1 = n_pair_c1 = n_train_c = n_test_c = np.nan
                if cf_res is not None:
                    tc2, pc2, n_pair_c2, _ntrc2, _ntec2 = cf_res
                else:
                    tc2 = pc2 = n_pair_c2 = np.nan

                rows_out.append({
                    "Mode":                mode_label,
                    "Model":               display,
                    "Condition":           tname,
                    "Best_CLF":            clf,
                    "n_folds_FM":          len(base_aucs)     if base_aucs     else np.nan,
                    "mean_FM":             _mean(base_aucs),
                    "std_FM":              _std(base_aucs),
                    "n_folds_FM+DK":       len(combined_aucs) if combined_aucs else np.nan,
                    "mean_FM+DK":          _mean(combined_aucs),
                    "std_FM+DK":           _std(combined_aucs),
                    "n_folds_Res_noleak":  len(res_aucs)      if res_aucs      else np.nan,
                    "mean_Res_noleak":     _mean(res_aucs),
                    "std_Res_noleak":      _std(res_aucs),
                    "t_combined_vs_FM":    _r3(t1),
                    "p_combined_above":    _r4(p1),
                    "sig_combined_above":  _stars(p1),
                    "t_FM_vs_Res":         _r3(t2),
                    "p_res_below":         _r4(p2),
                    "sig_res_below":       _stars(p2),
                    "u_combined_vs_FM":    _r3(u1),
                    "pu_combined_above":   _r4(pu1),
                    "sigu_combined_above": _stars(pu1),
                    "u_FM_vs_Res":         _r3(u2),
                    "pu_res_below":        _r4(pu2),
                    "sigu_res_below":      _stars(pu2),
                    # Corrected resampled t-test (Nadeau & Bengio 2003) | two-sided
                    "tc_combined_vs_FM":     _r3(tc1),
                    "pc_combined_vs_fm":     _r4(pc1),
                    "sigc_combined_vs_fm":   _stars(pc1),
                    "tc_FM_vs_Res":          _r3(tc2),
                    "pc_fm_vs_res":          _r4(pc2),
                    "sigc_fm_vs_res":        _stars(pc2),
                    "n_paired_corr_combined": int(n_pair_c1) if not (isinstance(n_pair_c1, float) and np.isnan(n_pair_c1)) else np.nan,
                    "n_paired_corr_res":      int(n_pair_c2) if not (isinstance(n_pair_c2, float) and np.isnan(n_pair_c2)) else np.nan,
                    "n_train_corr":           int(n_train_c) if not (isinstance(n_train_c, float) and np.isnan(n_train_c)) else np.nan,
                    "n_test_corr":            int(n_test_c) if not (isinstance(n_test_c, float) and np.isnan(n_test_c)) else np.nan,
                })

    df = pd.DataFrame(rows_out)

    for raw_col, fdr_col, sig_col in [
        ("p_combined_above",   "p_combined_above_fdr",   "sig_combined_above_fdr"),
        ("p_res_below",        "p_res_below_fdr",        "sig_res_below_fdr"),
        ("pu_combined_above",  "pu_combined_above_fdr",  "sigu_combined_above_fdr"),
        ("pu_res_below",       "pu_res_below_fdr",       "sigu_res_below_fdr"),
        ("pc_combined_vs_fm",  "pc_combined_vs_fm_fdr",  "sigc_combined_vs_fm_fdr"),
        ("pc_fm_vs_res",       "pc_fm_vs_res_fdr",       "sigc_fm_vs_res_fdr"),
    ]:
        df[fdr_col] = _apply_fdr(df[raw_col]).round(4)
        df[sig_col] = df[fdr_col].apply(_stars)

    csv_path = OUT_DIR / "statistics_plot4_noleak_targets.csv"
    df.to_csv(csv_path, index=False)
    print(f"[stats_plot4_noleak] wrote {csv_path}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_path = OUT_DIR / "statistics_plot4_noleak_targets.html"

    def _color_sig(val: str) -> str:
        return {
            "***": "background-color:#1a7a1a; color:white",
            "**":  "background-color:#4caf50; color:white",
            "*":   "background-color:#a5d6a7",
            "†":   "background-color:#e8f5e9",
        }.get(str(val), "")

    sig_cols  = [c for c in df.columns if c.startswith("sig")]
    float_fmt = {c: "{:.4f}" for c in df.select_dtypes("float").columns if c not in sig_cols}
    stat_fmt  = {c: "{:.3f}" for c in ["t_combined_vs_FM", "t_FM_vs_Res",
                                         "u_combined_vs_FM", "u_FM_vs_Res"]}
    float_fmt.update(stat_fmt)

    styled = (
        df.style
        .map(_color_sig, subset=sig_cols)
        .format(float_fmt, na_rep="")
        .set_caption(
            "Leakage-free residualization (RES_NO_LEAKAGE) | nested-CV AUC. "
            "Res = fold-internal FM-embedding standardization. "
            "T-test: independent one-tailed. Wilcoxon: Mann-Whitney U one-tailed. "
            "Nadeau-Bengio corrected resampled t-test: canonical two-sided. "
            "FDR: Benjamini-Hochberg. "
            "Test 1: H₁ FM+DK > FM (or FM+DK ≠ FM, corrected). "
            "Test 2: H₁ FM > FM_Res (or FM ≠ FM_Res, corrected). "
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
    print(f"[stats_plot4_noleak] wrote {html_path}")

    # ── Console summaries ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Full table:")
    print(df[["Mode", "Model", "Condition", "Best_CLF",
              "mean_FM", "mean_FM+DK", "mean_Res_noleak",
              "sig_combined_above", "sig_res_below"]].to_string(index=False))

    for mode_label in ["PCA", "non-PCA"]:
        sub = df[df["Mode"] == mode_label]
        for label, col in [
            (f"[{mode_label}] FM+DK > FM | t-test",       "p_combined_above"),
            (f"[{mode_label}] FM+DK > FM | t-test FDR",   "p_combined_above_fdr"),
            (f"[{mode_label}] FM+DK > FM | Wilcoxon",     "pu_combined_above"),
            (f"[{mode_label}] FM+DK > FM | Wilcoxon FDR", "pu_combined_above_fdr"),
            (f"[{mode_label}] FM > Res   | t-test",       "p_res_below"),
            (f"[{mode_label}] FM > Res   | t-test FDR",   "p_res_below_fdr"),
            (f"[{mode_label}] FM > Res   | Wilcoxon",     "pu_res_below"),
            (f"[{mode_label}] FM > Res   | Wilcoxon FDR", "pu_res_below_fdr"),
            (f"[{mode_label}] FM+DK ≠ FM | Nadeau-Bengio (two-sided)",     "pc_combined_vs_fm"),
            (f"[{mode_label}] FM+DK ≠ FM | Nadeau-Bengio (two-sided) + FDR", "pc_combined_vs_fm_fdr"),
            (f"[{mode_label}] FM ≠ Res   | Nadeau-Bengio (two-sided)",     "pc_fm_vs_res"),
            (f"[{mode_label}] FM ≠ Res   | Nadeau-Bengio (two-sided) + FDR", "pc_fm_vs_res_fdr"),
        ]:
            sig = sub[sub[col] < ALPHA]
            print(f"\n{'='*65}\n{label} (p < 0.05):")
            print("  (none)" if sig.empty else
                  sig[["Model", "Condition", col]].to_string(index=False))

    # ── 16 annotated plots (2 layouts × 2 modes × 4 tests) ───────────────────
    test_variants = [
        ("ttest",         "T-test"),
        ("ttest_fdr",     "T-test + FDR correction"),
        ("wilcoxon",      "Wilcoxon (MWU)"),
        ("wilcoxon_fdr",  "Wilcoxon + FDR correction"),
        ("corrected",     "Nadeau-Bengio t-test (two-sided)"),
        ("corrected_fdr", "Nadeau-Bengio t-test (two-sided) + FDR correction"),
    ]
    all_targets = TARGETS_CS + TARGETS_ETIO

    for pca, mode_tag in [(False, "nonpca"), (True, "pca")]:
        for test_key, test_label in test_variants:
            two_sided = test_key.startswith("corrected")
            annots_etio = _compute_annotations(TARGETS_ETIO, pca=pca,
                                               test_key=test_key, alpha=ALPHA)
            plot_2x2_annotated(
                TARGETS_ETIO, pca=pca,
                annots_by_target=annots_etio,
                out_path=OUT_DIR / f"plot4_noleak_etiology_{mode_tag}_annotated_{test_key}.png",
                test_label=test_label,
                two_sided=two_sided,
            )

            annots_all = _compute_annotations(all_targets, pca=pca,
                                              test_key=test_key, alpha=ALPHA)
            plot_4x2_annotated(
                pca=pca,
                annots_by_target=annots_all,
                out_path=OUT_DIR / f"plot4_noleak_combined_{mode_tag}_annotated_{test_key}.png",
                test_label=test_label,
                two_sided=two_sided,
            )

    # ── 3 boxplot 4×2 figures (non-PCA only) ──────────────────────────────────
    # (1) T-test one-tailed + FDR (boxplot version of the existing
    #     plot4_noleak_combined_nonpca_annotated_ttest_fdr.png).
    annots_t_fdr = _compute_annotations(all_targets, pca=False,
                                        test_key="ttest_fdr", alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_t_fdr,
        OUT_DIR / "plot4_noleak_combined_nonpca_box_ttest_fdr.png",
        test_label="T-test + FDR correction",
        pca=False,
        two_sided=False,
    )

    # (2) Nadeau-Bengio two-sided, no FDR.
    annots_nb = _compute_annotations(all_targets, pca=False,
                                     test_key="corrected", alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_nb,
        OUT_DIR / "plot4_noleak_combined_nonpca_box_corrected.png",
        test_label="Nadeau-Bengio two-sided",
        pca=False,
        two_sided=True,
    )

    # (3) Nadeau-Bengio two-sided + FDR correction.
    annots_nb_fdr = _compute_annotations(all_targets, pca=False,
                                         test_key="corrected_fdr", alpha=ALPHA)
    plot_4x2_box_annotated(
        annots_nb_fdr,
        OUT_DIR / "plot4_noleak_combined_nonpca_box_corrected_fdr.png",
        test_label="Nadeau-Bengio two-sided + FDR correction",
        pca=False,
        two_sided=True,
    )

    _write_box_counts_md(BASE / "plot4_combined_no_leakage_box_counts.md",
                         pca=False)

    print("\n[stats_plot4_noleak] Done.")


if __name__ == "__main__":
    main()
