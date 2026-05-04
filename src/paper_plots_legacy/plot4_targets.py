#!/usr/bin/env python3
"""2×2 subplot variants of plot4 across multiple targets.

Each output figure is a 2×2 grid of plot4-style panels (Baseline + 5 FMs × 3
variants). Missing entries (no data on disk) are kept as blank ticks with a
dashed-gray marker so it's obvious which runs still need to be produced.

Target mapping (user label → on-disk path segments):

    crs              → ["crs"]
    cs_6m            → ["cs_6m", "binary"]
    cs_1y            → ["cs_1y", "binary"]
    cs_2y            → ["cs_2y", "binary"]
    delay_VS         → ["etiology",      "vs_only"]   (acute vs chronic)
    delay_MCS        → ["etiology",      "mcs_only"]
    etiology_VS      → ["etiology_code", "vs_only"]   (ANOXIA vs TBI)
    etiology_MCS     → ["etiology_code", "mcs_only"]

Outputs (data/benchmark_results/new_results/PLOTS/):

    plot4_targets_nonpca.png     2×2: {crs, cs_6m, cs_1y, cs_2y}   non-PCA
    plot4_targets_pca.png        2×2: {crs, cs_6m, cs_1y, cs_2y}   PCA
    plot4_etiology_nonpca.png    2×2: {delay_VS, delay_MCS,
                                       etiology_VS, etiology_MCS}  non-PCA
    plot4_etiology_pca.png       2×2: same etiology/delay targets  PCA

FM set: BIOT, EEGPT, LaBram, NeuroLM, CBraMod (same as plot4_eegpt).
Classifier: random_forest.
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
OUT_DIR = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]
FM_MODELS = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CBraMod"]

DK_COLOR = "#555555"
MISSING_COLOR = "#b0b0b0"
FM_COLORS = {
    "BIOT":    "#1f77b4",
    "EEGPT":   "#8c564b",
    "LaBram":  "#2ca02c",
    "NeuroLM": "#9467bd",
    "CBraMod": "#ff7f0e",
}

TARGETS_CS = [
    ("CRS Diagnostic",  ["crs"]),
    ("6m", ["cs_6m", "binary_improvement"]),
    ("1y", ["cs_1y", "binary_improvement"]),
    ("2y", ["cs_2y", "binary_improvement"]),
]
TARGETS_ETIO = [
    ("Delay (VS)",       ["etiology",      "vs_only"]),
    ("Delay (MCS)",     ["etiology",      "mcs_only"]),
    ("Etiology (VS)",  ["etiology_code", "vs_only"]),
    ("Etiology (MCS)",["etiology_code", "mcs_only"]),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


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


def load_baseline(target_segs: list[str]) -> tuple[float, float] | None:
    """DK baseline: prefer in-fold-selected repeated CV; fall back to legacy."""
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


def _load_repeated_best_clf_fm(
    model: str, embedding_kind: str, target_segs: list[str]
) -> tuple[float, float] | None:
    """Priority lookup: nested_cv_repeated/best_clf/ (in-fold classifier selection)."""
    base = (BASE / model / "doc_patients" / embedding_kind)
    if embedding_kind == "EMBEDDING_DK_COMBINED_FM_PCA":
        base = base / "pca27"
    # DK+FM combined: try cv_nested_matched first, then cv_nested, then repeated
    for subdir in ("cv_nested_matched", "cv_nested"):
        p = base / Path(*target_segs) / subdir / "best_clf" / "classification_results.json"
        r = _macro_auc(p)
        if r is not None:
            return r
    p = base / Path(*target_segs) / "nested_cv_repeated" / "best_clf" / "classification_results.json"
    return _macro_auc(p)


def load_fm_embedding(
    model: str, embedding_kind: str, target_segs: list[str], classifier: str
) -> tuple[float, float] | None:
    # Priority: in-fold selection from repeated CV.
    if classifier in ("best_clf_repeated", "best_clf"):
        r = _load_repeated_best_clf_fm(model, embedding_kind, target_segs)
        if r is not None:
            return r
    if embedding_kind == "EMBEDDING_DK_COMBINED_FM_PCA":
        primary = (
            BASE / model / "doc_patients" / embedding_kind / "pca27"
            / Path(*target_segs)
            / "nested_cv" / classifier / "classification_results.json"
        )
    else:
        primary = (
            BASE / model / "doc_patients" / embedding_kind / Path(*target_segs)
            / "nested_cv" / classifier / "classification_results.json"
        )
    result = _macro_auc(primary)
    if result is not None:
        return result
    fallback = (
        BASE / model / "doc_patients" / embedding_kind / model
        / "doc_patients" / embedding_kind / Path(*target_segs)
        / "nested_cv" / classifier / "classification_results.json"
    )
    return _macro_auc(fallback)


def _load_res(path: Path, model: str, classifier: str) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    entry = d.get("results", {}).get(model, {}).get("classifiers", {}).get(classifier)
    if not entry:
        return None
    m = entry.get("mean_auc")
    s = entry.get("std_auc")
    if m is None:
        return None
    return float(m), float(s) if s is not None else 0.0


def _load_res_no_leakage_best_clf(
    model: str, target_segs: list[str], pca: bool
) -> tuple[float, float] | None:
    """Try merged repeated path, then per-target multi-model file → best_clf."""
    pca_label = "pca27" if pca else "no_pca"
    rep = (BASE / "RES_NO_LEAKAGE" / "cv_nested" / pca_label / "best_clf"
           / model / Path(*target_segs) / "results.json")
    if rep.is_file():
        try:
            with rep.open() as f:
                d = json.load(f)
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

    p = BASE / "RES_NO_LEAKAGE" / pca_label / Path(*target_segs) / "results.json"
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            d = json.load(f)
    except Exception:
        return None
    bc = d.get("results", {}).get(model, {}).get("classifiers", {}).get("best_clf")
    if not bc:
        return None
    m, s = bc.get("mean_auc"), bc.get("std_auc")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    return float(m), float(s) if s is not None else 0.0


def load_fm_res(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> tuple[float, float] | None:
    # Priority: in-fold selection from repeated CV (RES_NO_LEAKAGE).
    if classifier in ("best_clf_repeated", "best_clf"):
        r = _load_res_no_leakage_best_clf(model, target_segs, pca)
        if r is not None:
            return r
    if pca:
        path = BASE / "RESIDUALIZATION_DIM" / "pca27" / Path(*target_segs) / "results.json"
    else:
        path = BASE / "RESIDUALIZATION_DIM" / Path(*target_segs) / "results.json"
    return _load_res(path, model, classifier)


def _select_best_clf_for_fm(
    model: str, target_segs: list[str], pca: bool
) -> str:
    """Return the classifier key with the highest FM-only mean AUC.

    Prefers nested_cv_repeated/best_clf/ (in-fold selection) when available;
    falls back to post-hoc selection from per-classifier nested_cv/ results.
    """
    fm_kind = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    # Check if repeated best_clf results exist.
    r_repeated = _load_repeated_best_clf_fm(model, fm_kind, target_segs)
    if r_repeated is not None:
        return "best_clf_repeated"
    # Post-hoc fallback.
    best_clf, best_mean = "random_forest", -1.0
    for clf in CLASSIFIERS_TO_TRY:
        r = load_fm_embedding(model, fm_kind, target_segs, clf)
        if r is not None and r[0] > best_mean:
            best_clf, best_mean = clf, r[0]
    return best_clf


# ---------------------------------------------------------------------------
# Statistical annotation helpers
# ---------------------------------------------------------------------------


def _fold_aucs_from_json(path: Path) -> list[float] | None:
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


def _load_base_folds(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    if pca:
        p = (BASE / model / "doc_patients" / "EMBEDDING_FM_PCA_ONLY"
             / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        r = _fold_aucs_from_json(p)
        if r is not None:
            return r
        fb = (BASE / model / "doc_patients" / "EMBEDDING_FM_PCA_ONLY"
              / model / "doc_patients" / "EMBEDDING_FM_PCA_ONLY"
              / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        return _fold_aucs_from_json(fb)
    else:
        p = (BASE / model / "doc_patients" / "MLP_EMBEDDING"
             / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        r = _fold_aucs_from_json(p)
        if r is not None:
            return r
        fb = (BASE / model / "doc_patients" / "MLP_EMBEDDING"
              / model / "doc_patients" / "MLP_EMBEDDING"
              / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        return _fold_aucs_from_json(fb)


def _load_combined_folds(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    if pca:
        p = (BASE / model / "doc_patients" / "EMBEDDING_DK_COMBINED_FM_PCA" / "pca27"
             / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        r = _fold_aucs_from_json(p)
        if r is not None:
            return r
        flat = (BASE / model / "doc_patients" / "EMBEDDING_DK_COMBINED_FM_PCA"
                / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        return _fold_aucs_from_json(flat)
    else:
        p = (BASE / model / "doc_patients" / "EMBEDDING_DK_COMBINED"
             / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        r = _fold_aucs_from_json(p)
        if r is not None:
            return r
        fb = (BASE / model / "doc_patients" / "EMBEDDING_DK_COMBINED"
              / model / "doc_patients" / "EMBEDDING_DK_COMBINED"
              / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json")
        return _fold_aucs_from_json(fb)


def _load_res_folds(
    model: str, target_segs: list[str], pca: bool, classifier: str
) -> list[float] | None:
    if pca:
        path = BASE / "RESIDUALIZATION_DIM" / "pca27" / Path(*target_segs) / "results.json"
    else:
        path = BASE / "RESIDUALIZATION_DIM" / Path(*target_segs) / "results.json"
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception:
        return None
    clf_data = d.get("results", {}).get(model, {}).get("classifiers", {}).get(classifier)
    if not clf_data:
        return None
    aucs = clf_data.get("auc_per_fold")
    if not aucs:
        return None
    return [a for a in aucs if a is not None and not (isinstance(a, float) and np.isnan(a))]


def _welch_p(a: list[float], b: list[float], h1: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    t, p_two = _stats.ttest_ind(a, b, equal_var=False)
    if h1 == "a>b":
        return float(p_two / 2 if t > 0 else 1.0 - p_two / 2)
    return float(p_two / 2 if t < 0 else 1.0 - p_two / 2)


def _compute_panel_annotations(
    target_segs: list[str], pca: bool, alpha: float = 0.05
) -> dict[int, str]:
    """Return {row_index: color} for dots that should get a * annotation.

    Uses the same per-FM best classifier as build_rows() (mirrors plot1 logic).
    Blue * at combined if FM+DK > FM (p < alpha).
    Red  * at res     if FM  > Res  (p < alpha).
    """
    annots: dict[int, str] = {}
    for i, model in enumerate(FM_MODELS):
        clf = _select_best_clf_for_fm(model, target_segs, pca)
        base = _load_base_folds(model, target_segs, pca, clf)
        comb = _load_combined_folds(model, target_segs, pca, clf)
        res  = _load_res_folds(model, target_segs, pca, clf)
        if base and comb and _welch_p(comb, base, "a>b") < alpha:
            annots[2 + 3 * i] = "blue"
        if base and res and _welch_p(base, res, "a>b") < alpha:
            annots[3 + 3 * i] = "red"
    return annots


# ---------------------------------------------------------------------------
# Row building: each entry is (label, mean_or_None, std_or_None, color)
# ---------------------------------------------------------------------------


def build_rows(
    target_segs: list[str],
    pca: bool,
) -> list[tuple[str, float | None, float | None, str]]:
    rows: list[tuple[str, float | None, float | None, str]] = []

    # Baseline
    b = load_baseline(target_segs)
    if b is not None:
        rows.append(("DK", b[0], b[1], DK_COLOR))
    else:
        rows.append(("DK", None, None, DK_COLOR))

    if pca:
        variants = [
            ("",          "EMBEDDING_FM_PCA_ONLY"),
            ("+ DK", "EMBEDDING_DK_COMBINED_FM_PCA"),
            ("Res",      None),   # residualization (pca27)
        ]
    else:
        variants = [
            ("",           "MLP_EMBEDDING"),
            ("+ DK", "EMBEDDING_DK_COMBINED"),
            ("Res",      None),
        ]

    for model in FM_MODELS:
        color = FM_COLORS[model]
        # Select the classifier that maximises FM-only AUC (mirrors plot1).
        clf = _select_best_clf_for_fm(model, target_segs, pca)
        for suffix, embedding_kind in variants:
            label = f"{model} {suffix}".strip()
            if embedding_kind is None:
                r = load_fm_res(model, target_segs, pca=pca, classifier=clf)
            else:
                r = load_fm_embedding(model, embedding_kind, target_segs, clf)
            if r is None:
                rows.append((label, None, None, color))
            else:
                rows.append((label, r[0], r[1], color))

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panel(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
) -> int:
    """Draw one plot4-style panel on ax. Returns the count of missing ticks."""
    labels = [r[0] for r in rows]
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))
    missing = 0

    # Group-color bands (Baseline + each FM's triplet).
    gi = 0
    run_start = 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(
                    run_start - 0.5, i - 0.5,
                    color="gray", alpha=0.1, zorder=0,
                )
            gi += 1
            run_start = i

    ax.axhline(
        0.5, color="gray", linewidth=1.0, linestyle="--",
        alpha=0.7, zorder=1,
    )

    # Per-tick drawing: real points vs. blank placeholders.
    for xi, (_lbl, m, s, c) in zip(x, rows):
        if m is None:
            ax.plot(
                [xi], [0.5],
                marker="x", color=MISSING_COLOR, markersize=10,
                markeredgewidth=1.6, zorder=2,
            )
            missing += 1
        else:
            ax.errorbar(
                [xi], [m], yerr=[s or 0.0],
                fmt="D", color=c, capsize=3, markersize=6,
                linewidth=1.2, ecolor=c,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, 1.0)
    suffix = f"   [{missing} missing]" if missing else ""
    ax.set_title(f"{title}{suffix}", fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return missing


def plot_4x2_combined(
    pca: bool,
    out_path: Path,
) -> None:
    """Render a 4×2 combined figure stacking targets (rows 0-1) and etiology (rows 2-3).

    Layout — 2 columns, 4 rows, shared x and y:
      Row 0: CRS            | 6m
      Row 1: 1y             | 2y
      Row 2: Delay (VS)     | Delay (MCS)
      Row 3: Etiology (VS)  | Etiology (MCS)
    """
    all_targets = TARGETS_CS + TARGETS_ETIO  # 8 panels → 4 rows × 2 cols
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows = build_rows(tsegs, pca=pca)
        _plot_panel(ax, rows, title=tname)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_targets] wrote {out_path}")


def _plot_panel_annotated(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
    annots: dict[int, str],
) -> int:
    """Like _plot_panel but overlays coloured * for significant test results."""
    missing = _plot_panel(ax, rows, title)
    for xi, color in annots.items():
        if xi < len(rows):
            _lbl, m, s, _c = rows[xi]
            if m is not None:
                ax.text(
                    xi, m + (s or 0.0) + 0.04,
                    "*", color=color, ha="center", va="bottom",
                    fontsize=14, fontweight="bold",
                )
    return missing


def plot_4x2_combined_annotated(pca: bool, out_path: Path) -> None:
    """Like plot_4x2_combined but annotates significant comparisons.

    Blue * above FM+DK dot  → FM+DK significantly above FM (p < 0.05).
    Red  * above FM_Res dot → FM significantly above FM_Res (p < 0.05).
    """
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows   = build_rows(tsegs, pca=pca)
        annots = _compute_panel_annotations(tsegs, pca)
        _plot_panel_annotated(ax, rows, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    from matplotlib.lines import Line2D as _L2D
    legend_handles = [
        _L2D([], [], marker="$*$", color="blue",  markersize=11, linestyle="",
             label="FM+DK > FM  (p < 0.05)"),
        _L2D([], [], marker="$*$", color="red",   markersize=11, linestyle="",
             label="FM > FM Res  (p < 0.05)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01),
        bbox_transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_targets] wrote {out_path}")


def plot_2x2(
    target_list: list[tuple[str, list[str]]],
    pca: bool,
    out_path: Path,
    figure_title: str,
) -> dict[str, list[str]]:
    """Render a 2×2 figure. Returns {target → [missing labels]}."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharey=True, sharex=True)
    missing_report: dict[str, list[str]] = {}

    for ax, (tname, tsegs) in zip(axes.flat, target_list):
        rows = build_rows(tsegs, pca=pca)
        miss_labels = [r[0] for r in rows if r[1] is None]
        missing_report[tname] = miss_labels
        _plot_panel(ax, rows, title=tname)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    # Legend with FM colors + missing marker + chance.
    handles = [
        plt.Line2D([], [], color="gray", linestyle="--",
                   linewidth=1.0, label="chance"),
        plt.Line2D([], [], marker="x", linestyle="", color=MISSING_COLOR,
                   markersize=10, markeredgewidth=1.6, label="missing"),
        plt.Line2D([], [], marker="D", linestyle="", color=DK_COLOR,
                   markersize=10, label="Baseline"),
    ]
    for m in FM_MODELS:
        handles.append(
            plt.Line2D([], [], marker="D", linestyle="", color=FM_COLORS[m],
                       markersize=9, label=m)
        )
 #   fig.legend(
 #       handles=handles, loc="lower center",
 #       ncol=len(handles), fontsize=12,
 #       bbox_to_anchor=(0.5, -0.02), frameon=True,
 #   )

  #  fig.suptitle(figure_title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_targets] wrote {out_path}")
    return missing_report


# ---------------------------------------------------------------------------
# Missing-results checker + job generator
# ---------------------------------------------------------------------------

_RESULTS_ROOT = BASE
_MARKER_CSV = BASE.parents[1] / "original_DoC" / "nice_scalars_all.csv"
_PATIENT_LABELS = BASE.parents[1] / "metadata" / "metadata_patient_labels.csv"


def _save_plot4_missing_md_and_sh(out_md: Path, out_sh: Path) -> None:
    """For each FM × target, check whether DK+FM and FM-Res results exist for
    the best FM classifier.  Writes a .md summary and a .sh with condor_submit
    commands to fill the gaps.
    """
    all_targets = TARGETS_CS + TARGETS_ETIO

    # missing_rows: list of (pca, tname, model, clf, dk_missing, res_missing)
    missing_rows: list[tuple[bool, str, str, str, bool, bool]] = []

    for pca in (False, True):
        dk_kind = "EMBEDDING_DK_COMBINED_FM_PCA" if pca else "EMBEDDING_DK_COMBINED"
        for tname, target_segs in all_targets:
            for model in FM_MODELS:
                clf = _select_best_clf_for_fm(model, target_segs, pca)
                dk_ok  = load_fm_embedding(model, dk_kind, target_segs, clf) is not None
                res_ok = load_fm_res(model, target_segs, pca, clf) is not None
                if not dk_ok or not res_ok:
                    missing_rows.append((pca, tname, model, clf, not dk_ok, not res_ok))

    # ── .md ─────────────────────────────────────────────────────────────────
    md_lines = [
        "# Plot 4: Missing DK+FM and Residualization Results\n",
        "For each FM × target the 'Best CLF' is the one that maximises FM-only AUC "
        "(same selection as plot1).  ✗ = result file not found on disk.\n",
        "| PCA | Target | FM | Best CLF | DK+FM | Res |",
        "|-----|--------|----|----|-------|-----|",
    ]
    for pca, tname, model, clf, dk_miss, res_miss in missing_rows:
        pca_label = "yes" if pca else "no"
        dk_str  = "✗ missing" if dk_miss  else "✓"
        res_str = "✗ missing" if res_miss else "✓"
        md_lines.append(f"| {pca_label} | {tname} | {model} | {clf} | {dk_str} | {res_str} |")

    if not missing_rows:
        md_lines.append("| — | — | — | — | all present | all present |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[plot4_targets] wrote {out_md}")

    # ── .sh ─────────────────────────────────────────────────────────────────
    # Group missing DK+FM by (pca, model, clf) — one job runs all targets.
    import itertools as _it
    dk_jobs:  set[tuple[bool, str, str]] = set()
    res_jobs: set[tuple[bool, str, str]] = set()
    for pca, _tname, model, clf, dk_miss, res_miss in missing_rows:
        if dk_miss:
            dk_jobs.add((pca, model, clf))
        if res_miss:
            res_jobs.add((pca, model, clf))

    sh_lines = [
        "#!/bin/bash",
        "# Auto-generated by plot4_targets.py: submit missing DK+FM & Residualization jobs.",
        "# Review carefully before running — jobs are grouped by (PCA, FM, classifier).",
        "set -e",
        "",
    ]

    for pca, model, clf in sorted(dk_jobs):
        pca_flag = "--pca" if pca else ""
        tag = f"{model}_{'pca' if pca else 'nopca'}_{clf}"
        sh_lines += [
            f"# --- DK+FM: {model} | {'PCA' if pca else 'non-PCA'} | {clf} ---",
            f"condor_submit - << 'EOF_{tag}'",
            "universe          = vanilla",
            "request_cpus      = 16",
            "request_memory    = 48GB",
            "requirements      = (Arch == \"ppc64le\")",
            "getenv            = True",
            "environment       = \"PATH=$(PATH) HOME=$(HOME)\"",
            "+MaxRunTime       = 604800",
            "periodic_release  = (JobStatus == 5)",
            "on_exit_hold      = False",
            "executable        = /data/project/eeg_foundation/jobs/run_pca_study.sh",
            f"arguments         = --fm-models {model} "
            f"--feature-predicted crs etiology etiology_code cs_6m cs_1y cs_2y "
            f"--classifiers {clf} "
            f"--results-root {_RESULTS_ROOT} "
            f"--marker-csv {_MARKER_CSV} "
            f"--patient-labels {_PATIENT_LABELS} "
            f"--patient-labels-full {_PATIENT_LABELS} "
            f"--marker-reduction A --n-cv-folds 5 --random-state 42 "
            f"--no-shifts --skip-common-pool {pca_flag}",
            f"initialdir        = /data/project/eeg_foundation/src/doc_benchmark",
            f"output            = /data/project/eeg_foundation/logs/dk_missing_{tag}_$(Cluster).$(Process).out",
            f"error             = /data/project/eeg_foundation/logs/dk_missing_{tag}_$(Cluster).$(Process).err",
            f"log               = /data/project/eeg_foundation/logs/dk_missing_{tag}_$(Cluster).log",
            "queue",
            f"EOF_{tag}",
            "",
        ]

    for pca, model, clf in sorted(res_jobs):
        pca_flag = "--pca --pca-components 27" if pca else ""
        tag = f"{model}_{'pca' if pca else 'nopca'}_{clf}"
        sh_lines += [
            f"# --- Residualization: {model} | {'PCA' if pca else 'non-PCA'} | {clf} ---",
            f"condor_submit - << 'EOF_res_{tag}'",
            "universe          = vanilla",
            "request_cpus      = 16",
            "request_memory    = 48GB",
            "requirements      = (Arch == \"ppc64le\")",
            "getenv            = True",
            "environment       = \"PATH=$(PATH) HOME=$(HOME)\"",
            "+MaxRunTime       = 604800",
            "periodic_release  = (JobStatus == 5)",
            "on_exit_hold      = False",
            "executable        = /data/project/eeg_foundation/jobs/run_residualization_dim.sh",
            f"arguments         = --all-targets --model {model} "
            f"--results-root {_RESULTS_ROOT} "
            f"--marker-csv {_MARKER_CSV} "
            f"--patient-labels {_PATIENT_LABELS} "
            f"--output-dir {_RESULTS_ROOT}/RESIDUALIZATION_DIM "
            f"--reduction A --n-folds 5 --random-state 42 "
            f"--classifiers {clf} {pca_flag}",
            f"initialdir        = /data/project/eeg_foundation/src/doc_benchmark",
            f"output            = /data/project/eeg_foundation/logs/res_missing_{tag}_$(Cluster).$(Process).out",
            f"error             = /data/project/eeg_foundation/logs/res_missing_{tag}_$(Cluster).$(Process).err",
            f"log               = /data/project/eeg_foundation/logs/res_missing_{tag}_$(Cluster).log",
            "queue",
            f"EOF_res_{tag}",
            "",
        ]

    if not dk_jobs and not res_jobs:
        sh_lines.append("echo 'Nothing missing — all DK+FM and Residualization results present.'")

    out_sh.parent.mkdir(parents=True, exist_ok=True)
    out_sh.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    out_sh.chmod(0o755)
    print(f"[plot4_targets] wrote {out_sh}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = [
        (TARGETS_CS,   False,
            OUT_DIR / "plot4_targets_nonpca.png",
            "Random-Forest AUC across CS targets (non-PCA)"),
        (TARGETS_CS,   True,
            OUT_DIR / "plot4_targets_pca.png",
            "Random-Forest AUC across CS targets (PCA)"),
        (TARGETS_ETIO, False,
            OUT_DIR / "plot4_etiology_nonpca.png",
            "Random-Forest AUC across etiology/delay targets (non-PCA)"),
        (TARGETS_ETIO, True,
            OUT_DIR / "plot4_etiology_pca.png",
            "Random-Forest AUC across etiology/delay targets (PCA)"),
    ]

    # Combined 4×2 figures (targets top + etiology bottom).
    print("\n[plot4_targets] === plot4_combined_pca.png ===")
    plot_4x2_combined(pca=True,  out_path=OUT_DIR / "plot4_combined_pca.png")
    print("\n[plot4_targets] === plot4_combined_pca_annotated.png ===")
    plot_4x2_combined_annotated(pca=True, out_path=OUT_DIR / "plot4_combined_pca_annotated.png")
    print("\n[plot4_targets] === plot4_combined_nonpca.png ===")
    plot_4x2_combined(pca=False, out_path=OUT_DIR / "plot4_combined_nonpca.png")
    print("\n[plot4_targets] === plot4_combined_nonpca_annotated.png ===")
    plot_4x2_combined_annotated(pca=False, out_path=OUT_DIR / "plot4_combined_nonpca_annotated.png")

    all_missing: list[tuple[str, str, list[str]]] = []
    for target_list, pca, out_path, title in all_runs:
        print(f"\n[plot4_targets] === {out_path.name} ===")
        report = plot_2x2(target_list, pca=pca, out_path=out_path,
                          figure_title=title)
        for tname, miss in report.items():
            if miss:
                print(f"  [{tname}] MISSING ({len(miss)}): {', '.join(miss)}")
                all_missing.append((out_path.name, tname, miss))

    # Final summary of everything missing, across all 4 figures.
    print("\n" + "=" * 72)
    print("MISSING-DATA SUMMARY (what still needs to be run)")
    print("=" * 72)
    if not all_missing:
        print("  (nothing — all panels complete)")
    else:
        for fname, tname, miss in all_missing:
            print(f"[{fname} · {tname}]")
            for lbl in miss:
                print(f"    - {lbl}")

    _save_plot4_missing_md_and_sh(
        OUT_DIR / "plot4_missing_results.md",
        OUT_DIR.parent / "submit_missing_plot4_jobs.sh",
    )


if __name__ == "__main__":
    main()
