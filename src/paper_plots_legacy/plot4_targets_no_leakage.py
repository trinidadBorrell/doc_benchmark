#!/usr/bin/env python3
"""Leakage-free residualization variant of plot4_targets.py.

Identical layout and logic as plot4_targets.py, but the "Res" column reads
from RES_NO_LEAKAGE (fold-internal X standardization) instead of
RESIDUALIZATION_DIM.

Outputs (data/benchmark_results/new_results/PLOTS/):

    plot4_noleak_combined_nonpca.png          4×2 all targets, non-PCA
    plot4_noleak_combined_pca.png             4×2 all targets, PCA
    plot4_noleak_combined_nonpca_annotated.png  with significance stars
    plot4_noleak_combined_pca_annotated.png

    plot4_noleak_targets_nonpca.png   2×2: {crs, cs_6m, cs_1y, cs_2y}  non-PCA
    plot4_noleak_targets_pca.png      2×2: {crs, cs_6m, cs_1y, cs_2y}  PCA
    plot4_noleak_etiology_nonpca.png  2×2: delay/etiology targets       non-PCA
    plot4_noleak_etiology_pca.png     2×2: delay/etiology targets       PCA

Row structure per panel: DK | FM · FM+DK · FM Res (×5 models)
Residualization "Res" = leakage-free (RES_NO_LEAKAGE; fold-internal X scaling).
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


BASE    = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
OUT_DIR = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]
FM_MODELS = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CBraMod"]

DK_COLOR      = "#555555"
MISSING_COLOR = "#b0b0b0"
FM_COLORS = {
    "BIOT":    "#1f77b4",
    "EEGPT":   "#8c564b",
    "LaBram":  "#2ca02c",
    "NeuroLM": "#9467bd",
    "CBraMod": "#ff7f0e",
}

TARGETS_CS = [
    ("CRS Diagnostic", ["crs"]),
    ("6m",  ["cs_6m", "binary_improvement"]),
    ("1y",  ["cs_1y", "binary_improvement"]),
    ("2y",  ["cs_2y", "binary_improvement"]),
]
TARGETS_ETIO = [
    ("Delay (VS)",     ["etiology",      "vs_only"]),
    ("Delay (MCS)",    ["etiology",      "mcs_only"]),
    ("Etiology (VS)",  ["etiology_code", "vs_only"]),
    ("Etiology (MCS)", ["etiology_code", "mcs_only"]),
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
    base = BASE / model / "doc_patients" / embedding_kind
    if embedding_kind == "EMBEDDING_DK_COMBINED_FM_PCA":
        base = base / "pca27"
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
    if classifier in ("best_clf_repeated", "best_clf"):
        r = _load_repeated_best_clf_fm(model, embedding_kind, target_segs)
        if r is not None:
            return r
    if embedding_kind == "EMBEDDING_DK_COMBINED_FM_PCA":
        primary = (
            BASE / model / "doc_patients" / embedding_kind / "pca27"
            / Path(*target_segs) / "nested_cv" / classifier / "classification_results.json"
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


def _load_res_entry(path: Path, model: str, classifier: str) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    entry = d.get("results", {}).get(model, {}).get("classifiers", {}).get(classifier)
    if not entry:
        return None
    m = entry.get("mean_auc")
    s = entry.get("std_auc")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    return float(m), float(s) if s is not None else 0.0


def _load_res_noleak_best_clf(
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
    """Read from RES_NO_LEAKAGE (leakage-free residualization)."""
    if classifier in ("best_clf_repeated", "best_clf"):
        r = _load_res_noleak_best_clf(model, target_segs, pca)
        if r is not None:
            return r
    if pca:
        path = BASE / "RES_NO_LEAKAGE" / "pca27" / Path(*target_segs) / "results.json"
    else:
        path = BASE / "RES_NO_LEAKAGE" / "no_pca" / Path(*target_segs) / "results.json"
    return _load_res_entry(path, model, classifier)


def _select_best_clf_for_fm(model: str, target_segs: list[str], pca: bool) -> str:
    fm_kind = "EMBEDDING_FM_PCA_ONLY" if pca else "MLP_EMBEDDING"
    r_repeated = _load_repeated_best_clf_fm(model, fm_kind, target_segs)
    if r_repeated is not None:
        return "best_clf_repeated"
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
    """Read fold-level AUCs from RES_NO_LEAKAGE."""
    if pca:
        path = BASE / "RES_NO_LEAKAGE" / "pca27" / Path(*target_segs) / "results.json"
    else:
        path = BASE / "RES_NO_LEAKAGE" / "no_pca" / Path(*target_segs) / "results.json"
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
    annots: dict[int, str] = {}
    for i, model in enumerate(FM_MODELS):
        clf  = _select_best_clf_for_fm(model, target_segs, pca)
        base = _load_base_folds(model, target_segs, pca, clf)
        comb = _load_combined_folds(model, target_segs, pca, clf)
        res  = _load_res_folds(model, target_segs, pca, clf)
        if base and comb and _welch_p(comb, base, "a>b") < alpha:
            annots[2 + 3 * i] = "blue"
        if base and res and _welch_p(base, res, "a>b") < alpha:
            annots[3 + 3 * i] = "red"
    return annots


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def build_rows(
    target_segs: list[str], pca: bool
) -> list[tuple[str, float | None, float | None, str]]:
    rows: list[tuple[str, float | None, float | None, str]] = []

    b = load_baseline(target_segs)
    rows.append(("DK", b[0] if b else None, b[1] if b else None, DK_COLOR))

    if pca:
        variants = [
            ("",      "EMBEDDING_FM_PCA_ONLY"),
            ("+ DK",  "EMBEDDING_DK_COMBINED_FM_PCA"),
            ("Res",   None),
        ]
    else:
        variants = [
            ("",      "MLP_EMBEDDING"),
            ("+ DK",  "EMBEDDING_DK_COMBINED"),
            ("Res",   None),
        ]

    for model in FM_MODELS:
        color = FM_COLORS[model]
        clf   = _select_best_clf_for_fm(model, target_segs, pca)
        for suffix, embedding_kind in variants:
            label = f"{model} {suffix}".strip()
            if embedding_kind is None:
                r = load_fm_res(model, target_segs, pca=pca, classifier=clf)
            else:
                r = load_fm_embedding(model, embedding_kind, target_segs, clf)
            rows.append((label, r[0] if r else None, r[1] if r else None, color))

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_panel(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
) -> int:
    labels = [r[0] for r in rows]
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))
    missing = 0

    gi, run_start = 0, 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[run_start]:
            if gi % 2 == 0:
                ax.axvspan(run_start - 0.5, i - 0.5,
                           color="gray", alpha=0.1, zorder=0)
            gi += 1
            run_start = i

    ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

    for xi, (_lbl, m, s, c) in zip(x, rows):
        if m is None:
            ax.plot([xi], [0.5], marker="x", color=MISSING_COLOR,
                    markersize=10, markeredgewidth=1.6, zorder=2)
            missing += 1
        else:
            ax.errorbar([xi], [m], yerr=[s or 0.0], fmt="D", color=c,
                        capsize=3, markersize=6, linewidth=1.2, ecolor=c,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, 1.0)
    suffix = f"   [{missing} missing]" if missing else ""
    ax.set_title(f"{title}{suffix}", fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return missing


def _plot_panel_annotated(
    ax,
    rows: list[tuple[str, float | None, float | None, str]],
    title: str,
    annots: dict[int, str],
) -> int:
    missing = _plot_panel(ax, rows, title)
    for xi, color in annots.items():
        if xi < len(rows):
            _lbl, m, s, _c = rows[xi]
            if m is not None:
                ax.text(xi, m + (s or 0.0) + 0.04, "*",
                        color=color, ha="center", va="bottom",
                        fontsize=14, fontweight="bold")
    return missing


def _legend_handles():
    from matplotlib.lines import Line2D as _L
    return [
        _L([], [], color="gray", linestyle="--", linewidth=1.0, label="chance"),
        _L([], [], marker="x", linestyle="", color=MISSING_COLOR,
           markersize=10, markeredgewidth=1.6, label="missing"),
        _L([], [], marker="D", linestyle="", color=DK_COLOR, markersize=10, label="DK"),
    ] + [
        _L([], [], marker="D", linestyle="", color=FM_COLORS[m], markersize=9, label=m)
        for m in FM_MODELS
    ]


def _star_legend_handles(test_label: str):
    from matplotlib.lines import Line2D as _L
    return [
        _L([], [], marker="$*$", color="blue", markersize=11, linestyle="",
           label=f"FM+DK > FM  (p < 0.05, {test_label})"),
        _L([], [], marker="$*$", color="red",  markersize=11, linestyle="",
           label=f"FM > Res  (p < 0.05, {test_label})"),
    ]


def plot_4x2_combined(pca: bool, out_path: Path) -> None:
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows = build_rows(tsegs, pca=pca)
        _plot_panel(ax, rows, title=tname)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    mode = "PCA" if pca else "non-PCA"
    fig.suptitle(
        f"Leakage-free residualization — {mode}\n"
        "(Res = RES_NO_LEAKAGE: fold-internal FM-embedding standardization)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_noleak] wrote {out_path}")


def plot_4x2_combined_annotated(pca: bool, out_path: Path) -> None:
    all_targets = TARGETS_CS + TARGETS_ETIO
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, all_targets):
        rows   = build_rows(tsegs, pca=pca)
        annots = _compute_panel_annotations(tsegs, pca)
        _plot_panel_annotated(ax, rows, title=tname, annots=annots)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    fig.legend(
        handles=_star_legend_handles("Welch t-test"),
        loc="lower right", fontsize=9, framealpha=0.9, handlelength=1.2,
        bbox_to_anchor=(0.99, 0.01), bbox_transform=fig.transFigure,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_noleak] wrote {out_path}")


def plot_2x2(
    target_list: list[tuple[str, list[str]]],
    pca: bool,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharey=True, sharex=True)

    for ax, (tname, tsegs) in zip(axes.flat, target_list):
        rows = build_rows(tsegs, pca=pca)
        _plot_panel(ax, rows, title=tname)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC", fontsize=17)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot4_noleak] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[plot4_noleak] === 4×2 combined figures ===")
    plot_4x2_combined(
        pca=False, out_path=OUT_DIR / "plot4_noleak_combined_nonpca.png")
    plot_4x2_combined(
        pca=True,  out_path=OUT_DIR / "plot4_noleak_combined_pca.png")
    plot_4x2_combined_annotated(
        pca=False, out_path=OUT_DIR / "plot4_noleak_combined_nonpca_annotated.png")
    plot_4x2_combined_annotated(
        pca=True,  out_path=OUT_DIR / "plot4_noleak_combined_pca_annotated.png")

    print("\n[plot4_noleak] === 2×2 CS targets ===")
    plot_2x2(TARGETS_CS,   pca=False, out_path=OUT_DIR / "plot4_noleak_targets_nonpca.png")
    plot_2x2(TARGETS_CS,   pca=True,  out_path=OUT_DIR / "plot4_noleak_targets_pca.png")

    print("\n[plot4_noleak] === 2×2 Etiology/Delay targets ===")
    plot_2x2(TARGETS_ETIO, pca=False, out_path=OUT_DIR / "plot4_noleak_etiology_nonpca.png")
    plot_2x2(TARGETS_ETIO, pca=True,  out_path=OUT_DIR / "plot4_noleak_etiology_pca.png")

    print("\n[plot4_noleak] Done.")


if __name__ == "__main__":
    main()
