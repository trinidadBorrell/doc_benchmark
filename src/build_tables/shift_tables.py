#!/usr/bin/env python3
"""Build CRS shift comparison tables.

The table has two sections:
  1. Original rows  — Baseline + FM models using MLP_EMBEDDING results
     (matches the "CRS - AUC ± STD (No Combined Rows)" table exactly)
  2. Shifted rows   — for each FM model, three rows:
       {Model} (DK Shifted)   → EMBEDDING_DK_COMBINED_DK_SHIFTED
       {Model} (FM Shifted)   → EMBEDDING_DK_COMBINED_FM_SHIFTED
       {Model} (All Shifted)  → EMBEDDING_DK_COMBINED_BOTH_SHIFTED

Columns: SVM | MLP | Kernel Ridge | Random Forest
  (Shifted variants have no SVM — those cells show N/A.)

Output PNGs are saved to TABLES_DIR.

Usage:
    python shift_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["text.usetex"] = False

BASE = Path("/data/project/eeg_foundation/data/benchmark_results/new_results/")
TABLES_DIR = BASE / "TABLES"

FM_MODELS = ["TOTEM", "CBraMod", "LaBram", "NeuroLM"]
ALL_MODELS = ["MARKER_BASELINE"] + FM_MODELS

CLASSIFIERS = ["svm", "mlp", "kernel_ridge", "random_forest"]
CLASSIFIER_LABELS = {
    "svm": "SVM",
    "mlp": "MLP",
    "kernel_ridge": "Kernel Ridge",
    "random_forest": "Random Forest",
}

ROW_LABELS: dict[str, str] = {
    "MARKER_BASELINE": "Baseline",
    "TOTEM": "TOTEM",
    "CBraMod": "CBraMod",
    "LaBram": "LaBram",
    "NeuroLM": "NeuroLM",
}

# Shifted embedding kinds and their display suffixes.
SHIFT_VARIANTS = [
    ("EMBEDDING_DK_COMBINED", "Combined"),
    ("EMBEDDING_DK_COMBINED_DK_SHIFTED", "DK Shifted"),
    ("EMBEDDING_DK_COMBINED_FM_SHIFTED", "FM Shifted"),
    ("EMBEDDING_DK_COMBINED_BOTH_SHIFTED", "All Shifted"),
]

# Group colour bands (one per model group): original rows + 4 FM groups.
_GROUP_COLOURS = [
    "#ffffff",  # original block
    "#eaf2fb",  # TOTEM group
    "#eafbea",  # CBraMod group
    "#fef9e7",  # LaBram group
    "#fdf2f8",  # NeuroLM group
]


# ---------------------------------------------------------------------------
# Score reading helpers
# ---------------------------------------------------------------------------


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


def _read_classification_json(path: Path) -> dict[str, float | None]:
    """Read AUC/balanced-accuracy from a classification_results.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    auc_mean = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "mean"))
    auc_std = _to_float_or_none(_safe_get(data, "macro_average", "auc_score", "std"))
    bal_mean = _to_float_or_none(
        _safe_get(data, "macro_average", "balanced_accuracy", "mean")
    )
    bal_std = _to_float_or_none(
        _safe_get(data, "macro_average", "balanced_accuracy", "std")
    )

    if auc_mean is None:
        auc_mean = _to_float_or_none(data.get("test_auc_score"))
    if bal_mean is None:
        bal_mean = _to_float_or_none(data.get("test_balanced_accuracy"))

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "balanced_accuracy_mean": bal_mean,
        "balanced_accuracy_std": bal_std,
    }


def _read_summary_json(path: Path) -> dict[str, float | None]:
    """Read AUC/balanced-accuracy from a summary_metrics.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "auc_mean": _to_float_or_none(data.get("auc_mean")),
        "auc_std": _to_float_or_none(data.get("auc_std")),
        "balanced_accuracy_mean": _to_float_or_none(data.get("balanced_accuracy_mean")),
        "balanced_accuracy_std": _to_float_or_none(data.get("balanced_accuracy_std")),
    }


def _mlp_embedding_path(model: str, classifier: str) -> Path | None:
    """Resolve the MLP_EMBEDDING classification_results.json for a model/classifier."""
    if model == "MARKER_BASELINE":
        root = BASE / model
    else:
        root = BASE / model / "doc_patients" / "MLP_EMBEDDING"

    candidates = [root / "crs" / "nested_cv" / classifier / "classification_results.json"]

    for path in candidates:
        if path.is_file():
            return path
    return None


def _shifted_path(model: str, embedding_kind: str, classifier: str) -> Path | None:
    """Resolve a shifted variant result file (summary_metrics.json preferred)."""
    nested_cv = (
        BASE / model / "doc_patients" / embedding_kind / "crs" / "nested_cv" / classifier
    )
    for name in ("summary_metrics.json", "classification_results.json"):
        p = nested_cv / name
        if p.is_file():
            return p
    return None


def _read_any_path(path: Path) -> dict[str, float | None]:
    if path.name == "summary_metrics.json":
        return _read_summary_json(path)
    return _read_classification_json(path)


# ---------------------------------------------------------------------------
# Build row records
# ---------------------------------------------------------------------------


def _empty_scores() -> dict[str, float | None]:
    return {
        "auc_mean": None,
        "auc_std": None,
        "balanced_accuracy_mean": None,
        "balanced_accuracy_std": None,
    }


def build_rows() -> list[dict[str, object]]:
    """Return long-format records for every (row_key, classifier) combination."""
    rows: list[dict[str, object]] = []

    # ---- Original rows (MLP_EMBEDDING) ------------------------------------
    for model in ALL_MODELS:
        for classifier in CLASSIFIERS:
            path = _mlp_embedding_path(model, classifier)
            scores = _read_classification_json(path) if path is not None else _empty_scores()
            rows.append(
                {
                    "group": "original",
                    "row_key": model,
                    "row_label": ROW_LABELS[model],
                    "classifier": classifier,
                    **scores,
                }
            )

    # ---- Shifted rows (per FM model × 3 shift variants) -------------------
    for fm_model in FM_MODELS:
        for emb_kind, suffix in SHIFT_VARIANTS:
            row_key = f"{fm_model}_{emb_kind}"
            row_label = f"{ROW_LABELS[fm_model]} ({suffix})"
            for classifier in CLASSIFIERS:
                path = _shifted_path(fm_model, emb_kind, classifier)
                if path is not None:
                    scores = _read_any_path(path)
                else:
                    scores = _empty_scores()
                rows.append(
                    {
                        "group": fm_model,
                        "row_key": row_key,
                        "row_label": row_label,
                        "classifier": classifier,
                        **scores,
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _format_cell(mean_val, std_val, ref_mean=None) -> str:
    if pd.isna(mean_val):
        return "N/A"
    text = f"{mean_val:.3f}" if pd.isna(std_val) else f"{mean_val:.3f} \u00b1 {std_val:.3f}"
    if ref_mean is not None and not pd.isna(ref_mean):
        if float(mean_val) > float(ref_mean) + 1e-9:
            text = "\u2191 " + text
        elif float(mean_val) < float(ref_mean) - 1e-9:
            text = "\u2193 " + text
    return text


def _fm_model_of(row_key: str) -> str | None:
    """Return the FM model name if this is a shifted/combined row, else None."""
    for fm_model in FM_MODELS:
        if row_key.startswith(fm_model + "_"):
            return fm_model
    return None


def _build_row_order() -> list[str]:
    order: list[str] = list(ALL_MODELS)
    for fm_model in FM_MODELS:
        for emb_kind, _ in SHIFT_VARIANTS:
            order.append(f"{fm_model}_{emb_kind}")
    return order


def _group_of(row_key: str) -> str:
    """Return the colour group name for a row_key."""
    if row_key in ALL_MODELS:
        return "original"
    for fm_model in FM_MODELS:
        if row_key.startswith(fm_model + "_"):
            return fm_model
    return "original"


def render_shift_table(
    df: pd.DataFrame,
    score_prefix: str,
    title: str,
    output_path: Path,
) -> None:
    mean_col = "auc_mean" if score_prefix == "auc" else "balanced_accuracy_mean"
    std_col = "auc_std" if score_prefix == "auc" else "balanced_accuracy_std"

    pivot_mean = df.pivot(index="row_key", columns="classifier", values=mean_col)
    pivot_std = df.pivot(index="row_key", columns="classifier", values=std_col)
    label_map = df.drop_duplicates("row_key").set_index("row_key")["row_label"]

    full_row_order = _build_row_order()
    row_order = [r for r in full_row_order if r in pivot_mean.index]
    col_order = [c for c in CLASSIFIERS if c in pivot_mean.columns]

    pivot_mean = pivot_mean.reindex(index=row_order, columns=col_order)
    pivot_std = pivot_std.reindex(index=row_order, columns=col_order)

    # Standard means per FM model per classifier (MLP_EMBEDDING original rows).
    standard_mean: dict[str, dict[str, float | None]] = {}
    for fm_model in FM_MODELS:
        standard_mean[fm_model] = {}
        for classifier in col_order:
            if fm_model in pivot_mean.index:
                v = pivot_mean.at[fm_model, classifier]
                standard_mean[fm_model][classifier] = None if pd.isna(v) else float(v)
            else:
                standard_mean[fm_model][classifier] = None

    # Build cell text.
    table_text = []
    for row_key in row_order:
        display_row = [label_map.get(row_key, row_key)]
        fm_model = _fm_model_of(row_key)
        for classifier in col_order:
            ref = standard_mean[fm_model][classifier] if fm_model is not None else None
            display_row.append(
                _format_cell(
                    pivot_mean.at[row_key, classifier],
                    pivot_std.at[row_key, classifier],
                    ref_mean=ref,
                )
            )
        table_text.append(display_row)

    # Best value per column (for bold highlighting).
    best_means: dict[str, float | None] = {}
    for classifier in col_order:
        vals = pd.to_numeric(pivot_mean[classifier], errors="coerce")
        best_means[classifier] = None if vals.dropna().empty else float(vals.max())

    n_rows = len(table_text)
    fig_w = 11
    fig_h = 1.6 + 0.44 * max(n_rows, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = ["Model"] + [CLASSIFIER_LABELS[c] for c in col_order]
    table = ax.table(
        cellText=table_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.2)

    # Map group names to colours.
    group_order = ["original"] + FM_MODELS
    group_colour = {g: _GROUP_COLOURS[i] for i, g in enumerate(group_order)}

    # Header styling.
    for col_idx in range(len(col_labels)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#d0d0d0")
        cell.get_text().set_weight("bold")

    # Row styling: background colour per group + left-align model names.
    for row_idx, row_key in enumerate(row_order, start=1):
        group = _group_of(row_key)
        bg = group_colour.get(group, "#ffffff")
        for col_idx in range(len(col_labels)):
            table[(row_idx, col_idx)].set_facecolor(bg)
        table[(row_idx, 0)].get_text().set_ha("left")

    # Bold the best value per classifier column.
    for row_idx, row_key in enumerate(row_order, start=1):
        for col_idx, classifier in enumerate(col_order, start=1):
            mean_val = pivot_mean.at[row_key, classifier]
            best = best_means[classifier]
            if pd.isna(mean_val) or best is None:
                continue
            if abs(float(mean_val) - float(best)) < 1e-12:
                table[(row_idx, col_idx)].get_text().set_weight("bold")

    # Draw a separator line between the original block and the shifted blocks.
    n_original = sum(1 for rk in row_order if rk in ALL_MODELS)
    if n_original < n_rows:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        cell = table[(n_original + 1, 0)]
        cell_bbox = cell.get_window_extent(renderer)
        ax_bbox = ax.get_window_extent(renderer)
        y_frac = (cell_bbox.y1 - ax_bbox.y0) / ax_bbox.height
        ax.plot(
            [0, 1], [y_frac, y_frac],
            color="#555555", linewidth=1.2,
            transform=ax.transAxes, zorder=5,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    df = pd.DataFrame(rows)

    csv_path = TABLES_DIR / "shift_comparison_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    auc_out = TABLES_DIR / "table_crs_shift_auc_std.png"
    bal_out = TABLES_DIR / "table_crs_shift_balanced_accuracy_std.png"

    render_shift_table(
        df,
        "auc",
        "CRS - AUC \u00b1 STD (Shift Comparison)",
        auc_out,
    )
    render_shift_table(
        df,
        "balanced_accuracy",
        "CRS - Balanced Accuracy \u00b1 STD (Shift Comparison)",
        bal_out,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
