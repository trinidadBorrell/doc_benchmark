#!/usr/bin/env python3
"""Build classification result tables for all metrics and models.

Combines score extraction and table rendering in a single script.
Output PNGs are saved to TABLES_DIR (one AUC table + one Balanced Accuracy
table per metric).

Usage:
    python build_tables.py
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

TASKS = [
    {"metric": "crs", "metric_path": "crs", "variant": "main"},
    {"metric": "cs_1y", "metric_path": "cs_1y", "variant": "main"},
    {"metric": "cs_2y", "metric_path": "cs_2y", "variant": "main"},
    {"metric": "cs_6m", "metric_path": "cs_6m", "variant": "main"},
    {"metric": "etiology", "metric_path": "etiology", "variant": "main"},
    {"metric": "etiology_code", "metric_path": "etiology_code", "variant": "main"},
    {
        "metric": "cs_1y/binary_vs_to_mcs",
        "metric_path": "cs_1y/binary_vs_to_mcs",
        "variant": "binary_vs_to_mcs",
    },
    {
        "metric": "cs_2y/binary_vs_to_mcs",
        "metric_path": "cs_2y/binary_vs_to_mcs",
        "variant": "binary_vs_to_mcs",
    },
    {
        "metric": "cs_6m/binary_vs_to_mcs",
        "metric_path": "cs_6m/binary_vs_to_mcs",
        "variant": "binary_vs_to_mcs",
    },
    {
        "metric": "cs_1y/binary_mcs_to_conscious",
        "metric_path": "cs_1y/binary_mcs_to_conscious",
        "variant": "binary_mcs_to_conscious",
    },
    {
        "metric": "cs_2y/binary_mcs_to_conscious",
        "metric_path": "cs_2y/binary_mcs_to_conscious",
        "variant": "binary_mcs_to_conscious",
    },
    {
        "metric": "cs_6m/binary_mcs_to_conscious",
        "metric_path": "cs_6m/binary_mcs_to_conscious",
        "variant": "binary_mcs_to_conscious",
    },
    # Binary improvement: IMPROVED vs NON_IMPROVED (all baseline states)
    {
        "metric": "cs_1y/binary_improvement",
        "metric_path": "cs_1y/binary_improvement",
        "variant": "binary_improvement",
    },
    {
        "metric": "cs_2y/binary_improvement",
        "metric_path": "cs_2y/binary_improvement",
        "variant": "binary_improvement",
    },
    {
        "metric": "cs_6m/binary_improvement",
        "metric_path": "cs_6m/binary_improvement",
        "variant": "binary_improvement",
    },
    # Etiology filtered by baseline state
    {
        "metric": "etiology/vs_only",
        "metric_path": "etiology/vs_only",
        "variant": "etiology_vs_only",
    },
    {
        "metric": "etiology/mcs_only",
        "metric_path": "etiology/mcs_only",
        "variant": "etiology_mcs_only",
    },
    {
        "metric": "etiology_code/vs_only",
        "metric_path": "etiology_code/vs_only",
        "variant": "etiology_code_vs_only",
    },
    {
        "metric": "etiology_code/mcs_only",
        "metric_path": "etiology_code/mcs_only",
        "variant": "etiology_code_mcs_only",
    },
]

MODELS = ["MARKER_BASELINE", "TOTEM", "CBraMod", "LaBram", "NeuroLM"]
CLASSIFIERS = ["svm", "mlp", "kernel_ridge", "random_forest"]

ROW_LABELS = {
    "MARKER_BASELINE": "Baseline",
    "TOTEM": "TOTEM",
    "CBraMod": "CBraMod",
    "LaBram": "LaBram",
    "NeuroLM": "NeuroLM",
}

METRIC_TITLE = {
    "crs": "CRS",
    "cs_1y": "CS 1Y",
    "cs_2y": "CS 2Y",
    "cs_6m": "CS 6M",
    "etiology": "Etiology",
    "etiology_code": "Etiology Code",
    "cs_1y/binary_vs_to_mcs": "CS 1Y - Binary VS to MCS",
    "cs_2y/binary_vs_to_mcs": "CS 2Y - Binary VS to MCS",
    "cs_6m/binary_vs_to_mcs": "CS 6M - Binary VS to MCS",
    "cs_1y/binary_mcs_to_conscious": "CS 1Y - Binary MCS to Conscious",
    "cs_2y/binary_mcs_to_conscious": "CS 2Y - Binary MCS to Conscious",
    "cs_6m/binary_mcs_to_conscious": "CS 6M - Binary MCS to Conscious",
    "cs_1y/binary_improvement": "CS 1Y - Improvement",
    "cs_2y/binary_improvement": "CS 2Y - Improvement",
    "cs_6m/binary_improvement": "CS 6M - Improvement",
    "etiology/vs_only": "Etiology (VS only)",
    "etiology/mcs_only": "Etiology (MCS only)",
    "etiology_code/vs_only": "Etiology Code (VS only)",
    "etiology_code/mcs_only": "Etiology Code (MCS only)",
}

CLASSIFIER_ORDER = ["svm", "mlp", "kernel_ridge", "random_forest"]
CLASSIFIER_LABELS = {
    "svm": "SVM",
    "mlp": "MLP",
    "kernel_ridge": "Kernel Ridge",
    "random_forest": "Random Forest",
}

ROW_ORDER_MAIN = ["MARKER_BASELINE", "TOTEM", "CBraMod", "LaBram", "NeuroLM"]
ROW_ORDER_CRS = ROW_ORDER_MAIN + [
    "TOTEM_COMBINED_CRS",
    "CBraMod_COMBINED_CRS",
    "LaBram_COMBINED_CRS",
    "NeuroLM_COMBINED_CRS",
]


# ---------------------------------------------------------------------------
# Score extraction helpers
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


def candidate_json_paths(
    model: str,
    metric_path: str,
    embedding_kind: str,
    classifier: str,
) -> list[Path]:
    """Return candidate JSON paths in priority order."""
    metric_parts = Path(metric_path).parts

    if model == "MARKER_BASELINE":
        model_root = BASE / model
    else:
        model_root = BASE / model / "doc_patients" / embedding_kind

    candidates = [
        model_root
        / metric_path
        / "nested_cv"
        / classifier
        / "classification_results.json"
    ]

    if len(metric_parts) == 1 and metric_parts[0] in {"cs_1y", "cs_2y", "cs_6m"}:
        metric_name = metric_parts[0]
        candidates.extend(
            [
                model_root
                / metric_name
                / "multiclass"
                / "nested_cv"
                / classifier
                / "classification_results.json",
                model_root
                / metric_name
                / "binary"
                / "nested_cv"
                / classifier
                / "classification_results.json",
            ]
        )

    if len(metric_parts) == 1:
        candidates.append(
            model_root
            / metric_parts[0]
            / "binary"
            / "nested_cv"
            / classifier
            / "classification_results.json"
        )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def read_scores(json_path: Path) -> dict[str, float | None]:
    with json_path.open("r", encoding="utf-8") as f:
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


def resolve_json_path(
    model: str,
    metric_path: str,
    embedding_kind: str,
    classifier: str,
) -> Path | None:
    for path in candidate_json_paths(model, metric_path, embedding_kind, classifier):
        if path.is_file():
            return path
    return None


def _record(
    metric: str,
    row_key: str,
    row_label: str,
    variant: str,
    classifier: str,
    scores: dict[str, float | None] | None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "metric": metric,
        "row_key": row_key,
        "row_label": row_label,
        "variant": variant,
        "classifier": classifier,
        "auc_mean": None,
        "auc_std": None,
        "balanced_accuracy_mean": None,
        "balanced_accuracy_std": None,
    }
    if scores is not None:
        out.update(scores)
    return out


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for task in TASKS:
        metric = task["metric"]
        metric_path = task["metric_path"]
        variant = task["variant"]

        for model in MODELS:
            for classifier in CLASSIFIERS:
                json_path = resolve_json_path(model, metric_path, "MLP_EMBEDDING", classifier)
                scores = read_scores(json_path) if json_path is not None else None
                rows.append(
                    _record(
                        metric=metric,
                        row_key=model,
                        row_label=ROW_LABELS[model],
                        variant=variant,
                        classifier=classifier,
                        scores=scores,
                    )
                )

        # Combined rows only for CRS.
        if metric == "crs":
            for model in MODELS:
                if model == "MARKER_BASELINE":
                    continue
                for classifier in CLASSIFIERS:
                    json_path = resolve_json_path(
                        model, "crs", "EMBEDDING_DK_COMBINED", classifier
                    )
                    scores = read_scores(json_path) if json_path is not None else None
                    rows.append(
                        _record(
                            metric="crs",
                            row_key=f"{model}_COMBINED_CRS",
                            row_label=f"{ROW_LABELS[model]} (Combined)",
                            variant="combined_crs",
                            classifier=classifier,
                            scores=scores,
                        )
                    )

    return rows


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------


def _format_cell(mean_val, std_val) -> str:
    if pd.isna(mean_val):
        return "N/A"
    if pd.isna(std_val):
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} \u00b1 {std_val:.3f}"


def _row_order_for_metric(metric: str) -> list[str]:
    return ROW_ORDER_CRS if metric == "crs" else ROW_ORDER_MAIN


def render_score_table(
    metric_df: pd.DataFrame,
    metric: str,
    score_prefix: str,
    title: str,
    output_path: Path,
) -> None:
    mean_col = "auc_mean" if score_prefix == "auc" else "balanced_accuracy_mean"
    std_col = "auc_std" if score_prefix == "auc" else "balanced_accuracy_std"

    pivot_mean = metric_df.pivot(index="row_key", columns="classifier", values=mean_col)
    pivot_std = metric_df.pivot(index="row_key", columns="classifier", values=std_col)
    label_map = metric_df.drop_duplicates("row_key").set_index("row_key")["row_label"]

    row_order = [r for r in _row_order_for_metric(metric) if r in pivot_mean.index]
    col_order = [c for c in CLASSIFIER_ORDER if c in pivot_mean.columns]

    pivot_mean = pivot_mean.reindex(index=row_order, columns=col_order)
    pivot_std = pivot_std.reindex(index=row_order, columns=col_order)

    table_text = []
    for row_key in row_order:
        display_row = [label_map.get(row_key, row_key)]
        for classifier in col_order:
            display_row.append(
                _format_cell(
                    pivot_mean.at[row_key, classifier],
                    pivot_std.at[row_key, classifier],
                )
            )
        table_text.append(display_row)

    best_means: dict[str, float | None] = {}
    for classifier in col_order:
        vals = pd.to_numeric(pivot_mean[classifier], errors="coerce")
        best_means[classifier] = None if vals.dropna().empty else float(vals.max())

    n_rows = len(table_text)
    fig_w = 10
    fig_h = 1.4 + 0.48 * max(n_rows, 1)
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
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    for col_idx in range(len(col_labels)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#f0f0f0")
        cell.get_text().set_weight("bold")

    for row_idx in range(1, n_rows + 1):
        table[(row_idx, 0)].get_text().set_ha("left")

    for row_idx, row_key in enumerate(row_order, start=1):
        for col_idx, classifier in enumerate(col_order, start=1):
            mean_val = pivot_mean.at[row_key, classifier]
            best = best_means[classifier]
            if pd.isna(mean_val) or best is None:
                continue
            if abs(float(mean_val) - float(best)) < 1e-12:
                table[(row_idx, col_idx)].get_text().set_weight("bold")

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

    csv_path = TABLES_DIR / "model_metric_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    print("\nRows per metric:")
    print(df.groupby("metric").size().to_string())
    print()

    metrics = df["metric"].unique().tolist()
    for metric in metrics:
        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            print(f"Skipping {metric}: no rows.")
            continue

        metric_title = METRIC_TITLE.get(metric, metric)
        metric_slug = metric.replace("/", "__")

        auc_out = TABLES_DIR / f"table_{metric_slug}_auc_std.png"
        bal_out = TABLES_DIR / f"table_{metric_slug}_balanced_accuracy_std.png"

        render_score_table(metric_df, metric, "auc", f"{metric_title} - AUC \u00b1 STD", auc_out)
        render_score_table(
            metric_df,
            metric,
            "balanced_accuracy",
            f"{metric_title} - Balanced Accuracy \u00b1 STD",
            bal_out,
        )

        if metric == "crs":
            no_combined = metric_df[metric_df["variant"] != "combined_crs"].copy()
            auc_no_combined = TABLES_DIR / "table_crs_auc_std_no_combined.png"
            render_score_table(
                no_combined,
                metric,
                "auc",
                "CRS - AUC \u00b1 STD (No Combined Rows)",
                auc_no_combined,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
