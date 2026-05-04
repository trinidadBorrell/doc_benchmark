"""Plot classification AUC across layers with error bars.

Reads per-layer classification JSONs from ``linear_probing.py`` and/or
``non_linear_probing.py`` and draws one figure per model:

* **x-axis** — layer name
* **y-axis** — AUC (mean +/- std as error bars)
* **one group of bars per layer**, one bar per classifier
* **hatched bars** for non-linear probing when ``--nonlinear-dir`` is given

Optionally includes the last-layer embedding classification results from
the main pipeline as a reference horizontal band.

Usage
-----
::

    python plot_layerwise_crs_auc.py --output-dir /path/to/LINEAR_PROBING
    python plot_layerwise_crs_auc.py --output-dir /path/to/LINEAR_PROBING --models CbraMod
    python plot_layerwise_crs_auc.py \\
        --output-dir /path/to/LINEAR_PROBING \\
        --nonlinear-dir /path/to/NON_LINEAR_PROBING
"""

import argparse
import json
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120

# Must match linear_probing.py
MODEL_LAYERS = {
    "CbraMod": ["patch_emb", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"],
    "NeuroLM": [
        # VQ encoder
        "vq_emb",
        "vq_enc_patch_emb",
        "vq_enc_block_2", "vq_enc_block_5", "vq_enc_block_8", "vq_enc_block_11",
        # VQ quantizer
        "vq_encode_task", "vq_quantize",
        # VQ decoder (freq, 4 blocks)
        "vq_dec_freq_block_0", "vq_dec_freq_block_1",
        "vq_dec_freq_block_2", "vq_dec_freq_block_3",
        # GPT-2 backbone
        "gpt_0", "gpt_3", "gpt_6", "gpt_9", "gpt_11",
    ],
    "TOTEM": [
        "totem_enc_conv1", "totem_enc_conv2", "totem_enc_conv3",
        "totem_enc_residual", "totem_enc_z",
        "totem_vq_quantized",
        "totem_dec_conv1", "totem_dec_residual",
    ],
    "LaBram": [
        "vqnsp_enc_patch_embed",
        "vqnsp_enc_block_2",
        "vqnsp_enc_block_5",
        "vqnsp_enc_block_8",
        "vqnsp_enc_block_11",
        "vqnsp_encode_task",
        "vqnsp_quantize",
        "vqnsp_dec_block_0",
        "vqnsp_dec_block_1",
        "vqnsp_dec_block_2",
        "vqnsp_decode_task",
        "labram_patch_embed",
        "labram_block_2",
        "labram_block_5",
        "labram_block_8",
        "labram_block_11",
        "labram_norm",
    ],
    "BIOT": [
        "pre_transformer",
        "transformer_0",
        "transformer_1",
        "transformer_2",
        "transformer_3",
        "final_emb",
    ],
}

# Accept both spellings in CLI (CbraMod and CBraMod)
MODEL_ALIASES = {
    "CbraMod": "CbraMod",
    "CBraMod": "CbraMod",
    "NeuroLM": "NeuroLM",
    "TOTEM": "TOTEM",
    "LaBram": "LaBram",
    "BIOT": "BIOT",
}

CLF_NAMES = ["svm", "kernel_ridge", "random_forest"]
CLF_LABELS = {
    "svm": "SVM",
    "kernel_ridge": "KernelRidge",
    "random_forest": "RandomForest",
}
CLF_COLORS = {"svm": "#1f77b4", "kernel_ridge": "#ff7f0e", "random_forest": "#2ca02c"}

# Last-layer embedding classification results from the main pipeline
_BENCHMARK_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
EMBEDDING_CLF_PATHS = {
    "CbraMod": op.join(
        _BENCHMARK_ROOT,
        "CBraMod/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
    "NeuroLM": op.join(
        _BENCHMARK_ROOT,
        "NeuroLM/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
    "TOTEM": op.join(
        _BENCHMARK_ROOT,
        "TOTEM/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
    "LaBram": op.join(
        _BENCHMARK_ROOT,
        "LaBram/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
    "BIOT": op.join(
        _BENCHMARK_ROOT,
        "BIOT/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
}


# -- loaders ---------------------------------------------------------------


def load_classification_per_layer(output_dir, model):
    """Load per-layer classification JSONs.

    Returns
    -------
    results : dict
        ``{layer: {clf: {auc_mean, auc_std}}}``
    """
    clf_root = op.join(output_dir, "classification")
    results = {}
    layers_to_check = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    for layer in layers_to_check:
        layer_dir = op.join(clf_root, layer)
        if not op.isdir(layer_dir):
            continue
        layer_results = {}
        for clf in CLF_NAMES:
            path = op.join(layer_dir, f"{clf}_{model}_results.json")
            if not op.isfile(path):
                continue
            with open(path) as f:
                layer_results[clf] = json.load(f)
        if layer_results:
            results[layer] = layer_results
    return results


def load_embedding_classification(model):
    """Load last-layer embedding classification results.

    Returns
    -------
    results : dict or None
        ``{clf: {auc_mean, auc_std}}`` or None.
    """
    base = EMBEDDING_CLF_PATHS.get(model)
    if base is None or not op.isdir(base):
        return None
    results = {}
    for clf in CLF_NAMES:
        path = op.join(base, clf, "classification_results.json")
        if not op.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        # Pipeline format: macro_average.auc_score.{mean, std}
        macro = data.get("macro_average", {})
        auc = macro.get("auc_score", {})
        if "mean" in auc:
            results[clf] = {"auc_mean": auc["mean"], "auc_std": auc["std"]}
    return results if results else None


# -- plotter ---------------------------------------------------------------


def plot_classification_layers(
    layer_results, model, output_dir, emb_clf=None, nl_layer_results=None
):
    """Grouped bar plot: AUC per layer per classifier.

    Parameters
    ----------
    layer_results : dict
        ``{layer: {clf: {auc_mean, auc_std}}}`` — linear probing results.
    model : str
    output_dir : str
    emb_clf : dict or None
        ``{clf: {auc_mean, auc_std}}`` from the last-layer embedding.
    nl_layer_results : dict or None
        ``{layer: {clf: {auc_mean, auc_std}}}`` from non-linear probing.
        Plotted as hatched bars beside the linear bars.
    """
    if not layer_results and emb_clf is None and not nl_layer_results:
        print(f"   [{model}] No classification results found, skipping.", flush=True)
        return

    canonical = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    layers = [lyr for lyr in canonical if lyr in layer_results]

    if emb_clf is not None:
        layers_plot = layers + ["last-layer\nembedding"]
    else:
        layers_plot = list(layers)

    # When NL results exist, double the bar slots per layer
    has_nl = bool(nl_layer_results)
    n_clf = len(CLF_NAMES)
    # Each classifier gets 1 bar (linear) + optionally 1 bar (NL)
    bars_per_clf = 2 if has_nl else 1
    total_bars = n_clf * bars_per_clf
    bar_width = 0.8 / total_bars
    n_layers = len(layers_plot)
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * (2.2 if has_nl else 1.8)), 6))

    for ci, clf in enumerate(CLF_NAMES):
        # --- linear bars ---
        means, stds = [], []
        for layer in layers:
            r = layer_results.get(layer, {}).get(clf, {})
            means.append(r.get("auc_mean"))
            stds.append(r.get("auc_std"))
        if emb_clf is not None:
            r = emb_clf.get(clf, {})
            means.append(r.get("auc_mean"))
            stds.append(r.get("auc_std"))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        lin_slot = ci * bars_per_clf
        offset = (lin_slot - total_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=CLF_LABELS[clf],
            color=CLF_COLORS[clf],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

        if emb_clf is not None and clf in emb_clf:
            bars[-1].set_edgecolor("black")
            bars[-1].set_linewidth(1.5)
            bars[-1].set_hatch("//")

        # --- non-linear bars (hatched) ---
        if has_nl:
            nl_layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in nl_layer_results]
            nl_means = []
            nl_stds = []
            for layer in layers:
                r = nl_layer_results.get(layer, {}).get(clf, {}) if layer in (nl_layer_results or {}) else {}
                nl_means.append(r.get("auc_mean"))
                nl_stds.append(r.get("auc_std"))
            if emb_clf is not None:
                nl_means.append(None)
                nl_stds.append(None)

            nl_means = np.array(nl_means, dtype=float)
            nl_stds = np.array(nl_stds, dtype=float)

            nl_slot = ci * bars_per_clf + 1
            nl_offset = (nl_slot - total_bars / 2 + 0.5) * bar_width
            ax.bar(
                x + nl_offset,
                nl_means,
                bar_width,
                yerr=nl_stds,
                label=f"{CLF_LABELS[clf]} (NL)" if ci == 0 or True else "_nolegend_",
                color=CLF_COLORS[clf],
                alpha=0.5,
                capsize=3,
                edgecolor=CLF_COLORS[clf],
                linewidth=1.0,
                hatch="xx",
            )

    ax.axhline(
        0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.5, label="Chance"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers_plot, rotation=30, ha="right", fontsize=12)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUC", fontsize=14)
    title_suffix = " + NL (hatched)" if has_nl else ""
    ax.set_title(
        f"{model} \u2014 Classification AUC across layers (VS vs MCS){title_suffix}",
        fontsize=15,
    )
    ax.set_ylim(0.3, 1.05)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out_path = op.join(output_dir, f"classification_layers_{model}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}", flush=True)


# -- table renderer --------------------------------------------------------


def render_layer_table(
    layer_results, model, output_dir, emb_clf=None, nl_layer_results=None
):
    """Render a PNG table: rows = layers, columns = classifiers (AUC mean ± std).

    Parameters
    ----------
    layer_results : dict
        ``{layer: {clf: {auc_mean, auc_std}}}``
    model : str
    output_dir : str
    emb_clf : dict or None
        ``{clf: {auc_mean, auc_std}}`` from the last-layer embedding.
    nl_layer_results : dict or None
        ``{layer: {clf: {auc_mean, auc_std}}}`` from non-linear probing.
        Shown as additional columns with ``(NL)`` suffix.
    """
    canonical = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    layers = [lyr for lyr in canonical if lyr in layer_results]
    if emb_clf is not None:
        layers_all = layers + ["last-layer embedding"]
    else:
        layers_all = list(layers)

    if not layers_all:
        print(f"   [{model}] No data for table, skipping.", flush=True)
        return

    def _fmt(r):
        mean = r.get("auc_mean")
        std = r.get("auc_std")
        if mean is None:
            return "N/A"
        if std is None:
            return f"{mean:.3f}"
        return f"{mean:.3f} \u00b1 {std:.3f}"

    has_nl = bool(nl_layer_results)

    # Column headers: Layer | SVM | KR | RF | SVM(NL) | KR(NL) | RF(NL)
    linear_cols = [CLF_LABELS[c] for c in CLF_NAMES]
    nl_cols = [f"{CLF_LABELS[c]} (NL)" for c in CLF_NAMES] if has_nl else []
    col_labels = ["Layer"] + linear_cols + nl_cols

    table_text = []
    cell_means_lin = []
    cell_means_nl = []

    for layer in layers_all:
        if layer == "last-layer embedding":
            src_lin = emb_clf or {}
        else:
            src_lin = layer_results.get(layer, {})

        row = [layer]
        row_means_lin = []
        for clf in CLF_NAMES:
            r = src_lin.get(clf, {})
            row.append(_fmt(r))
            row_means_lin.append(r.get("auc_mean"))
        cell_means_lin.append(row_means_lin)

        if has_nl:
            src_nl = nl_layer_results.get(layer, {}) if layer != "last-layer embedding" else {}
            row_means_nl = []
            for clf in CLF_NAMES:
                r = src_nl.get(clf, {})
                row.append(_fmt(r))
                row_means_nl.append(r.get("auc_mean"))
            cell_means_nl.append(row_means_nl)

        table_text.append(row)

    # Best per classifier (over both linear and NL if present)
    all_col_means = cell_means_lin + (cell_means_nl if has_nl else [])

    best_per_lin_clf = []
    for ci in range(len(CLF_NAMES)):
        vals = [cell_means_lin[ri][ci] for ri in range(len(layers_all)) if cell_means_lin[ri][ci] is not None]
        best_per_lin_clf.append(max(vals) if vals else None)

    n_rows = len(table_text)
    fig_w = 4 + 2.5 * len(col_labels[1:])
    fig_h = 1.2 + 0.45 * max(n_rows, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=table_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    for col_idx in range(len(col_labels)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#f0f0f0")
        cell.get_text().set_weight("bold")

    for row_idx in range(1, n_rows + 1):
        table[(row_idx, 0)].get_text().set_ha("left")

    if emb_clf is not None:
        emb_row_idx = n_rows
        for col_idx in range(len(col_labels)):
            table[(emb_row_idx, col_idx)].set_facecolor("#fff8dc")

    # Shade NL columns lightly
    if has_nl:
        nl_col_start = 1 + len(CLF_NAMES)
        for row_idx in range(1, n_rows + 1):
            for col_idx in range(nl_col_start, len(col_labels)):
                table[(row_idx, col_idx)].set_facecolor("#f0f8ff")

    # Bold best linear value per classifier
    for ci in range(len(CLF_NAMES)):
        best = best_per_lin_clf[ci]
        if best is None:
            continue
        for ri in range(len(layers_all)):
            mean_val = cell_means_lin[ri][ci]
            if mean_val is not None and abs(float(mean_val) - float(best)) < 1e-9:
                table[(ri + 1, ci + 1)].get_text().set_weight("bold")

    ax.set_title(
        f"{model} \u2014 Classification AUC across layers (VS vs MCS)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    out_path = op.join(output_dir, f"classification_layers_table_{model}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   Saved: {out_path}", flush=True)


# -- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot classification AUC across layers with error bars.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/project/eeg_foundation/data/benchmark_results"
            "/new_results/LINEAR_PROBING"
        ),
        help="Results directory (same --output-dir used for linear_probing.py).",
    )
    parser.add_argument(
        "--nonlinear-dir",
        default=None,
        help=(
            "Non-linear probing results directory (--output-dir used for "
            "non_linear_probing.py).  When provided, NL results are overlaid "
            "as hatched bars AND standalone NL plots are saved to this directory."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CbraMod", "NeuroLM", "TOTEM", "LaBram", "BIOT"],
        help="Models to include.",
    )
    args = parser.parse_args()

    models = []
    for m in args.models:
        norm = MODEL_ALIASES.get(m)
        if norm is not None and norm not in models:
            models.append(norm)
    if not models:
        print("No valid models specified.")
        return

    print("=" * 60, flush=True)
    print("CLASSIFICATION AUC ACROSS LAYERS", flush=True)
    print(f"  output_dir    : {args.output_dir}", flush=True)
    print(f"  nonlinear_dir : {args.nonlinear_dir}", flush=True)
    print(f"  models        : {models}", flush=True)
    print("=" * 60, flush=True)

    for model in models:
        layer_results = load_classification_per_layer(args.output_dir, model)
        print(f"   [{model}] Loaded {len(layer_results)} layers (linear)", flush=True)

        emb_clf = load_embedding_classification(model)
        if emb_clf is not None:
            parts = []
            for clf, r in emb_clf.items():
                parts.append(f"{CLF_LABELS[clf]}={r['auc_mean']:.3f}")
            print(f"   [{model}] Last-layer embedding: {', '.join(parts)}", flush=True)
        else:
            print(
                f"   [{model}] No last-layer embedding classification found", flush=True
            )

        nl_layer_results = None
        if args.nonlinear_dir:
            nl_layer_results = load_classification_per_layer(args.nonlinear_dir, model)
            print(
                f"   [{model}] Loaded {len(nl_layer_results)} layers (non-linear)",
                flush=True,
            )

        # Combined plot (saved to linear dir)
        plot_classification_layers(
            layer_results, model, args.output_dir,
            emb_clf=emb_clf, nl_layer_results=nl_layer_results,
        )
        render_layer_table(
            layer_results, model, args.output_dir,
            emb_clf=emb_clf, nl_layer_results=nl_layer_results,
        )

        # Standalone NL plots (saved to nonlinear dir)
        if args.nonlinear_dir and nl_layer_results:
            plot_classification_layers(
                nl_layer_results, model, args.nonlinear_dir, emb_clf=None,
            )
            render_layer_table(
                nl_layer_results, model, args.nonlinear_dir, emb_clf=None,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
