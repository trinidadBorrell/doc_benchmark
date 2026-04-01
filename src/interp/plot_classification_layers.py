"""Plot classification AUC across layers with error bars.

Reads per-layer classification JSONs from ``linear_probing.py`` and draws
one figure per model:

* **x-axis** — layer name
* **y-axis** — AUC (mean +/- std as error bars)
* **one group of bars per layer**, one bar per classifier

Optionally includes the last-layer embedding classification results from
the main pipeline as a reference horizontal band.

Usage
-----
::

    python plot_classification_layers.py --output-dir /path/to/LINEAR_PROBING
    python plot_classification_layers.py --output-dir /path/to/LINEAR_PROBING --models CbraMod

Author: Trinidad Borrell <trinidad.borrell@gmail.com>
"""

import argparse
import json
import os.path as op

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120

# Must match linear_probing.py
MODEL_LAYERS = {
    "CbraMod": ["patch_emb", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"],
    "NeuroLM": ["gpt_0", "gpt_3", "gpt_6", "gpt_9", "gpt_11"],
}

# Accept both spellings in CLI (CbraMod and CBraMod)
MODEL_ALIASES = {
    "CbraMod": "CbraMod",
    "CBraMod": "CbraMod",
    "NeuroLM": "NeuroLM",
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
    for layer in MODEL_LAYERS[model]:
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


def plot_classification_layers(layer_results, model, output_dir, emb_clf=None):
    """Grouped bar plot: AUC per layer per classifier.

    Parameters
    ----------
    layer_results : dict
        ``{layer: {clf: {auc_mean, auc_std}}}``
    model : str
    output_dir : str
    emb_clf : dict or None
        ``{clf: {auc_mean, auc_std}}`` from the last-layer embedding.
    """
    if not layer_results and emb_clf is None:
        print(f"   [{model}] No classification results found, skipping.", flush=True)
        return

    layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in layer_results]

    # Add the embedding as a final "layer" if available
    if emb_clf is not None:
        layers_plot = layers + ["last-layer\nembedding"]
    else:
        layers_plot = list(layers)

    n_layers = len(layers_plot)
    n_clf = len(CLF_NAMES)
    bar_width = 0.8 / n_clf
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.8), 6))

    for ci, clf in enumerate(CLF_NAMES):
        means, stds = [], []
        for layer in layers:
            r = layer_results.get(layer, {}).get(clf, {})
            means.append(r.get("auc_mean"))
            stds.append(r.get("auc_std"))

        # Append embedding result
        if emb_clf is not None:
            r = emb_clf.get(clf, {})
            means.append(r.get("auc_mean"))
            stds.append(r.get("auc_std"))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        offset = (ci - n_clf / 2 + 0.5) * bar_width
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

        # If last bar is the embedding, highlight it
        if emb_clf is not None and clf in emb_clf:
            bars[-1].set_edgecolor("black")
            bars[-1].set_linewidth(1.5)
            bars[-1].set_hatch("//")

    ax.axhline(
        0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.5, label="Chance"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers_plot, rotation=30, ha="right", fontsize=12)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUC", fontsize=14)
    ax.set_title(
        f"{model} \u2014 Classification AUC across layers (VS vs MCS)",
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
        "--models",
        nargs="+",
        default=["CbraMod", "NeuroLM"],
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
    print(f"  output_dir : {args.output_dir}", flush=True)
    print(f"  models     : {models}", flush=True)
    print("=" * 60, flush=True)

    for model in models:
        layer_results = load_classification_per_layer(args.output_dir, model)
        print(f"   [{model}] Loaded {len(layer_results)} layers", flush=True)

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

        plot_classification_layers(
            layer_results, model, args.output_dir, emb_clf=emb_clf
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
