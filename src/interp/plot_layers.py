"""Combined R² across layers — one curve per layer, markers on x-axis.

Reads the per-layer regression ``summary.json`` files produced by
``linear_probing.py`` and/or ``non_linear_probing.py`` and draws a single
figure per model with:

* **x-axis** — neurophysiological markers (grouped by family)
* **y-axis** — R²
* **one curve per layer** (different colour)
* **dashed curves** for non-linear probing results when ``--nonlinear-dir``
  is provided

Usage
-----
::

    python plot_layers.py --output-dir /path/to/LINEAR_PROBING
    python plot_layers.py --output-dir /path/to/LINEAR_PROBING --models CbraMod
    python plot_layers.py \\
        --output-dir /path/to/LINEAR_PROBING \\
        --nonlinear-dir /path/to/NON_LINEAR_PROBING

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
    "NeuroLM": ["vq_emb", "gpt_0", "gpt_3", "gpt_6", "gpt_9", "gpt_11"],
}

# Classification constants (mirrors plot_classification_layers.py)
CLF_NAMES = ["svm", "kernel_ridge", "random_forest"]
CLF_LABELS = {"svm": "SVM", "kernel_ridge": "KernelRidge", "random_forest": "RandomForest"}
CLF_COLORS = {"svm": "#1f77b4", "kernel_ridge": "#ff7f0e", "random_forest": "#2ca02c"}

# Path to last-layer (full) embedding regressor results from the pipeline.
# These use the final embedding rather than individual intermediate layers.
_BENCHMARK_ROOT = "/data/project/eeg_foundation/data/benchmark_results/new_results"
EMBEDDING_REGRESSOR_PATHS = {
    "CbraMod": op.join(
        _BENCHMARK_ROOT,
        "CBraMod/doc_patients/MLP_EMBEDDING/regressor_results/summary.json",
    ),
    "NeuroLM": op.join(
        _BENCHMARK_ROOT,
        "NeuroLM/doc_patients/MLP_EMBEDDING/regressor_results/summary.json",
    ),
}
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

FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_COLORS = {
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",
    "wsmi": "#8b4513",
    "evoked": "#c55a11",
}

# Perceptually distinct palette for layers (up to 6)
LAYER_CMAP = plt.cm.viridis
LAYER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# -- helpers (duplicated from linear_probing.py to keep standalone) ---------


def _short_name(full_path):
    parts = full_path.split("/")
    return "_".join(parts[-2:])


def _marker_family(marker_name):
    if "KolmogorovComplexity" in marker_name or "PermutationEntropy" in marker_name:
        return "information_theory"
    if "PowerSpectralDensity" in marker_name:
        return "power_spectral_density"
    if "SymbolicMutualInformation" in marker_name:
        return "wsmi"
    return "evoked"


def _ordered_markers_by_family_and_mean(r2_per_layer):
    """Order markers by family, then ascending mean R² across layers."""
    all_markers = set()
    for dct in r2_per_layer.values():
        all_markers.update(dct.keys())

    def _mean_raw(marker):
        vals = [r2_per_layer[lyr].get(marker, np.nan) for lyr in r2_per_layer]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    families = {fam: [] for fam in FAMILY_ORDER}
    for marker in all_markers:
        fam = _marker_family(marker)
        families.setdefault(fam, []).append(marker)

    ordered = []
    for fam in FAMILY_ORDER:
        fam_markers = families.get(fam, [])
        fam_markers = sorted(fam_markers, key=lambda m: (_mean_raw(m), _short_name(m)))
        ordered.extend(fam_markers)
    return ordered


def _color_xticks_by_family(ax, marker_order):
    for tick, marker in zip(ax.get_xticklabels(), marker_order):
        fam = _marker_family(marker)
        tick.set_color(FAMILY_COLORS.get(fam, "black"))


# -- loader ----------------------------------------------------------------


def load_regression_per_layer(output_dir, model):
    """Load per-layer regression summaries.

    Returns
    -------
    r2_per_layer : dict
        ``{layer_key: {marker_name: r2_float}}``
    """
    reg_root = op.join(output_dir, "regression")
    r2_per_layer = {}
    layers_to_check = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    for layer in layers_to_check:
        path = op.join(reg_root, layer, model, "summary.json")
        if not op.isfile(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        r2_dict = {
            m: v["r2"]
            for m, v in metrics.items()
            if isinstance(v, dict) and not v.get("skipped", False) and "r2" in v
        }
        if r2_dict:
            r2_per_layer[layer] = r2_dict
    return r2_per_layer


def load_embedding_regressor(model):
    """Load the last-layer embedding regressor summary.

    Returns
    -------
    r2_dict : dict or None
        ``{marker_name: r2_float}`` or None if file not found.
    """
    path = EMBEDDING_REGRESSOR_PATHS.get(model)
    if path is None or not op.isfile(path):
        return None
    with open(path) as f:
        metrics = json.load(f)
    r2_dict = {
        m: v["r2"]
        for m, v in metrics.items()
        if isinstance(v, dict) and not v.get("skipped", False) and "r2" in v
    }
    return r2_dict if r2_dict else None


# -- plotter ---------------------------------------------------------------


def plot_r2_all_layers(
    r2_per_layer, model, output_dir, emb_r2=None, nl_r2_per_layer=None
):
    """One curve per layer, markers on x-axis, R² on y-axis.

    Parameters
    ----------
    r2_per_layer : dict
        ``{layer_key: {marker_name: r2}}`` — linear probing results.
    model : str
    output_dir : str
    emb_r2 : dict or None
        If given, ``{marker_name: r2}`` from the last-layer embedding
        regressor.  Plotted as a thick dashed black curve.
    nl_r2_per_layer : dict or None
        ``{layer_key: {marker_name: r2}}`` from non-linear probing.
        Plotted as dashed curves with ``(NL)`` suffix in the legend.
    """
    if not r2_per_layer and emb_r2 is None and not nl_r2_per_layer:
        print(f"   [{model}] No regression results found, skipping.", flush=True)
        return

    # Include all sources in the marker ordering computation
    all_curves = dict(r2_per_layer)
    if nl_r2_per_layer:
        all_curves.update(nl_r2_per_layer)
    if emb_r2 is not None:
        all_curves["__emb__"] = emb_r2

    marker_order = _ordered_markers_by_family_and_mean(all_curves)
    short_names = [_short_name(m) for m in marker_order]
    x = np.arange(len(marker_order))

    canonical = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    layers = [lyr for lyr in canonical if lyr in r2_per_layer]
    n_layers = len(layers)
    colors = [LAYER_CMAP(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 7))

    # -- linear probing layer curves (solid) --
    for li, layer in enumerate(layers):
        r2_dict = r2_per_layer[layer]
        y_raw = [r2_dict.get(m, np.nan) for m in marker_order]
        ax.plot(
            x,
            y_raw,
            label=layer,
            color=colors[li],
            marker=LAYER_MARKERS[li % len(LAYER_MARKERS)],
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            alpha=0.9,
        )

    # -- non-linear probing layer curves (dashed) --
    if nl_r2_per_layer:
        nl_layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in nl_r2_per_layer]
        nl_colors = [LAYER_CMAP(i / max(len(nl_layers) - 1, 1)) for i in range(len(nl_layers))]
        for li, layer in enumerate(nl_layers):
            r2_dict = nl_r2_per_layer[layer]
            y_raw = [r2_dict.get(m, np.nan) for m in marker_order]
            ax.plot(
                x,
                y_raw,
                label=f"{layer} (NL)",
                color=nl_colors[li],
                marker=LAYER_MARKERS[li % len(LAYER_MARKERS)],
                linestyle="--",
                linewidth=1.4,
                markersize=4,
                alpha=0.75,
            )

    # -- last-layer embedding curve --
    if emb_r2 is not None:
        y_emb = [emb_r2.get(m, np.nan) for m in marker_order]
        ax.plot(
            x,
            y_emb,
            label="last-layer embedding",
            color="black",
            marker="*",
            linestyle="--",
            linewidth=2.4,
            markersize=8,
            alpha=0.95,
            zorder=10,
        )

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Markers", fontsize=17)
    ax.set_ylabel("R\u00b2", fontsize=17)
    title_suffix = " + NL" if nl_r2_per_layer else ""
    ax.set_title(
        f"{model} \u2014 Raw R\u00b2 (Ridge regression{title_suffix}) across layers",
        fontsize=18,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=12)
    _color_xticks_by_family(ax, marker_order)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper left", fontsize=11, title="Layer", title_fontsize=12)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = op.join(output_dir, f"r2_all_layers_{model}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}", flush=True)


# -- classification loaders ------------------------------------------------


def load_classification_per_layer(output_dir, model):
    """Load per-layer classification JSONs.

    Returns
    -------
    dict : ``{layer: {clf: {auc_mean, auc_std}}}``
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
    dict or None : ``{clf: {auc_mean, auc_std}}``
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
        auc = data.get("macro_average", {}).get("auc_score", {})
        if "mean" in auc:
            results[clf] = {"auc_mean": auc["mean"], "auc_std": auc["std"]}
    return results if results else None


# -- AUC line plot ---------------------------------------------------------


def plot_auc_curves(clf_per_layer, model, output_dir, emb_clf=None, nl_clf_per_layer=None):
    """Line plot: AUC across layers, one curve per classifier with fill_between.

    Parameters
    ----------
    clf_per_layer : dict
        ``{layer: {clf: {auc_mean, auc_std}}}`` — linear probing results.
    model : str
    output_dir : str
    emb_clf : dict or None
        ``{clf: {auc_mean, auc_std}}`` for the last-layer embedding.
    nl_clf_per_layer : dict or None
        ``{layer: {clf: {auc_mean, auc_std}}}`` from non-linear probing.
        Plotted as dashed curves with ``(NL)`` suffix in the legend.
    """
    canonical = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    layers = [lyr for lyr in canonical if lyr in clf_per_layer]
    if emb_clf is not None:
        layers_plot = layers + ["last-layer\nembedding"]
    else:
        layers_plot = list(layers)

    if not layers_plot and not nl_clf_per_layer:
        print(f"   [{model}] No classification data for AUC plot, skipping.", flush=True)
        return

    x = np.arange(len(layers_plot))
    fig, ax = plt.subplots(figsize=(max(7, len(layers_plot) * 1.4), 5))

    # -- linear probing curves (solid) --
    for clf in CLF_NAMES:
        means, stds = [], []
        for layer in layers:
            r = clf_per_layer.get(layer, {}).get(clf, {})
            means.append(r.get("auc_mean", np.nan))
            stds.append(r.get("auc_std", np.nan))
        if emb_clf is not None:
            r = emb_clf.get(clf, {})
            means.append(r.get("auc_mean", np.nan))
            stds.append(r.get("auc_std", np.nan))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        color = CLF_COLORS[clf]

        ax.plot(
            x, means,
            label=CLF_LABELS[clf],
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)

    # -- non-linear probing curves (dashed) --
    if nl_clf_per_layer:
        nl_layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in nl_clf_per_layer]
        nl_x = np.arange(len(nl_layers))
        for clf in CLF_NAMES:
            means, stds = [], []
            for layer in nl_layers:
                r = nl_clf_per_layer.get(layer, {}).get(clf, {})
                means.append(r.get("auc_mean", np.nan))
                stds.append(r.get("auc_std", np.nan))
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)
            color = CLF_COLORS[clf]
            ax.plot(
                nl_x, means,
                label=f"{CLF_LABELS[clf]} (NL)",
                color=color,
                marker="s",
                linewidth=1.6,
                markersize=5,
                linestyle="--",
                alpha=0.75,
            )
            ax.fill_between(nl_x, means - stds, means + stds, color=color, alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(layers_plot, rotation=30, ha="right", fontsize=12)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUC", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    title_suffix = " + NL" if nl_clf_per_layer else ""
    ax.set_title(
        f"{model} \u2014 Classification AUC across layers (VS vs MCS){title_suffix}",
        fontsize=15,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out_path = op.join(output_dir, f"auc_curves_{model}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}", flush=True)


# -- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot R\u00b2 across all layers (one curve per layer).",
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
            "non_linear_probing.py).  When provided, NL curves are overlaid "
            "as dashed lines on the linear plots AND standalone NL plots are "
            "saved to this directory."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CbraMod", "NeuroLM"],
        help="Models to include.",
    )
    args = parser.parse_args()

    models = [m for m in args.models if m in MODEL_LAYERS]
    if not models:
        print("No valid models specified.")
        return

    print("=" * 60, flush=True)
    print("R\u00b2 ALL-LAYERS PLOT", flush=True)
    print(f"  output_dir     : {args.output_dir}", flush=True)
    print(f"  nonlinear_dir  : {args.nonlinear_dir}", flush=True)
    print(f"  models         : {models}", flush=True)
    print("=" * 60, flush=True)

    for model in models:
        r2_per_layer = load_regression_per_layer(args.output_dir, model)
        print(f"   [{model}] Loaded {len(r2_per_layer)} layers (linear)", flush=True)

        emb_r2 = load_embedding_regressor(model)
        if emb_r2 is not None:
            print(
                f"   [{model}] Loaded last-layer embedding regressor "
                f"({len(emb_r2)} markers)",
                flush=True,
            )
        else:
            print(f"   [{model}] No last-layer embedding regressor found", flush=True)

        # Load non-linear probing results if requested
        nl_r2_per_layer = None
        if args.nonlinear_dir:
            nl_r2_per_layer = load_regression_per_layer(args.nonlinear_dir, model)
            print(
                f"   [{model}] Loaded {len(nl_r2_per_layer)} layers (non-linear)",
                flush=True,
            )

        # Combined plot (saved to linear dir)
        plot_r2_all_layers(
            r2_per_layer, model, args.output_dir,
            emb_r2=emb_r2, nl_r2_per_layer=nl_r2_per_layer,
        )

        # Standalone NL plot (saved to nonlinear dir)
        if args.nonlinear_dir and nl_r2_per_layer:
            plot_r2_all_layers(
                nl_r2_per_layer, model, args.nonlinear_dir, emb_r2=None,
            )

        clf_per_layer = load_classification_per_layer(args.output_dir, model)
        print(
            f"   [{model}] Loaded {len(clf_per_layer)} layers (classification, linear)",
            flush=True,
        )
        emb_clf = load_embedding_classification(model)
        if emb_clf is not None:
            print(
                f"   [{model}] Loaded last-layer embedding classification "
                f"({list(emb_clf.keys())})",
                flush=True,
            )
        else:
            print(f"   [{model}] No last-layer embedding classification found", flush=True)

        nl_clf_per_layer = None
        if args.nonlinear_dir:
            nl_clf_per_layer = load_classification_per_layer(args.nonlinear_dir, model)
            print(
                f"   [{model}] Loaded {len(nl_clf_per_layer)} layers "
                f"(classification, non-linear)",
                flush=True,
            )

        # Combined AUC plot (saved to linear dir)
        plot_auc_curves(
            clf_per_layer, model, args.output_dir,
            emb_clf=emb_clf, nl_clf_per_layer=nl_clf_per_layer,
        )

        # Standalone NL AUC plot (saved to nonlinear dir)
        if args.nonlinear_dir and nl_clf_per_layer:
            plot_auc_curves(
                nl_clf_per_layer, model, args.nonlinear_dir, emb_clf=None,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
