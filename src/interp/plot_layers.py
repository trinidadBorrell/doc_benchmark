"""Combined R² across layers — one curve per layer, markers on x-axis.

Reads the per-layer regression ``summary.json`` files produced by
``linear_probing.py`` and draws a single figure per model with:

* **x-axis** — neurophysiological markers (grouped by family)
* **y-axis** — R²
* **one curve per layer** (different colour)

Usage
-----
::

    python plot_layers.py --output-dir /path/to/LINEAR_PROBING
    python plot_layers.py --output-dir /path/to/LINEAR_PROBING --models CbraMod

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
    for layer in MODEL_LAYERS[model]:
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


def plot_r2_all_layers(r2_per_layer, model, output_dir, emb_r2=None):
    """One curve per layer, markers on x-axis, R² on y-axis.

    Parameters
    ----------
    r2_per_layer : dict
        ``{layer_key: {marker_name: r2}}``
    model : str
    output_dir : str
    emb_r2 : dict or None
        If given, ``{marker_name: r2}`` from the last-layer embedding
        regressor.  Plotted as a thick dashed black curve labelled
        ``"last-layer embedding"``.
    """
    if not r2_per_layer and emb_r2 is None:
        print(f"   [{model}] No regression results found, skipping.", flush=True)
        return

    # Include embedding markers in the ordering computation
    all_curves = dict(r2_per_layer)
    if emb_r2 is not None:
        all_curves["__emb__"] = emb_r2

    marker_order = _ordered_markers_by_family_and_mean(all_curves)
    short_names = [_short_name(m) for m in marker_order]
    x = np.arange(len(marker_order))

    layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in r2_per_layer]
    n_layers = len(layers)
    colors = [LAYER_CMAP(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 7))

    # -- layer curves --
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
    ax.set_title(
        f"{model} \u2014 Raw R\u00b2 (Ridge regression) across layers",
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
    print(f"  output_dir : {args.output_dir}", flush=True)
    print(f"  models     : {models}", flush=True)
    print("=" * 60, flush=True)

    for model in models:
        r2_per_layer = load_regression_per_layer(args.output_dir, model)
        print(f"   [{model}] Loaded {len(r2_per_layer)} layers", flush=True)

        emb_r2 = load_embedding_regressor(model)
        if emb_r2 is not None:
            print(
                f"   [{model}] Loaded last-layer embedding regressor "
                f"({len(emb_r2)} markers)",
                flush=True,
            )
        else:
            print(f"   [{model}] No last-layer embedding regressor found", flush=True)

        plot_r2_all_layers(r2_per_layer, model, args.output_dir, emb_r2=emb_r2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
