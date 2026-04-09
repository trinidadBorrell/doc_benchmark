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
    "LaBram": [
        "vqnsp_enc_patch_embed",
        "vqnsp_enc_block_2",
        "vqnsp_enc_block_5",
        "vqnsp_enc_block_8",
        "vqnsp_enc_block_11",
        "vqnsp_encode_task",
        "vqnsp_quantize",
      #  "vqnsp_dec_block_0",
      #  "vqnsp_dec_block_1",
      #  "vqnsp_dec_block_2",
      #  "vqnsp_decode_task",
        "labram_patch_embed",
        "labram_block_2",
        "labram_block_5",
        "labram_block_8",
        "labram_block_11",
        "labram_norm",
    ],
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
    "LaBram": op.join(
        _BENCHMARK_ROOT,
        "LaBram/doc_patients/MLP_EMBEDDING/regressor_results/summary.json",
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
    "LaBram": op.join(
        _BENCHMARK_ROOT,
        "LaBram/doc_patients/MLP_EMBEDDING/crs/nested_cv",
    ),
}

FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_COLORS = {
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",
    "wsmi": "#8b4513",
    "evoked": "#c55a11",
}
FAMILY_DISPLAY = {
    "evoked": "Evoked",
    "wsmi": "Connectivity",
    "information_theory": "Information Theory",
    "power_spectral_density": "Power Spectral Density",
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
    r2_per_layer, model, output_dir, emb_r2=None, nl_r2_per_layer=None,
    scatter=False,
):
    """One curve (or scatter) per layer, markers on x-axis, R² on y-axis.

    Parameters
    ----------
    r2_per_layer : dict
        ``{layer_key: {marker_name: r2}}`` — linear probing results.
    model : str
    output_dir : str
    emb_r2 : dict or None
        If given, ``{marker_name: r2}`` from the last-layer embedding
        regressor.  Plotted as a thick dashed black curve / larger stars.
    nl_r2_per_layer : dict or None
        ``{layer_key: {marker_name: r2}}`` from non-linear probing.
        Plotted as dashed / open markers with ``(NL)`` suffix.
    scatter : bool
        When True, draw points only (no connecting lines).  Output filename
        gets a ``_scatter`` suffix.
    """
    if not r2_per_layer and emb_r2 is None and not nl_r2_per_layer:
        print(f"   [{model}] No regression results found, skipping.", flush=True)
        return

    ls_main = "" if scatter else "-"
    ls_nl   = "" if scatter else "--"
    ls_emb  = "" if scatter else "--"
    ms_main = 7  if scatter else 5
    ms_nl   = 6  if scatter else 4
    ms_emb  = 11 if scatter else 8
    lw_main = 0  if scatter else 1.6
    lw_nl   = 0  if scatter else 1.4
    lw_emb  = 0  if scatter else 2.4

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

    if model == "LaBram":
        fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 10))
    else:
        fig, ax = plt.subplots(figsize=(max(16, len(marker_order) * 0.65), 7))

    # -- linear probing layer curves / points --
    for li, layer in enumerate(layers):
        r2_dict = r2_per_layer[layer]
        y_raw = [r2_dict.get(m, np.nan) for m in marker_order]
        ax.plot(
            x, y_raw,
            label=layer,
            color=colors[li],
            marker=LAYER_MARKERS[li % len(LAYER_MARKERS)],
            linestyle=ls_main,
            linewidth=lw_main,
            markersize=ms_main,
            alpha=0.9,
        )

    # -- non-linear probing layer curves / points --
    if nl_r2_per_layer:
        nl_layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in nl_r2_per_layer]
        nl_colors = [LAYER_CMAP(i / max(len(nl_layers) - 1, 1)) for i in range(len(nl_layers))]
        for li, layer in enumerate(nl_layers):
            r2_dict = nl_r2_per_layer[layer]
            y_raw = [r2_dict.get(m, np.nan) for m in marker_order]
            ax.plot(
                x, y_raw,
                label=f"{layer} (NL)",
                color=nl_colors[li],
                marker=LAYER_MARKERS[li % len(LAYER_MARKERS)],
                linestyle=ls_nl,
                linewidth=lw_nl,
                markersize=ms_nl,
                alpha=0.75,
                fillstyle="none" if scatter else "full",
            )

    # -- last-layer embedding curve / points --
    if emb_r2 is not None:
        y_emb = [emb_r2.get(m, np.nan) for m in marker_order]
        ax.plot(
            x, y_emb,
            label="last-layer embedding",
            color="black",
            marker="*",
            linestyle=ls_emb,
            linewidth=lw_emb,
            markersize=ms_emb,
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
    suffix = "_scatter" if scatter else ""
    out_path = op.join(output_dir, f"r2_all_layers_{model}{suffix}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}", flush=True)


def animate_r2_layers(r2_per_layer, model, output_dir, fps=12, interp_frames=7,
                      scatter=False):
    """Animate R² curves sweeping from first to last layer with smooth morphing.

    Each pair of consecutive layers is bridged by ``interp_frames`` frames that
    linearly interpolate both y-values and colour, giving a continuous transition.
    Completed layers are kept as a faint trail in the background.

    Requires ``pillow`` (``pip install pillow``).
    Saved as ``r2_layers_animated_{model}.gif`` (or ``_scatter.gif`` when
    ``scatter=True``).
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print(
            f"   [{model}] matplotlib.animation unavailable, skipping GIF.",
            flush=True,
        )
        return

    if not r2_per_layer:
        print(f"   [{model}] No regression results for animation, skipping.", flush=True)
        return

    canonical = list(MODEL_LAYERS.get(model, []))
    layers = [lyr for lyr in canonical if lyr in r2_per_layer]
    if len(layers) < 2:
        print(
            f"   [{model}] Need ≥2 layers for animation, got {len(layers)}, skipping.",
            flush=True,
        )
        return

    marker_order = _ordered_markers_by_family_and_mean(r2_per_layer)
    short_names = [_short_name(m) for m in marker_order]
    x = np.arange(len(marker_order))

    n_layers = len(layers)
    colors = [LAYER_CMAP(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    y_arrays = [
        np.array([r2_per_layer[lyr].get(m, np.nan) for m in marker_order], dtype=float)
        for lyr in layers
    ]

    # ── build frame list ────────────────────────────────────────────────────
    # Each entry: (y_interp, rgba_color, marker_symbol_idx, layer_label, n_trail)
    # n_trail = how many completed layers to show as faint background curves
    frames = []

    for i in range(n_layers - 1):
        for t_idx in range(interp_frames):
            t = t_idx / interp_frames  # [0, 1)
            y = (1.0 - t) * y_arrays[i] + t * y_arrays[i + 1]
            c0, c1 = np.array(colors[i]), np.array(colors[i + 1])
            color = tuple((1.0 - t) * c0 + t * c1)
            label = layers[i] if t < 0.5 else layers[i + 1]
            frames.append((y, color, i, label, i))  # trail = layers 0..i-1

    # Final layer — hold for 1 second
    for _ in range(fps + 1):
        frames.append((y_arrays[-1], colors[-1], n_layers - 1, layers[-1], n_layers - 1))

    # ── figure setup ────────────────────────────────────────────────────────
    figsize = (max(16, len(marker_order) * 0.65), 10 if model == "LaBram" else 7)
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Markers", fontsize=17)
    ax.set_ylabel("R\u00b2", fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.5, len(marker_order) - 0.5)

    all_vals_flat = [v for y in y_arrays for v in y if not np.isnan(v)]
    ymin = min(min(all_vals_flat) - 0.05, -0.05) if all_vals_flat else -0.1
    ymax = (max(all_vals_flat) + 0.05) if all_vals_flat else 0.5
    ax.set_ylim(ymin, ymax)

    # Force draw so tick labels exist before colouring them
    fig.canvas.draw()
    _color_xticks_by_family(ax, marker_order)

    trail_lw     = 0   if scatter else 0.9
    trail_ms     = 4   if scatter else 3
    active_lw    = 0   if scatter else 1.8
    active_ms    = 7   if scatter else 5

    # Pre-initialise one line artist per layer for the trail (all hidden)
    trail_lines = []
    for i in range(n_layers):
        (tl,) = ax.plot(
            [], [],
            color=colors[i], linewidth=trail_lw, alpha=0.15, zorder=1,
            marker=LAYER_MARKERS[i % len(LAYER_MARKERS)], markersize=trail_ms,
        )
        trail_lines.append(tl)

    # The single morphing curve drawn on top
    (active_line,) = ax.plot(
        [], [], linewidth=active_lw, markersize=active_ms, alpha=0.95, zorder=2,
    )

    title_obj = ax.set_title("", fontsize=18)

    plt.tight_layout()

    def init():
        for tl in trail_lines:
            tl.set_data([], [])
            tl.set_visible(False)
        active_line.set_data([], [])
        return trail_lines + [active_line, title_obj]

    def update(frame_idx):
        y, color, sym_idx, label, n_trail = frames[frame_idx]

        for i, tl in enumerate(trail_lines):
            if i < n_trail:
                tl.set_data(x, y_arrays[i])
                tl.set_visible(True)
            else:
                tl.set_visible(False)

        active_line.set_data(x, y)
        active_line.set_color(color)
        active_line.set_marker(LAYER_MARKERS[sym_idx % len(LAYER_MARKERS)])
        title_obj.set_text(
            f"{model} \u2014 Raw R\u00b2 (Ridge regression) | {label}"
        )
        return trail_lines + [active_line, title_obj]

    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        init_func=init,
        blit=False,
        interval=1000 // fps,
    )

    gif_suffix = "_scatter" if scatter else ""
    out_path = op.join(output_dir, f"r2_layers_animated_{model}{gif_suffix}.gif")
    try:
        anim.save(out_path, writer="pillow", fps=fps, dpi=120)
        print(f"   Saved: {out_path}", flush=True)
    except Exception as e:
        print(f"   [WARN] Failed to save animated GIF: {e}", flush=True)
    finally:
        plt.close()


# -- R² by marker family across layers ------------------------------------


def plot_r2_by_family_layers(r2_per_layer, model, output_dir, scatter=False):
    """4-row subplot: mean R² ± std per marker family across layers.

    Each row corresponds to one marker family (Evoked, Connectivity,
    Information Theory, Power Spectral Density).  The curve shows the mean R²
    across all markers in that family per layer; ``fill_between`` shows ±1 std.

    Parameters
    ----------
    r2_per_layer : dict
        ``{layer_key: {marker_name: r2}}``
    model : str
    output_dir : str
    scatter : bool
        When True, draw points only (no connecting lines, no fill_between).
        Output filename gets ``_scatter`` suffix.
    """
    if not r2_per_layer:
        print(f"   [{model}] No regression results for family plot, skipping.", flush=True)
        return

    canonical = list(MODEL_LAYERS.get(model, [])) + ["last_layer"]
    layers = [lyr for lyr in canonical if lyr in r2_per_layer]
    if not layers:
        print(f"   [{model}] No layers found for family plot, skipping.", flush=True)
        return

    x = np.arange(len(layers))

    # Group all markers by family across all layers
    all_markers = set()
    for lyr in layers:
        all_markers.update(r2_per_layer[lyr].keys())

    family_markers = {fam: [] for fam in FAMILY_ORDER}
    for marker in all_markers:
        fam = _marker_family(marker)
        family_markers.setdefault(fam, []).append(marker)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(max(9, len(layers) * 1.5), 12),
        sharex=True,
    )

    ls = "" if scatter else "-"
    lw = 0  if scatter else 2.0
    ms = 8  if scatter else 6

    for ax, fam in zip(axes, FAMILY_ORDER):
        markers_in_fam = sorted(family_markers.get(fam, []))
        color = FAMILY_COLORS[fam]
        display = FAMILY_DISPLAY[fam]

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_ylim(-1.0, 1.0)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.25)

        if not markers_in_fam:
            ax.text(
                0.5, 0.5, "No markers in this family",
                transform=ax.transAxes, ha="center", va="center",
                color="gray", fontsize=11,
            )
            ax.set_title(display, fontsize=13, loc="left", pad=4, color=color)
            continue

        means, stds = [], []
        for lyr in layers:
            r2_dict = r2_per_layer[lyr]
            vals = [
                r2_dict[m] for m in markers_in_fam
                if m in r2_dict and not np.isnan(r2_dict[m])
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            x, means,
            color=color, marker="o", linestyle=ls, linewidth=lw,
            markersize=ms, alpha=0.9,
        )
        if not scatter:
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.20)

        ax.set_title(
            f"{display}  (n={len(markers_in_fam)} markers)",
            fontsize=13, loc="left", pad=4, color=color,
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(layers, rotation=30, ha="right", fontsize=11)
    axes[-1].set_xlabel("Layer", fontsize=13)

    fig.suptitle(
        f"{model} \u2014 Mean R\u00b2 per marker family across layers",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()

    suffix = "_scatter" if scatter else ""
    out_path = op.join(output_dir, f"r2_by_family_{model}{suffix}.png")
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


def plot_auc_curves(clf_per_layer, model, output_dir, emb_clf=None,
                    nl_clf_per_layer=None, scatter=False):
    """Line plot (or scatter): AUC across layers, one series per classifier.

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
        Plotted as dashed / open markers with ``(NL)`` suffix.
    scatter : bool
        When True, draw points only (no connecting lines).  ``fill_between``
        error bands are also omitted.  Output filename gets ``_scatter`` suffix.
    """
    ls_main = "" if scatter else "-"
    ls_nl   = "" if scatter else "--"
    lw_main = 0  if scatter else 2.0
    lw_nl   = 0  if scatter else 1.6
    ms_main = 8  if scatter else 6
    ms_nl   = 7  if scatter else 5

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

    # -- linear probing curves / points --
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
            linewidth=lw_main,
            linestyle=ls_main,
            markersize=ms_main,
        )
        if not scatter:
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)

    # -- non-linear probing curves / points --
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
                linewidth=lw_nl,
                linestyle=ls_nl,
                markersize=ms_nl,
                alpha=0.75,
                fillstyle="none" if scatter else "full",
            )
            if not scatter:
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
    suffix = "_scatter" if scatter else ""
    out_path = op.join(output_dir, f"auc_curves_{model}{suffix}.png")
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
        default=["CbraMod", "NeuroLM", "LaBram"],
        help="Models to include.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip animated GIF generation (r2_layers_animated_*.gif).",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frames per second for the animated GIF (default: 12).",
    )
    parser.add_argument(
        "--gif-interp-frames",
        type=int,
        default=18,
        help="Interpolation frames between each pair of consecutive layers (default: 18).",
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

        # Combined plot (saved to linear dir) — line + scatter versions
        for _scatter in (False, True):
            plot_r2_all_layers(
                r2_per_layer, model, args.output_dir,
                emb_r2=emb_r2, nl_r2_per_layer=nl_r2_per_layer,
                scatter=_scatter,
            )

        # R² by family — line + scatter versions
        for _scatter in (False, True):
            plot_r2_by_family_layers(
                r2_per_layer, model, args.output_dir, scatter=_scatter,
            )

        # Animated GIF — line + scatter versions
        if not args.skip_gif:
            for _scatter in (False, True):
                print(
                    f"   [{model}] Generating animated GIF "
                    f"({'scatter' if _scatter else 'line'}) …",
                    flush=True,
                )
                animate_r2_layers(
                    r2_per_layer, model, args.output_dir,
                    fps=args.gif_fps,
                    interp_frames=args.gif_interp_frames,
                    scatter=_scatter,
                )

        # Standalone NL plot (saved to nonlinear dir) — line + scatter versions
        if args.nonlinear_dir and nl_r2_per_layer:
            for _scatter in (False, True):
                plot_r2_all_layers(
                    nl_r2_per_layer, model, args.nonlinear_dir, emb_r2=None,
                    scatter=_scatter,
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

        # Combined AUC plot (saved to linear dir) — line + scatter versions
        for _scatter in (False, True):
            plot_auc_curves(
                clf_per_layer, model, args.output_dir,
                emb_clf=emb_clf, nl_clf_per_layer=nl_clf_per_layer,
                scatter=_scatter,
            )

        # Standalone NL AUC plot (saved to nonlinear dir) — line + scatter versions
        if args.nonlinear_dir and nl_clf_per_layer:
            for _scatter in (False, True):
                plot_auc_curves(
                    nl_clf_per_layer, model, args.nonlinear_dir, emb_clf=None,
                    scatter=_scatter,
                )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
