#!/usr/bin/env python3
"""Layer-wise R² summary across foundation models.

Produces two figures in data/benchmark_results/new_results/PLOTS/:

  - plot3_family_lines.png   (families × models, mean ± std line per family;
                               evoked row restricted to TimeLockedTopography
                               p1/p3a/p3b and ContingentNegativeVariation/default)
  - plot3_family_boxes.png   (families × models, boxplot + swarm + paired lines;
                               evoked row restricted to the same 4 markers)

Data source: per-layer ridge regression ``summary.json`` files written by
``src/interp/linear_probing.py`` to
``LINEAR_PROBING/regression/{layer}/{model}/summary.json``.

Grid layout (same for both figures):

  rows    → 4 marker families (Evoked, Connectivity, Info-theory, PSD)
  columns → models (BIOT, TOTEM, LaBraM, NeuroLM, CBraMod — EEGPT skipped
            when no data on disk)
  x-axis  → layer name for that model (columns do not share x)
  y-axis  → R² (shared across all panels)

Plot B additionally overlays a swarm plot and connects the same marker
across layers with thin paired lines so the reader can see layer-to-layer
correspondence per marker.
"""
from __future__ import annotations

import json
import os.path as op
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["axes.labelsize"] = "medium"


BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/paper_results",
))
LP_ROOT = BASE / "LINEAR_PROBING"
OUT = BASE / "PLOTS"

MODEL_ORDER = ["BIOT", "TOTEM", "LaBram", "EEGPT", "NeuroLM", "CbraMod"]
MODEL_DISPLAY = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBraM",
    "EEGPT": "EEGPT",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
}

MODEL_LAYERS = {
    "BIOT": [
        "pre_transformer",
        "transformer_0",
        "transformer_1",
        "transformer_2",
        "transformer_3",
    ],
    "TOTEM": [
        "totem_enc_conv1",
        "totem_enc_conv2",
        "totem_enc_conv3",
        "totem_enc_residual",
        "totem_enc_z",
        "totem_vq_quantized",
        "totem_dec_conv1",
        "totem_dec_residual",
    ],
    "LaBram": [
        "vqnsp_enc_patch_embed",
        "vqnsp_enc_block_2",
        "vqnsp_enc_block_5",
        "vqnsp_enc_block_8",
        "vqnsp_enc_block_11",
        "vqnsp_encode_task",
        "vqnsp_quantize",
        "labram_patch_embed",
        "labram_block_2",
        "labram_block_5",
        "labram_block_8",
        "labram_block_11",
        "labram_norm",
    ],
    "EEGPT": [],
    "NeuroLM": [
        "vq_emb",
        "vq_enc_patch_emb",
        "vq_enc_block_2",
        "vq_enc_block_5",
        "vq_enc_block_8",
        "vq_enc_block_11",
        "vq_encode_task",
        "vq_quantize",
        "vq_dec_freq_block_0",
        "vq_dec_freq_block_1",
        "vq_dec_freq_block_2",
        "vq_dec_freq_block_3",
        "gpt_0",
        "gpt_3",
        "gpt_6",
        "gpt_9",
        "gpt_11",
    ],
    "CbraMod": [
        "patch_emb",
        "layer_0",
        "layer_3",
        "layer_6",
        "layer_9",
        "layer_11",
    ],
}

LAYER_SHORT = {
    # BIOT
    "pre_transformer": "pre",
    "transformer_0": "T0",
    "transformer_1": "T1",
    "transformer_2": "T2",
    "transformer_3": "T3",
    "final_emb": "final",
    # TOTEM
    "totem_enc_conv1": "EC1",
    "totem_enc_conv2": "EC2",
    "totem_enc_conv3": "EC3",
    "totem_enc_residual": "ERes",
    "totem_enc_z": "Ez",
    "totem_vq_quantized": "VQq",
    "totem_dec_conv1": "DC1",
    "totem_dec_residual": "DRes",
    # LaBraM
    "vqnsp_enc_patch_embed": "VNpatch",
    "vqnsp_enc_block_2": "VNE2",
    "vqnsp_enc_block_5": "VNE5",
    "vqnsp_enc_block_8": "VNE8",
    "vqnsp_enc_block_11": "VNE11",
    "vqnsp_encode_task": "VNtask",
    "vqnsp_quantize": "VNq",
    "labram_patch_embed": "Lpatch",
    "labram_block_2": "L2",
    "labram_block_5": "L5",
    "labram_block_8": "L8",
    "labram_block_11": "L11",
    "labram_norm": "Lnorm",
    # NeuroLM
    "vq_emb": "VQemb",
    "vq_enc_patch_emb": "VQpatch",
    "vq_enc_block_2": "VQE2",
    "vq_enc_block_5": "VQE5",
    "vq_enc_block_8": "VQE8",
    "vq_enc_block_11": "VQE11",
    "vq_encode_task": "VQtask",
    "vq_quantize": "VQq",
    "vq_dec_freq_block_0": "VQD0",
    "vq_dec_freq_block_1": "VQD1",
    "vq_dec_freq_block_2": "VQD2",
    "vq_dec_freq_block_3": "VQD3",
    "gpt_0": "G0",
    "gpt_3": "G3",
    "gpt_6": "G6",
    "gpt_9": "G9",
    "gpt_11": "G11",
    # CBraMod
    "patch_emb": "patch",
    "layer_0": "L0",
    "layer_3": "L3",
    "layer_6": "L6",
    "layer_9": "L9",
    "layer_11": "L11",
}


# Evoked markers kept in the Evoked row of plot_family_lines (only).
# plot_family_boxes keeps the full evoked set unchanged.
EVOKED_LINES_WHITELIST = {
    "nice/marker/TimeLockedTopography/p1",
    "nice/marker/TimeLockedTopography/p3a",
    "nice/marker/TimeLockedTopography/p3b",
    "nice/marker/ContingentNegativeVariation/default",
}


FAMILY_ORDER = ["evoked", "wsmi", "information_theory", "power_spectral_density"]
FAMILY_DISPLAY = {
    "evoked": "E",
    "wsmi": "C",
    "information_theory": "IT",
    "power_spectral_density": "Spectral",
}
FAMILY_COLORS = {
    "evoked": "#c55a11",
    "wsmi": "#8b4513",
    "information_theory": "black",
    "power_spectral_density": "#7f3fbf",
}

# Per-model palette — must match plot4.py FM_COLORS so the two figures
# reuse the same color identity for each FM across the paper.
MODEL_COLORS = {
    "BIOT": "#1f77b4",
    "TOTEM": "#d62728",
    "LaBram": "#2ca02c",
    "NeuroLM": "#9467bd",
    "CbraMod": "#ff7f0e",
    "EEGPT": "#8c564b",
}


def _marker_family(marker_name: str) -> str:
    if "KolmogorovComplexity" in marker_name or "PermutationEntropy" in marker_name:
        return "information_theory"
    if "PowerSpectralDensity" in marker_name:
        return "power_spectral_density"
    if "SymbolicMutualInformation" in marker_name:
        return "wsmi"
    return "evoked"


def load_regression_per_layer(model: str) -> dict[str, dict[str, float]]:
    """Return ``{layer: {marker: r2}}`` for the given model."""
    reg_root = LP_ROOT / "regression"
    r2_per_layer: dict[str, dict[str, float]] = {}
    for layer in MODEL_LAYERS.get(model, []):
        path = reg_root / layer / model / "summary.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        r2_dict = {
            m: float(v["r2"])
            for m, v in metrics.items()
            if isinstance(v, dict) and not v.get("skipped", False) and "r2" in v
        }
        if r2_dict:
            r2_per_layer[layer] = r2_dict
    return r2_per_layer


# MLP_EMBEDDING on-disk dir (differs from model name only for CBraMod).
_MLP_EMB_DIR = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBram",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
    "EEGPT": "EEGPT",
}

# Pseudo-layer key used for the MLP_EMBEDDING last-layer regression summary.
LAST_LAYER_KEY = "mlp_last"
LAYER_SHORT[LAST_LAYER_KEY] = "Last"


def load_last_layer_r2(model: str) -> dict[str, float] | None:
    """Load ``{FM}/doc_patients/MLP_EMBEDDING/regressor_results/summary.json``.

    Returns ``{marker: r2}`` (skipped/missing entries dropped) or ``None`` if
    no regression summary exists for this model.
    """
    on_disk = _MLP_EMB_DIR.get(model, model)
    candidates = [
        BASE / on_disk / "doc_patients" / "MLP_EMBEDDING"
             / "regressor_results" / "summary.json",
        BASE / on_disk / "doc_patients" / "MLP_EMBEDDING" / on_disk
             / "doc_patients" / "MLP_EMBEDDING"
             / "regressor_results" / "summary.json",
    ]
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            r2_dict = {
                m: float(v["r2"])
                for m, v in metrics.items()
                if isinstance(v, dict) and not v.get("skipped", False)
                and "r2" in v
            }
            return r2_dict or None
    return None


def build_family_matrix(
    r2_per_layer: dict[str, dict[str, float]],
    markers_in_family: list[str],
    layers: list[str],
) -> np.ndarray:
    """Return an ``(n_markers, n_layers)`` R² matrix (NaN when missing)."""
    mat = np.full((len(markers_in_family), len(layers)), np.nan)
    for i, marker in enumerate(markers_in_family):
        for j, layer in enumerate(layers):
            val = r2_per_layer.get(layer, {}).get(marker)
            if val is not None and not np.isnan(val):
                mat[i, j] = val
    return mat


def family_markers(
    r2_per_layer: dict[str, dict[str, float]],
    fam: str,
) -> list[str]:
    seen: set[str] = set()
    for markers in r2_per_layer.values():
        seen.update(markers.keys())
    return sorted(m for m in seen if _marker_family(m) == fam)


def resolve_models() -> list[str]:
    """Keep only models that actually have at least one layer's summary.json."""
    present: list[str] = []
    for model in MODEL_ORDER:
        r2 = load_regression_per_layer(model)
        if r2:
            present.append(model)
        else:
            print(f"[plot3] dropping {model}: no regression summaries on disk")
    return present


# ---------------------------------------------------------------------------
# Plot A — mean ± std per family across layers (line plot)
# ---------------------------------------------------------------------------


def plot_family_lines(
    data: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    out_path: Path,
) -> None:
    n_rows = len(FAMILY_ORDER)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 1, 2.4 * n_rows + 1),
        sharey=True,
        squeeze=False,
    )

    for ci, model in enumerate(models):
        r2_per_layer = data[model]
        layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in r2_per_layer]
        x = np.arange(len(layers))
        short = [LAYER_SHORT.get(lyr, lyr) for lyr in layers]

        for ri, fam in enumerate(FAMILY_ORDER):
            ax = axes[ri, ci]
            color = FAMILY_COLORS[fam]

            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
            markers_in_fam = family_markers(r2_per_layer, fam)
            if fam == "evoked":
                markers_in_fam = [
                    m for m in markers_in_fam if m in EVOKED_LINES_WHITELIST
                ]
            mat = build_family_matrix(r2_per_layer, markers_in_fam, layers)

            if mat.size and not np.isnan(mat).all():
                means = np.nanmean(mat, axis=0)
                stds = np.nanstd(mat, axis=0)
                ax.plot(
                    x, means,
                    color=color, marker="o", linewidth=1.6,
                    markersize=5, alpha=0.95,
                )
                ax.fill_between(
                    x, means - stds, means + stds, color=color, alpha=0.20,
                )
            else:
                ax.text(
                    0.5, 0.5, "no markers",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9,
                )

            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="y", labelsize=9)

            if ri == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=14)
            if ri == n_rows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(short, rotation=45, ha="right", fontsize=10)
                ax.set_xlabel("Layer", fontsize=14)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(
                    f"{FAMILY_DISPLAY[fam]}\n$R^2$",
                    fontsize=12, color=color,
                )

    fig.suptitle(
        "Mean $R^2$ per marker family across layers",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot B — boxplot + swarm + paired lines
# ---------------------------------------------------------------------------


def _jitter(n: int, width: float = 0.18, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def plot_family_boxes(
    data: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    out_path: Path,
) -> None:
    n_rows = len(FAMILY_ORDER)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols + 1, 2.7 * n_rows + 1),
        sharey=True,
        squeeze=False,
    )

    for ci, model in enumerate(models):
        r2_per_layer = dict(data[model])  # copy so we can append "last" layer
        layers = [lyr for lyr in MODEL_LAYERS[model] if lyr in r2_per_layer]

        # Append the MLP_EMBEDDING last-layer regression as an extra column.
        last_r2 = load_last_layer_r2(model)
        if last_r2:
            r2_per_layer[LAST_LAYER_KEY] = last_r2
            layers.append(LAST_LAYER_KEY)
        else:
            print(f"[plot3]   {model}: no MLP_EMBEDDING last-layer regression")

        x = np.arange(1, len(layers) + 1)  # boxplot positions are 1-indexed
        short = [LAYER_SHORT.get(lyr, lyr) for lyr in layers]

        for ri, fam in enumerate(FAMILY_ORDER):
            ax = axes[ri, ci]
            # Columns colored per model (matches plot4_crs_rf_overview.png).
            color = MODEL_COLORS.get(model, FAMILY_COLORS[fam])

            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
            markers_in_fam = family_markers(r2_per_layer, fam)
            if fam == "evoked":
                markers_in_fam = [
                    m for m in markers_in_fam if m in EVOKED_LINES_WHITELIST
                ]
            mat = build_family_matrix(r2_per_layer, markers_in_fam, layers)

            if mat.size and not np.isnan(mat).all():
                # Boxplot — drop NaNs column-wise.
                box_data = [col[~np.isnan(col)] for col in mat.T]
                bp = ax.boxplot(
                    box_data,
                    positions=x,
                    widths=0.55,
                    showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color=color, linewidth=1.3),
                    whiskerprops=dict(color=color, linewidth=1.0, alpha=0.8),
                    capprops=dict(color=color, linewidth=1.0, alpha=0.8),
                    boxprops=dict(
                        facecolor=color, edgecolor=color, alpha=0.22, linewidth=1.0,
                    ),
                )
                _ = bp  # silence linter

                # Paired lines — one per marker across layers (same sample).
                for i in range(mat.shape[0]):
                    y = mat[i]
                    mask = ~np.isnan(y)
                    if mask.sum() < 2:
                        continue
                    ax.plot(
                        x[mask], y[mask],
                        color=color, linewidth=0.6, alpha=0.28, zorder=2,
                    )

                # Swarm points — jittered per column.
                for j, col in enumerate(mat.T):
                    mask = ~np.isnan(col)
                    if not mask.any():
                        continue
                    n = mask.sum()
                    jx = x[j] + _jitter(n, width=0.15, seed=j)
                    ax.scatter(
                        jx, col[mask],
                        s=11, color=color, alpha=0.75,
                        edgecolors="white", linewidths=0.4, zorder=3,
                    )
            else:
                ax.text(
                    0.5, 0.5, "no markers",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9,
                )

            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.25, axis="y")
            ax.tick_params(axis="y", labelsize=9)

            if ri == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=12)
            if ri == n_rows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Layer", fontsize=10)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(
                    f"{FAMILY_DISPLAY[fam]}\n$R^2$",
                    fontsize=10, color="black",
                )

            # Force a reasonable xlim so the boxes are centred.
            if len(layers):
                ax.set_xlim(0.3, len(layers) + 0.7)

    fig.suptitle(
        "$R^2$ by layer — boxplot + swarm + per-marker paired lines",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3] wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = resolve_models()
    print(f"[plot3] models present: {models}")

    data: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        data[model] = load_regression_per_layer(model)
        print(f"[plot3]   {model}: {len(data[model])} layers")

    plot_family_lines(data, models, OUT / "plot3_family_lines.png")
    plot_family_boxes(data, models, OUT / "plot3_family_boxes.png")


if __name__ == "__main__":
    main()
