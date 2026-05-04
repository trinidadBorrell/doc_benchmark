#!/usr/bin/env python3
"""Random-Forest AUC along FM layers vs. the DK baseline.

Produces two figures in data/benchmark_results/new_results/PLOTS/:

  - plot3_auc_layers_points.png
        One wide panel. x-axis: [DK, BIOT, TOTEM, LaBram, NeuroLM, CbraMod].
        At each FM tick we draw one error-bar point per layer, jittered
        horizontally, same color per FM (different colors across FMs).
        DK is a single error-bar point. No layer names are shown.

  - plot3_auc_layers_boxes.png
        2 x 3 grid (first cell = DK baseline panel, remaining 5 = FMs).
        Each FM subplot has one vertical error-bar box per layer on the
        x-axis (layer short name), subtitle = FM name, and the DK baseline
        is always drawn as the left-most box of each subplot for reference.

Random Forest is the only classifier considered.
Data sources:
  - FM layer AUC:
        LINEAR_PROBING/classification/{layer}/random_forest_{MODEL}_results.json
        ({"auc_mean": float, "auc_std": float, ...})
  - DK baseline AUC:
        MARKER_BASELINE/crs/nested_cv/random_forest/classification_results.json
        (data["macro_average"]["auc_score"]["mean"|"std"])
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot3a as plot3  # noqa: E402 — same-dir module for combined R² + AUC figure
import os


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"


BASE = Path(os.environ.get(
    "EEG_RESULTS_ROOT",
    "/data/project/eeg_foundation/data/benchmark_results/new_results",
))
LP_CLS = BASE / "LINEAR_PROBING" / "classification"
# Priority path: repeated CV with per-fold classifier selection
LP_REPEATED = BASE / "LINEAR_PROBING" / "nested_cv_repeated" / "best_clf"
# Paper-results root: source of the canonical CRS classification result for the
# "Last" tick in the AUC row (uses the cleaner nested_cv_repeated/best_clf path).
PAPER_RESULTS = Path(
    "/data/project/eeg_foundation/data/benchmark_results/paper_results"
)
OUT = BASE / "PLOTS"

CLASSIFIERS_TO_TRY = ["random_forest", "svm", "mlp", "xgboost", "kernel_ridge"]

MODEL_ORDER = ["BIOT", "TOTEM", "LaBram", "NeuroLM", "CbraMod"]
# Alternate column order: EEGPT replaces TOTEM. DK is still drawn as the AUC
# baseline (dotted reference line), not as a column, because the R² rows have
# no DK data.
MODEL_ORDER_EEGPT = ["BIOT", "LaBram", "EEGPT", "NeuroLM", "CbraMod"]

FAMILY_FULL_DISPLAY = {
    "evoked": "Evoked",
    "wsmi": "Connectivity",
    "information_theory": "Information\nTheory",
    "power_spectral_density": "Spectral",
}

MODEL_DISPLAY = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBraM",
    "EEGPT": "EEGPT",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
}

# One color per FM; DK gets its own gray.
DK_COLOR = "#555555"
FM_COLORS = {
    "BIOT": "#1f77b4",
    "TOTEM": "#d62728",
    "LaBram": "#2ca02c",
    "EEGPT": "#8c564b",
    "NeuroLM": "#9467bd",
    "CbraMod": "#ff7f0e",
}


MODEL_LAYERS = {
    "BIOT": [
        "pre_transformer",
#        "transformer_0",
        "transformer_1",
#        "transformer_2",
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
#        "vqnsp_enc_block_5",
        "vqnsp_enc_block_8",
#        "vqnsp_enc_block_11",
        "vqnsp_encode_task",
#        "vqnsp_quantize",
        "labram_patch_embed",
#        "labram_block_2",
        "labram_block_5",
#        "labram_block_8",
        "labram_block_11",
        "labram_norm",
    ],
    "NeuroLM": [
 #       "vq_emb",
        "vq_enc_patch_emb",
 #       "vq_enc_block_2",
        "vq_enc_block_5",
 #       "vq_enc_block_8",
        "vq_enc_block_11",
        "vq_encode_task",
 #       "vq_quantize",
 #       "vq_dec_freq_block_0",
        "vq_dec_freq_block_1",
 #       "vq_dec_freq_block_2",
        "vq_dec_freq_block_3",
        "gpt_0",
 #       "gpt_3",
        "gpt_6",
 #       "gpt_9",
        "gpt_11",
    ],
    "CbraMod": [
        "patch_emb",
        "layer_0",
 #       "layer_3",
        "layer_6",
 #       "layer_9",
        "layer_11",
    ],
    "EEGPT": [
        "patch_emb",
        "layer_0",
 #       "layer_2",
        "layer_4",
 #       "layer_6",
        "layer_7",
    ],
}


LAYER_SHORT = {
    "pre_transformer": "PreT",
    "transformer_0": "T0",
    "transformer_1": "T1",
    "transformer_2": "T2",
    "transformer_3": "T3",
    "final_emb": "final",
    "totem_enc_conv1": "EC1",
    "totem_enc_conv2": "EC2",
    "totem_enc_conv3": "EC3",
    "totem_enc_residual": "ERes",
    "totem_enc_z": "Ez",
    "totem_vq_quantized": "VQq",
    "totem_dec_conv1": "DC1",
    "totem_dec_residual": "DRes",
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
    "patch_emb": "patch",
    "layer_0": "L0",
    "layer_3": "L3",
    "layer_6": "L6",
    "layer_9": "L9",
    "layer_11": "L11",
    # EEGPT (shares patch_emb, layer_0, layer_6 keys with CBraMod)
    "layer_2": "L2",
    "layer_4": "L4",
    "layer_7": "L7",
 #   "encoder_out": "final",
}


# Full architectural pipeline per FM, in inference / data-flow order. Every
# possible layer position is listed exactly once so each slot has a unique
# fraction and no two layers collide. Sources:
#   BIOT      - model/biot.py: depth=4 transformer blocks (pre + 4 blocks).
#   CBraMod   - models/cbramod.py: n_layer=12 transformer blocks (+ patch_emb).
#   EEGPT     - README_TRINI.md / inference_eegpt_layers.py: depth=8.
#   TOTEM     - VQ-VAE inference: conv1→conv2→conv3→residual→z→VQ→
#               dec_conv1→dec_residual.
#   LaBraM    - modeling_pretrain.py: VQNSP depth=12 (+ patch_embed,
#               encode_task, quantize) chained into LaBraM depth=12
#               (+ patch_embed, norm).
#   NeuroLM   - train_vq.py / train_instruction.py / inference_*.py:
#               VQ encoder depth=12 (+ patch_emb, encode_task, quantize)
#               → VQ-decoder depth=4 → GPT depth=12.
# The MLP_EMBEDDING / linear-probing "Last" column is appended past the end
# of the pipeline (frac = 1L), giving it a unique slot too.
MODEL_PIPELINE: dict[str, list[str]] = {
    "BIOT": [
        "pre_transformer",
        *[f"transformer_{i}" for i in range(4)],
    ],
    "CbraMod": [
        "patch_emb",
        *[f"layer_{i}" for i in range(12)],
    ],
    "EEGPT": [
        "patch_emb",
        *[f"layer_{i}" for i in range(8)],
    ],
    "TOTEM": [
        "totem_enc_conv1", "totem_enc_conv2", "totem_enc_conv3",
        "totem_enc_residual", "totem_enc_z", "totem_vq_quantized",
        "totem_dec_conv1", "totem_dec_residual",
    ],
    "LaBram": [
        "vqnsp_enc_patch_embed",
        *[f"vqnsp_enc_block_{i}" for i in range(12)],
        "vqnsp_encode_task",
        "vqnsp_quantize",
        "labram_patch_embed",
        *[f"labram_block_{i}" for i in range(12)],
        "labram_norm",
    ],
    "NeuroLM": [
        "vq_enc_patch_emb",
        *[f"vq_enc_block_{i}" for i in range(12)],
        "vq_encode_task",
        "vq_quantize",
        *[f"vq_dec_freq_block_{i}" for i in range(4)],
        *[f"gpt_{i}" for i in range(12)],
    ],
}


def _arch_position(model: str, layer: str) -> tuple[int, int] | None:
    """(position, denominator) for ``layer`` in ``model``'s pipeline.

    The denominator is ``len(MODEL_PIPELINE[model])`` so that the trailing
    "Last" tick (linear probing on the MLP_EMBEDDING output, drawn one slot
    past the final architectural layer) lands at ``denom/denom = 1L``.
    Returns ``None`` if the layer is unknown for that model.
    """
    pipeline = MODEL_PIPELINE.get(model)
    if pipeline is None or layer not in pipeline:
        return None
    return (pipeline.index(layer), len(pipeline))


def _format_frac(frac: float) -> str:
    """Format a fraction as ``0L`` / ``0.5L`` / ``1L`` etc. (≤2 decimals)."""
    if frac <= 0:
        return "0"
    if frac >= 1:
        return "1"
    s = f"{frac:.2f}".rstrip("0").rstrip(".")
    return s 


def _layer_label(model: str, layer: str) -> str:
    """``{frac}L`` label for a layer (position in pipeline / pipeline length)."""
    if layer == plot3.LAST_LAYER_KEY:
        return "1"
    pos = _arch_position(model, layer)
    if pos is None:
        # Fallback: keep the legacy short name so unknowns are still readable.
        return LAYER_SHORT.get(layer, layer)
    num, denom = pos
    return _format_frac(num / denom)


def _load_dk_baseline() -> tuple[float | None, float | None]:
    """DK baseline AUC (mean, std) — picks the classifier with the highest mean."""
    best_m, best_s = None, None
    for clf in CLASSIFIERS_TO_TRY:
        path = (
            BASE / "MARKER_BASELINE" / "crs" / "nested_cv" / clf
            / "classification_results.json"
        )
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        auc = d.get("macro_average", {}).get("auc_score", {})
        m = auc.get("mean")
        s = auc.get("std")
        if m is not None and (best_m is None or float(m) > best_m):
            best_m = float(m)
            best_s = float(s) if s is not None else 0.0
    if best_m is None:
        print("[plot3_auc] DK baseline not found for any classifier")
    return best_m, best_s


def _load_fm_layers(model: str, pca: bool = False, pca_components: int = 27) -> list[tuple[str, float, float, str]]:
    """Return [(layer, auc_mean, auc_std, clf_name), ...] picking best clf per layer.

    Priority: repeated CV best-clf results → legacy per-clf post-hoc selection.
    """
    pca_tag = f"pca{pca_components}" if pca else "no_pca"
    out: list[tuple[str, float, float, str]] = []
    for layer in MODEL_LAYERS.get(model, []):
        # 1. Try repeated CV results (in-fold clf selection)
        repeated_path = LP_REPEATED / pca_tag / layer / f"{model}_results.json"
        if repeated_path.is_file():
            with repeated_path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            m = d.get("auc_mean")
            s = d.get("auc_std")
            if m is not None:
                out.append((layer, float(m), float(s) if s is not None else 0.0, "best_clf"))
                continue

        # 2. Legacy: post-hoc best classifier across per-clf files
        best_m, best_s, best_clf = None, 0.0, "random_forest"
        for clf in CLASSIFIERS_TO_TRY:
            path = LP_CLS / layer / f"{clf}_{model}_results.json"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            m = d.get("auc_mean")
            s = d.get("auc_std")
            if m is not None and (best_m is None or float(m) > best_m):
                best_m = float(m)
                best_s = float(s) if s is not None else 0.0
                best_clf = clf
        if best_m is not None:
            out.append((layer, best_m, best_s, best_clf))
    return out


# MLP_EMBEDDING on-disk dir (differs from LINEAR_PROBING naming for CBraMod).
_MLP_EMB_DIR = {
    "BIOT": "BIOT",
    "TOTEM": "TOTEM",
    "LaBram": "LaBram",
    "EEGPT": "EEGPT",
    "NeuroLM": "NeuroLM",
    "CbraMod": "CBraMod",
}


def _load_regression_eegpt() -> dict[str, dict[str, float]]:
    """R²-per-layer loader for EEGPT.

    ``plot3a.MODEL_LAYERS["EEGPT"]`` is an empty list (that script was never
    extended), so ``plot3.load_regression_per_layer("EEGPT")`` returns ``{}``.
    We re-implement it here using plot3b's EEGPT layer list.
    """
    reg_root = plot3.LP_ROOT / "regression"
    out: dict[str, dict[str, float]] = {}
    for layer in MODEL_LAYERS.get("EEGPT", []):
        path = reg_root / layer / "EEGPT" / "summary.json"
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
            out[layer] = r2_dict
    return out


def _load_last_layer_r2_new_results(model: str) -> dict[str, float] | None:
    """Last-layer R² loader that searches new_results.

    plot3a.load_last_layer_r2() reads from plot3a.BASE which points at
    paper_results — and paper_results has no MLP_EMBEDDING/regressor_results/
    summaries for LaBram/NeuroLM/CBraMod/TOTEM. The data exists under
    new_results, so this loader looks there explicitly.
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


def _load_fm_last_layer_paper(model: str) -> tuple[float | None, float | None, str]:
    """Last-layer CRS AUC from paper_results (nested_cv_repeated/best_clf).

    Path: paper_results/{FM}/doc_patients/MLP_EMBEDDING/crs/
          nested_cv_repeated/best_clf/classification_results.json
    Returns (mean, std, "best_clf") or (None, None, "best_clf").
    """
    on_disk = _MLP_EMB_DIR.get(model, model)
    path = (
        PAPER_RESULTS / on_disk / "doc_patients" / "MLP_EMBEDDING" / "crs"
        / "nested_cv_repeated" / "best_clf" / "classification_results.json"
    )
    if not path.is_file():
        return None, None, "best_clf"
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    auc = d.get("macro_average", {}).get("auc_score", {})
    m = auc.get("mean")
    s = auc.get("std")
    if m is None:
        # Top-level fallback for older summaries that store mean/std at root
        m = d.get("auc_mean")
        s = d.get("auc_std")
    if m is None:
        return None, None, "best_clf"
    return float(m), float(s) if s is not None else 0.0, "best_clf"


def _load_fm_last_layer(model: str) -> tuple[float | None, float | None, str]:
    """Final-layer MLP_EMBEDDING AUC (mean, std, best_clf) for the model.

    Candidate paths mirror plot1.py's MODEL_PATHS_MLP so that the 'Last' tick
    in the combined figure is always consistent with the CRS dot in plot1:
      - EEGPT: MLP-CLASSIFIER/EEGPT/doc_patients/MLP_EMBEDDING/  (same as plot1)
      - BIOT:  doubly-nested BIOT/doc_patients/MLP_EMBEDDING/BIOT/…  (same as plot1)
      - others: {FM}/doc_patients/MLP_EMBEDDING/
    """
    on_disk = _MLP_EMB_DIR.get(model, model)
    best_m, best_s, best_clf = None, 0.0, "random_forest"
    for clf in CLASSIFIERS_TO_TRY:
        candidates = []
        # EEGPT has its own dedicated output path (matches plot1's MODEL_PATHS_MLP).
        if model == "EEGPT":
            candidates.append(
                BASE / "MLP-CLASSIFIER" / "EEGPT" / "doc_patients" / "MLP_EMBEDDING"
                     / "crs" / "nested_cv" / clf / "classification_results.json"
            )
        # Standard path
        candidates.append(
            BASE / on_disk / "doc_patients" / "MLP_EMBEDDING" / "crs"
                 / "nested_cv" / clf / "classification_results.json"
        )
        # Doubly-nested fallback (BIOT)
        candidates.append(
            BASE / on_disk / "doc_patients" / "MLP_EMBEDDING" / on_disk
                 / "doc_patients" / "MLP_EMBEDDING" / "crs"
                 / "nested_cv" / clf / "classification_results.json"
        )
        for path in candidates:
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                auc = d.get("macro_average", {}).get("auc_score", {})
                m = auc.get("mean")
                s = auc.get("std")
                if m is not None and (best_m is None or float(m) > best_m):
                    best_m = float(m)
                    best_s = float(s) if s is not None else 0.0
                    best_clf = clf
                break
    if best_m is None:
        print(f"[plot3_auc] last-layer MLP_EMBEDDING not found for {model}")
    return best_m, best_s, best_clf


# ---------------------------------------------------------------------------
# (A) one wide panel: DK + one x-tick per FM, layers jittered
# ---------------------------------------------------------------------------


def plot_points(
    dk: tuple[float | None, float | None],
    fm_data: dict[str, list[tuple[str, float, float]]],
    out_path: Path,
) -> None:
    models_present = [m for m in MODEL_ORDER if fm_data.get(m)]
    xticks_labels = ["DK"] + [MODEL_DISPLAY[m] for m in models_present]
    n_groups = len(xticks_labels)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_groups + 4), 5))

    # chance line
    ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
               alpha=0.7, label="chance", zorder=1)

    # DK baseline (single point ± std)
    dk_m, dk_s = dk
    if dk_m is not None:
        ax.errorbar(
            [0], [dk_m], yerr=[dk_s or 0.0],
            fmt="o", color=DK_COLOR, capsize=5, markersize=7,
            linewidth=1.4, ecolor=DK_COLOR,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        )

    rng = np.random.default_rng(0)
    for gi, model in enumerate(models_present, start=1):
        layers = fm_data[model]
        n = len(layers)
        if n == 0:
            continue
        # Evenly spread layer points around the tick so they don't overlap.
        if n == 1:
            offsets = np.array([0.0])
        else:
            spread = min(0.35, 0.06 * (n - 1) + 0.18)
            offsets = np.linspace(-spread, spread, n)
            offsets += rng.uniform(-0.015, 0.015, size=n)
        xs = gi + offsets
        ys = np.array([m for _lyr, m, _s in layers])
        ys_err = np.array([s for _lyr, _m, s in layers])
        ax.errorbar(
            xs, ys, yerr=ys_err,
            fmt="o", color=FM_COLORS[model], capsize=4, markersize=7,
            linewidth=1.3, ecolor=FM_COLORS[model],
            markeredgecolor="white", markeredgewidth=0.5,
            alpha=0.92, zorder=3,
        )

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(xticks_labels, fontsize=11)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("AUC")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Legend: one swatch per FM + chance + DK
    handles = [
        plt.Line2D([], [], color="gray", linestyle="--", label="chance"),
        plt.Line2D([], [], marker="o", linestyle="", color=DK_COLOR,
                   markersize=7, label="DK"),
    ]
    for m in models_present:
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", color=FM_COLORS[m],
                       markersize=7, label=MODEL_DISPLAY[m])
        )
    ax.legend(handles=handles, loc="lower right",
              fontsize="medium", framealpha=0.9, ncols=2)

    ax.set_title("Random-Forest AUC along FM layers vs. DK baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3_auc] wrote {out_path}")


# ---------------------------------------------------------------------------
# (B) per-FM panels with one vertical box per layer + DK baseline at left
# ---------------------------------------------------------------------------


def _pseudo_box(ax, x: float, mean: float, std: float, color: str,
                width: float = 0.55) -> None:
    """Draw a mean/std "box" (rect) + median line + whisker caps at ``x``."""
    std = max(std, 1e-6)
    lo, hi = mean - std, mean + std
    # Box body (±1 std)
    ax.add_patch(plt.Rectangle(
        (x - width / 2, lo), width, 2 * std,
        facecolor=color, edgecolor=color, alpha=0.22, linewidth=1.0, zorder=4,
    ))
    # Median (mean) line
    ax.plot([x - width / 2, x + width / 2], [mean, mean],
            color=color, linewidth=1.6, zorder=4)
    # Whiskers (±2 std)
    w_lo, w_hi = mean - 2 * std, mean + 2 * std
    ax.plot([x, x], [lo, w_lo], color=color, linewidth=1.0, zorder=4)
    ax.plot([x, x], [hi, w_hi], color=color, linewidth=1.0, zorder=4)
    # Caps
    cap = width * 0.35
    ax.plot([x - cap, x + cap], [w_lo, w_lo], color=color, linewidth=1.0,
            zorder=4)
    ax.plot([x - cap, x + cap], [w_hi, w_hi], color=color, linewidth=1.0,
            zorder=4)


def plot_boxes(
    dk: tuple[float | None, float | None],
    fm_data: dict[str, list[tuple[str, float, float]]],
    out_path: Path,
    last_layer: dict[str, tuple[float | None, float | None]] | None = None,
) -> None:
    last_layer = last_layer or {}
    models_present = [m for m in MODEL_ORDER if fm_data.get(m)]
    n_cols = len(models_present)

    # Per-subplot width scales with layer count so crowded columns still read.
    widths = [0.3 * (len(fm_data[m]) + 2) + 2.0 for m in models_present]
    total_w = max(sum(widths), 14.0)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(total_w, 5),
        sharey=True,
        squeeze=False,
        gridspec_kw=dict(width_ratios=widths),
    )
    axes_row = axes[0]

    dk_m, dk_s = dk
    rng = np.random.default_rng(0)
    for ci, model in enumerate(models_present):
        ax = axes_row[ci]
        color = FM_COLORS[model]
        layers = fm_data[model]
        ll_m, ll_s = last_layer.get(model, (None, None))
        has_last = ll_m is not None
        labels = ["DK"] + [_layer_label(model, lyr) for lyr, _m, _s in layers]
        if has_last:
            labels.append("1")
        positions = np.arange(len(labels))

        # chance
        ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                   alpha=0.7, zorder=1)

        # DK baseline: point with std error bar at x=0
        if dk_m is not None:
            ax.errorbar(
                [positions[0]], [dk_m], yerr=[dk_s or 0.0],
                fmt="o", color=DK_COLOR, capsize=4, markersize=8,
                linewidth=1.4, ecolor=DK_COLOR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            )

        # Per-layer: point with std error bar
        if layers:
            xs = np.array([positions[j + 1] for j in range(len(layers))])
            ys = np.array([m for _lyr, m, _s in layers])
            es = np.array([s for _lyr, _m, s in layers])
            ax.errorbar(
                xs, ys, yerr=es,
                fmt="o", color=color, capsize=3, markersize=6,
                linewidth=1.1, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.5,
                alpha=0.95, zorder=3,
            )

        # Last-layer (MLP_EMBEDDING) result: star marker at the right edge
        if has_last:
            ax.errorbar(
                [positions[-1]], [ll_m], yerr=[ll_s or 0.0],
                fmt="*", color=color, capsize=4, markersize=14,
                linewidth=1.4, ecolor=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(MODEL_DISPLAY[model], fontsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ci == 0:
            ax.set_ylabel("AUC")

    fig.suptitle(
        "Random-Forest AUC per FM layer — points ± std "
        "(DK baseline at left; MLP_EMBEDDING last-layer = star marker at right)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3_auc] wrote {out_path}")


# ---------------------------------------------------------------------------
# (C) combined R² (4 family rows) + AUC (bottom row) per FM
# ---------------------------------------------------------------------------


def plot_combined(
    dk: tuple[float | None, float | None],
    fm_data: dict[str, list[tuple[str, float, float]]],
    last_layer_auc: dict[str, tuple[float | None, float | None]],
    r2_data: dict[str, dict[str, dict[str, float]]],
    r2_last: dict[str, dict[str, float]],
    out_path: Path,
    model_order: list[str] | None = None,
    exclude_marker_substrings: list[str] | None = None,
) -> None:
    """5×5 grid: 4 marker-family R² rows + 1 AUC row, one column per FM.

    Columns share x-axis (layers of that FM, with a trailing ``Last`` tick for
    MLP_EMBEDDING). Rows 1-4 share the R² scale; the AUC row is on [0, 1] and
    draws the DK baseline as a dashed horizontal reference line (it cannot be
    an x-tick because the R² rows have no DK column).

    ``model_order`` overrides the default column order (``MODEL_ORDER``) —
    used e.g. to produce the EEGPT-instead-of-TOTEM variant.
    """
    order = model_order if model_order is not None else MODEL_ORDER
    models_present = [
        m for m in order
        if fm_data.get(m) or r2_data.get(m)
    ]
    n_cols = len(models_present)
    family_rows = plot3.FAMILY_ORDER
    n_rows = len(family_rows) + 1  # +1 for AUC row at the bottom

    # Per-column layer list (R² layers ∪ AUC layers, in MODEL_LAYERS order).
    # "Last" is appended iff either modality has a last-layer value.
    col_layers: dict[str, list[str]] = {}
    col_has_last: dict[str, bool] = {}
    for model in models_present:
        r2_per_layer = r2_data.get(model, {})
        auc_layers = {lyr for lyr, _m, _s in fm_data.get(model, [])}
        # Use plot3b's MODEL_LAYERS: plot3a has "EEGPT": [] (because the
        # R²-only script wasn't extended), and we need the real layer order
        # here so EEGPT columns aren't empty in the combined figure.
        ordered = [
            lyr for lyr in MODEL_LAYERS.get(model, [])
            if lyr in r2_per_layer or lyr in auc_layers
        ]
        has_last = (r2_last.get(model) is not None) or (
            last_layer_auc.get(model, (None, None))[0] is not None
        )
        col_layers[model] = ordered
        col_has_last[model] = has_last

    widths = [max(3.0, 0.4 * len(col_layers[m]) + 2.2) for m in models_present]
    total_w = max(sum(widths), 14.0)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(total_w*0.8, 1.3 * n_rows + 1.2),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw=dict(width_ratios=widths),
    )

    dk_m, dk_s = dk

    for ci, model in enumerate(models_present):
        color = FM_COLORS[model]
        layers_all = list(col_layers[model])
        has_last = col_has_last[model]
        labels = [_layer_label(model, lyr) for lyr in layers_all]
        if has_last:
            labels.append("1")
        positions = np.arange(len(labels))

        # Prepare R²-per-layer dict with "Last" appended for this column.
        r2_per_layer = dict(r2_data.get(model, {}))
        last_r2 = r2_last.get(model)
        layers_r2 = list(layers_all)
        if has_last and last_r2:
            r2_per_layer[plot3.LAST_LAYER_KEY] = last_r2
            layers_r2.append(plot3.LAST_LAYER_KEY)
        elif has_last:
            # AUC has Last but R² doesn't — keep column width aligned, no box.
            layers_r2.append(plot3.LAST_LAYER_KEY)

        # ---- family R² rows (1-4) ------------------------------------------------
        for ri, fam in enumerate(family_rows):
            ax = axes[ri, ci]
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
            markers_in_fam = plot3.family_markers(r2_per_layer, fam)
            if fam == "evoked":
                markers_in_fam = [
                    m for m in markers_in_fam
                    if m in plot3.EVOKED_LINES_WHITELIST
                ]
            if exclude_marker_substrings:
                markers_in_fam = [
                    m for m in markers_in_fam
                    if not any(sub in m for sub in exclude_marker_substrings)
                ]
            mat = plot3.build_family_matrix(
                r2_per_layer, markers_in_fam, layers_r2,
            )

            if mat.size and not np.isnan(mat).all():
                box_positions = positions[: mat.shape[1]] if has_last else positions
                # Only draw a box where the column has at least one value.
                box_data, box_x = [], []
                for j, col in enumerate(mat.T):
                    mask = ~np.isnan(col)
                    if mask.any():
                        box_data.append(col[mask])
                        box_x.append(box_positions[j])
                if box_data:
                    ax.boxplot(
                        box_data,
                        positions=box_x,
                        widths=0.5,
                        showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color=color, linewidth=1.8),
                        whiskerprops=dict(color=color, linewidth=1.4, alpha=0.85),
                        capprops=dict(color=color, linewidth=1.4, alpha=0.85),
                        boxprops=dict(
                            facecolor=color, edgecolor=color,
                            alpha=0.22, linewidth=1.4,
                        ),
                    )

                # paired lines per marker
                for i in range(mat.shape[0]):
                    y = mat[i]
                    mask = ~np.isnan(y)
                    if mask.sum() < 2:
                        continue
                    ax.plot(
                        box_positions[mask], y[mask],
                        color=color, linewidth=0.9, alpha=0.32, zorder=2,
                    )
                # swarm
                for j, col in enumerate(mat.T):
                    mask = ~np.isnan(col)
                    if not mask.any():
                        continue
                    n = int(mask.sum())
                    jx = box_positions[j] + plot3._jitter(n, width=0.15, seed=j)
                    ax.scatter(
                        jx, col[mask],
                        s=22, color=color, alpha=0.78,
                        edgecolors="white", linewidths=0.5, zorder=3,
                    )
            else:
                ax.text(
                    0.5, 0.5, "no markers",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=12,
                )

            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.25, axis="y")
            ax.tick_params(axis="y", labelsize=10)

            if ri == 0:
                ax.set_title(plot3.MODEL_DISPLAY[model], fontsize=22)
            if ci == 0:
                ax.set_ylabel(
                 #   f"{plot3.FAMILY_DISPLAY[fam]} $R^2$",
                    f"$R^2$",
                    fontsize=24, color="black",
                )
            if ci == n_cols - 1:
                ax_r = ax.twinx()
                ax_r.set_ylim(ax.get_ylim())
                ax_r.set_yticks([])
                ax_r.tick_params(axis="y", labelsize=10)
                ax_r.set_ylabel(
                    FAMILY_FULL_DISPLAY[fam],
                    fontsize=15, color="black", labelpad=12,
                    rotation=0, ha="left", va="center",
                )
            # hide x tick labels on non-bottom rows (handled by sharex+below)
            plt.setp(ax.get_xticklabels(), visible=False)

        # ---- AUC row (last) ------------------------------------------------------
        ax_auc = axes[-1, ci]
        ax_auc.axhline(0.5, color="gray", linewidth=1.0, linestyle="--",
                       alpha=0.7, zorder=1)
        if dk_m is not None:
            ax_auc.axhline(dk_m, color=DK_COLOR, linewidth=1.2,
                           linestyle=":", alpha=0.9, zorder=1.5)

        layers = fm_data.get(model, [])
        layer_to_xy = {lyr: (m, s) for lyr, m, s in layers}
        # errorbars at per-layer positions
        xs, ys, es = [], [], []
        for j, lyr in enumerate(layers_all):
            if lyr in layer_to_xy:
                m, s = layer_to_xy[lyr]
                xs.append(positions[j])
                ys.append(m)
                es.append(s)
        if xs:
            ax_auc.errorbar(
                xs, ys, yerr=es,
                fmt="o", color=color, capsize=4, markersize=9,
                linewidth=1.6, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.7,
                alpha=0.95, zorder=3,
            )

        ll_m, ll_s = last_layer_auc.get(model, (None, None))
        if has_last and ll_m is not None:
            ax_auc.errorbar(
                [positions[-1]], [ll_m], yerr=[ll_s or 0.0],
                fmt="o", color=color, capsize=4, markersize=9,
                linewidth=1.6, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.7,
                alpha=0.95, zorder=3,
            )

        ax_auc.set_ylim(0.45, 0.9)
        ax_auc.grid(axis="y", linestyle=":", alpha=0.4)
        ax_auc.tick_params(axis="y", labelsize=10)
        ax_auc.set_xticks(positions)
        ax_auc.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
      #  ax_auc.set_xlabel("Layer", fontsize=30)
        if ci == 0:
            ax_auc.set_ylabel("AUC", fontsize=20, color="black")
        if ci == n_cols - 1:
            ax_auc_r = ax_auc.twinx()
            ax_auc_r.set_ylim(ax_auc.get_ylim())
            ax_auc_r.set_yticks([])
            ax_auc_r.tick_params(axis="y", labelsize=10)
            ax_auc_r.set_ylabel(
                "CRS\nDiagnostic",
                fontsize=15, color="black", labelpad=12,
                rotation=0, ha="left", va="center",
            )

        # Uniform xlim across all rows of this column.
        if len(positions):
            for ri in range(n_rows):
                axes[ri, ci].set_xlim(-0.6, len(positions) - 0.4)

    # Legend: DK baseline + chance (AUC row only, but place once in figure).
    dk_handle = plt.Line2D([], [], color=DK_COLOR, linestyle=":", linewidth=1.5,
                           label=f"DK baseline AUC={dk_m:.3f}"
                           if dk_m is not None else "DK baseline")
    chance_handle = plt.Line2D([], [], color="gray", linestyle="--",
                               linewidth=1.0, label="chance")
   # axes[-1, -1].legend(
   #     handles=[dk_handle, chance_handle],
   #     loc="lower right", fontsize=1, framealpha=0.9,
   # )

    # Add supxlabel BEFORE tight_layout so the layout reserves space for it
    # (otherwise rotated x-ticks of the AUC row collide with the label).
    fig.supxlabel(
        r"Relative network depth ($\ell\,/\,L$)",
        fontsize=22,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.08, bottom=0.12)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3_auc] wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _save_classifier_choices_md_3b(
    clf_choices_layers: dict[str, list[tuple[str, str, float]]],
    clf_choices_last: dict[str, tuple[str, float]],
    out_path: Path,
) -> None:
    """Save markdown table of best-classifier choices for plot3b AUC row.

    Each row also lists which classifiers have results (Found) and which are
    missing (Missing) so gaps are immediately visible.
    """
    def _layer_availability(model: str, layer: str) -> tuple[list[str], list[str]]:
        found, missing = [], []
        for clf in CLASSIFIERS_TO_TRY:
            path = LP_CLS / layer / f"{clf}_{model}_results.json"
            if path.is_file():
                import json as _json
                try:
                    d = _json.loads(path.read_text())
                    if d.get("auc_mean") is not None:
                        found.append(clf)
                        continue
                except Exception:
                    pass
            missing.append(clf)
        return found, missing

    def _last_layer_availability(model: str) -> tuple[list[str], list[str]]:
        on_disk = _MLP_EMB_DIR.get(model, model)
        found, missing = [], []
        for clf in CLASSIFIERS_TO_TRY:
            candidates = [
                BASE / on_disk / "doc_patients" / "MLP_EMBEDDING" / "crs"
                     / "nested_cv" / clf / "classification_results.json",
                BASE / on_disk / "doc_patients" / "MLP_EMBEDDING" / on_disk
                     / "doc_patients" / "MLP_EMBEDDING" / "crs"
                     / "nested_cv" / clf / "classification_results.json",
            ]
            if any(p.is_file() for p in candidates):
                found.append(clf)
            else:
                missing.append(clf)
        return found, missing

    lines = [
        "# Best Classifier per FM × Layer (Plot 3b AUC row)\n",
        "## Per-layer embeddings",
        "| FM | Layer | Best Classifier | Mean AUC | Found | Missing |",
        "|----|-------|-----------------|----------|-------|---------|",
    ]
    for model, rows in sorted(clf_choices_layers.items()):
        for layer, clf, mean in rows:
            found, missing = _layer_availability(model, layer)
            lines.append(
                f"| {model} | {layer} | {clf} | {mean:.4f} "
                f"| {', '.join(found) or '—'} | {', '.join(missing) or '—'} |"
            )

    lines += [
        "",
        "## Last-layer (MLP_EMBEDDING)",
        "| FM | Best Classifier | Mean AUC | Found | Missing |",
        "|----|-----------------|----------|-------|---------|",
    ]
    for model, (clf, mean) in sorted(clf_choices_last.items()):
        found, missing = _last_layer_availability(model)
        lines.append(
            f"| {model} | {clf} | {mean:.4f} "
            f"| {', '.join(found) or '—'} | {', '.join(missing) or '—'} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[plot3_auc] wrote {out_path}")


def _save_r2_latex_tables(
    r2_data: dict[str, dict[str, dict[str, float]]],
    r2_last: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    """One LaTeX table per FM: rows = markers, columns = layers ({frac}L + Last).

    Files: ``out_dir/r2_table_{FM}.tex``. Layer columns use the same
    architectural-fraction labels as the figures, so the table and the plot
    line up 1:1.
    """
    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    out_dir.mkdir(parents=True, exist_ok=True)
    for model, layers_dict in r2_data.items():
        # Architectural order, restricted to layers we actually have R² for.
        layers_arch = [
            lyr for lyr in MODEL_LAYERS.get(model, []) if lyr in layers_dict
        ]
        last = r2_last.get(model)
        layers_full = list(layers_arch) + ([plot3.LAST_LAYER_KEY] if last else [])
        if not layers_full:
            continue

        markers: set[str] = set()
        for lyr in layers_arch:
            markers.update(layers_dict.get(lyr, {}).keys())
        if last:
            markers.update(last.keys())
        marker_rows = sorted(markers)

        def _header_for(lyr: str) -> str:
            frac = _layer_label(model, lyr)
            name = "Last" if lyr == plot3.LAST_LAYER_KEY else lyr
            return f"{_esc(name)} ({frac})"

        headers = [_header_for(lyr) for lyr in layers_full]
        col_spec = "l" + "r" * len(layers_full)

        lines = [
            "% Auto-generated by plot3b.py — R^2 per marker x layer for "
            f"{MODEL_DISPLAY[model]}.",
            "\\begin{tabular}{%s}" % col_spec,
            "\\toprule",
            "Marker & " + " & ".join(headers) + r" \\",
            "\\midrule",
        ]
        for marker in marker_rows:
            row_cells = []
            for lyr in layers_full:
                if lyr == plot3.LAST_LAYER_KEY:
                    val = last.get(marker) if last else None
                else:
                    val = layers_dict.get(lyr, {}).get(marker)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row_cells.append("---")
                else:
                    row_cells.append(f"{val:.3f}")
            lines.append(_esc(marker) + " & " + " & ".join(row_cells) + r" \\")
        lines += ["\\bottomrule", "\\end{tabular}"]

        out_path = out_dir / f"r2_table_{model}.tex"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[plot3_auc] wrote {out_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dk = _load_dk_baseline()
    print(f"[plot3_auc] DK baseline: mean={dk[0]}, std={dk[1]}")
    fm_data: dict[str, list[tuple[str, float, float]]] = {}
    last_layer: dict[str, tuple[float | None, float | None]] = {}
    clf_choices_layers: dict[str, list[tuple[str, str, float]]] = {}
    clf_choices_last: dict[str, tuple[str, float]] = {}
    # Union MODEL_ORDER with the EEGPT variant so we load EEGPT data too.
    all_models = list(dict.fromkeys(MODEL_ORDER + MODEL_ORDER_EEGPT))
    for model in all_models:
        rows_with_clf = _load_fm_layers(model)
        if rows_with_clf:
            # strip clf_name for plot functions, collect for .md
            fm_data[model] = [(lyr, m, s) for lyr, m, s, _clf in rows_with_clf]
            clf_choices_layers[model] = [(lyr, clf, m) for lyr, m, _s, clf in rows_with_clf]
            print(f"[plot3_auc]   {model}: {len(rows_with_clf)} layers resolved")
        else:
            print(f"[plot3_auc]   {model}: no layers found — skipping")
        # Last layer: prefer the canonical paper_results CRS result so every
        # FM has a consistent value. Fall back to the legacy per-clf scan if
        # paper_results doesn't have one (e.g. TOTEM).
        ll_m, ll_s, ll_clf = _load_fm_last_layer_paper(model)
        if ll_m is None:
            ll_m, ll_s, ll_clf = _load_fm_last_layer(model)
        if ll_m is not None:
            last_layer[model] = (ll_m, ll_s)
            clf_choices_last[model] = (ll_clf, ll_m)
            print(f"[plot3_auc]   {model}: last-layer AUC={ll_m:.3f} ± {ll_s:.3f} [{ll_clf}]")

    _save_classifier_choices_md_3b(
        clf_choices_layers, clf_choices_last,
        OUT / "classifier_choices_plot3b.md",
    )

    plot_points(dk, fm_data, OUT / "plot3_auc_layers_points.png")
    plot_boxes(dk, fm_data, OUT / "plot3_auc_layers_boxes.png",
               last_layer=last_layer)

    # Combined figure: import R² data per FM from plot3.
    r2_data: dict[str, dict[str, dict[str, float]]] = {}
    r2_last: dict[str, dict[str, float]] = {}
    for model in all_models:
        r2 = plot3.load_regression_per_layer(model)
        # plot3.MODEL_LAYERS has "EEGPT": [] — fall back to our own layer list
        # so EEGPT R² data is loaded when plot3a's list is empty.
        if not r2 and model == "EEGPT":
            r2 = _load_regression_eegpt()
        if r2:
            r2_data[model] = r2
            print(f"[plot3_auc]   {model}: R² layers={len(r2)}")
        last_r2 = plot3.load_last_layer_r2(model)
        # plot3a.BASE points at paper_results, which has no MLP_EMBEDDING
        # regressor_results/ for LaBram/NeuroLM/CBraMod/TOTEM. Fall back to
        # new_results so the "Last" R² box is drawn for those columns too.
        if not last_r2:
            last_r2 = _load_last_layer_r2_new_results(model)
        if last_r2:
            r2_last[model] = last_r2
            print(f"[plot3_auc]   {model}: R² last-layer markers={len(last_r2)}")

    # BIOT: final_emb is the same as the MLP_EMBEDDING last layer — surface
    # it under the "Last" tick instead of as a separate column.
    biot_final_emb = (
        BASE / "LINEAR_PROBING" / "regression" / "final_emb" / "BIOT" / "summary.json"
    )
    if biot_final_emb.is_file():
        with biot_final_emb.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        r2_last["BIOT"] = {
            m: float(v["r2"])
            for m, v in metrics.items()
            if isinstance(v, dict) and not v.get("skipped", False) and "r2" in v
        }
        # also drop final_emb from the per-layer dict if it sneaked in
        r2_data.get("BIOT", {}).pop("final_emb", None)
        print(
            f"[plot3_auc]   BIOT: loaded final_emb R² → 'Last' "
            f"({len(r2_last['BIOT'])} markers)"
        )

    # EEGPT: encoder_out is the last layer — promote it to the "Last" tick
    # (same logic as BIOT/final_emb above; no separate MLP_EMBEDDING run exists).
    eegpt_encoder_out = (
        BASE / "LINEAR_PROBING" / "regression" / "encoder_out" / "EEGPT" / "summary.json"
    )
    if eegpt_encoder_out.is_file() and "EEGPT" not in r2_last:
        with eegpt_encoder_out.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        r2_last["EEGPT"] = {
            m: float(v["r2"])
            for m, v in metrics.items()
            if isinstance(v, dict) and not v.get("skipped", False) and "r2" in v
        }
        r2_data.get("EEGPT", {}).pop("encoder_out", None)
        print(
            f"[plot3_auc]   EEGPT: loaded encoder_out R² → 'Last' "
            f"({len(r2_last['EEGPT'])} markers)"
        )

    plot_combined(
        dk, fm_data, last_layer, r2_data, r2_last,
        OUT / "plot3_combined.png",
    )

    # EEGPT variant: same combined figure, but TOTEM is dropped and EEGPT
    # takes its slot. Column order: BIOT, LaBraM, EEGPT, NeuroLM, CBraMod
    # (DK stays as the AUC-row reference line).
    plot_combined(
        dk, fm_data, last_layer, r2_data, r2_last,
        OUT / "plot3_combined_eegpt.png",
        model_order=MODEL_ORDER_EEGPT,
    )

    # Same EEGPT variant, but Evoked-row R² points coming from
    # TimeLockedContrast/* and WindowDecoding/* markers are dropped.
    plot_combined(
        dk, fm_data, last_layer, r2_data, r2_last,
        OUT / "plot3_combined_eegpt_noContrast.png",
        model_order=MODEL_ORDER_EEGPT,
        exclude_marker_substrings=["TimeLockedContrast", "WindowDecoding"],
    )

    # Per-FM LaTeX R² tables (rows = markers, columns = layer fractions + Last).
    _save_r2_latex_tables(r2_data, r2_last, PAPER_RESULTS / "r2_tables")


if __name__ == "__main__":
    main()
