"""Plot MKNN(DK, FM) vs k for all foundation models, with permutation nulls.

Reads ``EMBEDDING_COMPARISON/component5_mknn_ksweep/{mknn_results,mknn_baseline}_{FM}.json``
and writes ``PLOTS/plot_mknn_k_sweep.png``.

Each FM gets two curves in its plot1 colour:
  • solid + markers — observed MKNN(k) with 95 % bootstrap CI band
  • dashed         — permutation null π(DK)(k) with 95 % CI band

Run locally (no GPU needed):
    source /data/project/eeg_foundation/.venv/bin/activate
    python src/doc_benchmark/src/paper_plots/plot_mknn_k_sweep.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BASE = Path(
    os.environ.get(
        "EEG_RESULTS_ROOT",
        "/data/project/eeg_foundation/data/benchmark_results/paper_results",
    )
)
MKNN_DIR = BASE / "EMBEDDING_COMPARISON" / "component5_mknn_ksweep"
OUT_PNG = BASE / "PLOTS" / "plot_mknn_k_sweep.png"

K_MIN = 10
K_MAX = 40

# (json-stem, display name) — JSON files are written under the FM dir name
# ("LaBram"), but the colour palette and legend use the canonical "LaBraM".
FMS: list[tuple[str, str]] = [
    ("BIOT", "BIOT"),
    ("LaBram", "LaBraM"),
    ("EEGPT", "EEGPT"),
    ("NeuroLM", "NeuroLM"),
    ("CBraMod", "CBraMod"),
]

# Match statistics_plot1.PER_TARGET_PALETTE.
COLOURS: dict[str, str] = {
    "BIOT":    "#1f77b4",
    "LaBraM":  "#2ca02c",
    "EEGPT":   "#8c564b",
    "NeuroLM": "#9467bd",
    "CBraMod": "#ff7f0e",
}


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (k, mean, ci_lo, ci_hi) sorted by k, restricted to [K_MIN, K_MAX]."""
    with open(path) as f:
        data = json.load(f)
    rows = sorted(data["k_results"].values(), key=lambda r: int(r["k"]))
    rows = [r for r in rows if K_MIN <= int(r["k"]) <= K_MAX]
    k = np.array([int(r["k"]) for r in rows])
    mean = np.array([float(r["mean"]) for r in rows])
    lo = np.array([float(r["ci_lower"]) for r in rows])
    hi = np.array([float(r["ci_upper"]) for r in rows])
    return k, mean, lo, hi


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    all_k: set[int] = set()

    for stem, display in FMS:
        results_path = MKNN_DIR / f"mknn_results_{stem}.json"
        baseline_path = MKNN_DIR / f"mknn_baseline_{stem}.json"
        if not results_path.is_file() or not baseline_path.is_file():
            print(f"[plot_mknn_k_sweep] missing for {stem} — skipping")
            continue

        colour = COLOURS[display]
        k_o, m_o, lo_o, hi_o = _load_curve(results_path)
        k_n, m_n, lo_n, hi_n = _load_curve(baseline_path)
        all_k.update(int(x) for x in k_o)

        ax.plot(k_o, m_o, color=colour, marker="o", markersize=4,
                linewidth=1.6, label=display)

        ax.plot(k_n, m_n, color=colour, linestyle="--", linewidth=1.2,
                alpha=0.85)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"MKNN(DK, FM)")
    if all_k:
        ax.set_xticks(sorted(all_k))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    fm_handles, fm_labels = ax.get_legend_handles_labels()
    null_proxy = Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                        label=r"permutation null $\pi(\mathrm{DK})$")
    ax.legend(
        handles=fm_handles + [null_proxy],
        labels=fm_labels + [null_proxy.get_label()],
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_mknn_k_sweep] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
