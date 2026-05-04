#!/usr/bin/env python3
"""Figure 2 — Layer-wise linear probing + CRS classification AUC (paper Section 3.2).

Reproduces ``plot3_combined_eegpt.png``: a 5-row × 5-column grid where rows 1-4
show ridge-regression R² for the four DK marker families (Evoked, Connectivity,
Information Theory, Spectral) at every probed FM layer, and row 5 shows the
nested-CV ROC AUC of CRS Diagnostic classifiers trained at each FM layer's
embedding.

Inputs (under ``--results-root``):
  - ``LINEAR_PROBING/regression/{layer}/{FM}/summary.json`` (per-marker R²)
  - ``LINEAR_PROBING/classification/{layer}/{FM}/...``     (per-layer AUC)

Output:
  - ``<output-dir>/plot3_combined_eegpt.png``  (paper Fig 2)
  - plus auxiliary panels (per-FM, AUC-only, R²-only)

Usage:
    python -m src.paper_plots.fig2_layerwise \\
        --results-root data/benchmark_results/paper_results \\
        --output-dir   data/benchmark_results/paper_results/PLOTS
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

PAPER_FILENAME = "plot3_combined_eegpt.png"
LEGACY_SCRIPT = (
    Path(__file__).resolve().parent.parent / "paper_plots_legacy" / "plot3b.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root of the paper-results tree (must contain LINEAR_PROBING/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the PNGs. Defaults to <results-root>/PLOTS.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "PLOTS").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LEGACY_SCRIPT.is_file():
        print(f"[fig2_layerwise] missing legacy script: {LEGACY_SCRIPT}", file=sys.stderr)
        return 1

    os.environ["EEG_RESULTS_ROOT"] = str(results_root)
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")

    paper_png = output_dir / PAPER_FILENAME
    print(f"[fig2_layerwise] paper Fig 2 -> {paper_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
