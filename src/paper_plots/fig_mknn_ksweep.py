#!/usr/bin/env python3
"""Appendix A.2 — MKNN(DK, FM) vs k for all foundation models.

Reproduces ``plot_mknn_k_sweep.png``: observed MKNN(k) curves with 95%
bootstrap CIs (solid + markers) and permutation-null curves with 95% CIs
(dashed) for k ∈ [10, 40].

Inputs (under ``--results-root``):
  - ``EMBEDDING_COMPARISON/component5_mknn_ksweep/mknn_results_{FM}.json``
  - ``EMBEDDING_COMPARISON/component5_mknn_ksweep/mknn_baseline_{FM}.json``

Output:
  - ``<output-dir>/plot_mknn_k_sweep.png``  (paper Appendix A.2)

Usage:
    python -m src.paper_plots.fig_mknn_ksweep \\
        --results-root data/benchmark_results/paper_results \\
        --output-dir   data/benchmark_results/paper_results/PLOTS
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

PAPER_FILENAME = "plot_mknn_k_sweep.png"
LEGACY_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "paper_plots_legacy"
    / "plot_mknn_k_sweep.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root of the paper-results tree (must contain "
        "EMBEDDING_COMPARISON/component5_mknn_ksweep/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the PNG. Defaults to <results-root>/PLOTS.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "PLOTS").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LEGACY_SCRIPT.is_file():
        print(
            f"[fig_mknn_ksweep] missing legacy script: {LEGACY_SCRIPT}",
            file=sys.stderr,
        )
        return 1

    os.environ["EEG_RESULTS_ROOT"] = str(results_root)
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")

    paper_png = output_dir / PAPER_FILENAME
    print(f"[fig_mknn_ksweep] paper Appendix A.2 -> {paper_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
