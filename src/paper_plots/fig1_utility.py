#!/usr/bin/env python3
"""Figure 1 — Utility AUC across six clinical tasks (paper Section 3.1).

Reproduces ``plot1_per_target_dot_ttest_fdr_directional.png``: nested-CV ROC AUC
for the five EEG foundation models and the domain-knowledge (DK) baseline on
the six clinical tasks (CRS Diagnostic, Etiology UWS/MCS, Prognosis 6m/1y/2y),
with FDR-corrected one-tailed t-tests against DK annotated as directional
significance markers.

Inputs (under ``--results-root``):
  - ``MARKER_BASELINE/``                      — DK baseline nested-CV scores
  - ``{FM}/doc_patients/MLP_EMBEDDING/``      — per-FM nested-CV scores
    where ``{FM}`` ∈ {BIOT, LaBram, EEGPT, NeuroLM, CBraMod}

Output:
  - ``<output-dir>/plot1_per_target_dot_ttest_fdr_directional.png``  (paper Fig 1)
  - plus the full grid of (test × FDR × encoding) ablation variants

Usage:
    python -m src.paper_plots.fig1_utility \\
        --results-root data/benchmark_results/paper_results \\
        --output-dir   data/benchmark_results/paper_results/PLOTS
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

PAPER_FILENAME = "plot1_per_target_dot_ttest_fdr_directional.png"
LEGACY_SCRIPT = (
    Path(__file__).resolve().parent.parent / "paper_plots_legacy" / "statistics_plot1.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root of the paper-results tree (must contain MARKER_BASELINE/ "
        "and {FM}/doc_patients/MLP_EMBEDDING/ subtrees).",
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
        print(f"[fig1_utility] missing legacy script: {LEGACY_SCRIPT}", file=sys.stderr)
        return 1

    os.environ["EEG_RESULTS_ROOT"] = str(results_root)
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")

    paper_png = output_dir / PAPER_FILENAME
    print(f"[fig1_utility] paper Fig 1 -> {paper_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
