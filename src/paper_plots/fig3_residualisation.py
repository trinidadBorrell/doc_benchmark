#!/usr/bin/env python3
"""Figure 3 — DK / FM / FM+DK / FM-residualised AUC (paper Section 3.2).

Reproduces ``plot4_noleak_combined_nonpca_annotated_ttest_fdr.png``: per-task
nested-CV ROC AUC comparing four conditions (DK alone, FM alone, FM ⊕ DK,
FM-residualised under fold-internal residualisation). Pairwise differences
(FM+DK vs FM, FM-residualised vs FM) are tested with a corrected paired
t-test and Benjamini-Hochberg FDR adjustment.

Inputs (under ``--results-root``):
  - ``MARKER_BASELINE/``                       — DK alone
  - ``{FM}/doc_patients/MLP_EMBEDDING/``       — FM alone
  - ``{FM}/doc_patients/EMBEDDING_DK_COMBINED/`` — FM ⊕ DK
  - ``{FM}/doc_patients/RES_NO_LEAKAGE/``      — FM residualised (no-leakage variant)

Output:
  - ``<output-dir>/plot4_noleak_combined_nonpca_annotated_ttest_fdr.png``  (paper Fig 3)
  - plus statistical-test ablation variants

Usage:
    python -m src.paper_plots.fig3_residualisation \\
        --results-root data/benchmark_results/paper_results \\
        --output-dir   data/benchmark_results/paper_results/PLOTS
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

PAPER_FILENAME = "plot4_noleak_combined_nonpca_annotated_ttest_fdr.png"
LEGACY_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "paper_plots_legacy"
    / "statistics_plot4_targets_no_leakage.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root of the paper-results tree (must contain MARKER_BASELINE/, "
        "MLP_EMBEDDING/, EMBEDDING_DK_COMBINED/, RES_NO_LEAKAGE/ subtrees).",
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
        print(
            f"[fig3_residualisation] missing legacy script: {LEGACY_SCRIPT}",
            file=sys.stderr,
        )
        return 1

    os.environ["EEG_RESULTS_ROOT"] = str(results_root)
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")

    paper_png = output_dir / PAPER_FILENAME
    print(f"[fig3_residualisation] paper Fig 3 -> {paper_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
