"""Canonical paper figures for the NeurIPS submission.

Each module corresponds to one figure (or set of related panels) in the paper:

- ``fig1_utility``           — Figure 1, Section 3.1 "Utility"
- ``fig2_layerwise``         — Figure 2, Section 3.2 "Representation"
- ``fig3_residualisation``   — Figure 3, Section 3.2 "Representation"
- ``fig_mknn_ksweep``        — Appendix A.2 MKNN k-sweep ablation

The four scripts are CLI wrappers around the canonical plotting code under
``src/paper_plots_legacy/``; they fix a clean argparse interface
(``--results-root`` / ``--output-dir``) and print the exact PNG filename that
appears in the paper.

The shared helper ``_corrected_ttest`` implements the Nadeau--Bengio corrected
paired t-test used throughout the statistical comparisons.
"""
