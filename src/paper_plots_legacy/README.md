# `paper_plots_legacy/`

> **Not part of the NeurIPS submission.** Use `src/paper_plots/` for the four
> figures that appear in the paper. This folder is preserved for transparency
> only — it contains exploratory variants, statistical-test ablations, and PCA
> versions of the analyses that informed the final figures but are not cited in
> the manuscript.

## Files

| File | Status | Notes |
|---|---|---|
| `plot1.py` | superseded | Earlier base of Fig 1 (per-target dot/box plots, no FDR annotations). The canonical version lives at `src/paper_plots/fig1_utility.py`. |
| `statistics_plot1.py` | superseded | Variant grid (t-test / Wilcoxon / corrected, with/without FDR, with/without PCA). Canonical paper variant is t-test + FDR (directional). |
| `plot3a.py` | superseded | R²-only (4 marker rows) layer-wise probe. Merged with `plot3b.py` into `src/paper_plots/fig2_layerwise.py`. |
| `plot3b.py` | superseded | Combines `plot3a.py` rows with the row-5 CRS classification AUC; uses fragile `sys.path.insert` to import `plot3a`. |
| `plot3c.py` | exploratory | MKNN bar plot (alternative rendering of Table 1). |
| `plot4.py` | superseded | Single-panel CRS overview of `FM` / `FM+DK` / `FM-residualised`. |
| `plot4_targets.py` | superseded | Extended to 8 targets but uses leakage-prone residualisation. |
| `plot4_targets_no_leakage.py` | superseded | Fold-internal residualisation; canonical version at `src/paper_plots/fig3_residualisation.py`. |
| `statistics_plot4_targets.py` | superseded | Statistical variants on the leakage-prone residualisation. |
| `statistics_plot4_targets_no_leakage.py` | superseded | Statistical variants on the no-leakage residualisation. Paper uses t-test + FDR (`plot4_noleak_combined_nonpca_annotated_ttest_fdr.png`). |
| `plot_mknn_k_sweep.py` | retained-and-mirrored | Appendix MKNN k-sweep plot. The canonical copy is `src/paper_plots/fig_mknn_ksweep.py`. |
| `_corrected_ttest.py` | retained-and-mirrored | Nadeau-Bengio corrected paired t-test helper. The canonical copy lives in `src/paper_plots/_corrected_ttest.py`. |

## Why keep this folder

These scripts produce the variant PNGs in
`data/benchmark_results/paper_results/PLOTS/` (e.g. `*_wilcoxon*.png`,
`*_corrected_*.png`, `*_pca*.png`) that did not make it into the paper. They are
useful for ablation reproducibility (showing that the paper's t-test + FDR
result is robust to test choice) but are **not required** to reproduce any
figure or table in the manuscript.

If you only want to reproduce the paper, ignore this folder and use
[`src/paper_plots/`](../paper_plots/).
