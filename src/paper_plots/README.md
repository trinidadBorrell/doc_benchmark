# `src/paper_plots/` — Canonical figures for the NeurIPS submission

One module per paper figure. Every script accepts a `--results-root` (the
top-level directory containing `MARKER_BASELINE/`, `MLP_EMBEDDING/`,
`EMBEDDING_DK_COMBINED/`, `RES_NO_LEAKAGE/`, `LINEAR_PROBING/`,
`EMBEDDING_COMPARISON/`) and an optional `--output-dir` (defaults to
`<results-root>/PLOTS`).

| Paper figure | Script | Output PNG |
|---|---|---|
| **Fig 1** — Utility AUC, 6 tasks (§3.1) | [`fig1_utility.py`](fig1_utility.py) | `plot1_per_target_dot_ttest_fdr_directional.png` |
| **Fig 2** — Layer-wise probing + CRS AUC (§3.2) | [`fig2_layerwise.py`](fig2_layerwise.py) | `plot3_combined_eegpt.png` |
| **Fig 3** — FM / FM+DK / FM-res across 6 tasks (§3.2) | [`fig3_residualisation.py`](fig3_residualisation.py) | `plot4_noleak_combined_nonpca_annotated_ttest_fdr.png` |
| **Appendix A.2** — MKNN k-sweep | [`fig_mknn_ksweep.py`](fig_mknn_ksweep.py) | `plot_mknn_k_sweep.png` |

The shared helper [`_corrected_ttest.py`](_corrected_ttest.py) implements the
Nadeau–Bengio (2003) variance-corrected paired t-test used in the FDR
comparisons (paper §3.2 "Corrected t-tests with Benjamini-Hochberg FDR
correction").

## Reproducing all four figures

```bash
RESULTS_ROOT=data/benchmark_results/paper_results
OUT=$RESULTS_ROOT/PLOTS

python -m src.paper_plots.fig1_utility           --results-root $RESULTS_ROOT --output-dir $OUT
python -m src.paper_plots.fig2_layerwise         --results-root $RESULTS_ROOT --output-dir $OUT
python -m src.paper_plots.fig3_residualisation   --results-root $RESULTS_ROOT --output-dir $OUT
python -m src.paper_plots.fig_mknn_ksweep        --results-root $RESULTS_ROOT --output-dir $OUT
```

Each script is a thin CLI wrapper around the canonical implementation under
[`../paper_plots_legacy/`](../paper_plots_legacy/) (preserved verbatim from the
research workflow). Wrappers set the `EEG_RESULTS_ROOT` environment variable
and invoke the legacy module via `runpy`; this keeps the statistics and
plotting code unchanged while exposing a clean argparse interface.

If you want the full grid of test / FDR / PCA ablation variants that did not
make it into the paper, run the legacy scripts directly:

```bash
EEG_RESULTS_ROOT=$RESULTS_ROOT python src/paper_plots_legacy/statistics_plot1.py
EEG_RESULTS_ROOT=$RESULTS_ROOT python src/paper_plots_legacy/statistics_plot4_targets_no_leakage.py
```
