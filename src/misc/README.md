# `src/misc/` — Modules outside the NeurIPS paper scope

Code that lives in this folder was developed during the research project but
**is not used by any of the seven evaluation steps or four figures of the
NeurIPS submission**. It is preserved here for transparency and for users who
want to reuse the broader DoC pipeline beyond the paper.

| Folder | Purpose | Why outside the paper |
|---|---|---|
| `general_metrics/` | MAPE, Pearson correlation, FFT analysis between original and reconstructed EEG | The paper does not report reconstruction-fidelity metrics. |
| `decoder/` | Temporal decoding (sliding-estimator + LogisticRegression) with statistical testing | Not in the paper's evaluation framework. |
| `global_analysis/` | OHBM-style topoplots, group-level qualitative analysis, biomarker comparisons | Used in earlier conference posters; not in the NeurIPS paper. |
| `markers_report/` | HTML report generator for per-subject DK marker visualisation | Diagnostic tool only; the paper aggregates DK markers numerically. |
| `build_tables/` | CRS shift-table builders | Supplementary diagnostic tables, not in the paper. |
| `interp/causal_tracing.py` | Causal-tracing experiments on FM embeddings | Exploratory; not in the paper. |
| `interp/embedding_steering.py` | Embedding-steering / activation editing | Exploratory; not in the paper. |
| `interp/non_linear_probing.py` | Non-linear (MLP/RF) probes against DK markers | The paper restricts itself to **linear** probes by design (§3.2). |
| `interp/residualization_dim.py` | Per-dimension residualisation variant | The paper uses fold-internal whole-vector residualisation (`src/interp/res_no_leakage/`). |
| `interp/residualization_embeddings.py` | Earlier residualisation variant fitting on the union of train+test | Superseded by `res_no_leakage` (paper §3.2 cites Snoek et al. 2019 / Rosenblatt et al. 2024 on data leakage). |
| `interp/launch_pool_jobs.py` | HTCondor batch launcher for pool-based interp jobs | Workflow helper; not in the paper. |
| `interp/run_*.sh`, `how_to_run.txt` | Convenience shell wrappers | Workflow helpers. |

## Reproducing the paper

If you only want to reproduce the paper's results, **ignore this folder**.
Use [`src/paper_plots/`](../paper_plots/) for the four figures and
`cookbooks/pipeline.py --paper-eval` for the seven evaluation steps.

## Running the legacy pipeline phases

The original `cookbooks/pipeline.py` exposed five phases (GENERAL_METRICS,
MLP_EMBEDDING, DECODER, MARKERS, MODEL). The MLP_EMBEDDING and MARKERS phases
are paper-relevant and remain at their original locations. The remaining three
(GENERAL_METRICS, DECODER, plus the optional HTML report under MARKERS) reach
into the modules in this folder and continue to function via the
`--general-metrics-only`, `--decoder-only`, and `--markers-only` flags. See
the top-level [`README.md`](../../README.md) for the full command surface.
