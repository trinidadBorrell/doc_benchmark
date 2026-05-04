# Can EEG Foundation Models Go Beyond Domain Knowledge in Reading Consciousness?

Reference implementation for the NeurIPS 2025 submission of the same title.
This repository evaluates five EEG foundation models (BIOT, LaBraM, EEGPT,
NeuroLM, CBraMod) against a validated domain-knowledge (DK) baseline across
six clinical tasks, on 300 recordings from 249 patients with disorders of
consciousness (DoC). Two complementary axes are measured:

1. **Utility** — zero-shot clinical classification performance vs.\ the DK.
2. **Representational audit** — layer-wise linear probing, fold-internal
   residualisation, and Mutual k-Nearest Neighbour (MKNN) alignment with
   the DK space.


---

## Repository layout

```
doc_benchmark/
├── cookbooks/
│   ├── pipeline.py              ← top-level orchestrator (paper + legacy phases)
│   ├── intermodel_results.py    ← cross-FM result aggregation
│   └── run_*.sh                 ← convenience shell wrappers
├── src/
│   ├── paper_plots/             ← canonical NeurIPS figures (Fig 1 / 2 / 3 / Appendix)
│   ├── paper_plots_legacy/      ← variant / ablation plot scripts (not in paper)
│   ├── model/                   ← dk_marker_baseline, fm_embedding_classifier,
│   │                              fm_plus_dk_classifier, embedding_comparison (MKNN)
│   ├── interp/                  ← linear_probing, plot_layerwise_{r2,crs_auc},
│   │                              res_no_leakage/fold_internal_residualisation
│   └── misc/                    ← exploratory modules NOT used by the paper
├── tests/                       ← pytest suite
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md  (this file)
```

The four files in `src/paper_plots/` produce the paper's three figures plus
the appendix MKNN k-sweep; everything else under `src/misc/` is kept for
transparency and is **not required** to reproduce any paper result.

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline targets Python 3.11 and was developed against the dependency
versions pinned in `requirements.txt` (MNE 1.5–1.10, scikit-learn ≥ 1.2,
NumPy 1.26.x, antropy, h5py, joblib, junifer).

---

## Foundation models

| Model | Pretraining (hrs) | Params | License | Repository |
|---|---:|---:|---|---|
| BIOT | ~5,000 | 3.3M | MIT | https://github.com/ycq091044/BIOT |
| LaBraM-B | ~2,534 | 5.8M | MIT | https://github.com/935963004/LaBraM |
| EEGPT | ~1,200 | 10M | Apache 2.0 | https://github.com/BINE022/EEGPT |
| NeuroLM-B | ~25,000 | 254M | MIT | https://github.com/935963004/NeuroLM |
| CBraMod | ~9,000 | 4M | MIT | https://github.com/wjq-learning/CBraMod |

Each FM is run **frozen, zero-shot** on the DoC cohort: per-FM input
preprocessing follows the upstream specifications (sampling rate, channel
subset, window length, patch size). For BIOT and EEGPT, the EGI-256 montage
is projected onto the model's target layout (10–20 / 10–10 respectively).
LaBraM, NeuroLM, and CBraMod operate on the full 256-channel array.
Activation hooks placed at architecturally meaningful intermediate layers
yield one mean-pooled fixed-dimensional embedding per recording per layer.

---

## Data

The clinical dataset (300 recordings × 249 DoC patients, EGI 256-channel,
local-global auditory oddball paradigm) **is not redistributed** for privacy
reasons; access is restricted to the originating institution. The pipeline is
fully runnable on any equivalent DoC EEG cohort that satisfies these
requirements:

- a `patient_labels.csv` with at minimum `patient_id` and one of
  `diagnostic_crs_final` (UWS/MCS), `etiology`, `cs_6m`, `cs_1y`, `cs_2y`;
- a per-recording scalar marker CSV produced by extracting the DK markers
  from §3.2 with the upstream
  [`nice`](https://github.com/nice-tools/nice) library;
- per-FM embedding files (one `.npy` / `.npz` per recording per layer)
  produced by each model's upstream extraction code.

Dataset summary (paper §3.1): 144 UWS / 156 MCS recordings, 40.7% anoxic /
25.7% TBI, 67.3% acute / 32.7% chronic. CRS-R diagnostic label assigned from
the best assessment across at least three sessions.

Pre-processing follows a configuration-driven MNE-Python pipeline: 0.5–45 Hz
bandpass, 50 Hz notch, automatic artifact rejection, bad-channel
interpolation, common-average re-referencing, stimulus-locked epoching.

---

## End-to-end reproduction

A single command runs all seven evaluation steps of the paper, in dependency
order, against an existing tree of pre-extracted FM embeddings and DK
markers:

```bash
python cookbooks/pipeline.py \
    --paper-eval --all \
    --main-path     /path/to/doc_eeg_data \
    --metadata-dir  /path/to/metadata \
    --mode patient --task lg \
    --results-subdir CBraMod/doc_patients \
    --embedding-data-dir   /path/to/embeddings \
    --emb-marker-csv       /path/to/baseline_stable_20210128_scalars.csv \
    --emb-patient-labels-full /path/to/patient_labels_with_controls.csv
```

`--paper-eval` is equivalent to running, in this order:

1. **FM_EMBEDDING** (paper step A — Utility nested CV across 6 tasks)
2. **PROBING** (paper steps B & C — layer-wise R² + per-layer CRS AUC)
3. **FM_PLUS_DK** (paper step D — FM ⊕ DK concatenation)
4. **RESIDUALISE** (paper step E — fold-internal residualisation)
5. **MKNN** (paper steps F & G — MKNN(k=20) + Appendix k-sweep)

---

## Per-step reproduction

Each step is independently re-runnable via a dedicated `--*-only` flag. CV
folds are patient-grouped and shared across all FMs (paper §3.2).

| Step | Paper § | Phase flag | Entry point | Outputs (under `<results-root>/`) | Compute (paper Appendix A) |
|---|---|---|---|---|---|
| **A** Utility nested CV (5×20, 6 tasks, 5 classifiers, FDR t-test vs DK) | §3.1 (Fig 1) | `--fm-embedding-only` | `src/model/fm_embedding_classifier.py` (called by `pipeline.py`) | `MLP_EMBEDDING/{target}/nested_cv/...` and `MARKER_BASELINE/...` | ~540 CPU-h / 30 jobs |
| **B** Layer-wise linear probing R² | §3.2 (Fig 2 rows 1–4) | `--probing-only` | `src/interp/linear_probing.py` | `LINEAR_PROBING/regression/{layer}/{FM}/summary.json` | ~580 CPU-h / 5 jobs |
| **C** Layer-wise CRS AUC | §3.2 (Fig 2 row 5) | `--probing-only` (same call) | `src/interp/linear_probing.py` | `LINEAR_PROBING/classification/{layer}/{FM}/...` | shared with **B** |
| **D** FM ⊕ DK concatenation | §3.2 (Fig 3) | `--fm-plus-dk-only` | `src/model/fm_plus_dk_classifier.py` | `EMBEDDING_DK_COMBINED/{target}/...` | ~600 CPU-h / 36 jobs |
| **E** Fold-internal residualisation | §3.2 (Fig 3) | `--residualise-only` | `src/interp/res_no_leakage/fold_internal_residualisation.py` | `RES_NO_LEAKAGE/{target}/...` | ~540 CPU-h / 30 jobs |
| **F** MKNN(k=20) + permutation null | §3.2 (Table 1) | `--mknn-only` | `src/model/embedding_comparison.py` | `EMBEDDING_COMPARISON/component5_mknn/` | ~15 CPU-h / 1 job |
| **G** MKNN k-sweep ablation | Appendix A.2 | `--mknn-only` (same call) | `src/model/embedding_comparison.py` | `EMBEDDING_COMPARISON/component5_mknn_ksweep/` | shared with **F** |

The full project compute (including FM feature extraction on a single GPU
and all preliminary / failed runs) is ≈ 6,000 CPU-hours and ≈ 35 P100-hours
across ≈ 130 jobs (paper Appendix A).

### Worked example — single step

```bash
# Step E: fold-internal residualisation only
python cookbooks/pipeline.py \
    --residualise-only --all \
    --main-path /path/to/doc_eeg_data --metadata-dir /path/to/metadata \
    --mode patient --task lg \
    --results-subdir CBraMod/doc_patients \
    --emb-marker-csv /path/to/baseline_stable_20210128_scalars.csv \
    --emb-patient-labels-full /path/to/patient_labels.csv
```

---

## Figure reproduction

Once `<results-root>` contains the artefacts produced by the per-step table
above, the four canonical figures in the paper are produced by:

| Paper figure | Command | Output PNG |
|---|---|---|
| **Fig 1** Utility AUC, 6 tasks (§3.1) | `python -m src.paper_plots.fig1_utility --results-root <root>` | `plot1_per_target_dot_ttest_fdr_directional.png` |
| **Fig 2** Layer-wise R² + CRS AUC (§3.2) | `python -m src.paper_plots.fig2_layerwise --results-root <root>` | `plot3_combined_eegpt.png` |
| **Fig 3** FM / FM+DK / FM-res (§3.2) | `python -m src.paper_plots.fig3_residualisation --results-root <root>` | `plot4_noleak_combined_nonpca_annotated_ttest_fdr.png` |
| **Appendix A.2** MKNN k-sweep | `python -m src.paper_plots.fig_mknn_ksweep --results-root <root>` | `plot_mknn_k_sweep.png` |

Default `--output-dir` is `<results-root>/PLOTS/`. See
[`src/paper_plots/README.md`](src/paper_plots/README.md) for details on what
each script consumes and the available ablation variants under
[`src/paper_plots_legacy/`](src/paper_plots_legacy/).

---

## Statistical protocols

All AUCs are reported as mean over 100 outer-fold evaluations (5 outer folds
× 20 repeats), with patient-level fold integrity preserved across all FMs.
Significance is assessed with the Nadeau–Bengio variance-corrected paired
t-test (one-tailed, see `src/paper_plots/_corrected_ttest.py`), and
Benjamini-Hochberg FDR correction is applied across the family of comparisons
shown in each figure. The MKNN permutation-based null is built from 1000
random row-shufflings of the DK matrix.

---

## Tests

```bash
pytest tests/ -v
```

The unit tests cover the CV-split utilities used across all paper
evaluations. They do not require the clinical dataset and run in under a
minute on a modest workstation.

---

## Linting

```bash
ruff check src/
ruff format src/
```

Configuration (line length 88, target Python 3.8+) lives in
`pyproject.toml`.


---

## Beyond the paper

The original DoC research pipeline supports four legacy phases that **are
not part of the NeurIPS submission** but remain runnable for users
interested in the broader workflow:

| Legacy phase | Purpose | Flag |
|---|---|---|
| GENERAL_METRICS | MAPE / Pearson / FFT between original and reconstructed EEG | `--general-metrics-only` |
| DECODER | Temporal decoding (sliding-estimator + LogisticRegression) | `--decoder-only` |
| MODEL | SVM binary classifier on DK markers (legacy variant) | `--model-only` |

These phases reach into modules under [`src/misc/`](src/misc/) and can be
invoked alongside or independently of `--paper-eval`. They are documented in
[`src/misc/README.md`](src/misc/README.md).
