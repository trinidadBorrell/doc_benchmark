# doc_benchmark

EEG analysis pipeline for benchmarking neurophysiological markers and classifying consciousness states (VS vs MCS) in Disorders of Consciousness (DoC) patients. 

## Quick Start

```bash
# Get a worker node first
condor_submit -i /data/project/eeg_foundation/jobs/interactive.submit
conda activate pytorch_ppc64le

# Run the full pipeline
cd /data/project/eeg_foundation/src/doc_benchmark
python cookbooks/pipeline.py \
    --main-path /data/project/eeg_foundation/data/CbraMod/recon_data_inference \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all \
    --results-subdir CBraMod/doc_patients \
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results
```

## Pipeline Phases

| Phase | Flag | Purpose |
|-------|------|---------|
| A. GENERAL_METRICS | `--general-metrics-only` | MAPE, Pearson correlation, FFT between original & reconstructed EEG |
| B. MLP_EMBEDDING | `--mlp-embedding-only` | MLP/RF/KernelRidge classification on foundation model embeddings, linear probing, dimensionality study of embedding space|
| C. DECODER | `--decoder-only` | Temporal decoding with SlidingEstimator + LogisticRegression |
| D. MARKERS | `--markers-only` | Junifer feature extraction → HDF5 → 120 scalars + topographies |
| E. MODEL | `--model-only` | SVM binary classification (VS vs MCS) |

Each phase is independent. Skip phases with `--skip-markers`, `--skip-decoder`, etc.

## Running the Pipeline

**Subject selection** (mutually exclusive):

```bash
--all                        # all subjects
--subject BA001              # single subject
--subjects BA001,BA002       # comma-separated list
--random 5                   # N random subjects
```

**Useful options:**

```bash
--save-time           # write per-step timing CSV to results/logs/
--keep-h5             # retain H5 files after markers phase
--dry-run             # show what would run without executing
--data-source CBraMod # override auto-detection (auto|CBraMod|TOTEM|LaBram|bids)
--results-subdir CBraMod/doc_patients
```

## Parallelism

Three composable levels:

| Level | How | Use |
|-------|-----|-----|
| 1. Sequential | default | debugging |
| 2. In-process | `--batch-size N` | N subjects concurrently within one run (capped at `cpu_count - 1`) |
| 3. Cluster | `condor_submit jobs/markers.submit` | distributes subjects across nodes; jobs coordinate via filesystem locks |

**Submit markers array job:**

```bash
# CBraMod (default)
condor_submit jobs/markers.submit

# LaBram
condor_submit jobs/markers.submit \
    dataset=labram \
    main_path=/data/project/eeg_foundation/data/LaBram/results_DoC_lg/recon_data_inference \
    results_subdir=LaBram/doc_patients

# NeuroLM
condor_submit jobs/markers.submit \
    dataset=neurolm \
    main_path=/data/project/eeg_foundation/data/NeuroLM-output/fif_data_target \
    results_subdir=NeuroLM/doc_patients \
    data_source=bids
```

4 nodes × 4 threads = up to 16 subjects processed concurrently. Jobs skip already-finished subjects (`finished.txt`) and use atomic locks (`processing.lock`) to avoid duplicate work.

**Monitor:**

```bash
condor_q
tail -f /data/project/eeg_foundation/logs/markers_cbramod_<cluster>.*.out
grep "^>>> Progress" /data/project/eeg_foundation/logs/markers_cbramod_<cluster>.*.out
```

## Marker Baseline Classifier

`src/model/baseline.py` trains SVM, MLP, Random Forest, and Kernel Ridge directly on pre-computed scalar markers (not embeddings) — the classical baseline for comparison.

**Prediction targets and modes:**

| Target | Modes |
|--------|-------|
| `crs` | binary VS vs MCS |
| `etiology` | all subjects · `vs_only` (UWS baseline) · `mcs_only` (MCS baseline) |
| `etiology_code` | all subjects · `vs_only` · `mcs_only` |
| `cs_6m/cs_1y/cs_2y` | `multiclass` · `binary` · `binary_death` · `binary_vs_to_mcs` · `binary_mcs_to_conscious` · `binary_improvement` |

`binary_improvement` labels all subjects as IMPROVED/NON_IMPROVED regardless of baseline (VS→MCS/CONSCIOUS or MCS→CONSCIOUS).

```bash
python src/model/baseline.py \
    --original-metadata /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../patient_labels.csv \
    --main-path /data/.../benchmark_results/new_results \
    [--full-cv --n-cv-folds 5] [--marker-reduction A|B|C|D]
```

## Output Structure

```
results/{results-subdir}/
├── GENERAL_METRICS/
├── MLP_EMBEDDING/
│   └── {classic_split,nested_cv}/{mlp,random_forest,kernel_ridge}/
│       ├── classification_results.json
│       └── {crs,etiology,cs_6m,cs_1y,cs_2y}/   (--full-metric-prediction)
├── DECODER/
├── MARKERS/
│   └── sub-{ID}/ses-{NUM}/
│       ├── finished.txt
│       ├── original/{scalars,topos}_*.npz
│       └── recon/{scalars,topos}_*.npz
├── MARKER_BASELINE/
│   ├── crs/{classic_split,nested_cv}/{svm,mlp,random_forest,kernel_ridge}/
│   ├── {etiology,etiology_code}/{classic_split,nested_cv,vs_only/,mcs_only/}/
│   └── {cs_6m,cs_1y,cs_2y}/{multiclass,binary,binary_death,binary_vs_to_mcs,binary_mcs_to_conscious,binary_improvement}/
├── MODEL/
└── logs/
```

## Post-Hoc Studies

These scripts are **not** part of the main pipeline. They are run manually after pipeline results are available. Each can be invoked as a standalone Python script with `--help` for full argument docs.

---

### Decoder Analysis (`src/decoder/analysis/`)

Run after the DECODER phase to add statistical testing and publication-quality plots.

| Script | Purpose |
|--------|---------|
| `analysis.py` | Per-timepoint Wilcoxon signed-rank tests against chance (0.5 AUC), with FDR correction |
| `viz.py` | Publication-quality AUC timeseries plots with significance shading |
| `peak_analysis.py` | Peak detection in per-subject AUC curves using prominence thresholding |

```bash
python src/decoder/analysis/analysis.py \
    --results-dir results/{subdir}/DECODER/decoding-global-{timestamp}/ \
    --output-dir results/{subdir}/DECODER/stats/

python src/decoder/analysis/viz.py \
    --results-dir results/{subdir}/DECODER/decoding-global-{timestamp}/ \
    --stats-file results/{subdir}/DECODER/stats/wilcoxon_results.csv \
    --output-dir results/{subdir}/DECODER/plots/
```

---

### Global Marker Analysis (`src/global_analysis/`)

Group-level analyses comparing original vs. reconstructed neurophysiological markers.

| Script | Purpose |
|--------|---------|
| `global_topoplots_minimal.py` | Six topographic comparison plots (orig/recon/diff) with FDR-corrected Wilcoxon tests and Spearman correlation heatmaps, broken down by diagnosis group |
| `individual_analysis.py` | Per-subject scalar/topographic metrics (correlation, MSE, NMSE) and GFP plots per event type |
| `statistical_analysis.py` | Permutation-based cluster tests, paired Wilcoxon tests, effect sizes (Cohen's d), and SSIM for topographic fidelity |
| `qualitative_analysis.py` | Time-frequency spectrograms and electrode×epoch heatmaps for a subset of representative subjects |
| `ohbm_biomarker_group_comparison.py` | 9×4 topographic grid (9 biomarkers × 4 diagnostic groups) with Spearman ρ and FDR stars |
| `ohbm_plots.py` | 4-row decoder results figure grouped by subject type (EMCS / MCS / UWS / COMA) |
| `control_rs_plots_CBraMod.py` | 3-column topographic grid (original / reconstructed / relative difference) for control resting-state data |

```bash
python src/global_analysis/global_topoplots_minimal.py \
    --results-dir results/{subdir}/MARKERS/ \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/{subdir}/GLOBAL_ANALYSIS/

python src/global_analysis/statistical_analysis.py \
    --results-dir results/{subdir}/MARKERS/ \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/{subdir}/GLOBAL_ANALYSIS/stats/
```

---

### Embedding Interpretability (`src/interp/`)

Layer-by-layer probing studies to understand what neurophysiological information is encoded at each depth of the foundation models.

| Script | Purpose |
|--------|---------|
| `linear_probing.py` | For each layer: (1) Ridge regression from embeddings to each marker scalar (R² per marker per layer) and (2) nested CV VS/MCS classification (AUC per layer). Can run in `--pool-only` mode to cache mean-pooled per-layer embeddings first. |
| `embedding_steering.py` | Trains per-marker linear probes, then steers embeddings along probe directions to measure the effect on VS/MCS predictions (representation engineering) |
| `residualization_embeddings.py` | Removes marker-predictable variance from embeddings by projecting out all probe directions, then re-classifies VS/MCS to quantify how much marker information was load-bearing for classification |
| `plot_layers.py` | R² curves across layers — one curve per layer, markers on x-axis |
| `plot_classification_layers.py` | Per-layer AUC grouped bar chart by classifier |
| `launch_pool_jobs.py` | Generates HTCondor submit files for parallel linear probing (pool → analysis → aggregate stages) |

```bash
# Step 1: pool mean embeddings per layer (parallelisable via HTCondor)
python src/interp/linear_probing.py \
    --results-root /data/.../benchmark_results/new_results \
    --model CBraMod --output-dir results/interp/CBraMod/ \
    --pool-only

# Step 2: per-layer regression + classification
python src/interp/linear_probing.py \
    --results-root /data/.../benchmark_results/new_results \
    --model CBraMod --output-dir results/interp/CBraMod/ \
    --layer 0   # repeat for each layer

# Step 3: embedding steering
python src/interp/embedding_steering.py \
    --results-root /data/.../benchmark_results/new_results \
    --model CBraMod \
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/interp/CBraMod/steering/

# Step 4: residualization
python src/interp/residualization_embeddings.py \
    --results-root /data/.../benchmark_results/new_results \
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/interp/residualization/
```

**Outputs per model:**
```
results/interp/{Model}/
├── pooled_chan_layers.npz          # cached per-layer pooled embeddings
├── layer_{N}/
│   ├── summary.json                # R² per marker
│   ├── classification_results.json # AUC per classifier
│   └── feature_importance_raw.png
├── steering/
│   ├── embedding_steering_results.json
│   ├── embedding_steering_summary.csv
│   └── causal_effects_comparison_r2_order.png
└── residualization/
    ├── results.json / results.csv
    └── auc_comparison.png
```

For cluster execution across all layers in parallel:
```bash
python src/interp/launch_pool_jobs.py --model CBraMod --n-jobs 8
```

---

### Embedding Comparison Studies (`src/model/`)

Cross-model studies comparing foundation model embeddings against each other and against domain-knowledge markers. All use shared CV splits for fair comparison.

| Script | Purpose |
|--------|---------|
| `embedding_comparison.py` | Four-component analysis: (1) noise-ceiling-corrected R² (how well FM embeddings predict DK markers), (2) RSA — pairwise RDM alignment (Spearman ρ + bootstrap CIs), (3) CKA — linear Centered Kernel Alignment, (4) dimensionality metrics (power-law exponent, participation ratio, effective rank, threshold-based n_80/90/95/99) |
| `learning_curve.py` | Performance vs. data budget: evaluates all models on common class-balanced session subsets from N=5 to N=full, with identical nested CV folds across models |
| `feature_importance_comparison.py` | Cross-model R² heatmap — which models best encode which markers (max-normalised per marker) |
| `dk_embeddings_classification.py` | FM + domain-knowledge combined classifier with permutation controls (FM-shifted / DK-shifted / both-shifted) to isolate each modality's contribution |

```bash
# Four-component embedding comparison
python src/model/embedding_comparison.py \
    --results-dir /data/.../benchmark_results/new_results \
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/embedding_comparison/

# Learning curve across data budgets
python src/model/learning_curve.py \
    --results-dir /data/.../benchmark_results/new_results \
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/learning_curve/ \
    --budgets 5,10,20,40,full

# FM + DK combined classification
python src/model/dk_embeddings_classification.py \
    --results-dir /data/.../benchmark_results/new_results \
    --marker-csv /data/.../baseline_stable_20210128_scalars.csv \
    --patient-labels /data/.../metadata_patient_labels.csv \
    --output-dir results/dk_embeddings/
```

---

### Multi-Model Aggregation (`cookbooks/intermodel_results.py`)

Re-runs MLP embedding classification and/or aggregates decoder results on the **intersection** of subjects available across all foundation models, ensuring identical CV folds for fair head-to-head comparison.

```bash
python cookbooks/intermodel_results.py \
    --results-root /data/.../benchmark_results/new_results \
    --output-root /data/.../benchmark_results/new_results/intermodel \
    --patient-labels /data/.../metadata_patient_labels.csv \
    [--decoder-only | --mlp-embedding-only]
```

Outputs land in `{output_root}/DECODER/intermodel-{timestamp}/` and `{output_root}/MLP-CLASSIFIER/intermodel-{timestamp}/`, one subdirectory per model.

---

## Acknowledgements

This project is made possible by the generous support of [Paris Brain Institute America](https://parisbraininstitute.org/). We are deeply grateful to the institute for funding this research and for their continued commitment to advancing the understanding and treatment of disorders of consciousness. Their support enables the scientific infrastructure and clinical insights that drive this work forward.
