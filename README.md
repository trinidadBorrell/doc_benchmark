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

`src/model/baseline.py` trains SVM, Random Forest, and Kernel Ridge directly on pre-computed scalar markers (not embeddings) — the classical baseline for comparison.

Prediction targets: `crs`, `etiology`, `cs_6m`, `cs_1y`, `cs_2y`.

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
│   └── {target}/{classic_split,nested_cv}/{svm,random_forest,kernel_ridge}/
├── MODEL/
└── logs/
```

## Acknowledgements

This project is made possible by the generous support of [Paris Brain Institute America](https://parisbraininstitute.org/). We are deeply grateful to the institute for funding this research and for their continued commitment to advancing the understanding and treatment of disorders of consciousness. Their support enables the scientific infrastructure and clinical insights that drive this work forward.
