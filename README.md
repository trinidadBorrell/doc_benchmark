# doc_benchmark

EEG analysis pipeline for benchmarking neurophysiological markers and classifying consciousness states (VS vs MCS) in Disorders of Consciousness (DoC) patients. Supports multiple EEG data formats (CBraMod, TOTEM, LaBram, standard BIDS) and orchestrates 5 analysis phases.

## Quick Start

```bash
# Activate the conda environment on a worker node
condor_submit -i /data/project/eeg_foundation/jobs/interactive.submit
conda activate pytorch_ppc64le   # miniforge3_ppc64le/envs/pytorch_ppc64le

# Run the full pipeline for all subjects
cd /data/project/eeg_foundation/src/doc_benchmark
python cookbooks/pipeline.py \
    --main-path /data/project/eeg_foundation/data/CbraMod/recon_data_inference \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all \
    --results-subdir CBraMod/doc_patients \
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results
```

## Pipeline Phases

The pipeline (`cookbooks/pipeline.py`) orchestrates 5 independent phases, each runnable alone or in combination:

| Phase | Flag | Module | Purpose |
|-------|------|--------|---------|
| A. GENERAL_METRICS | `--general-metrics-only` | `src/general_metrics/compute_metrics.py` | MAPE, Pearson correlation, FFT between original & reconstructed EEG |
| B. MLP_EMBEDDING | `--mlp-embedding-only` | `src/model/mlp_embedding_classifier.py` | MLP/RF/KernelRidge classification on foundation model embeddings; also supports marker regression and full clinical target prediction |
| C. DECODER | `--decoder-only` | `src/decoder/decoder.py` | Temporal decoding with SlidingEstimator + LogisticRegression |
| D. MARKERS | `--markers-only` | `src/markers/compute_markers_with_junifer.py` | Junifer feature extraction -> HDF5 -> 120 scalars + topographies |
| E. MODEL | `--model-only` | `src/model/support_vector_machine.py` | SVM binary classification (VS vs MCS) |

Skip individual phases with `--skip-general-metrics`, `--skip-decoder`, etc.

### MLP_EMBEDDING Modes (Phase B)

`mlp_embedding_classifier.py` supports three operating modes:

```bash
# Standard binary VS vs MCS classification (single split)
python src/model/mlp_embedding_classifier.py \
    --data-dir /path/to/embeddings \
    --patient-labels /path/to/patient_labels_with_controls.csv \
    --output-dir /path/to/out

# 5-fold nested CV
python src/model/mlp_embedding_classifier.py ... --full-cv --n-cv-folds 5

# Marker regression: predict each scalar marker from embeddings (Ridge)
python src/model/mlp_embedding_classifier.py ... \
    --marker-regression \
    --marker-csv /data/original_DoC/baseline_stable_20210128_scalars.csv \
    --marker-reduction A

# Full clinical target prediction (crs, etiology, cs_6m, cs_1y, cs_2y)
python src/model/mlp_embedding_classifier.py ... \
    --full-metric-prediction \
    --patient-labels-full /data/metadata/patient_labels.csv

# Subject intersection across two foundation models
python src/model/mlp_embedding_classifier.py ... \
    --use-subject-intersection \
    --embedding-dirs TOTEM=/path/to/totem CBraMod=/path/to/cbramod
```

### Markers Phase Detail (D1-D4)

The markers phase processes each subject through 4 steps:

1. **D1 (junifer)** - Feature extraction via Junifer YAML configs -> HDF5 file
2. **D2 (compute_data)** - Report data generation from H5 + FIF
3. **D3 (compute_scalars)** - Scalar aggregation: 4 variants per marker (mean/std x trimmean80/std), filtered by ROI
4. **D4 (compute_topographies)** - Topographic map extraction from H5

Steps D3 and D4 run concurrently (both read the H5 file independently).

## Running the Pipeline

### Subject Selection (mutually exclusive)

```bash
--all                 # All discovered subjects
--subject BA001       # Single subject
--subjects BA001,BA002,BA010   # Comma-separated list
--random 5            # N random subjects
```

### Phase Selection

```bash
# Run only one phase
--markers-only
--decoder-only
--general-metrics-only
--mlp-embedding-only
--model-only

# Skip specific phases (combine freely)
--skip-markers --skip-model
```

### Additional Options

```bash
--save-time           # Write per-step timing CSV to results/logs/ (crash-safe: flushed after every D-step)
--keep-h5             # Retain H5 files after markers phase (default: cleaned up)
--skip-clustering     # Skip cluster permutation tests in markers
--dry-run             # Show what would run without executing
--verbose             # Verbose logging
--data-source CBraMod # Override auto-detection (auto|CBraMod|TOTEM|LaBram|standard|suffix|bids)
--results-subdir CBraMod/doc_patients   # Subdirectory within results
--original-data-path /path/to/orig1 /path/to/orig2   # Separate original data locations
```

## Parallelism

There are three levels of parallelism, from simplest to most powerful:

### 1. Sequential (default)

One subject at a time. No concurrency. Good for debugging.

```bash
# On a worker node (conda activate pytorch_ppc64le   # miniforge3_ppc64le/envs/pytorch_ppc64le first)
python cookbooks/pipeline.py ... --markers-only --batch-size 1
```

### 2. In-Process Parallelism (`--batch-size N`)

Processes N subjects concurrently **within a single pipeline run** using a thread pool. Each subject still runs its D-steps sequentially (D1 -> D2 -> D3+D4), but multiple subjects overlap. Threads are I/O-bound (waiting on subprocesses), so there is no GIL contention.

```bash
# On a worker node (conda activate pytorch_ppc64le   # miniforge3_ppc64le/envs/pytorch_ppc64le first)
python cookbooks/pipeline.py \
    --main-path /data/project/eeg_foundation/data/CbraMod/recon_data_inference \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --all \
    --markers-only --batch-size 4 --save-time \
    --results-subdir CBraMod/doc_patients \
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results
```

`--batch-size` is automatically capped at `(cpu_count - 1)`.

> **Note:** All subprocess calls inside the pipeline use `sys.executable`, ensuring every child process (junifer, compute_scalars, etc.) runs in the same conda environment as the pipeline itself.

### 3. HTCondor Array Jobs (cluster-level)

Distribute subjects across multiple cluster nodes. Each node runs its own pipeline process with `--batch-size` for additional within-node parallelism. Jobs coordinate via filesystem locks — no pre-partitioning step required.

**Submit the array job:**

```bash
# CBraMod (default — no overrides needed)
condor_submit jobs/markers.submit

# LaBram
condor_submit jobs/markers.submit \
    dataset=labram \
    main_path=/data/project/eeg_foundation/data/LaBram/results_DoC_lg/recon_data_inference \
    results_subdir=LaBram/doc_patients

# NeuroLM (BIDS layout)
condor_submit jobs/markers.submit \
    dataset=neurolm \
    main_path=/data/project/eeg_foundation/data/NeuroLM-output/fif_data_target \
    results_subdir=NeuroLM/doc_patients \
    data_source=bids
```

This submits 4 identical HTCondor jobs (`queue 4`). Each runs `--all --batch-size 4` and they coordinate automatically:
- **`processing.lock`**: Atomic lock file created before processing a subject. Prevents two jobs from working on the same subject simultaneously.
- **`finished.txt`**: Permanent marker written after successful completion. Subjects with this marker are skipped on re-runs.

**Dataset quick reference:**

| Dataset | `dataset=` | `main_path=` | `results_subdir=` | Extra |
|---------|-----------|-------------|------------------|-------|
| CBraMod | `cbramod` *(default)* | `.../CbraMod/recon_data_inference` | `CBraMod/doc_patients` | — |
| LaBram | `labram` | `.../LaBram/results_DoC_lg/recon_data_inference` | `LaBram/doc_patients` | — |
| NeuroLM | `neurolm` | `.../NeuroLM-output/fif_data_target` | `NeuroLM/doc_patients` | `data_source=bids` |

**Monitor the jobs:**

```bash
# Check job status
condor_q                                  # overview
condor_q -nobatch                         # one line per job

# Tail live output (replace cbramod/12345 with your dataset/cluster ID)
tail -f /data/project/eeg_foundation/logs/markers_cbramod_12345.*.out

# Grep for progress lines across all jobs
grep "^>>> Progress" /data/project/eeg_foundation/logs/markers_cbramod_12345.*.out

# Check which subjects have finished so far
find /data/project/eeg_foundation/data/benchmark_results/new_results/CBraMod/doc_patients/MARKERS \
    -name finished.txt | wc -l
```

**Result:** 4 nodes x 4 threads = up to 16 subjects processed concurrently across the cluster.

**Files involved:**

| File | Purpose |
|------|---------|
| `cookbooks/run_markers.sh` | Generic HTCondor wrapper — parameters passed as env vars |
| `jobs/markers.submit` | HTCondor submit file: 16 CPUs, 32 GB, 7-day limit, `queue 4` |

**How parameterization works:**

`markers.submit` uses HTCondor's `$(macro:default_value)` inline syntax to embed defaults
directly in every reference:

```
environment = "MAIN_PATH=$(main_path:/path/to/CbraMod/...) ..."
output = .../markers_$(dataset:cbramod)_$(Cluster).$(Process).out
```

When you run `condor_submit markers.submit dataset=labram main_path=...`, HTCondor
registers those key=value pairs as macros *before* parsing the file, so `$(dataset:cbramod)`
expands to `labram` and `$(main_path:...)` expands to the LaBram path.  The `:default`
fallback is only used when the macro was never set (i.e., plain `condor_submit
markers.submit` with no overrides → CBraMod defaults).

> **Why not plain `=` assignments?**  In submit files, `key = value` always overwrites the
> macro table entry — including values already set by command-line arguments.  Command-line
> overrides would be silently ignored.
>
> **Why not `?=`?**  The conditional-assignment operator `?=` is valid in
> `condor_config` files but is **not recognised in submit description files** — it
> produces a parse error.  Inline `$(macro:default)` is the correct portable solution.

### Combining Levels

The three levels compose naturally:

```
Level 3 (cluster)    4 HTCondor jobs across 4 nodes
    |
Level 2 (in-process) Each job runs --batch-size 4 (4 threads)
    |
Level 1 (per-subject) D3 + D4 run concurrently within each subject
```

### Crash Safety

- **`finished.txt` markers**: Written after each subject completes successfully. Subjects with this marker are skipped on re-runs, making the pipeline idempotent.
- **`processing.lock`**: Atomic lock file (created with `O_CREAT | O_EXCL`) prevents two jobs from processing the same subject. Cleaned up after completion or failure so the subject can be retried.
- **Per-step timing flushes**: When `--save-time` is enabled, the timing CSV is flushed after every D-step (D1, D2, D3, D4), not just after each subject. If junifer gets killed mid-run, all completed steps are preserved.
- **Thread-safe timing**: The timing CSV writer uses a lock, so parallel subjects don't corrupt the file.

## HTCondor Job Files

| Submit File | Executable | Purpose |
|-------------|-----------|---------|
| `jobs/job.submit` | `cookbooks/run_pipeline.sh` | MLP embedding on test data |
| `jobs/decoder_rerun.submit` | `cookbooks/run_decoder_only.sh` | Re-run decoder phase |
| `jobs/analysis_viz.submit` | `cookbooks/run_analysis_viz.sh` | Analysis + visualization post-processing |
| `jobs/markers.submit` | `cookbooks/run_markers.sh` | Markers array job — parameterized (dataset, paths, data_source) |
| `jobs/interactive.submit` | (interactive) | Get an interactive worker node |
| `jobs/interactive_long.submit` | (interactive) | Long-running interactive session |

Common condor commands:

```bash
condor_submit jobs/markers.submit         # submit (CBraMod default)
condor_q                                  # check status
condor_q -nobatch                         # detailed status
condor_rm <cluster_id>                    # cancel all jobs in cluster
condor_tail -f <cluster_id>.<process>     # live output
```

## Output Directory Structure

```
results/{results-subdir}/
├── GENERAL_METRICS/       # metrics.json, plots
├── MLP_EMBEDDING/         # {classic_split,nested_cv}/{mlp,random_forest,kernel_ridge}/
│                          # classification_results.json, roc_curve.png
│                          # regressor_results/  (--marker-regression)
│                          # {crs,etiology,cs_6m,cs_1y,cs_2y}/  (--full-metric-prediction)
├── DECODER/               # decoding_results.pkl, accuracy plots, topographies
├── MARKERS/
│   └── sub-{ID}/
│       └── ses-{NUM}/
│           ├── finished.txt              # completion marker
│           ├── original/
│           │   ├── icm_complete_features.h5
│           │   ├── scalars_{ID}_ses-{NUM}_original.npz
│           │   └── topos_{ID}_ses-{NUM}_original.npz
│           └── recon/
│               ├── icm_complete_features.h5
│               ├── scalars_{ID}_ses-{NUM}_recon.npz
│               └── topos_{ID}_ses-{NUM}_recon.npz
├── MARKER_BASELINE/       # standalone baseline.py output
│   └── {crs,etiology,cs_6m,cs_1y,cs_2y}/
│       ├── classic_split/
│       │   └── {svm,random_forest,kernel_ridge}/
│       │       ├── classification_results.json
│       │       ├── classification_results.png
│       │       └── subject_predictions.csv
│       └── nested_cv/
│           └── {svm,random_forest,kernel_ridge}/...
├── MODEL/                 # classification_report.json, confusion_matrix.png
└── logs/
    ├── pipeline_{timestamp}.log
    └── timing_{timestamp}.csv            # (when --save-time is used)
```

## Marker Baseline Classifier (standalone)

`src/model/baseline.py` is a standalone script that trains SVM, Random Forest, and Kernel Ridge classifiers directly on pre-computed neurophysiological scalar markers (not on foundation model embeddings). It is the classical marker-based baseline for comparison.

**Prediction targets** — all five run in sequence:

| Target | Description |
|--------|-------------|
| `crs` | Binary VS vs MCS (UWS→VS, MCS+/MCS-→MCS) |
| `etiology` | Binary acute vs chronic |
| `cs_6m` | Outcome score at 6 months (multiclass + binary collapse) |
| `cs_1y` | Outcome score at 1 year (multiclass + binary collapse) |
| `cs_2y` | Outcome score at 2 years (multiclass + binary collapse) |

For outcome targets (`cs_6m/cs_1y/cs_2y`) both a multiclass run and a binary (VS vs MCS) collapse run are saved under `{target}/multiclass/` and `{target}/binary/`.

**Usage:**

```bash
# Classic split (default), all 5 targets, reduction A
python src/model/baseline.py \
    --original-metadata /data/project/eeg_foundation/data/original_DoC/baseline_stable_20210128_scalars.csv \
    --patient-labels /data/project/eeg_foundation/data/metadata/patient_labels.csv \
    --main-path /data/project/eeg_foundation/data/benchmark_results/new_results

# 5-fold nested CV, reduction B
python src/model/baseline.py \
    --original-metadata /path/to/scalars.csv \
    --patient-labels /path/to/patient_labels.csv \
    --main-path /path/to/results \
    --full-cv --n-cv-folds 5 --marker-reduction B
```

**Reduction map** (`--marker-reduction`):

| Letter | Path |
|--------|------|
| A (default) | `icm/lg/egi256/trim_mean80` |
| B | `icm/lg/egi256/std` |
| C | `icm/lg/egi256gfp/trim_mean80` |
| D | `icm/lg/egi256gfp/std` |

**Output** lands in `{main-path}/MARKER_BASELINE/{target}/{classic_split,nested_cv}/{svm,random_forest,kernel_ridge}/`.

## Development

### Lint

```bash
ruff check src/
ruff format src/
```

### Tests

```bash
pytest tests/ -v
pytest tests/test_compute_metrics.py -v
pytest tests/test_compute_markers_hdf5.py -v
```

## Acknowledgements

This project is supported by [Paris Brain Institute America](https://parisbraininstitute-america.org/).

### Data Source Auto-Detection

The pipeline auto-detects the data layout (override with `--data-source`):

1. **CBraMod** - `sub-{id}/ses-{num}/sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif`
2. **Standard** - `sub-{id}/ses-{num}/orig/*.fif` and `recon/*.fif`
3. **Suffix** - `*_original.fif` and `*_recon.fif` in same session dir
4. **BIDS** - `sub-{id}/ses-{num}/eeg/*.fif`
5. **Single-file** - any `.fif` in session dir (reconstructed-only fallback)
