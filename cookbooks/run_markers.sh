#!/bin/bash
# Generic HTCondor wrapper for the markers phase.
# All parameters are passed as environment variables by the submit file.
# Multiple jobs coordinate via processing.lock + finished.txt.
set -e

export PYTHON_EXE="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin/python"
export PATH="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin:$PATH"

# Required
: "${MAIN_PATH:?MAIN_PATH must be set in submit file}"
: "${RESULTS_SUBDIR:?RESULTS_SUBDIR must be set in submit file}"

# Optional with defaults
TASK="${TASK:-lg}"
MODE="${MODE:-patient}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd /data/project/eeg_foundation/src/doc_benchmark

CMD=(
    "$PYTHON_EXE" cookbooks/pipeline.py
    --all
    --main-path "$MAIN_PATH"
    --metadata-dir /data/project/eeg_foundation/data/metadata
    --mode "$MODE" --task "$TASK"
    --markers-only --save-time --keep-h5 --batch-size "$BATCH_SIZE"
    --results-subdir "$RESULTS_SUBDIR"
    --results-dir /data/project/eeg_foundation/data/benchmark_results/new_results
)

# Optional flags (only added when set)
[ -n "$DATA_SOURCE" ]         && CMD+=(--data-source "$DATA_SOURCE")
[ -n "$ORIGINAL_DATA_PATH" ]  && CMD+=(--original-data-path $ORIGINAL_DATA_PATH)

exec "${CMD[@]}"
