#!/bin/bash
# Generic HTCondor wrapper for the markers phase.
# All parameters are passed as environment variables by the submit file.
# Multiple jobs coordinate via processing.lock + finished.txt.
set -e

export PYTHON_EXE="${PYTHON_EXE:-python3}"

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

# Subject selection: use the explicit partition if provided (set by markers.submit
# via "queue partition from partitions.txt"), otherwise fall back to --all.
if [ -n "$SUBJECTS" ]; then
    echo "[run_markers] Using subject partition: $(echo "$SUBJECTS" | tr ',' '\n' | wc -l) subjects"
    CMD+=(--subjects "$SUBJECTS")
else
    CMD+=(--all)
fi

exec "${CMD[@]}"
