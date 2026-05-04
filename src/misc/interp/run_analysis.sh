#!/bin/bash
set -euo pipefail

MODEL=""
LAYER=""
DATA_DIR=""
OUTPUT_DIR=""
PATIENT_LABELS=""
MARKER_CSV=""
SPLITS_FILE=""
EXTRA_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)          MODEL="$2";          shift 2 ;;
    --layer)          LAYER="$2";          shift 2 ;;
    --data_dir)       DATA_DIR="$2";       shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
    --patient_labels) PATIENT_LABELS="$2"; shift 2 ;;
    --marker_csv)     MARKER_CSV="$2";     shift 2 ;;
    --splits_file)    SPLITS_FILE="$2";    shift 2 ;;
    --extra)          EXTRA_ARGS="$2";     shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$MODEL" ] || [ -z "$LAYER" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PATIENT_LABELS" ]; then
  echo "Usage: $0 --model M --layer L --data_dir D --output_dir O --patient_labels P [--marker_csv C] [--splits_file S] [--extra ARGS]" >&2
  exit 2
fi

echo "[$(date)] Starting run_analysis.sh"
echo "Model:          $MODEL"
echo "Layer:          $LAYER"
echo "Data dir:       $DATA_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Patient labels: $PATIENT_LABELS"
echo "Marker CSV:     $MARKER_CSV"
echo "Splits file:    $SPLITS_FILE"
echo "Extra args:     $EXTRA_ARGS"

# Activate conda
CONDA_SH="$HOME/miniforge3_ppc64le/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: conda.sh not found at $CONDA_SH" >&2
  exit 1
fi
set +u
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate pytorch_ppc64le
set -u

echo "[$(date)] Conda environment activated"

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMD="python -u $SCRIPT_DIR/linear_probing.py \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR \
  --patient-labels $PATIENT_LABELS \
  --models $MODEL \
  --layers $LAYER"

if [ -n "$MARKER_CSV" ]; then
  CMD="$CMD --marker-csv $MARKER_CSV"
fi

if [ -n "$SPLITS_FILE" ]; then
  CMD="$CMD --splits-file $SPLITS_FILE"
fi

if [ -n "$EXTRA_ARGS" ]; then
  CMD="$CMD $EXTRA_ARGS"
fi

echo "[$(date)] Running: $CMD"
eval "$CMD"

echo "[$(date)] Done."
