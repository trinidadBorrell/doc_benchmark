#!/bin/bash
set -euo pipefail

MODEL=""
SUBJECTS=""
DATA_DIR=""
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)      MODEL="$2";      shift 2 ;;
    --subjects)   SUBJECTS="$2";   shift 2 ;;
    --data_dir)   DATA_DIR="$2";   shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$MODEL" ] || [ -z "$SUBJECTS" ] || [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 --model <CbraMod|NeuroLM> --subjects <sub-X,...> --data_dir <path> --output_dir <path>" >&2
  exit 2
fi

# Convert '+' separator (used to avoid HTCondor comma conflicts) back to ','
SUBJECTS="${SUBJECTS//+/,}"

echo "[$(date)] Starting run_pool_nonlinear.sh"
echo "Model:      $MODEL"
echo "Subjects:   $SUBJECTS"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

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

python -u "$SCRIPT_DIR/non_linear_probing.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --models "$MODEL" \
  --pool-only \
  --subjects "$SUBJECTS"

echo "[$(date)] Done."
