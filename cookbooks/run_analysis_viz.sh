#!/bin/bash
set -e
set -x

export PYTHON_EXE="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin/python"
export PATH="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin:$PATH"

echo "Python: $($PYTHON_EXE --version)"

RESULTS_DIR="/data/project/eeg_foundation/data/benchmark_results/new_results/CBraMod/doc_patients/DECODER/decoding-global-20260219_175359"

echo "Running analysis.py on $RESULTS_DIR ..."
$PYTHON_EXE src/decoder/analysis/analysis.py --results-dir "$RESULTS_DIR"

echo "Running viz.py on $RESULTS_DIR ..."
$PYTHON_EXE src/decoder/analysis/viz.py --results-dir "$RESULTS_DIR" --mode DOC

echo "Done!"
