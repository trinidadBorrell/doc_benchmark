#!/bin/bash

# Exit on any error
set -e

# Print commands for debugging
set -x

# Use the Python directly from the conda environment
export PYTHON_EXE="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin/python"
export PATH="/home/triniborrell/miniforge3_ppc64le/envs/pytorch_ppc64le/bin:$PATH"

# Print debugging info
echo "Using Python directly from conda environment: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo "Python location: $PYTHON_EXE"

# Verify we can import key packages
$PYTHON_EXE -c "import sys; print('Python executable:', sys.executable)"
$PYTHON_EXE -c "import sys; print('Python path:', sys.path[:3])"

# Run the pipeline with --decoder-only to re-run just the decoder phase
$PYTHON_EXE cookbooks/pipeline.py --all \
    --main-path /data/project/eeg_foundation/data/CbraMod/recon_data_inference \
    --metadata-dir /data/project/eeg_foundation/data/metadata \
    --mode patient --task lg --data-source CBraMod \
    --results-subdir CBraMod/doc_patients \
    --original-data-path \
        /data/project/eeg_foundation/data/data_250Hz_EGI256/zero_shot_data/DOC/fifdata \
        /data/project/eeg_foundation/data/data_250Hz_EGI256/nice_epochs_from_cohen_2/nice_epochs/nice_epochs2 \
    --decoder-only
