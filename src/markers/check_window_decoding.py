import numpy as np
from pathlib import Path

# Load the NPZ file
npz_file = Path("/data/project/eeg_foundation/src/doc_benchmark/results/new_results/MARKERS/scalars_sub-CHR117_task-restEC_scalars.npz")

if npz_file.exists():
    data = np.load(npz_file)
    print(f"Total keys in NPZ file: {len(data.files)}")
    
    # Find window_decoding markers
    wd_keys = [k for k in data.files if 'window_decoding' in k]
    print(f"\nFound {len(wd_keys)} window_decoding markers:")
    for k in wd_keys:
        print(f"  {k} = {data[k]}")
else:
    print(f"File not found: {npz_file}")
