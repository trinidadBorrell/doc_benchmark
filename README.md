This repository provides a benchmark for assessing the performance of time-series foundation models in reconstructing and forecasting EEG data from patients with disorders of consciousness (DoC). The framework encompasses three main phases: (1) marker computation, (2) statistical analysis, and (3) machine learning model training. It is built on the NICE neurophysiology framework with enhanced support for variable electrode configurations.

## 🎯 General Aim

The `doc_benchmark` pipeline provides a complete workflow for processing and analyzing EEG data from consciousness studies, specifically designed for:

- **Consciousness State Classification**: Using DOC-Forest methodology to distinguish between different states of consciousness
- **Data Reconstruction Validation**: Comparing original vs reconstructed EEG data quality
- **Cross-Subject Analysis**: Training machine learning models across multiple subjects  
- **Marker Benchmarking**: Comprehensive statistical comparison of neurophysiological markers

## 🔧 Main Pipeline Components

### 1. **Markers Phase** 🧠
- **Computes EEG markers** from raw `.fif` files using the NICE framework
- **Extracts features** including power spectral density, entropy measures, connectivity metrics
- **Supports variable electrode counts** (32, 64, 128, 256 channels)
- **Processes both original and reconstructed data**

### 2. **Analysis Phase** 📊  
- **Individual subject analysis**: Statistical comparison between original vs reconstructed markers
- **Global analysis**: Cross-subject statistical analysis and visualization
- **Generates comprehensive reports** with metrics, plots, and statistical tests

### 3. **Models Phase** 🤖
- **Machine learning classification**: ExtraTrees models for consciousness state prediction
- **Cross-subject training**: Uses all subjects' data for robust model training
- **Multiple configurations**: Trains models on scalar/topographic features from original/reconstructed data
- **Performance evaluation**: ROC curves, confusion matrices, feature importance analysis

## 🔬 Based on NICE Framework

This pipeline is built on the [NICE (Neurophysiological connectome)](https://github.com/nice-tools/nice) framework:
- **Standardized EEG markers**: Power spectral density, permutation entropy, Kolmogorov complexity
- **Connectivity measures**: Coherence, phase-amplitude coupling  
- **Time-locked analysis**: Event-related potentials and contrast measures
- **Robust preprocessing**: Artifact rejection and channel interpolation

## 📦 Installation

### Prerequisites
- Python 3.9 
- Virtual environment recommended

### Step-by-Step Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install base requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install NICE package** (requires special handling):
   ```bash
   pip install -e git+https://github.com/nice-tools/nice@e54fe9eda1318e3612b9a9e87a8e5d85c4b9f589#egg=nice
   ```

4. **Verify installation**:
   ```bash
   python -c "from nice import Markers; print('✅ NICE installed successfully!')"
   ```

## 🚀 Quick Start

### Basic Usage
```bash
# Process a single subject through all phases
python cookbooks/pipeline.py --subject AD023 \
  --metadata-dir /path/to/metadata \
  --data-dir /path/to/fifdata

# Process all subjects with parallel processing
python cookbooks/pipeline.py --all \
  --metadata-dir /path/to/metadata \
  --data-dir /path/to/fifdata \
  --batch-size 4
```

### Phase-Specific Execution
```bash
# Only compute markers
python cookbooks/pipeline.py --all --markers-only \
  --metadata-dir /path/to/metadata \
  --data-dir /path/to/fifdata

# Only run models training (after markers are computed)
python cookbooks/pipeline.py --models-only \
  --metadata-dir /path/to/metadata

# Only run global analysis
python cookbooks/pipeline.py --global-only \
  --metadata-dir /path/to/metadata
```

## 📁 Expected Data Structure

```
data_dir/
└── sub-{ID}/
    └── ses-{SESSION}/
        ├── *_original.fif  # Original EEG data
        └── *_recon.fif     # Reconstructed EEG data

metadata_dir/
└── patient_labels_with_controls.csv #Subject labels and states
                                     #Columns of CSV: subject, session, state, recording_date, cs_1y, cs_2y, cs_6m

```

## 📊 Output Structure

```
results/
├── SUBJECTS/           # Individual subject results
│   └── sub-{ID}/
│       └── {session}/
│           ├── markers_variable/     # Raw markers (.hdf5)
│           ├── features_variable/    # Processed features (.npy)
│           └── compare_markers/      # Statistical comparisons
├── EXTRATREES/         # Machine learning models
│   └── models_*/
│       ├── scalar_original/
│       ├── scalar_reconstructed/
│       ├── topo_original/
│       └── topo_reconstructed/
└── GLOBAL/            # Cross-subject analysis
    └── global_results_*/
```
## Citation

If you use this pipeline, please cite:

```
Engemann D.A.*, Raimondo F.*, King JR., Rohaut B., Louppe G.,
Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
Robust EEG-based cross-site and cross-protocol classification of
states of consciousness. Brain. doi:10.1093/brain/awy251
```
## Acknowledgements

This work was supported by [Paris Brain Institute America](https://parisbraininstitute-america.org/)’s project on Consciousness mapping.

```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated log files
3. Examine the comparison reports for diagnostic information
4. Contact: trinidad.borrell@gmail.com
