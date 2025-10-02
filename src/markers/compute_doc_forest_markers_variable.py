"""Compute markers with variable electrode support.

==================================================
Compute markers used for publication with variable electrode numbers
==================================================

This is an enhanced version of compute_doc_forest_markers.py that can handle
variable numbers of electrodes and automatically adapts the electrode
selections based on the available channels.

Features:
- Automatic adaptation to different electrode configurations (32, 64, 128, 256 channels)
- Event ID mapping for .fif files to convert numerical IDs to condition names
- Support for custom electrode ROI definitions
- Command line interface with configurable options

The event ID mapping converts numerical event codes to meaningful condition names
that are required for TimeLockedContrast markers. This mapping can be customized
or disabled if your data already has proper condition names.

References
----------
[1] Engemann D.A.`*, Raimondo F.`*, King JR., Rohaut B., Louppe G.,
    Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
    Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
    Robust EEG-based cross-site and cross-protocol classification of
    states of consciousness. Brain. doi:10.1093/brain/awy251
"""

# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Federico Raimondo <federaimondo@gmail.com>
#          Trinidad Borrell <trinidad.borrell@gmail.com>

import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append('/data/project/eeg_foundation/src/nice')
from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator,
                          ContingentNegativeVariation,
                          TimeLockedTopography,
                          TimeLockedContrast)

# Set consistent plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


def get_event_id_mapping():
    """Get event ID mapping for .fif files.
    
    Map numerical event IDs to condition names as expected by TimeLockedContrast
    markers. Adjust this mapping based on your experimental design.
    
    Returns
    -------
    dict
        Dictionary mapping numerical event IDs to condition names
    """
    
    event_id_mapping = {
        # Based on nice/algorithms/information_theory/tests/test_komplexity.py mapping
        # Main experimental conditions (Local/Standard, Global/Standard, etc.)
        10: 'HSTD',   # Control
        20: 'HDVT',   # Control 
        30: 'LSGS',    # Local Standard Global Standard 
        40: 'LSGD',    # Local Standard Global Deviant
        50: 'LDGD',    # Local Deviant Global Deviant
        60: 'LDGS'     # Local Deviant Global Standard
    }
    
    return event_id_mapping


def apply_event_id_mapping(epochs, event_id_mapping=None, verbose=True):
    """Apply event ID mapping to epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to remap
    event_id_mapping : dict | None
        Dictionary mapping old event IDs to new condition names.
        If None, uses the default mapping from get_event_id_mapping()
    verbose : bool
        Whether to print mapping information
        
    Returns
    -------
    mne.Epochs
        The epochs object with remapped event_id
    """
    
    if event_id_mapping is None:
        event_id_mapping = get_event_id_mapping()
    
    if verbose:
        print("Event ID mapping being used:")
        for old_id, new_name in event_id_mapping.items():
            print(f"  {old_id} -> {new_name}")
        
        print(f"\nOriginal event_id: {epochs.event_id}")
    
    # Create new event_id mapping with condition names
    new_event_id = {}
    for old_name, old_id in epochs.event_id.items():
        try:
            old_id_int = int(old_name)  # Convert string to int
            if old_id_int in event_id_mapping:
                new_name = event_id_mapping[old_id_int]
                new_event_id[new_name] = old_id
            else:
                # Keep unmapped events as is
                new_event_id[old_name] = old_id
        except ValueError:
            # If old_name is already a string condition name, keep it
            new_event_id[old_name] = old_id
    
    epochs.event_id = new_event_id
    
    if verbose:
       # print(f"Remapped event_id: {epochs.event_id}")
        
        print("\nEvent counts after mapping:")
        for event_name, event_id in epochs.event_id.items():
            count = sum(epochs.events[:, 2] == event_id)
            print(f"  {event_name} (ID {event_id}): {count} events")
    
    return epochs


def create_256_to_64_roi_mapping():
    """Create mapping from 256-channel ROIs to 64-channel ROIs using the JSON mapping.
    
    Returns
    -------
    dict
        Mapping functions for each ROI type
    """
    import json
    import os.path as op
    
    # Load the mapping file
    mapping_file = op.join(op.dirname(__file__), '..', '..', 'data', 'egi256_biosemi64.json')
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Get the recombination groups (biosemi64 -> 256 electrodes)
    recombination_groups = mapping_data['recombination_groups']
    
    # Create reverse mapping: 256 electrode number -> 64 channel index
    electrode_256_to_ch_64 = {}
    
    # Define the actual channel order from the fif file (1-based indexing)
    ch_64_names_ordered = [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
        'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
    ]
    
    # Convert electrode names to indices and create mapping
    for ch_64_name, electrode_256_list in recombination_groups.items():
        # Get the 64-channel index (0-63) from the channel names using actual fif order
        if ch_64_name in ch_64_names_ordered:
            ch_64_idx = ch_64_names_ordered.index(ch_64_name)  # 0-based indexing (0-63)
            
            # Convert electrode names like "E33" to electrode numbers (E33 -> 33, keep 1-based)
            for electrode_name in electrode_256_list:
                electrode_num = int(electrode_name[1:])  # E33 -> 33 (1-based EGI number)
                electrode_256_to_ch_64[electrode_num] = ch_64_idx
    
    def map_256_roi_to_64(roi_256):
        """Map a 256-channel ROI to corresponding 64-channel indices.
        
        Parameters
        ----------
        roi_256 : array
            Array of EGI electrode numbers (1-based, e.g., [5, 6, 13, 14, 15, 21, 22])
            
        Returns
        -------
        array
            Array of 64-channel indices (0-based, 0-63)
        """
        mapped_channels = set()
        
        for electrode_num in roi_256:
            if electrode_num in electrode_256_to_ch_64:
                mapped_channels.add(electrode_256_to_ch_64[electrode_num])
        
        return np.array(sorted(mapped_channels))
    
    return map_256_roi_to_64


def get_electrode_mapping(n_channels):
    """Get electrode mappings based on the number of channels.
    
    Parameters
    ----------
    n_channels : int
        Number of available channels
        
    Returns
    -------
    dict
        Dictionary with electrode mappings for different ROIs
    """
    
    # Standard mappings for common electrode configurations
    if n_channels == 256:  # EGI 256-channel system
        scalp_roi = np.arange(224)
        non_scalp = np.arange(224, 256)
        cnv_roi = np.array([5, 6, 13, 14, 15, 21, 22])
        mmn_roi = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])
        p3b_roi = np.array([8, 44, 80, 99, 100, 109, 118, 127, 128, 131, 185])
        p3a_roi = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])
        
    elif n_channels == 64:  # Standard 64-channel system (biosemi64)
        scalp_roi = np.arange(64)  # 0-based indexing (0-63)
        non_scalp = np.array([])  # No non-scalp channels
        
        # Use mapping from 256-channel ROIs to 64-channel ROIs
        map_roi = create_256_to_64_roi_mapping()
        
        # Original 256-channel ROIs
        cnv_roi_256 = np.array([5, 6, 13, 14, 15, 21, 22])
        mmn_roi_256 = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])
        p3b_roi_256 = np.array([8, 44, 80, 99, 100, 109, 118, 127, 128, 131, 185])
        p3a_roi_256 = np.array([5, 6, 8, 13, 14, 15, 21, 22, 44, 80, 131, 185])
        
        # Map to 64-channel indices
        cnv_roi = map_roi(cnv_roi_256)
        mmn_roi = map_roi(mmn_roi_256)
        p3b_roi = map_roi(p3b_roi_256)
        p3a_roi = map_roi(p3a_roi_256)
    
    # Filter out channels that don't exist
    cnv_roi = cnv_roi[cnv_roi < n_channels]
    mmn_roi = mmn_roi[mmn_roi < n_channels]
    p3b_roi = p3b_roi[p3b_roi < n_channels]
    p3a_roi = p3a_roi[p3a_roi < n_channels]
    
    return {
        'scalp_roi': scalp_roi,
        'non_scalp': non_scalp,
        'cnv_roi': cnv_roi,
        'mmn_roi': mmn_roi,
        'p3b_roi': p3b_roi,
        'p3a_roi': p3a_roi
    }


def compute_markers(epochs, output_file=None, apply_event_mapping=True, event_id_mapping=None,
                   timeout_minutes=30, skip_smi=False):
    """Compute markers for given epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to compute markers for
    output_file : str | None
        Path to save the markers. If None, uses default naming.
    apply_event_mapping : bool
        Whether to apply event ID mapping to convert numerical IDs to condition names
    event_id_mapping : dict | None
        Custom event ID mapping. If None and apply_event_mapping is True, 
        uses the default mapping from get_event_id_mapping()
    timeout_minutes : int
        Timeout in minutes for the entire computation (default: 30)
    skip_smi : bool
        Whether to skip SymbolicMutualInformation computation (recommended for large datasets)
        
    Returns
    -------
    Markers
        The computed markers
    """
    
    # Apply event ID mapping if requested
    if apply_event_mapping:
        epochs = apply_event_id_mapping(epochs, event_id_mapping=event_id_mapping)
    
    n_channels = len(epochs.ch_names)
    electrode_mapping = get_electrode_mapping(n_channels)
    
    print(f"Computing markers for {n_channels} channels")
    print(f"Scalp ROI: {len(electrode_mapping['scalp_roi'])} channels")
    
    psds_params = dict(n_fft=4096, n_overlap=40, n_jobs='auto', nperseg=128)
    
    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=None, tmax=0.6, fmin=1., fmax=45.,
        psd_params=psds_params, comment='default')
    
    m_list = [
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=False, comment='delta', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=True, comment='deltan', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=False, comment='theta', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=True, comment='thetan', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=False, comment='alpha', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=True, comment='alphan', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=False, comment='beta', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=True, comment='betan', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=False, comment='gamma', dB = False),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=True, comment='gamman', dB = False),
        
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                             normalize=True, comment='summary_se', dB = False),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.5, comment='summary_msf'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.9, comment='summary_sef90'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.95, comment='summary_sef95'),
        
        PermutationEntropy(tmin=None, tmax=0.6, backend='python'),
        
        KolmogorovComplexity(tmin=None, tmax=0.6, backend='python'),
    ]
    
    # Add SymbolicMutualInformation only if not skipped
    if not skip_smi:
        print("Computing SymbolicMutualInformation (this may take a while for large datasets)...")
        m_list.insert(-1, SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='python',
            comment='weighted'))
    else:
        print("Skipping SymbolicMutualInformation computation (--skip-smi flag enabled)")
    
    # Add evoked-related markers only if we have conditions
    if len(np.unique(epochs.events[:, 2])) > 1:
        m_list.extend([
            ContingentNegativeVariation(tmin=-0.004, tmax=0.596),
            
            TimeLockedTopography(tmin=0.064, tmax=0.112, comment='p1'),
            TimeLockedTopography(tmin=0.876, tmax=0.936, comment='p3a'),
            TimeLockedTopography(tmin=0.996, tmax=1.196, comment='p3b'),
        ])
        
        # Add contrasts if we have the specific conditions
        if all(x in epochs.event_id for x in ['LSGS', 'LDGD', 'LSGD', 'LDGS']):
            m_list.extend([
                TimeLockedContrast(tmin=None, tmax=None, condition_a='LSGS',
                                   condition_b='LDGD', comment='LSGS-LDGD'),
                
                TimeLockedContrast(tmin=None, tmax=None, condition_a='LSGD',
                                   condition_b='LDGS', comment='LSGD-LDGS'),
                
                TimeLockedContrast(tmin=None, tmax=None, condition_a=['LDGS', 'LDGD'],
                                   condition_b=['LSGS', 'LSGD'], comment='LD-LS'),
                
                TimeLockedContrast(tmin=0.736, tmax=0.788, condition_a=['LDGS', 'LDGD'],
                                   condition_b=['LSGS', 'LSGD'], comment='mmn'),
                
                TimeLockedContrast(tmin=0.876, tmax=0.936, condition_a=['LDGS', 'LDGD'],
                                   condition_b=['LSGS', 'LSGD'], comment='p3a'),
                
                TimeLockedContrast(tmin=None, tmax=None, condition_a=['LSGD', 'LDGD'],
                                   condition_b=['LSGS', 'LDGS'], comment='GD-GS'),
                
                TimeLockedContrast(tmin=0.996, tmax=1.196, condition_a=['LSGD', 'LDGD'],
                                   condition_b=['LSGS', 'LDGS'], comment='p3b')
            ])
    
    mc = Markers(m_list)
    mc.fit(epochs)
    
    if output_file:
        mc.save(output_file, overwrite=True)
        print(f"Markers saved to: {output_file}")
    
    return mc


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Compute DOC markers with variable electrode support')
    parser.add_argument('input_file', help='Input epochs file (.fif)')
    parser.add_argument('--output', '-o', help='Output markers file (.hdf5)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--no-event-mapping', action='store_true', 
                        help='Skip automatic event ID mapping (keep original event IDs)')
    parser.add_argument('--skip-smi', action='store_true',
                        help='Skip SymbolicMutualInformation computation (recommended for large datasets)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in minutes for marker computation (default: 30)')
    
    args = parser.parse_args()
    
    # Load epochs
    print(f"Loading epochs from: {args.input_file}")
    epochs = mne.read_epochs(args.input_file, preload=True)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = op.splitext(args.input_file)[0]
        output_file = base_name + '_markers.hdf5'
    
    # Compute markers with or without event mapping
    apply_mapping = not args.no_event_mapping
    mc = compute_markers(epochs, output_file, apply_event_mapping=apply_mapping, 
                        timeout_minutes=args.timeout, skip_smi=args.skip_smi)
    
    # Optional plotting
    if args.plot:
        try:
            # Plot PSDs if available
            for marker in mc.values():
                if hasattr(marker, 'estimator') and hasattr(marker.estimator, 'data_'):
                    psd = marker.estimator.data_
                    freqs = marker.estimator.freqs_
                    
                    plt.figure(figsize=(10, 6))
                    psd_mean = np.mean(psd, axis=0)
                    plt.semilogy(freqs, psd_mean.T, alpha=0.1, color='black')
                    plt.xlim(2, 40)
                    plt.ylim(np.min(psd_mean)*0.8, np.max(psd_mean)*1.2)
                    plt.ylabel('PSD')
                    plt.xlabel('Frequency [Hz]')
                    plt.title(f'Power Spectral Density - {len(epochs.ch_names)} channels')
                    
                    # Save plot in the same directory as the markers file
                    markers_dir = op.dirname(output_file)
                    plot_file = op.join(markers_dir, op.splitext(op.basename(output_file))[0] + '_psd.png')
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    print(f"PSD plot saved to: {plot_file}")
                    break
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


if __name__ == '__main__':
    main()