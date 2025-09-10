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
import argparse

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
        30: 'LSGS',    # Local Standard Global Standard (highest count)
        40: 'LSGD',    # Local Standard Global Deviant
        50: 'LDGS',    # Local Deviant Global Standard
        60: 'LDGD'     # Local Deviant Global Deviant
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
        print(f"Remapped event_id: {epochs.event_id}")
        
        print("\nEvent counts after mapping:")
        for event_name, event_id in epochs.event_id.items():
            count = sum(epochs.events[:, 2] == event_id)
            print(f"  {event_name} (ID {event_id}): {count} events")
    
    return epochs


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
        
    elif n_channels == 128:  # EGI 128-channel system
        scalp_roi = np.arange(120)
        non_scalp = np.arange(120, 128)
        # Scale down the ROIs proportionally
        cnv_roi = np.array([2, 3, 6, 7, 8, 10, 11])
        mmn_roi = np.array([2, 3, 4, 6, 7, 8, 10, 11, 22, 40, 65, 92])
        p3b_roi = np.array([4, 22, 40, 49, 50, 54, 59, 63, 64, 65, 92])
        p3a_roi = np.array([2, 3, 4, 6, 7, 8, 10, 11, 22, 40, 65, 92])
        
    elif n_channels == 64:  # Standard 64-channel system
        scalp_roi = np.arange(64)
        non_scalp = np.array([])  # No non-scalp channels
        # Central and frontal electrodes for CNV
        cnv_roi = np.array([1, 2, 3, 5, 6])
        # Frontocentral electrodes for MMN
        mmn_roi = np.array([1, 2, 3, 5, 6, 11, 20, 32, 46])
        # Parietal electrodes for P3b
        p3b_roi = np.array([2, 11, 20, 24, 25, 27, 29, 31, 32, 46])
        # Frontocentral electrodes for P3a
        p3a_roi = np.array([1, 2, 3, 5, 6, 11, 20, 32, 46])
        
    elif n_channels == 32:  # Standard 32-channel system
        scalp_roi = np.arange(32)
        non_scalp = np.array([])
        # Simplified ROIs for 32 channels
        cnv_roi = np.array([0, 1, 2, 3])
        mmn_roi = np.array([0, 1, 2, 3, 5, 10, 16])
        p3b_roi = np.array([1, 5, 10, 12, 13, 15, 16])
        p3a_roi = np.array([0, 1, 2, 3, 5, 10, 16])
        
    else:
        # Default: use all channels as scalp ROI and create basic ROIs
        scalp_roi = np.arange(n_channels)
        non_scalp = np.array([])
        # Use first few channels for all ROIs as fallback
        n_roi = min(7, n_channels)
        cnv_roi = np.arange(n_roi)
        mmn_roi = np.arange(min(12, n_channels))
        p3b_roi = np.arange(min(11, n_channels))
        p3a_roi = np.arange(min(12, n_channels))
    
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


def compute_markers(epochs, output_file=None, apply_event_mapping=True, event_id_mapping=None):
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
                             normalize=False, comment='delta'),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=True, comment='deltan'),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=False, comment='theta'),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=True, comment='thetan'),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=False, comment='alpha'),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=True, comment='alphan'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=False, comment='beta'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=True, comment='betan'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=False, comment='gamma'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=True, comment='gamman'),
        
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                             normalize=True, comment='summary_se'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.5, comment='summary_msf'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.9, comment='summary_sef90'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.95, comment='summary_sef95'),
        
        PermutationEntropy(tmin=None, tmax=0.6, backend='python'),
        
        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend='python',
            comment='weighted'),
        
        KolmogorovComplexity(tmin=None, tmax=0.6, backend='python'),
    ]
    
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
    mc = compute_markers(epochs, output_file, apply_event_mapping=apply_mapping)
    
    # Optional plotting
    if args.plot:
        try:
            # Plot PSDs if available
            for marker in mc.values():
                if hasattr(marker, 'estimator') and hasattr(marker.estimator, 'data_'):
                    psd = marker.estimator.data_
                    freqs = marker.estimator.freqs_
                    
                    plt.figure(figsize=(10, 6))
                    plt.semilogy(freqs, np.mean(psd, axis=0).T, alpha=0.1, color='black')
                    plt.xlim(2, 40)
                    plt.ylim(1e-14, 1e-10)
                    plt.ylabel('log(psd)')
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