"""DOC-Forest with variable electrode support.

==================================================
Compute features from markers with dynamic ROI mapping
==================================================

This is an enhanced version of compute_doc_forest_features_from_markers.py
that can handle variable numbers of electrodes and automatically adapts
the electrode selections and ROI mappings based on the available channels.

For simplicity, we compute scalars using a trimmed mean (80%) across
epochs and the mean across channels, plus topographies.

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

import numpy as np
from scipy.stats import trim_mean
import os.path as op
import argparse
import mne

from nice import read_markers

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set_color_codes()


def trim_mean80(a, axis=0):  # noqa
    return trim_mean(a, proportiontocut=.1, axis=axis)


def entropy(a, axis=0):  # noqa
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])


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


def get_reduction_params(electrode_mapping):
    """Get reduction parameters adapted to the electrode configuration.
    
    Uses the exact same structure as the original NICE example but adapts
    the electrode selections to the available configuration.
    
    Parameters
    ----------
    electrode_mapping : dict
        Dictionary with electrode mappings
        
    Returns
    -------
    dict
        Reduction parameters adapted to the electrode configuration
    """
    
    scalp_roi = electrode_mapping['scalp_roi']
    cnv_roi = electrode_mapping['cnv_roi'] 
    mmn_roi = electrode_mapping['mmn_roi']
    p3b_roi = electrode_mapping['p3b_roi']
    p3a_roi = electrode_mapping['p3a_roi']
    
    # Set reduction functions (same as original)
    channels_fun = np.mean  # function to summarize channels
    epochs_fun = trim_mean80  # robust mean to summarize epochs
    
    # Use the exact same structure as the working example
    reduction_params = {}
    
    reduction_params['PowerSpectralDensity'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'frequency', 'function': np.sum}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensity/summary_se'] = {
        'reduction_func':
            [{'axis': 'frequency', 'function': entropy},
             {'axis': 'epochs', 'function': np.mean},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi}}

    reduction_params['PowerSpectralDensitySummary'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi}}

    reduction_params['PermutationEntropy'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi}}

    reduction_params['SymbolicMutualInformation'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels_y', 'function': np.median},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels_y': scalp_roi,
            'channels': scalp_roi}}

    reduction_params['KolmogorovComplexity'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi}}

    reduction_params['ContingentNegativeVariation'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun}],
        'picks': {
            'epochs': None,
            'channels': cnv_roi if len(cnv_roi) > 0 else scalp_roi}}

    reduction_params['TimeLockedTopography'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'times', 'function': np.mean}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'times', 'function': np.mean}],
        'picks': {
            'epochs': None,
            'channels': scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/mmn'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'times', 'function': np.mean}],
        'picks': {
            'epochs': None,
            'channels': mmn_roi if len(mmn_roi) > 0 else scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/p3a'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'times', 'function': np.mean}],
        'picks': {
            'epochs': None,
            'channels': p3a_roi if len(p3a_roi) > 0 else scalp_roi,
            'times': None}}

    reduction_params['TimeLockedContrast/p3b'] = {
        'reduction_func':
            [{'axis': 'epochs', 'function': epochs_fun},
             {'axis': 'channels', 'function': channels_fun},
             {'axis': 'times', 'function': np.mean}],
        'picks': {
            'epochs': None,
            'channels': p3b_roi if len(p3b_roi) > 0 else scalp_roi,
            'times': None}}
    
    return reduction_params


def create_montage_layout(n_channels):
    """Create a montage and layout for plotting based on channel count.
    
    Parameters
    ----------
    n_channels : int
        Number of channels
        
    Returns
    -------
    tuple
        montage, layout, pos
    """
    
    if n_channels == 256:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        ch_names = ['E{}'.format(i) for i in range(1, 257)]
    elif n_channels == 128:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        ch_names = ['E{}'.format(i) for i in range(1, 129)]
    elif n_channels == 64:
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:64]
    elif n_channels == 32:
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:32]
    else:
        # Create a generic montage
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:min(n_channels, len(montage.ch_names))]
        # Pad with generic names if needed
        while len(ch_names) < n_channels:
            ch_names.append(f'CH{len(ch_names) + 1}')
    
    info = mne.create_info(ch_names[:n_channels], 1, ch_types='eeg', montage=montage)
    layout = mne.channels.make_eeg_layout(info)
    pos = layout.pos[:, :2]
    
    return montage, layout, pos


def compute_features(markers_file, output_scalars=None, output_topos=None, 
                    n_channels=None, plot=True):
    """Compute features from markers with variable electrode support.
    
    Parameters
    ----------
    markers_file : str
        Path to the markers HDF5 file
    output_scalars : str | None
        Output file for scalar features (.npy)
    output_topos : str | None
        Output file for topographic features (.npy)
    n_channels : int | None
        Number of channels (if None, inferred from markers)
    plot : bool
        Whether to generate plots
        
    Returns
    -------
    tuple
        (scalars, topos) numpy arrays
    """
    
    print(f"Loading markers from: {markers_file}")
    fc = read_markers(markers_file)
    
    # Infer number of channels if not provided
    if n_channels is None:
        # Try to infer from the first marker's data shape
        for marker in fc.values():
            if hasattr(marker, 'data_') and marker.data_ is not None:
                if marker.data_.ndim >= 2:
                    n_channels = marker.data_.shape[1]  # Assuming (epochs, channels, ...)
                    break
        if n_channels is None:
            n_channels = 64  # Default fallback
    
    print(f"Processing {n_channels} channels")
    
    # Get electrode mapping and reduction parameters
    electrode_mapping = get_electrode_mapping(n_channels)
    reduction_params = get_reduction_params(electrode_mapping)
    
    print(f"Scalp ROI: {len(electrode_mapping['scalp_roi'])} channels")
    print(f"Available markers: {len(fc.keys())}")
    
    # Get available markers for debugging
    available_markers = list(fc.keys())
    print(f"Available markers: {len(available_markers)}")
    for i, marker in enumerate(available_markers):
        print(f"  {i+1:2d}. {marker}")
    print()
    
    # Filter reduction parameters to only include those with available markers
    print("Filtering reduction parameters to match available markers...")
    
    # Get available marker classes
    available_marker_classes = set()
    for marker_key in available_markers:
        # Extract the class from the marker key
        # "nice/marker/PowerSpectralDensity/delta" -> "PowerSpectralDensity"
        # "nice/marker/PowerSpectralDensity/summary_se" -> "PowerSpectralDensity/summary_se"
        parts = marker_key.split('/')
        if len(parts) >= 3:
            marker_class = parts[2]  # PowerSpectralDensity
            if len(parts) >= 4 and parts[3] == 'summary_se':
                marker_class += '/summary_se'  # PowerSpectralDensity/summary_se
            available_marker_classes.add(marker_class)
    
    print(f"Available marker classes: {sorted(available_marker_classes)}")
    
    # Filter reduction parameters
    filtered_reduction_params = {}
    for param_key, param_value in reduction_params.items():
        if param_key in available_marker_classes:
            filtered_reduction_params[param_key] = param_value
            print(f"  ✓ Using: {param_key}")
        else:
            print(f"  ✗ Skipping: {param_key} (no matching markers)")
    
    print(f"\nUsing {len(filtered_reduction_params)} reduction parameter sets")
    
    # Actually compute reductions 
    scalars = fc.reduce_to_scalar(filtered_reduction_params)
    topos = fc.reduce_to_topo(filtered_reduction_params)
    
    print(f'Computed {scalars.shape[0]} scalar markers')
    print(f'Computed {topos.shape[0]} topographic markers with {topos.shape[1]} channels')
    
    # Save outputs
    if output_scalars:
        np.save(output_scalars, scalars)
        print(f"Scalar features saved to: {output_scalars}")
    
    if output_topos:
        np.save(output_topos, topos)
        print(f"Topographic features saved to: {output_topos}")
    
    # Optional plotting
    if plot and topos.shape[0] > 0:
        try:
            plot_topographies(fc, topos, electrode_mapping, n_channels, 
                            op.splitext(markers_file)[0] + '_topos.png')
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return scalars, topos


def plot_topographies(fc, topos, electrode_mapping, n_channels, output_file):
    """Plot a selection of topographic markers.
    
    Parameters
    ----------
    fc : Markers
        The markers collection
    topos : array
        Topographic data
    electrode_mapping : dict
        Electrode mapping information
    n_channels : int
        Number of channels
    output_file : str
        Output file for the plot
    """
    
    # Select markers to plot (power spectral densities)
    to_plot = []
    marker_keys = list(fc.keys())
    
    for band in ['deltan', 'thetan', 'alphan', 'betan', 'gamman']:
        matching = [k for k in marker_keys if band in k]
        if matching:
            to_plot.append(matching[0])
    
    if len(to_plot) == 0:
        print("No suitable markers found for plotting")
        return
    
    # Limit to first 5 for plotting
    to_plot = to_plot[:5]
    idx = [marker_keys.index(x) for x in to_plot]
    names = [x.split('/')[-1] for x in to_plot]
    topos_to_plot = topos[idx]
    
    # Create montage and layout
    montage, layout, pos = create_montage_layout(n_channels)
    
    scalp_roi = electrode_mapping['scalp_roi']
    non_scalp = electrode_mapping['non_scalp']
    
    # Create plots
    n_axes = len(names)
    fig_kwargs = dict(figsize=(3 * n_axes, 4))
    fig, axes = plt.subplots(1, n_axes, **fig_kwargs)
    
    if n_axes == 1:
        axes = [axes]
    
    for ax, name, topo in zip(axes, names, topos_to_plot):
        # Ensure topo has the right length
        if len(topo) != n_channels:
            print(f"Warning: topo length {len(topo)} != n_channels {n_channels}")
            continue
            
        vmin = np.nanmin(topo[scalp_roi])
        vmax = np.nanmax(topo[scalp_roi])
        
        if len(non_scalp) > 0:
            topo[non_scalp] = vmin
        
        nan_idx = np.isnan(topo)
        
        if np.all(nan_idx):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue
        
        try:
            im, _ = mne.viz.topomap.plot_topomap(
                topo[~nan_idx], pos[~nan_idx], vmin=vmin, vmax=vmax, axes=ax,
                cmap='viridis', image_interp='nearest', sensors=False, contours=0)
            
            ax.set_title(name)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=(vmin, vmax))
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            print(f"Could not plot {name}: {e}")
            ax.text(0.5, 0.5, f'Error: {name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Topography plots saved to: {output_file}")
    plt.close()


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Compute DOC features with variable electrode support')
    parser.add_argument('markers_file', help='Input markers file (.hdf5)')
    parser.add_argument('--output-scalars', '-s', help='Output scalars file (.npy)')
    parser.add_argument('--output-topos', '-t', help='Output topographies file (.npy)')
    parser.add_argument('--n-channels', '-n', type=int, help='Number of channels')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Determine output files
    base_name = op.splitext(args.markers_file)[0]
    
    scalars_file = args.output_scalars or base_name + '_scalars.npy'
    topos_file = args.output_topos or base_name + '_topos.npy'
    
    # Compute features
    scalars, topos = compute_features(
        args.markers_file, 
        output_scalars=scalars_file,
        output_topos=topos_file,
        n_channels=args.n_channels,
        plot=args.plot
    )


if __name__ == '__main__':
    main()