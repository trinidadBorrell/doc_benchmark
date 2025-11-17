"""
ERP analysis computation functions.

This module contains all computation logic for ERP analysis,
orchestrating calls to evoked computation functions.
"""

import gc
import logging
from pathlib import Path
from mne import concatenate_epochs

logger = logging.getLogger(__name__)


def compute_erp_analysis_data(epochs, skip_clustering=False, output_dir="./tmp_computed_data"):
    """
    Compute all ERP data and save to pkl files (NO PLOTTING).
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object
    skip_clustering : bool
        Skip cluster permutation tests to speed up computation
    output_dir : str or Path
        Directory to save computed data
        
    Returns
    -------
    dict
        Dictionary with paths to saved data files
    """
    from ..viz import trim_mean80
    from .evokeds import compute_gfp, compute_cluster_test
    from .contrast import compute_contrast

    logger.info("Computing ERP analysis data...")
    
    output_paths = {}
    
    # Local and Global effects using real condition data
    local_conditions = [["LDGS", "LDGD"], ["LSGS", "LSGD"]]
    global_conditions = [["LDGD", "LSGD"], ["LSGS", "LDGS"]]
    time_shift = -0.6
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}

    # ========== LOCAL EFFECT COMPUTATIONS ==========
    # 1. Local GFP
    logger.info("Computing Local Effect GFP data...")
    local_gfp_path = Path(output_dir) / "local_effect_gfp.pkl"
    compute_gfp(
        epochs,
        conditions=local_conditions,
        shift_time=time_shift,
        event_times=event_times,
        output_path=local_gfp_path,
    )
    output_paths['local_gfp'] = local_gfp_path

    # 2. Local Contrast
    logger.info("Computing Local Effect contrast data...")
    local_contrast_path = Path(output_dir) / "local_effect_contrast.pkl"
    compute_contrast(
        epochs,
        conditions=local_conditions,
        method=trim_mean80,
        output_path=local_contrast_path,
    )
    output_paths['local_contrast'] = local_contrast_path
    
    # 3. Local Cluster Test (optional)
    if not skip_clustering:
        logger.info("Computing Local Effect cluster test...")
        local_deviant_epochs = concatenate_epochs(
            [epochs["LDGS"], epochs["LDGD"]]
        )
        local_standard_epochs = concatenate_epochs(
            [epochs["LSGS"], epochs["LSGD"]]
        )
        combined_epochs = concatenate_epochs(
            [local_deviant_epochs, local_standard_epochs]
        )
        combined_epochs.events[: len(local_deviant_epochs), 2] = 1
        combined_epochs.events[len(local_deviant_epochs) :, 2] = 2
        combined_epochs.event_id = {
            "LocalDeviant": 1,
            "LocalStandard": 2,
        }
        if combined_epochs.info.get("description") is None:
            combined_epochs.info["description"] = "egi/256"
        
        local_cluster_path = Path(output_dir) / "local_cluster_test.pkl"
        compute_cluster_test(
            combined_epochs,
            ["LocalDeviant", "LocalStandard"],
            ["Local Deviant", "Local Standard"],
            p_threshold=1e-2,
            f_threshold=10,
            shift_time=-0.6,
            event_times=event_times,
            output_path=local_cluster_path,
        )
        output_paths['local_cluster'] = local_cluster_path

    # ========== GLOBAL EFFECT COMPUTATIONS ==========
    # 4. Global GFP
    logger.info("Computing Global Effect GFP data...")
    global_gfp_path = Path(output_dir) / "global_effect_gfp.pkl"
    compute_gfp(
        epochs,
        conditions=global_conditions,
        shift_time=time_shift,
        event_times=event_times,
        output_path=global_gfp_path,
    )
    output_paths['global_gfp'] = global_gfp_path

    # 5. Global Contrast
    logger.info("Computing Global Effect contrast data...")
    global_contrast_path = Path(output_dir) / "global_effect_contrast.pkl"
    compute_contrast(
        epochs,
        conditions=global_conditions,
        method=trim_mean80,
        output_path=global_contrast_path,
    )
    output_paths['global_contrast'] = global_contrast_path
    
    # 6. Global Cluster Test (optional)
    if not skip_clustering:
        logger.info("Computing Global Effect cluster test...")
        global_standard_epochs = concatenate_epochs(
            [epochs["LSGS"], epochs["LDGS"]]
        )
        global_deviant_epochs = concatenate_epochs(
            [epochs["LSGD"], epochs["LDGD"]]
        )
        global_combined_epochs = concatenate_epochs(
            [global_standard_epochs, global_deviant_epochs]
        )
        global_combined_epochs.event_id = {
            "GlobalStandard": 1,
            "GlobalDeviant": 2,
        }
        global_combined_epochs.events[: len(global_standard_epochs), 2] = 1
        global_combined_epochs.events[len(global_standard_epochs) :, 2] = 2
        
        global_cluster_path = Path(output_dir) / "global_cluster_test.pkl"
        compute_cluster_test(
            global_combined_epochs,
            ["GlobalStandard", "GlobalDeviant"],
            ["Global Standard", "Global Deviant"],
            p_threshold=1e-2,
            f_threshold=10,
            shift_time=-0.6,
            event_times=event_times,
            output_path=global_cluster_path,
        )
        output_paths['global_cluster'] = global_cluster_path

    # ========== ROI ANALYSIS COMPUTATIONS ==========
    rois = ["Fz", "Cz", "Pz"]
    for roi_name in rois:
        # Local ROI
        logger.info(f"Computing Local Effect contrast for ROI: {roi_name}")
        local_roi_path = Path(output_dir) / f"local_effect_contrast_{roi_name}.pkl"
        compute_contrast(
            epochs,
            local_conditions,
            method=trim_mean80,
            roi_name=roi_name,
            output_path=local_roi_path,
        )
        output_paths[f'local_roi_{roi_name}'] = local_roi_path
        
        # Global ROI
        logger.info(f"Computing Global Effect contrast for ROI: {roi_name}")
        global_roi_path = Path(output_dir) / f"global_effect_contrast_{roi_name}.pkl"
        compute_contrast(
            epochs,
            global_conditions,
            method=trim_mean80,
            roi_name=roi_name,
            output_path=global_roi_path,
        )
        output_paths[f'global_roi_{roi_name}'] = global_roi_path
    
    logger.info("✅ ERP computations complete. All data saved to pkl files.")
    gc.collect()
    
    return output_paths
