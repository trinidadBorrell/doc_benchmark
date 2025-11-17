"""
Diagnostic computation functions.

This module contains all computation logic for diagnostic analysis.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_diagnostic_data(epochs, output_dir="./tmp_computed_data"):
    """
    Compute diagnostic data (GFP for diagnostic plot).
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object
    output_dir : str or Path
        Directory to save computed data
        
    Returns
    -------
    dict
        Dictionary with paths to saved data files
    """
    from .evokeds import compute_gfp
    
    logger.info("Computing diagnostic data...")
    
    output_paths = {}
    
    # Compute GFP for diagnostic plot
    logger.info("Computing GFP for diagnostic plot...")
    gfp_path = Path(output_dir) / "diagnostic_gfp.pkl"
    event_times = {0: "I", 150: "II", 300: "III", 450: "IV", 600: "V"}
    
    compute_gfp(
        epochs,
        conditions=["LSGS", "LSGD", "LDGS", "LDGD"],
        shift_time=-0.6,
        event_times=event_times,
        output_path=gfp_path,
    )
    
    output_paths['gfp'] = gfp_path
    logger.info("✅ Diagnostic data computed")
    
    return output_paths
