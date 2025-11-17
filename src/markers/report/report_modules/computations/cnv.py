"""
CNV (Contingent Negative Variation) computation functions.

This module contains all computation logic for CNV analysis.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_cnv_analysis_data(epochs, report_data, output_dir="./tmp_computed_data"):
    """
    Compute CNV analysis data and save to pkl (NO PLOTTING).
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object
    report_data : dict
        Report data dictionary containing CNV data
    output_dir : str or Path
        Directory to save computed data
        
    Returns
    -------
    Path or None
        Path to saved CNV data or None if computation failed
    """
    from ..viz import trim_mean80
    from .evokeds import compute_cnv_analysis
    from ..data_io import MarkerDataAdapter
    
    cnv_data = report_data.get("cnv_data")

    if cnv_data is None:
        logger.warning("No CNV data found")
        return None

    try:
        cnv_slopes_trials = cnv_data.get("cnv_slopes_trials")
        if cnv_slopes_trials is None:
            logger.warning("No CNV slopes trials data found")
            return None

        logger.info(
            f"Found CNV slopes trials data with shape: {cnv_slopes_trials.shape}"
        )

        # CNV data now has full 256 channels - no filtering needed
        cnv_slopes_eeg = cnv_slopes_trials

        # Get intercepts data too
        cnv_intercepts_trials = cnv_data.get("cnv_intercepts_trials")
        if cnv_intercepts_trials is None:
            logger.warning("No CNV intercepts trials data found")
            return None

        cnv_intercepts_eeg = cnv_intercepts_trials

        # CNV has full channel set matching epochs.info
        cnv_info = epochs.info.copy()
        
        # Create a marker adapter for CNV
        cnv_marker = MarkerDataAdapter(
            data=cnv_slopes_eeg,
            intercepts=cnv_intercepts_eeg,
            ch_info=cnv_info,
            name="CNV",
        )
        
        # Use egi/256 for ROI definitions (valid_roi filtering handles subset)
        equipment_config = 'egi/256'
        
        # Compute all CNV analysis data and save to pickle
        output_path = Path(output_dir) / "cnv_computed_data.pkl"
        compute_cnv_analysis(
            cnv=cnv_marker,
            epochs=epochs,
            outlines=equipment_config,
            reduction_func=trim_mean80,
            output_path=output_path,
        )
        logger.info(f"✅ CNV computation completed. Data saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to compute CNV data: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None
