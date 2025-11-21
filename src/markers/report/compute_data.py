#!/usr/bin/env python3
"""
Next-ICM Report - PHASE 1: Compute All Data

This script computes all analysis data and saves it to pickle files.
After running this, you can generate plots instantly using generate_plots.py.

Usage:
    python compute_data.py [--skip-clustering] [--subject_id SUBJECT_ID] \
                           [--h5_file H5_FILE] [--fif_file FIF_FILE] \
                           [--output_dir DIR]
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

# Import computation modules directly
from report_modules.computations.diagnostic import compute_diagnostic_data
from report_modules.computations.erp import compute_erp_analysis_data
from report_modules.computations.cnv import compute_cnv_analysis_data
from report_modules.computations.spectral import compute_spectral_analysis_data
from report_modules.computations.connectivity import compute_connectivity_analysis_data
from report_modules.computations.information_theory import compute_information_theory_data
from report_modules.computations.predictions import compute_prediction_data
from report_modules.data_io import ReportDataLoader


def main():
    """Compute all analysis data and save to pickle files"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PHASE 1: Compute all Next-ICM analysis data"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip cluster permutation tests to speed up computation",
    )
    parser.add_argument(
        "--subject_id",
        default="AA048",
        help="Subject ID for analysis",
    )
    parser.add_argument(
        "--h5_file",
        default="./input/icm_complete_features.h5",
        help="Path to HDF5 file with features",
    )
    parser.add_argument(
        "--fif_file",
        default="./input/03_artifact_rejected_eeg.fif",
        help="Path to FIF file with epochs data",
    )
    parser.add_argument(
        "--output_dir",
        default="./tmp_computed_data",
        help="Output directory for computed data (pickle files)",
    )
    parser.add_argument(
        "--task",
        default="lg",
        choices=["lg", "rs"],
        help="Paradigm task: 'lg' (Local-Global) or 'rs' (Resting State). Determines which training dataset to use.",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Compute all data using computation modules
    try:
        logger.info(f"Computing data for subject {args.subject_id}")
        logger.info(f"Task paradigm: {args.task.upper()}")
        logger.info(f"Skip clustering: {args.skip_clustering}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Create output directory
        computed_data_path = Path(args.output_dir)
        computed_data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using computed data directory: {computed_data_path}")
        
        # Load data
        logger.info("="*60)
        logger.info("Loading data...")
        logger.info("="*60)
        
        hdf5_file = Path(args.h5_file)
        fif_file = Path(args.fif_file)
        
        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
        if not fif_file.exists():
            raise FileNotFoundError(f"FIF file not found: {fif_file}")
        
        logger.info(f"Loading HDF5 features from: {hdf5_file}")
        logger.info(f"Loading epochs from: {fif_file}")
        
        loader = ReportDataLoader(hdf5_file, fif_file, skip_preprocessing=True)
        report_data = loader.load_all_data()
        
        # Load epochs object
        epoch_info = report_data.get("epoch_info")
        if epoch_info is not None:
            epochs = epoch_info
            epochs.info["description"] = "egi/256"
            logger.info("Loaded epochs and set montage to 'egi/256'")
        else:
            logger.error("No epochs object found in data!")
            sys.exit(1)

        # ========== PHASE 1: COMPUTE ALL DATA (NO PLOTTING) ==========
        logger.info("="*60)
        logger.info("PHASE 1: Computing all analysis data...")
        logger.info("="*60)

        logger.info("Computing diagnostic data...")
        compute_diagnostic_data(epochs, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing ERP analysis...")
        compute_erp_analysis_data(epochs, skip_clustering=args.skip_clustering, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing CNV analysis...")
        compute_cnv_analysis_data(epochs, report_data, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing spectral analysis...")
        compute_spectral_analysis_data(epochs, report_data, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing connectivity analysis...")
        compute_connectivity_analysis_data(epochs, report_data, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing information theory analysis...")
        compute_information_theory_data(epochs, report_data, output_dir=str(computed_data_path))
        gc.collect()

        logger.info("Computing predictions...")
        try:
            compute_prediction_data(report_data, task=args.task, output_dir=str(computed_data_path))
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Skipping predictions: {e}")
            logger.warning("⚠️  Trained models not found - predictions will be skipped")
        gc.collect()

        logger.info("="*60)
        logger.info("✅ PHASE 1 COMPLETE: All data computed and saved to pkl files")
        logger.info(f"✅ Computed data saved to: {computed_data_path}")
        logger.info("="*60)
        
        logger.info("✅ Data computation completed successfully!")
        print(f"\n{'='*80}")
        print(f"SUCCESS: Computed data saved to {computed_data_path}")
        print("")
        print("Next step: Generate plots with:")
        print(f"  python generate_plots.py --subject_id {args.subject_id} \\")
        print(f"    --h5_file {args.h5_file} --fif_file {args.fif_file} \\")
        print(f"    --data_dir {args.output_dir}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute data: {e}")
        import traceback
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
