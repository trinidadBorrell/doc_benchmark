#!/usr/bin/env python3
"""
Test script for compute_markers_with_junifer.py
Runs marker computation with detailed debugging.

This tests the MARKERS phase (HDF5 generation) of the pipeline.
"""

import sys
import os
import subprocess
import tempfile
import re
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration - EDIT THESE PATHS
RECON_DATA_PATH = "/data/project/eeg_foundation/data/LaBram/results_DoC/recon_data_inference"
ORIGINAL_DATA_PATH = "/data/project/eeg_foundation/data/data_250Hz_EGI256/nice_epochs_from_cohen_2/nice_epochs/nice_epochs2"
OUTPUT_DIR = "/data/project/eeg_foundation/src/doc_benchmark/results/LaBram/doc_patients/test_markers"
TASK = "lg"

# Path to the markers source
MARKERS_SRC = Path(__file__).parent.parent / "src" / "markers"
TEMPLATE_YAML = MARKERS_SRC / "input" / "icm_complete_individual_markers_local_global.yaml"

def print_header(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)

def print_section(msg):
    print("\n" + "-" * 50)
    print(msg)
    print("-" * 50)

def find_test_files():
    """Find a few test files to process."""
    recon_path = Path(RECON_DATA_PATH)
    orig_path = Path(ORIGINAL_DATA_PATH)
    
    print_header("STEP 1: Finding test files")
    
    # Find reconstructed files
    print(f"\nSearching for reconstructed files in: {recon_path}")
    recon_files = list(recon_path.glob("**/sub-*_ses-*_vqnsp_reconstructed_epo.fif"))
    print(f"Found {len(recon_files)} reconstructed files")
    
    # Find original files
    print(f"\nSearching for original files in: {orig_path}")
    orig_files = list(orig_path.glob("**/sub-*/ses-*/eeg/*_epo.fif"))
    print(f"Found {len(orig_files)} original files")
    
    # Return first few of each
    test_files = {
        'recon': recon_files[:2] if recon_files else [],
        'original': orig_files[:2] if orig_files else []
    }
    
    print("\nTest files selected:")
    print("  Reconstructed:")
    for f in test_files['recon']:
        print(f"    {f}")
    print("  Original:")
    for f in test_files['original']:
        print(f"    {f}")
    
    return test_files

def check_template_yaml():
    """Check if template YAML exists and show its contents."""
    print_header("STEP 2: Checking template YAML")
    
    print(f"\nTemplate YAML path: {TEMPLATE_YAML}")
    
    if not TEMPLATE_YAML.exists():
        print(f"ERROR: Template YAML not found!")
        
        # List available YAML files
        yaml_dir = MARKERS_SRC / "input"
        if yaml_dir.exists():
            print(f"\nAvailable YAML files in {yaml_dir}:")
            for f in yaml_dir.glob("*.yaml"):
                print(f"  {f.name}")
        return False
    
    print("Template YAML exists!")
    
    # Show contents
    print("\nTemplate YAML contents:")
    print("-" * 40)
    with open(TEMPLATE_YAML, 'r') as f:
        content = f.read()
        print(content[:2000])  # First 2000 chars
        if len(content) > 2000:
            print("... (truncated)")
    print("-" * 40)
    
    return True

def create_subject_yaml(fif_file, output_h5):
    """Create a subject-specific YAML configuration."""
    print_section(f"Creating YAML for: {fif_file.name}")
    
    # Read the template YAML as text
    with open(TEMPLATE_YAML, 'r') as f:
        yaml_text = f.read()
    
    # Get the paths we need
    fif_path = Path(fif_file).absolute()
    datadir = str(fif_path.parent)
    pattern = fif_path.name
    
    print(f"  FIF path: {fif_path}")
    print(f"  Data dir: {datadir}")
    print(f"  Pattern: {pattern}")
    print(f"  Output H5: {output_h5}")
    
    # Do simple string replacements
    yaml_text = yaml_text.replace(
        'uri: icm_complete_features.h5',
        f'uri: {output_h5}'
    )
    
    # Update the datadir line
    yaml_text = re.sub(
        r'datadir: "[^"]*"',
        f'datadir: "{datadir}"',
        yaml_text
    )
    
    # Update the pattern line
    yaml_text = re.sub(
        r'pattern: "[^"]*\.fif"',
        f'pattern: "{pattern}"',
        yaml_text
    )
    
    # Create temporary YAML file
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_yaml.write(yaml_text)
    temp_yaml.close()
    
    print(f"  Created temp YAML: {temp_yaml.name}")
    
    # Show the modified YAML
    print("\n  Modified YAML (first 1000 chars):")
    print("  " + "-" * 40)
    lines = yaml_text[:1000].split('\n')
    for line in lines:
        print(f"  {line}")
    print("  " + "-" * 40)
    
    return temp_yaml.name

def run_junifer(yaml_file):
    """Run junifer with the specified YAML configuration."""
    print_section(f"Running junifer")
    print(f"  YAML file: {yaml_file}")
    
    try:
        # Check if junifer is available
        result = subprocess.run(
            ["which", "junifer"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("ERROR: junifer not found in PATH!")
            return False
        print(f"  junifer location: {result.stdout.strip()}")
        
        # Run junifer
        print("\n  Running: junifer run " + yaml_file)
        print("  " + "-" * 40)
        
        process = subprocess.Popen(
            ["junifer", "run", yaml_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        output_lines = []
        for line in process.stdout:
            print(f"  {line.rstrip()}")
            output_lines.append(line)
        
        process.wait()
        print("  " + "-" * 40)
        print(f"  Return code: {process.returncode}")
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"ERROR running junifer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_file(fif_file, file_type):
    """Test marker computation for a single file."""
    print_header(f"Testing: {fif_file.name} ({file_type})")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR) / file_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract subject/session info
    name = fif_file.name
    if "_vqnsp_reconstructed_epo.fif" in name:
        # Reconstructed file: sub-{id}_ses-{num}_vqnsp_reconstructed_epo.fif
        parts = name.split("_")
        subject_id = parts[0].replace("sub-", "")
        session_id = parts[1].replace("ses-", "")
    else:
        # Original file: sub-{id}_ses-{num}_task-lg_acq-01_epo.fif
        parts = name.split("_")
        subject_id = parts[0].replace("sub-", "")
        session_id = parts[1].replace("ses-", "")
    
    print(f"  Subject: {subject_id}")
    print(f"  Session: {session_id}")
    
    # Output H5 file
    h5_file = output_dir / f"sub-{subject_id}_ses-{session_id}_{file_type}.h5"
    print(f"  Output H5: {h5_file}")
    
    # Create YAML
    temp_yaml = create_subject_yaml(fif_file, str(h5_file))
    
    try:
        # Run junifer
        success = run_junifer(temp_yaml)
        
        if success:
            print(f"\n✓ Successfully created H5 file")
            if h5_file.exists():
                print(f"  File size: {h5_file.stat().st_size / 1024:.1f} KB")
                
                # Try to read the H5 file
                try:
                    import h5py
                    with h5py.File(h5_file, 'r') as f:
                        print(f"  H5 keys: {list(f.keys())}")
                        for key in list(f.keys())[:5]:
                            print(f"    {key}: {f[key]}")
                except Exception as e:
                    print(f"  Could not read H5: {e}")
            else:
                print(f"  WARNING: H5 file not found at expected location!")
        else:
            print(f"\n✗ Failed to create H5 file")
        
        return success
        
    finally:
        # Cleanup temp YAML
        if Path(temp_yaml).exists():
            Path(temp_yaml).unlink()
            print(f"\n  Cleaned up temp YAML")

def main():
    print_header("TEST: COMPUTE MARKERS (HDF5)")
    print(f"Reconstructed data: {RECON_DATA_PATH}")
    print(f"Original data: {ORIGINAL_DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Template YAML: {TEMPLATE_YAML}")
    
    # Check paths exist
    if not Path(RECON_DATA_PATH).exists():
        print(f"ERROR: Reconstructed data path does not exist: {RECON_DATA_PATH}")
        return
    
    # Check template YAML
    if not check_template_yaml():
        return
    
    # Find test files
    test_files = find_test_files()
    
    if not test_files['recon'] and not test_files['original']:
        print("\nERROR: No test files found!")
        return
    
    # Test one reconstructed file
    print_header("STEP 3: Testing marker computation")
    
    results = []
    
    if test_files['recon']:
        print("\nTesting first reconstructed file...")
        success = test_single_file(test_files['recon'][0], 'recon')
        results.append(('recon', success))
    
    if test_files['original']:
        print("\nTesting first original file...")
        success = test_single_file(test_files['original'][0], 'original')
        results.append(('original', success))
    
    # Summary
    print_header("SUMMARY")
    for file_type, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {file_type}: {status}")

if __name__ == "__main__":
    main()
