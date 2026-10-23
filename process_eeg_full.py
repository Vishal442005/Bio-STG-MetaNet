"""
Process eeg_full dataset to extract all 122 subjects
Converts .tar.gz files to .gz format compatible with existing parser
Creates proper dataset structure for subject-independent evaluation
"""

import os
import tarfile
import gzip
import shutil
from pathlib import Path

# Configuration
EEG_FULL_DIR = "eeg_full"
OUTPUT_DIR = "UCI_FULL_PROCESSED"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each .tar.gz file
print(f"Processing files from {EEG_FULL_DIR}...")
print(f"Output to: {OUTPUT_DIR}")

tar_files = list(Path(EEG_FULL_DIR).glob("*.tar.gz"))
print(f"Found {len(tar_files)} .tar.gz files")

for tar_path in tar_files:
    # Extract subject ID from filename
    # Format: co2a0000364.tar.gz -> co2a0000364
    subject_id = tar_path.stem  # Removes .tar.gz
    
    # Create subject directory
    subject_dir = Path(OUTPUT_DIR) / subject_id
    subject_dir.mkdir(exist_ok=True)
    
    try:
        # Extract tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get all members
            members = tar.getmembers()
            
            for member in members:
                if member.isfile() and member.name.endswith('.gz'):
                    # Extract file content
                    f = tar.extractfile(member)
                    if f:
                        # Read content
                        content = f.read()
                        
                        # Create output path maintaining subject folder structure
                        # member.name is like "co2a0000364/co2a0000364.rd.010.gz"
                        parts = member.name.split('/')
                        if len(parts) == 2:
                            output_path = subject_dir / parts[1]
                            with open(output_path, 'wb') as out_f:
                                out_f.write(content)
                        f.close()
    except Exception as e:
        print(f"Error processing {tar_path}: {e}")

print("\n" + "="*70)
print("Dataset Processing Complete")
print("="*70)

# Verify results
output_subjects = [p.name for p in Path(OUTPUT_DIR).iterdir() if p.is_dir()]
alc_subjects = [s for s in output_subjects if 'co2a' in s.lower() or 'co3a' in s.lower()]
ctrl_subjects = [s for s in output_subjects if 'co2c' in s.lower()]

print(f"Total subjects processed: {len(output_subjects)}")
print(f"Alcoholic subjects: {len(alc_subjects)}")
print(f"Control subjects: {len(ctrl_subjects)}")

# Count total files
total_files = 0
for subj in output_subjects:
    total_files += len(list((Path(OUTPUT_DIR) / subj).glob("*.gz")))

print(f"Total .gz files: {total_files}")
print(f"\nDataset ready at: {OUTPUT_DIR}")
print("="*70)
