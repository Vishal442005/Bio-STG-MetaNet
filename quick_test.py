"""
Quick test to see what's happening with data loading
"""

import os
import glob

print("Checking dataset structure...")

# Count files
train_gz = len(glob.glob('SMNI_CMI_TRAIN/**/*.gz', recursive=True))
test_gz = len(glob.glob('SMNI_CMI_TEST/**/*.gz', recursive=True))

print(f"SMNI_CMI_TRAIN: {train_gz} .gz files")
print(f"SMNI_CMI_TEST: {test_gz} .gz files")
print(f"Total: {train_gz + test_gz} .gz files")

# Check class distribution
train_dirs = [d for d in os.listdir('SMNI_CMI_TRAIN') if os.path.isdir(os.path.join('SMNI_CMI_TRAIN', d))]
co2a_dirs = [d for d in train_dirs if 'co2a' in d.lower()]
co2c_dirs = [d for d in train_dirs if 'co2c' in d.lower()]

print(f"\nTraining subjects: {len(train_dirs)}")
print(f"  Alcoholic (co2a): {len(co2a_dirs)}")
print(f"  Control (co2c): {len(co2c_dirs)}")

# Count files per class
co2a_files = len(glob.glob('SMNI_CMI_TRAIN/co2a*/**/*.gz', recursive=True))
co2c_files = len(glob.glob('SMNI_CMI_TRAIN/co2c*/**/*.gz', recursive=True))

print(f"\nTraining files by class:")
print(f"  Alcoholic: {co2a_files}")
print(f"  Control: {co2c_files}")

print("\nDataset is quite large - loading 600+ files will take time.")