"""
Debug script to test data loading and identify issues
"""

import os
import glob
import numpy as np
from improved_data_loader import EEGDataLoader

print("="*60)
print("DEBUGGING DATA LOADING")
print("="*60)

# Check dataset sizes
print("\n1. Checking dataset sizes...")
print(f"SMNI_CMI_TRAIN directories: {len([d for d in os.listdir('SMNI_CMI_TRAIN') if os.path.isdir(os.path.join('SMNI_CMI_TRAIN', d))])}")
print(f"SMNI_CMI_TEST directories: {len([d for d in os.listdir('SMNI_CMI_TEST') if os.path.isdir(os.path.join('SMNI_CMI_TEST', d))])}")
print(f"Total .gz files in SMNI_CMI_TRAIN: {len(glob.glob('SMNI_CMI_TRAIN/**/*.gz', recursive=True))}")
print(f"Total .gz files in SMNI_CMI_TEST: {len(glob.glob('SMNI_CMI_TEST/**/*.gz', recursive=True))}")

# Test loading with small sample
print("\n2. Testing data loading with small sample...")
loader = EEGDataLoader()

# Load just 20 files total
X, y = loader.load_kaggle_dataset(max_files=20)
print(f"Loaded shape: {X.shape}, Labels: {y.shape}")
print(f"Class distribution: Alcoholic={np.sum(y==1)}, Control={np.sum(y==0)}")

# Test preprocessing
print("\n3. Testing preprocessing...")
X_processed = loader.preprocess_pipeline(X, filter=True, normalize=True, normalize_method='zscore')
print(f"Processed shape: {X_processed.shape}")

# Test with larger sample
print("\n4. Testing with larger sample (100 files)...")
X_large, y_large = loader.load_kaggle_dataset(max_files=100)
print(f"Loaded shape: {X_large.shape}, Labels: {y_large.shape}")
print(f"Class distribution: Alcoholic={np.sum(y_large==1)}, Control={np.sum(y_large==0)}")

print("\n" + "="*60)
print("DEBUGGING COMPLETE")
print("="*60)