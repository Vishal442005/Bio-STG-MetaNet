"""
Quick test of data loader functionality
"""

import os
import gzip
import pandas as pd
import numpy as np

# Test reading a single file
file_path = r"C:\Users\Visha\Desktop\Internshhip\SMNI_CMI_TRAIN\co2a0000364\co2a0000364.rd.000.gz"

print(f"Testing file: {file_path}")

try:
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print("First 20 lines:")
    for i, line in enumerate(lines[:20]):
        print(f"  {i}: {line.strip()}")
    
    # Parse data
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('S1') or line.startswith('FP1'):
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            try:
                trial_num = int(parts[0])
                sensor_pos = parts[1]
                sample_idx = int(parts[2])
                voltage = float(parts[3])
                data.append([trial_num, sensor_pos, sample_idx, voltage])
            except (ValueError, IndexError):
                continue
    
    print(f"\nParsed {len(data)} data points")
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['trial', 'sensor', 'sample', 'voltage'])
    print(f"DataFrame shape: {df.shape}")
    print(f"Unique sensors: {df['sensor'].nunique()}")
    print(f"Unique samples: {df['sample'].nunique()}")
    print(f"Sample range: {df['sample'].min()} to {df['sample'].max()}")
    
    # Get unique sensor names
    sensors = sorted(df['sensor'].unique())
    print(f"Sensors: {sensors[:10]}... (total {len(sensors)})")
    
    # Create mapping
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensors)}
    
    # Get samples
    samples = sorted(df['sample'].unique())
    print(f"Samples: {samples[:10]}... (total {len(samples)})")
    
    # Take first 256 samples
    if len(samples) > 256:
        samples = samples[:256]
    
    # Create matrix
    eeg_matrix = np.zeros((64, 256))
    
    for _, row in df.iterrows():
        sensor_idx = sensor_to_idx.get(row['sensor'], -1)
        sample_idx = row['sample']
        
        if sensor_idx >= 0 and sensor_idx < 64 and sample_idx < 256:
            eeg_matrix[sensor_idx, sample_idx] = row['voltage']
    
    print(f"\nEEG matrix shape: {eeg_matrix.shape}")
    print(f"Non-zero elements: {np.count_nonzero(eeg_matrix)}")
    print(f"Data range: {eeg_matrix.min():.2f} to {eeg_matrix.max():.2f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()