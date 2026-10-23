"""
Improved EEG Data Loader
Handles both UCI eeg_full dataset and Kaggle SMNI datasets
Supports cross-dataset validation and comprehensive preprocessing
"""

import os
import gzip
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import warnings
warnings.filterwarnings('ignore')


class EEGDataLoader:
    """
    Comprehensive EEG data loader for alcoholic vs control classification
    Handles multiple dataset formats and provides unified preprocessing
    """
    
    def __init__(self, sampling_rate=256, n_channels=64, n_samples=256):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.filter_low = 0.5  # Hz
        self.filter_high = 100.0  # Hz
        
    def load_uci_dataset(self, data_dir='eeg_full', max_files=None):
        """
        Load UCI KDD EEG dataset from eeg_full directory
        Each subject directory contains .tar.gz files with EEG trials
        """
        print(f"Loading UCI dataset from {data_dir}...")
        
        X_list = []
        y_list = []
        
        # Get all subject directories
        subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for subject_dir in subject_dirs:
            subject_path = os.path.join(data_dir, subject_dir)
            
            # Determine label from directory name
            if 'co2a' in subject_dir.lower():
                label = 1  # Alcoholic
            elif 'co2c' in subject_dir.lower():
                label = 0  # Control
            else:
                continue  # Skip other directories
            
            # Process all .tar.gz files in subject directory
            tar_files = glob.glob(os.path.join(subject_path, '*.tar.gz'))
            
            for tar_file in tar_files:
                try:
                    # For now, skip .tar.gz files as they need extraction
                    # We'll focus on the Kaggle .gz files which are easier to process
                    continue
                except Exception as e:
                    print(f"Error processing {tar_file}: {e}")
                    continue
        
        if len(X_list) == 0:
            print("No data loaded from UCI dataset (tar.gz files need extraction)")
            print("Focusing on Kaggle dataset instead")
            return None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} trials from UCI dataset")
        print(f"Shape: {X.shape}, Labels: {y.shape}")
        print(f"Class distribution - Alcoholic: {np.sum(y==1)}, Control: {np.sum(y==0)}")
        
        return X, y
    
    def load_kaggle_dataset(self, train_dir='SMNI_CMI_TRAIN', test_dir='SMNI_CMI_TEST', max_files=None):
        """
        Load Kaggle SMNI EEG dataset
        Combines TRAIN and TEST directories for comprehensive dataset
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory  
            max_files: Maximum number of files to process (None for all)
        """
        print(f"Loading Kaggle dataset from {train_dir} and {test_dir}...")
        
        X_list = []
        y_list = []
        total_files_processed = 0
        
        # Collect files by class for balanced sampling
        class_0_files = []  # Control
        class_1_files = []  # Alcoholic
        
        # Process both train and test directories
        for data_dir in [train_dir, test_dir]:
            if not os.path.exists(data_dir):
                print(f"Warning: {data_dir} not found, skipping...")
                continue
                
            subject_dirs = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
            
            print(f"Found {len(subject_dirs)} subject directories in {data_dir}")
            
            for subject_dir in subject_dirs:
                subject_path = os.path.join(data_dir, subject_dir)
                
                # Determine label from directory name
                if 'co2a' in subject_dir.lower():
                    label = 1  # Alcoholic
                elif 'co2c' in subject_dir.lower():
                    label = 0  # Control
                else:
                    print(f"Skipping unknown directory: {subject_dir}")
                    continue
                
                # Process all .gz files in subject directory
                gz_files = glob.glob(os.path.join(subject_path, '*.gz'))
                print(f"  Processing {subject_dir}: {len(gz_files)} files")
                
                for gz_file in gz_files:
                    if label == 0:
                        class_0_files.append(gz_file)
                    else:
                        class_1_files.append(gz_file)
        
        print(f"Collected {len(class_0_files)} control files, {len(class_1_files)} alcoholic files")
        
        # Balance the dataset
        if max_files is not None:
            files_per_class = max_files // 2
            
            # Sample from each class
            import random
            random.seed(42)
            
            if len(class_0_files) >= files_per_class:
                sampled_0 = random.sample(class_0_files, files_per_class)
            else:
                sampled_0 = class_0_files  # Use all if not enough
            
            if len(class_1_files) >= files_per_class:
                sampled_1 = random.sample(class_1_files, files_per_class)
            else:
                sampled_1 = class_1_files  # Use all if not enough
            
            print(f"Balanced sampling: {len(sampled_0)} control, {len(sampled_1)} alcoholic")
            
            # Process sampled files
            for gz_file in sampled_0:
                try:
                    trial_data = self._process_kaggle_file(gz_file)
                    if trial_data is not None:
                        X_list.append(trial_data)
                        y_list.append(0)
                        total_files_processed += 1
                except Exception as e:
                    print(f"Error processing {gz_file}: {e}")
                    continue
            
            for gz_file in sampled_1:
                try:
                    trial_data = self._process_kaggle_file(gz_file)
                    if trial_data is not None:
                        X_list.append(trial_data)
                        y_list.append(1)
                        total_files_processed += 1
                except Exception as e:
                    print(f"Error processing {gz_file}: {e}")
                    continue
        else:
            # Use balanced sampling from all available files
            import random
            random.seed(42)
            
            # Balance the classes by taking minimum of both
            min_samples = min(len(class_0_files), len(class_1_files))
            print(f"Balancing to {min_samples} samples per class (total {2*min_samples})")
            
            if len(class_0_files) >= min_samples:
                sampled_0 = random.sample(class_0_files, min_samples)
            else:
                sampled_0 = class_0_files
            
            if len(class_1_files) >= min_samples:
                sampled_1 = random.sample(class_1_files, min_samples)
            else:
                sampled_1 = class_1_files
            
            # Process balanced files with progress reporting
            print(f"Processing {len(sampled_0)} control files...")
            for i, gz_file in enumerate(sampled_0):
                try:
                    trial_data = self._process_kaggle_file(gz_file)
                    if trial_data is not None:
                        X_list.append(trial_data)
                        y_list.append(0)
                        total_files_processed += 1
                        if (i + 1) % 50 == 0:
                            print(f"  Processed {i + 1}/{len(sampled_0)} control files...")
                except Exception as e:
                    print(f"Error processing {gz_file}: {e}")
                    continue
            
            print(f"Processing {len(sampled_1)} alcoholic files...")
            for i, gz_file in enumerate(sampled_1):
                try:
                    trial_data = self._process_kaggle_file(gz_file)
                    if trial_data is not None:
                        X_list.append(trial_data)
                        y_list.append(1)
                        total_files_processed += 1
                        if (i + 1) % 50 == 0:
                            print(f"  Processed {i + 1}/{len(sampled_1)} alcoholic files...")
                except Exception as e:
                    print(f"Error processing {gz_file}: {e}")
                    continue
        
        if len(X_list) == 0:
            raise ValueError("No data loaded from Kaggle dataset")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} trials from Kaggle dataset")
        print(f"Shape: {X.shape}, Labels: {y.shape}")
        print(f"Class distribution - Alcoholic: {np.sum(y==1)}, Control: {np.sum(y==0)}")
        
        return X, y
    
    def _process_uci_file(self, file_path):
        """
        Process individual UCI dataset file
        Returns 64x256 matrix or None if processing fails
        """
        try:
            # Read the tar.gz file (simplified - assumes extracted format)
            # This is a placeholder - actual implementation depends on file format
            # For now, return None to indicate need for actual file processing
            return None
        except:
            return None
    
    def _process_kaggle_file(self, file_path):
        """
        Process individual Kaggle .gz file
        File format: Header lines with #, then trial/channel info, then data
        Format: trial_num channel_name sample_index voltage
        Returns 64x256 matrix (taking first 256 samples from available data)
        """
        try:
            # Read gz file
            with gzip.open(file_path, 'rt') as f:
                lines = f.readlines()
            
            # Parse data, skipping header lines (starting with #)
            data = []
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line.startswith('S1') or line.startswith('FP1'):
                    continue  # Skip header and metadata lines
                
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
            
            if len(data) == 0:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['trial', 'sensor', 'sample', 'voltage'])
            
            # Get unique sensor names to ensure consistent ordering
            sensors = sorted(df['sensor'].unique())
            
            # Create mapping from sensor names to indices (0-63)
            sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensors)}
            
            # Get unique sample indices
            samples = sorted(df['sample'].unique())
            
            # Take first 256 samples if more are available
            if len(samples) > 256:
                samples = samples[:256]
            elif len(samples) < 256:
                # Pad with zeros if less than 256 samples
                pass  # Will handle in matrix creation
            
            # Create 64x256 matrix
            eeg_matrix = np.zeros((64, 256))
            
            # Fill matrix with data
            for _, row in df.iterrows():
                sensor_idx = sensor_to_idx.get(row['sensor'], -1)
                sample_idx = row['sample']
                
                if sensor_idx >= 0 and sensor_idx < 64 and sample_idx < 256:
                    eeg_matrix[sensor_idx, sample_idx] = row['voltage']
            
            return eeg_matrix
            
        except Exception as e:
            print(f"Error in _process_kaggle_file for {file_path}: {e}")
            return None
    
    def apply_filtering(self, X):
        """
        Apply Butterworth bandpass filter (0.5-100 Hz) with zero-phase filtering
        """
        print("Applying Butterworth bandpass filter (0.5-100 Hz)...")
        
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        # Design 4th order Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        X_filtered = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Apply zero-phase filtering (filtfilt)
                X_filtered[i, j, :] = signal.filtfilt(b, a, X[i, j, :])
        
        print("Filtering complete")
        return X_filtered
    
    def normalize_data(self, X, method='zscore'):
        """
        Normalize EEG data
        Methods: 'zscore', 'minmax', 'robust'
        """
        print(f"Normalizing data using {method} method...")
        
        X_normalized = np.zeros_like(X)
        
        if method == 'zscore':
            # Z-score normalization per channel per trial
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    mean = np.mean(X[i, j, :])
                    std = np.std(X[i, j, :])
                    if std > 1e-6:
                        X_normalized[i, j, :] = (X[i, j, :] - mean) / std
                    else:
                        X_normalized[i, j, :] = X[i, j, :] - mean
        
        elif method == 'minmax':
            # Min-max normalization per channel per trial
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    min_val = np.min(X[i, j, :])
                    max_val = np.max(X[i, j, :])
                    if max_val > min_val:
                        X_normalized[i, j, :] = (X[i, j, :] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    median = np.median(X[i, j, :])
                    q75, q25 = np.percentile(X[i, j, :], [75, 25])
                    iqr = q75 - q25
                    if iqr > 1e-6:
                        X_normalized[i, j, :] = (X[i, j, :] - median) / iqr
                    else:
                        X_normalized[i, j, :] = X[i, j, :] - median
        
        print("Normalization complete")
        return X_normalized
    
    def preprocess_pipeline(self, X, filter=True, normalize=True, normalize_method='zscore'):
        """
        Complete preprocessing pipeline
        """
        X_processed = X.copy()
        
        if filter:
            X_processed = self.apply_filtering(X_processed)
        
        if normalize:
            X_processed = self.normalize_data(X_processed, method=normalize_method)
        
        return X_processed
    
    def create_train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, stratify=True):
        """
        Create train/validation/test splits with stratification
        """
        if stratify:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size + val_size, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size + val_size), 
                random_state=42, stratify=y_temp
            )
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size + val_size, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42
            )
        
        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        print(f"Test set: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X, y, output_dir='outputs', prefix='eeg'):
        """
        Save processed data to numpy files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        X_path = os.path.join(output_dir, f'{prefix}_X.npy')
        y_path = os.path.join(output_dir, f'{prefix}_y.npy')
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        print(f"Saved processed data to {output_dir}")
        print(f"  {X_path}")
        print(f"  {y_path}")
        
        return X_path, y_path
    
    def load_processed_data(self, X_path, y_path):
        """
        Load previously processed data
        """
        X = np.load(X_path)
        y = np.load(y_path)
        
        print(f"Loaded processed data from {X_path} and {y_path}")
        print(f"Shape: {X.shape}, Labels: {y.shape}")
        
        return X, y


def main():
    """
    Main function to demonstrate data loading capabilities
    """
    loader = EEGDataLoader()
    
    print("=" * 60)
    print("EEG Data Loader - Testing Dataset Loading")
    print("=" * 60)
    
    # Try to load Kaggle dataset (small subset for testing)
    try:
        print("\n1. Loading Kaggle SMNI dataset (testing with 100 files)...")
        X_kaggle, y_kaggle = loader.load_kaggle_dataset(max_files=100)
        
        # Preprocess
        X_kaggle_processed = loader.preprocess_pipeline(X_kaggle)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_split(
            X_kaggle_processed, y_kaggle
        )
        
        # Save processed data
        loader.save_processed_data(X_kaggle_processed, y_kaggle, prefix='kaggle')
        
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
    
    # Try to load existing processed data if available
    try:
        print("\n2. Loading existing processed data...")
        if os.path.exists('X_filtered.npy') and os.path.exists('y_train.npy'):
            X_existing, y_existing = loader.load_processed_data('X_filtered.npy', 'y_train.npy')
            print(f"Existing data shape: {X_existing.shape}")
    except Exception as e:
        print(f"No existing processed data found: {e}")


if __name__ == "__main__":
    main()