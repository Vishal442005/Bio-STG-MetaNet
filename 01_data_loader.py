"""
FILE: 01_data_loader.py
PROJECT: Alcoholic EEG Classification - Internship Research
PAPER: Classification of Alcoholic EEG Signals Using a Deep Learning Method
        Farsi et al., IEEE Sensors Journal, 2021 | DOI: 10.1109/JSEN.2020.3026830
PURPOSE: Parse and preprocess the UCI SMNI EEG dataset into NumPy tensors.
"""

import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
TRAIN_DIR = r"C:\Internship\SMNI_CMI_TRAIN"
TEST_DIR  = r"C:\Internship\SMNI_CMI_TEST"
OUTPUT_DIR = r"C:\Users\Visha\Desktop\Internshhip\data"

N_CHANNELS = 64
N_TIMESTEPS = 256
# ────────────────────────────────────────────────────────────


def read_eeg_file(filepath: str) -> pd.DataFrame:
    """
    Parses a single raw EEG .gz file from the SMNI UCI dataset.
    Each file contains multi-channel EEG recordings for one trial.

    Returns a pandas DataFrame with columns:
        - trial_number: int
        - sensor_position: str (electrode label e.g. FP1, FZ, ...)
        - sample_num: int (0-255, 256 time-steps at 256 Hz)
        - sensor_value: float (voltage in µV)
    """
    with gzip.open(filepath, 'rt') as f:
        lines = [
            line.strip().split()
            for line in f
            if not line.startswith('#') and line.strip()
        ]

    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value']
    df = pd.DataFrame(lines, columns=columns)

    # Type-cast for downstream numerical operations
    df['sensor_value'] = df['sensor_value'].astype(float)
    df['sample_num']   = df['sample_num'].astype(int)
    df['trial_number'] = df['trial_number'].astype(int)

    return df


def file_to_matrix(filepath: str) -> np.ndarray | None:
    """
    Converts a single .gz EEG file → 2D NumPy matrix.
    Shape: (N_CHANNELS=64, N_TIMESTEPS=256)
    Returns None if the file does not conform to expected dimensions.
    """
    try:
        df = read_eeg_file(filepath)
        matrix = df.pivot(
            index='sensor_position',
            columns='sample_num',
            values='sensor_value'
        ).values
        if matrix.shape == (N_CHANNELS, N_TIMESTEPS):
            return matrix
    except Exception as e:
        print(f"  [WARN] Skipping {filepath}: {e}")
    return None


def process_directory(base_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates over all subject folders in base_dir.
    Each subject folder name encodes its class:
        'co2a' prefix → Alcoholic (label = 1)
        'co2c' prefix → Control   (label = 0)

    Returns:
        X: ndarray of shape (N_trials, 64, 256)  — EEG tensors
        y: ndarray of shape (N_trials,)           — Binary labels
    """
    X_list, y_list = [], []
    base = Path(base_dir)

    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    subjects = sorted([d for d in base.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subject folders in '{base_dir}'")

    for subject_folder in subjects:
        # Binary labelling from folder name convention
        label = 1 if 'co2a' in subject_folder.name else 0
        class_name = 'Alcoholic' if label == 1 else 'Control'

        trial_files = list(subject_folder.glob('*.gz'))
        loaded = 0

        for trial_file in trial_files:
            matrix = file_to_matrix(str(trial_file))
            if matrix is not None:
                X_list.append(matrix)
                y_list.append(label)
                loaded += 1

        print(f"  [{class_name}] {subject_folder.name}: {loaded}/{len(trial_files)} trials loaded")

    X = np.array(X_list, dtype=np.float32)  # (N, 64, 256)
    y = np.array(y_list,  dtype=np.int64)   # (N,)

    print(f"\nDataset assembled:")
    print(f"  Total trials : {X.shape[0]}")
    print(f"  Tensor shape : {X.shape}  (Trials × Channels × TimeSteps)")
    print(f"  Alcoholic    : {y.sum()} trials")
    print(f"  Control      : {(y == 0).sum()} trials")

    return X, y


def save_tensors(X: np.ndarray, y: np.ndarray, split: str = 'train') -> None:
    """Saves processed tensors to disk in .npy format."""
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    np.save(str(out / f'X_{split}.npy'), X)
    np.save(str(out / f'y_{split}.npy'), y)
    print(f"Saved → {out / f'X_{split}.npy'}  {out / f'y_{split}.npy'}")


if __name__ == '__main__':
    print("=" * 60)
    print("  UCI SMNI EEG Dataset Loader")
    print("  Farsi et al. IEEE Sensors Journal 2021")
    print("=" * 60)

    # Process training set
    X_train, y_train = process_directory(TRAIN_DIR)
    save_tensors(X_train, y_train, split='train')

    # Process test set
    X_test, y_test = process_directory(TEST_DIR)
    save_tensors(X_test, y_test, split='test')

    print("\nData loading complete. Proceed to 02_dsp_filtering_mne.py")
