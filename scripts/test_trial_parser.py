import gzip
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from scipy.signal import butter, sosfiltfilt

def load_single_trial(filepath, channel_order, target_len=256):
    chan_data = {ch: [] for ch in channel_order}
    current_chan = None
    
    with gzip.open(filepath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2] == "chan":
                    current_chan = parts[1]
                continue
            
            if current_chan in chan_data:
                parts = line.strip().split()
                if len(parts) >= 4:
                    val = float(parts[3])
                    chan_data[current_chan].append(val)
                    
    signals = []
    for ch in channel_order:
        series = chan_data[ch]
        if len(series) >= target_len:
            signals.append(series[:target_len])
        elif len(series) > 0:
            padded = np.pad(series, (0, target_len - len(series)), mode='edge')
            signals.append(padded)
        else:
            signals.append(np.zeros(target_len))
            
    arr = np.array(signals, dtype=np.float32)
    sos = butter(4, [0.5, 45.0], btype='bandpass', fs=256, output='sos')
    arr_filt = sosfiltfilt(sos, arr, axis=-1)
    
    mean = np.mean(arr_filt, axis=-1, keepdims=True)
    std = np.std(arr_filt, axis=-1, keepdims=True)
    std[std == 0] = 1.0
    arr_norm = (arr_filt - mean) / std
    return arr_norm

sample = os.path.join("UCI_FULL_PROCESSED", "co2a0000364.tar", "co2a0000364.rd.000.gz")
from src.data.electrode_coords import EEG_CHANNELS
data = load_single_trial(sample, EEG_CHANNELS)
print("Parsed trial shape:", data.shape)
print("Mean:", np.mean(data), "Std:", np.std(data), "Min:", np.min(data), "Max:", np.max(data))
