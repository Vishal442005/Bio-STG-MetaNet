import mne
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from src.data.electrode_coords import EEG_CHANNELS, CLINICAL_19_CHANNELS

test_file = os.path.join("openneuro_ds005907", "sub-001", "eeg", "sub-001_task-ThreeArmedBandit_eeg.set")
raw = mne.io.read_raw_eeglab(test_file, preload=False)
openneuro_chs = raw.ch_names

# Map channels case-insensitively
openneuro_chs_upper = [ch.upper() for ch in openneuro_chs]
uci_chs_upper = [ch.upper() for ch in EEG_CHANNELS]

common = [ch for ch in EEG_CHANNELS if ch.upper() in openneuro_chs_upper]
clinical_common = [ch for ch in CLINICAL_19_CHANNELS if ch.upper() in openneuro_chs_upper]

print(f"Total UCI cortical channels: {len(EEG_CHANNELS)}")
print(f"Total OpenNeuro channels: {len(openneuro_chs)}")
print(f"Common channels ({len(common)}): {common}")
print(f"Clinical 19 common ({len(clinical_common)}/19): {clinical_common}")
