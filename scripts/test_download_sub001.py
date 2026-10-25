import os
import urllib.request
import mne

os.makedirs("openneuro_ds005907/sub-001/eeg", exist_ok=True)
base_s3 = "https://s3.amazonaws.com/openneuro.org/ds005907"

# 1. Download root files
for root_file in ["dataset_description.json", "participants.tsv", "participants.json", "README"]:
    local_path = os.path.join("openneuro_ds005907", root_file)
    if not os.path.exists(local_path):
        print(f"Downloading {root_file}...")
        urllib.request.urlretrieve(f"{base_s3}/{root_file}", local_path)

# 2. Download sub-001 files
sub_files = [
    "sub-001_coordsystem.json",
    "sub-001_electrodes.tsv",
    "sub-001_task-ThreeArmedBandit_channels.tsv",
    "sub-001_task-ThreeArmedBandit_eeg.json",
    "sub-001_task-ThreeArmedBandit_events.tsv",
    "sub-001_task-ThreeArmedBandit_eeg.set"
]

for sf in sub_files:
    local_sf = os.path.join("openneuro_ds005907", "sub-001", "eeg", sf)
    if not os.path.exists(local_sf):
        print(f"Downloading sub-001/eeg/{sf}...")
        url = f"{base_s3}/sub-001/eeg/{sf}"
        urllib.request.urlretrieve(url, local_sf)
        print(f"Downloaded {sf} ({os.path.getsize(local_sf) / (1024*1024):.2f} MB)")

# 3. Test load with MNE
set_file = os.path.join("openneuro_ds005907", "sub-001", "eeg", "sub-001_task-ThreeArmedBandit_eeg.set")
print(f"Testing MNE load on {set_file}...")
raw = mne.io.read_raw_eeglab(set_file, preload=False)
print("MNE Raw Info:")
print(raw.info)
print("Channel names count:", len(raw.ch_names))
print("Sample channels:", raw.ch_names[:10])
