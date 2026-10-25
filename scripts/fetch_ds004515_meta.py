import urllib.request
import json
import pandas as pd
import io

channels_url = "https://s3.amazonaws.com/openneuro.org/ds004515/sub-001/eeg/sub-001_task-ProbabilisticSelection_channels.tsv"
eeg_json_url = "https://s3.amazonaws.com/openneuro.org/ds004515/sub-001/eeg/sub-001_task-ProbabilisticSelection_eeg.json"

try:
    req = urllib.request.Request(channels_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        ch_df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), sep="\t")
        print("Channels count:", len(ch_df))
        print("Channel names (first 10):", ch_df["name"].head(10).tolist())
        print("Sampling rate & types:", ch_df["type"].value_counts().to_dict())

    req2 = urllib.request.Request(eeg_json_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req2, timeout=10) as resp:
        info = json.loads(resp.read().decode("utf-8"))
        print("EEG JSON Info:", {k: info.get(k) for k in ["SamplingFrequency", "EEGChannelCount", "PowerLineFrequency", "EEGReference"]})
except Exception as e:
    print("Error:", e)
