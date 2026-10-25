import pandas as pd
import os

events_path = os.path.join("openneuro_ds005907", "sub-001", "eeg", "sub-001_task-ThreeArmedBandit_events.tsv")
df = pd.read_csv(events_path, sep="\t")
print("Events shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(10))
if "trial_type" in df.columns:
    print("\nTrial types:", df["trial_type"].value_counts())
