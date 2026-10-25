import os

processed_dir = "UCI_FULL_PROCESSED"
subjects = os.listdir(processed_dir)
print(f"Total subject folders: {len(subjects)}")

alc_subjects = [s for s in subjects if "a" in s[3]]
ctl_subjects = [s for s in subjects if "c" in s[3]]
print(f"Alcoholic subjects (co2a...): {len(alc_subjects)}")
print(f"Control subjects (co2c...): {len(ctl_subjects)}")

total_trials = 0
for s in subjects:
    s_path = os.path.join(processed_dir, s)
    if os.path.isdir(s_path):
        trials = [f for f in os.listdir(s_path) if f.endswith(".gz")]
        total_trials += len(trials)
print(f"Total trial gz files across all subjects: {total_trials}")
