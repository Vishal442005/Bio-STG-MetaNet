import gzip
import os

sample_subject = os.path.join("UCI_FULL_PROCESSED", "co2a0000364.tar")
sample_file = os.path.join(sample_subject, os.listdir(sample_subject)[0])

channels = []
with gzip.open(sample_file, "rt") as f:
    for line in f:
        if line.startswith("#") and "chan" in line:
            parts = line.strip().split()
            # e.g., # FP1 chan 0
            channels.append(parts[1])

print(f"Extracted {len(channels)} channels:")
print(channels)
