import gzip
import os

sample_subject = os.path.join("UCI_FULL_PROCESSED", "co2a0000364.tar")
files = os.listdir(sample_subject)
print("Files in subject folder:", len(files), "Example:", files[0])

sample_file = os.path.join(sample_subject, files[0])
with gzip.open(sample_file, "rt") as f:
    lines = [f.readline() for _ in range(25)]
    print("First 25 lines of trial:")
    for line in lines:
        print(line.strip())
