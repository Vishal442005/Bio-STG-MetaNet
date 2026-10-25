"""
Multi-Threaded Resumable Downloader for Full OpenNeuro ds005907 Dataset
Downloads all 53 subjects (sub-001 to sub-053) + root BIDS metadata from public AWS S3.
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

BASE_S3 = "https://s3.amazonaws.com/openneuro.org/ds005907"
OUTPUT_DIR = "openneuro_ds005907"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Root BIDS files
ROOT_FILES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "task-ThreeArmedBandit_events.json",
    "README",
    "CHANGES"
]

# 2. All 53 subjects
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 54)]

FILE_SUFFIXES = [
    "coordsystem.json",
    "electrodes.tsv",
    "task-ThreeArmedBandit_channels.tsv",
    "task-ThreeArmedBandit_eeg.json",
    "task-ThreeArmedBandit_events.tsv",
    "task-ThreeArmedBandit_eeg.set"
]

def get_remote_file_size(url):
    """Fetches Content-Length from HTTP HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0

def download_file(item):
    rel_path, is_large = item
    url = f"{BASE_S3}/{rel_path.replace(os.sep, '/')}"
    local_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    remote_size = get_remote_file_size(url)
    
    # Check if already complete
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        if remote_size > 0 and local_size == remote_size:
            return rel_path, local_size, "skipped"
        elif remote_size == 0 and local_size > 0 and not is_large:
            return rel_path, local_size, "skipped"

    temp_path = local_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024) # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        
        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(temp_path, local_path)
        return rel_path, downloaded, "downloaded"
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return rel_path, 0, f"error: {e}"


def main():
    print(f"=== INITIATING FULL DOWNLOAD OF OPENNEURO ds005907 (53 SUBJECTS) ===")
    print(f"Destination: {os.path.abspath(OUTPUT_DIR)}")
    
    # Build list of files to download
    tasks = []
    for rf in ROOT_FILES:
        tasks.append((rf, False))
        
    for sub in SUBJECTS:
        for suf in FILE_SUFFIXES:
            rel = os.path.join(sub, "eeg", f"{sub}_{suf}")
            is_large = suf.endswith(".set")
            tasks.append((rel, is_large))
            
    print(f"Total files in queue: {len(tasks)} (53 subjects x 6 files + {len(ROOT_FILES)} root files)")
    
    total_bytes = 0
    start_time = time.time()
    completed_count = 0
    large_set_count = 0
    
    # Use 4 parallel threads for downloading
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, item): item for item in tasks}
        
        for future in as_completed(futures):
            item = futures[future]
            rel_path, bytes_dl, status = future.result()
            completed_count += 1
            total_bytes += bytes_dl
            
            if rel_path.endswith(".set") and status in ["downloaded", "skipped"]:
                large_set_count += 1
                elapsed = time.time() - start_time
                mb_so_far = total_bytes / (1024 * 1024)
                rate = mb_so_far / elapsed if elapsed > 0 else 0
                print(f"[{completed_count}/{len(tasks)}] {rel_path} ({status}) | "
                      f"Raw EEG Sets: {large_set_count}/53 | "
                      f"Total DL: {mb_so_far:.1f} MB | Speed: {rate:.2f} MB/s", flush=True)
            elif not rel_path.endswith(".set"):
                if status == "downloaded":
                    print(f"[{completed_count}/{len(tasks)}] Metadata: {rel_path} ({status})", flush=True)
            elif "error" in status:
                print(f"[RETRY NEEDED] {rel_path}: {status}", flush=True)

    elapsed_total = time.time() - start_time
    print(f"\n=======================================================")
    print(f"DOWNLOAD COMPLETE: {completed_count}/{len(tasks)} files processed in {elapsed_total/60:.2f} minutes.")
    print(f"Total data transferred: {total_bytes / (1024*1024):.2f} MB.")
    print(f"Total EEG subjects downloaded: {large_set_count}/53.")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
