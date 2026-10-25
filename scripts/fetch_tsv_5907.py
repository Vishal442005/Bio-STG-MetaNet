import urllib.request
import pandas as pd
import io

url = "https://s3.amazonaws.com/openneuro.org/ds005907/participants.tsv"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        tsv_text = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(tsv_text), sep="\t")
    print("=== ds005907 Participants ===")
    print("Total participants:", len(df))
    print(df.head())
    print("Columns:", list(df.columns))
    if "GROUP" in df.columns or "group" in df.columns:
        col = "GROUP" if "GROUP" in df.columns else "group"
        print(df[col].value_counts())
except Exception as e:
    print("ds005907 error:", e)
