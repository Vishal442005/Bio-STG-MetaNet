import urllib.request
import pandas as pd
import io

url = "https://s3.amazonaws.com/openneuro.org/ds004515/participants.tsv"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    tsv_text = resp.read().decode("utf-8")

df = pd.read_csv(io.StringIO(tsv_text), sep="\t")
print("Total participants:", len(df))
print(df["GROUP"].value_counts())
print("\nAUDIT Summary:")
print(df.groupby("GROUP")["AUDIT"].describe())
print("\nAge Summary:")
print(df.groupby("GROUP")["age"].describe())
print("\nSex Distribution:")
print(pd.crosstab(df["GROUP"], df["sex"]))
