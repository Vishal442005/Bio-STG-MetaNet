import urllib.request
import json

url = "https://s3.amazonaws.com/openneuro.org/ds004515/sub-001/eeg/"
# Let's check files in sub-001/eeg via crn graphql
graphql_url = "https://openneuro.org/crn/graphql"
query = """
{
  dataset(id: "ds004515") {
    latestSnapshot {
      files(prefix: "sub-001/eeg") {
        id
        filename
        size
      }
    }
  }
}
"""

req = urllib.request.Request(
    graphql_url,
    data=json.dumps({"query": query}).encode("utf-8"),
    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        files = data.get("data", {}).get("dataset", {}).get("latestSnapshot", {}).get("files", [])
        for f in files:
            print(f"File: {f['filename']}, Size: {f['size'] / (1024*1024):.2f} MB")
except Exception as e:
    print("Error:", e)
