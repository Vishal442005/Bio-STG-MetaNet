import urllib.request
import json

url = "https://openneuro.org/crn/graphql"

for ds_id in ["ds004515", "ds005907"]:
    query = f"""
    {{
      dataset(id: "{ds_id}") {{
        id
        latestSnapshot {{
          tag
          description {{
            Name
            Authors
          }}
          summary {{
            modalities
            subjects
            tasks
            size
            totalFiles
          }}
        }}
      }}
    }}
    """
    req = urllib.request.Request(
        url,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            snap = data.get("data", {}).get("dataset", {}).get("latestSnapshot", {})
            desc = snap.get("description", {})
            summ = snap.get("summary", {})
            print(f"=== {ds_id} ===")
            print("Name:", desc.get("Name"))
            print("Authors:", desc.get("Authors"))
            print("Subjects count:", len(summ.get("subjects", [])))
            print("Size (GB):", round(summ.get("size", 0) / (1024**3), 2))
            print("Tasks:", summ.get("tasks"))
    except Exception as e:
        print(f"Error {ds_id}:", e)
