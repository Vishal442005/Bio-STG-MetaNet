import urllib.request
import xml.etree.ElementTree as ET

for sub in ["sub-002", "sub-025", "sub-053"]:
    url = f"https://s3.amazonaws.com/openneuro.org/?prefix=ds005907/{sub}/eeg/&delimiter=/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
            ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
            contents = root.findall('s3:Contents', ns)
            keys = [c.find('s3:Key', ns).text.split('/')[-1] for c in contents]
            sizes = [int(c.find('s3:Size', ns).text)/(1024*1024) for c in contents]
            print(f"{sub} files: {list(zip(keys, [round(s, 2) for s in sizes]))}")
    except Exception as e:
        print(f"Error {sub}:", e)
