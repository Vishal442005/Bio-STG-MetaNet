import urllib.request
import xml.etree.ElementTree as ET

url = "https://s3.amazonaws.com/openneuro.org/?prefix=ds004515/sub-001/eeg/&delimiter=/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_data = resp.read()
        root = ET.fromstring(xml_data)
        # S3 namespace
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        contents = root.findall('s3:Contents', ns)
        print(f"Found {len(contents)} files:")
        for c in contents:
            key = c.find('s3:Key', ns).text
            size = int(c.find('s3:Size', ns).text)
            print(f"  {key} -> {size / (1024*1024):.2f} MB")
except Exception as e:
    print("Error:", e)
