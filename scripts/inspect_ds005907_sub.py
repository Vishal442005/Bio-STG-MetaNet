import urllib.request
import xml.etree.ElementTree as ET

url = "https://s3.amazonaws.com/openneuro.org/?prefix=ds005907/sub-001/&delimiter=/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_data = resp.read()
        root = ET.fromstring(xml_data)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        prefixes = root.findall('s3:CommonPrefixes', ns)
        subdirs = [p.find('s3:Prefix', ns).text for p in prefixes]
        print(f"sub-001 subdirectories:", subdirs)
        
        # also check recursive prefix
        url2 = "https://s3.amazonaws.com/openneuro.org/?prefix=ds005907/sub-001/eeg/"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            root2 = ET.fromstring(resp2.read())
            contents = root2.findall('s3:Contents', ns)
            print(f"sub-001/eeg files ({len(contents)}):")
            for c in contents:
                k = c.find('s3:Key', ns).text
                sz = int(c.find('s3:Size', ns).text)
                print(f"  {k} -> {sz / (1024*1024):.2f} MB")
except Exception as e:
    print("Error:", e)
