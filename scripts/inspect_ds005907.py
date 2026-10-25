import urllib.request
import xml.etree.ElementTree as ET

url = "https://s3.amazonaws.com/openneuro.org/?prefix=ds005907/&delimiter=/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_data = resp.read()
        root = ET.fromstring(xml_data)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        # Check CommonPrefixes (subfolders)
        prefixes = root.findall('s3:CommonPrefixes', ns)
        subdirs = [p.find('s3:Prefix', ns).text for p in prefixes]
        print(f"Subdirectories ({len(subdirs)}):", subdirs[:10])
        
        # Check root files
        contents = root.findall('s3:Contents', ns)
        print(f"Root files ({len(contents)}):")
        for c in contents[:10]:
            k = c.find('s3:Key', ns).text
            sz = int(c.find('s3:Size', ns).text)
            print(f"  {k} -> {sz / (1024*1024):.2f} MB")
except Exception as e:
    print("Error:", e)
