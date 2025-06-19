from pathlib import Path
import json
import re

root_dir = Path("data")               # adjust if your folder is elsewhere
pattern  = re.compile(r"_label_(\d+)\.png$")

file2label = {
    p.name: int(pattern.search(p.name).group(1))
    for p in root_dir.iterdir()
    if p.is_file() and pattern.search(p.name)
}

with open("data/file_to_label.json", "w") as fp:
    json.dump(file2label, fp, indent=2)
