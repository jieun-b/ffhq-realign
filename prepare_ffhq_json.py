import json
import os

with open("ffhq/ffhq-dataset-v2.json", "r") as f:
    data = json.load(f)

for k, v in data.items():
    # in_the_wild
    path = v["in_the_wild"]["file_path"]
    v["in_the_wild"]["file_path"] = f"in-the-wild-images/{os.path.basename(path)}"

with open("ffhq/ffhq-dataset-flat.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
