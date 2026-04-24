import glob
import json
import re
import sys

bottle_files = glob.glob("*.json")
if not bottle_files:
    print("No bottle JSON found, skipping.")
    sys.exit(0)

with open(bottle_files[0]) as f:
    data = json.load(f)

formula_data = next(iter(data.values()))
bottle = formula_data["bottle"]
root_url = bottle["root_url"]
cellar = bottle.get("cellar", ":any_skip_relocation")
tags = bottle["tags"]

lines = ['  bottle do', f'    root_url "{root_url}"']
for tag, info in tags.items():
    sha = info["sha256"]
    lines.append(f'    sha256 cellar: {cellar}, {tag}: "{sha}"')
lines.append("  end")
bottle_block = "\n".join(lines)

formula_path = "Formula/amlx.rb"
with open(formula_path) as f:
    text = f.read()

if "bottle do" in text:
    text = re.sub(r"  bottle do\n.*?  end\n", bottle_block + "\n", text, flags=re.DOTALL)
else:
    text = re.sub(r"(  license [^\n]+\n)", rf"\1\n{bottle_block}\n", text)

with open(formula_path, "w") as f:
    f.write(text)

print(f"Bottle block written for tags: {list(tags.keys())}")
