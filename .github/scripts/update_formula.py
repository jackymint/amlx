import os
import re

formula_path = "Formula/amlx.rb"
version = os.environ["VERSION"]
source_url = os.environ["SOURCE_URL"]
source_sha256 = os.environ["SOURCE_SHA256"]

with open(formula_path) as f:
    text = f.read()

text = re.sub(
    r'url "https://github\.com/[^"]*"',
    f'url "{source_url}"',
    text,
)

text = re.sub(
    r'^  sha256 "[^"]*"',
    f'  sha256 "{source_sha256}"',
    text,
    count=1,
    flags=re.MULTILINE,
)

text = re.sub(
    r'^  version "[^"]*"',
    f'  version "{version}"',
    text,
    flags=re.MULTILINE,
)

with open(formula_path, "w") as f:
    f.write(text)

print(f"url     → {source_url}")
print(f"sha256  → {source_sha256}")
print(f"version → {version}")
