import os
import re

formula_path = "Formula/amlx.rb"
version = os.environ["VERSION"]
binary_url = os.environ["BINARY_URL"]
binary_sha256 = os.environ["BINARY_SHA256"]

with open(formula_path) as f:
    text = f.read()

# Update url line
text = re.sub(
    r'url "https://github\.com/[^"]+/releases/download/[^"]*"',
    f'url "{binary_url}"',
    text,
)

# Update sha256 line (first occurrence, the main one)
text = re.sub(
    r'^  sha256 "[^"]*"',
    f'  sha256 "{binary_sha256}"',
    text,
    count=1,
    flags=re.MULTILINE,
)

# Update version line
text = re.sub(
    r'^  version "[^"]*"',
    f'  version "{version}"',
    text,
    flags=re.MULTILINE,
)

with open(formula_path, "w") as f:
    f.write(text)

print(f"url     → {binary_url}")
print(f"sha256  → {binary_sha256}")
print(f"version → {version}")
