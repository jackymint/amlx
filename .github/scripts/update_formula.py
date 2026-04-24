import os
import re

formula_path = "Formula/amlx.rb"
url = os.environ["TARBALL_URL"]
sha256 = os.environ["SHA256"]

with open(formula_path) as f:
    text = f.read()

text = re.sub(
    r'url "https://github\.com/[^/]+/amlx/archive/refs/tags/v[^"]*"',
    f'url "{url}"',
    text,
)

text = re.sub(
    r'(url "https://github\.com/[^"]+\.tar\.gz"\n  sha256 )"[^"]*"',
    rf'\1"{sha256}"',
    text,
)

with open(formula_path, "w") as f:
    f.write(text)

print(f"url    → {url}")
print(f"sha256 → {sha256}")
