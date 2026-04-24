#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import amlx" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -e '.[mlx]'
fi

echo "Starting amlx at http://127.0.0.1:8000/"
amlx serve --host 127.0.0.1 --port 8000
