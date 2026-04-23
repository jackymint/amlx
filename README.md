# amlx

MacBook-first local AI inference server inspired by oMLX.

## Goals

- OpenAI-compatible API for local coding agents
- 3-layer cache (RAM + SQLite + paged SSD blocks)
- Continuous batching scheduler for concurrent local requests
- Easy local operation on Apple Silicon MacBooks
- Clean adapter interface for MLX runtime integration

## Quickstart (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
amlx serve --engine echo --host 127.0.0.1 --port 8000
```

Open dashboard UI:

```bash
open http://127.0.0.1:8000/
```

Download models from dashboard:

- Open `Model Download Center`
- Search models by name/tag/id in the search box (online from Hugging Face)
- Browse results with pagination (`Prev` / `Next`) to avoid long scrolling
- Each model shows machine fit (`Good fit`, `Runs but tight`, `Not suitable`)
- Click `Download` on a model
- Track progress in `Download Jobs`
- Remove models from memory with `Unload` and remove disk installs with `Delete`
- Installed models appear in `Installed Models`

Chat from dashboard:

- Use `Chat Console` for multi-turn conversation UI
- Press `Enter` to send, `Shift+Enter` for newline
- `Clear` resets local chat history in the browser
- Runtime chips show `configured model` vs `loaded` status separately

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen2.5-coder:7b",
    "messages": [{"role": "user", "content": "Write a Python function for Fibonacci."}],
    "max_tokens": 128,
    "temperature": 0.2
  }' | jq
```

Cache stats:

```bash
curl http://127.0.0.1:8000/v1/cache/stats
```

Tune for MacBook throughput:

```bash
amlx serve \
  --engine echo \
  --max-batch-size 12 \
  --batch-wait-ms 25 \
  --max-memory-cache-items 1024 \
  --block-chars 4096
```

Custom model storage path:

```bash
amlx serve --models-dir ~/.amlx/models
```

## Use MLX runtime

Install optional dependency:

```bash
pip install -e '.[mlx]'
```

Run with MLX adapter:

```bash
amlx serve --engine mlx --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

## Roadmap

- True paged KV cache blocks on SSD
- Model warm pool and prefill reuse
- Menu bar app
