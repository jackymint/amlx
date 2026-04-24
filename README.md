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
pip install -e '.[mlx]'
amlx serve --host 127.0.0.1 --port 8000
```

Open dashboard UI:

```bash
open http://127.0.0.1:8000/
```

Download models from dashboard:

- Open `DOWNLOAD MODEL`
- Search models by name/tag/id in the search box (online from Hugging Face)
- Browse results with pagination (`Prev` / `Next`) to avoid long scrolling
- Each model shows machine fit (`Good fit`, `Runs but tight`, `Not suitable`)
- Click `Download` on a model
- Track progress in `Download Jobs`
- Remove models from memory with `Unload` and remove disk installs with `Delete`
- Installed models appear in `Installed Models`
- Train models from `Training` tab using LoRA/QLoRA jobs (line-based samples + epochs)
- Completed fine-tune adapters are auto-applied to matching models during chat inference
- Upload `.json` / `.jsonl` dataset in `Train Loaded Model` to parse and train automatically

Chat from dashboard:

- Use `Chat` for multi-turn conversation UI
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

Tool calling (OpenAI-style):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Call get_weather for Bangkok {\"city\":\"Bangkok\"}"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather by city",
          "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
        }
      }
    ],
    "tool_choice": "required"
  }' | jq
```

Thinking (reasoning summary) support:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role":"user","content":"Explain cache layers briefly"}],
    "thinking": {"enabled": true, "summary": "auto"},
    "reasoning_effort": "medium"
  }' | jq
```

When enabled, response includes:
- `choices[0].message.thinking`
- `usage.reasoning_tokens`

## Training Guidelines

Use these guidelines for better training quality, especially with tool-calling workloads:

1. Keep one intent per sample
- Each line/record should represent a single behavior (for example: "weather query => call `get_weather`").

2. Prefer deterministic phrasing
- Use stable wording patterns for repeated intents so model behavior is easier to steer.

3. Add explicit tool intent cues
- Include samples such as "If user asks current weather, call `get_weather` with city JSON."
- Keep tool name spelling exactly the same as request `tools[].function.name`.

4. Pair intent with argument shape
- Include examples containing valid JSON snippets matching your tool schema.
- Example: `{"city":"Bangkok"}` not free-form text.

5. Separate policy from facts
- Put response style rules (tone, length, safety) in dedicated samples.
- Put domain facts (product, SOP, business rules) in different samples.

6. Start small, iterate
- Begin with 20-50 clean samples, train, test on real prompts, then add missing cases.

7. Keep epochs conservative
- Use `1-2` epochs first; raise only when behavior is still weak.

8. Remember runtime contract
- Training in this project runs real LoRA/QLoRA fine-tuning jobs and outputs adapter weights.
- For reliable tool invocation, still send `tools` + `tool_choice` in each chat request.

Cache stats:

```bash
curl http://127.0.0.1:8000/v1/cache/stats
```

Tune for MacBook throughput:

```bash
amlx serve \
  --max-batch-size 12 \
  --batch-wait-ms 25 \
  --max-memory-cache-items 1024 \
  --block-chars 4096
```

Custom model storage path:

```bash
amlx serve --models-dir ~/.amlx/models
```

## Configure MLX model

Set configured MLX model:

```bash
amlx serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

## Roadmap

- True paged KV cache blocks on SSD
- Model warm pool and prefill reuse
- Menu bar app
