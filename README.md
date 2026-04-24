# amlx

MacBook-first local AI inference server with OpenAI-compatible API, built for Apple Silicon.

## Features

- **OpenAI-compatible API** — drop-in for local coding agents and tools
- **3-layer prefix cache** — RAM (LRU) + SQLite disk + paged SSD blocks
- **Continuous batching scheduler** — handles concurrent requests efficiently
- **LoRA / QLoRA fine-tuning** — train loaded models directly from the dashboard
- **Quantization** — reduce model size on-device
- **Dashboard UI** — download, chat, train, and quantize from the browser
- **Tool calling** — OpenAI-style `tools` + `tool_choice`
- **Thinking / reasoning** — summarized reasoning tokens in response
- **GPU duty-cycle cap** — keep your MacBook cool under sustained load
- **Homebrew install** — one-line install via tap

## Install

**Homebrew (recommended)**

```bash
brew tap jackymint/amlx
brew install amlx
```

**pip**

```bash
pip install 'amlx[mlx]'
```

**From source**

```bash
git clone https://github.com/jackymint/amlx
cd amlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx]'
```

> Requires macOS + Apple Silicon (MLX backend).

## Quickstart

```bash
amlx serve
```

Open dashboard:

```
http://127.0.0.1:8000/
```

## Dashboard

| Tab | What you can do |
|-----|----------------|
| **Home** | Live metrics — cache hits, batch scheduler stats, throughput pulse |
| **Chat** | Multi-turn chat with any loaded model |
| **Download Model** | Search Hugging Face, check device fit, download & manage models |
| **Train Loaded Model** | Upload `.json` / `.jsonl` dataset → LoRA/QLoRA fine-tune |
| **Quantize** | Quantize installed models on-device |

**Download flow**

1. Open **Download Model** tab
2. Search by name, id, or tag
3. Each result shows machine fit: `Good fit` / `Runs but tight` / `Not suitable`
4. Click **Download** — track progress in **Download Jobs**
5. Loaded models appear in **Loaded In Memory**
6. Remove from memory with **Unload**, remove from disk with **Delete**

**Chat**

- Press `Enter` to send, `Shift+Enter` for newline
- `Clear` resets browser chat history
- Runtime chips show `configured model` vs `loaded` status

**Training**

1. Choose model and profile name
2. Set epochs (start with 1–2)
3. Upload `.json` or `.jsonl` dataset
4. Fine-tuned adapter weights are auto-applied during chat inference
5. Click **Export** to save the merged model

## API

**Health check**

```bash
curl http://127.0.0.1:8000/health
```

**Chat completion**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Write a Python Fibonacci function."}],
    "max_tokens": 128,
    "temperature": 0.2
  }' | jq
```

**Tool calling**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is the weather in Bangkok?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather by city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
      }
    }],
    "tool_choice": "required"
  }' | jq
```

**Thinking / reasoning**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Explain cache layers briefly."}],
    "thinking": {"enabled": true, "summary": "auto"},
    "reasoning_effort": "medium"
  }' | jq
```

Response includes `choices[0].message.thinking` and `usage.reasoning_tokens`.

**Cache stats**

```bash
curl http://127.0.0.1:8000/v1/cache/stats
```

## CLI Options

```
amlx serve [OPTIONS]

  --model TEXT                 Configured model identifier
  --host TEXT                  Bind host (default: 127.0.0.1)
  --port INT                   Bind port (default: 8000)
  --cache-dir PATH             Cache directory (default: ~/.amlx/cache)
  --models-dir PATH            Model storage directory (default: ~/.amlx/models)
  --max-memory-cache-items INT In-memory LRU cache capacity (default: 512)
  --max-batch-size INT         Scheduler batch size (default: 8)
  --batch-wait-ms INT          Scheduler flush wait in ms (default: 20)
  --block-chars INT            Paged block size in chars (default: 4096)
  --gpu-limit-percent INT      GPU duty cycle cap 20–100% (default: 100)
  --log-level TEXT             Uvicorn log level (default: info)
```

**Tune for throughput**

```bash
amlx serve \
  --max-batch-size 12 \
  --batch-wait-ms 25 \
  --max-memory-cache-items 1024 \
  --block-chars 4096
```

**Custom paths**

```bash
amlx serve \
  --models-dir ~/.amlx/models \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

## Training Guidelines

1. **One intent per sample** — each record should represent a single behavior
2. **Deterministic phrasing** — stable wording makes model behavior easier to steer
3. **Explicit tool intent cues** — e.g. `"If user asks weather, call get_weather with {\"city\":\"<city>\"}."`
4. **Pair intent with argument shape** — include valid JSON matching your tool schema
5. **Separate policy from facts** — keep response style rules and domain facts in different samples
6. **Start small** — 20–50 clean samples, train, test, then expand
7. **Conservative epochs** — start with 1–2; raise only when behavior is still weak
8. **Runtime contract** — always send `tools` + `tool_choice` in chat requests for reliable tool invocation

**Dataset format**

```jsonl
{"text": "If user asks weather in Bangkok, call get_weather with {\"city\":\"Bangkok\"}."}
{"prompt": "weather question", "response": "call get_weather with city JSON"}
{"instruction": "forecast intent", "output": "call get_weather_forecast with city and days"}
```

## Roadmap

- True paged KV cache blocks on SSD
- Model warm pool and prefill reuse
- Menu bar app
