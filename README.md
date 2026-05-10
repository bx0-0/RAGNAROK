# Kaggle Ollama Gateway

Run powerful open-source LLMs on **free Kaggle GPUs** with a public OpenAI-compatible API endpoint.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/runs%20on-Kaggle-20BEFF)](https://www.kaggle.com)

---

## What does this do?

1. Runs **Ollama** inside a Kaggle Notebook (free 2x T4 GPUs, 30GB VRAM)
2. Wraps Ollama's API with an **OpenAI-compatible** endpoint
3. Exposes it publicly via a **Cloudflare Tunnel**
4. You get a working `https://...trycloudflare.com/v1` endpoint for any OpenAI-compatible tool

Works with: **Claude Code, Codex, OpenCode, Cursor, VSCode AI extensions**, and any OpenAI agent framework.

---

## Quick Start

### On Kaggle

```bash
git clone https://github.com/yourusername/kaggle-ollama-gateway.git
cd kaggle-ollama-gateway

# Option A: Interactive config file
nano config/settings.env    # edit model & settings
bash start.sh

# Option B: CLI arguments
bash start.sh --model qwen3:8b --max-concurrent 2
```

That's it. You'll get a public URL like:
```
🌐  https://abcd1234.trycloudflare.com/v1
```

### Use the endpoint

```bash
curl https://YOUR-URL.trycloudflare.com/v1/models

curl -X POST https://YOUR-URL.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Project Structure

```
kaggle-ollama-gateway/
├── start.sh                  # Main entry — run this
├── config/
│   └── settings.env          # Edit model, tokens, concurrency
├── scripts/
│   ├── setup.sh              # Install dependencies
│   ├── install_model.sh      # Pull Ollama model
│   └── tunnel.sh             # Cloudflare tunnel + watchdog
├── src/
│   ├── __init__.py
│   ├── server.py             # FastAPI OpenAI-compatible server
│   ├── utils.py              # Helpers: message conversion, JSON
│   └── logging.py            # Logging config
├── examples/
│   ├── test_api.py           # Quick API test
│   └── claude_code.md        # Claude Code integration guide
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

Edit `config/settings.env`:

```env
# Model to load (must be available on Ollama)
MODEL_NAME=qwen3:8b

# Max concurrent requests (Kaggle T4: 1-2 recommended)
MAX_CONCURRENT=1

# Context window size
NUM_CTX=68768

# Max tokens to generate
NUM_PREDICT=16384

# Batch size
NUM_BATCH=2444

# Keep model loaded (set to -1 for always, or "60m" for 60 min)
KEEP_ALIVE=60m
```

Or override via CLI:
```bash
bash start.sh --model llama3.3:70b --max-concurrent 2 --num-ctx 32768
```

---

## Supported Models (Kaggle T4 friendly)

| Model | Size | Quality | Load Time |
|-------|------|---------|-----------|
| `qwen3:8b` | 8B | ⭐⭐⭐ | ~3 min |
| `qwen2.5:7b` | 7B | ⭐⭐⭐ | ~3 min |
| `llama3.3:70b` | 70B | ⭐⭐⭐⭐ | ~15 min |
| `mistral:7b` | 7B | ⭐⭐ | ~3 min |
| `deepseek-r1:7b` | 7B | ⭐⭐⭐ | ~3 min |

---

## Examples

### Claude Code

```bash
claude --api-model claude-sonnet-4-20250514 \
  --api-base https://YOUR-URL.trycloudflare.com/v1 \
  --api-key any-value
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-URL.trycloudflare.com/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

See `examples/` for more.

---

## Troubleshooting

### Model fails to download
- Check Kaggle internet: `!ping -c 1 ollama.com`
- Retry: `bash scripts/install_model.sh`
- Try a smaller model

### Tunnel URL not appearing
- Wait up to 60 seconds
- Check: `!ps aux | grep cloudflared`
- Restart: `bash scripts/tunnel.sh`

### Port 8000 already in use
```bash
fuser -k 8000/tcp
```

### Ollama won't start
- Check disk: `!df -h`
- Free space: `!rm -rf /root/.cache/`
- Restart: `pkill ollama && ollama serve`

### "Server is busy" (429 error)
- Only 1 request at a time by default
- Set `MAX_CONCURRENT=2` in config for 2 simultaneous

---

## License

MIT
