<p align="center">
  <img src="assets/RAGNAROK.png" alt="RAGNAROK" width="370">
</p>

<h1 align="center">
  <img src="assets/dragon.png" width="50" align="center"> RAGNAROK — GPU Model Gateway
</h1>

<p align="center">
  Run powerful open-source LLMs on <strong>free Kaggle / Colab GPUs</strong> with a public OpenAI-compatible API.
  Now with <strong>TTS</strong> (text-to-speech) and <strong>auto garbage collection</strong> for memory safety.
</p>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/runs%20on-Kaggle-20BEFF)](https://www.kaggle.com)
[![Colab](https://img.shields.io/badge/runs%20on-Colab-F9AB00)](https://colab.research.google.com)
[![TTS](https://img.shields.io/badge/TTS-OmniVoice%20%7C%20Inflect-purple.svg)](docs/tts-api.md)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-white.svg?logo=ollama&labelColor=white)](https://ollama.com)
[![OmniVoice](https://img.shields.io/badge/OmniVoice-600%2B_Langs-blue.svg)](https://huggingface.co/k2-fsa/OmniVoice)
[![Inflect](https://img.shields.io/badge/Inflect-CPU_English-orange.svg)](https://huggingface.co/owensong/Inflect-Nano-v2)
[![GC](https://img.shields.io/badge/GC-Auto%20Eviction-green.svg)]()

</div>

---

## <img src="assets/animal.png" width="60" align="middle"> Overview

1. Runs **Ollama** inside Kaggle / Colab notebooks (free GPUs)
2. Wraps Ollama's API with an **OpenAI-compatible** endpoint
3. Exposes it publicly via a **Cloudflare Tunnel**
4. You get a working `https://*.trycloudflare.com/v1` URL for any OpenAI client
5. **TTS endpoints** — text-to-speech with OmniVoice (600+ languages) or Inflect v2 (English, CPU-only)
6. **Auto GC** — idle models auto-evict from memory after configurable timeout
7. **Streaming control** — client disconnects cancel the generation; list active generations + stop/unload endpoints for ops

Works with: **Codex · OpenCode · Cursor · VSCode AI extensions · Pi Agent · any OpenAI agent framework**

---

## <img src="assets/fire.png" width="50"> Quick Start (Kaggle / Colab)

Run these commands directly in a notebook cell:

```bash
!git clone https://github.com/bx0-0/RAGNAROK.git
%cd RAGNAROK
!bash start.sh --model qwen3.6:35b --verbose-log True --num-ctx 100000
```

You'll receive a public URL like:

```
🌐  https://pct-drums-partnerships-chosen.trycloudflare.com/v1
```

### Use the endpoint

```bash
# List models
curl https://YOUR-URL.trycloudflare.com/v1/models

# Chat completion (streaming)
curl -X POST https://YOUR-URL.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# TTS speech generation
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"مرحبا بالعالم","voice_instruct":"male, young adult"}' \
  --output hello.wav
```

---

## <img src="assets/written.png" width="50"> Examples

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-URL.trycloudflare.com/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="qwen3.6:35b",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

### Pi Agent Configuration

Add the gateway as a custom provider in `.pi/agent/models.json`:

```json
{
  "providers": {
    "myapi2": {
      "baseUrl": "https://YOUR-URL.trycloudflare.com/v1",
      "api": "openai-completions",
      "apiKey": "sk-anything",
      "models": [{"id": "qwen3.6:35b", "name": "Qwen 27B", "contextWindow": 80000, "input": ["text"]}]
    }
  }
}
```

---

## <img src="assets/search.png" width="50" align="middle"> ragnrok CLI

A lightweight Bash CLI for managing models and files on Kaggle / Colab / local — Hugging Face downloads, a managed file listing, and Ollama model creation (including MTP + associated files). No Python dependencies; `create` reuses the gateway's existing model-management path.

```bash
# Download model file(s) from Hugging Face into RAGNAROK storage
./ragnrok hf --download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF \
    Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf --max-workers 8

# List RAGNAROK-managed files
./ragnrok --list

# Create an Ollama model with a renderer/parser
./ragnrok create model.gguf --parser my-parser --render my-renderer

# MTP: first file = main model, the rest are associated files
./ragnrok create --mtp model.gguf mmproj-1.gguf mmproj-2.gguf --parser p --render r
```

> `--parser NAME` / `--render NAME` each require an explicit value (a renderer/parser name). `--mtp` is a mode flag: it does not classify files. See [docs/cli-flags.md](docs/cli-flags.md#ragnrok-cli).

---

## Documentation Index

| Topic | Link | Description |
|---|---|---|
| 📂 **Project Structure** | [docs/arch.md](docs/arch.md) | Folder layout, architecture diagram, GC system |
| ⚙️ **Configuration & CLI** | [docs/cli-flags.md](docs/cli-flags.md) | All CLI flags, env vars, settings.env reference |
| 🔍 **Model Sources** | [docs/install.md](docs/install.md) | Ollama library, HF GGUF, platform setup (Kaggle/Colab/local) |
| 🎙️ **TTS API** | [docs/tts-api.md](docs/tts-api.md) | Speech endpoints, voice_instruct reference, examples |
| 🔌 **API Reference** | [docs/api-reference.md](docs/api-reference.md) | All OpenAI-compatible endpoints |
| 🛑 **Streaming Control** | [docs/api-reference.md](docs/api-reference.md#streaming-lifecycle--control) | Disconnect cancellation, stop / unload endpoints |
| 🖥️ **Platform Guides** | [docs/install.md](docs/install.md) | Kaggle 30GB, Colab 15GB, local Linux setup |
| 🧰 **ragnrok CLI** | [docs/cli-flags.md](docs/cli-flags.md#ragnrok-cli) | HF downloads, file listing, model creation (incl. MTP) |

---

## <img src="assets/troubleshooting.png" width="50" align="middle"> Troubleshooting

| Problem | Fix |
|---|---|
| Model fails to download | `!ping -c 1 ollama.com` → `bash scripts/install_model.sh` |
| Tunnel URL not appearing | Wait 60s → `cat /tmp/cloudflared.log \| tail -20` |
| Port 8000 already in use | `fuser -k 8000/tcp` |
| "Server is busy" (429) | Set `MAX_CONCURRENT=3` or use `--max-concurrent 3` |
| Ollama won't start | `df -h` → `rm -rf /root/.cache/` → `pkill ollama && ollama serve` |

### Server Logs

```bash
cat /tmp/gateway-server.log       # Server stdout/stderr
cat /tmp/gateway-requests.log     # Request log (if VERBOSE_LOG=True)
cat /tmp/ollama-pull.log          # Model download progress
```

---

## <img src="assets/data.png" width="50"> Architecture

```
Client (OpenAI SDK / Pi Agent / curl)
    │  HTTPS
    ▼
Cloudflare Tunnel (cloudflared)
    │  HTTP
    ▼
FastAPI Server (uvloop + httptools)
    │  Async Ollama client (connection pool: 2000 max)
    ▼
Ollama (localhost:11434) ──→ GPU inference
```

---

**[MIT License](LICENSE)** — Created by **Saber Mohamed**
