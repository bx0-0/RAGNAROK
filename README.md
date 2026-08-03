<p align="center">
  <img src="assets/RAGNAROK.png" alt="RAGNAROK" width="370">
</p>

<h1 align="center">
  <img src="assets/dragon.png" width="50" align="center"> RAGNAROK — GPU Model Gateway
</h1>

<p align="center">
  Run open-source LLMs on <strong>free Kaggle / Colab GPUs</strong> with a public OpenAI-compatible API.
  Now with <strong>TTS</strong> (text-to-speech) and <strong>auto garbage collection</strong>.
</p>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/runs%20on-Kaggle-20BEFF)](https://www.kaggle.com)
[![Colab](https://img.shields.io/badge/runs%20on-Colab-F9AB00)](https://colab.research.google.com)
[![TTS](https://img.shields.io/badge/TTS-OmniVoice%20%2B%20Inflect-purple.svg)](docs/tts-api.md)

</div>

---

## Quick Start

```bash
git clone https://github.com/bx0-0/RAGNAROK.git && cd RAGNAROK
bash start.sh --model qwen3.5:9b
```

Wait for the tunnel URL, then test:

```bash
# Chat (OpenAI-compatible)
curl https://YOUR-URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Hello!"}]}'

# TTS speech
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"مرحبا بالعالم","voice_instruct":"male, young adult"}' \
  --output hello.wav
```

## Features

| Feature | Details |
|---|---|
| OpenAI-compatible API | Chat completions, streaming, tools, embeddings |
| TTS (OmniVoice) | 600+ languages, voice design, GPU-accelerated |
| TTS (Inflect v2) | English-only, runs on CPU, ~16-38 MB models |
| Auto GC | Idle model eviction to prevent OOM |
| Cloudflare tunnel | Free public HTTPS endpoint from Kaggle/Colab |

## Documentation

| Topic | Link |
|---|---|
| **Platform Setup** (Kaggle / Colab / Local) | [docs/install.md](docs/install.md) |
| **CLI Flags** & Config | [docs/cli-flags.md](docs/cli-flags.md) |
| **TTS API** & Voice Reference | [docs/tts-api.md](docs/tts-api.md) |
| **API Reference** (all endpoints) | [docs/api-reference.md](docs/api-reference.md) |
| **Architecture** & Plugin System | [docs/arch.md](docs/arch.md) |

## Testing

```bash
# Unit tests (no server needed)
python -m pytest tests/test_sse.py tests/test_utils.py -v

# Live tests (requires running gateway)
python -m pytest tests/test_live.py -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Model fails to download | `bash scripts/install_model.sh` |
| Tunnel URL missing | Wait 60s; check `cat /tmp/cloudflared.log \| tail -20` |
| Port 8000 in use | `fuser -k 8000/tcp` |
| "Server is busy" (429) | Use `--max-concurrent 3` |
| Server logs | `cat /tmp/gateway-server.log` |

---

**[MIT License](LICENSE)** — Created by **Saber Mohamed**
