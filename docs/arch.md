# Architecture & Features

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

## Core Features

- **Semaphore-based concurrency control** — immediate 429 rejection when busy
- **Token batching** — 100ms SSE accumulation reduces network overhead; tool calls force immediate flush
- **Queue-based stream consumer** — decouples Ollama generator from SSE yields to prevent coroutine nesting
- **SSE envelope caching** — `lru_cache(16)` pre-builds JSON templates per model, replaces only deltas at runtime
- **Automatic retry** — up to 2 retries on empty streams or upstream crashes (configurable via `RETRY_ON_EMPTY`)
- **Tool use support** — full OpenAI function calling with system prompt injection for chunked file writing

## Garbage Collection

Auto-evicts idle models after configurable timeout (`GC_IDLE_TIMEOUT` env var, default 600s). Sweep loop checks every `GC_SWEEP_INTERVAL` seconds.

See [src/gc.py](../src/gc.py) and `--gc-timeout` CLI flag.

## TTS Plugin System

Engines implement `AbstractTTSEngine` (`load`, `synthesize`, `unload`, `is_loaded`) and self-register at import time via `register_self()`. Adding a new engine requires zero edits to existing code.

```
src/tts/
├── __init__.py          # registry + register_self() calls
├── base.py              # AbstractTTSEngine interface
├── omnivoice_engine.py  # OmniVoice (600+ langs)
└── inflect_engine.py    # Inflect v2 (English, CPU)
```

See [TTS API Reference](tts-api.md).

## Project Structure

```
src/
├── config.py            # env var defaults + CLI overrides
├── gc.py                # ModelGC singleton
├── logging.py           # verbose request log
├── server.py            # FastAPI lifespan, startup
├── sse.py               # SSE framing
├── state.py             # Ollama state + warmup
├── streaming.py         # async stream handling
├── utils.py             # message conversion helpers
├── errors.py            # centralized error responses
├── models/              # Pydantic request/response schemas
├── routes/              # FastAPI APIRouter modules
└── tts/                 # TTS engine plugins
scripts/
├── setup.sh             # install Ollama, cloudflared, deps
└── install_tts.sh       # download TTS models + deps
```
