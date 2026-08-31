# <img src="../assets/folder.png" width="30" align="middle"> Project Structure & Architecture

```
kaggle-ollama-gateway/
├── start.sh                  # Main entry — run this
├── assets/
│   └── RAGNAROK.png          # Logo
├── config/
│   └── settings.env          # Model, tokens, concurrency settings
├── scripts/
│   ├── setup.sh              # Install Ollama + Python deps + cloudflared
│   ├── install_model.sh      # Pull Ollama model(s)
│   ├── install_tts.sh        # Download TTS models (OmniVoice + Inflect)
│   ├── tunnel_orchestrate.sh # Cloudflare tunnel lifecycle manager
│   └── tunnel_*.sh           # Tunnel healthcheck, start, watchdog helpers
├── src/
│   ├── server.py             # FastAPI app + lifespan + uvloop config
│   ├── config.py             # Env-based configuration (TTS, GC settings)
│   ├── gc.py                 # Model garbage collector — auto-evicts idle models
│   ├── state.py              # Async Ollama client, semaphore, warmup
│   ├── routes/               # OpenAI-compatible endpoints
│   │   ├── chat.py           # POST /v1/chat/completions (stream + non-stream)
│   │   ├── tts.py            # POST /v1/audio/speech + unload/list engines
│   │   ├── models.py         # GET /v1/models
│   │   ├── embeddings.py     # POST /v1/embeddings
│   │   └── health.py         # GET /health
│   ├── models/               # Pydantic request/response schemas
│   │   ├── chat.py           # ChatCompletionRequest
│   │   ├── shared.py         # Message, Choice, Tool schemas
│   │   └── tts.py            # SpeechRequest (per-request TTS params)
│   ├── tts/                  # TTS plugin system
│   │   ├── __init__.py       # Engine registry — add engines here
│   │   ├── base.py           # AbstractTTSEngine interface
│   │   ├── omnivoice_engine.py  # OmniVoice (600+ langs, voice design)
│   │   └── inflect_engine.py    # Inflect v2 (English only, ~16MB)
│   ├── streaming.py          # SSE generator with token batching + retries
│   ├── sse.py                # SSE protocol helpers + envelope caching
│   ├── utils.py              # OpenAI ↔ Ollama message conversion
│   ├── errors.py             # Centralized error responses
│   └── logging.py            # Async-safe dual logging (file + verbose stdout)
├── tests/                    # Test suite
├── examples/
│   └── claude_code.md        # Integration guides
├── requirements.txt
└── README.md
```

## <img src="../assets/data.png" width="40" align="middle"> How It Works

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

### Data Flow

1. **Client sends** an OpenAI-compatible `POST /v1/chat/completions` request
2. **Cloudflare Tunnel** decrypts HTTPS → forwards as HTTP to FastAPI on port 8000
3. **FastAPI** converts the OpenAI format to Ollama's internal API, applies concurrency limits
4. **Ollama** runs inference on the GPU and streams tokens back
5. **Streaming layer** batches tokens into SSE frames with 100ms accumulation windows

## <img src="../assets/animal.png" width="40" align="middle"> Key Features

- **Semaphore-based concurrency control** — immediate 429 rejection when busy, no queueing
- **Token batching** — 100ms SSE accumulation reduces network overhead; tool calls force immediate flush
- **Queue-based stream consumer** — decouples Ollama generator from SSE yields to prevent coroutine nesting
- **SSE envelope caching** — `lru_cache(16)` pre-builds JSON templates per model, replaces only deltas at runtime
- **Automatic retry** — up to 2 retries on empty streams or upstream crashes (configurable via `RETRY_ON_EMPTY`)
- **Tool use support** — full OpenAI function calling with system prompt injection for chunked file writing
- **Client-disconnect cancellation** — a `DisconnectAwareStreamingResponse` watches the ASGI `http.disconnect` event in parallel with the body stream; on disconnect it cancels the streaming task, which cancels the Ollama pusher and closes the per-request stream so generation stops immediately. The shared Ollama client is never closed; the concurrency slot is released exactly once.
- **Generation control** — per-request stop (POST /v1/chat/completions/{request_id}/stop) and model unload (POST /v1/models/unload). Stop cancels one generation; unload stops that model's generations and evicts weights via keep_alive: 0. Separate operations.

## TTS Plugin Architecture

TTS engines implement `AbstractTTSEngine` (`load`, `synthesize`, `unload`, `is_loaded`) and self-register at import time. Adding a new engine requires **zero edits** to existing code — just drop a new `.py` in `src/tts/` and call `register_self()`.

```
Request → routes/tts.py → registry.get(name) → engine.synthesize(text, **kwargs) → WAV bytes
                                         ↓
                                   ModelGC tracks idle time → auto-evicts after timeout
```

See [TTS API Reference](tts-api.md) for endpoints and examples.

## Garbage Collection

`ModelGC` is a singleton that monitors loaded models (LLM + TTS) and auto-evicts those idle beyond `GC_IDLE_TIMEOUT` seconds (default 600s / 10 min). Sweep runs every `GC_SWEEP_INTERVAL` seconds in the background.

- Models are **touched** on every request, resetting the idle timer
- Manual unload via `POST /v1/audio/unload` or GC sweep both go through the same path
- Eviction frees GPU/CPU memory for the next incoming request



## Tests for streaming control

tests/test_disconnect.py exercises the full cancel lifecycle against a fake Ollama
(tests/fake_ollama.py): normal streaming, FIN disconnect, RST disconnect, explicit stop,
unload, and concurrent-generation isolation. Run with a fake Ollama on :11434 and the gateway
on :8120:

```bash
FAKE_PORT=11434 python3 tests/fake_ollama.py &   # fake Ollama
PORT=8120 python3 -m src &                        # gateway
python3 tests/test_disconnect.py
```

> Verified **against the fake Ollama** (abort-on-client-close is emulated). Not a real-GPU
> VRAM-release test.