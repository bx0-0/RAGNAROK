# ⚙️ Configuration & CLI Flags

All flags override `config/settings.env` values. Repeat `--model` to load multiple models.

## Model & GPU

| Flag | Description | Default |
|---|---|---|
| `--model <name>` | Ollama model or `hf.co/...` (repeatable for multi-model) | `qwen3.5:9b` |
| `--max-concurrent <n>` | Max simultaneous requests before 429 rejection | `1` |
| `--num-ctx <n>` | Context window in tokens | `16384` |
| `--num-predict <n>` | Max tokens to generate per response | `16384` |
| `--num-batch <n>` | Decoding batch size for throughput tuning | `2444` |
| `--flash-attn <bool>` | Enable flash attention (reduces VRAM usage) | `True` |
| `--num-gpu <n>` | GPU layers (-1 = all on GPU, 0 = CPU only) | `-1` |
| `--keep-alive <dur>` | Keep model in RAM after last request (`60m`, `-1` = forever) | `60m` |

## Server

| Flag | Description | Default |
|---|---|---|
| `--port <n>` | FastAPI listening port | `8000` |
| `--debug` | Enable debug-level logging | off |
| `--verbose-log` | Live request log printed to terminal in real-time | off |

## TTS

| Flag | Description | Default |
|---|---|---|
| `--tts-enabled <bool>` | Enable or disable all TTS endpoints | `true` |
| `--tts-engine <name>` | Default engine when `"model"` is omitted: `omnivoice` or `inflect` | `omnivoice` |
| `--tts-device <gpu>` | OmniVoice device placement: `cuda` or `cpu` | `cuda` |
| `--tts-variant <v>` | Inflect model size: `nano` (~16MB) or `micro` (~38MB) | `nano` |
| `--gc-timeout <s>` | Seconds before an idle model is auto-evicted (0 = never) | `600` |

## settings.env Reference

Edit `config/settings.env` for persistent configuration. CLI flags take precedence:

```env
# === Model ===
MODEL_NAME=qwen3.6:35b
MAX_CONCURRENT=3
NUM_CTX=100000
NUM_PREDICT=16384
NUM_BATCH=3000
FLASH_ATTN=True
KEEP_ALIVE=60m

# === Server ===
PORT=8000

# === TTS ===
TTS_ENABLED=true
TTS_DEFAULT_ENGINE=omnivoice
TTS_OMNIVOICE_DEVICE=cuda
TTS_INFLECT_VARIANT=nano
TTS_MAX_CHARS=5000

# === GC ===
GC_IDLE_TIMEOUT=600
GC_SWEEP_INTERVAL=60
```

### Config Resolution Order

Values are resolved in this priority (highest first):

1. **CLI flag** on `bash start.sh` command line
2. **settings.env** file (`source config/settings.env`)
3. **Hardcoded default** in `start.sh` variable declaration
