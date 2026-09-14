# ⚙️ Configuration & CLI Flags

All flags override `config/settings.env` values. Repeat `--model` to load multiple models.

## Model & GPU

| Flag | Description | Default |
|---|---|---|
| `--model <name>` | Ollama model or `hf.co/...` (repeatable for multi-model) | `qwen3.5:9b` |
| `--max-concurrent <n>` | Max simultaneous requests before 429 rejection | `2` |
| `--num-ctx <n>` | Context window in tokens | `16384` |
| `--num-predict <n>` | Max tokens to generate per response | `16384` |
| `--num-batch <n>` | Decoding batch size for throughput tuning | `500` |
| `--draft-num-predict <n>` | Draft tokens per step for MTP speculative decoding (sent as an Ollama option) | `4` |
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
DRAFT_NUM_PREDICT=4
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
TTS_MIN_GPU_FREE_GB=7

# === GC ===
GC_IDLE_TIMEOUT=600
GC_SWEEP_INTERVAL=60
```

### Config Resolution Order

Values are resolved in this priority (highest first):

1. **CLI flag** on `bash start.sh` command line
2. **settings.env** file (`source config/settings.env`)
3. **Hardcoded default** in `start.sh` variable declaration

---

## ragnrok CLI

The `ragnrok` command (repo root) is a lightweight Bash CLI. No Python CLI framework or new dependencies — `create` reuses the gateway's existing `ModelManager.create_from_modelfile` (the `ollama create -f` path that accepts the custom `RENDERER` / `PARSER` directives).

| Command | Description |
|---|---|
| `ragnrok hf --download REPO [FILE...] [--max-workers N]` | Download file(s) from a Hugging Face repo into RAGNAROK storage |
| `ragnrok --list` | List RAGNAROK-managed files (NAME / TYPE / SIZE), with LOCATION at the bottom |
| `ragnrok create model.gguf [--parser NAME] [--render NAME] [--name NAME]` | Create an Ollama model from a local GGUF |
| `ragnrok create --mtp model.gguf [assoc...] [--parser NAME] [--render NAME] [--name NAME]` | MTP / associated-file mode: first file is the main model, the rest are associated files |

### Flags

| Flag | Requires value | Description |
|---|---|---|
| `--download` | REPO (then optional FILE...) | `hf` subcommand — repository to download |
| `--max-workers N` | yes | Parallel download workers |
| `--mtp` | no (mode flag) | Treats 2nd+ positionals as associated files (generic pass-through; does not classify them). Also adds `PARAMETER draft_num_predict 4` to the Modelfile |
| `--parser NAME` | **yes** | Set the PARSER (a renderer/parser name); validated as a simple identifier |
| `--render NAME` | **yes** | Set the RENDERER (a renderer/parser name); validated as a simple identifier |
| `--name NAME` | **yes** | Override the derived Ollama model name |

> `--parser` / `--render` take an **explicit value** (e.g. `qwen3.8`) — they are not boolean toggles. Values are validated as a simple identifier (alnum, dot, dash, underscore).
>
> **Storage location:** Kaggle → `/kaggle/working/.ragnrok`, Colab → `/tmp/.ragnrok`, local → `~/.ragnrok`. Override with `RAGNROK_STORAGE_DIR`.
