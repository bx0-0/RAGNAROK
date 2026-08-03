# CLI Flags

All flags override `config/settings.env` values. Repeat `--model` for multiple models.

## Model & GPU

| Flag | Description | Default |
|---|---|---|
| `--model <name>` | Ollama model or `hf.co/...` (repeatable) | `qwen3.5:9b` |
| `--max-concurrent <n>` | Max simultaneous requests | `1` |
| `--num-ctx <n>` | Context window tokens | `16384` |
| `--num-predict <n>` | Max generation tokens | `16384` |
| `--num-batch <n>` | Decoding batch size | `2444` |
| `--flash-attn <bool>` | Enable flash attention | `True` |
| `--num-gpu <n>` | GPU layers (-1 = all) | `-1` |
| `--keep-alive <dur>` | Model keep-alive duration | `60m` |

## Server

| Flag | Description | Default |
|---|---|---|
| `--port <n>` | FastAPI port | `8000` |
| `--debug` | Enable debug logging | off |
| `--verbose-log` | Live request log in terminal | off |

## TTS

| Flag | Description | Default |
|---|---|---|
| `--tts-enabled <bool>` | Enable/disable TTS | `true` |
| `--tts-engine <name>` | Default engine: `omnivoice` or `inflect` | `omnivoice` |
| `--tts-device <gpu>` | OmniVoice device: `cuda` or `cpu` | `cuda` |
| `--tts-variant <v>` | Inflect variant: `nano` or `micro` | `nano` |
| `--gc-timeout <s>` | GC idle eviction timeout (seconds) | `600` |
