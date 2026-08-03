# 🎙️ TTS API Reference

Two engines available via a plugin system — **OmniVoice** (600+ languages, GPU) and **Inflect v2** (English only, CPU).

All engine parameters are sent in the request body. No CLI args needed for per-request tuning.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `POST` | `/v1/audio/unload` | Unload engine(s) from memory |
| `GET`  | `/v1/audio/engines` | List available engines + GC status |

---

## Speech Request Schema

### Required Fields

| Field | Type | Description |
|---|---|---|
| `model` | string | Engine name: `"omnivoice"` or `"inflect"` |
| `input` | string | Text to synthesize (max 5000 chars by default) |

### Optional Fields (per-request overrides)

| Field | Type | Default | Engine(s) | Description |
|---|---|---|---|---|
| `response_format` | string | `"wav"` | Both | Output format: `wav` or `mp3` |
| `speed` | float | `1.0` | Both | Playback speed (0.25–4.0) |
| `voice_instruct` | string | `"male, young adult"` | OmniVoice | Voice design tags |
| `num_step` | int | `16` | OmniVoice | Diffusion steps: 8 (fast) → 32 (quality) |
| `guidance_scale` | float | `2.0` | OmniVoice | Classifier-free guidance: 0.1–5.0 |
| `variation` | float | `0.667` | Inflect | Prosody randomness: 0.0 (robotic) → 1.0 (expressive) |
| `seed` | int | `7` | Inflect | Random seed for reproducibility |

---

## OmniVoice Voice Design Reference

The `voice_instruct` field accepts comma-separated English or Chinese tags. **Never mix languages.**

### <img src="../assets/search.png" width="24" align="middle"> English Items

| Category | Tags |
|---|---|
| Gender | `male`, `female` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Pitch | `high pitch`, `moderate pitch`, `low pitch`, `very high pitch`, `very low pitch` |
| Accent | `american accent`, `australian accent`, `british accent`, `canadian accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |
| Special | `whisper` |

### Chinese Items

`东北话`, `中年`, `中音调`, `云南话`, `低音调`, `儿童`, `四川话`, `女`, `宁夏话`, `少年`, `极低音调`, `极高音调`, `桂林话`, `河南话`, `济南话`, `甘肃话`, `男`, `石家庄话`, `老年`, `耳语`, `贵州话`, `陕西话`, `青岛话`, `青年`, `高音调`

### Rules

- Use only English **or** only Chinese — never mix
- **English:** join with `, ` (comma + space), e.g. `'male, young adult'`
- **Chinese:** join with `，` (full-width comma), e.g. `'男，中年'`

---

## Examples

### <img src="../assets/fire.png" width="24" align="middle"> OmniVoice (Arabic)

```bash
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "السلام عليكم ورحمة الله",
    "voice_instruct": "female, british accent",
    "speed": 0.9,
    "num_step": 24,
    "guidance_scale": 2.5
  }' --output out.wav
```

### <img src="../assets/fire.png" width="24" align="middle"> Inflect (English)

```bash
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inflect",
    "input": "Hello world! This is a text-to-speech test.",
    "speed": 1.2,
    "variation": 0.8,
    "seed": 42
  }' --output out.wav
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-URL/v1",
    api_key="not-needed",
)

# OmniVoice
resp = client.audio.speech.create(
    model="omnivoice",
    input="Hello world!",
    voice_instruct="female, british accent",
    speed=1.2,
)
resp.write_to_file("speech.wav")
```

---

## Memory Management

TTS models load **lazily** on first request and auto-evict after idle timeout (default 600s / 10 min).

### Manual Unload

```bash
# Unload a specific engine
curl -X POST https://YOUR-URL/v1/audio/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "omnivoice"}'

# Unload all TTS engines at once
curl -X POST https://YOUR-URL/v1/audio/unload \
  -H "Content-Type: application/json" \
  -d '{}'

# Check which engines are loaded / unloaded
curl https://YOUR-URL/v1/audio/engines
```

### How It Works

```
First request → model downloads if missing → loads into memory → GC timer starts
Each request → resets idle timer (touch)
Idle > 600s   → GC sweep evicts model, frees memory
Next request  → reloads from cache (fast, no re-download)
```
