# TTS API Reference

All engine parameters are sent in the request body — no CLI args needed.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `POST` | `/v1/audio/unload` | Unload engine(s) from memory |
| `GET`  | `/v1/audio/engines` | List available engines + status |

## Speech Request Body

```json
{
  "model": "omnivoice",           // engine name: omnivoice | inflect
  "input": "Hello world!",       // text to synthesize
  "response_format": "wav",      // wav (default) | mp3
  "voice_instruct": "male, young adult",  // OmniVoice only
  "num_step": 16,                // OmniVoice: 8-32 diffusion steps
  "guidance_scale": 2.0,         // OmniVoice: 0.1-5.0 CFG scale
  "speed": 1.0,                  // Both engines: 0.25-4.0
  "variation": 0.667,            // Inflect only: 0.0-1.0
  "seed": 7                      // Inflect only: reproducibility
}
```

### Engine-Specific Params

**OmniVoice** — `voice_instruct`, `num_step`, `guidance_scale`

Valid English items: `american accent`, `australian accent`, `british accent`, `canadian accent`, `child`, `chinese accent`, `elderly`, `female`, `high pitch`, `indian accent`, `japanese accent`, `korean accent`, `low pitch`, `male`, `middle-aged`, `moderate pitch`, `portuguese accent`, `russian accent`, `teenager`, `very high pitch`, `very low pitch`, `whisper`, `young adult`

Valid Chinese items: `东北话`, `中年`, `中音调`, `云南话`, `低音调`, `儿童`, `四川话`, `女`, `宁夏话`, `少年`, `极低音调`, `极高音调`, `桂林话`, `河南话`, `济南话`, `甘肃话`, `男`, `石家庄话`, `老年`, `耳语`, `贵州话`, `陕西话`, `青岛话`, `青年`, `高音调`

> Use only English **or** only Chinese. Join English with `, ` (comma+space), Chinese with `，` (full-width comma).

**Inflect** — `variation`, `seed`

## Examples

### curl

```bash
# OmniVoice (Arabic)
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "السلام عليكم",
    "voice_instruct": "male, young adult",
    "speed": 0.9,
    "num_step": 24
  }' --output out.wav

# Inflect (English)
curl -X POST https://YOUR-URL/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inflect",
    "input": "Hello world!",
    "speed": 1.2,
    "variation": 0.8
  }' --output out.wav
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-URL/v1",
    api_key="not-needed",
)

resp = client.audio.speech.create(
    model="omnivoice",
    input="Hello world!",
    voice_instruct="female, british accent",
    speed=1.2,
)
resp.write_to_file("speech.wav")
```

## Memory Management

Models load lazily on first request and auto-evict after idle timeout (default 600s).

```bash
# Unload a specific engine
curl -X POST https://YOUR-URL/v1/audio/unload \
  -d '{"model": "omnivoice"}'

# Unload all TTS engines
curl -X POST https://YOUR-URL/v1/audio/unload \
  -d '{}'

# Check status
curl https://YOUR-URL/v1/audio/engines
```
