# 🔌 API Reference — All Endpoints

Full OpenAI-compatible endpoint reference. Base URL example: `https://YOUR-URL.trycloudflare.com`

---

## Chat Completions

**`POST /v1/chat/completions`**

Standard OpenAI chat completions with full streaming, tool calls, and function calling support.

### Request Body

Same as [OpenAI API](https://platform.openai.com/docs/api-reference/chat):

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | ✅ | Model name (e.g. `qwen3.5:9b`) |
| `messages` | array | ✅ | Array of `{role, content}` message objects |
| `stream` | boolean | ❌ | Enable SSE streaming (default: `false`) |
| `max_tokens` | integer | ❌ | Max tokens to generate |
| `tools` | array | ❌ | OpenAI-compatible tool/function definitions |
| `reasoning_effort` | string | ❌ | Thinking level: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `thinking` | boolean | ❌ | Enable/disable thinking (bool models). `reasoning_effort` wins if both set |

### Example

```bash
curl -X POST https://YOUR-URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs"}],
    "stream": true,
    "reasoning_effort": "low"
  }'
```

### Reasoning Effort

`reasoning_effort` maps to Ollama's top-level `think` field. Mapping lives in one place — `THINK_LEVEL_MAP` in `src/config.py` — so adding a new level is a one-line edit:

| Client sends | Ollama `think` | Notes |
|---|---|---|
| `"none"` | `false` | thinking off |
| `"minimal"` | `"low"` | |
| `"low"` | `"low"` | |
| `"medium"` | `"medium"` | |
| `"high"` | `"high"` | |
| `"xhigh"` | `"high"` | |
| anything else | `true` | model default level |
| *(omitted)* | *(omitted)* | Ollama uses the model default |

The reasoning trace comes back as `reasoning_content` on the message (non-stream) or as `delta.reasoning_content` chunks (stream), interleaved before the final content — same as DeepSeek/OpenAI-style APIs.

> Models without thinking support simply ignore the field. Requires `ollama>=0.5.0` client.

---

## Models

**`GET /v1/models`**

List all loaded models with their metadata.

### Example

```bash
curl https://YOUR-URL/v1/models
```

### Response

```json
{
  "object": "list",
  "data": [
    {"id": "qwen3.6:35b", "object": "model", "owned_by": "local"}
  ]
}
```

---

## Embeddings

**`POST /v1/embeddings`**

Generate vector embeddings for text via Ollama's built-in embedding API.

### Example

```bash
curl -X POST https://YOUR-URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

---

## Health Check

**`GET /health`**

Server status, loaded models, and default model.

### Example

```bash
curl https://YOUR-URL/health
```

### Response

```json
{
  "status": "ready",
  "models": ["qwen3.6:35b"],
  "default": "qwen3.6:35b"
}
```

---

## TTS Endpoints

See [TTS API Reference](tts-api.md) for full documentation including voice reference and examples.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `POST` | `/v1/audio/unload` | Unload engine(s) from memory |
| `GET`  | `/v1/audio/engines` | List available engines + GC status |

---

## Error Responses

All endpoints return standard HTTP status codes:

| Code | Meaning | When |
|---|---|---|
| 200 | OK | Success |
| 400 | Bad Request | Invalid JSON, missing required fields |
| 429 | Too Many Requests | Server busy (concurrency limit reached) |
| 500 | Internal Error | Model generation failed, TTS engine crashed |
