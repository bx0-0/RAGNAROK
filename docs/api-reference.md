# API Reference

Full OpenAI-compatible endpoints.

## Chat

**`POST /v1/chat/completions`**

Standard OpenAI chat completions. Supports streaming, tool calls, and function calling.

Request body: same as OpenAI API (`model`, `messages`, `stream`, `max_tokens`, `tools`, etc.)

```bash
curl -X POST https://YOUR-URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Models

**`GET /v1/models`**

List loaded models.

```bash
curl https://YOUR-URL/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {"id": "qwen3.6:35b", "object": "model", "owned_by": "local"}
  ]
}
```

## Embeddings

**`POST /v1/embeddings`**

Generate embeddings via Ollama's embedding API.

## Health

**`GET /health`**

Server status and loaded model list.

```bash
curl https://YOUR-URL/health
```

Response:
```json
{
  "status": "ready",
  "models": ["qwen3.6:35b"],
  "default": "qwen3.6:35b"
}
```

## TTS

See [TTS API Reference](tts-api.md).
