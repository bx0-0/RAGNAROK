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

## Streaming Lifecycle & Control

The streaming API is fully OpenAI-compatible and unchanged: `stream: true` still returns
`data: {...}

` SSE frames terminated by `data: [DONE]

`.

### Client-disconnect cancellation

When a streaming client disconnects unexpectedly (FIN, RST, Wi-Fi drop, or a proxy/tunnel
timeout), the gateway detects the disconnect via the ASGI `http.disconnect` event and
cancels the in-flight generation:

```
client disconnect -> disconnect detection (ASGI http.disconnect)
  -> streaming task cancelled -> Ollama stream closed -> generation stops
  -> request cleanup -> concurrency slot released
```

Previously a client disconnect left the Ollama generation running until the hard stream
timeout, wasting GPU. This path stops it immediately.

> The model stays loaded. A client disconnect only releases the active generation — it does
> **not** evict model weights from memory. See **Unload a model** below.

### Request ID

Every streaming response carries an `x-request-id` response header (a short opaque id).
Use it to target that specific in-flight generation with the stop endpoint.

### List active generations

**`GET /v1/chat/completions/active`**

Read-only operational endpoint: returns one lightweight entry per in-flight
streaming generation. Useful for discovering which requests are running and
targeting one (or all) with the stop endpoint below. Listing is non-mutating:
calling it never cancels, releases, or unloads anything.

| | |
|---|---|
| **Path** | `/v1/chat/completions/active` |
| **Body** | none |
| **200** | `{"object":"list","count":N,"data":[...]}` |

`data` items (one per active generation):

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Same id carried in the `x-request-id` response header; feed this to the stop endpoint |
| `model` | string | Model being generated (e.g. `qwen3.5:9b`) |
| `status` | string | `running` (normal) or `cancelling` (a stop/unload is in flight) |
| `elapsed_seconds` | number | Seconds since the generation started (0.01s precision) |

```bash
# Discover active generations
curl -s https://YOUR-URL/v1/chat/completions/active

# -> {"object":"list","count":1,"data":[
#      {"request_id":"84c2ef9f","model":"qwen3.5:9b",
#       "status":"running","elapsed_seconds":3.42} ]}
```

**Discover-and-stop workflow:**

1. List active generations to find the id you want to target.
2. Copy the `request_id`.
3. Stop that specific generation with the stop endpoint.
4. Re-list to confirm it disappeared.

```bash
BASE=https://YOUR-URL
RID=$(curl -s $BASE/v1/chat/completions/active | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["request_id"])')
curl -X POST $BASE/v1/chat/completions/$RID/stop
curl -s $BASE/v1/chat/completions/active   # -> count: 0 (it's gone)
```

### Stop a generation

**`POST /v1/chat/completions/{request_id}/stop`**

Cancels a single in-flight streaming generation. Affects **one request only**; other
concurrent generations continue unaffected. Does **not** unload the model.

| | |
|---|---|
| **Path param** | `request_id` — value of the `x-request-id` header from the stream |
| **Body** | none |
| **200** | `{"status":"stopped","request_id":"<id>"}` |
| **404** | `{"error":{"message":"No active stream with id <id>","type":"not_found"}}` (id unknown or already finished) |

```bash
# Capture the request id from the streaming response header, then stop it
RID=$(curl -s -D - -o /dev/null -X POST https://YOUR-URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"long task"}],"stream":true}' \
  | awk 'tolower($1)=="x-request-id:"{print $2}')
curl -X POST "https://YOUR-URL/v1/chat/completions/$RID/stop"
```

### Unload a model

**`POST /v1/models/unload`**

Explicitly evicts a model from Ollama's active residency. First stops any in-flight
generations for that model (frees GPU compute), then issues Ollama `keep_alive: 0` to drop
the weights, and reports the post-unload state.

| | |
|---|---|
| **Body** | `{"model":"qwen3.5:9b"}` (optional — defaults to the first configured model) |
| **Scope** | All in-flight generations for that model, then the model weights |
| **200** | `{"status":"unloaded","model":"<name>","stopped_streams":N,"still_loaded":[...]}` |

`still_loaded` lists models Ollama still reports as resident after the evict (empty when
the model is fully dropped). Use it to confirm the unload actually took effect.

```bash
curl -X POST https://YOUR-URL/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:9b"}'
```

> **Disconnect vs unload — separate operations.**
> - **Stop / client disconnect**: cancels the active generation; model weights may stay
>   resident in memory for reuse.
> - **Unload**: stops that model's active generations **and** evicts the weights from memory.

> **Security note:** like the rest of this API, these control endpoints are not
> authenticated. They can stop or evict model execution, so they must be kept behind the
> same trusted network boundary as the rest of the gateway (e.g. the Cloudflare tunnel is
> the only public front door). Do not expose them on a wide-open interface without access
> control.

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
