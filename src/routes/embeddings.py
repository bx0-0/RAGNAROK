"""POST /v1/embeddings — OpenAI-compatible embedding endpoint."""

import time

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.config import MODEL_NAME, OLLAMA_BASE_URL
from src.state import _get_state
from src.logging import log_request_start, log_request
from src.utils import _read_body, _fast_id

router = APIRouter()


@router.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    state = _get_state(request)
    request_id = _fast_id()
    start_time = time.monotonic()
    await log_request_start(request_id, "POST", "/v1/embeddings")

    try:
        body = orjson.loads(await _read_body(request))
    except Exception:
        await log_request(request_id, "POST", "/v1/embeddings", 400, 0, 0, 0, "BAD_JSON")
        return Response(status_code=400, content=b'{"error":"Invalid JSON"}', media_type="application/json")

    model = body.get("model", MODEL_NAME)
    input_data = body.get("input")
    if not input_data:
        await log_request(request_id, "POST", "/v1/embeddings", 400, 0, 0, 0, "MISSING_INPUT")
        return Response(
            status_code=400,
            content=orjson.dumps({"error": {"message": "'input' is required", "type": "invalid_request_error"}}),
            media_type="application/json",
        )

    if isinstance(input_data, str):
        input_data = [input_data]

    payload = {"model": model, "input": input_data}

    try:
        resp = await state.http_client.embed(**payload)
        elapsed = round(time.monotonic() - start_time, 2)
        embeddings = resp.embeddings or []
        # Determine embedding dimension from first vector
        dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], (list, tuple)) else 0

        data = [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ]
        await log_request(request_id, "POST", "/v1/embeddings", 200, elapsed, len(input_data) * dim, dim, "EMBED")
        return Response(
            status_code=200,
            content=orjson.dumps({
                "object": "list",
                "data": data,
                "model": model,
                "usage": {"prompt_tokens": len(input_data) * dim, "total_tokens": len(input_data) * dim},
            }),
            media_type="application/json",
        )
    except Exception as e:
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/embeddings", 502, elapsed, 0, 0, f"ERR:{str(e)[:40]}")
        return Response(
            status_code=502,
            content=orjson.dumps({"error": {"message": str(e), "type": "upstream_error"}}),
            media_type="application/json",
        )
