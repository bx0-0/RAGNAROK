"""GET /v1/models — list available models.
POST /v1/models/unload — stop active gens for a model, then unload it from VRAM.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.config import _MODEL_LIST, MODEL_NUM_CTX, MODEL_NAME
from src.state import _get_state, ask_ollama_unload
from src.logging import logger

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "owned_by": "local",
            "context_length": MODEL_NUM_CTX.get(model),
        } for model in _MODEL_LIST],
    }


@router.post("/v1/models/unload")
async def unload_model_endpoint(request: Request):
    """Unload a model from VRAM.

    1. Stop any in-flight generations for that model (frees GPU compute).
    2. Issue Ollama keep_alive=0 to evict weights from memory.
    3. Report post-unload state via /api/ps.

    Body: {"model": "qwen3.5:9b"}  (optional; defaults to first configured model)
    """
    state = _get_state(request)

    # Parse model from JSON body
    model = MODEL_NAME
    try:
        import orjson
        body = orjson.loads(await request.body())
        if isinstance(body, dict) and body.get("model"):
            model = body["model"]
    except Exception:
        pass  # no body or bad body → default model

    # 1. Stop active gens for this model
    stopped = await state.stop_streams_for_model(model)

    # 2. Ask Ollama to unload
    await ask_ollama_unload(state.http_client, model)

    # 3. Report post-unload state
    remaining = []
    try:
        resp = await state.http_client.ps()
        base = model.split(":")[0]
        loaded_after = [
            {"name": p.name, "size_vram": p.size_vram}
            for p in (resp.models or [])
            if (p.name or "").startswith(base)
        ]
        remaining = loaded_after
    except Exception as e:
        remaining = [f"ps error: {e}"]

    logger.info(f"UNLOAD model={model} stopped_streams={stopped} still_loaded={len(remaining)}")
    return JSONResponse({
        "status": "unloaded",
        "model": model,
        "stopped_streams": stopped,
        "still_loaded": remaining,
    })
