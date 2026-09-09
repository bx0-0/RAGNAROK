"""GET /v1/models — list available models.
POST /v1/models/unload — stop active gens for a model, then unload it from VRAM.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import orjson

from src.config import _MODEL_LIST, MODEL_NUM_CTX, MODEL_NAME
from src.state import _get_state, ask_ollama_unload
from src.models.repair import RepairRequest
from src.repair_manager import RepairError
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


@router.post("/v1/models/repair")
async def repair_model_endpoint(request: Request):
    """Explicit, user-driven runtime repair.

    Creates a derived Ollama model that reuses the same underlying weights
    (`FROM <original>`) but applies the user-supplied `renderer` /
    `parser`. RAGNAROK never guesses the renderer/parser and never repairs
    automatically — the caller supplies the runtime configuration.

    After the derived model is created and verified, RAGNAROK routes future
    requests for the original name to the repaired model and deletes the
    original (only after the mapping is safely updated).

    Body: {"model": "...", "renderer": "...", "parser": "..."}
    (renderer / parser optional; at least one required)
    """
    state = _get_state(request)
    if state.repairs is None:
        return JSONResponse(status_code=503, content={"error": {"message": "repair subsystem unavailable", "type": "server_error"}})

    try:
        body = orjson.loads(await request.body())
        req = RepairRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid repair request", "type": "invalid_request_error",
                               "detail": str(e)[:160]}},
        )

    try:
        result = await state.repairs.repair(req.model, req.renderer, req.parser)
    except RepairError as e:
        code = e.code
        status = 400 if code == "INVALID_PARAMS" else 500
        logger.warning(f"REPAIR FAILED model={req.model} code={code} err={e}")
        return JSONResponse(
            status_code=status,
            content={"error": {"message": str(e), "type": "repair_error",
                               "code": code, "repairable": True,
                               "repair_endpoint": "/v1/models/repair"}},
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"REPAIR crashed model={req.model}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Repair failed: {e}", "type": "server_error",
                               "code": "REPAIR_FAILED"}},
        )

    logger.info(f"REPAIR OK original={req.model} derived={result['repaired_model']}")
    return JSONResponse(status_code=200, content=result)
