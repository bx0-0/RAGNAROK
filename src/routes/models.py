"""GET /v1/models — list available models."""

from fastapi import APIRouter

from src.config import _MODEL_LIST

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "owned_by": "local",
        } for model in _MODEL_LIST],
    }
