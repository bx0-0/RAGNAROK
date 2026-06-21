"""GET /health — health check endpoint."""

from fastapi import APIRouter, Request

from src.state import _get_state
from src.config import _MODEL_LIST, MODEL_NAME

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    state = _get_state(request)
    return {
        "status": "ready" if state.is_warm else "warming",
        "models": _MODEL_LIST,
        "default": MODEL_NAME,
    }
