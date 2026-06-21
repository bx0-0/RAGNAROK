"""Register all route routers on the FastAPI app."""

from fastapi import FastAPI

from src.routes.models import router as _models_router
from src.routes.health import router as _health_router
from src.routes.chat import router as _chat_router
from src.routes.embeddings import router as _embeddings_router


def register_routers(app: FastAPI):
    app.include_router(_models_router)
    app.include_router(_health_router)
    app.include_router(_chat_router)
    app.include_router(_embeddings_router)
