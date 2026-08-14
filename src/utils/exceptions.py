"""Exceptions for HTTP mapping."""

from http import HTTPStatus
from fastapi import Request
from fastapi.responses import Response
from src.utils.gpu import InsufficientVRAMError

def handle_insufficient_vram(request: Request, exc: InsufficientVRAMError) -> Response:
    return Response(
        content=f'GPU memory insufficient ({exc.free_gb:.2f} GB free < {exc.required_gb} GB). '
               f'Unload large models or increase VRAM.',
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        media_type='text/plain',
    )
