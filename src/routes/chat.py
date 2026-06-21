"""POST /v1/chat/completions — main chat endpoint + non-stream handler."""

import time
import asyncio
from functools import lru_cache

import orjson
import uvloop
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.logging import setup_logging, logger, _open_log_fh, log_request_start, log_request, _log_fh as _gw_log_fh
from src.utils import (
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
    _fast_id,
)
from src.errors import _RATE_LIMIT_RESPONSE, _BAD_JSON_RESPONSE
from src.streaming import handle_stream
