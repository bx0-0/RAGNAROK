"""Utility helpers for the gateway.

- helpers: message conversion, streaming helpers
- gpu: VRAM queries and guards
- exceptions: custom errors for HTTP mapping
"""

from . import helpers
from . import gpu
from . import exceptions

# Re-export common helpers so `from src.utils import X` keeps working
from .helpers import (
    _fast_id,
    _read_body,
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
)
