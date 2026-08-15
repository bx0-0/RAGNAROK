"""Utility helpers for the gateway.

- helpers: message conversion, streaming helpers
- gpu: VRAM queries and guards
"""

from . import helpers
from . import gpu

# Re-export common helpers so `from src.utils import X` keeps working
from .helpers import (
    fast_id,
    read_body,
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
)
