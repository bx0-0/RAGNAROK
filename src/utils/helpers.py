import os
import orjson

from fastapi import Request

from src.logging import logger


def _fast_id():
    return os.urandom(4).hex()


def extract_text_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def convert_messages_to_ollama(messages, has_tools=False):
    result = []
    append = result.append

    # System prompt for tools — prevents the model from writing large files at once
    if has_tools:
        tool_system_prompt = (
            "CRITICAL: When using tools that write to files, you MUST write the file "
            "in SMALL CHUNKS (max 50 lines at a time). Do NOT attempt to write the "
            "entire file content in a single tool call. Write the first chunk, then "
            "stop, and you will be asked to continue. This is a system requirement."
        )
        append({"role": "system", "content": tool_system_prompt})
    # End system prompt

    for m in messages:
        role = m["role"]
        content = m.get("content")

        if role == "tool":
            if isinstance(content, str):
                tool_content = content
            elif content is not None:
                tool_content = orjson.dumps(content).decode()
            else:
                tool_content = ""
            tool_msg = {"role": "tool", "content": tool_content}
            tc_id = m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            append(tool_msg)
        elif role == "assistant":
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                text = ""
            asst_msg = {"role": "assistant", "content": text}
            tool_calls = m.get("tool_calls")
            if tool_calls:
                ollama_tc = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = orjson.loads(args)
                        except Exception:
                            args = {}
                    ollama_tc.append({
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args,
                        }
                    })
                asst_msg["tool_calls"] = ollama_tc
            append(asst_msg)
        elif role == "system":
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            elif content is None:
                content = ""
            append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, str):
                append({"role": "user", "content": content})
            elif isinstance(content, list):
                ollama_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        ollama_content.append(item)
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        # Strip base64 prefix so Ollama accepts it
                        if url.startswith("data:image/"):
                            base64_part = url.split(",", 1)
                            if len(base64_part) > 1:
                                url = base64_part[1]  # Keep only clean base64
                        # End of cleanup
                        ollama_content.append({"type": "image_url", "image_url": url})
                text = " ".join(
                    item.get("text", "")
                    for item in ollama_content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
                images = [
                    item.get("image_url", "")
                    for item in ollama_content
                    if isinstance(item, dict) and item.get("type") == "image_url"
                ]
                if text or images:
                    msg = {"role": "user", "content": text}
                    if images:
                        msg["images"] = images
                    append(msg)
                else:
                    append({"role": "user", "content": ""})
            else:
                append({"role": "user", "content": ""})
    return result


def format_tool_calls_openai(ollama_tcs):
    openai_tcs = []
    for idx, tc in enumerate(ollama_tcs):
        try:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, dict):
                try:
                    args = orjson.dumps(args).decode("utf-8")
                except Exception:
                    args = "{}"
            openai_tcs.append({
                "index": idx,
                "id": tc.get("id") or f"call_{_fast_id()}",
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            })
        except Exception as e:
            logger.error(f"Tool format error: {e}")
    return openai_tcs


# ─── Chunked body reader (avoids buffering huge prompts in RAM) ───
async def _read_body(request: Request, max_size_mb: int = 50) -> bytes:
    """Read request body without loading >max_size_mb all at once.
    Returns raw bytes. Raises ValueError if Content-Length exceeds limit."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size_mb * 1024 * 1024:
        raise ValueError(f"Body too large: {int(content_length)} bytes")
    # Read in 64KB chunks — FastAPI/Starlette already streams internally
    chunks = []
    async for chunk in request.stream():
        chunks.append(chunk)
        total = sum(len(c) for c in chunks)
        if total > max_size_mb * 1024 * 1024:
            raise ValueError(f"Body exceeds {max_size_mb}MB limit")
    return b"".join(chunks)
