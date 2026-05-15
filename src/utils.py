import os
import orjson

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


def convert_messages_to_ollama(messages):
    result = []
    append = result.append
    for m in messages:
        role = m["role"]
        content = m.get("content")

        if role == "tool":
            tool_msg = {"role": "tool", "content": content if isinstance(content, str) else ""}
            tc_id = m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            append(tool_msg)
        elif role == "assistant":
            # Extract text from content
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
            append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, str) and content.strip():
                append({"role": "user", "content": content})
            elif isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
                if text.strip():
                    append({"role": "user", "content": text})
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


