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


def convert_messages_to_ollama(messages, has_tools=False):
    result = []
    append = result.append

    # ═══ System Prompt للـ Tools — يمنع الموديل يكتب ملفات كبيرة مرة واحدة ═══
    if has_tools:
        tool_system_prompt = (
            "CRITICAL: When using tools that write to files, you MUST write the file "
            "in SMALL CHUNKS (max 50 lines at a time). Do NOT attempt to write the "
            "entire file content in a single tool call. Write the first chunk, then "
            "stop, and you will be asked to continue. This is a system requirement."
        )
        append({"role": "system", "content": tool_system_prompt})
    # ═══ End System Prompt ═══

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
