import os
import orjson


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
    ollama_messages = []
    for m in messages:
        role = m.get("role")
        content = extract_text_content(m.get("content"))

        if role == "tool":
            tool_msg = {"role": "tool", "content": content or ""}
            tool_call_id = m.get("tool_call_id")
            if tool_call_id:
                tool_msg["tool_call_id"] = tool_call_id
            ollama_messages.append(tool_msg)
        elif role == "assistant":
            asst_msg = {"role": "assistant", "content": content or ""}
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
            ollama_messages.append(asst_msg)
        elif role == "system":
            ollama_messages.append({"role": "system", "content": content})
        elif role == "user":
            if content and content.strip():
                ollama_messages.append({"role": "user", "content": content})
    return ollama_messages


def format_tool_calls_openai(ollama_tcs):
    openai_tcs = []
    for idx, tc in enumerate(ollama_tcs):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, dict):
            args = orjson.dumps(args).decode("utf-8")
        openai_tcs.append({
            "index": idx,
            "id": tc.get("id") or f"call_{_fast_id()}",
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "arguments": args,
            },
        })
    return openai_tcs
