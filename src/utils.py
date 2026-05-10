import uuid
import json

try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")
    def json_loads(text):
        return orjson.loads(text)
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    def json_loads(text):
        return json.loads(text)

def extract_text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(texts) if texts else ""
    return str(content)

def convert_messages_to_ollama(messages):
    ollama_messages = []
    for m in messages:
        role = m.get("role")
        content = extract_text_content(m.get("content"))

        if role == "tool":
            tool_msg = {"role": "tool", "content": content or ""}
            if "tool_call_id" in m:
                tool_msg["tool_call_id"] = m["tool_call_id"]
            ollama_messages.append(tool_msg)
        elif role == "assistant":
            asst_msg = {"role": "assistant", "content": content or ""}
            if "tool_calls" in m and m["tool_calls"]:
                ollama_tc = []
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json_loads(args)
                        except Exception:
                            args = {}
                    ollama_tc.append(
                        {
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": args,
                            }
                        }
                    )
                asst_msg["tool_calls"] = ollama_tc
            ollama_messages.append(asst_msg)
        elif role == "system":
            ollama_messages.append({"role": "system", "content": content})
        elif role == "user" and content.strip():
            ollama_messages.append({"role": "user", "content": content})
    return ollama_messages

def format_tool_calls_openai(ollama_tcs):
    openai_tcs = []
    for idx, tc in enumerate(ollama_tcs):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, dict):
            args = json_dumps(args)
        openai_tcs.append(
            {
                "index": idx,
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            }
        )
    return openai_tcs
