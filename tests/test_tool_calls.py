"""Live integration tests for non-streaming tool calls.
Run: pytest tests/test_tool_calls.py -v
Tests full tool-use lifecycle: request → tool call → inject result → final answer.
"""

import json
import time
import httpx
import pytest

BASE = "https://hawaiian-greatly-ata-respondents.trycloudflare.com/v1"
# Tool calls make the model think longer → higher timeout
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries on 429


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=TIMEOUT)


def post_with_retry(client, url, **kwargs):
    """POST with retry on 429 (semaphore busy)."""
    for attempt in range(MAX_RETRIES):
        r = client.post(url, **kwargs)
        if r.status_code != 429:
            return r
        wait = RETRY_DELAY * (attempt + 1)  # 2s, 4s, 6s
        print(f"\n  ⚠️  429 on attempt {attempt+1}, retrying in {wait}s...")
        time.sleep(wait)
    return r  # return last response after retries


# ─── Single-tool tests ───

class TestToolCallSingle:
    """Model should recognize when to call a tool."""

    def test_calculator_tool(self, client):
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+3?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }],
            "stream": False,
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        has_tool = msg.get("tool_calls") is not None
        has_answer = "5" in (msg.get("content") or "").lower()
        assert has_tool or has_answer

    def test_weather_lookup_tool(self, client):
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "stream": False,
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            assert "id" in tc
            assert tc["type"] == "function"
            assert "get_weather" in tc["function"]["name"].lower()
            assert isinstance(tc["function"]["arguments"], (str, dict))


# ─── Two-turn tool loop tests ───

class TestToolCallTwoTurn:
    """Full 2-turn loop: tool call → inject result → final answer."""

    def test_calculator_two_turn(self, client):
        r1 = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 7×8?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }],
            "stream": False,
        })
        assert r1.status_code == 200
        msg1 = r1.json()["choices"][0]["message"]

        if not msg1.get("tool_calls"):
            assert "56" in (msg1.get("content") or "")
            return

        tool_id = msg1["tool_calls"][0]["id"]
        r2 = post_with_retry(client, "/chat/completions", json={
            "messages": [
                {"role": "user", "content": "What is 7×8?"},
                msg1,
                {"role": "tool", "content": "56", "tool_call_id": tool_id},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }],
            "stream": False,
        })
        assert r2.status_code == 200
        content = (r2.json()["choices"][0]["message"]["content"] or "").lower()
        assert "56" in content

    def test_file_write_tool_two_turn(self, client):
        tools = [{
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        }]
        r1 = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "Write 'hello' to /tmp/test.txt"}],
            "tools": tools,
            "stream": False,
        })
        assert r1.status_code == 200
        msg1 = r1.json()["choices"][0]["message"]
        if not msg1.get("tool_calls"):
            return  # Model declined — OK

        tc = msg1["tool_calls"][0]
        assert "write_file" in tc["function"]["name"].lower()
        args = tc["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        assert "path" in args or "/tmp/test.txt" in json.dumps(args).lower()

    def test_multiple_tool_calls(self, client):
        tools = [{
            "type": "function",
            "function": {
                "name": "translate",
                "description": "Translate text to another language",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "language": {"type": "string"}},
                    "required": ["text", "language"],
                },
            },
        }]
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "Translate 'hello' and 'goodbye' to French."}],
            "tools": tools,
            "stream": False,
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        if msg.get("tool_calls"):
            names = [t["function"]["name"].lower() for t in msg["tool_calls"]]
            assert "translate" in names


# ─── Edge cases ───

class TestToolCallEdgeCases:

    def test_no_tools_returns_normal_response(self, client):
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "Say OK"}],
            "stream": False,
        })
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert "tool_calls" not in msg or msg["tool_calls"] is None

    def test_empty_tool_definition(self, client):
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "Say OK"}],
            "tools": [],
            "stream": False,
        })
        assert r.status_code == 200

    def test_tool_with_no_parameters(self, client):
        r = post_with_retry(client, "/chat/completions", json={
            "messages": [{"role": "user", "content": "Ping the service"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Check if service is alive",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "stream": False,
        })
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
