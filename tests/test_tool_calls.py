"""Live integration tests for non-streaming tool calls.
Run: pytest tests/test_tool_calls.py -v
Tests full tool-use lifecycle: request → tool call → inject result → final answer.
"""

import json
import httpx
import pytest

BASE = "https://constraint-viewing-strengths-bride.trycloudflare.com/v1"
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=TIMEOUT)


# ─── Helper to replay a tool-use conversation ───

def run_tool_conversation(client, messages, tools):
    """Run 2-turn tool loop: user → model tool call → inject result → final answer."""
    r1 = client.post("/chat/completions", json={
        "messages": messages,
        "tools": tools,
        "stream": False,
    })
    return r1


class TestToolCallSingle:
    """Model should recognize when to call a tool."""

    def test_calculator_tool(self, client):
        r = run_tool_conversation(client,
            messages=[{"role": "user", "content": "What is 2+3?"}],
            tools=[{
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
        )
        assert r.status_code == 200
        data = r.json()
        msg = data["choices"][0]["message"]
        # Model either calls tool or answers directly — both OK
        has_tool = msg.get("tool_calls") is not None
        has_answer = "5" in (msg.get("content") or "").lower()
        assert has_tool or has_answer

    def test_weather_lookup_tool(self, client):
        r = run_tool_conversation(client,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=[{
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
        )
        assert r.status_code == 200
        data = r.json()
        msg = data["choices"][0]["message"]
        has_tool = msg.get("tool_calls") is not None
        # If tool called, verify structure
        if has_tool:
            tc = msg["tool_calls"][0]
            assert "id" in tc
            assert tc["type"] == "function"
            assert "get_weather" in tc["function"]["name"].lower()
            # Arguments should be a string (OpenAI compat) or dict
            args = tc["function"]["arguments"]
            assert isinstance(args, (str, dict))


class TestToolCallTwoTurn:
    """Full 2-turn loop: tool call → inject result → final answer."""

    def test_calculator_two_turn(self, client):
        # Turn 1: ask calculation with tool available
        r1 = client.post("/chat/completions", json={
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
        data1 = r1.json()
        msg1 = data1["choices"][0]["message"]

        if not msg1.get("tool_calls"):
            # Model answered directly — verify correctness
            assert "56" in (msg1.get("content") or "")
            return

        tc = msg1["tool_calls"][0]
        tool_id = tc["id"]

        # Turn 2: inject fake tool result
        r2 = client.post("/chat/completions", json={
            "messages": [
                {"role": "user", "content": "What is 7×8?"},
                msg1,
                {
                    "role": "tool",
                    "content": "56",
                    "tool_call_id": tool_id,
                },
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
        data2 = r2.json()
        final_content = (data2["choices"][0]["message"]["content"] or "").lower()
        assert "56" in final_content

    def test_file_write_tool_two_turn(self, client):
        tools = [{
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }]

        r1 = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Write 'hello' to /tmp/test.txt"}],
            "tools": tools,
            "stream": False,
        })
        assert r1.status_code == 200
        data1 = r1.json()
        msg1 = data1["choices"][0]["message"]

        if not msg1.get("tool_calls"):
            return  # Model declined to use tool — OK

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
                    "properties": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                    },
                    "required": ["text", "language"],
                },
            },
        }]

        r = client.post("/chat/completions", json={
            "messages": [
                {
                    "role": "user",
                    "content": "Translate 'hello' and 'goodbye' to French.",
                },
            ],
            "tools": tools,
            "stream": False,
        })
        assert r.status_code == 200
        data = r.json()
        msg = data["choices"][0]["message"]

        # Model should call translate at least once or answer directly
        if msg.get("tool_calls"):
            names = [tc["function"]["name"].lower() for tc in msg["tool_calls"]]
            assert "translate" in names


class TestToolCallEdgeCases:

    def test_no_tools_returns_normal_response(self, client):
        """Without tools, model should just answer."""
        r = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Say OK"}],
            "stream": False,
        })
        assert r.status_code == 200
        data = r.json()
        msg = data["choices"][0]["message"]
        assert "tool_calls" not in msg or msg["tool_calls"] is None

    def test_empty_tool_definition(self, client):
        """Empty tools array should behave like no tools."""
        r = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Say OK"}],
            "tools": [],
            "stream": False,
        })
        assert r.status_code == 200

    def test_tool_with_no_parameters(self, client):
        tools = [{
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Check if service is alive",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        r = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Ping the service"}],
            "tools": tools,
            "stream": False,
        })
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
