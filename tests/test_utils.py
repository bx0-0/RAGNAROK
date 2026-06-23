"""Tests for message conversion: OpenAI format <-> Ollama format."""

import pytest

from src.utils import (
    convert_messages_to_ollama,
    format_tool_calls_openai,
    extract_text_content,
)


# ─── extract_text_content ───

class TestExtractTextContent:
    def test_plain_string(self):
        assert extract_text_content("hello") == "hello"

    def test_none(self):
        assert extract_text_content(None) == ""

    def test_list_with_text(self):
        items = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert extract_text_content(items) == "a b"

    def test_list_mixed_types(self):
        items = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        assert extract_text_content(items) == "hello"

    def test_non_string_non_list(self):
        assert extract_text_content(42) == "42"


# ─── convert_messages_to_ollama (no tools) ───

class TestConvertNoTools:
    def test_simple_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = convert_messages_to_ollama(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_system_then_user(self):
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_assistant_message(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there!"},
        ]
        result = convert_messages_to_ollama(msgs)
        assert len(result) == 2
        assert result[1]["content"] == "there!"

    def test_system_none_content(self):
        msgs = [{"role": "system", "content": None}]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["content"] == ""

    def test_user_list_text_items(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["content"] == "a b"

    def test_user_with_base64_image(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ABC123"},
                    },
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        msg = result[0]
        assert msg["content"] == "look"
        assert len(msg["images"]) == 1
        # base64 prefix stripped
        assert msg["images"][0] == "ABC123"

    def test_user_with_plain_image_url(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        msg = result[0]
        assert msg["content"] == ""
        assert msg["images"][0] == "https://example.com/img.png"

    def test_user_message_with_no_content(self):
        msgs = [{"role": "user", "content": None}]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["content"] == ""


# ─── convert_messages_to_ollama (with tools) ───

class TestConvertWithTools:
    def test_injects_tool_system_prompt(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = convert_messages_to_ollama(msgs, has_tools=True)
        assert result[0]["role"] == "system"
        assert "SMALL CHUNKS" in result[0]["content"]
        assert len(result) == 2

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": '{"path":"a.py"}',
                        },
                    }
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]["function"]
        assert tc["name"] == "edit_file"
        # Arguments string parsed into dict
        assert isinstance(tc["arguments"], dict)
        assert tc["arguments"]["path"] == "a.py"

    def test_tool_result_message(self):
        msgs = [
            {
                "role": "tool",
                "content": "file written",
                "tool_call_id": "call_1",
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "file written"
        assert result[0]["tool_call_id"] == "call_1"

    def test_tool_result_with_dict_content(self):
        msgs = [
            {
                "role": "tool",
                "content": {"status": "ok"},
                "tool_call_id": "call_2",
            }
        ]
        result = convert_messages_to_ollama(msgs)
        # Non-string content serialized to JSON string
        assert '"status"' in result[0]["content"]

    def test_assistant_tool_call_bad_json_fallback(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bad",
                            "arguments": "not-json!!!",
                        },
                    }
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        tc = result[0]["tool_calls"][0]["function"]
        assert tc["arguments"] == {}

    def test_assistant_list_content(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            }
        ]
        result = convert_messages_to_ollama(msgs)
        assert result[0]["content"] == "part1 part2"

    def test_full_conversation_round_trip(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "write a file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": '{"path":"x.py","content":"hi"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "ok",
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "Done!"},
        ]
        result = convert_messages_to_ollama(msgs)
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]


# ─── format_tool_calls_openai ───

class TestFormatToolCallsOpenAI:
    def test_basic(self):
        ollama_tcs = [
            {
                "id": "call_1",
                "function": {
                    "name": "edit_file",
                    "arguments": {"path": "a.py"},
                },
            }
        ]
        result = format_tool_calls_openai(ollama_tcs)
        assert len(result) == 1
        tc = result[0]
        assert tc["index"] == 0
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        # Arguments serialized to string for OpenAI
        assert isinstance(tc["function"]["arguments"], str)
        import json
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed["path"] == "a.py"

    def test_string_arguments_passthrough(self):
        ollama_tcs = [
            {
                "id": "call_1",
                "function": {
                    "name": "f",
                    "arguments": '{"x":1}',
                },
            }
        ]
        result = format_tool_calls_openai(ollama_tcs)
        assert result[0]["function"]["arguments"] == '{"x":1}'

    def test_missing_id_generates_one(self):
        ollama_tcs = [
            {
                "function": {"name": "f", "arguments": {}},
            }
        ]
        result = format_tool_calls_openai(ollama_tcs)
        assert result[0]["id"].startswith("call_")

    def test_multiple_indexed(self):
        ollama_tcs = [
            {"id": "a", "function": {"name": "f1", "arguments": {}}},
            {"id": "b", "function": {"name": "f2", "arguments": {}}},
        ]
        result = format_tool_calls_openai(ollama_tcs)
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    def test_empty_list(self):
        assert format_tool_calls_openai([]) == []


# ─── Pydantic validation ───

import pytest
from src.models.chat import ChatCompletionRequest


class TestChatCompletionRequest:
    def test_rejects_empty_messages(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(messages=[])

    def test_accepts_one_message(self):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert len(req.messages) == 1
