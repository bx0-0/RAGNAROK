"""Focused parser tests — lock the src.parser module boundary.

The moved code is already covered in tests/test_streaming.py (via the
src.streaming re-exports). These tests import directly from src.parser to
guarantee the new module is the real home of the parsing/normalization
responsibility, and to pin the tool-call contract (the part involved in the
original double-format regression) at the parser level.
"""
from types import SimpleNamespace as NS

from src.parser import ParsedChunk, parse_chunk, format_tool_calls


def _tool_call(name="read", args='{"path":"x"}', cid="id-1"):
    return NS(function=NS(name=name, arguments=args), id=cid)


def _chunk(content=None, thinking="", tool_calls=None, done=False,
           prompt=0, evalc=0, reason=""):
    return NS(
        message=NS(content=content, thinking=thinking, tool_calls=tool_calls),
        done=done, prompt_eval_count=prompt, eval_count=evalc, done_reason=reason,
    )


class TestParseChunkFromNewHome:

    def test_plain_content(self):
        p = parse_chunk(_chunk(content="hello"))
        assert isinstance(p, ParsedChunk)
        assert p.content == "hello"
        assert p.thinking == ""
        assert not p.tool_calls
        assert p.done is False

    def test_list_content_joins_text_parts_only(self):
        parts = [{"type": "text", "text": "a"},
                 {"type": "text", "text": "b"},
                 {"type": "tool_call", "text": "drop"}]
        assert parse_chunk(_chunk(content=parts)).content == "a b"

    def test_thinking_and_usage(self):
        p = parse_chunk(_chunk(thinking="t", done=True, prompt=3, evalc=7, reason="stop"))
        assert p.thinking == "t" and p.done is True
        assert p.prompt_eval_count == 3 and p.eval_count == 7 and p.done_reason == "stop"

    def test_none_message_yields_empty(self):
        # A message=None chunk is Ollama's empty final marker; parse_chunk
        # returns a default ParsedChunk (done is NOT read off the chunk).
        p = parse_chunk(NS(message=None, done=True, prompt_eval_count=0,
                           eval_count=0, done_reason=""))
        assert p.content == "" and not p.tool_calls

    def test_tool_calls_preformatted_to_openai_shape(self):
        p = parse_chunk(_chunk(tool_calls=[
            _tool_call("read", '{"p":"a"}', "i-1"),
            _tool_call("write", "{}", None),
        ]))
        assert len(p.tool_calls) == 2
        assert p.tool_calls[0]["id"] == "i-1"
        assert p.tool_calls[0]["function"]["name"] == "read"
        assert p.tool_calls[1]["id"].startswith("call_")  # generated when absent
        assert p.tool_calls[0]["type"] == "function"


class TestFormatToolCallsFromNewHome:

    def test_indices_args_and_ids(self):
        out = format_tool_calls([
            _tool_call("a", '{"x":1}', "i0"),
            _tool_call("b", '{"y":2}', None),
        ], 5)
        assert [o["index"] for o in out] == [5, 6]
        assert out[0]["id"] == "i0"
        assert out[1]["id"].startswith("call_")
        assert out[0]["function"]["arguments"] == '{"x":1}'

    def test_dict_arguments_serialized(self):
        out = format_tool_calls([_tool_call("f", {"k": 1})], 0)
        assert out[0]["function"]["arguments"] == '{"k":1}'

    def test_idempotent_on_already_formatted_dicts(self):
        # parse_chunk pre-formats; formatting again must NOT degrade to ?/{}
        raw = _chunk(tool_calls=[_tool_call("read", '{"path":"/x"}', "c1")])
        once = parse_chunk(raw).tool_calls
        again = format_tool_calls(once, 0)
        assert again[0]["function"]["name"] == "read"
        assert again[0]["function"]["arguments"] == '{"path":"/x"}'
        assert again[0]["id"] == "c1"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
