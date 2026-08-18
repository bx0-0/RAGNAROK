"""Per-model num_ctx: config parsing, validation, and the request path.

Each test sets env vars (via monkeypatch, auto-restored), then reloads
``src.config`` through the shared ``cfg`` fixture. The fixture also reloads
config after teardown so every test starts and ends with a clean module.

Tests:
  1-5. parsing / validation rules
  6.  backward-compat default (omitted NUM_CTX)
  7.  the real request path: active_model -> model_opts -> to_ollama_payload
      -> build_chat_kwargs carries the mapped num_ctx (both stream & non-stream
      use build_chat_kwargs, so asserting the kwargs covers both).
  8.  warmup opts use the mapped ctx + num_predict=1
"""

import importlib
import os

import pytest


@pytest.fixture()
def cfg(monkeypatch):
    """Reload src.config against the current env.

    ``monkeypatch.setenv``/``delenv`` inside each test set the env for the
    duration of the test and are auto-reverted by pytest. This fixture yields
    the freshly reloaded module, and on teardown clears any test-specific env
    then reloads once more so no invalid/stale state leaks between tests.
    """
    c = importlib.reload(importlib.import_module("src.config"))
    yield c
    # teardown: drop test-specific overrides so the reload below always sees a
    # valid env (protects against teardown running before monkeypatch reverts).
    os.environ.pop("NUM_CTX", None)
    os.environ.pop("MODEL_NAME", None)
    importlib.reload(importlib.import_module("src.config"))


# ── 1. one model + one ctx ───────────────────────────────────────────────────
def test_one_model_one_ctx(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "m1")
    monkeypatch.setenv("NUM_CTX", "8192")
    c = importlib.reload(cfg)
    assert c.MODEL_NUM_CTX == {"m1": 8192}


# ── 2. multiple models + one ctx -> broadcast ────────────────────────────────
def test_multi_model_one_ctx_broadcast(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a b c")
    monkeypatch.setenv("NUM_CTX", "40000")
    c = importlib.reload(cfg)
    assert c.MODEL_NUM_CTX == {"a": 40000, "b": 40000, "c": 40000}


# ── 3. N models + N ctxs -> positional mapping (order) ──────────────────────
def test_multi_model_positional(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a b c")
    monkeypatch.setenv("NUM_CTX", "100000 50000 32000")
    c = importlib.reload(cfg)
    assert c.MODEL_NUM_CTX == {"a": 100000, "b": 50000, "c": 32000}


# ── 4. too few ctx values -> clear error ────────────────────────────────────
def test_too_few_ctx_values_error(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a b c")
    monkeypatch.setenv("NUM_CTX", "100000 50000")
    with pytest.raises(ValueError, match="NUM_CTX accepts"):
        importlib.reload(cfg)


# ── 5. too many ctx values -> clear error ───────────────────────────────────
def test_too_many_ctx_values_error(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a b c")
    monkeypatch.setenv("NUM_CTX", "100000 50000 32000 16000")
    with pytest.raises(ValueError, match="NUM_CTX accepts"):
        importlib.reload(cfg)


# ── 6. omitted --num-ctx -> default 16384 broadcast (backward-compat) ───────
def test_omitted_num_ctx_defaults_broadcast(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "m1 m2 m3")
    monkeypatch.delenv("NUM_CTX", raising=False)
    c = importlib.reload(cfg)
    assert c.MODEL_NUM_CTX == {"m1": 16384, "m2": 16384, "m3": 16384}
    # and it propagates through the request config path, not just the mapping:
    assert c.model_opts("m2")["num_ctx"] == 16384


# ── 7. END-TO-END: active_model receives its mapped num_ctx through the real
#     config -> payload -> build_chat_kwargs path (stream + non-stream) ───────
def test_end_to_end_active_model_gets_mapped_num_ctx(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "qwen3.8:27b llama3.1:8b mistral:7b")
    monkeypatch.setenv("NUM_CTX", "100000 50000 32000")
    c = importlib.reload(cfg)
    # downstream models.chat imports config values at import time — refresh it
    import src.models.chat as mchat
    importlib.reload(mchat)

    req = mchat.ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    msgs = [{"role": "user", "content": "hi"}]

    for active, expected in (
        ("qwen3.8:27b", 100000),
        ("llama3.1:8b", 50000),
        ("mistral:7b", 32000),
    ):
        payload = req.to_ollama_payload(active, "60m", c.model_opts(active), msgs)
        kwargs = mchat.build_chat_kwargs(payload, stream=False)   # non-stream path
        assert kwargs["model"] == active
        assert kwargs["options"]["num_ctx"] == expected, active
        # stream path shares build_chat_kwargs; confirm it carries the same value
        s_kwargs = mchat.build_chat_kwargs(payload, stream=True)
        assert s_kwargs["options"]["num_ctx"] == expected, active
        # every other option is still present (shared template intact)
        assert kwargs["options"]["num_batch"] == c.NUM_BATCH
        assert "num_gpu" in kwargs["options"]
        assert "flash_attn" in kwargs["options"]


def test_warmup_opts_use_mapped_ctx_and_predict_one(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a b")
    monkeypatch.setenv("NUM_CTX", "9999 1111")
    c = importlib.reload(cfg)
    wa = c.model_opts("a", warmup=True)
    wb = c.model_opts("b", warmup=True)
    assert wa["num_ctx"] == 9999 and wb["num_ctx"] == 1111
    assert wa["num_predict"] == 1 and wb["num_predict"] == 1
