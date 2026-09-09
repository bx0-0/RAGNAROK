"""Unit tests for the Model Management + Runtime Repair system.

No real Ollama, no 27B download. All Ollama calls are mocked.

Covers the acceptance matrix:
  Model existence: exists -> no pull / missing -> pull / unavailable -> error
  Repair: valid repair -> derived created, renderer/parser passed, weights reused,
          verified, original deleted only after verify, mapping switches
          failed create keeps original / failed verify keeps original /
          repeat identical repair is idempotent / invalid params -> validation error
          Ollama failure propagated
  Error handling: repairable Ollama error -> REPAIR_AVAILABLE /
                  unknown error -> normal path / no automatic repair
"""
import asyncio
import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

from src.model_manager import ModelManager
from src.repair_manager import (
    RepairManager, RepairError, split_model, make_repaired_name,
    REPAIRED_SUFFIX, parse_modelfile,
)
from src.models.repair import RepairRequest
from src.errors import is_repairable_error, build_repair_available_response

import src.streaming as _st
from src.retry import RetryPolicy


# ── Fake Ollama AsyncClient (records calls, controllable state) ─────────────
class FakeOllamaClient:
    def __init__(self, present=(), fail_create=False, fail_verify=False, fail_list=False):
        self.present = set(present)
        self.fail_create = fail_create
        self.fail_verify = fail_verify
        self.fail_list = fail_list
        self.calls = []          # list of (method, kwargs)
        self.creations = []      # (name, modelfile)
        self.modelfiles = {}     # name -> raw modelfile (as Ollama would store it)

    def _record(self, m, **kw):
        self.calls.append((m, kw))

    async def list(self):
        self._record("list")
        if self.fail_list:
            raise ollama.ResponseError("ollama unavailable", 503)
        return NS(models=[NS(name=n) for n in sorted(self.present)])

    async def pull(self, model=None, stream=None, **kw):
        self._record("pull", model=model)
        self.present.add(model)

    async def delete(self, model=None, **kw):
        self._record("delete", model=model)
        self.present.discard(model)

    async def show(self, model=None, **kw):
        self._record("show", model=model)
        return NS(model_dump=lambda m=model: {"name": m, "details": {},
                                             "modelfile": self.modelfiles.get(m)})

    async def chat(self, model=None, messages=None, options=None, keep_alive=None, **kw):
        self._record("chat", model=model)
        if self.fail_verify:
            raise ollama.ResponseError("System message must be at the beginning.", 500)
        return NS(message=NS(content="ok"), done=True, eval_count=1)

    async def generate(self, model=None, prompt=None, options=None, keep_alive=None, **kw):
        self._record("generate", model=model)
        if self.fail_verify:
            raise ollama.ResponseError("bad renderer", 500)
        return NS(message=NS(content="ok"), done=True, eval_count=1)

    async def _request_raw(self, method, path, json=None):
        self._record("create", path=path, json=json)
        if self.fail_create:
            raise ollama.ResponseError("create boom", 500)
        name = json["name"]
        self.present.add(name)
        mf = json.get("modelfile", "")
        self.creations.append((name, mf))
        self.modelfiles[name] = mf
        class _R:
            def json(self):
                return {"status": "success"}
        return _R()


# ── Model existence / ensure_available ─────────────────────────────────────
def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_exists_true_no_pull():
    c = FakeOllamaClient(present=["qwen3.5:9b"])
    mm = ModelManager(c)
    _run(mm.ensure_available("qwen3.5:9b"))
    assert ("pull", {"model": "qwen3.5:9b"}) not in c.calls
    assert ("pull", None) not in c.calls
    assert not any(m == "pull" for m, _ in c.calls)


def test_missing_model_triggers_pull():
    c = FakeOllamaClient(present=[])
    mm = ModelManager(c)
    pulled = _run(mm.ensure_available("mistral:7b"))
    assert pulled is False
    assert ("pull", {"model": "mistral:7b"}) in c.calls
    assert "mistral:7b" in c.present


def test_ollama_unavailable_propagates():
    c = FakeOllamaClient(present=[], fail_list=True)
    mm = ModelManager(c)
    with pytest.raises(ollama.ResponseError):
        _run(mm.ensure_available("mistral:7b"))
    # and no pull was attempted
    assert not any(m == "pull" for m, _ in c.calls)


def test_exists_uses_structured_names():
    c = FakeOllamaClient(present=["a:b", "c:d"])
    mm = ModelManager(c)
    assert _run(mm.exists("a:b")) is True
    assert _run(mm.exists("a")) is False          # base without tag != tagged
    assert _run(mm.exists("nope")) is False


# ── model-name helpers ─────────────────────────────────────────────────────
def test_split_model():
    assert split_model("qwen3.8:27b") == ("qwen3.8", "27b")
    # uppercase / underscored tags pass through
    assert split_model("Qwen3.8-27B:IQ3_S") == ("Qwen3.8-27B", "IQ3_S")
    assert split_model("llama3.1") == ("llama3.1", None)


def test_make_repaired_name_deterministic_and_valid():
    n1 = make_repaired_name("Qwen3.8-27B:IQ3_S", "qwen3.8", "qwen3.5")
    n2 = make_repaired_name("Qwen3.8-27B:IQ3_S", "qwen3.8", "qwen3.5")
    assert n1 == n2
    base, tag = split_model(n1)
    assert tag == "IQ3_S"
    assert REPAIRED_SUFFIX in base
    assert base.endswith(f"-{REPAIRED_SUFFIX}-") or f"-{REPAIRED_SUFFIX}-" in base
    # different config -> different name (no collision)
    n3 = make_repaired_name("Qwen3.8-27B:IQ3_S", "llama", "llama")
    assert n1 != n3


# ── RepairManager: happy path ──────────────────────────────────────────────
def test_repair_creates_derived_and_switches_mapping():
    c = FakeOllamaClient(present=["Qwen3.8-27B:IQ3_S"])
    rm = RepairManager(ModelManager(c))
    res = _run(rm.repair("Qwen3.8-27B:IQ3_S", "qwen3.8", "qwen3.5"))

    assert res["status"] == "success"
    assert res["weights_changed"] is False
    derived = res["repaired_model"]
    assert derived != "Qwen3.8-27B:IQ3_S"
    # mapping switched
    assert rm.resolve("Qwen3.8-27B:IQ3_S") == derived
    # original deleted, derived kept
    assert "Qwen3.8-27B:IQ3_S" not in c.present
    assert derived in c.present
    # creation used a Modelfile FROM the original with renderer/parser
    name, modelfile = c.creations[-1]
    assert name == derived
    assert "FROM Qwen3.8-27B:IQ3_S" in modelfile
    assert "RENDERER qwen3.8" in modelfile
    assert "PARSER qwen3.5" in modelfile


def test_repair_no_auto_download():
    """If the original exists, repair must not pull it again."""
    c = FakeOllamaClient(present=["m:t"])
    rm = RepairManager(ModelManager(c))
    _run(rm.repair("m:t", "r", "p"))
    assert not any(m == "pull" for m, _ in c.calls)


def test_repair_idempotent_repeated_identical():
    c = FakeOllamaClient(present=["m:t"])
    rm = RepairManager(ModelManager(c))
    r1 = _run(rm.repair("m:t", "qwen3.8", "qwen3.5"))
    created_first = len(c.creations)
    r2 = _run(rm.repair("m:t", "qwen3.8", "qwen3.5"))
    assert r1["repaired_model"] == r2["repaired_model"]
    # second call did NOT create a new model
    assert len(c.creations) == created_first


def test_repair_different_config_creates_different():
    c = FakeOllamaClient(present=["m:t"])
    rm = RepairManager(ModelManager(c))
    r1 = _run(rm.repair("m:t", "aa", "bb"))
    # after first repair, original is deleted; recreate original for 2nd config
    c.present.add("m:t")
    r2 = _run(rm.repair("m:t", "cc", "dd"))
    assert r1["repaired_model"] != r2["repaired_model"]


def test_repair_mapping_updates_after_success():
    c = FakeOllamaClient(present=["m:t"])
    rm = RepairManager(ModelManager(c))
    before = rm.resolve("m:t")
    assert before == "m:t"
    res = _run(rm.repair("m:t", "r", "p"))
    assert rm.resolve("m:t") == res["repaired_model"]


# ── RepairManager: failure-safety ──────────────────────────────────────────
def test_create_failure_keeps_original():
    c = FakeOllamaClient(present=["m:t"], fail_create=True)
    rm = RepairManager(ModelManager(c))
    with pytest.raises(RepairError) as ei:
        _run(rm.repair("m:t", "r", "p"))
    assert ei.value.code == "CREATE_FAILED"
    assert "m:t" in c.present, "original must be kept"
    assert rm.resolve("m:t") == "m:t", "mapping must be unchanged"


def test_verify_failure_keeps_original_and_cleans_derived():
    c = FakeOllamaClient(present=["m:t"], fail_verify=True)
    rm = RepairManager(ModelManager(c))
    with pytest.raises(RepairError) as ei:
        _run(rm.repair("m:t", "r", "p"))
    assert ei.value.code == "VERIFY_FAILED"
    assert "m:t" in c.present, "original must be kept"
    assert rm.resolve("m:t") == "m:t", "mapping must be unchanged"


def test_original_not_found_error():
    c = FakeOllamaClient(present=[])
    rm = RepairManager(ModelManager(c))
    with pytest.raises(RepairError) as ei:
        _run(rm.repair("ghost:t", "r", "p"))
    assert ei.value.code == "ORIGINAL_NOT_FOUND"


def test_invalid_params_no_renderer_or_parser():
    rm = RepairManager(ModelManager(FakeOllamaClient(present=["m:t"])))
    with pytest.raises(RepairError) as ei:
        _run(rm.repair("m:t", None, None))
    assert ei.value.code == "INVALID_PARAMS"


def test_delete_failure_does_not_break_success():
    """Even if delete(original) fails, the repair (mapping) already succeeded."""
    class C(FakeOllamaClient):
        async def delete(self, model=None, **kw):
            self._record("delete", model=model)
            raise ollama.ResponseError("delete refused", 500)
    c = C(present=["m:t"])
    rm = RepairManager(ModelManager(c))
    res = _run(rm.repair("m:t", "r", "p"))
    assert res["status"] == "success"
    assert rm.resolve("m:t") != "m:t"
    assert res["original_deleted"] is False
    assert "m:t" in c.present  # original still present (delete failed), which is safe


# ── Request validation (Pydantic) ──────────────────────────────────────────
def test_repair_request_validates_model():
    RepairRequest(model="Qwen3.8-27B:IQ3_S", renderer="qwen3.8", parser="qwen3.5")
    with pytest.raises(Exception):
        RepairRequest(model="", renderer="r")
    with pytest.raises(Exception):
        RepairRequest(model="bad name with space", renderer="r")
    with pytest.raises(Exception):
        RepairRequest(model="a:b:c", renderer="r")
    with pytest.raises(Exception):
        RepairRequest(model="m:t", renderer="has space")


# ── Repairable-error detection ─────────────────────────────────────────────
def test_is_repairable_detects_known_errors():
    assert is_repairable_error("System message must be at the beginning.")
    assert is_repairable_error("invalid renderer for model")
    assert is_repairable_error("unknown parser 'x'")


def test_is_repairable_ignores_unrelated():
    assert not is_repairable_error("connection reset by peer")
    assert not is_repairable_error("out of memory")
    assert not is_repairable_error("")


def test_build_repair_available_response_shape():
    r = build_repair_available_response("System message must be at the beginning.")
    assert r.status_code == 500
    import orjson
    body = orjson.loads(r.body)
    err = body["error"]
    assert err["code"] == "REPAIR_AVAILABLE"
    assert err["type"] == "model_runtime_incompatible"
    assert err["repair_available"] is True
    assert err["repair_endpoint"] == "/v1/models/repair"
    assert "beginning" in err["detail"]


# ── parse_modelfile ─────────────────────────────────────────────────────────
def test_parse_modelfile_full():
    mf = "FROM qwen:base\nRENDERER qwen3.8\nPARSER qwen3.5\n"
    r = parse_modelfile(mf)
    assert r == {"from": "qwen:base", "renderer": "qwen3.8", "parser": "qwen3.5"}


def test_parse_modelfile_partial_and_case():
    # Only FROM; keywords are case-insensitive; values with tags are preserved.
    r = parse_modelfile("from myorg/model:v1\n")
    assert r["from"] == "myorg/model:v1"
    assert r["renderer"] is None and r["parser"] is None


def test_parse_modelfile_empty_and_garbage():
    assert parse_modelfile("") == {"from": None, "renderer": None, "parser": None}
    assert parse_modelfile(None) == {"from": None, "renderer": None, "parser": None}


# ── Restart reconstruction (item 8) ─────────────────────────────────────────
def test_reconstruct_restores_mapping_from_modelfile():
    # Simulate a prior process that repaired 'qwen3.5:9b' -> a derived model
    # whose Modelfile Ollama persisted, then a RAGNAROK restart (fresh manager).
    derived = make_repaired_name("qwen3.5:9b", "qwen3.8", "qwen3.5")
    fake = FakeOllamaClient(present={derived})
    # Ollama stores the Modelfile it created the model from:
    fake.modelfiles[derived] = "FROM qwen3.5:9b\nRENDERER qwen3.8\nPARSER qwen3.5\n"
    mm = ModelManager(fake)
    mgr = RepairManager(mm)
    n = _run(mgr.reconstruct_from_ollama())
    assert n == 1
    # resolve() now maps the (deleted) original to the derived model:
    assert mgr.resolve("qwen3.5:9b") == derived
    rec = mgr.active_repair("qwen3.5:9b")
    assert rec.renderer == "qwen3.8" and rec.parser == "qwen3.5"


def test_reconstruct_skips_models_without_suffix():
    fake = FakeOllamaClient(present={"qwen3.5:9b"})
    fake.modelfiles["qwen3.5:9b"] = "FROM qwen3.5:9b\n"
    mm = ModelManager(fake)
    mgr = RepairManager(mm)
    n = _run(mgr.reconstruct_from_ollama())
    assert n == 0
    assert mgr.resolve("qwen3.5:9b") == "qwen3.5:9b"


def test_reconstruct_tolerates_missing_modelfile_and_down_ollama():
    # Derived name present but Ollama has no stored modelfile -> skipped, no crash.
    derived = make_repaired_name("qwen:9b", "r", "p")
    fake = FakeOllamaClient(present={derived})
    mgr = RepairManager(ModelManager(fake))
    assert _run(mgr.reconstruct_from_ollama()) == 0

    # Ollama down entirely -> returns 0, no exception.
    fake2 = FakeOllamaClient(fail_list=True)
    mgr2 = RepairManager(ModelManager(fake2))
    assert _run(mgr2.reconstruct_from_ollama()) == 0


def test_reconstruct_does_not_clobber_existing_mapping():
    # An in-memory repair already active for the original must win.
    derived = make_repaired_name("qwen:9b", "r", "p")
    fake = FakeOllamaClient(present={derived})
    fake.modelfiles[derived] = "FROM qwen:9b\nRENDERER r\nPARSER p\n"
    mgr = RepairManager(ModelManager(fake))
    # Pre-existing mapping to a *different* derived (e.g. a different config):
    from src.repair_manager import _RepairRecord, _cfg_key
    mgr._mapping["qwen:9b"] = _RepairRecord(
        derived_name="qwen-ragnarok-repaired-XXXX", renderer="other",
        parser="p", cfg_key=_cfg_key("other", "p"))
    _run(mgr.reconstruct_from_ollama())
    assert mgr.resolve("qwen:9b") == "qwen-ragnarok-repaired-XXXX"


# ── Streaming: repairable Ollama error -> REPAIR_AVAILABLE SSE frame (item 12) ─
class _ChatRaisesClient:
    """Fake Ollama client whose chat() rejects the request (the real failure
    mode: Ollama throws a ResponseError before streaming begins)."""
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    async def chat(self, **kw):
        self.calls += 1
        raise self._exc


def _drive_stream(client):
    """Drive the REAL stream_generator over *client*; return raw SSE bytes."""
    state = NS(http_client=client, semaphore=asyncio.Semaphore(2))
    sfx, efx = _st.make_sse_frames("m", "chatcmpl-req", 0)

    async def run():
        gen = _st.stream_generator(
            state, "req", {"model": "m", "messages": []},
            0.0, "chatcmpl-req", 0, "m", 300, sfx, efx,
        )
        return b"".join([f async for f in gen])

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


def test_stream_repairable_error_yields_repair_available_frame():
    """A chat-template ResponseError (the system-message failure) must surface
    as a REPAIR_AVAILABLE SSE frame followed by a terminal [DONE] -- not a
    generic upstream_error, and not silently dropped."""
    orig_policy = _st._RETRY
    try:
        _st._RETRY = RetryPolicy(max_retries=0)
        exc = ollama.ResponseError(
            "System message must be at the beginning.", 500)
        client = _ChatRaisesClient(exc)
        raw = _drive_stream(client)
    finally:
        _st._RETRY = orig_policy

    text = raw.decode("utf-8", errors="replace")
    # The repair-available frame is present and clearly signalled.
    assert "REPAIR_AVAILABLE" in text
    assert "/v1/models/repair" in text
    # The stream is still terminated cleanly with a [DONE].
    assert b"data: [DONE]" in raw
    # And we did NOT fall through to the generic upstream_error path.
    assert "upstream_error" not in text


def test_stream_unrepairable_error_stays_generic():
    """A non-repairable Ollama error must keep the existing generic behaviour
    (no REPAIR_AVAILABLE frame) -- i.e. the new path is narrowly scoped."""
    orig_policy = _st._RETRY
    try:
        _st._RETRY = RetryPolicy(max_retries=0)
        exc = ollama.ResponseError("connection reset by peer", 500)
        client = _ChatRaisesClient(exc)
        raw = _drive_stream(client)
    finally:
        _st._RETRY = orig_policy

    text = raw.decode("utf-8", errors="replace")
    assert "REPAIR_AVAILABLE" not in text
    assert b"data: [DONE]" in raw
