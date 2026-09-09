
# ── HTTP endpoint tests (route wiring + validation + integration) ───────────
# Mount ONLY the models router on a bare FastAPI app with a fake gw state.
# Avoids importing src.server (which pulls uvloop / heavy deps).
import asyncio as _asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.model_manager import ModelManager
from src.repair_manager import RepairManager
from tests.test_model_repair import FakeOllamaClient


def _client_with_gw(fake_ollama, state=None):
    from src.routes.models import router as models_router
    from src.state import GatewayState
    if state is None:
        state = GatewayState()
    state.http_client = fake_ollama
    state.models = ModelManager(fake_ollama)
    state.repairs = RepairManager(state.models)
    app = FastAPI()
    app.state.gw = state
    app.include_router(models_router)
    return TestClient(app), state


def _run2(coro):
    return _asyncio.new_event_loop().run_until_complete(coro)


def test_endpoint_repair_success():
    fake = FakeOllamaClient(present=["Qwen3.8-27B:IQ3_S"])
    client, state = _client_with_gw(fake)
    r = client.post("/v1/models/repair", json={
        "model": "Qwen3.8-27B:IQ3_S", "renderer": "qwen3.8", "parser": "qwen3.5",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["weights_changed"] is False
    assert body["repaired_model"] != "Qwen3.8-27B:IQ3_S"
    # client-visible model name is still the ORIGINAL
    assert body["active_model"] == "Qwen3.8-27B:IQ3_S"
    # mapping switched
    assert state.repairs.resolve("Qwen3.8-27B:IQ3_S") == body["repaired_model"]


def test_endpoint_repair_invalid_params_returns_400():
    fake = FakeOllamaClient(present=["m:t"])
    client, _ = _client_with_gw(fake)
    r = client.post("/v1/models/repair", json={"model": "m:t"})
    assert r.status_code == 400
    assert "renderer" in r.text or "parser" in r.text or r.json()["error"]["code"] == "INVALID_PARAMS"


def test_endpoint_repair_bad_model_name_400():
    fake = FakeOllamaClient(present=["m:t"])
    client, _ = _client_with_gw(fake)
    r = client.post("/v1/models/repair", json={"model": "bad name", "renderer": "r"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_endpoint_repair_original_not_found_500():
    fake = FakeOllamaClient(present=[])
    client, _ = _client_with_gw(fake)
    r = client.post("/v1/models/repair", json={"model": "ghost", "renderer": "r"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "ORIGINAL_NOT_FOUND"
    assert r.json()["error"]["repairable"] is True


def test_endpoint_repair_create_failure_500_keeps_original():
    fake = FakeOllamaClient(present=["m:t"], fail_create=True)
    client, _ = _client_with_gw(fake)
    r = client.post("/v1/models/repair", json={"model": "m:t", "renderer": "r"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "CREATE_FAILED"
    assert "m:t" in fake.present
