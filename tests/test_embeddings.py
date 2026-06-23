"""Live integration tests for the /v1/embeddings endpoint.
Run: pytest tests/test_embeddings.py -v
Requires an embedding model loaded (e.g. embeddinggemma:300m).
"""

import json
import httpx
import pytest

BASE = "https://constraint-viewing-strengths-bride.trycloudflare.com/v1"
EMBEDDING_MODEL = "embeddinggemma:300m"
TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=TIMEOUT)


import pytest


class TestEmbeddingsBasic:
    def test_single_string_input(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "Hello world",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        emb = data["data"][0]
        assert emb["object"] == "embedding"
        assert emb["index"] == 0
        assert isinstance(emb["embedding"], list)
        assert len(emb["embedding"]) > 0

    def test_list_of_strings(self, client):
        texts = ["first doc", "second doc", "third doc"]
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["data"]) == 3
        for i, emb in enumerate(data["data"]):
            assert emb["index"] == i
            assert isinstance(emb["embedding"], list)

    def test_embedding_consistency(self, client):
        """Same input should produce the same embedding vector."""
        texts = ["test consistency"]
        r1 = client.post("/embeddings", json={"model": EMBEDDING_MODEL, "input": texts})
        r2 = client.post("/embeddings", json={"model": EMBEDDING_MODEL, "input": texts})
        e1 = r1.json()["data"][0]["embedding"]
        e2 = r2.json()["data"][0]["embedding"]
        assert len(e1) == len(e2)
        for a, b in zip(e1, e2):
            assert abs(a - b) < 1e-5

    def test_usage_included(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "test",
        })
        assert r.status_code == 200
        data = r.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_model_in_response(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "hi",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == EMBEDDING_MODEL

    def test_missing_input_returns_400(self, client):
        r = client.post("/embeddings", json={})
        assert r.status_code == 400

    def test_empty_input_list(self, client):
        """Empty list may succeed with zero embeddings or fail — both OK."""
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": [],
        })
        assert r.status_code in (200, 400, 500)

    def test_long_text(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "This is a longer test sentence with multiple words. " * 50,
        })
        assert r.status_code == 200

    def test_special_characters(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "Hello! @world #test 🚀 <b>html</b>",
        })
        assert r.status_code == 200

    def test_unicode(self, client):
        r = client.post("/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "こんにちは 你好 مرحبا שלום",
        })
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
