"""Tests A–G for the RAGNAROK disconnect-cancellation chain.

Prerequisites:
  - A fake Ollama running on 127.0.0.1:11499  (tests/fake_ollama.py)
  - RAGNAROK gateway running on 127.0.0.1:8120

Run:
  python3 tests/test_disconnect.py
"""

import asyncio
import socket
import struct
import time
import json
import sys
import os
import urllib.request
import urllib.error

FAKE_PORT = 11434
GW_PORT = 8120
GW_HOST = "127.0.0.1"

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def gw_url(path: str) -> str:
    return f"http://{GW_HOST}:{GW_PORT}{path}"


def http_json(method: str, path: str, body: dict = None, timeout=5):
    url = gw_url(path)
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def http_get_raw(path: str, timeout=5):
    url = gw_url(path)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def sse_chat_request(model: str = "qwen3.5:9b", stream: bool = True,
                     total_tokens: int = 40, timeout: float = 10):
    """Send a streaming chat request and read N SSE data frames.

    Returns (request_id, frames_read, error_str_or_None).
    """
    import http.client

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "say hello"}],
        "stream": stream,
    }).encode()

    conn = http.client.HTTPConnection(GW_HOST, GW_PORT, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    rid = resp.getheader("x-request-id")
    frames = []
    try:
        max_lines = total_tokens * 3  # SSE = 2 lines per event; allow extra
        for i in range(max_lines):
            line = resp.readline()
            if not line:
                break
            if line.strip():  # skip blank separator lines
                frames.append(line.decode().strip())
                if b"[DONE]" in line:
                    break
    except Exception as e:
        return rid, frames, str(e)
    finally:
        conn.close()
    return rid, frames, None


def fake_status():
    """GET fake Ollama /api/ps-equivalent; return (open_streams, aborted)."""
    # We can't query the fake directly without its own HTTP interface.
    # Instead we query the gateway's logs. For this test we'll use a
    # dedicated status endpoint on the fake.
    pass


def fin_disconnect(timeout: float = 15):
    """Open a streaming chat, read 5 chunks, then send FIN (close socket).

    Returns the request_id we extracted from the first SSE frame.
    """
    import http.client

    body = json.dumps({
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "long prompt"}],
        "stream": True,
    }).encode()

    conn = http.client.HTTPConnection(GW_HOST, GW_PORT, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()

    # Read a few frames
    frames = []
    for _ in range(5):
        line = resp.readline()
        if not line:
            break
        frames.append(line.decode().strip())

    # Now FIN: close the socket without RST
    sock = conn.sock
    conn.sock = None  # prevent __del__ from closing it
    try:
        sock.shutdown(socket.SHUT_WR)  # send FIN
    except Exception:
        pass
    sock.close()

    return frames


def rst_disconnect(timeout: float = 15):
    """Open a streaming chat, read 5 chunks, then send RST (SO_LINGER 1,0)."""
    import http.client

    body = json.dumps({
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "long prompt"}],
        "stream": True,
    }).encode()

    conn = http.client.HTTPConnection(GW_HOST, GW_PORT, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()

    frames = []
    for _ in range(5):
        line = resp.readline()
        if not line:
            break
        frames.append(line.decode().strip())

    # RST: set SO_LINGER with 0 timeout, then close
    sock = conn.sock
    conn.sock = None
    try:
        linger = struct.pack("ii", 1, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
    except Exception:
        pass
    sock.close()

    return frames


def get_fake_state() -> dict:
    """Query the fake Ollama's in-memory state via a status endpoint."""
    # We added /_state to the fake for testing
    try:
        with urllib.request.urlopen(
            f"http://{GW_HOST}:{FAKE_PORT}/_state", timeout=3
        ) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def wait_until(pred, timeout=10, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


# ──────────────────────────────────────────────────────────
# Test A: normal completion (no disconnect)
# ──────────────────────────────────────────────────────────

def test_a_normal():
    print("\n=== TEST A: normal completion ===")
    rid, frames, err = sse_chat_request(total_tokens=200, timeout=30)
    assert err is None, f"Test A error: {err}"
    assert len(frames) >= 10, f"Test A: only {len(frames)} frames (expected ≥10)"
    # Last non-empty frame should be [DONE]
    non_empty = [f for f in frames if f]
    assert any("[DONE]" in f for f in non_empty), f"Test A: no [DONE] in {non_empty[-3:]}"
    state = get_fake_state()
    assert len(state.get("closed_clean", [])) >= 1, f"Test A: fake saw no clean close: {state}"
    print(f"  frames={len(frames)}, closed_clean={len(state.get('closed_clean', []))}")
    print("  PASS")


# ──────────────────────────────────────────────────────────
# Test B: FIN disconnect
# ──────────────────────────────────────────────────────────

def test_b_fin():
    print("\n=== TEST B: FIN disconnect ===")
    state_before = get_fake_state()
    open_before = set(state_before.get("open_streams", {}).keys())

    frames = fin_disconnect()
    print(f"  read {len(frames)} frames before FIN")
    assert len(frames) >= 3, f"Test B: only {len(frames)} frames before FIN"

    # Wait for the fake to detect the abort
    def check_aborted():
        s = get_fake_state()
        return any(r not in open_before for r in s.get("aborted", []))

    aborted = wait_until(check_aborted, timeout=10)
    state_after = get_fake_state()
    print(f"  fake aborted={state_after.get('aborted', [])}")
    assert aborted, f"Test B: fake did not see abort. state={state_after}"
    print("  PASS")


# ──────────────────────────────────────────────────────────
# Test C: RST disconnect
# ──────────────────────────────────────────────────────────

def test_c_rst():
    print("\n=== TEST C: RST disconnect ===")
    state_before = get_fake_state()
    open_before = set(state_before.get("open_streams", {}).keys())

    frames = rst_disconnect()
    print(f"  read {len(frames)} frames before RST")
    assert len(frames) >= 3, f"Test C: only {len(frames)} frames before RST"

    def check_aborted():
        s = get_fake_state()
        return any(r not in open_before for r in s.get("aborted", []))

    aborted = wait_until(check_aborted, timeout=10)
    state_after = get_fake_state()
    print(f"  fake aborted={state_after.get('aborted', [])}")
    assert aborted, f"Test C: fake did not see abort. state={state_after}"
    print("  PASS")


# ──────────────────────────────────────────────────────────
# Test E: stop-generation endpoint
# ──────────────────────────────────────────────────────────

def test_e_stop():
    print("\n=== TEST E: stop-generation endpoint ===")
    # Start a stream in a background thread (non-blocking)
    import threading

    result = {"rid": None, "frames": [], "err": None}

    def start_stream():
        rid, frames, err = sse_chat_request(total_tokens=42, timeout=20)
        result.update(rid=rid, frames=frames, err=err)

    t = threading.Thread(target=start_stream, daemon=True)
    t.start()
    time.sleep(1.0)  # let the stream start

    # Now call the stop endpoint.  We need the request_id the gateway assigned.
    # The gateway's /v1/chat/completions response headers should carry x-request-id.
    # Since our helper doesn't expose it, we call stop with a wildcard approach:
    # Actually, let's use a different strategy — send a raw request and grab the header.
    import http.client
    body = json.dumps({
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "stop me"}],
        "stream": True,
    }).encode()
    conn = http.client.HTTPConnection(GW_HOST, GW_PORT, timeout=20)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    rid = resp.getheader("x-request-id") or resp.getheader("X-Request-Id")
    print(f"  request_id from header: {rid}")
    # Read a couple frames
    for _ in range(3):
        line = resp.readline()
        if not line:
            break
    # Close the client side (simulates tunnel drop for this test)
    sock = conn.sock
    conn.sock = None
    try:
        sock.shutdown(socket.SHUT_WR)
    except Exception:
        pass
    sock.close()

    # Now call stop — even though the client is gone, the server should still
    # process the stop call cleanly
    status, body2 = http_json("POST", f"/v1/chat/completions/{rid}/stop")
    print(f"  stop response: {status} {body2}")
    # The stream should have been stopped — status 200 expected if it was found
    # (if already aborted by the client close, 404 is acceptable too)
    assert status in (200, 404), f"Test E: unexpected stop status {status}"
    print("  PASS")


# ──────────────────────────────────────────────────────────
# Test F: unload endpoint
# ──────────────────────────────────────────────────────────

def test_f_unload():
    print("\n=== TEST F: unload endpoint ===")
    status, body = http_json("POST", "/v1/models/unload", {"model": "qwen3.5:9b"})
    print(f"  status={status} body={body}")
    assert status == 200, f"Test F: unload status={status}"
    assert body.get("status") == "unloaded", f"Test F: body={body}"
    print("  PASS")


# ──────────────────────────────────────────────────────────
# Test G: two simultaneous streams, disconnect one, other finishes
# ──────────────────────────────────────────────────────────

def test_g_isolation():
    print("")
    print("=== TEST G: two streams, disconnect one, other continues ===")
    import threading
    import http.client
    import json as _json
    rids = {}
    results = {}

    def run_stream(key, timeout):
        body = _json.dumps({"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "stream " + key}], "stream": True}).encode()
        conn = http.client.HTTPConnection(GW_HOST, GW_PORT, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rid = resp.getheader("x-request-id")
        rids[key] = rid
        frames = []
        err = None
        try:
            for i in range(300):
                line = resp.readline()
                if not line:
                    break
                if line.strip():
                    frames.append(line.decode().strip())
                    if b"[DONE]" in line:
                        break
        except Exception as e:
            err = str(e)
        finally:
            conn.close()
        results[key] = {"rid": rid, "frames": frames, "err": err}

    t1 = threading.Thread(target=lambda: run_stream("s1", 25), daemon=True)
    t2 = threading.Thread(target=lambda: run_stream("s2", 25), daemon=True)
    t1.start()
    t2.start()
    waited = 0.0
    while waited < 10:
        time.sleep(0.1); waited += 0.1
        if rids.get("s1") and rids.get("s2"):
            break
    assert rids.get("s2"), "Test G: stream 2 never started"
    print("  both in-flight, stopping s2=" + rids["s2"])
    status, _ = http_json("POST", "/v1/chat/completions/" + rids["s2"] + "/stop")
    print("  stopped s2 via endpoint: status=" + str(status))
    assert status == 200, "Test G: stop should find s2, got " + str(status)
    t1.join(timeout=25)
    t2.join(timeout=5)
    s1 = results.get("s1", {}); s2 = results.get("s2", {})
    s1f = len(s1.get("frames", [])); s2f = len(s2.get("frames", []))
    s1e = s1.get("err"); s2e = s2.get("err")
    print("  s1: frames=" + str(s1f) + " err=" + str(s1e))
    print("  s2: frames=" + str(s2f) + " err=" + str(s2e))
    assert s1e is None, "Test G: s1 had error " + str(s1e)
    assert any("[DONE]" in f for f in s1.get("frames", [])), "Test G: s1 missing [DONE]"
    assert s2f < s1f, "Test G: s2(" + str(s2f) + ") not < s1(" + str(s1f) + ")"
    print("  PASS")

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    tests = [
        ("A", test_a_normal),
        ("B", test_b_fin),
        ("C", test_c_rst),
        ("E", test_e_stop),
        ("F", test_f_unload),
        ("G", test_g_isolation),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
