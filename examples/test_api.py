#!/usr/bin/env python3
"""
Quick test — send a request to your gateway and verify it works.

Usage:
    python3 examples/test_api.py
    python3 examples/test_api.py --url https://your-abc.trycloudflare.com/v1
"""

import argparse
import sys

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

DEFAULT_URL = "http://localhost:8000/v1"

def test_models(url):
    print(f"Testing GET {url}/models ...")
    resp = httpx.get(f"{url}/models", timeout=10)
    resp.raise_for_status()
    print(f"  Response: {resp.json()}")
    return True

def test_chat(url):
    print(f"Testing POST {url}/chat/completions ...")
    resp = httpx.post(
        f"{url}/chat/completions",
        json={
            "model": "any",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say hello in 5 words."},
            ],
            "max_tokens": 50,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    print(f"  Response: {msg.get('content', '(empty)')}")
    print(f"  Tokens: {data.get('usage', {})}")
    return True

def test_chat_stream(url):
    print(f"Testing streaming POST {url}/chat/completions ...")
    received = []
    with httpx.stream(
        "POST",
        f"{url}/chat/completions",
        json={
            "model": "any",
            "messages": [
                {"role": "user", "content": "Count to 3, one word per line."},
            ],
            "stream": True,
            "max_tokens": 30,
        },
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                import json
                data = json.loads(payload)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    received.append(content)
                    print(content, end="", flush=True)
            except Exception:
                pass
    print()
    print(f"  Full response: {''.join(received)}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test Kaggle Ollama Gateway")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL (default: localhost:8000/v1)")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    passed = 0
    for name, fn in [
        ("List models", test_models),
        ("Chat (non-stream)", test_chat),
        ("Chat (stream)", test_chat_stream),
    ]:
        try:
            fn(base)
            print(f"  ✅ {name} passed\n")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} failed: {e}\n")

    print(f"Results: {passed}/3 passed")
    sys.exit(0 if passed == 3 else 1)

if __name__ == "__main__":
    main()
