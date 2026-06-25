#!/bin/bash
#
# Setup script — Install all dependencies
#

set -e

echo "  ├─ Cleaning old processes..."
if pgrep -f cloudflared >/dev/null 2>&1; then pkill -f cloudflared; fi
if pgrep -f "src.server" >/dev/null 2>&1; then pkill -f "src.server"; fi
if fuser 8000/tcp >/dev/null 2>&1; then fuser -k 8000/tcp; fi
# Do not force-re-download cloudflared; existence check handles it later

echo "  ├─ Installing zstd (required by Ollama)..."
apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq zstd > /dev/null 2>&1 || true

echo "  ├─ Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    max_retries=3
    for attempt in $(seq 1 $max_retries); do
        echo "  │  Attempt ${attempt}/${max_retries}..."
        if curl -fsSL --http1.1 https://ollama.com/install.sh | sh > /dev/null 2>&1; then
            echo "  │  ✅ Ollama installed successfully"
            break
        fi
        if [ "$attempt" -eq "$max_retries" ]; then
            echo "  └─ ❌ Failed to install Ollama after ${max_retries} attempts. Please install it manually:"
            echo "     curl -fsSL --http1.1 https://ollama.com/install.sh | sh"
            exit 1
        fi
        echo "  │  ⚠️  Retry in 3s..."
        sleep 3
    done
else
    echo "  │  ℹ️  Ollama already installed"
fi

echo "  ├─ Installing Python packages..."
pip install -q -r requirements.txt

echo "  ├─ Downloading cloudflared..."
if [ ! -f "cloudflared" ] || [ ! -x "cloudflared" ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
    echo "  │  ✅ cloudflared downloaded"
else
    echo "  │  ℹ️  cloudflared already present"
fi

echo "  └─ Done."
