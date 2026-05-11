#!/bin/bash
#
# Setup script — Install all dependencies
#

set -e

echo "  ├─ Cleaning old processes..."
pkill -f cloudflared 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
rm -f cloudflared 2>/dev/null || true

echo "  ├─ Installing zstd & toilet (required by Ollama & banners)..."
apt-get update -qq && apt-get install -y -qq zstd toilet toilet-fonts fonts-extra > /dev/null 2>&1 || true

echo "  ├─ Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    echo "  │  ✅ Ollama installed"
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
