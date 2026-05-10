#!/bin/bash
#
# Install / pull the selected Ollama model
#

set -e

MODEL="${MODEL_NAME:-qwen3:8b}"

echo "  ├─ Stopping any existing Ollama..."
pkill ollama 2>/dev/null || true
sleep 2

echo "  ├─ Starting Ollama serve..."
ollama serve > /dev/null 2>&1 &
sleep 3

echo "  ├─ Pulling model: $MODEL"
if ollama list | grep -q "^$MODEL "; then
    echo "  │  ℹ️  Model already cached"
else
    ollama pull "$MODEL"
    echo "  │  ✅ Model downloaded"
fi

echo "  └─ Done."
