# Claude Code Integration Guide

## Setup

Replace `YOUR-URL` with your Cloudflare tunnel URL from the gateway output.

```bash
claude \
  --api-model claude-sonnet-4-20250514 \
  --api-base https://YOUR-URL.trycloudflare.com/v1 \
  --api-key not-needed
```

## Shell Profile (Permanent)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export ANTHROPIC_BASE_URL=https://YOUR-URL.trycloudflare.com/v1
export ANTHROPIC_API_KEY=not-needed
```

Then run Claude Code normally.

## Notes

- The model name in requests doesn't matter — the gateway routes everything to your configured Ollama model.
- For best results, use a 7B-14B model for Claude Code compatibility.
