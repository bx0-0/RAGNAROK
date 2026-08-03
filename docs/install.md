# Platform Setup

## Kaggle (Recommended)

- **GPU:** 2× T4 (**30GB VRAM**)
- **Session:** Up to 30 hours
- **No restrictions** on cloudflared
- Suitable for models up to ~27B with generous context windows

```python
!git clone https://github.com/bx0-0/RAGNAROK.git
%cd RAGNAROK
!bash start.sh --model qwen3.6:35b --verbose-log True --num-ctx 100000
```

## Google Colab (Free Tier)

- **GPU:** 1× T4 (**15GB VRAM**)
- **Session:** Up to 12 hours, auto-disconnects
- ⚠️ Only models under 15GB VRAM will load correctly
- Use smaller models (7B–8B) and moderate `--num-ctx` (16384–32768)

```python
!git clone https://github.com/bx0-0/RAGNAROK.git
%cd RAGNAROK
!bash start.sh --model qwen3.5:9b --verbose-log True --num-batch 2000 --num-ctx 32768 --max-concurrent 2
```

## Local Linux

```bash
git clone https://github.com/bx0-0/RAGNAROK.git
cd RAGNAROK
pip install -r requirements.txt
./scripts/setup.sh   # installs Ollama + cloudflared
bash start.sh --model qwen3.5:9b
```
