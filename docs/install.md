# 🖥️ Platform Setup & Model Sources

## <img src="../assets/search.png" width="30" align="middle"> Model Sources

### 1. Ollama Library Models

Any model from the [Ollama library](https://ollama.com/library). Just specify the name:

```bash
bash start.sh --model qwen3.6:35b
```

### 2. HuggingFace GGUF Models

Pull any GGUF model directly from HF with a quantization tag. This is great for models not in Ollama's library:

```bash
bash start.sh --model hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0
```

> ⚠️ **GPU size matters.** Colab free tier has 15GB VRAM — only models that fit will load. Kaggle offers 30GB VRAM for larger models. Check model quantization sizes before pulling.

Browse [bartowski GGUF repos](https://huggingface.co/bartowski) for heavily optimized quantized models.

---

## <img src="../assets/fire.png" width="30" align="middle"> Kaggle (Recommended)

- **GPU:** 2× T4 (**30GB VRAM**)
- **Session:** Up to 30 hours
- **No restrictions** on cloudflared tunneling
- Suitable for models up to ~27B with generous context windows

```python
!git clone https://github.com/bx0-0/RAGNAROK.git
%cd RAGNAROK
!bash start.sh --model qwen3.6:35b --verbose-log True --num-ctx 100000
```

### Why Kaggle?

| Feature | Kaggle | Colab Free |
|---|---|---|
| VRAM | **30GB** (2× T4) | 15GB (1× T4) |
| Session | Up to 30h | Up to 12h |
| Cloudflare tunnel | No restrictions | Sometimes blocked |
| Idle disconnect | Rare | Aggressive |

---

## <img src="../assets/fire.png" width="30" align="middle"> Google Colab (Free Tier)

- **GPU:** 1× T4 (**15GB VRAM**)
- **Session:** Up to 12 hours, auto-disconnects on inactivity
- ⚠️ Only models under ~10GB VRAM will load reliably with context room
- Use smaller models (7B–8B) and moderate `--num-ctx`

```python
!git clone https://github.com/bx0-0/RAGNAROK.git
%cd RAGNAROK
!bash start.sh --model qwen3.5:9b --verbose-log True --num-batch 2000 --num-ctx 32768 --max-concurrent 2
```

---

## Local Linux

Run on your own machine with GPU or CPU:

```bash
git clone https://github.com/bx0-0/RAGNAROK.git
cd RAGNAROK
pip install -r requirements.txt
./scripts/setup.sh          # installs Ollama + cloudflared
bash start.sh --model qwen3.5:9b
```

> **CPU-only mode:** Set `--num-gpu 0` to run entirely on CPU (slow but works for small models).

---

## VRAM Guidelines

| Model Size | Min VRAM | Recommended Context |
|---|---|---|
| 1B–3B | 6GB | 32K+ tokens |
| 7B–8B | 10GB | 16K–32K tokens |
| 9B–14B | 16GB | 8K–16K tokens |
| 27B–35B | 24GB | 4K–8K tokens |

> Rule of thumb: Each 1B parameters ≈ ~2GB VRAM at Q4 quantization. Increase `--num-ctx` only if VRAM allows.
