# vllm-2xb50

## qwen35-embed-quant.patch

4-line patch enabling quantized (`embed_tokens`) loading for Qwen3.5 / Qwen3.8-27B W4A16-AutoRound models on XPU.

- Fixes: `ValueError` when loading `born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast` (int8 embed_tokens, int4 lm_head) on XPU.
- Passes `quant_config` + `prefix` to `VocabParallelEmbedding` in `qwen3_5.py` (model) and `qwen3_5_mtp.py` (MTP predictor).
- Verified `git apply --check` clean on upstream vllm main (29af8bd, 2026-07-30).
- Applied automatically by `build.sh` after `docker.patch` (image tag gets `-p` when docker.patch applies).