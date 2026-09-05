# vllm-2xb50

## qwen35-embed-quant.patch

4-line patch enabling quantized (`embed_tokens`) loading for Qwen3.5 / Qwen3.8-27B W4A16-AutoRound models on XPU.

- Fixes: `ValueError` when loading `born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast` (int8 embed_tokens, int4 lm_head) on XPU.
- Passes `quant_config` + `prefix` to `VocabParallelEmbedding` in `qwen3_5.py` (model) and `qwen3_5_mtp.py` (MTP predictor).
- Verified `git apply --check` clean on upstream vllm main (29af8bd, 2026-07-30).
- Applied automatically by `build.sh` after `docker.patch` (image tag gets `-p` when docker.patch applies).
## qwen35-mtp-draft-vocab.patch

Enable the MTP drafter's **vocab-truncated draft head** (40 960 tokens) for
`born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast` (and other Qwen3.5/3.8 W4A16
checkpoints that ship `mtp_draft_vocab_ids.pt` + `mtp.draft_lm_head.*`).

- Apply **after** `qwen35-embed-quant.patch` (the build does this automatically;
  image tag gets `-nomtp` if it fails to apply).
- Without it, the MTP drafter constructs the **full ~248k-row** `lm_head`
  even though the checkpoint only provides 40 960 `draft_lm_head` rows, so
  the drafter head is left unpopulated and speculative decoding is
  effectively random.
- With it: a 40 960-row `ParallelLMHead` is built, the checkpoint's
  `mtp.draft_lm_head.*` weights are mapped onto it, and
  `compute_logits` scatters draft logits back to the full vocab
  (`-inf` elsewhere). Speculative decoding stays **exact** — only the
  drafter's proposal distribution changes.
- Disable with `MTP_DRAFT_VOCAB=0`. Log line on load:
  `MTP drafter uses a 40960-token draft head`.
- Validated `git apply --check` clean on upstream vllm main (29af8bd) and on
  `Suppressor72/vllm` `fix/mamba-retention-eagle-boundary` (74a6570), after the
  embed patch.
