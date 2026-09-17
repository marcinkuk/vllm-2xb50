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


## patches/vllm-mtp-draft-group-annotation-55390-56026.patch

Combines **vLLM PR #55390** (MTP draft-group *trailing-layer* annotation) and
**PR #56026** (MTP draft-group *prefix-scope* annotation) into a single patch
for `vllm/v1/core/kv_cache_utils.py` + its test file, re-anchored to current
upstream `vllm-project/vllm` `main` (applied on top of `88afb777008`).

### Why it is needed (0% prefix-cache reuse)

Qwen3.5 / Qwen3.8-27B register their MTP drafter layers under a **separate
top-level module prefix** (`mtp.…`) while the target model uses `model.…`.
With speculative decoding + block drop, `_annotate_eagle_groups` had no way to
tell which KV-cache groups hold the drafter's attention, so the coordinator
treated **every** group (including the Mamba/GDN groups) as draft groups.
Cross-request prefix-cache reuse for those groups then silently dropped to
**0%**.

### What it does

`_annotate_eagle_groups` now detects drafter groups in order of preference:
1. **Spec-driven** - `non_causal_multi_token_decode` marker (Kimi-K3 DSpark).
2. **Trailing-layer** (opt-in via `_uses_trailing_mtp_layers`, PR #55390) -
   flag the group holding the last registered layer.
3. **Drafter-prefix fallback** (PR #56026) - when 1-2 flag nothing and the
   groups partition `kv_cache_spec` exactly: if the last registered layer's
   top-level prefix differs from the target's (`mtp.` vs `model.`), flag every
   group holding one of those layers. The Mamba/GDN groups stay unflagged and
   keep serving prefix hits. Same-prefix (EAGLE) drafters are not detected and
   fall back to the conservative all-groups behaviour (with a warning).

### Apply

```sh
# from a fresh vllm-project/vllm checkout at or after 88afb777008
git apply patches/vllm-mtp-draft-group-annotation-55390-56026.patch
```

- Validated `git apply --check` clean on upstream main `88afb777008`
  (`[CPU][s390x] Pin protobuf ...`); compiles, and all 16 MTP/EAGLE draft-group
  tests in `tests/v1/core/test_kv_cache_utils.py` pass (the trailing-layer and
  prefix-scope tests from both PRs).
- Supersedes both upstream PRs (both were `dirty`/`open` against current main,
  so neither applied cleanly). See the upstream comment on PR #55390.
- Not applied by `build.sh` (that path builds from `docker.patch`); apply it
  manually to a vllm checkout before your own build if you need it.
