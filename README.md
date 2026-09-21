# vllm-2xb50

XPU build recipe + patch set for **Qwen3.5 / Qwen3.8-27B** (MTP speculative
decoding, W4A16-AutoRound). The single source of truth for the patches that
`build.sh` pulls and applies on top of `vllm-project/vllm` `main`.

## Patch necessity audit — 2026-09-17

Verified against `vllm-project/vllm` **main @ `9612f77077`** (the then-latest
upstream) using the GitHub API for PR status and a content check on current
main. **All four patches below apply strictly (`git apply`) in this order on
that main, and the patched tree compiles + `import vllm` succeeds.**

| # | Patch | Upstream status | Needed? | Notes |
|---|-------|-----------------|---------|-------|
| 1 | `patches/vllm-mtp-draft-group-annotation-55390-56026.patch` | **PR #55390 open** + **PR #56026 open** (neither merged) | **YES — until both merge** | Fixes 0% prefix-cache reuse (see below). One strict `git apply`. |
| 2 | `patches/vision-tower-cpu-offload.patch` | Not upstream | **YES** | Qwen3-VL tower CPU offload (`VLLM_VISION_CPU_OFFLOAD_GB`). |
| 3 | `qwen35-embed-quant.patch` | Not upstream | **YES** | Quantized `embed_tokens` on XPU (W4A16 AutoRound). |
| 4 | `qwen35-mtp-draft-vocab.patch` | Not upstream | **YES** | 40 960-token MTP drafter head. **Apply after #3.** |

Upstream PRs that are **no longer needed** (their content is already in main,
so their raw `.patch` files fail to apply — do not re-add them):

- **PR #54713** — `[BugFix] Retain both replay boundaries so an EAGLE resend of a
  block-aligned prompt still hits`. **MERGED** (closed, `merged=true`,
  merge commit `b28c3e1568bfae930f61d4b24940e47528c85d4a`, 2026-09-10).
  Its `get_replay_boundaries() -> tuple[...]` change is already in main, so the
  old `#54713` ("prefixhit") step was **removed** from `build.sh`.

> To re-audit later: `curl -s -H "Authorization: Bearer $GITHUB_TOKEN"
> https://api.github.com/repos/vllm-project/vllm/pulls/<N>` → check `merged` /
> `merged_at`; and `git apply --check <patch>` on a fresh `main`.

## build.sh

Run from the root of a `vllm-project/vllm` checkout. It is self-contained: it
resets to `origin/main`, pulls each patch from **this** repo, and applies them
in order. **Every patch that fails to apply aborts the build** (loud failure,
never a silent skip).

```
1  combined MTP draft-group annotation (#55390 + #56026)   -> -mtpeagle
2  vision-tower CPU offload                                -> -visionoffload
3  embed-quant (W4A16 embed_tokens)                        -> -noembed
4  MTP draft-vocab (40960-token head)  [after #3]          -> -mtp
   docker build -f docker/Dockerfile.xpu -t vllm-intel-xpu:<DATE>-<HASH><tags>
```

The old per-PR patches `#55390`, `#56026`, `#54713` were replaced by the single
combined patch (#1); the old `-prefixhit` / `-swa-offload` / `-hybrid` steps are
gone (see the audit table above for why).

## patches/vision-tower-cpu-offload.patch

Offloads the Qwen3-VL **vision tower** to CPU so the LLM weights fit in XPU
memory, controlled by `VLLM_VISION_CPU_OFFLOAD_GB` (GiB of the tower to move
off-device; `0`/unset = disabled).

- Not upstream; no equivalent in `vllm-project/vllm` main.
- Git-format; applies with `git apply` from the repo root.
- Verified `git apply` clean on upstream vllm main `9612f77077` (2026-09-17).
- Applied by `build.sh` as step 2 (`-visionoffload` tag).

## qwen35-embed-quant.patch

Patch enabling quantized (`embed_tokens`) loading for Qwen3.5 / Qwen3.8-27B
W4A16-AutoRound models on XPU.

- Fixes: `ValueError` when loading `born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast`
  (int8 `embed_tokens`, int4 `lm_head`) on XPU.
- Passes `quant_config` + `prefix` to `VocabParallelEmbedding` in `qwen3_5.py`
  (model) and `qwen3_5_mtp.py` (MTP predictor).
- Git-format; applies with `git apply` from the repo root.
- Verified `git apply` clean on upstream vllm main `9612f77077` (2026-09-17).
- Applied by `build.sh` as step 3 (`-noembed` tag).
## qwen35-mtp-draft-vocab.patch

Enable the MTP drafter's **vocab-truncated draft head** (40 960 tokens) for
`born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast` (and other Qwen3.5/3.8 W4A16
checkpoints that ship `mtp_draft_vocab_ids.pt` + `mtp.draft_lm_head.*`).

- Apply **after** `qwen35-embed-quant.patch` (the build does this automatically,
  step 4 / `-mtp` tag). Its pre-image includes the embed patch's lines, so
  applying it before embed fails.
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
- Now a clean **git-format** patch (`vllm/` prefix); applies with a plain
  `git apply` from the repo root. (Earlier copies were `patch -p1` format
  without the `vllm/` prefix and required `patch -p1 -d vllm`; those no
  longer `git apply`.)
- Verified `git apply` clean on upstream vllm main `9612f77077` (2026-09-17),
  after the embed patch.


## patches/vllm-mtp-draft-group-annotation-55390-56026.patch

Combines **vLLM PR #55390** (MTP draft-group *trailing-layer* annotation) and
**PR #56026** (MTP draft-group *prefix-scope* annotation) into a single patch
for `vllm/v1/core/kv_cache_utils.py` + its test file, re-anchored to current
upstream `vllm-project/vllm` `main` (validated on `9612f77077`, 2026-09-17).

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
# from a fresh vllm-project/vllm checkout at or after 9612f77077
git apply patches/vllm-mtp-draft-group-annotation-55390-56026.patch
```

- Validated `git apply` clean on upstream main `9612f77077` (2026-09-17);
  compiles, and all MTP/EAGLE draft-group tests in
  `tests/v1/core/test_kv_cache_utils.py` pass (the trailing-layer and
  prefix-scope tests from both PRs).
- Supersedes both upstream PRs (both still `open`/unmerged against current
  main, and their raw `.patch` files no longer apply cleanly due to context
  drift — so this combined patch is the local replacement). See the upstream
  comment on PR #55390.
- **Applied by `build.sh` as step 1** (image tag gets `-mtpeagle`). Re-audit
  before each build: once **#55390 and #56026 are both merged**, delete this
  patch and the build.sh step — main will carry the fix natively.

## Tests

`testy/symm_rendezvous_test.py` — checks whether the torch symmetric-memory
(Level-Zero) transport comes up on this GPU pair, i.e. whether the Triton
one-shot all-reduce (patch [8], `VLLM_XPU_TRITON_ALLREDUCE=1`) can engage at
all on B50/B70. Three legs: (0) oneCCL baseline, (1) `symm.rendezvous`
(the 09-21 "L0 error 45" spot), (2) an all_reduce through the mapped peer
buffers. Needs the `-tritonar` image; no model load, a few seconds.

Run standalone (fresh container) or alongside the live server:

```sh
# standalone (fresh container, both render nodes, the compose's CCL_* env)
# --entrypoint python overrides the image's ENTRYPOINT (vllm); the image
# must be a -tritonar one (leg 1 imports the module patch [8] adds)
docker run --rm \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=1073741824 \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1073741824 \
  -v "$PWD/testy/symm_rendezvous_test.py":/tmp/symm_test.py:ro \
  --entrypoint python vllm-intel-xpu:TAG /tmp/symm_test.py

# alongside the live server (no downtime; see the script's docstring for
# the residual-risk note and the watch/restart procedure; --entrypoint
# python is needed because exec uses the IMAGE's entrypoint, not the
# container's compose override)
docker cp testy/symm_rendezvous_test.py <container>:/tmp/symm_test.py
docker exec -e MASTER_PORT=29617 --entrypoint python <container> /tmp/symm_test.py
```

Verdicts: `ONECCL_BASELINE_FAIL` = your env (not the transport) is broken;
`SYMM_TRANSPORT_FAIL (<phase>)` = keep the flag off, oneCCL is your transport;
`SYMM_TRANSPORT_OK` = the Triton path is viable, enable the flag at the next
planned restart.
