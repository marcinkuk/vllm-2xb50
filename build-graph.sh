#!/bin/bash
# build-graph.sh — vLLM XPU image for Qwen3.5/3.8-27B with CUDA/XPU GRAPHS enabled
#                 on dual-Intel Arc (B50/B70, Battlemage) at TP=2 + MTP speculative decoding.
#
# This is build.sh PLUS the XPU-graph enablers. It applies the same 4 base
# patches as build.sh, then the 3 XPU-specific fixes that make TP=2 + MTP +
# graphs actually run on Battlemage, then the graph memory-profiling patch so
# the worker budgets GPU memory for graph capture. Every patch is pulled from
# THIS repo (marcinkuk/vllm-2xb50) — the vault is the single source of truth.
#
# WHAT THIS UNLOCKS (vs. build.sh, which ships eager-only)
#   VLLM_XPU_ENABLE_XPU_GRAPH=1 now actually engages CUDA-graph capture on XPU
#   for TP=2 + MTP instead of forcing eager for every kernel. The graph path in
#   current main (merged #34482 "Support CUDAGraph on XPU", #38193 disable-by-
#   default, #43043 usage) is complete; what was missing for this exact config
#   is what the XPU-specific patches below supply:
#     [6] getmem   : getMemoryInfo zero-free fallback on XPU        (#53990, open)
#     [7] grammar  : keep grammar-bitmask copies on the right stream (#53997, open)
#     [8] tritonar : TP=2 fused allreduce via Triton, envs-gated flag
#                    (upstream analog #54768, open). The flag
#                    VLLM_XPU_TRITON_ALLREDUCE is now DECLARED in vllm/envs.py
#                    (bool, default 0) and the gate reads envs.VLLM_XPU_TRITON_ALLREDUCE,
#                    so enabling it no longer triggers vLLM's "Unknown env var"
#                    warning and the flag participates in env validation.
#     [9] memprof  : let the XPU worker profile + budget graph-capture
#                    memory (was hard-excluded to CUDA-like platforms only)
#     [9b] mtphitfix : MTP/EAGLE + GDN prefix-cache corruption fix
#                     (port of open upstream #57128; drop_eagle_block was
#                     ignored, so a "hit" could reuse unverified draft Mamba
#                     state — the silent-corruption root cause). NEW as of
#                     the 2026-09-20 audit; apply AFTER the 8-patch chain.
#
# RUNTIME (set on the serving process, e.g. in the container entrypoint):
#   VLLM_XPU_ENABLE_XPU_GRAPH=1     <- the switch that turns graphs ON
#   VLLM_XPU_TRITON_ALLREDUCE=1     <- optional; enables the TP=2 Triton AR (patch [8]); declared env flag, off by default
#
# NOTE on patch [8] tritonar: it ADDS new files (xpu_triton_all_reduce.py etc).
#   If a tree already carries those untracked files, `git apply` of the file-
#   creation hunks fails as an apparent "conflict" (false reject). `git clean -fdx`
#   the tree first (or apply on a truly clean checkout) — the patch itself is fine.
#
# ALLREDUCE TRANSPORTS ON THIS STACK (two distinct failure modes, different fixes)
#   (a) oneCCL / Level-Zero IPC — the DEFAULT transport (dist.all_reduce). The
#       07-28 FATAL crash was here: oneCCL zeMemOpenIpcHandle -> ZE_RESULT_ERROR_
#       INVALID_ARGUMENT inside all_reduce; the engine died. Mitigations are the
#       commented oneCCL env vars further down (CCL_ZE_CLOSE_IPC_WA, CCL_ZE_CACHE=0,
#       CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0, CCL_ATL_TRANSPORT=ofi, CCL_ZE_
#       IPC_EXCHANGE=sockets, CCL_TOPO_P2P_ACCESS=0). These are the ones to tune
#       if the FATAL oneCCL IPC crash resurfaces — NOT patch [8].
#   (b) Triton symmetric-memory one-shot — OPTIONAL, patch [8], now gated on the
#       envs-declared VLLM_XPU_TRITON_ALLREDUCE (off by default). The 09-21 error
#       ("L0 error 45 in symm.rendezvous" + "XPU Triton all-reduce init failed;
#       using oneCCL") is this path's init failing on the B50/B70 pair — it is
#       NON-FATAL: the try/except falls back to oneCCL, so serving continues. Do
#       NOT "fix" (b) with (a)'s oneCCL env vars; they are different transports.
#   RECOMMENDATION: leave VLLM_XPU_TRITON_ALLREDUCE unset (off) until the
#       symm.rendezvous L0 transport is verified working on a B50/B70 pair (it
#       works where P2P + symm-mem are supported, e.g. some B70 configs); when
#       on, it only takes over small 1024-aligned bf16 decode all-reduces and
#       falls back to oneCCL for everything else. Rebuild with the new patch
#       ([8] now declares the flag in vllm/envs.py, so the "Unknown env var"
#       warning from the 09-21 log is gone either way).
#
# CROSS-REFERENCES (upstream tracking — all still OPEN as of 2026-09-21)
#   #56917  "[Feature]: TP=2 graph capture + MTP speculative decoding crash on
#            Arc B70 — fix already exists upstream, unmerged"  (== this stack)
#   #54768  "[XPU] Route small TP all-reduces to a Level Zero IPC kernel" —
#            the open upstream analog of patch [8] tritonar. (Do not confuse
#            with #53989, which is a different PR: fused QK-norm+RoPE+gate.)
#   #53990 / #53997  CySpiegel XPU PRs (covered by patches [6]/[7])
#   #57128  MTP/EAGLE + GDN prefix-cache corruption fix — covered by [9b]; the
#            full PR rewrites stale base (sink_blocks / manager registry) so only
#            the minimal find_longest_cache_hit hunk is adopted locally.
#   #53912  "[Bug]: MTP + prefix caching corruption (empty/repeated output)" —
#            the symptom #9b addresses on this exact config.
#   imryanpurdy/Qwen3.8-27B-4x-Intel-B70s  — same model, MTP5 + graphs shipped,
#            144 tok/s, bit-identical canary (docs/CAMPAIGN-2026-09-10-GRAPHS.md)
#
# VERIFY BEFORE TRUSTING (canary): corruption is config-specific, so re-run the
#   deterministic canary (5 prompts, temp=0, sha256 of the 64-token completion)
#   against eager before shipping any graphs build. The 0002/memprof patch only
#   changes memory profiling, not numerics — but the graph path itself must be
#   canaried on YOUR build.
#
# Apply order matters: embed-quant [4] BEFORE mtp-vocab [5]; the XPU patches
# [6]-[8] are independent of the MTP block but applied after it to match the
# validated 8-patch flow (mtpeagle,vision,embed,mtpvocab,getmem,grammar,
# tritonar,memprof), with [9b] mtphitfix last (it touches
# single_type_kv_cache_manager.py, which no earlier patch modifies). Every
# patch FAILS THE BUILD loudly if it no longer applies (upstream drift); it
# never silently skips.
#
# VALIDATED 2026-09-20: all 8 original patches strict `git apply` on
#   vllm-project/vllm main @ 17e50b9b76 / 9679173788 (also 4868312); the 9th
#   (mtphitfix) applies on all three plus on top of the 8-patch chain.
#   RE-VALIDATED 2026-09-21: full 9-patch chain (incl. [9b] mtphitfix) strict
#   `git apply` on current vllm main @ f05b88751, and [9b] alone also applies on
#   27757dde02 / 9679173788 / 4868312. Upstream #57128 (source) and #53912 (bug)
#   both still OPEN/unmerged, so [9b] remains required.
#   RE-VALIDATED 2026-09-21 (again): full 9-patch chain strict `git apply` on
#   newest vllm main @ 04c1f4a4079 (2026-09-21). The new commits since 8902dbb
#   (04c1f4a4 ROCm SWA, 0b7f11a1 routed-experts aux output, 0aee727f CI) are
#   ROCm/CI only -- no XPU/GDN/MTP/attention source changes, so none of [1]-[9b]
#   was superseded. gpu_worker.py lost the enable_return_routed_experts block
#   but patch [9] (graph mem-profiling, ~line 581) is in a separate region and
#   still applies. GDN prefill backend resolves to Triton on XPU (CUDA-only
#   fast paths in _resolve_gdn_prefill_backend); decode uses the FLA
#   fused_recurrent_gated_delta_rule_packed_decode Triton kernel, so
#   #57565 (Mamba-SSU B70 tuned configs) does not accelerate this model.
#   Runtime TP=2+MTP+graphs not yet canaried on real B50/B70 hardware.
#   RE-VALIDATED 2026-09-21 (3rd): full 9-patch chain strict `git apply` on
#   newest vllm main @ db7f1f67 (2026-09-21 12:55 UTC). The 4 commits since
#   04c1f4a4 (7268f6e3 multimodal cache staleness, 15859bb3 structured-output
#   test reorg, 82daf9f5 ROCm CI parity, db7f1f67 XPU CI Ray-UT deselect) touch
#   no XPU/GDN/MTP/attention source — none of [1]-[9b] superseded. Upstream PR
#   states re-checked: #54768 / #53990 / #53997 / #57128 / #57565 all still OPEN,
#   issues #53912 / #56917 still OPEN.
#   RE-VALIDATED 2026-09-21 (4th): full 9-patch chain strict `git apply` on
#   newest vllm main @ 4f145167 (2026-09-21 11:29 UTC). [8] tritonar was
#   reworked to gate on envs.VLLM_XPU_TRITON_ALLREDUCE (flag now declared in
#   vllm/envs.py: TYPE_CHECKING bool default 0 + environment_variables lambda)
#   instead of a raw os.environ read, so the "Unknown env var:
#   VLLM_XPU_TRITON_ALLREDUCE" warning is gone and the flag is validated. The
#   symm.rendezvous L0 error (09-21, non-fatal, oneCCL fallback engaged) is
#   transport-specific: on B50/B70 the Triton path can fail at init, so it stays
#   OFF by default and is opt-in per hardware. PR states re-confirmed via GitHub
#   API: #54768 (analog of [8]) / #53990 / #53997 / #57128 / #57565 / #55390 /
#   #56026 all still OPEN/unmerged; issues #53912 / #56917 still OPEN. No patch
#   superseded by upstream as of 4f145167.

# 1. Hard reset to a clean state and pull the latest upstream code
docker builder prune -a -f
docker image prune -a -f

git switch main
git fetch origin
git reset --hard origin/main
git clean -fdx
git pull

HASH=$(git rev-parse --short HEAD)
DATE=$(date +%Y-%m-%d_%H-%M)
NAME=${DATE}-${HASH}
V=marcinkuk/vllm-2xb50   # this repo — single source of the patches

# 2. MTP draft-group / Mamba+EAGLE boundary fix (#55390 + #56026 COMBINED).
curl -L "https://raw.githubusercontent.com/${V}/main/patches/vllm-mtp-draft-group-annotation-55390-56026.patch" -o /tmp/mtpeagle.patch
git apply /tmp/mtpeagle.patch || { echo "FATAL: mtpeagle (55390+56026) patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-mtpeagle

# 3. Vision-tower CPU offload (VLLM_VISION_CPU_OFFLOAD_GB). Not upstream.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/vision-tower-cpu-offload.patch" -o /tmp/visionoffload.patch
git apply /tmp/visionoffload.patch || { echo "FATAL: vision-offload patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-visionoffload

# 4. Embed-quantization (W4A16 AutoRound: quantized embed_tokens on XPU). Not upstream.
#    git-format; MUST be applied before mtp-vocab (step 5).
curl -L "https://raw.githubusercontent.com/${V}/main/qwen35-embed-quant.patch" -o /tmp/embed-quant.patch
git apply /tmp/embed-quant.patch || { echo "FATAL: embed-quant patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-noembed

# 5. MTP vocab-truncated draft head (40960-token drafter head). Not upstream.
#    MUST come AFTER step 4 (embed): its pre-image includes the embed patch's
#    lines in qwen3_5_mtp.py.
curl -L "https://raw.githubusercontent.com/${V}/main/qwen35-mtp-draft-vocab.patch" -o /tmp/mtp-vocab.patch
git apply /tmp/mtp-vocab.patch || { echo "FATAL: mtp-vocab patch no longer applies on ${HASH} (apply AFTER embed)"; exit 1; }
NAME=${NAME}-mtp

# 6. XPU getMemoryInfo zero-free fallback (CySpiegel #53990, open).
#    Prevents the worker's memory probe from returning 0 free bytes on Arc
#    (getMemInfo() can report 0 before the device is fully resident), which
#    would abort startup or starve KV/graph memory.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-getmemoryinfo-fallback.patch" -o /tmp/getmem.patch
git apply /tmp/getmem.patch || { echo "FATAL: xpu-getmemoryinfo-fallback patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-getmem

# 7. XPU grammar-bitmask stream fix (CySpiegel #53997, open).
#    Keeps grammar bitmask copies on the correct stream so structured-output /
#    guided-decoding masks are valid when the model runs under graph capture.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-grammar-bitmask-stream-fix.patch" -o /tmp/grammar.patch
git apply /tmp/grammar.patch || { echo "FATAL: xpu-grammar-bitmask-stream-fix patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-grammar

# 8. XPU Triton allreduce for TP=2 (upstream analog #54768, open). Opt-in at
#    runtime via VLLM_XPU_TRITON_ALLREDUCE=1 (declared in vllm/envs.py, so no
#    "unknown env var" warning); engages only for world_size == 2. If the
#    symm.rendezvous L0 transport fails on your GPU pair it falls back to
#    oneCCL automatically (init is wrapped in try/except), so enabling it is
#    safe to leave ON.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-triton-allreduce-tp2.patch" -o /tmp/tritonar.patch
git apply /tmp/tritonar.patch || { echo "FATAL: xpu-triton-allreduce-tp2 patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-tritonar

# 9. XPU CUDA-graph memory profiling (net-new; enables graphs to be budgeted).
#    The base gpu_worker determine_available_memory() only called
#    model_runner.profile_cudagraph_memory() for CUDA-like platforms (XPU was
#    hard-excluded with "see #39977"). This lets the XPU worker profile its
#    graph-capture footprint so it reserves room for the captured graphs
#    instead of running short on KV memory when graphs engage under TP=2.
#    Safe on XPU: the capture/profiling path runs through the persistent
#    torch.cuda->torch.xpu shim in xpu_model_runner.py (_torch_cuda_wrapper).
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-cudagraph-memory-profiling.patch" -o /tmp/memprof.patch
git apply /tmp/memprof.patch || { echo "FATAL: xpu-cudagraph-memory-profiling patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-graphmemprof

# 9b. MTP/EAGLE prefix-cache corruption fix (port of upstream #57128, still open/dirty).
#     In MambaManager.find_longest_cache_hit the drop_eagle_block flag was ACCEPTED
#     but IGNORED, so under MTP/EAGLE speculative decoding a "hit" could reuse a Mamba
#     state that still holds UNVERIFIED draft state from a rejected draft position.
#     That is the silent-corruption root cause (symptom: empty or repeated-character
#     output, issue #53912) — it bites exactly this config: GDN (mamba) prefix caching
#     ON + MTP ON. Fix: when drop_eagle_block is set, skip only the FIRST (most
#     recent) checkpoint the finder matches, then keep scanning for the next
#     (older, committed) one — instead of blanking the whole search tail. Self-contained
#     42-line change, verified to `git apply` on 9679173788 / 4868312 / 27757dde02 and
#     on top of the 8-patch chain.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-mtp-prefix-hit-fix.patch" -o /tmp/mtp-hitfix.patch
git apply /tmp/mtp-hitfix.patch || { echo "FATAL: xpu-mtp-prefix-hit-fix patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-mtphitfix

# 10. Build the XPU image (graphs-capable).
docker build --cpuset-cpus="0" --memory="16g" --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${NAME} .

echo vllm-intel-xpu:${NAME}
echo
echo "Serve with:  VLLM_XPU_ENABLE_XPU_GRAPH=1 [VLLM_XPU_TRITON_ALLREDUCE=1] vllm serve ..."
echo "Remember the canary: 5 deterministic prompts, temp=0, sha256 vs eager before trusting graphs."
