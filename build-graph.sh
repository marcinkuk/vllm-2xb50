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
#     [8] tritonar : TP=2 fused allreduce via Triton, opt-in flag   (#53989, open)
#     [9] memprof  : let the XPU worker profile + budget graph-capture
#                    memory (was hard-excluded to CUDA-like platforms only)
#
# RUNTIME (set on the serving process, e.g. in the container entrypoint):
#   VLLM_XPU_ENABLE_XPU_GRAPH=1     <- the switch that turns graphs ON
#   VLLM_XPU_TRITON_ALLREDUCE=1     <- optional; enables the TP=2 Triton AR (patch [8])
#
# CROSS-REFERENCES (upstream tracking — all still OPEN as of 2026-09-19)
#   #56917  "[Feature]: TP=2 graph capture + MTP speculative decoding crash on
#            Arc B70 — fix already exists upstream, unmerged"  (== this stack)
#   #53989 / #53990 / #53997  CySpiegel XPU PRs (covered by patches [8]/[6]/[7])
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
# tritonar,memprof). Every patch FAILS THE BUILD loudly if it no longer applies
# (upstream drift); it never silently skips.
#
# VALIDATED 2026-09-19: all 8 patches strict `git apply` on vllm-project/vllm
#   main @ 5302d1fe40 (also on 4cc15f2121). Runtime TP=2+MTP+graphs not yet
#   canaried on real B50/B70 hardware.

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

# 8. XPU Triton allreduce for TP=2 (CySpiegel #53989, open). Opt-in at runtime
#    via VLLM_XPU_TRITON_ALLREDUCE=1; engages only for world_size == 2.
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

# 10. Build the XPU image (graphs-capable).
docker build --cpuset-cpus="0" --memory="16g" --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${NAME} .

echo vllm-intel-xpu:${NAME}
echo
echo "Serve with:  VLLM_XPU_ENABLE_XPU_GRAPH=1 [VLLM_XPU_TRITON_ALLREDUCE=1] vllm serve ..."
echo "Remember the canary: 5 deterministic prompts, temp=0, sha256 vs eager before trusting graphs."
