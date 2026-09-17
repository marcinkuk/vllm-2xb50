#!/bin/bash
# build.sh — vLLM XPU image for Qwen3.5/3.8-27B (MTP speculative decoding)
#
# VERIFIED 2026-09-17 against vllm-project/vllm main @ 9612f77077 (latest).
# This file is a TEMPLATE: copy it into a vllm-project/vllm checkout and run
# it from that checkout's root. It is SELF-CONTAINED — every patch is pulled
# from THIS repo (marcinkuk/vllm-2xb50), so the vault is the single source of
# truth for the patch set.
#
# Patch necessity audit (2026-09-17, GitHub API + content check on current main):
#   #54713  MERGED upstream (b28c3e1568, 2026-09-10, "[BugFix] Retain both replay
#           boundaries so an EAGLE resend of a block-aligned prompt still hits").
#           Its get_replay_boundaries() -> tuple[...] change is already in main,
#           so the patch no longer applies.  => NOT APPLIED (was the old
#           'prefixhit' step; removed).
#   #55390  open / #56026 open  -> both covered by the single COMBINED patch in
#           step 2 (a strict superset of both; 12 draft-group tests pass). It
#           fixes the 0% prefix-cache reuse for Qwen3.5/3.8 (separately-prefixed
#           MTP drafter 'mtp.' vs 'model.').  => NEEDED until both merge upstream.
#   vision-tower-cpu-offload / embed-quant / mtp-draft-vocab: NOT upstream  => keep.
#
# Apply order matters: embed-quant (4) BEFORE mtp-vocab (5) — that patch's
# context includes a line added by the embed patch. Every patch below FAILS THE
# BUILD loudly if it no longer applies (upstream drift); it never silently skips.

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
#    Applies strictly (no --3way) on current main. Once #55390/#56026 are merged
#    upstream, remove this whole step.
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
#    This is now a clean git-format patch (vllm/ prefix, self-contained: it adds
#    the drafter-head code and treats the embed patch's lines as context), so it
#    applies with a plain 'git apply' from the repo root. It MUST come AFTER
#    step 4 (embed): its pre-image includes the embed patch's
#    'quant_config=quant_config'/'prefix=...embed_tokens' lines in
#    qwen3_5_mtp.py, so applying it before embed fails.
curl -L "https://raw.githubusercontent.com/${V}/main/qwen35-mtp-draft-vocab.patch" -o /tmp/mtp-vocab.patch
git apply /tmp/mtp-vocab.patch || { echo "FATAL: mtp-vocab patch no longer applies on ${HASH} (apply AFTER embed)"; exit 1; }
NAME=${NAME}-mtp

# 6. Build the XPU image
docker build --cpuset-cpus="0" --memory="16g" --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${NAME} .

echo vllm-intel-xpu:${NAME}
