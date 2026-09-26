#!/bin/bash
# build-graph.sh — vLLM XPU image for Qwen3.5/3.8-27B with CUDA/XPU GRAPHS enabled
#                 on dual-Intel Arc (B50/B70, Battlemage) at TP=2 + MTP speculative decoding.
#
# This is build.sh PLUS the XPU-graph enablers. It applies the same 4 base
# patches as build.sh, then the XPU-specific fixes that make TP=2 + MTP +
# graphs actually run on Battlemage (XPU CUDA-graphs are default-on upstream
# since #51600/dcfc17e0b, 2026-09-24, so no env switch is needed at serve
# time). Every patch is pulled from THIS repo (marcinkuk/vllm-2xb50) — the
# vault is the single source of truth.
#
# WHAT THIS UNLOCKS (vs. build.sh, which ships eager-only)
#   XPU CUDA-graph capture is DEFAULT-ON for this config since upstream
#   #51600 (merged 2026-09-24, dcfc17e0b): it deleted the old
#   VLLM_XPU_ENABLE_XPU_GRAPH opt-in env var, flipped XPUPlatform to
#   graphs-by-default (opt out with --enforce-eager), and made the worker
#   budget graph-capture memory on XPU (which is what our former [9] memprof
#   patch used to inject). What still needs patching for this exact config
#   (TP=2 + MTP + GDN prefix caching on Battlemage) is:
#     [6] getmem   : getMemoryInfo zero-free fallback on XPU        (#53990, open)
#     [7] grammar  : keep grammar-bitmask copies on the right stream (#53997, open)
#     [9b] mtphitfix : MTP/EAGLE + GDN prefix-cache corruption fix
#                     (port of open upstream #57128; drop_eagle_block was
#                     ignored, so a "hit" could reuse unverified draft Mamba
#                     state — the silent-corruption root cause; the symptom
#                     is issue #53912).
#     [9c] fstier   : bounded capacity + LRU eviction for the fs (disk) KV
#                     tier (vendored open upstream #54327, head 8cd8ebf1):
#                     without it the unbounded fs tier writes until the
#                     /vllm_prefix_cache volume quota is hit (09-25 log:
#                     123x [Errno 122] "Disk quota exceeded" + 129 short
#                     writes, all in _r0). Inert unless the serve
#                     --kv-transfer-config fs tier sets "max_bytes".
#                     Applied last.
#   [8] tritonar  : DISABLED 2026-09-24 — TP=2 one-shot Triton symmetric-
#                   memory allreduce (opt-in VLLM_XPU_TRITON_ALLREDUCE,
#                   upstream analog #54768 still open). Never functional on
#                   the B50/B70 pair (symm.rendezvous L0 error 45, always
#                   falling back to oneCCL); torch 2.14.0 (vLLM main's pin)
#                   added fused/async-TP XPU symm-mem ops but not the raw
#                   rendezvous transport this patch uses, so nothing
#                   upstream made it work. oneCCL is the allreduce transport.
#
# RUNTIME (set on the serving process, e.g. in the container entrypoint):
#   nothing required — XPU graphs are default-on since #51600 (2026-09-24);
#   pass --enforce-eager to run the eager baseline / canary. [9c] fstier
#   needs no env either, but it is INERT unless the serve --kv-transfer-
#   config fs tier sets "max_bytes" (see step 9c and the patch README).
#
# NOTE on patch [8] tritonar (DISABLED in the build since 2026-09-24, see step
#   8): when/if re-enabled, it ADDS new files (xpu_triton_all_reduce.py etc).
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
#   (b) Triton symmetric-memory one-shot — patch [8], DISABLED in the build as
#       of 2026-09-24 (was: OPTIONAL, gated on the envs-declared
#       VLLM_XPU_TRITON_ALLREDUCE, off by default). The 09-21 error
#       ("L0 error 45 in symm.rendezvous" + "XPU Triton all-reduce init failed;
#       using oneCCL") is this path's init failing on the B50/B70 pair, and it
#       never got fixed on this hardware: torch 2.14.0 (the vLLM-main pin)
#       added fused/async-TP XPU symm-mem ops (via intel/torch-xpu-ops #3747)
#       and IPC-handle sharing in XPUCachingAllocator, but NOT the raw
#       symm.empty/rendezvous transport this patch uses, and the upstream
#       analog #54768 is still open. oneCCL (a) is now the sole allreduce
#       transport in this build. Do NOT "fix" (b) with (a)'s oneCCL env vars;
#       they are different transports. Re-enable [8] only after a live B50/B70
#       canary of VLLM_XPU_TRITON_ALLREDUCE=1 passes (when on, it only takes
#       over small 1024-aligned bf16 decode all-reduces and falls back to
#       oneCCL for everything else).
#
# CROSS-REFERENCES (upstream tracking — states as of 2026-09-24)
#   #51600  "[XPU] enable XPU GRAPH by default" — MERGED 2026-09-24 (dcfc17e0b).
#            Superseded our [9] memprof patch and deleted the
#            VLLM_XPU_ENABLE_XPU_GRAPH env var (graphs are now default-on;
#            opt out with --enforce-eager). See RE-VERIFIED note below.
#   #56917  "[Feature]: TP=2 graph capture + MTP speculative decoding crash on
#            Arc B70 — fix already exists upstream, unmerged"  (== this stack)
#   #54768  "[XPU] Route small TP all-reduces to a Level Zero IPC kernel" —
#            open upstream analog of patch [8] tritonar, which is DISABLED in
#            this build since 2026-09-24 (see step 8). (Do not confuse
#            with #53989, which is a different PR: fused QK-norm+RoPE+gate.)
#   #53990 / #53997  CySpiegel XPU PRs (covered by patches [6]/[7]) — still open
#   #57128  MTP/EAGLE + GDN prefix-cache corruption fix — covered by [9b]; the
#            full PR rewrites stale base (sink_blocks / manager registry) so only
#            the minimal find_longest_cache_hit hunk is adopted locally.
#   #53912  "[Bug]: MTP + prefix caching corruption (empty/repeated output)" —
#            the symptom [9b] addresses on this exact config.
#   #54327  "[Feature][KV Offload] Add bounded capacity and LRU eviction to the
#            filesystem tier" — still OPEN/unmerged (re-checked 2026-09-26,
#            head 8cd8ebf1, base 10e6a7f2). Vendored locally as [9c] fstier
#            because the unbounded fs tier is failing in production (disk-
#            quota exhaustion of the /vllm_prefix_cache volume, 09-25 log).
#            Once MERGED upstream, drop step 9c — main will carry it natively.
#   imryanpurdy/Qwen3.8-27B-4x-Intel-B70s  — same model, MTP5 + graphs shipped,
#            144 tok/s, bit-identical canary (docs/CAMPAIGN-2026-09-10-GRAPHS.md)
#
# VERIFY BEFORE TRUSTING (canary): corruption is config-specific, so re-run the
#   deterministic canary (5 prompts, temp=0, sha256 of the 64-token completion)
#   against eager before shipping any graphs build. [9b] mtphitfix fixes a
#   cache-logic bug (not a memory/profiling change), so the canary still
#   measures it; the graph path itself must be canaried on YOUR build.
#
# Apply order matters: embed-quant [4] BEFORE mtp-vocab [5]; the XPU patches
# [6]-[7] are independent of the MTP block but applied after it, giving the
# validated 7-patch flow (mtpeagle,vision,embed,mtpvocab,getmem,grammar,
# mtphitfix) — [8] tritonar is DISABLED (2026-09-24) and [9] memprof was
# superseded by upstream #51600 (2026-09-24). [9b] mtphitfix touches
# single_type_kv_cache_manager.py and [9c] fstier only
# vllm/v1/kv_offload/tiering/fs/manager.py (+ its test + docs), which no
# earlier patch modifies, so [9b] then [9c] go last, in that order.
# Every applied patch FAILS THE BUILD loudly if it no longer applies
# (upstream drift); none silently skip.
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
#   API: #54768 (analog of [8]) / #53990 / #53997 / #57128 / #57565 all still
#   OPEN/unmerged (#55390 is now MERGED — see next block); issues #53912 /
#   #56917 still OPEN. No patch superseded by upstream as of 4f145167.
#   RE-VALIDATED 2026-09-23: full 9-patch chain strict `git apply` on newest
#   vllm main @ c961121519. Two upstream drifts found and re-hunked (only two;
#   the other 7 patches were clean against c961121519 with no changes):
#     (1) #55390 (Mamba+EAGLE positional draft-grouping) MERGED 2026-09-22, so
#         step [2] was the STALE COMBINED 55390+56026 patch. It is now SPLIT to
#         the standalone 0001-56026-on-current-main.patch, which carries ONLY
#         the #56026 delta on top of current main: flag every KV group holding
#         a separately-prefixed drafter's layers, and key the all-groups draft
#         fallback warning on use_eagle_block_drop().
#     (2) patch [8] tritonar re-hunked: xpu_communicator.py is now TRACKED in
#         upstream main and (a) already imports `vllm.envs as envs` (the old
#         `from vllm import envs` import hunk is now redundant -> dropped) and
#         (b) all_reduce() now opens with a VLLM_BATCH_INVARIANT /
#         _fixed_rank_sum guard, so the Triton fast-path is inserted AFTER that
#         guard instead of directly before `output = input_.clone()`. The new
#         kernel file xpu_triton_all_reduce.py and the envs.py flag
#         (VLLM_XPU_TRITON_ALLREDUCE: bool=False, off by default) are unchanged.
#         Runtime behavior identical: opt-in, TP=2 only, try/except -> oneCCL.
#   RE-VALIDATED 2026-09-23 (audit of a real failure): a build run that printed
#   "FATAL: mtpeagle (55390+56026) patch no longer applies on 6dc34b6334" came
#   from a STALE pre-split copy of this script (the combined 55390+56026 patch).
#   Re-running the CURRENT script re-fetches the split
#   0001-56026-on-current-main.patch, which applies clean on 6dc34b6334 and on
#   every newer main — no code change needed for that FATAL. The audit also
#   pinned two base-era requirements that are now enforced up front by the
#   gate in step 1b, so an out-of-era base fails with an actionable message:
#     (a) the 56026 patch's two files match main blob-for-blob from the #55390
#         merge (0bce411a, 2026-09-22) through 8b660ce96 (2026-09-23); the
#         patch is the #56026 delta ON TOP of #55390, so main must carry
#         _uses_trailing_mtp_layers() in vllm/v1/core/kv_cache_utils.py.
#     (b) [8] tritonar was re-hunked onto the VLLM_BATCH_INVARIANT guard that
#         #55881 (e4340e41c) introduced at the top of
#         xpu_communicator.all_reduce(); on bases BEFORE that commit the
#         xpu_communicator.py hunk fails (the new-file/envs hunks are fine).
#         The user's 6dc34b6334 base is exactly such a base (it predates
#         e4340e41c): there [8] would have been the NEXT failure after [2].
#   Full 9-patch chain strict `git apply` re-verified clean on current
#   vllm main @ 8b660ce96 (2026-09-23 14:16 UTC). PR states re-checked:
#   #55390 MERGED, #56026 still OPEN; #54768 / #53990 / #53997 / #57128 still
#   OPEN. All 9 raw.githubusercontent.com patch URLs live (HTTP 200).
#   RE-VERIFIED 2026-09-25: the 7-patch chain (mtpeagle, vision, embed-quant,
#   mtp-vocab, getmem, grammar, mtphitfix) strict `git apply` clean on current
#   vllm main @ 0908116dd and e33de821c; era-gate PASS. Changes to THIS script:
#     (1) [9] graphmemprof DROPPED — superseded by #51600 (dcfc17e0b, merged
#         2026-09-24): XPU graphs are now default-on and gpu_worker budgets
#         graph-capture memory on XPU (exactly what [9] injected); its patch
#         no longer applies (the base lines it expected are gone). The
#         VLLM_XPU_ENABLE_XPU_GRAPH env var it documented was deleted from
#         vllm/envs.py, so the old 'Serve with' line is obsolete — graphs run
#         without any env switch; pass --enforce-eager for the eager canary.
#     (2) [8] tritonar DISABLED — no upstream fix made it work: #54768 still
#         open, and torch 2.14.0 (vLLM main's current pin) added fused/async-TP
#         XPU symm-mem ops (via intel/torch-xpu-ops #3747) and IPC-handle
#         sharing in XPUCachingAllocator, but NOT the raw symm.empty/rendezvous
#         transport this patch uses (the B50/B70 pair still hits L0 error 45 in
#         symm.rendezvous; it always fell back to oneCCL). oneCCL remains the
#         allreduce transport; optional oneAPI tuning (CCL_ATL_TRANSPORT=ofi,
#         CCL_ZE_IPC_EXCHANGE=sockets, CCL_TOPO_P2P_ACCESS=0,
#         CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0) only if the FATAL oneCCL
#         IPC crash resurfaces. Re-enable [8] only after a live B50/B70 canary
#         of VLLM_XPU_TRITON_ALLREDUCE=1 passes.
#     (3) era-gate narrowed to the single _uses_trailing_mtp_layers marker
#         (the #55390 merge, 0bce411a, 2026-09-22): [8] tritonar was the only
#         patch anchored to the VLLM_BATCH_INVARIANT guard (#55881), so that
#         marker is no longer a build requirement.
#   Upstream states re-checked 2026-09-25: #55390 and #55881 MERGED (the
#   VLLM_BATCH_INVARIANT guard is still in xpu_communicator.py on main);
#   #56026 / #53990 / #53997 / #54768 / #57128 / #57565 all still OPEN; issues
#   #53912 and #56917 still OPEN. All 7 live patch URLs HTTP 200.
#   RE-VERIFIED 2026-09-26 (post-incident): added [9c] fstier — vendored
#   upstream PR #54327 (head 8cd8ebf1, still OPEN/unmerged/blocked, re-checked
#   2026-09-26): bounded capacity (max_bytes) + LRU eviction for the fs tier.
#   The 8-patch chain (the 7 above + fstier) strict `git apply` clean on
#   newest vllm main @ 31f2e70cd (2026-09-26) — and earlier on 3b4566c5cf;
#   all 7 prior patches' pre-images clean on both (no upstream drift in their
#   files since 0908116dd/e33de821c; the 2 commits since 3b4566c5, #58830
#   multimodal-processor security gate + #58609 CI split, touch no
#   patch-era file). fs-tier test suite on the patched tree (CPU sandbox,
#   re-run on 31f2e70cd): 44 passed / 11 skipped, 0 failed (baseline on
#   unpatched 31f2e70cd: 33 passed / 11 skipped; the 55/44 teardown
#   "errors" are CPU-only-box `torch.accelerator.empty_cache` noise,
#   present identically in both runs; all 11 new bounded-capacity tests
#   pass). Motivating failure: 09-25 production log (container 59) —
#   123x [Errno 122] "Disk quota exceeded" + 129 short-write errors, all in
#   /vllm_prefix_cache/<model>_<digest>_r0/ (fs-tier disk-quota exhaustion;
#   XPU KV usage peaked at 96% but is not the cause), cumulative
#   store_bytes 25.6 GiB vs load_bytes 310 GiB, external prefix-cache hit
#   23.7-92.1% — the fs tier is doing its job, it just has no capacity bound.

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

# 1b. Base-era gate: the retained patches are re-hunked for a specific upstream
#     era, so check the base's era BEFORE trying to apply anything, and fail
#     with an actionable message instead of a cryptic per-patch "patch does
#     not apply". (This is the "patch no longer applies" guard, made specific.)
#     The [2] mtpeagle patch is the #56026 delta ON TOP of merged #55390
#     (0bce411a, 2026-09-22), so its two files (kv_cache_utils.py and its test)
#     only match main from that merge onward — verified blob-for-blob from
#     0bce411a through e33de821c (2026-09-24). The VLLM_BATCH_INVARIANT marker
#     (#55881) was the anchor of the now-DISABLED [8] tritonar patch, so it is
#     no longer a build requirement (it is still in main; kept here for
#     reference only, not checked).
era_ok=1
for marker in \
  "vllm/v1/core/kv_cache_utils.py:_uses_trailing_mtp_layers"; do
  f="${marker%%:*}"; pat="${marker##*:}"
  if ! git grep -q "$pat" -- "$f"; then
    era_ok=0
    echo "NOTE: base ${HASH} predates the #55390 merge (0bce411a, 2026-09-22)."
  fi
done
if [ "$era_ok" != 1 ]; then
  echo "FATAL: base ${HASH} is outside the verified patch era (needs main at/after"
  echo "       0bce411a, the #55390 merge, 2026-09-22). Fix: 'git fetch origin &&"
  echo "       git reset --hard origin/main' and re-run. Known-good main for"
  echo "       this patch set: 0bce411a (2026-09-22) .. e33de821c (2026-09-24,"
  echo "       verified)."
  exit 1
fi

# 2. MTP separately-prefixed-drafter KV-group fix (#56026, still open). NOTE:
#    this used to be the COMBINED 55390+56026 patch, but #55390 (Mamba+EAGLE
#    positional draft-grouping) is now MERGED upstream (2026-09-22), so only the
#    #56026 delta remains. It flags every KV group holding a separately-prefixed
#    drafter's layers as a draft group, and keys the all-groups draft fallback
#    warning on use_eagle_block_drop(). Standalone git-format patch; re-hunked
#    onto current main c961121519 (2026-09-23).
curl -L "https://raw.githubusercontent.com/${V}/main/patches/0001-56026-on-current-main.patch" -o /tmp/mtpeagle.patch
git apply /tmp/mtpeagle.patch || { echo "FATAL: 56026 patch no longer applies on ${HASH}."; \
  echo "       The 56026 patch is the #56026 delta ON TOP of merged #55390 (0bce411a);" \
  echo "       its two files (kv_cache_utils.py, test_kv_cache_utils.py) must match main" \
  echo "       blob-for-blob, verified for 0bce411a .. 8b660ce96. Upstream drift:" \
  echo "       re-hunk the patch onto current main (update the 'index' hashes) and" \
  echo "       re-run. (If you saw 'mtpeagle (55390+56026)' here, your script copy is" \
  echo "       stale — pull this repo and re-run; #55390 is merged, patch is split.)"; exit 1; }
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

# 8. [DISABLED 2026-09-24] XPU Triton allreduce for TP=2 (upstream analog
#    #54768, still OPEN). Not applied in this build: it never worked on the
#    B50/B70 pair (symm.rendezvous L0 error 45 -> oneCCL fallback on every
#    boot), and no upstream change fixed that transport — torch 2.14.0 (the
#    vLLM-main pin) gained fused/async-TP XPU symm-mem ops (via
#    intel/torch-xpu-ops #3747) and XPUCachingAllocator IPC-handle sharing,
#    but not the raw symm.empty/rendezvous path this patch exercises, and
#    #54768 is unmerged. Allreduce runs over oneCCL (see the (a) transport
#    note in the header for the optional oneAPI tuning envs). Re-enable by
#    un-commenting — after a live B50/B70 canary of
#    VLLM_XPU_TRITON_ALLREDUCE=1 proves the init succeeds:
# curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-triton-allreduce-tp2.patch" -o /tmp/tritonar.patch
# git apply /tmp/tritonar.patch || { echo "FATAL: xpu-triton-allreduce-tp2 patch no longer applies on ${HASH}."; \
#   echo "       If 'xpu_communicator.py: patch does not apply' is the cause, the base" \
#   echo "       predates the VLLM_BATCH_INVARIANT guard (#55881, e4340e41c) that this" \
#   echo "       re-hunk anchors on. Fix: build on main at/after e4340e41c (current main" \
#   echo "       qualifies); the new-file + envs.py hunks are era-independent."; exit 1; }
# NAME=${NAME}-tritonar
# (The patch file itself stays in the vault as a reference artifact.)

# 9. [DROPPED 2026-09-24] XPU CUDA-graph memory profiling — superseded by
#    upstream #51600 (merged 2026-09-24, commit dcfc17e0b), which made XPU
#    graphs default-on AND made gpu_worker determine_available_memory() call
#    profile_cudagraph_memory() on XPU (the exact line [9] used to add) and
#    dropped the "XPU stays excluded (see #39977)" clause. The patch no
#    longer applies on main at/after dcfc17e0b (verified: it fails on
#    gpu_worker.py), and the VLLM_XPU_ENABLE_XPU_GRAPH env var it used to
#    document is gone from vllm/envs.py — graphs are now on by default, opt
#    out with --enforce-eager.

# 9b. MTP/EAGLE prefix-cache corruption fix (port of upstream #57128, still open/dirty).
#     In MambaManager.find_longest_cache_hit the drop_eagle_block flag was ACCEPTED
#     but IGNORED, so under MTP/EAGLE speculative decoding a "hit" could reuse a Mamba
#     state that still holds UNVERIFIED draft state from a rejected draft position.
#     That is the silent-corruption root cause (symptom: empty or repeated-character
#     output, issue #53912) — it bites exactly this config: GDN (mamba) prefix caching
#     ON + MTP ON. Fix: when drop_eagle_block is set, skip only the FIRST (most
#     recent) checkpoint the finder matches, then keep scanning for the next
#     (older, committed) one — instead of blanking the whole search tail. Self-contained
#     42-line change, applied last in the chain (it touches
#     single_type_kv_cache_manager.py, which no earlier patch modifies); verified to
#     `git apply` on current main (0908116dd / e33de821c, 2026-09-24).
curl -L "https://raw.githubusercontent.com/${V}/main/patches/xpu-mtp-prefix-hit-fix.patch" -o /tmp/mtp-hitfix.patch
git apply /tmp/mtp-hitfix.patch || { echo "FATAL: xpu-mtp-prefix-hit-fix patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-mtphitfix

# 9c. Bounded capacity + LRU eviction for the fs (disk) KV tier (vendored
#     upstream PR #54327, head 8cd8ebf1, still OPEN/unmerged). Adds a
#     `max_bytes` param to the fs-tier manager: when set, the tier evicts
#     least-recently-used blocks to stay under the bound before a store,
#     protects blocks in an active load/store, and skips a cache write when a
#     batch cannot fit (without failing the request). Applied LAST in the chain
#     (it only touches vllm/v1/kv_offload/tiering/fs/manager.py + its test + the
#     usage doc, which no earlier patch modifies). The bound is PER RANK
#     DIRECTORY (<model>_<digest>_r<rank>) and requires exclusive ownership of
#     it; it is INERT unless the serve --kv-transfer-config fs tier sets
#     "max_bytes" — see the final echo and the patch README.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/fs-tier-max-bytes-54327.patch" -o /tmp/fstier-maxbytes.patch
git apply /tmp/fstier-maxbytes.patch || { echo "FATAL: fs-tier-max-bytes-54327 patch no longer applies on ${HASH}."; \
  echo "       The patch vendors open upstream #54327 (head 8cd8ebf1) against the" \
  echo "       tiering fs manager (manager.py + test + docs). Upstream drift:" \
  echo "       re-hunk the patch onto current main (update the 'index' hashes) and" \
  echo "       re-run."; exit 1; }
NAME=${NAME}-fstier

# 11. Build the XPU image (graphs-capable).
docker build --cpuset-cpus="0" --memory="16g" --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${NAME} .

echo vllm-intel-xpu:${NAME}
echo
echo "Serve with:  vllm serve ... (XPU graphs default-on since #51600/2026-09-24;"
echo "             no env switch needed; --enforce-eager for the eager canary)"
echo "[9c] fstier (PR #54327) ships bounded-capacity + LRU eviction for the fs"
echo "     KV tier. It is INERT unless you enable it on the serve fs tier, e.g.:"
echo "       --kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\","
echo "         \"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\","
echo "         \"cpu_bytes_to_use\":12884901888,\"secondary_tiers\":[{\"type\":\"fs\","
echo "         \"root_dir\":\"/vllm_prefix_cache\",\"n_read_threads\":8,\"n_write_threads\":8,"
echo "         \"max_bytes\":<per-rank-dir byte cap>]}}'"
echo "     The bound is per <model>_<digest>_r<rank> dir and requires exclusive"
echo "     ownership of it (no multi-engine sharing in bounded mode). Size it to"
echo "     the volume's per-dir quota so LRU eviction happens BEFORE [Errno 122] quota."
echo "Remember the canary: 5 deterministic prompts, temp=0, sha256 vs eager before trusting graphs."
