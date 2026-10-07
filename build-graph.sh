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
#                     Applied before [9d].
#     [9d] hybridprefill : hybrid-model prefill misclassified as uniform
#                     decode (port of open upstream #47123, head 41edfaf).
#                     Qwen3.5/3.8 are HYBRID (linear_attention mamba layers
#                     interleaved with full_attention): in _is_uniform_decode a
#                     request whose prompt length is exactly 1+num_spec_tokens
#                     (e.g. MTP n=3 -> a 4-token prompt) is misread as a
#                     uniform-decode / spec step, the mamba (GDN) state is not
#                     zeroed, and the forward returns garbage. This adds
#                     _compute_force_uniform_decode: hybrid + any prefill in
#                     the batch -> force False (correct), non-hybrid or
#                     pure-decode -> None (old heuristic). Same corruption
#                     symptom class as [9b]/#53912 but a DIFFERENT root cause
#                     (prefill misclassification vs prefix-cache-hit reuse),
#                     so it complements rather than replaces [9b]. No-op for
#                     non-hybrid models. Applied before [9e].
#     [9e] turboquantspec : TurboQuant spec-decode CUDA-graph fix (port of
#                     open upstream #53406, head 673af7f5; fixes issue #52475:
#                     MTP + turboquant_* KV = repetition collapse / IMA). The
#                     TQ backend declared _cudagraph_support = UNIFORM_BATCH,
#                     so MTP verify batches (uniform query_len = 1 + n_spec =
#                     4 for n=3) got FULL-captured with dummy metadata
#                     (seq_lens = 1): the TQ verify path is NOT graph-
#                     capturable (CPU-resident metadata, per-request Python
#                     prefill loop, supports_spec_as_decode=False), so
#                     garbage attention was baked into the graph. This
#                     downgrades the level to UNIFORM_SINGLE_TOKEN_DECODE so
#                     verify batches run uncaptured, correctly. Self-contained
#                     (one file, no overlap with [1]-[9d]). Applied before
#                     [10]; drop when #53406 merges.
#     [10] onercclreset : reset oneCCL's collective chain after every
#                     graph-capture warmup (port of the CLOSED-UNMERGED
#                     upstream #58415, with the hook moved out of the dead
#                     after-yield location into the model-runner capture
#                     loop). Without it, a captured graph leaves the oneCCL
#                     chain bound to the graph's completion event and the
#                     NEXT capture's torch.xpu.synchronize() aborts with
#                     UR_RESULT_ERROR_DEVICE_LOST — vllm #58388 and, since
#                     #56531 (10-06), the startup death #60379 (TP=2 + MTP +
#                     GDN, the exact 10-06 build). No-op on TP=1.
#   Tag-suffix aliases (Docker tags cap at 128 chars, see the step-11 guard):
#   [3] -> -vision, [9d] -> -hybrid, [9e] -> -tq. The full names stay in each
#   step's comment/echo; the tag only needs to be readable/unique.
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
#   [9e] turboquantspec (PR #53406) fixes TurboQuant + MTP (issue #52475); it
#   takes effect when the serve command passes a turboquant_* --kv-cache-dtype
#   (see the final echo).
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
#            the symptom [9b] (prefix-cache-hit reuse) AND [9d] (prefill
#            misclassified as uniform decode) both address on this config;
#            the two root causes are independent, so both patches stay.
#   #47123  "[Bugfix] Fix misclassification of prefill as uniform decode for
#            hybrid models" — still OPEN/unmerged (head 41edfaf). Vendored
#            locally as [9d] hybridprefill: Qwen3.5/3.8 are hybrid, so a
#            prompt of exactly 1+num_spec_tokens was read as a uniform-decode
#            spec step (mamba state never zeroed -> garbage). No-op for
#            non-hybrid models; complements [9b]. Once MERGED upstream, drop
#            step 9d — main will carry it natively.
#   #54327  "[Feature][KV Offload] Add bounded capacity and LRU eviction to the
#            filesystem tier" — still OPEN/unmerged (re-checked 2026-09-26
#            three times, no PR activity since 2026-09-20, base 10e6a7f2 is
#            stale vs current main). Vendored locally as [9c] fstier because
#            the unbounded fs tier is failing in production (disk-quota
#            exhaustion of the /vllm_prefix_cache volume, 09-25 log: 123x
#            [Errno 122] + 129 short writes, all in the _r0 dir). Once
#            MERGED upstream, drop step 9c — main will carry it natively.
#   #58388  "[Bug][XPU][MRV2] engine dies when the first request after startup
#            is a large prefill" — oneCCL leaves the stream Recording after a
#            graph capture; the fix PR #58415 (reset oneCCL's collective chain
#            after capture) was CLOSED UNMERGED. Patch [10] onercclreset is a
#            local port of #58415 with its placement fixed (the upstream hook
#            sat after the yield in GroupCoordinator.graph_capture, a
#            generator — dead code). [10] instead resets in the model-runner
#            capture loop after the eager warmup, before the capture
#            __enter__. Needed by both #58388 (post-capture prefill) and
#            #60379 (the 10-06 startup death on the FIRST PIECEWISE capture
#            with TP=2 + MTP + GDN, introduced by #56531). No-op on TP=1;
#            degrades to a no-op call if upstream lands #58415 — then drop
#            step 10, like [9] was dropped for #51600.
#   #56531  "[Bugfix] Route every speculation-capable row through the
#            speculative path" — merged 2026-10-06 (0eac152707). The 48h
#            regression that made the first capture record oneCCL collectives
#            (use_spec_decode became `speculative_config is not None`); with
#            the #58415-class oneCCL bug unfixed, it kills startup on XPU/TP2.
#            NOT reverted here: it is a legitimate correctness fix, and [10]
#            makes its capture-time collectives safe.
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
# single_type_kv_cache_manager.py, [9c] fstier only
# vllm/v1/kv_offload/tiering/fs/manager.py (+ its test + docs), and [9d]
# hybridprefill only vllm/v1/worker/gpu_model_runner.py (+ its test), [9e]
# turboquantspec only vllm/v1/attention/backends/turboquant_attn.py, and [10]
# onercclreset only vllm/distributed/device_communicators/xpu_communicator.py
# + vllm/v1/worker/gpu/cudagraph_utils.py — none of the earlier ENABLED
# patches modify any of those files ([8] tritonar also touches
# xpu_communicator.py but is DISABLED), so [9b] then [9c] then [9d] then [9e]
# then [10] go last, in that order.
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
#   pass). Motivating failure: 09-25 production log (container 59, window
#   09:07..11:13) — 123x [Errno 122] "Disk quota exceeded" + 129 short-write
#   errors = 252 "block I/O failed", all in
#   /vllm_prefix_cache/<model>_<digest>_r0/ (fs-tier disk-quota exhaustion;
#   XPU KV usage peaked at 96% but is not the cause); over that window the
#   per-interval kv_offload_store_bytes/load_bytes sum to ~27.5 / ~333 GiB
#   and the external prefix-cache hit rate ran 23.7-92.1% — the fs tier is
#   doing its job, it just has no capacity bound.
#   RE-VERIFIED 2026-09-26 (3rd): full 8-patch chain strict `git apply` clean
#   on newest vllm main @ 7d8c5fe9a (2026-09-26 15:37 UTC). The 7 commits since
#   ad6817b68 (#58786 Anthropic thinking w/ P/D, #58754 Anthropic inline-system
#   merge detect, #58046 mypy typing for Qwen/Qianfan, #58749 CI cudagraph-mode
#   overhead, #58594 GLM5.3 sparse-indexer attn, #58810 CI batch submission,
#   #58499 DSV4.1 ViT cudagraph replay) touch NO patch-era file: the only ones
#   in a patch-era file are #58046's qwen3_5.py/qwen3_5_mtp.py hunks, which are
#   annotation-only (ClassVar, get_multimodal_config(), MultiModalFeatureSpec)
#   far from the [4]/[5] anchor regions — all 8 patches apply byte-clean, no
#   re-hunk needed. fs-tier suite on the patched 7d8c5fe9a tree (CPU sandbox):
#   44 passed / 11 skipped, 0 failed — identical to the ad6817b68 run (the 55
#   "errors" are the same CPU-only teardown noise, present in the unpatched
#   baseline too). PR #54327 re-checked a third time: still OPEN, no activity
#   since 2026-09-20. The incident evidence file
#   (_vllm-59-vllm-server-1_logs - przepelnienie cache.txt, container 59,
#   window 09-25 09:07..11:13) re-confirmed 123x [Errno 122] + 129
#   "Short write: expected 55705600 bytes, wrote 33685504" I/O failures, all in
#   the <model>_<digest>_r0 dir — the exact failure [9c] fstier bounds.
#   RE-VERIFIED 2026-09-30: full 8-patch chain strict `git apply` clean on
#   newest vllm main @ c4df37d (2026-09-30; v0.30.1rc0 tagged at 2df122e6).
#   The two commits since fc2c801a (cff08b4 GHSA-4hhp-h66f chat-template DoS
#   fix, c4df37d ROCm mori-build) touch no patch-era file (the envs.py diff
#   is +6 unrelated lines), so the re-hunk is valid 5463fe49 .. c4df37d.
#   ONE re-hunk needed: [2] mtpeagle — upstream #57652 (5463fe49, merged
#   2026-09-30, "Expand replicated_layout detection to multi-group MLA")
#   appended test_kv_cache_groups_tp_replicas to the END of
#   tests/v1/core/test_kv_cache_utils.py, breaking the patch's EOF-anchored
#   test hunk (@@ -4383,3). kv_cache_utils.py was untouched by #57652 and
#   still applied at the same offset; only the test hunk's anchor/context
#   re-aimed (@@ -4734,3), all 54 added lines byte-identical, verified
#   blob-for-blob against 2df122e6/fc2c801a (the two test+kvutils blobs are
#   identical on both tips). Because the re-anchored test pre-image only
#   exists from 5463fe49 onward, the step-1b era-gate now checks a SECOND
#   marker, kv_cache_groups_tp_replicas in kv_cache_utils.py (introduced by
#   #57652; proven absent at 5463fe49^): era is now 5463fe49 .. c4df37d.
#   (The original 09-30 dry-run caught exactly this FATAL before any other
#   patch was tried.) PR states re-checked 2026-09-30: #56026 / #53990 /
#   #53997 / #57128 / #54327 / #54768 all still OPEN (no activity); #55390 /
#   #51600 still MERGED; issues #53912 / #56917 still OPEN. envs.py/xpu.py
#   moved again since 8b660ce96 (ROCm/GLM/KV-offloading only) —
#   VLLM_XPU_ENABLE_XPU_GRAPH still absent from envs.py, so [8] stays
#   DISABLED (re-enable only after a live B50/B70 canary of
#   VLLM_XPU_TRITON_ALLREDUCE=1) and [9] stays DROPPED. Watch: #57128 merge
#   -> drop [7b]; #54327 merge -> drop [9c]; #56026 merge -> drop [2] and
#   relax the era-gate (kv_cache_groups_tp_replicas stays a hard marker only
#   while the re-anchored test pre-image depends on the #57652 EOF).
#   RE-VERIFIED 2026-10-05 (added [9d] hybridprefill): the existing 8-patch
#   chain strict `git apply` clean on current vllm main @ 710ac56e (2026-10-05);
#   era-gate PASS (_uses_trailing_mtp_layers + kv_cache_groups_tp_replicas both
#   present). Added step [9d] = port of open upstream PR #47123 (head 41edfaf),
#   vendored verbatim as patches/hybrid-prefill-uniform-decode-47123.patch (2
#   files: vllm/v1/worker/gpu_model_runner.py + its test). It fixes a prefill
#   being misclassified as a uniform-decode / spec step in HYBRID models
#   (Qwen3.5/3.8 have linear_attention mamba layers interleaved with
#   full_attention): a prompt of exactly 1+num_spec_tokens (MTP n=3 -> 4 tokens)
#   triggered _is_uniform_decode, the mamba/GDN state was never zeroed, and the
#   forward returned garbage. The port adds _compute_force_uniform_decode:
#   hybrid + any prefill in the batch -> force False; non-hybrid or pure-decode
#   -> None (old heuristic preserved) — a NO-OP for non-hybrid models. It is an
#   INDEPENDENT root cause from [9b] mtphitfix (#57128, prefix-cache-hit reuse);
#   both fix the #53912/#56917 "empty/repeated output" symptom class, so both
#   stay. It only touches gpu_model_runner.py + its test (no earlier patch
#   does), so it applies on top of [1]-[9c] with no collision; verified `git
#   apply --check` clean on 710ac56e and it re-applies after all 8 preceding
#   patches in the chain. PR #47123 re-checked OPEN/unmerged (head 41edfaf,
#   created 2026-06-30) — so [9d] is required now; DROP step 9d once #47123
#   merges upstream.
#   RE-VERIFIED 2026-10-07 (added [10] onercclreset): local port of the
#   CLOSED-UNMERGED upstream #58415 (reset oneCCL's collective chain after
#   capture), with the hook moved out of the dead after-yield location
#   (GroupCoordinator.graph_capture is a generator) into the model-runner
#   capture loop, after the eager warmup and before the torch.cuda.graph
#   __enter__. Added after the 10-06 startup-crash regression (vllm #60379:
#   first PIECEWISE capture dies with UR_RESULT_ERROR_DEVICE_LOST under
#   TP=2 + MTP + GDN; 48h bisect -> #56531, which makes zero-draft rows take
#   the speculative path and thus record oneCCL collectives inside the
#   capture window). [10] strict `git apply` clean on top of the full
#   [2]-[9e] chain on both main @ df417f780a (2026-10-07) and 4ea0c28bc (the
#   10-06 build base); both touched files (xpu_communicator.py,
#   cudagraph_utils.py) had zero upstream commits since the patch was
#   re-hunked, and both py_compile clean. The call site is a getattr()
#   probe, so the patch degrades to a no-op call if upstream ever lands
#   #58415 or an equivalent; when that happens drop step 10 (same lifecycle
#   as [9] for #51600).

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
#     only match main from that merge onward. Since the 2026-09-30 re-hunk the
#     test hunk is anchored on the EOF that #57652 (5463fe49, merged
#     2026-09-30) appended — test_kv_cache_groups_tp_replicas, whose helper
#     kv_cache_groups_tp_replicas() it calls landed in kv_cache_utils.py in
#     that same commit — so the pre-image now REQUIRES main at/after 5463fe49.
#     Two era markers are checked: _uses_trailing_mtp_layers (the #55390
#     merge, 0bce411a) and kv_cache_groups_tp_replicas (the #57652 merge,
#     5463fe49) — the latter is the binding one. The VLLM_BATCH_INVARIANT
#     marker (#55881) was the anchor of the now-DISABLED [8] tritonar patch,
#     so it is no longer a build requirement (it is still in main; kept here
#     for reference only, not checked).
era_ok=1
for marker in \
  "vllm/v1/core/kv_cache_utils.py:_uses_trailing_mtp_layers" \
  "vllm/v1/core/kv_cache_utils.py:kv_cache_groups_tp_replicas"; do
  f="${marker%%:*}"; pat="${marker##*:}"
  if ! git grep -q "$pat" -- "$f"; then
    era_ok=0
    if [ "$pat" = "kv_cache_groups_tp_replicas" ]; then
      echo "NOTE: base ${HASH} predates the #57652 merge (5463fe49, 2026-09-30)."
    else
      echo "NOTE: base ${HASH} predates the #55390 merge (0bce411a, 2026-09-22)."
    fi
  fi
done
if [ "$era_ok" != 1 ]; then
  echo "FATAL: base ${HASH} is outside the verified patch era (needs main at/after"
  echo "       5463fe49, the #57652 merge, 2026-09-30 — the [2] mtpeagle test"
  echo "       hunk is anchored on the EOF #57652 appended). Fix: 'git fetch origin &&"
  echo "       git reset --hard origin/main' and re-run. Known-good main for"
  echo "       this patch set: 5463fe49 (2026-09-30) .. c4df37d (2026-09-30,"
  echo "       verified)."
  exit 1
fi

# 2. MTP separately-prefixed-drafter KV-group fix (#56026, still open). NOTE:
#    this used to be the COMBINED 55390+56026 patch, but #55390 (Mamba+EAGLE
#    positional draft-grouping) is now MERGED upstream (2026-09-22), so only the
#    #56026 delta remains. It flags every KV group holding a separately-prefixed
#    drafter's layers as a draft group, and keys the all-groups draft fallback
#    warning on use_eagle_block_drop(). Standalone git-format patch; re-hunked
#    onto current main c4df37d (2026-09-30). Re-hunk 09-30: #57652 (5463fe49)
#    appended test_kv_cache_groups_tp_replicas to the END of
#    test_kv_cache_utils.py, which shifted this patch's EOF-anchored test hunk
#    (kv_cache_utils.py itself was untouched by #57652 and still applies at the
#    same offset) — only the test hunk's anchor/context changed, all 54 added
#    lines are byte-identical to the 2026-09-23 re-hunk.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/0001-56026-on-current-main.patch" -o /tmp/mtpeagle.patch
git apply /tmp/mtpeagle.patch || { echo "FATAL: 56026 patch no longer applies on ${HASH}."; \
  echo "       The 56026 patch is the #56026 delta ON TOP of merged #55390 (0bce411a);" \
  echo "       its two files (kv_cache_utils.py, test_kv_cache_utils.py) must match main" \
  echo "       blob-for-blob, verified for 5463fe49 .. c4df37d (2026-09-30). Upstream" \
  echo "       drift: re-hunk the patch onto current main (update the 'index' hashes and" \
  echo "       the hunk line numbers) and re-run; if the new drift lands in the test" \
  echo "       file's EOF again, only the test hunk's anchor needs re-aiming. (If you saw" \
  echo "       'mtpeagle (55390+56026)' here, your script copy is stale — pull this repo" \
  echo "       and re-run; #55390 is merged, patch is split.)"; exit 1; }
NAME=${NAME}-mtpeagle

# 3. Vision-tower CPU offload (VLLM_VISION_CPU_OFFLOAD_GB). Not upstream.
curl -L "https://raw.githubusercontent.com/${V}/main/patches/vision-tower-cpu-offload.patch" -o /tmp/visionoffload.patch
git apply /tmp/visionoffload.patch || { echo "FATAL: vision-offload patch no longer applies on ${HASH}"; exit 1; }
NAME=${NAME}-vision

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
# The patch lives in THIS vault repo; the raw URL only serves it once the
# vault's main is pushed. Prefer a local vault checkout (VAULT_LOCAL=/path/to/
# vllm-2xb50) and fall back to the raw URL; a 404 (not pushed yet) fails with
# a different, actionable message than upstream drift (fetched, no apply).
FSTIER_PATCH=""
if [ -n "${VAULT_LOCAL:-}" ] && [ -f "${VAULT_LOCAL}/patches/fs-tier-max-bytes-54327.patch" ]; then
  FSTIER_PATCH="${VAULT_LOCAL}/patches/fs-tier-max-bytes-54327.patch"
else
  curl -fL "https://raw.githubusercontent.com/${V}/main/patches/fs-tier-max-bytes-54327.patch" -o /tmp/fstier-maxbytes.patch 2>/dev/null
  [ -s /tmp/fstier-maxbytes.patch ] && FSTIER_PATCH=/tmp/fstier-maxbytes.patch
fi
if [ -z "$FSTIER_PATCH" ]; then
  echo "FATAL: cannot fetch patches/fs-tier-max-bytes-54327.patch (raw URL 404 —"
  echo "       the vault's main carrying [9c] is not pushed yet). Fix: 'git push"
  echo "       origin main' in marcinkuk/vllm-2xb50, or set"
  echo "       VAULT_LOCAL=/path/to/local/vllm-2xb50-checkout and re-run."; exit 1;
fi
git apply "$FSTIER_PATCH" || { echo "FATAL: fs-tier-max-bytes-54327 patch no longer applies on ${HASH}."; \
  echo "       The patch vendors open upstream #54327 (head 8cd8ebf1) against the" \
  echo "       tiering fs manager (manager.py + test + docs). Upstream drift:" \
  echo "       re-hunk the patch onto current main (update the 'index' hashes) and" \
  echo "       re-run."; exit 1; }
NAME=${NAME}-fstier

# 9d. Hybrid-model prefill misclassified as uniform decode (port of upstream
#     PR #47123, head 41edfaf, still OPEN/unmerged). Vendored verbatim from the
#     PR (2 files, git-format): vllm/v1/worker/gpu_model_runner.py + its test.
#     Qwen3.5/3.8 are HYBRID (mamba/linear_attention layers interleaved with
#     full_attention), so the _is_uniform_decode heuristic could misread a
#     request whose prompt length is exactly 1+num_spec_tokens (MTP n=3 -> a
#     4-token prompt) as a uniform-decode / spec step: the mamba (GDN) state
#     was never zeroed and the forward produced garbage. The PR adds
#     _compute_force_uniform_decode, consulted in _determine_batch_execution_
#     and_padding: hybrid + any prefill in the batch -> force False (correct);
#     non-hybrid or a pure-decode batch -> None (the old heuristic is
#     preserved). So it is a NO-OP for non-hybrid models and never makes a
#     decode-only batch slower. It is an INDEPENDENT root cause from [9b]
#     mtphitfix (#57128: prefix-cache "hit" reusing unverified draft Mamba
#     state) — both hit the #53912 "empty/repeated output" symptom class on
#     this exact config, so both stay. Self-contained: it only touches
#     gpu_model_runner.py + its test, which no earlier patch in the chain
#     modifies, so it applies cleanly on top of [1]-[9c]. Drop it once
#     #47123 is MERGED upstream (main will then carry the fix natively).
#     Fetched the same way as [9c] (local VAULT_LOCAL first, raw URL fallback,
#     clear FATAL if the vault's main carrying [9d] is not pushed yet).
HYBRID_PATCH=""
if [ -n "${VAULT_LOCAL:-}" ] && [ -f "${VAULT_LOCAL}/patches/hybrid-prefill-uniform-decode-47123.patch" ]; then
  HYBRID_PATCH="${VAULT_LOCAL}/patches/hybrid-prefill-uniform-decode-47123.patch"
else
  curl -fL "https://raw.githubusercontent.com/${V}/main/patches/hybrid-prefill-uniform-decode-47123.patch" -o /tmp/hybrid-prefill.patch 2>/dev/null
  [ -s /tmp/hybrid-prefill.patch ] && HYBRID_PATCH=/tmp/hybrid-prefill.patch
fi
if [ -z "$HYBRID_PATCH" ]; then
  echo "FATAL: cannot fetch patches/hybrid-prefill-uniform-decode-47123.patch (raw URL 404 —"
  echo "       the vault's main carrying [9d] is not pushed yet). Fix: 'git push"
  echo "       origin main' in marcinkuk/vllm-2xb50, or set"
  echo "       VAULT_LOCAL=/path/to/local/vllm-2xb50-checkout and re-run."; exit 1;
fi
git apply "$HYBRID_PATCH" || { echo "FATAL: hybrid-prefill-uniform-decode-47123 patch no longer applies on ${HASH}."; \
  echo "       The patch ports open upstream #47123 (head 41edfaf): it adds" \
  echo "       _compute_force_uniform_decode to vllm/v1/worker/gpu_model_runner.py" \
  echo "       (the prefill-as-uniform-decode misclassification in hybrid models)" \
  echo "       plus test_compute_force_uniform_decode in" \
  echo "       tests/v1/worker/test_gpu_model_runner.py. Upstream drift:" \
  echo "       re-hunk the patch onto current main (update the 'index' hashes and" \
  echo "       hunk line numbers) and re-run."; exit 1; }
NAME=${NAME}-hybrid

# 9e. TurboQuant spec-decode CUDA-graph fix (port of upstream PR #53406, head
#     673af7f5, still OPEN/unmerged; fixes issue #52475). Vendored verbatim
#     from the PR (1 file, git-format): vllm/v1/attention/backends/
#     turboquant_attn.py. The TQ backend declared _cudagraph_support =
#     UNIFORM_BATCH, so MTP verify batches (uniform query_len = 1 +
#     num_speculative_tokens; = 4 for our n=3) were FULL-captured with dummy
#     metadata (seq_lens filled with 1). The TQ verify path is not graph-
#     capturable (CPU-resident metadata, per-request Python prefill loop,
#     supports_spec_as_decode=False), so empty/garbage attention got baked
#     into the graph: silent repetition collapse for num_speculative_tokens
#     > 1, illegal memory access for == 1 (reported on Qwen3.8-27B GDN
#     hybrid + MTP — this exact model family). This downgrades the level to
#     UNIFORM_SINGLE_TOKEN_DECODE: verify batches run uncaptured (correct),
#     pure 1-token decodes still capture (no decode-throughput change).
#     Applies when the serve command passes a turboquant_* --kv-cache-dtype
#     (default bf16 KV never instantiates the TQ backend) — see the final
#     echo. Self-contained: it only touches turboquant_attn.py, which no
#     earlier patch in the chain modifies, so it applies cleanly on top of
#     [1]-[9d]. Fetched the same way as [9c]/[9d] (local VAULT_LOCAL first,
#     raw URL fallback, clear FATAL if the vault's main carrying [9e] is not
#     pushed yet). Drop it once #53406 is MERGED upstream (main will then
#     carry the fix natively); re-check the head weekly.
TQSPECPATCH=""
if [ -n "${VAULT_LOCAL:-}" ] && [ -f "${VAULT_LOCAL}/patches/turboquant-specdecode-cg-53406.patch" ]; then
  TQSPECPATCH="${VAULT_LOCAL}/patches/turboquant-specdecode-cg-53406.patch"
else
  curl -fL "https://raw.githubusercontent.com/${V}/main/patches/turboquant-specdecode-cg-53406.patch" -o /tmp/turboquant-specdecode-cg.patch 2>/dev/null
  [ -s /tmp/turboquant-specdecode-cg.patch ] && TQSPECPATCH=/tmp/turboquant-specdecode-cg.patch
fi
if [ -z "$TQSPECPATCH" ]; then
  echo "FATAL: cannot fetch patches/turboquant-specdecode-cg-53406.patch (raw URL 404 —" \
  echo "       the vault's main carrying [9e] is not pushed yet). Fix: 'git push" \
  echo "       origin main' in marcinkuk/vllm-2xb50, or set" \
  echo "       VAULT_LOCAL=/path/to/local/vllm-2xb50-checkout and re-run."; exit 1;
fi
git apply "$TQSPECPATCH" || { echo "FATAL: turboquant-specdecode-cg-53406 patch no longer applies on ${HASH}."; \
  echo "       The patch ports open upstream #53406 (head 673af7f5): _cudagraph_" \
  echo "       "support" UNIFORM_BATCH -> UNIFORM_SINGLE_TOKEN_DECODE in" \
  echo "       vllm/v1/attention/backends/turboquant_attn.py (spec-decode verify" \
  echo "       batches were FULL-captured with dummy metadata; see issue #52475)." \
  echo "       Upstream drift: re-hunk the patch onto current main (update the" \
  echo "       'index' hashes and hunk line numbers) and re-run."; exit 1; }
NAME=${NAME}-tq

# 10. oneCCL collective-chain reset after each graph-capture warmup (port of
#     upstream #58415 — CLOSED UNMERGED — with the dead-code-after-yield
#     placement fixed: the reset is called from the model-runner capture loop
#     right AFTER the eager warmup and BEFORE the torch.cuda.graph __enter__,
#     not after the GroupCoordinator.graph_capture yield). After an XPU graph
#     capture, oneCCL's collective chain is bound to the captured graph's
#     completion event; the next capture's torch.xpu.synchronize() then aborts
#     with UR_RESULT_ERROR_DEVICE_LOST. One tiny eager all-reduce replaces that
#     event with an ordinary one. Required on TP>=2 (this setup: 2x B50).
#     This is what the 2026-10-06 startup crash (issue #60379, regression from
#     #56531 + MTP/GDN) hits. No-op when world_size<=1 or already capturing;
#     degrades to a no-op call if upstream lands #58415 (the call site probes
#     the method with getattr), so it is safe to leave in across re-hunks.
#     Re-hunked onto main @ df417f780a (2026-10-07); the two touched files
#     (xpu_communicator.py, cudagraph_utils.py) had zero upstream drift in
#     that era, so the patch stays green until one of them changes. Fetched
#     the same way as [9c]/[9d]/[9e] (local VAULT_LOCAL first, raw URL
#     fallback, clear FATAL if the vault's main carrying [10] is not pushed
#     yet).
ONERCCLRESETPATCH=""
if [ -n "${VAULT_LOCAL:-}" ] && [ -f "${VAULT_LOCAL}/patches/xpu-onerccl-capture-reset.patch" ]; then
  ONERCCLRESETPATCH="${VAULT_LOCAL}/patches/xpu-onerccl-capture-reset.patch"
else
  curl -fL "https://raw.githubusercontent.com/${V}/main/patches/xpu-onerccl-capture-reset.patch" -o /tmp/onerccl-reset.patch 2>/dev/null
  [ -s /tmp/onerccl-reset.patch ] && ONERCCLRESETPATCH=/tmp/onerccl-reset.patch
fi
if [ -z "$ONERCCLRESETPATCH" ]; then
  echo "FATAL: cannot fetch patches/xpu-onerccl-capture-reset.patch (raw URL 404 —" \
  echo "       the vault's main carrying [10] is not pushed yet). Fix: 'git push" \
  echo "       origin main' in marcinkuk/vllm-2xb50, or set" \
  echo "       VAULT_LOCAL=/path/to/local/vllm-2xb50-checkout and re-run."; exit 1;
fi
git apply "$ONERCCLRESETPATCH" || { echo "FATAL: xpu-onerccl-capture-reset patch no longer applies on ${HASH}."; \
  echo "       The patch adds XpuCommunicator.reset_after_graph_capture() and a call site" \
  echo "       in vllm/v1/worker/gpu/cudagraph_utils.py (after the eager warmup, before" \
  echo "       the PIECEWISE/FULL capture). It anchors on the VLLM_BATCH_INVARIANT branch" \
  echo "       of XpuCommunicator.all_reduce and on the 'CG Capture: mode=' warmup/capture" \
  echo "       block of CudaGraphManager.capture. Upstream drift: re-hunk (update the" \
  echo "       'index' hashes + context) and re-run."; exit 1; }
NAME=${NAME}-onercclreset

# 11. Build the XPU image (graphs-capable).
#     Tag-length guard: Docker image tags cap at 128 chars, and the NAME
#     accumulates one short suffix per applied patch, so adding a patch can
#     silently push the tag over the cap and fail only at this docker build
#     step (10-07: 136 chars -> "invalid reference format"). Keep new patch
#     suffixes SHORT (see the -hybrid / -tq aliases above) and check here.
if [ ${#NAME} -gt 120 ]; then
  echo "FATAL: image tag vllm-intel-xpu:${NAME} is ${#NAME} chars (Docker cap 128)."
  echo "       Shorten a patch's NAME suffix in its step above (the full name"
  echo "       stays in that step's comments/echo; the tag only needs to be"
  echo "       unique enough to read)."
  exit 1
fi
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
echo "[9e] turboquantspec (PR #53406) is the MTP-safety fix for the TurboQuant"
echo "     attention backend; it is active when the serve command uses a"
echo "     turboquant_* --kv-cache-dtype, e.g.:"
echo "       --kv-cache-dtype turboquant_4bit_nc   (4-bit keys w/ norm-corr | 4-bit values,"
echo "         the balanced default; 'nc' = norm correction)"
echo "       or turboquant_k8v4 (8-bit keys) / turboquant_k3v4_nc (3-bit keys) / turboquant_3bit_nc."
echo "     It fixes the TurboQuant + MTP (n=3) repetition-collapse / IMA bug (issue #52475)."
echo "     The default --kv-cache-dtype (auto/bf16) does not touch it."
echo "[10] onercclreset ports the unmerged upstream #58415 (fixed placement): after"
echo "     every graph-capture warmup it issues one tiny eager all-reduce to reset"
echo "     oneCCL's collective chain, so the next capture's torch.xpu.synchronize()"
echo "     cannot hit UR_RESULT_ERROR_DEVICE_LOST. Fixes the 10-06 startup crash"
echo "     (vllm #60379, TP=2 + MTP + GDN) and vllm #58388. No-op on TP=1."
echo "Remember the canary: 5 deterministic prompts, temp=0, sha256 vs eager before trusting graphs."
