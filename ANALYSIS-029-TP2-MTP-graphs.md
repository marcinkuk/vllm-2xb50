# vLLM 0.29 + XPU + TP=2 + MTP + XPU graphs — determination

Date: 2026-07-30. Scope: does any **existing** PR / patch / project already
enable the exact stack *vLLM 0.29, XPU, TP=2, MTP speculative decoding, XPU
graphs*? Method: GitHub API + raw patch fetch + local `git apply` validation
on two fresh checkouts (`/tmp/vllm-base` = pin `9612f77077`, `/tmp/vllm-v029`
= `v0.29.0`/`98dff2a81`) + inspection of the CySpiegel fork image/env.

## TL;DR

**No single existing merged PR, patch, or project fully enables the exact
target stack as "vLLM 0.29 + XPU + TP=2 + MTP + XPU graphs".** The closest
working reference is the **CySpiegel fork / this vault's 7-patch stack on the
0.30-dev pin `9612f77077`** — not true vLLM 0.29.0.

| Candidate | Version it enables | XPU graph @ TP=2 | MTP | Status |
|---|---|---|---|---|
| vLLM upstream `main` / any release | any | **no** — #51600 (enable by default) still OPEN; no graph-safe L0 all-reduce path | yes (natively) | closest official, but no XPU graphs |
| CySpiegel fork (d76d53ba8, built on the 0.30-dev pin) | **0.30-dev** (`0.30.0.dev326+g9612f77077`) | **yes — demonstrated** (in-image env enables it, runs TP=2) | yes | **closest viable path** |
| This vault, 7-patch stack | **0.30-dev** pin only (see below) | yes, **if** the 3 XPU patches + env block are used | yes | functionally equivalent to fork (same work rebased) |
| True **vLLM v0.29.0** (`98dff2a81`) | v0.29.0 | only after **rebasing the whole stack** — current 7-patch set **FAILS** on it | yes (after rebase) | requires rewrite work |

**Exact match: none. Closest viable: 0.30-dev pin + vault 7-patch stack
+ fork's in-image env block.**

## Why "vLLM 0.29" in the title is a misnomer here

- The vault's patch stack is anchored to upstream pin **`9612f77077`**
  (`git describe`: `v0.29.1rc0-326-g9612f770`), which setuptools-scm reports
  as **`0.30.0.dev326+g9612f77077`** — i.e. **pre-0.30 dev, not v0.29**:
  326 commits *after* `v0.29.1rc0`, 6 commits *before* `v0.30.0rc1`
  (tag distance).
- The README `# base: vllm@v0.29.0`-style comment / "v0.29" wording is
  **stale** — the stack was rebased against the 0.30-dev pin.
- **The current ordered 7-patch set does NOT apply on true v0.29.0.**
  Validated 2026-07-30 (`/tmp/val029.sh`):

  | Patch | v0.29.0 | 0.30-dev pin |
  |---|---|---|
  | `vllm-mtp-draft-group-annotation-55390-56026.patch` | **FAIL** (first failure, `kv_cache_utils.py:2117`) | OK |
  | `qwen35-mtp-draft-vocab.patch` | **FAIL** | OK |
  | `xpu-grammar-bitmask-stream-fix.patch` | **FAIL** | OK |
  | `vision-tower-cpu-offload.patch` | OK | OK |
  | `qwen35-embed-quant.patch` | OK | OK |
  | `xpu-getmemoryinfo-fallback.patch` | OK | OK |
  | `xpu-triton-allreduce-tp2.patch` | (in the 7; OK on pin) | OK |

  Ordered full-stack result: pin = **PASS (7/7, all 10 files parse)**,
  v0.29.0 = **FAIL**.
- **Consequence:** if *true* vLLM 0.29.0 is a hard requirement, the
  affected patches (at minimum the draft-group, MTP-draft-vocab, and
  grammar-bitmask ones) must be **rebased/rewritten for v0.29.0** before
  the exact "0.29 + graphs" stack can exist. That is new work; nothing
  published today provides it.

## XPU graph path — what actually gates it (no hidden TP blocker)

- `vllm/platforms/xpu.py::check_and_update_config` forces
  `compilation_config.cudagraph_mode = CUDAGraphMode.NONE` when
  `not supports_xpu_graph()` **or** `not envs.VLLM_XPU_ENABLE_XPU_GRAPH`.
  `supports_xpu_graph()` = `is_torch_equal_or_newer("2.11.0.dev")`
  (`vllm/utils/torch_utils.py`).
- The "XPU graph is single-GPU" wording is a **`warning_once` only —
  not a hard TP>1 block**. The real constraint is a **graph-capturable
  all-reduce path**: TP all-reduces run *inside* the captured graph, so the
  communicator must be capture-safe (the stock oneCCL/SYCL path is not —
  that is exactly what `CCL_ENABLE_SYCL_KERNELS=0` "fixes" by killing
  graph capture).
- **Proof of concept: the CySpiegel fork runs TP=2 + XPU graphs.**
  Its in-image env (Dockerfile):
  `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_USE_V2_MODEL_RUNNER=0`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TRITON_INTEL_DEVICE_ARCH=bmg`,
  `CCL_ZE_IPC_EXCHANGE=pidfd`, `FI_PROVIDER_PATH=/opt/venv/lib`.
- The fork's XPU diff (`ca90b9e7d..d76d53ba8`) is **functionally equivalent
  to this vault's patch set** (same work rebased):
  `xpu_communicator.py` changes, added `xpu_triton_all_reduce.py`
  (= `patches/xpu-triton-allreduce-tp2.patch`), `xpu.py` `getMemoryInfo`
  fallback (= `patches/xpu-getmemoryinfo-fallback.patch`).

## Upstream PRs — none merged, none complete

| PR | Content | Status (2026-07-30) | Use |
|---|---|---|---|
| **#51600** | enable XPU graph **by default** (`VLLM_XPU_ENABLE_XPU_GRAPH` default `False` in the pin) | **OPEN, not in the pin** | would remove the need for the env flag once merged |
| **#53997** (CySpiegel) | XPU grammar-bitmask stream fix in `vllm/v1/worker/gpu/structured_outputs.py` | **OPEN** | upstream twin of vault `xpu-grammar-bitmask-stream-fix.patch` (content matches); re-audit before each build |
| **#54768** (CySpiegel) | **L0 IPC all-reduce** for small-TP all-reduces — the graph-capturable TP path | **OPEN**; applies **cleanly** onto pin `9612f77077` | the forward-looking fix for TP+graphs; **blocked on kernels** (below) |
| kernels **PR #570** | adds `vllm_xpu_kernels/p2p.py` (+ C++/IPC) | **OPEN, not in `0.1.14.1`** | **prerequisite for #54768** — `vllm_xpu_kernels==0.1.14.1` (used by both pin and v0.29.0) **lacks the `p2p` module** |

So even the *most* complete individual upstream contribution (#54768) does
not work standalone today: it needs an **unreleased kernels build**
(≥ post-#570), and it still leaves MTP-specific fixes (draft-group
annotation, draft vocab) to #55390/#56026 (both OPEN — covered by the
vault's combined patch).

## Conflicts in the current vault setup

1. **`1_env.txt` is misaligned with graph capture** (the production file):
   - `CCL_ENABLE_SYCL_KERNELS=0` — **graph-capture killer** (forces
     non-capturable SYCL kernels out; only works with `--enforce-eager`);
   - `VLLM_XPU_ENABLE_XPU_GRAPH=1` — **commented out** (disabled);
   - `CCL_ZE_IPC_EXCHANGE=sockets` — conflicts with the fork's
     `pidfd`; the full set of CCL mitigations is active while the graph
     path is off;
   - fork block missing: `TRITON_INTEL_DEVICE_ARCH=bmg`,
     `VLLM_USE_V2_MODEL_RUNNER=0`, `FI_PROVIDER_PATH=/opt/venv/lib`.
   - `2_env.txt` (the CCL-mitigations-off variant) is the one closer to
     fork-compatible — it keeps `CCL_TOPO_P2P_ACCESS=0` but drops the
     rest of the mitigation block.
   - **Fix:** drop `CCL_ENABLE_SYCL_KERNELS=0` (and the other CCL
     mitigations that fight the L0/IPC path), and enable the graph/MTP env
     block, i.e. mirror the fork's in-image block listed above.
2. **The command still carries `--enforce-eager`** (both env files) — this
   alone forces `CUDAGraphMode.NONE` regardless of every flag above.
   It must go for graphs to run.
3. **MTP is off in the command**: the `--speculative-config
   '{"method":"mtp","num_speculative_tokens":3}'` line is commented out in
   `1_env.txt`/`2_env.txt`.
4. **3 of the 7 vault patches are orphaned:** `build.sh` applies only 4
   (draft-group, vision-offload, embed-quant, mtp-vocab). The three XPU
   ones — `xpu-triton-allreduce-tp2`, `xpu-grammar-bitmask-stream-fix`,
   `xpu-getmemoryinfo-fallback` — are **not applied by any build step**
   (no reference in `build.sh`, README, or env files). The running image
   tag `2026-07-28_06-41-03a2d03367` also does not match the current
   `build.sh` output format (`<DATE>-<7-char-hash>` + `-mtpeagle -
   visionoffload -noembed -mtp` suffixes; that short hash is 10 chars and
   carries no suffixes) so it predates that build.sh revision, or comes
   from the CySpiegel fork's builder. Either way there is no proof the
   running image contains the 3 XPU patches, and the current `build.sh`
   as written would not add them. (`build.sh` is intentionally unmodified
   per task constraint; wiring these steps in is a follow-up decision for
   the vault owner.)
5. **Stale base comment:** `# base: vllm@v0.29.0`-style wording in the
   vault overstates the pin; the actual base is the 0.30-dev pin
   `9612f77077`.

## Recommendation

1. **Accept the 0.30-dev pin `9612f77077` as the base** (the version the
   patch stack and the fork both target). It is 326 commits past
   `v0.29.1rc0` and 6 from `v0.30.0rc1` — the right trade: v0.30.0 final
   doesn't exist yet as a buildable tag, and v0.29.0 can't take the stack.
2. **Build from the full 7-patch set** (all validated `git apply` clean on
   the pin, 2026-07-30), i.e. wire the 3 orphaned XPU patches into the
   build — or build from the **CySpiegel fork image directly** (equivalent
   work, already packaged with the correct env).
3. **Adopt the fork's in-image env block** in `1_env.txt` (drop
   `CCL_ENABLE_SYCL_KERNELS=0` + the full CCL mitigation set, set
   `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_USE_V2_MODEL_RUNNER=0`,
   `CCL_ZE_IPC_EXCHANGE=pidfd`, `TRITON_INTEL_DEVICE_ARCH=bmg`,
   `FI_PROVIDER_PATH=/opt/venv/lib`), and **remove `--enforce-eager`** and
   **uncomment the MTP speculative-config** line.
4. **Re-audit the open PRs before each build** (GitHub API, as the README
   audit does): #55390/#56026 (→ drop patch 1 when merged), #53997
   (→ drop `xpu-grammar-bitmask-stream-fix.patch`), #51600 (→ drop the
   `VLLM_XPU_ENABLE_XPU_GRAPH` flag when merged), #54768 + kernels #570
   (→ drop the Triton-AR / IPC patches once the L0 IPC path lands with
   `p2p` in a released `vllm_xpu_kernels`).

## Reproduce

```sh
# v0.29.0 failure (3 of 7 patches fail)
bash /tmp/val029.sh          # runs ordered git-apply check on /tmp/vllm-v029

# pin success (7/7)
cd /tmp/vllm-base && git status --short   # 10 files modified = the 7 patches
```

Working trees: `/tmp/vllm-base` (pin, 7 patches applied, kept for
inspection), `/tmp/val029.sh` (validation script). No changes to
`vllm-project/vllm` source were committed; `build.sh` in this vault was
**not modified** (constraint respected).
