# "Take the latest vLLM and adapt the patches for the graph path" — determination

Date: 2026-07-30. Continues ANALYSIS-029 (which fixed the "vLLM 0.29" misnomer and
proved the 7-patch stack applies on the 0.30-dev pin `9612f77077`). This one answers
the follow-up: **pull the *latest* vLLM main (`a8d1aa9c99`) and adapt the patch stack
for the XPU-graph execution path** — i.e. make *vLLM-latest + XPU + TP=2 + MTP +
XPU graphs* actually run on graphs rather than the currently-captured eager runs.
Method: fresh `git apply` validation of all 7 vault patches on `a8d1aa9c99`, source
read of the graph gating in `vllm/platforms/xpu.py` + `vllm/config/vllm.py`, and a
live GitHub-API audit of the 5 relevant open PRs plus the two reference projects
(SergioB dual-B70 cookbook, CySpiegel fork) + the vllm-xpu-kernels release history.

## TL;DR

**No single *merged/released* artifact is a turnkey "TP=2 + MTP + XPU graphs" stack.
But every piece that such a stack needs already exists and is identified — they are
all currently *open PRs or pip-installable releases*, and the vault's own 3 graph-path
patches **already apply cleanly on the latest main** (verified today, no rewrite
needed).** So "take the latest vLLM and adapt the patches for graphs" decomposes into:

1. **Patch adaptation to latest main — already DONE.** All 7 vault patches (the 4
   Qwen/mtpeagle ones `build.sh` wires today **plus** the 3 graph-path ones that are
   currently *not* wired) apply in order on `a8d1aa9c99`, and every `.py` still parses.
   The 3 graph patches are self-contained and **not** redundant (main lacks them).
2. **The env must flip from "graph off" to "graph on."** The captured runs are eager
   because `--enforce-eager` (a **hard** gate) + `VLLM_XPU_ENABLE_XPU_GRAPH` is
   commented out, and the oneCCL block is the *wrong* class of fix (handle-exchange
   overrides instead of the SYCL simple-threshold contract).
3. **MTP correctness needs `vllm_xpu_kernels 0.1.15.x`, not the pinned `0.1.14.1`**
   (the GDN "ragged speculative token traversal" fix `#600` is post-0.1.14.1).
4. **The clean, intended path** is `#51600 + #56013` (torch 2.14 → Level-Zero graph,
   removes the single-GPU restriction) + `#57291` (kernels bump). All still open.

**So: the exact config is *credible and composable today* from open PRs + a pip
kernel bump + the vault's validated patches + an env flip, but it is *not yet a
released/merged turnkey stack*, and no captured run in *this* recipe's environment
proves it end-to-end yet.** The closest published proof that it works is **PR
#54768**, whose test plan *is* the user's config and whose results include
**Qwen3.8-27B GPTQ-INT4 + MTP + TP=2 + XPU graph on 2× Arc Pro B70**.

---

## 1. "vLLM 0.29" vs. the base vs. the latest main (recap, one line)

| Ref | `git describe` / version | role |
|---|---|---|
| `9612f77077` | `v0.29.1rc0-326-g9612f77077` → setuptools `0.30.0.dev326` | the pin the vault stack is currently anchored to |
| `a8d1aa9c99` | latest `main` (post-0.30-dev) | **"the latest vLLM"** the user asked for |

Critical fact verified today: **`a8d1aa9c99` and `9612f77077` have byte-identical
`requirements/xpu.txt`** (`torch==2.13.0`, `triton==3.7.2+xpu`,
`vllm_xpu_kernels==0.1.14.1`) **and byte-identical `supports_xpu_graph()`**
(`is_torch_equal_or_newer("2.11.0.dev")`). So the graph path is *structurally
unchanged* between the pin and latest main — the torch-2.14 / Level-Zero-graph jump
(`#56013`) is **not** in main yet. "Adapting to latest" is therefore mostly a
*patch-applies + env* question, not a torch-version question.

## 2. Do the vault's patches apply on the latest main? (verified 2026-07-30)

Fresh clone of `vllm-project/vllm @ a8d1aa9c99`; each patch applied with `git apply`
(strict, no `--3way`), in the `build.sh` order; then a full `ast.parse` sweep of every
`vllm/**/*.py`:

| Patch (vault) | applies **alone** on a8d1aa9c99 | applies **in build.sh order** | redundant on main? |
|---|---|---|---|
| `vllm-mtp-draft-group-annotation-55390-56026.patch` (mtpeagle) | OK | OK | no |
| `vision-tower-cpu-offload.patch` | OK | OK | no |
| `qwen35-embed-quant.patch` | OK | OK | no |
| `qwen35-mtp-draft-vocab.patch` | **FAIL alone** (needs embed first) | OK (after embed) | no |
| **`xpu-getmemoryinfo-fallback.patch`** (graph-path) | OK | OK | no (vllm-side wrapper; kernel `#499` is a different layer) |
| **`xpu-grammar-bitmask-stream-fix.patch`** (graph-path) | OK | OK | **no** — main has no `use_copy_stream`/`is_xpu` gate (this is upstream `#53997`) |
| **`xpu-triton-allreduce-tp2.patch`** (graph-path) | OK | OK | **no** — patch *creates* `xpu_triton_all_reduce.py`; main has neither file nor the flag (this is upstream `#54768`'s analog) |

- **Ordered result: all 7 apply on `a8d1aa9c99`; 0 Python parse failures.** The only
  "FAIL" is `qwen35-mtp-draft-vocab.patch` applied *in isolation* — that is the
  documented dependency on `qwen35-embed-quant.patch` being applied first (it passes
  in the real build.sh order, exactly as on the base pin).
- **Consequence:** the "take the latest and adapt the patches" work item is
  *effectively complete as far as the patch text goes* — the stack does not need
  rebase/rewrite for latest main. The remaining work is **wiring + env + kernel pin**
  (see §8), i.e. decisions, not patch surgery.

## 3. What actually gates XPU graphs on the latest main (4 gates, in order)

Reading `vllm/config/vllm.py` + `vllm/platforms/xpu.py` on `a8d1aa9c99`:

- **Gate A — torch version:** `supports_xpu_graph()` = `is_torch_equal_or_newer(
  "2.11.0.dev")`. `torch==2.13.0` → **passes** (both pin and latest main).
- **Gate B — the env var:** `check_and_update_config` sets `cudagraph_mode = NONE`
  when `not envs.VLLM_XPU_ENABLE_XPU_GRAPH`, logging *"XPU Graph is disabled by
  environment variable, please set VLLM_XPU_ENABLE_XPU_GRAPH=1"* — **this is the
  exact line in the captured logs.** The var is currently *commented out* in both
  `1_env.txt` and `2_env.txt`. → **must be set to `1`.**
- **Gate C — `--enforce-eager` (the #1 real blocker):** in `vllm/config/vllm.py`,
  `if self.model_config.enforce_eager: compilation_config.cudagraph_mode =
  CUDAGraphMode.NONE` (twice). This runs and **nothing after it re-enables a graph
  mode** — `check_and_update_config` can only *downgrade to NONE or warn*, it never
  sets a non-NONE mode. So **`--enforce-eager` overrides the graph env var and is the
  hard reason the captured runs are eager.** → **must be removed** to get graphs.
- **Gate D — the real multi-GPU limitation (warning, not a gate):** the `else`
  branch (supports AND env set) only *warns* *"XPU Graph support is experimental and
  currently only supports single-GPU execution."* It does **not** force NONE. So on
  latest main, setting the env + dropping `--enforce-eager` will *attempt* TP=2
  graph capture despite the warning. **Whether that capture actually succeeds is the
  torch/kernel L0-graph interop question** — the real multi-GPU constraint — not a
  config gate. (See §4–§6.)

**Net:** to even *attempt* graphs the recipe needs `VLLM_XPU_ENABLE_XPU_GRAPH=1`
**and** no `--enforce-eager`. Both are currently the opposite in the captured env.

## 4. The oneCCL contract that makes TP=2 *start and record* under a graph

The vault's captured env uses the **wrong class** of oneCCL mitigation:
`CCL_ZE_IPC_EXCHANGE=sockets` and `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` are
*handle-exchange* overrides. The authoritative **SergioB dual-B70 cookbook**
(`docs/DUAL-B70-TP2.md`, master) and **PR #54768's test plan** both say the working
contract is the **SYCL simple-algorithm thresholds** (which keep practical collectives
off the peer-device-IPC path the `xe` driver rejects on non-P2P desktop dual-card
layouts), and explicitly *not* the handle-exchange overrides:

```bash
-e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
-e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
-e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
-e CCL_SYCL_ALLTOALL_TMP_BUF=1
# DO NOT add CCL_ZE_IPC_EXCHANGE / CCL_ATL_TRANSPORT / FI_PROVIDER / CCL_ATL_SHM
#   (they pick how handles are exchanged, not which algorithm is chosen;
#    some combos return the runtime to the failing path)
```
Docker: `--device /dev/dri --ipc=host --cap-add SYS_PTRACE` (vault already has
`ipc: host`; `SYS_PTRACE` permits oneCCL's cross-rank pidfd exchange under seccomp —
needs confirming on the runtime).

This is the same workaround class Intel documented for dual-B60 without P2P
(`intel/llm-scaler#594`). **PR #54768 states it plainly:** *"with its SYCL kernels
and `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` and `CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD`
raised, **TP=2 starts and records**, and from 64 KiB up oneCCL is level with this
path or faster."* So oneCCL **can** be recorded into an XPU graph on this stack —
once the thresholds are raised.

## 5. MTP correctness requires `vllm_xpu_kernels 0.1.15.x`, not the pinned `0.1.14.1`

| Kernel build | upload date | contains GDN `#600` fix? |
|---|---|---|
| **`0.1.14.1`** (pinned by *both* base and latest main) | **2026-08-28** | **NO** |
| `0.1.15.1` | 2026-09-17 03:35 | **yes** (one day after the fix) |
| `0.1.15.3` (PyPI latest) | 2026-09-17 14:17 | **yes** |

The fix is **`da16a5595c` "[GDN] Fix ragged speculative token traversal (#600)"**,
merged **2026-09-16** to official `vllm-xpu-kernels` main. Verified by ancestry:
`da16a5595c` **is in main, NOT in the `0.1.14.1` tag**. (The CySpiegel-fork GDN
commits `06e0d7f`/`2c86542` are *not* in official main — they're the fork's own
rebase of the same class of fix; the official one to rely on is `#600`/`da16a5595c`.)

**Consequence:** for *correct* MTP (no ragged-spec-decode corruption) on latest main
you must **override the kernel pin to `0.1.15.1` or `0.1.15.3`** (both pip-installable
now), or wait for **PR #57291** ("[XPU] Bump kernels to 0.1.15.1") to merge, or
source-build `vllm-xpu-kernels` from main. This is the `KERNEL_MTP_PIN` open question
— resolved: **the pin is the blocker; `0.1.15.3` is the fix.**

## 6. Direct answer: is there a PR / patch / project for *exactly* this config?

| Candidate | What it gives you for *TP=2 + MTP + graphs* | Status |
|---|---|---|
| **vllm #51600** "[XPU] enable XPU GRAPH by default" | **The intended mechanism.** *"PyTorch XPU 2.14 switches XPU Graph from SYCL Graph to Level Zero Graph, addressing previous multi-GPU limitations and high memory usage."* Removes the env var **and** the single-GPU restriction. Depends on torch 2.14. | **OPEN** (needs #56013) |
| **vllm #56013** "[XPU] upgrade to PyTorch 2.14" | The torch half of #51600. | **OPEN** |
| **vllm #54768** "[XPU] Route small TP all-reduces to a Level Zero IPC kernel" | **The decisive proof.** Its test plan *is* the user's config (`CCL_SYCL_*_SIMPLE_THRESHOLD` + `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `--tensor-parallel-size 2`), and its results include **Qwen3.8-27B GPTQ-INT4, 2× Arc Pro B70, XPU graph, with MTP** (84.14 oneCCL → 85.20 with the L0-IPC kernel, +1.3%). Upstream analog of the vault's `xpu-triton-allreduce-tp2.patch`. | **OPEN**; the L0-IPC *kernel* needs a `vllm-xpu-kernels#570` build, but the **oneCCL-threshold graph-recording part works with stock `0.1.15.x`** |
| **vllm #53997** "[XPU][V2] Keep grammar-bitmask copies on the current stream under XPU graphs" | Fixes the structured-output/graph crash; the vault's `xpu-grammar-bitmask-stream-fix.patch` is the local equivalent. | **OPEN** |
| **vllm #57291** "[XPU] Bump kernels to 0.1.15.1" | Brings main onto the MTP-correct kernels (§5). | **OPEN** |
| **SergioB `intel-arc-pro-b70-inference-cookbook`** (master) | The published **dual-B70 TP2 project**: worker-affinity patch, the oneCCL threshold contract (§4), Docker flags, and a measured TP2/TP4/PP contract. **But its published stance for multi-GPU is *compile-only* (no graph)** — *"XPU graph capture has been refused for multi-GPU execution on the tested image generations, so TP2/PP2 uses compile-only execution"* — and its recorded graph-capture failure (`wait ... event associated with a command graph`) is on the **FP8-MoE** path, not the user's dense-27B int4. | **PUBLISHED** (but = compile-only, pre-graph) |
| **CySpiegel fork / this vault** (7-patch stack on 0.30-dev) | The closest *working recipe*; but the **captured runs are eager** (graph off, §3). | **local** |

**So: as a *merged/released turnkey stack* — no. As a *composed stack from open PRs
+ a pip kernel bump + the vault's validated patches + an env flip* — yes, and the
composition is fully specified below.**

## 7. The nuance that makes "yes, it can work" credible *for this model*

The cookbook's graph-**refusal** was (a) on the older `vllm-xpu-kernels 0.1.12.3`
image gen and (b) specifically the **FP8-MoE** capture path. The user's target is
**dense Qwen3.8-27B (int4/GPTQ) + MTP** — which is *exactly* the model PR #54768
tested **successfully under XPU graph on 2× B70, including with MTP**. So the
published third-party evidence that "TP=2 + MTP + XPU graph works" is *directly on
the user's model class*, just on a nightly build (pre-merge) with `0.1.15.x` kernels
and the §4 threshold contract.

## 8. The exact recipe to make it run on graphs (RECOMMENDATION — not applied to the vault)

Two options. **Neither was applied** to `build.sh`/`1_env.txt`/`2_env.txt` (per the
no-edit constraint); these are the specified changes to make in a *scratch* copy first.

**Option 1 — conservative, on latest main `a8d1aa9c99` / torch 2.13 (works today if the
L0 interop cooperates):**
- **build.sh:** wire in the 3 graph-path patches so the full 7-patch stack is applied:
  keep steps 2–5 (mtpeagle, visionoffload, embed-quant, mtp-vocab) and **add**
  `xpu-getmemoryinfo-fallback`, `xpu-grammar-bitmask-stream-fix`,
  `xpu-triton-allreduce-tp2` (each `git apply` with the same loud-fail guard).
- **kernel pin:** override `vllm_xpu_kernels==0.1.14.1` → **`0.1.15.3`** (MTP
  correctness, §5).
- **env (both workers):**
  - **set** `VLLM_XPU_ENABLE_XPU_GRAPH=1`;
  - **remove** `--enforce-eager`;
  - **add** the §4 oneCCL block (`CCL_SYCL_*_SIMPLE_THRESHOLD=4294967296`,
    `CCL_SYCL_ALLTOALL_TMP_BUF=1`);
  - **remove** `CCL_ZE_IPC_EXCHANGE=sockets` and `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`
    (wrong class, §4);
  - **keep** `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `--ipc=host`; **add/confirm**
    `--cap-add SYS_PTRACE`;
  - **optionally** set `VLLM_XPU_TRITON_ALLREDUCE=1` to use the vault's local
    Triton one-shot all-reduce for the ≤64 KiB decode collectives (the
    `xpu-triton-allreduce-tp2.patch` path) — the #54768 upstream analog instead
    needs the `vllm-xpu-kernels#570` build.
- **Risk:** torch 2.13 multi-GPU graph capture is still "experimental /
  single-GPU-restricted" per the warning; it *records* on the #54768 nightly with
  `0.1.15.x` kernels, but may refuse on some driver/torch combos. Known-good
  fallback = **compile-only** (torch.compile, `--enforce-eager` off but no graph)
  per the cookbook.

**Option 2 — the clean, intended path (forward-looking, no local graph patches needed
for the *mechanism*):** wait for **`#56013` (torch 2.14) + `#51600` (graph by
default, single-GPU restriction removed) + `#57291` (kernels 0.1.15.1)** to merge.
Then the stack is just: the Qwen/MTP patches (which survive, §2) + the §4 oneCCL
thresholds + no `--enforce-eager`. The grammar/all-reduce graph patches become
redundant once their upstream PRs (`#53997`/`#54768`) land. This is the path to
*keep watching*; it removes the "experimental single-GPU" caveat entirely.

## 9. What is verified vs. still open risk

**Verified (2026-07-30, this session):**
- All 7 vault patches apply in order on `a8d1aa9c99`; 0 `.py` parse failures; the 3
  graph patches are self-contained and not redundant on main.
- `xpu.txt` and `supports_xpu_graph()` are **identical** between base `9612f77077` and
  latest `a8d1aa9c99` (torch 2.13 / kernels 0.1.14.1 / ≥2.11.0.dev) → the graph *path*
  is structurally unchanged; the jump to Level-Zero graph is **not** in main yet.
- `--enforce-eager` is a **hard** gate that overrides `VLLM_XPU_ENABLE_XPU_GRAPH`
  (it forces `cudagraph_mode=NONE` and nothing re-enables it); the "single-GPU"
  message is a **warning**, not a gate.
- The 5 relevant PRs (#51600, #56013, #54768, #53997, #57291) are **all open / unmerged**.
- GDN `#600` fix is in kernels main but **not** in `0.1.14.1`; `0.1.15.1/.3` post-date
  it (Sep 17) and are pip-installable.
- The oneCCL threshold contract (§4) and the "do not add handle-exchange overrides"
  warning are sourced from the dual-B70 cookbook + #54768.

**Open risk (NOT yet demonstrated in this recipe):**
- **No captured run in *this* vault's environment proves TP=2 + MTP + XPU graphs
  completes on the user's actual B50/B70 hardware with a latest-main build.** The
  closest third-party proof (#54768) is on a *nightly*, *pre-merge*, with a
  *custom kernel build*. Confidence that it **can** work is high (it's the user's
  exact model class, tested on 2× B70 with MTP + graph), but it is not yet shown in
  this specific recipe/driver combo. First smoke-test should watch for the
  "XPU Graph is disabled…" / capture lines and a `--enforce-eager`-free server that
  actually enters graph mode, and compare eager-vs-graph tok/s before trusting it.

## 10. Recommended next steps (priority order)

1. Confirm the runtime has `SYS_PTRACE` (and `ipc: host`) — oneCCL needs it.
2. In a **scratch copy** of the recipe (not the vault), apply **Option 1** from §8.
3. Smoke-test **TP=2 + MTP** with graphs on; watch the server log for graph-capture
   success (vs. the FP8-style `wait ... event associated with a command graph` error)
   and for the oneCCL threshold effect; compare eager-vs-graph throughput.
4. If capture **refuses** (multi-GPU graph on this torch/driver), fall back to
   **compile-only** (the cookbook's known-good path) and keep MTP; that is a usable,
   if slower, config.
5. Track **`#51600` / `#56013` / `#57291`**; when they land, re-validate the stack on
   the new pin and drop the now-redundant graph patches (`#53997`/`#54768` land the
   grammar + all-reduce fixes upstream).

## Appendix — patch ↔ upstream correspondence

| Vault patch | Upstream | Note |
|---|---|---|
| `xpu-triton-allreduce-tp2.patch` | **#54768** | local = Triton one-shot `OneShotAllReduce`; upstream = Level-Zero IPC kernel (needs `vllm-xpu-kernels#570`). Same intent: a graph-recordable 2-rank TP all-reduce. |
| `xpu-grammar-bitmask-stream-fix.patch` | **#53997** | near-identical intent (keep bitmask H2D on the current stream under XPU graph). |
| `xpu-getmemoryinfo-fallback.patch` | kernels **#499** | vllm-side wrapper fallback vs. kernel-side C++ fix; different layers — likely a no-op belt-and-suspenders now, verify before keeping. |
| (graph enable / torch 2.14) | **#51600 + #56013** | the mechanism itself; env var + single-GPU restriction removed in #51600. |
| (MTP kernel correctness) | **#57291 + kernels #600** | pin `0.1.15.x`. |
