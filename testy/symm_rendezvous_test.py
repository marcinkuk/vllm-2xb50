#!/usr/bin/env python
"""Does the torch symmetric-memory (Level-Zero) transport come up on THIS GPU pair?

This is the exact init path of the Triton one-shot all-reduce (patch [8],
xpu_triton_all_reduce.OneShotAllReduce): enable_symm_mem_for_group ->
symm.empty -> symm.rendezvous -> get_buffer. That `rendezvous` call is the one
that threw "L0 error 45" on 2026-09-21 on a B50/B70 pair. If it constructs here,
VLLM_XPU_TRITON_ALLREDUCE=1 can actually engage; if it raises, you stay on oneCCL.

It runs the PATCHED class itself (not a re-implementation), so it needs an image
with patch [8] applied (the build-graph.sh output). A few seconds; no model load.

Three legs, so a failure tells you WHICH layer is broken:
  leg0  oneCCL: xccl all_reduce + barrier — the same collectives vLLM uses for
        prefill/barriers. If THIS fails, the symm verdict below is meaningless:
        your env (not the L0 transport) is the problem — e.g. if you run the
        image with bare `docker run` instead of the compose, you are missing
        the compose's CCL_SYCL_ALLGATHERV/ALLREDUCE_SIMPLE_THRESHOLD vars.
  leg1  symm:   enable_symm_mem_for_group + empty + rendezvous + get_buffer.
        A throw here == the 09-21 L0 error; the transport's limit on this pair.
  leg2  symm:   one all_reduce through the mapped peer buffers (sum check).

Run inside the XPU image with BOTH render nodes visible, e.g. (from the repo
dir; the mount means no rebuild is needed for script edits):

  docker run --rm \
    --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
    -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=1073741824 \
    -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1073741824 \
    -v "$PWD/testy/symm_rendezvous_test.py":/tmp/symm_test.py:ro \
    --entrypoint python \
    vllm-intel-xpu:TAG \
    /tmp/symm_test.py

Two gotchas:
  * `--entrypoint python` is REQUIRED: these images set ENTRYPOINT to `vllm`
    (the compose nulls it out with `entrypoint: []`). Without the override
    you get "vllm: error: unrecognized arguments: /tmp/symm_test.py".
  * the image must carry patch [8] tritonar (tag suffix `-tritonar`), because
    leg 1 imports xpu_triton_all_reduce.OneShotAllReduce — a module that
    patch [8] adds. An older image (pre-tritonar, e.g. a tag ending in
    `-noembed-mtp`) fails leg 1 with ImportError: that is a wrong-image
    verdict, NOT a transport verdict; leg 0 (oneCCL) does not depend on any
    local patch.

Final line:
  RESULT: ONECCL_BASELINE_FAIL <exc> -> oneCCL is broken in THIS env; fix the
      env (compose vars / CCL settings) and re-run before judging the transport
  RESULT: SYMM_TRANSPORT_FAIL (<phase>) <exc> -> oneCCL works, L0 symm doesn't:
      keep VLLM_XPU_TRITON_ALLREDUCE=0 and use oneCCL with your CCL_* mitigations
  RESULT: SYMM_TRANSPORT_OK  -> the Triton AR path is viable; try the flag ON
  (a SYMM_TRANSPORT_CORRUPT line = buffers map but data is wrong; treat as FAIL)
For the full answer (correctness + latency vs oneCCL, 20k+200k iters) instead run
the patched module's own harness:

  python -m vllm.distributed.device_communicators.xpu_triton_all_reduce

Running it ALONGSIDE a live server (no downtime, no rebuild):

  1) docker cp testy/symm_rendezvous_test.py <container>:/tmp/symm_test.py
  2) pick a lull (no long generation in flight on the server)
  3) docker exec -e MASTER_PORT=29617 --entrypoint python <container> /tmp/symm_test.py
     (--entrypoint is needed here too: exec uses the IMAGE's entrypoint — vllm —
      not the container's compose override)

docker exec inherits the container's environment (your CCL_* vars) and the
same render nodes, so leg 0 sees the exact server conditions. The test adds
only a few MB and a few small kernels for a few seconds; it uses its own
process group and port, so it does not touch the server's.
Residual risk: the 07-28 crash was a level-zero IPC driver bug, and leg 1
does a second L0 IPC exchange while the server is busy. Low probability, but
watch `docker logs -f <container>` during the run: if you see
zeMemOpenIpcHandle / L0 errors in the SERVER log, abort, `docker restart
<container>` (restart: unless-stopped covers it; KV cache is lost, agents
reconnect), and keep VLLM_XPU_TRITON_ALLREDUCE=0.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")


def _oneccl_baseline(rank: int, dev) -> None:
    """Leg 0: prove XCCL collectives work in this env before judging symm."""
    t = torch.full((8192,), 1.0, dtype=torch.bfloat16, device=dev)
    dist.all_reduce(t)
    if not torch.allclose(t, torch.full_like(t, 2.0)):
        raise RuntimeError(f"xccl all_reduce corrupted: first4={t[:4].tolist()}")
    dist.barrier()
    print(f"[rank {rank}] leg0 oneCCL all_reduce+barrier OK")


def _worker(rank: int, world: int) -> None:
    dev = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    phase = "oneCCL baseline"
    try:
        _oneccl_baseline(rank, dev)

        # Leg 1: the exact patched init. A throw here == the 09-21 L0 error 45.
        phase = "symm rendezvous"
        from vllm.distributed.device_communicators.xpu_triton_all_reduce import (
            OneShotAllReduce,
        )

        ar = OneShotAllReduce(dist.group.WORLD, dev)

        # Leg 2: prove the mapped peer buffers exchange data end-to-end:
        # rank0 sends 1.0, rank1 sends 2.0 -> both must observe the sum 3.0.
        phase = "symm all_reduce"
        t = torch.full((5120,), float(rank + 1), dtype=torch.bfloat16, device=dev)
        out = ar.all_reduce(t)
        ok = torch.allclose(out, torch.full_like(out, 3.0))
        print(f"[rank {rank}] OneShotAllReduce OK; all_reduce correctness={bool(ok)}")
        dist.barrier()
        if rank == 0:
            print("RESULT: SYMM_TRANSPORT_OK" if ok else "RESULT: SYMM_TRANSPORT_CORRUPT")
        if not ok:
            sys.exit(1)
    except Exception as e:  # noqa: BLE001 - we WANT to report the transport failure
        if phase == "oneCCL baseline":
            print(f"[rank {rank}] RESULT: ONECCL_BASELINE_FAIL {type(e).__name__}: {e}")
        else:
            print(f"[rank {rank}] RESULT: SYMM_TRANSPORT_FAIL ({phase}) {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    world = 2
    mp.spawn(_worker, args=(world,), nprocs=world, join=True)
