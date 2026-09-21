#!/usr/bin/env python
"""Does the torch symmetric-memory (Level-Zero) transport come up on THIS GPU pair?

This is the exact init path of the Triton one-shot all-reduce (patch [8],
xpu_triton_all_reduce.OneShotAllReduce): enable_symm_mem_for_group ->
symm.empty -> symm.rendezvous -> get_buffer. That `rendezvous` call is the one
that threw "L0 error 45" on 2026-09-21 on a B50/B70 pair. If it constructs here,
VLLM_XPU_TRITON_ALLREDUCE=1 can actually engage; if it raises, you stay on oneCCL.

It runs the PATCHED class itself (not a re-implementation), so it needs an image
with patch [8] applied (the build-graph.sh output). Two seconds; no model load.

Run inside the XPU image with BOTH render nodes visible, e.g.:

  docker run --rm \
    --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
    vllm-intel-xpu:TAG \
    python /workspace/symm_rendezvous_test.py

Read the final line:
  RESULT: SYMM_TRANSPORT_OK  -> the Triton AR path is viable; try the flag ON
  RESULT: SYMM_TRANSPORT_FAIL <exc>  -> L0 symm doesn't come up here; keep flag OFF
For the full answer (correctness + latency vs oneCCL, 20k+200k iters) instead run
the patched module's own harness:

  python -m vllm.distributed.device_communicators.xpu_triton_all_reduce
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")


def _worker(rank: int, world: int) -> None:
    dev = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    try:
        from vllm.distributed.device_communicators.xpu_triton_all_reduce import (
            OneShotAllReduce,
        )

        # This __init__ performs enable_symm_mem_for_group + empty + rendezvous +
        # get_buffer. A throw here == the 09-21 L0 error; that's the transport's
        # limit on this pair, independent of the model.
        ar = OneShotAllReduce(dist.group.WORLD, dev)

        # Prove the mapped peer buffers actually exchange data end-to-end:
        # rank0 sends 1.0, rank1 sends 2.0 -> both must observe the sum 3.0.
        t = torch.full((5120,), float(rank + 1), dtype=torch.bfloat16, device=dev)
        out = ar.all_reduce(t)
        ok = torch.allclose(out, torch.full_like(out, 3.0))
        print(f"[rank {rank}] OneShotAllReduce init OK; all_reduce correctness={bool(ok)}")
        dist.barrier()
        if rank == 0:
            print("RESULT: SYMM_TRANSPORT_OK" if ok else "RESULT: SYMM_TRANSPORT_CORRUPT")
        if not ok:
            sys.exit(1)
    except Exception as e:  # noqa: BLE001 - we WANT to report the transport failure
        print(f"[rank {rank}] RESULT: SYMM_TRANSPORT_FAIL {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    world = 2
    mp.spawn(_worker, args=(world,), nprocs=world, join=True)
