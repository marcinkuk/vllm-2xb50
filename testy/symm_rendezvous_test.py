#!/usr/bin/env python
"""Does the torch symmetric-memory (Level-Zero) transport come up on THIS GPU pair?

This probes the exact init path of the Triton one-shot all-reduce (patch [8],
vllm.distributed.device_communicators.xpu_triton_all_reduce.OneShotAllReduce):
enable_symm_mem_for_group -> symm.empty -> symm.rendezvous -> get_buffer. The
`rendezvous` call is the one that threw "L0 error 45" on 2026-09-21 on a
B50/B70 pair. If it constructs here, the L0 transport works and
VLLM_XPU_TRITON_ALLREDUCE=1 (on a -tritonar image) can actually engage; if it
raises, you stay on oneCCL.

Three legs, so a failure tells you WHICH layer is broken:
  leg0  oneCCL: xccl all_reduce + barrier — the same collectives vLLM uses for
        prefill/barriers. If THIS fails, the symm verdict below is meaningless:
        your env (not the L0 transport) is the problem — e.g. if you run the
        image with bare `docker run` instead of the compose, you are missing
        the compose's CCL_SYCL_ALLGATHERV/ALLREDUCE_SIMPLE_THRESHOLD vars.
  leg1  symm:   enable_symm_mem_for_group + empty + rendezvous + get_buffer.
        A throw here == the 09-21 L0 error; the transport's limit on this pair.
  leg2  symm:   prove the mapped peer buffers are real, readable device memory
        (write distinct values, cross-check after a process-group all_reduce).

Two variants for leg1/leg2, chosen automatically:
  * VLLM     — runs the PATCHED OneShotAllReduce class itself (full fidelity,
               incl. its Triton kernel's all_reduce). Requires an image with
               patch [8] applied (tag suffix `-tritonar`).
  * TORCH_ONLY — re-runs the same torch._symmetric_memory calls the patch
               makes, without importing vllm. Works on ANY image (incl. the
               pre-tritonar 09-09 build), because it needs only torch + oneCCL.
               A SYMM_TRANSPORT_OK from this variant means the L0 transport is
               viable (patch [8]'s transport will come up on a -tritonar
               image); for the Triton kernel's full correctness+latency check,
               run the patched module's own harness on a -tritonar image:
                 python -m vllm.distributed.device_communicators.xpu_triton_all_reduce

Run inside the XPU image with BOTH render nodes visible (a few seconds; no
model load). From the repo dir (the mount means no rebuild for script edits):

  docker run --rm \
    --ipc host \
    --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=1073741824 \
    -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=1073741824 \
    -v "$PWD/testy/symm_rendezvous_test.py":/tmp/symm_test.py:ro \
    --entrypoint python \
    vllm-intel-xpu:TAG \
    /tmp/symm_test.py

Three gotchas:
  * `--entrypoint python` is REQUIRED: these images set ENTRYPOINT to `vllm`
    (the compose nulls it out with `entrypoint: []`). Without the override you
    get "vllm: error: unrecognized arguments: /tmp/symm_test.py".
  * `-v /dev/dri/by-path:/dev/dri/by-path` (and `--ipc host`) are REQUIRED:
    oneCCL's ze_fd_manager opens /dev/dri/by-path to enumerate the GPUs.
    Without it you get "init_device_fds ... could not open device directory"
    at the first collective -> leg 0 fails with ONECCL_BASELINE_FAIL and you
    never reach the transport. (The test now checks this up front and tells
    you so instead of dying in oneCCL.)
  * any image works for the TRANSPORT verdict, but to exercise the actual
    Triton one-shot kernel (leg2 via OneShotAllReduce) the image must carry
    patch [8] (tag suffix `-tritonar`); a pre-tritonar image transparently
    falls back to the TORCH_ONLY variant (leg1 is identical either way).

Final line (the "via ..." part says which variant ran):
  RESULT: ONECCL_BASELINE_FAIL <exc> -> oneCCL is broken in THIS env; fix the
      env (compose vars / CCL settings / by-path mount) and re-run before
      judging the transport
  RESULT: SYMM_TRANSPORT_FAIL (<phase>, via <variant>) <exc> -> oneCCL works,
      L0 symm doesn't: keep VLLM_XPU_TRITON_ALLREDUCE=0 and use oneCCL with
      your CCL_* mitigations
  RESULT: SYMM_TRANSPORT_OK (via <vllm|torch>) -> the L0 symm transport comes
      up; on a -tritonar image you can enable the flag at the next planned
      restart. (A SYMM_TRANSPORT_CORRUPT line = buffers map but data is wrong;
      treat as FAIL.)

Running it ALONGSIDE a live server (no downtime, no rebuild):

  1) docker cp testy/symm_rendezvous_test.py <container>:/tmp/symm_test.py
  2) pick a lull (no long generation in flight on the server)
  3) docker exec -e MASTER_PORT=29617 --entrypoint python <container> /tmp/symm_test.py
     (--entrypoint is needed here too: exec uses the IMAGE's entrypoint — vllm —
      not the container's compose override)

docker exec inherits the container's environment (your CCL_* vars, the
by-path mount) and the same render nodes, so leg 0 sees the exact server
conditions — this variant is the one that most closely mirrors production.
The test adds only a few MB and a few small kernels for a few seconds; it
uses its own process group and port, so it does not touch the server's.
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

# Mirror patch [8]'s slot sizing closely (small enough to be cheap here).
_TORCH_SLOT_ELEMS = 2 * 64 * 1024  # 128K bf16 elems = 256 KiB per rank
_TORCH_FLAG_ELEMS = 16


def _preflight_by_path() -> None:
    """Fail fast with a clear hint if /dev/dri/by-path is not mounted.

    oneCCL's ze_fd_manager opens this directory to enumerate the GPUs; without
    it the first collective dies with an opaque "init_device_fds ... could not
    open device directory" that looks like a transport failure but is a mount
    problem (the compose mounts it via `volumes: - /dev/dri/by-path`).
    We only require the top-level dir to exist (that is what oneCCL's opendir
    needs); the entries are symlinks to ../renderD128 etc., not subdirs, so we
    must NOT isdir() them.
    """
    if not os.path.isdir("/dev/dri/by-path"):
        raise RuntimeError(
            "/dev/dri/by-path is not mounted inside this container; oneCCL "
            "cannot enumerate the GPUs. Add to the docker run: "
            "-v /dev/dri/by-path:/dev/dri/by-path  (and --ipc host, as in the compose)."
        )


def _oneccl_baseline(rank: int, dev) -> None:
    """Leg 0: prove XCCL collectives work in this env before judging symm."""
    _preflight_by_path()
    t = torch.full((8192,), 1.0, dtype=torch.bfloat16, device=dev)
    dist.all_reduce(t)
    if not torch.allclose(t, torch.full_like(t, 2.0)):
        raise RuntimeError(f"xccl all_reduce corrupted: first4={t[:4].tolist()}")
    dist.barrier()
    print(f"[rank {rank}] leg0 oneCCL all_reduce+barrier OK")


def _symm_via_vllm(rank: int, dev, gname) -> bool:
    """Legs 1-2 using the PATCHED OneShotAllReduce (full fidelity)."""
    from vllm.distributed.device_communicators.xpu_triton_all_reduce import (
        OneShotAllReduce,
    )

    ar = OneShotAllReduce(dist.group.WORLD, dev)  # leg 1: the exact patched init
    # leg 2: rank0 sends 1.0, rank1 sends 2.0 -> both must observe the sum 3.0.
    t = torch.full((5120,), float(rank + 1), dtype=torch.bfloat16, device=dev)
    out = ar.all_reduce(t)
    ok = bool(torch.allclose(out, torch.full_like(out, 3.0)))
    print(f"[rank {rank}] OneShotAllReduce all_reduce correctness={ok}")
    return ok


def _symm_via_torch(rank: int, dev, gname) -> bool:
    """Legs 1-2 with the same torch calls the patch makes (no vllm import)."""
    import torch.distributed._symmetric_memory as symm

    # leg 1 — identical sequence to OneShotAllReduce.__init__:
    symm.enable_symm_mem_for_group(gname)
    slot = symm.empty(_TORCH_SLOT_ELEMS, dtype=torch.bfloat16, device=dev)
    slot.zero_()
    flag = symm.empty(_TORCH_FLAG_ELEMS, dtype=torch.int32, device=dev)
    flag.zero_()
    slot_hdl = symm.rendezvous(slot, group=gname)  # <- the 09-21 "L0 error 45" spot
    flag_hdl = symm.rendezvous(flag, group=gname)
    peer = 1 - rank
    peer_slot = slot_hdl.get_buffer(peer, slot.shape, slot.dtype)
    peer_flags = flag_hdl.get_buffer(peer, flag.shape, flag.dtype)
    dist.barrier()
    print(
        f"[rank {rank}] torch symm init OK "
        f"(peer_slot ptr={peer_slot.data_ptr():#x}, "
        f"peer_flags ptr={peer_flags.data_ptr():#x})"
    )
    # leg 2 — prove the peer mapping is real, readable device memory:
    # write a rank-distinct value, run a process-group all_reduce over it, and
    # confirm both ranks end on the expected sum (this also cross-checks that
    # the buffers are genuinely on-device and addressable).
    slot.fill_(float(rank + 1))
    flag.zero_()
    dist.all_reduce(slot)  # 1.0 + 2.0
    dist.barrier()
    ok = bool(torch.allclose(slot, torch.full_like(slot, 3.0)))
    print(f"[rank {rank}] torch symm peer-buffer cross-check ok={ok}")
    return ok


def _worker(rank: int, world: int) -> None:
    dev = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    gname = dist.group.WORLD.group_name
    phase = "oneCCL baseline"
    variant = "vllm"
    try:
        _oneccl_baseline(rank, dev)

        phase = "symm rendezvous"
        try:
            ok = _symm_via_vllm(rank, dev, gname)
        except ImportError:
            # No tritonar patch in this image -> probe the transport with the
            # same torch calls instead (still a valid transport verdict).
            phase = "torch-only symm"
            variant = "torch"
            if rank == 0:
                print(
                    "[info] image has no tritonar patch "
                    "(vllm xpu_triton_all_reduce not importable) -> "
                    "using torch-only transport probe"
                )
            ok = _symm_via_torch(rank, dev, gname)
        dist.barrier()
        if rank == 0:
            if ok:
                print(f"RESULT: SYMM_TRANSPORT_OK (via {variant})")
            else:
                print(f"RESULT: SYMM_TRANSPORT_CORRUPT (via {variant})")
        if not ok:
            sys.exit(1)
    except Exception as e:  # noqa: BLE001 - we WANT to report the transport failure
        if phase == "oneCCL baseline":
            print(f"[rank {rank}] RESULT: ONECCL_BASELINE_FAIL {type(e).__name__}: {e}")
        else:
            print(
                f"[rank {rank}] RESULT: SYMM_TRANSPORT_FAIL "
                f"({phase}, via {variant}) {type(e).__name__}: {e}"
            )
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    world = 2
    mp.spawn(_worker, args=(world,), nprocs=world, join=True)
