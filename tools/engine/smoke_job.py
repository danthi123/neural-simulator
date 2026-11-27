"""The engine smoke job (runs ON a node, no model): import the named modules, record provenance, count its own launches.

    python -m tools.engine.smoke_job --seed 1 --out OUT --imports torch,rwkv --sleep 20

Host guard: refuses unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1 (set by the engine's job script on pool / w3090 nodes) or
ENGINE_LOCAL_TEST=1 (the local test backend). The launch counter lives one level above the attempt dir (the smoke's own jobs
dir, cleared at the start of each smoke), so a duplicated launch shows as count 2. Written atomically; exits 0 even when an
import fails (the failure is a recorded field, read by the smoke verdict).
"""
import argparse
import importlib
import json
import os
import socket
import sys
import time


def guard(env=None):
    env = os.environ if env is None else env
    if not any(env.get(k) == "1" for k in ("POOL_NODE", "BRAIN_ALLOW_W3090", "ENGINE_LOCAL_TEST")):
        raise SystemExit("smoke_job: refuses to run unless POOL_NODE=1, BRAIN_ALLOW_W3090=1 or ENGINE_LOCAL_TEST=1")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imports", default="torch,rwkv")
    ap.add_argument("--sleep", type=float, default=0)
    a = ap.parse_args(argv)
    guard()
    t0 = time.time()
    cnt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.out))), f"count_s{a.seed}")
    with open(cnt, "a") as fh:
        fh.write(f"{os.getpid()}\n")
    count = len(open(cnt).read().split())
    mods = {}
    for m in [x for x in a.imports.split(",") if x]:
        try:
            mod = importlib.import_module(m)
            mods[m] = {"ok": True, "version": str(getattr(mod, "__version__", "?"))}
        except Exception as e:                                    # recorded, not raised: the verdict names the node
            mods[m] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    try:
        cg = open("/proc/self/cgroup").read().strip().splitlines()[-1]
    except OSError:
        cg = None
    time.sleep(a.sleep)                                           # long enough for the driver to crash and reattach
    torch_info = {}
    if mods.get("torch", {}).get("ok"):
        import torch
        torch_info = {"threads": torch.get_num_threads(), "cuda": bool(torch.cuda.is_available())}
    rec = {"seed": a.seed, "host": socket.gethostname(), "host_env": os.environ.get("HOST"), "commit": os.environ.get("COMMIT"),
           "device": os.environ.get("BRAIN_DEVICE", "cpu"), "precision": "n/a (no model)", "python": sys.version.split()[0],
           "imports": mods, "torch": torch_info, "count": count, "cgroup": cg,
           "model_dir": os.environ.get("BRAIN_MODEL_DIR"), "wall_s": round(time.time() - t0, 2)}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
