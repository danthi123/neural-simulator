"""The night as its own process: load a brain home, replay what was lived since its last sleep, save, exit.

python -m brain.night --home DIR [--model 0.1b|0.4b|1.5b|PATH] [--device cuda] [--steps N --lr X --kl-weight K]

Same semantics as "restart = night" (brain/brain.py): nothing pending -> nothing happens. It runs as a separate GPU
job (adopt-1.5b plan step 7) so chat starts without a blocking sleep; queue it with `bash tools/brain_gpu.sh night`.
Holds the home's lock, so a chat cannot live the same brain while it sleeps (and vice versa). Prints one JSON line:
episodes replayed, seconds, the card's peak memory (torch.cuda.max_memory_allocated) and the memory accounting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from brain.brain import Brain


def run(home: str, model: str | None = None, device: str = "cpu", cortex=None, force: bool = False,
        log=None, **sleep_kw) -> dict:
    if not os.path.exists(os.path.join(home, "meta.json")):
        raise FileNotFoundError(f"{home}: no brain lives here (a night needs a lived home)")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    b = Brain(home, device=device, cortex=cortex, cortex_model=model, sleep_on_restart=False, lock=True,
              sleep_log=log, **{k: v for k, v in sleep_kw.items() if v is not None})
    try:
        pending = len(b.pending())
        slept = False
        if pending or force:
            b.sleep()
            slept = True
            b.save()
        out = {"home": home, "cortex_model": b.cortex_model, "pending": pending, "slept": slept,
               "night": b.meta.get("sleeps", 0), "sleep_cfg": b.sleep_cfg, "seconds": round(time.time() - t0, 1),
               "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
               "memory": b.memory_report()}
    finally:
        b.release()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default=None, help="checkpoint path or size tag; default: the default cortex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--kl-weight", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="sleep even with nothing pending")
    a = ap.parse_args(argv)
    out = run(a.home, a.model, a.device, force=a.force, log=lambda m: print(m, file=sys.stderr, flush=True),
              sleep_steps=a.steps, sleep_lr=a.lr, sleep_kl_weight=a.kl_weight)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
