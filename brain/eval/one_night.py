# Capability: Lifelong learning
# Record: brain/research/2026-11-21-sleep-language-keep-prereg.md (section 2 step-0 gate)
# Expected-fail: none (a harness helper: copies a home and sleeps one night)
"""One sleep night on an already-saved home, under a chosen keep set (K-P step-0 gate,
brain/research/2026-11-21-sleep-language-keep-prereg.md section 2).

    from brain.eval.one_night import one_night
    one_night(src_home, dst_home, keep_bank=None | "prior", cortex=<Cortex>, sleep_seed=S, **life_brain_kw)

The snapshot `src_home` is never touched: it is copied to `dst_home` (which must not exist), the copy is opened with
sleep_on_restart off, sleeps ONE night with seed sleep_seed + sleeps (the same seed for both arms), and is saved.
Run it twice from the same snapshot (keep_bank=None = arm B, keep_bank="prior" = arm K-P) and read both dst homes.
For keep_bank="prior" on a home saved without it, the copy's meta.json is marked first (the explicit switch-on that
Brain refuses to make by itself in a home that already slept), and the bank is sampled once into dst_home/keep_bank.json.
Pass the life's own Brain kwargs (sleep_steps, sleep_lr, sleep_kl_weight, adapter_rank, inquiry, ...) as brain_kw.
"""
from __future__ import annotations

import json
import os
import shutil

from brain.brain import Brain


def one_night(src_home: str, dst_home: str, keep_bank: str | None = None, cortex=None, sleep_seed: int = 0, **brain_kw) -> dict:
    if os.path.exists(dst_home):
        raise FileExistsError(dst_home)
    shutil.copytree(src_home, dst_home, ignore=shutil.ignore_patterns(".lock"))
    mp = os.path.join(dst_home, "meta.json")
    with open(mp) as fh:
        meta = json.load(fh)
    if keep_bank is None:
        meta.pop("keep_bank", None)
    else:
        meta["keep_bank"] = keep_bank
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=1)
    log: list = []
    b = Brain(dst_home, cortex=cortex, sleep_seed=sleep_seed, keep_bank=keep_bank, sleep_log=log.append,
              **{**brain_kw, "sleep_on_restart": False})
    try:
        losses = b.sleep()
        b.save()
        return {"home": dst_home, "night": b.meta["sleeps"], "keep_bank": keep_bank, "keep_record": b.last_keep_record,
                "mean_replay": sum(r for r, _ in losses) / len(losses) if losses else None,
                "mean_keep": sum(k for _, k in losses) / len(losses) if losses else None}
    finally:
        b.release()
