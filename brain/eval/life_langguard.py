# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (an instrument)
"""The --lang-guard read-out (design section 5): KL and NLL of the adapters-on cortex vs the pretrained cortex on the frozen
held-out set brain/eval/life_heldout.py, after every trunk night."""
from __future__ import annotations

import os
import shutil
import time

import torch
import torch.nn.functional as F

from brain.eval.life_heldout import HELDOUT


def measure(cortex, base, texts=None) -> dict:
    """KL (adapters-on vs pretrained, the direction and form of sleep.py's keep term) and per-token NLL under both, mean
    over the texts. Read-only: plastic.enabled and the grad mode are restored, no RNG is drawn, `base` is not mutated."""
    texts = HELDOUT if texts is None else texts
    p = cortex.plastic
    if p is None:
        return {"kl": 0.0, "nll_on": None, "nll_off": None, "n_texts": len(texts), "seconds": 0.0, "no_adapters": True}
    t0 = time.time()
    was = p.enabled
    kls, non, noff = [], [], []
    try:
        with torch.no_grad():
            for t in texts:
                ids = cortex.pipe.encode(t)
                tgt = torch.tensor(ids[1:])
                p.enabled = False
                off = F.log_softmax(p(ids, base)[0].float(), -1)
                p.enabled = True
                on = F.log_softmax(p(ids, base)[0].float(), -1)
                kls.append(float(F.kl_div(on, off, log_target=True, reduction="batchmean")))
                non.append(float(F.nll_loss(on[:-1], tgt.to(on.device))))
                noff.append(float(F.nll_loss(off[:-1], tgt.to(off.device))))
    finally:
        p.enabled = was
    m = lambda v: round(sum(v) / len(v), 6)
    return {"kl": m(kls), "nll_on": m(non), "nll_off": m(noff), "n_texts": len(texts),
            "seconds": round(time.time() - t0, 1)}


def night_read(home: str, cortex, night: int, sim, brain_kw: dict, work: str, weekend: bool = False) -> dict:
    """The language guard of one trunk night: measure() on a disk copy of the slept trunk home (opened like a probe copy,
    sleep_on_restart=False), so the trunk, the shared cortex's adapter switch and the lived days are untouched. The copy
    is opened in its own world-clock morning and the global RNG state is restored afterwards."""
    import random
    import numpy as np
    from brain.brain import Brain
    from brain.eval import life
    cp = os.path.join(work, f"langguard_L{night}")
    shutil.rmtree(cp, ignore_errors=True)
    shutil.copytree(home, cp)
    sim.to(life.day_start(night + 1, weekend))
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    b = Brain(cp, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
    try:
        assert not b.pending(), "the trunk has an unslept day: the language guard must read a slept home"
        return measure(b.cortex, [x.float() for x in b.base])
    finally:
        b.release()
        shutil.rmtree(cp, ignore_errors=True)
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
