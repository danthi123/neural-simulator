"""Sleep: replay hippocampal episodes into the cortex's slow synapses (complementary learning systems).

Interleaved replay (record brief: research/findings/2026-08-26-cls-interleaved-consolidation-6seed-GO-...): every
replayed batch mixes stored episodes with ordinary experience, and a KL term holds ordinary behaviour to the
pretrained cortex, so learning new facts does not overwrite what the cortex already does. Episodes replay in
proportion to their strength (use, salience), so often-used memories consolidate most.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def _ids(cortex, text):
    return cortex.pipe.encode(text)


def sleep(plastic, base_state, episodes, generic, *, steps=200, lr=5e-4, kl_weight=1.0, seed=0, log=None):
    """episodes: list of (text, strength). generic: ordinary conversation texts. Returns the per-step losses."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    c = plastic.cortex
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    weights = [max(s, 1e-3) for _, s in episodes]
    gen_ids = [_ids(c, g) for g in generic]
    # the pretrained cortex's behaviour on ordinary text, held as the protection target
    plastic.enabled = False
    with torch.no_grad():
        ref = [F.log_softmax(plastic(ids, base_state)[0], -1) for ids in gen_ids]
    plastic.enabled = True
    losses = []
    for step in range(steps):
        text, _ = rng.choices(episodes, weights=weights)[0]
        ids = _ids(c, text)
        logits, _ = plastic(ids, base_state)
        replay = F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=logits.device))
        j = rng.randrange(len(gen_ids))
        g_logits, _ = plastic(gen_ids[j], base_state)
        keep = F.kl_div(F.log_softmax(g_logits, -1), ref[j], log_target=True, reduction="batchmean")
        loss = replay + kl_weight * keep
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append((float(replay), float(keep)))
        if log and step % 20 == 0:
            log(f"sleep step {step}: replay {float(replay):.3f} keep {float(keep):.4f}")
    return losses
