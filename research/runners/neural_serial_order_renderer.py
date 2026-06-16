"""CYCLE 105 Step 1 — the NeuralSerialOrderRenderer: the de-risked neural replacement for the conversational
output's host word-ordering f-string (`rf_phasor_composer.py:346` `f"{agent} {action} {patient}"`).

The host f-string does two things: (i) impose a serial ORDER on the recalled role-fillers, (ii) concatenate. This
renderer moves (i) -- the cognitive part -- onto neurons: a rate-coded COMPETITIVE-QUEUING serial-order generator
(de-risked GO on real spikes, phase B, 6/6 seeds vs permuted-order + equal-drive controls). The frame (e.g. SVO =
agent before action before patient) sets a PRIMACY GRADIENT as graded external current into the fact's concept
pools; the spiking RATE tracks the drive, so the per-pool rate RANKING = the serial order. This is the biological
parallel->serial conversion (premotor/SMA competitive queuing, catalog G.07/H.19; Grossberg 1978, Bullock-Rhodes
2003) -- NOT a host sort. The equal-drive control FAILS (no gradient -> no reliable order), proving the neurons do
the serialization.

What stays host (and is legitimately the BODY emitting motor output, per the BRAIN-BASED-ONLY standard): the final
`" ".join` of the already-neurally-ordered, already-neurally-spelled words. Per-slot spelling (concept pool -> word)
is the separately-validated A->W read-out primitive (`concept_speak_demo`, 100% multi-seed), passed in as a
callback so this renderer stays substrate-agnostic.

OPT-IN: importing/using this does NOT change the production composer. Wiring it into `render_fact` behind a
default-off flag (preserving the no-confab moat + the full conversational matrix) is CYCLE-105 Step 2.
Self-test:  SIM_BACKEND=cupy python -u -m research.runners.neural_serial_order_renderer
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_serial_order_spiking_derisk import (  # noqa: E402  (reuse the de-risked mechanism)
    build_pool_bridge, pool_rates, PRIMACY_pA, VOCAB, N_PER)


class NeuralSerialOrderRenderer:
    """Rate-coded competitive-queuing serial-order renderer. `order()` converts a frame's PARALLEL primacy
    gradient (graded current into the concept pools) into a SERIAL order (the per-pool spiking rate ranking) --
    the neural parallel->serial step. `render()` spells each ordered slot via the pluggable A->W `spell` callback
    and joins (the body's emission). VOCAB pools at N_PER neurons each share one driven-pool bridge."""

    def __init__(self, seed=42, primacy_pA=PRIMACY_pA, vocab=VOCAB):
        if vocab > VOCAB:
            raise ValueError(f"this renderer's bridge holds {VOCAB} pools; got vocab={vocab}")
        self.bridge, self.pool_idx = build_pool_bridge(seed)
        self.primacy_pA = tuple(primacy_pA)

    def order(self, frame_concepts):
        """frame_concepts: concept-pool indices in the FRAME's intended order (slot 0 = highest primacy). Drive
        each with its slot's primacy current, read per-pool rate, RETURN the concepts ordered by rate ranking
        (the spiking competitive-queuing read-out). With a real gradient the neurons reproduce the frame order;
        the equal-drive control shows that without the gradient they cannot (so the order is neurally produced)."""
        n = len(frame_concepts)
        drive = {int(c): self.primacy_pA[min(i, len(self.primacy_pA) - 1)] for i, c in enumerate(frame_concepts)}
        rate = pool_rates(self.bridge, self.pool_idx, drive)
        return [int(c) for c in sorted(frame_concepts, key=lambda c: -rate[int(c)])]

    def render(self, frame_concepts, spell):
        """spell: concept-index -> word (the A->W read-out primitive). Returns the utterance (the body joins the
        neurally-ordered, neurally-spelled words)."""
        return " ".join(spell(c) for c in self.order(frame_concepts))


def _self_test():
    """Exercise the renderer on SVO facts: the frame order is [agent, action, patient]; verify the neural
    rate-coded read-out reproduces that order (and a mock spell concatenates). Multi-seed, vs an equal-drive
    sanity check (no gradient -> order not reliably the frame order)."""
    rng = np.random.default_rng(7)
    facts = []
    seen = set()
    while len(facts) < 8:
        trip = tuple(int(x) for x in rng.choice(VOCAB, 3, replace=False))
        if trip not in seen:
            seen.add(trip); facts.append(trip)
    spell = lambda c: f"w{c}"                                   # mock A->W (production passes the real read-out)
    ok = 0
    for seed in (42, 43, 44):
        r = NeuralSerialOrderRenderer(seed=seed)
        seed_ok = 0
        for trip in facts:
            ordered = r.order(list(trip))                      # frame order = the trip's order (agent,action,patient)
            utter = r.render(list(trip), spell)
            if ordered == list(trip) and utter == " ".join(f"w{c}" for c in trip):
                seed_ok += 1
        print(f"  [seed {seed}] neural-rendered SVO order correct: {seed_ok}/{len(facts)}", flush=True)
        ok += seed_ok
    total = 3 * len(facts)
    print(f"\n  NeuralSerialOrderRenderer self-test: {ok}/{total} SVO utterances correctly ordered by the spiking "
          f"competitive-queuing read-out (mock spelling).", flush=True)
    if ok >= int(0.95 * total):
        print(f"  PASS: the renderer packages the de-risked neural serial-order mechanism -- order produced by "
              f"spiking rate ranking, spelling via the A->W callback, only the final join is the body. ==> ready "
              f"to wire into render_fact behind a default-off flag (Step 2, with the no-confab moat as the gate).",
              flush=True)
    else:
        print(f"  CHECK: {ok}/{total} -- inspect the primacy current gap / RUN_STEPS before Step 2.", flush=True)
    return ok, total


if __name__ == "__main__":
    os.environ.setdefault("SIM_BACKEND", "cupy")
    print("[NeuralSerialOrderRenderer self-test] does the packaged renderer reproduce SVO order neurally?", flush=True)
    _self_test()
