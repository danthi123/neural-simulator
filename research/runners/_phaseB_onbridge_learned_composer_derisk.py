"""On-bridge binding build STEP 2 (cheap-first, NO protected sim/ edit) — does the on-bridge LEARNED spiking binder
do real CONVERSATIONAL store / who-what Q&A / abstention (the no-confab moat), not just systematicity recall?

STEP 1 (`_phaseB_onbridge_frlf_bundle_derisk`, 6-seed GO 0.969) proved the fixed-role + learned-filler BUNDLED bind
survives real LIF spiking at D_h=256. STEP 2 asks the production question: wrap that on-bridge binding in the
composer's conversational API -- store an SVO fact as a spiking-read bundled composite, answer "what does <agent>
<action>?" / "who <action> <patient>?" by unbinding the cued roles + cleanup, and ABSTAIN (return None) when no
stored fact matches -- exactly the RFPhasorComposer interface, but with the LEARNED spiking binder instead of the
fixed exact-inverse phasor algebra. If who/what recall is high AND the moat holds (never-stored cue -> abstain,
permuted cue -> no false match), the learned spiking binder is conversation-capable; the full production wiring
(synaptic read-out, conversation-trained weights) is the next step.

THE BIND IS ON-BRIDGE: store() drives the analog 3-way bundle onto the LIF ON/OFF populations and keeps the
spiking-read (rate) composite; query unbinds from that spiking composite. The read-out (the learned W_O cleanup)
is numpy -- the SAME scaffold/fast-path split as STEP 1 and the production composer's numpy cleanup.

GATE (3 seeds, escalate to 6): who+what recall >> chance AND the moat holds 6/6 (never-stored cue abstains;
permuted cue does not false-match a stored answer). Reuse-by-import (STEP-1 LIF substrate + FixedRoleLearnedFiller
Binder + systematicity codes). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_learned_composer_derisk [--dh 256] [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.cortex_learned_binder_systematicity_probe import make_role_codes, native_argmax  # noqa: E402
from research.runners._phaseB_fixed_role_learned_filler_bundling_derisk import FixedRoleLearnedFillerBinder  # noqa: E402
import research.runners._phaseB_onbridge_bind_nonlinearity_derisk as onb  # noqa: E402

F = 16                 # the shared concept pool (agents/actions/patients drawn from it)
N_TRAIN_STEPS = 24000  # bundle-aware read-out training (== the A/B)
N_FACTS = 5            # SVO facts in the KB
LR = 0.005


class OnBridgeLearnedComposer:
    """The on-bridge learned spiking binder wrapped in the composer conversational API (store / query_patient /
    query_agent / the moat). The bind/bundle is real LIF spiking (ON/OFF populations); the read-out is the learned
    W_O cleanup (numpy fast path, as STEP 1 + the production composer)."""

    def __init__(self, codes, seed, D_h, bridge, pos_idx, neg_idx):
        self.seed = int(seed)
        self.fillers = codes[:F]
        self.D_in = self.fillers.shape[1]
        self.D_h = D_h
        self.bridge, self.pos_idx, self.neg_idx = bridge, pos_idx, neg_idx
        roles = make_role_codes(3, self.D_in, seed)                          # agent / action / patient
        rng_pm1 = np.random.default_rng(seed * 31 + 5)
        R_proj = rng_pm1.standard_normal((self.D_in, D_h)) / np.sqrt(self.D_in)
        self.role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)           # [3, D_h] fixed +-1 self-inverse
        self.binder = FixedRoleLearnedFillerBinder(D_in=self.D_in, role_pm1=self.role_pm1, D_h=D_h, lr=LR,
                                                   lam=1e-4, seed=seed)
        self._train()
        self._calibrate()
        self.kb = []                                                         # (fact_tuple, spiking composite)

    def _train(self):
        """Bundle-aware read-out training over RANDOM SVO triples (full concept coverage -> the read-out recovers
        ANY concept in ANY role, the composer's requirement)."""
        rng = np.random.default_rng(self.seed * 999 + 1)
        for _ in range(N_TRAIN_STEPS):
            ids = [int(x) for x in rng.choice(F, 3, replace=False)]
            self.binder.train_fact_step([0, 1, 2], ids, self.fillers, int(rng.integers(3)))

    def _analog_bundle(self, ids):
        return sum(self.role_pm1[r] * (self.fillers[ids[r]] @ self.binder.W_F) for r in range(3))

    def _calibrate(self):
        rng = np.random.default_rng(13)
        demo = self._analog_bundle([int(x) for x in rng.choice(F, 3, replace=False)])
        np_mag = float(np.mean(np.abs(demo)) + 1e-9)
        lif = onb.lif_onoff(self.bridge, self.pos_idx, self.neg_idx, demo, onb.DRIVE_SCALE)
        self.cal = np_mag / (float(np.mean(lif)) + 1e-9)

    def _spiking_composite(self, ids):
        """Drive the analog 3-way bundle onto the LIF ON/OFF populations; return the spiking-read signed composite."""
        analog = self._analog_bundle(ids)
        lif = onb.lif_onoff(self.bridge, self.pos_idx, self.neg_idx, analog, onb.DRIVE_SCALE) * self.cal
        return lif[:self.D_h] - lif[self.D_h:]

    def store(self, agent, action, patient):
        self.kb.append(((agent, action, patient), self._spiking_composite([agent, action, patient])))

    def _unbind_concept(self, comp, role):
        return int(native_argmax(self.binder.unbind(comp, role), self.fillers))   # role in {0:agent,1:action,2:patient}

    def query_patient(self, agent, action):
        """'what does <agent> <action>?' -> patient concept, or None (abstain) if no stored fact matches the cue."""
        for _fact, comp in self.kb:
            if self._unbind_concept(comp, 0) == agent and self._unbind_concept(comp, 1) == action:
                return self._unbind_concept(comp, 2)
        return None

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> agent concept, or None (abstain)."""
        for _fact, comp in self.kb:
            if self._unbind_concept(comp, 1) == action and self._unbind_concept(comp, 2) == patient:
                return self._unbind_concept(comp, 0)
        return None


def run_seed(codes, seed, D_h):
    bridge, pos_idx, neg_idx = onb.build_bind_bridge(D_h, seed)
    comp = OnBridgeLearnedComposer(codes, seed, D_h, bridge, pos_idx, neg_idx)
    rng = np.random.default_rng(seed * 101 + 7)
    # N_FACTS facts with DISTINCT agents and distinct (agent,action) cues (so who/what have unique answers)
    agents = list(rng.choice(F, N_FACTS, replace=False))
    facts = []
    used = set()
    for a in agents:
        while True:
            ac = int(rng.integers(F)); p = int(rng.integers(F))
            if (a, ac) not in used and len({int(a), ac, p}) == 3:
                used.add((int(a), ac)); facts.append((int(a), ac, p)); break
    for (a, ac, p) in facts:
        comp.store(a, ac, p)

    what_ok = sum(int(comp.query_patient(a, ac) == p) for (a, ac, p) in facts)
    who_ok = sum(int(comp.query_agent(ac, p) == a) for (a, ac, p) in facts)
    # MOAT 1 — a never-stored (agent, action) cue must abstain (None).
    stored_cues = {(a, ac) for (a, ac, p) in facts}
    moat_abstain = moat_n = 0
    for _ in range(20):
        a = int(rng.integers(F)); ac = int(rng.integers(F))
        if (a, ac) in stored_cues:
            continue
        moat_n += 1
        moat_abstain += int(comp.query_patient(a, ac) is None)
    # MOAT 2 — permuted cue: query each fact's agent with a DIFFERENT fact's action -> must NOT return that
    # other fact's patient as a confident answer (no false bind). Counts as clean if None or != the wrong patient.
    perm_clean = 0
    for i, (a, ac, p) in enumerate(facts):
        wrong = facts[(i + 1) % len(facts)]
        ans = comp.query_patient(a, wrong[1])              # agent a, but another fact's action
        perm_clean += int(ans is None or ans != wrong[2])
    row = {"seed": seed, "what": what_ok / N_FACTS, "who": who_ok / N_FACTS,
           "moat_abstain": moat_abstain / max(moat_n, 1), "moat_n": moat_n,
           "perm_clean": perm_clean / len(facts)}
    print(f"  [seed {seed} D_h={D_h}] what {row['what']:.2f} | who {row['who']:.2f} | "
          f"moat-abstain {row['moat_abstain']:.2f} ({moat_abstain}/{moat_n}) | perm-clean {row['perm_clean']:.2f}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_phaseB_onbridge_learned_composer.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[on-bridge learned composer de-risk] does the on-bridge LEARNED spiking binder do who/what Q&A + the "
          f"no-confab moat at D_h={args.dh}? seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, args.dh) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    what, who, moat, perm = m("what"), m("who"), m("moat_abstain"), m("perm_clean")
    chance = 1.0 / F
    n_moat = sum(int(r["moat_abstain"] >= 0.99) for r in rows)
    n_perm = sum(int(r["perm_clean"] >= 0.99) for r in rows)
    print(f"\n{'='*98}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds, D_h={args.dh}): what {what:.3f} | who {who:.3f} | "
          f"moat-abstain {moat:.3f} ({n_moat}/{len(seeds)} clean) | perm-clean {perm:.3f} ({n_perm}/{len(seeds)}) "
          f"| chance {chance:.3f}", flush=True)
    go = what >= 0.80 and who >= 0.80 and n_moat == len(seeds) and n_perm == len(seeds)
    if go:
        print(f"  GO: the on-bridge LEARNED spiking binder is CONVERSATION-CAPABLE -- who/what recall "
              f"{who:.2f}/{what:.2f} >> chance {chance:.2f}, the no-confab moat holds (never-stored cue abstains "
              f"{n_moat}/{len(seeds)}, permuted cue clean {n_perm}/{len(seeds)}). The learned binder does real "
              f"store/Q&A/abstention on real spikes. ==> the idealized exact-inverse algebra is replaceable by the "
              f"learned spiking binder; the production wiring (synaptic read-out + conversation training) is next.",
              flush=True)
    elif what >= 0.6 and who >= 0.6 and n_moat == len(seeds):
        print(f"  PARTIAL: who/what recall works ({who:.2f}/{what:.2f}) + moat holds, but recall is below 0.80 -- "
              f"localize (more read-out training / population size / D_h).", flush=True)
    else:
        print(f"  NEGATIVE: the on-bridge learned composer does not cleanly do Q&A+moat (what {what:.2f} who "
              f"{who:.2f} moat {n_moat}/{len(seeds)}) -- localize before the production wiring.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}", flush=True)
    out = {"verdict": "GO" if go else ("PARTIAL" if (what >= 0.6 and who >= 0.6 and n_moat == len(seeds)) else "NEGATIVE"),
           "D_h": args.dh, "seeds": seeds, "what": what, "who": who, "moat_abstain": moat, "perm_clean": perm,
           "n_moat_clean": n_moat, "n_perm_clean": n_perm, "chance": chance, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
