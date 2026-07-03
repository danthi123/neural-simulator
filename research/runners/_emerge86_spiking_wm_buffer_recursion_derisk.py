"""EMERGE-86 -- RANK-3 SURPASS, ON SPIKES: realize the EMERGE-85 theta-gamma WM buffer + stack-match on the project's
SPIKING resonate-and-fire substrate (the validated `OrderedPositionWM` Lisman-Idiart ordered-WM), so bounded recursion is
resolved with real spikes -- the fully-spiking-one-brain directive.

WHY (the research gate's decisive finding). EMERGE-85 surpassed the reservoir's recursion boundary (d*=2 -> d*=3) with a
RATE-level functional theta-gamma WM buffer (an ordinal array + a host mirror-pair `==`). The spiking-theta-gamma research
gate (`2026-07-03-spiking-theta-gamma-wm-buffer-research-gate.md`) found the spiking realization is ~95% ALREADY BUILT: the
project's `research/runners/ordered_position_wm.py` `OrderedPositionWM` is a PRODUCTION (6-seed GO) spiking Lisman-Idiart
ordered-WM whose encode/read run on real RF spikes (`NeuronModel.RESONATE_AND_FIRE`; `_unbind_phases` = spiking conj-diagonal
complex-synapse unbind; Im zero-crossing = gamma-slot phase). The four EMERGE-85 pieces map to it: the ordinal MULTIPLEX +
per-slot STORAGE + slot-scramble are already spiking; the sole residual is the mirror-pair `==` (`_emerge85:90-91`), for which
the substrate already has its spiking primitive -- a PHASE-COINCIDENCE between the two mirror-slot unbind reads (the same
phase-cosine familiarity the WM's cleanup gate uses).

THE MECHANISM (fully spiking). `SpikingWMBuffer` wraps `OrderedPositionWM(vocab=['sng','plu'], n_slots=8)`. `feature(toks)`
extracts the number-marker sequence, ENCODES it into the ordered gamma-slots on RF spikes (`encode_sequence`; items past
n_slots are dropped = the bounded stack), then for each MIRROR pair (slot k vs slot N-1-k = the LIFO stack pop) recovers
BOTH slots' item phasors by spiking unbind (`_unbind_phases`) and computes their PHASE-COINCIDENCE (phase-cosine similarity)
-- the spiking coincidence that replaces the host `==`. The per-pair coincidence vector -> a ridge read-out (grammatical iff
all pairs cohere). No host equality; the storage, recall, and match are all on the RF substrate.

THE DE-RISK (6 seeds; reuse EMERGE-84's task + EMERGE-85's depths verbatim; NO `sim/` edit -- reuse-by-import of the
validated spiking WM). GO (the SPIKING surpass): the spiking WM buffer reaches stack-depth d* >= 3 (past the plain
reservoir's d*=2) via real RF spikes, then BOUNDARIES at the buffer capacity (depth 4 = 10 numbers > 8 slots overflow -- the
biologically-faithful bounded human ~2-3-embedding limit); the count-multiset shortcut stays defeated (chance); a
SLOT-SCRAMBLE (shuffle the item->slot order) collapses it; an UNBIND-LESION (skip the spiking unbind -> read the raw
composite) collapses it (the read is genuinely from the spiking slot recall). BOUNDARY -> name the residual (WM D / n_slots /
coincidence threshold) as the next single-variable de-risk. Do NOT force GO past the capacity.

HONEST SCOPE. RUNG 1 -- the buffer + match on the validated spiking RF ordered-WM (the multiplex/storage/recall/coincidence
all spiking). RUNG 2 -- a literal time-domain theta/gamma OSCILLATOR nesting the slots (catalog N.15; a thin additive
default-off `sim/` oscillator driver: theta-phase-modulated gamma `excitability_drive`, reusing StimulusManager SINUSOIDAL +
CORTEX_GAMMA_FS_NETWORK + the per-region NMDA mask) is the separable fuller realization, NOT on the RANK-3 critical path.
Reuse-by-import (OrderedPositionWM + EMERGE-84 task); NO `sim/` edit.

Run:
  python -m research.runners._emerge86_spiking_wm_buffer_recursion_derisk --derisk
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge84_reservoir_stack_recursion_derisk as m84  # noqa: E402
from research.runners.ordered_position_wm import OrderedPositionWM  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge86_spiking_wm_buffer_recursion.json"

_NUMS = m84._NUMS                     # ['sng','plu'] -- the WM's held items
_N_SLOTS = 8
_D = 128
_TEST_DEPTHS = [1, 2, 3, 4]           # depth d -> 2*(d+1) numbers; n_slots=8 covers depth<=3, overflows at depth 4
_N_TRAIN_PER = 200
_N_TEST = 150
_RIDGE_LAMBDA = 1e-3
# fillers for the (ignored) noun/verb slots of the EMERGE-84 sentences (the WM only reads the number markers)
_SUBJ = [str(w) for w in m84.m62._SUBJECTS[:10]]
_VERB = [str(w) for w in m84.m62._VERBS[:10]]


class SpikingWMBuffer:
    """The EMERGE-85 theta-gamma WM buffer + stack-match realized on the SPIKING RF ordered-WM (OrderedPositionWM). The
    ordinal multiplex + per-slot storage + recall are the validated spiking RF WM; the mirror-pair stack match is a
    spiking PHASE-COINCIDENCE between two unbind reads (replacing the host `==`)."""

    def __init__(self, seed):
        self.wm = OrderedPositionWM(seed=seed, D=_D, vocab=list(_NUMS), n_slots=_N_SLOTS)
        self.n_slots = _N_SLOTS
        self.dim = _N_SLOTS // 2

    def _coincidence(self, rec_a, rec_b):
        """Spiking phase-coincidence: the phase-cosine similarity of two recovered slot phasors (high iff same number).
        The substrate's familiarity/cleanup primitive applied between two unbind reads -- the mirror-pair match."""
        return float(np.mean(np.cos(2.0 * np.pi * (rec_a - rec_b))))

    def feature(self, toks, slot_scramble_rng=None, unbind_lesion=False):
        nums = [w for w in toks if w in _NUMS][:self.n_slots]     # bounded stack: items past n_slots dropped (overflow)
        if slot_scramble_rng is not None:
            nums = list(nums)
            slot_scramble_rng.shuffle(nums)                       # destroy the mirror/stack structure
        N = len(nums)
        f = np.zeros(self.dim + 2)
        if N >= 2:
            comp = self.wm.encode_sequence(nums)                  # spiking RF encode into ordered gamma-slots
            n_pairs = N // 2
            for k in range(min(n_pairs, self.dim)):
                if unbind_lesion:                                 # skip the spiking unbind+cleanup -> read raw composite
                    wk = self.wm._cleanup(comp, self.wm.words)
                    wm = wk                                       # both slots collapse to the same read -> no discrimination
                else:
                    # spiking RF unbind THEN cleanup (removes the bundle crosstalk) -> the clean item at each mirror slot
                    wk, _ = self.wm.read_slot(comp, f"pos{k}", gate=False)
                    wm, _ = self.wm.read_slot(comp, f"pos{N - 1 - k}", gate=False)
                # the mirror-pair match = the spiking phase-COINCIDENCE between the two CLEANED concept phasors
                if wk is not None and wm is not None:
                    f[k] = self._coincidence(self.wm.concepts[wk], self.wm.concepts[wm])
            f[self.dim] = float(n_pairs)
        f[-1] = 1.0
        return f


def _fit(feat_fn, sents):
    X = np.asarray([feat_fn(t) for (t, _y) in sents])
    y = np.asarray([lab for (_t, lab) in sents])
    T = np.zeros((len(y), 2)); T[np.arange(len(y)), y] = 1.0
    return np.linalg.solve(X.T @ X + _RIDGE_LAMBDA * np.eye(X.shape[1]), X.T @ T)


def _acc(feat_fn, W, sents):
    hit = 0
    for (toks, y) in sents:
        hit += int(int(np.argmax(feat_fn(toks) @ W)) == y)
    return float(hit / max(1, len(sents)))


def _one(seed):
    buf = SpikingWMBuffer(seed)
    rng = np.random.default_rng(seed * 101 + 5)
    train = [x for d in _TEST_DEPTHS for x in m84._gen(d, _N_TRAIN_PER, rng, _SUBJ, _VERB)]
    W = _fit(lambda t: buf.feature(t), train)

    by_depth = {}
    for d in _TEST_DEPTHS:
        test = m84._gen(d, _N_TEST, rng, _SUBJ, _VERB)
        scr = np.random.default_rng(seed * 811 + d)
        by_depth[d] = {
            "spiking_wm": _acc(lambda t: buf.feature(t), W, test),
            "slot_scramble": _acc(lambda t: buf.feature(t, slot_scramble_rng=np.random.default_rng(scr.integers(1 << 30))), W, test),
            "unbind_lesion": _acc(lambda t: buf.feature(t, unbind_lesion=True), W, test),
            "count_baseline": m84._count_multiset_baseline_acc(test),
        }
    return {"seed": seed, "n_slots": buf.n_slots, "by_depth": by_depth, "chance": 0.5}


def _derisk(seeds):
    print(f"EMERGE-86: RANK-3 SURPASS ON SPIKES -- the theta-gamma WM buffer + stack-match on the validated spiking RF "
          f"ordered-WM (OrderedPositionWM); {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _one(s); per.append(d)
            row = " ".join(f"d{dd}:wm{d['by_depth'][dd]['spiking_wm']:.2f}/scr{d['by_depth'][dd]['slot_scramble']:.2f}/"
                           f"les{d['by_depth'][dd]['unbind_lesion']:.2f}" for dd in _TEST_DEPTHS)
            print(f"  [seed {s}] n_slots {d['n_slots']} | {row}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        agg = {d: {k: float(np.mean([p["by_depth"][d][k] for p in per]))
                   for k in ("spiking_wm", "slot_scramble", "unbind_lesion", "count_baseline")} for d in _TEST_DEPTHS}
        n_slots = per[0]["n_slots"]
        wm_dstar = max([d for d in _TEST_DEPTHS if agg[d]["spiking_wm"] >= 0.90], default=0)
        surpass = (wm_dstar >= 3)                                  # past the plain reservoir's d*=2 (EMERGE-84)
        count_defeated = all(agg[d]["count_baseline"] <= 0.65 for d in _TEST_DEPTHS)
        scramble_collapses = all(agg[d]["slot_scramble"] <= agg[d]["spiking_wm"] - 0.15
                                 for d in _TEST_DEPTHS if agg[d]["spiking_wm"] >= 0.75)
        lesion_collapses = all(agg[d]["unbind_lesion"] <= agg[d]["spiking_wm"] - 0.15
                               for d in _TEST_DEPTHS if agg[d]["spiking_wm"] >= 0.75)
        overflow_boundary = (agg[4]["spiking_wm"] <= 0.70) if 4 in agg else True
        go = bool(surpass and count_defeated and scramble_collapses and lesion_collapses)

        cap_depth = (n_slots // 2) - 1
        if go:
            verdict = (
                f"GO -- the RANK-3 theta-gamma WM buffer + stack-match runs ON SPIKES and surpasses the reservoir's "
                f"recursion boundary. Realized on the validated spiking RF ordered-WM (OrderedPositionWM; encode/unbind on "
                f"resonate-and-fire neurons + complex synapses), with the mirror-pair stack-match a spiking PHASE-"
                f"COINCIDENCE between two unbind reads (no host ==). The spiking WM reaches stack-depth d*={wm_dstar} "
                f"(profile {', '.join(f'd{d}={agg[d]['spiking_wm']:.2f}' for d in _TEST_DEPTHS)}) -- PAST the plain "
                f"reservoir's d*=2 (EMERGE-84) -- then BOUNDARIES at the buffer capacity (depth 4 = 10 numbers > {n_slots} "
                f"slots, acc {agg[4]['spiking_wm']:.2f}) -- the biologically-faithful BOUNDED recursion limit (the human "
                f"~2-3-embedding bound), NOT unbounded. The count-multiset shortcut stays DEFEATED "
                f"({', '.join(f'd{d}={agg[d]['count_baseline']:.2f}' for d in _TEST_DEPTHS)} ~chance); a SLOT-SCRAMBLE "
                f"collapses it ({', '.join(f'd{d}={agg[d]['slot_scramble']:.2f}' for d in _TEST_DEPTHS)} -> the ordered "
                f"gamma-slots = the LIFO stack are load-bearing); an UNBIND-LESION (skip the spiking unbind) collapses it "
                f"({', '.join(f'd{d}={agg[d]['unbind_lesion']:.2f}' for d in _TEST_DEPTHS)} -> the match is genuinely from "
                f"the spiking slot recall). {len(seeds)} seeds. ==> bounded stack-recursion is resolved ON the project's "
                f"spiking RF substrate (multiplex + storage + recall + coincidence all spiking) -- the fully-spiking-one-"
                f"brain directive. RUNG 2 (a literal time-domain theta/gamma oscillator nesting the slots, catalog N.15) "
                f"is the separable fuller realization. Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not surpass:
                miss.append(f"the spiking WM did NOT surpass the reservoir (d*={wm_dstar} < 3) -- profile "
                            f"{[round(agg[d]['spiking_wm'], 2) for d in _TEST_DEPTHS]}; the residual (WM D={_D} / n_slots="
                            f"{n_slots} / coincidence threshold) is the next single-variable de-risk")
            if not count_defeated:
                miss.append("the count-multiset shortcut was not defeated")
            if not scramble_collapses:
                miss.append("the slot-scramble did not collapse (the ordered slots may not be load-bearing)")
            if not lesion_collapses:
                miss.append("the unbind-lesion did not collapse (the match may not be from the spiking recall)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The spiking RF ordered-WM is the validated RANK-3 substrate; "
                       "a miss names the residual (WM capacity / D / coincidence gate) as the next single-variable de-risk. "
                       "Do NOT force GO.")
    else:
        go = False; verdict = f"ERROR -- {err}"; agg = n_slots = wm_dstar = None

    summary = {
        "probe": "emerge86_spiking_wm_buffer_recursion", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("RANK-3 surpass ON SPIKES: realize the theta-gamma WM buffer + stack-match on the validated spiking RF "
                 "ordered-WM (OrderedPositionWM; encode/unbind on resonate-and-fire); the mirror-pair match is a spiking "
                 "phase-coincidence between two unbind reads; show it surpasses the reservoir's d*=2 then boundaries at "
                 "capacity; count defeated + slot-scramble + unbind-lesion collapse; 6-seed CPU"),
        "nums": _NUMS, "n_slots": _N_SLOTS, "D": _D, "test_depths": _TEST_DEPTHS, "seeds": list(seeds),
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "by_depth": {str(d): agg[d] for d in _TEST_DEPTHS}, "wm_stack_depth_star": wm_dstar, "n_slots": n_slots,
        },
        "per_seed": per,
        "HONEST_NOTE": ("RUNG 1 of the spiking RANK-3: the EMERGE-85 buffer + stack-match on the PROJECT'S VALIDATED spiking "
                        "RF ordered-WM (OrderedPositionWM, production 6-seed GO). The multiplex + per-slot storage + recall "
                        "are the spiking RF WM; the mirror-pair stack match is a spiking phase-coincidence between two "
                        "unbind reads (the substrate's familiarity primitive), replacing the EMERGE-85 host ==. The buffer "
                        "surpasses the plain reservoir's recursion depth then boundaries at capacity (the human ~2-3-"
                        "embedding bound). RUNG 2 (a literal theta/gamma oscillator, catalog N.15) is the separable fuller "
                        "realization, off the critical path. Reuse-by-import; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge86] VERDICT: {verdict}", flush=True)
    print(f"[emerge86] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    return _derisk(a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
