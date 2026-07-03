"""EMERGE-80 -- the RANK-1.5 SPIKING-LSM PORT: realize the EMERGE-78/79 form->role RESERVOIR on the project's SPIKING
neuron model (a recurrent Izhikevich liquid-state machine), replacing the rate `tanh` echo-state pool -- the fully-spiking
one-brain directive, on pre-registered EMERGE-6b ground.

WHY (the RANK-2 scoping verdict). EMERGE-78/79 established the fronto-striatal reservoir as a LEARNED, genuinely non-local
replacement for the hand form->role labeler -- but at RATE level (a numpy `tanh` echo-state pool). The RANK-2 scoping
(`2026-07-03-rank2-production-reservoir-residual-scoping.md`) found the production side already self-organized (whack-a-mole
to add a "production reservoir") and named the highest-leverage next move: PORT the already-GO rate reservoir onto the
SPIKING substrate -- the non-negotiable fully-spiking end state, pre-registered by the EMERGE-6b "reservoir + trained
read-out" gate. This is a genuine LIQUID-STATE MACHINE (Maass 2002): a recurrent pool of SPIKING neurons (the project's
Izhikevich 2007 model + fixed-random recurrent connectivity) driven by the discovered closed-class configuration; a trained
ridge read-out over the pool's population activity maps the whole-sequence spiking state -> per-slot thematic role.

THE MECHANISM. `SpikingLSM` mirrors the EMERGE-78 `Reservoir` API (a `final_state(U)` returning the read-out feature vector),
so it DROPS INTO the entire EMERGE-78 harness (construction generators, final-state slot read-out, the governing-cue +
symmetric-window baselines, hand-labeler control, anti-cheats) with the reservoir SWAPPED. Only the pool changes: a
fixed-random recurrent Izhikevich pool (C=100/k=0.7/vr=-60/vt=-40/a=0.03/b=-2/c=-50/d=100, v_peak=35 -- the standard RS
cortical operating point) driven token-by-token; the feature = per-neuron spike-COUNT over the whole sequence (the
population rate vector, the LSM read-out).

THE DE-RISK (6 seeds; rate-level CPU/numpy -- the Izhikevich pool runs on numpy; reuse the EMERGE-78 harness; NO `sim/` edit):
  RUNG 1 (this file): does the SPIKING pool retain enough to do the CORE form->role map (the EMERGE-78 comprehension task)?
    * (A) CONSOLIDATION -- the spiking LSM LEARNS the full form->role map (train role acc) matching the rate reservoir;
    * (B) NON-LOCAL -- the spiking LSM resolves the relative-clause HEAD where the governing-cue + symmetric-window
      baselines are at chance (the EMERGE-78 gate; the pool's recurrence integrates the whole sequence on spikes);
    * controls: rel-head scramble -> chance; SPIKING-ness proven by the pool being genuinely active (mean spikes/neuron > 0)
      + a POOL-SILENCE lesion (zero the recurrent + input drive) collapsing the read-out (the read is from spikes, not a
      static bias); the hand-labeler control (None on multi-arg).
  GO bar: train >= 0.90 AND rel-head reservoir >= 0.85 while BOTH baselines <= 0.65 AND scramble collapses AND the pool is
  genuinely spiking (active + silence-lesion collapses). If the spiking pool CANNOT learn the map / resolve the non-local
  (spikes destroy the fine state a near-critical tanh pool kept) -> honest BOUNDARY naming the exact deficit (pool
  operating point / read-out / needs the RF resonate-and-fire complex-synapse pool instead of Izhikevich) as the next
  single-variable de-risk. Do NOT force GO.

HONEST SCOPE. RUNG 1 is a DIRECT recurrent Izhikevich pool (the project's spiking neuron model) -- a faithful spiking LSM,
CPU-runnable, cheap-first. RUNG 2 (follow-on) puts it on a full `SimulationBridge` region (the on-substrate realization).
The EMERGE-79 distal-cue MEMORY-DEPTH (how far a spiking pool holds a 1-bit cue -- likely SHORTER than the near-critical
tanh reservoir) is a separate characterization (a spiking pool's fading memory is typically shorter). Reuse-by-import
(EMERGE-78 harness + the project's Izhikevich params); NO `sim/` edit.

Run:
  python -m research.runners._emerge80_spiking_lsm_port_derisk --demo
  python -m research.runners._emerge80_spiking_lsm_port_derisk --derisk
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

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
# reuse the ENTIRE EMERGE-78 harness (only the reservoir is swapped for the spiking LSM)
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _make_sentence, _slot_data, _fit_slots, _slot_acc, _fit_gov_baseline, _gov_acc,
    _fit_symwin, _symwin_acc, _hand_labeler_none, _TRAIN_KINDS, _RELHEAD_KINDS, _N_TRAIN_PER_CONSTRUCTION, _ROLES,
)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge80_spiking_lsm_port.json"

# Izhikevich 2007 RS cortical operating point (the project's default RS pyramidal; sim/enums.py IZH2007_RS_CORTICAL_PYRAMIDAL)
_IZH = dict(C=100.0, k=0.7, vr=-60.0, vt=-40.0, a=0.03, b=-2.0, c=-50.0, d=100.0, v_peak=35.0)
_DT = 0.5
_N_POOL = 500
_T_STEP = 30                      # sim sub-steps per input token (15 ms at dt=0.5 -> enough for Izhikevich to integrate)
_REC_DENSITY = 0.1
_REC_SCALE = 22.0                 # recurrent synaptic current scale (pA per presynaptic spike, summed)
_IN_SCALE = 350.0                 # input drive scale (pA per active closed-class/OPEN input dim; > RS rheobase ~70-100 pA)
_BIAS = 55.0                      # tonic background current (keeps the pool near-threshold / fluctuation-driven, LSM regime)


class SpikingLSM:
    """A recurrent Izhikevich liquid-state machine (Maass 2002) with the EMERGE-78 `Reservoir` API. `final_state(U)`
    drives the pool token-by-token with the encoded input sequence and returns the per-neuron spike-COUNT vector over the
    whole sequence (the population read-out feature). Fixed-random recurrent + input weights, deterministic per seed."""

    def __init__(self, in_dim, seed, n=_N_POOL):
        rng = np.random.default_rng(seed * 7919 + 3)
        self.n = n
        self.W_in = (rng.random((n, in_dim)) * 2 - 1) * _IN_SCALE
        Wr = (rng.random((n, n)) * 2 - 1)
        Wr = Wr * (rng.random((n, n)) < _REC_DENSITY)
        np.fill_diagonal(Wr, 0.0)
        self.W_rec = Wr * _REC_SCALE
        self._last_mean_spikes = 0.0

    def final_state(self, U, silence=False):
        p = _IZH
        v = np.full(self.n, p["vr"], dtype=np.float64)
        u = np.zeros(self.n, dtype=np.float64)
        spikes = np.zeros(self.n, dtype=np.float64)
        count = np.zeros(self.n, dtype=np.float64)
        total_steps = 0
        for t in range(len(U)):
            drive = np.zeros(self.n) if silence else (self.W_in @ U[t])
            for _ in range(_T_STEP):
                rec = np.zeros(self.n) if silence else (self.W_rec @ spikes)
                I = drive + rec + (0.0 if silence else _BIAS)   # tonic bias keeps the pool responsive (LSM regime)
                dv = (p["k"] * (v - p["vr"]) * (v - p["vt"]) - u + I) / p["C"]
                du = p["a"] * (p["b"] * (v - p["vr"]) - u)
                v = v + dv * _DT
                u = u + du * _DT
                fired = v >= p["v_peak"]
                spikes = fired.astype(np.float64)
                v = np.where(fired, p["c"], v)
                u = np.where(fired, u + p["d"], u)
                count += spikes
                total_steps += 1
        self._last_mean_spikes = float(count.mean())
        return count / max(1, total_steps)         # per-neuron population rate over the whole sequence


def _derisk_one(seed):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    enc = Encoder(discovered)
    lsm = SpikingLSM(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(_N_TRAIN_PER_CONSTRUCTION)]
    Ws = _fit_slots(lsm, enc, train)
    gov_tab, gov_def = _fit_gov_baseline(train)
    sw_tab, sw_def = _fit_symwin(enc, train)

    train_acc = _slot_acc(lsm, enc, Ws, [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(30)])

    rel = [_make_sentence(k, rng, subj, verb, obj) for k in _RELHEAD_KINDS for _ in range(150)]
    relhead_res = _slot_acc(lsm, enc, Ws, rel, only_slot=0)
    relhead_gov = _gov_acc(gov_tab, gov_def, rel, only_slot=0)
    relhead_symwin = _symwin_acc(enc, sw_tab, sw_def, rel, only_slot=0)
    rel_full = _slot_acc(lsm, enc, Ws, rel)

    scr = np.random.default_rng(seed * 613 + 7)
    relhead_scramble = _slot_acc(lsm, enc, Ws, rel, scramble_rng=scr, only_slot=0)

    # SPIKING-ness: the pool is genuinely active, and a POOL-SILENCE lesion (zero all drive) collapses the read-out
    _ = lsm.final_state(enc.encode(_make_sentence("transitive", rng, subj, verb, obj)[0]))
    mean_spikes = lsm._last_mean_spikes
    S_sil, Y_sil = _slot_data_silence(lsm, enc, rel)
    silence_acc = _silence_acc(Ws, S_sil, Y_sil, only_slot=0)

    hand_acc, hand_none = _hand_labeler_none(discovered, rng, subj, verb, obj, n=40)

    return {
        "seed": seed, "n_pool": lsm.n, "mean_spikes_per_neuron": mean_spikes,
        "train_acc": train_acc, "relhead_reservoir": relhead_res, "relhead_gov_baseline": relhead_gov,
        "relhead_symwin_baseline": relhead_symwin, "rel_full": rel_full, "relhead_scramble": relhead_scramble,
        "silence_lesion_acc": silence_acc, "hand_labeler_acc": hand_acc, "hand_none": hand_none,
        "chance_binary": 0.5,
    }


def _slot_data_silence(lsm, enc, sentences):
    """Collect the silence-lesioned final states at each content slot (drive zeroed -> the read-out has no spikes to read
    -> must collapse if the read is genuinely from pool activity)."""
    from collections import defaultdict
    S, Y = defaultdict(list), defaultdict(list)
    from research.runners._emerge78_reservoir_form_to_role_derisk import _ROLE_IDX
    for toks, roles in sentences:
        f = np.concatenate([lsm.final_state(enc.encode(toks), silence=True), [1.0]])
        for k, t in enumerate(sorted(roles)):
            S[k].append(f); Y[k].append(_ROLE_IDX[roles[t]])
    return S, Y


def _silence_acc(Ws, S, Y, only_slot=None):
    hit = tot = 0
    for k in S:
        if only_slot is not None and k != only_slot:
            continue
        if k not in Ws:
            continue
        X = np.asarray(S[k]); y = np.asarray(Y[k])
        hit += int((np.argmax(X @ Ws[k], axis=1) == y).sum()); tot += len(y)
    return float(hit / max(1, tot))


def _demo(seed=42):
    print("\n=== EMERGE-80 -- SPIKING-LSM PORT: a recurrent IZHIKEVICH liquid-state machine replaces the rate tanh "
          "reservoir; does the SPIKING pool learn the form->role map + resolve the non-local rel-clause head? ===\n",
          flush=True)
    d = _derisk_one(seed)
    print(f"  pool {d['n_pool']} Izhikevich RS | mean spikes/neuron over a sentence: {d['mean_spikes_per_neuron']:.2f}")
    print(f"  (A) CONSOLIDATION train role acc: {d['train_acc']:.3f}")
    print(f"  (B) NON-LOCAL rel-head: spiking-LSM {d['relhead_reservoir']:.3f} vs gov(left) {d['relhead_gov_baseline']:.3f} "
          f"/ symwin(+-2) {d['relhead_symwin_baseline']:.3f} (chance {d['chance_binary']:.2f})  [full rel {d['rel_full']:.3f}]")
    print(f"  rel-head scramble {d['relhead_scramble']:.3f} | POOL-SILENCE lesion {d['silence_lesion_acc']:.3f} "
          f"(collapse = read is from spikes) | hand {d['hand_labeler_acc']:.3f}\n")


def _derisk(seeds):
    print(f"EMERGE-80 de-risk: SPIKING-LSM PORT (recurrent Izhikevich liquid-state machine) of the EMERGE-78 form->role "
          f"reservoir; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s); per.append(d)
            print(f"  [seed {s}] spikes/neuron {d['mean_spikes_per_neuron']:.2f} | train {d['train_acc']:.3f} | REL-HEAD "
                  f"lsm {d['relhead_reservoir']:.3f}/gov {d['relhead_gov_baseline']:.3f}/sym {d['relhead_symwin_baseline']:.3f} "
                  f"(full {d['rel_full']:.3f}) | scr {d['relhead_scramble']:.3f} | silence {d['silence_lesion_acc']:.3f} | "
                  f"hand {d['hand_labeler_acc']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        train, relhead, gov, symwin = m("train_acc"), m("relhead_reservoir"), m("relhead_gov_baseline"), m("relhead_symwin_baseline")
        scramble, silence, spikes, rel_full = m("relhead_scramble"), m("silence_lesion_acc"), m("mean_spikes_per_neuron"), m("rel_full")
        hand = m("hand_labeler_acc")
        chanceb = per[0]["chance_binary"]

        active = spikes > 0.5
        consolidation_ok = train >= 0.90
        nonlocal_ok = (relhead >= 0.85 and gov <= 0.65 and symwin <= 0.65)
        scramble_ok = scramble <= chanceb + 0.18
        silence_ok = (relhead - silence) >= 0.20             # the read is genuinely from pool spikes
        go = bool(active and consolidation_ok and nonlocal_ok and scramble_ok and silence_ok)

        if go:
            verdict = (
                f"GO -- the EMERGE-78/79 form->role RESERVOIR ports to the SPIKING substrate: a recurrent IZHIKEVICH "
                f"liquid-state machine (the project's RS cortical operating point, fixed-random recurrence, genuinely "
                f"active at {spikes:.2f} spikes/neuron per sentence) LEARNS the full form->role map (train role acc "
                f"{train:.3f} >= 0.90) via a ridge read-out over its whole-sequence population spike-counts, AND resolves "
                f"the non-local RELATIVE-CLAUSE HEAD ({relhead:.3f}) where BOTH the governing-cue baseline ({gov:.3f}) AND "
                f"the symmetric +-2 window ({symwin:.3f}) are at chance (~{chanceb:.2f}; full rel-clause role {rel_full:.3f}) "
                f"-- the pool's spiking recurrence integrates the whole sequence. Controls: rel-head WORD-ORDER-SCRAMBLE "
                f"{scramble:.3f} -> reads structure; POOL-SILENCE lesion (zero all drive) collapses the read to {silence:.3f} "
                f"(drop {relhead-silence:.3f}) -> the read-out is genuinely from POOL SPIKES, not a static bias. The hand "
                f"labeler scores {hand:.3f} on the multi-arg shapes. {len(seeds)} seeds. ==> the reservoir form->role "
                f"mechanism is realized on the project's SPIKING neuron model (a liquid-state machine), not just a rate "
                f"tanh pool -- the fully-spiking-one-brain directive. RUNG 2 (on a full SimulationBridge region) + the "
                f"distal-cue memory-DEPTH characterization are the follow-ons. Reuse EMERGE-78 harness; NO sim/ edit.")
        else:
            miss = []
            if not active:
                miss.append(f"the Izhikevich pool is nearly SILENT ({spikes:.2f} spikes/neuron) -- the operating point "
                            f"(input/recurrent scale) needs tuning; the read-out has no signal")
            if not consolidation_ok:
                miss.append(f"the spiking LSM did NOT learn the form->role map (train {train:.3f} < 0.90) -- spikes may "
                            f"destroy the fine state a near-critical tanh pool kept; next: tune the pool / try the RF "
                            f"resonate-and-fire complex-synapse pool")
            if not nonlocal_ok:
                miss.append(f"the spiking LSM did NOT resolve the non-local rel-head (lsm {relhead:.3f} vs gov {gov:.3f} / "
                            f"symwin {symwin:.3f}) -- the spiking pool's whole-sequence integration is weaker than the "
                            f"tanh reservoir's")
            if not scramble_ok:
                miss.append(f"rel-head scramble {scramble:.3f} did not collapse")
            if not silence_ok:
                miss.append(f"POOL-SILENCE lesion did not collapse the read ({relhead:.3f} vs {silence:.3f}) -- the "
                            f"read-out may be reading a static bias, not pool spikes")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The spiking-LSM PORT of the rate reservoir hits a deficit; "
                       "the exact residual (pool operating point / read-out / Izhikevich-vs-RF-resonate-and-fire) is the "
                       "next single-variable de-risk. The RATE reservoir (EMERGE-78/79) stands; this characterizes the "
                       "spiking realization. Do NOT force GO.")
    else:
        go = False; verdict = f"ERROR -- {err}"
        train = relhead = gov = symwin = scramble = silence = spikes = rel_full = hand = None

    summary = {
        "probe": "emerge80_spiking_lsm_port", "verdict": verdict, "go": bool(go) if err is None else False,
        "mechanism": ("port the EMERGE-78/79 rate echo-state reservoir onto the project's SPIKING neuron model: a "
                      "recurrent Izhikevich 2007 RS liquid-state machine (Maass 2002; fixed-random recurrence + input) "
                      "driven by the EMERGE-62 discovered closed-class configuration; a ridge read-out over the pool's "
                      "whole-sequence population spike-COUNTS -> per-slot thematic role. Mirrors the EMERGE-78 Reservoir "
                      "API so it drops into the entire EMERGE-78 harness. Reuse-by-import; NO sim/ edit."),
        "task": ("does the SPIKING pool retain enough to learn the form->role map (train) + resolve the non-local "
                 "relative-clause head (vs governing-cue + symmetric-window baselines at chance), with the pool genuinely "
                 "spiking (active + pool-silence-lesion collapse) + scramble collapse; 6-seed; rate CPU"),
        "izh_params": _IZH, "n_pool": _N_POOL, "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "mean_spikes_per_neuron": spikes, "train_acc": train, "relhead_reservoir": relhead,
            "relhead_gov_baseline": gov, "relhead_symwin_baseline": symwin, "rel_full": rel_full,
            "relhead_scramble": scramble, "silence_lesion_acc": silence, "hand_labeler_acc": hand,
        },
        "per_seed": per,
        "HONEST_NOTE": ("RUNG 1 of the spiking port: a DIRECT recurrent Izhikevich pool (the project's spiking neuron "
                        "model), CPU-runnable, cheap-first. GO = the reservoir form->role mechanism survives on spikes "
                        "(learns the map + non-local, pool genuinely active + silence-lesion collapses). BOUNDARY = the "
                        "spiking realization has a deficit (names the residual: operating point / read-out / RF-vs-Izh). "
                        "RUNG 2 (full SimulationBridge region) + the distal-cue memory-DEPTH (a spiking pool's fading "
                        "memory is typically SHORTER than a near-critical tanh reservoir) are follow-ons. Reuse EMERGE-78 "
                        "harness; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge80] VERDICT: {verdict}", flush=True)
    print(f"[emerge80] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
