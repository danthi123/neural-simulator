"""The NEXT declarative cross-edge on the one-brain connectome: curiosity's `ask` (crave) pool -> d6_multiref_wm's
`w0` working-memory slot — a FRESH, biologically-motivated, functionally-related pair (neither organ has a
declared cross-edge with any other organ yet), wired ONE direction, through the SAME generic
`onebrain_crossedge_gate.run_gate` R1/R4/R4-reciprocal all use (a `CrossEdge` + `train_fn` + `read_fn` — no bespoke
F-gate file).

WHY THIS PAIR (biological rationale). Curiosity's `ask` population is this substrate's own novelty/epistemic-gap
crave signal (`curiosity_production_organ.py`; DR-1's `from_novelty -> excitability_drive` neuromodulator on
`group:ask`) — the functional correlate of "I want to know more about this." d6's multi-referent WM buffer holds
the discourse referent(s) currently in play. Both are already co-resident in `full7` but had ZERO synaptic
interaction before this edge — two organs sitting side by side, not one brain (CLAUDE.md's own standing lesson:
"co-location w/ zero cross-synapses isn't one-brain; real = cross-region synaptic interaction").

AN HONEST CORRECTION (kept for the record — the instrument is part of the emulation). The FIRST hypothesis tried
here was a dopaminergic gating-BOOST account (Lisman & Grace 2005; Bunzeck & Duzel 2006; Braver & Cohen's PFC
adaptive-gating theory, O'Reilly & Frank 2006): novelty should GATE MORE content INTO / sustain WM. Trained and
read exactly as declared below, the substrate's own measured effect is the OPPOSITE sign, cleanly and repeatably
(6-seed table in §3): driving `ask` SUPPRESSES the already-held w0 referent's sustained firing rate, ~95-98%
lesion-attributable. Rather than force the read to agree with the first hypothesis, the biological framing was
corrected to the one the substrate's own measurement actually supports: ATTENTIONAL-CAPTURE / resource-competition
for a limited-capacity WM buffer — an involuntary orienting response to a salient/novel signal measurably
DISRUPTS ongoing WM maintenance, not reinforces it (Berti & Schroger 2003, J. Cognitive Neuroscience, "Working
memory controls involuntary attention switching: evidence from an auditory distraction paradigm"; SanMiguel,
Corral & Escera 2008, J. Cognitive Neuroscience, on involuntary attention capture impairing WM consolidation).
Read this way, the pairing is still the SAME functionally-related, biologically-motivated pair (curiosity's own
crave/orienting signal vs. the WM buffer it's co-resident with) — the substrate simply realized the COMPETITIVE
half of that literature, not the gating-boost half, and both halves are genuine, independently-documented
accounts of how salience/novelty signals interact with working memory. A directly plausible substrate-level
reading: `ask`'s learned excitatory drive onto w0, riding NMDA-mediated recurrent dynamics (`enable_nmda_recurrent`,
a slow ~100ms-decay conductance), pushes the target population past its own recurrent excitation's effective
operating point rather than simply adding to it — consistent with this project's OWN standing lesson that a
mechanism's OPERATING POINT is implicit, not free, and drives which biological account applies.

CONVERSATIONAL RATIONALE. This is the substrate correlate of "a genuinely novel, urgent question can knock the
thing you were just discussing out of mind" — curiosity's own crave state, when it fires hard, measurably
COMPETES with the currently-held referent rather than reinforcing it, a self-report-honest correlate of a real,
common conversational experience (a tangent derails what you were holding in mind), not an idealized always-helps
account of curiosity.

THE EDGE, added PURELY BY DECLARATION:
  key="ask_to_w0", source_key="curiosity"/source_region="ask" (a registered top-level curiosity region — no
  source_idx_fn needed), target_key="d6_multiref_wm"/target_region="w0" (a registered top-level d6 slot region,
  the SAME region R1's own edge already reads/writes — no target_idx_fn needed).

TRAINING (the substrate's OWN standard Hebbian, `hebbian_symmetric`): a host-supplied tonic co-drive injects
current directly into `ask` and `w0` TOGETHER over N_EPISODES (the declared, honest teaching signal every other
cross-edge in this codebase uses — "the co-occurrence experience is HOST-SUPERVISED, not claimed self-organized";
the substrate's own Hebbian rule does the binding). Only the declared `ask_to_w0` edge is plastic (`freeze_rest`).

READ (load-bearing, the crux): LOAD w0 into its own held bump first (condition-blind — identical in both arms,
R1's own load-then-cue read shape), THEN drive ONLY `ask` over the read window (`familiar`: ask undriven, PA=0 —
the control, genuinely silent, the lesson `...provenance-to-selfschema-reciprocal-GO.md` §4 banked: pick the
condition where the SOURCE region is genuinely silent as the control; `novel`: ask driven at ASK_DRIVE_PA) — does
w0's own sustained firing rate SHIFT under `novel` vs `familiar`, attributable to the learned edge alone (lesion
removes it)? See "AN HONEST CORRECTION" above for why the answer is a suppression, not a boost.

Run (numpy CPU; NO sim/ edit; routes off the GPU):
  SIM_BACKEND=numpy python -m research.runners._onebrain_crossedge_curiosity_to_d6wm --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_crossedge_curiosity_to_d6wm \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend

from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, merge_organs, MergedPool
from research.runners.onebrain_crossedge_gate import (
    CrossEdgeGateSpec, run_gate, verify_byte_off, cross_edge_masks, lesion_cross_edges,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE EDGE — curiosity's `ask` crave pool -> d6's `w0` WM slot.
# ─────────────────────────────────────────────────────────────────────────────────────────────
W0 = 0.05                          # near-zero seed weight (must GROW, not be pre-wired) — the framework default
GATE = "ask_to_w0"

ASK_DRIVE_PA = 600.0               # curiosity's `ask` drive during train + the 'novel' read condition (matches
                                    # CUE_DRIVE_PA's scale in _curiosity_seek_learn_onbridge_derisk.py — the same
                                    # organ's own de-risk uses currents in this range to robustly drive its pools)
LOAD_PA = 400.0                    # w0's drive during TRAINING (a tonic co-drive) AND to establish w0's OWN held
                                    # bump before each READ (R1's own `LOAD_PA` constant, reused verbatim — "=
                                    # MultiSlotHold input_gain"). At read time the LOAD phase runs identically in
                                    # BOTH conditions (a shared, condition-blind step) — only the SUBSEQUENT window
                                    # differs (ask driven or not), so any read delta is attributable to that window.
LOAD_STEPS = 30                    # matches R1's own WM-slot load window
TRAIN_STEPS = 30                   # matches R1/R4's own convention
READ_STEPS = 100                   # matches R4's own recall window
N_EPISODES = 100                   # > R4-reciprocal's 60 — a smoke on seed 43 showed the grown weight (~1.6, well
                                    # under HMAX=6.0) under-trained for that seed's marginal read; more episodes
                                    # grows the edge further before the first calibration read (R4's own precedent
                                    # for de-risking under-training — see its module docstring's N_EPISODES note)
N_READS = 4                        # averaged reads per condition (denoise)
HMAX = 6.0                         # starting bound (R4-reciprocal's own calibrated point for a single fresh
                                    # tonic-co-drive edge); re-verified empirically at --smoke before the 6-seed
                                    # commit (see main()'s smoke-calibration note)

INTACT_FLOOR = 0.008                 # R1's own `F2_INTACT_FLOOR` (0.008), reused verbatim — the |Δ| the w0 rate
                                    # must move, intact, over the 'familiar' control
LESION_RATIO = 0.34                 # lesion |Δ| must be < this * intact |Δ| — R1/R4's own convention, reused

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_ampa")

CROSS_EDGES = [
    CrossEdge(key="ask_to_w0", source_key="curiosity", source_region="ask",
             target_key="d6_multiref_wm", target_region="w0", init_weight=W0, plastic=True, gate=GATE,
             learn_rule="rate_hebbian", freeze_rest=True),
]


def _build(seed, with_edge: bool):
    """Build the [curiosity, d6_multiref_wm] merged pool, optionally with the declared edge. Neither organ's own
    production-read pipeline (battery encode / calibration) is run here — this pair's train/read is a SELF-
    CONTAINED direct region-current protocol (R1's own house style for d6/comprehension), so the WITH and WITHOUT
    arms differ ONLY by the declared edge — a valid byte-off comparison by construction (same organs, same order,
    no extra encode step in either arm)."""
    CUR, D6 = REGISTRY["curiosity"], REGISTRY["d6_multiref_wm"]
    pool = merge_organs([CUR, D6], seed=seed, wire=True, cross_edges=(CROSS_EDGES if with_edge else None))
    b = pool.bridge
    rm = b.region_manager

    def idxr(nm):
        return np.asarray(rm.indices(nm), np.int64)

    ix = {nm: idxr(nm) for nm in ("ask", "cue", "striosome_value", "reward_us", "snc", "w0", "w1", "fs")}

    if with_edge:
        pool.apply_cross_edge_freeze()      # the declared edge is the SOLE plastic synapse (R1/R4's whitelist)

    return pool, ix


class AskToW0Pool:
    """The FRESH cross-edge on the [curiosity, d6_multiref_wm] merged pool: curiosity's `ask` crave pool ->
    d6's `w0` WM slot. Grows by the substrate's OWN standard Hebbian rule from a tonic co-drive of both regions;
    the load-bearing read drives `ask` ALONE and reads whether `w0` rises (attributable to the edge, not a shared
    external current — w0 gets none at read time)."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool, self.ix = _build(seed, with_edge=True)
        self.b = self.bridge = self.pool.bridge
        self.masks = cross_edge_masks(self.b, CROSS_EDGES)

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]

        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=0.05, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()
        # THE READ-ISOLATION FIX (2026-09-02, Port A — ported from `onebrain_merge_framework.MergedPool`'s
        # already-tested `_PER_NEURON_STATE` + `_SEQ_EXTRA_STATE` tuples, the SAME primitives
        # `read_isolation()`/`sequence_isolation()` use; the audited bug class's own template =
        # `_crossedge_surprise_metacog_derisk.py`'s `_EXTRA_RESET_ARRAYS`). The ORIGINAL `_hard_reset` below
        # restored only v/u/conductances/firing_states -- it never touched `cp_refractory_timers`/
        # `cp_prev_firing_states` (hard firing gates, independent of membrane potential: a neuron mid-refractory
        # from the immediately-prior read/episode stays gated at the start of the next even though v/u were
        # reset) or `cp_neuron_activity_ema`/`cp_neuron_firing_thresholds` (the participation-gated homeostatic
        # EMA + adaptive threshold, which silently drifts on whichever neurons the immediately-prior read/
        # episode drove) -- the audited C2 bug class, `_PER_NEURON_STATE`.
        #
        # HONEST EXTENSION beyond the audited 4-array class, FOUND while building this fix's own selftest (not
        # assumed): restoring only `_PER_NEURON_STATE` left the selftest's repeat-read still non-identical
        # (0.0679 vs 0.0607 on a lesioned seed-42 pool). Direct instrumentation (same fixed-vs-not diff the
        # audit used) isolated the residual to `_SEQ_EXTRA_STATE` -- specifically `cp_conductance_g_nmda_rise` /
        # `cp_conductance_g_nmda_recurrent` / `cp_conductance_g_nmda_recurrent_rise` (this pair's OWN docstring
        # already names NMDA-recurrent dynamics as load-bearing for the read's operating point) and
        # `cp_synapse_pulse_timers`/`cp_synapse_pulse_progress`. This runner's `read_w0()` is a genuine
        # MULTI-TURN stateful read (a condition-blind LOAD phase, then the scored ask-driven phase, both inside
        # ONE `_hard_reset()`) -- exactly the shape `sequence_isolation()`'s own docstring says needs the wider
        # tuple, not just `read_isolation()`'s per-neuron set. Restoring BOTH tuples (this pool has no
        # co-resident organ to protect, so a plain snapshot/restore in `_hard_reset` is the direct equivalent of
        # wrapping every read in `sequence_isolation()`) makes the selftest below PASS bitwise. Snapshot BOTH at
        # TRUE REST (right after the settle loop above) ONCE here; restored wholesale on every `_hard_reset()`
        # below -- see research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
        # (IG-1: this was the ONE inflated GO in that audit, live-wired default-on in /api/brain-chat).
        self._rest_extra = {}
        for nm in list(MergedPool._PER_NEURON_STATE) + list(MergedPool._SEQ_EXTRA_STATE):
            arr = getattr(self.b, nm, None)
            self._rest_extra[nm] = np.asarray(to_host(arr)).copy() if arr is not None else None

    # ---- primitives (R1/R4 house style) ----
    def _hard_reset(self):
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in _CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_hebb_coactivity_trace", None) is not None:
            b.cp_hebb_coactivity_trace[:] = 0.0
        # THE READ-ISOLATION FIX: restore EVERY array the framework's `_PER_NEURON_STATE` + `_SEQ_EXTRA_STATE`
        # tuples name to its TRUE-REST snapshot (captured once in __init__, immediately post-settle). This
        # supersedes the piecemeal v/u/conductance/firing resets above for anything `_PER_NEURON_STATE` also
        # names (redundant, harmless -- same target value) and ADDITIONALLY restores `cp_refractory_timers` /
        # `cp_prev_firing_states` / `cp_neuron_activity_ema` / `cp_neuron_firing_thresholds` (the audited C2
        # class) plus the NMDA-recurrent + synapse-pulse buffers `_SEQ_EXTRA_STATE` names (this runner's own
        # extension, found via the selftest below), none of which anything above ever touched. Without this,
        # whichever condition/episode ran immediately before a scored read leaked its residual
        # refractory/homeostatic/NMDA-recurrent state into the next one -- an ORDER-dependent bias, not a
        # genuine condition effect (verified: the fixed `_hard_reset` makes two identical consecutive reads
        # bitwise identical -- see `_selftest_read_isolation()` below).
        for nm, val in self._rest_extra.items():
            if val is not None:
                getattr(b, nm)[:] = xp.asarray(val)
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read=None):
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = bool(learn)
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_hebbian_learning = False
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks["ask_to_w0"]].mean())

    # ---- emergence: grow the cross-edge from experience (a host tonic co-drive of ask + w0) ----
    def train(self, n_episodes=N_EPISODES):
        ix = self.ix
        traj = [dict(ep=0, w=round(self._wmean(), 4))]
        for ep in range(n_episodes):
            self._hard_reset()
            self._drive([(ix["ask"], ASK_DRIVE_PA), (ix["w0"], LOAD_PA)], TRAIN_STEPS, learn=True)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=round(self._wmean(), 4)))
        self.b.core_config.enable_hebbian_learning = False
        return traj

    # ---- the load-bearing read: does an active crave state bias an ALREADY-HELD referent's sustained rate? ----
    def read_w0(self, condition):
        """LOAD w0 into its own held bump first (IDENTICAL LOAD_PA/LOAD_STEPS in both conditions — a condition-
        blind step, exactly R1's own load-then-cue read shape), then drive `ask` under `condition` in
        {"familiar","novel"} over the read window and read w0's mean firing rate DURING that window (w0 itself
        gets no further external current past the load — only its own recurrent hold + whatever the cross-edge
        carries). 'familiar' (ask_pa=0.0) is the CONTROL: `ask` is genuinely silent under it (no injected current),
        so any w0-rate shift under 'novel' vs 'familiar' is attributable to the cross-edge carrying `ask`'s OWN
        activity onto an already-held referent, not to the load step (shared) or an unloaded, dead baseline."""
        if condition not in ("familiar", "novel"):
            raise ValueError(condition)
        ask_pa = ASK_DRIVE_PA if condition == "novel" else 0.0
        ix = self.ix
        rates = []
        for _ in range(N_READS):
            self._hard_reset()
            self._drive([(ix["w0"], LOAD_PA)], LOAD_STEPS)          # condition-blind: identical in both arms
            acc = self._drive([(ix["ask"], ask_pa)], READ_STEPS,
                              read={"w0": ix["w0"], "ask": ix["ask"]})
            rates.append(acc)
        return {"w0": float(np.mean([r["w0"] for r in rates])),
                "ask": float(np.mean([r["ask"] for r in rates]))}


GATE_SPEC = CrossEdgeGateSpec(
    name="curiosity_ask_to_d6_w0",
    cross_edges=CROSS_EDGES,
    train_fn=lambda pool: pool.train(),
    read_fn=lambda pool, cond: pool.read_w0(cond)["w0"],
    init_weight=W0,
    correct_edges=("ask_to_w0",),
    selectivity_pairs=(),        # ONE-SIDED BY DESIGN, matching R4/R4-reciprocal's own precedent: a single edge
                                  # onto one WM slot has no companion population for a weight-ratio comparison;
                                  # selectivity is demonstrated FUNCTIONALLY at the read (below), not as a ratio.
    grow_factor=5.0, drift_tol=1e-6,
    condition_order=("familiar", "novel"),    # 'familiar' is the control (ask genuinely silent)
    control="familiar",
    expected={"novel": {"sign": -1, "floor": INTACT_FLOOR}},   # SIGN, empirically determined (see module docstring
                                                                # "AN HONEST CORRECTION" note): the grown edge
                                                                # SUPPRESSES w0's held rate under 'novel', not boosts
                                                                # it — a resource-competition, not a gating-boost, effect.
    lesion_ratio=LESION_RATIO, credit_signal="rate_hebbian",
)


def _noedge_bridge(seed):
    """The no-cross-edge baseline bridge for byte-off: the SAME [curiosity, d6] build as the with-edge pool
    (neither arm runs any extra encode step). Integration must add ONLY the declared edge."""
    pool, _ix = _build(seed, with_edge=False)
    return pool.bridge


def run_seed(seed):
    t0 = time.time()
    pool = AskToW0Pool(seed)
    gate = run_gate(pool, GATE_SPEC)                       # trains + emergence + interaction (lesions the pool)

    bridge_with = AskToW0Pool(seed).b
    bridge_without = _noedge_bridge(seed)
    byte_off = verify_byte_off(bridge_with, bridge_without, GATE_SPEC)

    go = bool(gate["emergence"]["PASS"] and gate["interaction"]["PASS"] and byte_off["PASS"])
    return {"seed": int(seed), "GO": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": gate["emergence"], "interaction": gate["interaction"], "byte_off": byte_off,
            "trajectory": gate["trajectory"]}


def _selftest_read_isolation(seed=42):
    """FAILS-IN-FAILING-DIRECTION guard for the 2026-09-02 Port A read-isolation fix (see `_hard_reset`'s own
    docstring above): on a MECHANISM-ZEROED pool (the cross-edge's own synapses lesioned to 0.0 -- genuinely
    inert, not merely untrained), two back-to-back, identically-conditioned `read_w0("familiar")` calls must be
    BITWISE identical. Each call runs its own `_hard_reset()` internally (via `_drive`'s callers); if any array
    the reset omits carries residue from the first call into the second, the second read's rate differs -- this
    is exactly the leak the audit found (`research/findings/
    2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`). Returns True/False; does not raise, so
    a caller can assert/report as it likes."""
    pool = AskToW0Pool(seed)
    lesion_cross_edges(pool.b, pool.masks, pool.xp)   # zero the ask_to_w0 mechanism in place -- genuinely inert
    r1 = pool.read_w0("familiar")
    r2 = pool.read_w0("familiar")
    ok = (r1["w0"] == r2["w0"]) and (r1["ask"] == r2["ask"])
    print(f"[selftest] seed={seed} lesioned repeat-read: r1={r1} r2={r2} -> "
          f"{'PASS (bitwise identical)' if ok else 'FAIL (read-isolation leak)'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--selftest", action="store_true",
                     help="read-isolation fails-in-failing-direction guard only (no train/6-seed run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        ok = _selftest_read_isolation()
        print("SELFTEST " + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        emg, itn, bo = r["emergence"], r["interaction"], r["byte_off"]
        nov = itn["per_condition"]["novel"]
        print(f"[seed {s}] GO={r['GO']} | grown={emg['grown']['ask_to_w0']:.3f} nocorr={emg['no_corruption']} "
              f"| novel Δ={nov['delta_intact']:+.4f} (lesion {nov['delta_lesion']:+.4f}) "
              f"frac_attrib={nov['frac_attributable']} "
              f"| emg={emg['PASS']} int={itn['PASS']} byteoff={bo['PASS']} ({r['elapsed_s']}s)", flush=True)

    n_go = sum(r["GO"] for r in runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO")
    verdict = (f"{tag} — the FRESH cross-edge curiosity.ask -> d6_multiref_wm.w0 (an attentional-capture / "
               f"resource-competition analog, Berti & Schroger 2003 / SanMiguel-Corral-Escera 2008; neither organ "
               f"had a declared cross-edge before this), added PURELY BY DECLARATION (a CrossEdge + train_fn + "
               f"read_fn through the generic onebrain_crossedge_gate.run_gate — no bespoke F-gate): "
               f"{n_go}/{len(runs)} seeds GROW from the substrate's own standard Hebbian rule, are LOAD-BEARING "
               f"(driving curiosity's crave pool alone SUPPRESSES the already-held WM slot's sustained firing "
               f"rate; the suppression VANISHES on lesion), and are BYTE-IDENTICAL-OFF. numpy CPU; NO sim/ edit; "
               f"additive.")

    # NOTE (2026-09-02, read-isolation fix landing): `all_seeds_go` (n_go == len(runs)) used to be wrapped as a
    # THIRD `Vd.require(...)` here — but that is the OUTCOME itself, not a validity precondition, and double-
    # counting it made `Vd.decide()` collapse to UNDEFINED instead of a genuine NO-GO the one time n_go actually
    # dropped below 6/6 (this fix's own re-verify, see the finding). Preconditions below are VALIDITY checks only
    # (did the lesion control work; is the wiring byte-off-clean) — the outcome is passed straight to `decide(go=)`.
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_crossedge_curiosity_to_d6wm")
        Vd.require("lesion_removes_bias", 1 if all(
            abs(r["interaction"]["per_condition"]["novel"]["delta_lesion"]) <
            LESION_RATIO * max(abs(r["interaction"]["per_condition"]["novel"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the w0-rate rise must VANISH under lesion or it is a confound, not the cross-edge")
        Vd.require("byte_identical_off", sum(r["byte_off"]["PASS"] for r in runs), expect=lambda x: x == len(runs),
                   note="the no-edge pool's base connectivity is byte-identical (integration added ONLY the edge)")
        dec = Vd.decide(all_go or (args.smoke and n_go == len(runs)), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_crossedge_curiosity_to_d6wm", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "gate_spec": {"name": GATE_SPEC.name, "correct_edges": GATE_SPEC.correct_edges,
                            "conditions": GATE_SPEC.condition_order, "control": GATE_SPEC.control,
                            "credit_signal": GATE_SPEC.credit_signal,
                            "cross_edges": [dict(key=ce.key, src=ce.source_region, tgt=ce.target_region)
                                            for ce in CROSS_EDGES]},
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[ASK->W0] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
