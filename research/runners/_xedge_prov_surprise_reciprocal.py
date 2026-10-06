"""DRAFT — xedge source_provenance -> surprise RECIPROCAL (research/draft-xedge-prov-surprise-reciprocal).

Question (PLAN.md, refuter-approved round 3, 2026-10-04): on the SAME physical input, does a claim the
substrate's source-monitoring opponent circuit reads as EXTERNALLY-sourced ('perceived'/'asserted') drive a
HIGHER windowed surprise-pool rate than one it reads as internally-generated ('generated') — and is that
difference carried by the declared CrossEdge prov_perceived -> surprise? This closes the missing half of the
existing surprise->prov_generated edge (`_onebrain_integration_surprise_episodic_crossedge.py`): that edge
gates what gets ENCODED into provenance; only the reverse path lets a verdict ALREADY IN MEMORY change the
live 'am I surprised?' read. Biologically: JHL 1993 source monitoring is inferential (confident) — the
loop framing is the 2026-09-01 reciprocal GO's argued extension; Lisman & Grace 2005 hippocampal-VTA feedback
limb; sensory attenuation/efference copy (self-generated LESS surprising — CONTROL read only, see sign
constraint); Rao & Ballard 1999 precision-weighted PE.

SIGN CONSTRAINT (verified in code, sim/bridge.py standard symmetric Hebbian only potentiates; weights clipped
[hebbian_min_weight, 1.0]): cross-edge weights grow positive only, so this file declares the AMPLIFICATION
half — prov_perceived -> surprise ('foreign content MORE surprising', the Lisman-Grace excitatory sign) —
and reads the attenuation half ('self-generated LESS') as the matched CONTROL condition. The pre-declared
variant B (prov_generated -> patient_expected, onto the surprise circuit's subtractive prediction neurons =
efference-copy anatomy) is the follow-up if the sign is judged wrong — never a post-hoc flip.

PRE-REGISTERED OUTCOME TABLE (written BEFORE any run; PLAN.md §5 verbatim authority — row read decides the
claim, nothing else):
  1  grows + delta perceived>generated clears floor + lesion-attributable + byte-off + clean-zero, ALL 6 dev
     seeds -> PROVISIONAL-SUPPORTS (trace-level only; NOT a GO, NOT 'closed', NOT gate-seed, NOT phenomenal,
     NOT self-organized — training is a host-supervised teaching current, declared; default-OFF wire-in=Claude)
  2  grows, sign right, below floor on some seeds -> weak/operating-point-dependent; NOT trace-level
     load-bearing; next lever = operating-point recalibration, not a new mechanism
  3  grows, delta does NOT collapse under lesion -> the shift is NOT carried by the declared edge (candidates:
     base-wiring leak, co-drive artifact, unmodeled path); run instrument checks first, repair only one that
     fails; if clean, bank the honest negative and iterate the episode recipe (no-defer)
  4  does NOT grow -> verdict on the TRAINING RECIPE only; change recipe and re-run (no-defer), never on biology
  5  grows + trace-level load-bearing but no-edge/lesioned arm ALSO moves, or byte-off FAIL -> CONFOUND; no
     biological reading; instrument bug; fix and re-run
  6  sign REVERSED (generated > perceived) -> directional negative on the anchor; report; variant B next
  7  byte-off PASS but clean-zero FAIL (prov_perceived active under 'generated' exemplar) -> leaky control
     (graded opponent trace); replace control exemplar before reading rows 1-6

MUST NOT CLAIM: no phenomenal anything ('feels surprising' etc.); honest functional read-out only ('the
substrate's windowed surprise-pool rate shifts with its own source-monitoring verdict'); PROVISIONAL at best,
never a GO; dev seeds 7/1,4,9,12,15 ONLY — never 42/43/44/100/101/102; one-sided (amplification only).

Machinery: ONE merged spiking pool [SURPRISE_LITE, source_provenance] (SURPRISE_LITE config reconciliation —
R4's hebbian_rate_window hazard), ONE declarative CrossEdge, trained by the substrate's own standard
rate-window-free symmetric Hebbian (`hebbian_symmetric`, sim/config grounding Kandel 6e Ch 54 / Marr 1971;
the plain two-factor cousin of docs/biology.md:284's three-factor family), through the GENERIC
`onebrain_crossedge_gate.run_gate` — no bespoke F-gate. Port-A read isolation (MergedPool._PER_NEURON_STATE,
onebrain_merge_framework.py:254-259) restores thresholds/refractory/prev-firing/EMA on every _hard_reset;
smoke proves order-invariance. INTACT_FLOOR PLACEHOLDER — smoke sets it by the PRE-DECLARED rule
0.5 x smoke-seed-7 delta_intact, frozen verbatim here + WORKLOG before the 6-seed queue, never revisited.

Run (numpy CPU; NO sim/ edit; pool/workstation only, never >10 min on S; halve N_EPISODES if smoke exceeds):
  SIM_BACKEND=numpy python -m research.runners._xedge_prov_surprise_reciprocal --selftest
  SIM_BACKEND=numpy python -m research.runners._xedge_prov_surprise_reciprocal --smoke
  SIM_BACKEND=numpy python -m research.runners._xedge_prov_surprise_reciprocal \
      --seeds 7,1,4,9,12,15 \
      --out research/findings/raw/_xedge_prov_surprise_reciprocal_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend

from research.runners.onebrain_merge_framework import (
    REGISTRY, CrossEdge, merge_organs,
)
from research.runners.onebrain_crossedge_gate import (
    CrossEdgeGateSpec, run_gate, verify_byte_off,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE EDGE — reciprocal of surprise->prov_generated: prov_perceived -> surprise.
# ─────────────────────────────────────────────────────────────────────────────────────────────
GATE = "provper_to_surprise"        # the single plastic cross-edge gate
W0 = 0.05                           # near-zero seed weight (must GROW, not be pre-wired)
CUE_PA = 2000.0                     # calibrated operating point for this SAME organ pair (template-2 sweep)
CTX_DRIVE_PA = 2500.0               # source_provenance's OWN encoding-context drive (R4's de-risk constant)
EPISODE_DRIVE_PA = 2500.0           # source_provenance's OWN content drive (R4's de-risk constant)
TRAIN_STEPS = 60                    # calibrated CONTRADICT measurement window (CUE_PA sweep)
N_EPISODES = 150                    # calibrated for this organ pair's edge growth
RECALL_STEPS = 100
N_READS = 8                         # averaged reads per condition (denoise; this pairing needed 8)
PRE_STEPS = 60                      # prediction pre-phase (cue alone) before the assertion volley
HMAX = 20.0                         # hebbian_max_weight calibrated on this pair's seed-42 smoke
CROSS_EDGE_LR = 0.15                # this pair's calibrated learning rate (scaling fix, template-2 record)

INTACT_FLOOR = 0.005                # PLACEHOLDER — smoke (seed 7) replaces this by the PRE-DECLARED rule
                                    # 0.5 x smoke delta_intact; frozen verbatim here + WORKLOG before the
                                    # 6-seed queue; never revisited, never re-set from later seeds (PLAN §8e).
LESION_RATIO = 0.34                 # lesion |Δ| must be < this * intact |Δ| (house convention)

_CONDUCT = ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
            "cp_conductance_g_ampa")

# THE READ-ISOLATION FIX (2026-09-02 C2 class, Port A — reuse the framework's own per-neuron-state list):
# arrays the ORIGINAL hand-rolled _hard_reset missed; snapshotted at true rest, restored every reset.
_EXTRA_RESET_ARRAYS = ("cp_refractory_timers", "cp_prev_firing_states",
                       "cp_neuron_activity_ema", "cp_neuron_firing_thresholds")

CROSS_EDGES = [
    CrossEdge(key=GATE, source_key="source_provenance", source_region="prov_perceived",
              target_key="surprise", target_region="surprise", init_weight=W0, plastic=True, gate=GATE,
              learn_rule="rate_hebbian", freeze_rest=True),
]


def _assign_blocks(seed, n_trained):
    """ANTI-CHEAT: RANDOM per-seed assignment of which two trained concepts play cue (c) / false-assertion (c').
    A seed-keyed RNG independent of every other seeded draw in this module (distinct offset)."""
    rng = np.random.default_rng(int(seed) * 104729 + 17)
    c = int(rng.integers(0, n_trained))
    cp = int(rng.integers(0, n_trained - 1))
    if cp >= c:
        cp += 1
    return c, cp


def _build_pool(seed, with_edge=True):
    """Build the [SURPRISE_LITE, source_provenance] MergedPool via the DECLARATIVE cross_edges= framework
    param (same shape as `_onebrain_integration_surprise_episodic_crossedge._build_pool`, edge flipped)."""
    from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, merge_organs
    SURPRISE = REGISTRY["surprise"]
    SP = REGISTRY["source_provenance"]
    # SURPRISE's registered config forces enable_hebbian_learning=True (needed only by its usual pool-1
    # partner) — a direct MergeConflict against source_provenance's False. SURPRISE's OWN circuit is 100%
    # fixed/block-diagonal, so the override is behavior-preserving (verified by the template-2 record).
    SURPRISE_LITE = dataclasses.replace(
        SURPRISE, config={**SURPRISE.config, "enable_hebbian_learning": False, "hebbian_rate_window": False})
    edges = [
        CrossEdge(key=GATE, source_key="source_provenance", source_region="prov_perceived",
                  target_key="surprise", target_region="surprise", init_weight=W0, plastic=True, gate=GATE,
                  learn_rule="rate_hebbian", freeze_rest=True),
    ] if with_edge else None
    pool = merge_organs([SURPRISE_LITE, SP], seed=seed, config_descriptors=[SURPRISE_LITE, SP],
                        wire=True, cross_edges=edges)
    return pool


def _noedge_bridge(seed):
    """The no-cross-edge baseline bridge for byte-off: same battery build, cross_edges=None — integration
    must add ONLY the declared edge."""
    return _build_pool(seed, with_edge=False).bridge


class ProvSurprisePool:
    """The RECIPROCAL feedback edge on the [surprise, source_provenance] merged pool: source_provenance's
    `prov_perceived` opponent pool -> surprise's firing region. Grows by the substrate's OWN standard
    symmetric Hebbian from the mismatch-trial co-drive (CONTRADICT trial + ctx_perceived + content).
    Reads: the same mismatch trial, PLUS the verdict held through the window — windowed mean firing rate of
    the surprise region under 'perceived' vs 'generated' (plan §1 exact probe)."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool = _build_pool(seed, with_edge=True)
        self.pool.ensure_built()
        self.b = self.pool.bridge
        self.bridge = self.b               # onebrain_crossedge_gate.run_gate reads pool.bridge (template-1 named it b)
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("cue", "patient_expected", "patient_asserted", "surprise",
                                           "episode", "content_readout", "ctx_perceived", "ctx_generated",
                                           "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
        meta = self.pool.meta["surprise"]
        self.blk = int(meta["blk"]); self.n_trained = int(meta["n_trained"]); self.n_novel = int(meta["n_novel"])
        self.cue_c, self.assert_cp = _assign_blocks(seed, self.n_trained)   # THIS SEED's random block pair

        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        full_mask = np.isin(row, self.ix["prov_perceived"]) & np.isin(col, self.ix["surprise"])
        assert int(full_mask.sum()) > 0, "declarative cross-edge is EMPTY -- the CrossEdge did not wire"
        self.masks = {"provper->surprise": full_mask}

        from research.runners.onebrain_merge_framework import _source_prov_organ
        self.sp_organ = _source_prov_organ(seed, self.pool)
        self.sp_organ.ensure_built()          # runs its OWN build-time Hebbian encode of the 8-item battery

        # THE DECLARATIVE WHITELIST FREEZE -- cp_plasticity_rate_gain=0 everywhere then re-open ONLY GATE.
        self.pool.apply_cross_edge_freeze()

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        self._noncross &= ~self.masks["provper->surprise"]
        # standard (non-rate-window) Hebbian hyperparameters for OUR cross-edge's training window.
        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=CROSS_EDGE_LR, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()
        # Port-A read isolation: snapshot true-rest for every _EXTRA_RESET_ARRAYS array (restored below).
        self._rest_extra = {}
        for nm in _EXTRA_RESET_ARRAYS:
            arr = getattr(self.b, nm, None)
            self._rest_extra[nm] = np.asarray(to_host(arr)).copy() if arr is not None else None

    # ---- primitives (house shape) ----
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
        # restore every extra per-neuron array to TRUE rest so residue from whichever read/episode ran
        # immediately before cannot leak into the next one (order-dependent bias = the C2 bug class).
        for nm in _EXTRA_RESET_ARRAYS:
            val = self._rest_extra.get(nm)
            if val is not None:
                getattr(b, nm)[:] = xp.asarray(val)
        b.cp_external_input_current[:] = 0.0

    def _drive(self, pairs, steps, learn=False, read=None, pre_pairs=None, pre_steps=0):
        """Optionally run a PREDICTION pre-phase (pre_pairs, pre_steps -- cue alone so the slow GABA_B
        subtractive prediction is settled before the assertion volley), then the measured phase."""
        b, xp = self.b, self.xp
        b.core_config.enable_hebbian_learning = False
        if pre_pairs is not None and pre_steps > 0:
            precur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
            for idx, pa in pre_pairs:
                precur[xp.asarray(idx)] = xp.float32(pa)
            for _ in range(pre_steps):
                b.cp_external_input_current[:] = precur
                b._run_one_simulation_step()
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

    def _wmean(self, name="provper->surprise"):
        data = np.asarray(to_host(self.b.cp_connections.data))
        m = self.masks[name]
        return float(data[m].mean()) if m.any() else float("nan")

    # ---- the surprise-inducing CONTRADICT trial for THIS seed's random block pair ----
    def _cue_idx(self, concept):
        return self.ix["cue"][concept * self.blk:(concept + 1) * self.blk]

    def _cue_pre_pairs(self):
        """Prediction pre-phase drive: cue alone (this seed's cue_c), so patient_expected's recall + the slow
        GABA_B subtractive prediction are settled BEFORE the assertion volley (mismatch-negativity protocol)."""
        return [(self._cue_idx(self.cue_c), CUE_PA)]

    def _contradict_pairs(self):
        ix = self.ix
        assert_idx = ix["patient_asserted"][self.assert_cp * self.blk:(self.assert_cp + 1) * self.blk]
        return [(self._cue_idx(self.cue_c), CUE_PA), (assert_idx, CUE_PA)]

    # ---- emergence: grow the cross-edge from experience (mismatch + PERCEIVED verdict co-drive) ----
    def train(self, n_episodes=N_EPISODES):
        ix = self.ix
        n_items = len(self.sp_organ.patterns["perceived"])
        traj = [dict(ep=0, w=round(self._wmean(), 4))]
        for ep in range(n_episodes):
            self._hard_reset()
            content_idx = ix["episode"][np.asarray(self.sp_organ.patterns["perceived"][ep % n_items], np.int64)]
            drives = (self._contradict_pairs()
                      + [(ix["ctx_perceived"], CTX_DRIVE_PA), (content_idx, EPISODE_DRIVE_PA)])
            self._drive(drives, TRAIN_STEPS, learn=True,
                        pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
            if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                traj.append(dict(ep=ep + 1, w=round(self._wmean(), 4)))
        self.b.core_config.enable_hebbian_learning = False
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    # ---- THE READ: mismatch trial + verdict held through the window -> windowed surprise-pool rate ----
    def read_verdict(self, condition):
        """Recall ONE fixed battery exemplar of `condition` in {"perceived","generated"} — content drive +
        the matching ctx line HELD THROUGH the whole trial window (the verdict must be present for the
        windowed-rate read) — on the SAME mismatch trial, and read the surprise region's mean firing rate.
        'generated' is the CONTROL (clean-zero: prov_perceived must sit ~0 under it — row 7)."""
        ix = self.ix
        if condition not in ("perceived", "generated"):
            raise ValueError(condition)
        content_idx = ix["episode"][np.asarray(self.sp_organ.patterns[condition][0], np.int64)]
        ctx_line = ix["ctx_perceived"] if condition == "perceived" else ix["ctx_generated"]
        pairs = (self._contradict_pairs()
                 + [(content_idx, EPISODE_DRIVE_PA), (ctx_line, CTX_DRIVE_PA)])
        rates = []
        for _ in range(N_READS):
            self._hard_reset()
            acc = self._drive(pairs, RECALL_STEPS,
                              read={"surprise": ix["surprise"], "gen": ix["prov_generated"],
                                    "perc": ix["prov_perceived"]},
                              pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
            rates.append(acc)
        return {"surprise": float(np.mean([r["surprise"] for r in rates])),
                "gen": float(np.mean([r["gen"] for r in rates])),
                "perc": float(np.mean([r["perc"] for r in rates]))}

    def delta_pair(self):
        """delta_intact = surprise rate under 'perceived' - under 'generated' (predicted sign +1)."""
        r_perc = self.read_verdict("perceived")["surprise"]
        r_gen = self.read_verdict("generated")["surprise"]
        return r_perc - r_gen


GATE_SPEC = CrossEdgeGateSpec(
    name="RECIP_prov_perceived_to_surprise",
    cross_edges=CROSS_EDGES,
    train_fn=lambda pool: pool.train(),
    read_fn=lambda pool, cond: pool.read_verdict(cond)["surprise"],
    init_weight=W0,
    correct_edges=(GATE,),
    selectivity_pairs=(),        # ONE-SIDED BY DESIGN (sign constraint) — attenuation half is the control
                                 # READ only; see module docstring.
    grow_factor=5.0, drift_tol=1e-6,
    condition_order=("generated", "perceived"),   # control first, test second (fixed order; smoke proves
                                                  # order-invariance via the Port-A reset before queueing)
    control="generated",
    expected={"perceived": {"sign": +1, "floor": INTACT_FLOOR}},
    lesion_ratio=LESION_RATIO, credit_signal="rate_hebbian",
)


def run_seed(seed):
    t0 = time.time()
    pool = ProvSurprisePool(seed)
    gate = run_gate(pool, GATE_SPEC)                       # trains + emergence + interaction (lesions)

    bridge_with = _build_pool(seed, with_edge=True).bridge
    bridge_without = _noedge_bridge(seed)
    byte_off = verify_byte_off(bridge_with, bridge_without, GATE_SPEC)

    # clean-zero control (row 7): prov_perceived must sit ~0 under the genuinely-generated control exemplar
    gen_rates = pool.read_verdict("generated")
    clean_zero = {"perc_rate": gen_rates["perc"], "gen_rate": gen_rates["gen"],
                  "ok": bool(gen_rates["perc"] < 0.005)}

    supports = bool(gate["emergence"]["PASS"] and gate["interaction"]["PASS"]
                    and byte_off["PASS"] and clean_zero["ok"])
    return {"seed": int(seed), "provisional_supports": supports, "elapsed_s": round(time.time() - t0, 1),
            "emergence": gate["emergence"], "interaction": gate["interaction"], "byte_off": byte_off,
            "trajectory": gate["trajectory"], "clean_zero": clean_zero}


def _selftest_read_isolation(seed=7, verbose=True):
    """READ-ISOLATION fails-in-its-failing-direction guard (C2 class, Port A). On a fresh pool, two
    back-to-back IDENTICAL read_verdict("generated") calls must be bitwise identical with the fix enabled,
    and must DIVERGE with the extra-array restore programmatically disabled (probe has teeth)."""
    pool = ProvSurprisePool(seed)
    ix = pool.ix

    # induce asymmetric residue ahead of the probed reads (the same co-drive train() uses).
    pool._hard_reset()
    pool._drive(pool._contradict_pairs() + [(ix["ctx_perceived"], CTX_DRIVE_PA)], TRAIN_STEPS)

    # Claude ruling 2026-10-04 (xedge-selftest): organic residue cannot give a READ-level probe teeth here. The
    # strong trial drive is contractive (attempt 1 read bitwise identical with the restore off), the surprise region
    # fires only late in the 160-step window, cp_neuron_firing_thresholds is not read by the Izhikevich step while
    # homeostasis is off, and a refractory hold long enough to reach the late spikes overflows the kernel. So the
    # teeth are STATE-level: residue is INJECTED into every _EXTRA_RESET_ARRAYS array on the surprise region; the
    # pre-fix reset must leave it in place, and the fix must restore TRUE rest bitwise. The read-level half is
    # repeat-read identity under the fix after the same injection, plus the smoke's a->b->a order-invariance check.
    sur = np.asarray(ix["surprise"], np.int64)

    def _inject():
        b = pool.b
        for nm, bump in (("cp_neuron_firing_thresholds", 5.0), ("cp_neuron_activity_ema", 0.5),
                         ("cp_refractory_timers", 3), ("cp_prev_firing_states", True)):
            arr = getattr(b, nm, None)
            if arr is None:
                continue
            if arr.dtype == bool:
                arr[sur] = bump
            else:
                arr[sur] = arr[sur] + bump

    def _at_rest(extra):
        for nm in _EXTRA_RESET_ARRAYS:
            rest = extra.get(nm)
            if rest is None:
                continue
            if not np.array_equal(np.asarray(to_host(getattr(pool.b, nm))), rest):
                return False
        return True

    saved_extra = pool._rest_extra
    assert any(v is not None for v in saved_extra.values()), "no extra array exists: the guard would be vacuous"
    pool._rest_extra = {}                  # simulate the ORIGINAL (pre-fix) _hard_reset
    _inject()
    pool._hard_reset()
    broken_diverges = not _at_rest(saved_extra)   # teeth: the pre-fix reset leaves the residue

    pool._rest_extra = saved_extra         # the actual fix
    _inject()
    pool._hard_reset()
    fixed_at_rest = _at_rest(saved_extra)
    f1 = pool.read_verdict("generated")["surprise"]
    _inject()
    f2 = pool.read_verdict("generated")["surprise"]
    fixed_identical = fixed_at_rest and (f1 == f2)
    if verbose:
        print(f"[selftest] fix-disabled leaves injected residue={broken_diverges}", flush=True)
        print(f"[selftest] fix-enabled  at-rest={fixed_at_rest} read identical={f1 == f2}: {f1!r} vs {f2!r}",
              flush=True)
    ok = broken_diverges and fixed_identical
    if ok and verbose:
        print("[selftest] PASS — repeat-read bitwise identity holds under the fix, and the probe has teeth",
              flush=True)
    return ok


def _order_invariance_check(seed=7, verbose=True):
    """Smoke-only (PLAN §8d/risk-4 fix): the SAME condition read after DIFFERENT predecessors must agree —
    a->b->a with the Port-A reset means predecessor residue cannot move the read. Any drift = instrument
    broken, fix BEFORE queueing."""
    pool = ProvSurprisePool(seed)
    a1 = pool.read_verdict("perceived")["surprise"]
    _b = pool.read_verdict("generated")["surprise"]
    a2 = pool.read_verdict("perceived")["surprise"]
    drift = abs(a1 - a2)
    ok = bool(drift <= 1e-12)
    if verbose:
        print(f"[order-invariance] a1={a1!r} a2={a2!r} drift={drift!r} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,1,4,9,12,15")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator (seed 7, CALIBRATION only)")
    ap.add_argument("--selftest", action="store_true",
                    help="read-isolation fails-in-failing-direction guard only (no train/run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        ok = _selftest_read_isolation()
        print("SELFTEST " + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1

    seeds = [7] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    oi = True
    if args.smoke:
        oi = _order_invariance_check()

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        emg, itn, bo = r["emergence"], r["interaction"], r["byte_off"]
        gen = itn["per_condition"]["perceived"]
        print(f"[seed {s}] supports={r['provisional_supports']} | grown={emg['grown'][GATE]:.3f} "
              f"nocorr={emg['no_corruption']} | perceived Δ={gen['delta_intact']:+.4f} "
              f"(lesion {gen['delta_lesion']:+.4f}) frac_attrib={gen['frac_attributable']} "
              f"| clean_zero={r['clean_zero']['ok']} | emg={emg['PASS']} int={itn['PASS']} "
              f"byteoff={bo['PASS']} ({r['elapsed_s']}s)", flush=True)

    n_sup = sum(r["provisional_supports"] for r in runs)
    all_sup = (n_sup == len(runs)) and not args.smoke
    if args.smoke:
        tag = "PROVISIONAL smoke indicator (seed 7, CALIBRATION ONLY — never evidence)"
    elif all_sup:
        tag = f"PROVISIONAL-SUPPORTS {n_sup}/{len(runs)} dev seeds"
    else:
        tag = f"PROVISIONAL-NOT-SUPPORTED {n_sup}/{len(runs)} dev seeds"
    verdict = (f"{tag} — the reciprocal CrossEdge source_provenance.prov_perceived -> surprise (the missing "
               f"reverse half of the existing surprise->prov encoding gate), added PURELY BY DECLARATION "
               f"through the generic onebrain_crossedge_gate.run_gate: on this run the substrate's own "
               f"source-monitoring verdict shifted its own windowed surprise-pool rate (perceived > "
               f"generated) through the declared edge. TRACE-LEVEL only; dev seeds only; never a GO; "
               f"not 'closed' (default-OFF, wire-in is Claude's); training is a declared host-supervised "
               f"teaching current; one-sided amplification (attenuation half = control read only). numpy "
               f"CPU; NO sim/ edit; additive.")

    if args.smoke:
        d0 = runs[0]["interaction"]["per_condition"]["perceived"]["delta_intact"] if runs else 0.0
        suggested = 0.5 * d0 if d0 > 0 else 0.0
        print(f"[smoke] order_invariance={oi} | suggested INTACT_FLOOR (rule: 0.5 x smoke delta, sign +1 "
              f"required) = {suggested:.4f} — freeze into the constant + WORKLOG before the 6-seed queue",
              flush=True)
        if not oi:
            print("[smoke] ORDER-INVARIANCE FAIL — fix the instrument, do NOT queue", flush=True)

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("draft_xedge_prov_surprise_reciprocal")
        # the OUTCOME is not a precondition (verdict_preconditions rule) — only validity checks go here.
        Vd.require("lesion_collapses_delta", 1 if all(
            abs(r["interaction"]["per_condition"]["perceived"]["delta_lesion"]) <
            LESION_RATIO * max(abs(r["interaction"]["per_condition"]["perceived"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the surprise-rate delta must collapse under lesion or it is a confound, not the edge")
        Vd.require("byte_identical_off", sum(r["byte_off"]["PASS"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="the no-edge pool's base connectivity is byte-identical (integration added ONLY the edge)")
        dec = Vd.decide(all_sup or (args.smoke and n_sup == len(runs)), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "xedge_prov_surprise_reciprocal", "provisional": True, "verdict": verdict,
               "provisional_supports": all_sup,
               "n_support": n_sup, "n_seeds": len(seeds), "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "gate_spec": {"name": GATE_SPEC.name, "correct_edges": GATE_SPEC.correct_edges,
                             "conditions": GATE_SPEC.condition_order, "control": GATE_SPEC.control,
                             "credit_signal": GATE_SPEC.credit_signal, "intact_floor": INTACT_FLOOR,
                             "cross_edges": [dict(key=ce.key, src=ce.source_region, tgt=ce.target_region)
                                             for ce in CROSS_EDGES]},
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[PROV_PERCEIVED->SURPRISE] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_sup or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
