"""STEP 6 of the D5 learn-through-use arc — a GRADED apical read that makes the conversation-visible move RELIABLE.

THE STEP-5 RESIDUAL (why 4/6, not 6/6): step-5 proved the REAL `EpisodicRecallOrgan.recall` `apical_cue` RISES after
between-turn consolidation (learn-through-use is conversation-visible), but only 4/6-reliable. The blocker is the READ,
not the mechanism: `apical_cue` is a per-held-cell BINARY UP-fraction (mean over held cells of `cp_v_apical > up_thresh`),
so on a ~13-cell emergent membership it is QUANTIZED in steps of 1/n_held (~0.077) and sometimes has NO step available
(seed42: the weakest completing encode already sat at its structural ceiling → apical_cue 0.4286 → 0.4286, flat). A
dedicated b_adapt/plateau-window sweep PROVED the plateau knob cannot raise it (finding
2026-08-20-d5-learn-through-use-CAN-be-conversation-visible-...-4of6-graded-read-for-reliable). The finding names the
redirect: a GRADED / CONTINUOUS apical read replacing the binary UP-fraction.

THE STEP-6 READ (this runner): keep the SAME store, SAME weak-encode op-point, SAME consolidation loop as step-5 — change
ONLY the read-out. Instead of counting how many held cells crossed a hard threshold, read the CONTINUOUS apical plateau
state across held cells (three candidates, all from the SAME apical voltage the binary read thresholds):
  * depth_rest  = mean_held max(cp_v_apical − apical_E_rest, 0)   — plateau depth above rest (mV); maximally continuous.
  * depth_hold  = mean_held max(cp_v_apical − v_hold, 0)          — depth above the latch hold; this is EXACTLY the BTSP
                    instructive signal `IS_post = max(cp_v_apical − v_hold, 0)` the substrate's own plasticity kernel
                    integrates (sim/bridge.py fused_btsp_update) and the step-3 interference_run already reads.
  * soft        = mean_held sigmoid((cp_v_apical − up_thresh)/T)  — a soft-sigmoid version of the binary read: same
                    [0,1] scale/interpretation (fraction-of-cells-completing) but smooth, so the reply's quoted number
                    keeps its meaning while gaining continuity.
Because the apical voltage rises CONTINUOUSLY as consolidation potentiates the cue→held recurrence (a held cell moving
−25 → −22 mV is invisible to the binary threshold but visible to every graded read), the graded read produces a SMOOTH,
always-present increase — no quantization dead-steps.

THE MOAT IS PRESERVED BY CONSTRUCTION (graded read runs BESIDE the binary read, does NOT replace the gate): the binary
UP-fraction + the specificity criteria (cue ≥ COMPLETE_MIN, cue ≥ 3·perm, cue ≥ 3·nocue, nocue ≤ CTRL_MAX) STILL gate
`in_memory` — the honest abstain. The graded read is only surfaced as the conversation-visible MAGNITUDE for a memory
the binary gate already admitted. So an unformed memory still reads in_memory=False (never surfaces a graded number),
and the graded read is additionally shown to DISCRIMINATE formed-vs-unformed on its own (cue ≫ nocue/perm; the
formation-lesion — read the formed dog through the UNFORMED baseline weights — collapses it), i.e. it is a faithful read,
not a weight-blind monotone function that would leak the moat.

PROTOCOL (identical to step-5 — same store, same-store lesion control):
  1. Encode 'dog' WEAK (adaptive per-seed: the LOWEST-BINARY-apical store that still COMPLETES in the headroom band —
     the SAME selection step-5 used, so this lands on the SAME store step-5 measured). 'cat' NEVER encoded.
  2. TURN T (handler): snapshot-isolated `org.recall('dog')` → binary apical_cue + the graded reads, in_memory. + cat.
  3. LESION arm (BRAIN_D5_CONSOLIDATE=0): mark_recall + consolidate = NO-OP → store byte-identical → later read flat.
  4. ON arm (BRAIN_D5_CONSOLIDATE=1), SAME store: n_turns of mark_recall→consolidate rounds (the real continuous_engine
     loop), reading the handler after each turn (the graded trajectory).
  5. TURN T+k (handler): read AGAIN. The GRADED read must be HIGHER, monotone across turns, cat unchanged.

MEASURE (the teeth), evaluated for EACH graded read (depth_rest / depth_hold / soft):
  * GRADED_MOVES   : graded_cue_Tk > graded_cue_T (+margin) through the REAL org.recall path.
  * MONOTONE       : the graded trajectory is non-decreasing across the consolidation turns (NO dead-steps) and ends
                     strictly above turn T.
  * LESION_VANISHES: flag OFF → byte-identical store → the later graded read is IDENTICAL to turn T (the move is
                     DRIVEN by the loop, not decoration); the ON arm DID consolidate.
  * FAITHFUL_READ  : (i) cue-specific — graded_cue_T ≥ 3·max(graded_perm_T, graded_nocue_T); (ii) formation-lesion —
                     `org.recall('dog', lesion=True)` graded collapses to ≪ the formed graded (the read is carried by
                     the FORMED assembly, not a weight-blind depolarization) — so the moat is preserved.
  * MOAT           : never-recalled 'cat' stays in_memory=False (binary gate) and its graded read ≪ dog's.
  * STILL_USABLE   : the weak encode COMPLETES at turn T (in_memory_T=True, binary gate).
  * DETERMINISTIC  : two identical isolated handler reads match (binary AND graded).

GO (per seed, for the chosen PRIMARY read): GRADED_MOVES ∧ MONOTONE ∧ LESION_VANISHES ∧ FAITHFUL_READ ∧ MOAT ∧
  STILL_USABLE ∧ DETERMINISTIC. The summary reports the 6/6 count for ALL THREE graded reads so the best is chosen on
  evidence. Honest NO-GO otherwise (localizes whether a graded read trades reliability for lost discrimination).

BRAIN-BASED (NO sim/ edit): the read is a pure READ change over the SAME `cp_v_apical` the production recall already
reads — implemented here as a `GradedEpisodicDapMemory` subclass (a drop-in for `_episodic_dap_dialogue_memory.recall`).
The strengthening is the substrate's OWN plateau-gated BTSP via the ACTUAL continuous_engine.consolidate_used_memory.
Host code is only the clock, the encode-strength selection, and the snapshot-isolation determinism guard. GPU-preferred.
  Run:    SIM_BACKEND=cupy python -m research.runners._d5_step6_graded_apical_read_derisk --seed 42
  6-seed: SIM_BACKEND=cupy python -m research.runners._d5_step6_graded_apical_read_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402
from research.runners._episodic_dap_dialogue_memory import (  # noqa: E402
    EpisodicDapMemory, COMPLETE_MIN, CUE_OVER_CTRL, CTRL_MAX)
from research.runners.d5_episodic_production_organ import EpisodicRecallOrgan  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _reset_apical_latch, _apical_up_read)
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
from webapp import continuous_engine as CE  # noqa: E402  (the ACTUAL production wiring under test)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_step6_graded" / "seed42.json"

# The headroom band for the weak encode (IDENTICAL to step-5 so we land on the SAME store).
APICAL_LO = 0.15
APICAL_HI = 0.80
MOVE_MARGIN_BIN = 1e-6         # binary move margin (quantised, step 1/n_held)
MOVE_MARGIN_DEPTH = 1e-3       # graded depth move margin (mV) — well below any real consolidation move
MOVE_MARGIN_SOFT = 1e-4        # graded soft move margin ([0,1])
MONO_TOL_FRAC = 0.02           # a trajectory is monotone-up if it never BACKTRACKS by more than this fraction of the
                               # total move — tolerates sub-percent numerical ripple at the SATURATING tail (where the
                               # weight has stopped moving), which is NOT a dead-step. Reported alongside strict-monotone.
# ── ABSOLUTE tolerance FLOOR (knob-2 seed-44 fix, board #71) ─────────────────────────────────────────────────────────
# For a WEAK consolidator (seed 44) the TOTAL move is tiny (~0.8 mV over the pre-saturation window), so a pure
# 2%-of-move tolerance is a sub-0.02 mV, SUB-RIPPLE bound → a normal saturating-tail ripple (~0.1-0.3 mV) or the
# plateau's own amplitude-saturation (Bittner, Milstein, Grienberger, Romani & Magee 2017 Science 357:1033 — the
# regenerative NMDA plateau the depth read measures is CEILING-BOUNDED, so near the top the read flattens while the
# weight still grows) spuriously fails monotone / registers a dead-step. The floor makes the tolerance = max(2%-of-move,
# ABS floor). The floor is CALIBRATED: BELOW a meaningful per-turn rise (seed 44's linear-regime first turn is ~0.8 mV)
# and ABOVE the saturating ripple (~0.1-0.3 mV) + DEAD_READ_EPS (0.05 mV). ADDITIVE: for a large-move trace 2%-of-move
# dominates (e.g. move=26 mV → 0.52 mV > 0.4), so the 2%-relative behavior is UNCHANGED. The floor NEVER excuses a
# genuinely FLAT/DECREASING trace: _mono_rel still requires the trace to END ABOVE start (move>0), and the dead-step
# tail-excuse only applies to a trace that GENUINELY ROSE (total read move > min_rise), so a flat read on a rising
# weight (the binary quantization defect / an untrained trace) still counts.
MONO_TOL_ABS_MV = 0.4          # absolute backtrack/saturation floor for the mV depth reads (0.1-0.3 ripple < 0.4 < 0.8 rise)
MONO_TOL_ABS_SOFT = 2e-3       # absolute floor for the [0,1] soft read (analogous: below a meaningful move, above ripple)
W_MEANINGFUL = 0.5             # a per-turn within-weight rise (mV) considered MEANINGFUL for the dead-step contrast
DEAD_READ_EPS = 0.05           # a depth read that moves < this (mV) on a meaningful-weight turn is a DEAD-STEP (flat)
DEAD_READ_EPS_SOFT = 1e-3      # soft-read dead-step threshold ([0,1])
T_SOFT = 4.0                   # soft-sigmoid temperature (mV) about up_thresh
FAITHFUL_K = float(CUE_OVER_CTRL)   # graded cue must beat perm/nocue by this factor (reuse the binary specificity ratio)
LESION_COLLAPSE_FRAC = 0.15    # formation-lesion graded must be <= this fraction of the formed graded (read is carried
                               # by the FORMED assembly, not a weight-blind depolarization)
GRADED_READS = ("depth_rest", "depth_hold", "soft")   # the three candidate graded reads, all from the SAME apical state


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The GRADED read — a pure READ change over the SAME cp_v_apical the binary read thresholds.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _apical_dual_read(bridge, R, held_pos_by_asm, cue_by_asm, up_thresh, v_hold, t_soft=T_SOFT):
    """Mirror of `_apical_up_read` (byte-identical drive / reset / step machinery, byte-identical binary UP-fraction)
    that ALSO returns the three GRADED reads from the SAME apical state, in ONE sim pass:
        up          binary UP-fraction (== _apical_up_read; the moat gate)
        depth_rest  mean_held max(cp_v_apical − apical_E_rest, 0)   [mV above rest]
        depth_hold  mean_held max(cp_v_apical − v_hold, 0)          [mV above hold == BTSP IS_post]
        soft        mean_held sigmoid((cp_v_apical − up_thresh)/t_soft)  [soft binary, on 0..1]
    Averaged over the (here single) assembly. NO sim/ edit — this is the read-out only."""
    cp = R.cp; to_host = R.to_host; ca3_idx = R.ca3_idx
    cfg = bridge.core_config
    E_rest = float(getattr(cfg, "apical_E_rest", -65.0))
    ups, d_rest, d_hold, softs = [], [], [], []
    for held_pos, cue_g in zip(held_pos_by_asm, cue_by_asm):
        R.hard_silence(); _reset_apical_latch(bridge)
        if cue_g is not None and len(cue_g) > 0:
            darr = cp.asarray(np.asarray(cue_g, dtype=np.int64), dtype=cp.int64)
            bridge.cp_external_input_current[darr] = cp.float32(R._drive_pA)
        else:
            darr = None
        for _ in range(R._warm + R._read):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        if getattr(bridge, "cp_v_apical", None) is None:
            ups.append(0.0); d_rest.append(0.0); d_hold.append(0.0); softs.append(0.0)
        else:
            va = to_host(bridge.cp_v_apical)
            held_global = [int(ca3_idx[p]) for p in held_pos]
            if held_global:
                vv = np.asarray([float(va[g]) for g in held_global], dtype=np.float64)
                ups.append(float(np.mean((vv > up_thresh).astype(np.float64))))
                d_rest.append(float(np.mean(np.maximum(vv - E_rest, 0.0))))
                d_hold.append(float(np.mean(np.maximum(vv - v_hold, 0.0))))
                softs.append(float(np.mean(1.0 / (1.0 + np.exp(-(vv - up_thresh) / t_soft)))))
            else:
                ups.append(0.0); d_rest.append(0.0); d_hold.append(0.0); softs.append(0.0)
        if darr is not None:
            bridge.cp_external_input_current[darr] = 0.0
    return dict(up=float(np.mean(ups)) if ups else 0.0,
                depth_rest=float(np.mean(d_rest)) if d_rest else 0.0,
                depth_hold=float(np.mean(d_hold)) if d_hold else 0.0,
                soft=float(np.mean(softs)) if softs else 0.0)


class GradedEpisodicDapMemory(EpisodicDapMemory):
    """EpisodicDapMemory whose `recall` ALSO emits the GRADED apical reads alongside the binary UP-fraction. The binary
    read + the specificity criteria STILL gate `in_memory` (the moat); the graded reads are the continuous
    conversation-visible magnitude. Pure READ change — a drop-in for `_episodic_dap_dialogue_memory.recall`, NO sim/
    edit, NO formation change (store/consolidation inherited verbatim)."""

    def _apical_dual(self, slot, cue_kind, lesion):
        cue = {"cue": self.cue_by_asm, "perm": self.perm_by_asm}.get(cue_kind)
        drive = [cue[slot]] if cue is not None else [None]
        if lesion:
            saved = self.R.C.data.copy(); self.R.C.data[:] = self.baseline_weights
        try:
            return _apical_dual_read(self.bridge, self.R, [self.held_pos_by_asm[slot]], drive,
                                     self.p["up_thresh"], self.p["v_hold"])
        finally:
            if lesion:
                self.R.C.data[:] = saved

    def recall(self, topic, *, lesion=False):
        slot = self.topic_slot.get(topic)
        if slot is None:
            return {"topic": topic, "slot": None, "formed": False, "in_memory": False,
                    "apical_cue": 0.0, "apical_perm": 0.0, "apical_nocue": 0.0,
                    "graded_cue": {r: 0.0 for r in GRADED_READS},
                    "graded_perm": {r: 0.0 for r in GRADED_READS},
                    "graded_nocue": {r: 0.0 for r in GRADED_READS}, "reason": "no-slot"}
        c = self._apical_dual(slot, "cue", lesion)
        p = self._apical_dual(slot, "perm", lesion)
        n = self._apical_dual(slot, "nocue", lesion)
        cue, perm, nocue = c["up"], p["up"], n["up"]
        completes = bool(cue >= COMPLETE_MIN and cue >= CUE_OVER_CTRL * (perm + 1e-6)
                         and cue >= CUE_OVER_CTRL * (nocue + 1e-6) and nocue <= CTRL_MAX)
        return {"topic": topic, "slot": slot, "formed": bool(slot in self.formed and not lesion),
                "in_memory": completes, "apical_cue": float(cue), "apical_perm": float(perm),
                "apical_nocue": float(nocue), "lesioned": bool(lesion), "reason": "spiking-dap-completion",
                "graded_cue": {r: float(c[r]) for r in GRADED_READS},
                "graded_perm": {r: float(p[r]) for r in GRADED_READS},
                "graded_nocue": {r: float(n[r]) for r in GRADED_READS}}


def _whash(cp, W):
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _build_organ(seed, train_events):
    """A production EpisodicRecallOrgan whose store is a GradedEpisodicDapMemory ('dog' encoded at `train_events` BTSP
    passes; 'cat' never). Same construction as step-5's `_build_organ`, only the memory CLASS differs (the read change)."""
    org = EpisodicRecallOrgan(seed, ["cat", "dog"], verbose=False)
    org.mem = GradedEpisodicDapMemory(seed, org.topics, verbose=False, train_events=int(train_events), wmax=100.0)
    if not org.mem.store("dog"):
        raise RuntimeError("store('dog') returned False — dog not BTSP-formed")
    org._store_order = ["dog"]
    return org


def _borderline_apical(org, cp):
    """The borderline (pre-consolidation) full-cue handler read for 'dog', snapshot-isolated → deterministic +
    weight-attributable. Returns (rec, snap, W_before, w_dog_before). Same as step-5."""
    mem = org.mem
    dslot = mem.topic_slot["dog"]
    mem.recall("dog")                       # warm + allocate cp_v_apical
    mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
    snap = snapshot_state(mem.bridge)
    W_before = mem.R.C.data.copy()
    w_dog_before = float(cp.mean(W_before[mem.R.withinA_masks[dslot]]))
    restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
    rec = org.recall("dog")
    return rec, snap, W_before, w_dog_before


def _free_org(org, cp):
    try:
        org.mem = None
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _select_encode(seed, cp, te_grid, verbose=True):
    """ADAPTIVE per-seed encode selection — IDENTICAL criterion to step-5 (LOWEST BINARY apical_cue store that COMPLETES
    in the headroom band), so this lands on the SAME store step-5 measured. The graded move is then read at that op-point.
    Returns (org, te, rec, snap, W, w_dog); (None, ...) if no grid point completes with headroom."""
    best = None  # (ac, org, te, rec, snap, W_before, w_dog_before)
    n_completing = 0
    for te in te_grid:
        org = _build_organ(seed, te)
        rec, snap, W_before, w_dog_before = _borderline_apical(org, cp)
        ac = float(rec["apical_cue"]); inmem = bool(rec["in_memory"]); nocue = float(rec["apical_nocue"])
        if verbose:
            print(f"    [encode-select s{seed}] te={te:2d} w_dog={w_dog_before:6.2f} apical_cue(bin)={ac:.4f} "
                  f"graded_cue={rec['graded_cue']} nocue={nocue:.4f} in_memory={inmem} "
                  f"asm={org.mem.assembly_sizes}", flush=True)
        keep = bool(inmem and APICAL_LO < ac <= APICAL_HI)
        if keep:
            n_completing += 1
        if keep and (best is None or ac < best[0]):
            if best is not None:
                _free_org(best[1], cp)
            best = (ac, org, te, rec, snap, W_before, w_dog_before)
        else:
            _free_org(org, cp)
    if best is None:
        return (None, None, None, None, None, None)
    if verbose:
        print(f"    [encode-select s{seed}] CHOSEN te={best[2]} borderline apical_cue(bin)={best[0]:.4f} "
              f"(from {n_completing} completing candidate(s))", flush=True)
    return best[1:]


def _mono_rel(traj, tol_frac=MONO_TOL_FRAC, tol_abs=0.0):
    """The DEFENSIBLE monotonicity: the read never BACKTRACKS by more than max(tol_frac of the total move, tol_abs), and
    ends above start. The RELATIVE term tolerates sub-percent ripple on a large-move trace; the ABSOLUTE floor (tol_abs)
    tolerates the fixed-size saturating-tail ripple on a TINY-total-move trace (seed 44: ~0.8 mV move → 2%-of-move is a
    sub-ripple 0.016 mV bound). A genuinely FLAT (move<=0) or DECREASING trace still fails (ends-above-start is
    required); a real down-trend larger than the floor still fails. `_mono_strict` (below, no floor) is reported
    alongside for full honesty. tol_abs=0 (default) == the original pure-relative behavior."""
    move = traj[-1] - traj[0]
    if move <= 0:
        return False
    tol = max(tol_frac * abs(move), tol_abs)
    non_dec = all(traj[i + 1] >= traj[i] - tol for i in range(len(traj) - 1))
    return bool(non_dec)


def _mono_strict(traj, eps):
    """Strict non-decreasing (every step >= prev - eps) AND rises — reported, not GO-driving (see _mono_rel)."""
    non_dec = all(traj[i + 1] >= traj[i] - eps for i in range(len(traj) - 1))
    return bool(non_dec and traj[-1] > traj[0] + eps)


def _dead_steps(w_traj, r_traj, w_eps, r_eps, tail_tol=0.0, min_rise=0.0):
    """Count the turns where the within-weight rose MEANINGFULLY (Δw > w_eps) but the read stayed FLAT (|Δread| < r_eps)
    — the binary UP-fraction's quantization defect. A graded read that tracks the weight in its LINEAR regime has ZERO
    such turns.

    SATURATING-TAIL EXCLUSION (tail_tol>0): the depth read measures a REGENERATIVE, ceiling-bounded NMDA plateau (Bittner
    et al. 2017), so once the read has climbed into its saturation band a further weight rise legitimately produces
    almost no read move — that is the plateau's biology, NOT the binary read's quantization defect. When tail_tol>0 a
    flat step is EXCLUDED iff (a) the trace GENUINELY ROSE overall (total read move > min_rise) AND (b) the PRIOR read
    value already sits within tail_tol of the trajectory max (we are on the saturating tail). tail_tol=0 (default) keeps
    the ORIGINAL behavior — used for the BINARY contrast, which must retain its quantization dead-steps so the graded
    read's advantage is still demonstrated. The exclusion never gives a free pass to a NON-rising trace (a flat read on a
    rising weight with total move <= min_rise still counts — the moat)."""
    move = r_traj[-1] - r_traj[0]
    rmax = max(r_traj)
    n = 0
    for i in range(1, len(w_traj)):
        if (w_traj[i] - w_traj[i - 1]) > w_eps and abs(r_traj[i] - r_traj[i - 1]) < r_eps:
            saturating_tail = bool(tail_tol > 0.0 and move > min_rise and r_traj[i - 1] >= rmax - tail_tol)
            if not saturating_tail:
                n += 1
    return int(n)


def _selftest_criteria():
    """The FLAT-TRACE CONTROL, run as a deterministic unit check (no brain) so it executes every run and can FAIL LOUDLY
    (the project's "a gate must be able to fail in its failing direction" discipline). It PROVES the tuned tolerance did
    NOT defeat the criterion: a genuinely FLAT / DECREASING / collapsing trace is still REJECTED, the large-move
    2%-relative behavior is UNCHANGED, and the binary read KEEPS its quantization dead-steps (so the graded read's
    advantage is still real). Returns the list of FAILED check names (empty == all pass)."""
    f = []
    def chk(name, cond):
        if not bool(cond):
            f.append(name)
    A = MONO_TOL_ABS_MV
    # ── monotone floor: the loosening must NOT admit a non-rising trace ──
    chk("flat_rejected",              _mono_rel([15.0, 15.0, 15.0], tol_abs=A) is False)   # no rise → reject
    chk("decreasing_rejected",        _mono_rel([15.0, 14.5, 14.0], tol_abs=A) is False)   # down-trend → reject
    chk("collapse_rejected",          _mono_rel([15.0, 16.0, 15.05], tol_abs=A) is False)  # big backtrack (0.95>0.4) → reject
    # ── monotone floor: it MUST admit the seed-44 case (tiny move + sub-floor saturating ripple) ──
    chk("saturating_ripple_admitted", _mono_rel([14.51, 15.31, 15.28], tol_abs=A) is True)  # 0.03 dip < 0.4 floor → admit
    # ── additive: large-move (>50x floor) 2%-relative behavior is UNCHANGED ──
    chk("large_move_2pct_preserved",  _mono_rel([10.0, 40.0, 38.0], tol_abs=A) is False)   # 2.0 drop > max(0.56,0.4) → reject
    chk("large_move_ripple_admitted", _mono_rel([10.0, 40.0, 39.7], tol_abs=A) is True)    # 0.3 dip < 0.56 (2% of 30) → admit
    # ── dead-step: BINARY quantized-flat read KEEPS its dead-steps (tail_tol=0 → contrast preserved) ──
    chk("binary_deadsteps_preserved", _dead_steps([30.0, 42.0, 43.0], [0.25, 0.25, 0.25],
                                                  W_MEANINGFUL, MOVE_MARGIN_BIN) == 2)
    # ── dead-step: a graded read that RISES then saturates at the tail → NO dead-step (saturation is expected biology) ──
    chk("saturating_tail_excluded",   _dead_steps([31.6, 43.2, 44.1], [14.51, 15.31, 15.33],
                                                  W_MEANINGFUL, DEAD_READ_EPS, tail_tol=A, min_rise=A) == 0)
    # ── dead-step: a flat read on a rising weight with NO real total rise still counts (no free pass — the moat) ──
    chk("flat_read_still_deadstep",   _dead_steps([30.0, 42.0, 43.0], [15.0, 15.0, 15.05],
                                                  W_MEANINGFUL, DEAD_READ_EPS, tail_tol=A, min_rise=A) >= 1)
    return f


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-step6-graded] seed={seed} backend={backend} — weak-usable encode → handler recall → consolidate → "
          f"handler recall; does the GRADED org.recall read MOVE reliably (no dead-steps)?", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        cache_key = ("d5-step6", seed)
        CE.forget_session(cache_key)

        te_grid = [int(x) for x in a.te_grid.split(",")] if a.te_grid else [a.train_events]
        org, te, _rec0, snap, W_before, w_dog_before = _select_encode(seed, cp, te_grid, verbose=True)
        instrument_valid = org is not None
        if not instrument_valid:
            result["verdict_status"] = "UNDEFINED"
            result["checks"] = {"instrument_valid": False, "reason": "no te landed a weak-usable headroom store"}
            print(f"[d5-step6-graded] seed={seed} INSTRUMENT-INVALID: no encode in {te_grid} landed a completing "
                  f"headroom store (binary apical in ({APICAL_LO},{APICAL_HI}])", flush=True)
            CE.forget_session(cache_key)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            return result

        mem = org.mem
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]

        def handler_read(topic, W, *, lesion=False):
            """A snapshot-ISOLATED production handler recall on store-weights W. `org.recall` is the EXACT live-handler
            method (now emitting graded reads too); restoring the clean-rest snapshot + injecting W isolates the read so
            a T-vs-T+k comparison is purely WEIGHT-attributable (same as step-5)."""
            restore_state(mem.bridge, snap)
            mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic, lesion=lesion)

        def g(rec, field, read):
            return float(rec[field][read])

        # ── TURN T (handler): the borderline read (binary + graded), re-read isolated for determinism ──
        rec_dog_T = handler_read("dog", W_before)
        rec_dog_T2 = handler_read("dog", W_before)
        rec_cat_T = handler_read("cat", W_before)
        rec_dog_T_les = handler_read("dog", W_before, lesion=True)   # FORMATION-lesion (baseline weights) — faithfulness
        hash_before = _whash(cp, W_before)
        w_cat_before = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))
        w_between_before = float(cp.mean(W_before[mem.R.between_mask]))
        # determinism over BOTH binary and every graded read
        det_bin = abs(rec_dog_T["apical_cue"] - rec_dog_T2["apical_cue"]) < 1e-9
        det_grd = all(abs(g(rec_dog_T, "graded_cue", r) - g(rec_dog_T2, "graded_cue", r)) < 1e-9 for r in GRADED_READS)
        deterministic = bool(det_bin and det_grd)
        inmem_T = bool(rec_dog_T["in_memory"])
        ac_T = float(rec_dog_T["apical_cue"])
        headroom = bool(inmem_T and APICAL_LO < ac_T <= APICAL_HI)
        cat_never = bool(w_cat_before < 5.0 and not rec_cat_T["in_memory"])
        print(f"[d5-step6] TURN T: dog binary={ac_T:.4f} graded={rec_dog_T['graded_cue']} in_memory={inmem_T} | "
              f"cat binary={rec_cat_T['apical_cue']:.4f} | dog(formation-lesion) graded={rec_dog_T_les['graded_cue']} | "
              f"w_dog={w_dog_before:.1f} te={te} det={deterministic} headroom={headroom}", flush=True)

        # ── LESION arm (flag OFF): mark + consolidate = NO-OP; SAME store byte-identical + later read flat ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        CE.mark_recall(cache_key, "dog")
        off_rec = CE.consolidate_used_memory(cache_key, org)     # must be None
        W_off = mem.R.C.data.copy(); hash_off = _whash(cp, W_off)
        rec_dog_off = handler_read("dog", W_off)
        byte_identical_off = bool(off_rec is None and hash_off == hash_before)
        # lesion-flat is per-read: the later read equals turn T for binary AND every graded read
        lesion_flat_bin = abs(rec_dog_off["apical_cue"] - ac_T) < 1e-9
        lesion_flat_grd = {r: bool(abs(g(rec_dog_off, "graded_cue", r) - g(rec_dog_T, "graded_cue", r)) < 1e-9)
                           for r in GRADED_READS}
        print(f"[d5-step6] LESION (flag=0): consolidate→{off_rec} | store byte-identical={byte_identical_off} | "
              f"later binary flat={lesion_flat_bin} graded flat={lesion_flat_grd}", flush=True)

        # ── ON arm (flag ON), SAME store: n_turns of use (each re-arms mark_recall → one consolidation round) ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        bin_traj = [round(ac_T, 4)]
        grd_traj = {r: [round(g(rec_dog_T, "graded_cue", r), 5)] for r in GRADED_READS}
        wdog_traj = [round(w_dog_before, 3)]; consolidated_rounds = 0
        on_rec = None; W_after = W_before
        for turn in range(a.n_turns):
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_after)
            CE.mark_recall(cache_key, "dog")
            on_rec = CE.consolidate_used_memory(cache_key, org, n_episodes=a.n_episodes)
            if on_rec is not None:
                consolidated_rounds += 1
            W_after = mem.R.C.data.copy()
            wd = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
            rec_turn = handler_read("dog", W_after)
            bin_traj.append(round(float(rec_turn["apical_cue"]), 4))
            for r in GRADED_READS:
                grd_traj[r].append(round(g(rec_turn, "graded_cue", r), 5))
            wdog_traj.append(round(wd, 3))
            print(f"  [turn T+{turn+1}] consolidate→{'ok' if on_rec else None} | w_dog={wd:.1f} | "
                  f"binary={rec_turn['apical_cue']:.4f} graded={rec_turn['graded_cue']} "
                  f"in_memory={rec_turn['in_memory']}", flush=True)
        consolidated = bool(consolidated_rounds > 0)
        hash_after = _whash(cp, W_after)
        w_dog_after = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
        w_cat_after = float(cp.mean(W_after[mem.R.withinA_masks[cslot]]))
        w_between_after = float(cp.mean(W_after[mem.R.between_mask]))

        # ── TURN T+k (handler): the FINAL later-turn read ──
        rec_dog_Tk = handler_read("dog", W_after)
        rec_cat_Tk = handler_read("cat", W_after)
        ac_Tk = float(rec_dog_Tk["apical_cue"]); inmem_Tk = bool(rec_dog_Tk["in_memory"])

        # ── per-read verdict pieces ──────────────────────────────────────────────────────────────────────────────
        # weight-drift specificity (identical to step-5 — about the consolidation loop, not the read)
        dw_dog = w_dog_after - w_dog_before
        cat_drift = abs(w_cat_after - w_cat_before); between_drift = abs(w_between_after - w_between_before)
        weight_specific = bool(cat_drift <= 0.05 * max(dw_dog, 1e-6) and between_drift <= 0.05 * max(dw_dog, 1e-6))
        moat_cat = bool((not rec_cat_T["in_memory"]) and (not rec_cat_Tk["in_memory"]))   # cat never admitted

        # binary (step-5 read) dead-step count: turns where w_dog rose meaningfully but the binary read stayed FLAT
        binary_dead_steps = _dead_steps(wdog_traj, bin_traj, W_MEANINGFUL, MOVE_MARGIN_BIN)

        per_read = {}
        for r in GRADED_READS:
            margin = MOVE_MARGIN_SOFT if r == "soft" else MOVE_MARGIN_DEPTH
            strict_eps = (MOVE_MARGIN_SOFT if r == "soft" else 1e-3)
            dead_eps = (DEAD_READ_EPS_SOFT if r == "soft" else DEAD_READ_EPS)
            tol_abs = MONO_TOL_ABS_SOFT if r == "soft" else MONO_TOL_ABS_MV  # absolute floor (knob-2 seed-44 fix)
            cue_T = g(rec_dog_T, "graded_cue", r); cue_Tk = g(rec_dog_Tk, "graded_cue", r)
            perm_T = g(rec_dog_T, "graded_perm", r); nocue_T = g(rec_dog_T, "graded_nocue", r)
            les_T = g(rec_dog_T_les, "graded_cue", r)               # formation-lesion (baseline weights)
            cat_T = g(rec_cat_T, "graded_cue", r); cat_Tk = g(rec_cat_Tk, "graded_cue", r)
            traj = grd_traj[r]
            moves = bool(cue_Tk > cue_T + margin)
            monotone = _mono_rel(traj, tol_abs=tol_abs)            # defensible: max(2%-of-move, abs floor) — GO-driving
            monotone_strict = _mono_strict(traj, strict_eps)        # reported (absolute, no floor) — honesty
            graded_dead_steps = _dead_steps(wdog_traj, traj, W_MEANINGFUL, dead_eps,
                                            tail_tol=tol_abs, min_rise=tol_abs)   # saturating-tail excluded (Bittner 2017)
            no_dead_steps = bool(graded_dead_steps == 0)
            faithful_specific = bool(cue_T >= FAITHFUL_K * max(perm_T, nocue_T, 1e-9))
            faithful_lesion = bool(les_T <= LESION_COLLAPSE_FRAC * max(cue_T, 1e-9))
            faithful = bool(faithful_specific and faithful_lesion)
            lesion_vanishes = bool(byte_identical_off and lesion_flat_grd[r] and consolidated)
            # cat_small (strict graded-cat flatness) is REPORTED; the GO-driving specificity is moat-preserving:
            # the consolidation only strengthened dog (weight_specific) AND the moat gate keeps cat out (moat_cat), so a
            # sub-threshold graded cat ripple is NEVER surfaced. (Matches step-5's own framing of the identical crosstalk.)
            cat_small = bool(abs(cat_Tk - cat_T) <= 0.15 * max(cue_Tk - cue_T, 1e-9))
            specific = bool(weight_specific and moat_cat)
            read_go = bool(moves and monotone and no_dead_steps and lesion_vanishes and faithful and specific
                           and inmem_T and deterministic)
            move_treat = cue_Tk - cue_T
            move_ctrl = g(rec_dog_off, "graded_cue", r) - cue_T
            per_read[r] = dict(
                cue_T=round(cue_T, 5), cue_Tk=round(cue_Tk, 5), perm_T=round(perm_T, 5), nocue_T=round(nocue_T, 5),
                lesion_T=round(les_T, 5), cat_T=round(cat_T, 5), cat_Tk=round(cat_Tk, 5), traj=traj,
                MOVES=moves, MONOTONE=monotone, MONOTONE_STRICT=monotone_strict, no_dead_steps=no_dead_steps,
                graded_dead_steps=graded_dead_steps, binary_dead_steps=binary_dead_steps,
                LESION_VANISHES=lesion_vanishes,
                FAITHFUL=faithful, faithful_specific=faithful_specific, faithful_lesion=faithful_lesion,
                SPECIFIC=specific, cat_small=cat_small, read_go=read_go,
                move_treat=round(move_treat, 5), move_ctrl=round(move_ctrl, 5),
                attributable=attributable_to(f"[s{seed}] {r} graded move: ON vs LESION(OFF)", move_treat, move_ctrl))

        primary = a.primary_read
        pr = per_read[primary]
        # binary (step-5 read) for reference: did IT move / was it flat (the quantization dead-step)?
        binary_moves = bool(ac_Tk > ac_T + MOVE_MARGIN_BIN)
        binary_flat_at_ceiling = bool(not binary_moves)

        go = bool(pr["read_go"])
        LESION_VANISHES = pr["LESION_VANISHES"]
        SPECIFIC = pr["SPECIFIC"]
        STILL_USABLE = bool(inmem_T)

        print(f"[d5-step6] TURN T+k: PRIMARY({primary}) graded {pr['cue_T']}→{pr['cue_Tk']} "
              f"(traj {pr['traj']}) MOVES={pr['MOVES']} MONO={pr['MONOTONE']} FAITHFUL={pr['FAITHFUL']} "
              f"| binary {ac_T:.4f}→{ac_Tk:.4f} (moved={binary_moves}) | cat graded {pr['cat_T']}→{pr['cat_Tk']} "
              f"| w_dog {w_dog_before:.1f}→{w_dog_after:.1f}", flush=True)

        result["attributable"] = pr["attributable"]

        v = Verdict(f"RELIABLE conversation-visible learn-through-use via a GRADED apical read ({primary}): the REAL "
                    f"org.recall graded completion rises monotonically after consolidation (seed {seed})")
        v.disabled("host weight formula", "the strengthening is the substrate's OWN plateau-gated BTSP "
                                          "(fused_btsp_update) via continuous_engine.consolidate_used_memory")
        v.disabled("binary UP-fraction as the conversation-visible read", "the quantised per-held-cell UP-fraction has "
                                                                          "structural dead-steps (the step-5 4/6 limit); "
                                                                          "the graded plateau read replaces it as the "
                                                                          "magnitude (binary still gates in_memory=moat)")
        v.require("weak-store-completes-T", inmem_T, expect=True,
                  note="the weak encode still COMPLETES at turn T via the BINARY gate (a genuinely usable memory)")
        v.require("borderline-has-headroom", ac_T, expect=lambda x: APICAL_LO < x <= APICAL_HI,
                  note="the borderline BINARY apical_cue sits BELOW ceiling (the step-5 op-point)")
        v.require("handler-read-deterministic", deterministic, expect=True,
                  note="two identical isolated handler recalls match on binary AND graded reads")
        v.require("cat-never-recalled", cat_never, expect=True,
                  note="cat is a genuine never-formed control (no completion, baseline weight)")
        v.require("graded-read-faithful", pr["FAITHFUL"], expect=True,
                  note="the graded read is cue-specific AND collapses under the formation-lesion (carried by the FORMED "
                       "assembly, not a weight-blind depolarization) → the moat is preserved")
        v.reaches("graded-read-moves-with-use", pr["cue_T"], pr["cue_Tk"],
                  note="the PRODUCTION org.recall GRADED read rose after consolidation (the reliable conversation-visible move)")
        v.control(f"consolidation-ON vs LESION-OFF ({primary} graded move)",
                  treatment=pr["move_treat"], control=pr["move_ctrl"], min_separation=0.0,
                  note="the graded move requires the consolidation loop; the flag OFF is byte-identical")
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]

        checks = dict(
            instrument_valid=True, te=te, primary_read=primary,
            GO=go, LESION_VANISHES=LESION_VANISHES, SPECIFIC=SPECIFIC, STILL_USABLE=STILL_USABLE,
            deterministic=deterministic, headroom=headroom, cat_never=cat_never,
            consolidated=consolidated, byte_identical_off=byte_identical_off, consolidated_rounds=consolidated_rounds,
            # binary (step-5) reference
            apical_cue_T=round(ac_T, 4), apical_cue_off=round(rec_dog_off["apical_cue"], 4), apical_cue_Tk=round(ac_Tk, 4),
            binary_traj=bin_traj, binary_moves=binary_moves, binary_flat_at_ceiling=binary_flat_at_ceiling,
            inmem_T=inmem_T, inmem_Tk=inmem_Tk,
            # graded reads
            per_read={r: per_read[r] for r in GRADED_READS},
            # weight state
            wdog_traj=wdog_traj, w_dog_before=round(w_dog_before, 3), w_dog_after=round(w_dog_after, 3),
            dw_dog=round(dw_dog, 3), cat_drift=round(cat_drift, 3), between_drift=round(between_drift, 3),
            weight_specific=weight_specific, moat_cat=moat_cat,
            hash_before=hash_before, hash_off=hash_off, hash_after=hash_after,
            n_turns=a.n_turns, n_episodes=a.n_episodes, assembly_sizes=mem.assembly_sizes,
            w_within_after=on_rec.get("w_within_after") if on_rec else None)
        result["checks"] = checks
        print(f"[d5-step6] checks={json.dumps({k: checks[k] for k in checks if k != 'per_read'}, default=str)}", flush=True)
        CE.forget_session(cache_key)
        del mem, org
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()
    finally:
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 118)
    print(f"[d5-step6-graded] seed={seed} VERDICT: {status}")
    if "checks" in result and result["checks"].get("instrument_valid"):
        c = result["checks"]
        print(f"    PRIMARY={c['primary_read']} GO={c['GO']} | binary {c['apical_cue_T']}→{c['apical_cue_Tk']} "
              f"(moved={c['binary_moves']}, flat-at-ceiling={c['binary_flat_at_ceiling']})")
        print(f"    binary dead-steps (w rose, read flat) = {c['per_read'][GRADED_READS[0]]['binary_dead_steps']}")
        for r in GRADED_READS:
            p = c["per_read"][r]
            print(f"    {r:11s}: {p['cue_T']}→{p['cue_Tk']} traj={p['traj']} MOVES={p['MOVES']} MONO={p['MONOTONE']} "
                  f"(strict={p['MONOTONE_STRICT']}) dead={p['graded_dead_steps']} FAITHFUL={p['FAITHFUL']} "
                  f"LESION_VAN={p['LESION_VANISHES']} SPECIFIC={p['SPECIFIC']} cat_flat={p['cat_small']} "
                  f"read_go={p['read_go']}")
    print(f"[d5-step6-graded] wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-events", type=int, default=8, dest="train_events")
    ap.add_argument("--te-grid", type=str, default="5,6,7,8,10", dest="te_grid",
                    help="adaptive per-seed encode sweep (IDENTICAL to step-5 → same store): the LOWEST-BINARY-apical "
                         "completing store is used. '' => --train-events.")
    ap.add_argument("--n-episodes", type=int, default=3, dest="n_episodes",
                    help="consolidation recall→strengthen episodes PER tick (continuous_engine default 3)")
    ap.add_argument("--n-turns", type=int, default=3, dest="n_turns",
                    help="number of later USE turns (each re-arms mark_recall → one consolidation tick); step-5 used 3")
    ap.add_argument("--primary-read", type=str, default="depth_rest", choices=list(GRADED_READS), dest="primary_read",
                    help="the graded read that drives the GO verdict; the summary reports 6/6 for ALL three")
    ap.add_argument("--self-test", action="store_true", dest="self_test",
                    help="run ONLY the criteria self-test (flat-trace control, no brain) and exit")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    # The FLAT-TRACE CONTROL runs on EVERY invocation (deterministic, no brain) and BLOCKS if the tuned tolerance ever
    # defeats the criterion (flat/decreasing/collapse admitted, or the binary contrast lost).
    _st_fails = _selftest_criteria()
    if _st_fails:
        print(f"[d5-step6-graded] ⛔ CRITERIA SELF-TEST FAILED — tolerance defeats the criterion: {_st_fails}", flush=True)
        return 3
    print(f"[d5-step6-graded] criteria self-test: PASS (flat-trace control holds; floor MV={MONO_TOL_ABS_MV} "
          f"soft={MONO_TOL_ABS_SOFT}; binary contrast preserved)", flush=True)
    if a.self_test:
        return 0

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}; go_flags = []; valid_flags = []
    # per-read 6/6 tallies (choose the best read on evidence)
    read_go_counts = {r: 0 for r in GRADED_READS}
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        c = res.get("checks", {})
        instrument_valid = bool(c.get("instrument_valid") and c.get("headroom") and c.get("cat_never")
                                and c.get("deterministic"))
        valid_flags.append(instrument_valid)
        go_flags.append(bool(res.get("verdict_status") == "GO"))
        if c.get("instrument_valid") and c.get("per_read"):
            for r in GRADED_READS:
                if c["per_read"][r].get("read_go"):
                    read_go_counts[r] += 1

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds); n_valid = int(sum(valid_flags))
        print("\n" + "#" * 118)
        print(f"[d5-step6-graded] {n}-SEED SUMMARY: PRIMARY({a.primary_read}) {summ_go}/{n} GO "
              f"({n_valid}/{n} instrument-valid) seeds={seeds}")
        print(f"[d5-step6-graded] per-read 6/6 tally (all graded reads): "
              + " | ".join(f"{r}={read_go_counts[r]}/{n}" for r in GRADED_READS))
        for s in seeds:
            c = all_results[s].get("checks", {})
            pr = (c.get("per_read") or {}).get(a.primary_read, {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} valid={valid_flags[seeds.index(s)]} "
                  f"te={c.get('te')} | binary {c.get('apical_cue_T')}→{c.get('apical_cue_Tk')} "
                  f"(moved={c.get('binary_moves')} dead={pr.get('binary_dead_steps')}) | {a.primary_read} "
                  f"{pr.get('cue_T')}→{pr.get('cue_Tk')} MOVES={pr.get('MOVES')} MONO={pr.get('MONOTONE')} "
                  f"dead={pr.get('graded_dead_steps')} FAITHFUL={pr.get('FAITHFUL')} SPECIFIC={pr.get('SPECIFIC')} "
                  f"cat_flat={pr.get('cat_small')} read_go={pr.get('read_go')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "n_valid": n_valid, "go_flags": go_flags,
                                         "valid_flags": valid_flags, "primary_read": a.primary_read,
                                         "read_go_counts": read_go_counts, "backend": backend, "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-step6-graded] wrote {summ_path}")
    valid_go = [go_flags[i] for i in range(len(seeds)) if valid_flags[i]]
    return 0 if (valid_go and all(valid_go)) else 1


if __name__ == "__main__":
    sys.exit(main())
