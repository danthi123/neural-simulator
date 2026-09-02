"""One-brain INTEGRATION cross-edge #1 — D2 SURPRISE -> E2 WORLD-MODEL: an ERROR-GATED FORWARD-MODEL UPDATE.

THE EDGE (design rank #1, `2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`). The E2
affective world-model (`worldmodel_production_organ.py` / `_affective_world_model_derisk.py`, 6/6 GO) declares its
own honest residual: "TEACHER-DRIVEN: the transition is LEARNED (Hebbian co-fire) but not self-organized from
conversation" -- its `state -> pred_{pos,neg}` valence transition is trained ONCE by a host schedule
(`train_transition`) and then FROZEN; nothing lets the model RE-LEARN online, and nothing makes that re-learning
CONDITIONAL on the model actually being wrong. This runner de-risks the edge that closes that residual: the
world-model's OWN spiking prediction-error unit (its `surprise_{pos,neg}` pools -- the D2-class expectation-
violation signal INSIDE the predictive-coding circuit) becomes the THIRD FACTOR that GATES plasticity on the
existing `state -> pred` transition. The forward model updates ITSELF, but ONLY when it is surprised.

THE MECHANISM (per update turn, on a target state s0 that currently predicts valence v0):
  1. PREDICTION+ASSERTION READ: cue s0 (establish the top-down prediction), then drive s0 + the OBSERVED valence;
     read the surprise pools' windowed firing `surp_hz = rate(surprise_pos)+rate(surprise_neg)`. This is the
     brain's OWN error signal for this turn (a `cp_firing_states` read, never a host compare of obs vs pred).
  2. THIRD-FACTOR GATE: the transition Hebbian window OPENS iff `surp_hz >= gate_threshold` -- the plasticity is
     gated by the spiking prediction-error, exactly the shipped surprise-gates-plasticity pattern
     (`2026-09-01-surprise-episodic-encode-decision-crossedge-GO.md`; Lisman & Grace 2005 novelty->DA->LTP gate;
     Rao & Ballard 1999 error drives generative-model learning).
  3. UPDATE (only if gated open): co-fire state s0 with the pred pool matching the OBSERVED valence (the
     environment delivers the observed valence as drive -- the legitimate sensory boundary, EXACTLY how
     `train_transition`/`run_update_on_error` already teach), Hebbian strengthens `state[s0] -> pred_{observed}`
     (bounded by hebbian_max_weight), then the gate RE-FREEZES so every read is a frozen forward pass.

WHY THIS IS ERROR-GATED, NOT A HOST SCHEDULE (the load-bearing distinction, tested by three intact/lesion arms
run through the IDENTICAL gating code):
  * SURPRISING sequence (observed valence VIOLATES the current prediction): surprise FIRES -> gate opens -> the
    transition to the observed pool GROWS -> the prediction SHIFTS toward the new observation -> surprise falls
    silent -> the gate self-closes (predictive-coding error-minimisation; learning STOPS once the model agrees).
  * EXPECTED sequence (observed valence CONFIRMS the prediction) under the SAME gating code: surprise stays
    silent (the prediction cancels the observation) -> the gate NEVER opens -> NO update. The ONLY thing that
    differs from the surprising arm is whether the brain's error unit fired -- so the learning is owned by the
    neural surprise, not by "the flag is on".
  * LESION (`SURPRISE_WORLDMODEL_LESION=1`): the obs->surprise sensory-drive edges are zeroed, so the prediction-
    error unit CANNOT fire -> `surp_hz ~ 0` -> the gate never opens -> no update, even on the surprising
    sequence. SPECIFICITY control: on that same lesioned circuit, HOST-FORCING the gate open still updates the
    transition (the plasticity pathway is intact) -- proving the lost update is due to the severed surprise
    SIGNAL, not a disabled transition.
  * BYTE-OFF (`SURPRISE_WORLDMODEL_UPDATE=0`): the current teacher-driven path -- the model is FROZEN after its
    initial training (no online update runs at all). The transition weights and the queryable prediction are
    byte-identical to the built+trained organ, and the circuit connectivity is byte-identical (the edge adds NO
    synapse -- it is a third-factor GATE on the existing transition, host gating logic around neural spikes).

WHAT IS NEURAL vs THE DECLARED HOST BOUNDARY (honest, per docs/TERMS.md):
  * NEURAL: the surprise SIGNAL (a `cp_firing_states[surprise]` rate); the transition weights (Hebbian-plastic,
    the substrate's own rule with Miller-MacKay competition); the queryable prediction (a two-pool spike-rate
    difference). The GATE is DRIVEN by the neural surprise (lesioning it removes the update).
  * DECLARED BOUNDARY (NOT `self-organized`): the teach DIRECTION is the OBSERVED valence delivered as a sensory
    drive (the environment boundary, identical to the shipped organ's own training); the gate THRESHOLD is a
    build-time host calibration (the same boundary `surprise_production_organ` declares: "the DECISION ... is a
    threshold on that spiking rate"); WHICH state is the target and WHEN turns arrive are host/teacher scaffold.
    So this is `host-supervised`/`error-GATED`, NOT `self-organized`.

DE-RISK ONLY: reuse-by-import of `_affective_world_model_derisk`; NO sim/ edit; NO production wiring; NO default
flip; additive. numpy CPU for the smoke (routes off the GPU); the 6-seed verify runs on the GPU queue.

Run:
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_worldmodel_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_worldmodel_derisk --calibrate --seed 7
  SIM_BACKEND=numpy python -m research.runners._crossedge_surprise_worldmodel_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_crossedge_surprise_worldmodel_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # bare invocation stays on CPU; the queued cupy run overrides via env
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import statistics as _st
import time
from pathlib import Path

import numpy as np

from sim.backend import get_backend, to_host
from tools.lab import attributable_to, lever

from research.runners._affective_world_model_derisk import (
    build_world_model_circuit,
    train_transition,
    _drive_read,
    _hard_reset,
    _idx,
    _valence_map,
)

# ── protocol constants (world-model read protocol; inherited from the E2 organ's validated operating point) ──
N_STATES = 6
N_REPS = 22           # initial (teacher-scheduled) transition training reps — the E2 organ default
UPDATE_TURNS = 16     # online error-gated update turns presented for the target state
HOLD = 40             # Hebbian co-fire window per gated update step (train_transition's own window)
CUE_PA = 1000.0       # state cue (must cross every state block's per-neuron threshold; E2's verified value)
OBS_PA = 400.0        # observed-valence sensory drive (E2's decoupled value: surprise fires MODERATELY)
TEACH_PA = 1000.0     # teach-pool drive during a gated update step (train_transition's value)
PRE_STEPS = 60        # prediction pre-phase (top-down prediction settles before the assertion volley)
READ_STEPS = 60       # surprise assertion-read window
GATE_FRAC = 0.35      # gate threshold = expected + GATE_FRAC*(violated-expected): stays open until surprise is
                       # ~65% cancelled (the prediction substantially shifted), then self-closes
TARGET_STATE = 0      # s0 (declared: WHICH state is the target is host/teacher scaffold)

# GO floors. Frozen here; a 6-seed cupy soak is the decisive test (this smoke is an indicator).
SHIFT_FLOOR = 5.0     # the surprising-arm prediction must shift at least this far (Hz) toward the observed valence
                       # (the effect is ~100Hz, ~25x the ~4Hz read-noise floor — so this is a wide margin, not tight)
WGROW_MULT = 3.0      # the observed-pool transition weight must grow to >= this multiple of its post-train baseline
WGROW_ABS = 0.10      # ... AND grow by at least this absolute amount (a real, not floating-point, change)
ATTRIB_MIN = 0.80     # fraction of the shift attributable to the surprise-gated update (vs lesion / vs expected)


REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  connection-matrix helpers (orientation-robust; backend-safe — modify cp_connections.data in place via xp,
#  the run_gate.lesion_cross_edges house style, so this works identically on numpy and cupy sparse matrices).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _region_idx_set(bridge, name):
    return set(int(i) for i in _idx(bridge, name))


def _coo_rowcol(bridge):
    coo = bridge.cp_connections.tocoo()
    return np.asarray(to_host(coo.row)), np.asarray(to_host(coo.col))


def _mask_between(bridge, row, col, pre_idx, post_idx):
    """Orientation-robust boolean mask over cp_connections.data for edges between two index sets (either
    (pre->post) or (post->pre), because this engine's COO orientation is not asserted anywhere)."""
    pre = np.fromiter(pre_idx, dtype=np.int64) if not isinstance(pre_idx, np.ndarray) else pre_idx
    post = np.fromiter(post_idx, dtype=np.int64) if not isinstance(post_idx, np.ndarray) else post_idx
    m1 = np.isin(row, pre) & np.isin(col, post)
    m2 = np.isin(row, post) & np.isin(col, pre)
    return m1 | m2


def _state_block_idx(bridge, meta, s):
    st = _idx(bridge, "state")
    blk = meta["blk"]
    return st[s * blk:(s + 1) * blk]


def _transition_weight(bridge, meta, s, pred_pool):
    """Mean weight of the transition edges state-block[s] <-> pred_pool (orientation-robust). This is the
    world-model's OWN forward-model weight the surprise gate teaches — no new synapse is introduced."""
    row, col = _coo_rowcol(bridge)
    pre = _state_block_idx(bridge, meta, s)
    post = _idx(bridge, pred_pool)
    m = _mask_between(bridge, row, col, pre, post)
    if not m.any():
        return 0.0
    data = np.asarray(to_host(bridge.cp_connections.data))
    return float(data[m].mean())


def _zero_obs_to_surprise(bridge, xp):
    """LESION: zero the obs_{pos,neg} -> surprise_{pos,neg} sensory-drive edges, so the prediction-error unit
    cannot fire -> the third-factor gate never opens. Leaves the state<->pred transition pathway fully intact
    (the specificity control host-forces the gate open on this same circuit and still updates). Backend-safe:
    modifies cp_connections.data in place (keeps the CSR structure). Returns the count zeroed."""
    row, col = _coo_rowcol(bridge)
    obs = _region_idx_set(bridge, "obs_pos") | _region_idx_set(bridge, "obs_neg")
    surp = _region_idx_set(bridge, "surprise_pos") | _region_idx_set(bridge, "surprise_neg")
    obs_a = np.fromiter(obs, dtype=np.int64)
    surp_a = np.fromiter(surp, dtype=np.int64)
    m = _mask_between(bridge, row, col, obs_a, surp_a)
    data = np.asarray(to_host(bridge.cp_connections.data)).copy()
    n = int(m.sum())
    data[m] = 0.0
    bridge.cp_connections.data = xp.asarray(data, dtype=bridge.cp_connections.data.dtype)
    return n


def _conn_hash(bridge):
    d = np.asarray(to_host(bridge.cp_connections.data))
    return (int(d.shape[0]), float(np.abs(d).sum()), hash(d.tobytes()))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  spiking reads (reuse the E2 organ's drive/read primitives verbatim — same operating point)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _pred_diff(bridge, idx_map, xp, s):
    """Queryable prediction for state s: rate(pred_pos) - rate(pred_neg) (a spike-rate difference)."""
    _hard_reset(bridge)
    pr = _drive_read(bridge, idx_map, {"state": (s, CUE_PA)}, PRE_STEPS, xp, ["pred_pos", "pred_neg"])
    return float(pr["pred_pos"] - pr["pred_neg"])


def _read_surprise(bridge, idx_map, xp, s, obs_sign):
    """The brain's OWN error signal for observing `obs_sign` in state s: prediction pre-phase (state cue), then
    assertion (state + observed valence), read cp_firing_states[surprise_{pos,neg}]."""
    obs_region = "obs_pos" if obs_sign > 0 else "obs_neg"
    _hard_reset(bridge)
    r = _drive_read(bridge, idx_map, {"state": (s, CUE_PA), obs_region: (None, OBS_PA)},
                    READ_STEPS, xp, ["surprise_pos", "surprise_neg"],
                    pre_drives={"state": (s, CUE_PA)}, pre_steps=PRE_STEPS)
    return float(r["surprise_pos"] + r["surprise_neg"])


def _gated_update_step(bridge, cfg, idx_map, xp, s, obs_sign, gate_threshold, *, force_open=False):
    """ONE error-gated update turn. Read the surprise; if it clears the gate (or is host-forced), OPEN the
    transition Hebbian window for exactly one co-fire of state s + pred_{observed}, then RE-FREEZE. Returns
    (surp_hz, gate_open)."""
    surp_hz = _read_surprise(bridge, idx_map, xp, s, obs_sign)
    gate_open = bool(force_open or (surp_hz >= gate_threshold))
    if gate_open:
        teach = "pred_pos" if obs_sign > 0 else "pred_neg"
        cfg.enable_hebbian_learning = True                 # open the gate for exactly this credited step
        _hard_reset(bridge)
        _drive_read(bridge, idx_map, {"state": (s, CUE_PA), teach: (None, TEACH_PA)}, HOLD, xp, [])
        cfg.enable_hebbian_learning = False                # re-freeze -> every read is a frozen forward pass
    return surp_hz, gate_open


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  one seed
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _build_and_train(seed):
    """Build the E2 world-model circuit and run its OWN (teacher-scheduled) initial transition training, then
    freeze — exactly the state the production organ is in before any conversation. Returns everything the
    update arms need."""
    xp, _ = get_backend()
    bridge, cfg, meta = build_world_model_circuit(seed, n_states=N_STATES)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in REGIONS}
    vmap = _valence_map(seed, meta["n_states"])
    train_transition(bridge, cfg, idx_map, meta, xp, vmap, n_reps=N_REPS)
    cfg.enable_hebbian_learning = False
    return xp, bridge, cfg, meta, idx_map, vmap


def _calibrate_gate(bridge, idx_map, xp, s, v0):
    """Build-time gate threshold from the frozen circuit: expected (obs==v0) surprise low, violated (obs==-v0)
    high; threshold sits GATE_FRAC into the gap (declared host boundary — a threshold on a spiking rate)."""
    exp_hz = _read_surprise(bridge, idx_map, xp, s, v0)
    vio_hz = _read_surprise(bridge, idx_map, xp, s, -v0)
    thr = exp_hz + GATE_FRAC * max(vio_hz - exp_hz, 0.0)
    return thr, exp_hz, vio_hz


def _run_update_arm(seed, mode):
    """mode: 'surprising' | 'expected' | 'lesion' | 'lesion_forced' | 'byteoff'."""
    xp, bridge, cfg, meta, idx_map, vmap = _build_and_train(seed)
    s = TARGET_STATE
    v0 = int(vmap[s])
    obs_pool = "pred_neg" if v0 > 0 else "pred_pos"        # the OBSERVED valence's pool (opposite of v0)
    same_pool = "pred_pos" if v0 > 0 else "pred_neg"       # the initially-trained pool (predicts v0)

    thr, exp_hz, vio_hz = _calibrate_gate(bridge, idx_map, xp, s, v0)

    pred_before = _pred_diff(bridge, idx_map, xp, s)
    w_obs_before = _transition_weight(bridge, meta, s, obs_pool)
    w_same_trained = _transition_weight(bridge, meta, s, same_pool)   # reference: the trained forward weight

    conn_before = _conn_hash(bridge)

    lesion_n = 0
    if mode in ("lesion", "lesion_forced"):
        lesion_n = _zero_obs_to_surprise(bridge, xp)
    # surprising world for every update arm except 'expected' (which confirms the current prediction)
    obs_sign = v0 if mode == "expected" else -v0
    force = (mode == "lesion_forced")

    surp_trace, gate_opens = [], 0
    if mode != "byteoff":
        for _t in range(UPDATE_TURNS):
            surp_hz, opened = _gated_update_step(bridge, cfg, idx_map, xp, s, obs_sign, thr, force_open=force)
            surp_trace.append(round(surp_hz, 2))
            gate_opens += int(opened)

    pred_after = _pred_diff(bridge, idx_map, xp, s)
    w_obs_after = _transition_weight(bridge, meta, s, obs_pool)
    conn_after = _conn_hash(bridge)

    # signed shift of the prediction TOWARD the observed valence (obs_sign). For 'expected', obs_sign==v0, so a
    # (non-)shift toward the already-predicted valence; for the others obs_sign==-v0 (toward the new observation).
    shift = (pred_after - pred_before) * obs_sign
    flipped = bool(np.sign(pred_after) == np.sign(-v0) and abs(pred_after) > 1e-6) if mode != "expected" else False
    return {
        "mode": mode, "seed": int(seed), "v0": v0, "target_state": s,
        "gate_threshold": round(thr, 3), "calib_expected_hz": round(exp_hz, 3), "calib_violated_hz": round(vio_hz, 3),
        "lesion_edges_zeroed": lesion_n,
        "pred_before": round(pred_before, 3), "pred_after": round(pred_after, 3), "shift_toward_obs": round(shift, 3),
        "pred_sign_flipped": flipped,
        "w_obs_before": round(w_obs_before, 4), "w_obs_after": round(w_obs_after, 4),
        "w_same_trained": round(w_same_trained, 4),
        "w_obs_grew_frac": (round(w_obs_after / w_same_trained, 3) if w_same_trained > 1e-6 else None),
        "gate_opens": gate_opens, "surprise_trace": surp_trace,
        "surprise_max": (max(surp_trace) if surp_trace else 0.0),
        "conn_byte_identical_pre_post": bool(conn_before == conn_after),
    }


def run_seed(seed, verbose=True):
    t0 = time.time()
    surprising = _run_update_arm(seed, "surprising")
    expected = _run_update_arm(seed, "expected")
    lesion = _run_update_arm(seed, "lesion")
    lesion_forced = _run_update_arm(seed, "lesion_forced")
    byteoff = _run_update_arm(seed, "byteoff")

    # (a) EMERGENCE: the observed-pool transition weight GREW (structural fact, exact) from its near-zero
    #     post-train baseline via the substrate's own Hebbian rule, gated OPEN by the surprise, AND the queryable
    #     prediction shifted toward the observation. The update SELF-LIMITS (surprise cancels as the model
    #     assimilates), so the weight need not reach the fully-trained magnitude — the growth + the functional
    #     shift are the load-bearing facts, not a fraction of the trained pool.
    dw = surprising["w_obs_after"] - surprising["w_obs_before"]
    wgrow = bool(surprising["gate_opens"] >= 1
                 and surprising["w_obs_after"] >= WGROW_MULT * max(surprising["w_obs_before"], 1e-6)
                 and dw >= WGROW_ABS)
    shift_surprising = surprising["shift_toward_obs"]
    emergence = bool(wgrow and shift_surprising >= SHIFT_FLOOR)

    # (b) ERROR-GATED SELECTIVITY: the EXPECTED arm (same gating code) does NOT update — the gate never opens
    #     because the brain is not surprised (the ONLY difference from the surprising arm). Anchored on the
    #     UNAMBIGUOUS structural facts: no gate opened AND the transition weight is exactly unchanged.
    expected_no_update = bool(expected["gate_opens"] == 0
                              and abs(expected["w_obs_after"] - expected["w_obs_before"]) < 1e-9)

    # (b) LESION-VANISHES: severing the obs->surprise signal keeps surprise silent -> the gate never opens ->
    #     no update (weight exactly unchanged), even on the surprising sequence.
    lesion_no_update = bool(lesion["gate_opens"] == 0 and lesion["surprise_max"] < 1.0
                            and abs(lesion["w_obs_after"] - lesion["w_obs_before"]) < 1e-9)

    # SPECIFICITY: on the SAME lesioned circuit, host-forcing the gate open still updates -> the transition
    # pathway is intact; the lost update is due to the severed surprise SIGNAL, not disabled plasticity.
    lesion_specific = bool(lesion_forced["shift_toward_obs"] >= SHIFT_FLOOR and lesion_forced["gate_opens"] > 0)

    # (c) ATTRIBUTABLE: the surprising-arm shift is owned by the surprise-gated update — vs the lesion (severed
    #     signal) AND vs the expected arm (same machinery, no surprise).
    frac_vs_lesion = attributable_to(f"seed{seed} surprise-gated update shift vs lesion",
                                     shift_surprising, lesion["shift_toward_obs"])
    frac_vs_expected = attributable_to(f"seed{seed} surprise-gated update shift vs expected (same gating code)",
                                       shift_surprising, expected["shift_toward_obs"])
    attributable = bool(frac_vs_lesion is not None and frac_vs_lesion >= ATTRIB_MIN
                        and frac_vs_expected is not None and frac_vs_expected >= ATTRIB_MIN)

    # (d) BYTE-OFF: flag off = the current frozen path — no update runs (gate_opens==0), so the transition
    #     weights are EXACTLY unchanged from the built+trained circuit and the circuit connectivity is
    #     byte-identical (the edge adds NO synapse — it is a third-factor GATE, host logic around neural spikes).
    #     Asserted in the data (exact weight + connectivity-hash equality), not inferred, and NOT keyed on the
    #     prediction read (which carries a few-Hz read-noise floor irrelevant to whether any weight moved).
    byte_off = bool(byteoff["gate_opens"] == 0
                    and abs(byteoff["w_obs_after"] - byteoff["w_obs_before"]) < 1e-9
                    and byteoff["conn_byte_identical_pre_post"])

    go = bool(emergence and expected_no_update and lesion_no_update and lesion_specific
              and attributable and byte_off)

    if verbose:
        lever(f"seed{seed} prediction shift surprising->lesion",
              shift_surprising, lesion["shift_toward_obs"])
        print(f"[seed {seed}] GO={go} elapsed={time.time()-t0:.1f}s")
        print(f"    thr={surprising['gate_threshold']} (exp={surprising['calib_expected_hz']} "
              f"vio={surprising['calib_violated_hz']})  v0={surprising['v0']}")
        print(f"    SURPRISING  shift={surprising['shift_toward_obs']:+.2f} flip={surprising['pred_sign_flipped']} "
              f"w_obs {surprising['w_obs_before']}->{surprising['w_obs_after']} "
              f"(frac {surprising['w_obs_grew_frac']}) opens={surprising['gate_opens']}/{UPDATE_TURNS} "
              f"surp_trace[0:4]={surprising['surprise_trace'][:4]}")
        print(f"    EXPECTED    shift={expected['shift_toward_obs']:+.2f} opens={expected['gate_opens']} "
              f"surp_max={expected['surprise_max']:.2f}  -> no_update={expected_no_update}")
        print(f"    LESION      shift={lesion['shift_toward_obs']:+.2f} opens={lesion['gate_opens']} "
              f"surp_max={lesion['surprise_max']:.2f} zeroed={lesion['lesion_edges_zeroed']} "
              f"-> vanishes={lesion_no_update}")
        print(f"    LES_FORCED  shift={lesion_forced['shift_toward_obs']:+.2f} opens={lesion_forced['gate_opens']} "
              f"-> transition_intact={lesion_specific}")
        print(f"    BYTEOFF     shift={byteoff['shift_toward_obs']:+.2f} conn_identical="
              f"{byteoff['conn_byte_identical_pre_post']} -> byte_off={byte_off}")
        print(f"    emergence={emergence} attributable={attributable} "
              f"(vs_lesion={frac_vs_lesion} vs_expected={frac_vs_expected})")

    return {
        "seed": int(seed), "GO": go, "elapsed_s": round(time.time() - t0, 1),
        "emergence": emergence, "expected_no_update": expected_no_update,
        "lesion_no_update": lesion_no_update, "lesion_specific": lesion_specific,
        "attributable": attributable,
        "frac_attributable_vs_lesion": (None if frac_vs_lesion is None else round(frac_vs_lesion, 4)),
        "frac_attributable_vs_expected": (None if frac_vs_expected is None else round(frac_vs_expected, 4)),
        "byte_off": byte_off, "wgrow": wgrow,
        "arms": {"surprising": surprising, "expected": expected, "lesion": lesion,
                 "lesion_forced": lesion_forced, "byteoff": byteoff},
    }


def calibrate(seed):
    xp, bridge, cfg, meta, idx_map, vmap = _build_and_train(seed)
    s = TARGET_STATE
    v0 = int(vmap[s])
    thr, exp_hz, vio_hz = _calibrate_gate(bridge, idx_map, xp, s, v0)
    print(f"[calibrate seed={seed}] state={s} v0={v0:+d}  expected_surprise={exp_hz:.3f}Hz  "
          f"violated_surprise={vio_hz:.3f}Hz  gate_threshold={thr:.3f} (GATE_FRAC={GATE_FRAC})")
    return thr, exp_hz, vio_hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="2-seed indicator (42,43)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.seed)
        return 0

    seeds = [42, 43] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    runs = [run_seed(s) for s in seeds]
    n_go = sum(r["GO"] for r in runs)
    all_go = (n_go == len(runs))

    if args.smoke:
        tag = "SMOKE-GO (indicator; PARTIAL-pending-6seed-cupy-soak)" if all_go else "SMOKE-NO-GO"
    else:
        tag = "GO" if all_go else "NO-GO"
    verdict = (f"{tag} — D2 surprise (the world-model's OWN spiking prediction-error unit) GATES an online update "
               f"of the E2 forward model's state->pred transition: {n_go}/{len(runs)} seeds show the transition "
               f"GROW toward the observed valence on a SURPRISING sequence (prediction shifts), NO update on an "
               f"EXPECTED sequence (same gating code, surprise silent), the update VANISH when the obs->surprise "
               f"signal is lesioned (transition still plastic under host-forced gate -> specific), the shift "
               f"ATTRIBUTABLE to the surprise-gated update, and byte-identical-off. Teach direction + gate "
               f"threshold are declared host boundaries (NOT self-organized). numpy CPU; NO sim/ edit; additive.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("crossedge_surprise_worldmodel_derisk")
        Vd.require("all_seeds_go", n_go, expect=lambda x: x == len(runs),
                   note="emergence + expected-no-update + lesion-vanishes + specificity + attributable + byte-off")
        Vd.require("emergence_grows_transition", sum(r["emergence"] for r in runs), expect=lambda x: x == len(runs),
                   note="the observed-pool transition weight grows and the prediction shifts toward the observation")
        Vd.require("error_gated_selectivity", sum(r["expected_no_update"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="the EXPECTED arm (identical gating code) performs NO update — learning is gated by the "
                        "brain's own surprise, not by the flag")
        Vd.require("lesion_vanishes", sum(r["lesion_no_update"] for r in runs), expect=lambda x: x == len(runs),
                   note="severing obs->surprise closes the gate -> no update")
        Vd.require("lesion_specific_transition_intact", sum(r["lesion_specific"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="host-forcing the gate open on the lesioned circuit still updates -> the lost update is "
                        "the severed surprise SIGNAL, not disabled plasticity")
        Vd.require("byte_identical_off", sum(r["byte_off"] for r in runs), expect=lambda x: x == len(runs),
                   note="flag off = the frozen current path; no weight/prediction change; connectivity identical")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "crossedge_surprise_worldmodel_derisk", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "smoke": bool(args.smoke), "preconditions": preconditions,
               "constants": {"N_STATES": N_STATES, "N_REPS": N_REPS, "UPDATE_TURNS": UPDATE_TURNS, "HOLD": HOLD,
                             "GATE_FRAC": GATE_FRAC, "SHIFT_FLOOR": SHIFT_FLOOR, "WGROW_MULT": WGROW_MULT,
                             "WGROW_ABS": WGROW_ABS, "ATTRIB_MIN": ATTRIB_MIN, "TARGET_STATE": TARGET_STATE},
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[SURPRISE->WORLDMODEL error-gated update] VERDICT: {verdict}\n" + "=" * 100,
          flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
