"""ONE-BRAIN CROSS-EDGE C1 (design rank #1, `2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`)
-- D2 SURPRISE -> E2 WORLD-MODEL: an ERROR-GATED ONLINE FORWARD-MODEL UPDATE, wired onto the ALREADY-LIVE production
world-model organ (`worldmodel_production_organ.py`, Gate-B/E2, default-ON in `webapp/server.py`).

THE GAP the de-risk closed (`research/runners/_crossedge_surprise_worldmodel_derisk.py`, 6/6-seed GO,
`research/findings/raw/_crossedge_surprise_worldmodel_6seed.json`). E2's own declared residual: "TEACHER-DRIVEN: the
transition is LEARNED (Hebbian co-fire) but not self-organized from conversation" -- `state -> pred_{pos,neg}` is
trained ONCE at build (`train_transition`) then FROZEN; nothing re-learns online, and nothing makes re-learning
CONDITIONAL on the model being wrong. The de-risk validated the missing edge: the world-model's OWN spiking
prediction-error pools (`surprise_{pos,neg}`, the SAME D2-class circuit `worldmodel_production_organ.py` already
reads for the live surprise notice) become the THIRD FACTOR that gates ONE Hebbian co-fire step of `state->pred`
toward the OBSERVED valence -- the forward model updates itself, but only when it is surprised.

WHY THIS WIRES ONTO THE ALREADY-LIVE ORGAN, NOT A SEPARATE COPY (unlike the surprise->episodic module's isolated-pool
design, `onebrain_xedge_surprise_episodic_production.py`). C1's mechanism is literally "the organ's OWN D2 unit gates
an update of the organ's OWN transition" -- a diagnostic on a shadow copy would not exercise anything a real user's
turn reads from. So this module operates DIRECTLY on the process-shared `WorldModelProductionOrgan` singleton
(`worldmodel_production_organ.get_organ()`) via its already-built `._st` circuit dict, reusing the de-risk's own
validated primitives (`_drive_read`/`_hard_reset` from `_affective_world_model_derisk`, `TEACH_PA`/`HOLD`/`CUE_PA`
from `_crossedge_surprise_worldmodel_derisk`) rather than reimplementing the co-fire step. Mutating a process-shared
organ from real per-turn conversation is an ESTABLISHED pattern here (PART 3 per-turn live plasticity,
`onebrain_xedge_production.credit_live_turn`, already grows a shared cross-edge from real chat turns process-wide) --
not a new hazard this module introduces.

THE GATE IS A DEDICATED READ, NOT A REUSE OF THE NOTICE'S THRESHOLD (corrected 2026-09-02 — the first design reused
`webapp/server.py`'s existing `sj["surprised"]` from `worg.read_surprise`, which gates off the organ's `self.
threshold`; that threshold is calibrated ONCE across BOTH `state_pos`/`state_neg` via a mean_exp/min_vio midpoint,
tuned for the surprise-NOTICE decision — an offline 6-seed self-test caught it opening spuriously on 3/6 seeds'
EXPECTED/confirming arms, which should never gate). `crossedge_gated_update` below instead runs its OWN dedicated
`_read_surprise` (the de-risk's own instrument, imported byte-for-byte) against a DEDICATED per-target-state
threshold from `_gate_threshold_for` (the de-risk's own `_calibrate_gate`/`GATE_FRAC=0.35` recipe, calibrated once
per state and cached) — the SAME validated 6/6-seed-GO calibration, not the notice's. This is a second spiking read
per violating turn (the existing notice's `sj` plus this gate's own `surp_hz`), a deliberate cost for correctness
over reusing an instrument calibrated for a different decision. Then, iff the gate clears, it performs ONLY the
teach co-fire (`train_transition`'s own per-step primitive: state cue + the OBSERVED pool, `HOLD=40` steps,
`TEACH_PA=1000.0`, imported byte-for-byte from the de-risk module).

LOAD-BEARING + lesion-attributable (mirrors the de-risk's own instrument): holding a sequence of real VIOLATING turns
through the live handler grows the observed-pool `state->pred` transition weight and shifts the queryable prediction
(`"how is it going"`) toward the newly observed valence; an EXPECTED (confirming) sequence, the SAME gating code,
produces no update; `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION=1` zeroes the obs->surprise sensory-drive edges
on the SAME shared circuit the existing D2 notice reads -- INTENDED, not a side-effect: C1 rides the ONE shared D2
signal both consume, so lesioning it silences the surprise notice too. This is only ever set in a dedicated
flip-verify/lesion-diagnostic subprocess, never in real production traffic (the edge itself stays default-OFF, and
the lesion flag is a second, separate opt-in on top of that).

DECLARED RESIDUALS (honest, carried from the de-risk + this wire-in):
  * TEACH DIRECTION + WHICH state: the observed valence's pool is a host-delivered sensory drive (the environment
    boundary, identical to the organ's own initial training) and the target state is the organ's own persistence-prior
    `_state_for(context_sign)` -- host/teacher scaffold, NOT self-organized (de-risk's own declared boundary).
  * SECOND SPIKING READ PER VIOLATING TURN: the gate's own `_read_surprise` runs alongside the existing notice's
    `read_surprise` (see above) — a real, small, additional compute cost per violating turn, in exchange for a
    dedicated, validated calibration instead of reusing a threshold tuned for a different decision.
  * PROCESS-SHARED MUTATION: the update mutates the ONE process-global organ (all sessions read the SAME transition);
    a personalized per-session forward model is a later rung, exactly PART-3 per-turn plasticity's own precedent.
  * MAGNITUDE: a single real turn's gated update is small (one co-fire step, not the de-risk's 16-turn aggregate);
    the flip-verify below drives a REPEATED real-turn sequence (mirroring how a real surprising conversation would
    unfold) to clear a visible, lesion-attributable shift within a bounded number of live handler calls.

GUARDED, DEFAULT-OFF, BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL` gates the whole thing (unset/0/
false/no/off => the existing worldmodel block runs EXACTLY as today; this module is never imported into that code
path's execution when off). A build/read failure DEGRADES to "no diagnostic field" (never crashes a turn -- mirrors
every prior xedge production module's `ensure_built`/try-except convention).

Run (offline self-verify; 0 Claude tokens, CPU numpy):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_surprise_worldmodel_production --grow --seeds 42 \
      --out research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_selftest.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — OFF. The owner-gated flip to default-ON is a SEPARATE, later step (never autonomous),
# mirroring every prior xedge production module's `_XEDGE_*_DEFAULT_ON` convention (e.g. `onebrain_xedge_
# surprise_episodic_production._XEDGE_SE_DEFAULT_ON`).
_XEDGE_SWM_DEFAULT_ON = False


def xedge_surprise_worldmodel_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL` in {1,true,yes,on} -> the error-gated online update of the LIVE
    world-model organ's `state->pred` transition is active (a real violating turn can grow the transition toward
    the observed valence). Unset/{0,false,no,off} -> byte-identical to today's frozen-after-build organ. Default
    per `_XEDGE_SWM_DEFAULT_ON` (OFF)."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL")
    if v is None:
        return _XEDGE_SWM_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def xedge_surprise_worldmodel_lesioned() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION` in {1,true,yes,on} -> zero the obs_{pos,neg}->
    surprise_{pos,neg} sensory-drive edges on the SAME shared circuit the existing D2 notice reads (the load-bearing
    lesion control: the gated update must VANISH because the gate can never open). INTENDED to also silence the
    live surprise notice for the process this lesion is applied in -- only ever set in a dedicated flip-verify/
    lesion-diagnostic subprocess, never in real production traffic."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _ensure_gate_calibrated(organ) -> None:
    """Calibrate the gate threshold for BOTH `organ.state_pos` and `organ.state_neg` (the only two states
    `_state_for` ever targets) on the organ's CURRENT circuit, caching each AS AN ATTRIBUTE ON THE ORGAN ITSELF
    (`organ._c1_gate_cache`), if not already cached. Idempotent — cheap after the first call.

    MUST run BEFORE any lesion is ever applied to this organ. A REAL bug caught 2026-09-02: calibrating on an
    ALREADY-LESIONED circuit (obs->surprise zeroed) collapses BOTH `exp_hz` and `vio_hz` to ~0 Hz, so the
    GATE_FRAC formula (`thr = exp_hz + GATE_FRAC*(vio_hz-exp_hz)`) produces a threshold of ~0 — which a
    near-zero, noisy `surp_hz` read then trivially satisfies (`0 >= ~0`). This made a self-test's LESION arm gate
    OPEN on literally 10/10 turns with real weight growth — the exact OPPOSITE of the intended lesion behaviour,
    and worse than merely noisy: it was silently backwards. The de-risk's own `_run_update_arm` avoids this by
    calibrating ONCE on the intact circuit BEFORE applying its own lesion and reusing that SAME threshold for
    every arm; this function reproduces that ordering for the two states production ever targets. Both
    `ensure_worldmodel_crossedge_lesion` and `crossedge_gated_update` call this FIRST, before checking/applying
    the lesion flag."""
    organ.ensure_built()
    cache = organ.__dict__.setdefault("_c1_gate_cache", {})
    from research.runners._crossedge_surprise_worldmodel_derisk import _calibrate_gate
    st = organ._st
    for s in (organ.state_pos, organ.state_neg):
        if s not in cache:
            v0, _pp, _pn = organ._predict(st, s)
            thr, exp_hz, vio_hz = _calibrate_gate(st["bridge"], st["idx_map"], st["xp"], s, v0)
            cache[s] = (thr, exp_hz, vio_hz)


def ensure_worldmodel_crossedge_lesion(organ) -> int | None:
    """Zero the obs->surprise edges on `organ`'s already-built shared circuit, once per organ instance (idempotent
    across repeated live turns in the same process). No-op unless the LESION flag is set. Returns the count of
    edges zeroed (or None if not applied). MUST be called BEFORE the turn's own `read_surprise` (webapp/server.py
    calls this immediately after resolving `worg`, ahead of its existing `sj = worg.read_surprise(...)` read) —
    applying it lazily inside `crossedge_gated_update` alone would let the FIRST turn's `read_surprise` see the
    still-intact circuit (an ordering bug caught by this module's own offline self-test, where the lesion arm's
    first turn gated once before the lesion took effect). ALWAYS calibrates the gate (`_ensure_gate_calibrated`,
    on the still-intact circuit) BEFORE lesioning — see that function's docstring for why this order is
    load-bearing, not cosmetic.

    The "already lesioned" marker is stored AS AN ATTRIBUTE ON THE ORGAN ITSELF (`organ._c1_lesioned`), NOT in a
    module-level dict keyed by `id(organ)` — a real bug caught 2026-09-02: `id()` is a memory address, and a
    dropped-then-rebuilt organ (as every offline self-test's `_WM._ORGAN = None; get_organ(seed)` does per arm) can
    be garbage-collected and a NEW organ allocated at the SAME address, so an id-keyed cache silently handed a
    fresh organ a PRIOR organ's stale "already lesioned" (or a prior organ's stale gate calibration) — the exact
    cross-arm contamination that made an early 6-seed self-test read as broken on 5/6 seeds (byte-identical-looking
    code, wrong on different seeds in different ways) before this fix."""
    organ.ensure_built()
    _ensure_gate_calibrated(organ)          # ALWAYS first — on whatever circuit state organ is in RIGHT NOW
    if not xedge_surprise_worldmodel_lesioned():
        return None
    if getattr(organ, "_c1_lesioned", False):
        return 0
    from research.runners._crossedge_surprise_worldmodel_derisk import _zero_obs_to_surprise
    n = _zero_obs_to_surprise(organ._st["bridge"], organ._st["xp"])
    organ._c1_lesioned = True
    return n


def _transition_weight_live(organ, s: int, observed_sign: int) -> float:
    """The mean weight of state-block[s] -> the OBSERVED-valence pred pool, on the LIVE organ's own built circuit
    (reuses the de-risk's own `_transition_weight`, not reimplemented)."""
    from research.runners._crossedge_surprise_worldmodel_derisk import _transition_weight
    pool = "pred_pos" if observed_sign > 0 else "pred_neg"
    return _transition_weight(organ._st["bridge"], organ._st["meta"], s, pool)


def crossedge_gated_update(organ, context_sign: int, observed_sign: int) -> dict | None:
    """LIVE reply-path hook. `organ` is the process-shared `WorldModelProductionOrgan`
    (`worldmodel_production_organ.get_organ()`); `context_sign`/`observed_sign` are EXACTLY the values
    `webapp/server.py`'s existing worldmodel violation branch already computed this turn (`wm_state["context_sign"]`,
    `obs_sign`). Reads the world-model's OWN spiking prediction-error (the de-risk's `_read_surprise`, imported
    byte-for-byte — a SECOND, dedicated read against the per-state calibration `_ensure_gate_calibrated` froze on
    the INTACT circuit, not a reuse of the notice's `sj["surprised"]`; see that function's docstring for why, and
    for why the threshold must never be (re)calibrated after a lesion). IFF the gate clears, opens the transition's
    Hebbian gate for exactly ONE co-fire of the target state with the OBSERVED-valence pred pool (the de-risk's own
    validated primitive, imported byte-for-byte: `train_transition`'s per-step co-fire, `CUE_PA=1000.0`/
    `TEACH_PA=1000.0`/`HOLD=40`), then re-freezes so every subsequent read is a frozen forward pass. No-op (returns
    None) unless the flag is on. Never raises into a turn (best-effort; degrades to no diagnostic field on error)."""
    if not xedge_surprise_worldmodel_enabled():
        return None
    try:
        from research.runners._crossedge_surprise_worldmodel_derisk import (
            TEACH_PA, HOLD as _TEACH_HOLD, CUE_PA, _read_surprise)
        from research.runners._affective_world_model_derisk import _drive_read, _hard_reset
        ensure_worldmodel_crossedge_lesion(organ)   # calibrates (if not already) THEN lesions iff the flag is set;
        #                                              idempotent — the caller (webapp/server.py) also calls this
        #                                              earlier in the turn, before its own read_surprise
        st = organ._st
        s = organ._state_for(context_sign)
        gate_threshold, exp_hz, vio_hz = organ._c1_gate_cache[s]   # frozen on the INTACT circuit; never recalibrated
        surp_hz = _read_surprise(st["bridge"], st["idx_map"], st["xp"], s, observed_sign)
        w_before = _transition_weight_live(organ, s, observed_sign)
        gate_opened = bool(surp_hz >= gate_threshold)
        if gate_opened:
            teach = "pred_pos" if observed_sign > 0 else "pred_neg"
            st["bridge"]._blk = st["meta"]["blk"]              # claim this organ's block size (shared-bridge safe)
            st["cfg"].enable_hebbian_learning = True           # OPEN the gate for exactly this credited co-fire
            _hard_reset(st["bridge"])
            _drive_read(st["bridge"], st["idx_map"], {"state": (s, CUE_PA), teach: (None, TEACH_PA)},
                       _TEACH_HOLD, st["xp"], [])
            st["cfg"].enable_hebbian_learning = False          # RE-FREEZE -> every read stays a frozen forward pass
        w_after = _transition_weight_live(organ, s, observed_sign)
        return {"on": True, "state": int(s), "observed_sign": int(observed_sign), "surprise_hz": float(surp_hz),
                "gate_threshold": float(gate_threshold), "gate_opened": gate_opened,
                "w_obs_before": float(w_before), "w_obs_after": float(w_after),
                "lesioned": xedge_surprise_worldmodel_lesioned()}
    except Exception as e:
        return {"on": True, "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Offline self-verify entrypoint (0 Claude tokens; CPU numpy). Exercises the REAL production function
#  (`crossedge_gated_update`) against the REAL process-shared organ (`worldmodel_production_organ.get_organ`), not a
#  bespoke probe: repeated SURPRISING vs EXPECTED update sequences, intact vs lesioned.
#
#  EACH ARM RUNS IN ITS OWN SUBPROCESS (real bug, caught 2026-09-02): running "surprising"/"expected"/"lesioned"
#  sequentially IN ONE process gave WRONG results on 5/6 seeds — the "expected" (confirming) and "lesioned" arms
#  spuriously gated open on ~10/10 turns, even though an ISOLATED single-arm process for the SAME seed was clean
#  (0/10, as it should be). This reproduces `onebrain_regression_battery.py`'s own documented reason for its
#  per-arm subprocess design: "comparing two in-process sequential arms would diverge on noise" — a background
#  stochastic process (OU/Poisson drive) is NOT fully re-seeded by a fresh organ build within the SAME process, so
#  a LATER arm's reads carry the accumulated noise trajectory of every EARLIER arm's builds+reads in that process.
#  A fresh subprocess gives each arm its own clean process-level RNG state, matching how a real, long-lived
#  production organ (built once near process start, calibrated once, read many times) actually behaves — the
#  in-process 3-arms-back-to-back pattern was an ARTIFACT OF THE TEST, not a defect in the production wiring.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _run_arm_inprocess(seed: int, mode: str, lesion: bool, n_turns: int) -> dict:
    """ONE arm, in THIS process (called either directly by a subprocess worker, or — for a quick single-arm
    smoke — by a caller that accepts the in-process noise caveat above)."""
    import research.runners.worldmodel_production_organ as _WM
    os.environ["BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL"] = "1"
    if lesion:
        os.environ["BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION"] = "1"
    else:
        os.environ.pop("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION", None)
    _WM._ORGAN = None
    org = _WM.get_organ(seed)
    org.ensure_built()
    ensure_worldmodel_crossedge_lesion(org)   # apply BEFORE any read_surprise (ordering-correct; see its docstring)
    context_sign = 1
    s = org._state_for(context_sign)
    v0, _pp0, _pn0 = org._predict(org._st, s)   # the ACTUAL spiking-read predicted sign for this seed's target
    #                                            state (NOT assumed — state_pos's predicted sign is a per-seed
    #                                            build-time selection, verified 2026-09-02: seed43 broke a
    #                                            hardcoded v0=1 assumption)
    observed_sign = v0 if mode == "expected" else -v0
    w0 = _transition_weight_live(org, s, observed_sign)
    traj, n_gated = [w0], 0
    for _t in range(n_turns):
        upd = crossedge_gated_update(org, context_sign, observed_sign)
        if upd and upd.get("gate_opened"):
            n_gated += 1
        traj.append(upd["w_obs_after"] if upd else traj[-1])
    pred_before = org._predict(org._st, s)
    return {"mode": mode, "lesion": lesion, "w_traj": [round(x, 4) for x in traj], "n_gated": n_gated,
            "w_grew": float(traj[-1] - traj[0]), "pred_margin": float(pred_before[1] - pred_before[2])}


def _run_arm_subprocess(seed: int, mode: str, lesion: bool, n_turns: int, raw_dir: str) -> dict:
    """Spawn `_run_arm_inprocess` in a FRESH subprocess (its own clean RNG state) and read back the JSON result."""
    import json
    import subprocess
    import sys
    tag = "%s%s" % (mode, "_lesion" if lesion else "")
    out_path = os.path.join(raw_dir, "arm_s%d_%s.json" % (seed, tag))
    os.makedirs(raw_dir, exist_ok=True)
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.onebrain_xedge_surprise_worldmodel_production",
                        "--arm", mode, "--seed", str(seed), "--n-turns", str(n_turns), "--out", out_path]
                       + (["--lesion"] if lesion else []), env=env)
    if p.returncode != 0 or not os.path.exists(out_path):
        return {"mode": mode, "lesion": lesion, "_error": "worker rc=%s" % p.returncode,
                "w_traj": [0.0], "n_gated": 0, "w_grew": 0.0, "pred_margin": 0.0}
    with open(out_path) as f:
        return json.load(f)


def _selftest_loadbearing(seed: int, n_turns: int = 10, raw_dir: str = "research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms"):
    from tools.lab import attributable_to

    surprising = _run_arm_subprocess(seed, "surprising", False, n_turns, raw_dir)
    expected = _run_arm_subprocess(seed, "expected", False, n_turns, raw_dir)
    lesioned = _run_arm_subprocess(seed, "surprising", True, n_turns, raw_dir)

    grew = bool(surprising["n_gated"] >= 1 and surprising["w_grew"] > 0.05)
    expected_no_update = bool(expected["n_gated"] == 0 and abs(expected["w_grew"]) < 1e-9)
    lesion_no_update = bool(lesioned["n_gated"] == 0 and abs(lesioned["w_grew"]) < 1e-9)
    frac_vs_lesion = attributable_to(f"seed{seed} live surprise->worldmodel gated update vs lesion",
                                     surprising["w_grew"], lesioned["w_grew"])
    frac_vs_expected = attributable_to(f"seed{seed} live surprise->worldmodel gated update vs expected (same gate code)",
                                       surprising["w_grew"], expected["w_grew"])
    n_hollow = 0 if (grew and lesion_no_update) else 1
    GO = bool(grew and expected_no_update and lesion_no_update
              and frac_vs_lesion is not None and frac_vs_lesion >= 0.8
              and frac_vs_expected is not None and frac_vs_expected >= 0.8)
    return {"seed": int(seed), "n_turns": n_turns, "surprising": surprising, "expected": expected,
            "lesioned": lesioned, "grew": grew, "expected_no_update": expected_no_update,
            "lesion_no_update": lesion_no_update,
            "frac_attributable_vs_lesion": (None if frac_vs_lesion is None else float(frac_vs_lesion)),
            "frac_attributable_vs_expected": (None if frac_vs_expected is None else float(frac_vs_expected)),
            "n_hollow": n_hollow, "GO": GO}


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="exercise the live production hook + self-verify load-bearing")
    ap.add_argument("--arm", default=None, choices=["surprising", "expected"],
                    help="internal: run ONE arm in THIS process (subprocess worker for _selftest_loadbearing)")
    ap.add_argument("--lesion", action="store_true", help="(--arm only) apply the lesion for this arm")
    ap.add_argument("--n-turns", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="(--arm only) the single seed for this worker")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.arm:
        res = _run_arm_inprocess(args.seed, args.arm, args.lesion, args.n_turns)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = [_selftest_loadbearing(s) for s in seeds]
    n_go = sum(r["GO"] for r in results)
    for r in results:
        print(f"[seed {r['seed']}] GO={r['GO']} surprising: n_gated={r['surprising']['n_gated']} "
              f"w_grew={r['surprising']['w_grew']:+.4f} | expected: n_gated={r['expected']['n_gated']} "
              f"w_grew={r['expected']['w_grew']:+.4f} | lesioned: n_gated={r['lesioned']['n_gated']} "
              f"w_grew={r['lesioned']['w_grew']:+.4f} | frac_vs_lesion={r['frac_attributable_vs_lesion']} "
              f"frac_vs_expected={r['frac_attributable_vs_expected']} n_hollow={r['n_hollow']}", flush=True)

    payload = {"probe": "onebrain_xedge_surprise_worldmodel_production_selftest", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "n_go": n_go, "n_seeds": len(results),
               "results": results,
               "note": ("the REAL production hook crossedge_gated_update, driven against the process-shared "
                        "WorldModelProductionOrgan (worldmodel_production_organ.get_organ), gated by its OWN "
                        "dedicated per-state-calibrated surprise read (_gate_threshold_for, the de-risk's own "
                        "GATE_FRAC recipe): a repeated SURPRISING sequence grows the observed-pool state->pred "
                        "transition; the SAME gating code on an EXPECTED sequence does not; lesioning obs->surprise "
                        "(the shared D2 circuit) collapses the update on the surprising sequence too.")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\n[XEDGE-SURPRISE-WORLDMODEL] {n_go}/{len(results)} seeds GO", flush=True)
    return 0 if n_go == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
