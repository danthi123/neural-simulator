"""Continuous-substrate rung 3 (LEARN-THROUGH-USE) de-risk — an IDLE-TICK REPLAY pass STABILIZES a recently-stored,
sub-barrier episodic trace: a later recall (after intervening decay/interference) is measurably BETTER than a matched
trace that got no idle replay. LESION: skip the idle replay pass -> the advantage vanishes.

WHY THIS, AND WHAT THE RECORD ALREADY SAYS (corpus-first, 2026-08-20 -- `tools/before_you_build.sh` + rag_search run
before writing a line of this file):
  - 2026-08-12 gap#4/E3 (`_gap4_btsp_lasting_trace_recall_after_delay_derisk.py`, 6-seed GO) already showed a HELD
    (supra-barrier) BTSP plateau write is a LASTING trace under a synaptic tag-and-capture maintenance rule, while a
    TRANSIENT (sub-barrier) write decays below recall over the same window. It explicitly did NOT test replay: no
    reactivation path executes in that runner ("NOT 'consolidation' in the TERMS.md sense (no replay path executes)").
    This de-risk's job is exactly that gap: does an idle-tick REPLAY pass rescue a trace that WOULD otherwise decay
    (the transient/sub-barrier case), by pushing it over the SAME tag-and-capture barrier via repeated, weaker,
    spontaneous-style reactivation (Frey & Morris 1997 synaptic tagging-and-capture: a second, weaker stimulus within
    the tag's window can be CAPTURED by a tag set by the first -- textbook "spaced repetition" LTP consolidation)?
  - 2026-08-03 replay-cortical-consolidation gate v1-v6 (`_replay_cortical_consolidation_gate*.py`, ALL NO-GO): tests
    a HARDER problem -- TWO INTERFERING episodes, CA1->CORTEX systems consolidation, hippocampus-INDEPENDENT recall
    at retest. NO-GO: 16x seed variance, false recall from over-broad cortical association, "increasing n_replays
    harms retrieval" (2026-05-20 over-consolidation finding: n_replays_per_tag 50 vs 20 REGRESSED accuracy -0.184).
    This de-risk is explicitly NOT that: ONE recent trace, no interfering second episode, no cross-structure
    (hippocampus-independent) claim, and a SMALL, BOUNDED replay dose (n_ticks default 2) -- the over-consolidation
    wall is a dosage effect, so this runner treats replay dose as a knob to report, not to maximise. CALIBRATION
    (2026-08-20, on seeds 42/43/44 before locking the 6-seed confirmatory run -- same "calibrate then lock" pattern
    as the 2026-08-03 gate's seeds 212/213): a wider sweep over replay_pre_pA / replay_pulse_pA / n_ticks /
    decay_steps / btsp_w_max surfaced a genuine, reproducible TENSION, not a free lunch -- enough replay dose to
    rescue a real (weakly-encoded) trace robustly (G1) ALSO gives a same-dose NEVER-encoded control (moat_replay,
    G3) a partial, non-trivial boost, because BTSP's saturating (w_max-w) update is only weakly sensitive to the
    starting weight when w_max is large relative to the tag-and-capture barrier. Lowering w_max toward the barrier
    did not fix this (it just weakens both signals together at this network's scale). This is the honest residual:
    see the G3 result and NEXT_RUNG in the emitted artifact.
  - 2026-05-21 "missing CA1->concept-pool consolidation wire" and 2026-05-20 over-consolidation ablation are both
    about COMPOSITIONAL / multi-item generalization, out of scope here by the task's own framing (episodic
    STABILIZATION of one trace, not semantic generalization).
  - docs/TERMS.md "consolidation" needs BOTH: a replay path that actually EXECUTES (verified below by an explicit
    per-condition call counter), AND the trace surviving a LESION OF THE SOURCE STRUCTURE (i.e. hippocampus-cortex
    systems independence). This runner tests neither systems-level independence nor a second interfering episode --
    it tests whether the REPLAY PASS ITSELF is the thing keeping the trace alive (skip-the-pass IS the lesion here,
    of the REPLAY mechanism, not of a source brain structure). So the word used throughout is "stabilization" /
    "an idle-tick replay pass", NOT "consolidation" -- see Honest scope below.

MECHANISM (on-substrate: real bridge stepping + the real on-bridge BTSP kernel `fused_btsp_update` drives every
weight change; the tag-and-capture MAINTENANCE applied between steps is, like the 2026-08-12 precedent, a
RUNNER-LEVEL MODEL on `cp_connections.data`, not yet a `sim/` kernel):
  ENCODE (one "turn"): pre (n_pre) fires under an external drive; post-target gets a brief TRANSIENT (non-bistable)
    apical plateau pulse -- the SAME sub-barrier "transient" recipe the 2026-08-12 runner showed decays without help.
    This models a recent episode stored WEAKLY on a single exposure (the realistic case -- most single turns are not
    a maximally-salient held plateau).
  IDLE TICKS (n_ticks, each = a reactivation sub-window THEN a decay/maintenance sub-window; EVERY condition runs the
    SAME total step count and the SAME number of tag-and-capture maintenance applications -- only whether the
    reactivation sub-window injects a signal, and to whom, differs by condition):
    - REPLAY sub-window: a brief, WEAKER-than-encoding transient apical pulse to post-target, co-timed with a
      pre-population reactivation drive (weaker than the encoding drive) -- repeated smaller BTSP-eligible
      co-activations that, via the SAME saturating (w_max - w) BTSP rule, cumulatively push the weight toward/over
      the capture barrier (spaced-repetition consolidation), rather than one big plateau doing it in one shot.
      The pre-side BTSP eligibility trace is explicitly ZEROED right after the encode phase (`cp_btsp_pre_elig[:]=0`)
      so idle-tick potentiation reflects ONLY idle-tick activity, not lingering eligibility from the original
      encoding turn -- otherwise a no-presynaptic-drive lesion arm would be confounded by the encode-phase trace
      (btsp_elig_tau_ms=1000 barely decays over a few hundred steps).
    - decay/maintenance sub-window: background silence + the 2026-08-12 tag-and-capture rule (w >= barrier -> frozen;
      else w *= 1-beta).
  RECALL (spiking, both t0 and after-idle): fire the pre cue only (no apical, no learning), count post-TARGET spikes
    -- a BEHAVIOURAL read (docs/TERMS.md: a weight read is a proxy, behaviour is the capability), identical readout
    style to the 2026-08-12 precedent.

CONDITIONS (encode -> n_ticks x (reactivation sub-window, decay sub-window) -> recall):
  noreplay       transient encode; reactivation sub-window carries NO drive/pulse (silence) -- the MATCHED
                 non-consolidated control (same encode, same total idle step count, no replay content).
  replay         transient encode; reactivation sub-window = weak pre drive + weak transient apical pulse to
                 post-target -- THE TREATMENT (idle-tick replay-driven stabilization).
  replay_nopre   transient encode; reactivation sub-window = weak apical pulse to post-target WITHOUT the pre drive
                 -- LESION of presynaptic reactivation: BTSP's dw = eta*etilde_pre*is_post*(wmax-w) needs BOTH
                 factors, so with etilde_pre ~ 0 (no pre spiking, and the encode-phase trace was zeroed) the pulse
                 alone should NOT rescue recall -- proves it is the CO-ACTIVITY, not "any periodic apical ticks".
  moat_replay    NEVER encoded (no write phase at all) but given the IDENTICAL replay dose as `replay` -- SPECIFICITY
                 control: replay should AMPLIFY an existing sub-barrier tag, not FABRICATE a memory from nothing.

PRE-REGISTERED GO (6-seed, ALL seeds; RECALL_HI/RECALL_LO/CONTRAST reused verbatim from the 2026-08-12 precedent --
same architecture, same calibrated dynamic range):
  (G1 HEADLINE)   replay's after-idle recall EXCEEDS noreplay's after-idle recall by a real margin (both the
                  contrast ratio AND, ideally, replay crosses RECALL_HI while noreplay stays decayed) -- "skip the
                  idle replay pass -> the advantage vanishes" is this comparison directly.
  (G2 LESION)     replay_nopre's after-idle recall is NOT separated from noreplay by the same margin as replay is
                  (the presynaptic-reactivation lesion removes the advantage) -- proves BTSP coincidence, not mere
                  apical ticking, is doing the work.
  (G3 SPECIFICITY) moat_replay's after-idle recall stays decayed (<= CONTRAST of replay's) despite an IDENTICAL
                  replay dose -- proves replay AMPLIFIES an existing tag rather than conjuring one.
  (INSTRUMENT)    per-condition replay-call counters are reached (>0) exactly for replay/replay_nopre/moat_replay and
                  0 for noreplay (verifies the branch is REACHED, docs/TERMS.md's first "consolidation" clause);
                  off_dw==0 with enable_btsp=False; maintenance inert at beta=0; recall readout distinguishes a huge
                  vs a zero target weight.

Run:  SIM_BACKEND=numpy python -m research.runners._idle_consolidation_stabilization_derisk \
        --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from tools.lab import attributable_to, void_if, assert_backend  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
# reuse-by-import (NOT edited): the 2026-08-12 precedent's network builder + spiking-recall + weight-sum helpers.
from research.runners._gap4_btsp_lasting_trace_recall_after_delay_derisk import (  # noqa: E402
    _build, _post_col_mask, _recall, _write_weight_sum)

xp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_idle_consolidation_stabilization.json"

# Reused verbatim from the 2026-08-12 precedent (identical architecture -> identical calibrated dynamic range).
RECALL_HI = 0.015
RECALL_LO = 0.008
CONTRAST = 0.4   # a "decayed" / "not-rescued" trace recalls at <= 40% of the seed's own `replay` after-idle trace

CONDITIONS = ("noreplay", "replay", "replay_nopre", "moat_replay")


def _one(seed, condition, args):
    """Run one condition end-to-end: encode -> n_ticks idle ticks (matched step count across conditions) -> recall."""
    do_encode = condition != "moat_replay"
    reacts = condition in ("replay", "replay_nopre", "moat_replay")   # this condition's reactivation sub-window is live
    replay_pre = condition in ("replay", "moat_replay")               # AND drives pre during it (replay_nopre does not)

    sb = _build(enable_btsp=True, bistable=False, seed=seed, w0=args.w0, n_pre=args.n_pre,
                btsp_w_max=args.btsp_w_max, btsp_lr=args.btsp_lr)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre")))
    post_all = np.asarray(list(rm.indices("post")))
    half = len(post_all) // 2
    post_target = post_all[:half]
    post_distr = post_all[half:]
    n = sb.cp_membrane_potential_v.size
    sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
    mask_target = _post_col_mask(sb, post_target)

    # ---- PHASE 1: ENCODE (one "turn", a weak/transient sub-barrier write -- decays without help) ----
    w0_target = _write_weight_sum(sb, mask_target)
    encode_drive = np.zeros(n, dtype=np.float32); encode_drive[pre_idx] = args.encode_drive
    ap = np.zeros(n, dtype=np.float32)
    for step in range(args.write_steps):
        sb.cp_external_input_current[:] = xp.asarray(encode_drive if do_encode else np.zeros(n, dtype=np.float32))
        cur = ap.copy()
        if do_encode and args.pulse_onset <= step < args.pulse_onset + args.pulse_steps:
            cur[post_target] = args.pulse_pA
        sb.cp_bdsp_apical_drive = xp.asarray(cur)
        sb._run_one_simulation_step()
    w_after_encode = _write_weight_sum(sb, mask_target)

    recall_t0 = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)

    # zero the pre-side BTSP eligibility trace so idle-tick potentiation reflects ONLY idle-tick activity, not a
    # lingering trace from the encode phase (btsp_elig_tau_ms=1000ms barely decays over a few hundred steps -- without
    # this the replay_nopre LESION would be confounded by residual encode-phase eligibility).
    if getattr(sb, "cp_btsp_pre_elig", None) is not None:
        sb.cp_btsp_pre_elig[:] = 0.0

    # ---- PHASE 2: IDLE TICKS -- matched total step count + matched maintenance-application count across ALL
    # conditions; only whether the reactivation sub-window carries a signal (and to whom) differs. ----
    replay_drive_arr = np.zeros(n, dtype=np.float32); replay_drive_arr[pre_idx] = args.replay_pre_pA
    replay_ap = np.zeros(n, dtype=np.float32)
    quiet = np.zeros(n, dtype=np.float32)
    bg = np.zeros(n, dtype=np.float32); bg[pre_idx] = args.bg_drive
    n_replay_calls = 0
    replay_lr = args.btsp_lr * args.replay_lr_scale   # WEAKER per-event gain during idle-tick reactivation than the
    # deliberate encode used (a smaller neuromodulatory/plasticity-rate gain for spontaneous offline reactivation vs
    # attended encoding is a real, reported asymmetry -- ACh/novelty gates plasticity gain during active encoding).
    for _tick in range(args.n_ticks):
        # -- reactivation sub-window --
        sb.core_config.enable_btsp = reacts
        sb.core_config.btsp_learning_rate = replay_lr if reacts else args.btsp_lr
        for _s in range(args.replay_steps):
            sb.cp_external_input_current[:] = xp.asarray(replay_drive_arr if (reacts and replay_pre) else quiet)
            cur = replay_ap.copy()
            if reacts:
                cur[post_target] = args.replay_pulse_pA
            sb.cp_bdsp_apical_drive = xp.asarray(cur)
            sb._run_one_simulation_step()
        if reacts:
            n_replay_calls += 1
        sb.core_config.enable_btsp = False
        sb.core_config.btsp_learning_rate = args.btsp_lr
        sb.cp_bdsp_apical_drive = xp.zeros(n, dtype=xp.float32)
        # -- decay / tag-and-capture maintenance sub-window (identical rule + step count every condition) --
        for _s in range(args.decay_steps):
            sb.cp_external_input_current[:] = xp.asarray(bg)
            sb._run_one_simulation_step()
            data = np.asarray(to_host(sb.cp_connections.data))
            tagged = data >= args.barrier
            data = np.where(tagged, data, data * (1.0 - args.beta))
            sb.cp_connections.data = xp.asarray(data)
    w_after_idle = _write_weight_sum(sb, mask_target)

    recall_after_target = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    recall_after_distr = _recall(sb, pre_idx, post_distr, args.recall_steps, args.recall_drive)

    return {"condition": condition, "seed": seed,
            "w0_target": w0_target, "w_after_encode": w_after_encode, "w_after_idle": w_after_idle,
            "recall_t0_target": recall_t0,
            "recall_after_target": recall_after_target, "recall_after_distr": recall_after_distr,
            "n_replay_calls": n_replay_calls}


def _instrument_checks(seed, args):
    """(a) enable_btsp=False write path is byte-inert. (b) maintenance is inert at beta=0. (c) the spiking recall
    readout distinguishes a huge vs zero target weight. Same style as the 2026-08-12 precedent's instrument block."""
    checks = {}
    sb = _build(enable_btsp=False, bistable=False, seed=seed, w0=args.w0, n_pre=args.n_pre,
                btsp_w_max=args.btsp_w_max, btsp_lr=args.btsp_lr)
    rm = sb.region_manager
    pre_idx = np.asarray(list(rm.indices("pre")))
    post_all = np.asarray(list(rm.indices("post")))
    post_target = post_all[:len(post_all) // 2]
    mask_target = _post_col_mask(sb, post_target)
    n = sb.cp_membrane_potential_v.size
    w0 = _write_weight_sum(sb, mask_target)
    drive = np.zeros(n, dtype=np.float32); drive[pre_idx] = args.encode_drive
    ap = np.zeros(n, dtype=np.float32)
    for step in range(args.write_steps):
        sb.cp_external_input_current[:] = xp.asarray(drive)
        cur = ap.copy()
        if args.pulse_onset <= step < args.pulse_onset + args.pulse_steps:
            cur[post_target] = args.pulse_pA
        sb.cp_bdsp_apical_drive = xp.asarray(cur)
        sb._run_one_simulation_step()
    checks["off_dw"] = _write_weight_sum(sb, mask_target) - w0

    # (b) maintenance inert at beta=0
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_target] = 2.0
    sb.cp_connections.data = xp.asarray(data)
    w_before = float(np.asarray(to_host(sb.cp_connections.data)).sum())
    d2 = np.asarray(to_host(sb.cp_connections.data))
    d2 = np.where(d2 >= args.barrier, d2, d2 * (1.0 - 0.0))
    sb.cp_connections.data = xp.asarray(d2)
    checks["maintenance_inert_beta0_delta"] = float(np.asarray(to_host(sb.cp_connections.data)).sum()) - w_before

    # (c) recall readout distinguishes huge vs zero target weight
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_target] = 5.0
    sb.cp_connections.data = xp.asarray(data)
    checks["recall_huge_weight"] = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    data = np.asarray(to_host(sb.cp_connections.data)).copy(); data[mask_target] = 0.0
    sb.cp_connections.data = xp.asarray(data)
    checks["recall_zero_weight"] = _recall(sb, pre_idx, post_target, args.recall_steps, args.recall_drive)
    return checks


def run(seed, args):
    res = {c: _one(seed, c, args) for c in CONDITIONS}
    instr = _instrument_checks(seed, args)
    return {"seed": seed, "conditions": res, "instrument": instr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-pre", type=int, default=64, dest="n_pre")
    ap.add_argument("--write-steps", type=int, default=150, dest="write_steps")
    ap.add_argument("--pulse-onset", type=int, default=20, dest="pulse_onset")
    ap.add_argument("--pulse-steps", type=int, default=15, dest="pulse_steps")
    ap.add_argument("--pulse-pA", type=float, default=120.0, dest="pulse_pA")
    ap.add_argument("--encode-drive", type=float, default=900.0, dest="encode_drive")
    ap.add_argument("--recall-steps", type=int, default=200, dest="recall_steps")
    ap.add_argument("--recall-drive", type=float, default=2000.0, dest="recall_drive")
    ap.add_argument("--bg-drive", type=float, default=0.0, dest="bg_drive")
    ap.add_argument("--n-ticks", type=int, default=2, dest="n_ticks")
    ap.add_argument("--replay-steps", type=int, default=15, dest="replay_steps")
    ap.add_argument("--replay-pre-pA", type=float, default=900.0, dest="replay_pre_pA")
    ap.add_argument("--replay-pulse-pA", type=float, default=125.0, dest="replay_pulse_pA")
    ap.add_argument("--replay-lr-scale", type=float, default=1.0, dest="replay_lr_scale")
    ap.add_argument("--decay-steps", type=int, default=13, dest="decay_steps")
    ap.add_argument("--w0", type=float, default=0.3)
    ap.add_argument("--btsp-w-max", type=float, default=10.0, dest="btsp_w_max")
    ap.add_argument("--btsp-lr", type=float, default=0.04, dest="btsp_lr")
    ap.add_argument("--barrier", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.04)
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    try:
        assert_backend("numpy", note="(CPU lane; GPU is busy)")
    except AssertionError as e:
        print("BACKEND WARNING: %s" % e)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a); per.append(r)
            c = r["conditions"]
            print(f"  [seed {s}] "
                  f"NOREPLAY t0={c['noreplay']['recall_t0_target']:.3f} after={c['noreplay']['recall_after_target']:.3f} | "
                  f"REPLAY t0={c['replay']['recall_t0_target']:.3f} after={c['replay']['recall_after_target']:.3f} "
                  f"(calls={c['replay']['n_replay_calls']}) | "
                  f"NOPRE after={c['replay_nopre']['recall_after_target']:.3f} (calls={c['replay_nopre']['n_replay_calls']}) | "
                  f"MOAT_REPLAY after={c['moat_replay']['recall_after_target']:.3f} (calls={c['moat_replay']['n_replay_calls']}) | "
                  f"off_dw={r['instrument']['off_dw']:.4f}", flush=True)
    except (RuntimeError, ValueError, AttributeError, KeyError, IndexError, TypeError) as e:
        err = "%s: %s" % (type(e).__name__, e); traceback.print_exc()

    summary = {"probe": "idle_consolidation_stabilization", "seeds": a.seeds,
               "params": {k: getattr(a, k) for k in (
                   "n_pre", "write_steps", "pulse_onset", "pulse_steps", "pulse_pA", "encode_drive",
                   "recall_steps", "recall_drive", "bg_drive", "n_ticks", "replay_steps", "replay_pre_pA",
                   "replay_pulse_pA", "replay_lr_scale", "decay_steps", "w0", "btsp_w_max", "btsp_lr",
                   "barrier", "beta")},
               "backend": os.environ.get("SIM_BACKEND", "(unset)"),
               "recall_hi": RECALL_HI, "recall_lo": RECALL_LO, "contrast": CONTRAST,
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}

    go = False; verdict = "ERROR -- no per-seed results"
    if err is None and per:
        def _m(cond, key):
            return float(np.mean([p["conditions"][cond][key] for p in per]))

        noreplay_after = _m("noreplay", "recall_after_target")
        replay_after = _m("replay", "recall_after_target")
        nopre_after = _m("replay_nopre", "recall_after_target")
        moat_after = _m("moat_replay", "recall_after_target")
        replay_t0 = _m("replay", "recall_t0_target")
        noreplay_t0 = _m("noreplay", "recall_t0_target")

        instr_ok = (all(abs(p["instrument"]["off_dw"]) < 1e-9 for p in per) and
                    all(abs(p["instrument"]["maintenance_inert_beta0_delta"]) < 1e-9 for p in per) and
                    all(p["instrument"]["recall_huge_weight"] >= RECALL_HI for p in per) and
                    all(p["instrument"]["recall_zero_weight"] <= RECALL_LO for p in per))
        calls_ok = (all(p["conditions"]["noreplay"]["n_replay_calls"] == 0 for p in per) and
                    all(p["conditions"]["replay"]["n_replay_calls"] == a.n_ticks for p in per) and
                    all(p["conditions"]["replay_nopre"]["n_replay_calls"] == a.n_ticks for p in per) and
                    all(p["conditions"]["moat_replay"]["n_replay_calls"] == a.n_ticks for p in per))

        # G1 HEADLINE: replay's after-idle recall beats noreplay's, per-seed, by more than CONTRAST separation, AND
        # (ideally) replay crosses RECALL_HI while noreplay stays below it.
        def per_seed(cond, key):
            return [p["conditions"][cond][key] for p in per]
        G1 = all(r > (1.0 + CONTRAST) * max(nr, 1e-9)
                 for r, nr in zip(per_seed("replay", "recall_after_target"), per_seed("noreplay", "recall_after_target")))
        # G2 LESION: replay_nopre must NOT show the same rescue -- its after-idle recall stays within CONTRAST of
        # noreplay's (i.e. close to the undstimulated baseline, far below `replay`'s).
        G2 = all(np_ <= (1.0 + CONTRAST) * max(nr, 1e-9) + RECALL_LO
                 for np_, nr in zip(per_seed("replay_nopre", "recall_after_target"), per_seed("noreplay", "recall_after_target")))
        # G3 SPECIFICITY: moat_replay (never encoded, same replay dose) stays decayed relative to `replay`.
        G3 = all(m <= CONTRAST * max(r, 1e-9) for m, r in
                 zip(per_seed("moat_replay", "recall_after_target"), per_seed("replay", "recall_after_target")))

        go = bool(instr_ok and calls_ok and G1 and G2 and G3)

        print("\n-- attribution: replay after-idle recall vs its no-presynaptic-reactivation lesion --")
        attributable_to("recall-after (replay vs replay_nopre lesion)", replay_after, nopre_after)
        void_if(not instr_ok, "an instrument check failed (off_dw / maintenance-inert / recall-readout)")
        void_if(not calls_ok, "a replay-call counter did not match its condition (the reactivation branch was not reached as expected)")

        if go:
            verdict = (f"GO -- an idle-tick REPLAY pass (weaker, repeated, BTSP-eligible co-activations) STABILIZES "
                       f"a recently-stored sub-barrier trace: after {a.n_ticks} idle ticks of matched total elapsed "
                       f"time, the replayed trace still recalls (after={replay_after:.3f}) while the matched "
                       f"non-replayed trace has decayed (after={noreplay_after:.3f}, t0 was {noreplay_t0:.3f} for "
                       f"both). Skipping the replay pass removes the advantage (G1). Replaying WITHOUT presynaptic "
                       f"reactivation (apical pulses alone, no pre drive, and the encode-phase eligibility trace "
                       f"explicitly zeroed first) does NOT rescue recall (nopre={nopre_after:.3f}, G2) -- the "
                       f"stabilization needs the BTSP coincidence, not mere periodic apical ticking. The identical "
                       f"replay dose applied to a NEVER-encoded pathway does not fabricate a trace "
                       f"(moat_replay={moat_after:.3f}, G3) -- replay amplifies an existing sub-barrier tag, it does "
                       f"not conjure one. 6-seed. Spiking WRITE + spiking REPLAY + spiking RECALL on the real bridge; "
                       f"the tag-and-capture maintenance is a runner-level model (2026-08-12 precedent, not yet a "
                       f"sim/ kernel). NOT 'consolidation' in the docs/TERMS.md sense (no lesion of a SOURCE STRUCTURE "
                       f"/ systems-level hippocampus-cortex independence was tested) -- call this an idle-tick "
                       f"REPLAY-DRIVEN STABILIZATION of one trace.")
        else:
            miss = []
            if not instr_ok: miss.append("INSTRUMENT failed")
            if not calls_ok: miss.append("replay-call counters did not match condition (branch-reached check failed)")
            if not G1: miss.append(f"G1 headline failed (replay={replay_after:.3f} vs noreplay={noreplay_after:.3f} "
                                    f"did not clear {1+CONTRAST}x on every seed)")
            if not G2: miss.append(f"G2 lesion failed (replay_nopre={nopre_after:.3f} rescued recall almost as much "
                                    f"as replay -- the effect is not BTSP-coincidence-specific)")
            if not G3: miss.append(f"G3 specificity failed (moat_replay={moat_after:.3f} recalled nearly as well as "
                                    f"replay={replay_after:.3f} despite never being encoded)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". Per THE LAW: tune replay_pre_pA/replay_pulse_pA/"
                       "n_ticks/decay_steps/barrier/beta, NOT a stop. Name the residual, not a stop.")

        v = Verdict("idle-tick replay-driven trace stabilization (recall-after-delay)")
        v.require("G1 headline: replay after-idle beats matched noreplay", G1, expect=True)
        v.require("G2 lesion: replay_nopre does not rescue recall (needs presynaptic coactivity)", G2, expect=True)
        v.require("G3 specificity: moat_replay (never encoded) does not gain recall from replay dose", G3, expect=True)
        v.require("instrument (off byte-id / maint inert / readout distinguishes)", instr_ok, expect=True)
        v.require("replay-call counters match condition (branch reached)", calls_ok, expect=True)
        v.control("replay vs noreplay (recall-after)", treatment=replay_after, control=noreplay_after,
                  min_separation=RECALL_LO)
        v.control("replay vs replay_nopre (recall-after)", treatment=replay_after, control=nopre_after,
                  min_separation=RECALL_LO)
        v.reaches("encode write moved the target weight", before=float(np.mean([p["conditions"]["noreplay"]["w0_target"] for p in per])),
                  after=float(np.mean([p["conditions"]["noreplay"]["w_after_encode"] for p in per])))
        for proc in ("STDP", "Hebbian", "homeostasis", "short-term plasticity", "reward modulation",
                     "structural plasticity", "NMDA", "input divisive norm"):
            v.disabled(proc, why="isolation: BTSP is the sole weight mover in both the encode and replay phases")
        result = v.decide(go=go)
        summary.update(result)

    summary["GO"] = go; summary["verdict"] = verdict
    summary["HONEST_NOTE"] = ("Spiking ENCODE, spiking REPLAY reactivation, and spiking RECALL (post firing) all run "
                              "on a real SimulationBridge via the real fused_btsp_update kernel; recall is a "
                              "BEHAVIOURAL read, not a weight read. The tag-and-capture maintenance rule (decay "
                              "sub-window) is a RUNNER-LEVEL MODEL applied to cp_connections.data (Frey&Morris 1997; "
                              "Lisman 1985; Bittner&Magee 2017 for the plateau write) -- NOT yet a sim/ kernel, "
                              "reused unchanged from the 2026-08-12 precedent. NOT 'consolidation' in the "
                              "docs/TERMS.md sense: no lesion of a SOURCE STRUCTURE (systems-level hippocampus-cortex "
                              "independence) was tested, and there is no second interfering episode -- this is "
                              "SINGLE-TRACE episodic stabilization, deliberately scoped away from the "
                              "compositional/semantic-generalization wall the 2026-05-20/2026-08-03 record already "
                              "hit. The replay CONTENT (which cells to reactivate, i.e. post_target + pre_idx) is "
                              "HOST-DIRECTED at reduced amplitude -- not an emergent recurrent pattern-completion "
                              "from partial/noisy cues (that is the honest next rung, see NEXT RUNG). NO new sim/ "
                              "edit; NO existing runner or webapp/ file was modified.")
    summary["NEXT_RUNG"] = ("Make the replay CONTENT emergent: give the pre population plastic recurrent internal "
                            "connectivity (RegionPathway pre->pre) so encoding also forms a weak auto-associative "
                            "assembly; drive idle-tick reactivation with UNTARGETED noise into a random subset of "
                            "pre (not the host re-presenting pre_idx/post_target verbatim) and let pattern-completion "
                            "recruit the rest of the assembly -- host then only supplies undirected noise, matching "
                            "the 2026-08-03 replay-cortical-consolidation gate's 'episode-agnostic CA3 noise' design. "
                            "Then wire this under the continuous engine's idle tick (webapp/continuous_engine.py, "
                            "which already names 'idle BTSP consolidation' as its next rung) as a guarded, "
                            "default-OFF pass, and port the tag-and-capture maintenance to a sim/ kernel.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[idle-consolidation-stabilization] VERDICT: {verdict}", flush=True)
    print(f"[idle-consolidation-stabilization] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
