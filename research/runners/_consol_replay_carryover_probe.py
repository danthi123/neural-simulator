"""CANDIDATE C: is the winner of replay window w determined by CARRY-OVER state from window w-1?
(2026-07-28, targeted probe)

THE OPEN FACT
-------------
`coactivation_replay` drives fact i's slot with 1400 pA for 30 steps, yet the DRIVEN slot wins its own
window only 15/27 times (chance 9/27), while competition inside each window is near-exclusive
(winner 400-1100 spikes, losers 0-12). Already EXCLUDED with the lever verified live: NMDA-attractor
gating, weighted-vs-count coincidence drive, plateau self-regen, hebbian_max_weight.

WHAT THIS PROBE TESTS
---------------------
`coactivation_replay` runs its 30-step bursts BACK-TO-BACK with NO quiet period: it zeroes
`cp_external_input_current`, stimulates, and steps. Every piece of neuronal STATE therefore crosses the
window boundary intact:
  * cp_recovery_variable_u  (Izhikevich adaptation; +d per spike, decays at rate a)
  * cp_conductance_g_e / g_i / g_nmda / g_nmda_rise (synaptic residue)
  * cp_membrane_potential_v, cp_refractory_timers
  * cp_neuron_firing_thresholds (enable_homeostasis defaults TRUE)
  * cp_v_apical (+ the coincidence plateau latch)
A slot that just emitted ~1000 spikes enters the NEXT window deeply adapted; a slot that sat silent
enters it rested. If that asymmetry is larger than the 1400 pA cue, the winner is set by history, not
by the cue -- which is exactly the observed symptom.

NOTE the NMDA-attractor exclusion does NOT cover this: that lever gates the comp_attr_s->comp_attr_s
`nmda_slow` self-loop synapses only. `u`, `g_e`, `v`, thresholds and the apical/plateau state are
untouched by it, and the slot's OWN internal AMPA recurrence (BrainRegion internal_density=0.20,
exc_weight_mean=2.0) is NOT under that gate either.

TWO MEASUREMENTS, both required (a correlation alone would not be causal, an intervention alone would
not localize):
  (1) OBSERVATIONAL -- record per-slot state at every window boundary of the REAL replay and ask
      whether winner(w) is predicted by winner(w-1) / by the pre-window state, and compare that
      predictor's accuracy against the cue's ("winner = driven").
  (2) CAUSAL -- insert a QUIET GAP of G steps between windows (drive zeroed) so the carried state
      decays, and re-measure driven-slot-wins + the CONTINUOUS driven-slot spike share. The gap is the
      lever; its effect on the carried state is measured explicitly so a null cannot be an inert lever.

LEVER VERIFICATION (today's rule): the probe prints the per-slot pre-window state SPREAD (a continuous
quantity) for gap=0 vs gap=G. If the gap does not measurably shrink the carried asymmetry, the arm is
reported as UNINTERPRETABLE, not as a null. Firing DURING the gap is also recorded -- if the slots keep
firing through it the gap is not quiet and the arm is likewise not a null.

NO sim/ edit. Monkeypatches only this process's step call, and restores it.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_replay_carryover_probe --seed 42 --gap 0
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_replay_carryover_probe --seed 42 --gap 200
"""
import os, sys, json, time, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from types import SimpleNamespace
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay,
    CONSOLIDATED_FACTS, _try_tgate)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
BURST = 30

# state arrays sampled at each window boundary. None-guarded: several are only allocated under
# particular config flags, and a missing array must read as "absent", never as zero.
STATE_ARRAYS = ["cp_recovery_variable_u", "cp_membrane_potential_v", "cp_conductance_g_e",
                "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
                "cp_refractory_timers", "cp_neuron_firing_thresholds", "cp_v_apical"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=3,
                    help="3 reproduces the 9-window/seed baseline the 15/27 figure came from.")
    ap.add_argument("--gap", type=int, default=0,
                    help="CAUSAL LEVER: quiet steps inserted between 30-step windows (drive zeroed). "
                         "0 = the shipped back-to-back replay.")
    ap.add_argument("--gap-attractor-off", action="store_true",
                    help="close nmda_attractor DURING the gap only, so a latched slot cannot self-sustain "
                         "through it. Use if slots are measured to keep firing during the gap.")
    ap.add_argument("--hebb-max", type=float, default=2.5,
                    help="baseline value the 15/27 figure was measured at; 20.0 is the raised-bound arm.")
    ap.add_argument("--weighted-coincidence", action="store_true",
                    help="cfg.coincidence_weighted_drive. Set EXPLICITLY both ways (comp_dendritic "
                         "defaults it True, so an ON-only flag is a no-op).")
    ap.add_argument("--freeze-plasticity", action="store_true",
                    help="FOLLOW-UP LEVER (candidate D): disable Hebbian + BTSP for the whole replay. The "
                         "quiet-gap arm excluded FAST state carry-over, but the plastic ca1->slot and "
                         "pool->slot weights also accumulate ACROSS windows and a 100 ms gap does not decay "
                         "them. If freezing them restores driven-slot targeting, the write is corrupting its "
                         "own targeting. Lever verified by reading ca1->slot mean weight pre/post replay.")
    ap.add_argument("--out", default="research/findings/raw/cortical_store")
    args = ap.parse_args()
    t_start = time.time()

    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,
             comp_btsp=True, comp_btsp_lr=0.0005, comp_btsp_wmax=2000.0, comp_btsp_elig_tau=30.0,
             comp_no_pool_slot=False, comp_pool_slot_weight=1.5, comp_attractor_slots=N,
             comp_per_slot_fs=False, enable_hebbian=True)
    b = build_substrate(args.seed, SimpleNamespace(**a))
    b.core_config.hebbian_max_weight = float(args.hebb_max)
    b.core_config.enable_stdp = False
    b.core_config.coincidence_weighted_drive = bool(args.weighted_coincidence)
    print(f"  CONFIG: hebbian_max_weight={b.core_config.hebbian_max_weight}  "
          f"coincidence_weighted_drive={b.core_config.coincidence_weighted_drive}  "
          f"enable_homeostasis={b.core_config.enable_homeostasis}  dt={b.core_config.dt_ms}  "
          f"gap={args.gap} steps ({args.gap * b.core_config.dt_ms:.1f} ms)")

    rm = b.region_manager
    slot = {i: np.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=np.int64) for i in range(N)}
    print(f"  slots: {[len(slot[i]) for i in sorted(slot)]} neurons each; "
          f"burst={BURST} steps ({BURST * b.core_config.dt_ms:.1f} ms)")

    tags, _ = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    print(f"  encode done ({time.time() - t_start:.0f}s)")

    if args.freeze_plasticity:
        b.core_config.enable_hebbian_learning = False
        b.core_config.enable_btsp = False
        b.core_config.enable_stdp = False
        print(f"  LEVER: plasticity FROZEN (hebbian={b.core_config.enable_hebbian_learning} "
              f"btsp={b.core_config.enable_btsp} stdp={b.core_config.enable_stdp})")
    from research.runners.nmda_compositional_consolidation import _mean_gate_weight as _mgw
    try:
        w_pre = float(_mgw(b, "ca1_to_comp_attr"))
    except Exception as _e:
        w_pre = float("nan"); print(f"  ca1->slot weight read failed: {_e}")

    # ---------------- instrumentation ----------------
    fire = {j: [] for j in sorted(slot)}     # per-step per-slot spike sum (window steps only)
    gap_fire = []                            # per-gap per-slot spike sum (proves the gap is quiet)
    boundary = []                            # per-slot state at the END of each window (== entering w+1)
    step_ct = {"n": 0}
    orig_step = b._run_one_simulation_step

    def _sample_state():
        """Per-slot mean of every allocated state array. One host transfer per array."""
        rec = {}
        for nm in STATE_ARRAYS:
            arr = getattr(b, nm, None)
            if arr is None:
                rec[nm] = None
                continue
            h = to_host(arr).astype(np.float64).ravel()
            if h.shape[0] < int(b.cp_membrane_potential_v.shape[0]):
                rec[nm] = None      # partially-allocated (e.g. per-branch) -> not slot-indexable
                continue
            rec[nm] = [float(h[slot[j]].mean()) for j in sorted(slot)]
        return rec

    drive_at_win = []    # per-slot mean cp_external_input_current at the FIRST step of each window

    def sampling_step(*a_, **k_):
        # ⚠️ SAMPLE THE DRIVE *BEFORE* THE STEP. This is the "verify the lever moved" rule applied to the
        # cue itself: the whole framing assumes the driven slot really receives +slot_drive_pA. Measure it,
        # do not assume it. (Sampled pre-step because the step may consume/overwrite the array.)
        if step_ct["n"] % BURST == 0:
            _ext = to_host(b.cp_external_input_current).astype(np.float64)
            drive_at_win.append([float(_ext[slot[j]].mean()) for j in sorted(slot)])
        r = orig_step(*a_, **k_)
        _fs = to_host(b.cp_firing_states)
        for j in sorted(slot):
            fire[j].append(float(_fs[slot[j]].sum()))
        step_ct["n"] += 1
        if step_ct["n"] % BURST == 0:
            boundary.append(_sample_state())
            if args.gap > 0:
                # QUIET GAP: zero the drive and run `gap` RAW steps (orig_step, so they are not counted
                # as window steps and do not pollute the per-window firing record).
                saved = b.cp_external_input_current.copy()
                b.cp_external_input_current[:] = 0.0
                if args.gap_attractor_off:
                    _try_tgate(b, "nmda_attractor", 0.0)
                gf = [0.0] * N
                for _ in range(int(args.gap)):
                    orig_step()
                    _g = to_host(b.cp_firing_states)
                    for j in sorted(slot):
                        gf[j] += float(_g[slot[j]].sum())
                gap_fire.append(gf)
                if args.gap_attractor_off:
                    _try_tgate(b, "nmda_attractor", 1.0)
                b.cp_external_input_current[:] = saved
                # post-gap state = what the NEXT window actually starts from
                boundary[-1] = {("post_" + k): v for k, v in _sample_state().items()} | boundary[-1]
        return r

    b._run_one_simulation_step = sampling_step
    try:
        coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles), args.seed,
                            coactivate=True, attractor_on=True)
    finally:
        b._run_one_simulation_step = orig_step
    print(f"  replay done ({time.time() - t_start:.0f}s), {step_ct['n']} window steps sampled")
    try:
        w_post = float(_mgw(b, "ca1_to_comp_attr"))
    except Exception:
        w_post = float("nan")
    _moved = abs(w_post - w_pre) > 1e-9
    print(f"  LEVER-CHECK ca1->slot mean weight: {w_pre:.6f} -> {w_post:.6f}  "
          f"({'CHANGED' if _moved else 'UNCHANGED'}); freeze_plasticity={args.freeze_plasticity}"
          + ("  ⚠️ FREEZE DID NOT HOLD — arm uninterpretable" if (args.freeze_plasticity and _moved) else "")
          + ("  ⚠️ NO WRITE HAPPENED even unfrozen — arm uninterpretable"
             if (not args.freeze_plasticity and not _moved) else ""))

    # ---------------- window reconstruction (identical RNG to coactivation_replay) ----------------
    # ⛔ THE BUG THIS PROBE FOUND (2026-07-28). `coactivation_replay` keeps ONE list alive and shuffles it
    # IN PLACE each cycle:  order = list(range(len(facts))) ; for _c in ...: rng.shuffle(order)
    # Fisher-Yates permutes the CURRENT arrangement, so shuffling an already-shuffled list yields a
    # DIFFERENT permutation of the identity than shuffling a fresh sorted list from the same RNG stream.
    # `_consol_replay_apical_probe.py` (the source of the "driven slot wins 15/27" figure) reconstructed it
    # as `o = list(range(N)); rng.shuffle(o)` -- a FRESH list per cycle. Cycle 1 agrees (both start [0,1,2]);
    # every later cycle is mislabelled. Ground truth is the CUE-VERIFY block below (argmax of the actual
    # cp_external_input_current), which matches THIS reconstruction and not the fresh-list one.
    rng = np.random.default_rng(int(args.seed) + 777)
    _ord = list(range(N))
    order = []
    for _c in range(int(args.cycles)):
        rng.shuffle(_ord); order.extend(list(_ord))
    n_win = min(len(order), len(fire[0]) // BURST)
    tot, winner, share = [], [], []
    for w in range(n_win):
        sl = slice(w * BURST, (w + 1) * BURST)
        t = [float(np.asarray(fire[j][sl]).sum()) for j in sorted(slot)]
        tot.append(t)
        winner.append(int(np.argmax(t)))
        share.append(t[order[w]] / sum(t) if sum(t) > 0 else float("nan"))
    driven = order[:n_win]

    print("\n  [CUE-VERIFY] per-slot mean cp_external_input_current at the first step of each window "
          "(the driven slot MUST carry ~slot_drive_pA=1400 and the others ~0):")
    _cue_ok = 0
    for w in range(min(n_win, len(drive_at_win))):
        d = drive_at_win[w]
        ok = int(np.argmax(d)) == driven[w] and d[driven[w]] > 100.0
        _cue_ok += ok
        if w < 4 or not ok:
            print(f"    w{w:2d} driven={driven[w]}  ext_pA={[round(x,1) for x in d]}  "
                  f"{'ok' if ok else '⛔ CUE NOT ON THE DRIVEN SLOT'}")
    print(f"    => cue landed on the driven slot in {_cue_ok}/{min(n_win, len(drive_at_win))} windows")

    print("\n  (WINDOWS) spikes per slot | driven | winner")
    for w in range(n_win):
        mark = "OK " if winner[w] == driven[w] else "MIS"
        prev = f" prev_winner={winner[w-1]}" if w else " prev_winner=-"
        print(f"    w{w:2d} {[int(x) for x in tot[w]]}  driven={driven[w]}  winner={winner[w]}  "
              f"{mark}  share={share[w]:.3f}{prev}")

    n_cue = sum(1 for w in range(n_win) if winner[w] == driven[w])
    # PER-CYCLE: each cycle is one full permutation of the N facts. If cycle 1 targets perfectly and later
    # cycles decay, the corrupting quantity ACCUMULATES across windows (weights) rather than being a fast
    # state that a quiet gap would reset.
    per_cycle = [sum(1 for w in range(c * N, min((c + 1) * N, n_win)) if winner[w] == driven[w])
                 for c in range((n_win + N - 1) // N)]
    print(f"\n  [PER-CYCLE] driven-slot wins per cycle (each /{N}): {per_cycle}")
    print(f"\n  [BASELINE] driven-slot wins: {n_cue}/{n_win} (chance {n_win/N:.1f}/{n_win})")
    print(f"  [BASELINE] CONTINUOUS driven-slot spike share: mean={np.nanmean(share):.4f} "
          f"(chance {1.0/N:.4f}, perfect 1.0)")

    # ---------------- (1) OBSERVATIONAL: is winner(w) predicted by window w-1? ----------------
    trans = [(winner[w - 1], winner[w], driven[w]) for w in range(1, n_win)]
    n_t = len(trans)
    persist = sum(1 for p, c, _ in trans if c == p)
    cue_t = sum(1 for _, c, d in trans if c == d)
    print(f"\n  [C-1 PERSISTENCE] winner(w)==winner(w-1): {persist}/{n_t} "
          f"(chance {n_t/N:.1f}/{n_t})  -- >chance = the previous winner LATCHES")
    print(f"  [C-1 ADAPTATION ] winner(w)==winner(w-1) BELOW chance would mean the previous winner is "
          f"ADAPTED OUT (Izhikevich u / homeostatic threshold).")
    print(f"  [C-1 CUE        ] winner(w)==driven(w) over the same transitions: {cue_t}/{n_t}")
    # conditional: does the cue work better when the driven slot was NOT the previous winner?
    for cond, lab in ((True, "driven WAS prev winner"), (False, "driven was NOT prev winner")):
        sub = [(p, c, d) for p, c, d in trans if (d == p) == cond]
        if sub:
            print(f"      | {lab:26s} n={len(sub):2d}  driven wins {sum(1 for _,c,d in sub if c==d)}/{len(sub)}")
    # 3x3 contingency prev-winner x current-winner
    cm = np.zeros((N, N), dtype=int)
    for p, c, _ in trans:
        cm[p, c] += 1
    print(f"  [C-1 CONTINGENCY] rows=winner(w-1), cols=winner(w):\n{cm}")

    # ---------------- (1b) does the PRE-WINDOW STATE predict the winner? ----------------
    print("\n  [C-2 PRE-WINDOW STATE -> WINNER]  (state sampled at the end of window w-1; for gap>0 the "
          "post-gap 'post_*' state is what window w actually starts from)")
    prefix = "post_" if args.gap > 0 else ""
    for nm in STATE_ARRAYS:
        key = prefix + nm
        vals = [bd.get(key) for bd in boundary[:n_win - 1]]
        if not vals or any(v is None for v in vals):
            print(f"    {nm:32s} : not allocated / not slot-indexable")
            continue
        V = np.asarray(vals, dtype=np.float64)              # (n_t, N)
        hi = sum(1 for k in range(n_t) if int(np.argmax(V[k])) == trans[k][1])
        lo = sum(1 for k in range(n_t) if int(np.argmin(V[k])) == trans[k][1])
        spread = float(np.mean((V.max(1) - V.min(1)) / np.maximum(np.abs(V).max(1), 1e-9)))
        print(f"    {nm:32s} : argmax->winner {hi}/{n_t}  argmin->winner {lo}/{n_t}  "
              f"(chance {n_t/N:.1f})  rel-spread={spread:.4f}  per-slot mean={np.round(V.mean(0),3).tolist()}")

    # ---------------- (2) CAUSAL LEVER VERIFICATION ----------------
    if args.gap > 0:
        gf = np.asarray(gap_fire, dtype=np.float64)
        print(f"\n  [LEVER] gap={args.gap} steps ({args.gap*b.core_config.dt_ms:.1f} ms). Slot spikes DURING "
              f"the gap (must be ~0 for the gap to be quiet): per-gap totals mean={gf.sum(1).mean():.1f} "
              f"max={gf.sum(1).max():.1f}")
        for nm in STATE_ARRAYS:
            pre = [bd.get(nm) for bd in boundary]
            post = [bd.get("post_" + nm) for bd in boundary]
            if any(v is None for v in pre) or any(v is None for v in post):
                continue
            P, Q = np.asarray(pre), np.asarray(post)
            sp_pre = float(np.mean(P.max(1) - P.min(1)))
            sp_post = float(np.mean(Q.max(1) - Q.min(1)))
            print(f"    {nm:32s} : cross-slot spread  end-of-window {sp_pre:.4f} -> post-gap {sp_post:.4f} "
                  f"({'DECAYED' if sp_post < sp_pre * 0.9 else 'UNCHANGED — lever inert on this variable'})")

    res = dict(seed=args.seed, gap=args.gap, cycles=args.cycles, n_windows=n_win,
               hebb_max=args.hebb_max, weighted_coincidence=bool(args.weighted_coincidence),
               gap_attractor_off=bool(args.gap_attractor_off),
               driven_wins=n_cue, driven_share_mean=float(np.nanmean(share)),
               winner=winner, driven=list(map(int, driven)), share=list(map(float, share)),
               totals=[[float(x) for x in t] for t in tot],
               persistence=persist, n_transitions=n_t, cue_over_transitions=cue_t,
               contingency=cm.tolist(),
               gap_fire=[[float(x) for x in g] for g in gap_fire],
               argv=sys.argv[1:])
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fn = f"{args.out}/carryover_seed{args.seed}_gap{args.gap}{'_atoff' if args.gap_attractor_off else ''}.json"
    Path(fn).write_text(json.dumps(res, indent=2))
    print(f"\nCARRYOVER-PROBE DONE -> {fn}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
