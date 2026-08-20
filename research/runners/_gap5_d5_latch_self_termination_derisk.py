"""STEP 2 of the D5 learn-through-use arc: can D5's PERSISTENT dendritic-dAP latch be made to SELF-TERMINATE into a
DISCRETE, SWR-like reactivation window -- completing the assembly (held members co-fire) then returning to baseline --
WITHOUT losing the completion or the specificity?

CONTEXT (read research/findings/2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read.md):
  STEP 1 (verified NO-GO): the Ecker AdEx SOMA recurrence at D5's real scale (~15 cells, real BTSP weights ~85) is too
  weak to complete a stored assembly -- 0.000 held-out completion everywhere. BUT D5's OWN per-cell dendritic-dAP latch
  DOES complete at this scale (that is why D5 reads via the latch, not soma recurrence). THE RESIDUAL the latch leaves:
  it is PERSISTENT (KIR-latched apical UP state). Learn-through-use / BTSP write-back (step 3) needs a DISCRETE,
  self-terminating reactivation window (an SWR-like transient: ignite -> co-fire -> die to baseline) -- both to be
  biologically faithful (compressed SWR events, buzsaki swr-sequence-replay) AND so a persistent UP state does not
  block normal operation / drive unbounded runaway potentiation during write-back.

THE MECHANISM (this runner):
  * SUBSTRATE: a GENUINE production D5 EpisodicDapMemory store (n_ca3=2000). 'dog' is BTSP-formed the real way
    (emergent DG-selected ~11-16 cell assembly, heterogeneous within-recurrence w~80-85); 'cat' is NEVER formed
    (baseline w~1.5). The reactivation READ is D5's OWN dendritic-dAP apical latch (cp_v_apical / fused_coincidence_
    plateau + self_regen + KIR), byte-untouched.
  * SELF-TERMINATION GATE (default-off; a=0/b=0 -> byte-identical persistent latch): an Ecker-`b`-style spike/dAP-
    triggered ADAPTATION current on the APICAL compartment (biology: a dendritic Ca-activated K / AHP / SK current;
    research/biology/dendritic-plateau-coincidence-burst.md names the plateau's Ca2+/NMDA basis, and the SWR envelope
    swr-sequence-replay is the self-terminating transient it must produce). A per-cell adaptation variable `w`
    INCREMENTS by `b_adapt` on every step the apical is in the UP/plateau state (a dAP event) and DECAYS with tau_w;
    it feeds back as a hyperpolarizing current on cp_v_apical (dv_adapt = -(dt/tau_apical)*w). Once w accumulates
    enough it knocks the apical below the KIR bistable band -> the plateau's v-gated self_regen sigmoid collapses ->
    the latch drops to DOWN. With the cue already removed there is no drive to re-ignite -> it rests silent = a
    bounded, discrete transient. Attaches RUNNER-SIDE (mutates bridge.cp_v_apical between steps, exactly as the organ's
    own _reset_apical_latch does) -- NO sim/ edit; byte-identical when b_adapt=0.

  PROTOCOL (why sustained-then-free, not a brief kick): the organ's production recall reliably ignites the latch with a
  SUSTAINED cue; a brief kick left to self-sustain is a knife-edge (unreliable ignition). So Phase-1 = SUSTAINED cue
  (ignite_steps) guarantees the latch LATCHES (= completion); Phase-2 = FREE window (window_steps, cue removed) is where
  self-termination is observed. The persistent latch (adapt-OFF) stays UP the whole free window; the gate (adapt-ON)
  self-terminates within it.

  READ CONTAMINATION (the organ's own warning: many live-mutated reads on one bridge are UNRELIABLE): fixed by a
  SNAPSHOT/RESTORE -- snapshot every mutable cp_ state array once at a clean rest (post-store, cp_v_apical allocated),
  restore it byte-identically before EVERY read. Verified: 5/5 repeated dog-OFF reads identical (else read#2 onward
  spuriously fail to ignite). This is the instrument's determinism guarantee, asserted as a precondition.

MEASURE (the teeth), dog adapt-ON vs adapt-OFF (= the gate LESION), same store, snapshot-restored:
  * COMPLETION_SURVIVES: the held-out members STILL co-fire in the FREE window (win_peak_ON >= COMPLETE_MIN and not
    degraded vs the baseline persistent latch) -- a self-termination that also kills completion is a FAIL.
  * SELF_TERMINATES: free-window activity returns to baseline (term_ratio_ON low) while the baseline PERSISTS
    (term_ratio_OFF high) -- the contrast is the point.
  * DISCRETE: the transient is a BOUNDED event (up_ms in [tens, < the window]) -- not a single spike, not persistent.
  * SPECIFIC: never-formed 'cat', identical dose, does NOT co-fire (dog >> cat).
  GO = COMPLETION_SURVIVES and SELF_TERMINATES and DISCRETE and SPECIFIC. Honest NO-GO otherwise (localizes whether the
  latch can be gated at all -> a different termination mechanism).

Reuse-by-import: EpisodicDapMemory (the production D5 organ) + _reset_apical_latch. NO sim/ edit. GPU-preferred.
  Run:    SIM_BACKEND=cupy python -m research.runners._gap5_d5_latch_self_termination_derisk \
              --seed 42 --out research/findings/raw/_d5_latch_selfterm/seed42.json
  6-seed: SIM_BACKEND=cupy python -m research.runners._gap5_d5_latch_self_termination_derisk \
              --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
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
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_latch_selfterm" / "seed42.json"

# --- GO thresholds ----------------------------------------------------------------------------------------------------
COMPLETE_MIN = 0.40        # free-window held-out co-firing FRACTION that counts as "the latch completed the assembly"
COMPLETE_KEEP = 0.75       # completion must survive the gate: win_peak_ON >= COMPLETE_KEEP * win_peak_OFF
TERM_MAX = 0.25            # adapt-ON free-window tail/peak -- self-termination (<= this = terminated)
PERSIST_MIN = 0.70         # adapt-OFF free-window tail/peak -- the baseline persistent latch (the thing to terminate)
DISCRETE_MIN_MS = 15.0     # the UP transient must last at least this (not a single spike)
CAT_MAX = 0.10             # never-formed cat free-window co-firing (specificity)
DOG_OVER_CAT = 3.0


def snapshot_state(bridge):
    """Copy every mutable cp_ ndarray + the sparse connection .data + current_time_step -> a clean-rest snapshot the
    reads restore byte-identically (the organ warns reused-bridge multi-reads contaminate; this is the determinism
    guarantee). Excludes sparse matrices themselves (identified by a .tocoo attr); their .data is copied separately."""
    snap = {}
    for name in dir(bridge):
        if not name.startswith("cp_"):
            continue
        try:
            arr = getattr(bridge, name)
        except Exception:  # noqa: BLE001 -- some cp_ attrs are properties that can raise; skip them
            continue
        if arr is None or hasattr(arr, "tocoo"):
            continue
        if hasattr(arr, "dtype") and hasattr(arr, "shape") and hasattr(arr, "copy"):
            try:
                snap[name] = arr.copy()
            except Exception:  # noqa: BLE001
                pass
    if getattr(bridge, "cp_connections", None) is not None:
        snap["__conn_data__"] = bridge.cp_connections.data.copy()
    # ZERO the spike-history in the frozen state: hard_silence does NOT reset cp_prev_firing_states, so a prior read's
    # last-step spikes (e.g. cat cells) would otherwise be frozen in and, on restore, drive feedback inhibition that
    # suppresses the next assembly's ignition. A clean rest has no spikes pending.
    for name in ("cp_prev_firing_states", "cp_firing_states"):
        if name in snap:
            snap[name][:] = False
    snap["__t__"] = int(bridge.runtime_state.current_time_step)
    return snap


def restore_state(bridge, snap):
    for name, arr in snap.items():
        if name.startswith("__"):
            continue
        cur = getattr(bridge, name, None)
        if cur is not None and hasattr(cur, "shape") and cur.shape == arr.shape:
            cur[:] = arr
        else:
            setattr(bridge, name, arr.copy())
    if "__conn_data__" in snap and getattr(bridge, "cp_connections", None) is not None:
        bridge.cp_connections.data[:] = snap["__conn_data__"]
    bridge.runtime_state.current_time_step = snap["__t__"]


def reactivate_latch(mem, slot, snap, *, b_adapt, tau_w, tau_apical, cue_pa, ignite_steps, window_steps, up_thresh):
    """Snapshot-restore -> SUSTAINED cue (ignite_steps) -> FREE window (window_steps, cue removed). Optionally apply the
    Ecker-`b` apical adaptation gate. Returns the ignition/free-window completion + self-termination metrics + a
    down-sampled held-apical-UP trace. cp_v_apical is D5's own bistable latch; the held cells are the NON-cue members."""
    bridge = mem.bridge; R = mem.R; cp = R.cp
    cfg = bridge.core_config
    E_rest = float(getattr(cfg, "apical_E_rest", -65.0)); dt = float(cfg.dt_ms)
    cue_g = mem.cue_by_asm[slot]
    held_cp = cp.asarray(np.asarray([int(R.ca3_idx[p]) for p in mem.held_pos_by_asm[slot]], dtype=np.int64))
    n_held = int(held_cp.size)

    restore_state(bridge, snap)                      # byte-identical clean start
    _reset_apical_latch(bridge)
    darr = cp.asarray(np.asarray(cue_g, dtype=np.int64), dtype=cp.int64)
    bridge.cp_external_input_current[darr] = cp.float32(cue_pa)

    up_trace = []; soma_trace = []; w_adapt = None; w_max = 0.0
    total = ignite_steps + window_steps
    for t in range(total):
        if t == ignite_steps:
            bridge.cp_external_input_current[darr] = 0.0     # remove cue -> FREE window
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        va = bridge.cp_v_apical
        if va is not None:
            if b_adapt > 0.0:
                if w_adapt is None:
                    w_adapt = cp.zeros_like(va)
                is_up = (va > cp.float32(up_thresh)).astype(cp.float32)   # dAP/plateau-active event indicator
                w_adapt += cp.float32(b_adapt) * is_up                    # increment per dAP-active step (Ecker b)
                w_adapt -= cp.float32(dt / tau_w) * w_adapt               # decay tau_w
                va -= cp.float32(dt / tau_apical) * w_adapt               # apical AHP/SK adaptation current (in place)
                w_max = max(w_max, float(w_adapt[held_cp].max()))
            up_trace.append(float((va[held_cp] > cp.float32(up_thresh)).astype(cp.float32).mean()))
        else:
            up_trace.append(0.0)
        fs = bridge.cp_firing_states
        soma_trace.append(float(fs[held_cp].astype(cp.float32).mean()) if fs is not None else 0.0)
    bridge.cp_external_input_current[darr] = 0.0

    up = np.asarray(up_trace); soma = np.asarray(soma_trace)
    win = up[ignite_steps:]
    ignite_peak = float(up[:ignite_steps].max())
    win_peak = float(win.max())
    win_tail = float(win[int(0.70 * len(win)):].mean())
    term_ratio = win_tail / (win_peak + 1e-6)
    up_ms = float((win > 0.15).sum() * dt)                  # ms of the free window the latch stays UP (transient width)
    window_ms = float(window_steps * dt)
    # down-sample the trace for the artifact (every 5th step)
    trace_ds = [round(float(x), 3) for x in up[::5]]
    return dict(ignite_peak=round(ignite_peak, 4), win_peak=round(win_peak, 4), win_tail=round(win_tail, 4),
                term_ratio=round(term_ratio, 4), up_ms=up_ms, window_ms=window_ms,
                duty=round(up_ms / (window_ms + 1e-6), 4), soma_peak=round(float(soma.max()), 4),
                w_max=round(w_max, 3), n_held=n_held, trace_ds=trace_ds)


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 110)
    print(f"[d5-latch-selfterm] seed={seed} backend={backend} -- build genuine D5 store, ignite the dendritic latch, "
          f"gate it with apical adaptation", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        mem = EpisodicDapMemory(seed, topics=["cat", "dog"], verbose=True)
        stored = mem.store("dog")
        if not stored:
            raise RuntimeError("mem.store('dog') returned False -- dog was not BTSP-formed")
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]
        up_thresh = mem.p["up_thresh"]
        rec_dog = mem.recall("dog")     # warms + allocates cp_v_apical + a fresh 3-read sequence ending clean-ish
        w_within_dog = float(cp.mean(mem.R.C.data[mem.R.withinA_masks[dslot]]))
        # --- the ONE frozen snapshot: right after recall('dog') + hard_silence + reset. This clean-rest state reliably
        # IGNITES the latch (verified: 5/5 identical reads); all arms RESTORE it byte-identically, so OFF vs ON differ
        # ONLY by the gate. It must be taken here, BEFORE any other topic's read (a cat read contaminates the residual
        # state that gates ignition) and taken ONCE (re-snapshotting after a read captures a contaminated state that no
        # longer ignites -- the organ's documented reused-bridge read contamination). recall('cat') runs AFTER: the arms
        # restore the frozen snap, so post-snapshot contamination is irrelevant. ---
        mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)
        rec_cat = mem.recall("cat")     # D5-native cat verdict (info only; bridge contamination now irrelevant)
        info = dict(n_ca3=int(mem.n_ca3), assembly_sizes=mem.assembly_sizes, dog_slot=dslot, cat_slot=cslot,
                    dog_size=int(mem.assembly_sizes[dslot]), cat_size=int(mem.assembly_sizes[cslot]),
                    w_within_dog=round(w_within_dog, 2), d5_recall_dog=rec_dog, d5_recall_cat=rec_cat, up_thresh=up_thresh)
        result["extract"] = info
        print(f"[d5-latch-selfterm] store: dog={info['dog_size']} cells (w_within~{info['w_within_dog']}), "
              f"cat={info['cat_size']} cells (never-formed); d5 recall dog in_memory={rec_dog['in_memory']} "
              f"apical_cue={rec_dog['apical_cue']:.3f} cat={rec_cat['in_memory']}", flush=True)

        rk = dict(tau_apical=a.tau_apical, cue_pa=a.cue_pa, ignite_steps=a.ignite_steps,
                  window_steps=a.window_steps, up_thresh=up_thresh)
        # --- the arms (each snapshot-restored to the frozen good state) ---
        dog_off = reactivate_latch(mem, dslot, snap, b_adapt=0.0, tau_w=a.tau_w, **rk)             # persistent (gate LESION)
        dog_off2 = reactivate_latch(mem, dslot, snap, b_adapt=0.0, tau_w=a.tau_w, **rk)            # determinism check
        dog_on = reactivate_latch(mem, dslot, snap, b_adapt=a.b_adapt, tau_w=a.tau_w, **rk)        # self-terminating
        cat_off = reactivate_latch(mem, cslot, snap, b_adapt=0.0, tau_w=a.tau_w, **rk)             # never-formed control
        cat_on = reactivate_latch(mem, cslot, snap, b_adapt=a.b_adapt, tau_w=a.tau_w, **rk)
        result["dog_off"] = dog_off; result["dog_on"] = dog_on
        result["cat_off"] = cat_off; result["cat_on"] = cat_on
        result["dog_off_repeat"] = {k: dog_off2[k] for k in ("ignite_peak", "win_peak", "term_ratio", "up_ms")}

        # instrument determinism: the two adapt-OFF reads must be byte-identical (snapshot-restore works)
        deterministic = bool(abs(dog_off["win_peak"] - dog_off2["win_peak"]) < 1e-9
                             and abs(dog_off["term_ratio"] - dog_off2["term_ratio"]) < 1e-9)

        # -------- the measured teeth --------
        COMPLETION_SURVIVES = bool(dog_on["win_peak"] >= COMPLETE_MIN
                                   and dog_on["win_peak"] >= COMPLETE_KEEP * dog_off["win_peak"])
        SELF_TERMINATES = bool(dog_on["term_ratio"] <= TERM_MAX and dog_off["term_ratio"] >= PERSIST_MIN)
        DISCRETE = bool(DISCRETE_MIN_MS <= dog_on["up_ms"] < dog_on["window_ms"])
        SPECIFIC = bool(cat_on["win_peak"] <= CAT_MAX
                        and dog_on["win_peak"] >= DOG_OVER_CAT * (cat_on["win_peak"] + 1e-6))
        go = COMPLETION_SURVIVES and SELF_TERMINATES and DISCRETE and SPECIFIC

        # attribute the self-termination to the gate: term_strength = 1 - term_ratio (higher = more terminated)
        ts_on = 1.0 - dog_on["term_ratio"]; ts_off = 1.0 - dog_off["term_ratio"]
        attr = attributable_to(f"[s{seed}] latch self-termination caused by the apical adaptation gate (ON vs OFF-lesion)",
                               ts_on, ts_off)
        result["attributable_self_termination"] = attr

        # -------- earned verdict (preconditions = INSTRUMENT validity; teeth drive go=) --------
        v = Verdict(f"D5 dendritic latch self-terminates into a discrete SWR-like window (seed {seed})")
        v.disabled("BDSP/Hebbian/STDP learning", "readout plasticity FROZEN -- this is a READ + a self-termination gate, not learning")
        v.disabled("soma-recurrence completion", "step-1 verified NO-GO at this scale; the read is D5's per-cell dendritic latch")
        v.require("d5-store-formed-dog", rec_dog["in_memory"], expect=True,
                  note="the genuine production D5 store BTSP-formed dog and it completes on D5's own read")
        v.require("instrument-deterministic", deterministic, expect=True,
                  note="two adapt-OFF reads byte-identical -> snapshot/restore removed the reused-bridge read contamination")
        v.require("latch-completes-baseline", dog_off["win_peak"], expect=lambda x: x >= COMPLETE_MIN,
                  note="the baseline (adapt-OFF) latch autonomously completes the held-out assembly in the free window")
        v.require("baseline-persists", dog_off["term_ratio"], expect=lambda x: x >= PERSIST_MIN,
                  note="the baseline latch STAYS UP the whole free window -> there is a persistent latch to terminate")
        v.reaches("adaptation-reaches-apical", 0.0, dog_on["w_max"],
                  note="the gate's adaptation variable actually accumulates on the apical compartment (w_max>0)")
        v.reaches("gate-moves-freewindow-tail", dog_off["win_tail"], dog_on["win_tail"],
                  note="the gate changes the free-window latch read-out (persistent tail -> terminated tail)")
        v.require("cat-silent-control", cat_off["win_peak"], expect=lambda x: x <= CAT_MAX,
                  note="the never-formed cat is a proper silent control (validity for the specificity tooth)")
        decided = v.decide(go=go)
        result["verdict"] = decided
        result["verdict_status"] = decided["status"]

        checks = dict(COMPLETION_SURVIVES=COMPLETION_SURVIVES, SELF_TERMINATES=SELF_TERMINATES,
                      DISCRETE=DISCRETE, SPECIFIC=SPECIFIC,
                      dog_off_win_peak=dog_off["win_peak"], dog_on_win_peak=dog_on["win_peak"],
                      dog_off_term_ratio=dog_off["term_ratio"], dog_on_term_ratio=dog_on["term_ratio"],
                      dog_on_up_ms=dog_on["up_ms"], dog_off_up_ms=dog_off["up_ms"],
                      dog_on_duty=dog_on["duty"], dog_off_duty=dog_off["duty"],
                      cat_on_win_peak=cat_on["win_peak"], dog_on_w_max=dog_on["w_max"],
                      dog_on_soma_peak=dog_on["soma_peak"])
        result["checks"] = checks
        print(f"[d5-latch-selfterm] checks={checks}", flush=True)
        del mem
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 110)
    print(f"[d5-latch-selfterm] seed={seed} VERDICT: {status}")
    if "checks" in result:
        c = result["checks"]
        print(f"    dog OFF: win_peak={c['dog_off_win_peak']} term_ratio={c['dog_off_term_ratio']} (persistent)")
        print(f"    dog ON : win_peak={c['dog_on_win_peak']} term_ratio={c['dog_on_term_ratio']} up_ms={c['dog_on_up_ms']} "
              f"(discrete transient) | cat ON win_peak={c['cat_on_win_peak']}")
    print(f"[d5-latch-selfterm] wrote {out_path}")
    print("=" * 110)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--b-adapt", type=float, default=0.8, dest="b_adapt",
                    help="Ecker-b apical adaptation increment per dAP-active step (0 = off = persistent latch)")
    ap.add_argument("--tau-w", type=float, default=150.0, dest="tau_w", help="adaptation decay time-constant (ms)")
    ap.add_argument("--tau-apical", type=float, default=15.0, dest="tau_apical", help="apical membrane tau (ms)")
    ap.add_argument("--cue-pa", type=float, default=300.0, dest="cue_pa")
    ap.add_argument("--ignite-steps", type=int, default=150, dest="ignite_steps",
                    help="sustained-cue ignition phase (ms at dt=1) -- guarantees the latch latches")
    ap.add_argument("--window-steps", type=int, default=500, dest="window_steps",
                    help="free window (cue removed) -- where self-termination is observed")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}; go_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds)
        print("\n" + "#" * 110)
        print(f"[d5-latch-selfterm] {n}-SEED SUMMARY: {summ_go}/{n} GO  seeds={seeds}  go_flags={go_flags}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} "
                  f"dog_on_win_peak={c.get('dog_on_win_peak')} dog_on_term={c.get('dog_on_term_ratio')} "
                  f"dog_off_term={c.get('dog_off_term_ratio')} up_ms={c.get('dog_on_up_ms')} "
                  f"cat_on={c.get('cat_on_win_peak')}")
        print("#" * 110)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "go_flags": go_flags, "backend": backend,
                                         "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-latch-selfterm] wrote {summ_path}")
    return 0 if all(go_flags) else 1


if __name__ == "__main__":
    sys.exit(main())
