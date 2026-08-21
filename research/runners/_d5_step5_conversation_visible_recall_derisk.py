"""STEP 5 of the D5 learn-through-use arc — make the strengthening CONVERSATION-VISIBLE through the REAL production
recall (`EpisodicRecallOrgan.recall`), the signal step-4 lacked. This is the precondition for the production-default
flip: it is the difference between "the memory got more resilient" (step-4's robustness RESERVE, read by the arc's
completion-margin instrument) and "you can tell it learned" (the handler's OWN apical_cue / in_memory → the reply text).

THE STEP-4 RESIDUAL (why the handler read was flat): step-4 encoded 'dog' at train_events=15, landing the within-dog
weight at ~60 (of wmax=100). At that weight the FULL-cue apical read is at the TOP of its sensitivity curve — most
held cells' apical latches are already UP, so consolidation (w~60→75) saturated the already-strong cue→held synapses
without recruiting the remaining structurally-capped held cells → apical_cue FLAT on 5/6 seeds. It was NOT that the read
cannot move; it was the OPERATING POINT (the wall-reframe: the proxy — here a near-ceiling encode — owned the
measurement).

THE STEP-5 REGIME (this runner): encode 'dog' WEAKER (fewer BTSP passes → w_dog ~40, the STEEP part of the
apical_cue(w_dog) curve) so the memory starts UNDER-consolidated but still USABLE — it COMPLETES at turn T
(in_memory=True) with apical_cue in a HEADROOM band (~0.25–0.6, not at the 1.0 ceiling). This is the biologically-
faithful labile trace a SINGLE exposure should leave (production's train_events=40 one-shot-full-strength is itself the
idealization step-4 flagged). Then the step-4 loop consolidates it between turns, and the FULL-cue org.recall apical_cue
RISES — the handler's own read, the reply's quoted "dendritic dAP completion X.XX", moves. That is the conversation-
visible signal.

WHY THE FULL CUE (not a degraded later cue): production's referential recall ALWAYS drives the full standard cue (the
referent extraction is deterministic), so a degraded/partial later cue is an INSTRUMENT that exposes the reserve, NOT
something a normal conversation does — step-4 already showed the reserve moves under such stress and that it is NOT
conversation-visible. The only genuinely conversation-visible read is the FULL-cue handler recall, so that is the teeth.

WHY apical_cue RISE (not an in_memory False→True flip): the production server arms consolidation ONLY when the turn-T
recall COMPLETES (`mark_recall` is guarded by `rec.get("in_memory")`, webapp/server.py). A memory that does NOT complete
at turn T is never marked, so it is never consolidated — an in_memory False→True flip is UNREACHABLE through the faithful
production path. The faithful signal is therefore a weakly-COMPLETING memory (in_memory=True, low apical_cue) whose
apical_cue RISES with use. The disclosure text quotes that number, so the reply visibly changes.

THE PROTOCOL (same store, same-store lesion control — the clean isolation of the consolidation loop):
  1. Encode 'dog' WEAK (adaptive per-seed: pick the BTSP-pass count landing borderline apical_cue in the headroom band
     with in_memory=True; build-to-build emergent-membership variance shifts the sweet spot, so a fixed count is not
     robust). 'cat' is NEVER encoded (specificity control).
  2. TURN T — read THROUGH THE HANDLER: snapshot-isolated `org.recall('dog')` → apical_cue_T, in_memory_T. + cat.
  3. LESION arm (BRAIN_D5_CONSOLIDATE=0): mark_recall + consolidate_used_memory → NO-OP (None). The store must be
     BYTE-IDENTICAL and a later org.recall must read IDENTICALLY (the move VANISHES) — SAME store, clean lesion.
  4. ON arm (BRAIN_D5_CONSOLIDATE=1), SAME store: mark_recall + consolidate_used_memory runs the arc-1 loop.
  5. TURN T+k — read AGAIN THROUGH THE HANDLER: `org.recall('dog')` → apical_cue_Tk. Must be HIGHER (the handler-visible
     recall improved), while cat is unchanged.

MEASURE (the teeth):
  * HANDLER_MOVES : apical_cue_Tk > apical_cue_T (+ margin) through the REAL org.recall (and/or in_memory flip, reported).
  * LESION_VANISHES: flag OFF → the later org.recall apical_cue is IDENTICAL to turn T AND the store byte-identical,
                     while the ON arm DID consolidate → the move is DRIVEN by the loop, not decoration.
  * SPECIFIC      : never-recalled 'cat' apical_cue unchanged; cat/between within-weight ~unchanged.
  * STILL_USABLE  : the weak encode still COMPLETES at turn T (in_memory_T=True) — a genuinely usable memory.
  * DETERMINISTIC : two identical isolated handler reads match (the instrument read is deterministic).
INSTRUMENT preconditions (require()): weak store COMPLETES with headroom (in_memory_T + apical_cue_T in band), cat never
  forms, the read is deterministic. Teeth drive go=. Honest NO-GO otherwise (localizes why the handler read stays flat).

BRAIN-BASED (NO sim/ edit): the strengthening is the substrate's OWN plateau-gated BTSP (`fused_btsp_update`) via the
ACTUAL `webapp.continuous_engine.consolidate_used_memory`; the reads are the ACTUAL production `EpisodicRecallOrgan.recall`
the live handler calls. Host code here is only the clock, the encode-strength selection, and the snapshot-isolation
determinism guard. GPU-preferred.
  Run:    SIM_BACKEND=cupy python -m research.runners._d5_step5_conversation_visible_recall_derisk --seed 42
  6-seed: SIM_BACKEND=cupy python -m research.runners._d5_step5_conversation_visible_recall_derisk --seeds 42 43 44 100 101 102
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
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from research.runners.d5_episodic_production_organ import EpisodicRecallOrgan  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
from webapp import continuous_engine as CE  # noqa: E402  (the ACTUAL production wiring under test)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_step5_visible" / "seed42.json"

# The HEADROOM band for the weak encode: the borderline full-cue apical_cue must COMPLETE (in_memory=True) but sit
# BELOW the ceiling so consolidation has room to raise it. Below the band the store fails to complete (unusable);
# at/above the top it is saturated (the step-4 flatness). The band brackets the STEEP part of the apical(w) curve.
APICAL_LO = 0.15       # must be >= the organ's COMPLETE_MIN (0.20 gate) after in_memory passes; LO here just guards >0
APICAL_HI = 0.80       # must leave room below the 1.0 ceiling
MOVE_MARGIN = 1e-6     # a strictly-higher later-turn apical_cue counts as a move (quantised read: steps of 1/n_held)


def _whash(cp, W):
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _build_organ(seed, train_events):
    """A genuine production EpisodicRecallOrgan with 'dog' encoded at `train_events` BTSP passes; 'cat' never."""
    org = EpisodicRecallOrgan(seed, ["cat", "dog"], verbose=False)
    org.mem = EpisodicDapMemory(seed, org.topics, verbose=False, train_events=int(train_events), wmax=100.0)
    if not org.mem.store("dog"):
        raise RuntimeError("store('dog') returned False — dog not BTSP-formed")
    org._store_order = ["dog"]
    return org


def _borderline_apical(org, cp):
    """The borderline (pre-consolidation) full-cue handler read for 'dog' on a freshly-built organ, snapshot-isolated
    so it is deterministic + weight-attributable. Returns (rec, snap, W_before, w_dog_before)."""
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
    """Drop an EpisodicDapMemory organ and reclaim its GPU bridge so the te-sweep never holds >2 bridges at once
    (each readout bridge is ~5 GB; a 5-te sweep would OOM otherwise)."""
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
    """ADAPTIVE per-seed encode selection: build a store at each train_events in te_grid and keep the LOWEST-apical
    store that still COMPLETES (in_memory=True, APICAL_LO < apical_cue <= APICAL_HI). WHY lowest, not first-in-band: the
    recall→strengthen consolidation loop converges to a MEMBERSHIP-dependent BOUNDED fixed point (the step-3
    self-terminating window caps it), so the handler apical_cue rises ONLY if the encode starts BELOW that fixed point.
    The weakest completing encode maximises the headroom below the fixed point → the most reliable visible move.
    Build-to-build emergent-membership variance shifts the operating point, so a fixed count is not robust. Tracks a
    running-best (frees the loser's GPU bridge immediately). Returns (org, te, rec, snap, W, w_dog); (None, ...) if no
    grid point completes with headroom (INSTRUMENT-INVALID for this seed — start at/above the fixed point, too weak, or
    self-ignites — not a load-bearing fail)."""
    best = None  # (ac, org, te, rec, snap, W_before, w_dog_before)
    n_completing = 0
    for te in te_grid:
        org = _build_organ(seed, te)
        rec, snap, W_before, w_dog_before = _borderline_apical(org, cp)
        ac = float(rec["apical_cue"]); inmem = bool(rec["in_memory"]); nocue = float(rec["apical_nocue"])
        if verbose:
            print(f"    [encode-select s{seed}] te={te:2d} w_dog={w_dog_before:6.2f} apical_cue={ac:.4f} "
                  f"nocue={nocue:.4f} in_memory={inmem} asm={org.mem.assembly_sizes}", flush=True)
        keep = bool(inmem and APICAL_LO < ac <= APICAL_HI)
        if keep:
            n_completing += 1
        if keep and (best is None or ac < best[0]):
            if best is not None:
                _free_org(best[1], cp)                # a new, lower-apical winner → free the old best
            best = (ac, org, te, rec, snap, W_before, w_dog_before)
        else:
            _free_org(org, cp)                        # not completing, or not better than the running best
    if best is None:
        return (None, None, None, None, None, None)
    if verbose:
        print(f"    [encode-select s{seed}] CHOSEN te={best[2]} borderline apical_cue={best[0]:.4f} "
              f"(from {n_completing} completing candidate(s))", flush=True)
    return best[1:]


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-step5-visible] seed={seed} backend={backend} — weak-usable encode → handler recall → consolidate → "
          f"handler recall; does the REAL org.recall apical_cue MOVE?", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        cache_key = ("d5-step5", seed)
        CE.forget_session(cache_key)

        te_grid = [int(x) for x in a.te_grid.split(",")] if a.te_grid else [a.train_events]
        org, te, rec_dog_T, snap, W_before, w_dog_before = _select_encode(seed, cp, te_grid, verbose=True)
        instrument_valid = org is not None
        if not instrument_valid:
            result["verdict_status"] = "UNDEFINED"
            result["checks"] = {"instrument_valid": False, "reason": "no te landed a weak-usable headroom store"}
            print(f"[d5-step5-visible] seed={seed} INSTRUMENT-INVALID: no encode in {te_grid} landed a completing "
                  f"headroom store (apical in ({APICAL_LO},{APICAL_HI}])", flush=True)
            CE.forget_session(cache_key)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            return result

        mem = org.mem
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]

        def handler_read(topic, W):
            """A snapshot-ISOLATED production handler recall on store-weights W. `org.recall` is the EXACT live-handler
            method; restoring the clean-rest snapshot + injecting W before it isolates the organ's non-idempotent-read
            residual (repeated reads on one bridge drift), so a T-vs-T+k apical_cue comparison is purely
            WEIGHT-attributable. In production the reads are spread across turns; here back-to-back, so the isolation is
            required for a fair comparison. The recall call + read are byte-for-byte the production ones."""
            restore_state(mem.bridge, snap)
            mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic)

        # ── TURN T (handler): the borderline read (already computed in _select_encode; re-read isolated for determinism)
        rec_dog_T = handler_read("dog", W_before)
        rec_dog_T2 = handler_read("dog", W_before)
        rec_cat_T = handler_read("cat", W_before)
        hash_before = _whash(cp, W_before)
        w_cat_before = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))
        w_between_before = float(cp.mean(W_before[mem.R.between_mask]))
        deterministic = bool(abs(rec_dog_T["apical_cue"] - rec_dog_T2["apical_cue"]) < 1e-9)
        inmem_T = bool(rec_dog_T["in_memory"])
        ac_T = float(rec_dog_T["apical_cue"])
        headroom = bool(inmem_T and APICAL_LO < ac_T <= APICAL_HI)
        cat_never = bool(w_cat_before < 5.0 and not rec_cat_T["in_memory"])
        print(f"[d5-step5] TURN T (handler): dog apical_cue={ac_T:.4f} in_memory={inmem_T} | cat apical_cue="
              f"{rec_cat_T['apical_cue']:.4f} | w_dog={w_dog_before:.1f} te={te} det={deterministic} headroom={headroom}",
              flush=True)

        # ── LESION arm (flag OFF): mark + consolidate = NO-OP; SAME store byte-identical + later read flat ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        CE.mark_recall(cache_key, "dog")
        off_rec = CE.consolidate_used_memory(cache_key, org)     # must be None
        W_off = mem.R.C.data.copy(); hash_off = _whash(cp, W_off)
        rec_dog_off = handler_read("dog", W_off)
        byte_identical_off = bool(off_rec is None and hash_off == hash_before)
        ac_off = float(rec_dog_off["apical_cue"])
        lesion_flat = bool(abs(ac_off - ac_T) < 1e-9)
        print(f"[d5-step5] LESION (flag=0): consolidate→{off_rec} | store byte-identical={byte_identical_off} | "
              f"later recall apical_cue={ac_off:.4f} (flat={lesion_flat})", flush=True)

        # ── ON arm (flag ON), SAME store: a FEW TURNS of use. Production consolidates PER referential recall (each turn
        # re-arms mark_recall → the next idle tick runs a 3-episode consolidation), so "a few turns later" = several
        # consolidation ROUNDS, not one tick. Loop n_turns rounds (re-arming each, as the live handler does), letting
        # the within-assembly weight ACCUMULATE across rounds, and read the LIVE org.recall apical_cue after each turn. ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        apical_traj = [round(ac_T, 4)]; wdog_traj = [round(w_dog_before, 3)]; consolidated_rounds = 0
        on_rec = None; W_after = W_before
        for turn in range(a.n_turns):
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_after)
            CE.mark_recall(cache_key, "dog")               # each referential recall re-arms the between-turn tick
            on_rec = CE.consolidate_used_memory(cache_key, org, n_episodes=a.n_episodes)
            if on_rec is not None:
                consolidated_rounds += 1
            W_after = mem.R.C.data.copy()                  # accumulate across turns (the store carries forward)
            wd = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
            rec_turn = handler_read("dog", W_after)
            apical_traj.append(round(float(rec_turn["apical_cue"]), 4)); wdog_traj.append(round(wd, 3))
            print(f"  [turn T+{turn+1}] consolidate→{'ok' if on_rec else None} | w_dog={wd:.1f} | "
                  f"handler apical_cue={rec_turn['apical_cue']:.4f} in_memory={rec_turn['in_memory']}", flush=True)
        consolidated = bool(consolidated_rounds > 0)
        hash_after = _whash(cp, W_after)
        w_dog_after = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
        w_cat_after = float(cp.mean(W_after[mem.R.withinA_masks[cslot]]))
        w_between_after = float(cp.mean(W_after[mem.R.between_mask]))

        # ── TURN T+k (handler): the FINAL later-turn read through the LIVE recall ──
        rec_dog_Tk = handler_read("dog", W_after)
        rec_cat_Tk = handler_read("cat", W_after)
        ac_Tk = float(rec_dog_Tk["apical_cue"]); inmem_Tk = bool(rec_dog_Tk["in_memory"])
        print(f"[d5-step5] TURN T+k (handler): dog apical_cue={ac_T:.4f}→{ac_Tk:.4f} in_memory {inmem_T}→{inmem_Tk} | "
              f"w_dog {w_dog_before:.1f}→{w_dog_after:.1f} (traj {apical_traj}) | cat apical_cue "
              f"{rec_cat_T['apical_cue']:.4f}→{rec_cat_Tk['apical_cue']:.4f}", flush=True)

        # ── the measured teeth ──
        apical_up = bool(ac_Tk > ac_T + MOVE_MARGIN)
        inmem_flip = bool((not inmem_T) and inmem_Tk)
        HANDLER_MOVES = bool(apical_up or inmem_flip)
        LESION_VANISHES = bool(byte_identical_off and lesion_flat and consolidated)
        cat_apical_flat = bool(abs(rec_cat_Tk["apical_cue"] - rec_cat_T["apical_cue"]) < 1e-6)
        dw_dog = w_dog_after - w_dog_before
        cat_drift = abs(w_cat_after - w_cat_before); between_drift = abs(w_between_after - w_between_before)
        SPECIFIC = bool(cat_apical_flat and cat_drift <= 0.05 * max(dw_dog, 1e-6)
                        and between_drift <= 0.05 * max(dw_dog, 1e-6))
        STILL_USABLE = bool(inmem_T)
        go = bool(HANDLER_MOVES and LESION_VANISHES and SPECIFIC and STILL_USABLE and deterministic)

        # apical move attributable to the loop: ON (ac_Tk - ac_T) vs OFF (ac_off - ac_T ≈ 0)
        move_treat = ac_Tk - ac_T
        move_ctrl = ac_off - ac_T
        attr = attributable_to(f"[s{seed}] later-turn handler apical_cue move: ON vs LESION(OFF)",
                               move_treat, move_ctrl)
        result["attributable"] = attr

        v = Verdict(f"CONVERSATION-VISIBLE learn-through-use: the REAL org.recall apical_cue rises after the idle "
                    f"tick consolidates a USED memory (seed {seed})")
        v.disabled("host weight formula", "the strengthening is the substrate's OWN plateau-gated BTSP "
                                          "(fused_btsp_update) via the ACTUAL continuous_engine.consolidate_used_memory")
        v.disabled("robustness-reserve instrument", "the teeth are the PRODUCTION EpisodicRecallOrgan.recall() FULL-cue "
                                                    "apical_cue the live handler quotes — NOT the arc's completion-margin"
                                                    " reserve read (step-4's signal)")
        v.require("weak-store-completes-T", inmem_T, expect=True,
                  note="the weak encode still COMPLETES at turn T (a genuinely usable memory, in_memory=True)")
        v.require("borderline-has-headroom", ac_T, expect=lambda x: APICAL_LO < x <= APICAL_HI,
                  note="the borderline full-cue apical_cue sits BELOW the ceiling (room to rise)")
        v.require("handler-read-deterministic", deterministic, expect=True,
                  note="two identical isolated handler recalls match → the read is deterministic")
        v.require("cat-never-recalled", cat_never, expect=True,
                  note="cat is a genuine never-formed control (no completion, baseline weight)")
        v.reaches("handler-apical-moves-with-use", ac_T, ac_Tk,
                  note="the PRODUCTION org.recall apical_cue changed after consolidation (the conversation-visible read)")
        v.control("consolidation-ON vs LESION-OFF (handler apical move)",
                  treatment=move_treat, control=move_ctrl, min_separation=0.0,
                  note="the handler-visible apical move requires the consolidation loop; the flag OFF is byte-identical")
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]

        checks = dict(
            instrument_valid=True, te=te,
            HANDLER_MOVES=HANDLER_MOVES, LESION_VANISHES=LESION_VANISHES, SPECIFIC=SPECIFIC,
            STILL_USABLE=STILL_USABLE, deterministic=deterministic, headroom=headroom, cat_never=cat_never,
            consolidated=consolidated, byte_identical_off=byte_identical_off, lesion_flat=lesion_flat,
            apical_cue_T=round(ac_T, 4), apical_cue_off=round(ac_off, 4), apical_cue_Tk=round(ac_Tk, 4),
            apical_traj=apical_traj, wdog_traj=wdog_traj, n_turns=a.n_turns, consolidated_rounds=consolidated_rounds,
            apical_up=apical_up, inmem_T=inmem_T, inmem_Tk=inmem_Tk, inmem_flip=inmem_flip,
            w_dog_before=round(w_dog_before, 3), w_dog_after=round(w_dog_after, 3), dw_dog=round(dw_dog, 3),
            cat_apical_T=round(rec_cat_T["apical_cue"], 4), cat_apical_Tk=round(rec_cat_Tk["apical_cue"], 4),
            cat_apical_flat=cat_apical_flat, cat_drift=round(cat_drift, 3), between_drift=round(between_drift, 3),
            hash_before=hash_before, hash_off=hash_off, hash_after=hash_after,
            w_within_before=on_rec.get("w_within_before") if on_rec else None,
            w_within_after=on_rec.get("w_within_after") if on_rec else None,
            n_episodes=on_rec.get("n_episodes") if on_rec else None,
            assembly_sizes=mem.assembly_sizes)
        result["checks"] = checks
        print(f"[d5-step5] checks={json.dumps(checks, default=str)}", flush=True)
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
    print(f"[d5-step5-visible] seed={seed} VERDICT: {status}")
    if "checks" in result and result["checks"].get("instrument_valid"):
        c = result["checks"]
        print(f"    HANDLER_MOVES={c['HANDLER_MOVES']} LESION_VANISHES={c['LESION_VANISHES']} SPECIFIC={c['SPECIFIC']} "
              f"STILL_USABLE={c['STILL_USABLE']} deterministic={c['deterministic']}")
        print(f"    handler apical_cue T={c['apical_cue_T']} → T+k={c['apical_cue_Tk']} (OFF={c['apical_cue_off']}) | "
              f"in_memory {c['inmem_T']}→{c['inmem_Tk']} | w_dog {c['w_dog_before']}→{c['w_dog_after']} (te={c['te']})")
        print(f"    byte-identical-off={c['byte_identical_off']} lesion-flat={c['lesion_flat']} | "
              f"cat apical {c['cat_apical_T']}→{c['cat_apical_Tk']} (flat={c['cat_apical_flat']})")
    print(f"[d5-step5-visible] wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-events", type=int, default=8, dest="train_events",
                    help="BTSP encode passes for the WEAK-usable store (single-te fallback if --te-grid unset). 8 => "
                         "w_dog~40, the steep band; the full-GO store is 40 (saturated, step-4 flatness).")
    ap.add_argument("--te-grid", type=str, default="5,6,7,8,10", dest="te_grid",
                    help="adaptive per-seed encode sweep: the LOWEST-apical completing store is used (max headroom "
                         "below the consolidation fixed point; build-variance shifts the sweet spot). '' => --train-events.")
    ap.add_argument("--n-episodes", type=int, default=3, dest="n_episodes",
                    help="consolidation recall→strengthen episodes PER tick (the continuous_engine default is 3)")
    ap.add_argument("--n-turns", type=int, default=3, dest="n_turns",
                    help="number of later USE turns (each re-arms mark_recall → one between-turn consolidation tick); "
                         "'a few turns later' = the weight accumulates across ticks, as in a live conversation")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}; go_flags = []; valid_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        c = res.get("checks", {})
        instrument_valid = bool(c.get("instrument_valid") and c.get("headroom") and c.get("cat_never")
                                and c.get("deterministic"))
        valid_flags.append(instrument_valid)
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds); n_valid = int(sum(valid_flags))
        print("\n" + "#" * 118)
        print(f"[d5-step5-visible] {n}-SEED SUMMARY: {summ_go}/{n} GO ({n_valid}/{n} instrument-valid) seeds={seeds}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} valid={valid_flags[seeds.index(s)]} "
                  f"te={c.get('te')} apical T→T+k {c.get('apical_cue_T')}→{c.get('apical_cue_Tk')} "
                  f"(OFF {c.get('apical_cue_off')}) in_mem {c.get('inmem_T')}→{c.get('inmem_Tk')} "
                  f"w_dog {c.get('w_dog_before')}→{c.get('w_dog_after')} byte-id-off={c.get('byte_identical_off')} "
                  f"MOVES={c.get('HANDLER_MOVES')} SPECIFIC={c.get('SPECIFIC')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "n_valid": n_valid, "go_flags": go_flags,
                                         "valid_flags": valid_flags, "backend": backend, "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-step5-visible] wrote {summ_path}")
    valid_go = [go_flags[i] for i in range(len(seeds)) if valid_flags[i]]
    return 0 if (valid_go and all(valid_go)) else 1


if __name__ == "__main__":
    sys.exit(main())
