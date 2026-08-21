"""STEP 4 of the D5 learn-through-use arc — the PRODUCTION-INTEGRATION rung (the mission's integration-to-production
SPINE). The arc-1 recall→self-terminating-window→BTSP loop is wired UNDER `webapp/continuous_engine.py`'s idle tick
(additive, default-OFF behind BRAIN_D5_CONSOLIDATE, byte-identical when off, ON-path rollback-safe). It STRENGTHENS the
ROBUSTNESS RESERVE of a memory the brain USED: a memory recalled during a turn, CONSOLIDATED between turns by the idle
tick, survives a bigger within-recurrence lesion / completes from a sparser cue in a LATER turn — on the organ's REAL
store — and that improvement VANISHES when the consolidation loop is lesioned (the flag OFF), leaving the store
byte-identical.

⚠ HONEST BOUNDARY (adversarial-verification correction 2026-08-20 — do NOT overstate): the gain lives in the store's
robustness RESERVE, read by the arc's `reactivate(strengthen=False)` completion-MARGIN instrument — NOT in the handler's
own recall OUTPUT. The handler-visible signals (`apical_cue` / `in_memory` → the reply) are FLAT pre/post on 5/6 seeds
(only 1 seed moves), so this is NOT yet load-bearing on the CONVERSATION content (the owner's "metadata moves,
conversation doesn't" bar, feedback_faculties_must_drive_not_observe). What is proven: the wiring + the mechanism,
safe-when-off; finding a regime where the handler-visible recall actually moves is the step-5 residual (before any flip).

THE WIRING UNDER TEST (the real production code path, NOT a re-derivation of the mechanism step-3 already proved):
  * `EpisodicRecallOrgan.recall(topic)` — the EXACT call the live handler makes (webapp/server.py, the referential
    turn: `rec = eorg.recall(ref, ...)`). Its `apical_cue` is the brain's spiking dendritic-dAP completion read.
  * `webapp.continuous_engine.mark_recall(cache_key, topic)` — the handler marks the topic a turn RECALLED.
  * `webapp.continuous_engine.consolidate_used_memory(cache_key, organ)` — the idle tick's between-turn step: runs
    the arc-1 step-3 loop on the organ's REAL store (`mem.R.C.data` by object identity), strengthening the used
    memory via the substrate's OWN plateau-gated BTSP. Default-OFF behind BRAIN_D5_CONSOLIDATE.

THE PROTOCOL (same store, same-store lesion control = a clean isolation of the consolidation loop):
  1. Build a genuine production `EpisodicRecallOrgan`; encode 'dog' to BORDERLINE strength (train_events, robustness
     headroom — the honest scope: a freshly-encoded, not-yet-consolidated memory is exactly what learn-through-use
     fills; production currently forms full-strength, itself an idealization). 'cat' is NEVER encoded (specificity).
  2. TURN T — read the memory THROUGH THE HANDLER: `org.recall('dog')` (apical_cue) + robustness (max-lesion-
     survived, min-cue-current, the arc's completion read on the organ's real store). Capture W_before + its hash.
  3. LESION arm (BRAIN_D5_CONSOLIDATE=0): `mark_recall` + `consolidate_used_memory` → a NO-OP (returns None).
     The store must be BYTE-IDENTICAL (hash unchanged) and a later `org.recall` must read IDENTICALLY (the improvement
     VANISHES) — measured on the SAME store, so this is the clean lesion isolation.
  4. ON arm (BRAIN_D5_CONSOLIDATE=1), on the SAME store: `mark_recall` + `consolidate_used_memory` runs the loop →
     the within-assembly recurrence potentiates.
  5. TURN T+k — read AGAIN THROUGH THE HANDLER: `org.recall('dog')` + robustness. The used memory must be MORE
     robust/accessible (higher apical_cue, and/or survives a lesion it did not before / completes from a sparser cue).

MEASURE (the teeth):
  * LIVE_STRENGTHENS : a later-turn recall is more robust/accessible after consolidation (apical_cue up, and/or
                       max-lesion-survived up, and/or min-cue-current down) — through the handler recall.
  * LESION_VANISHES  : with the flag OFF the later-turn read is IDENTICAL to turn T (no improvement) AND the store is
                       byte-identical → the improvement is DRIVEN by the consolidation loop, not decoration.
  * SPECIFIC         : the never-recalled 'cat' recall is unchanged; the dog gain does not spill to cat/between.
  * DETERMINISTIC    : two identical handler reads on one store match (the instrument is deterministic).
INSTRUMENT preconditions (require()): store formed with headroom, the handler recall COMPLETES pre-consolidation, cat
  never forms, the read is deterministic. Teeth drive go=. Honest NO-GO otherwise (localizes what integration misses).

Reuse-by-import (NO sim/ edit): the production D5 organ + the arc's step-2/3 helpers + the ACTUAL continuous_engine
consolidation. GPU-preferred. Multi-seed with the step-3 headroom precondition gate (build-to-build emergent-membership
variance means not every draw yields a headroom store — those are INSTRUMENT-INVALID trials, not load-bearing fails).
  Run:    SIM_BACKEND=cupy python -m research.runners._d5_live_consolidation_integration_derisk --seed 42
  6-seed: SIM_BACKEND=cupy python -m research.runners._d5_live_consolidation_integration_derisk --seeds 42 43 44 100 101 102
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
from research.runners._gap5_d5_learn_through_use_derisk import (  # noqa: E402
    max_lesion_survived, min_cue_current, LESION_GRID)
from webapp import continuous_engine as CE  # noqa: E402  (the ACTUAL production wiring under test)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_live_consolidation" / "seed42.json"

COMPLETE_MIN = 0.40   # the arc's completion criterion for the robustness reads (matches step-3)

# the cfg fields the reactivate-based robustness reads mutate — saved/restored so every HANDLER recall sees clean cfg
_CFG_KEYS = (
    "enable_hebbian_learning", "enable_stdp", "enable_structural_plasticity", "enable_bdsp", "enable_btsp",
    "btsp_learning_rate", "btsp_w_max", "btsp_w_min", "btsp_elig_tau_ms", "btsp_hetero_dep", "btsp_milstein_k_dep",
    "btsp_mean_subtract", "btsp_dog_a_dep", "btsp_elig_tau_slow_ms", "btsp_win_gate_theta", "btsp_elig_exponent",
    "btsp_elig_hard_thresh", "coincidence_plateau_v_hold")


def _whash(cp, W):
    """Deterministic content hash of the store weights (host bytes)."""
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _save_cfg(cfg):
    return {k: getattr(cfg, k, None) for k in _CFG_KEYS}


def _restore_cfg(cfg, saved):
    for k, v in saved.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass


def _robustness(mem, dslot, snap, W, cue_full, rk):
    """max-lesion-survived + min-cue-current on the organ's store (the arc's completion read), cfg-clean around it."""
    cfg = mem.bridge.core_config
    saved = _save_cfg(cfg)
    try:
        ml = max_lesion_survived(mem, dslot, snap, W, cue_full, rk)
        mc = min_cue_current(mem, dslot, snap, W, cue_full, rk)
    finally:
        _restore_cfg(cfg, saved)
    return ml, mc


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-live-consolidation] seed={seed} backend={backend} — production organ + continuous_engine tick; "
          f"recall→consolidate→later-recall through the LIVE handler", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        cp, _ = get_backend()
        cache_key = ("d5-live-test", seed)
        CE.forget_session(cache_key)

        # ── build the PRODUCTION episodic organ; encode 'dog' BORDERLINE (headroom), 'cat' never (specificity) ──
        org = EpisodicRecallOrgan(seed, ["cat", "dog"], verbose=False)
        org.mem = EpisodicDapMemory(seed, org.topics, verbose=False, train_events=a.train_events, wmax=100.0)
        if not org.mem.store("dog"):
            raise RuntimeError("store('dog') returned False — dog not BTSP-formed")
        org._store_order = ["dog"]
        mem = org.mem
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]

        # a clean transient-rest snapshot for the reactivate robustness reads (weights injected per read)
        mem.recall("dog")                       # warm + allocate cp_v_apical
        mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)
        up_thresh = mem.p["up_thresh"]; v_hold = mem.p["v_hold"]
        rk = dict(tau_w=150.0, tau_apical=15.0, cue_pa=300.0, ignite_steps=80, window_steps=500,
                  up_thresh=up_thresh, v_hold=v_hold, btsp_lr=0.02, btsp_w_max=100.0,
                  btsp_elig_tau_ms=1000.0, b_adapt=0.8)
        cue_full = np.asarray(mem.cue_by_asm[dslot], dtype=np.int64)
        W_before = mem.R.C.data.copy(); hash_before = _whash(cp, W_before)
        w_dog_before = float(cp.mean(W_before[mem.R.withinA_masks[dslot]]))
        w_cat_before = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))

        def handler_read(topic, W):
            """A snapshot-ISOLATED production handler recall on store-weights W. `org.recall` is the EXACT live-handler
            method; restoring the clean-rest snapshot + injecting W before it isolates the organ's non-idempotent-read
            residual (repeated reads on one bridge drift down — the step-2 'reads contaminate' warning), so a T-vs-T+k
            apical_cue comparison is purely WEIGHT-attributable (the load-bearing question). In production the reads are
            spread across turns; here they are back-to-back, so the isolation is required for a fair comparison."""
            restore_state(mem.bridge, snap)
            mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic)

        # ── TURN T: read the memory THROUGH THE HANDLER (the exact production call, snapshot-isolated) ──
        rec_dog_T = handler_read("dog", W_before)
        rec_dog_T2 = handler_read("dog", W_before)   # determinism of the isolated handler read
        rec_cat_T = handler_read("cat", W_before)
        ml_before, mc_before = _robustness(mem, dslot, snap, W_before, cue_full, rk)
        deterministic = bool(abs(rec_dog_T["apical_cue"] - rec_dog_T2["apical_cue"]) < 1e-9)
        headroom = bool(0.0 <= ml_before < max(LESION_GRID))       # completes un-lesioned, not saturated
        recall_completes = bool(rec_dog_T["in_memory"])
        cat_never = bool(w_cat_before < 5.0 and not rec_cat_T["in_memory"])
        print(f"[d5-live] TURN T (handler): dog apical_cue={rec_dog_T['apical_cue']:.4f} in_memory="
              f"{rec_dog_T['in_memory']} | cat apical_cue={rec_cat_T['apical_cue']:.4f} | max-lesion={ml_before} "
              f"min-cue={mc_before}pA | w_dog={w_dog_before:.1f} | det={deterministic} headroom={headroom}", flush=True)

        # ── LESION arm (flag OFF): mark + consolidate = NO-OP; the SAME store must stay byte-identical + reads flat ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"
        # restore the pristine post-recall clean rest before the tick — replicates PRODUCTION (the bridge sits idle
        # between the recall turn and the tick, with no intervening reads; the arc's snapshot/restore discipline).
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        CE.mark_recall(cache_key, "dog")
        off_rec = CE.consolidate_used_memory(cache_key, org)     # must be None
        W_off = mem.R.C.data.copy(); hash_off = _whash(cp, W_off)
        rec_dog_off = handler_read("dog", W_off)
        ml_off, mc_off = _robustness(mem, dslot, snap, W_off, cue_full, rk)
        byte_identical_off = bool(off_rec is None and hash_off == hash_before)
        lesion_flat = bool(abs(rec_dog_off["apical_cue"] - rec_dog_T["apical_cue"]) < 1e-9
                           and ml_off == ml_before and mc_off == mc_before)
        print(f"[d5-live] LESION (flag=0): consolidate→{off_rec} | store byte-identical={byte_identical_off} | "
              f"later recall apical_cue={rec_dog_off['apical_cue']:.4f} (flat={lesion_flat}) ml={ml_off} mc={mc_off}",
              flush=True)

        # ── ON arm (flag ON), SAME store: mark + consolidate runs the loop → the used memory strengthens ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
        # again restore the pristine post-recall rest (production idle condition) before the tick fires
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        CE.mark_recall(cache_key, "dog")
        on_rec = CE.consolidate_used_memory(cache_key, org)      # a record (the tick actually ran)
        consolidated = bool(on_rec is not None)
        W_after = mem.R.C.data.copy(); hash_after = _whash(cp, W_after)
        w_dog_after = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
        w_cat_after = float(cp.mean(W_after[mem.R.withinA_masks[cslot]]))
        w_between_before = float(cp.mean(W_before[mem.R.between_mask]))
        w_between_after = float(cp.mean(W_after[mem.R.between_mask]))

        # ── TURN T+k: read AGAIN THROUGH THE HANDLER (the later turn, snapshot-isolated) ──
        rec_dog_Tk = handler_read("dog", W_after)
        rec_cat_Tk = handler_read("cat", W_after)
        ml_after, mc_after = _robustness(mem, dslot, snap, W_after, cue_full, rk)
        print(f"[d5-live] TURN T+k (handler): dog apical_cue={rec_dog_T['apical_cue']:.4f}→"
              f"{rec_dog_Tk['apical_cue']:.4f} | max-lesion {ml_before}→{ml_after} | min-cue {mc_before}→{mc_after}pA "
              f"| w_dog {w_dog_before:.1f}→{w_dog_after:.1f} | cat apical_cue {rec_cat_T['apical_cue']:.4f}→"
              f"{rec_cat_Tk['apical_cue']:.4f}", flush=True)

        # ── the measured teeth ──
        # Robustness SCORE (higher = more robust/accessible): rewards a larger surviving-lesion AND a smaller
        # completion-cue. The full-cue apical_cue read SATURATES at 1.0 for a completing store, so the sensitive
        # load-bearing signal is robustness-under-stress (survives a lesion / completes from a sparser cue) — exactly
        # the arc's measure. apical_cue is still reported (and moves on builds where it is NOT saturated).
        def _robscore(ml, mc):
            return float(ml) * 1000.0 - float(mc)
        rob_before = _robscore(ml_before, mc_before)
        rob_after = _robscore(ml_after, mc_after)
        rob_off = _robscore(ml_off, mc_off)
        rob_treat = rob_after - rob_before       # ON: later-turn robustness gain
        rob_ctrl = rob_off - rob_before          # OFF (lesion): must be ~0
        apical_up = bool(rec_dog_Tk["apical_cue"] > rec_dog_T["apical_cue"] + 1e-6)
        lesion_up = bool(ml_after > ml_before)
        cue_down = bool(mc_after < mc_before)
        LIVE_STRENGTHENS = bool(apical_up or lesion_up or cue_down)
        LESION_VANISHES = bool(byte_identical_off and lesion_flat and consolidated and abs(rob_ctrl) < 1e-9)
        cat_apical_flat = bool(abs(rec_cat_Tk["apical_cue"] - rec_cat_T["apical_cue"]) < 1e-6)
        cat_drift = abs(w_cat_after - w_cat_before); between_drift = abs(w_between_after - w_between_before)
        dw_dog = w_dog_after - w_dog_before
        SPECIFIC = bool(cat_apical_flat and cat_drift <= 0.05 * max(dw_dog, 1e-6)
                        and between_drift <= 0.05 * max(dw_dog, 1e-6))
        go = bool(LIVE_STRENGTHENS and LESION_VANISHES and SPECIFIC and deterministic)

        attr = attributable_to(f"[s{seed}] later-turn robustness gain: WITH-consolidation(ON) vs LESION(OFF)",
                               rob_treat, rob_ctrl)
        result["attributable"] = attr

        # ── earned verdict ──
        v = Verdict(f"LIVE learn-through-use: a used memory is more robust in a later turn via the idle tick (seed {seed})")
        v.disabled("host weight formula", "the strengthening is the substrate's OWN plateau-gated BTSP "
                                          "(fused_btsp_update), written back to the organ's store by object identity")
        v.disabled("standalone probe", "the T and T+k reads are the PRODUCTION EpisodicRecallOrgan.recall() the live "
                                       "handler calls; consolidation is the ACTUAL continuous_engine tick function")
        v.require("d5-store-formed-dog", w_dog_before, expect=lambda x: x > 20.0,
                  note="'dog' BTSP-formed to a borderline within-weight (headroom to strengthen)")
        v.require("handler-recall-completes-T", recall_completes, expect=True,
                  note="the live handler recall COMPLETES the assembly before consolidation (a real memory to use)")
        v.require("borderline-has-headroom", ml_before, expect=lambda x: 0.0 <= x < max(LESION_GRID),
                  note="the store completes un-lesioned but is fragile to lesion (robustness headroom)")
        v.require("handler-read-deterministic", deterministic, expect=True,
                  note="two identical handler recalls match → the instrument read is deterministic")
        v.require("cat-never-recalled", cat_never, expect=True,
                  note="cat is a genuine never-formed control (no completion, baseline weight)")
        v.reaches("robustness-moves-with-use", float(ml_before), float(ml_after),
                  note="the later-turn max-lesion-survived robustness read changed after consolidation")
        v.control("consolidation-ON vs LESION-OFF (later-turn robustness gain)",
                  treatment=rob_treat, control=rob_ctrl, min_separation=0.0,
                  note="the later-turn robustness gain requires the consolidation loop; the flag OFF is byte-identical")
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]

        checks = dict(
            LIVE_STRENGTHENS=LIVE_STRENGTHENS, LESION_VANISHES=LESION_VANISHES, SPECIFIC=SPECIFIC,
            deterministic=deterministic, headroom=headroom, recall_completes=recall_completes, cat_never=cat_never,
            consolidated=consolidated, byte_identical_off=byte_identical_off, lesion_flat=lesion_flat,
            apical_cue_T=round(rec_dog_T["apical_cue"], 4), apical_cue_off=round(rec_dog_off["apical_cue"], 4),
            apical_cue_Tk=round(rec_dog_Tk["apical_cue"], 4), apical_up=apical_up,
            max_lesion_before=ml_before, max_lesion_after=ml_after, lesion_up=lesion_up,
            min_cue_before=mc_before, min_cue_after=mc_after, cue_down=cue_down,
            max_lesion_off=ml_off, min_cue_off=mc_off,
            rob_before=round(rob_before, 2), rob_after=round(rob_after, 2), rob_off=round(rob_off, 2),
            rob_treat=round(rob_treat, 2), rob_ctrl=round(rob_ctrl, 2),
            w_dog_before=round(w_dog_before, 3), w_dog_after=round(w_dog_after, 3), dw_dog=round(dw_dog, 3),
            cat_apical_T=round(rec_cat_T["apical_cue"], 4), cat_apical_Tk=round(rec_cat_Tk["apical_cue"], 4),
            cat_apical_flat=cat_apical_flat, cat_drift=round(cat_drift, 3), between_drift=round(between_drift, 3),
            hash_before=hash_before, hash_off=hash_off, hash_after=hash_after,
            w_within_before=on_rec.get("w_within_before") if on_rec else None,
            w_within_after=on_rec.get("w_within_after") if on_rec else None,
            n_episodes=on_rec.get("n_episodes") if on_rec else None,
            n_cue_full=int(len(cue_full)), assembly_sizes=mem.assembly_sizes)
        result["checks"] = checks
        print(f"[d5-live] checks={json.dumps(checks, default=str)}", flush=True)
        CE.forget_session(cache_key)
        del mem, org
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()
    finally:
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"   # leave the flag OFF (default) for any following run

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 118)
    print(f"[d5-live-consolidation] seed={seed} VERDICT: {status}")
    if "checks" in result:
        c = result["checks"]
        print(f"    LIVE_STRENGTHENS={c['LIVE_STRENGTHENS']} LESION_VANISHES={c['LESION_VANISHES']} "
              f"SPECIFIC={c['SPECIFIC']} deterministic={c['deterministic']} headroom={c['headroom']}")
        print(f"    handler apical_cue T={c['apical_cue_T']} → T+k={c['apical_cue_Tk']} (OFF={c['apical_cue_off']}) | "
              f"max-lesion {c['max_lesion_before']}→{c['max_lesion_after']} | min-cue "
              f"{c['min_cue_before']}→{c['min_cue_after']}pA | w_dog {c['w_dog_before']}→{c['w_dog_after']}")
        print(f"    byte-identical-off={c['byte_identical_off']} lesion-flat={c['lesion_flat']} | "
              f"cat apical {c['cat_apical_T']}→{c['cat_apical_Tk']} (flat={c['cat_apical_flat']}) cat_drift={c['cat_drift']}")
    print(f"[d5-live-consolidation] wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-events", type=int, default=15, dest="train_events",
                    help="BTSP encode passes for the borderline store (15 => completes un-lesioned but lesion-fragile "
                         "= robustness headroom; the full-GO store is 40)")
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
        instrument_valid = bool(c.get("headroom") and c.get("recall_completes") and c.get("cat_never")
                                and c.get("deterministic"))
        valid_flags.append(instrument_valid)
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds); n_valid = int(sum(valid_flags))
        print("\n" + "#" * 118)
        print(f"[d5-live-consolidation] {n}-SEED SUMMARY: {summ_go}/{n} GO ({n_valid}/{n} instrument-valid) seeds={seeds}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} valid={bool(c.get('headroom') and c.get('recall_completes') and c.get('cat_never') and c.get('deterministic'))} "
                  f"apical T→T+k {c.get('apical_cue_T')}→{c.get('apical_cue_Tk')} (OFF {c.get('apical_cue_off')}) "
                  f"ml {c.get('max_lesion_before')}→{c.get('max_lesion_after')} mc {c.get('min_cue_before')}→"
                  f"{c.get('min_cue_after')} byte-id-off={c.get('byte_identical_off')} lesion-flat={c.get('lesion_flat')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "n_valid": n_valid, "go_flags": go_flags,
                                         "valid_flags": valid_flags, "backend": backend, "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-live-consolidation] wrote {summ_path}")
    # GO iff every INSTRUMENT-VALID seed is GO (invalid draws are not load-bearing fails)
    valid_go = [go_flags[i] for i in range(len(seeds)) if valid_flags[i]]
    return 0 if (valid_go and all(valid_go)) else 1


if __name__ == "__main__":
    sys.exit(main())
