"""FLIP-GATE verification: the GRADED apical read wired into the PRODUCTION EpisodicRecallOrgan.recall makes
learn-through-use conversation-visible at the PRODUCTION op-point — the last precondition before flipping
`BRAIN_D5_CONSOLIDATE` default 0->1.

This differs from the step-6 de-risk (`_d5_step6_graded_apical_read_derisk.py`, GradedEpisodicDapMemory subclass at a
WEAK adaptive encode) in the two ways that matter for a PRODUCTION flip:
  1. It uses the REAL production classes: EpisodicRecallOrgan.note_topic -> EpisodicDapMemory.store (the ACTUAL
     production encode strength, train_events=40) -> EpisodicRecallOrgan.recall (now emitting the graded reads) ->
     recall_disclosure (the ACTUAL reply text). No subclass, no weak-encode selection — exactly what a live turn runs.
  2. It checks the ANTI-HOLLOW bar at the REPLY level: the recall_disclosure STRING must change (recall strength mV
     rises) between a used and an un-used memory, and that difference must VANISH under the consolidation lesion
     (BRAIN_D5_CONSOLIDATE=0). If the reply does not change, the flip is NOT load-bearing and MUST NOT happen.

TEETH (per seed, primary read = depth_hold):
  * STILL_USABLE     : the production-encoded 'dog' COMPLETES at turn T (binary gate in_memory=True).
  * BINARY_BYTE_ID   : rec['apical_cue'] (from the dual read) == a direct `_apical_up_read` on the same weights
                       (the moat gate is byte-identical to HEAD — the graded read did not perturb it).
  * GRADED_MOVES     : depth_hold rises turn T -> T+k through the REAL org.recall (learn-through-use is visible).
  * MONOTONE         : the depth_hold trajectory is non-decreasing across consolidation turns (relative tol).
  * FAITHFUL         : (i) cue-specific graded_cue >= 3*max(perm,nocue); (ii) formation-lesion -> graded ~0.000
                       (the read is carried by the FORMED assembly, not a weight-blind depolarization) -> the moat.
  * MOAT             : never-recalled 'cat' stays in_memory=False and its graded read << dog's.
  * BYTE_ID_OFF      : flag OFF -> consolidate returns None -> store hash unchanged -> the later graded read is
                       IDENTICAL to turn T (the move is DRIVEN by the loop, not decoration).
  * REPLY_CHANGES    : recall_disclosure(rec_Tk) != recall_disclosure(rec_T) (the surfaced strength mV rose in the
                       reply STRING), AND recall_disclosure(rec_off) == recall_disclosure(rec_T) (vanishes under lesion).

GO (per seed): STILL_USABLE ∧ BINARY_BYTE_ID ∧ GRADED_MOVES ∧ MONOTONE ∧ FAITHFUL ∧ MOAT ∧ BYTE_ID_OFF ∧ REPLY_CHANGES
∧ DETERMINISTIC. Honest NO-GO otherwise (the flip is blocked).

  Run:    SIM_BACKEND=cupy python -m research.runners._d5_graded_production_flip_verify --seed 42
  6-seed: SIM_BACKEND=cupy python -m research.runners._d5_graded_production_flip_verify --seeds 42 43 44 100 101 102
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
from research.runners.d5_episodic_production_organ import (  # noqa: E402
    EpisodicRecallOrgan, recall_disclosure, SURFACED_GRADED_READ)
from research.runners._episodic_dap_dialogue_memory import GRADED_READS  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _reset_apical_latch, _apical_up_read)
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
from webapp import continuous_engine as CE  # noqa: E402  (the ACTUAL production wiring under test)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_graded_prodflip" / "seed42.json"

MOVE_MARGIN = 1e-3        # depth_hold move margin (mV) — well below any real consolidation move
MONO_TOL_FRAC = 0.02      # relative monotonicity tolerance (tail numerical ripple, not a dead-step)
FAITHFUL_K = 3.0          # graded cue must beat perm/nocue by this factor (the binary specificity ratio)
LESION_COLLAPSE_FRAC = 0.15


def _whash(cp, W):
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _mono_rel(traj, tol_frac=MONO_TOL_FRAC):
    move = traj[-1] - traj[0]
    if move <= 0:
        return False
    tol = tol_frac * abs(move)
    return bool(all(traj[i + 1] >= traj[i] - tol for i in range(len(traj) - 1)))


def _gh(rec, field):
    """The surfaced graded read (depth_hold) out of a recall record's cue/perm/nocue graded dict."""
    return float((rec.get(field) or {}).get(SURFACED_GRADED_READ, 0.0))


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-prodflip] seed={seed} backend={backend} te={a.train_events} n_ep={a.n_episodes} — PRODUCTION path: "
          f"note_topic->recall->consolidate->recall; does the surfaced GRADED recall strength rise in the REPLY?",
          flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    cache_key = ("d5-prodflip", seed)
    try:
        cp, _ = get_backend()
        CE.forget_session(cache_key)

        # ── PRODUCTION encode: note_topic('dog') at the real store strength (train_events=40); 'cat' never spoken ──
        org = EpisodicRecallOrgan(seed, ["cat", "dog"], verbose=False)
        org._ensure_built()
        # default train_events = production GO_DEFAULTS (40); allow probing the op-point below/above it
        org.mem.p["train_events"] = int(a.train_events)
        wrote = org.note_topic("dog")
        if not wrote:
            raise RuntimeError("note_topic('dog') did not form the assembly")
        mem = org.mem
        dslot = mem.topic_slot["dog"]; cslot = mem.topic_slot["cat"]

        # clean-rest snapshot for isolated, deterministic, weight-attributable handler reads (the step-5/6 guard)
        mem.recall("dog")  # warm/allocate cp_v_apical
        mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)
        W_before = mem.R.C.data.copy()
        hash_before = _whash(cp, W_before)
        w_dog_before = float(cp.mean(W_before[mem.R.withinA_masks[dslot]]))
        w_cat_before = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))
        w_between_before = float(cp.mean(W_before[mem.R.between_mask]))

        def handler_read(topic, W, *, lesion=False):
            """The EXACT production recall (EpisodicRecallOrgan.recall), snapshot-isolated on store-weights W so a
            T-vs-T+k comparison is purely WEIGHT-attributable."""
            restore_state(mem.bridge, snap)
            mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic, lesion=lesion)

        # ── TURN T (production recall + disclosure) ──
        rec_T = handler_read("dog", W_before)
        rec_T2 = handler_read("dog", W_before)
        rec_cat_T = handler_read("cat", W_before)
        rec_dog_T_les = handler_read("dog", W_before, lesion=True)  # formation-lesion (baseline weights)
        disc_T = recall_disclosure(rec_T)

        # BINARY BYTE-IDENTITY: the dual read's apical_cue == a direct _apical_up_read on the same weights (moat gate).
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        up_direct = _apical_up_read(mem.bridge, mem.R, [mem.held_pos_by_asm[dslot]],
                                    [mem.cue_by_asm[dslot]], mem.p["up_thresh"])
        binary_byte_id = bool(abs(float(rec_T["apical_cue"]) - float(up_direct)) < 1e-12)

        inmem_T = bool(rec_T["in_memory"])
        ac_T = float(rec_T["apical_cue"])
        gh_T = _gh(rec_T, "graded_cue")
        gh_perm_T = _gh(rec_T, "graded_perm"); gh_nocue_T = _gh(rec_T, "graded_nocue")
        gh_les_T = _gh(rec_dog_T_les, "graded_cue")
        gh_cat_T = _gh(rec_cat_T, "graded_cue")
        det_bin = abs(rec_T["apical_cue"] - rec_T2["apical_cue"]) < 1e-9
        det_grd = abs(gh_T - _gh(rec_T2, "graded_cue")) < 1e-9
        deterministic = bool(det_bin and det_grd)
        cat_never = bool(w_cat_before < 5.0 and not rec_cat_T["in_memory"])
        print(f"[d5-prodflip] TURN T: dog binary={ac_T:.4f} depth_hold={gh_T:.3f} in_memory={inmem_T} "
              f"(byte_id_binary={binary_byte_id} up_direct={up_direct:.4f}) | perm={gh_perm_T:.3f} nocue={gh_nocue_T:.3f} "
              f"| formation-lesion depth_hold={gh_les_T:.3f} | cat depth_hold={gh_cat_T:.3f} | w_dog={w_dog_before:.1f}",
              flush=True)
        print(f"[d5-prodflip]   reply@T: {disc_T}", flush=True)

        # ── LESION arm (flag OFF): mark + consolidate = NO-OP; store byte-identical + later read/reply flat ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "0"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        CE.mark_recall(cache_key, "dog")
        off_rec = CE.consolidate_used_memory(cache_key, org)  # must be None
        W_off = mem.R.C.data.copy(); hash_off = _whash(cp, W_off)
        rec_off = handler_read("dog", W_off)
        disc_off = recall_disclosure(rec_off)
        byte_identical_off = bool(off_rec is None and hash_off == hash_before)
        reply_flat_off = bool(disc_off == disc_T)
        print(f"[d5-prodflip] LESION(flag=0): consolidate->{off_rec} | store byte-identical={byte_identical_off} | "
              f"reply flat={reply_flat_off}", flush=True)

        # ── ON arm (flag ON), SAME store: n_turns of use (each re-arms mark_recall -> one consolidation round) ──
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        gh_traj = [round(gh_T, 5)]; wdog_traj = [round(w_dog_before, 3)]
        consolidated_rounds = 0; W_after = W_before; on_rec = None
        for turn in range(a.n_turns):
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_after)
            CE.mark_recall(cache_key, "dog")
            on_rec = CE.consolidate_used_memory(cache_key, org, n_episodes=a.n_episodes)
            if on_rec is not None:
                consolidated_rounds += 1
            W_after = mem.R.C.data.copy()
            wd = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
            rec_turn = handler_read("dog", W_after)
            gh_traj.append(round(_gh(rec_turn, "graded_cue"), 5))
            wdog_traj.append(round(wd, 3))
            print(f"  [turn T+{turn+1}] consolidate->{'ok' if on_rec else None} | w_dog={wd:.1f} | "
                  f"binary={rec_turn['apical_cue']:.4f} depth_hold={_gh(rec_turn,'graded_cue'):.3f} "
                  f"in_memory={rec_turn['in_memory']}", flush=True)
        consolidated = bool(consolidated_rounds > 0)
        hash_after = _whash(cp, W_after)
        w_dog_after = float(cp.mean(W_after[mem.R.withinA_masks[dslot]]))
        w_cat_after = float(cp.mean(W_after[mem.R.withinA_masks[cslot]]))
        w_between_after = float(cp.mean(W_after[mem.R.between_mask]))

        # ── TURN T+k (production recall + disclosure) ──
        rec_Tk = handler_read("dog", W_after)
        rec_cat_Tk = handler_read("cat", W_after)
        disc_Tk = recall_disclosure(rec_Tk)
        gh_Tk = _gh(rec_Tk, "graded_cue"); ac_Tk = float(rec_Tk["apical_cue"]); inmem_Tk = bool(rec_Tk["in_memory"])
        print(f"[d5-prodflip]   reply@T+k: {disc_Tk}", flush=True)

        # ── teeth ──────────────────────────────────────────────────────────────────────────────────────────────
        graded_moves = bool(gh_Tk > gh_T + MOVE_MARGIN)
        monotone = _mono_rel(gh_traj)
        faithful_specific = bool(gh_T >= FAITHFUL_K * max(gh_perm_T, gh_nocue_T, 1e-9))
        faithful_lesion = bool(gh_les_T <= LESION_COLLAPSE_FRAC * max(gh_T, 1e-9))
        faithful = bool(faithful_specific and faithful_lesion)
        dw_dog = w_dog_after - w_dog_before
        cat_drift = abs(w_cat_after - w_cat_before); between_drift = abs(w_between_after - w_between_before)
        weight_specific = bool(cat_drift <= 0.05 * max(dw_dog, 1e-6) and between_drift <= 0.05 * max(dw_dog, 1e-6))
        moat_cat = bool((not rec_cat_T["in_memory"]) and (not rec_cat_Tk["in_memory"])
                        and _gh(rec_cat_Tk, "graded_cue") <= 0.15 * max(gh_Tk, 1e-9))
        specific = bool(weight_specific and moat_cat)
        lesion_vanishes = bool(byte_identical_off and abs(_gh(rec_off, "graded_cue") - gh_T) < 1e-9 and consolidated)
        reply_changes = bool(disc_Tk != disc_T and reply_flat_off)
        binary_moves = bool(ac_Tk > ac_T + 1e-6)

        go = bool(inmem_T and binary_byte_id and graded_moves and monotone and faithful and specific
                  and lesion_vanishes and reply_changes and deterministic)

        move_treat = gh_Tk - gh_T
        move_ctrl = _gh(rec_off, "graded_cue") - gh_T
        attrib = attributable_to(f"[s{seed}] depth_hold graded move: ON vs LESION(OFF)", move_treat, move_ctrl)

        v = Verdict(f"PRODUCTION-path learn-through-use is conversation-visible via the graded depth_hold read: the "
                    f"REAL EpisodicRecallOrgan.recall + recall_disclosure surface a recall strength that RISES after "
                    f"consolidation (seed {seed}, production encode te={a.train_events})")
        v.disabled("host weight formula", "the strengthening is the substrate's OWN plateau-gated BTSP via "
                                          "continuous_engine.consolidate_used_memory")
        v.disabled("binary UP-fraction as the conversation-visible read", "quantised (step-5 flat-5/6); the graded "
                                                                          "depth_hold is the surfaced magnitude, the "
                                                                          "binary still gates in_memory (moat)")
        v.require("weak-store-completes-T", inmem_T, expect=True,
                  note="the production-encoded 'dog' COMPLETES at turn T via the BINARY gate")
        v.require("binary-byte-identical", binary_byte_id, expect=True,
                  note="the dual read's apical_cue == a direct _apical_up_read (the moat gate is unchanged)")
        v.require("handler-read-deterministic", deterministic, expect=True,
                  note="two identical isolated production recalls match on binary AND graded")
        v.require("cat-never-recalled", cat_never, expect=True, note="cat is a genuine never-formed control")
        v.require("graded-read-faithful", faithful, expect=True,
                  note="cue-specific AND collapses under the formation-lesion -> the moat is preserved")
        v.require("reply-changes-and-vanishes", reply_changes, expect=True,
                  note="the recall_disclosure STRING rises with use AND is byte-flat under the consolidation lesion")
        v.reaches("graded-strength-rises-with-use", gh_T, gh_Tk,
                  note="the PRODUCTION recall strength (depth_hold mV) rose after consolidation")
        v.control("consolidation-ON vs LESION-OFF (depth_hold move)", treatment=move_treat, control=move_ctrl,
                  min_separation=0.0, note="the graded move requires the consolidation loop; the flag OFF is byte-identical")
        decided = v.decide(go=go)
        result["verdict"] = decided; result["verdict_status"] = decided["status"]; result["attributable"] = attrib

        checks = dict(
            te=a.train_events, n_episodes=a.n_episodes, n_turns=a.n_turns, GO=go,
            STILL_USABLE=inmem_T, BINARY_BYTE_ID=binary_byte_id, up_direct=round(float(up_direct), 4),
            GRADED_MOVES=graded_moves, MONOTONE=monotone, FAITHFUL=faithful,
            faithful_specific=faithful_specific, faithful_lesion=faithful_lesion,
            MOAT=moat_cat, weight_specific=weight_specific, SPECIFIC=specific,
            BYTE_ID_OFF=byte_identical_off, LESION_VANISHES=lesion_vanishes, REPLY_CHANGES=reply_changes,
            deterministic=deterministic, cat_never=cat_never, consolidated=consolidated,
            consolidated_rounds=consolidated_rounds,
            apical_cue_T=round(ac_T, 4), apical_cue_Tk=round(ac_Tk, 4), binary_moves=binary_moves,
            depth_hold_T=round(gh_T, 4), depth_hold_Tk=round(gh_Tk, 4), depth_hold_traj=gh_traj,
            depth_hold_perm_T=round(gh_perm_T, 4), depth_hold_nocue_T=round(gh_nocue_T, 4),
            depth_hold_lesion_T=round(gh_les_T, 4), depth_hold_cat_T=round(gh_cat_T, 4),
            depth_hold_cat_Tk=round(_gh(rec_cat_Tk, "graded_cue"), 4),
            wdog_traj=wdog_traj, w_dog_before=round(w_dog_before, 3), w_dog_after=round(w_dog_after, 3),
            dw_dog=round(dw_dog, 3), cat_drift=round(cat_drift, 3), between_drift=round(between_drift, 3),
            hash_before=hash_before, hash_off=hash_off, hash_after=hash_after,
            reply_T=disc_T, reply_off=disc_off, reply_Tk=disc_Tk,
            assembly_sizes=mem.assembly_sizes, move_treat=round(move_treat, 5), move_ctrl=round(move_ctrl, 5))
        result["checks"] = checks
        print(f"[d5-prodflip] checks={json.dumps({k: checks[k] for k in checks if not str(k).startswith('reply')}, default=str)}",
              flush=True)
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
    print("=" * 118)
    print(f"[d5-prodflip] seed={seed} VERDICT: {result.get('verdict_status')} (wrote {out_path})")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-events", type=int, default=40, dest="train_events",
                    help="the production encode strength (GO_DEFAULTS=40 -> what a live note_topic uses)")
    ap.add_argument("--n-episodes", type=int, default=1, dest="n_episodes",
                    help="consolidation episodes per tick (the _D5_EPISODES the flip will set)")
    ap.add_argument("--n-turns", type=int, default=3, dest="n_turns")
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
        n = len(seeds); n_go = int(sum(go_flags))
        print("\n" + "#" * 118)
        print(f"[d5-prodflip] {n}-SEED SUMMARY (production encode te={a.train_events}, n_episodes={a.n_episodes}): "
              f"{n_go}/{n} GO  seeds={seeds}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} | binary {c.get('apical_cue_T')}->"
                  f"{c.get('apical_cue_Tk')} (moved={c.get('binary_moves')}) | depth_hold {c.get('depth_hold_T')}->"
                  f"{c.get('depth_hold_Tk')} traj={c.get('depth_hold_traj')} MOVES={c.get('GRADED_MOVES')} "
                  f"MONO={c.get('MONOTONE')} FAITHFUL={c.get('FAITHFUL')} MOAT={c.get('MOAT')} "
                  f"BYTE_ID_OFF={c.get('BYTE_ID_OFF')} REPLY_CHANGES={c.get('REPLY_CHANGES')} GO={c.get('GO')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": n_go, "go_flags": go_flags, "backend": backend,
                                         "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-prodflip] wrote {summ_path}")
    return 0 if (go_flags and all(go_flags)) else 1


if __name__ == "__main__":
    sys.exit(main())
