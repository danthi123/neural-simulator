"""ONE-BRAIN XEDGE -- LONG CONVERSATION SOAK (extends PART 3's per-turn live plasticity to production scale).

WHY. `BRAIN_ONEBRAIN_XEDGE`+`BRAIN_ONEBRAIN_XEDGE_LEARN` are now PRODUCTION-DEFAULT-ON
(`research/findings/2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.md`, commit
`fe1911f2`), which makes the d6-WM->comprehension cross-edge grow ONE in-brain self-supervised credited
step PER REAL CHAT TURN (`onebrain_xedge_production.credit_live_turn_from_comprehension`, PART 3,
`2026-08-27-onebrain-xedge-per-turn-live-plasticity-GO.md`). That finding's own "what this advances"
section names the residual explicitly: it was verified on a 24-turn / 3-seed protocol; "production
conversations are longer than the verify protocol." This runner is that soak.

NO NEW LEARNING RULE. Every credited step below calls the SAME atom PART 2/3 verified
(`XedgeProductionPool.credit_live_turn` -> `_credit_turn_step`): read the brain's OWN `amb_read`
resolution (frozen), and IFF confident (|margin|>conf) drive `teach_{resolved}` for one DA-gated credited
episode -- gate OPENED for exactly that step then RE-FROZEN. No host label ever writes a weight. This file
adds ONLY measurement: longer trajectories, checkpointed probes, and one new session shape (teach-then-
distract) to ask a question PART 3 never posed -- does unrelated intervening chat activity erode an
earlier turn's teaching (catastrophic interference)?

FOUR CHECKS (the flip's own named residual):
  1. BOUNDEDNESS over the FULL trajectory (every credited turn, not just start/end) -- <= stdp_w_max
     (F3, `HMAX`), no runaway, no collapse-to-zero on an intact (non-lesioned) taught session.
  2. NO-DEGRADATION-OVER-TURNS -- comprehension quality at EARLY vs LATE checkpoints within the SAME long
     session: (a) the xedge-focus WM-resolved balanced margin/read-rate (the PART-3 headline instrument),
     and (b) a GENERAL comprehension check on WELL-FORMED (content-decisive, WM-independent) items via
     `corg.judge(...).comprehended` -- catches the failure mode of the per-turn plasticity poisoning
     comprehension broadly, not just the taught item.
  3. SUSTAINED LOAD-BEARING at turn ~60 AND at the final turn (60/100), not just at turn 24: re-run the
     PART-3 agent-taught-vs-patient-taught role-flip probe at each checkpoint through the SAME long
     session, confirming the taught role still SIGNS the later comprehension read deep into the
     conversation.
  4. DRIFT / CATASTROPHIC-INTERFERENCE -- teach role X on the focus pool (w0) for the first `teach_turns`
     turns, then run UNRELATED per-turn credited activity on a DIFFERENT candidate pool (w1) for the rest
     of the (long) conversation, and check whether w0's taught edge / its later read decays. This is a
     stability-plasticity test PART 3 never ran (it only ever taught ONE pool for the whole session).

CPU/numpy only (`SIM_BACKEND=numpy`, set BEFORE any sim import); seeded via `cfg.seed` (the R3Pool
machinery this reuses is the SAME seeded-construction path PART 2/3 verified -- see
`tests/test_determinism.py::TestSubstrateActuallySeeded`). Reuses `onebrain_xedge_production`'s pool +
comprehension organ + `_spiking_comprehension_monitor_derisk.build_battery` -- no `sim/` edit, additive.

Run (single seed, quick):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_long_conversation_soak --seeds 42 \
      --n-turns 100 --out research/findings/raw/_onebrain_xedge_long_soak_seed42.json

Run (the full 6-seed soak this finding reports):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_long_conversation_soak \
      --seeds 42,43,44,100,101,102 --n-turns 100 \
      --out research/findings/raw/_onebrain_xedge_long_soak_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners import onebrain_xedge_production as X
from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS, HMAX
from research.runners._spiking_comprehension_monitor_derisk import build_battery


def _fresh_pool(seed: int):
    """A brand-new per-turn-live pool (mirrors `_selftest_perturn.run_session`'s reset pattern): the module
    global + the comprehension organ singleton are BOTH cleared so each session starts from an
    independently-BUILT (seeded) substrate, never a leftover from a previous session in this process."""
    X._POOL = None
    from research.runners import comprehension_production_organ as _CO
    _CO._ORGAN = None
    os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
    os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"
    X.set_live_per_turn(True)
    pool = X.get_xedge_pool(seed)
    assert pool is not None and pool.ok and pool.live_per_turn, "per-turn pool failed to build"
    return pool


def _probe(pool, corg, ambig_items, well_items, foc):
    """ONE frozen-forward-pass probe of comprehension quality (never credits -- no `_episode`/teach call).
    (a) xedge-focus quality: the content-cancelled WM-resolved balanced margin (the EXACT quantity PART 3's
    `_wm_resolved_role` thresholds) for the HELD `foc` vs a no-edge baseline pool, plus the decision-level
    resolved-role rate through the REAL `repair_target` path on ambiguous items. (b) GENERAL quality: the
    `comprehended` rate on WELL-FORMED (content-decisive, WM-independent) items -- a control that should
    stay high/flat regardless of what the cross-edge is doing, so a drop here would flag the per-turn
    plasticity poisoning comprehension broadly."""
    amb = pool.pool.xedge_amb_read
    cues = pool.pool.xedge_balanced_cues
    base_pool = pool.pool.xedge_base_pool
    baseline = float(amb(base_pool, cues)["margin"])
    wm_margins = [float(amb(foc, cues)["margin"]) for _ in range(3)]
    delta = float(np.mean(wm_margins)) - baseline

    pool.pool.xedge_focus = foc
    reads = []
    for (_lab, tag, n0, v, n1) in ambig_items:
        r = corg.repair_target(f"{n0} {v} {n1}")
        reads.append({"item": f"{n0}/{v}/{n1}", "tag": tag, "role": (r.get("role") if r else None),
                     "wm_resolved": bool(r.get("wm_resolved")) if r else False})

    well_rows = []
    for (_lab, tag, n0, v, n1) in well_items:
        j = corg.judge(f"{n0} {v} {n1}")
        well_rows.append({"item": f"{n0}/{v}/{n1}", "tag": tag,
                          "comprehended": bool(j.get("comprehended")) if j else False,
                          "margin": float(j.get("margin", 0.0)) if j else None})

    return {"baseline_margin": round(baseline, 4), "wm_delta_margin": round(delta, 4),
            "wm_resolved_reads": sum(r["wm_resolved"] for r in reads), "n_ambig": len(reads), "reads": reads,
            "general_comprehended": sum(r["comprehended"] for r in well_rows), "n_well": len(well_rows),
            "well_rows": well_rows}


def _run_soak_session(seed: int, direction: str, n_turns: int, checkpoints, ambig_items, well_items,
                      lesion_plasticity: bool = False, interference: bool = False, teach_turns: int = 10,
                      conf: float = 0.02):
    """A single LONG conversation session. Non-interference: teach `direction` on the focus pool (w0) for
    the WHOLE `n_turns`-turn session (PART 3's design, just longer + checkpointed). Interference: teach
    `direction` on w0 for `teach_turns`, then spend the REST of the (long) conversation on UNRELATED
    per-turn credited activity that holds a DIFFERENT candidate pool (w1, alternating agent/patient
    content) -- w0 receives no further direct teaching -- and keep probing w0's taught edge/read
    throughout, to see whether the intervening unrelated turns erode it."""
    pool = _fresh_pool(seed)
    corg = pool.comp_organ
    corg.ensure_built()
    foc = CAND_POOLS[0]
    distractor_foc = CAND_POOLS[1]
    pool.set_focus(foc)
    role_key = f"{foc}->{'A' if direction == 'agent' else 'P'}"

    checkpoints = sorted(set(checkpoints) | {0, n_turns})
    weight_traj = [round(float(pool.cross_weights[role_key]), 4)]
    probe_traj = [{"turn": 0, **_probe(pool, corg, ambig_items, well_items, foc)}]

    for t in range(1, n_turns + 1):
        if interference and t > teach_turns:
            distractor_direction = "agent" if (t % 2 == 0) else "patient"
            pool.credit_live_turn(distractor_direction, conf=conf, lesion_plasticity=False, focus=distractor_foc)
        else:
            pool.credit_live_turn(direction, conf=conf, lesion_plasticity=lesion_plasticity, focus=foc)
        weight_traj.append(round(float(pool.cross_weights[role_key]), 4))
        if t in checkpoints:
            probe_traj.append({"turn": t, **_probe(pool, corg, ambig_items, well_items, foc)})

    return {
        "seed": seed, "direction": direction, "n_turns": n_turns, "focus": foc,
        "interference": interference, "teach_turns": (teach_turns if interference else None),
        "distractor_focus": (distractor_foc if interference else None),
        "lesion_plasticity": lesion_plasticity,
        "w_start": weight_traj[0], "w_min": round(min(weight_traj), 4), "w_max": round(max(weight_traj), 4),
        "w_final": weight_traj[-1], "weight_traj": weight_traj,
        "bounded_F3": bool(max(weight_traj) <= HMAX + 1e-6),
        "probe_traj": probe_traj, "n_live_credited": pool.n_live_credited,
    }


def _soak_seed(seed: int, n_turns: int = 100, conf: float = 0.02):
    """The full per-seed soak: agent-taught / patient-taught / lesion (all full-length, checkpointed) +
    ONE teach-then-distract interference session. Returns every raw session plus the 4 derived verdicts."""
    batt = build_battery(seed, n_per_cond=3)
    ambig_items = [it for it in batt if it[0] == 0 and "ambig" in it[1]][:5]
    well_items = [it for it in batt if it[0] == 1][:5]

    # the taught WM-resolved read is NOT established immediately -- it's a threshold dynamic (measured on the
    # smoke run: 0/5 resolved at turn 10, 5/5 resolved by turn 15-24, matching PART 3's own 24-turn endpoint).
    # `teach_turns=30` is chosen to be safely PAST that establishment point, so the interference session's
    # "post-teach" checkpoint is testing a genuinely-established taught state, not pre-establishment noise.
    TEACH_TURNS = 30 if n_turns >= 40 else max(10, n_turns // 3)
    # checkpoints kept DELIBERATELY SPARSE (each checkpoint costs 10 frozen-read probes x4 sessions): only the
    # turns each of the 4 checks actually needs -- 10 (general-comprehension early reference), TEACH_TURNS
    # (xedge-established reference + interference post-teach snapshot), 60 (the task's explicit "turn 60"
    # sustained-load-bearing requirement, when reachable), and n_turns (the final/late reference everywhere).
    checkpoints = sorted({10, TEACH_TURNS} | ({60} if n_turns >= 60 else set()) | {n_turns})

    agent_sess = _run_soak_session(seed, "agent", n_turns, checkpoints, ambig_items, well_items)
    patient_sess = _run_soak_session(seed, "patient", n_turns, checkpoints, ambig_items, well_items)
    lesion_sess = _run_soak_session(seed, "agent", n_turns, checkpoints, ambig_items, well_items,
                                    lesion_plasticity=True)
    interference_sess = _run_soak_session(seed, "agent", n_turns, checkpoints, ambig_items, well_items,
                                          interference=True, teach_turns=TEACH_TURNS)

    def probe_at(sess, turn):
        # nearest checkpoint AT OR AFTER `turn` (checkpoints always include n_turns, so this never misses).
        for p in sess["probe_traj"]:
            if p["turn"] >= turn:
                return p
        return sess["probe_traj"][-1]

    eps = max(0.004, 3.0 * abs(agent_sess["probe_traj"][0]["baseline_margin"]))

    # ---- 1. BOUNDEDNESS (full trajectory, all 4 sessions) ----
    all_w = agent_sess["weight_traj"] + patient_sess["weight_traj"] + lesion_sess["weight_traj"] + \
        interference_sess["weight_traj"]
    bounded_F3 = bool(max(all_w) <= HMAX + 1e-6)
    grew_from_baseline = bool(agent_sess["w_final"] > 0.5 and agent_sess["w_start"] <= 0.06 and
                              patient_sess["w_final"] > 0.5 and patient_sess["w_start"] <= 0.06)
    lesion_did_not_grow = bool(lesion_sess["w_final"] <= 0.06)
    # ATTRIBUTION (whose difference IS the long-soak weight growth?): the taught (INTACT, full n_turns) session
    # and the LESIONED (frozen-gate) session run the IDENTICAL credit path -- measuring both is not the same as
    # asking whose the growth was (gap#5). `attributable_to` forces the subtraction: the intact-vs-lesion delta
    # must be (near-)100% owned by the per-turn plasticity, not by anything the credit path does with the gate
    # frozen.
    from tools.lab import attributable_to
    frac_weight = attributable_to(f"seed{seed} long-soak xedge weight growth: taught(intact, full {n_turns}-turn "
                                  "session) vs lesion(frozen gate)",
                                  agent_sess["w_final"] - agent_sess["w_start"],
                                  lesion_sess["w_final"] - lesion_sess["w_start"])

    # ---- 2. NO-DEGRADATION-OVER-TURNS (early vs late, within the SAME taught session) ----
    # two DIFFERENT "early" references: (a) turn=10 for the GENERAL (WM-independent) comprehension check --
    # that quality should be flat from turn 0, no establishment threshold applies; (b) turn=30 (`TEACH_TURNS`,
    # empirically past the WM-resolved-read establishment threshold -- see the comment above) for the
    # xedge-focus quality check, so "no degradation" compares ESTABLISHED quality to late quality, not
    # pre-establishment noise to late quality (which would be meaningless).
    early_a = probe_at(agent_sess, 10)
    late_a = probe_at(agent_sess, n_turns)
    early_p = probe_at(patient_sess, 10)
    late_p = probe_at(patient_sess, n_turns)
    est_a = probe_at(agent_sess, TEACH_TURNS)
    est_p = probe_at(patient_sess, TEACH_TURNS)
    xedge_established = bool(est_a["wm_resolved_reads"] >= 3 and est_a["wm_delta_margin"] > eps and
                             est_p["wm_resolved_reads"] >= 3 and est_p["wm_delta_margin"] < -eps)
    # xedge-focus quality must not COLLAPSE late vs its established checkpoint (some growth/noise is fine; a
    # drop to near-zero resolved reads or a sign flip on the taught session is the failure mode).
    xedge_quality_ok = bool(late_a["wm_resolved_reads"] >= max(1, est_a["wm_resolved_reads"] - 1) and
                            late_p["wm_resolved_reads"] >= max(1, est_p["wm_resolved_reads"] - 1) and
                            late_a["wm_delta_margin"] > eps and late_p["wm_delta_margin"] < -eps)
    # general (WM-independent) comprehension must stay >= early, minus a 1-item slack for read noise.
    general_ok = bool(late_a["general_comprehended"] >= early_a["general_comprehended"] - 1 and
                      late_p["general_comprehended"] >= early_p["general_comprehended"] - 1)
    no_degradation = bool(xedge_quality_ok and general_ok)

    # ---- 3. SUSTAINED LOAD-BEARING at turn ~60 and at the final turn ----
    a60, p60 = probe_at(agent_sess, 60), probe_at(patient_sess, 60)
    a_final, p_final = probe_at(agent_sess, n_turns), probe_at(patient_sess, n_turns)
    sustained_60 = bool(a60["wm_delta_margin"] > eps and p60["wm_delta_margin"] < -eps)
    sustained_final = bool(a_final["wm_delta_margin"] > eps and p_final["wm_delta_margin"] < -eps)

    # ---- 4. DRIFT / CATASTROPHIC-INTERFERENCE (teach-then-distract) ----
    post_teach = probe_at(interference_sess, TEACH_TURNS)     # right after the (established) teaching block
    post_distract = probe_at(interference_sess, n_turns)      # after (n_turns-TEACH_TURNS) UNRELATED turns
    w_post_teach = interference_sess["weight_traj"][TEACH_TURNS]
    w_post_distract = interference_sess["w_final"]
    w_drift = round(w_post_distract - w_post_teach, 4)
    margin_drift = round(post_distract["wm_delta_margin"] - post_teach["wm_delta_margin"], 4)
    # teaching must itself be ESTABLISHED at the post-teach checkpoint before "preserved" is a meaningful
    # question (see the establishment-threshold note above) -- reported explicitly so an unestablished
    # teach phase never gets silently read as "preserved" or "not preserved".
    teaching_established = bool(post_teach["wm_delta_margin"] > eps and post_teach["wm_resolved_reads"] >= 3)
    # "preserved" = the taught role STILL signs the read after distraction (same sign, not collapsed toward
    # the eps floor) -- reported regardless of outcome (a first-class result, see the finding's drift check).
    taught_preserved = bool(teaching_established and post_distract["wm_delta_margin"] > eps and
                            post_distract["wm_resolved_reads"] >= 3)

    GO = bool(bounded_F3 and grew_from_baseline and lesion_did_not_grow and no_degradation and
             sustained_60 and sustained_final)

    return {
        "seed": seed, "n_turns": n_turns, "eps": round(eps, 4), "checkpoints": checkpoints,
        "agent_session": agent_sess, "patient_session": patient_sess, "lesion_session": lesion_sess,
        "interference_session": interference_sess,
        "bounded_F3": bounded_F3, "grew_from_baseline": grew_from_baseline,
        "lesion_did_not_grow": lesion_did_not_grow,
        "frac_attributable_to_per_turn_plasticity": (None if frac_weight is None else float(frac_weight)),
        "no_degradation_over_turns": no_degradation, "xedge_established_by_teach_turns": xedge_established,
        "xedge_quality_ok": xedge_quality_ok, "general_comprehension_ok": general_ok,
        "early_late_agent": {"early_turn10": early_a, "established_checkpoint": est_a, "late_final": late_a},
        "early_late_patient": {"early_turn10": early_p, "established_checkpoint": est_p, "late_final": late_p},
        "sustained_load_bearing_turn60": sustained_60, "sustained_load_bearing_final": sustained_final,
        "checkpoint60_agent": a60, "checkpoint60_patient": p60,
        "drift_check": {
            "teach_turns": TEACH_TURNS, "teaching_established": teaching_established,
            "w_post_teach": w_post_teach, "w_post_distract_final": w_post_distract, "w_drift": w_drift,
            "margin_post_teach": post_teach["wm_delta_margin"], "margin_post_distract": post_distract["wm_delta_margin"],
            "margin_drift": margin_drift, "taught_role_preserved_after_distraction": taught_preserved,
        },
        "GO": GO,
    }


def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-turns", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = []
    for s in seeds:
        r = _soak_seed(s, n_turns=args.n_turns, conf=args.conf)
        d = r["drift_check"]
        print(f"[seed {s}] bounded_F3={r['bounded_F3']} grew={r['grew_from_baseline']} "
              f"lesion_flat={r['lesion_did_not_grow']} no_degradation={r['no_degradation_over_turns']} "
              f"sustained@60={r['sustained_load_bearing_turn60']} sustained@final={r['sustained_load_bearing_final']} "
              f"GO={r['GO']}", flush=True)
        print(f"    weights: agent {r['agent_session']['w_start']}->{r['agent_session']['w_final']} "
              f"(min={r['agent_session']['w_min']} max={r['agent_session']['w_max']}) | "
              f"patient {r['patient_session']['w_start']}->{r['patient_session']['w_final']} "
              f"(min={r['patient_session']['w_min']} max={r['patient_session']['w_max']}) | "
              f"lesion {r['lesion_session']['w_start']}->{r['lesion_session']['w_final']}")
        print(f"    early(t=10) vs late(t={args.n_turns}) agent wm_delta_margin: "
              f"{r['early_late_agent']['early_turn10']['wm_delta_margin']:+.4f} -> "
              f"{r['early_late_agent']['late_final']['wm_delta_margin']:+.4f}  "
              f"general_comprehended {r['early_late_agent']['early_turn10']['general_comprehended']}/"
              f"{r['early_late_agent']['early_turn10']['n_well']} -> "
              f"{r['early_late_agent']['late_final']['general_comprehended']}/"
              f"{r['early_late_agent']['late_final']['n_well']}")
        print(f"    drift-check (teach@0-{d['teach_turns']} on w0, distract@{d['teach_turns']+1}-{args.n_turns} on w1): "
              f"established={d['teaching_established']} w_drift={d['w_drift']:+.4f} margin_drift={d['margin_drift']:+.4f} "
              f"taught_role_preserved={d['taught_role_preserved_after_distraction']}")
        results.append(r)

    n_go = sum(r["GO"] for r in results)
    payload = {"probe": "onebrain_xedge_long_conversation_soak", "seeds": seeds, "n_turns": args.n_turns,
               "conf": args.conf, "backend": os.environ.get("SIM_BACKEND", "numpy"),
               "results": results, "n_go": n_go, "n_seeds": len(seeds),
               "note": ("Extends PART 3's per-turn live plasticity (2026-08-27-onebrain-xedge-per-turn-live-"
                        "plasticity-GO, 24 turns/3 seeds) to a LONG conversation soak -- validates the "
                        "production-default-ON flip's own named residual (production conversations are "
                        "longer than the verify protocol). Same credit atom (`credit_live_turn` -> "
                        "`_credit_turn_step`), no host label, no new learning rule. Adds: full-trajectory "
                        "boundedness, early-vs-late comprehension-quality (xedge-focus AND general "
                        "WM-independent well-formed-item comprehension), sustained load-bearing at turn "
                        "60 and the final turn, and a teach-then-distract catastrophic-interference check "
                        "(teach w0 for 10 turns, then run unrelated credited activity on w1 for the rest of "
                        "the conversation, and re-probe w0).")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\nTOTAL: {n_go}/{len(seeds)} seeds GO", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
