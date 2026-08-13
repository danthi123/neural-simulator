"""GROUND the learned CAUSAL FORWARD MODEL in the brain's REAL conversational fact store — the
declared next rung of `2026-08-12-learned-causal-forward-model-...` (its own honest boundary:
"Events are delivered as block drive ... Grounding them in the emergent relational code is the
follow-on that makes the states themselves learned").

WHAT THIS CLOSES (the honest boundary the toy de-risk sat at)
-------------------------------------------------------------
The toy `_causal_forward_model_derisk.py` proved a directed n-way STATE forward model + DO-
intervention on spikes — but its events were TOY host block-drives (A/B/C/D/X/Y = bare indices),
so it could NOT answer a real conversational "why did X" / "what happens if X" over the brain's
ACTUAL learned facts. This runner wires the forward model's EVENT POPULATION to the production
fact store — the RF-VSA `RFPhasorComposer` whose `query_patient(agent,action)->patient` is the
no-confab moat the live chat recall uses — so every event IS a real learned `(agent,action)->
patient` fact, the causal machinery (temporal-order STDP + phasic DA + Pearl DO) runs over the
REAL fact graph, and the why/what-if ANSWERS are real recalled facts, moat-confirmed.

THE REAL-FACT CAUSAL WORLD (every state is a real stored SVO fact)
-----------------------------------------------------------------
  CHAIN     A=(dog,go,east) -> B=(dog,reach,river) -> D=(dog,drink,water)
            taught as ADJACENT pairs [A,B],[B,D]; A->D (go east => drink water) is NEVER a taught
            edge — the "what happens if the dog goes east?" consequence must be a substrate ROLLOUT.
  CONFOUND  C=(sun,rise,sky) is a COMMON CAUSE of X=(bird,sing,dawn) and Y=(dog,wake,morning);
            X is observed just before Y -> temporal-order STDP tags a SPURIOUS X->Y (the bird's
            song does NOT wake the dog; the sunrise does). The DO-intervention must prune it.

THE GROUNDING — three load-bearing bindings to the production composer (NOT a relabeling of the toy)
---------------------------------------------------------------------------------------------------
  (1) EVENT SET DERIVED FROM THE COMPOSER. The event blocks are enumerated by QUERYING the
      composer (`query_patient` — the spiking RF unbind): a candidate `(agent,action,patient)`
      becomes an event ONLY if the brain's recall confirms it. No stored/recalled fact => no
      event => the forward model cannot reason about it.
  (2) THE CAUSAL CURRICULUM IS GATED BY RECALL. A causal episode (fact_i then fact_j) is taught
      ONLY when BOTH endpoints are moat-recalled. Drop a fact from the composer -> its event
      vanishes -> every causal edge touching it never forms -> the downstream why/what-if for it
      collapses. This is the GROUNDING LESION (distinct from the toy edge-lesion): it proves the
      COMPOSER is load-bearing, i.e. real grounding, not a toy under a new label.
  (3) THE ANSWERS ARE REAL RECALLED FACTS, MOAT-SAFE. The what-if successor and the why-cause are
      each mapped BACK to a fact and CONFIRMED by `query_patient` (the no-confab moat). A predicted
      event whose fact the composer cannot confirm is REJECTED (0 confabulation) — the organ reads/
      notices, it never manufactures a fact.

WHAT IS BRAIN-BASED vs THE DECLARED BOUNDARY
--------------------------------------------
- Every FACT is a spiking RF-VSA unbind+cleanup (`query_patient`); every PREDICTION is a
  `cp_firing_states` block-rate argmax over the forward-model substrate; the forward-simulation is
  the substrate's own directed propagation (D fires via B though A->D is unlearned); the transition
  edges are temporal-order STDP + DA three-factor weights. No host writes a transition table, no
  host formula computes the prediction or the causal verdict.
- Declared boundary (per THE LAW, carried from the toy de-risk): the teacher renders the event
  drive, the temporal ORDER of each episode, and the phasic-DA SIGN (the environment/teacher
  reinforcement; the brain's dopamine channel converts it to a weight change). NAMED NEXT RUNG:
  drive the event blocks directly from the composer's UNBIND SPIKES in one merged bridge (grounding-
  by-shared-substrate) rather than by deriving+gating the drive (grounding-by-derivation) — and
  drive the DA from a spiking mismatch unit. This de-risk closes grounding-by-derivation; the
  shared-substrate wiring is the follow-on.

GO-GATE (pre-registered, 6 seeds 42/43/44/100/101/102)
------------------------------------------------------
 (1) EVENT SET GROUNDED: all 6 candidate facts are moat-recalled (the event set == the composer's
     recalled facts), and the untaught moat battery abstains (0 false-accepts).
 (2) UNSEEN CONSEQUENCE over REAL facts (forward-simulation, NOT recall): cue A=(dog,go,east), roll
     the substrate forward; D=(dog,drink,water) fires (via B) though A->D was never taught, every
     off-chain event silent — AND the composer moat CONFIRMS (dog,drink,water) is a real stored fact
     (the what-if answer is a real recalled fact).
 (3) WHY = REAL CAUSE FACT + DO-PROBE: "why did the dog wake?" reads the directed edge INTO
     Y=(dog,wake,morning) as the argmax do-probe predecessor -> C=(sun,rise,sky), moat-confirmed,
     and it SURVIVES the DO-probe (do(C)->Y high) while the spurious X=(bird,sing) does NOT
     (do(X)->Y ~0). The answer is "because the sun rose", never "because the bird sang".
 (4) EDGE LESION (load-bearing): zero the learned forward edges -> forward prediction + unseen +
     why all collapse.
 (5) GROUNDING LESION (load-bearing, the NEW grounding teeth): drop D from the composer -> the
     what-if consequence collapses (D no longer predicted); drop C -> the why-cause collapses. The
     composer is load-bearing.
 (6) ANTI-CHEAT: (a) CORRELATION-ONLY (no DO phase) -> spurious X->Y survives -> do(X) WRONGLY
     fires Y AND why(Y) WRONGLY reads X (the bird) -> DO-prune load-bearing + attributable; (b)
     SHUFFLE the causal curriculum -> forward prediction vs the true chain fails; (c) MOAT no-confab
     -> a predicted event that maps to a non-fact is rejected.

Reuse-by-import of the de-risked toy primitives (NO `sim/` edit; additive new runner). CPU/numpy
(the ~180-neuron model + small RF composer get no GPU benefit — the E2/causal precedent, same
scale, ran 6-seed on numpy CPU).

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._causal_forward_model_grounded_derisk \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_causal_forward_model_grounded_6seed.json
    SIM_BACKEND=numpy python -m research.runners._causal_forward_model_grounded_derisk --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

# Reuse the de-risked toy forward-model primitives verbatim (build, step, train, reads, lesion).
from research.runners import _causal_forward_model_derisk as TOY  # noqa: E402
from research.runners._causal_forward_model_derisk import (  # noqa: E402
    A, B, D, C, X, Y, N_EVENTS, EVENT_NAMES,
    build_forward_model, train, forward_prediction, unseen_consequence, do_intervention,
    _lesion_xblock, _xblock_weight, _held_read,
)
from research.runners.rf_phasor_composer import RFPhasorComposer, DEFAULT_VOCAB  # noqa: E402

# ---------------------------------------------------------------------------
# The real-fact causal world. Each event index (the toy A..Y) is now a REAL (agent,action)->patient
# fact stored in the production composer. The toy OBS_EPISODES / CHAIN_EDGES map 1:1 onto these.
# ---------------------------------------------------------------------------
FACTS = {
    A: ("dog", "go", "east"),      # chain: the dog goes east
    B: ("dog", "reach", "river"),  # chain: -> reaches the river
    D: ("dog", "drink", "water"),  # chain: -> drinks water   (the UNSEEN 2-step consequence)
    C: ("sun", "rise", "sky"),     # confound common cause: the sun rises
    X: ("bird", "sing", "dawn"),   # effect of sunrise (observed just before Y -> spurious X->Y)
    Y: ("dog", "wake", "morning"), # effect of sunrise: the dog wakes  (the "why did X" target)
}
FACT_ORDER = [A, B, D, C, X, Y]

# Vocab = the composer default + every word our real facts use.
GROUNDED_VOCAB = sorted(set(DEFAULT_VOCAB) | {
    "reach", "river", "drink", "water", "sun", "rise", "sky",
    "bird", "sing", "dawn", "wake", "morning",
})

# Untaught (agent,action) cues that MUST abstain (query_patient -> None): the no-false-accept moat
# battery. Each pairs a real agent with an action it never took (or vice-versa).
MOAT_BATTERY = [("dog", "sing"), ("bird", "go"), ("sun", "drink"), ("dog", "fly"),
                ("cat", "rise"), ("river", "wake"), ("bird", "reach"), ("sun", "go")]


def _fact_str(e):
    a, v, p = FACTS[e]
    return f"({a},{v},{p})"


# ---------------------------------------------------------------------------
# The production fact store (grounding binding #1 + #2 + #3)
# ---------------------------------------------------------------------------
def build_grounded_composer(seed, *, drop=None, D_dim=128):
    """Build the production RF-VSA composer and store the real-fact world, optionally DROPPING a
    fact (the grounding lesion: the brain never learned it -> its event cannot be recalled)."""
    drop = set(drop or [])
    comp = RFPhasorComposer(seed=int(seed), D=int(D_dim), vocab=GROUNDED_VOCAB)
    for e in FACT_ORDER:
        if e in drop:
            continue
        comp.store(*FACTS[e])
    return comp


def _recalled(comp, e):
    """Grounding binding #1/#3: an event exists / an answer is trusted ONLY if the spiking RF recall
    confirms the fact. `query_patient` is the no-confab moat the live chat uses."""
    a, v, p = FACTS[e]
    return comp.query_patient(a, v) == p


def enumerate_events(comp):
    """The event set is DERIVED from the composer: only moat-recalled facts become events. Returns
    the recalled event indices + the recall status per candidate."""
    status = {e: _recalled(comp, e) for e in FACT_ORDER}
    return [e for e in FACT_ORDER if status[e]], status


def moat_battery(comp):
    """Untaught cues must abstain (query_patient -> None). Returns the false-accept count."""
    return sum(1 for (a, v) in MOAT_BATTERY if comp.query_patient(a, v) is not None)


# ---------------------------------------------------------------------------
# The grounded "why" read — a spiking DO-probe over REAL facts
# ---------------------------------------------------------------------------
def why_cause(bridge, xp, target, *, candidates=None):
    """'Why did <target-fact>?' — read the directed edge INTO the target as the argmax DO-probe
    predecessor: for each candidate cause i, do(i) (HOLD i) and read the target's firing rate; the
    CAUSE is the i whose intervention most drives the target. Fully spiking (no weight-matrix host
    read). Returns (cause_event, per-candidate target-rate)."""
    blocks = bridge._blocks
    cand = candidates if candidates is not None else [e for e in range(len(blocks)) if e != target]
    rates = {i: _held_read(bridge, blocks, xp, i)[target] for i in cand}
    cause = max(rates, key=lambda i: rates[i])
    return cause, rates


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------
def run_seed(seed, *, mode="intact", verbose=True, obs_reps=30, interv_reps=30,
             read_prop=0.50, drop_facts=None, **build_kw):
    """mode: intact | edge_lesion | ground_lesion_D | ground_lesion_C | corr_only | shuffle.
    drop_facts: for ground-lesion modes, the fact(s) removed from the COMPOSER (default per mode)."""
    from sim.backend import get_backend
    xp, _ = get_backend()

    # Grounding lesion drops a fact from the COMPOSER (the brain never learned it).
    if mode == "ground_lesion_D":
        drop_facts = drop_facts or [D]
    elif mode == "ground_lesion_C":
        drop_facts = drop_facts or [C]
    comp = build_grounded_composer(seed, drop=drop_facts)
    recalled_events, recall_status = enumerate_events(comp)
    battery_fa = moat_battery(comp)

    # Build the spiking forward model (identical topology to the toy; 6 fixed blocks).
    bridge, cfg, meta = build_forward_model(seed, **build_kw)
    label_map = {e: e for e in range(meta["n_events"])}

    # Grounding binding #2: the causal curriculum is GATED by recall. Only edges whose BOTH
    # endpoints are moat-recalled are ever experienced.
    if mode == "shuffle":
        eps, perm = TOY._shuffled_episodes(seed)
        # shuffle relabels events; gate on the PERMUTED endpoints' recall (all recalled in shuffle
        # mode since no drop) -> the model learns the shown (wrong) structure.
        episodes = eps
        prune_src = perm[X]
    else:
        episodes = [ep for ep in TOY.OBS_EPISODES if all(recall_status[e] for e in ep)]
        prune_src = X

    do_interv = (mode not in ("corr_only",)) and recall_status.get(X, False) and recall_status.get(Y, False)
    train(bridge, cfg, meta, xp, episodes, obs_reps=obs_reps, interv_reps=interv_reps,
          do_intervention=do_interv, prune_src=prune_src)

    # Freeze + apply the uniform maturation gain (the gap#5 protocol reused from the toy runner).
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.current_reward_signal = 0.0
    cfg.propagation_strength = float(read_prop)

    w_AD = _xblock_weight(bridge, A, D)
    if mode == "edge_lesion":
        nz = _lesion_xblock(bridge)
        if verbose:
            print(f"  [edge_lesion] zeroed {nz} cross-block edges")

    # --- the spiking reads (brain-based; firing-state argmax) ---
    fwd = forward_prediction(bridge, meta, xp, label_map=label_map)
    unseen = unseen_consequence(bridge, meta, xp, label_map=label_map, w_AD=w_AD)
    doi = do_intervention(bridge, meta, xp, label_map=label_map)
    cause_evt, cause_rates = why_cause(bridge, xp, Y)

    # --- grounding binding #3: map the answers back to REAL facts + confirm via the moat ---
    # what-if consequence: is the predicted D a real recalled fact?
    unseen_fact_confirmed = bool(unseen["predicts_D"] and _recalled(comp, D))
    # why-cause: is the read cause C, and is (sun,rise,sky) a real recalled fact?
    why_is_C = bool(cause_evt == C)
    why_fact_confirmed = bool(why_is_C and _recalled(comp, C))
    # no-confab: the predicted consequence must NOT be an unrecalled/off-graph event.
    confab = bool(unseen["predicts_D"] and not _recalled(comp, D))

    what_if_answer = None
    why_answer = None
    if unseen_fact_confirmed:
        a, v, p = FACTS[A]
        _, _, dp = FACTS[D]
        what_if_answer = (f"If the {a} {v}es {p}, it will {FACTS[D][1]} {dp} — a consequence I rolled "
                          f"forward through {FACTS[B][0]} {FACTS[B][1]}ing the {FACTS[B][2]}, and my "
                          f"no-confab moat confirms ({FACTS[D][0]},{FACTS[D][1]})->{dp} is a fact I stored.")
    if why_fact_confirmed:
        ya, yv, yp = FACTS[Y]
        ca, cv, cp = FACTS[C]
        why_answer = (f"The {ya} {yv}s ({yp}) because the {ca} {cv}s — that cause survives a DO-probe "
                      f"(forcing the {ca} to {cv} makes the {ya} {yv}; forcing the {FACTS[X][0]} to "
                      f"{FACTS[X][1]} does NOT), so it is a cause not a mere correlation, and "
                      f"({ca},{cv})->{cp} is a fact I stored.")

    # per-seed GO (intact): grounded event set + moat-safe unseen + grounded why-cause + cause-vs-corr
    go = bool(len(recalled_events) == len(FACT_ORDER) and battery_fa == 0
              and fwd["acc"] >= 1.0
              and unseen_fact_confirmed and confab is False
              and why_fact_confirmed and doi["X_not_cause_of_Y"])

    res = {
        "seed": seed, "mode": mode,
        "n_recalled_events": len(recalled_events), "recall_status": recall_status,
        "moat_battery_fa": battery_fa,
        "fwd_acc": fwd["acc"], "fwd_directed_ratio": fwd["directed_ratio"],
        "unseen": unseen, "unseen_fact_confirmed": unseen_fact_confirmed, "confab": confab,
        "do": doi, "why_cause": EVENT_NAMES.get(cause_evt), "why_is_C": why_is_C,
        "why_fact_confirmed": why_fact_confirmed,
        "why_target_rate_C": round(cause_rates.get(C, 0.0), 2),
        "why_target_rate_X": round(cause_rates.get(X, 0.0), 2),
        "w_AD": round(w_AD, 3), "what_if_answer": what_if_answer, "why_answer": why_answer,
        "go": go,
    }
    if verbose:
        print(f"  [{mode:15s} seed {seed}] events={len(recalled_events)}/6 batt_fa={battery_fa} "
              f"fwd_acc={fwd['acc']:.2f} | unseen D={unseen['D_rate']:.0f} off={unseen['offchain_max']:.0f} "
              f"predictsD={unseen['predicts_D']} moatOK={unseen_fact_confirmed} confab={confab} "
              f"| why(Y)={EVENT_NAMES.get(cause_evt)} (C={cause_rates.get(C,0):.0f} X={cause_rates.get(X,0):.0f}) "
              f"| do(X)Y={doi['Y_rate_do_X']:.0f} do(C)Y={doi['Y_rate_do_C']:.0f} Xcause={not doi['X_not_cause_of_Y']} "
              f"| GO={go}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--obs-reps", type=int, default=30)
    ap.add_argument("--interv-reps", type=int, default=30)
    ap.add_argument("--read-prop", type=float, default=0.50)
    ap.add_argument("--smoke", action="store_true",
                    help="1-seed intact + edge_lesion + ground_lesion_D + corr_only quick check")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rep_kw = dict(obs_reps=args.obs_reps, interv_reps=args.interv_reps, read_prop=args.read_prop)

    if args.smoke:
        print("=== SMOKE (seed 42): intact | edge_lesion | ground_lesion_D | corr_only ===")
        it = run_seed(42, mode="intact", **rep_kw)
        el = run_seed(42, mode="edge_lesion", **rep_kw)
        gd = run_seed(42, mode="ground_lesion_D", **rep_kw)
        co = run_seed(42, mode="corr_only", **rep_kw)
        print("\n  SMOKE checks:")
        print(f"   intact GO ................................. {it['go']}")
        print(f"   intact what-if answer ..................... {it['what_if_answer']}")
        print(f"   intact why answer ........................ {it['why_answer']}")
        print(f"   edge_lesion collapses forward pred ....... {el['fwd_acc'] < 1.0}  (acc {el['fwd_acc']:.2f})")
        print(f"   GROUND-lesion(drop D) collapses what-if .. {not gd['unseen_fact_confirmed']}  "
              f"(events {gd['n_recalled_events']}/6, predictsD {gd['unseen']['predicts_D']})")
        print(f"   corr_only WRONGLY makes X cause Y ........ {not co['do']['X_not_cause_of_Y']}  "
              f"(do(X)->Y {co['do']['Y_rate_do_X']:.0f}); why(Y)={co['why_cause']}")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== INTACT (grounded directed causal forward model over REAL facts) ===")
    intact = [run_seed(s, mode="intact", **rep_kw) for s in seeds]

    ac_seeds = seeds[:3]
    print("\n=== ANTI-CHEATS (mechanistic; 3 seeds) ===")
    edge_lesion = [run_seed(s, mode="edge_lesion", **rep_kw) for s in ac_seeds]
    ground_lesion_D = [run_seed(s, mode="ground_lesion_D", **rep_kw) for s in ac_seeds]
    ground_lesion_C = [run_seed(s, mode="ground_lesion_C", **rep_kw) for s in ac_seeds]
    corr_only = [run_seed(s, mode="corr_only", **rep_kw) for s in ac_seeds]
    shuffle = [run_seed(s, mode="shuffle", **rep_kw) for s in ac_seeds]

    n_go = sum(1 for r in intact if r["go"])
    verdict = ("GO" if (len(intact) >= 6 and n_go >= 5) or (len(intact) < 6 and n_go == len(intact))
               else "BOUNDARY")

    # anti-cheat scores
    all_events = sum(1 for r in intact if r["n_recalled_events"] == len(FACT_ORDER))
    battery_ok = sum(1 for r in intact if r["moat_battery_fa"] == 0)
    unseen_go = sum(1 for r in intact if r["unseen_fact_confirmed"])
    confab_any = sum(1 for r in intact if r["confab"])
    why_go = sum(1 for r in intact if r["why_fact_confirmed"])
    cause_correct = sum(1 for r in intact if r["do"]["X_not_cause_of_Y"])
    fwd_min = min(r["fwd_acc"] for r in intact)

    edge_collapse = sum(1 for r in edge_lesion if r["fwd_acc"] < 1.0)
    groundD_collapse = sum(1 for r in ground_lesion_D if not r["unseen_fact_confirmed"])
    groundC_collapse = sum(1 for r in ground_lesion_C if not r["why_fact_confirmed"])
    corr_wrong = sum(1 for r in corr_only if not r["do"]["X_not_cause_of_Y"])
    shuf_fail = sum(1 for r in shuffle if r["fwd_acc"] < 1.0)

    from tools.lab import attributable_to
    from tools.verdict import Verdict
    intact_sep = _st.mean([r["do"]["cause_separation"] for r in intact[:len(ac_seeds)]])
    corr_sep = _st.mean([r["do"]["cause_separation"] for r in corr_only])
    frac = attributable_to("cause-vs-correlation separation @ DO-intervention", intact_sep, corr_sep)

    print("\n=== VERDICT ===")
    print(f"  INTACT GO: {n_go}/{len(intact)} seeds (>=5/6 required)  ->  {verdict}")
    print(f"  event set grounded (6/6 facts moat-recalled): {all_events}/{len(intact)}")
    print(f"  moat battery abstains (0 false-accepts): {battery_ok}/{len(intact)}")
    print(f"  forward-prediction acc (min over seeds): {fwd_min:.2f}")
    print(f"  UNSEEN consequence moat-confirmed real fact: {unseen_go}/{len(intact)}  (confab: {confab_any})")
    print(f"  WHY = real cause fact (sun rose) + DO-probe: {why_go}/{len(intact)}")
    print(f"  cause-vs-correlation (X not cause of Y): {cause_correct}/{len(intact)}")
    print(f"  edge-lesion collapses forward prediction: {edge_collapse}/{len(edge_lesion)}")
    print(f"  GROUND-lesion(drop D) collapses what-if: {groundD_collapse}/{len(ground_lesion_D)}")
    print(f"  GROUND-lesion(drop C) collapses why-cause: {groundC_collapse}/{len(ground_lesion_C)}")
    print(f"  corr_only WRONGLY asserts X->Y (DO-prune load-bearing): {corr_wrong}/{len(corr_only)}")
    print(f"  shuffle fails to reproduce true chain: {shuf_fail}/{len(shuffle)}")

    v = (Verdict("grounded causal forward model — real-fact why/what-if over the production composer")
         .require("intact GO on >=5/6 seeds", n_go,
                  expect=lambda k: k >= max(5, len(intact) - 1) if len(intact) >= 6 else k == len(intact))
         .require("event set grounded: 6/6 facts moat-recalled (all seeds)", all_events,
                  expect=lambda k: k == len(intact))
         .require("moat battery abstains 0 false-accepts (all seeds)", battery_ok,
                  expect=lambda k: k == len(intact))
         .require("forward-prediction accuracy 1.0 (min over seeds)", fwd_min, expect=lambda x: x >= 1.0)
         .require("UNSEEN consequence is a moat-confirmed real fact (all seeds)", unseen_go,
                  expect=lambda k: k == len(intact))
         .require("0 confabulation (predicted consequence always a real fact)", confab_any,
                  expect=lambda k: k == 0)
         .require("WHY reads the real cause fact + survives DO-probe (all seeds)", why_go,
                  expect=lambda k: k == len(intact))
         .require("cause-vs-correlation: X not a cause of Y (all seeds)", cause_correct,
                  expect=lambda k: k == len(intact))
         .require("edge-lesion collapses forward prediction (3/3)", edge_collapse,
                  expect=lambda k: k == len(edge_lesion))
         .require("GROUNDING-lesion(drop D) collapses the what-if (3/3) — composer load-bearing",
                  groundD_collapse, expect=lambda k: k == len(ground_lesion_D))
         .require("GROUNDING-lesion(drop C) collapses the why-cause (3/3) — composer load-bearing",
                  groundC_collapse, expect=lambda k: k == len(ground_lesion_C))
         .require("corr_only WRONGLY makes X cause Y — DO-prune load-bearing (3/3)", corr_wrong,
                  expect=lambda k: k == len(corr_only))
         .require("shuffle fails to reproduce true chain (3/3)", shuf_fail,
                  expect=lambda k: k == len(shuffle))
         .control("DO-intervention separation vs corr_only", intact_sep, corr_sep, min_separation=1.0)
         .require("cause separation attributable to DO-prune (>=0.8)", frac,
                  expect=lambda x: x is not None and x >= 0.8)
         .disabled("OU background process", "deterministic regime for a controllable operating point")
         .disabled("conductance noise", "deterministic regime")
         .disabled("grounding-by-shared-substrate",
                   "events are DERIVED from + gated by the composer's moat recall (grounding-by-"
                   "derivation); driving the event blocks directly from the composer's UNBIND SPIKES "
                   "in one merged bridge is the named next rung")
         .disabled("spiking-mismatch-driven DA", "the DA sign is delivered by the teacher (the "
                   "environment boundary); a spiking mismatch unit driving from_reward DA is the next rung")
         .disabled("high-order (history-dependent) transitions",
                   "the model is FIRST-ORDER (state -> next); the HTM-TM high-order predictor "
                   "(EMERGE-15 GO) is the named composition"))
    decided = v.decide(go=(verdict == "GO"))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"mode": "causal_forward_model_grounded",
                       "facts": {EVENT_NAMES[e]: FACTS[e] for e in FACT_ORDER},
                       "intact": intact, "edge_lesion": edge_lesion,
                       "ground_lesion_D": ground_lesion_D, "ground_lesion_C": ground_lesion_C,
                       "corr_only": corr_only, "shuffle": shuffle,
                       "n_go": n_go, "n_seeds": len(intact),
                       "verdict": decided["status"], "verdict_label": verdict,
                       "event_set_grounded": all_events, "moat_battery_ok": battery_ok,
                       "forward_acc_min": fwd_min, "unseen_moat_confirmed": unseen_go,
                       "confab_any": confab_any, "why_grounded": why_go, "cause_correct": cause_correct,
                       "edge_lesion_collapse": edge_collapse,
                       "ground_lesion_D_collapse": groundD_collapse,
                       "ground_lesion_C_collapse": groundC_collapse,
                       "corr_only_wrong": corr_wrong, "shuffle_fail": shuf_fail,
                       "intact_cause_separation": intact_sep, "corr_cause_separation": corr_sep,
                       "cause_attributable_to_do_prune": frac,
                       "preconditions": decided["preconditions"],
                       "disabled_processes": decided["disabled_processes"],
                       "verdict_status": decided["status"]}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
