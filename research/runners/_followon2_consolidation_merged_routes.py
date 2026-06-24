"""FOLLOW-ON #2 — CONSOLIDATION: wire the two GO cross-region routes into MergedNavConvAgent, validate per-route.

The two interactions were GO standalone in SEPARATE behavioral-task runners; this CONSOLIDATES them onto the deployed
`MergedNavConvAgent` itself (the agent's OWN methods now carry the functional integration), per the scoping
`research/findings/2026-06-23-functional-one-brain-integration-scoping.md` §I-4-c / §I-5-b. Each route is validated
SEPARATELY (route ENGAGES + nav Δ~0 + moat 0-FA + LESION-collapses) — opt-in-then-default-if-clean; an HONEST verdict
(keep opt-in + characterize) if a default-on flip would regress nav or weaken the moat.

  Route A (language->action, COMMAND_GATE): `MergedNavConvAgent(co_resident_command_route=True).command_move(dir)` —
    a PARSED spoken command opens the command_route gate (parser action-ensemble firing) -> the learned word->cortex
    route steers the BG cascade. ENGAGES: COUPLED follows the command >> ISOLATED-NAV (gate closed) >> LESION (route cut).
  Route B (perception->memory/compose): `MergedNavConvAgent(co_resident_composer=True, co_resident_perception=True)
    .perceive_and_ground(obj)` — a perceived object's live cortex_it rate is grounded into the composer codebook ->
    composable on the `rf` slice. ENGAGES: held-out COMPOSE >> memorization floor; LESION (grounding->random) collapses.

TRACTABLE + FOREGROUND: each route builds ONE small merged agent (~1-2 min) and runs a SHORT decision/compose battery.
NO sim/ edit (the routes are reuse-by-import of the GO standalone primitives, wired onto the agent). GPU (SIM_BACKEND=cupy).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sim.backend import get_backend


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ── ROUTE A: language->action (COMMAND_GATE) on the merged agent ──────────────────────────────────────────────
def validate_route_a(seed=42, decisions=6):
    """Build MergedNavConvAgent(co_resident_command_route=True), couple done in __init__, and run a short command
    battery: COUPLED (parse_first), ISOLATED-NAV (gate closed), then a LESION agent. The route ENGAGES iff COUPLED
    follows the commanded direction >> ISOLATED-NAV and >> LESION. nav Δ~0 + moat 0-FA are checked separately
    (the nav cascade is the SAME default cascade; the moat is the conversational composer, array-disjoint)."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    from research.runners.spoken_instruction_nav import default_schedule, DIRECTION_WORDS
    from research.runners.g11_bg_runner import ACTION_NAMES, N_ACTIONS
    xp, backend = get_backend()
    chance = 1.0 / N_ACTIONS
    print(f"\n[routeA] ===== seed {seed} (backend={backend}) — language->action COMMAND_GATE on the merged agent =====",
          flush=True)
    t0 = time.time()
    # the command-route route needs the nav cascade cortex_X pools + the spiking-WTA readout (forced on by the flag).
    # vocab: include the direction words so the parser + the route's word codes are coherent (the default 16-word probe
    # vocab already covers north/east/south/west).
    agent = MergedNavConvAgent(seed=seed, co_resident_command_route=True)
    print(f"[routeA] built agent in {time.time() - t0:.1f}s "
          f"({agent._merged_bridge.core_config.num_neurons} neurons, readout={agent._handles['cmd_readout_region']}_X)",
          flush=True)

    # a short schedule of commanded directions (changes across the battery).
    schedule = default_schedule(n_phases=N_ACTIONS, seed=seed)
    per_dir = max(1, decisions // len(schedule))
    cmds = [d for d in schedule for _ in range(per_dir)]
    print(f"[routeA] command battery ({len(cmds)} decisions): {[DIRECTION_WORDS[d] for d in cmds]}", flush=True)

    def _run(parse_first):
        ok = 0
        moves = {a: 0 for a in ACTION_NAMES}
        none = 0
        for d in cmds:
            chosen, _ = agent.command_move(d, parse_first=parse_first)
            if chosen is None:
                none += 1
            else:
                moves[chosen] += 1
                if chosen == d:
                    ok += 1
        return ok / max(1, len(cmds)), moves, none

    coupled_acc, coupled_moves, coupled_none = _run(parse_first=True)
    print(f"[routeA]  COUPLED      acc-vs-commanded = {coupled_acc:.3f}  (moves={coupled_moves}, none={coupled_none})",
          flush=True)
    isonav_acc, _, _ = _run(parse_first=False)
    print(f"[routeA]  ISOLATED-NAV acc-vs-commanded = {isonav_acc:.3f}  (gate held CLOSED)", flush=True)

    # LESION: a FRESH agent (clean), command_route cut, parser fires -> the route is severed -> chance.
    t1 = time.time()
    agent_les = MergedNavConvAgent(seed=seed, co_resident_command_route=True)
    n_lesioned = agent_les.lesion_command_route()
    les_ok = 0
    for d in cmds:
        chosen, _ = agent_les.command_move(d, parse_first=True)
        if chosen is not None and chosen == d:
            les_ok += 1
    lesion_acc = les_ok / max(1, len(cmds))
    print(f"[routeA]  LESION       acc-vs-commanded = {lesion_acc:.3f}  ({n_lesioned} command_route synapses zeroed, "
          f"parser firing) [built in {time.time() - t1:.1f}s]", flush=True)

    # moat 0-FA on the lesion-free coupled agent (the conversational composer is array-disjoint from the nav route).
    agent.composer.kb = []
    agent.hear("dog go north")
    moat_abstain = (agent.what_does("river", "look") is None)
    pos_recall = (agent.what_does("dog", "go") == "north")
    print(f"[routeA]  MOAT         abstain-on-unstored={moat_abstain}  pos-recall(dog go->north)={pos_recall}",
          flush=True)

    engages = (coupled_acc >= 0.50 and coupled_acc >= isonav_acc + 0.20 and coupled_acc >= lesion_acc + 0.20)
    lesion_collapses = (lesion_acc <= 0.40)
    moat_ok = bool(moat_abstain and pos_recall)
    return {
        "route": "A_language_to_action_command_gate", "seed": int(seed), "backend": backend, "chance": chance,
        "wired": True, "method": "MergedNavConvAgent.command_move",
        "coupled_acc": coupled_acc, "isolated_nav_acc": isonav_acc, "lesion_acc": lesion_acc,
        "n_lesioned_synapses": int(n_lesioned), "n_decisions": len(cmds),
        "engages": bool(engages), "lesion_collapses": bool(lesion_collapses),
        "moat_abstain": bool(moat_abstain), "pos_recall": bool(pos_recall), "moat_0fa": moat_ok,
    }


# ── ROUTE B: perception->memory/compose on the merged agent ───────────────────────────────────────────────────
def validate_route_b(seed=42):
    """Build MergedNavConvAgent(co_resident_composer=True, co_resident_perception=True), ground perceived objects via
    perceive_and_ground (the agent's OWN method), COMPOSE held-out facts, and check held-out compose >> floor, moat
    0-FA, and LESION (grounding->random) collapse. Reuses navigate_to_compose_then_answer's held-out/lesion scorers."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS, N_OBJECTS
    from research.runners.navigate_to_compose_then_answer import (
        ACTIONS, D, _held_out_compose_score, _lesion_recompose_score, _moat_check,
    )
    xp, backend = get_backend()
    chance = 1.0 / N_OBJECTS
    print(f"\n[routeB] ===== seed {seed} (backend={backend}) — perception->compose on the merged agent =====",
          flush=True)
    t0 = time.time()
    # vocab = the perceived objects + the compose verbs (so the composer codebook + the moat actions are coherent).
    vocab = list(OBJECT_WORDS) + ACTIONS
    agent = MergedNavConvAgent(seed=seed, vocab=vocab, co_resident_composer=True, co_resident_perception=True)
    print(f"[routeB] built agent in {time.time() - t0:.1f}s ({agent._merged_bridge.core_config.num_neurons} neurons, "
          f"cortex_it base={int(agent._handles['cortex_it_indices'][0])})", flush=True)

    # GROUND every object via the agent's perceive_and_ground (the in-episode perception->codebook grounding). A real
    # episode traverses to a SUBSET; here (the consolidation unit-test) we ground all objects to exercise the wiring.
    grounded = []
    first_sample = None
    for obj in OBJECT_WORDS:
        rate, phases = agent.perceive_and_ground(obj)
        if float(np.asarray(rate).sum()) > 0.0:
            grounded.append(obj)
        if first_sample is None:
            first_sample = (obj, np.asarray(rate).copy(), phases.copy())
    print(f"[routeB]  grounded {len(grounded)}/{N_OBJECTS} objects via perceive_and_ground: {grounded}", flush=True)

    # PROVENANCE: the grounded code in the codebook == the live-rate projection (not a host-set phasor), and the
    # composer is the co-resident MergedRFComposer bound to the merged bridge.
    prov_ok = False
    if first_sample is not None:
        obj0, _r0, ph0 = first_sample
        prov_ok = bool(np.allclose(agent.composer.concepts[obj0], ph0)
                       and agent.composer._merged is agent._merged_bridge)
    print(f"[routeB]  PROVENANCE   grounded-code==live-rate-projection & composer bound to merged bridge: {prov_ok}",
          flush=True)

    # held-out compose vs memorization floor (compose != recall): COMPOSE generalizes to never-composed pairings.
    clean, floor, n_held, held_composites = _held_out_compose_score(agent.composer, grounded, seed)
    print(f"[routeB]  COMPOSE      held-out clean {clean:.3f} | mem-floor {floor:.3f} (chance {chance:.3f}) | "
          f"n_held_out={n_held}", flush=True)

    # the no-confab moat + a positive recall control.
    moat_ok, moat_tot, pos, stored, absent = _moat_check(agent.composer, grounded)
    print(f"[routeB]  MOAT         abstain {moat_ok}/{moat_tot} on {absent}  |  pos-recall {pos}/1 (stored {stored[:1]})",
          flush=True)

    # LESION the grounded-code map (grounded objects -> random codes), re-cleanup the SAME composites -> collapse.
    rng = np.random.default_rng(seed * 7919 + 3)
    for o in grounded:
        agent.composer.concepts[o] = rng.uniform(0.0, 1.0, D)
    lesion_clean, _ = _lesion_recompose_score(agent.composer, grounded, held_composites)
    print(f"[routeB]  LESION       grounded->random, re-cleanup SAME composites: {lesion_clean:.3f} "
          f"(was {clean:.3f}; should collapse)", flush=True)

    engages = (len(grounded) >= 2 and clean >= 0.90 and clean >= floor + 0.30)
    lesion_collapses = (lesion_clean <= floor + 1e-9 or lesion_clean <= chance + 0.10)
    moat_0fa = bool(moat_ok == moat_tot and moat_tot >= 1 and pos == 1)
    moat_breach = bool(moat_tot >= 1 and moat_ok < moat_tot)
    return {
        "route": "B_perception_to_compose", "seed": int(seed), "backend": backend, "chance": chance,
        "wired": True, "method": "MergedNavConvAgent.perceive_and_ground",
        "n_grounded": len(grounded), "grounded": grounded,
        "compose_clean": clean, "compose_floor": floor, "n_held_out": n_held,
        "moat_ok": moat_ok, "moat_tot": moat_tot, "pos_recall": pos, "moat_absent": absent,
        "lesion_compose": lesion_clean, "provenance_ok": prov_ok,
        "engages": bool(engages), "lesion_collapses": bool(lesion_collapses),
        "moat_0fa": moat_0fa, "moat_breach": moat_breach,
    }


def _route_verdict(r):
    if r.get("moat_breach"):
        return "MOAT_BREACH"
    if r["engages"] and r["lesion_collapses"] and r["moat_0fa"]:
        return "GO"
    if r["engages"] and (r["lesion_collapses"] or r["moat_0fa"]):
        return "PARTIAL"
    return "HONEST_NEGATIVE"


def main():
    ap = argparse.ArgumentParser(description="FOLLOW-ON #2 consolidation: wire+validate the 2 GO cross-region routes "
                                             "onto MergedNavConvAgent (per-route engage/lesion/nav/moat).")
    ap.add_argument("--route", choices=["a", "b", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decisions", type=int, default=6, help="route A command decisions")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_followon2_consolidation_merged_routes.json")
    args = ap.parse_args()

    _, backend = get_backend()
    results = {}
    if args.route in ("a", "both"):
        ra = validate_route_a(seed=args.seed, decisions=args.decisions)
        ra["verdict"] = _route_verdict(ra)
        results["route_a"] = ra
        print(f"[routeA] verdict = {ra['verdict']}", flush=True)
    if args.route in ("b", "both"):
        rb = validate_route_b(seed=args.seed)
        rb["verdict"] = _route_verdict(rb)
        results["route_b"] = rb
        print(f"[routeB] verdict = {rb['verdict']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # merge into any existing file (so route a and route b can be run separately and accumulate).
    existing = {}
    if os.path.exists(args.out):
        try:
            existing = json.load(open(args.out))
        except Exception:
            existing = {}
    existing.setdefault("task", "FOLLOW-ON #2 consolidation: wire the 2 GO cross-region routes onto MergedNavConvAgent")
    existing["date"] = "2026-06-24"
    existing["backend"] = backend
    existing["scope"] = ("research/runners/nav_conv_merged_bridge.py — MergedNavConvAgent + build_merged_nav_conv_bridge "
                         "(opt-in co_resident_command_route / co_resident_perception + command_move/perceive_and_ground "
                         "methods); reuse-by-import of the GO standalone primitives; NO sim/ edit")
    existing.update(results)
    with open(args.out, "w") as f:
        json.dump(_ser(existing), f, indent=2)
    print(f"\n[followon2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
