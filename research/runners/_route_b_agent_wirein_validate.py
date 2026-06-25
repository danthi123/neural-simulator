"""Route-B Option-1 AGENT wire-in re-validation (purity #1, 2026-06-25).

Validates the SAME Option-3 bars (held-out compose >> floor, LESION collapses, MOAT 0-FA, provenance grounding-is-
SPIKES) but THROUGH the AGENT surface `MergedNavConvAgent(co_resident_composer=True, co_resident_composer_kind=
'onebrain', co_resident_perception=True)` with the (now-default) gen_spikes grounding — i.e. the agent's OWN
`perceive_and_ground` grounds the percept via the LEARNED gen_perception->gen_concept convergence (spikes-only, no
host-`M`), writing to `composer.comp.concepts` (the codebook the onebrain binds/cleanups read), and the held-out
compose / lesion / moat anti-cheats run on the agent's co-resident composer. Plus an rf-path regression check (the rf
composer still grounds + composes via the agent, host_m, no gen stack forced).

Reuses the GO standalone anti-cheat scorers from `navigate_to_compose_then_answer` (reuse-by-import) so the bars are
byte-for-byte the validated ones; this just feeds them the AGENT's (bridge, composer, gen handles) instead of the
standalone build_compose_bridge's. NO `sim/` edit; runner-layer only.

The MOAT is NEVER weakened: a breach is a HARD STOP (the per-seed verdict + the aggregate go to MOAT_BREACH).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend

from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS, N_OBJECTS
from research.runners.navigate_to_see_then_answer import default_object_layout
from research.runners.navigate_to_compose_then_answer import (
    D, _held_out_compose_score, _lesion_recompose_score, _moat_check, _provenance_check,
    lesion_gen_convergence, _codebook, _algebra,
)


def _ground_objects_via_agent(agent, objects):
    """GROUND each object through the AGENT's OWN `perceive_and_ground` (the in-episode perception->codebook grounding).
    Captures the FIRST grounded object's (source_vec, phases, source_kind) for the provenance assert (mirroring the
    standalone's provenance_sample). Returns (grounded, provenance_sample)."""
    grounded = []
    prov = None
    for obj in objects:
        source_vec, phases = agent.perceive_and_ground(obj)
        if obj not in grounded:
            grounded.append(obj)
        if prov is None:
            # the source_kind the standalone _perceive_and_ground would set (gen_spikes -> gen_concept_spikes).
            sk = "gen_concept_spikes" if agent.perception_grounding == "gen_spikes" else "cortex_it_rate_host_M"
            prov = {"obj": obj, "source": np.asarray(source_vec).copy(), "phases": np.asarray(phases).copy(),
                    "source_kind": sk}
    return grounded, prov


def _agent_handles_for_provenance(agent):
    """Build the minimal `handles` dict the standalone `_provenance_check` consults against the agent (grounding mode,
    composer_kind, gen_proj for gen_spikes)."""
    h = {"grounding": agent.perception_grounding, "composer_kind": agent.co_resident_composer_kind}
    if agent.perception_grounding == "gen_spikes":
        h["gen_proj"] = agent._handles["gen_proj"]
    return h


def run_seed(seed, composer_kind="onebrain", grounding="gen_spikes"):
    xp, backend = get_backend()
    chance = 1.0 / N_OBJECTS
    print(f"\n[routeB-agent] ===== seed {seed} (backend={backend}, composer={composer_kind}, grounding={grounding}) =====",
          flush=True)

    layout = default_object_layout(seed)
    # ground the first 3 layout objects (the same subset the standalone episode reaches) — through the AGENT.
    sorted_cells = sorted(layout.keys(), key=lambda c: c[0])
    objects = [layout[c] for c in sorted_cells[:3]]
    print(f"[routeB-agent] objects to ground via agent.perceive_and_ground: {objects}", flush=True)

    vocab = list(OBJECT_WORDS) + ["chase", "near"]
    agent = MergedNavConvAgent(seed=seed, vocab=vocab, co_resident_composer=True,
                               co_resident_composer_kind=composer_kind, co_resident_perception=True,
                               perception_grounding=grounding)
    print(f"[routeB-agent] agent built: composer={type(agent.composer).__name__} "
          f"grounding={agent.perception_grounding} gen_resident={agent.co_resident_generalization} "
          f"codebook={'composer.comp.concepts' if _codebook(agent.composer) is not agent.composer else 'composer.concepts'}",
          flush=True)

    composer = agent.composer

    # --- GROUND via the agent, then COMPOSE held-out facts on the co-resident composer (anti-cheat 2). ---
    grounded, prov_sample = _ground_objects_via_agent(agent, objects)
    print(f"[routeB-agent]  GROUND   {len(grounded)} objects via agent.perceive_and_ground: {grounded}", flush=True)

    clean, floor, n_held, held_composites = _held_out_compose_score(composer, grounded, seed)
    print(f"[routeB-agent]  COMPOSE  held-out clean {clean:.3f} | mem-floor {floor:.3f} (chance {chance:.3f}) | "
          f"n_held_out={n_held}", flush=True)

    # --- MOAT (anti-cheat 4) + positive recall, through the agent's composer. ---
    moat_ok, moat_tot, pos, stored, absent = _moat_check(composer, grounded)
    print(f"[routeB-agent]  MOAT     abstain {moat_ok}/{moat_tot} on {absent}  |  pos-recall {pos}/1 "
          f"(stored {stored[:1]})", flush=True)

    # --- PROVENANCE (anti-cheat 3): the grounded code is spikes-only (gen_concept SPIKES via the learned convergence),
    #     the composer is the agent's merged composer, the bind ran on the bridge's RF complex synapses. ---
    prov = {}
    if prov_sample is not None:
        h_prov = _agent_handles_for_provenance(agent)
        prov = _provenance_check(agent._merged_bridge, composer, h_prov, prov_sample["source"],
                                 prov_sample["phases"], prov_sample["obj"], prov_sample["source_kind"])

    # --- LESION (anti-cheat 1): gen_spikes -> SEVER the learned convergence + RE-GROUND (degenerate codes); host_m ->
    #     restore grounded objects to random codes. Then re-cleanup the SAME held-out composites -> compose collapses. ---
    if grounding == "gen_spikes":
        n_cut = lesion_gen_convergence(agent._merged_bridge, agent._handles["gen"])
        for o in grounded:
            agent.perceive_and_ground(o)        # RE-GROUND off the now-severed convergence (via the agent)
        lesion_clean, _ = _lesion_recompose_score(composer, grounded, held_composites)
        print(f"[routeB-agent]  LESION   sever gen_perception->gen_concept ({n_cut} syn) + re-ground, re-cleanup: "
              f"held-out compose {lesion_clean:.3f} (was {clean:.3f}; should collapse toward chance)", flush=True)
    else:
        rng = np.random.default_rng(seed * 7919 + 3)
        cb = _codebook(composer)
        for o in grounded:
            cb.concepts[o] = rng.uniform(0.0, 1.0, D)
        if hasattr(composer, "_store_dirty"):
            composer._store_dirty = True
        if hasattr(composer, "_csr_cache"):
            composer._csr_cache = {}
        lesion_clean, _ = _lesion_recompose_score(composer, grounded, held_composites)
        print(f"[routeB-agent]  LESION   grounded->random codes, re-cleanup: held-out compose {lesion_clean:.3f} "
              f"(was {clean:.3f}; should collapse toward chance)", flush=True)

    # --- verdict (the Option-3 bars, through the agent). ---
    n_grounded = len(grounded)
    compose_go = (n_grounded >= 2 and clean >= 0.90 and clean >= floor + 0.30)
    moat_ok_all = (moat_ok == moat_tot and moat_tot >= 1 and pos == 1)
    moat_breach = (moat_tot >= 1 and moat_ok < moat_tot)
    lesion_ok = (lesion_clean <= floor + 1e-9 or lesion_clean <= chance + 0.10)
    grounding_is_spikes = bool(prov.get("grounded_code_is_spikes_only", False)) if grounding == "gen_spikes" else None
    go = bool(compose_go and moat_ok_all and lesion_ok and (grounding != "gen_spikes" or grounding_is_spikes))

    return {
        "seed": int(seed), "backend": backend, "chance": chance, "grounding": grounding,
        "composer_kind": composer_kind, "via": "MergedNavConvAgent.perceive_and_ground",
        "n_grounded": n_grounded, "grounded": grounded,
        "compose_clean": clean, "compose_floor": floor, "n_held_out": n_held,
        "moat_ok": moat_ok, "moat_tot": moat_tot, "pos_recall": pos, "moat_absent": absent, "stored": stored,
        "lesion_compose": lesion_clean, "provenance": prov,
        "compose_go": compose_go, "moat_ok_all": moat_ok_all, "moat_breach": moat_breach,
        "lesion_ok": lesion_ok, "grounding_is_spikes": grounding_is_spikes, "go": go,
    }


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


def main():
    ap = argparse.ArgumentParser(description="Route-B Option-1 AGENT wire-in re-validation (the Option-3 bars THROUGH "
                                             "MergedNavConvAgent.perceive_and_ground).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--composer", type=str, default="onebrain", choices=["rf", "onebrain"])
    ap.add_argument("--grounding", type=str, default="gen_spikes", choices=["gen_spikes", "host_m"])
    ap.add_argument("--out", type=str, default="research/findings/raw/_route_b_agent_wirein_validate.json")
    args = ap.parse_args()

    _, backend = get_backend()
    results = [run_seed(s, composer_kind=args.composer, grounding=args.grounding) for s in args.seeds]
    any_breach = any(r["moat_breach"] for r in results)
    all_go = all(r["go"] for r in results)
    verdict = "MOAT_BREACH" if any_breach else ("GO" if all_go else "NO-GO")
    mean_clean = float(np.mean([r["compose_clean"] for r in results]))
    mean_floor = float(np.mean([r["compose_floor"] for r in results]))

    print(f"\n{'=' * 100}", flush=True)
    print(f"[routeB-agent] {len(results)} seed(s) via AGENT ({args.composer}, {args.grounding}): held-out compose "
          f"{mean_clean:.3f} | mem-floor {mean_floor:.3f}  ==>  [{verdict}]", flush=True)
    for r in results:
        print(f"[routeB-agent]   seed {r['seed']}: grounded={r['n_grounded']} compose={r['compose_clean']:.3f} "
              f"floor={r['compose_floor']:.3f} lesion={r['lesion_compose']:.3f} moat={r['moat_ok']}/{r['moat_tot']} "
              f"pos={r['pos_recall']} spikes={r['grounding_is_spikes']} -> "
              f"{'GO' if r['go'] else ('MOAT_BREACH' if r['moat_breach'] else 'NO-GO')}", flush=True)
    print(f"{'=' * 100}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_ser({"verdict": verdict, "backend": backend, "composer": args.composer, "grounding": args.grounding,
                        "mean_clean": mean_clean, "mean_floor": mean_floor, "results": results}), f, indent=2,
                  default=str)
    print(f"[routeB-agent] wrote {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else (3 if verdict == "MOAT_BREACH" else 1))


if __name__ == "__main__":
    main()
