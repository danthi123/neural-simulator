"""Close-out validation for the 3G neural-discourse-planner DEFAULT-ON flip on the --rich runtime path.

This is the TRACTABLE foreground validator for the close-out flip (brain_chat_tui --rich now defaults the
spiking dlPFC discourse-planner ON, with --no-neural-planner the host escape). It re-confirms the GO 3G result
on a SINGLE tiny brain (far fewer bridge builds than the full --neural-derisk battery, which exceeds the 5-min
foreground cap under GPU contention), checking exactly the flip's claims:

  (1) --rich NEURAL path is reachable + substantive: the neural-planner RichAnswerComposer (== the runtime
      `--rich` default wiring) gives >=2 brain-sourced multi-sentence answers, neural-ordered, on the rich turns;
  (2) MOAT 0-FA under the neural planner: the untaught/general cues still ABSTAIN;
  (3) the HOST escape (`--no-neural-planner` => neural_planner=False) restores the host path (also substantive);
  (4) the neural selection is LOAD-BEARING (3G core): lesion the dlPFC ordering -> the elaboration COLLAPSES
      (intact >=1 fact -> lesioned 0), AND it is ON-TOPIC (every neural-selected fact within the topic's 2-hop
      graph neighbourhood, foregrounding a direct associate);
  (5) QUALITY PARITY: the neural per-turn sentence counts are >= the host's (no quality regression).

Reuse-by-import, NO `sim/` edit. Reuses rich_answer_composer's own tiny-brain builder + the smoke script.

  SIM_BACKEND=cupy python -m research.runners._closeout_3G_planner_default_validate --seed 42 \
      --out research/findings/raw/_closeout_3G_neural_planner_default.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_closeout_3G_neural_planner_default.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    from research.runners.rich_answer_composer import (
        RichAnswerComposer, _build_smoke_chat, _SMOKE_SCRIPT, _topic_neighborhood,
    )

    t0 = time.time()
    rich_kinds = ("rich", "followup")

    # Build the (expensive) tiny-brain ChatBrain ONCE and SHARE it across all conditions -- the MultiTurnAgent
    # bridge is the dominant build cost; the RichAnswerComposer wrappers + the planner's spreading-controller are
    # cheap by comparison. Each condition gets a FRESH composer wrapper (so discourse thread state never carries),
    # but they all wrap the SAME underlying brain/agent. (This is what makes the validation tractable in one
    # foreground slice; the full --neural-derisk builds a fresh brain per condition and exceeds the 5-min cap.)
    chat = _build_smoke_chat(a.seed, use_multiturn=True)
    stored = set(tuple(f) for f in RichAnswerComposer(chat, neural_planner=False)._stored_facts())

    def _run_script(neural_planner):
        rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                                  neural_planner=neural_planner, planner_seed=a.seed)
        rows = []
        for utt, kind in _SMOKE_SCRIPT:
            r = rich.answer(utt)
            rows.append({"you": utt, "kind": kind, "answer": r["answer"], "abstained": r["abstained"],
                         "facts": r["facts"], "n_sentences": r["n_sentences"], "topic": r["topic"],
                         "all_brain_sourced": all(tuple(f) in stored for f in r["facts"])})
        return rows

    # --- run the smoke script under the NEURAL planner (== the --rich runtime default) ---
    neural_rows = _run_script(neural_planner=True)
    # --- run the SAME script under the HOST planner (the --no-neural-planner escape) ---
    host_rows = _run_script(neural_planner=False)

    neural_rich = [r for r in neural_rows if r["kind"] in rich_kinds]
    host_rich = [r for r in host_rows if r["kind"] in rich_kinds]
    abstain_neural = [r for r in neural_rows if r["kind"] == "abstain"]

    # (1) neural path reachable + substantive on every rich turn
    neural_substantive = all((not r["abstained"]) and r["n_sentences"] >= 2 and r["all_brain_sourced"]
                             for r in neural_rich)
    neural_min_sentences = min((r["n_sentences"] for r in neural_rich if not r["abstained"]), default=0)
    # (2) moat 0-FA under neural
    moat_held_neural = all(r["abstained"] for r in abstain_neural)
    moat_breaches = [(r["you"], r["answer"]) for r in abstain_neural if not r["abstained"]]
    # (3) host escape: substantive too + abstains
    host_substantive = all((not r["abstained"]) and r["n_sentences"] >= 2 for r in host_rich)
    abstain_host = [r for r in host_rows if r["kind"] == "abstain"]
    moat_held_host = all(r["abstained"] for r in abstain_host)
    host_escape_ok = host_substantive and moat_held_host
    # (5) quality parity: neural >= host per turn
    counts = [(h["you"], h["n_sentences"], n["n_sentences"]) for h, n in zip(host_rich, neural_rich)]
    parity = all(n >= 2 and n >= h for _u, h, n in counts)

    # (4) LESION + ON-TOPIC (the 3G core), on the SAME shared brain (fresh composer wrappers)
    probe = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                               neural_planner=True, planner_seed=a.seed)
    g_intact = probe._elaboration_facts("memory", exclude=[])
    # on-topic: every neural-selected elaboration fact within the topic's 2-hop graph neighbourhood
    graph = probe.composer._assoc_graph()
    nbhd = _topic_neighborhood(graph, "memory", hops=2)
    on_topic = all((f[0] in nbhd or f[2] in nbhd) for f in g_intact)
    direct_nbrs = set(graph.get("memory", {}).keys()) | {"memory"}
    foregrounds_direct = any((f[0] in direct_nbrs or f[2] in direct_nbrs) for f in g_intact)
    # lesion the dlPFC ordering -> the elaboration component must collapse to nothing
    les = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                             neural_planner=True, planner_seed=a.seed)
    les._planner.ordered_associates = lambda topic, avoid=(): []
    g_les = les._elaboration_facts("memory", exclude=[])
    lesion_collapses = len(g_intact) >= 1 and len(g_les) == 0

    go = bool(neural_substantive and moat_held_neural and host_escape_ok and parity
              and on_topic and foregrounds_direct and lesion_collapses)

    if go:
        verdict = (
            f"GO -- the --rich runtime NEURAL discourse-planner is REACHABLE + DEFAULT-correct: every rich turn is "
            f"substantive (>= {neural_min_sentences} brain-sourced, neural-ordered sentences, >= the host count), "
            f"the no-confab moat STILL ABSTAINS on all {len(abstain_neural)} untaught/general cues, the "
            f"--no-neural-planner host escape restores the host path (substantive + moat-held), the neural "
            f"selection is LOAD-BEARING (elaboration {len(g_intact)} -> {len(g_les)} facts under dlPFC lesion), "
            f"and every neural-selected fact is ON-TOPIC (within memory's 2-hop neighbourhood, foregrounding a "
            f"direct associate). Quality-parity, NOT output-identity -- the neural latency rank surfaces a "
            f"different-but-equally-valid order than the host argmax. 3G re-confirmed on the runtime wiring."
        )
    else:
        bits = []
        if not neural_substantive:
            bits.append(f"neural NOT substantive on rich turns: "
                        f"{[(r['you'], r['n_sentences'], r['abstained']) for r in neural_rich]}")
        if not moat_held_neural:
            bits.append(f"MOAT LEAK under neural planner: {moat_breaches}")
        if not host_escape_ok:
            bits.append(f"HOST escape broken (substantive={host_substantive}, moat={moat_held_host})")
        if not parity:
            bits.append(f"quality regression (host vs neural counts {counts})")
        if not (on_topic and foregrounds_direct):
            bits.append(f"NOT on-topic (on_topic={on_topic}, foregrounds_direct={foregrounds_direct}, "
                        f"g_intact={g_intact})")
        if not lesion_collapses:
            bits.append(f"LESION did NOT collapse (intact={len(g_intact)}, lesioned={len(g_les)})")
        verdict = "HONEST-NEGATIVE -- " + " || ".join(bits)

    res = {
        "probe": "closeout_3G_neural_planner_default_on_rich_runtime",
        "flip": "brain_chat_tui --rich now DEFAULTS the spiking dlPFC discourse-planner ON "
                "(neural_planner=True); --no-neural-planner is the host escape. numpy-CPU nuance: HOST default on "
                "the numpy backend (the planner builds a per-topic SimulationBridge), neural-ON the GPU default.",
        "backend": os.environ.get("SIM_BACKEND"),
        "seed": a.seed,
        "GO": go,
        "verdict": verdict,
        "neural_min_sentences": neural_min_sentences,
        "neural_substantive": neural_substantive,
        "moat_held_neural": moat_held_neural,
        "moat_breaches": moat_breaches,
        "host_escape_ok": host_escape_ok,
        "host_substantive": host_substantive,
        "moat_held_host": moat_held_host,
        "quality_parity": parity,
        "sentence_counts_host_vs_neural": [{"q": u, "host": h, "neural": n} for u, h, n in counts],
        "on_topic": on_topic,
        "foregrounds_direct_associates": foregrounds_direct,
        "lesion_collapses": lesion_collapses,
        "elaboration_intact_facts": g_intact,
        "elaboration_lesioned_facts": g_les,
        "topic_neighborhood": sorted(nbhd),
        "neural_transcript": neural_rows,
        "host_transcript": host_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100, flush=True)
    print("[3G default flip] NEURAL (--rich default) vs HOST (--no-neural-planner) -- same questions:", flush=True)
    print("=" * 100, flush=True)
    for h, n in zip(host_rows, neural_rows):
        print(f"  you>          {h['you']}", flush=True)
        ht = "[ABSTAIN]" if h["abstained"] else f"[{h['n_sentences']}s]"
        nt = "[ABSTAIN]" if n["abstained"] else f"[{n['n_sentences']}s]"
        print(f"  HOST>         {h['answer']}  {ht}", flush=True)
        print(f"  NEURAL>       {n['answer']}  {nt}", flush=True)
        print("", flush=True)
    print("=" * 100, flush=True)
    print(f"[3G default flip] lesion: elaboration {len(g_intact)} -> {len(g_les)} (collapses={lesion_collapses}); "
          f"on-topic={on_topic} foregrounds-direct={foregrounds_direct}; moat-neural={moat_held_neural} "
          f"host-escape={host_escape_ok}; elapsed {res['elapsed_s']}s", flush=True)
    print(f"[3G default flip] VERDICT: {verdict}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
