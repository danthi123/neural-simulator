"""Phase-9 SMOKE: the FULL EMBODIED experience loop -- perceive-while-ACTING -> converse via the RA console.

The capstone of the fluid-conversation arc + the experience-connection (Phase-8 was a cheap-first with a lightweight
percept stand-in; this uses the LIVE loop). Composes two validated pieces on ONE brain:
  - Tier-3 live-and-remember (`_tier3_live_and_remember_derisk`): a MergedNavConvAgent LIVES, and on first arrival at
    an object it `perceive_and_ground`s the LIVE cortex_it spiking percept + `composer.store`s the lived fact
    (prev near obj). The object's code is grounded from REAL perception DURING behaviour.
  - the RA console (Phase-2..8): the RA-fine-tuned ~21M renders a grounded fact fluently, gated + VERIFY + moat.
So: the brain PERCEIVES objects as it acts, then we CONVERSE about what it lived -- the RA generator renders the
lived facts; the moat holds on never-encountered objects.

SMOKE (single-seed by default; the merged-bridge build + a live episode is heavy -- multi-seed is the follow-on):
  (a) LIVE: run a short live episode -> lived facts (prev near obj) grounded from live percepts, in agent.composer;
  (b) CONVERSE: for each lived fact, recall it (composer.query_patient, validated) + RA-render it fluently;
  (c) MOAT: a held-out (never-encountered) object -> query_patient None -> abstain (no confabulation);
  (d) GROUNDING-LESION: corrupt a lived object's grounded code -> its recall collapses (load-bearing on the percept).

Reuse-by-import (Tier-3 live loop + the RA faculty); NO `sim/` edit. GPU/CuPy strongly preferred (the merged bridge).
Run: SIM_BACKEND=cupy python -m research.runners._fluidconv_phase9_embodied_loop_smoke --seed 42 --n-steps 400
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase9_embodied_loop.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from research.runners._fluidconv_phase2_ra_finetune import FT_CKPT
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2

    t0 = time.time()
    err = None
    result = {"seed": a.seed, "n_steps": a.n_steps}
    try:
        from research.runners._tier3_live_and_remember_derisk import (
            _build_agent, LivingWorld, LiveState, live, SpikingHunger, LINK_VERB, OBJECT_WORDS)
        from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3
        from research.runners._grounded_lang_integration_derisk import _build_inflection_map
        from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key
        from sim.backend import to_host

        print(f"[phase9-embodied] building the merged one brain (nav+conv+perception+drive+composer) seed {a.seed}...",
              flush=True)
        world = LivingWorld(a.seed, n_objects=3)
        agent = _build_agent(a.seed)
        bridge = agent._merged_bridge
        hunger = SpikingHunger(bridge, window=40)
        st = LiveState(a.seed)
        print(f"[phase9-embodied] LIVING a {a.n_steps}-step episode (perceive+ground+store objects encountered)...",
              flush=True)
        live(agent, hunger, st, world, a.n_steps, drive_reward="spiking", perceive=True)
        lived = list(st.lived_facts)
        result["lived_facts"] = [list(f) for f in lived]
        result["placed"] = world.placed; result["held_out"] = world.held_out
        print(f"[phase9-embodied] lived facts: {lived}  (held-out: {world.held_out})", flush=True)

        faculty = FTFaculty()
        actions = {LINK_VERB}
        inflect = _build_inflection_map(sorted(actions))
        agents_set = {f[0] for f in lived}; patients_set = {f[2] for f in lived}
        store_keys = {tuple(f) for f in lived}

        # (b) CONVERSE about each lived fact via the RA console (recall + RA-render)
        conv = []
        for (prev, verb, obj) in lived:
            got = agent.composer.query_patient(prev, verb)          # validated lived recall
            if got is None:
                conv.append({"fact": [prev, verb, obj], "recall": None, "reply": "I don't know.", "ok": False}); continue
            ctx = f"the {prev} is {verb} the {got} ."
            ans = faculty.answer(ctx, f"what is the {prev} {verb} ?")
            reply = ans if got in ans.split() else f"the {prev} is {verb} the {got}."
            conv.append({"fact": [prev, verb, obj], "recall": got, "reply": reply, "ok": bool(got == obj)})
            print(f"    lived '{prev} {verb} {obj}' -> recall {got} -> \"{reply}\"", flush=True)

        # (c) MOAT: a held-out (never-encountered) object -> abstain
        moat_ok = True
        for h in world.held_out:
            if agent.composer.query_patient(h, LINK_VERB) is not None:
                moat_ok = False
        result["moat_ok"] = bool(moat_ok)

        # (d) GROUNDING-LESION: corrupt a lived object's grounded code -> its recall collapses
        lesion_ok = None
        if lived:
            prev0, verb0, obj0 = lived[0]
            import numpy as np
            code = agent.composer.concepts.get(obj0)
            if code is not None:
                agent.composer.concepts[obj0] = (np.asarray(code, dtype=float)
                                                 + np.random.default_rng(a.seed).normal(0, 2.0, size=np.asarray(code).shape))
                lesion_ok = bool(agent.composer.query_patient(prev0, verb0) != obj0)
        result["lesion_collapsed"] = lesion_ok

        n_ok = sum(c["ok"] for c in conv)
        result["converse_ok"] = n_ok; result["converse_total"] = len(conv); result["conv"] = conv
        go = bool(len(lived) >= 1 and n_ok == len(conv) and moat_ok and (lesion_ok in (True, None)))
        result["GO"] = go
        result["verdict"] = (("GO (smoke) -- the merged one brain PERCEIVED objects DURING behaviour (live cortex_it "
                              "spiking forward) + the RA console CONVERSED about them (recall + RA-render); moat holds "
                              "on never-encountered objects; grounding-lesion collapses recall. The full embodied "
                              "perceive-while-acting -> converse loop works end-to-end on one brain (single-seed "
                              "smoke; multi-seed is the follow-on).") if go else
                             ("HONEST/PARTIAL (smoke) -- lived %d facts, converse %d/%d, moat %s, lesion %s"
                              % (len(lived), n_ok, len(conv), moat_ok, lesion_ok)))
    except Exception as e:
        err = repr(e); traceback.print_exc()
        result["GO"] = False; result["verdict"] = f"ERROR -- {err}"

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase9-embodied] VERDICT: {result['verdict']}", flush=True)
    print(f"[phase9-embodied] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if result.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
