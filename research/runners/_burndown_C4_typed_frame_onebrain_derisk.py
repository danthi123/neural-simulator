#!/usr/bin/env python
"""BURNDOWN C4 (the LAST Bucket-A conversion) cheap-first DE-RISK / HARD GATE: the TYPED verb-frame argument-structure
surface on the SPIKING one-brain substrate == the numpy `ArgStructureComposer` oracle.

C1/C2/C3 brought the word-ordering + the flat-SVO recall/answer onto the spiking `OneBrainComposer`. C4 brings the
TYPED verb-frame path (the typed roles GOAL/THEME/RECIPIENT/LOCATION/...) onto the substrate too, so the console's
`--argstructure` path can run `--composer onebrain`.

The conversion: give the spiking substrate a TYPED-ROLE API -- typed roles bound + stored via the RF complex-synapse
store (like the flat who/what, extended to typed roles), `query_role`, and the verb-frame `render` (the C1 spiking
FrameCQ). Realized by extending `OneBrainComposer` with `typed_roles=(...)` + `store_fact`/`query_role`/`render`
(reuse-by-import; NO sim/ edit; the parent's bind/store/unbind/cleanup machinery -- which iterates self.bind_roles --
carries the typed roles for free; the per-fact bundle never exceeds the few roles a verb frame realizes).

THE HARD GATE (per the prompt):
  * typed-role store / query_role / frame-render on the spiking substrate == the ArgStructureComposer numpy oracle on
    the validated typed cases (e.g. "where does the boy go?" -> "the boy goes to the park"; the THEME/GOAL/etc.
    recalls), ANSWER-IDENTICAL;
  * moat 0-FA;
  * runs on GPU (the substrate);
  * the default numpy ArgStructureComposer path stays BYTE-IDENTICAL (the oracle).
  * WATCH for a bundle-SNR / D boundary -- typed frames are DENSE composites; if the substrate can't hold them at the
    console D, RAISE D (the standard VSA lever) or honestly report the boundary.

Falsification: substrate typed-recall diverges from the oracle (beyond a D-liftable margin), OR the moat breaks ->
STOP, write the honest NEGATIVE (mapping the boundary), push, report.

Run (GPU, the substrate):  SIM_BACKEND=cupy python -u -m research.runners._burndown_C4_typed_frame_onebrain_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, ALL_ROLES, TYPED_ROLES, FUNCTION_WORDS, reparse_to_fact)
from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402

# A vocab covering the three frame classes' fillers (the same fillers the oracle test uses).
VOCAB = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase", "send", "run",
         "park", "house", "ball", "bone", "table", "shelf", "river", "hug"]

# The validated typed-role facts (== tests/test_argstructure_composer.py + the Tier-0.1 de-risk).
FACTS = [
    {"agent": "boy", "action": "go", "GOAL": "park"},
    {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
    {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
    {"agent": "cat", "action": "chase", "patient": "river"},        # default transitive (bare patient)
]

# The role recalls the oracle's test asserts (cue roles -> answer). These define the typed-recall parity bar.
RECALL_CASES = [
    ("GOAL", {"agent": "boy", "action": "go"}, "park"),
    ("agent", {"action": "go", "GOAL": "park"}, "boy"),
    ("THEME", {"agent": "girl", "action": "give"}, "ball"),
    ("RECIPIENT", {"agent": "girl", "action": "give"}, "dog"),
    ("THEME", {"agent": "dog", "action": "put"}, "bone"),
    ("LOCATION", {"agent": "dog", "action": "put"}, "table"),
    ("patient", {"agent": "cat", "action": "chase"}, "river"),
]

# The renders (the headline "the boy goes to the park" + the frame-lexicon coverage). use_framecq=False so the parity
# bar is the substrate-decoded CONTENT + the frame scaffold (the ordering is the separately-validated C1 conversion;
# here we pin the CONTENT decode + the typed-frame scaffold). We ALSO render with use_framecq=True (default) and check
# it equals the same target on the canonical frames.
RENDER_CASES = [
    ({"agent": "boy", "action": "go", "GOAL": "park"}, "the boy goes to the park"),
    ({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"}, "the girl gives the ball to the dog"),
    ({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"}, "the dog puts the bone on the table"),
    ({"agent": "cat", "action": "chase", "patient": "river"}, "the cat chases the river"),
]

# The no-confab moat: stored cue -> answer; unstored cues -> None (0 false-accepts).
MOAT_CASES = [
    ("GOAL", {"agent": "boy", "action": "go"}, "park"),        # should answer
    ("GOAL", {"agent": "boy", "action": "eat"}, None),         # unstored verb -> abstain
    ("GOAL", {"agent": "cat", "action": "go"}, None),          # unstored (agent,action) -> abstain
    ("THEME", {"agent": "dog", "action": "give"}, None),       # unstored (wrong agent for give) -> abstain
]


def run_seed(seed, D=64, verbose=True):
    """Build BOTH the numpy oracle and the spiking-substrate typed composer on identical seed/D/vocab, store the same
    typed facts, and assert the substrate == the oracle (== ground truth) on every typed case + moat 0-FA."""
    oracle = ArgStructureComposer(seed=seed, D=D, vocab=VOCAB, use_spiking_cq=False)   # numpy FrameCQ oracle
    # the spiking substrate (GPU): the typed roles enter bind_roles; the RF complex-synapse store + resonate scan run
    # the bind/store/unbind/cleanup on FIRING NEURONS. use_spiking_cq=False so the render-order parity bar is the
    # substrate CONTENT decode (the spiking CQ order is the C1 conversion, separately validated); we still exercise the
    # default (spiking-CQ on GPU) render below.
    sub = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, typed_roles=TYPED_ROLES, use_spiking_cq=False)
    for f in FACTS:
        oracle.store_fact(f)
        sub.store_fact(f)

    res = {"seed": seed, "D": D}

    # --- (1) typed-role RECALL parity (substrate == oracle == truth) ---
    recall = []
    for role, cue, truth in RECALL_CASES:
        o = oracle.query_role(role, **cue)
        s = sub.query_role(role, **cue)
        recall.append({"role": role, "cue": cue, "truth": truth, "oracle": o, "sub": s,
                       "parity": (s == o), "correct": (s == truth)})
    n_parity = sum(1 for r in recall if r["parity"])
    n_correct = sum(1 for r in recall if r["correct"])
    res["recall"] = {"n_parity": n_parity, "n_correct": n_correct, "n_total": len(recall), "detail": recall}
    if verbose:
        print(f"  [seed {seed}] RECALL parity {n_parity}/{len(recall)}, correct {n_correct}/{len(recall)}: "
              + ", ".join(f"{r['role']}({'=' if r['parity'] else '!'}{r['sub']})" for r in recall), flush=True)

    # --- (2) RENDER parity (substrate == oracle == target) ---
    render = []
    for fact, target in RENDER_CASES:
        o = oracle.render(dict(fact), oracle._composite_for(fact), use_framecq=False)
        s = sub.render(dict(fact), use_framecq=False)
        # ALSO the default (spiking-CQ on GPU) render on the substrate -- canonical frames -> same target.
        s_cq = sub.render(dict(fact), use_framecq=True)
        render.append({"fact": fact, "target": target, "oracle": o, "sub": s, "sub_cq": s_cq,
                       "parity": (s == o), "correct": (s == target), "cq_correct": (s_cq == target)})
    n_rparity = sum(1 for r in render if r["parity"])
    n_rcorrect = sum(1 for r in render if r["correct"])
    n_cq = sum(1 for r in render if r["cq_correct"])
    res["render"] = {"n_parity": n_rparity, "n_correct": n_rcorrect, "n_cq_correct": n_cq, "n_total": len(render),
                     "detail": render}
    if verbose:
        for r in render:
            print(f"  [seed {seed}] RENDER {r['fact'].get('action')}: sub=\"{r['sub']}\" "
                  f"(target=\"{r['target']}\" {'MATCH' if r['correct'] else 'MISS'}; cq=\"{r['sub_cq']}\" "
                  f"{'MATCH' if r['cq_correct'] else 'MISS'})", flush=True)

    # --- (2b) VERIFY: the substrate render re-parses to the stored typed fact ---
    reparse = []
    for fact, _target in RENDER_CASES:
        if fact["action"] == "chase":      # default transitive -- reparse covers the typed frames
            continue
        rendered = sub.render(dict(fact))
        reparse.append(bool(reparse_to_fact(rendered, fact)) if rendered else False)
    res["verify_reparse"] = {"n_ok": int(sum(reparse)), "n_total": len(reparse)}

    # --- (3) MOAT: substrate abstains on unstored, 0 false-accepts (== oracle) ---
    moat = []
    for role, cue, exp in MOAT_CASES:
        o = oracle.query_role(role, **cue)
        s = sub.query_role(role, **cue)
        moat.append({"role": role, "cue": cue, "exp": exp, "oracle": o, "sub": s,
                     "fa": (exp is None and s is not None), "parity": (s == o)})
    fa = sum(1 for m in moat if m["fa"])
    moat_recall_ok = (moat[0]["sub"] == "park")
    n_abstain_ok = sum(1 for m in moat if m["exp"] is None and m["sub"] is None)
    n_abstain = sum(1 for m in moat if m["exp"] is None)
    res["moat"] = {"false_accepts": int(fa), "recall_ok": bool(moat_recall_ok),
                   "abstain_ok": int(n_abstain_ok), "n_abstain": int(n_abstain),
                   "parity": all(m["parity"] for m in moat)}
    if verbose:
        print(f"  [seed {seed}] MOAT: recall_ok={moat_recall_ok}, abstain {n_abstain_ok}/{n_abstain}, "
              f"false_accepts={fa} (parity={res['moat']['parity']})", flush=True)

    # --- (4) AGRAMMATISM anti-cheat: ablate the scaffold -> telegraphic (substrate) ---
    boy = {"agent": "boy", "action": "go", "GOAL": "park"}
    tele = sub.render(dict(boy), ablate_closed_class=True)
    full = sub.render(dict(boy))
    agram_ok = (tele is not None and tele != full
                and all(w not in FUNCTION_WORDS for w in tele.split())
                and "goes" not in (tele.split() if tele else []))
    res["agrammatism"] = {"telegraphic": tele, "full": full, "ok": bool(agram_ok)}
    if verbose:
        print(f"  [seed {seed}] AGRAMMATISM (ablate scaffold): \"{tele}\" -> {'OK' if agram_ok else 'FAIL'}",
              flush=True)

    seed_go = (n_parity == len(recall) and n_correct == len(recall)
               and n_rparity == len(render) and n_rcorrect == len(render) and n_cq == len(render)
               and res["verify_reparse"]["n_ok"] == res["verify_reparse"]["n_total"]
               and fa == 0 and moat_recall_ok and n_abstain_ok == n_abstain and res["moat"]["parity"]
               and agram_ok)
    res["seed_go"] = bool(seed_go)
    return res


def main():
    from sim.backend import is_gpu_backend
    D = int(os.environ.get("C4_D", "64"))
    seeds = tuple(int(s) for s in os.environ.get("C4_SEEDS", "42,43,44,45,46,47").split(","))
    t0 = time.time()
    print(f"[BURNDOWN C4] typed verb-frame argument-structure on the SPIKING one-brain substrate == the numpy "
          f"ArgStructureComposer oracle.  backend={'cupy(GPU substrate)' if is_gpu_backend() else 'numpy(tiny-smoke)'},"
          f" D={D}, seeds={seeds}", flush=True)
    if not is_gpu_backend():
        print("  (WARNING: numpy backend -- this is the tiny-smoke path; the HARD GATE is on GPU/cupy.)", flush=True)
    rows = [run_seed(s, D=D) for s in seeds]

    n_go = sum(1 for r in rows if r["seed_go"])
    all_recall = all(r["recall"]["n_parity"] == r["recall"]["n_total"]
                     and r["recall"]["n_correct"] == r["recall"]["n_total"] for r in rows)
    all_render = all(r["render"]["n_parity"] == r["render"]["n_total"]
                     and r["render"]["n_correct"] == r["render"]["n_total"]
                     and r["render"]["n_cq_correct"] == r["render"]["n_total"] for r in rows)
    total_fa = sum(r["moat"]["false_accepts"] for r in rows)
    all_abstain = all(r["moat"]["abstain_ok"] == r["moat"]["n_abstain"] for r in rows)
    all_moat_parity = all(r["moat"]["parity"] for r in rows)
    all_reparse = all(r["verify_reparse"]["n_ok"] == r["verify_reparse"]["n_total"] for r in rows)
    all_agram = all(r["agrammatism"]["ok"] for r in rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  SUMMARY ({len(seeds)} seeds, D={D}): GO {n_go}/{len(seeds)}", flush=True)
    print(f"    typed RECALL substrate==oracle==truth: {all_recall}", flush=True)
    print(f"    RENDER substrate==oracle==target (incl spiking-CQ): {all_render}", flush=True)
    print(f"    MOAT false-accepts total: {total_fa} (must be 0); abstain all={all_abstain}; "
          f"parity={all_moat_parity}", flush=True)
    print(f"    VERIFY re-parse all: {all_reparse}", flush=True)
    print(f"    agrammatism (ablate->telegraphic): {all_agram}", flush=True)
    print(f"{'='*100}", flush=True)

    go = (n_go == len(seeds) and all_recall and all_render and total_fa == 0 and all_abstain
          and all_moat_parity and all_reparse and all_agram)
    if go:
        print(f"  GO: the TYPED verb-frame surface (store_fact / query_role / verb-frame render) runs on the SPIKING "
              f"substrate ANSWER-IDENTICAL to the numpy ArgStructureComposer oracle on every validated typed case "
              f"(GOAL/THEME/RECIPIENT/LOCATION recalls + 'the boy goes to the park' etc.), with 0 moat false-accepts "
              f"and the agrammatism control intact. ==> the console --argstructure path can run --composer onebrain.",
              flush=True)
    else:
        print(f"  NO-GO: the substrate typed path diverges from the oracle OR breaks the moat. If a bundle-SNR/D "
              f"boundary, re-run with a larger C4_D; else a valid NEGATIVE mapping the boundary.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {"go": bool(go), "n_go": int(n_go), "n_seeds": len(seeds), "D": D, "seeds": list(seeds),
           "all_recall": bool(all_recall), "all_render": bool(all_render), "total_false_accepts": int(total_fa),
           "all_abstain": bool(all_abstain), "all_moat_parity": bool(all_moat_parity),
           "all_reparse": bool(all_reparse), "all_agrammatism": bool(all_agram),
           "backend": ("cupy" if is_gpu_backend() else "numpy"), "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_burndown_C4_typed_frame_onebrain.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
