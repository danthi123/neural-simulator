#!/usr/bin/env python
"""Tier 0.1 + 0.2 BUILD verification (run AFTER the Step-1 de-risk GO).

Exercises the PRODUCTION argstructure_composer.py end-to-end:
  (1) on the REAL first-chat brain (grounded codes from brain3000pos_w7000.npz) with REAL corpus typed-role facts
      (extracted with the preposition kept) -- store/recall/render the boy-go-to-the-park fact + the moat;
  (2) the AGRAMMATISM anti-cheat (ablate the closed-class scaffold -> telegraphic) on the real brain;
  (3) the FIXED-CAPACITY WM (0.2): the WM substrate neuron-count is CONSTANT as vocab grows (16 vs 320 vs 3000) --
      the balloon is gone by construction;
  (4) the verb-frame lexicon COVERAGE (go/give/put/default) on a tiny vocab.

Run:  SIM_BACKEND=numpy python -u -m research.runners._tier0_argstructure_build_verify
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
    ArgStructureComposer, FixedCapacityDiscourseWM, reparse_to_fact, FUNCTION_WORDS)

BRAIN = os.path.join(_REPO, "bridges", "firstchat", "brain3000pos_w7000.npz_seed42.npz")
TYPED_FACTS = os.path.join(_REPO, "research", "findings", "raw", "_tier0_typed_facts.json")


def load_grounded_codes(npz_path, words):
    """Load {word: grounded-phasor[D]} for the requested words from a first-chat brain bundle."""
    # allow_pickle: our OWN first-chat brain artifact (trusted, self-generated; the vocab array is dtype=object) --
    # the same convention as _corpus_svo_extract.load_vocab. NOT an untrusted source.
    d = np.load(npz_path, allow_pickle=True)
    vocab = [str(w).lower() for w in d["vocab"]]
    grounded = np.asarray(d["grounded"])
    idx = {w: i for i, w in enumerate(vocab)}
    out = {}
    for w in words:
        if w in idx:
            out[w] = grounded[idx[w]].astype(float)
    return out, int(grounded.shape[1])


def test_real_brain():
    """Store/recall/render real corpus typed-role facts on the REAL brain's grounded codes + the moat."""
    print("\n[1] REAL BRAIN (grounded codes) + REAL corpus typed-role facts", flush=True)
    facts_all = json.load(open(TYPED_FACTS, encoding="utf-8"))
    # pick the clean GOAL/LOCATION facts (prep to/on/into) -- the target argument-structure cases
    clean = [f for f in facts_all if f.get("prep") in ("to", "on", "into")
             and any(k in f for k in ("GOAL", "LOCATION", "RECIPIENT"))]
    # build the vocab from every word these facts mention
    words = set()
    for f in clean:
        for k, v in f.items():
            if k not in ("count", "attest", "prep") and isinstance(v, str):
                words.add(v)
    grounded, D = load_grounded_codes(BRAIN, words)
    vocab = sorted(words)
    print(f"    {len(clean)} clean typed-role facts; vocab {len(vocab)} words; brain D={D}; "
          f"{len(grounded)}/{len(vocab)} grounded by the brain", flush=True)
    comp = ArgStructureComposer(seed=42, D=D, vocab=vocab, grounded_codes=grounded)
    for f in clean:
        fact = {k: v for k, v in f.items() if k not in ("count", "attest", "prep")}
        comp.store_fact(fact)

    # recall + render every stored fact; the moat: abstain on an unstored cue
    n_recall_ok, n_render_ok, n_reparse_ok = 0, 0, 0
    boy_park_text = None
    for f in clean:
        fact = {k: v for k, v in f.items() if k not in ("count", "attest", "prep")}
        verb, ag = fact["action"], fact["agent"]
        obl_role = next((r for r in ("GOAL", "LOCATION", "RECIPIENT", "patient") if r in fact), None)
        rec = comp.query_role(obl_role, agent=ag, action=verb)
        if rec == fact[obl_role]:
            n_recall_ok += 1
        rendered = comp.render(fact, comp._composite_for(fact))
        if reparse_to_fact(rendered, fact):
            n_reparse_ok += 1
        # a fluent render must carry function words (not telegraphic)
        if any(w in FUNCTION_WORDS for w in rendered.split()):
            n_render_ok += 1
        if ag == "boy" and verb == "go" and fact.get("GOAL") == "park":
            boy_park_text = rendered
    n = len(clean)
    print(f"    recall typed role: {n_recall_ok}/{n}  | render fluent: {n_render_ok}/{n}  | "
          f"VERIFY re-parse: {n_reparse_ok}/{n}", flush=True)
    # show a few renders
    for f in clean[:6]:
        fact = {k: v for k, v in f.items() if k not in ("count", "attest", "prep")}
        print(f"      ({fact['agent']} {fact['action']} {[v for k,v in fact.items() if k not in ('agent','action')]})"
              f" -> \"{comp.render(fact, comp._composite_for(fact))}\"", flush=True)
    if boy_park_text:
        print(f"    >>> the headline fact: \"{boy_park_text}\"", flush=True)

    # MOAT: an unstored cue must abstain (None); 0 false-accepts over a battery of never-stored cues
    unstored = [("boy", "give"), ("dog", "go"), ("park", "go"), ("cat", "fly")]
    fa = sum(1 for ag, vb in unstored if comp.query_role("GOAL", agent=ag, action=vb) is not None)
    print(f"    MOAT false-accepts on {len(unstored)} unstored cues: {fa} (must be 0)", flush=True)

    # AGRAMMATISM (anti-cheat): ablate scaffold -> telegraphic (no function words, no tense), differs from full
    agram_ok = True
    if boy_park_text:
        bf = {"agent": "boy", "action": "go", "GOAL": "park"}
        tele = comp.render(bf, comp._composite_for(bf), ablate_closed_class=True)
        differs = tele != boy_park_text
        no_fw = all(w not in FUNCTION_WORDS for w in tele.split())
        no_tense = "goes" not in tele.split()
        agram_ok = differs and no_fw and no_tense
        print(f"    AGRAMMATISM (ablate scaffold): \"{tele}\"  (differs={differs}, no-func-words={no_fw}, "
              f"no-tense={no_tense} -> {'OK' if agram_ok else 'FAIL'})", flush=True)

    ok = (n_recall_ok == n and n_reparse_ok == n and n_render_ok == n and fa == 0 and agram_ok)
    return {"name": "real_brain", "ok": bool(ok), "n_facts": n, "n_recall_ok": n_recall_ok,
            "n_reparse_ok": n_reparse_ok, "n_render_fluent": n_render_ok, "false_accepts": int(fa),
            "agrammatism_ok": bool(agram_ok), "boy_park_render": boy_park_text}


def test_fixed_wm_constant_neuron_count():
    """Tier 0.2: the WM substrate neuron-count is CONSTANT as vocab grows -- the balloon is gone."""
    print("\n[2] FIXED-CAPACITY WM (0.2): neuron-count constant across vocab sizes", flush=True)
    counts = {}
    balloon = {}
    for V in (16, 320, 3000):
        vocab = [f"w{i}" for i in range(V)]
        wm = FixedCapacityDiscourseWM(seed=42, D=64, vocab=vocab, n_slots=4)
        wm.hold(vocab[:3])              # hold a 3-item ordered sequence (triggers the RF bind/bundle/unbind)
        _ = wm.read(0)
        counts[V] = wm.wm_neuron_count()
        balloon[V] = max(600, 60 * V)   # what content_selection_spiking.py would allocate
    const = (len(set(counts.values())) == 1)
    print(f"    fixed WM neuron-count: {counts}  ->  CONSTANT={const}", flush=True)
    print(f"    (the OLD balloon n=max(600,60*len(vocab)) would have been: {balloon})", flush=True)
    return {"name": "fixed_wm", "ok": bool(const), "neuron_counts": counts, "balloon_counts": balloon}


def test_frame_coverage():
    """Verb-frame lexicon coverage: go/give/put/default each render with their frame's scaffold."""
    print("\n[3] FRAME LEXICON coverage (tiny vocab)", flush=True)
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase",
             "park", "ball", "bone", "table", "river"]
    comp = ArgStructureComposer(seed=42, D=64, vocab=vocab)
    cases = [
        ({"agent": "boy", "action": "go", "GOAL": "park"}, "the boy goes to the park"),
        ({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
         "the girl gives the ball to the dog"),
        ({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
         "the dog puts the bone on the table"),
        ({"agent": "cat", "action": "chase", "patient": "river"}, "the cat chases the river"),
    ]
    n_ok = 0
    for fact, target in cases:
        comp.store_fact(fact)
        rendered = comp.render(fact, comp._composite_for(fact))
        ok = rendered == target
        n_ok += int(ok)
        print(f"    {fact['action']:7s} -> \"{rendered}\"  ({'MATCH' if ok else 'MISMATCH vs '+target})", flush=True)
    return {"name": "frame_coverage", "ok": n_ok == len(cases), "n_ok": n_ok, "n_total": len(cases)}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("[Tier 0.1+0.2 BUILD verification] production argstructure_composer on the real brain + the fixed WM",
          flush=True)
    results = [test_real_brain(), test_fixed_wm_constant_neuron_count(), test_frame_coverage()]
    all_ok = all(r["ok"] for r in results)
    print(f"\n{'='*100}", flush=True)
    for r in results:
        print(f"  {r['name']:18s}: {'PASS' if r['ok'] else 'FAIL'}", flush=True)
    print(f"  BUILD VERIFY: {'PASS' if all_ok else 'FAIL'}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"all_ok": bool(all_ok), "results": results}
    path = os.path.join(_REPO, "research", "findings", "raw", "_tier0_argstructure_build_verify.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
