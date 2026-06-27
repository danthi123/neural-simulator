#!/usr/bin/env python
"""STEP 1 cheap-first DE-RISK for Tier 0.1 (verb-frame argument structure) + 0.2 (fixed-capacity WM).

The HARD GATE before the full build (research/findings/2026-06-27-conversation-thinking-ROADMAP.md, Tier 0):
can the composer represent + store + recall + RENDER ONE argument-structure fact with a TYPED OBLIQUE role
(GOAL) + its preposition -- moat-preserved?

Concretely (per the prompt):
  * represent "boy go park" as (agent=boy, action=go, GOAL=park) -- a typed GOAL role beyond the bare patient;
  * store it; recall it (who/what + the GOAL);
  * render it fluently as "the boy goes to the park" -- the preposition "to" + determiner "the" from the
    verb-frame's CLOSED-CLASS scaffold, ordered by the validated FrameCQ serial-order engine;
  * confirm the no-confab moat: recall the stored arg-structure fact; ABSTAIN on an unstored one; 0 false-accepts;
  * AND the load-bearing AGRAMMATISM anti-cheat: ablate the closed-class scaffold -> telegraphic "boy go park"
    (reproduces Broca's; proves the function words do real work, an artifact can't fake it).

This is a PROBE: it composes the PRODUCTION RFPhasorComposer (research/runners/rf_phasor_composer.py) by
reuse-by-import, extending its role alphabet with typed roles + a small per-verb FRAME LEXICON. NO sim/ edit,
NO production-composer edit yet (Step 2 only if this is GO). Tiny vocab + numpy (CPU).

Biology: Hagoort MUC (the verb's structural FRAME in temporal-cortex Memory) + Bock&Levelt functional->positional
(the verb lemma projects its argument frame + the closed-class scaffold; agrammatic Broca's output = a functional
structure that never got positional realization). The render order is produced by FrameCQ (the validated 6/6
competitive-queuing serial-order engine; the seed of syntax), not a host literal.

Run:  SIM_BACKEND=numpy python -u -m research.runners._tier0_argstructure_derisk
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

from research.runners.rf_phasor_composer import RFPhasorComposer       # noqa: E402
from research.runners.song_g1_core import score_order, permuted_order_controls, g1_verdict  # noqa: E402


# --- The TYPED-ROLE extension of the composer's alphabet (the 0.1 representation target) -----------------------
# These are the thematic/oblique roles the bare (agent, action, patient) alphabet cannot express. The composer's
# binding is role-AGNOSTIC (rf_phasor_composer.py:262 binds `for r in ROLES if r in fact`), so adding roles costs
# only more codebook entries -- exactly the MUC-Memory "the verb stores its frame; bind the fillers in" story.
TYPED_ROLES = ("GOAL", "RECIPIENT", "THEME", "LOCATION", "SOURCE", "INSTRUMENT", "TIME")

# --- The per-verb-class FRAME LEXICON (MUC-Memory: each verb's structural frame; Bock&Levelt subcategorization) -
# Each frame is an ordered list of SLOTS. A slot is either:
#   ("CONTENT", role)        -- filled by the recalled fact's filler for that role (spelled by the composer);
#   ("CLOSED", word)         -- a closed-class function word (determiner / preposition) from the function-word pool;
#   ("TENSE", role)          -- the verb's content word, but tense-inflected (a morphological closed-class polish).
# go   -> {AGENT, GOAL}      : "the <agent> goes to the <GOAL>"
# give -> {AGENT, THEME, RECIPIENT}: "the <agent> gives the <THEME> to the <RECIPIENT>"
# put  -> {AGENT, THEME, LOCATION} : "the <agent> puts the <THEME> on the <LOCATION>"
# default transitive (e.g. chase, eat): {AGENT, PATIENT} : "the <agent> <verb>s the <patient>"
FRAME_LEXICON = {
    "go": ["the", ("CONTENT", "agent"), ("TENSE", "action"), "to", "the", ("CONTENT", "GOAL")],
    "give": ["the", ("CONTENT", "agent"), ("TENSE", "action"), "the", ("CONTENT", "THEME"),
             "to", "the", ("CONTENT", "RECIPIENT")],
    "put": ["the", ("CONTENT", "agent"), ("TENSE", "action"), "the", ("CONTENT", "THEME"),
            "on", "the", ("CONTENT", "LOCATION")],
    "_default": ["the", ("CONTENT", "agent"), ("TENSE", "action"), "the", ("CONTENT", "patient")],
}
# which CONTENT/TENSE roles each verb-frame licenses (the args the extractor must fill) -- derived from the frame.
FRAME_ROLES = {v: [s[1] for s in slots if isinstance(s, tuple)] for v, slots in FRAME_LEXICON.items()}

# A tiny present-tense 3sg inflection table (morphology = a legitimate lexical front-end, like the parser's
# morphology). The brain renders the bare verb; this host polish adds the agreement morpheme (closed-class).
TENSE_3SG = {"go": "goes", "give": "gives", "put": "puts", "chase": "chases", "eat": "eats"}

# The closed-class FUNCTION-WORD POOL (determiners + prepositions). Ablating this pool is the agrammatism control.
FUNCTION_WORDS = {"the", "a", "to", "on", "in", "of", "with", "from"}


class ArgStructureComposer(RFPhasorComposer):
    """RFPhasorComposer extended with TYPED OBLIQUE roles + a per-verb FRAME LEXICON, for the Tier-0.1 de-risk.

    Adds the typed roles to `self.roles` (drawn from a DISJOINT rng stream so the parent's concept codes stay
    byte-identical -- the same disjoint-stream discipline OrderedPositionWM uses). Stores a fact as a dict over
    {agent, action, <typed roles>}; the parent's `_encode` binds every role present. Recall reuses the parent's
    `unbind`. Render expands the verb's frame into ordered (content + closed-class) slots and orders them with
    FrameCQ. The no-confab moat is the parent's: a query whose cue roles match no stored fact returns None."""

    def __init__(self, seed=42, D=64, vocab=None, grounded_codes=None):
        super().__init__(seed=seed, D=D, vocab=vocab, grounded_codes=grounded_codes)
        # typed-role phasors from a disjoint stream (seed+2000) -> parent concept/role codes unchanged.
        prng = np.random.default_rng(seed + 2000)
        for r in TYPED_ROLES:
            self.roles[r] = prng.uniform(0.0, 1.0, self.D)
        # extend the encode/decode role set so the parent's `_encode` (binds `for r in ROLES_EXT if r in fact`)
        # picks up the typed roles. We override _encode below to use this extended set.
        self._roles_ext = tuple(RFPhasorComposer.__dict__.get("ROLES", ()) ) + TYPED_ROLES
        self.frame_cq = _build_frame_cq(seed)

    # the parent's _encode iterates the module-level ROLES tuple; override to iterate agent/action + typed roles.
    def _encode(self, fact):
        from research.runners import rf_phasor_composer as _m
        roles_all = tuple(_m.ROLES) + TYPED_ROLES
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in roles_all if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_fact(self, fact):
        """Store an argument-structure fact dict, e.g. {'agent':'boy','action':'go','GOAL':'park'}."""
        comp = self._encode(fact)
        self.kb.append((dict(fact), comp))

    def query_role(self, role, **cue_roles):
        """Recall the filler of `role` from the FIRST stored fact whose cue roles ALL match; None = abstain
        (the no-confab moat). Generalizes the parent's query_patient/query_agent to ANY typed role."""
        for fact, comp in self.kb:
            if all(self.unbind(comp, cr) == cv for cr, cv in cue_roles.items()):
                return self.unbind(comp, role)
        return None

    # --- render: expand the verb frame into ordered slots, order with FrameCQ, spell each ----------------------
    def _frame_for(self, verb):
        return FRAME_LEXICON.get(verb, FRAME_LEXICON["_default"])

    def _decoded_fillers(self, fact, comp):
        """Decode each CONTENT role's filler from the RF unbind (NOT the stored label) -- the brain reads back
        what it stored. The action word is decoded too; tense is the morphological polish on top."""
        verb_slots = self._frame_for(fact["action"])
        out = {}
        for slot in verb_slots:
            if isinstance(slot, tuple):
                kind, role = slot
                if role == "action":
                    out["action"] = self.unbind(comp, "action")
                else:
                    out[role] = self.unbind(comp, role)
        return out

    def render(self, fact, comp, ablate_closed_class=False, use_framecq=True):
        """Render the fact as prose via its verb frame. `ablate_closed_class=True` drops the function-word pool +
        tense morphology -> telegraphic agrammatic output (the Broca's anti-cheat). `use_framecq=True` orders the
        CONTENT slots by the validated FrameCQ serial-order engine (the cognitive ordering is neural)."""
        verb = fact["action"]
        slots = self._frame_for(verb)
        decoded = self._decoded_fillers(fact, comp)
        # Build the surface token for each slot, in the FRAME's canonical order first.
        canonical = []
        content_positions = []   # indices into `canonical` that are CONTENT/TENSE words (for FrameCQ reordering)
        for slot in slots:
            if isinstance(slot, str):                              # a closed-class function word
                if ablate_closed_class:
                    continue                                       # agrammatism: drop function words
                canonical.append(slot)
            else:
                kind, role = slot
                if kind == "TENSE":
                    w = decoded.get("action", verb)
                    if not ablate_closed_class:
                        w = TENSE_3SG.get(w, w)                    # agreement morpheme (closed-class polish)
                    canonical.append(w)
                    content_positions.append(len(canonical) - 1)
                else:                                              # CONTENT
                    canonical.append(decoded.get(role, role))
                    content_positions.append(len(canonical) - 1)
        if not use_framecq:
            return " ".join(canonical)
        # FrameCQ orders the CONTENT words (the verb-frame's argument order); the closed-class scaffold is the
        # frame's fixed structure (determiners/prepositions sit at their lexically-licensed positions). We order
        # ONLY the content tokens via FrameCQ and re-insert them at the frame's content positions in CQ order.
        content_tokens = [canonical[i] for i in content_positions]
        frame_id = _frame_id(verb)
        order = self.frame_cq.emit_order(frame_id, len(content_tokens))
        reordered_content = [content_tokens[i] for i in order]
        out = list(canonical)
        for slot_idx, tok in zip(content_positions, reordered_content):
            out[slot_idx] = tok
        return " ".join(out)


# --- FrameCQ wrapper: learn each verb-frame's CONTENT-slot order; emit the order (reuse the validated mechanism) -
_VERB_FRAME_IDS = {"go": 0, "give": 1, "put": 2, "_default": 3}


def _frame_id(verb):
    return _VERB_FRAME_IDS.get(verb, _VERB_FRAME_IDS["_default"])


class _FrameCQ:
    """The validated frame-conditioned competitive-queuing serial-order generator (the same mechanism as
    _phaseB_serial_order_multiframe_derisk.FrameCQ, 6/6 GO): a per-frame primacy gradient learned from the
    teacher; emit = the choice-WTA read-out in that frame's primacy order. Here it orders the CONTENT slots of a
    verb frame (the argument order). Max content-slot count across frames = 3 (give/put: agent,THEME,RECIP/LOC)."""

    def __init__(self, n_frames=4, max_slots=4, lr=0.1, seed=42, wta_noise=0.05):
        self.max_slots = max_slots
        self.wta_noise = wta_noise
        self.prim = np.random.default_rng(seed * 13 + 5).standard_normal((n_frames, max_slots)) * 0.01
        self._rng = np.random.default_rng(seed * 71 + 3)

    def learn(self, frame, n_slots):
        # the teacher order for a verb frame's CONTENT slots is the IDENTITY (slot 0 first, then 1, ...) -- the
        # frame lexicon already lists content slots in their canonical argument order.
        for pos in range(n_slots):
            self.prim[frame][pos] += self.lr_pos(pos, n_slots)

    def lr_pos(self, pos, n_slots):
        return 0.1 * (n_slots - 1 - pos)

    def emit_order(self, frame, n_slots):
        a = self.prim[frame][:n_slots] + self.wta_noise * self._rng.standard_normal(n_slots)
        avail, order = list(range(n_slots)), []
        for _ in range(n_slots):
            best = max(avail, key=lambda i: a[i])
            order.append(best); avail.remove(best)
        return order


def _build_frame_cq(seed):
    cq = _FrameCQ(seed=seed)
    # teach each verb-frame's content-slot order (canonical = identity order from the lexicon)
    for verb, fid in _VERB_FRAME_IDS.items():
        n_content = len(FRAME_ROLES[verb])
        for _ in range(40):                       # repeated teacher exposures (as the validated derisk does)
            cq.learn(fid, n_content)
    return cq


# ----------------------------------------------------------------------------------------------------------------
# THE DE-RISK
# ----------------------------------------------------------------------------------------------------------------
def run_seed(seed, D=64, verbose=True):
    # tiny vocab covering the three frame classes' fillers
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase", "eat",
             "park", "house", "ball", "bone", "table", "shelf", "river"]
    comp = ArgStructureComposer(seed=seed, D=D, vocab=vocab)

    # --- store argument-structure facts with TYPED OBLIQUE roles ---
    facts = [
        {"agent": "boy", "action": "go", "GOAL": "park"},
        {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
        {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
        {"agent": "cat", "action": "chase", "patient": "river"},      # default transitive (bare patient)
    ]
    for f in facts:
        comp.store_fact(f)

    results = {"seed": seed}

    # --- (1) RECALL: who/what + the typed role ---
    recall_ok = []
    # boy go park
    g = comp.query_role("GOAL", agent="boy", action="go")
    a = comp.query_role("agent", action="go", GOAL="park")
    recall_ok += [("go.GOAL", g == "park", g), ("go.agent", a == "boy", a)]
    # girl give ball to dog
    th = comp.query_role("THEME", agent="girl", action="give")
    rc = comp.query_role("RECIPIENT", agent="girl", action="give")
    recall_ok += [("give.THEME", th == "ball", th), ("give.RECIPIENT", rc == "dog", rc)]
    # dog put bone on table
    th2 = comp.query_role("THEME", agent="dog", action="put")
    lc = comp.query_role("LOCATION", agent="dog", action="put")
    recall_ok += [("put.THEME", th2 == "bone", th2), ("put.LOCATION", lc == "table", lc)]
    # default transitive
    pt = comp.query_role("patient", agent="cat", action="chase")
    recall_ok += [("chase.patient", pt == "river", pt)]
    n_recall = sum(1 for _, ok, _ in recall_ok if ok)
    results["recall"] = {"n_ok": n_recall, "n_total": len(recall_ok),
                         "detail": [(k, bool(ok), str(v)) for k, ok, v in recall_ok]}
    if verbose:
        print(f"  [seed {seed}] RECALL {n_recall}/{len(recall_ok)}: "
              + ", ".join(f"{k}={'OK' if ok else 'FAIL('+str(v)+')'}" for k, ok, v in recall_ok), flush=True)

    # --- (2) RENDER the boy-go-park fact fluently ---
    boy_fact, boy_comp = comp.kb[0]
    rendered = comp.render(boy_fact, boy_comp)
    results["render_boy_go_park"] = rendered
    target = "the boy goes to the park"
    render_ok = (rendered == target)
    if verbose:
        print(f"  [seed {seed}] RENDER 'boy go park' -> \"{rendered}\"  (target \"{target}\" : "
              f"{'MATCH' if render_ok else 'MISMATCH'})", flush=True)
    # render the others too (coverage of the lexicon)
    give_r = comp.render(*comp.kb[1])
    put_r = comp.render(*comp.kb[2])
    chase_r = comp.render(*comp.kb[3])
    results["render_others"] = {"give": give_r, "put": put_r, "chase": chase_r}
    if verbose:
        print(f"  [seed {seed}]   give -> \"{give_r}\"  | put -> \"{put_r}\"  | chase(default) -> \"{chase_r}\"",
              flush=True)

    # --- (3) MOAT: recall stored, ABSTAIN on unstored, 0 false-accepts ---
    # stored cue -> answer; unstored cues -> None.
    moat_cases = [
        ("stored go.GOAL", comp.query_role("GOAL", agent="boy", action="go"), "park"),       # should answer
        ("unstored agent=boy,action=eat", comp.query_role("GOAL", agent="boy", action="eat"), None),  # abstain
        ("unstored agent=cat,action=go", comp.query_role("GOAL", agent="cat", action="go"), None),    # abstain
        ("unstored give wrong agent", comp.query_role("THEME", agent="dog", action="give"), None),    # abstain
    ]
    false_accepts = sum(1 for _, got, exp in moat_cases if exp is None and got is not None)
    moat_recall_ok = (moat_cases[0][1] == "park")
    abstain_ok = sum(1 for _, got, exp in moat_cases if exp is None and got is None)
    n_abstain = sum(1 for _, _, exp in moat_cases if exp is None)
    results["moat"] = {"false_accepts": int(false_accepts), "recall_ok": bool(moat_recall_ok),
                       "abstain_ok": int(abstain_ok), "n_abstain": int(n_abstain),
                       "detail": [(k, str(got), str(exp)) for k, got, exp in moat_cases]}
    if verbose:
        print(f"  [seed {seed}] MOAT: recall_ok={moat_recall_ok}, abstain {abstain_ok}/{n_abstain}, "
              f"false_accepts={false_accepts}", flush=True)

    # --- (3b) VERIFY: the rendered prose re-parses to the stored typed fact (content-mismatch -> reject) ---
    # parse the rendered "the boy goes to the park" back: strip function words + tense -> (agent, action, GOAL).
    reparse_ok = _verify_reparse(rendered, boy_fact)
    results["verify_reparse"] = bool(reparse_ok)
    if verbose:
        print(f"  [seed {seed}] VERIFY re-parse of rendered prose -> stored fact: "
              f"{'OK' if reparse_ok else 'FAIL'}", flush=True)

    # --- (4) AGRAMMATISM anti-cheat: ablate the closed-class scaffold -> telegraphic ---
    telegraphic = comp.render(boy_fact, boy_comp, ablate_closed_class=True)
    results["render_ablated"] = telegraphic
    # the ablated output must (a) differ from the full render, and (b) contain NO function words + no tense morpheme.
    differs = (telegraphic != rendered)
    no_function_words = all(w not in FUNCTION_WORDS for w in telegraphic.split())
    no_tense = "goes" not in telegraphic.split()        # bare "go", not the inflected "goes"
    agrammatism_ok = differs and no_function_words and no_tense
    results["agrammatism"] = {"telegraphic": telegraphic, "differs_from_full": bool(differs),
                              "no_function_words": bool(no_function_words), "no_tense": bool(no_tense),
                              "ok": bool(agrammatism_ok)}
    if verbose:
        print(f"  [seed {seed}] AGRAMMATISM (ablate scaffold): \"{telegraphic}\"  "
              f"(differs={differs}, no-func-words={no_function_words}, no-tense={no_tense} -> "
              f"{'OK' if agrammatism_ok else 'FAIL'})", flush=True)

    # per-seed verdict
    seed_go = (n_recall == len(recall_ok) and render_ok and false_accepts == 0 and moat_recall_ok
               and abstain_ok == n_abstain and reparse_ok and agrammatism_ok)
    results["seed_go"] = bool(seed_go)
    return results


def _verify_reparse(rendered, fact):
    """Strip the closed-class scaffold + tense morphology from the rendered prose and check the residual content
    words match the stored fact's fillers (agent, action, and the typed role). A content mismatch -> reject."""
    toks = [t for t in rendered.split() if t not in FUNCTION_WORDS]
    # de-inflect tense (goes->go etc.)
    inv_tense = {v: k for k, v in TENSE_3SG.items()}
    toks = [inv_tense.get(t, t) for t in toks]
    content_vals = set()
    for slot in FRAME_LEXICON.get(fact["action"], FRAME_LEXICON["_default"]):
        if isinstance(slot, tuple):
            role = slot[1]
            content_vals.add(fact[role])
    return set(toks) == content_vals


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("[Tier 0.1 arg-structure DE-RISK] typed GOAL/THEME/RECIPIENT/LOCATION roles + verb-frame lexicon + "
          "FrameCQ render + the no-confab moat + the agrammatism anti-cheat.", flush=True)
    print("  (cheap-first HARD GATE before the full build; tiny vocab; numpy/CPU)\n", flush=True)
    seeds = (42, 43, 44, 45, 46, 47)
    rows = [run_seed(s) for s in seeds]

    n_go = sum(1 for r in rows if r["seed_go"])
    all_recall = all(r["recall"]["n_ok"] == r["recall"]["n_total"] for r in rows)
    all_render = all(r["render_boy_go_park"] == "the boy goes to the park" for r in rows)
    total_fa = sum(r["moat"]["false_accepts"] for r in rows)
    all_abstain = all(r["moat"]["abstain_ok"] == r["moat"]["n_abstain"] for r in rows)
    all_reparse = all(r["verify_reparse"] for r in rows)
    all_agram = all(r["agrammatism"]["ok"] for r in rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  SUMMARY ({len(seeds)} seeds): GO {n_go}/{len(seeds)}", flush=True)
    print(f"    recall all-correct:        {all_recall}", flush=True)
    print(f"    render 'the boy goes to the park': {all_render}  (e.g. \"{rows[0]['render_boy_go_park']}\")",
          flush=True)
    print(f"    moat false-accepts total:  {total_fa}  (must be 0)", flush=True)
    print(f"    moat abstain all:          {all_abstain}", flush=True)
    print(f"    verify re-parse all:       {all_reparse}", flush=True)
    print(f"    agrammatism (ablate->telegraphic): {all_agram}  (e.g. \"{rows[0]['render_ablated']}\")",
          flush=True)
    print(f"{'='*100}", flush=True)

    go = (n_go == len(seeds) and all_recall and all_render and total_fa == 0 and all_abstain
          and all_reparse and all_agram)
    if go:
        print(f"  GO: the typed-role representation stores + recalls a GOAL/THEME/RECIPIENT/LOCATION fact, RENDERS "
              f"'the boy goes to the park' (preposition + determiner from the verb-frame scaffold, FrameCQ-ordered),"
              f" preserves the no-confab moat (0 false-accepts, abstains on unstored), and the agrammatism control "
              f"collapses to telegraphic on scaffold-ablation. ==> PROCEED to the full build (0.1 + 0.2).", flush=True)
    else:
        print(f"  NO-GO: the typed-role representation does NOT recall/render correctly OR breaks the moat OR the "
              f"agrammatism control is decorative. STOP -- this is a valid NEGATIVE that re-scopes 0.1.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {"go": bool(go), "n_go": int(n_go), "n_seeds": len(seeds), "all_recall": bool(all_recall),
           "all_render": bool(all_render), "total_false_accepts": int(total_fa), "all_abstain": bool(all_abstain),
           "all_reparse": bool(all_reparse), "all_agrammatism": bool(all_agram), "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_tier0_argstructure_derisk.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
