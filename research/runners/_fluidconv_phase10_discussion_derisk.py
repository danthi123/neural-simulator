"""Phase-10 DE-RISK: OPEN-ENDED grounded DISCUSSION -- discuss ideas/concepts using RELEVANT knowledge + synthesis
(the owner's "talk in depth about ideas the brain has relevant info on but no explicit answer").

The gap (measured): the fluid console does one-fact-lookup + abstains on cross-fact synthesis ("how are dogs and cats
different?" -> "I don't know") or concept discussion ("tell me about predators" -> "I don't know"). Per the scoping
(`2026-07-01-open-ended-grounded-discussion-scoping.md`), the retrieval half is ~90% built; the #1 cheap-first is to
RETRIEVE the topic's grounded neighbourhood (the association-graph facts-about/mentioning + the dlPFC discourse
planner's on-topic associates -- the SAME machinery the GO DiscursiveTurn uses) -> condition the RA-fine-tuned 21M on
the MULTIPLE retrieved facts -> generate ONE fluent multi-sentence grounded DISCUSSION -> VERIFY every asserted
known-entity SVO is grounded (allow non-fact connective/opinion glue; DROP ungrounded fact-claims). Moat reframed from
hard-abstain to grounded-elaboration-with-hedging (per `feedback_moat_not_hard`): the brain discusses what it RELATES,
and hedges/says-where-it-ends rather than fabricating.

Biology: spreading activation (Collins-Loftus / ACT-R) selects the material; PFC Control (Hagoort MUC / catalog G.08)
sequences it; the re-parse VERIFY is the gist-editable-but-not-fabricating guard.

METRICS (>=3 seeds): (a) DISCUSS = an open/concept/compare question the gate-first path ABSTAINS on now -> a
multi-sentence answer citing >=2 grounded facts about the topic's neighbourhood (strictly richer than 1-fact);
(b) GROUNDED = 0 ungrounded known-entity fact-claims in the discussion (VERIFY drops any); (c) LESION = a topic with
an EMPTY neighbourhood -> honest hedge/abstain (not fabrication); (d) PERMUTED = retrieve the WRONG topic's
neighbourhood -> the discussion is about the wrong thing (retrieval is load-bearing); (e) CONFAB-PROBE = inject an
ungrounded fact into the render -> VERIFY drops it.

GO = discuss (>=2 grounded facts, richer than 1-fact) + 0 ungrounded + lesion-hedges + permuted-load-bearing, >=3
seeds. Reuse-by-import; NO `sim/` edit.
Run: python -m research.runners._fluidconv_phase10_discussion_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase10_discussion.json"

# a small KB with CATEGORY facts (so concept discussion is possible) + object facts + attributes. The brain has
# RELEVANT info on "predator" (dog/fox/cat are predators) but no exact "tell me about predators" answer -> it must
# DISCUSS from the neighbourhood.
KB = [
    ("dog", "eat", "meat"), ("dog", "chase", "cat"), ("dog", "is", "predator"),
    ("cat", "eat", "fish"), ("cat", "chase", "mouse"), ("cat", "is", "predator"),
    ("fox", "eat", "rabbit"), ("fox", "chase", "rabbit"), ("fox", "is", "predator"),
    ("bird", "eat", "seed"), ("bird", "is", "prey"),
    ("mouse", "eat", "seed"), ("mouse", "is", "prey"),
]


class Discussant:
    """Retrieve a topic's grounded neighbourhood (association-graph) -> RA-synthesize a multi-fact discussion ->
    VERIFY grounded. The retrieval mirrors the GO DiscursiveTurn's facts-about/mentioning gather; the synthesis is
    the RA generator conditioned on the MULTIPLE retrieved facts; VERIFY keeps the moat."""

    def __init__(self, seed, kb):
        self.kb = list(kb)
        self.agents = {f[0] for f in kb}; self.patients = {f[2] for f in kb}; self.actions = {f[1] for f in kb}
        self.inflect = _build_inflection_map(sorted(self.actions))
        self.store_keys = {tuple(f) for f in kb}
        vocab = sorted(self.agents | self.patients | self.actions)
        self.agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
        for (a, v, p) in kb:
            self.agent.hear(f"{a} {v} {p}")
        self.faculty = FTFaculty()

    def neighbourhood(self, topic, permute_topic=None):
        """The topic's grounded neighbourhood: facts where topic is agent OR patient (association-graph adjacency).
        For a CATEGORY topic (e.g. 'predator'), the members' facts too (X is predator -> X's facts). permute_topic
        (anti-cheat) retrieves a DIFFERENT topic's neighbourhood."""
        t = permute_topic or topic
        facts = [list(f) for f in self.kb if t in (f[0], f[2])]
        members = [f[0] for f in self.kb if f[2] == t]          # X is <t> -> X is a member of category t
        for m in members:
            facts += [list(f) for f in self.kb if f[0] == m and list(f) not in facts]
        return facts

    def discuss(self, question, topic, permute_topic=None, inject_confab=None, max_facts=7):
        """Retrieve the topic's neighbourhood -> render EACH fact SEPARATELY via the RA generator (the validated
        FAITHFUL single-fact render -- NOT a multi-fact context, which makes the 21M confabulate by mixing entities)
        -> per-sentence VERIFY -> concatenate the verified sentences into a discussion paragraph (moat: an ungrounded
        render is DROPPED). This is the GO DiscursiveTurn's per-sentence approach; the paragraph is the multi-fact
        grounded discussion. Non-fact glue (the frame) is free; only asserted known-entity SVOs are policed."""
        nb = self.neighbourhood(topic, permute_topic=permute_topic)
        render_facts = list(nb[:max_facts])
        if inject_confab:                                       # confab-probe: try to also assert a FALSE fact
            render_facts = render_facts + [list(inject_confab)]
        if not render_facts:
            return {"question": question, "topic": topic, "neighbourhood": [], "reply": "I don't know much about that.",
                    "grounded_svos": [], "ungrounded": [], "n_grounded": 0, "hedged": True}
        sentences = []; grounded = []; ungrounded = []
        for (a, v, p) in render_facts:
            q = f"what is the {a} ?" if v == "is" else f"what does the {a} {v} ?"   # copula gets the "what is" frame
            one = self.faculty.answer(f"the {a} {_v3(v)} {p} .", q)                  # faithful single-fact render
            svos = _extract_all_svos(one, self.agents, self.actions, self.patients, self.inflect)
            g = [s for s in svos if _fact_key(s) in self.store_keys]
            u = [s for s in svos if _fact_key(s) not in self.store_keys]
            if u or not g:                                      # VERIFY: DROP a sentence that asserts an ungrounded
                ungrounded += u                                 # (or no) grounded fact -> the moat (never emit it)
                continue
            sentences.append(one.strip()); grounded += g
        frame = f"Here's what I know about the {topic}:" if not permute_topic else f"Here's what I know:"
        reply = (frame + " " + " ".join(sentences)).strip() if sentences else "I don't know much about that."
        return {"question": question, "topic": topic, "neighbourhood": nb, "reply": reply,
                "grounded_svos": grounded, "ungrounded": ungrounded,
                "n_grounded": len({tuple(s) for s in grounded}), "hedged": (not sentences)}


def run(seed):
    d = Discussant(seed, KB)
    # (a) DISCUSS: open/concept/compare questions the gate-first path abstains on
    discuss = {}
    discuss["concept"] = d.discuss("tell me about predators", "predator")          # concept/category
    discuss["about"] = d.discuss("tell me about the dog", "dog")                    # rich single-topic
    # (b) grounded + (a) richness computed below
    # (c) LESION: an empty-neighbourhood topic -> honest hedge
    lesion = d.discuss("tell me about dragons", "dragon")
    # (d) PERMUTED: retrieve the WRONG topic -> load-bearing
    permuted = d.discuss("tell me about the dog", "dog", permute_topic="bird")
    # (e) CONFAB-PROBE: inject a FALSE fact into the context -> VERIFY drops it
    confab = d.discuss("tell me about the dog", "dog", inject_confab=("dog", "eat", "fish"))  # dog eats meat, NOT fish

    concept_ok = bool(discuss["concept"]["n_grounded"] >= 2 and not discuss["concept"]["ungrounded"])
    about_ok = bool(discuss["about"]["n_grounded"] >= 2 and not discuss["about"]["ungrounded"])
    lesion_ok = bool(lesion["hedged"] or lesion["n_grounded"] == 0)
    # permuted load-bearing: the permuted discussion's grounded facts are about the WRONG topic (bird), not dog
    perm_about_dog = any(s[0] == "dog" for s in permuted["grounded_svos"])
    permuted_ok = bool(not perm_about_dog)     # retrieving bird's neighbourhood -> the discussion isn't about the dog
    # confab-probe: the injected false fact (dog eat fish) must NOT appear as a grounded assertion (it's ungrounded)
    confab_leaked = any(tuple(s) == ("dog", "eat", "fish") for s in confab["grounded_svos"])
    confab_ok = bool(not confab_leaked)

    return {"seed": seed, "discuss": discuss, "lesion": lesion, "permuted": permuted, "confab": confab,
            "concept_ok": concept_ok, "about_ok": about_ok, "lesion_ok": lesion_ok,
            "permuted_ok": permuted_ok, "confab_ok": confab_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time(); err = None; per_seed = []
    try:
        for s in a.seeds:
            r = run(s); per_seed.append(r)
            print(f"  [seed {s}] concept {r['concept_ok']} | about {r['about_ok']} | lesion-hedge {r['lesion_ok']} | "
                  f"permuted-loadbearing {r['permuted_ok']} | confab-dropped {r['confab_ok']}", flush=True)
            print(f"      Q 'tell me about predators' -> \"{r['discuss']['concept']['reply']}\" "
                  f"(grounded facts: {r['discuss']['concept']['n_grounded']})", flush=True)
            print(f"      Q 'tell me about the dog'   -> \"{r['discuss']['about']['reply']}\" "
                  f"(grounded facts: {r['discuss']['about']['n_grounded']})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        disc_ok = all(r["concept_ok"] and r["about_ok"] for r in per_seed)
        les_ok = all(r["lesion_ok"] for r in per_seed)
        perm_ok = all(r["permuted_ok"] for r in per_seed)
        conf_ok = all(r["confab_ok"] for r in per_seed)
        go = bool(disc_ok and les_ok and perm_ok and conf_ok)
        verdict = (("GO -- OPEN-ENDED grounded DISCUSSION: an open/concept/compare question the gate-first path "
                    "abstains on now yields a multi-sentence answer citing >=2 grounded facts from the topic's "
                    "neighbourhood (RA-synthesized), 0 ungrounded fact-claims (moat via VERIFY), an empty-neighbourhood "
                    "topic HEDGES (no fabrication), the retrieval is load-bearing (permuted -> wrong-topic discussion), "
                    "and an injected false fact is DROPPED by VERIFY. >=3 seeds. The brain DISCUSSES what it relates, "
                    "grounded + traceable, instead of one-fact-lookup + abstain.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if disc_ok else [f"discuss concept/about {[(r['concept_ok'], r['about_ok']) for r in per_seed]} "
                                            "(the RA synthesis didn't cite >=2 grounded facts, or leaked an ungrounded claim)"]) +
                       ([] if les_ok else ["lesion did not hedge (fabricated on an empty neighbourhood)"]) +
                       ([] if perm_ok else ["permuted not load-bearing"]) +
                       ([] if conf_ok else ["confab-probe leaked (VERIFY missed an injected false fact)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase10_discussion", "GO": go, "verdict": verdict,
               "resolves": "open-ended grounded discussion: retrieve the topic's neighbourhood -> RA-synthesize a "
                           "multi-fact discussion -> VERIFY grounded; moat as honest hedging, not hard-abstain.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "grounded multi-fact discussion + retrieval-augmented synthesis over the stored+adjacent "
                                 "neighbourhood is achievable; free abstractive synthesis / open-world inference BEYOND "
                                 "the retrieved facts is the field's genuine wall -- the honest hedge (discuss what it "
                                 "relates, flag guesses, say where knowledge ends) is the deliverable at that boundary."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase10-discussion] VERDICT: {verdict}", flush=True)
    print(f"[phase10-discussion] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
