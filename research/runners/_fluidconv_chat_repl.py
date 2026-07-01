"""The fluid-conversation CONSOLE -- one coherent chat loop tying Phases 2-5 together (the owner's console-not-
dashboard priority). Talk to the brain like an LLM: ask grounded questions, use pronouns across turns, TEACH it new
facts, and it abstains ("I don't know") on what it hasn't learned.

  QUESTION  "what does the dog eat?" / "what does it chase?" (pronoun) / "who eats meat?" / "does the dog eat meat?"
            -> interrogative parse -> brain GATE (moat gate-FIRST) -> RA-fine-tuned 21M focused answer -> VERIFY.
  STATEMENT "the wolf eats rabbit" / "wolf eat rabbit"  -> hear (LEARN) -> "ok, i learned that the wolf eats rabbit."
  UNTAUGHT  -> "I don't know."   (the no-confab moat)

Assembles: `MultiTurnAgent` (multi-turn anaphora, Phase 4) + `FTFaculty` (the RA render/QA fine-tuned generator,
Phase 2) + the Phase-3 gate->answer->VERIFY + Phase-5 growth. The BRAIN does comprehension + knowledge + grounding +
moat; the minimized (~21M) brain-trained brain-gated generator does fluency. Reuse-by-import; NO sim/ edit.

Run (scripted smoke / demo):
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --demo
  SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --script "what does the dog eat?|the wolf eats rabbit|what does the wolf eat?"
Run (interactive): ... (no --script/--demo -> reads stdin; blank line or 'quit' exits)
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

from research.runners.multi_turn_agent import MultiTurnAgent  # noqa: E402
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT, SUBJECTS as FT_SUBJECTS, OBJECTS as FT_OBJECTS  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_chat_repl_demo.json"
_QWORDS = {"what", "who", "does", "do", "is", "are", "tell", "can", "could", "why", "how", "when", "where", "?"}
_PRON = {"it", "its", "they", "them", "that"}
_STOP = {"the", "a", "an", "does", "do", "did", "the", "to", "of", "please"}


class FluidChat:
    """One coherent fluid-conversation agent (Phases 2-5 assembled)."""

    def __init__(self, seed=42, extra_vocab=None):
        with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
            self.cur = json.load(fh)
        facts = self.cur.get("facts", [])
        self.agents = {f[0] for f in facts}
        self.patients = {f[2] for f in facts}
        self.actions = {f[1] for f in facts}
        self.inflect = _build_inflection_map(sorted(self.actions))
        # a generous pre-allocated vocab so new facts can be TAUGHT (composer codes are fixed at build): curriculum +
        # the fine-tune's broad subject/object pools + any extra.
        vocab = set(_collect_vocab(self.cur)) | set(FT_SUBJECTS) | set(FT_OBJECTS) | set(extra_vocab or [])
        self.vocab = sorted(vocab)
        # referents (for anaphora) must stay small (one 40-neuron attractor/referent in n=600) -> a curated set
        referents = sorted(set(sorted(self.agents)[:6]) | set(list(self.patients)[:4]))
        self.mta = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in self.vocab},
                                  seed=seed, defer_planner=True, enable_biased_competition=False, composer_kind="rf")
        _teach(self.mta.agent, self.cur)
        self.store_keys = {tuple(f) for f in facts}
        self.faculty = FTFaculty()
        self.npar = self.faculty.npar

    def _content(self, toks):
        subj = next((t for t in toks if t in self.agents or t in self.vocab and t not in _STOP and t not in self.actions), None)
        verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
        return subj, verb

    def _is_question(self, toks):
        return bool(set(toks) & _QWORDS)

    def _answer(self, subj, verb):
        """Phase-3 turn: GATE -> RA-render -> VERIFY. Writes the answer as the salient referent (Phase-4)."""
        p = self.mta.agent.what_does(subj, verb)
        if p is None:
            return None, "I don't know."
        ctx = f"the {subj} {_v3(verb)} {p} ."
        ans = self.faculty.answer(ctx, f"what does the {subj} {verb} ?")
        svos = _extract_all_svos(ans, self.agents, self.actions, self.patients, self.inflect)
        ung = [s for s in svos if _fact_key(s) not in self.store_keys]
        verified = bool((([subj, verb, p] in svos) or (p in ans.split())) and not ung)
        reply = ans if verified else f"The {subj} {_v3(verb)} {p}."
        if p in self.mta.referents:
            self.mta._write_referent(p)
        return p, reply

    def turn(self, text):
        """One conversation turn: statement -> learn; question -> gate->answer->verify; untaught -> abstain."""
        raw = text.strip()
        toks = [t.strip("?.!,") for t in raw.lower().split()]
        toks = [t for t in toks if t]
        if not toks:
            return "?"
        if self._is_question(toks):
            # QUESTION. resolve a pronoun agent via the held referent; else a concrete subject.
            has_pron = any(t in _PRON for t in toks)
            verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
            subj = next((t for t in toks if t in self.agents), None)
            if has_pron and subj is None:
                subj = self.mta._resolve("it", query_verb=verb)
                if subj is None:
                    return "I don't know."
            if subj is None or verb is None:
                return "I don't know."
            _p, reply = self._answer(subj, verb)
            return reply
        # STATEMENT -> LEARN (growth). parse S V O over the vocab.
        subj = next((t for t in toks if t in self.vocab and t not in _STOP and self.inflect.get(t) not in self.actions), None)
        verb = next((self.inflect.get(t) for t in toks if self.inflect.get(t) in self.actions), None)
        obj = None
        if verb is not None:
            after = toks[toks.index(next(t for t in toks if self.inflect.get(t) == verb)) + 1:]
            obj = next((t for t in after if t in self.vocab and t not in _STOP), None)
        if subj and verb and obj:
            self.mta.hear(f"{subj} {verb} {obj}")
            self.store_keys.add((subj, verb, obj))
            # the learned subject/object become known entities so LATER questions find them (growth)
            self.agents.add(subj); self.patients.add(obj)
            return f"ok, i learned that the {subj} {_v3(verb)} {obj}."
        return "sorry, i didn't understand that."


DEMO = [
    "what does the dog chase?",      # -> the dog chases cat.   (writes 'cat')
    "what does it eat?",             # -> it=cat -> the cat eats fish.  (anaphora, Phase 4)
    "the wolf eats rabbit",          # -> ok, learned  (growth, Phase 5)
    "what does the wolf eat?",       # -> the wolf eats rabbit.  (learned fact usable)
    "what does the lion eat?",       # -> I don't know.  (moat)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="run the canned demo transcript (Q&A + anaphora + growth + moat)")
    ap.add_argument("--script", default=None, help="'|'-separated turns to run then exit")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    try:
        chat = FluidChat(seed=a.seed)
        print(f"[fluid-chat] ready -- brain (comprehension+knowledge+moat) + a ~{chat.npar:.0f}M brain-gated "
              f"generator (fluency). dev={chat.faculty.device}\n", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"ERROR: {e}"); return 1

    turns = DEMO if a.demo else (a.script.split("|") if a.script else None)
    transcript = []
    if turns is not None:
        for t in turns:
            reply = chat.turn(t)
            transcript.append({"you": t.strip(), "brain": reply})
            print(f"  you>   {t.strip()}\n  brain> {reply}", flush=True)
        # a light self-check for the canned demo
        go = None
        if a.demo:
            def _said(i, sub):
                return sub in transcript[i]["brain"].lower()
            go = bool(_said(0, "cat") and _said(1, "fish") and "learned" in transcript[2]["brain"].lower()
                      and _said(3, "rabbit") and "know" in transcript[4]["brain"].lower())
            print(f"\n  [demo self-check] Q&A + anaphora(it->cat->fish) + growth(wolf) + moat(lion) all correct: {go}",
                  flush=True)
        out = {"probe": "fluidconv_chat_repl", "seed": a.seed, "demo": bool(a.demo), "transcript": transcript,
               "demo_all_correct": go, "npar_M": round(chat.npar, 1),
               "elapsed_seconds": round(time.time() - t0, 1)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  [saved] {a.out}", flush=True)
        return 0 if (go is None or go) else 1
    # interactive
    print("  (interactive; blank line or 'quit' to exit)\n", flush=True)
    while True:
        try:
            line = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        print(f"  brain> {chat.turn(line)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
