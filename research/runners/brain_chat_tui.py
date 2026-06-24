"""brain_chat_tui — an easy TUI to LOAD a developed/trained brain and hold a MULTI-TURN conversation with it.

The owner uses this to TALK to a developed brain (e.g. the self-knowledge brain). It LOADS the EXACT developed
brain (its grounded concept codes + its stored facts + its vocab) and runs a multi-turn chat loop:

    prompt -> parse the question (self-aliases resolved, anaphora resolved from the discourse buffer)
           -> RECALL from the brain (who/what/yes-no/describe/reason via the agent; the no-confab GATE)
           -> RENDER fluently (default: the OFF-BRIDGE Qwen grounded-language faculty, gate->constrain->verify,
              loaded ONCE + kept warm; --stub-renderer uses the template-stub, GPU-FREE, for the CPU smoke)
           -> print the answer, OR "I don't know about that." on abstention (the MOAT).

The render is GATED + VERIFIED: the brain supplies + verifies the CONTENT (the moat holds EVEN WITH a real
generative LLM in the loop); the faculty's only job is fluent surface form.

LOAD SOURCES (auto-detected from --load):
  * a `developed_brain_io` BUNDLE directory (brain.json + grounded_codes.npz + facts.json + lineage/) -- the
    self-contained "developed brain" the develop loop / a save_developed_brain call writes. THE GENERIC PATH.
  * the SELF-KNOWLEDGE brain: a `_self_knowledge_grounded_codes.json` codes blob (+ the curriculum it was
    developed on) -- the brain reconstructs on the learned codes and re-teaches the curriculum facts. Pass the
    codes .json (or just `--self-knowledge` to use the default codes path).
  * NOTHING / a tiny fallback (the GPU-FREE smoke): build a tiny CPU brain from a handful of facts.

COMMANDS in the chat loop:
  /raw      toggle the brain's OWN neural renderer (no LLM) -- the unvarnished brain (raw recalled triple).
  /facts    list what the brain knows (its stored facts).
  /help     show the commands.
  /quit     exit (also: /exit, /q, Ctrl-D).

SELF-REFERENCE: 'you'/'your'/'I'/'me'/'it' map to the agent 'brain' so 'what are you?' / 'how do you learn?'
resolve against the brain's self-facts.

REUSE-BY-IMPORT, NO `sim/` edit. The OFF-BRIDGE Qwen faculty is the runtime fluent renderer (used when the owner
runs it for real with a free GPU); the GPU-FREE smoke validates the BRAIN side on CPU with the template-stub.

Usage:
    # talk to a saved developed brain (real, with the off-bridge Qwen renderer, free GPU):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --load <developed-brain-dir-or-codes.json>

    # the self-knowledge brain (after `_self_knowledge_demo` saved its codes):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --self-knowledge

    # GPU-FREE smoke (template-stub renderer, scripted stdin):
    SIM_BACKEND=numpy python -m research.runners.brain_chat_tui --stub-renderer --tiny-demo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.developed_brain_io import (  # noqa: E402
    load_developed_brain, is_developed_brain_bundle,
)

# default self-knowledge artifacts (so `--self-knowledge` works with no path)
_SK_CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")
_SK_CURRICULUM = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_self_knowledge.json")


# ============================================================================================================
# Self-reference + a free-text question -> a (kind, cue) the brain answers against its stored SVO facts.
# (The keyword->fact matcher is faithful: it routes a question to the stored fact whose WORDS the question
# mentions, synonym-resolved; an unmatched question ABSTAINS -- the no-confab moat. Ported from the
# self-knowledge demo's router so a plain English question resolves, while carrying ZERO project knowledge.)
# ============================================================================================================

DEFAULT_SELF_ALIASES = {"you", "your", "yours", "i", "me", "my", "it", "its", "yourself", "itself"}

_STOP = {"what", "who", "does", "do", "the", "a", "an", "is", "are", "of", "to", "from", "that", "how",
         "did", "will", "can", "and", "with", "in", "on", "for", "by", "as", "be", "this", "these",
         "those", "there", "here", "prevent", "prevents", "tell", "about", "say", "know", "knows"}

_QUESTION_SYNONYMS = {
    "learn": {"learns", "learning"}, "learns": {"learns"},
    "forget": {"forgetting", "replays", "replay", "remembers"}, "forgetting": {"forgetting"},
    "remember": {"remembers", "memory"}, "memory": {"memory", "remembers", "consolidates"},
    "lie": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "lying": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "guess": {"moat", "confabulation", "refuses", "guessing"},
    "use": {"uses"}, "uses": {"uses"}, "using": {"uses"},
    "teach": {"teaches"}, "teaches": {"teaches"}, "taught": {"teaches"},
    "store": {"stores", "remembers", "composer"}, "speak": {"phrases", "faculty", "answers"},
    "answer": {"answers", "remembers"}, "think": {"uses", "neurons"}, "work": {"uses", "runs"},
    "consolidate": {"consolidates"}, "grow": {"grows", "develops", "tiers"},
    "develop": {"develops", "daily"}, "made": {"has", "uses", "neurons", "spikes"}, "make": {"has", "uses"},
}


class QuestionRouter:
    """Map a free-text question to a stored SVO fact (the GATE cue), resolving self-aliases. Decisive only when a
    CONTENT keyword of the question appears in some fact (a bare self-alias match is not enough -> abstain)."""

    def __init__(self, self_aliases=None):
        self.self_aliases = set(self_aliases) if self_aliases else set(DEFAULT_SELF_ALIASES)

    def _resolve_self(self, word):
        w = word.lower().strip(".,!?")
        return "brain" if w in self.self_aliases else w

    def keywords(self, question):
        toks = [self._resolve_self(t) for t in re.findall(r"[a-zA-Z]+", question.lower())]
        kws = set()
        for t in toks:
            if t in _STOP and t != "brain":
                continue
            kws.add(t)
            kws |= _QUESTION_SYNONYMS.get(t, set())
        return kws, toks

    def match_fact(self, question, stored_facts):
        """Return (gate_svo or None, score). The best stored fact by content-keyword overlap; an identity question
        ('what are you') routes to a defining 'brain has/is/uses ...' fact."""
        kws, toks = self.keywords(question)
        content_kws = kws - {"brain"}
        is_identity_q = ("brain" in kws and not content_kws
                         and any(w in {"be", "are", "is", "am"} for w in toks))
        if is_identity_q:
            # a defining fact about the brain, in preference order (covers base + 3rd-person inflected verbs)
            for want in ("has", "have", "is", "uses", "use"):
                for (a, v, p) in stored_facts:
                    if a == "brain" and v == want:
                        return [a, v, p], 1
            # fall back to ANY fact whose agent is 'brain' (the brain's own self-statement)
            for (a, v, p) in stored_facts:
                if a == "brain":
                    return [a, v, p], 1
        best, best_score = None, 0
        for (a, v, p) in stored_facts:
            ftoks = {a, v, p}
            content_hits = len(content_kws & ftoks)
            brain_hit = 1 if ("brain" in kws and "brain" in ftoks) else 0
            score = content_hits * 10 + brain_hit
            if content_hits >= 1 and score > best_score:
                best, best_score = (a, v, p), score
        return (list(best) if best is not None else None), best_score


# ============================================================================================================
# The fluent renderers (default = the off-bridge Qwen; --stub-renderer = the template-stub, GPU-free).
# Both expose `render_svo(a, v, p) -> (surface, asserted_svo_or_None)`; the TUI gate->constrain->verify wraps them.
# ============================================================================================================

class StubRenderer:
    """The GPU-FREE template-stub faculty (the P3 `TemplateStubFaculty`): renders a gated SVO into a fluent
    surface form CONSTRAINED to the fact's own words, and exposes the canonical content SVO it asserts (what
    VERIFY re-parses). Stands in for the real Qwen renderer in the CPU smoke -- NO model download, deterministic."""

    name = "template-stub (GPU-free)"

    def __init__(self):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self._fac = TemplateStubFaculty()

    def render_svo(self, a, v, p):
        surface, asserted = self._fac.render_svo(a, v, p)
        return surface, asserted


class QwenRenderer:
    """The OFF-BRIDGE Qwen-0.5B grounded-language faculty (the spiking forward, reused-by-import from the
    integration de-risk). Loaded ONCE + kept warm. `render_svo` returns the generated prose + None for the
    asserted SVO (the TUI re-parses the PROSE to recover the asserted content -- the genuine VERIFY of a real
    generative model's output)."""

    name = "off-bridge Qwen-0.5B (spiking forward)"

    def __init__(self, T=16, max_new_tokens=24, seed=42):
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[tui] WARNING: CUDA not available -- the Qwen renderer will be slow on CPU.", flush=True)
        self._fac = SpikingQwenFaculty(T=T, max_new_tokens=max_new_tokens, seed=seed, device=device)
        self.load_seconds = self._fac.load_seconds

    def render_svo(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo(a, v, p)
        return surface, None      # asserted SVO recovered by the TUI's re-parse of the prose

    def render_svo_regen(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo_regen(a, v, p)
        return surface, None


# ============================================================================================================
# The chat brain: wraps a loaded conversational agent + the router + the renderer + the gate/constrain/verify.
# ============================================================================================================

class ChatBrain:
    def __init__(self, agent, *, self_aliases=None, renderer=None, verbose_thinking=True):
        # agent is a MultiTurnAgent (preferred, for anaphora) or a BrainConversationalAgent
        self.agent = agent
        self.inner = getattr(agent, "agent", agent)             # the BrainConversationalAgent
        self.is_multiturn = hasattr(agent, "held_referent")     # MultiTurnAgent exposes this
        self.router = QuestionRouter(self_aliases=self_aliases)
        self.renderer = renderer
        self.verbose_thinking = verbose_thinking
        self.raw_mode = False                                   # /raw toggles the brain's own renderer (no LLM)
        # the brain's stored facts (string-only roles) + content-token sets for the VERIFY re-parse
        self._refresh_facts()

    def _refresh_facts(self):
        comp = self.inner.composer
        self.stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in comp.kb
                             if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
        self.agents_set = {a for a, _, _ in self.stored_facts}
        self.actions_set = {v for _, v, _ in self.stored_facts}
        self.patients_set = {p for _, _, p in self.stored_facts}
        from research.runners._grounded_lang_integration_derisk import _build_inflection_map
        self.inflect = _build_inflection_map(sorted(self.actions_set))

    # --- the GATE: a free-text question -> a verified stored SVO fact, or None (abstain) ---
    def gate(self, question):
        """Resolve the question to a stored fact and VERIFY it against the spiking recall. Returns
        (gate_svo or None). An anaphor in the question is resolved from the discourse WM (multi-turn)."""
        # resolve anaphora in the question FIRST (multi-turn): replace a leading 'it'/'that'/'they' with the held
        # referent, so a follow-up 'what does it eat' uses the prior turn's referent.
        q = self._resolve_anaphora(question)
        gate_svo, _score = self.router.match_fact(q, self.stored_facts)
        if gate_svo is None:
            return None
        a, v, p = gate_svo
        # VERIFY the matcher's pick against the brain's SPIKING recall (the answer must be the spiking memory's)
        recalled = self.inner.what_does(a, v)
        if recalled == p:
            # write the answer's salient referent (the PATIENT/object) into the discourse WM so a NEXT-turn pronoun
            # resolves to it -- exactly as MultiTurnAgent.hear() writes only the patient. We treat a CONCRETE entity
            # (one that is itself the AGENT of some fact -- i.e. something the brain can say more about) as the
            # discourse referent; this matches the validated single-referent anaphora pattern (a fresh referent
            # dominates the WM) and avoids polluting the WM with abstract patients (e.g. 'spikes'/'words') that are
            # not salient pronoun antecedents. The no-confab moat is unaffected.
            if isinstance(p, str) and p in self.agents_set:
                self._note_referent(p)
            return [a, v, p]
        return None

    def _resolve_anaphora(self, question):
        """If the question's first content token is a pronoun and the discourse WM holds a referent, substitute it
        (multi-turn anaphora). Only the MultiTurnAgent has a WM loop; otherwise pass the question through."""
        if not self.is_multiturn:
            return question
        anaphors = {"it", "that", "they", "them", "this"}
        toks = question.split()
        for i, t in enumerate(toks):
            tl = t.lower().strip(".,!?")
            if tl in anaphors:
                ref = self.agent.held_referent()[0]
                if ref is not None:
                    toks[i] = ref
                    return " ".join(toks)
        return question

    def _note_referent(self, word):
        """Write a referent into the discourse WM (multi-turn), so a later pronoun resolves to it."""
        if self.is_multiturn and isinstance(word, str):
            try:
                self.agent._write_referent(word)
            except Exception:
                pass

    # --- the CONSTRAIN + VERIFY render of a gated fact into fluent prose ---
    def render(self, gate_svo):
        """Render the gated SVO into a fluent sentence (CONSTRAIN) and VERIFY the content re-parses to the gated
        fact. Returns the verified fluent string, or the brain's raw triple on a verify miss / raw mode / no
        renderer. NEVER emits unverified generative prose as the answer."""
        a, v, p = gate_svo
        if self.raw_mode or self.renderer is None:
            return self._raw(gate_svo)
        surface, asserted = self.renderer.render_svo(a, v, p)
        if self._verify(surface, asserted, gate_svo):
            return surface
        # a generative renderer can DRIFT: try a tighter re-prompt once (if supported), else speak the raw fact
        if hasattr(self.renderer, "render_svo_regen"):
            surface2, asserted2 = self.renderer.render_svo_regen(a, v, p)
            if self._verify(surface2, asserted2, gate_svo):
                return surface2
        return self._raw(gate_svo) + "   [unverified render -> spoke the brain's raw fact]"

    def _verify(self, surface, asserted, gate_svo):
        """VERIFY: re-parse the rendered content back into an SVO and require it to MATCH the gated fact. For the
        stub, `asserted` is the canonical content SVO; for Qwen, `asserted` is None -> re-parse the PROSE."""
        if asserted is None:
            from research.runners._grounded_lang_integration_derisk import _extract_svo_from_prose
            asserted = _extract_svo_from_prose(surface, self.agents_set, self.actions_set,
                                               self.patients_set, self.inflect)
            if asserted is None:
                return False
        parsed = self.inner.parse(asserted, voice="active")
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        return rsvo == list(gate_svo)

    def _raw(self, gate_svo):
        """The brain's OWN renderer: the raw recalled triple as a plain sentence (no LLM)."""
        return " ".join(str(x) for x in gate_svo)

    # --- the full turn ---
    def answer(self, question):
        """One conversational turn: GATE (recall + abstain) -> CONSTRAIN+VERIFY render. Returns
        (answer_string, abstained_bool)."""
        gate_svo = self.gate(question)
        if gate_svo is None:
            return "I don't know about that.", True
        return self.render(gate_svo), False

    def list_facts(self):
        """The brain's stored facts (for /facts)."""
        self._refresh_facts()
        return list(self.stored_facts)


# ============================================================================================================
# Loading a developed brain from the various sources.
# ============================================================================================================

def _load_self_knowledge(codes_path, curriculum_path, seed, use_multiturn, enable_neural_render):
    """Reconstruct the self-knowledge brain: build a BrainConversationalAgent/MultiTurnAgent on the saved learned
    grounded codes + teach the curriculum facts. Returns (agent, self_aliases, n_facts)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    with open(os.path.abspath(curriculum_path), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    # the full taught fact set as SVO (facts + attribute_facts as (noun, 'is', adj))
    facts = [tuple(f) for f in cur.get("facts", [])]
    facts += [(noun, "is", adj) for noun, adj in cur.get("attribute_facts", [])]
    # vocab: the concept set + general-knowledge + untaught fall-backs (so the moat abstains STRUCTURALLY)
    vocab = set(["is"])
    for a, v, p in facts:
        vocab.update([a, v, p])
    vocab |= {"france", "paris", "two", "plus", "four", "romeo", "juliet", "wrote", "shakespeare",
              "color", "blue", "legs", "has", "many"}
    for probe in cur.get("deliberately_untaught_project_facts", {}).get("probes", []):
        for w in probe:
            if isinstance(w, str) and w != "?":
                vocab.add(w)
    vocab = sorted(vocab)
    grounded = None
    if codes_path and os.path.exists(codes_path):
        with open(codes_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
        print(f"[tui] loaded {len(grounded)} developed grounded codes from "
              f"{os.path.relpath(codes_path, _REPO)}", flush=True)
    else:
        print("[tui] no developed codes file found -- the brain answers the taught facts on its own seed codes "
              "(run _self_knowledge_demo to develop + save the learned codes).", flush=True)
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        actions = {v for _a, v, _p in facts} | {"is"}
        referents = [w for w in vocab if w not in actions]
        # size the WM loop to hold every referent (2x headroom) so a large vocabulary does NOT overrun the
        # pattern budget (the SpikingLoopContextBuffer holds n/pattern_size patterns) -- same rule as
        # _longitudinal_develop_loop.build_agent.
        pattern_size = 40
        wm_n = max(600, 2 * pattern_size * max(1, len(referents)))
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts,
                               grounded_codes=grounded if grounded else None, seed=seed,
                               wm_n=wm_n, wm_pattern_size=pattern_size,
                               enable_neural_render=enable_neural_render, composer_kind="rf",
                               enable_biased_competition=False)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                         grounded_codes=grounded if grounded else None,
                                         composer_kind="rf", enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    n = 0
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
        n += 1
    aliases = set(cur.get("self_reference", {}).get("agent_aliases", [])) | DEFAULT_SELF_ALIASES
    return agent, aliases, n


def _build_tiny_demo(seed, use_multiturn, enable_neural_render):
    """A tiny CPU brain for the GPU-FREE smoke: a handful of self-facts + a couple of object facts. Mirrors the
    self-knowledge shape so the smoke exercises self-reference + the moat + multi-turn anaphora."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    # base-form verbs so the template-stub's 3rd-person inflection reads cleanly (use->uses, learn->learns).
    # 'cat' is the OBJECT of (dog chase cat) AND the SUBJECT of (cat eat fish) -- the validated chainable-referent
    # pattern so 'what does it eat' resolves 'it'->cat (the dog's chase-object) and answers 'fish'.
    facts = [
        ("brain", "use", "spikes"),
        ("brain", "learn", "words"),
        ("brain", "store", "memory"),
        ("dog", "chase", "cat"),
        ("cat", "eat", "fish"),
    ]
    actions = {v for _a, v, _p in facts}
    vocab = sorted({w for f in facts for w in f} | {"river", "bird", "fish"})  # extra encodable, never-stored
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        referents = [w for w in vocab if w not in actions]
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts, seed=seed,
                               enable_neural_render=enable_neural_render, composer_kind="rf",
                               enable_biased_competition=False)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf",
                                         enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return agent, DEFAULT_SELF_ALIASES, len(facts)


def load_brain(args):
    """Resolve --load / --self-knowledge / --tiny-demo into (agent, self_aliases, n_facts, source_desc)."""
    use_mt = not args.no_multiturn
    nr = args.neural_render
    # explicit developed-brain bundle directory
    if args.load and is_developed_brain_bundle(args.load):
        agent, manifest = load_developed_brain(args.load, use_multiturn=use_mt, enable_neural_render=nr)
        aliases = set(manifest.get("self_aliases") or []) | DEFAULT_SELF_ALIASES
        n = manifest.get("n_facts", len(getattr(agent, "agent", agent).composer.kb))
        return agent, aliases, n, f"developed-brain bundle: {args.load}"
    # self-knowledge brain (explicit flag, or a --load pointing at a codes .json)
    if args.self_knowledge or (args.load and str(args.load).endswith(".json")):
        codes = args.load if (args.load and str(args.load).endswith(".json")) else _SK_CODES
        curriculum = args.curriculum or _SK_CURRICULUM
        agent, aliases, n = _load_self_knowledge(codes, curriculum, args.seed, use_mt, nr)
        return agent, aliases, n, f"self-knowledge brain (codes={os.path.relpath(codes, _REPO) if os.path.exists(codes) else 'seed-codes'})"
    # tiny CPU demo (the GPU-FREE smoke)
    if args.tiny_demo or not args.load:
        agent, aliases, n = _build_tiny_demo(args.seed, use_mt, nr)
        return agent, aliases, n, "tiny CPU demo brain"
    raise FileNotFoundError(f"--load {args.load!r} is neither a developed-brain bundle nor a codes .json")


# ============================================================================================================
# The renderer factory.
# ============================================================================================================

def build_renderer(args):
    """Build the fluent renderer: the off-bridge Qwen (default) or the template-stub (--stub-renderer / smoke)."""
    if args.stub_renderer:
        return StubRenderer()
    if args.no_renderer:
        return None
    return QwenRenderer(T=args.T, max_new_tokens=args.max_new_tokens, seed=args.seed)


# ============================================================================================================
# The interactive REPL.
# ============================================================================================================

_BANNER = """\
============================================================================
  BRAIN CHAT  --  talk to a developed brain about what it knows
============================================================================
  Source : {source}
  Knows  : {n_facts} facts   |   Renderer: {renderer}
  Self   : 'you'/'your'/'I'/'me'/'it' map to the brain (ask 'what are you?')
  Moat   : the brain ABSTAINS ('I don't know about that.') on anything it
           was not taught -- it never makes things up.
  Commands: /facts  /raw  /help  /quit
============================================================================
"""

_HELP = """\
  /facts   list the facts the brain knows
  /raw     toggle the brain's OWN renderer (no LLM) -- raw recalled triple
  /help    show this help
  /quit    exit  (also /exit, /q, Ctrl-D)
"""


def _print_facts(chat):
    facts = chat.list_facts()
    if not facts:
        print("  (the brain knows no facts.)", flush=True)
        return
    print(f"  the brain knows {len(facts)} facts:", flush=True)
    for a, v, p in facts:
        print(f"    - {a} {v} {p}", flush=True)


def run_repl(chat, source, n_facts):
    rname = chat.renderer.name if chat.renderer is not None else "(none -- raw brain triples)"
    print(_BANNER.format(source=source, n_facts=n_facts, renderer=rname), flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[tui] bye.", flush=True)
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "/q", "quit", "exit"):
            print("[tui] bye.", flush=True)
            break
        if low in ("/help", "help", "?"):
            print(_HELP, flush=True)
            continue
        if low == "/facts":
            _print_facts(chat)
            continue
        if low == "/raw":
            chat.raw_mode = not chat.raw_mode
            print(f"  [raw mode {'ON -- the brain speaks its own raw triples (no LLM)' if chat.raw_mode else 'OFF -- fluent rendering'}]",
                  flush=True)
            continue
        if chat.verbose_thinking and chat.renderer is not None and not chat.raw_mode:
            print("  brain> thinking...", flush=True)
        ans, abstained = chat.answer(line)
        tag = "  (abstained -- the moat)" if abstained else ""
        print(f"brain> {ans}{tag}\n", flush=True)


# ============================================================================================================
# The GPU-FREE scripted SMOKE.
# ============================================================================================================

def run_smoke(chat, source, n_facts, out_path):
    """Scripted multi-turn turns (incl. anaphora + abstention + self-reference) on the tiny CPU brain with the
    template-stub renderer. Verifies the TUI loads + converses + the moat abstains + multi-turn anaphora works."""
    # the scripted multi-turn conversation. Each entry: (utterance, expectation-kind).
    # 'anaphora' uses the prior turn's referent; 'abstain' must hit the moat; 'self' is a self-reference question.
    script = [
        ("what are you", "answer"),              # self-reference: 'you' -> brain ('brain uses spikes')
        ("how do you learn", "answer"),          # self-reference synonym: learn -> learns ('brain learns words')
        ("what does the brain store", "answer"),  # direct self-fact ('brain store memory')
        ("what does the dog chase", "answer"),   # object fact -> the answer 'cat' is a chainable referent -> WM
        ("what does it eat", "anaphora"),        # anaphora: 'it' -> cat (the dog's chase-object) -> 'fish'
        ("what does the dragon do", "abstain"),  # untaught subject -> the moat abstains
        ("who wrote romeo and juliet", "abstain"),  # general knowledge never taught -> abstain (the firewall)
        ("what is the capital of france", "abstain"),  # Qwen knows this; the brain must NOT (firewall)
    ]
    transcript = []
    for utterance, kind in script:
        gate_svo = chat.gate(utterance)          # peek the gate so the transcript records what the brain recalled
        ans, abstained = (chat.answer(utterance) if gate_svo is None
                          else (chat.render(gate_svo), False))
        transcript.append({"you": utterance, "kind": kind, "gate_svo": gate_svo,
                           "brain": ans, "abstained": abstained})

    # checks
    self_q = transcript[0]
    self_answered = (not self_q["abstained"]) and self_q["gate_svo"] is not None and self_q["gate_svo"][0] == "brain"
    learn_q = next((t for t in transcript if t["you"] == "how do you learn"), None)
    learn_answered = bool(learn_q and not learn_q["abstained"]
                          and learn_q["gate_svo"] is not None and learn_q["gate_svo"][1] == "learn")
    # anaphora (RIGOROUS): the 'what does it eat' turn must have RESOLVED 'it' to the EXACT prior referent ('cat',
    # the dog's chase-object) AND answered the cat-eat-fish fact. A resolution to anything but 'cat', or an
    # abstention, FAILS -- so a spurious WM read cannot pass.
    anaphora_turn = next(t for t in transcript if t["you"] == "what does it eat")
    resolved_to = chat._resolve_anaphora("what does it eat")
    anaphora_resolved = (("cat" in resolved_to.split()) and ("it" not in resolved_to.split())
                         and (not anaphora_turn["abstained"])
                         and anaphora_turn["gate_svo"] == ["cat", "eat", "fish"])
    # abstention turns must abstain (the moat)
    abstain_turns = [t for t in transcript if t["kind"] == "abstain"]
    moat_held = all(t["abstained"] for t in abstain_turns)
    # at least the self + object facts answered (the brain converses)
    answered = [t for t in transcript if t["kind"] == "answer" and not t["abstained"]]
    converses = len(answered) >= 3

    go = bool(self_answered and learn_answered and anaphora_resolved and moat_held and converses)

    verdict = (
        f"GO -- the TUI loads a saved/tiny brain + holds a multi-turn conversation: self-reference resolves "
        f"('what are you' -> {self_q['gate_svo']}), learn-synonym resolves to the 'brain learn words' fact, "
        f"multi-turn anaphora binds 'it' -> {resolved_to!r} (the dog's chase-object 'cat') and answers "
        f"['cat','eat','fish'], the no-confab moat abstains on all {len(abstain_turns)} untaught/general cues "
        f"(incl. 'capital of France' the LLM knows but the brain must not), and {len(answered)} fact turns "
        f"answered. Renderer={chat.renderer.name if chat.renderer else 'raw'}. READY for the owner to --load the "
        f"real developed brain (with the off-bridge Qwen renderer)."
        if go else
        f"PARTIAL/SNAG -- self_answered={self_answered} learn_answered={learn_answered} "
        f"anaphora_resolved={anaphora_resolved} (resolved={resolved_to!r}) moat_held={moat_held} "
        f"converses={converses} ({len(answered)} fact turns). See the transcript for the localize."
    )

    res = {
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "source": source,
        "renderer": (chat.renderer.name if chat.renderer is not None else "raw brain triples"),
        "n_facts": n_facts,
        "self_reference_answered": self_answered,
        "learn_synonym_answered": learn_answered,
        "multiturn_anaphora_resolved": anaphora_resolved,
        "anaphora_resolved_to": resolved_to,
        "moat_held": moat_held,
        "n_abstain_turns": len(abstain_turns),
        "n_answer_turns": len(answered),
        "converses": converses,
        "transcript": transcript,
        "tui_features": [
            "load a developed brain (codes + facts + vocab) from a developed_brain_io bundle, OR the self-knowledge "
            "codes+curriculum, OR a tiny CPU fallback",
            "multi-turn chat: GATE (recall + abstain) -> CONSTRAIN+VERIFY fluent render (off-bridge Qwen default; "
            "template-stub for the GPU-free smoke) -> answer or 'I don't know about that.'",
            "multi-turn anaphora (it/that/they -> the prior referent via the MultiTurnAgent discourse WM)",
            "self-reference (you/your/I/me/it -> the brain) so 'what are you' / 'how do you learn' resolve",
            "commands: /raw (brain's own renderer, no LLM), /facts (list knowledge), /help, /quit",
            "the no-confab moat: the brain abstains on anything it was not taught (verified at the recall layer)",
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # print the transcript
    print("\n" + "=" * 90, flush=True)
    print("[tui SMOKE] scripted multi-turn transcript:", flush=True)
    print("=" * 90, flush=True)
    for t in transcript:
        gate = "" if t["gate_svo"] is None else f"   (recalled: {t['gate_svo']})"
        atag = "  [ABSTAIN]" if t["abstained"] else ""
        print(f"  you>   {t['you']}", flush=True)
        print(f"  brain> {t['brain']}{atag}{gate}", flush=True)
    print("=" * 90, flush=True)
    print(f"[tui SMOKE] VERDICT: {verdict}", flush=True)
    print(f"[tui SMOKE] saved {os.path.relpath(out_path, _REPO)}", flush=True)
    return res


# ============================================================================================================
# main.
# ============================================================================================================

def main():
    ap = argparse.ArgumentParser(description="Talk to a developed/trained brain (multi-turn).")
    ap.add_argument("--load", default=None,
                    help="a developed-brain bundle DIR (brain.json+...) OR a grounded-codes .json (self-knowledge).")
    ap.add_argument("--self-knowledge", action="store_true",
                    help="load the self-knowledge brain (default codes + curriculum).")
    ap.add_argument("--curriculum", default=None,
                    help="curriculum .json for the self-knowledge brain (default: _curriculum_self_knowledge.json).")
    ap.add_argument("--tiny-demo", action="store_true",
                    help="build a tiny CPU brain from a handful of facts (GPU-free fallback / smoke).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-multiturn", action="store_true",
                    help="use the bare BrainConversationalAgent (no discourse WM / anaphora).")
    ap.add_argument("--neural-render", action="store_true",
                    help="enable the brain's own spiking serial-order renderer (slow).")
    # renderer
    ap.add_argument("--stub-renderer", action="store_true",
                    help="use the GPU-FREE template-stub renderer (the CPU smoke); default is the off-bridge Qwen.")
    ap.add_argument("--no-renderer", action="store_true",
                    help="no fluent renderer (the brain speaks its own raw triples).")
    ap.add_argument("--T", type=int, default=16, help="off-bridge Qwen rate-code pool budget (16=GO).")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="Qwen surface-form length cap.")
    # smoke
    ap.add_argument("--smoke", action="store_true",
                    help="run the scripted GPU-FREE smoke (no interactive input) + write the JSON verdict.")
    ap.add_argument("--out", default="research/findings/raw/_brain_chat_tui_smoke.json",
                    help="smoke JSON output path.")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    # load the brain
    agent, aliases, n_facts, source = load_brain(a)
    # build the renderer (the smoke forces the stub if neither flag set)
    if a.smoke and not a.stub_renderer and not a.no_renderer:
        a.stub_renderer = True   # the GPU-free smoke uses the template-stub by default
    renderer = build_renderer(a)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=renderer)

    if a.smoke:
        res = run_smoke(chat, source, n_facts, os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out)
        return 0 if res["go"] else 1

    run_repl(chat, source, n_facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
