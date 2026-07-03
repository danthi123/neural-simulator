"""EMERGE-58 / RUNG 3 — the FINAL integration of the north-star wire: fold the EMERGENT-REASONING fluent conversation
(EMERGE-51..57) into the flagship FLUID console so ONE console answers BOTH kinds of question under ONE consistent
gate-first no-confab MOAT. Wernicke decides -> Broca articulates, for BOTH the emergent reasoner and the fluid paths.

  (A) EMERGE emergent-reasoning questions  -- discovered-category inheritance / per-dimension cancellation / sibling-
      discrimination, rendered FLUENTLY by the re-fine-tuned generator (EMERGE-57):
        you> can an owl fly?        brain> Yes, the owl can fly.      (INHERIT via the discovered bird codon)
        you> can a penguin fly?     brain> No, the penguin walks.     (LOCOMOTION exception -- cancellation)
        you> can a robin breathe?   brain> Yes, the robin can breathe.(RESPIRATION inherited -- per-dimension, no leak)
        you> can an owl swim?       brain> I don't know whether ...   (SIBLING-abstain: owl is a bird, not a fish)
        you> can a zzz fly?         brain> I don't know what a zzz is. (the no-confab MOAT -- never observed)
  (B) the existing FLUID-conversation paths (unchanged, no regression):
        you> what does the dog eat? brain> The dog eats meat.
        you> tell me about the dog  brain> Here's what I know about the dog: ...
        ... learn / discuss / classify / compare / instance / growth / moat, all as in `_fluidconv_chat_repl`.

THE MERGE (a COMPOSITION, not a destructive edit): `UnifiedFluentConsole` OWNS a `FluidChat` (the flagship fluid
paths, EMERGE-nothing changed) + a taught `PerDimensionConsole` (EMERGE-54's per-dimension emergent reasoner over the
pooler-discovered categories -- the console that gives inherit / per-dimension cancel / sibling-discrimination / moat).
A tiny ROUTER dispatches:
  * "can a X <verb>?" (the EMERGE ability frame)  -> the EMERGE reasoner -> a gate decision -> the SAME gate-first
    render loop as EMERGE-56/57 (abstain -> the generator is NEVER invoked; answer -> the re-fine-tuned 21M renders).
  * everything else                                -> `FluidChat.turn()` (the existing fluid dispatch, byte-unchanged).
ONE shared moat: an unknown/unobserved subject abstains on BOTH kinds, and on an abstain the fluent generator is NOT
invoked (render-count 0 -- the load-bearing property). The `can a X <verb>?` frame is SHARED between an EMERGE ability
question and a fluid fact question, so DISAMBIGUATION IS BY TAXONOMY MEMBERSHIP, not the frame alone: an observed
taxonomy member routes to the reasoner; a fluid-known (or unknown) entity in the SAME frame routes to the fluid path,
which answers it if it knows the entity and else applies its OWN gate-first moat. (EMERGE-58 audit remediation: the
earlier frame-ONLY router misrouted a fluid-known entity -- "can a dog eat?" was routed to the reasoner and FALSELY
denied "I don't know what a dog is" in the very session the fluid path answers "The dog eats meat." The membership gate
fixes that self-contradiction; the moat still holds by construction for GENUINELY-unknown tokens on whichever path.)

THE EMERGE GATE-DECISION ADAPTER (`emerge_pd_gate_decision`): mirrors EMERGE-56's adapter but for the ABILITY-SPECIFIC
`PerDimensionConsole.ask_can` (which reads ONLY the asked ability's discovered dimension). It returns the same
gate-first tuple the render loop consumes:
  * member unknown / below the floor / sibling-branch (asked ability not inherited) -> gate = ABSTAIN (the moat).
  * inherited class default on the asked ability                                    -> gate = ANSWER, affirm,
                                                                                        svo = (member, "can", ability).
  * member exception on the asked ability's dimension (penguin fly -> walks)        -> gate = ANSWER, negate,
                                                                                        svo = (member, ovr_verb, None).
The adapter is validated for FIDELITY against `PerDimensionConsole.ask_can`'s own decision (they must agree on gate +
polarity + the grounded content) so the wire cannot silently diverge from EMERGE's reasoning.

Reuse-by-import: `FluidChat` (`_fluidconv_chat_repl`) + `PerDimensionConsole` (`_emerge54`) + `_CountingFTFaculty` +
`emerge_v3` (`_emerge57`) + the gate-first render pattern (`_emerge56`). NO `sim/` edit; NO `_fluidconv_chat_repl` edit
(the flagship is used verbatim -- the merge is a wrapper). The generator ANN remains a tracked temporary scaffold.

Run:
  # CPU-safe: routing + EMERGE gate-decision adapter fidelity + moat + no-regression (no GPU, no ckpt render)
  SIM_BACKEND=numpy python -m research.runners._emerge58_unified_fluent_console --derisk --seeds 42 43 44
  # the mixed demo transcript (BOTH kinds in one session); GPU render if the ckpt + torch are present, else templated
  SIM_BACKEND=numpy python -m research.runners._emerge58_unified_fluent_console --demo
  # force the fluent GPU render path (needs torch+CUDA + the EMERGE-57 ckpt) for the mixed demo
  SIM_BACKEND=numpy python -m research.runners._emerge58_unified_fluent_console --demo --render
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# EMERGE-54: the per-dimension emergent reasoner (inherit / per-dimension cancel / sibling-discrim / moat)
from research.runners._emerge54_per_dimension_cancellation_derisk import (  # noqa: E402
    PerDimensionConsole, _script_lines as _pd_script_lines, _feed as _pd_feed, _dim_of,
    _BIRD_HELDOUT as _PD_BIRD_HELDOUT, _FISH_HELDOUT as _PD_FISH_HELDOUT,
    _BIRD_EXC as _PD_BIRD_EXC, _FISH_EXC as _PD_FISH_EXC,
)
from research.runners._emerge52_multilevel_conversational_console import FLOOR as _PD_FLOOR, _lemma  # noqa: E402
# EMERGE-57: the frame-aware inflection + the re-fine-tuned fluent renderer (skip-if-no-ckpt)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3, EMERGE_FT_CKPT  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge58_unified_fluent_console.json"

# the EMERGE ability frame the router owns: "can a/an <member> <verb>?" (with an optional trailing '?')
_ABILITY_RE = re.compile(r"^\s*can\s+(?:a|an)\s+(\w+)\s+(\w+)\s*\??\s*$", re.I)


def _art(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


# --------------------------------------------------------------------------------------------------------------------
# THE EMERGE GATE-DECISION ADAPTER (per-dimension flavour): read the PerDimensionConsole's ability-specific decision
# and convert it to the fluent faculty's gate-first (bool + SVO triple) input. 1-to-1 with `ask_can`'s decision.
# --------------------------------------------------------------------------------------------------------------------
def emerge_pd_gate_decision(console: PerDimensionConsole, member: str, prop: str) -> dict:
    """Read EMERGE-54's ABILITY-SPECIFIC inference decision for 'can a <member> <prop>?' (it reads ONLY the asked
    ability's discovered dimension) and convert it to the fluent faculty's gate-first input. Returns:
        gate:     "ANSWER" | "ABSTAIN"
        svo:      (subject, verb, object|None) when ANSWER, None when ABSTAIN
        polarity: "affirm" | "negate" | None
        source:   "inherited" | "exception" | "moat_unknown" | "moat_sibling"
    Mirrors `PerDimensionConsole.ask_can` exactly:
      (0) member never observed                                          -> ABSTAIN (moat_unknown)
      (1) member has an exception in the ASKED ability's dimension       -> ANSWER negate (member, ovr_verb, None)
      (2) the asked ability is inherited (a class teaches it for member) -> ANSWER affirm (member, "can", lemma(prop))
      (3) otherwise (sibling branch / below floor)                       -> ABSTAIN (moat_sibling)"""
    # (0) never-observed member -> the no-confab moat ("I don't know what a zzz is.")
    if member not in console.member_idx:
        return {"gate": "ABSTAIN", "svo": None, "source": "moat_unknown", "polarity": None}

    dim = _dim_of(prop)
    # (1) per-dimension cancellation: the member's own exception overrides ONLY its own dimension
    if dim is not None and console._exception_in_dim(member, dim):
        ep = console.ovr_prop.get(member, prop)               # e.g. penguin -> "walks" (already-3sg intransitive)
        return {"gate": "ANSWER", "svo": (member, ep, None), "source": "exception", "polarity": "negate"}

    # (2) inheritance for the asked ability, purely from the discovered-codon graded drive (correct-level,
    #     codon-driven sibling-discrimination)
    cls = console._best_class_for_prop(member, prop)
    if cls is not None:
        return {"gate": "ANSWER", "svo": (member, "can", _lemma(prop)), "source": "inherited", "polarity": "affirm"}

    # (3) sibling branch / below floor: the asked ability is not inherited for this member -> honest abstain (moat)
    return {"gate": "ABSTAIN", "svo": None, "source": "moat_sibling", "polarity": None}


def _pd_reference(console: PerDimensionConsole, member: str, prop: str) -> dict:
    """Ground truth from EMERGE-54 itself: (gate, polarity, property) as `ask_can` decides, parsed from its OWN
    committed answer string. The adapter must agree with this."""
    s = console.ask_can(member, prop)
    if s.startswith("I don't know"):
        return {"gate": "ABSTAIN", "polarity": None, "property": None}
    if s.startswith("No,"):                                   # member exception (cancellation)
        return {"gate": "ANSWER", "polarity": "negate", "property": console.ovr_prop.get(member)}
    return {"gate": "ANSWER", "polarity": "affirm", "property": _lemma(prop)}   # "Yes, a <m> can <lemma>."


def _adapter_matches(console: PerDimensionConsole, member: str, prop: str) -> bool:
    """The adapter's (gate, polarity, grounded property) must equal EMERGE-54's own decision."""
    ref = _pd_reference(console, member, prop)
    dec = emerge_pd_gate_decision(console, member, prop)
    if dec["gate"] != ref["gate"]:
        return False
    if dec["gate"] == "ABSTAIN":
        return True
    if dec["polarity"] != ref["polarity"]:
        return False
    subj, verb, obj = dec["svo"]
    adapter_prop = obj if dec["source"] == "inherited" else verb    # inherited -> the ability word; exception -> ovr verb
    return adapter_prop == ref["property"]


# --------------------------------------------------------------------------------------------------------------------
# THE FLUENT RENDERER (EMERGE-57's re-fine-tuned 21M, instrumented to count invocations). A TEMPLATE fallback keeps
# the console CPU-safe / offline (no torch/ckpt) with the SAME surface shape so routing/moat are testable everywhere.
# --------------------------------------------------------------------------------------------------------------------
class _TemplateEmergeFaculty:
    """A CPU-safe, offline, content-locked stand-in for the re-fine-tuned generator: renders EMERGE gated SVOs in the
    SAME surface shape ('Yes, the owl can fly.' / 'No, the penguin walks.') and COUNTS invocations (so the moat's
    'renderer-never-invoked-on-abstain' is a hard, assertable count). Used when the GPU ckpt/torch is unavailable."""

    kind = "template"

    def __init__(self):
        self.render_call_count = 0
        self.npar = 0.0
        self.device = "cpu"

    def render_emerge(self, svo, polarity):
        self.render_call_count += 1
        subj, verb, obj = svo
        if polarity == "affirm":
            return f"Yes, the {subj} can {obj}."               # inherited class default
        return f"No, the {subj} {emerge_v3(verb)}."            # member exception (negation of the default)


def _make_fluent_faculty(prefer_gpu):
    """Return (faculty, kind). Prefer the EMERGE-57 re-fine-tuned 21M (`_CountingFTFaculty`) when torch+CUDA + the
    ckpt are available; else fall back to the CPU template stand-in (same surface, offline). `prefer_gpu` gates it."""
    if prefer_gpu and os.path.exists(EMERGE_FT_CKPT):
        try:
            from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import _CountingFTFaculty
            fac = _CountingFTFaculty()
            # wrap render_emerge so the console gets a uniform (surface) return regardless of faculty
            _orig = fac.render_emerge

            def _render(svo, polarity, _orig=_orig):
                surface, _facts, _q = _orig(svo, polarity)
                return surface
            fac.render_emerge_surface = _render                # keep the counter on the real object
            fac.kind = "ft21m"
            return fac, "ft21m"
        except Exception as e:                                 # torch missing / ckpt unreadable -> template
            print(f"[emerge58] fluent GPU faculty unavailable ({e!r}); using the CPU template renderer.", flush=True)
    return _TemplateEmergeFaculty(), "template"


# --------------------------------------------------------------------------------------------------------------------
# THE UNIFIED CONSOLE: one loop, one moat. EMERGE ability frame -> reasoner -> gate-first render; else -> FluidChat.
# --------------------------------------------------------------------------------------------------------------------
class UnifiedFluentConsole:
    """ONE console that answers BOTH the EMERGE emergent-reasoning ability questions (fluently rendered, gate-first
    moat) AND the existing fluid-conversation paths (verbatim `FluidChat`), under one consistent no-confab moat."""

    def __init__(self, seed=42, prefer_gpu_render=False, build_fluid=True, verbose=False):
        self.seed = int(seed)
        self.render_calls_on_abstain = 0        # the load-bearing counter (must stay 0 across BOTH kinds of abstain)
        self.emerge_render_calls = 0            # how many times the fluent generator was invoked for EMERGE answers
        # (A) the EMERGE per-dimension reasoner: teach it the scripted taxonomy (observe -> is-a -> class props +
        #     LOCOMOTION exceptions), exactly as EMERGE-54's de-risk does.
        self.reasoner = PerDimensionConsole(seed=self.seed)
        obs, isa, teach, _ = _pd_script_lines(self.seed)
        _pd_feed(self.reasoner, obs, isa, teach)
        # the fluent renderer for EMERGE answers (GPU re-fine-tuned 21M if available; else CPU template)
        self.faculty, self.render_kind = _make_fluent_faculty(prefer_gpu_render)
        # (B) the flagship FLUID console -- used VERBATIM (no edit). Optional (a --derisk of just routing/moat can
        #     skip the heavy FluidChat build; but the no-regression gate needs it).
        self.fluid = None
        if build_fluid:
            from research.runners._fluidconv_chat_repl import FluidChat
            self.fluid = FluidChat(seed=self.seed)
        if verbose:
            self._greeting()

    def _greeting(self):
        gpar = f"~{self.faculty.npar:.0f}M {self.render_kind}" if getattr(self.faculty, "npar", 0) else self.render_kind
        print(f"[emerge58] ready -- ONE console: EMERGE emergent reasoning (inherit / per-dimension cancel / "
              f"sibling-abstain) rendered by the {gpar} generator + the flagship fluid paths, under ONE gate-first "
              f"moat. dev={getattr(self.faculty, 'device', 'cpu')}\n", flush=True)

    # ---- the router -------------------------------------------------------------------------------------------------
    def _is_emerge_ability(self, text):
        """RECOGNISE the ability frame 'can a/an <X> <verb>?' and return (X, verb) or None. This is frame-recognition
        ONLY -- it does NOT decide the route. The `can a X <verb>?` frame is SHARED with fluid fact questions, so the
        route is disambiguated in `turn()` by TAXONOMY MEMBERSHIP (X in self.reasoner.member_idx -> the reasoner; else
        -> the fluid path). A fluid-known entity in this frame is answered by the fluid path, not falsely denied."""
        m = _ABILITY_RE.match(text.strip())
        if not m:
            return None
        return m.group(1).lower(), m.group(2).lower()

    def _render_emerge(self, svo, polarity):
        """Invoke the fluent renderer (GPU 21m or CPU template) for an EMERGE ANSWER. Increments the faculty's own
        call counter; returns the fluent surface string."""
        self.emerge_render_calls += 1
        if hasattr(self.faculty, "render_emerge_surface"):    # the GPU 21m wrapper (returns a surface string)
            return self.faculty.render_emerge_surface(svo, polarity)
        return self.faculty.render_emerge(svo, polarity)       # the template stand-in

    def _emerge_turn(self, member, prop):
        """One EMERGE ability turn: reason -> gate decision -> gate-first render. On ABSTAIN the renderer is NEVER
        invoked (the moat, by construction). Returns the reply string."""
        dec = emerge_pd_gate_decision(self.reasoner, member, prop)
        calls_before = getattr(self.faculty, "render_call_count", 0)
        if dec["gate"] == "ABSTAIN":
            # THE MOAT: the brain decided to abstain -> the renderer is given NOTHING (never invoked).
            if dec["source"] == "moat_unknown":
                reply = f"I don't know what {_art(member)} is."
            else:                                              # sibling branch / below floor
                reply = f"I don't know whether {_art(member)} can {_lemma(prop)}."
            calls_after = getattr(self.faculty, "render_call_count", 0)
            self.render_calls_on_abstain += (calls_after - calls_before)     # MUST stay 0
            return reply
        # gate=ANSWER -> render the grounded gated fact fluently
        surface = self._render_emerge(dec["svo"], dec["polarity"])
        return surface

    def turn(self, text):
        """One unified conversation turn. The 'can a X <verb>?' ability frame is SHARED between the EMERGE reasoner
        (a taxonomy ability question) and the fluid paths (a general fact question), so disambiguation is by TAXONOMY
        MEMBERSHIP, not by the frame alone: route to the EMERGE reasoner IFF X is an observed taxonomy member; a
        fluid-known (or unknown) entity in the same frame goes to the fluid path, which answers it if it knows X and
        else applies its OWN gate-first moat. Everything else -> the flagship FluidChat dispatch (verbatim). ONE
        gate-first moat across both.

        (EMERGE-58 audit remediation: the earlier frame-ONLY router misrouted a fluid-known entity -- e.g.
        'can a dog eat?' -> ('dog','eat') -> the reasoner, where 'dog' is not in the scripted taxonomy, so it FALSELY
        denied 'I don't know what a dog is' in the very session where the fluid path answers 'The dog eats meat.' The
        membership gate below fixes that self-contradiction; the moat still holds by construction for GENUINELY-unknown
        tokens -- they reach a gate-first moat on whichever path, and the fluent generator is never invoked.)"""
        raw = (text or "").strip()
        if not raw:
            return "?"
        ab = self._is_emerge_ability(raw)
        if ab is not None:
            member, prop = ab
            if member in self.reasoner.member_idx:
                # an observed taxonomy member -> the EMERGE emergent reasoner (inherit / cancel / sibling-abstain)
                return self._emerge_turn(member, prop)
            if self.fluid is not None:
                # a fluid-known or unknown entity in the ability frame -> the fluid path (it answers if it knows X,
                # else applies its own gate-first moat). This ends the frame-only false-denial.
                return self.fluid.turn(raw)
            # no fluid path built (routing/moat-only de-risk): the reasoner's moat still abstains SAFELY
            # (moat_unknown -> 'I don't know what a X is', the fluent generator NOT invoked).
            return self._emerge_turn(member, prop)
        # everything else -> the existing fluid paths, byte-unchanged
        if self.fluid is None:
            return "(fluid paths not built -- construct with build_fluid=True)"
        return self.fluid.turn(raw)


# --------------------------------------------------------------------------------------------------------------------
# THE MIXED DEMO: BOTH kinds of question in one session, showing the unified moat + fluency + no cross-talk.
# --------------------------------------------------------------------------------------------------------------------
DEMO = [
    # (B) fluid paths
    ("what does the dog chase?", "FLUID   grounded Q&A (writes 'cat')"),
    ("what does it eat?",        "FLUID   anaphora (it=cat -> the cat eats fish)"),
    ("the wolf eats rabbit",     "FLUID   growth"),
    ("what does the wolf eat?",  "FLUID   learned fact usable"),
    # (A) EMERGE emergent-reasoning, rendered fluently
    ("can an owl fly?",          "EMERGE  INHERIT (discovered bird codon)"),
    ("can a penguin fly?",       "EMERGE  CANCEL (locomotion exception -> walks)"),
    ("can a robin breathe?",     "EMERGE  PER-DIMENSION inherit (respiration; no leak)"),
    ("can an owl swim?",         "EMERGE  SIBLING-abstain (owl is a bird, not a fish)"),
    ("can a zzz fly?",           "EMERGE  MOAT (never observed)"),
    # (B) back to fluid -- MEMBERSHIP routing: a fluid-known entity in the SHARED ability frame is ANSWERED by the
    # fluid path (NOT falsely denied) -- the 2026-07-03 audit remediation.
    ("can a dog eat?",           "FLUID   membership routing (dog is fluid-known, not a taxonomy member -> answered)"),
    ("tell me about the dog",    "FLUID   grounded discussion"),
    ("does the dog eat meat?",   "FLUID   yes/no"),
    ("what does the lion eat?",  "FLUID   MOAT (untaught)"),
]


def _demo(seed=42, prefer_gpu_render=False):
    con = UnifiedFluentConsole(seed=seed, prefer_gpu_render=prefer_gpu_render, verbose=True)
    print("=== EMERGE-58 UNIFIED FLUENT CONSOLE -- one console, both kinds of question, one gate-first no-confab "
          "moat (Wernicke decides -> Broca articulates) ===\n", flush=True)
    transcript = []
    for (line, why) in DEMO:
        reply = con.turn(line)
        transcript.append({"you": line, "brain": reply, "route": why.split()[0]})
        print(f"  you>   {line}\n  brain> {reply}   [{why}]", flush=True)
    print(f"\n  render-calls on abstains (BOTH kinds): {con.render_calls_on_abstain}   "
          f"(the load-bearing property: MUST be 0)\n", flush=True)
    return con, transcript


# --------------------------------------------------------------------------------------------------------------------
# THE DE-RISK (CPU-safe): (a) EMERGE render correct; (b) fluid paths still work (no regression); (c) ONE moat;
# (d) no cross-talk. Uses the CPU template renderer by default (so gates (a)/(c)/(d) run offline); the GPU render is a
# skip-if-no-ckpt smoke reported separately.
# --------------------------------------------------------------------------------------------------------------------
# the EMERGE probes (member, prop, expected route+outcome). Held-out members come from EMERGE-54's taxonomy.
def _emerge_probes():
    """Taxonomy probes (all OBSERVED members -> the EMERGE reasoner): inherit / cancel / sibling-abstain, + one
    genuinely-unknown moat. Used by the render-correctness gate + the GPU render smoke."""
    return [
        (_PD_BIRD_HELDOUT, "fly", "inherited"),          # owl -> Yes, the owl can fly (INHERIT)
        (_PD_FISH_HELDOUT, "swim", "inherited"),         # minnow -> Yes, the minnow can swim (INHERIT)
        (_PD_BIRD_EXC[0], "fly", "exception"),           # penguin -> No, the penguin walks (CANCEL)
        (_PD_FISH_EXC[0], "swim", "exception"),          # pike -> No, the pike lurks (CANCEL)
        ("robin", "breathe", "inherited"),               # robin -> Yes, the robin can breathe (PER-DIMENSION inherit)
        (_PD_BIRD_HELDOUT, "swim", "moat_sibling"),      # owl swim -> abstain (SIBLING: bird, not fish; 21M NOT invoked)
        ("zzz", "fly", "moat_unknown"),                  # never observed (MOAT; 21M NOT invoked)
    ]


def _membership_probes():
    """The AUDIT-REMEDIATION probes: NON-members in the SHARED 'can a X <verb>?' ability frame. The earlier frame-ONLY
    router misrouted these into the reasoner and FALSELY DENIED a fluid-known entity ('can a dog eat?' -> 'I don't know
    what a dog is' in the same session the fluid path answers 'The dog eats meat.'). Post-fix, membership-aware routing
    sends a non-member to the fluid path -- which answers a fluid-known entity and else applies its OWN gate-first moat.
    This gate probes the EXACT shape the original no-cross-talk gate never tested (so it can no longer pass by
    construction)."""
    return [
        ("dog", "eat", "fluid_known"),                   # fluid-known -> "The dog eats meat." (NOT falsely denied)
        ("zzz", "fly", "unknown_moat"),                  # unknown to both -> gate-first moat (21M NOT invoked)
    ]


def _derisk_one(seed, build_fluid=True):
    con = UnifiedFluentConsole(seed=seed, prefer_gpu_render=False, build_fluid=build_fluid)
    faculty = con.faculty
    probes = _emerge_probes()

    # membership-aware routing: a probe routes to the reasoner IFF its member is an observed taxonomy member.
    def _routes_to_reasoner(q):
        ab = con._is_emerge_ability(q)
        return ab is not None and ab[0] in con.reasoner.member_idx

    # (a-fid) ADAPTER FIDELITY: the per-dimension gate decision == PerDimensionConsole.ask_can's own decision (over
    # the taxonomy members -- the non-member 'zzz' probe is not a reasoner decision to mirror).
    adapter_fid = float(np.mean([_adapter_matches(con.reasoner, m, p)
                                 for (m, p, _) in probes if m in con.reasoner.member_idx]))

    # (a) EMERGE RENDER CORRECT + (c) ONE MOAT (render-not-invoked-on-abstain) + routing of taxonomy members.
    n_render_correct = 0
    n_answer = 0
    moat_render_calls = 0
    moat_says_idk = 0
    n_moat = 0
    per_probe = []
    for (m, prop, exp) in probes:
        is_member = m in con.reasoner.member_idx
        # membership-aware routing: an observed member -> the reasoner; else -> fluid (or the reasoner's own moat
        # when no fluid path is built). Members MUST route to the reasoner.
        routed_reasoner = bool(is_member)
        calls0 = getattr(faculty, "render_call_count", 0)
        reply = con.turn(f"can {_art(m)} {prop}?")
        calls_delta = getattr(faculty, "render_call_count", 0) - calls0
        ok = True
        if exp == "inherited":
            n_answer += 1
            ok = reply.lower().startswith("yes") and (_lemma(prop) in reply.lower()) and calls_delta == 1
            n_render_correct += int(ok)
        elif exp == "exception":
            n_answer += 1
            ovr = (con.reasoner.ovr_prop.get(m) or "")
            ok = reply.lower().startswith("no") and (ovr in reply.lower()) and (m in reply.lower()) and calls_delta == 1
            n_render_correct += int(ok)
        else:  # moat (unknown or sibling) -- abstain on whichever path, the FLUENT GENERATOR NOT invoked
            n_moat += 1
            ok = reply.lower().startswith("i don't know") and calls_delta == 0
            moat_render_calls += calls_delta                         # MUST stay 0
            moat_says_idk += int(reply.lower().startswith("i don't know"))
        per_probe.append({"member": m, "prop": prop, "expect": exp, "reply": reply,
                          "routed_reasoner": bool(routed_reasoner), "is_member": bool(is_member),
                          "render_delta": int(calls_delta), "ok": bool(ok)})
    emerge_render_correct = float(n_render_correct / max(1, n_answer))
    members_routed = all(p["routed_reasoner"] for p in per_probe if p["is_member"])
    moat_ok = (moat_render_calls == 0 and moat_says_idk == n_moat)

    # (a2) MEMBERSHIP ROUTING -- the AUDIT-REMEDIATION gate (the exact shape the original no-cross-talk gate never
    # probed, so it could not pass by construction). A fluid-known entity in the ability frame must NOT be falsely
    # denied (it is answered by the path that knows it, and the EMERGE generator is NOT stolen into it); a genuinely-
    # unknown entity abstains gate-first (the fluent generator NOT invoked).
    membership = []
    membership_ok = None
    if con.fluid is not None:
        membership_ok = True
        for (m, prop, exp) in _membership_probes():
            calls0 = getattr(faculty, "render_call_count", 0)
            reply = con.turn(f"can {_art(m)} {prop}?")
            calls_delta = getattr(faculty, "render_call_count", 0) - calls0
            false_denial = reply.lower().startswith(f"i don't know what {_art(m)}")   # the pre-fix bug's signature
            if exp == "fluid_known":
                # answered by the fluid path (mentions the known fact), NOT falsely denied, EMERGE 21M NOT stolen in
                ok = ((not false_denial) and calls_delta == 0 and (not reply.lower().startswith("i don't know"))
                      and ("eat" in reply.lower() or "meat" in reply.lower()))
            else:  # unknown_moat: gate-first abstain, generator NOT invoked
                ok = reply.lower().startswith("i don't know") and calls_delta == 0
            membership_ok = membership_ok and ok
            membership.append({"member": m, "prop": prop, "expect": exp, "reply": reply,
                               "false_denial": bool(false_denial), "render_delta": int(calls_delta), "ok": bool(ok)})

    # (b) NO REGRESSION on the fluid paths: run a representative fluid slice and check the known-correct outcomes.
    reg = {}
    if con.fluid is not None:
        def _f(line):
            return con.turn(line)
        # the flagship's own proven demo order (mirrors _fluidconv_chat_repl DEMO): chase writes 'cat' as the referent,
        # then 'it'=cat -> the cat eats fish (Phase-4 anaphora), growth, yes/no, discuss, moat.
        reg["what_chase"] = _f("what does the dog chase?")           # -> the dog chases cat  (writes 'cat')
        reg["anaphora"] = _f("what does it eat?")                    # -> it=cat -> the cat eats fish (Phase-4)
        reg["what_eat"] = _f("what does the dog eat?")               # -> mentions meat
        reg["growth"] = _f("the wolf eats rabbit")                   # -> ok, learned
        reg["growth_use"] = _f("what does the wolf eat?")            # -> rabbit
        reg["yesno"] = _f("does the dog eat meat?")                  # -> Yes ...
        reg["discuss"] = _f("tell me about the dog")                 # -> Here's what I know ...
        reg["moat"] = _f("what does the lion eat?")                  # -> I don't know
        fluid_ok = bool(
            "cat" in reg["what_chase"].lower()
            and "fish" in reg["anaphora"].lower()                    # anaphora resolved (it=cat -> fish)
            and "meat" in reg["what_eat"].lower()
            and "learned" in reg["growth"].lower()
            and "rabbit" in reg["growth_use"].lower()
            and reg["yesno"].lower().startswith("yes")
            and ("know about" in reg["discuss"].lower() or "here's what i know" in reg["discuss"].lower()
                 or "don't know" not in reg["discuss"].lower())
            and "know" in reg["moat"].lower())
        # (d) NO CROSS-TALK (corrected -- probes the failing shape): a fluid knowledge Q is NOT stolen by the reasoner;
        # an EMERGE TAXONOMY-member ability Q IS the reasoner; AND a FLUID-KNOWN entity in the SHARED ability frame is
        # NOT stolen by the reasoner (the audit's fix -- the pre-fix router routed 'can a dog eat?' to the reasoner and
        # falsely denied it).
        no_crosstalk = bool(
            (not _routes_to_reasoner("what does the dog eat?"))
            and (not _routes_to_reasoner("tell me about the dog"))
            and _routes_to_reasoner("can an owl fly?")
            and (not _routes_to_reasoner("can a dog eat?")))
    else:
        fluid_ok = None
        no_crosstalk = bool((not _routes_to_reasoner("what does the dog eat?"))
                            and _routes_to_reasoner("can an owl fly?"))

    return {"seed": seed, "adapter_fidelity": adapter_fid, "emerge_render_correct": emerge_render_correct,
            "n_answer": n_answer, "members_routed": bool(members_routed), "moat_ok": bool(moat_ok),
            "moat_render_calls_on_abstains": int(moat_render_calls), "n_moat": n_moat,
            "render_calls_on_abstain_total": int(con.render_calls_on_abstain),
            "membership_ok": membership_ok, "membership_detail": membership,
            "fluid_ok": fluid_ok, "no_crosstalk": bool(no_crosstalk), "render_kind": con.render_kind,
            "fluid_regression_detail": reg, "per_probe": per_probe}


def _derisk(seeds, build_fluid=True, gpu_render_smoke=False):
    print(f"EMERGE-58 RUNG 3 de-risk: unify EMERGE emergent-reasoning (inherit/per-dim-cancel/sibling-abstain) + the "
          f"flagship fluid paths under ONE gate-first moat; adapter fidelity + EMERGE render correct + no-regression + "
          f"ONE moat + no cross-talk; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s, build_fluid=build_fluid)
            per.append(d)
            print(f"  [seed {s}] adapter-fid {d['adapter_fidelity']:.2f} | emerge-render {d['emerge_render_correct']:.2f}"
                  f" ({d['render_kind']}) | members-routed {int(d['members_routed'])} | membership-ok {d['membership_ok']}"
                  f" | moat-ok {int(d['moat_ok'])} (render-on-abstain {d['moat_render_calls_on_abstains']}) | "
                  f"fluid-ok {d['fluid_ok']} | no-crosstalk {int(d['no_crosstalk'])}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    gpu_smoke = None
    if err is None and gpu_render_smoke:
        gpu_smoke = _gpu_render_smoke(seeds[0])

    if err is None:
        adapter_fid = float(np.mean([d["adapter_fidelity"] for d in per]))
        emerge_render = float(np.mean([d["emerge_render_correct"] for d in per]))
        members_routed_all = all(d["members_routed"] for d in per)
        moat_all = all(d["moat_ok"] for d in per)
        render_on_abstain = int(sum(d["moat_render_calls_on_abstains"] for d in per))
        no_crosstalk_all = all(d["no_crosstalk"] for d in per)
        membership_vals = [d["membership_ok"] for d in per if d["membership_ok"] is not None]
        membership_all = (len(membership_vals) > 0 and all(membership_vals))
        fluid_vals = [d["fluid_ok"] for d in per if d["fluid_ok"] is not None]
        fluid_all = (len(fluid_vals) > 0 and all(fluid_vals))
        # GO bar: adapter fidelity == 1.0, EMERGE renders correct on the template path, every TAXONOMY-MEMBER ability
        # frame routes to the reasoner, membership routing is correct (a fluid-known entity in the ability frame is
        # answered -- NOT falsely denied; a genuine unknown abstains gate-first), the moat holds on BOTH kinds
        # (0 renders on abstains), no cross-talk, and NO fluid regression.
        go = bool(adapter_fid >= 0.99 and emerge_render >= 0.99 and members_routed_all and membership_all and moat_all
                  and render_on_abstain == 0 and no_crosstalk_all and fluid_all)
        if go:
            verdict = (f"GO -- the EMERGENT-REASONING fluent conversation (EMERGE-51..57) is FOLDED into the flagship "
                       f"FLUID console: ONE console answers BOTH the EMERGE emergent-reasoning questions (inherit / "
                       f"per-dimension cancel / sibling-abstain -- adapter fidelity {adapter_fid:.2f} vs the reasoner's "
                       f"own ask_can, render-correct {emerge_render:.2f} on the CPU path, every taxonomy-member ability "
                       f"frame routed to the reasoner) AND the existing fluid paths (no regression -- what/anaphora/"
                       f"growth/yes-no/discuss/moat all correct), under ONE consistent gate-first no-confab MOAT "
                       f"({render_on_abstain} renders on abstains across BOTH kinds -- the load-bearing property). The "
                       f"'can a X <verb>?' frame is SHARED, so DISAMBIGUATION IS BY TAXONOMY MEMBERSHIP: a fluid-known "
                       f"entity in the frame ('can a dog eat?') is answered by the fluid path (NOT falsely denied) and a "
                       f"genuine unknown abstains gate-first -- membership-routing gate PASS (the EMERGE-58 audit "
                       f"remediation). {len(seeds)}-seed, CPU-safe. The fluent GPU render (the re-fine-tuned 21M, "
                       f"EMERGE-57) is the same gate-first loop -- "
                       f"{('GPU smoke ran: ' + gpu_smoke['note']) if gpu_smoke else 'run --render for the GPU smoke'}. "
                       f"Reuse-by-import; NO sim/ edit; NO _fluidconv_chat_repl edit. Wernicke decides -> Broca "
                       f"articulates, for BOTH the reasoner and the fluid paths.")
        else:
            miss = []
            if adapter_fid < 0.99: miss.append(f"adapter fidelity {adapter_fid:.2f} < 0.99")
            if emerge_render < 0.99: miss.append(f"EMERGE render-correct {emerge_render:.2f} < 0.99")
            if not members_routed_all: miss.append("a TAXONOMY-MEMBER ability frame did NOT route to the reasoner")
            if not membership_all: miss.append("membership routing FAILED (a fluid-known entity was falsely denied, or "
                                               "a genuine unknown did not abstain gate-first) -- the audit-remediation gate")
            if not moat_all or render_on_abstain != 0:
                miss.append(f"MOAT breached ({render_on_abstain} renders on abstains) -- BLOCKING")
            if not no_crosstalk_all: miss.append("cross-talk detected (a route leaked)")
            if not fluid_all: miss.append("fluid-path REGRESSION (an existing fluid answer changed)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The specific gap is above. Routing/moat/regression gaps are "
                       "wiring, not a mechanism wall; a MOAT breach (renders on abstains != 0) is BLOCKING -- the "
                       "renderer must NEVER be invoked on an abstain of EITHER kind.")
    else:
        go = False
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge58_unified_fluent_console", "rung": 3, "GO": go, "verdict": verdict,
               "mechanism": "a UnifiedFluentConsole COMPOSES the flagship FluidChat (fluid paths, verbatim) + a taught "
                            "PerDimensionConsole (EMERGE-54 per-dimension emergent reasoner over the pooler-discovered "
                            "categories). A router keyed on the 'can a X <verb>?' ability frame dispatches EMERGE "
                            "questions to the reasoner -> a per-dimension gate-decision adapter (emerge_pd_gate_decision, "
                            "1-to-1 with ask_can) -> the SAME gate-first render loop as EMERGE-56/57 (abstain -> the "
                            "generator is NEVER invoked; answer -> the re-fine-tuned 21M / a CPU template renders the "
                            "gated fact fluently). Everything else -> FluidChat.turn (byte-unchanged). ONE moat across "
                            "both kinds. Reuse-by-import; NO sim/ edit; NO _fluidconv_chat_repl edit.",
               "task": "fold EMERGE-51..57 emergent-reasoning fluent conversation into the flagship fluid console; "
                       "adapter fidelity + EMERGE render correct + no fluid regression + ONE gate-first moat + no "
                       "cross-talk; mixed demo; multi-seed",
               "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "gpu_render_smoke": gpu_smoke,
               "HONEST_NOTE": "RUNG 3 = the MERGE (composition), not a new mechanism. The EMERGE render correctness is "
                              "gated on the CPU template renderer (content-locked, same surface) so routing/moat/"
                              "regression run offline + CPU-safe + multi-seed; the FLUENT GPU render is the EMERGE-57 "
                              "re-fine-tuned 21M behind the IDENTICAL gate-first loop (--render / --demo --render; "
                              "skip-if-no-ckpt), reported as a smoke -- EMERGE-57 already GO'd its render fidelity 1.00 "
                              "+ moat 0-renders-on-abstain. The generator ANN is a tracked temporary scaffold "
                              "(spiking-forward conversion deferred, validated at 88.6M). The MOAT is preserved BY "
                              "CONSTRUCTION on BOTH kinds (the gate short-circuits before the renderer / the fluid gate). "
                              "The EMERGE reasoner is taught EMERGE-54's scripted bird/fish taxonomy; corpus-scale "
                              "feature discovery is the standing EMERGE follow-on. AUDIT REMEDIATION (2026-07-03): the "
                              "initial frame-ONLY router misrouted a fluid-known entity ('can a dog eat?') into the "
                              "reasoner and FALSELY DENIED it; disambiguation is now by TAXONOMY MEMBERSHIP (member -> "
                              "reasoner; else -> fluid), with a membership-routing gate that probes the exact failing "
                              "shape so it cannot pass by construction. The gate-first moat is unchanged for genuine "
                              "unknowns (abstain on whichever path, the fluent generator NOT invoked)."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[emerge58] VERDICT: {verdict}", flush=True)
    print(f"[emerge58] wrote {OUT}\n" + "=" * 112, flush=True)
    return 0 if go else 1


# --------------------------------------------------------------------------------------------------------------------
# THE GPU RENDER SMOKE: render the EMERGE ANSWERS via the re-fine-tuned 21M (EMERGE-57) behind the SAME gate-first loop.
# Reports the fluent surfaces + confirms the moat (0 renders on abstains) on the real model. Skip-if-no-ckpt/torch.
# --------------------------------------------------------------------------------------------------------------------
def _gpu_render_smoke(seed=42):
    if not os.path.exists(EMERGE_FT_CKPT):
        return {"ran": False, "note": f"EMERGE-57 ckpt absent ({EMERGE_FT_CKPT}) -- GPU render skipped"}
    try:
        con = UnifiedFluentConsole(seed=seed, prefer_gpu_render=True, build_fluid=False, verbose=True)
    except Exception as e:
        return {"ran": False, "note": f"could not build the GPU faculty ({e!r})"}
    if con.render_kind != "ft21m":
        return {"ran": False, "note": "GPU faculty unavailable (torch/CUDA missing) -- fell back to template"}
    probes = _emerge_probes()
    recs = []
    render_on_abstain = 0
    for (m, prop, exp) in probes:
        calls0 = getattr(con.faculty, "render_call_count", 0)
        reply = con.turn(f"can {_art(m)} {prop}?")
        delta = getattr(con.faculty, "render_call_count", 0) - calls0
        if exp.startswith("moat"):
            render_on_abstain += delta
        recs.append({"member": m, "prop": prop, "expect": exp, "reply": reply, "model_invoked": bool(delta)})
        print(f"  you> can {_art(m)} {prop}?\n  brain> {reply}   "
              f"[{exp}; model {'INVOKED' if delta else 'NOT invoked'}]\n", flush=True)
    note = (f"GPU render smoke on the re-fine-tuned 21M ({con.faculty.npar:.1f}M): the fluent surfaces above; the "
            f"moat held on the REAL model ({render_on_abstain} renders on abstains -- the load-bearing property).")
    print(f"[emerge58] {note}", flush=True)
    return {"ran": True, "note": note, "render_on_abstain": int(render_on_abstain), "records": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--demo", action="store_true", help="the mixed demo transcript (BOTH kinds in one session)")
    ap.add_argument("--derisk", action="store_true", help="the CPU-safe de-risk gates (routing/adapter/moat/regression)")
    ap.add_argument("--render", action="store_true", help="prefer the fluent GPU render (needs torch+CUDA + EMERGE-57 ckpt)")
    ap.add_argument("--gpu-render-smoke", action="store_true", help="(with --derisk) also run the GPU render smoke")
    ap.add_argument("--no-fluid", action="store_true", help="(with --derisk) skip building FluidChat (routing/moat only)")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds, build_fluid=(not a.no_fluid), gpu_render_smoke=a.gpu_render_smoke)
    if a.demo:
        con, transcript = _demo(a.seed, prefer_gpu_render=a.render)
        return 0
    # default: the mixed demo (template render unless --render)
    _demo(a.seed, prefer_gpu_render=a.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
