"""Tier 2.5 -- PRODUCTION tense/aspect as a bound fact-tag (the temporal-representation slice of Tier 2).

WHAT THIS IS (honest scope)
  A bare SVO triple has no TIME: "boy go park" cannot say whether the boy WENT, GOES, or WILL GO. Tense is the
  cheapest temporal-representation step (front-4 #3 target; Levelt's positional level inserts tense; Hagoort MUC's
  closed-class scaffold). This module binds a TENSE role-tag (PAST / PRESENT / FUTURE, cleaned ONLY against a
  3-word codebook) onto each stored fact, reads it back at render time, and DRIVES the surface verb form:
    PAST -> "went" / PRESENT -> "goes" / FUTURE -> "will go".

  HONEST BOUNDARY (NOT claimed here): this is tense/aspect as a single 3-valued metadata TAG on a fact (the
  cheapest viable temporal layer). It is NOT a full event-semantics / Reichenbach reference-time / Davidsonian
  event-argument representation (perfect vs progressive aspect, relative tense, event chaining over a timeline).
  Those are the deferred temporal-reasoning frontier. PBWM gating (the OTHER half of front-4 #3 -- BG
  disinhibition / transmission_gate / DA-RPE re-targeted as a WM input gate, O'Reilly-Frank) is a SEPARATE control
  layer and is a deferred follow-on, NOT in this representation-only task.

THE MECHANISM IT COMPOSES (reuse-by-import -- the PROVEN polarity/negation + common-ground tag; NO research gate)
  RFPhasorComposer binds a POLARITY role (AFFIRM/NEGATE) onto a fact, cleaned ONLY against a 2-word tag codebook
  (rf_phasor_composer.py:159-162, 528-544, 802-822), and reads it back at query time (ask_yes_no). The just-built
  CommonGroundComposer (commit 43f6bda4) is the SAME bind-a-tag-on-a-role-cleaned-against-a-small-codebook pattern
  at a SHARED/PRIVATE role. Tense is that pattern AGAIN at a TENSE role: PAST/PRESENT/FUTURE cleaned only against
  `tense_words`. PLUS the no-confab moat: a never-stored fact reads tag=None (no fabricated tense); a render over an
  unknown subject returns None.

DESIGN (NO sim/ edit, NO existing-composer edit -- SUBCLASSES ArgStructureComposer):
  ArgStructureComposer already produces agreement morphology: its `_decode_unit_word` TENSE branch reads the verb
  from the RF unbind (`unbind(comp,'action')`) and applies the PRESENT-3sg inflection (`TENSE_3SG`). This module:
    * ADDS a TENSE role + a 3-word codebook `tense_words = ['PAST','PRESENT','FUTURE']` (drawn from a DISJOINT rng
      stream seed+5252 so the parent's concept/role codes stay byte-identical).
    * `store_tensed(fact, tense=...)`: encode the fact AND bind (TENSE (x) tense_tag) onto it (the polarity-tag
      pattern). The tag bind goes through the parent's RF `_bind` -> `_resonate`, so it is bound on the spiking
      resonate-and-fire substrate exactly as the validated polarity/common-ground tags are (NOT a host flag).
    * `read_tense(fact)`: the bound PAST/PRESENT/FUTURE tag of the first fact whose cue roles (agent+action[+the
      obliques present]) match, by the parent's RF unbind + cleanup against `tense_words`. None = abstain (moat).
    * OVERRIDES `_decode_unit_word`'s TENSE branch: READ the bound tense tag and inflect the verb ACCORDINGLY --
      PAST via an irregular/regular past table, PRESENT via the parent's TENSE_3SG, FUTURE via "will <bare verb>".
      So the bound tag DRIVES the rendered surface verb form. The render is otherwise the parent's frame path
      (FrameCQ orders the content slots; the closed-class scaffold supplies determiners/prepositions).

Biology: tense is the closed-class scaffold the positional level inserts (Bock & Levelt 1994; Garrett); a bound
metadata role is the project's validated precedent (polarity, common-ground). The 3-valued tag is the file-card's
temporal slot of Discourse Representation Theory; the surface inflection is the morphophonological-encoding host
polish (a legitimate lexical front-end, like the parser's morphology -- the brain renders the bare verb tag, the
inflection table adds the morpheme).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer,
    ALL_ROLES,
    TENSE_3SG,
    frame_for,
    realized_units,
)

# The tense tag codebook -- the temporal markers, cleaned ONLY against this 3-word list (NOT the main vocab),
# exactly as polarity is cleaned only against pol_words. The composer's binding is role-agnostic, so the tag is
# just three more concept-codebook entries on a dedicated role.
TENSE_WORDS = ("PAST", "PRESENT", "FUTURE")
TENSE_ROLE = "TENSE_TAG"

# A PAST-tense inflection table (irregular + regular -ed). Morphology = a legitimate lexical front-end (like the
# parser's morphology + the parent's present-tense TENSE_3SG). The brain renders the bare verb + the bound TENSE
# tag; this host polish adds the correct past morpheme.
TENSE_PAST = {
    "go": "went", "come": "came", "walk": "walked", "run": "ran", "give": "gave", "send": "sent",
    "put": "put", "chase": "chased", "eat": "ate", "see": "saw", "like": "liked", "have": "had",
    "make": "made", "take": "took", "find": "found", "look": "looked", "want": "wanted", "stop": "stopped",
}


def inflect(verb, tense):
    """Surface verb form for (bare verb, tense tag). PAST -> irregular/regular -ed; PRESENT -> 3sg (-s);
    FUTURE -> 'will <bare verb>'. Unknown verbs fall back to a regular rule (so the tag still DRIVES the form)."""
    t = (tense or "PRESENT").upper()
    if t == "PAST":
        return TENSE_PAST.get(verb, verb + "ed")
    if t == "FUTURE":
        return "will " + verb            # the auxiliary is the closed-class future marker; bare verb after it
    return TENSE_3SG.get(verb, verb)     # PRESENT (default)


class TenseAspectComposer(ArgStructureComposer):
    """ArgStructureComposer extended with a bound PAST/PRESENT/FUTURE TENSE tag that DRIVES the rendered verb form.

    The tag bind/unbind go through the parent's RF ops (_bind / _unbind_phases -> _resonate), so the tense tag is
    bound/read on the resonate-and-fire substrate -- exactly as the validated polarity (AFFIRM/NEGATE) and
    common-ground (SHARED/PRIVATE) tags are. On CuPy that runs on the GPU RF substrate; on NumPy the same RF
    dynamics loop runs on CPU (the == test-oracle path).

    The no-confab moat is preserved: a never-stored cue reads tense=None (abstain -- no fabricated tense), and the
    underlying who/what query abstains on an unstored (agent,action) cue. The default-tense store is byte-compatible
    with the parent's store_fact for facts that carry no tense (tense defaults to PRESENT)."""

    def __init__(self, seed=42, D=64, vocab=None, grounded_codes=None, framecq_seed=None):
        super().__init__(seed=seed, D=D, vocab=vocab, grounded_codes=grounded_codes, framecq_seed=framecq_seed)
        # ADD the TENSE role + its tag codebook to the parent's dicts (exactly as polarity adds AFFIRM/NEGATE and
        # common-ground adds SHARED/PRIVATE). Use a DISJOINT rng stream so the parent's codes stay byte-identical.
        prng = np.random.default_rng(seed + 5252)
        self.tense_words = list(TENSE_WORDS)
        if TENSE_ROLE not in self.roles:
            self.roles[TENSE_ROLE] = prng.uniform(0.0, 1.0, self.D)
        for tag in self.tense_words:
            if tag not in self.concepts:
                self.concepts[tag] = prng.uniform(0.0, 1.0, self.D)

    # --- storage: the polarity-tag pattern, on the TENSE role -------------------------------------------------
    def _encode_tensed(self, fact, tense):
        """Encode an argument-structure fact WITH a bound tense tag. Mirrors ArgStructureComposer._encode (binds
        every role present in ALL_ROLES) but appends one more bound role (TENSE (x) tag). The tag bind goes through
        the same _bind the parent uses, so it runs on the RF spiking substrate (the parent's _bind uses _resonate)."""
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ALL_ROLES if r in fact]
        bounds.append(self._bind(self.roles[TENSE_ROLE], self.concepts[tense]))
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_tensed(self, fact, tense="PRESENT"):
        """Store an argument-structure fact dict WITH a PAST/PRESENT/FUTURE tense tag bound onto it. `fact` is a
        dict over {agent, action, <typed roles>} (as ArgStructureComposer.store_fact expects); `tense` in
        {'PAST','PRESENT','FUTURE'}. The fact is stored in the parent's kb (so query_role/render/the moat all keep
        working unchanged); the tense is an EXTRA bound role read by read_tense + the render."""
        tn = str(tense).upper()
        if tn not in self.tense_words:
            raise ValueError(f"tense must be one of {self.tense_words}, got {tense!r}")
        fact = dict(fact)
        fact["_tense"] = tn       # host-side label on the fact-dict (the AUTHORITATIVE copy is the bound tag)
        comp = self._encode_tensed(fact, tn)
        self.kb.append((fact, self._store_substrate(comp) if self.enable_substrate_store else comp))

    # --- read-back: the bound tag, by RF unbind + cleanup against tense_words (the polarity-read pattern) ------
    def read_tense(self, fact):
        """The bound PAST/PRESENT/FUTURE tag of the FIRST stored fact whose CUE roles match, by the composer's RF
        unbind + cleanup against `tense_words`. The cue roles are agent + action + every oblique/typed role present
        in `fact` (so two facts with the same agent+action but different goals stay distinguishable). None if no
        fact matches (the no-confab moat -- a never-stored fact has no tense to read; we do NOT fabricate one).
        The unbind/cleanup is the SAME machinery ask_yes_no uses to read the polarity tag."""
        cue_roles = {r: fact[r] for r in ALL_ROLES if r in fact}
        for f, handle in self.kb:
            comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
            if all(self.unbind(comp, cr) == cv for cr, cv in cue_roles.items()):
                return self.unbind(comp, TENSE_ROLE, self.tense_words)
        return None

    # --- render: the bound tense tag DRIVES the surface verb form ---------------------------------------------
    def _decode_unit_word(self, unit, fact, comp):
        """Override the parent's TENSE branch: decode the verb from the RF unbind AND inflect it by the BOUND tense
        tag (read from the composite via the RF unbind + cleanup), not the host-default present-3sg. CONTENT slots
        are the parent's (the role's filler decoded from the composite)."""
        kind, role, _lead = unit
        if kind == "TENSE":
            verb = self.unbind(comp, "action")
            tense = self.unbind(comp, TENSE_ROLE, self.tense_words)   # READ the bound tag off the substrate
            return inflect(verb, tense)
        return self.unbind(comp, role)

    def render_tensed(self, fact, comp=None, ablate_closed_class=False, use_framecq=True, lesion_tense=False):
        """Render the fact as tensed prose via its verb frame, with the surface verb form DRIVEN by the bound tense
        tag. `lesion_tense=True` SEVERS the tense read (forces the PRESENT default) -- the lesion control proving the
        tag does real work (the rendered tense then collapses to present regardless of the stored tag). Otherwise the
        parent's frame render (FrameCQ ordering + closed-class scaffold + the moat on a missing composite)."""
        if not lesion_tense:
            return self.render(fact, comp=comp, ablate_closed_class=ablate_closed_class, use_framecq=use_framecq)
        # lesion: temporarily make _decode_unit_word read PRESENT (sever the tense role read).
        saved = self._decode_unit_word

        def _present_only(unit, f, c):
            kind, role, _lead = unit
            if kind == "TENSE":
                return inflect(self.unbind(c, "action"), "PRESENT")
            return self.unbind(c, role)
        self._decode_unit_word = _present_only
        try:
            return self.render(fact, comp=comp, ablate_closed_class=ablate_closed_class, use_framecq=use_framecq)
        finally:
            self._decode_unit_word = saved


# ---------------------------------------------------------------------------------------------------------------
# A small console probe: store a few tensed facts, then show the tensed renders + the moat. Reuse-by-import else.
# ---------------------------------------------------------------------------------------------------------------
def _console(seed=42, D=64):
    c = TenseAspectComposer(seed=seed, D=D)
    facts = [
        ({"agent": "boy", "action": "go", "GOAL": "park"}, "PAST"),
        ({"agent": "cat", "action": "run", "GOAL": "home"}, "PRESENT"),
        ({"agent": "dog", "action": "come", "GOAL": "home"}, "FUTURE"),
    ]
    # need the content words in the vocab so codes exist; rebuild with an extended vocab if absent.
    need = sorted({w for fct, _ in facts for w in (fct["agent"], fct["action"], fct.get("GOAL"))} | set(c.words))
    c = TenseAspectComposer(seed=seed, D=D, vocab=need)
    for fct, tn in facts:
        c.store_tensed(fct, tense=tn)

    print("=" * 92)
    print("TIER 2.5 tense/aspect console -- a bound PAST/PRESENT/FUTURE tag DRIVES the surface verb form")
    print(f"  (seed={seed}, D={D})")
    print("=" * 92)
    for fct, tn in facts:
        read = c.read_tense(fct)
        rendered = c.render_tensed(fct)
        lesioned = c.render_tensed(fct, lesion_tense=True)
        print(f"  {fct['agent']} {fct['action']} {fct.get('GOAL')} [{tn:7s}] -> read={read!r:9s} | "
              f"render: {rendered!r}")
        print(f"      (lesion-tense -> {lesioned!r})")
    print("  Moat (a never-stored fact has no tense, and the query abstains):")
    for fct in [{"agent": "boy", "action": "stop", "GOAL": "river"}, {"agent": "dog", "action": "look", "GOAL": "park"}]:
        print(f"    read_tense({fct['agent']} {fct['action']} {fct.get('GOAL')}) = {c.read_tense(fct)!r}  "
              f"(query_role abstains: {c.query_role('GOAL', agent=fct['agent'], action=fct['action']) is None})")
    print("=" * 92)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tier 2.5 tense/aspect (PAST/PRESENT/FUTURE tag -> surface verb form).")
    ap.add_argument("--console", action="store_true", help="run the tensed-render console probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
    _console(seed=args.seed, D=args.D)
