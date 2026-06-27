"""Tier 2.4 -- PRODUCTION minimal common-ground: shared-vs-private fact tagging -> AUDIENCE DESIGN (the cheapest
theory-of-mind slice).

WHAT THIS IS (honest scope)
  A competent speaker tracks what is MUTUALLY KNOWN (common ground) vs what only THEY know (private), and tailors
  what they SAY to it. AUDIENCE DESIGN = VOLUNTEER private facts (new to the listener); SUPPRESS-or-merely-ACKNOWLEDGE
  shared facts (the listener already knows them). This needs NO recursive belief reasoning -- only a per-fact
  SHARED/PRIVATE tag bound onto each fact + a read at response time. It is the smallest viable step toward modelling
  the interlocutor (front-2 target #3; Clark & Brennan 1991 grounding; Duff & Brown-Schmidt 2012 common ground is
  hippocampal/declarative; Stephens-Silbert-Hasson 2010 speaker-listener coupling is its inter-brain signature).

  HONEST BOUNDARY (NOT claimed here): this is NOT full ToM / false-belief reasoning (the recursive Bayesian
  agent-modelling wall, front-2 §3/§6). It is a single-bit-per-fact listener model (shared vs private). The harder
  ToM (a divergent BELIEF store, Sally-Anne; RSA implicature) is the deferred, research-gated frontier above it.

THE MECHANISM IT COMPOSES (reuse-by-import -- the PROVEN polarity/negation tag; NO research gate needed)
  RFPhasorComposer already binds a POLARITY role (AFFIRM/NEGATE) onto a stored fact, cleaned ONLY against a 2-word
  tag codebook (`pol_words`), and reads it back at query time (ask_yes_no). Common-ground is the SAME pattern at a
  NEW role: a SHARED/PRIVATE tag bound onto each fact (cleaned only against `cg_words = ['SHARED','PRIVATE']`), read
  at response time to decide audience design. PLUS the no-confab moat: a never-stored cue -> abstain; never fabricate
  a fact OR a tag (an unstored fact reads tag=None, not a guess).

DESIGN (NO sim/ edit, NO composer edit -- SUBCLASSES RFPhasorComposer):
  * store_cg(agent, action, patient, common_ground='SHARED'|'PRIVATE', polarity=None): store the fact with the
    common-ground tag bound onto it (the polarity-tag pattern, on a dedicated `commonground` role so it composes
    independently of negation). The tag bind/unbind run THROUGH the parent's RF resonate-and-fire ops (_bind /
    _unbind_phases both call _resonate), so on the spiking substrate the tag is bound/read in spikes -- exactly as
    the validated polarity tag is (NOT a host-side flag; the composer's ops ARE the RF substrate ops).
  * read_common_ground(agent, action, patient): the bound SHARED/PRIVATE tag of the first matching fact, by the
    composer's RF unbind + cleanup against `cg_words`; None if no fact matches (the moat -- no fabricated tag).
  * should_volunteer(agent, action, patient): audience design -- True iff the fact is PRIVATE (new to the listener);
    False iff SHARED (already known); None iff unknown (abstain -- the clarification trigger, not a guess).
  * describe_audience_designed(agent): generate the agent's stored sentence ONLY if it is new to the listener
    (PRIVATE); for a SHARED fact, return an ACKNOWLEDGEMENT ("as you know, ...") instead of re-stating it; None if
    no fact / unknown subject (the no-confab moat -- no invented sentence about an unknown subject).

  The `commonground` role + the `cg_words` codebook are ADDED in __init__ (the parent's roles/concepts dicts are
  extended, exactly as polarity adds AFFIRM/NEGATE). Everything else (bind/unbind/cleanup/scan/moat) is the parent's.

Biology: common ground is hippocampal/declarative (Duff & Brown-Schmidt 2012 -- amnesics fail to establish/use it,
syntax spared); the DMN/mentalizing network tailors output to it (front-2 §5/§10); the per-fact tag is the
file-card "grounded" flag of Discourse Representation Theory (Kamp 1981 / Heim 1982). Reuses the role-binding the
project validated for polarity (a bound metadata role is precedent).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

# The common-ground tag codebook -- the SHARED/PRIVATE markers, cleaned ONLY against this 2-word list (NOT the main
# vocab), exactly as the polarity tags are cleaned only against pol_words. SHARED = listener already knows the fact;
# PRIVATE = only the speaker knows it (so it is the informative thing to volunteer).
CG_WORDS = ("SHARED", "PRIVATE")
CG_ROLE = "commonground"


class CommonGroundComposer(RFPhasorComposer):
    """Minimal common-ground: a SHARED/PRIVATE tag bound onto each fact (the polarity-tag pattern) -> audience design.

    The tag bind/unbind go through the parent's RF ops (_bind / _unbind_phases -> _resonate), so the tag is
    bound/read on the resonate-and-fire substrate -- exactly as the validated polarity (AFFIRM/NEGATE) tag is. On
    the CuPy backend that runs on the GPU RF substrate; on the NumPy backend the same RF dynamics loop runs on CPU
    (the == test-oracle path). The `use_spiking_bind` arg is accepted for API symmetry with the other Tier-2
    composers and is forwarded to the parent (it does not change the tag path, which is always the RF ops).

    The no-confab moat is preserved: a never-stored cue reads tag=None (abstain -- no fabricated tag), and the
    underlying who/what query abstains on an unstored (agent,action) cue."""

    def __init__(self, seed=42, D=64, vocab=None, use_spiking_bind=False, **kwargs):
        super().__init__(seed=seed, D=D, vocab=vocab, **kwargs)
        self.use_spiking_bind = bool(use_spiking_bind)   # API symmetry; the tag path is always the parent's RF ops
        # ADD the common-ground role + its tag codebook to the parent's dicts (exactly as polarity adds AFFIRM/NEGATE).
        # Use the composer's OWN rng-stream-equivalent draws so the codes are deterministic per seed.
        rng = np.random.default_rng(seed + 4242)   # a distinct stream so we don't disturb the parent's codes
        self.cg_words = list(CG_WORDS)
        if CG_ROLE not in self.roles:
            self.roles[CG_ROLE] = rng.uniform(0.0, 1.0, self.D)
        for tag in self.cg_words:
            if tag not in self.concepts:
                self.concepts[tag] = rng.uniform(0.0, 1.0, self.D)

    # --- storage: the polarity-tag pattern, on the commonground role -----------------------------------------
    def _encode_cg(self, fact, common_ground):
        """Encode a fact WITH a bound common-ground tag. Mirrors RFPhasorComposer._encode but appends one more
        bound role (commonground (x) tag). The tag bind goes through the same _bind the parent uses, so it runs on
        the RF spiking substrate when use_spiking_bind is set (the parent's _bind always uses _resonate)."""
        from research.runners.rf_phasor_composer import ROLES
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ROLES if r in fact]
        bounds.append(self._bind(self.roles[CG_ROLE], self.concepts[common_ground]))
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_cg(self, agent, action, patient, common_ground="SHARED", polarity=None):
        """Store an SVO fact with a SHARED/PRIVATE common-ground tag bound onto it (the cheapest listener model).
        `common_ground` in {'SHARED','PRIVATE'}. The fact is stored via the parent's kb (so query_patient/query_agent
        /ask_yes_no all keep working unchanged); the tag is an EXTRA bound role read by read_common_ground."""
        cg = str(common_ground).upper()
        if cg not in self.cg_words:
            raise ValueError(f"common_ground must be one of {self.cg_words}, got {common_ground!r}")
        # build the same fact-dict the parent's store() builds (so attribute/clause/polarity handling is identical)
        fact = {"agent": agent, "action": action}
        if self._is_clause_like(patient):
            fact["patient"] = patient
        elif isinstance(patient, tuple):
            adjs, noun = patient
            adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
            fact["patient"] = noun
            fact["attribute"] = adjs[0]
            if len(adjs) > 1:
                fact["attribute2"] = adjs[1]
        else:
            fact["patient"] = patient
        if polarity is not None:
            fact["polarity"] = polarity
        fact["_cg"] = cg     # record the tag on the fact-dict too (host-side label; the AUTHORITATIVE copy is bound)
        comp = self._encode_cg(fact, cg)
        self.kb.append((fact, self._store_substrate(comp) if self.enable_substrate_store else comp))

    @staticmethod
    def _is_clause_like(x):
        return getattr(x, "_fields", None) == ("agent", "action", "patient")

    # --- read-back: the bound tag, by RF unbind + cleanup against cg_words (the polarity-read pattern) --------
    def read_common_ground(self, agent, action, patient):
        """The bound SHARED/PRIVATE tag of the FIRST stored fact whose full SVO matches (agent+action+patient), by
        the composer's RF unbind + cleanup against `cg_words`. None if no fact matches (the no-confab moat -- a
        never-stored fact has no tag to read; we do NOT fabricate one). The unbind/cleanup is the SAME machinery
        ask_yes_no uses to read the polarity tag."""
        for fact, comp in self._iter_facts():
            if (self.unbind(comp, "agent") == agent
                    and self.unbind(comp, "action") == action
                    and self.unbind(comp, "patient") == patient):
                return self.unbind(comp, CG_ROLE, self.cg_words)
        return None

    # --- audience design ------------------------------------------------------------------------------------
    def should_volunteer(self, agent, action, patient):
        """Audience design: should the agent VOLUNTEER this fact to the listener?
          PRIVATE (listener doesn't know it)  -> True  (volunteer -- it is informative)
          SHARED  (listener already knows it)  -> False (suppress / merely acknowledge -- don't re-explain)
          unknown (never stored)               -> None  (abstain -- the clarification trigger, not a guess)
        This is the single decision that distinguishes audience-designed output from a tag-blind speaker (who must
        either tell everything or suppress everything)."""
        tag = self.read_common_ground(agent, action, patient)
        if tag is None:
            return None
        return tag == "PRIVATE"

    def describe_audience_designed(self, agent, order_fn=None, acknowledge_shared=True):
        """Generate the agent's stored sentence WITH audience design over the FIRST fact whose agent matches:
          * the fact is PRIVATE -> VOLUNTEER it: render the full sentence (the parent's render_fact path; the word
            order is neural when order_fn is set).
          * the fact is SHARED  -> SUPPRESS the re-statement: return an ACKNOWLEDGEMENT ("as you know, <agent> ...")
            when acknowledge_shared (the competent move -- reference common ground, don't re-explain), or None to
            stay silent when acknowledge_shared=False.
          * no fact / unknown subject -> None (the no-confab moat -- no invented sentence about an unknown subject).
        Returns (text_or_None, decision) where decision in {'volunteer','acknowledge','suppress','abstain'}."""
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "agent") == agent:
                tag = self.unbind(comp, CG_ROLE, self.cg_words)
                ac = self.unbind(comp, "action")
                pt = self._render(comp, "patient", fact["patient"], order_fn=order_fn)
                adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    pt = " ".join(adjs + [pt])
                if tag == "PRIVATE":
                    words = [agent, ac, pt]
                    if order_fn is not None:
                        sent = " ".join(words[i] for i in order_fn(len(words)))
                    else:
                        sent = f"{agent} {ac} {pt}"
                    return sent, "volunteer"
                # SHARED -> do not re-state; acknowledge or stay silent
                if acknowledge_shared:
                    return f"as you know, {agent} {ac} {pt}", "acknowledge"
                return None, "suppress"
        return None, "abstain"      # no fact about this subject -> abstain (no fabrication)


# ---------------------------------------------------------------------------------------------------------------
# A small console probe: store a few SHARED + PRIVATE facts, then show audience-designed responses (the demo the
# report quotes). Reuse-by-import otherwise.
# ---------------------------------------------------------------------------------------------------------------
def _console(seed=42, D=64, use_spiking_bind=False):
    cg = CommonGroundComposer(seed=seed, D=D, use_spiking_bind=use_spiking_bind)
    facts = [
        ("dog", "go", "north", "SHARED"),     # listener already knows this
        ("cat", "run", "south", "PRIVATE"),   # only the speaker knows this -> volunteer
        ("dog", "look", "river", "PRIVATE"),
        ("cat", "stop", "east", "SHARED"),
    ]
    for a, act, pt, tag in facts:
        cg.store_cg(a, act, pt, common_ground=tag)

    print("=" * 90)
    print("MINIMAL COMMON-GROUND console -- audience design (volunteer PRIVATE, suppress/ack SHARED)")
    print(f"  (seed={seed}, D={D}, spiking_bind={use_spiking_bind})")
    print("  Listener model:")
    for a, act, pt, tag in facts:
        print(f"    {a} {act} {pt:7s}  [{tag}]")
    print("=" * 90)
    print("  Audience-designed responses (per subject):")
    for a in ("dog", "cat"):
        text, decision = cg.describe_audience_designed(a)
        print(f"    describe {a!r}: [{decision}] {text!r}")
    print("  Per-fact volunteer decision (audience design tracks the tag):")
    for a, act, pt, tag in facts:
        v = cg.should_volunteer(a, act, pt)
        word = {True: "VOLUNTEER (private->tell)", False: "SUPPRESS (shared->known)", None: "ABSTAIN"}[v]
        print(f"    {a} {act} {pt:7s} [{tag:7s}] -> {word}")
    print("  Moat (a never-stored fact has no tag, and the query abstains):")
    for a, act, pt in [("dog", "stop", "small"), ("cat", "come", "big")]:
        print(f"    read_common_ground({a},{act},{pt}) = {cg.read_common_ground(a, act, pt)!r}  "
              f"(query_patient abstains: {cg.query_patient(a, act) is None})")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Minimal common-ground (shared/private tag -> audience design).")
    ap.add_argument("--console", action="store_true", help="run the audience-design console probe")
    ap.add_argument("--spiking", action="store_true", help="run the tag bind/unbind through the real RF spiking bind")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
    _console(seed=args.seed, D=args.D, use_spiking_bind=args.spiking)
