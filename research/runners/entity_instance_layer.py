"""Tier 1.1 — the entity-instance / discourse-referent layer (PRODUCTION module).

Promoted from the GO de-risk (research/findings/2026-06-27-tier1-entity-instances-GO.md;
research/runners/_tier1_entity_instances_derisk.py, 6/6 seeds, all anti-cheats load-bearing). Turns the brain's
TYPE-keyed knowledge ("the concept boy") into INSTANCE tracking ("this boy vs that boy") so "which boy?" is
genuinely answerable — the KEYSTONE of the conversation-depth roadmap (fronts 1+2 BOTH ranked it #1).

THE MECHANISM (reuse-by-import; NO `sim/` edit; NO production-composer edit — this WRAPS the deployed composer).
An entity INSTANCE token is a PHASOR code minted as the TYPE code blended with a per-instance sparse "barcode"
(the hippocampal episodic index / DG-sparsified token; Quian-Quiroga concept cells = the TYPE, the barcode = the
individuating index; the SHIPPED D.14 engram API is functionally this barcode). In the complex domain:

    z(boy#i) = normalize( (1-alpha) * z_type[boy] + alpha * z_barcode_i ),   instance_phases = angle(z) / 2pi

  * alpha = 0     -> the pure type code -> ALL boy#i IDENTICAL  == the MERGE LESION (DG separation OFF).
  * alpha = 0.7 (default, the DG operating point) -> instances near the random-floor decorrelation, STILL
    type-linked (so a cue-driven CA3 completion could recover "a boy"; "which boy?" filters candidates by type).
  * OVERLAP-REJECTION on the barcode draw (the project's own 320-concept sparse-codes recovery path) realizes DG
    pattern-separation + adult-neurogenesis "fine pattern separation": redraw until decorrelated from same-type
    siblings -> reliable separation (lifted the 2 unlucky-collision seeds 4/6 -> 6/6 in the de-risk).

The instance codes are INJECTED into the deployed RFPhasorComposer's `concepts` dict (the composer is
concept-AGNOSTIC for binding -- rf_phasor_composer.py:262), so a fact attaches to the INSTANCE via the SAME
spiking RF bind/unbind the production composer uses (`agent = boy#1`, not the bare type "boy"). A
Discourse-Representation-Theory file-card (`_tokens` type metadata + `_held` referent registry, capacity ~7 per
Lisman-Idiart) maps surface refs -> tokens. "which X?" is a biased-competition WTA over the type's candidate
instances, scored by which one's distinguishing fact matches the cue (the de-risked biased_competition_buffer.py
pattern: a clean winner, or abstain on a tie -- the no-confab moat).

BIOLOGY: hippocampal episodic-index "barcode" (eLife 2024 PMC11429605) binds co-active concept TYPES into an
individuated TOKEN; DG pattern separation (catalog D.12) keeps same-type instances decorrelated; CA3 pattern
completion (D.13) recovers the right one from a partial cue; Tonegawa engram (D.14, SHIPPED) = the barcode;
Eichenbaum-Cohen items-in-context (D.02) = the discourse-referent store; Desimone-Duncan / Wong-Wang biased
competition = the multi-candidate WTA; DRT/file-card (Kamp 1981) = the surface-ref -> token map.

HONEST SCOPE (per the GO finding): the barcode "structure" is a developmental-random wiring rule (the genome-style
self-organization the project accepts); the bind/unbind attaching facts is the validated spiking RF FHRR primitive;
the which-X candidate SCORING is a host loop (the same scaffold biased_competition_buffer.py flags for
neuralization). Capacity is the biology-faithful Lisman-Idiart ~7 active referents. Multi-REFERENT bare-pronoun
disambiguation among several held referents still needs the biased-competition WTA + finer agreement cues
(2026-06-17-multireferent-disambiguation-NEGATIVE.md).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The composer's typed oblique roles (Tier 0.1) so a fact can bind GOAL/RECIPIENT/... to an instance agent.
TYPED_ROLES = ("GOAL", "RECIPIENT", "THEME", "LOCATION", "SOURCE", "INSTRUMENT", "TIME")
# the full role set a fact may bind (the composer's base alphabet + the typed obliques). The composer's `_encode`
# iterates only its module-level ROLES, so the layer encodes over this extended set.
ALL_ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2") + TYPED_ROLES

ALPHA_DEFAULT = 0.7        # the DG operating point (near-floor instance separation, retained type-linkage)
REJECT_COS = 0.12          # overlap-rejection threshold: redraw until decorrelated from same-type siblings
REJECT_TRIES = 40          # best-of-N (never an infinite loop)
WM_CAPACITY = 7            # Lisman-Idiart active-referent ceiling for the file-card

_PAST = {"go": "went", "eat": "ate", "see": "saw", "run": "ran", "come": "came", "give": "gave",
         "take": "took", "make": "made", "find": "found", "have": "had", "chase": "chased",
         "throw": "threw", "say": "said", "feel": "felt", "build": "built", "hear": "heard",
         "sing": "sang", "fly": "flew", "become": "became", "put": "put", "sit": "sat",
         "stand": "stood", "hold": "held", "tell": "told", "get": "got", "leave": "left",
         "meet": "met", "fall": "fell", "drink": "drank", "swim": "swam", "ride": "rode",
         "write": "wrote", "draw": "drew", "grow": "grew", "know": "knew", "fight": "fought"}


def past_tense(verb):
    """A small irregular-past table + the regular -ed/-d default (a closed-class morphology polish, like the
    parser's morphology; the brain decodes the bare verb)."""
    if verb in _PAST:
        return _PAST[verb]
    return verb + ("d" if verb.endswith("e") else "ed")


def phase_blend(type_phases, barcode_phases, alpha):
    """Blend type + barcode phasors in the COMPLEX domain, return phases in [0,1). alpha=0 -> pure type (the merge
    lesion); alpha>0 -> the type pulled toward the per-instance barcode (the DG-sparsified individuating index)."""
    zt = np.exp(2j * np.pi * np.asarray(type_phases))
    zb = np.exp(2j * np.pi * np.asarray(barcode_phases))
    z = (1.0 - alpha) * zt + alpha * zb
    return (np.angle(z) / (2.0 * np.pi)) % 1.0


def phase_cos(a, b):
    """Mean phase-cosine similarity in [-1, 1] (the composer's own cleanup metric)."""
    return float(np.mean(np.cos(2.0 * np.pi * (np.asarray(a) - np.asarray(b)))))


class EntityInstanceLayer:
    """Wraps a deployed RFPhasorComposer (or any composer exposing `concepts`/`words`/`roles`/`_bind`/`_bundle`/
    `_filler_phases`/`unbind`/`kb`) with entity-instance allocation + the discourse file-card + which-X
    disambiguation. Reuse-by-import: ADDITIVE to the composer (it never edits the composer's own facts/codes;
    instance codes are NEW dict entries, instance facts are NEW kb appends).

    Typical use:
        layer = EntityInstanceLayer(composer)          # the console's already-built RFPhasorComposer
        b1 = layer.allocate("boy")                     # "a boy ..."     -> a fresh token 'boy#1'
        b2 = layer.allocate("boy")                     # "another boy"   -> 'boy#2', pattern-separated
        layer.store_fact(b1, "go", GOAL="park")        # boy#1 went to the park
        layer.store_fact(b2, "eat", patient="apple")   # boy#2 ate the apple
        tok, _ = layer.which("boy", action="go", GOAL="park")   # -> 'boy#1'  (None on tie/no-match = the moat)
        layer.describe_distinguisher(b1)               # -> "went to the park"
    """

    def __init__(self, composer, alpha=ALPHA_DEFAULT, barcode_seed=None, capacity=WM_CAPACITY):
        self.comp = composer
        self.alpha = float(alpha)
        self.capacity = int(capacity)
        # register the typed oblique roles on the composer if missing (disjoint rng stream -> base codes unchanged).
        seed = int(getattr(composer, "seed", 42))
        _trng = np.random.default_rng(seed + 2000)
        for r in TYPED_ROLES:
            if r not in self.comp.roles:
                self.comp.roles[r] = _trng.uniform(0.0, 1.0, self.comp.D)
        # a DISJOINT rng stream for the per-instance barcodes (composer type/role codes stay byte-identical).
        bseed = (seed + 7000) if barcode_seed is None else int(barcode_seed)
        self._barcode_rng = np.random.default_rng(bseed)
        self._inst_count = {}          # type -> count (stable token ids)
        self._tokens = {}              # token -> {"type":..., "attrs":{...}}  (the DRT type metadata)
        self._held = []                # the discourse-referent file-card (introduced refs, in order; capacity-bounded)

    # -- allocation: indefinite "a boy" -> a fresh instance token (a DG-sparsified barcode over the type) ---------
    def _draw_barcode_code(self, type_name):
        """The instance code = type blended with a per-instance barcode, OVERLAP-REJECTED against same-type siblings
        (DG pattern-separation / adult-neurogenesis fine separation). Best-of-N if a decorrelated draw isn't found."""
        existing = [self.comp.concepts[t] for t in self.instances_of(type_name)]
        zt = self.comp.concepts[type_name]
        best, best_worst = None, np.inf
        for _ in range(REJECT_TRIES):
            code = phase_blend(zt, self._barcode_rng.uniform(0.0, 1.0, self.comp.D), self.alpha)
            worst = max((phase_cos(code, e) for e in existing), default=-1.0)
            if worst < best_worst:
                best, best_worst = code, worst
            if worst < REJECT_COS:
                return code
        return best

    def allocate(self, type_name, attrs=None):
        """Allocate a NEW instance token of `type_name` (an indefinite "a boy"/"another boy"). Returns the token id
        ('boy#1'). The token's barcode code is injected into the composer's `concepts` + `words` (so it binds + is a
        cleanup candidate), and it enters the discourse file-card (capacity-bounded, FIFO eviction)."""
        if type_name not in self.comp.concepts:
            raise KeyError(f"unknown type {type_name!r}")
        self._inst_count[type_name] = self._inst_count.get(type_name, 0) + 1
        token = f"{type_name}#{self._inst_count[type_name]}"
        self.comp.concepts[token] = self._draw_barcode_code(type_name)
        if token not in self.comp.words:
            self.comp.words.append(token)
        self._tokens[token] = {"type": type_name, "attrs": dict(attrs or {})}
        self._note_referent(token)
        return token

    def instances_of(self, type_name):
        """Candidate instance tokens of a type (the WTA candidate set for 'which X?')."""
        return [t for t, m in self._tokens.items() if m["type"] == type_name]

    def is_instance(self, token):
        return token in self._tokens

    def type_of(self, token):
        m = self._tokens.get(token)
        return m["type"] if m else None

    # -- the discourse file-card (capacity-bounded; the active referents) -----------------------------------------
    def _note_referent(self, token):
        if token in self._held:
            self._held.remove(token)
        self._held.append(token)
        if len(self._held) > self.capacity:        # Lisman-Idiart ~7 active referents -> FIFO eviction
            self._held = self._held[-self.capacity:]

    def reset_discourse(self):
        """Clear the file-card (a new discourse): a pronoun then has no antecedent (abstains)."""
        self._held = []

    # -- store a fact keyed to the INSTANCE token (the composer's spiking RF bind) --------------------------------
    def _encode_fact(self, fact):
        bounds = [self.comp._bind(self.comp.roles[r], self.comp._filler_phases(fact[r]))
                  for r in ALL_ROLES if r in fact]
        return self.comp._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_fact(self, agent_token, action, patient=None, **typed_roles):
        """Store {agent: <instance token>, action, patient/typed roles...}. The agent is the INSTANCE token (not the
        bare type) — this is what makes the fact about THIS boy. Encoded over the EXTENDED role set so typed obliques
        (GOAL, ...) bind via the same spiking RF bind. Touching the agent marks it the active referent."""
        fact = {"agent": agent_token, "action": action}
        if patient is not None:
            fact["patient"] = patient
        fact.update(typed_roles)
        comp = self._encode_fact(fact)
        self.comp.kb.append((fact, comp))
        if self.is_instance(agent_token):
            self._note_referent(agent_token)

    # -- which X? — the biased-competition WTA over the type's candidate instances --------------------------------
    def which(self, type_name, sever_binding=False, **cue_roles):
        """Resolve 'which <type> <cue>?' to the candidate instance whose DISTINGUISHING fact matches all cue roles.

        Biased-competition WTA over candidates: for each candidate instance, score how well a stored fact with THAT
        instance as agent matches the cue (each matched cue role = evidence; the match is the spiking RF
        unbind+cleanup). Winner = the uniquely-best-matching instance; a TIE or NO full match -> (None, scores) (the
        no-confab moat). Returns (token | None, {token: score}).

        sever_binding=True is the BINDING-LESION control (every fact a candidate for every instance) -> the cue no
        longer individuates -> wrong/abstain.
        """
        candidates = self.instances_of(type_name)
        if not candidates:
            return None, {}
        scores = {}
        for tok in candidates:
            best = 0
            for fact, comp in self.comp.kb:
                if (not sever_binding) and self.comp.unbind(comp, "agent") != tok:
                    continue
                best = max(best, sum(1 for r, v in cue_roles.items() if self.comp.unbind(comp, r) == v))
            scores[tok] = best
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_tok, top_s = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        if top_s < len(cue_roles) or top_s <= runner:      # tie or no full match -> abstain
            return None, scores
        return top_tok, scores

    def describe_distinguisher(self, token):
        """An instance's distinguishing fact, rendered as a prose RELATIVE clause ('went to the park' / 'ate the
        apple') — what a clarification says. Decoded from the RF unbind (not the stored labels). None if no fact."""
        for fact, comp in self.comp.kb:
            if self.comp.unbind(comp, "agent") != token:
                continue
            action = self.comp.unbind(comp, "action")
            if "GOAL" in fact:
                return f"{past_tense(action)} to the {self.comp.unbind(comp, 'GOAL')}"
            if "patient" in fact:
                return f"{past_tense(action)} the {self.comp.unbind(comp, 'patient')}"
            return past_tense(action)
        return None

    def clarify_which(self, type_name):
        """Build the upgraded "which X?" clarification text from the type's instances' distinguishing facts — e.g.
        'the one that went to the park, or the one that ate the apple?'. Returns (text | None, n_instances): None
        when the type has <2 distinguishable instances (the caller falls back to the honest generic Tier-0.4 line)."""
        toks = self.instances_of(type_name)
        descs = [(t, self.describe_distinguisher(t)) for t in toks]
        descs = [(t, d) for t, d in descs if d]
        if len(descs) < 2:
            return None, len(descs)
        clauses = [f"the one that {d}" for _t, d in descs]
        if len(clauses) == 2:
            text = f"{clauses[0]}, or {clauses[1]}?"
        else:
            text = ", ".join(clauses[:-1]) + f", or {clauses[-1]}?"
        return text, len(descs)

    def answer_which(self, type_name, **cue_roles):
        """Resolve which-X AND render the natural answer 'the <type> that <distinguisher>' (e.g. 'the boy that went
        to the park'); None on tie/no-match (the moat -> the caller asks the clarification instead)."""
        tok, _scores = self.which(type_name, **cue_roles)
        if tok is None:
            return None, None
        return tok, f"the {type_name} that {self.describe_distinguisher(tok)}"

    # -- pronoun / definite resolution: pattern-complete to the held referent (the file-card) ---------------------
    def resolve_pronoun(self, type_name=None, bias_token=None):
        """A definite/pronoun ('the boy'/'he') resolves to the HELD discourse referent. With a `type_name`, only
        held referents of that type are candidates; a `bias_token` (a biased-competition winner) is preferred; else
        the most-recently-introduced matching referent (the default accessibility — a single salient referent
        resolves unambiguously). EMPTY file-card -> None (no antecedent -> abstain, never confabulate). Multi-referent
        ties without a bias are the documented biased-competition WTA case (not resolved here)."""
        held = [t for t in self._held if (type_name is None or self.type_of(t) == type_name)]
        if not held:
            return None
        if bias_token is not None and bias_token in held:
            return bias_token
        return held[-1]
