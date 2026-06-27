"""Tier 2.1-A -- PRODUCTION factored-relation analogy (A:B::C:? over EXPLICIT factored relations).

Promoted from the GO de-risk (research/findings/2026-06-27-tier2.1A-factored-relation-analogy-GO.md;
scratch probe t21A_factored_relation_derisk.py): the analogy mechanism (unbind -> transform -> apply ->
cleanup; Komer-Stewart / Eliasmith-SPA) is sound on the composer's OWN spiking ops, and it WORKS when the
relation lives on an EXPLICIT FACTORED axis (the ADDITIVE-codes 1.000 case) -- the regime-A unlock the analogy
research gate identified (2026-06-27-analogy-representation-research-gate.md, option (a)). This is distinct from
analogy over RAW learned concept codes (regime B), which is the corpus-scale frontier and stays NO-GO
(2026-06-27-tier2.1-analogy-NEGATIVE.md): producing meaningful relational geometry on learned codes is the
open problem; the composer mechanism is ready when the codes are.

WHAT THIS IS (honest scope):
  Analogy/inference over relations the agent is GIVEN as an EXPLICIT FACTORED structure -- each item is a point
  in a shared factored attribute space (e.g. king = {GENDER:male, SEM:royal}, queen = {GENDER:female, SEM:royal}).
  The relation (the analogy transform) = the OFFSET along a factored axis (gender-flip, capital_of, past-tense).
  Realized through the composer's role-binding; the analogy transform-extract + apply RUN ON the resonate-and-fire
  spiking substrate (== numpy reference, 6 seeds). The no-confab moat is preserved: an un-grounded analogy
  (unregistered item, or a transform whose cleanup confidence is below the familiarity gate) ABSTAINS (None).

  WORKS FOR: BIJECTIVE (one-to-one) relations -- gender, capital_of, tense, comparative, antonym, and any 1:1
  parallel mapping with a shared offset. king:queen :: prince:? -> princess; paris:france :: rome:? -> italy.
  HONEST BOUNDARY (documented in the findings): MANY-TO-ONE relations (is_a / taxonomy: dog->mammal, cat->mammal)
  are NOT a single clean additive offset (the instance offset is entity-specific), so they are NOT served by this
  vector-offset mechanism -- they need a set-membership / category-readout mechanism (a separate build). And the
  RAW-CODE-GEOMETRY form (king-man~queen-woman emerging on LEARNED codes) is the regime-B corpus frontier.

DESIGN (reuse-by-import; NO sim/ edit; NO production-composer edit -- SUBCLASSES RFPhasorComposer):
  * register_item(name, **attrs): give an item its factored attributes (the explicit relation structure). Its code
    is the SUM of its per-attribute-VALUE phasors (the validated 'add' factored form -- a clean additive axis per
    attribute; this dodges the bundle-of-binds superposition-crosstalk wall, 2026-06-16 bundling-NEGATIVE / the
    de-risk's 'bind' mode which FAILS).
  * analogy(a, b, c): EXTRACT the transform T = phasor(b) (x) conj(phasor(a)) from the EXAMPLE pair (NOT a named
    relation -- genuine analogy), APPLY rec = T (x) phasor(c), CLEANUP over the registered codebook (operands
    excluded). The transform-extract + apply go through the RF spiking bind/conj when use_spiking_bind=True (the
    composer's _resonate); numpy phase arithmetic otherwise (== the spiking result, the CPU/test-oracle path).
  * The familiarity-gate moat: cleanup confidence below `abstain_sim` (or any operand unregistered) -> None.

Biology: TEM structure/content factorisation (Whittington-Behrens 2020 -- the relation is an explicit factored
axis, NOT read off raw similarity); rLPFC relational integration over WM-held structured representations
(Bunge/Wendelken); the Eliasmith-SPA / Komer-Stewart spiking VSA analogy recipe over role-filler structure.
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


class FactoredRelationAnalogy(RFPhasorComposer):
    """Analogy A:B::C:? over EXPLICIT factored-relation structure, on the RF phasor substrate.

    Items are registered with factored attributes; each item's code = the sum of its per-attribute-value phasors
    (the validated additive factored form). `analogy(a, b, c)` extracts the relation transform from the (a, b)
    example pair and applies it to c, cleaning up over the registered codebook -- the transform-extract + apply
    run through the composer's spiking bind/conj when `use_spiking_bind=True`. The no-confab moat abstains (None)
    on an un-grounded analogy.

    `abstain_sim` (default 0.5): the cleanup-confidence familiarity gate. The de-risk measured correct-analogy
    cleanup sims ~1.0 (numpy) / ~0.995 (spiking) and random-query sims ~0.08, so 0.5 cleanly separates a grounded
    analogy from an un-grounded one. None when the gate is below threshold -> the moat is preserved."""

    def __init__(self, seed=42, D=256, use_spiking_bind=False, abstain_sim=0.5):
        # The factored item codes REPLACE the parent's concept dict; the parent vocab is unused (we manage codes
        # ourselves), so we pass a 1-word vocab to keep the parent's construction cheap.
        super().__init__(seed=seed, D=D, vocab=["_"])
        self.use_spiking_bind = bool(use_spiking_bind)
        self.abstain_sim = float(abstain_sim)
        self._attr_rng = np.random.default_rng(seed + 7777)
        # the factored item store: name -> {axis: value}; the value-phase codebook; the item codebook.
        self.item_attrs = {}                  # name -> {axis: value}
        self._val_phase = {}                  # (axis, value) -> phases[D]
        self.item_code = {}                   # name -> additive factored phases[D]  (REPLACES self.concepts here)

    # --- registration ---------------------------------------------------------------------------------------
    def _value_phase(self, axis, value):
        key = (axis, value)
        if key not in self._val_phase:
            self._val_phase[key] = self._attr_rng.uniform(0.0, 1.0, self.D)
        return self._val_phase[key]

    def register_item(self, name, **attrs):
        """Register an item with its factored attributes (the explicit/given relation structure). Its code is the
        additive sum of its per-attribute-value phasors -- the relation between two items that differ on ONE axis
        is then a clean shared offset (the validated regime-A representation)."""
        if not attrs:
            raise ValueError(f"item {name!r} needs at least one factored attribute")
        self.item_attrs[name] = dict(attrs)
        comps = [self._value_phase(ax, v) for ax, v in attrs.items()]
        self.item_code[name] = np.sum(comps, axis=0) % 1.0

    def register_pair(self, axis, src, tgt, src_attrs=None, tgt_attrs=None, shared=None):
        """Convenience: register a BIJECTIVE relation pair on `axis` -- src and tgt share their other attributes
        (`shared`, an iterable of (axis, value)) and differ on `axis` (src->'src', tgt->'tgt' unless overridden).
        Lets a caller declare 'king:queen on GENDER, shared SEM=royal_hi' in one call."""
        shared = dict(shared or {})
        sa = dict(shared); sa.update(src_attrs or {axis: f"{axis}_src_{src}"})
        ta = dict(shared); ta.update(tgt_attrs or {axis: f"{axis}_tgt_{tgt}"})
        self.register_item(src, **sa)
        self.register_item(tgt, **ta)

    def _phasor_of(self, item):
        return self._to_phasor(self.item_code[item])

    # --- the analogy operation (transform-extract -> apply -> cleanup) --------------------------------------
    def _spiking_subtract(self, x_phases, y_phases):
        """T = x (x) conj(y) via the RF conj synapse (a phase SUBTRACT through resonate-and-fire). y is unbound."""
        D = self.D
        zx = self._to_phasor(x_phases)
        zy_conj = np.conj(self._to_phasor(y_phases))
        conns = [(D + k, k, zy_conj[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zx
        return self._resonate(2 * D, conns, kick)[D:]

    def _cleanup_factored(self, rec_phases, candidates):
        """Nearest registered item to `rec_phases` over `candidates`, plus its (mean-cos) confidence in [-1,1]."""
        sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - self.item_code[w])))) for w in candidates]
        j = int(np.argmax(sims))
        return candidates[j], float(sims[j])

    def analogy(self, a, b, c, candidates=None, exclude_operands=True, return_score=False):
        """Solve  a : b :: c : ?  over the registered factored codebook. Returns the answer item (or None = abstain
        when an operand is unregistered, or the cleanup confidence is below `abstain_sim` -- the no-confab moat).

        T = phasor(b) (x) conj(phasor(a))   # the relation transform, EXTRACTED from the example pair (genuine
                                            #   analogy -- the relation is never named to the apply step)
        rec = T (x) phasor(c)               # apply to C
        D   = cleanup(rec) over candidates, operands {a,b,c} excluded by default.

        With use_spiking_bind=True the transform-extract (B (x) conj A) AND the apply (T (x) C) run through the RF
        resonate-and-fire bind/conj (the composer's _resonate); numpy phase arithmetic otherwise (== spiking)."""
        if any(x not in self.item_code for x in (a, b, c)):
            return (None, None) if return_score else None       # moat: an unregistered operand -> abstain
        cands = list(candidates) if candidates is not None else list(self.item_code.keys())
        if exclude_operands:
            cands = [w for w in cands if w not in (a, b, c)]
        if not cands:
            return (None, None) if return_score else None
        if self.use_spiking_bind:
            t_phases = self._spiking_subtract(self.item_code[b], self.item_code[a])   # T through RF conj synapse
            rec = self._bind(t_phases, self.item_code[c])                             # apply through the RF bind
        else:
            zt = self._phasor_of(b) * np.conj(self._phasor_of(a))
            rec = (np.angle(zt * self._phasor_of(c)) / (2.0 * np.pi)) % 1.0
        pred, sim = self._cleanup_factored(rec, cands)
        if sim < self.abstain_sim:
            return (None, sim) if return_score else None         # moat: low-confidence -> abstain (un-grounded)
        return (pred, sim) if return_score else pred


# ---------------------------------------------------------------------------------------------------------------
# A small curated relation knowledge base for the console probe (the explicit factored relations the agent KNOWS).
# Each family is BIJECTIVE; the analogy answer is a real item of the family. (Taxonomy / is_a is intentionally
# ABSENT -- it is the documented many-to-one boundary, not served by this vector-offset mechanism.)
# ---------------------------------------------------------------------------------------------------------------
def build_knowledge_base(seed=42, D=256, use_spiking_bind=False, abstain_sim=0.5):
    """A FactoredRelationAnalogy preloaded with gender / capital / tense / comparative / antonym relations as
    explicit factored structure. Returns the composer. The console probe queries it."""
    kb = FactoredRelationAnalogy(seed=seed, D=D, use_spiking_bind=use_spiking_bind, abstain_sim=abstain_sim)

    # GENDER (gender-flip; SEM shared within a pair)
    gender = [("king", "queen", "royal_hi"), ("prince", "princess", "royal_lo"), ("man", "woman", "person"),
              ("actor", "actress", "perform"), ("uncle", "aunt", "kin"), ("boy", "girl", "young"),
              ("lord", "lady", "noble"), ("waiter", "waitress", "serve")]
    for m, f, sem in gender:
        kb.register_item(m, GENDER="male", SEM=sem)
        kb.register_item(f, GENDER="female", SEM=sem)

    # CAPITAL_OF (city->country; region shared)
    capital = [("paris", "france", "fr"), ("rome", "italy", "it"), ("berlin", "germany", "de"),
               ("madrid", "spain", "es"), ("lisbon", "portugal", "pt"), ("vienna", "austria", "at"),
               ("athens", "greece", "gr"), ("oslo", "norway", "no")]
    for city, country, reg in capital:
        kb.register_item(city, REGION=reg, ROLE="city")
        kb.register_item(country, REGION=reg, ROLE="country")

    # PAST-TENSE (present->past; lemma shared)
    tense = [("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("talk", "talked"),
             ("open", "opened"), ("close", "closed"), ("start", "started"), ("call", "called")]
    for pres, past in tense:
        kb.register_item(pres, TENSE="present", LEMMA=pres)
        kb.register_item(past, TENSE="past", LEMMA=pres)

    # COMPARATIVE (base->comparative; quality shared)
    comparative = [("big", "bigger", "size"), ("small", "smaller", "size2"), ("fast", "faster", "speed"),
                   ("slow", "slower", "speed2"), ("warm", "warmer", "temp"), ("cold", "colder", "temp2"),
                   ("tall", "taller", "height"), ("short", "shorter", "height2")]
    for base, comp_, qual in comparative:
        kb.register_item(base, DEGREE="base", QUAL=qual)
        kb.register_item(comp_, DEGREE="comparative", QUAL=qual)

    return kb


# --- relation-family lookup for the console (which items belong to which family, for nicer prompts) -------------
_FAMILY_AXES = {"GENDER": "GENDER", "CAPITAL": "ROLE", "TENSE": "TENSE", "COMPARATIVE": "DEGREE"}


def _console(seed=42, D=256, use_spiking_bind=False):
    kb = build_knowledge_base(seed=seed, D=D, use_spiking_bind=use_spiking_bind)
    print("=" * 78)
    print("FACTORED-RELATION ANALOGY console -- 'A is to B as C is to ?'")
    print(f"  (seed={seed}, D={D}, spiking_bind={use_spiking_bind}, abstain<{kb.abstain_sim})")
    print("  Known BIJECTIVE relations: GENDER, CAPITAL_OF, PAST-TENSE, COMPARATIVE.")
    print("  Many-to-one (is_a / taxonomy) is the documented BOUNDARY -> will ABSTAIN.")
    print("=" * 78)
    demos = [
        ("king", "queen", "prince", "princess"),       # gender
        ("man", "woman", "actor", "actress"),          # gender
        ("paris", "france", "rome", "italy"),          # capital_of
        ("berlin", "germany", "oslo", "norway"),       # capital_of
        ("walk", "walked", "jump", "jumped"),          # past-tense
        ("play", "played", "open", "opened"),          # past-tense
        ("big", "bigger", "fast", "faster"),           # comparative
        ("warm", "warmer", "tall", "taller"),          # comparative
        ("dog", "mammal", "robin", "?"),               # UNKNOWN items (is_a not in KB) -> abstain
        ("king", "queen", "zzz_unknown", "?"),         # unregistered C -> abstain
    ]
    for a, b, c, expect in demos:
        ans, sim = kb.analogy(a, b, c, return_score=True)
        verdict = "ABSTAIN (no grounded relation)" if ans is None else f"{ans}  (conf {sim:.3f})"
        mark = ""
        if expect != "?":
            mark = "  OK" if ans == expect else f"  <-- expected {expect}"
        print(f"  {a:8s} : {b:10s} :: {c:12s} : {verdict}{mark}")
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Factored-relation analogy (A:B::C:?) over explicit relations.")
    ap.add_argument("--console", action="store_true", help="run the 'A is to B as C is to ?' console probe")
    ap.add_argument("--spiking", action="store_true", help="run the analogy op through the real RF spiking bind")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=256)
    args = ap.parse_args()
    # The console probe is the only entry point (the module is otherwise reuse-by-import). --console is accepted
    # for explicitness; a bare invocation runs it too.
    _console(seed=args.seed, D=args.D, use_spiking_bind=args.spiking)
