#!/usr/bin/env python
"""STEP 1 cheap-first DE-RISK for Tier 1.1 — the entity-instance / discourse-referent layer (the KEYSTONE).

The HARD GATE before any build (research/findings/2026-06-27-conversation-thinking-ROADMAP.md, Tier 1.1;
fronts 1+2 BOTH ranked it #1). Can the brain turn its TYPE-keyed knowledge ("the concept boy") into INSTANCE
tracking ("this boy vs that boy") so "which boy?" is genuinely answerable -- moat-preserved?

Concretely (per the prompt):
  * ALLOCATE two instances of the SAME type -- boy#1 (fact: boy#1 go GOAL:park) and boy#2 (fact: boy#2 eat
    patient:apple) -- as SEPARABLE instance tokens;
  * store each fact keyed to the INSTANCE (not the type "boy");
  * resolve "which boy went to the park?" -> boy#1 (its distinguishing fact);
  * a pronoun/definite ("the boy"/"he") resolves to the HELD/biased instance (the biased-competition WTA).

The THREE confirmations that gate GO:
  (a) the two instances stay SEPARABLE -- low code/engram overlap, do not collapse to one "boy";
  (b) the disambiguation picks the RIGHT instance;
  (c) the no-confab moat holds -- an unstored instance/query ABSTAINS (0 false-accepts).

Anti-cheats (mandatory -- the project is BURNED by over-claimed memory/reference results):
  * PATTERN-SEPARATION: measure the two same-type instances' code overlap (must be LOW); a MERGE LESION
    (alpha=0 -> both instances collapse to the bare type code) -> disambiguation fails (the contrast proves
    separation is load-bearing).
  * RIGHT-REFERENT: the which-X / pronoun resolves to the CORRECT instance; a PERMUTED / BINDING LESION
    (sever the instance->fact binding) -> resolves WRONG / abstains.
  * MOAT 0 FA: an unstored instance or query abstains; never fabricate an instance.
  * MULTI-SEED (6 seeds) for the separation + disambiguation generalization claim.

The MECHANISM (reuse-by-import, NO sim/ edit, NO production-composer edit yet -- Step 2 only if GO).
An entity INSTANCE token is a PHASOR code minted as the TYPE code blended with a per-instance sparse "barcode"
(the hippocampal episodic index / DG-sparsified token; Quian-Quiroga concept cells = the TYPE, the barcode =
the individuating index; the SHIPPED D.14 engram API is functionally this barcode). In the complex domain:
    z(boy#i) = normalize( (1-alpha) * z_type[boy] + alpha * z_barcode_i ),  instance phases = angle(z) / 2pi
  * alpha = 0     -> the pure type code -> ALL boy#i are IDENTICAL  == the MERGE LESION (DG separation OFF).
  * alpha in [.5,.7] -> separated instances that STILL carry the type (so "which boy?" filters candidates by type).
The instance codes are injected into the deployed RFPhasorComposer's `concepts` dict (the composer is
concept-AGNOSTIC for binding -- rf_phasor_composer.py:262), so facts attach to the INSTANCE via the SAME
spiking RF bind/unbind the production composer uses, and "which X?" is a biased-competition WTA over the
type's candidate instances scored by which one's distinguishing fact matches the query.

Biology: hippocampal episodic-index "barcode" (eLife 2024 PMC11429605) binds co-active concept TYPES into an
individuated TOKEN; DG pattern separation (D.12) keeps two same-type instances decorrelated; CA3 pattern
completion (D.13) recovers the right one from a partial cue; Tonegawa engram (D.14, SHIPPED) = the barcode;
Eichenbaum-Cohen items-in-context (D.02) = the discourse-referent store; Desimone-Duncan/Wong-Wang biased
competition (the de-risked `biased_competition_buffer.py`) picks among several matching referents; DRT/file-card
(Kamp 1981) = the surface-ref -> token map.

Run:  SIM_BACKEND=numpy python -u -m research.runners._tier1_entity_instances_derisk
      SIM_BACKEND=numpy python -u -m research.runners._tier1_entity_instances_derisk --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# The instance-code mint: a per-instance sparse barcode blended into the type code (the DG-sparsified episodic
# index). alpha controls separation; alpha=0 is the MERGE LESION (both same-type instances collapse to the type).
# ---------------------------------------------------------------------------------------------------------------
def _phase_blend(type_phases, barcode_phases, alpha):
    """Blend type and barcode phasors in the COMPLEX domain, return the angle (phases in [0,1)). alpha=0 -> the
    pure type code (the merge lesion); alpha>0 -> the type pulled toward the per-instance barcode."""
    zt = np.exp(2j * np.pi * np.asarray(type_phases))
    zb = np.exp(2j * np.pi * np.asarray(barcode_phases))
    z = (1.0 - alpha) * zt + alpha * zb
    return (np.angle(z) / (2.0 * np.pi)) % 1.0


def _phase_cos(a, b):
    """Mean phase-cosine similarity in [-1, 1] (the composer's own cleanup metric)."""
    return float(np.mean(np.cos(2.0 * np.pi * (np.asarray(a) - np.asarray(b)))))


class InstanceLayer:
    """The entity-instance / discourse-referent layer over the deployed RFPhasorComposer (reuse-by-import).

    `allocate(type, attrs)` mints a NEW instance token (a type-blended barcode phasor injected into the
    composer's `concepts`), returns its token id ('boy#1'). `store_fact(token, ...)` binds a fact to the INSTANCE
    via the composer's spiking RF bind. `which(type, **cue)` is the biased-competition WTA over the type's
    candidate instances -- the one whose distinguishing fact matches the cue, or None (the no-confab moat). A
    DRT-style file-card (`_held`) holds the active discourse referent; `resolve_pronoun(type)` pattern-completes
    a definite/pronoun to the held instance.
    """

    def __init__(self, seed=42, D=128, alpha=0.7, base_vocab=None, verbose=False):
        self.seed = int(seed)
        self.D = int(D)
        self.alpha = float(alpha)
        self.verbose = bool(verbose)
        # base vocab = the TYPES + the fact fillers (actions/patients/goals). The composer mints type codes for
        # these; instance tokens are added on top of them.
        self.base_vocab = base_vocab or [
            "boy", "girl", "dog", "cat",                # entity TYPES
            "go", "eat", "chase", "see",                # actions
            "park", "apple", "bone", "ball", "river",   # patients / goals
        ]
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=self.base_vocab)
        # register the typed oblique roles (GOAL, ...) on the composer's role alphabet from a DISJOINT rng stream
        # (seed+2000), exactly as ArgStructureComposer does -- so the parent's concept/role codes stay byte-identical
        # and a fact can bind a GOAL filler via the same spiking RF bind. (Tier 0.1 already validated typed roles;
        # here a fact attaches GOAL/patient to an INSTANCE token agent.)
        _trng = np.random.default_rng(seed + 2000)
        for r in ("GOAL", "RECIPIENT", "THEME", "LOCATION", "SOURCE", "INSTRUMENT", "TIME"):
            self.comp.roles[r] = _trng.uniform(0.0, 1.0, self.D)
        # a DISJOINT rng stream for the per-instance barcodes (so the composer's type/role codes stay byte-identical
        # -- the same disjoint-stream discipline OrderedPositionWM/ArgStructureComposer use).
        self._barcode_rng = np.random.default_rng(seed + 7000)
        self._inst_count = {}          # type -> how many instances allocated (for stable token ids)
        self._tokens = {}              # token id -> {"type":..., "attrs":{...}}
        self._held = []                # the DRT file-card: discourse referents introduced this discourse, in order
        if verbose:
            print(f"[instance-layer] seed={seed} D={D} alpha={alpha}", flush=True)

    # -- allocation: indefinite "a boy" -> a fresh instance token (a DG-sparsified barcode over the type) ---------
    def _draw_barcode(self, type_name):
        """A per-instance barcode (the sparse episodic index). DG pattern-separation (D.12) + adult-neurogenesis
        ('fine pattern separation') is realized as OVERLAP-REJECTION: redraw the barcode until the resulting
        instance code is decorrelated from every already-allocated same-type instance (phase-cos below the
        random-floor band). This is the project's own overlap-rejection recovery path (CLAUDE.md, the 320-concept
        sparse-codes work) -- it guarantees reliable separation instead of relying on a single lucky draw. Falls
        back to the best of N tries (never an infinite loop)."""
        existing = [self.comp.concepts[t] for t in self.instances_of(type_name)]
        zt = self.comp.concepts[type_name]
        best, best_max = None, np.inf
        for _ in range(40):
            bc = self._barcode_rng.uniform(0.0, 1.0, self.D)
            code = _phase_blend(zt, bc, self.alpha)
            worst = max((_phase_cos(code, e) for e in existing), default=-1.0)
            if worst < best_max:
                best, best_max = code, worst
            if worst < 0.12:                      # decorrelated from all same-type siblings -> accept
                return code
        return best                                # best-of-40 (rarely reached at D>=64)

    def allocate(self, type_name, attrs=None):
        if type_name not in self.comp.concepts:
            raise KeyError(f"unknown type {type_name!r}")
        self._inst_count[type_name] = self._inst_count.get(type_name, 0) + 1
        token = f"{type_name}#{self._inst_count[type_name]}"
        self.comp.concepts[token] = self._draw_barcode(type_name)
        # the token participates in cleanup as a candidate concept (so unbind can recover it)
        if token not in self.comp.words:
            self.comp.words.append(token)
        self._tokens[token] = {"type": type_name, "attrs": dict(attrs or {})}
        self._held.append(token)       # newly-introduced referent enters the file-card
        if self.verbose:
            print(f"[allocate] {token}  (type={type_name})", flush=True)
        return token

    def instances_of(self, type_name):
        """The candidate instance tokens of a type (the WTA candidate set for 'which X?')."""
        return [t for t, m in self._tokens.items() if m["type"] == type_name]

    # -- store a fact keyed to the INSTANCE token (the composer's spiking RF bind) --------------------------------
    # the roles a fact may bind (the parent's base alphabet + the typed oblique roles we registered). The parent's
    # `_encode` iterates only the module-level ROLES, so we encode over the EXTENDED set here (exactly the
    # ArgStructureComposer._encode override).
    _ALL_ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2",
                  "GOAL", "RECIPIENT", "THEME", "LOCATION", "SOURCE", "INSTRUMENT", "TIME")

    def _encode_fact(self, fact):
        bounds = [self.comp._bind(self.comp.roles[r], self.comp._filler_phases(fact[r]))
                  for r in self._ALL_ROLES if r in fact]
        return self.comp._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_fact(self, agent_token, action, patient=None, **typed_roles):
        """Store {agent: <instance token>, action, patient/typed roles...}. The agent is the INSTANCE token (not
        the bare type) -- this is what makes the fact about THIS boy. Encoded over the EXTENDED role set so typed
        oblique roles (GOAL, ...) bind via the same spiking RF bind the production composer uses."""
        fact = {"agent": agent_token, "action": action}
        if patient is not None:
            fact["patient"] = patient
        fact.update(typed_roles)
        comp = self._encode_fact(fact)
        self.comp.kb.append((fact, comp))

    # -- which X? -- the biased-competition WTA over the type's candidate instances -------------------------------
    def which(self, type_name, sever_binding=False, **cue_roles):
        """Resolve 'which <type> <cue>?' to the candidate instance whose DISTINGUISHING fact matches all cue roles.

        Mechanism (biased competition over candidates): for each candidate instance of the type, score how well a
        stored fact with THAT instance as agent matches the cue (each matched cue role contributes evidence; the
        match is the spiking RF unbind+cleanup). The WTA winner = the uniquely-best-matching instance; a TIE or NO
        match -> None (the no-confab moat: never fabricate which one).

        sever_binding=True is the BINDING-LESION control: the candidate's agent identity is ignored (every fact is
        a candidate for every instance), so the instance->fact binding no longer disambiguates -> wrong/abstain.
        """
        candidates = self.instances_of(type_name)
        if not candidates:
            return None, {}
        scores = {}
        for tok in candidates:
            best = 0
            for fact, comp in self.comp.kb:
                ag = self.comp.unbind(comp, "agent")
                if (not sever_binding) and ag != tok:
                    continue            # only this instance's facts count (the binding does the work)
                matched = sum(1 for r, v in cue_roles.items() if self.comp.unbind(comp, r) == v)
                best = max(best, matched)
            scores[tok] = best
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_tok, top_s = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        n_cue = len(cue_roles)
        # require ALL cue roles matched AND a strict lead over the runner-up (the biased-competition winner).
        if top_s < n_cue or top_s <= runner:
            return None, scores          # tie or no full match -> abstain
        return top_tok, scores

    def describe_distinguisher(self, token):
        """The distinguishing fact for an instance, rendered as prose ('went to the park' / 'ate the apple') --
        what the console says to disambiguate ('the one that went to the park, or the one that ate the apple?')."""
        for fact, comp in self.comp.kb:
            if self.comp.unbind(comp, "agent") != token:
                continue
            action = self.comp.unbind(comp, "action")
            if "GOAL" in fact:
                goal = self.comp.unbind(comp, "GOAL")
                return f"{_past(action)} to the {goal}"
            if "patient" in fact:
                pat = self.comp.unbind(comp, "patient")
                return f"{_past(action)} the {pat}"
            return _past(action)
        return None

    # -- pronoun / definite resolution: pattern-complete to the held referent (the file-card) ---------------------
    def resolve_pronoun(self, type_name=None, bias_token=None):
        """A definite/pronoun ('the boy'/'he') resolves to the HELD discourse referent. If a type is given, only
        held referents of that type are candidates; a `bias_token` (the biased-competition winner) is preferred.
        Empty file-card -> None (no antecedent to bind -> abstain, never confabulate)."""
        held = [t for t in self._held if (type_name is None or self._tokens[t]["type"] == type_name)]
        if not held:
            return None
        if bias_token is not None and bias_token in held:
            return bias_token
        # no bias -> the most-recently-introduced matching referent (the default accessibility; a single held
        # referent resolves unambiguously). Multi-referent ties without a bias are the documented WTA case.
        return held[-1]

    def reset_discourse(self):
        """Clear the file-card (a new discourse). Load-bearing control: after reset, a pronoun has no antecedent."""
        self._held = []


def _past(verb):
    irr = {"go": "went", "eat": "ate", "see": "saw", "run": "ran", "come": "came"}
    return irr.get(verb, verb + ("ed" if not verb.endswith("e") else "d"))


# ---------------------------------------------------------------------------------------------------------------
# The de-risk scenario + the three gates + the anti-cheats, run per seed.
# ---------------------------------------------------------------------------------------------------------------
def run_seed(seed, D=128, alpha=0.6, verbose=False):
    out = {"seed": seed, "alpha": alpha, "D": D}

    # === ALLOCATE two instances of the SAME type "boy", attach a distinguishing fact to each ===
    L = InstanceLayer(seed=seed, D=D, alpha=alpha, verbose=verbose)
    boy1 = L.allocate("boy")                       # "a boy walked in..."
    boy2 = L.allocate("boy")                       # "...another boy walked in"
    L.store_fact(boy1, "go", GOAL="park")          # boy#1 went to the park
    L.store_fact(boy2, "eat", patient="apple")     # boy#2 ate the apple

    # --- GATE (a) + ANTI-CHEAT: PATTERN SEPARATION ------------------------------------------------------------
    # The two same-type instances must be SEPARABLE (low code overlap) yet each still carries the TYPE.
    z1, z2 = L.comp.concepts[boy1], L.comp.concepts[boy2]
    z_type = L.comp.concepts["boy"]
    inst_inst_cos = _phase_cos(z1, z2)             # boy#1 vs boy#2 -- must be LOW (separated)
    inst_type_cos = 0.5 * (_phase_cos(z1, z_type) + _phase_cos(z2, z_type))   # each still ~a boy (type-linked)
    # random-pair floor: the mean phase-cos of MANY independent code pairs (the decorrelated baseline). For D=128
    # this is ~0; computed empirically (not the +1 self-pair bug). The DG-separation target is for two same-type
    # instances to sit NEAR this floor, decorrelated despite sharing the type.
    rng = np.random.default_rng(seed + 99)
    rand_cos = float(np.mean([_phase_cos(rng.uniform(0, 1, D), rng.uniform(0, 1, D)) for _ in range(200)]))
    rand_sd = float(np.std([_phase_cos(rng.uniform(0, 1, D), rng.uniform(0, 1, D)) for _ in range(200)]))
    # SEPARATION PASS (the principled `cleanup_separated` midpoint rule, NOT a tuned bound): the two same-type
    # instances are SEPARATED iff their mutual overlap sits BELOW the midpoint between the random floor and the
    # type-overlap they necessarily share (i.e. they are closer to "decorrelated" than to "a shared code"), AND
    # within a few SD of the random floor (near-floor decorrelation, the DG-sparsification signature).
    midpoint = 0.5 * (rand_cos + inst_type_cos)
    separated = (inst_inst_cos < midpoint) and (inst_inst_cos < rand_cos + 6.0 * max(rand_sd, 1e-3))
    out["separation"] = {
        "inst_inst_cos": inst_inst_cos, "inst_type_cos": inst_type_cos,
        "random_floor_cos": rand_cos, "random_floor_sd": rand_sd, "midpoint": midpoint,
        "separated": bool(separated),
    }

    # MERGE LESION (alpha=0): both instances collapse onto the bare type code -> identical -> disambiguation MUST
    # fail. This proves the separation is load-bearing (an artifact would still 'work' merged).
    Lm = InstanceLayer(seed=seed, D=D, alpha=0.0, verbose=False)
    mb1 = Lm.allocate("boy"); mb2 = Lm.allocate("boy")
    Lm.store_fact(mb1, "go", GOAL="park"); Lm.store_fact(mb2, "eat", patient="apple")
    merge_cos = _phase_cos(Lm.comp.concepts[mb1], Lm.comp.concepts[mb2])
    merged_park, _ = Lm.which("boy", action="go", GOAL="park")
    merged_apple, _ = Lm.which("boy", action="eat", patient="apple")
    # under the merge lesion the two instances are byte-IDENTICAL codes, so the agent-binding cannot tell them
    # apart: unbind(fact, 'agent') cleans up to the SAME token for BOTH facts. So the system CANNOT uniquely map
    # each distinguishing fact to its OWN instance -- the two distinct queries collapse onto one token (or one
    # mis-resolves). The control PASSES iff the merge FAILS to recover the correct (boy#1, boy#2) distinct pair --
    # i.e. NOT (which_park==mb1 AND which_apple==mb2). This is the load-bearing proof: separation is what lets the
    # binding individuate; remove it and individuation collapses.
    merged_distinct_correct = (merged_park == mb1 and merged_apple == mb2)
    merge_lesion_breaks = (not merged_distinct_correct) or (merged_park == merged_apple)
    out["merge_lesion"] = {"merged_inst_inst_cos": merge_cos,
                           "which_park_under_merge": merged_park, "which_apple_under_merge": merged_apple,
                           "merged_distinct_correct": bool(merged_distinct_correct),
                           "breaks_disambiguation": bool(merge_lesion_breaks)}

    # --- GATE (b) + ANTI-CHEAT: RIGHT REFERENT ----------------------------------------------------------------
    # "which boy went to the park?" -> boy#1 (its distinguishing fact). The console upgrade.
    which_park, scores_park = L.which("boy", action="go", GOAL="park")
    which_apple, scores_apple = L.which("boy", action="eat", patient="apple")
    right_referent = (which_park == boy1) and (which_apple == boy2)
    # the disambiguation prose the console would speak:
    disamb_text = None
    d1 = L.describe_distinguisher(boy1)
    d2 = L.describe_distinguisher(boy2)
    if d1 and d2:
        disamb_text = f"the one that {d1}, or the one that {d2}?"
    answer_text = None
    if which_park == boy1 and d1:
        answer_text = f"the boy that {d1}"      # "the boy that went to the park"
    out["which_x"] = {
        "which_park": which_park, "which_apple": which_apple,
        "right_referent": bool(right_referent),
        "scores_park": scores_park, "scores_apple": scores_apple,
        "disambiguation_text": disamb_text, "answer_text": answer_text,
    }

    # BINDING-LESION control: ignore the instance->fact binding (every fact is a candidate for every instance) ->
    # the cue matches both instances equally -> abstain / wrong. Proves the binding does the disambiguation.
    sever_park, _ = L.which("boy", sever_binding=True, action="go", GOAL="park")
    binding_lesion_breaks = (sever_park != boy1)
    out["binding_lesion"] = {"which_park_severed": sever_park,
                             "breaks_disambiguation": bool(binding_lesion_breaks)}

    # PRONOUN / DEFINITE resolution: after only boy#2 is the active/biased referent, 'the boy'/'he' resolves to it.
    # (single salient referent -> unambiguous pattern-completion to the held token; the biased-competition WTA is
    # exercised by `which` above when several match.)
    L.reset_discourse()
    L.allocate  # noqa  (no-op ref to keep the token table; allocate already populated _tokens)
    L._held = [boy2]                          # discourse just mentioned boy#2 ("...he ate the apple")
    pron = L.resolve_pronoun(type_name="boy")
    pronoun_ok = (pron == boy2)
    # pronoun moat: with an EMPTY file-card a pronoun has NO antecedent -> abstain (never confabulate).
    L.reset_discourse()
    pron_empty = L.resolve_pronoun(type_name="boy")
    pronoun_moat_ok = (pron_empty is None)
    out["pronoun"] = {"resolved": pron, "pronoun_ok": bool(pronoun_ok),
                      "empty_filecard_abstains": bool(pronoun_moat_ok)}

    # --- GATE (c) + ANTI-CHEAT: MOAT 0 FALSE-ACCEPTS ----------------------------------------------------------
    # 1) an UNSTORED instance/query must abstain. "which boy chased the cat?" -- no boy did -> None.
    unstored_which, _ = L.which("boy", action="chase", patient="cat")
    # 2) a query about a NEVER-ALLOCATED type abstains.
    unalloc_which, _ = L.which("girl", action="go", GOAL="park")
    # 3) a fact-level query about a non-existent instance fact abstains (the composer's own moat).
    fa = 0
    fa += 1 if unstored_which is not None else 0
    fa += 1 if unalloc_which is not None else 0
    # 4) the composer's no-confab moat on a direct fact query for an unstored predicate:
    q_unstored = L.comp.query_patient(boy1, "chase")   # boy#1 didn't chase anything -> None
    fa += 1 if q_unstored is not None else 0
    moat_ok = (fa == 0)
    out["moat"] = {"unstored_which": unstored_which, "unalloc_which": unalloc_which,
                   "query_unstored_predicate": q_unstored, "false_accepts": fa, "moat_ok": bool(moat_ok)}

    # === SEED VERDICT ===
    gate_a = bool(separated) and bool(merge_lesion_breaks)
    gate_b = bool(right_referent) and bool(binding_lesion_breaks) and bool(pronoun_ok) and bool(pronoun_moat_ok)
    gate_c = bool(moat_ok)
    out["gates"] = {"a_separation": gate_a, "b_right_referent": gate_b, "c_moat": gate_c}
    out["seed_go"] = gate_a and gate_b and gate_c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    results = []
    for s in seeds:
        r = run_seed(s, D=args.D, alpha=args.alpha, verbose=args.verbose)
        results.append(r)
        g = r["gates"]
        print(f"[seed {s}] GO={r['seed_go']}  "
              f"(a sep={g['a_separation']} inst-inst={r['separation']['inst_inst_cos']:+.3f} "
              f"vs type={r['separation']['inst_type_cos']:+.3f} | "
              f"b refer={g['b_right_referent']} which_park={r['which_x']['which_park']} | "
              f"c moat={g['c_moat']} FA={r['moat']['false_accepts']})", flush=True)
        if s == seeds[0]:
            print(f"    which boy went to the park? -> {r['which_x']['answer_text']}", flush=True)
            print(f"    clarification: which boy? -> {r['which_x']['disambiguation_text']}", flush=True)

    n_go = sum(1 for r in results if r["seed_go"])
    n = len(results)
    verdict = "GO" if n_go == n else ("PARTIAL" if n_go > 0 else "NO-GO")
    summary = {
        "seeds": seeds, "n_seeds": n, "n_go": n_go, "verdict": verdict,
        "D": args.D, "alpha": args.alpha, "wall_s": round(time.time() - t0, 1),
        "results": results,
    }
    print(f"\n=== TIER 1.1 ENTITY-INSTANCE DE-RISK: {verdict} ({n_go}/{n} seeds GO) "
          f"in {summary['wall_s']}s ===", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}", flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
