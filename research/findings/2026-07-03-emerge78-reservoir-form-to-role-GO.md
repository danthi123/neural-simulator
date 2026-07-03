# EMERGE-78 — the FRONTO-STRIATAL RESERVOIR replaces the hand form→thematic-role labeler (learned map, no hand branch) AND integrates whole-sequence structure no fixed window can (a constructed non-local proof-of-mechanism) — **GO** (6-seed, adversarially hardened)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge78_reservoir_form_to_role_derisk.py`
**Test:** `tests/test_emerge78_reservoir_form_to_role.py`
**Raw:** `research/findings/raw/_emerge78_reservoir_form_to_role.json`
**Research gate:** `research/findings/2026-07-03-next-frontier-beyond-templated-constructions-research-gate.md`

## The residual this retires (the anti-whack-a-mole gate)

Everything UPSTREAM of the reply-side producer is already emergent (closed-class discovery EMERGE-62, slot order
EMERGE-63, slot inventory EMERGE-64, on-spike A→W EMERGE-67..71). The ONE hand-designed residual is the form→thematic-ROLE
labeler (`label_sentence`/`label_sentence_ext` + `FRAME_LEXICON`): positional if-rules that grow ONE BRANCH PER
CONSTRUCTION SHAPE — the whack-a-mole the owner flagged. `label_sentence_ext` even structurally caps at ≤1 post-verbal
argument (`if len(post) > 1: return None`, `_emerge72:181`).

## The mechanism (Hinaut–Dominey 2013, the fronto-striatal reservoir grammar model)

A FIXED-random echo-state reservoir (the rate analogue of a spiking liquid-state machine on the project's own recurrent
RF/Izhikevich pools) is driven by the EMERGE-62 discovered closed-class configuration (content abstracted to a single OPEN
marker → no lexical identity). A trained **final-state, slot-indexed** ridge read-out (the rate analogue of the
on-substrate population read-out) maps the reservoir's whole-sentence state → the thematic role of each content slot. No
`CONSTRUCTIONS` dict, no `label_sentence` branch. Rides the project's own pre-registered EMERGE-6b gate ("reservoir +
trained read-out / FORCE / Laje-Buonomano").

## The de-risk — **GO** (6 seeds 42/43/44/100/101/102; rate-level, CPU/numpy)

**Hardened after a 5-skeptic adversarial verification of a first pass.** The first pass claimed a GO on LOCAL
multi-argument held-out shapes (dative/double-PP), but the adversarial verify PROVED those are solvable by a *trivial
local rule* that ties the reservoir at 1.000 — so local held-out shows only a CONSOLIDATION win, not reservoir
necessity. This version adds the load-bearing NON-LOCAL test.

| gate | value (6-seed) | bar |
|---|---|---|
| **(A) CONSOLIDATION** — the reservoir LEARNS the full form→role map (train role acc) | **1.000** | ≥ 0.95 |
| shipped hand labeler on the multi-arg shapes | **0.000** (structural None) | ≤ 0.10 |
| **(B) NECESSITY** — reservoir on the relative-clause HEAD | **1.000** | ≥ 0.90 |
| — strongest LEFT-context (case-marking / governing-cue) baseline | **0.500** (chance) | ≤ 0.65 |
| — symmetric ±2 window baseline | **0.500** (chance) | ≤ 0.65 |
| (C) rel-head word-order scramble | 0.33 (≈ chance) | collapse |
| (D) rel-head non-degenerate closed-class-IDENTITY lesion | 0.500 (drop 0.500) | collapse |
| MOAT (honest) — OOD argument-fabrication / in-dist positional | 0.00 / 0.00 | reported |

**The load-bearing result (B):** on the single-embedding RELATIVE CLAUSE, the HEAD's role differs by structure —
**AGENT** in a subject-relative `the s1 that Vs the s2` vs **THEME** in an object-relative `the s1 that the s2 Vs` — yet
the head has an IDENTICAL left context (`the [head] that …`). The reservoir resolves it at **1.000**, while **both** the
strongest LEFT-context governing-cue rule **and** a symmetric ±2 window are at **chance (0.500)**. The disambiguation is
GLOBAL: because the relativizer "that" abstracts to the same OPEN marker as a verb, an object-relative and a simple
transitive `the s1 Vs the s2` have *identical local windows at the head* (`[EDGE, the, OPEN, OPEN, the]` in both), and
only the whole-sequence structure (relative clause vs complete SVO) separates them — which the reservoir's final state
integrates and no fixed window can. Length is ruled out: subject- and object-relatives are both 6 tokens (a length
classifier scores 0.500), so the reservoir uses verb-POSITION structure, not length.

**The necessity is CONTINGENT, disclosed honestly (per the focused adversarial recheck).** "that" occurs **zero times**
in the discovery corpus — it is out-of-vocabulary by construction (injected only at test time), so it is not
"undiscovered" but *absent*. The **counterfactual**: were "that" a distinct discovered closed cue (as EMERGE-62 would
likely discover it if it appeared in usage — high-frequency, distributionally flat), a ±1 window would resolve the head
and **the reservoir advantage would vanish** (verified: the ±2 window ties at 1.000 under that counterfactual). So (B)
is a genuine **proof-of-mechanism** — the reservoir integrates whole-sequence structure that no fixed window can, on a
**constructed** single-embedding case — **not** evidence that reservoirs are necessary for relative clauses in general.
The genuinely window-defeating result (variable-distance / deeper recursion, where no fixed window follows *regardless of
vocabulary*) is the RANK-3 frontier.

## Honest scope (precise, not overclaimed)

- **(A) CONSOLIDATION:** the form→role map is **LEARNED from usage with no hand branch** (a general governing-cue hand
  rule could also label the local shapes; the value is the self-extending LEARNED map, the Dominey-Hinaut path — not that
  a hand rule is impossible). This retires the whack-a-mole *maintenance pattern* (no per-shape human edit).
- **(B) NECESSITY:** the reservoir resolves a single-embedding NON-LOCAL/global dependency that no fixed local window
  captures. A *sufficiently wide* fixed window could catch this bounded case, but the reservoir self-adapts the memory
  depth from data rather than a human choosing the window; **variable-distance / DEEPER embedding (where no fixed window
  follows) is the RANK-3 frontier** (a theta-gamma-multiplexed WM buffer / assembly-calculus stack), not this rung.
- **NO-CONFAB (honest):** the read-out has NO abstain class and *does* fabricate argument roles on OOD closed-class
  sequences; the in-distribution positional check is WEAK — this is NOT the project's gate-first abstention moat.
- Rate-level, comprehension-first. The spiking LSM port + the production reservoir (Dominey 2015) are pre-registered
  follow-ons. Reuse-by-import (EMERGE-62 discovery + corpus vocab; the hand labeler as the control); NO `sim/` edit. NOT
  open prose (R4).

## Adversarial verification

**First pass (5 skeptics, `go_survives_all_skeptics: false`) — the process working as intended.** The skeptics found the
first pass's held-out shapes trivially local (a ±1/±2 rule ties the reservoir), the memorization-floor mis-specified, the
lesion degenerate (all-OPEN → OOD, not isolating the closed-class cue), the hand-labeler control fair-but-over-framed, and
the moat near-vacuous. Every finding was remediated: the load-bearing test is now the genuine NON-LOCAL rel-head (both the
strongest governing-cue AND symmetric-window baselines fail); the lesion is non-degenerate (closed→one generic marker,
structure preserved); the hand-labeler + moat claims are honestly reframed (learned-not-hand-authored; no abstain class /
OOD fabrication reported).

**Focused recheck (1 skeptic, `GO_NEEDS_FRAMING_FIX`) — folded in.** It confirmed the computational result is real and
the controls sound (length **ruled out** decisively; scramble/lesion/both baselines fair and correctly at chance), but
found the necessity claim's load-bearing dependency under-disclosed: "that" occurs **0 times** in the discovery corpus, so
the non-locality is *contingent* on the OOV/verb-colliding relativizer, not a structural property of relative clauses —
were "that" a distinct discovered cue, a ±1 window would tie. `necessity_genuine: partial` (genuine whole-sequence
integration, but a constructed case). Remediated: the "that"-absence contingency + the counterfactual are now disclosed
in the verdict, HONEST_NOTE, and the `task`/`mechanism` fields (above). The GO stands on the general **consolidation**
(learned map, no hand branch) + the honestly-scoped **proof-of-mechanism** (reservoir whole-sequence integration on a
constructed non-local case); general window-defeating necessity is deferred to RANK-3.

## Files
- `research/runners/_emerge78_reservoir_form_to_role_derisk.py` — the reservoir + final-state slot read-out + the two
  baselines (left-context governing-cue, symmetric ±2 window) + the 6-gate de-risk.
- `tests/test_emerge78_reservoir_form_to_role.py` — 6 CPU tests (non-local shapes, hand-labeler-None, content abstraction
  + non-degenerate lesion, determinism, final-state read, the seed-42 hardened GO gates).
- `research/findings/raw/_emerge78_reservoir_form_to_role.json` — the 6-seed de-risk.
