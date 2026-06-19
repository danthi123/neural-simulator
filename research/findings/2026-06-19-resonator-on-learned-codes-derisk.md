# Resonator nested-decode on the production LEARNED 320 codes — de-risk (2026-06-19)

**One cheap CPU/numpy de-risk before the conversational-scaling consolidation build.** Verdict feeds the
scope decision in `research/findings/2026-06-19-conversational-scaling-next-lever-scoping.md` (commit
`4927e09a`).

## The question (the only thing this tests)

The scoping recommends CONSOLIDATING validated-but-shelved richer capabilities into the production
`OneBrainComposer` — attributed entities, the **F=3 resonator nested / two-attribute decode**, and
multi-frame comprehension. The resonator does two-attribute decode **GO 6/6** in
`research/runners/nested_composition_agent.py`, BUT with a documented caveat: **it needs CLEAN PHASOR codes
and degrades on correlated codes.** Production conversation runs on the **LEARNED 320 codes** (the
stream / PPMI cortex, `_phaseB_stream_codes_320_neural_seed{42,43,44}.npy`), which carry **semantic
correlation**. So:

> Does the F=3 resonator's nested / two-attribute decode **HOLD** on the production's LEARNED 320 codes, or
> does learned-code correlation **degrade** convergence?

This decides whether the consolidation ships the resonator-nested path on the production codes, or ships
attributed + multi-frame only (which do not need the resonator) and records a boundary.

## Method (numpy / CPU, no `sim/` edit, no GPU)

`research/runners/_resonator_on_learned_codes_probe.py`.

- **Resonator agent under test:** `NestedCompositionAgent` (reuse-by-import) — the exact validated nested-decode
  capability (FLAT noun, ONE-attribute via the **2-factor** resonator, TWO-attribute "big red ball" via the
  **3-factor** resonator with random restarts), decoded through the production `query_patient` crosstalk-
  subtraction path.
- **Codes are fed as the composer would bind them.** Two sources, both as phasor codebooks:
  - **CLEAN (ceiling):** random uniform phasors — the resonator's documented best case (what the GO used).
  - **LEARNED (production-grounded):** the genuine learned-from-conversation 320 stream-cortex codes
    (`(320, 300)` real PPMI, on-bridge "neural" read), grounded into composer phases by the **verbatim
    production map** `grounded_phases(code) = angle(proj @ code) / 2π`
    (`consolidated_320_conversation_demo.py`, the step-3 grounding). These are EXACTLY the phasors the
    production composer binds.
- **Vocab from the real 320 taxonomy** (every token exists in the learned codes): 12 nouns
  (`animals_pets`+`toys`+`places`), 8 verbs (`motion_actions`+`manipulate`), 8 adjs
  (`colors`+`sizes`+`texture_temp`).
- **Dimension:** D = **2048**. The resonator nested capability fundamentally needs D ≥ 2048 — at the
  production composer's *default* D=128 even the CLEAN two-attribute decode MISSES (verified: it misfires into
  a clause decode). This is the GO test's dimension and matches the doc's "clause-in-clause needs D≥2048". The
  learned codes are grounded to 2048-dim phases through the same projection map (a 2048×300 fixed complex
  projection). This deliberately isolates the ONE question (does **correlation** degrade the decode?) from the
  orthogonal "D too small" effect. **Cost recorded:** consolidating the resonator path requires the composer
  run the nested ops at this higher D.
- 3 agent seeds (42/43/44), 8 facts per kind per seed.

## Result (3 seeds, correct / total)

| Capability | CLEAN (ceiling) | LEARNED (production-grounded) | Verdict |
|---|---|---|---|
| **flat** patient | 24/24 (100.0%) | **24/24 (100.0%)** | **HOLDS** |
| **one-attribute** (2-factor resonator) | 24/24 (100.0%) | **24/24 (100.0%)** | **HOLDS** |
| **two-attribute** (3-factor resonator) | 24/24 (100.0%) | **7/24 (29.2%)** | **DEGRADES** |

Rough chance for exact two-attribute (noun × adj-pair): **0.30%**. So the degraded learned-code two-attribute
recovery (29.2%) is still ~**100× chance**, but **far below the 100% clean ceiling**.

Per-seed (learned codes), with the grounded-vocab off-diagonal phasor cosine (the correlation the resonator
sees):

| agent seed | flat | one-attr | two-attr | grounded vocab off-diag cos (mean / max) |
|---|---|---|---|---|
| 42 | 8/8 | 8/8 | 4/8 | 0.159 / 0.532 |
| 43 | 8/8 | 8/8 | 1/8 | 0.162 / 0.508 |
| 44 | 8/8 | 8/8 | 2/8 | 0.148 / 0.429 |

(For reference, the raw learned 320 codes carry off-diagonal cosine mean ~0.03 but **max ~0.83–0.89** — a real
"dog/cat are related" semantic tail; the random projection spreads it to the ~0.15-mean / ~0.5-max seen above.)

## Mechanism (why two-attribute degrades but one-attribute does not)

A failure-by-adj-pair diagnostic (seed 42, all 28 adj-pairs on a fixed noun) makes the mechanism precise:

- The overwhelming failure signature is **`(adjₐ, adj_b) → "adjₐ noun"`** — i.e. the **noun is recovered
  correctly and ONE of the two attributes is dropped** (e.g. `(big, hot) → "big ball"`, `(small, cold) →
  "small ball"`). It is not a wholesale collapse to noise.
- The two recovered pairs had **low adj-adj grounded cosine** (0.03, 0.10); failures skewed higher (mean MISS
  adj-adj cos **0.195** vs OK **0.068**). So correlation does hurt — but the deeper cause is that on the
  correlated, lower-effective-dimensionality grounded codes, the **3-factor permutation-symmetric resonator
  (with restarts selected on reconstruction residual) reliably locks onto only ONE adjective** instead of
  disambiguating both. The two adjective factors share a codebook; the restart-residual tie-break that breaks
  their permutation symmetry on clean codes is what the correlation defeats.
- The **2-factor** resonator (one-attribute) has no permutation symmetry to break and stays **perfect** on the
  learned codes. FLAT cleanup is **perfect**. The fragility is specific to the **3-factor / two-attribute**
  decode.

## VERDICT

**PARTIAL — HOLDS for flat + one-attribute; the F=3 two-attribute (3-factor) decode DEGRADES on the learned
codes (100% clean → 29% learned, ~100× chance but well below ceiling).**

The degradation is driven by the learned codes' semantic correlation defeating the 3-factor resonator's
permutation-symmetry restart tie-break; the resonator collapses a two-attribute entity to one attribute (noun
preserved).

## Recommended consolidation scope

1. **SHIP into the production `OneBrainComposer` consolidation, on the learned 320 codes:**
   - **Attributed entities (single-attribute, the 2-factor resonator)** — 100% on the learned codes, == clean.
     This is the "big cat" capability and it is production-safe on the real codes.
   - **Multi-frame comprehension** — independent of the resonator; ship as scoped.
   - (FLAT who/what is already production; unaffected — 100%.)

2. **DO NOT ship the two-attribute / F=3 nested-decode path on the learned codes as a reliable capability.**
   Record the boundary: *two-attribute attributed entities ("big red ball") decode at ~100% on CLEAN phasor
   codes but only ~29% on the production learned/grounded codes (3-seed), because semantic correlation defeats
   the 3-factor resonator's permutation tie-break (it collapses to one attribute).* If two-attribute entities
   are later prioritized, the precise, specified next moves are: (a) **decorrelate / whiten the grounded
   phases** before the 3-factor decode (note: this is the known point-neuron-substrate whitening problem — the
   project's standing whitening blocker), or (b) a **stronger 3-factor restart schedule / residual margin** to
   survive ~0.5-max correlation, or (c) **distinct per-attribute role tags** (turn the two attributes into two
   *named* role-bindings rather than a commutative shared-codebook product, which removes the permutation
   symmetry entirely — at the cost of fixed attribute slots). All are follow-ons, not blockers for the
   consolidation.

3. **Dimension note for whoever wires the resonator path:** the nested ops need **D ≥ 2048**; the production
   composer's default phasor dimension is **D=128**, at which even the clean two-attribute decode fails. The
   single-attribute (2-factor) path's D requirement should be confirmed at the production D before shipping
   (this probe established the *correlation* answer at the resonator's viable D=2048; the single-attribute path
   was perfect there).

## Files

- Probe: `research/runners/_resonator_on_learned_codes_probe.py`
- Raw results: `research/findings/raw/_resonator_on_learned_codes.json`
- Production grounding map reused verbatim: `research/runners/consolidated_320_conversation_demo.py`
  (`_projection`, `grounded_phases`)
- Capability under test: `research/runners/nested_composition_agent.py`
- Learned codes: `research/findings/raw/_phaseB_stream_codes_320_neural_seed{42,43,44}.npy`
