---
type: finding
status: live
date: 2026-06-03
mechanism: fhrr
---

# Recursive clause nesting RESOLVES — depth-3 perfect on phasor FHRR (2026-06-03)

**One line:** A clause used as an argument — "dog see (cat chase bird)" — decodes perfectly on the
phasor FHRR substrate, and the capacity extends to **recursion depth 3** (a clause inside a clause
inside a clause, 7 fillers) across vocabularies up to 64 words per kind. This is genuine syntactic
recursion on the substrate whose flat-distinct form scored 0.000 on *any* nesting.

This is the third and strongest nesting result of the Direction-A arc, after the resonator decoder
(`2026-06-03-resonator-decoder-...`) and multi-modifier attribution
(`2026-06-03-multi-modifier-attribution-...`). The deep-research synthesis flagged nesting / multi-hop
SNR as THE wall (the hierarchical-320 shortcut scored 0.000 on structured facts and was retracted).
This note shows the wall is passable for embedded clauses.

## Why a clause is decoded differently from "big red ball"

- A **multi-modifier** entity ("big red ball" = adj₁⊗adj₂⊗noun) is a *product of unknown factors* —
  it needs the resonator network (and the two same-codebook adjectives create permutation symmetry,
  handled by restarts).
- A **nested clause** ("(cat chase bird)") is a *bundle of role-products* where the roles
  (AGENT / ACTION / PATIENT) are *known, fixed* vectors. So it is decoded by **recursive unbinding** —
  no resonator needed. Unbind the outer PATIENT role to get the inner clause, then unbind that clause's
  own roles. The only open question was whether the signal survives the accumulated crosstalk of
  multiple bundle levels.

```
inner = phasor( AGENT⊗cat + ACTION⊗chase + PATIENT⊗bird )        ("cat chase bird")
fact  = phasor( AGENT⊗dog + ACTION⊗see   + PATIENT⊗inner )        ("dog see (cat chase bird)")
decode: unbind outer AGENT/ACTION → dog, see;
        unbind outer PATIENT → inner-estimate → unbind its AGENT/ACTION/PATIENT → cat, chase, bird.
```

## Pre-registered frozen probe (depth 2) — RESOLVES

`research/findings/raw/_recursive_clause_probe.py` (D=1024, M=8, 20 trials):

| Metric | Result |
|---|---|
| Outer level recovered (agent + action) | 1.00 |
| Inner clause recovered (agent + action + patient) | 1.00 |
| **Full 5-filler recovery** | **1.00** |
| Control (outer patient read as a flat noun — must fail) | 0.20 (chance 0.125; well below the 0.50 gate) |

The control stays low: the patient slot is a clause, not a flat noun, so reading it as a flat noun
fails — confirming the recovery used the genuine nested structure, not a leak.

## Capacity sweep (depth × vocabulary) — the honest boundary

Full-recovery fraction (40 trials each, D=1024):

| Recursion depth (fillers) | M=8 | M=16 | M=32 | M=64 |
|---|---|---|---|---|
| 2 (5 fillers) | 1.00 | 1.00 | 1.00 | 1.00 |
| **3 (7 fillers)** | **1.00** | **1.00** | **1.00** | **1.00** |
| 4 (9 fillers) | 0.70 | 0.60 | 0.38 | 0.38 |
| 5 (11 fillers) | 0.03 | 0.00 | 0.03 | 0.00 |

**Depth 3 is perfect across all tested vocabularies.** Depth 4 degrades; depth 5 breaks. The limit is
the dimension-bounded bundle SNR (each nesting level adds crosstalk; ~D=1024 supports the signal of
~7 nested fillers cleanly), so the boundary moves up with D — depth 4–5 are expected to recover at
larger D. Vocabulary size barely matters within a depth (the cleanup is the easy part); recursion
depth is the real cost.

### Smell-test note (a perfect score earns extra scrutiny)

The first capacity sweep reported ~0.05 everywhere — a **false negative from a comparison bug** in the
probe (the construction list was innermost-first but the decode list was outermost-first; comparing the
two reversed lists never matched). The decode was correct all along; the frozen probe, which checks each
filler individually rather than as an ordered list, correctly showed 1.00. Lesson logged: a surprising
*negative* in an exploratory sweep deserves the same scrutiny as a surprising positive — here the bug was
in the harness, not the substrate.

## Significance

- **Depth 3 exceeds human center-embedding.** "The rat the cat the dog chased killed ate" is a depth-3
  center-embedding and is already near-incomprehensible to humans; the substrate decodes depth-3
  embedded clauses at 100%.
- This is **real syntactic recursion** (a clause filling an argument slot), the compositional capability
  the flat-distinct 320 substrate fundamentally lacked (dense real-Hadamard binding is non-invertible →
  0.000 on nesting). It lives on the phasor FHRR substrate, where binding is invertible.
- It composes with the existing nesting: a patient can be a flat concept, a one- or two-attribute entity,
  or an embedded clause — all decoded by the same unbind-and-detect machinery.

## Files

- `research/findings/raw/_recursive_clause_probe.py` — the frozen depth-2 RESOLVES probe.
- Capacity sweep was inline (this note records the table).

## Agent integration (same day) — single embedded clause is the robust agent capability

The capability is now exposed in the unified conversational agent
(`research/runners/nested_composition_agent.py`): a fact's patient can be a `Clause` namedtuple
("dog see (cat chase bird)"), decoded by recursive unbinding. The agent must *auto-detect* the slot's
kind (no flag tells it), using a clause detector — a verb component is present only in a clause
(ACTION-unbind confidence: clause 0.247–0.316 vs flat/attributed ≤0.077, a clean split; validated
cheap-first).

**Robust (tested, claimed), at the agent's default D=2048:**

| Case | Agent multi-seed | Mechanism |
|---|---|---|
| Single clause, flat args ("dog see (cat chase bird)") | 12/12 | clause detector + recursive unbinding |
| Attribute inside a clause ("dog see (cat chase (big bird))") | 12/12 | inside-clause model comparison (below) |
| Clause-in-clause ("dog eat (cat chase (bird hold ball))") | 11/12 | depth-2 boundary occasionally costs a seed |

- `test_embedded_clause_multi_seed_robust`, `test_attribute_inside_clause_multi_seed_robust` (6/6 each),
  `test_clause_in_clause_depth_2` (≥5/6). Auto-distinguished from flat/attributed patients.

**The key fix — inside-clause model comparison.** A first integration decoded clause arguments
flat-only (no resonator), which *confabulated* the base noun for an attributed argument (cleaning
`adj⊗bird` against nouns gives a random noun, ~1–2/6). The correct policy: inside a clause, decode an
argument as a flat noun **or** a one-attribute entity by **comparing which model explains the vector
better** — the flat-noun cleanup confidence vs the resonator's reconstruction residual — with no fixed
threshold (a fixed threshold can't separate flat nouns at depth 2 from attributed arguments at depth 1).
This made attribute-in-clause robust (0/6 → 6/6) without regressing clause-in-clause. The two attributes
inside a clause are out of scope (the 1-vs-2 residual escalation is skipped), so an inside-clause
attributed argument is always one adjective + noun.

**The remaining boundary (documented, NOT over-claimed):** two or more levels of clause nesting
(clause-in-clause is 11/12; three levels break). The raw substrate recurses to depth 3 with *known* role
structure (the probe + sweep above); the agent's *auto-detection* adds a per-level kind-decision whose
error compounds, so its robust depth is ~2 embedded clauses. The depth-2 case needs D ≥ 2048 (at D=1024
it is ~5/6); "more capacity wins" — raising D lifts the agent's robust nesting depth, exactly as the SNR
analysis predicts.

## Biology-faithful capstone — the recursive clause decodes IN SPIKES

`research/findings/raw/_spiking_recursive_clause_probe.py` (D=256, M=8, 12 trials) builds and decodes
"dog see (cat chase bird)" *entirely* with the genuine resonate-and-fire FHRR ops (`rf_bind` / `rf_unbind`
/ `rf_bundle`; Izhikevich 2001 / Frady-Sommer 2019 phasor spikes):

| Metric | Result |
|---|---|
| Outer level recovered (agent + action) | 1.00 |
| Inner clause recovered (agent + action + patient) | 1.00 |
| **Full 5-filler spiking recovery** | **1.00** |
| Control (outer patient as a flat noun) | 0.00 (must stay low) |

So recursive clause nesting is biology-faithful, not just a numpy convenience — it works through two
bundle levels on the genuine spiking substrate, even at D=256.

**Smell-test (a perfect spiking score earns scrutiny):** the first run scored inner-clause 0.00 with a
correct outer 1.00 — a real signal of a second-level break. Diagnosed: an intermediate `rf_resonate`
cleanup between the two unbind levels corrupts the phase structure the second unbind needs (with resonate
0.00, without it 1.00, at D=256/512/1024). The fix is to unbind the raw `rf_unbind` output directly (the
same pattern the existing `_spiking_nested_fact_probe` uses). The 1.00 is genuine — the control at 0.00
confirms the decode uses the nested structure, not a leak.

## Verdict

**RESOLVES (substrate, depth 3) + agent capability (embedded clause incl. attributed arguments, robust).**
Recursive clause nesting works to depth 3 (perfect, vocab-robust) on the phasor FHRR substrate with a
clean dimension-limited boundary at depth 4–5. At its default D=2048 the unified agent robustly decodes a
single embedded clause (12/12), an attributed argument inside a clause (12/12, via inside-clause model
comparison), and one level of clause-in-clause (11/12) — with two-or-more clause-nesting levels the
documented boundary. A genuine path past the nesting / multi-hop wall for embedded clauses.
