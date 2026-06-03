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

**Robust (tested, claimed): a single embedded clause with flat-noun arguments — 6/6 seeds.**
- `test_embedded_clause_multi_seed_robust`: "dog see (cat chase bird)" decoded 6/6.
- Auto-distinguished from flat and attributed patients (`test_clause_vs_flat_and_attributed_distinguished`).

**Boundary (documented, NOT claimed):** the *agent's auto-detection* is robust to **one** embedded
clause, not the substrate's depth-3. Two distinct error sources compound the raw-substrate capacity:

| Case | Agent multi-seed | Why |
|---|---|---|
| Single clause, flat args ("cat chase bird") | 6/6 | within SNR; detection clean |
| Clause-in-clause ("cat chase bird hold ball") | ~5/6 | detection error compounds per level |
| Attribute inside a clause ("cat chase (big bird)") | unreliable | the resonator on a doubly-crosstalk'd `adj⊗noun` is past SNR |

The agent degrades **gracefully, not confabulatorily**: inside a clause it decodes arguments as flat
nouns only (no resonator), so an attributed inner argument loses its adjective rather than inventing a
wrong attribute. The distinction is honest: the **raw substrate** with *known* role structure recurses
to depth 3 (the probe + sweep above); the **agent's auto-detection** adds a per-level kind-decision whose
error compounds, so its robust depth is one embedded clause.

## Verdict

**RESOLVES (substrate, depth 3) + agent capability (one embedded clause, robust).** Recursive clause
nesting works to depth 3 (perfect, vocab-robust) on the phasor FHRR substrate with a clean
dimension-limited boundary at depth 4–5; the unified agent exposes a single embedded clause with flat
arguments robustly (6/6 seeds) and documents the deeper-nesting / attribute-in-clause boundary honestly.
A genuine path past the nesting / multi-hop wall for embedded clauses.
