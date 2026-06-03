# Multi-modifier attribution ("big red ball") RESOLVES via restart-selected resonator — 2026-06-03

**One line:** The nested-composition agent now stores and answers facts whose patient is a
*two-attribute* entity ("big red ball"), not just one attribute ("red ball"), and detects the
attribute count automatically. Validated cheap-first (algebra) then at the agent level 6/6 seeds.

This extends Direction A (the resonator decoder — see
`2026-06-03-resonator-decoder-cheap-first-RESOLVES-the-multi-factor-nesting-decode-in-algebra.md`).
A two-attribute patient is the product `adj₁ ⊗ adj₂ ⊗ noun`. Because the two adjective factors are
drawn from the **same** codebook, the decode hits the classic resonator repeated-factor failure —
permutation symmetry. This note records the honest cheap-first arc that resolved it and the model-
selection signal that lets the agent auto-detect depth.

## Cheap-first arc (numpy phasor algebra, D=1024, M=12 — a conservative over-count of the agent's vocab)

The probe is the decisive falsification-first step before any agent change.

| Step | Method | Two-attribute decode success |
|---|---|---|
| 1 | Naive 3-factor resonator (symmetric init) | **0.00** — both adjective factors collapse to one (permutation symmetry) |
| 2 | + random symmetry-breaking init | 0.43 — better, **not robust** |
| 3 | + K random restarts, **select best by reconstruction residual** | K=4 → 0.67; K=8 → 0.80; **K=16 → 0.93** |

Step 3 is the documented fix for repeated-codebook resonators (restart + reconstruction-residual
selection). At the agent's real adjective-vocabulary size (M=5, *easier* than this M=12 probe) the
floor is higher than 0.93.

### Model selection: how the agent knows the attribute count without being told

The same kind of confidence signal that already separates flat from nested (the cleanup confidence)
extends to separating **one attribute from two**: the **2-factor reconstruction residual**
`|⟨adj⊗noun, p⟩| / D`.

| Patient | 2-factor residual (mean) | min / max |
|---|---|---|
| One attribute `adj ⊗ noun` | **0.998** | min 0.996 |
| Two attributes `adj ⊗ adj ⊗ noun` | **0.114** | max 0.140 |

The gap is enormous and clean (0.996 vs 0.140 — any threshold in ~0.2–0.9 separates them). A 2-factor
resonator fits a true one-attribute product almost perfectly but a two-attribute product poorly,
because the second adjective has nowhere to go in a 2-factor model. So depth detection is:

1. flat cleanup confidence ≥ `flat_threshold` → **flat** noun;
2. else 2-factor resonator residual ≥ `resid_threshold` (0.5) → **one** attribute;
3. else 3-factor resonator with restarts → **two** attributes.

All three branches use one honest confidence/residual signal — the same principle as the no-confab
abstention threshold, just extended to attribute count.

## Agent-level validation (with bundle crosstalk — the real test)

The probes above used clean products. The agent unbinds the patient role from a *bundle* of three
role-bindings, so the decoded patient carries crosstalk from the agent + action bindings. At the
agent's default D=1024, K=16 restarts:

- `test_two_modifier_attributed_patient` — "cat see (big red) ball" → "big red ball" ✓
- `test_single_and_two_modifier_distinguished_automatically` — same noun "ball", one-attribute
  "red ball" vs two-attribute "big red ball", both correct, depth auto-detected ✓
- `test_two_modifier_canonical_vocab_order` — stored reversed ("red","big") still renders "big red
  ball" ✓
- `test_two_modifier_multi_seed_robust` — **6/6 seeds** (42–47) decode "small cold river" ✓

13/13 tests pass (9 original + 4 new). Demo decodes flat / one-attribute / two-attribute patients and
still abstains on the unknown.

## Honest scope

- **Adjective order is not recoverable.** Binding is commutative, so the two-attribute decode returns
  the *set* of modifiers; the agent renders them in canonical vocabulary order ("big red ball", never
  "red big ball"). This matches the semantics — adjective order in such attributive lists is largely
  free — but it is a genuine limitation, not a hidden one.
- This is the **phasor FHRR** substrate, where the resonator works. The flat-distinct 320 substrate's
  dense real-Hadamard binding is non-invertible and cannot nest at all (the 0.000-class failure;
  `_resonator_real320_probe`). The capability lives on the substrate that supports it.
- The decode is restart-based (K=16): more compute than a single pass, but sub-second per query at this
  scale and fully deterministic (fixed restart seed).

## Files

- `research/runners/nested_composition_agent.py` — `_filler` (multi-modifier product), `_resonator2`
  (returns residual), `_resonator3` (restart-selected repeated-factor decode), `query_patient` (3-way
  depth model-selection), `_render`/`_fact_concepts` (multi-modifier rendering in canonical order).
- `tests/test_nested_composition_agent.py` — 4 new multi-modifier tests (13 total).
- Cheap-first probes were inline (this note records their verdicts); the standing nested-fact probes
  remain in `research/findings/raw/_resonator_nested_fact_probe.py` + `_spiking_nested_fact_probe.py`.

## Verdict

**RESOLVES.** Two-attribute attributed entities decode robustly (6/6 seeds) with the attribute count
detected automatically. The unified conversational agent now spans flat facts, one- and two-attribute
nested entities, who/what Q&A, dialogue planning, and abstention — all on one biology-faithful phasor
substrate.
