# Step 2b — RF composer co-resident on the one bridge: COMPLETE

> **Result: COMPLETE.** The fact-binding composer now runs on a slice of the SAME merged
> navigation+conversation bridge. With step 2a already complete, this finishes **roadmap step 2 — navigation and
> conversation consolidated onto one brain.** No new engine (`sim/`) edit; the only engine change is the
> owner-approved sliced phase-neuron operation, still default-off byte-identical.

## What step 2b did

Step 2a put navigation, the sentence-comprehension parser, and the dialogue planner onto one
`SimulationBridge`, but kept the **composer** — the part that binds words into facts and recalls them — on its own
separate network. Step 2b brings the composer onto the same bridge too: it runs its bind / recall operations on a
dedicated block of neurons (the `rf` region) of the merged bridge.

The composer uses a different neuron model (a phase-based "resonate-and-fire" neuron) than navigation's standard
model. They cannot be advanced in the same engine step. The owner-approved engine change lets the composer's
operations touch **only** its own block of neurons; navigation's step harmlessly overwrites the composer's idle
neurons between operations (the composer re-initializes each operation and keeps its memory in a separate set of
synapses), and the composer's operations leave navigation's neurons untouched.

Implementation (all in `research/runners/`, no `sim/` edit):
- `MergedRFComposer` (a subclass of the production composer) overrides only the low-level resonate step to address
  the merged bridge's `rf` slice (shift the operation's indices, build a full-width kick, install the complex
  weights, kick with the rf-region mask, read the slice).
- `build_merged_nav_conv_bridge(co_resident_rf=True)` reserves the `rf` region (no pathways into navigation,
  appended last so the other regions' indices are unchanged).
- `MergedNavConvAgent(co_resident_composer=True)` wires it in (default off = step-2a behaviour preserved).

## Acceptance gates — all green

| Gate | What it checks | Result |
|---|---|---|
| **1 — bit-exactness + isolation** (CPU) | a bind, and a bind→recall round-trip, on the rf slice equal the standalone composer **exactly**; the co-resident navigation neurons' state is **byte-identical** across the operation | **PASS** — `tests/test_merged_rf_composer_coresident.py` 5/5 (atol 1e-9; navigation state byte-identical) |
| **2 — conversational capability** (GPU, production D=128) | the full conversational matrix runs with the composer co-resident: comprehension, fact memory + Q&A, **abstention (refuses to make up answers)**, voice-invariance, negation/yes-no, embedded clauses, dialogue planning, generation; + the co-residence anti-cheat | **PASS** — `tests/test_nav_conv_step2b_coresident.py` 7/7 (81 s on RTX 3090) |
| **3 — navigation not regressed with the rf region** (GPU) | a flagship navigation episode on the merged bridge with the rf region present scores the same as without it | **PASS** — `gate6_merged_rf_seed42.json` = **2.0000**, per-phase `[0.4956, 0.5044, 0.4956, 0.5044]` — **byte-identical** to standalone and merged-no-rf (Δ = 0) |

Gate 1 is the merged-`n` form of the earlier coexistence proof (the "5b edited-version"): the masked operation is
correct **and** byte-isolated. Gate 3 confirms empirically what the structure guarantees — the rf region has no
navigation edges and is idle during the navigation episode, so it cannot perturb navigation (the same reason the
parser and dialogue-planner regions left navigation byte-identical in gate (a)).

## Status: roadmap step 2 is done

Navigation and conversation — comprehension, fact memory + question answering, abstention, negation, clauses,
dialogue planning, generation — now all run as separate, non-overlapping groups of neurons on **one**
`SimulationBridge` with one update loop, capability-equivalent to the separate brains. The navigation half learns
continuously while the conversational half stays frozen and unchanged, and the no-make-up-answers guarantee holds.

**Honest scope (unchanged):** this is a *consolidation* of the existing capabilities onto one substrate, not a new
capability. The composer's binding is still the clean, exactly-invertible vector-symbolic algebra — a principled
idealization of what a real cortex would learn, not a learned cortex. Replacing that idealization with a learned
spiking-cortical binding is **step 3 (the true cortex)**, the remaining frontier, deliberately deferred to its own
arc.

## Trail

- Step 2a complete (gate a PASS + gate b GREEN): `2026-06-10-step2a-nav-gate-a-PASS-3of6-byte-identical.md`
- 2b code: commit `3448afc8` (composer + builder + agent) + `e5e4d1a3` (gate-3 plumbing)
- 2b plan (trust-but-verified execution-ready): `docs/plans/2026-06-10-step2b-rf-coresident-implementation.md`
- The owner-approved engine change (default-off byte-identical):
  `2026-06-10-unification-sliced-RF-ops-edit-byte-review.md`
- Architecture (plain language): `docs/ARCHITECTURE_nav_conv_merge.md`
