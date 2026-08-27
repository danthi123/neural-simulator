---
type: finding
status: live
mechanism: gnw-three-organ-bus organ-C LTM-tier exemption (BRAIN_GNW_ORGANB_LTM_EXEMPT reused, default-OFF de-risk)
lane: integration
date: 2026-08-27
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_organc_ltm_exempt_derisk/organc_ltm_exempt_6seed_verdict.json
  - research/findings/raw/_organc_ltm_exempt_derisk/byte_identical_check_result.json
runner: research/findings/raw/_organc_ltm_exempt_derisk/{verify_derisk.py,byte_identical_check.py,build_verdict.py}
---

# Organ-C LTM-tier exemption — the mirror-image fix, closing the 3-organ bus's remaining residual from the organ-B de-risk (6-seed GO, moat verified)

**Verdict: GO (6/6 seeds).** A **default-OFF** de-risk, reusing the SAME flag as its organ-B counterpart. Behind
`BRAIN_GNW_ORGANB_LTM_EXEMPT` (unset = byte-identical to today, proven below), the GNW three-organ bus's organ C
— the spiking "comprehension monitor" that must corroborate the recalled proposition before the 3-way consensus
can ignite — now corroborates instead of vetoing when the recall's PROVENANCE is the stable cortical LTM tier.
This closes the residual [`research/findings/2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md`](2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md)
explicitly reported but declined to fix unilaterally: with the organ-B exemption alone, the 3-organ bus's
`chelsea_fc|country` probe still abstained, now via `abstain_reason=consensus_veto_organ_c_non_comprehension`
instead of organ B's. **Production default is unchanged — this is a de-risk, not a flip; the owner decides
whether/when to promote it.**

## Research-first (read before touching code)

- [`research/findings/2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md`](2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md)
  — the organ-B de-risk this arc mirrors, including its own "Honest residuals" section naming this exact gap:
  "The 3-organ bus still vetoes every LTM fact, now via organ C instead of organ B... organ C's `brain_vocab`
  needs its own LTM-awareness... NOT built here."
- [`research/findings/2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md`](2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md)
  — the original diagnostic (Bug 1 comprehension word-order, fixed separately; Bug 2 organ B's expectation gap,
  the parent of this arc).
- `webapp/gnw_three_organ_bus.py` read in full: `_real_vocab_competence`, `_comprehension_vote`,
  `three_organ_combine`, and the module's own DE-RISK docstring section (which already named this gap as "a
  genuinely separate gap, not fixed here" before this session).
- `webapp/gnw_two_organ_bus.py`'s `organb_ltm_exempt_enabled` / `_organ_a_recall` / the source-tier check —
  the pattern this arc mirrors for organ C (reused by import, not re-derived).
- `research/runners/tiered_fact_store.py::query_patient_source` — the additive method (already built by the
  organ-B de-risk) that tells a stable-LTM recall apart from a conversational-buffer one; unchanged here.

## The mechanism — reusing the SAME flag for BOTH organs, on purpose

Per the task's explicit steer, this arc does NOT introduce a second flag. `BRAIN_GNW_ORGANB_LTM_EXEMPT` now
governs organ B's exemption (unchanged, `gnw_two_organ_bus.py`) AND organ C's exemption (new, `gnw_three_organ_bus.py`) —
**one owner switch fixes the whole 3-organ stack.**

1. `three_organ_combine` already computed `ltm_exempt_applied = bool(organb_ltm_exempt and recall_source == "ltm")`
   for organ B (built by the organ-B de-risk). This arc reuses the SAME boolean for organ C — no new provenance
   probe, no new source-tier logic.
2. `_comprehension_vote(..., ltm_exempt: bool = False)` — new keyword-only parameter, default `False`. When
   `ltm_exempt=True`, the function returns `votes=True`, `organ_c_comprehended=True`,
   `organ_c_ltm_exempt_applied=True` WITHOUT calling `_real_vocab_competence` or consulting the D4
   `ComprehensionProductionOrgan`'s spiking margin at all — neither instrument is invoked for this recall.
3. `ltm_exempt=False` (the default) is the byte-identical pre-existing code path: `_real_vocab_competence` runs
   exactly as before, falling to the D4 spiking-margin read on an OOV proposition exactly as before.

**Why corroborate, mirroring organ B's own justification exactly.** A stored LTM fact's thematic roles are
already resolved by its own stored engram — the brain knows `chelsea_fc` is the agent, `country` the relation,
`united_kingom` the patient, BECAUSE it stored that triple. Comprehension of an ALREADY-RECALLED proposition is
not "can a buffer-calibrated vocabulary-membership cue see these tokens" (organ C's own domain question for an
UNKNOWN, freshly-asserted proposition) — it is a category error to apply that instrument to pre-consolidated
knowledge the instrument was never built to judge, exactly the argument the organ-B finding already made for the
surprise monitor.

## The moat — verified rigorously (the critical check)

**Organ A's own miss is still the moat's real gate, and this lever never touches it.** `three_organ_combine`
returns `abstain_reason="primary_recall_miss"` and returns EARLY, before organ B OR organ C (exempted or not)
is even consulted, whenever `query_patient`/`query_patient_source` returns `None` — i.e. whenever the fact
genuinely is not in either tier. Proof, 6/6 seeds, a fact guaranteed absent from both tiers
(`definitely_not_a_stored_entity_xyz` / `definitely_not_a_stored_relation_xyz`):

| seed | flag OFF | flag ON |
|---|---|---|
| 42/43/44/100/101/102 | `committed=None`, `abstain_reason="primary_recall_miss"` | `committed=None`, `abstain_reason="primary_recall_miss"` — IDENTICAL |

**The moat holds identically with the flag on or off, every seed, on the 3-organ bus.**

## Genuine LTM facts now commit on the 3-organ bus, 6/6 seeds

The same three probes the organ-B de-risk used (`chelsea_fc|country` — the diagnostic's headline case — plus two
more drawn live from the store):

| | flag OFF (today / pre-fix) | flag ON (this de-risk) |
|---|---|---|
| `committed` | `None` | the correct stored patient |
| `abstain_reason` | `consensus_veto_organ_b_withheld` (organ B fails first; organ C's own read is also `False` for the same reason, but never gets to author the reason) | — (none; committed) |
| `organ_b_confirmed` | `False` | `True` |
| `organ_c_votes` | `False` | `True` |
| `organ_c_ltm_exempt_applied` | `False` | `True` |
| `recall_source` | `null` | `"ltm"` |

Identical across all 6 seeds — 18/18 (agent,action) x seed cells flip from vetoed to committed, with the exact
stored patient, never a different one.

## Conversational-buffer recalls: completely untouched

A fact taught mid-conversation into the buffer tier (`zzz_test_agent / zzz_test_action / zzz_test_patient`) is
read through `three_organ_combine` with the flag OFF and ON: `committed`, `organ_b_confirmed`, `organ_c_votes`,
and `organ_c_real_vocab_known` are IDENTICAL in both arms, 6/6 seeds, and `recall_source == "buffer"` /
`organb_ltm_exempt_applied == False` / `organ_c_ltm_exempt_applied == False` confirm the exemption path was never
entered for either organ.

## The 2-organ bus: unaffected (this arc never touches `gnw_two_organ_bus.py`)

Spot-checked on the same probes, same seeds: `two_organ_combine`'s already-GO behavior (from the organ-B de-risk)
is unchanged by loading `gnw_three_organ_bus.py` in-process — no shared cache or module-level state leaks between
the two buses.

## Byte-identical when OFF — hash-asserted against the PRE-PATCH code at HEAD, not inferred

Per `docs/TERMS.md`'s bar, `byte_identical_check.py` loads the actual PRE-PATCH `webapp/gnw_three_organ_bus.py`
straight from `git show HEAD` (the exact code merged by the organ-B de-risk, BEFORE this session's organ-C edits)
as a standalone module, and SHA-256-hashes the OLD module's `three_organ_combine` output against the NEW module's
output with `organb_ltm_exempt=False`, across a probe panel (the LTM headline fact, an unstored fact, a
buffer-taught fact) — plus a full `chat.gate()` NL-question panel through BOTH an OLD-module-installed gate and a
NEW-module-installed gate with the env var genuinely unset.

**Result: 0 diffs, every probe, both the tuple-level and the full NL-question level**
(`research/findings/raw/_organc_ltm_exempt_derisk/byte_identical_check_result.json`).

## Constraints honored

- **Additive, default-OFF, byte-identical-when-off**: proven above by hash, not inferred.
- **Production default NOT flipped.** `two_organ_enabled()` / `three_organ_enabled()` / `_organ_discriminates()`
  — every existing default-on gate — untouched; `tools/gates/production_integration.check([...])` on the changed
  files returns `[]`.
- **No `sim/` edit.** All changes are in `webapp/gnw_three_organ_bus.py` (organ B's own module,
  `webapp/gnw_two_organ_bus.py`, is untouched by this arc).
- **numpy, small ChatBrain** (`_build_tiny_demo`, `SIM_BACKEND=numpy`) throughout — no GPU brain build, per the
  memory-safety constraint (a concurrent GPU brain build was in flight).
- **6-seed** (42/43/44/100/101/102) for the moat + commit + buffer-untouched + 2-organ-bus-unaffected checks.
- **One flag governs both organs.** No second env var was added; `BRAIN_GNW_ORGANB_LTM_EXEMPT` alone now unblocks
  organ B (`gnw_two_organ_bus.py`, pre-existing) AND organ C (`gnw_three_organ_bus.py`, this arc) for a genuine
  LTM-sourced recall, leaving a conversational-buffer recall and the unstored-fact moat untouched on both buses.

## Honest residuals / next steps (owner review, not actioned here)

1. **Promotion to default-on is an owner call**, exactly as instructed. This finding gives a moat-verified,
   6-seed-GO switch that currently unblocks the FULL knowledge-core on the 3-organ bus (today's install target
   when `BRAIN_GNW_3ORGAN` is on) for every genuine LTM fact, while leaving an absent fact abstaining and every
   conversational-buffer recall untouched.
2. `gnw_bus_shadow.py`'s own organ C (the diagnostic's Bug 3, a narrower reverse-binding defect in the N-organ
   bus one layer further down) is unrelated to this lever and still open — not on the critical path (neither the
   2-organ nor the 3-organ bus delegates down to it in production today).

## Commands to reproduce

```bash
# 6-seed moat + commit + organ-C-exemption + buffer-untouched + 2-organ-bus-unaffected sweep
SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_organc_ltm_exempt_derisk/verify_derisk.py

# byte-identical-when-off proof (hashes the PRE-PATCH code at HEAD against the new code, flag off)
SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_organc_ltm_exempt_derisk/byte_identical_check.py

# wrap the 6-seed sweep's checks in tools.verdict.Verdict (produces the cited GO artifact)
.venv/bin/python research/findings/raw/_organc_ltm_exempt_derisk/build_verdict.py

# manual spot-check, the diagnostic's own headline question, flag on vs off, full 3-organ install
SIM_BACKEND=numpy BRAIN_GNW_ORGANB_LTM_EXEMPT=1 .venv/bin/python -c "
import os
os.environ.setdefault('BRAIN_LTM_SHIP_DEFAULT', '1')
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
from research.runners.developed_brain_io import _inner_agent
from research.runners.tiered_fact_store import TieredFactStore
from research.runners.sharded_phasor_store import ShardedPhasorStore
from webapp.gnw_three_organ_bus import install_three_organ_gate

agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind='onebrain')
ltm = ShardedPhasorStore.load(os.path.expanduser('~/Projects/sim-data/knowledge_bundles/wikidata_core_15k'))
inner = _inner_agent(agent); inner.composer = TieredFactStore(inner.composer, ltm)
chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
install_three_organ_gate(chat, seed=42)
print(chat.gate('what country is chelsea fc from'))   # -> ['chelsea_fc', 'country', 'united_kingom'] with the flag ON
"
```
