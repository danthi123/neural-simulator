---
type: finding
status: live
mechanism: gnw-two-organ-bus organ-B LTM-tier exemption (BRAIN_GNW_ORGANB_LTM_EXEMPT, default-OFF de-risk)
lane: integration
date: 2026-08-27
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_organb_ltm_exempt_derisk/organb_ltm_exempt_6seed_verdict.json
  - research/findings/raw/_organb_ltm_exempt_derisk/byte_identical_check_result.json
runner: research/findings/raw/_organb_ltm_exempt_derisk/{verify_derisk.py,byte_identical_check.py,build_verdict.py}
---

# Organ-B LTM-tier exemption — a default-OFF de-risk closing Bug 2 of the 2026-08-27 knowledge-in-chat-veto diagnostic (6-seed GO, moat verified)

**Verdict: GO (6/6 seeds).** A **default-OFF** de-risk. Behind a NEW flag `BRAIN_GNW_ORGANB_LTM_EXEMPT` (unset =
byte-identical to today, proven below), the GNW two-organ bus's organ B — the spiking "surprise monitor" that must
CORROBORATE organ A's recall before the coincidence can ignite — now corroborates instead of unconditionally
withholding when the recall's PROVENANCE is the stable cortical LTM tier. This closes
[Bug 2 of the 2026-08-27 diagnostic](2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md):
organ B's expectation registry `e_B` is built exclusively from the small conversational-buffer tier, so it holds
`expected=None` for every one of the shipped 15,000-fact Wikidata LTM core and vetoes ALL of them, correct or not.
**Production default is unchanged — this is a de-risk, not a flip; the owner decides whether/when to promote it.**

## Research-first (read before touching code)

- [`research/findings/2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md`](2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md)
  — the full two-layer trace. Bug 1 (comprehension word-order) is already fixed and merged
  (`2406e5aff`/`3666e01af`). Bug 2 (this finding's target) named three candidate fixes and explicitly declined to
  pick one unilaterally, "a judgment call about the anti-crosstalk moat's scope". Bug 4 (the 3-organ bus's hollow
  default-on flip) was independently fixed at `af5ca298` and is confirmed genuinely default-on in this session's
  own traces (see the 3-organ section below).
- `webapp/gnw_two_organ_bus.py` (organ B / the two-organ coincidence bus) and `webapp/gnw_three_organ_bus.py`
  (organ C / the three-organ consensus) read in full, plus how `webapp/server.py` installs both (~4088-4160).
- `research/runners/tiered_fact_store.py` and `research/runners/sharded_phasor_store.py` read in full to
  understand how `TieredFactStore` distinguishes a small conversational-BUFFER answer from a routed cortical-LTM
  answer (the buffer is checked first; the LTM is consulted only on a buffer abstain).
- `research/runners/_gnw_two_distinct_organs_derisk.py`'s `organ_b_confirms`: `exp is None -> organ B cannot form
  an expectation for this cue -> no vote` — the exact line the diagnostic's root cause traces to.
- `docs/TERMS.md` (byte-identical must be hash-asserted, not inferred from code; GO needs the gate's own verdict)
  and `docs/FAILURE_GATE_MATRIX.md` (`gates/production_integration`, `gates/single_seed`,
  `gates/claim_verdict_consistency`, `gates/verdict_preconditions`) consulted so this finding's claims stay
  gate-clean — verified directly (`tools/gates/production_integration.check(...)` on the three changed files
  returns `[]`).

## The mechanism (which of the diagnostic's 3 candidates, and why)

The diagnostic named three options. This builds **option 2** ("exempt LTM-routed recalls from organ B's
participation"), the one it flagged as least moat-risky, with the actual crosstalk exposure narrowed further than
the diagnostic's own framing:

1. **`TieredFactStore.query_patient_source(agent, action, order_fn=None)`** (`research/runners/tiered_fact_store.py`,
   purely additive — no existing method touched) returns `(patient, source)` with `source` in `{"buffer", "ltm",
   None}`, reusing the SAME per-tier calls `query_patient`'s own `_tiered()` already makes (no extra composer
   read).
2. **`gnw_two_organ_bus._organ_a_recall(composer, agent, action, *, ltm_exempt)`** — when `ltm_exempt=False`
   (the default), this is the byte-identical single `composer.query_patient(agent, action)` call, unchanged. Only
   when `ltm_exempt=True` AND the composer exposes `query_patient_source` does it probe the tier.
3. **`two_organ_combine(..., organb_ltm_exempt=False)`** — new keyword-only parameter, default `False`. When the
   recall's source is `"ltm"` AND the flag is on, organ B's vote is hard-set to `confirmed=True` WITHOUT calling
   `organ_b_confirms` at all (nothing in its registry to read against; the organ's own spiking corroboration read
   is never invoked for this recall). A conversational-buffer recall (`source == "buffer"`) is completely
   untouched — organ B still reads its own `e_B` expectation exactly as today.
4. The identical wiring is reused in `webapp/gnw_three_organ_bus.py`'s `three_organ_combine` (organ C's own
   comprehension vote is untouched — see below).

**Why corroborate rather than "organ B abstains from the vote" (the diagnostic's third candidate).** The
two-organ bus's ignition math is a calibrated 2-of-2 coincidence knee; introducing a third outcome ("no vote")
would require re-deriving `coincidence_hop`'s threshold, which the diagnostic itself flagged as the riskiest
option. Corroborating is the narrower change: it makes organ B agree exactly where its own honest read WOULD be
"nothing to disagree with" — the LTM tier represents pre-consolidated knowledge the surprise monitor was never
trained on and structurally cannot form an opinion about, so "no expectation" is being read as consent, not as
silence being defaulted to yes for an opinion the organ actually holds.

## The moat — verified rigorously (the critical check)

**Organ A's own miss is the moat's real gate, and the flag never touches it.** `two_organ_combine` returns
`abstain_reason="primary_recall_miss"` and returns EARLY, before organ B (exempted or not) is even consulted,
whenever `composer.query_patient`/`query_patient_source` returns `None` — i.e. whenever the fact genuinely is not
in either tier. Proof, 6/6 seeds, a fact guaranteed absent from both tiers
(`definitely_not_a_stored_entity_xyz` / `definitely_not_a_stored_relation_xyz`):

| seed | flag OFF | flag ON |
|---|---|---|
| 42/43/44/100/101/102 | `committed=None`, `abstain_reason="primary_recall_miss"` | `committed=None`, `abstain_reason="primary_recall_miss"` — IDENTICAL |

Same result on the three-organ bus (`_moat_unstored`, 6/6 seeds): `committed=None`,
`abstain_reason="primary_recall_miss"`. **The moat holds identically with the flag on or off, on both buses,
every seed.**

## Genuine LTM facts now commit (2-organ bus), 6/6 seeds

Three real facts pulled straight from the shipped `wikidata_core_15k` bundle (`chelsea_fc|country` — the
diagnostic's headline case — plus two more drawn live from the store, `frank_lincoln_wright|instance_of` and
`harold_clayton_lloyd|instance_of`):

| | flag OFF (today) | flag ON (the de-risk) |
|---|---|---|
| `committed` | `None` | the correct stored patient (`united_kingom`, `human_specie`, `human_specie`) |
| `abstain_reason` | `consensus_veto_organ_b_withheld` | — (none; committed) |
| `recall_source` | `null` (flag off never probes it) | `"ltm"` |
| `organb_ltm_exempt_applied` | — | `True` |

Identical across all 6 seeds — 18/18 (agent,action) x seed cells flip from vetoed to committed, with the exact
stored patient, never a different one (organ A's own recall is unchanged by this lever; only organ B's vote
changes).

## Conversational-buffer recalls: completely untouched (the discipline the flag exists to enforce)

A fact taught mid-conversation into the buffer tier (`zzz_test_agent / zzz_test_action / zzz_test_patient`,
`store()` called on the live composer) is read through `two_organ_combine` with the flag OFF and ON: `committed`,
`organ_b_confirmed`, and `organ_b_surprise_hz` are IDENTICAL in both arms, 6/6 seeds, and `recall_source ==
"buffer"` / `organb_ltm_exempt_applied == False` confirm the exemption path was never entered. (In this
particular test the freshly-taught fact happens to fall outside organ B's process-cached pre-registered block
range and reads `organ_b_confirmed=False` in BOTH arms — an existing, unmodified property of organ B's own
lazy-build caching, not something this lever changes; the point of the check is that flag on vs off makes ZERO
difference to a buffer-sourced recall, which held.)

## Byte-identical when OFF — hash-asserted against the PRE-PATCH code at HEAD, not inferred

Per `docs/TERMS.md`'s bar ("byte-identical... asserted in the data, never inferred from reading the code"),
`byte_identical_check.py` loads the actual PRE-PATCH `webapp/gnw_two_organ_bus.py` / `webapp/gnw_three_organ_bus.py`
straight from `git show HEAD` as standalone modules (not just "the same function with a kwarg defaulted"), and
SHA-256-hashes the OLD module's `two_organ_combine`/`three_organ_combine` output against the NEW module's output
with `organb_ltm_exempt=False`, across a probe panel (the LTM headline fact, an unstored fact, a buffer-taught
fact) — plus a full `chat.gate()` NL-question panel (`"what country is chelsea fc from"`, `"what is chelsea fc"`,
`"who are you"`) through BOTH an OLD-module-installed gate and a NEW-module-installed gate with the env var
genuinely unset.

**Result: 0 diffs, every probe, both buses, both the tuple-level and the full NL-question level**
(`research/findings/raw/_organb_ltm_exempt_derisk/byte_identical_check_result.json`).

## The three-organ bus: organ B's exemption reaches it too — organ C blocks LTM facts for a SEPARATE, un-fixed reason

Per the task's explicit request, the SAME `organb_ltm_exempt` lever was threaded through
`gnw_three_organ_bus.three_organ_combine` (reusing `_organ_a_recall` + the identical exemption logic). Result,
6/6 seeds, all 3 LTM probes: **organ B's vote is exempted correctly** (`organ_b_confirmed=True`,
`organb_ltm_exempt_applied=True`) — but the 3-organ consensus still ABSTAINS, now for a DIFFERENT reason:
`abstain_reason="consensus_veto_organ_c_non_comprehension"`.

**Root cause (traced, not fixed — reported per the task's instruction not to unilaterally patch it).** Organ C's
veto authority is `_real_vocab_competence(agent, action, cand, brain_vocab)`
(`webapp/gnw_three_organ_bus.py`), and `brain_vocab` comes from the SAME buffer-only `_chat_concepts(chat)` organ
B's registry uses — so `chelsea_fc` / `country` / `united_kingom` are ALL "unknown" to organ C too
(`organ_c_real_vocab_known=False`, `organ_c_unknown_tokens=[...]` naming all three). Organ C then falls to its
OOV fallback, the D4 `ComprehensionProductionOrgan`'s spiking margin: `.competent(...)` correctly judges this a
"fully out-of-vocabulary" case (`organ_c_competent=True` — genuine noise, not a partial/unreliable read) and
reads the margin — but the margin is **0.0** (no cue signal at all for words this organ's toy cue-lexicon has
never seen) against a calibrated threshold of **0.2486** (`organ_c_threshold` in the cited artifact), so it
withholds. This is uniform across every LTM
probe and every seed (`organ_c_margin: 0.0` in 18/18 cells) — not noise, a structural gap of the SAME shape as
organ B's (a component whose registry/vocabulary is buffer-scoped, meeting a knowledge-core concept it was never
built to judge). **Not fixed here** — it is a separate mechanism (organ C's real-vocab / D4-comprehension read,
not organ B's expectation registry) and fixing it is exactly the kind of unilateral moat-scope decision this
task was told not to make without owner review, mirrored from the diagnostic's own discipline for Bug 2.

## Constraints honored

- **Additive, default-OFF, byte-identical-when-off**: proven above by hash, not inferred.
- **Production default NOT flipped.** `two_organ_enabled()` / `three_organ_enabled()` / `_organ_discriminates()`
  — every existing default-on gate — untouched; `tools/gates/production_integration.check([...])` on all three
  changed files returns `[]` (no ledger/anchor/claim violation).
- **No `sim/` edit.** All changes are in `webapp/gnw_two_organ_bus.py`, `webapp/gnw_three_organ_bus.py`, and
  `research/runners/tiered_fact_store.py` (a new additive method, no existing method's body touched).
- **numpy, small ChatBrain** (`_build_tiny_demo`, `SIM_BACKEND=numpy`) throughout — no GPU brain build, per the
  memory-safety constraint (a concurrent GPU brain build was in flight).
- **6-seed** (42/43/44/100/101/102) for the moat + commit + buffer-untouched + 3-organ checks, per project
  standard.

## Honest residuals / next steps (not actioned here — owner review, per the task's own instruction)

1. **The 3-organ bus still vetoes every LTM fact**, now via organ C instead of organ B. If the owner wants
   LTM facts to commit on the 3-organ bus, organ C's `brain_vocab` needs its own LTM-awareness (a parallel
   gap to the one this finding closes for organ B) — NOT built here.
2. **`gnw_bus_shadow.py`'s organ C** (the diagnostic's Bug 3, a different, narrower reverse-binding defect in the
   N-organ bus) is unrelated to this lever and still open — not on the critical path (that bus never delegates
   down to it in production today).
3. **Promotion to default-on** is an owner call, exactly as instructed. This finding gives a moat-verified,
   6-seed-GO switch (`BRAIN_GNW_ORGANB_LTM_EXEMPT=1`) that currently unblocks the 2-organ bus (today's
   production default) for every genuine LTM fact while leaving an absent fact abstaining and every
   conversational-buffer recall untouched.

## Commands to reproduce

```bash
# 6-seed moat + commit + buffer-untouched + 3-organ sweep
SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_organb_ltm_exempt_derisk/verify_derisk.py

# byte-identical-when-off proof (hashes the PRE-PATCH code at HEAD against the new code, flag off)
SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_organb_ltm_exempt_derisk/byte_identical_check.py

# wrap the 6-seed sweep's checks in tools.verdict.Verdict (produces the cited GO artifact)
.venv/bin/python research/findings/raw/_organb_ltm_exempt_derisk/build_verdict.py

# manual spot-check, the diagnostic's own headline question, flag on vs off
SIM_BACKEND=numpy BRAIN_GNW_ORGANB_LTM_EXEMPT=1 .venv/bin/python -c "
import os
os.environ.setdefault('BRAIN_LTM_SHIP_DEFAULT', '1')
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
from research.runners.developed_brain_io import _inner_agent
from research.runners.tiered_fact_store import TieredFactStore
from research.runners.sharded_phasor_store import ShardedPhasorStore
from webapp.gnw_two_organ_bus import install_two_organ_gate

agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind='onebrain')
ltm = ShardedPhasorStore.load(os.path.expanduser('~/Projects/sim-data/knowledge_bundles/wikidata_core_15k'))
inner = _inner_agent(agent); inner.composer = TieredFactStore(inner.composer, ltm)
chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
install_two_organ_gate(chat, seed=42)
print(chat.gate('what country is chelsea fc from'))   # -> ['chelsea_fc', 'country', 'united_kingom'] with the flag ON
"
```
