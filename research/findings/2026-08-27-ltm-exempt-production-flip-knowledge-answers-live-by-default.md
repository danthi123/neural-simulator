---
type: finding
status: live
date: 2026-08-27
mechanism: gnw-consensus-ltm-exempt-production-flip
lane: E-language / knowledge-in-chat
artifacts:
  - research/findings/raw/_organb_ltm_exempt_derisk/organb_ltm_exempt_6seed_verdict.json
  - research/findings/raw/_organc_ltm_exempt_derisk/organc_ltm_exempt_6seed_verdict.json
runner: research/findings/raw/_organc_ltm_exempt_derisk/verify_derisk.py
---

# Knowledge-base facts now answer in live chat BY DEFAULT: the GNW-consensus LTM-exemption is flipped to production-default

**One-line:** `BRAIN_GNW_ORGANB_LTM_EXEMPT` (the ONE flag that governs both organ B on the 2-organ bus and organ C
on the 3-organ bus) is flipped from default-OFF to **default-ON** (`webapp/gnw_two_organ_bus.py::organb_ltm_exempt_enabled`
fallback `""`→`"1"`), so the live `/api/brain-chat` consensus stack no longer vetoes genuine long-term-memory
(15k-fact) recalls — a knowledge-base fact the brain actually holds now COMMITS instead of returning "I don't know".
`BRAIN_GNW_ORGANB_LTM_EXEMPT=0` is the byte-identical escape to the pre-flip behavior. Flipped autonomously per the
owner's explicit approval (2026-08-27, "handle them all autonomously, don't defer").

## What this closes (the #1 knowledge-in-chat blocker)

The deep trace (`2026-08-27-knowledge-in-live-chat-veto-comprehension-and-gnw-organb-expectation-gap.md`) found the
live consensus organs each built expectations ONLY from the small conversational buffer, never the LTM tier, so
BOTH organ B (surprise monitor) and organ C (comprehension) withheld on EVERY LTM recall → the Q=2/Q=3 consensus
abstained on every knowledge-base fact, right or wrong. Two de-risks fixed this behind the shared default-OFF flag:
- organ B — `2026-08-27-organb-ltm-exempt-derisk-6seed-GO.md` (GO 6/6, moat held, byte-identical-off, merged `9402f0ae`).
- organ C — `2026-08-27-organc-ltm-exempt-derisk-6seed-GO.md` (GO 6/6, moat held, byte-identical-off, merged `97d03a11`).

This flip makes that verified-safe "flag-on" behavior the production default.

## Why the flip is safe (the moat is intact BY THE SAME PROOF)

The de-risks' critical proof is that a NON-EXISTENT fact still ABSTAINS with the flag ON, 6/6 seeds, on BOTH buses —
because organ A's own recall-miss (`primary_recall_miss`) short-circuits BEFORE organ B or C is consulted, so the
exemption can never manufacture an answer the store does not hold. That proof was measured on exactly the flag-ON
state that is now the default. Conversational-buffer recalls are untouched (organ B/C's own vote is identical
flag-on/off for buffer facts). The anti-confab moat is therefore preserved by construction, not weakened.

## Verification of the flip itself

Artifacts (the two merged de-risk verdicts this flip promotes):
`research/findings/raw/_organb_ltm_exempt_derisk/organb_ltm_exempt_6seed_verdict.json` and
`research/findings/raw/_organc_ltm_exempt_derisk/organc_ltm_exempt_6seed_verdict.json` (both GO 6/6, moat held).

Gate-level (direct): with `BRAIN_GNW_ORGANB_LTM_EXEMPT` genuinely unset (the production default),
`organb_ltm_exempt_enabled()` returns **True**; `=0` returns **False** (the byte-identical escape); `=1` True.
Organ C on the 3-organ bus imports and reads the SAME `organb_ltm_exempt_enabled`, so one flag governs the whole
stack. The underlying mechanism is unchanged from the two merged de-risks (6/6 GO each); this commit only changes
the default and the two docstrings.

## Honest residual (a SEPARATE gap, not blocking this flip)

Removing the consensus veto is necessary but not always sufficient end-to-end: the shipped 15k LTM bundle keys some
country entities as `<name>_portal` / `<name>_core` rather than the bare `<name>` a user types, so LTM RETRIEVAL can
still return empty for those exact surface forms independent of this flip (flagged by the open-ended-wiring agent).
That is a topic/key-routing gap in the fact store, a follow-up — not a defect in the consensus fix. For facts that
retrieve correctly (e.g. `chelsea_fc|country`), the recall now commits live.

Functional read-outs only; no phenomenal-experience claim.
