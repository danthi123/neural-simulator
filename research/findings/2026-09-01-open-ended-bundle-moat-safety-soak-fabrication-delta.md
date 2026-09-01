---
type: finding
status: positive
date: 2026-09-01
mechanism: BRAIN_OPEN_ENDED bundle moat-safety soak — fabrication-rate delta across parent-only / +NP-entailment / +both-children arms, on brain-known / brain-unknown / Qwen-known-brain-unknown topic classes
lane: open-ended-honesty
seeds: [42]
seed-waiver: >
  A descriptive real-traffic soak (fabrication rates + a structural byte-identical check), not a stochastic
  GO. The load-bearing findings are STRUCTURAL and deterministic — (a) the moat children are byte-identical
  across all 3 arms on the dangerous/unknown classes because open_ended_chat.py returns _base_post_filter
  BEFORE either child is consulted, and (b) the base filter abstains on every unknown/dangerous topic — both
  reproduce identically on any seed. The rates are measured over n=32 topics (12 known / 10 unknown / 10
  dangerous) through the real /api/brain-chat path at seed 42. It informs an OWNER UX decision (#112); no
  production default was flipped.
artifacts:
  - research/findings/raw/_open_ended_bundle_moat_soak_full.json
  - research/runners/_open_ended_bundle_moat_safety_soak.py
external: NO-EXTERNAL-NEEDED — an internal real-traffic moat-safety measurement over this repo's own chat path.
---

# BRAIN_OPEN_ENDED bundle moat-safety soak: the flip is fabrication-SAFE (base filter covers the dangerous class), but it trades grounded known-fact recall for free generation the WKV mouth cannot yet ground

**Artifact:** `research/findings/raw/_open_ended_bundle_moat_soak_full.json` (3-arm × 3-class soak, n=32, real handler).

## The decision this soak informs (owner-gated, not flipped)

The `BRAIN_OPEN_ENDED` bundle (#112) would make the DEFAULT `/api/brain-chat` reply a free, first-person
generation (WKV spiking mouth in-vocab, Qwen fallback), guarded by two moat children —
`BRAIN_OPEN_ENDED_NP_ENTAILMENT` (spiking NP-entailment post-filter) + `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`.
The flip-plan (`2026-09-01-production-default-flip-plan.md`) named the exact next action: a real-traffic
moat-safety soak of the bundle (fabrication rate with vs without the children, on brain-known / brain-unknown
/ Qwen-known-brain-unknown topics), to hand the owner the fabrication-rate delta before any flip.

## Results (3 arms — parent-only A / +NP-entailment B / +both-children C — × 3 topic classes, n=32)

**1. The moat children are STRUCTURALLY INERT on the dangerous + unknown classes.** All three arms are
byte-identical on every unknown and every dangerous topic. Cause (confirmed on generated text, not just
argued from source): `webapp/open_ended_chat.py`'s post_filter takes `if not known: return
_base_post_filter(...)` BEFORE `np_entailment_enabled()` is consulted, and the gen-time-honesty path is
itself gated on `known`. So the two children can only ever affect KNOWN topics.

**2. The base filter already makes the dangerous class fabrication-safe.** On every unknown and every
dangerous (Qwen-known / brain-unknown) topic, raw generation fabricates (fabrication_rate_raw 1.0) but the
BASE filter catches it: **fabrication_rate_filtered 0.0, abstain_rate 1.0** in all three arms. No fabrication
on a brain-unknown topic reaches the user, with or without the children.

**3. Known topics: no fabrication, but free generation replaces exact recall.** On the 12 known topics:
held_out_violation_rate **0.0** (the free reply never asserts a false held-out fact — moat-safe) in all arms,
but recall_preservation **0.0**. That 0.0 is largely a FORM-generator coverage fact, not a moat failure: the
WKV mouth's V=1000 TinyStories vocabulary structurally cannot name a Wikidata entity's real facts (5/12
replies went through it), and even the Qwen-written subset (`recall_preservation_qwen_only` 0.0) produced
conversational text rather than the exact stored `(agent, relation, patient)`.

## Verdict for the owner (#112)

**The bundle flip is fabrication-SAFE** — 0 fabrication reaches the user on any class (dangerous/unknown
abstain via the base filter; known never violates a held-out fact). **The moat children (NP_ENTAILMENT /
GEN_TIME_HONESTY) matter only for KNOWN topics** — they are inert on exactly the dangerous class the
flip-plan worried about, because the base filter fires first. So "flip the children WITH the parent for
dangerous-class safety" is unnecessary; the base filter already covers it.

**The real cost of the flip is NOT fabrication — it is a grounding regression on KNOWN topics.** Flipping the
bundle replaces the grounded strict/rich composer's exact-fact recall with free conversational generation
whose in-vocab mouth (WKV, V=1000) cannot express the 15k-Wikidata KB's facts. So on a topic the brain
actually knows, the default reply would become conversational-but-fact-thin rather than the exact stored fact.
That is the genuine owner UX call: prefer free open-ended conversation (accepting thinner factual grounding on
known topics until the WKV mouth's vocab/grounding covers the KB) vs keep grounded recall as the default.

## Next rungs

- The clean unlock is the WKV mouth's KB grounding/vocab coverage (so free generation on a known topic can
  still surface the brain's real fact) — then the flip is a strict win. Until then it is a UX tradeoff.
- A per-arm KNOWN-topic soak once the WKV mouth covers the KB would re-measure recall_preservation as a real
  (not form-limited) number.
