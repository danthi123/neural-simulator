---
type: finding
status: verified
date: 2026-09-17
mechanism: retire multi_turn_agent_v2.py's separate host anaphor word-list (_ANAPHORS = {"it","that","them","they",
  "this"}) — the SAME spiking CA3 pattern-completion anaphor detector already shipped as V1's sole detection path is
  wired as V2's sole path; the host list is DELETED (no fallback, mirroring the 2026-09-16 V1 host-removal)
integration_faculty: spiking-anaphor-detection (research variant multi_turn_agent_v2)
lane: language (scaffold-retirement follow-on)
seeds: [42, 43, 44, 100, 101, 102]
verdict: RETIRED (research-only). The 2026-09-16 V1 host-removal finding named multi_turn_agent_v2.py's own separate
  host anaphor list as the un-retired follow-on. multi_turn_agent_v2 is research-only (imported by 3 research runners:
  _phaseB_cross_sentence_coherence_derisk, _phaseB_reconsolidation_update_derisk, multi_turn_ordered_wm_demo; NOT the
  webapp production path). The spiking detector's ANAPHORS list is provably identical to the deleted host set, so the
  swap is answer-preserving: a 6-seed byte-identical differential (narrate() + resolve() over a full multi-turn
  scenario, both host-list and spiking-organ paths) is IDENTICAL at every scenario field and every token, all 6 seeds
  (all_scenario_identical + all_token_level_identical + byte_identical all True). The host `_ANAPHORS` set is DELETED;
  the spiking CA3 organ is V2's SOLE anaphor-detection path (a substrate error propagates, no host fallback — exactly
  like V1). Not a headline scaffold_retired increment (V2 is a research variant, not a production faculty ledger row);
  it is the clean follow-on the V1 finding named.
runner: research/runners/_v2_anaphor_retire_verify.py
artifacts:
  - research/findings/raw/_v2_anaphor_retire_verify/differential_result.json
external: NO-EXTERNAL-NEEDED — a host-scaffold DELETE confirmed answer-preserving by a byte-identical differential; the
  spiking CA3 pattern-completion detector it uses is already de-risked (2026-09-09 GO) + shipped default-on in V1.
builds_on:
  - research/findings/2026-09-16-host-removal-novelty-anaphor-qroute-RETIRED-byte-identical-differential-GO.md
---

# multi_turn_agent_v2 host anaphor word-list RETIRED — spiking CA3 detector is the sole path (byte-identical)

V1 (`MultiTurnAgent`) retired its host anaphor word-list on 2026-09-16, moving anaphor DETECTION onto the spiking CA3
pattern-completion organ (`spiking_anaphor_detection_organ.SpikingAnaphorDetectorOrgan`), and that finding explicitly
named V2's own separate `_ANAPHORS = {"it","that","them","they","this"}` list as the un-retired follow-on. This closes
it.

## What changed (commit 661901a9a)

`multi_turn_agent_v2.py`: the host `_ANAPHORS` set is DELETED; `_anaphor_is()` (a verbatim mirror of
`MultiTurnAgent._anaphor_is`) lazily builds the spiking CA3 detector once per session and is the SOLE detection path —
a substrate error propagates, no host fallback. The organ's `ANAPHORS` list is identical to the deleted host set, so on
clean typed anaphors + non-anaphor content words the spiking decision agrees token-for-token.

## Verify (research/findings/raw/_v2_anaphor_retire_verify/differential_result.json)

A 6-seed byte-identical differential runs a full multi-turn scenario (narrate + resolve + moat/abstain + a
non-anaphor passthrough) through BOTH the retired host-list path and the spiking-organ path and compares every
scenario field + every token: `all_scenario_identical=True`, `all_token_level_identical=True`, `byte_identical=True`,
verdict GO, across seeds 42/43/44/100/101/102. The retirement is answer-preserving.

## Scope

multi_turn_agent_v2 is a research variant (3 research importers; not the webapp production path), so this is not a
headline scaffold_retired increment — it is the clean host-scaffold follow-on the V1 finding named, removing a
duplicate host cognition path so the one spiking detector is the sole anaphor-detection substrate across both agents.
