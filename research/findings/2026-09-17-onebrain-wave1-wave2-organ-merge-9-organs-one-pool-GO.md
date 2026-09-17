---
type: finding
status: verified
date: 2026-09-17
mechanism: onebrain organ-merge — wiring comprehension + source_provenance (Wave-1) then self_schema + curiosity +
  causal_whatif (Wave-2) onto the SINGLE shared spiking pool that already holds surprise/worldmodel/metacog/pragmatic,
  via each organ's get_organ() checking wave{1,2}_pool_enabled()/get_wave{N}_pool() (mirrors the shipped single_pool
  branch; additive + default-off)
integration_faculty: onebrain-merge-organs
lane: one-brain substrate consolidation
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO 6/6 for BOTH waves — the shared spiking pool now validates at NINE co-resident cortical organs (up from
  the 4 already default-on via single_pool). Wave-1 organread (comprehension + source_provenance + the 4 shipped):
  all_go 6/6 — every organ byte-identical merged-vs-coresident (max_coresidence_delta 0.0), answer-preserving, and
  faculty-alive, with the gain-0 freeze + legacy-diverges anti-cheats 6/6. Wave-2 organread (adds self_schema +
  curiosity + causal_whatif, 9 organs total): all_go 6/6, every one of the 9 organs alive + answer-same +
  coresidence-byte-identical, plus n_wave1_carried_read_byte_identical 6/6 (Wave-1's organs stay byte-identical under
  the larger pool). The wirings are CHERRY-PICKED to main DEFAULT-OFF (byte-identical when off); the DEFAULT-ON FLIP
  (making these organs share the pool in production) is a reviewed follow-on gated on the integrated /api/brain-chat
  no-regression battery (post-gaming). Wave-3 (+prospective_memory scope-reduced, +d6_multiref_wm, N=7002) organread
  is IN-FLIGHT on the pool.
runner: research/runners/_onebrain_wave1_organread_verify.py + _onebrain_wave2_organread_verify.py
artifacts:
  - research/findings/raw/_onebrain_wave1/organread_6seed.json
  - research/findings/raw/_onebrain_wave2/organread_6seed.json
external: NO-EXTERNAL-NEEDED — this consolidates ALREADY-de-risked organs onto one already-validated pooling framework
  (the single_pool pattern shipped default-on 2026-09-05); the organread gate is an internal byte-identity/answer/
  faculty-alive check, not a new mechanism.
builds_on:
  - research/findings/2026-09-02-onebrain-wave1-comprehension-provenance-merge-3seed-smoke.md
---

# One-brain organ merge — 9 cortical organs on ONE shared spiking pool, validated GO 6/6 (Waves 1+2)

The one-brain goal is to move every faculty onto the SHARED spiking substrate. The single_pool merge (surprise,
worldmodel, metacog, pragmatic) shipped default-on 2026-09-05. This lands the next two rungs, each wiring more organs'
get_organ() onto the same growing pool, mirroring that pattern (additive, default-off, byte-identical when off).

## Wave-1 (comprehension + source_provenance → 6 organs on the pool)

`research/findings/raw/_onebrain_wave1/organread_6seed.json`: all_go 6/6. Aggregate n_gate_a/b/c 6, n_gain0_freeze 6,
n_legacy_diverges 6, n_go 6. Every organ (the 4 shipped + comprehension + source_provenance) reports n_alive 6,
n_answer_same 6, n_coresidence_byte_identical 6, max_coresidence_delta 0.0. HONEST SCOPE: only comprehension's
get_organ() wiring is committed to production — source_provenance's PRODUCTION wrapper (SourceProvenanceHonestyMonitor)
has a different online-incremental API and was correctly NOT force-wired (a separate build); the organread GO here
validates the POOL mechanism for source_provenance's read organ, not the shipped wrapper's re-wiring.

## Wave-2 (self_schema + curiosity + causal_whatif → 9 organs on the pool)

`research/findings/raw/_onebrain_wave2/organread_6seed.json`: all_go 6/6. All NINE organs alive 6 / answer_same 6 /
coresidence_byte_identical 6; n_wave1_carried_read_byte_identical 6 (the earlier organs stay byte-identical as the pool
grows); gain-0 freeze + legacy-diverges 6/6. The 3 new organs' get_organ() wirings are committed default-off.

## Scope + follow-on

Committed to main default-off (acab6f4b2 Wave-2 organs, 223ed3995 Wave-1 comprehension). The MERGE is de-risked GO
for 9 organs; the production DEFAULT-ON flip (retiring the per-organ co-resident bridges for the pooled path) is the
follow-on, gated on the integrated /api/brain-chat battery + a wave flip-regression (post-gaming). Wave-3 (11 organs,
N=7002) organread is running on the pool; it appends when it lands. This is genuine one-brain consolidation: 9
cortical organs computing on a single shared spiking substrate, answer-preserving.
