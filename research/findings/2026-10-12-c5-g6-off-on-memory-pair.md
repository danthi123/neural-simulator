---
type: finding
status: provisional
claim_check: measured (exact integers read from the two staged c5pair RSS sidecar copies named in artifacts; delegate verdict per RULING models; C5 status per RULING c5-pair-read-1012; awaiting Claude review)
date: 2026-10-12
lane: c5 / g6 baseline (the idle-replay organ's memory cost; C5 status UNDEFINED per RULING c5-pair-read-1012)
mechanism: The g6 OFF/ON whole-brain battery pair on one AWS r7i.4xlarge (rev 18ec1f86d, seed 42, arms concurrent) measuring the organ's per-turn RSS/hwm cost against the <= 2 GB/turn target; idle replay never fired, so this settles the OFF-curve and the resident cost, not C5
seeds: [42]
artifacts:
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair.json
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair.json
  - research/findings/raw/_regression_battery/c5_offcurve_diag/c5pair_PROVENANCE.txt
  - research/findings/raw/_regression_battery/c5_offcurve_diag/c5pair_free_at_end.txt
verdict: C5 = UNDEFINED (RULING c5-pair-read-1012: the replay-fired precondition fails, circuits_built=0 throughout, OFF/ON replies byte-identical). Within scope: resident ON-OFF max 1332 KiB vs the 2 GB/turn target = met, organ quiescent, single seed 42. <!--derived-->
---

# RESULT: g6 OFF/ON whole-brain battery pair — the idle-replay organ's memory cost is ~zero

**Date:** 2026-10-12 · **Status:** PROVISIONAL (agent-filed, verdict pending Claude's review) ·
**Seeds:** 42 (single seed; no generalization claim)

**C5 STATUS: UNDEFINED (RULING `c5-pair-read-1012`): idle replay never fired in g6 — the ON
sidecar shows `circuits_built=0` throughout and the OFF/ON replies are byte-identical — so the
replay-fired precondition fails and this is NOT a C5 memory pass.** Within its own scope the
ordered ≤ 2 GB/turn <!--derived--> measurement passes: one MiMo Pro delegate (RULING `models`) read
max ON−OFF = 1332 KiB <!--derived--> (0.0013 GiB <!--derived-->, ~1500x <!--derived--> headroom);
the organ was loaded-but-quiescent, so this bounds resident/loaded cost only, and single seed 42
bars any generalization.

## Per-turn numbers (exact integers)

<!--derived--> each cell below is a readout of
research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl and
research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl (copied from the
AWS landing folder named in c5pair_PROVENANCE.txt); the ON−OFF columns are their differences.

| turn | OFF rss_kb | ON rss_kb | ON−OFF rss (KiB) | OFF hwm_kb | ON hwm_kb | ON−OFF hwm (KiB) |
|---|---|---|---|---|---|---|
| emo | 6234688 | 6236020 | +1332 | 6393712 | 6393948 | +236 |
| bc_a | 11690200 | 11690860 | +660 | 11783108 | 11783064 | −44 |
| bc_b | 11689416 | 11690064 | +648 | 11783108 | 11783064 | −44 |
| rich_well | 17075536 | 17075724 | +188 | 17224424 | 17223756 | −668 |
| rich_open | 22416700 | 22415792 | −908 | 22459292 | 22458480 | −812 |

<!--derived--> Target check: 1332 KiB ≤ 2097152 KiB (2 GB) on rss and 812 KiB ≤ 2097152 KiB on hwm,
all five turns.

## Preconditions

- Both arms (pid 5160 OFF, pid 5161 ON, organ hosted in-process
  <!--derived--> research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl)
  ran **concurrently** on one AWS r7i.4xlarge (128 GB <!--derived-->), revision `18ec1f86d`, seed 42,
  launched by Claude per RULING `c5-off-baseline`; `pair.done` present, both sidecars carry all five
  `battery_turn` records (research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl,
  research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl).
- End-of-battery memory: 121 GB free of 123 GB <!--derived-->
  research/findings/raw/_regression_battery/c5_offcurve_diag/c5pair_free_at_end.txt — no memory
  pressure, so neither arm was clipped by the host (unlike the workstation's 20 GB <!--derived-->
  user-slice OOM at rich_open that killed the earlier cap-18/cap-30 arms; environment, not a
  property of the pair — research/coordination/LOCAL_LLM_RUNBOOK.md ruling `wspool-kills-1012`).
- The ≤ 2 GB/turn <!--derived--> organ-cost target is the one fixed by RULING `c5-wspool-26-1012`
  (research/coordination/LOCAL_LLM_RUNBOOK.md). The C5-level precondition (replay fired in ON) is
  NOT met here — see the C5 STATUS line and RULING `c5-pair-read-1012`.

## Scope (delegate + rulings)

- Measured: per-turn end-of-turn `rss_kb` and `hwm_kb`, all five g6 battery turns (emo, bc_a, bc_b,
  rich_well, rich_open), arms concurrent on the 128 GB host.
- Shows: the organ's resident/loaded memory cost is effectively zero — max |ON−OFF| = 1332 KiB rss
  <!--derived-->, 812 KiB hwm <!--derived-->, vs the ≤ 2 GB/turn target (met by >1500x
  <!--derived-->).
- Does NOT show: active replay cost. The organ was quiescent during the battery
  (`circuits_built=0`, `circ_k=null` in all five organ turn records <!--derived-->
  research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl — the battery
  drives the brain continuously, so idle-replay never triggers). This bounds loaded/resident cost
  only.
- Single seed (42): the six-seed rule (42/43/44/100/101/102) applies before any broader claim.
- The workstation deaths (20.9 GB anon-rss vs a then-20 GB <!--derived-->
  research/coordination/LOCAL_LLM_RUNBOOK.md ruling `wspool-kills-1012`) were environment; the AWS
  pair is the measurement of record.
- The OFF-curve this pair settles (RULING `c5-pair-read-1012`): RSS 6.2/11.7/11.7/17.1/22.4 GB
  <!--derived--> with the flag OFF, about +5.3 GB per new turn KIND (bc_a→bc_b adds 0
  <!--derived-->) — battery/worker retention, a separate efficiency question, not C5
  (research/coordination/LOCAL_LLM_RUNBOOK.md).

## Provenance

`AWS r7i.4xlarge (128 GB), rev 18ec1f86d, g6 OFF and ON run concurrently with the battery_turn +
organ RSS sidecars (RULING c5-off-baseline); launched by Claude 2026-10-12` — quoted verbatim from
the landing folder's PROVENANCE.txt, copied into this finding's artifacts below as
c5pair_PROVENANCE.txt (the 304 MB per-arm logs stayed in the landing folder).

artifacts:
- research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl
- research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl
- research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair.json
- research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair.json
- research/findings/raw/_regression_battery/c5_offcurve_diag/c5pair_PROVENANCE.txt
- research/findings/raw/_regression_battery/c5_offcurve_diag/c5pair_free_at_end.txt