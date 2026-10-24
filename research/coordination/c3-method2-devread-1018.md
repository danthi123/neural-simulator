# C3 method-2 dev read (background OU noise OFF) — PROVISIONAL

**Date:** 2026-10-18 · **Read by:** one MiMo Pro delegate (RULING models), driver-persisted · **Status:** PROVISIONAL dev-read, dev seeds 1-8 only — NOT a GO, NOT a gate result.

## Preconditions

- Method-2 artifacts (flag ON): `research/findings/raw/_curiosity_lp/m2_dev/battery_m2_dev_s1.json` .. `battery_m2_dev_s8.json`, all with `*.prov.json`, at revision `58c1598f10b2f5ea4508ccd20bab301d774a6bf9` (`research/c3-method2-noou`), `PYTHONHASHSEED=0` recorded by the sidecar env filter; formal 3-run byte-gate PASSED 2026-10-18 12:42 (byte-identical modulo `wall_s`).
- Method-1 reference (flag OFF, frozen): `research/findings/raw/_curiosity_lp/battery_dev1-8.json` (aggregate verdict NO-GO at `w_topic_ask=80`, `tau_hab_ms=12000`).
- Bars spec: `docs/drafts/curiosity-lp/PLAN.md`; bar strings and the aggregate formula: `research/runners/_curiosity_lp_trace_derisk.py` (~lines 175-200, ~700-710; `tools/lab.py attributable_to = (t-c)/t`).
- Config equality checked: both arms `w_topic_ask=80.0`, `tau_hab_ms=12000.0`.

## Sanity

PASS. All 8 method-2 artifacts carry `no_ou: true`; method-1 has no `no_ou` key (flag off). Seeds 1-8 present exactly once each, no duplicates/missing. Method-2 files carry `bars` but no `aggregate`; aggregates below were combined from the 8 per-seed records with the runner's own formulas.

## Per-bar table

```
Bar                     method-1 (1-8)     method-2 (1-8 combined)  pass condition                       verdict
B1_first_novel_asked    1/8 fail           8/8 true                 asked on g_eel, 8/8                  FLIPPED fail->pass
B1b_every_first_gap     0/8 fail           8/8 true                 first probe of every novel topic,8/8 FLIPPED fail->pass
B2_stop_by_4th          8/8                8/8                      not asked on u_toad_4, 8/8           HELD pass
B3_answered_not_reasked 8/8                8/8                      not re-asked on lp_ibex_2, 0/8       HELD pass
B4_moats_silent         8/8 (moat_asks=0)  8/8 (moat_asks=0)        0 of 24 moat asks                    HELD pass
B5_lesion_continues     8/8                8/8                      lesion asks u_toad_4 8/8, differs 8/8 HELD pass
B5_attribution          t=32 c=0 f=1.0     t=22 c=0 f=1.0           fraction >= 0.5                      HELD pass (1.0 both)
B6_perm_order           0/8 fail           8/8 true                 B1b,B2,B3,B4 hold, 8/8              FLIPPED fail->pass
B7_perm_exposure        1/8 fail           8/8 true                 >=2 of m_moose/m_goat/m_yak and     FLIPPED fail->pass
                                                                    none of g_eel/g_crab/g_newt, 8/8
B8_null                 8/8                8/8                      intact_b == intact vector, 8/8      HELD pass
B9_drive_lesion         8/8                8/8                      0 asks every turn, 8/8              HELD pass
```

Per-seed consistency (method-2): every bar 8/8 — no 7/8 anywhere, seeds 1-8 identical on all bars. (Method-1 per-seed: B1 true only seed 8; B7 true only seed 4.) Informational: `stage2_c3` discriminates 1/8 -> 8/8; `asks_before_stop` 0 on all method-1 seeds -> 1-2 on method-2 seeds (2,1,1,1,2,1,1,1).

## Aggregate

Method-1 frozen: NO-GO — four bar failures (B1 1/8, B1b 0/8, B6 0/8, B7 1/8); B2/B3/B4/B5/B8/B9 8/8. Method-2 combined: every bar 8/8 — all four failures flip, holds hold, `B5_attribution` fraction 1.0. Under method-2 the all-bars-hold pattern the aggregate verdict logic keys on (`_curiosity_lp_trace_derisk.py` line 710) is reached on dev seeds 1-8.

## Verdict (PROVISIONAL)

Turning the OU background noise off (method-2, `--no-ou`) flips B1, B1b, B6 and B7 from fail to 8/8 pass while B2, B3, B4, B5, B8 and B9 hold unchanged: the frozen method-1 NO-GO does not carry over to method-2, and the OU background process is the leading dev-read suspect for the four method-1 failures. Dev seeds 1-8, single-flag change, no six-seed claim. Gate-seed prereg and any bar/scorer change remain Claude's (section 2).