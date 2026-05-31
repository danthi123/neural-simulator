# P4 multi-hop directional fix: RESCUES the hub-crowding bottleneck per the frozen bar (multi-seed OUT full-2hop at fan-in 8 = 0.583 >= 0.50, vs undirected ANY = 0.000) -- the located cheap fix WORKS -- BUT the rescue is BIMODAL / seed-dependent (seed 42 = 8/8, seed 44 = 6/8, seed 43 = 0/8). The directional filter correctly isolates the outgoing edge on every seed; seed 43's total failure is weak UNDERLYING big_red binding on that seed's bridge, not a filter bug. So directional filtering removes the hop-2 hub-crowding bottleneck (clear improvement over undirected) and EXPOSES the residual per-seed binding-quality variance as multi-hop's next limit. Directional multi-hop is a REAL but NOT-UNIFORMLY-ROBUST capability.

> ## SUPERSEDED (2026-05-31): the real fix was the teacher_pA bug, not the directional filter
> The bimodality's root cause was found to be a teacher_pA bug (sparse encode_pair silently used teacher=100
> instead of the configured 500). Fixing it made undirected multi-hop at fan-in 8 jump 0.000 -> 0.750, which
> makes the DIRECTIONAL filter (this finding's subject) roughly NEUTRAL -- now slightly NEGATIVE (0.708 vs
> 0.750). The directional filter treated the SYMPTOM (hub-crowding); the teacher fix treated the CAUSE (weak
> bindings). See finding 2026-05-31-P4-multihop-POST-teacher-fix-the-bug-fix-was-the-real-win-directional-
> filter-now-neutral.md. The directional filter is retained (semantic + harmless) but de-framed from "the
> fix" to "a neutral semantic choice." This finding's RESCUED-but-BIMODAL result was at the BUGGY teacher=100;
> read it through this supersession.

**Date:** 2026-05-31
**Status:** Controller verdict on the decisive multi-seed (42/43/44) directional-fix run, the follow-up to the hub-reuse DEGRADES-WITH-FANIN finding. RESCUED per the pre-registered (unmoved) bar, with the bimodality foregrounded per scrutinize-a-PASS-harder-than-a-FAIL. Throwaway probe (research/findings/raw/_multihop_directional_probe.py); g20_multibridge.py byte-unmodified.

## The fix tested

The hub-reuse test located the DEGRADES bottleneck at hop-2 hub-crowding: querying a crowded hub returns its many INCOMING noun-edges and buries the one OUTGOING edge, because multitag retrieval is UNDIRECTED. The tags are name-ordered ("remember a is b" -> tag "a_b", cue-first), so the fix filters the hop-2 hub query to tags where the hub is the FIRST token (hub_*), isolating the outgoing edge. The probe runs BOTH direction='any' (reproduces DEGRADES) and direction='out' (the fix) in one controlled pass.

## Decisive measurement (full_2hop, per seed, fan-in 8)

| seed | ANY (undirected) | OUT (directional fix) |
|---|---|---|
| 42 | 0/8 = 0.00 | 8/8 = 1.00 |
| 43 | 0/8 = 0.00 | **0/8 = 0.00** |
| 44 | 0/8 = 0.00 | 6/8 = 0.75 |
| **mean** | **0.000** | **0.583** |

Full table (multi-seed mean full_2hop): fan-in 2 ANY 1.000 / OUT 1.000; fan-in 4 ANY 0.333 / OUT 0.333 (per-seed: 42 OUT 2/4, 43 OUT 4/4, 44 OUT 0/4 -- high variance, hop-1-limited); fan-in 8 ANY 0.000 / OUT 0.583.

PRE-REGISTERED VERDICT (bar set + frozen before the run): directional RESCUES if OUT full_2hop at fan-in 8 >= 0.50 multi-seed. 0.583 >= 0.50 => **RESCUED**. Bar NOT moved.

## Controller scrutiny (the PASS is real per the bar but BIMODAL -- foregrounded, not buried)

1. The fix mechanically works on every seed: the per-trial logs show direction='out' querying hub "big" returns ['red', ...] (red surfaces) on seeds 42 and 44, vs undirected [tree, river, apple] (red buried). The big_red tag is correctly isolated on ALL THREE seeds (HUB->C tag printed "big->red = big_red" each seed) -- so the filter mechanics are sound everywhere.

2. Seed 43's 0/8 is NOT a filter bug -- it is weak UNDERLYING binding. On seed 43's bridgeC the big_red engram does not recall "red" strongly enough to reach top-3 even when its tag is cleanly isolated. This is the project's pervasive per-seed structural variance (the sparse bridges are seed-specific; a given pair's binding strength varies by seed). Evidence it is binding-not-filter: seed 43 fan-in 4 directional is 4/4 (the hot_dry binding IS good on seed 43), but fan-in 8 big_red is 0/8 (that specific binding is weak on seed 43).

3. The PASS is FRAGILE at the bar: mean 0.583 is carried by seed 42 (1.00) + seed 44 (0.75); if seed 42 were ~0.50 the mean would dip below 0.50. So RESCUED is honest per the frozen bar but is NOT a claim of uniform robustness.

4. Unambiguous part: directional filtering is a STRICT improvement over undirected at fan-in 8 (0.583 vs 0.000) and never hurts (it only removes irrelevant incoming edges from the hub query; single-hop and low-fan-in are unchanged). The mechanism-level bottleneck (hub-crowding) is genuinely removed.

## Honest capability statement + disposition

Directional multi-hop reasoning on the validated multitag stack is a REAL capability: filtering the hop-2 hub query by encode-direction removes the hub-crowding wall that collapsed undirected multi-hop (0.000 -> 0.583 at fan-in 8). It is NOT uniformly robust: where the underlying pairwise binding is weak on a given seed, even a cleanly-isolated outgoing edge does not recall its target (seed 43 = 0/8). So the residual limit has MOVED from a mechanism-level problem (hub-crowding, now fixed) to a substrate-level one (per-seed binding-quality variance) -- the same variance that gates the whole conversational stack.

Disposition: the directional filter is a clean, validated, low-risk improvement (strict win, never hurts) -- worth integrating into the shipped query_concept as an optional directional mode + a multi-hop chat command, with HONEST framing (robust where binding is good; gated by per-seed binding quality). This is instrumental P4 capability progress on the working stack. The deeper biology-translatable deliverable remains the DG separation-vs-reliability BOUNDARY; the multitag stack (directional or not) is the oracle-shortcut-in-another-form, not a substrate biologization.

## Discipline

Throwaway probe only; g20_multibridge.py byte-unmodified; reuse-by-import; GPU/CuPy (no numpy). Pre-registered three-state bar set before the run, NOT moved by the bimodal result (0.583 >= 0.50 = RESCUED is reported as-is, with the bimodality + fragility foregrounded). The smoke (seed 42) was run before the decisive multi-seed (grounding-before-decisive). The PASS was scrutinized harder than a FAIL: the seed-43 total failure was diagnosed (binding-quality, not filter) rather than averaged away. Wall 1225s (~20 min), 3 seeds.
