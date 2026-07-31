---
type: finding
status: corrected
date: 2026-05-26
---

# Direction 4 NEGATIVE INVALIDATED: same class of cross-bridge uniformity bug as Direction 5 had — activity vectors at each word position are BYTE-IDENTICAL across all 5 bridges (L2 = 59.25 / 60.39 / 67.59 by position; matches exactly across A_nouns / B_verbs / C_adj / D_spatial / E_functional). The D4 substrate-geometry conclusion is therefore unsupported; D4 needs a re-test after fixing cross-bridge uniformity (analog of the D5 sparse-pattern fix but in a different code path).

**Date:** 2026-05-26 ~00:30 EDT
**Status:** DIRECTION_4_NEGATIVE INVALIDATED at the diagnostic level; the underlying conclusion "bio_brain_regions cross-bridge does NOT engage" cannot stand on this data; D4 re-test with cross-bridge uniformity fix is required to determine the true status.

## What the bug looks like in D4

Inspecting D4 smoke activity caches (research/findings/raw/direction_4_cache/activity_smoke_*_seed42.npz):

```
Position 0 (first word of each bridge):
  A_nouns[apple]:   shape=(8, 1600) mean=0.0743 L2=59.2537
  B_verbs[go]:      shape=(8, 1600) mean=0.0743 L2=59.2537  <-- IDENTICAL
  C_adj[big]:       shape=(8, 1600) mean=0.0743 L2=59.2537  <-- IDENTICAL
  D_spatial[north]: shape=(8, 1600) mean=0.0743 L2=59.2537  <-- IDENTICAL
  E_functional[i]:  shape=(8, 1600) mean=0.0743 L2=59.2537  <-- IDENTICAL

Position 1:
  All 5 bridges: mean=0.0780 L2=60.3904  <-- IDENTICAL across bridges

Position 2:
  All 5 bridges: mean=0.0938 L2=67.5944  <-- IDENTICAL across bridges
```

Direct verification: `np.array_equal(A_nouns[apple], B_verbs[go]) == True`. The activity vectors are byte-identical.

WITHIN a bridge, different word positions DO have different activity (position 0 != position 1), so the bug is specifically the cross-bridge uniformity, not a global flat-line.

## Same class of bug as D5

D5 had: K-of-N sparse patterns generated with seed=base_seed → all 5 bridges produced identical patterns → cross-bridge discrimination mathematically impossible. **Fix**: bridge-specific seed offsets (commit c4e18f2).

D4 has: orthogonal lang_input codes for each word position AND lang_input→pool weight matrices both seeded by the same base_seed across bridges. Position-i word in A_nouns has the SAME orthogonal code AND the SAME randomly-initialized weight matrix as position-i word in B_verbs. The captured pool activity ends up byte-identical.

The D4 cross-bridge probe constructs the 80-concept union by concatenating per-bridge activity, but since the activity is repeated 5x (16 distinct patterns repeated across 5 bridges = only 16 distinct positions in the union), discrimination over an apparent 80-concept space is mathematically impossible.

## Implications

1. **The 2026-05-25 D4 NEGATIVE finding is INVALIDATED.** It documented "bio_brain_regions cross-bridge does NOT engage" — that conclusion is unsupported because the probe was operating on duplicate inputs.

2. **The 2026-05-25 D4 global_mean diagnostic is also INVALIDATED** (it concluded the cheap centring fix doesn't help because the substrate is geometry-limited; actually the inputs were identical so no centring could help).

3. **D5's "REVERSAL of substrate-geometry hypothesis" is now strengthened**: both D4 and D5 had cross-bridge uniformity bugs; D5's bug was caught and fixed; D4's bug is now diagnosed but not yet fixed. The substrate-geometry hypothesis is even less supported than I claimed in the D5 reversal findings doc.

4. **Pillar n=105 (D3 V=32) is UNAFFECTED**: D3 is a SINGLE-substrate result, not multi-bridge. Verified by independent diagnostic: D3 V=32 cosine(north, east) = 0.1555 (low; words are distinct within-substrate).

5. **Pillar n=95 (G.20 sparse cross-bridge) is UNAFFECTED**: G.20 sparse uses K-of-N patterns derived per-(bridge, seed) — distinct file size + distinct patterns verified earlier in the arc.

6. **Direction M (320-concept production deliverable) is UNAFFECTED**: G.20 sparse 5-bridge production tier uses different file sizes per bridge (55891-55892 KB), suggesting distinct training; Direction R-v3 capacity envelope passed.

## Fix needed

The D4 5-bridge runner needs to either:
- (a) Use bridge-specific seeds for build_biological_brain_regions (parallel of D5's seed-offset fix; substantial since the protected builder takes a single seed)
- (b) Use bridge-specific orthogonal codes (modify the code-generation step to incorporate bridge identity)
- (c) BOTH

The simplest fix: pass `seed = base_seed + _BRIDGE_OFFSETS[bridge_label]` to BOTH the bridge construction AND the orthogonal-code generation. Mirrors D5's fix.

## Pre-registered next concrete actions

1. **Apply D5-analog seed-offset fix to D4 5-bridge runner** (~30 min CPU + verification)
2. **Re-train D4 smoke** with the fix (~80-95 min GPU; would compete with in-flight D5 production; queue for when D5 production completes)
3. **Re-run D4 cross-bridge probe** on the bug-fixed cache
4. **If D4 bugfix smoke is PASS or PARTIAL**: D4 also achieves cross-bridge composition; pillar n=95-analog on bio_brain_regions dedicated-pool architecture
5. **If D4 bugfix smoke is still NEGATIVE**: the dedicated-pool-only architecture genuinely doesn't support cross-bridge composition (confirms the D5 hybrid's shared sparse pool is necessary)

## Discipline

- Bug discovered through user-prompted re-diligence ("looks like it finished and you failed to pick up on it?" → cascaded into re-checking D4 with the same diligence applied to D5)
- Same class of systematic bug, same diagnostic approach (cross-bridge equality check on cached activity)
- Pillar n=105 verified UNAFFECTED (single-substrate; different word activity)
- Direction M deliverable UNAFFECTED (different file sizes per bridge)
- Bar UNCHANGED 0.80 throughout
- Honest propagation: D4 NEGATIVE finding is INVALIDATED (not silently revised); the D4 re-test is queued

## Files

- D4 cache directory: `research/findings/raw/direction_4_cache/`
- Original D4 NEGATIVE finding (now INVALIDATED): `research/findings/2026-05-25-DIRECTION-4-5bridge-SMOKE-NEGATIVE-bio_brain_regions-cross-bridge-doesnt-engage-multi-seed-chance-level.md`
- D4 global_mean diagnostic (also INVALIDATED): commit ca5b000
- D5 bug fix that revealed the pattern: commit c4e18f2
- D5 bugfix PARTIAL finding (now strengthened): `research/findings/2026-05-25-DIRECTION-5-HYBRID-BUGFIX-PARTIAL-pattern-uniqueness-bug-fixed-substrate-is-NOT-the-limit-mirror-of-pillar-n95.md`
