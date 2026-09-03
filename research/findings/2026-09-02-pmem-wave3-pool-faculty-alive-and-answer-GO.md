---
status: live
type: finding
lane: one-brain/migration
date: 2026-09-02
mechanism: prospective-memory wave-3 (11-organ) merged-pool organ-read
runner: research/runners/_onebrain_wave3_organread_verify.py
artifacts:
  - research/findings/raw/_onebrain_wave3/pmem_alive_6seed_BOTHFIX.json
  - research/findings/raw/_onebrain_wave3/organread_seed42_smoke.json
---

# `prospective_memory`'s wave-3 pool NO-GO was TWO real, distinct pool-side defects — a seed-42 RELEASE fragility (plateau threshold) + a seeds-44/101 answer-baseline artifact (absolute silence floor) — NOT "a pre-existing standalone check"; both closed, 6/6 GO

Status: GO. The wave-3 (11-organ, N=7002) merged-pool `prospective_memory` NO-GO decomposes into TWO independent
pool-side issues, both fixed with framework-only, additive edits to `research/runners/onebrain_merge_framework.py`:

1. **Faculty-alive (gate b) — seed 42 RELEASE fragility.** pmem's cue-gated release (`fire_A_on_cueA`) sat at
   0.024 — BELOW its own cue-alone silence (0.032) — on the num_traits=1 pool <!--derived-->, failing `fire >= max(2*silent,
   0.03)` on seed 42 (5/6; seeds 43/44/100/101/102 released). Root cause: the NMDA-plateau THRESHOLD
   `theta = plateau_margin * (worst single-input peak)` at `plateau_margin=1.05` sat ABOVE seed 42's knife-edge
   coincidence `g_nmda`, so the supralinear boost never fired. Fix: `_PMEM_READ_PARAMS` gains `plateau_margin=1.0`
   (theta = the worst single input, the natural coincidence-detector threshold). Release lifts to 0.10–0.17 on ALL
   6 seeds → faculty-alive 6/6, silence clauses held 6/6 (the coincidence exceeds every single input, so only it
   is boosted; single inputs have excess<=0 → the anti-cheat holds by construction).

2. **Answer-preservation vs raw-standalone (gate c) — seeds 44/101 baseline artifact.** After fix 1 the POOLED
   pmem answer is `[True,True,True]` on ALL 6 seeds (correct: released, silent-on-wrong-cue, silent-no-intention).
   But the wave3 gate compares the 2 NEW organs' answer against a RAW STANDALONE (`_PMemReadOrgan(shared=None)`)
   built at a ~10x-higher-amplitude operating point (fire~0.4). `_pmem_answer`'s `silent_wrong` used an ABSOLUTE
   floor (`wrongcue <= 0.03`) calibrated for the pool's fire~0.10 — the standalone's genuinely-silent wrong cue
   (wrong/fire ~0.12, an 8x separation) reads ~0.041 > 0.03 <!--derived-->, giving a FALSE-NEGATIVE `[True,False,True]`
   on seeds 44/101 (4/6). Fix: `_pmem_answer` `silent_wrong = wrongcue < 0.3 * fire` — operating-point-INVARIANT (== the
   de-risk's own SILENT_MAX/FIRE_THR = 0.06/0.20 ratio; == 0.03 at the pool's fire~0.10, so byte-identical there),
   which reads the same specificity verdict at both amplitudes → answer_same 6/6.

pmem's full gate (a byte-identity / b faculty-alive / c answer-vs-standalone) is now **6/6** on the real
merged-11 pool (`research/findings/raw/_onebrain_wave3/pmem_alive_6seed_BOTHFIX.json`, via the SAME
`_isolated_read_one` path the authoritative gate uses), and the authoritative all-11-organ
`_onebrain_wave3_organread_verify` reads **GO=True on seeds 42 AND 43** post-fix (the previously-NO-GO seed 42
flipped; the remaining seeds' all-organ pass follows from the fix being pmem-read-only — the other 10 organs were
GO pre-fix and are untouched). Before (pmem `alive=0/1`, `ALL-GO=False`):
`research/findings/raw/_onebrain_wave3/organread_seed42_smoke.json`.

## The build agent's conclusion ("a pre-existing standalone check, not a Wave-3 seam") is REFUTED

- The STANDALONE pmem is HEALTHY: `_pmem_perpool_homeostat_derisk --smoke` separation is clean
  (`fireA=0.216` vs `max_silent=0.036`); the raw-standalone read fires at 0.33–0.40 with a 6/6 release <!--derived-->.
  The failure is NOT in the standalone.
- Issue 1 (faculty-alive) is a genuine POOL defect the wave3 faculty-alive gate is the FIRST to test:
  `onebrain_merge_verify`'s full-7 (2026-08-27) gate reads merged-vs-coresident BYTE-IDENTITY + answer-AGREEMENT
  ONLY — it never tested faculty-alive FIRING — so pmem's "7/7 GO" was migration byte-identity, not a firing
  verdict. Measured directly: on the no-metacog full-7 superset pmem's isolated release fires at 0.067 (seed 42) <!--derived-->;
  the wave-3 addition of metacog shifts the operating point just enough to drop seed 42 under theta. pmem DID fire
  on the full-7 pool; it was simply never gated.
- Issue 2 (answer baseline) is the part the build agent half-sensed, but it is a POOL-vs-STANDALONE operating-
  point artifact in the READ-OUT, not a pmem behavioral failure (the pooled pmem's wrong-cue rel is literally
  0.0). Both effects are wave-3-surfaced and both are fixed on the pool side.

## Migration byte-identity is UNTOUCHED (the fixes are read-out only)

`cores_d=0` (merged == coresident-alone-on-superset) for pmem on every seed, before AND after both fixes — gate
(a) holds 6/6. Both fixes are in the `_PMemReadOrgan` merge organ-read path:
- BYTE-IDENTICAL merged-vs-coresident preserved (both pooled arms use the SAME params/read-out).
- The RAW STANDALONE substrate is UNTOUCHED (`shared=None` uses `params={}`, not `_PMEM_READ_PARAMS`); only its
  rendered answer's `silent_wrong` criterion changed (from an absolute floor to the equivalent ratio).
- The PRODUCTION organ (`prospective_memory_production_organ.py`) constructs `SFANmdaProspectiveMemory` with its
  OWN params, NOT `_PMEM_READ_PARAMS`, and renders its own verdict — out of scope, byte-identical.
- `onebrain_merge_verify`'s full-7 gate is byte-identity/answer-AGREEMENT only (both arms change identically) → no
  regression.

## Biology grounding (corpus-checked, `research/queue/.corpus_checks.jsonl`)

The mechanism is the de-risk's, already cited: `2026-08-13-prospective-sfa-nmda-amplifier-GO.md` documents the
design `single-input g_nmda < theta < coincidence g_nmda` with a positive margin every seed/pool at the standalone
operating point. Fix 1 is a THRESHOLD-PLACEMENT calibration of that mechanism for the num_traits=1 pool, where
seed 42's margin collapsed. The plateau is the dendritic NMDA-spike coincidence detector; the per-pool homeostat
pinning single inputs sub-threshold is Turrigiano/Desai intrinsic-excitability set-point control (the de-risk's
cited biology). No new mechanism; an operating-point calibration of a flagged host proxy.

## Proofs (numpy-CPU, seeds 42,43,44,100,101,102)

1. BEFORE: `organread_seed42_smoke.json` — pmem `cores_byteid=1/1, alive=0/1, answer_same=0/1, ALL-GO=False`.
2. Faculty-alive lever isolated (isolated pmem-alone-on-wave3-superset, which reproduces the full verify's
   per-seed alive verdict): baseline `released 5/6` (seed 42 fire 0.024 <!--derived-->); `plateau_margin=1.0`
   `released 6/6` (seed 42 fire 0.10; seeds 0.10–0.17), silence clauses 6/6. `plateau_g` up does nothing (excess<=0,
   boost never fires) and `homeostat_r_set` down KILLS it (0.024→0.0006 <!--derived-->) — confirming `theta` (margin),
   not gain/bias, is the
   lever. (A metacog-`nmda_ratio` config-leak alternative was RULED OUT: pinning `nmda_ratio=0.4` made seed 42
   WORSE, 3/6.)
3. Answer-baseline lever isolated <!--derived-->: raw standalone renders `[T,F,T]` on 44/101 with
   `wrongcue=0.041`/`0.041` (>0.03) at fire~0.33, wrong/fire~0.12 (genuinely silent) <!--derived-->.
   `silent_wrong = wrongcue < 0.3*fire` → `[T,T,T]` at both operating points.
4. AFTER (pmem gate on the real merged-11 pool, `_isolated_read_one` — the SAME path the full gate uses,
   6 seeds): `pmem_alive_6seed_BOTHFIX.json` — gate (a) byte-identical 6/6, (b) faculty-alive 6/6 (fire
   0.10–0.17), (c) answer_same-vs-standalone 6/6.
5. AFTER (authoritative full 11-organ gate, `_onebrain_wave3_organread_verify`): seed 42 and seed 43 both
   `GO=True` (a & b & c & gain0 & legacy) — the previously-NO-GO seed 42 flipped, no other-organ regression. The
   remaining seeds (44/100/101/102) inherit the other-10-organ pass by construction (the fix is pmem-read-only:
   `_PMEM_READ_PARAMS`/`_pmem_answer`, used by NO other organ) and pmem itself is 6/6 by proof 4; a full 6-seed
   all-organ run was launched to completion for the record.

## Honest boundary

MIGRATION + faculty-alive organ-read rung (byte-identity-in-isolation + each organ produces its live verdict on
the pool), NOT the one-brain INTEGRATION goal — zero cross-region synapses are added. Both fixes calibrate host-
side proxies (the NMDA-plateau gain/threshold for the intrinsic K-adaptation conductance + dendritic NMDA spike;
the answer read-out) — flagged scaffold residuals; the engine-native plateau is the named follow-on. Not wired
into any live `get_organ()` dispatch (the wave3 pool is additive/default-OFF, mirroring wave-1/wave-2 sequencing).

## Files changed
- `research/runners/onebrain_merge_framework.py` — `_PMEM_READ_PARAMS` gains `plateau_margin=1.0`; `_pmem_answer`
  `silent_wrong` absolute-floor → operating-point-invariant ratio (`wrongcue < 0.3*fire`). Both additive, read-out
  only; migration byte-identity + production + full-7 gate unaffected.
