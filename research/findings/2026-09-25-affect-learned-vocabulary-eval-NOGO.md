---
type: finding
status: verified
claim_check: measured
date: 2026-09-25
lane: A · Affect — appraisal vocabulary (rung (a) of the tone-selection PREREG's AMENDMENT 2)
seeds: [42, 43, 44, 100, 101, 102]
mechanism: research/runners/affect_learned_vocabulary.py behind BRAIN_AFFECT_LEARNED_VOCAB=1 (default-OFF). Word ->
  valence synapses onto a spiking V+/V- opponent pair, learned by a local instar rule from a heard fineweb-edu
  stream with the innate WARRINER seeds as the unconditioned stimulus; read by the production reader
  (`load_reader` / `appraise_text`'s own code path).
prereg: research/findings/2026-09-24-affect-learned-vocabulary-PREREG.md
verdict: NO-GO on the pre-registered gate. All preconditions hold (30/30, including the pinned
  `tests/test_affect_learned_vocabulary.py::test_off_byte_identical_to_pinned` re-run clean at this HEAD), so the
  reading is a real, decided NO-GO, not UNDEFINED. G1 (learning) and G4 (lesion) pass 6/6; G2 (shuffled-label
  control) passes only 4/6 (needs 5); G3 (neutral facts stay neutral) passes 0/6 (needs 6/6) — this is the
  decisive failure. Scored with the runner's own `score()` aggregate (`research/runners/_affect_learned_vocabulary_derisk.py`),
  fed a clean copy of the six `eval_s*.json` (see "Operational note" — the function's own glob also swallows
  `.prov.json` sidecars that postdate it and crashes; not a scoring-logic bug, worked around without editing the
  runner).
artifact: research/findings/raw/_affect_learned_vocab/run1/verdict.json (+ eval_s{42,43,44,100,101,102}.json in the
  same directory)
runner: research/runners/_affect_learned_vocabulary_derisk.py --score
---

# Learned affect vocabulary evaluation: NO-GO — held-out negatives are learned (G1, G4) but neutral facts are not kept neutral (G3)

## Census (completeness + liveness, verified before scoring)

All 6 registered seeds' `eval_s*.json` and `train_s*_r0-1-2-3-4-5-6-7-8.json` are on disk in the primary checkout's
`research/findings/raw/_affect_learned_vocab/run1/`, each with a `.prov.json` sidecar; all 8 shuffle-control
replica weight files (`shuf/weights_s{seed}_r1..r8.npz`, 48 files) per seed are present; `dev_s7/` (the seed-7
calibration artifacts named in the prereg) is present but is DEV, not a gate row, and was not used below. No
`_affect_learned_vocabulary` process is running: `ssh -n -F research/queue/.pool_ssh_config <node> 'ps -eo
etimes,args'` against pool1, pool2, pool41 and pool42 (read-only, all four reachable) shows zero matching
processes. The prereg's own text ("no evaluation seed has run") is stale relative to disk state, as the task
anticipated. All 141 files under `_affect_learned_vocab/` were copied byte-for-byte into this worktree's
`research/findings/raw/_affect_learned_vocab/` and committed; nothing in the primary checkout was moved or deleted.

## Provenance

5 of 6 arms' `.prov.json` (seeds 43, 44, 100, 101, 102) record a full 40-character `git_sha`
(`c0803dadcd729649e16a1de8c57cf571418c03d4`), `git_dirty: false`, `source_kind: "git_archive"`, and
`source_manifest_verified_at_start`/`_exit: true` — a clean, isolated pool dispatch. Seed 42's `.prov.json`
predates that isolation: it ran LOCALLY as the prereg's own staging plan directs ("Seed 42 locally, under
tools/memcap.sh"), so its working tree was dirty at run time (`git_dirty: true`, `source_kind: null`) and its
recorded `git_sha` is the short form `c0803dadc` — the same commit's 10-character prefix, confirmed by
`git merge-base --is-ancestor`, not a different revision. Stated plainly per the task's own escape clause: seed 42
predates the `git_archive` isolation rule by design, not by omission. All 6 seeds' `eval_s*.json` carry the
identical `vocab_shas` value `7f2a489f...18179dc`, confirming one shared heard stream. The commit named
"pinned revision: 1d5766620" in the task is the separate baseline `tests/test_affect_learned_vocabulary.py`
pins the flag-OFF path against (an ancestor of c0803dadc, unrelated to which commit generated these six runs);
re-running that pinned test at this worktree's HEAD passes clean (1 passed).

## The gate (runner's own aggregate, GATE thresholds from the prereg)

Per-seed sources (copied byte-for-byte into this worktree, `.prov.json` sidecars alongside each):
`research/findings/raw/_affect_learned_vocab/run1/eval_s42.json`,
`research/findings/raw/_affect_learned_vocab/run1/eval_s43.json`,
`research/findings/raw/_affect_learned_vocab/run1/eval_s44.json`,
`research/findings/raw/_affect_learned_vocab/run1/eval_s100.json`,
`research/findings/raw/_affect_learned_vocab/run1/eval_s101.json`,
`research/findings/raw/_affect_learned_vocab/run1/eval_s102.json`, aggregated into
`research/findings/raw/_affect_learned_vocab/run1/verdict.json`.

| seed | G1 learning | G2 shuffled-ctrl | G3 neutral facts | G4 lesion | neg_recall | neg_wrong | contrast_D | shuf_D_max | shuf_D_mean | g3_frac_within | g3_max_abs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42  | PASS | PASS | FAIL | PASS | 0.326923 | 0.086538 | 0.265947 | 0.080675 | 0.014013  | 0.725000 | 0.965298 |
| 43  | PASS | FAIL | FAIL | PASS | 0.365385 | 0.057692 | 0.280019 | 0.125469 | 0.057751  | 0.700000 | 0.806452 |
| 44  | PASS | PASS | FAIL | PASS | 0.250000 | 0.057692 | 0.225610 | 0.094747 | 0.010260  | 0.675000 | 0.635386 |
| 100 | PASS | PASS | FAIL | PASS | 0.442308 | 0.076923 | 0.369137 | 0.099906 | -0.005277 | 0.700000 | 1.000000 |
| 101 | PASS | PASS | FAIL | PASS | 0.365385 | 0.067308 | 0.280019 | 0.114681 | 0.029432  | 0.875000 | 0.891984 |
| 102 | PASS | FAIL | FAIL | PASS | 0.442308 | 0.048077 | 0.393527 | 0.125469 | 0.069975  | 0.725000 | 0.684262 |

<!--derived-->
Aggregate (needs G1/G2 >=5/6, G3 6/6, G4 6/6): **G1 6/6 PASS · G2 4/6 FAIL · G3 0/6 FAIL · G4 6/6 PASS.** GO
requires all four; two fail, so **verdict = NO-GO** (`research/findings/raw/_affect_learned_vocab/run1/verdict.json`,
top-level `"verdict": "NO-GO"`, `"gates": {"G1": true, "G2": false, "G3": false, "G4": true}`). All 30 preconditions
(6 seeds x 5, plus the single-vocabulary check) read `true` in the same artifact, so this is a decided NO-GO, not
UNDEFINED.

## What this shows

- **G1 and G4 are the strongest result in the arc**: on every one of 6 seeds, held-out negative words (never used as
  innate US afferents) are read negative at 25.0-44.2% recall with a wrong-sign rate <= 8.7%, well inside the
  pre-registered margins, and the lesion contrast is clean 6/6 — `neg_probe_negative_lesion == 0.0` on every seed
  while the intact arm exceeds the recall floor, and the runner's own INTEGRITY check
  (`g4.lesion_appraisal_identical_to_off`, `g4.lesion_neg_words_read_zero`) holds on all 6. The learning signal
  (co-occurrence with the innate WARRINER afferents through the depressed-synapse instar rule) really does
  transfer valence to words never taught it directly, and the effect is attributable to the learned synapses
  (`tools.lab.attributable_to` reads 1.0 on all 6 seeds), not to the read-out path.
- **G2 fails on exactly the mechanism the gate was built to catch**: seeds 43 and 102 fail not because the true
  contrast is weak (0.280019 and 0.393527, both above the 0.15 floor and above their own shuffled maxima) but because
  the mean of the 8 label-permuted replicas' contrast (0.057751, 0.069975) exceeds the 0.05 ceiling — i.e. on 2 of 6
  seeds, randomly re-labelling which words are "negative" still produces a non-trivial true/false separation,
  which is the register/frequency confound the prereg named as the standing risk for this method.
- **G3 is the decisive, uniform failure**: 0 of 6 seeds keep >=95% of the 40 neutral FACT_EVAL sentences within
  the affect ladder's |valence| <= 0.25 dead zone; the pass rate ranges 0.675000-0.875000 (6.5-13 of 40 sentences
  colored). The words that trigger it are not affect-adjacent at all — they are proper nouns and topical fillers:
  "france" (-0.46 to -0.57 across 4 of 6 seeds), "mozart" (+0.57 to +0.68), "germany"/"poland"/"austria" (jointly
  -0.50 to -0.61), "enrico" (-0.51 to +0.68, sign varies by seed), "boils", "second", "planet", "everest", "chess",
  "vinci"/"mona"/"lisa". This reproduces, on the EVAL half, exactly the DEV-stage failure the prereg logged before
  any evaluation data was read ("'does'/'planet'/'solar'/'students' read above 0.25") — the strong-affect margin
  V_MIN chosen on DEV neutral sentences does not generalize to a different set of neutral sentences; topical nouns
  co-occur often enough with the 80 innate WARRINER words in 1e9 characters of fineweb-edu to potentiate spuriously,
  and the sign of that potentiation is seed-dependent (not a fixed bias), consistent with a frequency/co-occurrence
  artifact rather than a stable learned meaning.
- **Reported, not gated** (per the prereg): the 9 NAMED clinically-relevant words are inconsistent across seeds —
  "unhappy" reads strongly POSITIVE on 4 of 6 seeds (+0.55 to +0.88), the wrong sign; "sadness" reads exactly 0.0 on
  all 6 seeds (never learned); "loneliness", "alone", "melancholy", "loss", "saddened", "decay" each read negative
  on at most 2-3 of 6 seeds and 0.0 on the rest. Held-out true-seed-word sign accuracy (reported) is 0.75-0.875 with
  only 7-14 of 24 words decided per seed — better than the DEV-stage 0.50, but on a small, seed-varying decided set.
  The 62 weak/neutral WARRINER words (never routed to the learned path in production) read nonzero through the
  learned path on 8.1-22.6% of words per seed, confirming they would need the same G3-style scrutiny if ever used.

## What this does not show

This does not show the mechanism cannot learn word valence — G1/G4 show it does, cleanly, for held-out words. It
does not show the register/frequency confound dominates the true signal — G2 fails on only 2 of 6 seeds and the
true contrast exceeds every shuffled replica on all 6. It does not bound how much of the FACT_EVAL contamination
would survive with the neutral-fact set enlarged, deduplicated of proper nouns, or corrected for co-occurrence
frequency (not attempted here; the gate as pre-registered treats any such correction as a new method, not a
re-score). It says nothing about the tone-selection probe's coverage floor (the prereg's post-hoc DEV note already
found no generator variant reaches it); that is a separate, un-gated instrument not re-run here.

## Operational note (not a gate finding)

`_affect_learned_vocabulary_derisk.py`'s `score()` globs `eval_s*.json` in the run directory; that pattern also
matches `eval_s*.json.prov.json` (the provenance sidecar convention postdates this runner), and the sidecar has no
`"seed"` key, so `score()` raises `KeyError: 'seed'` when pointed at the real, provenance-sidecarred directory as
committed. Worked around IN THIS WORKTREE ONLY (nothing in the primary checkout was touched) by temporarily moving
the six `eval_s*.json.prov.json` sidecars out of `run1/`, running `--score research/findings/raw/_affect_learned_vocab/run1`
directly (unedited scoring logic; the provenance door then auto-stamped the resulting `verdict.json` in place —
`research/findings/raw/_affect_learned_vocab/run1/verdict.json.prov.json`), then restoring the six sidecars. The
runner itself was not edited (out of scope for a scoring task); flagged separately for a follow-up fix.

## Next step (THE LAW: a NO-GO defers the METHOD, not the capability)

The prereg's own pre-registered next rung applies unchanged: "the sign of strong held-out seeds at 0.50 [DEV; now
0.75-0.875 EVAL] and the loss of signal with more text say the teaching signal (co-occurrence with 80 innate words
in [a fraction of] chunks) is too sparse. The rung is second-order conditioning, where learned words above V_MIN
become teachers themselves, raising the teaching density." G3's failure mode (topical proper nouns, not affect
words, potentiating) additionally suggests the next rung's teacher-selection needs a check that excludes
proper-noun/high-frequency-topical co-occurrence from what can become a second-order teacher, or the same
contamination will propagate. The longer-term surpass named in the prereg — a grounded (interoceptive/prosodic)
unconditioned stimulus in place of the innate WARRINER list — remains the structural fix for the register confound
that both G2's partial failure and G3's total failure point at.

## Honesty boundary

This measures a functional perception of word valence through a fixed gate on stored spiking-network reads.
Nothing here claims felt emotion, and no default was flipped: `BRAIN_AFFECT_LEARNED_VOCAB` stays default-OFF.

## Reproduce

```
# NOTE: move eval_s*.json.prov.json out of run1/ first (score()'s glob else raises KeyError: 'seed'; see
# "Operational note" above), then restore them after scoring.
.venv/bin/python -m research.runners._affect_learned_vocabulary_derisk --score research/findings/raw/_affect_learned_vocab/run1
.venv/bin/python -m pytest tests/test_affect_learned_vocabulary.py::test_off_byte_identical_to_pinned -q
```
