---
type: finding
status: live
claim_check: synthesis
date: 2026-09-25
mechanism: lexicon v1 -- corpus positional-frame PPMI graph + label-spreading (host-computed noun/non-noun
  category score), spike-RELAYED into a two uncoupled pools (amendment A1(a); the "spiking WTA" makes no
  spiking-decision claim -- the decision is v2, lexicon_spiking_frame_category.py). Teacher-supervised seed set
  (k=12/class, drawn per-CV-seed from the hand lists); "self-organized" does not apply per docs/TERMS.md.
lane: language (D6 referent-lexicon, v1 host-label-spreading de-risk)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO -- one pre-registered evidence gate fails (G7 organ-level reply recall, mean 0.7778 < 0.80); all
  other evidence gates and all integrity smokes pass, on independent re-computation via the runner's own
  pre-registered `score()` entrypoint run on this branch.
artifacts:
  - research/findings/raw/_lexicon_learned_referent/lexicon_referent_s42_43.json
  - research/findings/raw/_lexicon_learned_referent/lexicon_referent_s44_100.json
  - research/findings/raw/_lexicon_learned_referent/lexicon_referent_s101_102.json
  - research/findings/raw/_lexicon_learned_referent/verdict.json
  - (+ matching .prov.json sidecars for every file above, including verdict.json.prov.json)
external: NO-EXTERNAL-NEEDED -- reads already-run, already-committed-elsewhere pool artifacts; scores them with
  the pre-registered scorer already present on `main`; no new run, no new mechanism, no sim/ edit.
builds_on:
  - research/runners/_lexicon_learned_referent_derisk.py (on `main` since commit 33e94a512, which carries
    amendment A1 verbatim -- gate thresholds unchanged from the original pre-registration)
  - a byte-identical-in-substance finding already committed on `research/language-lane-next` (NOT merged to
    `main`): research/findings/2026-09-23-lexicon-referent-v1-host-label-spreading-6seed-NO-GO-and-relabel.md
    (see "Relationship to prior work" below) -- that branch also carries these same three raw artifacts plus a
    prior `verdict.json`, both untouched by this finding
  - research/findings/2026-09-23-cpu-lane-harvest-language-lexicon-referent-v2-S2-scored-6seed-GO.md (same
    branch; the successor mechanism, v2, scored separately)
next_command: "none -- scored. To re-derive: .venv/bin/python -m research.runners._lexicon_learned_referent_derisk
  --score research/findings/raw/_lexicon_learned_referent (runner already on main; no checkout needed)."
---

# Lexicon v1 (learned referent detector): 6-seed NO-GO on organ recall, confirmed on `main` (2026-09-25)

## What was measured, and how this finding was produced

`research/runners/_lexicon_learned_referent_derisk.py` carries its own pre-registration in its module docstring
(G1-G7, written before any 6-seed result was opened; amended as A1 on 2026-09-23T09:45, thresholds unchanged).
The three raw per-seed artifacts in `research/findings/raw/_lexicon_learned_referent/` (seeds 42+43, 44+100,
101+102 -- all 6 of `SEEDS6`) were produced on the pool at git revision `fc8a0e10a03c88bbb3231c27cc92a24a8e37b59e`
2026-09-23 09:25-09:26 (per-file `.prov.json`: `git_sha: "fc8a0e10a03c88bbb3231c27cc92a24a8e37b59e"` in full,
`source_kind: "git_archive"`, `git_dirty: false`, on all three files). These artifacts sit UNTRACKED in the
primary checkout (`/home/dant123/Projects/sim`); they were copied byte-for-byte (`diff -q`, all 6 files incl.
`.prov.json` sidecars, identical) into this branch's own worktree and are committed here for the first time on
any `main`-descended branch.

**Liveness check (2026-09-25, before scoring):** `ssh -n -F research/queue/.pool_ssh_config <node> 'ps -eo
etimes,args | grep _lexicon_learned_referent'` against `pool1`, `pool2`, `pool41`, `pool42` (all four reachable
and alive, each running unrelated jobs -- `load_bearing_fraction`, `d6_capacity_curve`,
`onebrain_regression_battery`) returned zero matches on every node: no `_lexicon_learned_referent_derisk` process
is running anywhere. Combined with all 6 `SEEDS6` present across exactly 3 paired files, the battery is COMPLETE
and not in flight.

**Scoring:** `.venv/bin/python -m research.runners._lexicon_learned_referent_derisk --score
research/findings/raw/_lexicon_learned_referent`, run on this branch against the copied files, using the
scorer already on `main` (added by commit `33e94a512`, which carries amendment A1 verbatim -- "the thresholds
above are UNCHANGED; only the READING of gates ... change"). Output: `verdict.json` (+ auto-stamped
`.prov.json` sidecar via `research/runners/__init__.py`'s provenance door), committed alongside the raw data.

## Per-gate table (independently re-computed from `research/findings/raw/_lexicon_learned_referent/verdict.json`)

| gate (kind) | value | threshold | pass |
|---|---|---|---|
| G1 learned held-out balanced acc, mean (evidence) | 0.9294 | >= 0.80 | yes |
| G1 min over seeds (evidence) | 0.9267 | >= 0.75 | yes |
| G2 shuffled-graph control mean (evidence) | 0.4942 | <= 0.60 | yes |
| G3 learned - max(shuffled, freq-only, label-permuted), mean (evidence) | 0.3979 | >= 0.15 | yes |
| **G7 organ: learned recovers BOTH held-out referents, mean (evidence)** | **0.7778** | **>= 0.80** | **NO** |
| G7 false-positive in-scope, mean (evidence) | 0.0 | <= 0.20 | yes |
| G4 spiking-vs-offline agreement, min (integrity smoke) | 1.0 | -- | yes |
| G4 spiking balanced acc, mean (integrity smoke) | 0.9263 | -- | yes |
| G5 lesion abstain rate, min (integrity smoke) | 1.0 | -- | yes |
| G6 determinism (integrity smoke) | true, **partial** (graph not rebuilt in these artifacts) | -- | yes |
| G7 hand-table baseline in-scope, mean (integrity smoke) | 0.0 | -- | yes |
| G7 lesioned in-scope, mean (integrity smoke) | 0.0 | -- | yes |

`complete_6seed: true`, `integrity_smokes_pass: true` -- both stated preconditions hold, so the single failing
evidence gate (G7 recover-mean) is the entire and sole cause of the NO-GO. Per-seed G7 recover rate (42, 43, 44,
100, 101, 102; 12 organ trials/seed): 0.83, 0.92, 0.67, 0.67, 0.92, 0.67 -- three of six seeds land at 0.67. A
trial needs BOTH held-out nouns admitted, so recall is close to per-word accuracy applied twice; at ~0.93
word-level balanced accuracy some pairs still lose one member.

## What this shows

The host frame-graph label-spreading score (v1) is a strong open-vocabulary noun/non-noun detector at the
word level: G1-G3 clear their bars by a wide margin (learned 0.9294 vs. the best control at 0.5316, a 0.398 gap
against a 0.15 bar), and this holds on every one of the 6 seeds (G1 min 0.9267). The "spiking" wrapper around it
passes its own integrity checks (G4/G5), but per amendment A1(a) that wrapper is a RELAY, not a spiking decision
-- the two pools it drives are uncoupled and one is fed directly by `np.sign()` of the host score, so G4/G5 and
G7's hand/lesion arms pass BY CONSTRUCTION and are correctly demoted to integrity smokes in `verdict.json`, not
evidence. v1 therefore makes no "fully spiking" claim regardless of G7's outcome (`docs/TERMS.md`: "fully
spiking" requires every step between sensation and action to be neurons/synapses; here the decision-bearing step
is a host `np.sign()` call).

## What this does NOT show

The one gate that measures the capability at the level that matters -- does the detector's output let the D6
multi-referent working-memory organ actually recover an UNSEEN pair of nouns from a sentence it was never
tuned on -- fails its pre-registered bar. Mean recover-both-referents across 6 seeds is 0.7778 against a >= 0.80
requirement; half the seeds (44, 100, 102) sit at 0.67. This is not a marginal instrument or measurement problem:
`complete_6seed` and `integrity_smokes_pass` both read true, the false-positive control is clean (0.0 across all
seeds), and G6's "partial" status (the artifacts predate amendment A1(d)'s graph-rebuild fix) affects only an
INTEGRITY smoke, not the failing evidence gate. The shortfall is real and specific to the reply-level, two-word
joint-recall test.

## Relationship to prior work (why this is a confirmation, not a fresh discovery)

A near-identical finding, `research/findings/2026-09-23-lexicon-referent-v1-host-label-spreading-6seed-NO-GO-
and-relabel.md`, already exists -- but only on `research/language-lane-next` (unmerged; confirmed via `git
ls-tree origin/research/language-lane-next`, which also carries these same three raw artifacts plus its own
prior `verdict.json` and a `v1_null_6seed.json` report-only null this finding does not reproduce). Independently
re-running `--score` here, using the scorer that has SINCE landed on `main` (commit `33e94a512`, amendment A1
carried over verbatim), reproduces every gate value in that finding's table exactly (G7 mean 0.7778, G1 mean
0.9294, G3 gap 0.3979, etc.) -- this is a confirmation of an already-banked result, made available from `main`
for the first time, not a new measurement. No claim here should be read as first discovery of the NO-GO; credit
for that belongs to the 2026-09-23 work on `research/language-lane-next`.

Separately, `main` has since gained `research/runners/_d6_learned_referent_env_flag_derisk.py` (commit
`33e94a512`, R3/R4/R5 gates, its own distinct pre-registration) which wires this same host label-spreading
lexicon into a production-adjacent `BRAIN_LEARNED_REFERENT_LEXICON`/`_LESION` env-flag route, default OFF. That
is a DIFFERENT battery with its own gates and is not scored by this finding; it is named here only so a reader
does not mistake "the mechanism has a production route" for "the reply-level capability gate passed" -- it did
not, on this (G1-G7) pre-registration.

## Next step per THE LAW (banking the method, not the capability)

**Banked NO-GO: the METHOD** -- host frame-graph label-spreading, spike-relayed, feeding the D6 organ's
reply-level recall test -- fails its own pre-registered reply-level bar (G7, mean 0.7778 < 0.80) on 6/6 seeds,
reproducibly. This is a verdict on that method, not a license to drop the CAPABILITY (an open-vocabulary
referent lexicon that lets the multi-referent WM organ recover unseen noun pairs from a sentence).

**The capability is not abandoned; a successor already has its own GO, with a caveat.** v2
(`lexicon_spiking_frame_category.py` / `_lexicon_spiking_referent_derisk.py`, Hebbian(Oja)-learned feedforward
frame->category synapses driving a coupled two-pool spiking WTA -- a genuine spiking decision, not a relay)
scored 6-seed GO on all 8 of its own pre-registered gates in
`research/findings/2026-09-23-cpu-lane-harvest-language-lexicon-referent-v2-S2-scored-6seed-GO.md` (also only on
`research/language-lane-next`, not yet on `main`). v2's organ-level analogue of this finding's failing gate
(its S7, "learned recover mean") reads 0.7639 -- essentially the SAME raw recall as v1's failing 0.7778 -- but
v2's own pre-registration gates S7 at mean >= 0.60, not >= 0.80. **This threshold difference is flagged, not
resolved, here**: v2 passing its organ-recall gate is not evidence that the underlying reply-level recall
improved over v1; on the numbers alone it is marginally worse. Before either mechanism is treated as closing
this capability, the two organ-recall bars (0.80 here, 0.60 for v2) should be reconciled against a single
standard, and/or the actual bottleneck should be re-examined: at ~0.79-0.93 word-level balanced accuracy on
EITHER mechanism, a two-word-AND'd organ trial caps out well short of 0.80 recall, which may mean the organ's
binding step (not the referent detector's word-level accuracy) is the real constraint worth instrumenting next.
