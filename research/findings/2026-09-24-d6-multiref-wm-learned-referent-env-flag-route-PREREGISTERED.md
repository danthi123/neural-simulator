---
type: preregistration
status: preregistered
date: 2026-09-24
mechanism: The PRODUCTION wiring of the learned open-vocab referent lexicon into the D6 multi-referent WM organ
  (`research/runners/_d6_learned_referent_env_flag_derisk.py`) -- routing `BRAIN_LEARNED_REFERENT_LEXICON` /
  `BRAIN_LEARNED_REFERENT_LESION` through `d6_multiref_wm_production_organ._flag_learned_referent_lexicon()` ->
  `lexicon_spiking_frame_category.get_lexicon()` (a module-level singleton), not the direct `.referent_lexicon`
  attribute injection the mechanism gate already banked GO.
lane: E -- Language (learned referent lexicon -> multi-referent WM route)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PREREGISTERED -- no evaluation run has happened yet under this file. Seed 42 runs LOCALLY under
  `tools/memcap.sh`/`tools/mem_ok.sh` immediately after this commit; seeds 43/44/100/101/102 are to be staged on
  the pool (pool41 + pool42, isolated revision), not awaited synchronously in this session. See AMENDMENT 1: all six
  seeds (42 included) now run at one amended revision on one recorded input.
artifacts: []
external: none new -- this file verifies a WIRING, not a biological claim; the detector's own biology binding is
  unchanged from `_lexicon_spiking_referent_derisk.py`'s already-banked GO.
builds_on:
  - `research/findings/2026-09-23-cpu-lane-harvest-language-lexicon-referent-v2-S2-scored-6seed-GO.md` (the
    mechanism GO via direct `.referent_lexicon` attribute injection; this file exercises the PRODUCTION indirection
    that result never went through).
  - the 2026-09-23 harvest's own "next rung" note: route the referent extraction through the LEARNED lexicon
    behind a default-off flag, so wm-binding-advanced can be exercised on nouns the hand list lacks (the
    allfixes2 battery missed "owl").
review_corrections_applied:
  - "R4's lever check used `tools.lab.lever(..., required=True)`, which raises `LeverError` and crashes the
    process BEFORE the per-seed JSON is written whenever intact_rate == lesion_rate -- exactly the two realistic
    failing outcomes R4 exists to catch (a broken wire: both arms at 0.0; a lesion the route ignores: both arms
    at the same nonzero rate). Fixed: the lever check now uses `required=False` (factored into `_r4_gate`,
    unit-tested in `tests/test_d6_learned_referent_env_flag_derisk.py`), records `r4_lever_moved` in the
    artifact, and folds it into `r4_pass` -- an unmoved lever now records as a FAILED gate, never a crash, so
    `score()` sees NO-GO instead of INCOMPLETE."
  - "R5 was pre-registered as 'report-only, not gated' in an earlier draft of this text but `score()` put
    `R5_singleton_built_all_seeds` inside the `passed` set anyway. Fixed: `score()` now separates a `report_only`
    dict from the gated `evidence` dict; R5 sits only in `report_only` and cannot flip the verdict, because R1
    passing already implies a built lexicon and a check that cannot independently fail must never sit inside a
    verdict."
  - "R3/R4's prose said 'mean rate >= 0.60 ... gated at >= 0.50' / 'falls to <= 0.20' while `score()` actually
    gates on min-per-seed / max-per-seed. Fixed: the runner's own docstring now states the gate exactly as
    `score()` computes it (min(rates) >= 0.50 for R3, max(rates) <= 0.20 AND lever-moved for R4)."
  - "Declared explicitly (was previously only a docstring aside): `get_lexicon()` is called with no seed argument
    from `d6_multiref_wm_production_organ.py` and always trains at seed=42 regardless of which conversational
    seed the calling organ was built with. All 6 organ seeds in R3/R4 therefore exercise the SAME ONE
    lexicon/detector -- this is 1 lexicon x 6 organ-seed population draws, NOT 6 independently-trained
    substrates. See 'Honest residuals' below."
---

# D6 learned-referent lexicon, the PRODUCTION env-flag route: pre-registration (before any evaluation run)

**This is a pre-registration only**, committed before the first evaluation run this file governs
(`tools/gates/prereg_before_run.py`). It fixes the mechanism, the exact gates (matching `score()` exactly, after
three corrections from adversarial review — see `review_corrections_applied` above), and the honest residuals,
before seed 42 (local) or any pool-staged seed is read for verdict purposes.

## Why this is the genuine next rung, not a re-derivation of the banked mechanism GO

`_lexicon_spiking_referent_derisk.py` already banked a 6-seed GO for the MECHANISM (a coupled spiking WTA,
Hebbian frame->category synapses, load-bearing learned edge) by injecting the trained lexicon directly onto a
fresh organ's `.referent_lexicon` attribute. That is not the production route: production never sets
`.referent_lexicon` directly, it reads `BRAIN_LEARNED_REFERENT_LEXICON` from the process environment through
`d6_multiref_wm_production_organ._flag_learned_referent_lexicon()` -> `lexicon_spiking_frame_category.get_lexicon()`
(a module-level singleton, always seed=42, built once per process). That indirection was never exercised
end-to-end before this file: a bug in the env-var parse, in the singleton's lazy-build guard, or in `get_organ()`'s
per-session construction leaving a stale `.referent_lexicon` from an earlier test would be invisible to the
direct-injection gate above and would still ship broken. This file is the wiring check the 2026-09-23 harvest
named as the "next rung".

## The gates, each stating the realistic outcome that FAILS it (matching `score()` exactly)

- **R1 ROUTE-ON, held-out capability.** `BRAIN_LEARNED_REFERENT_LEXICON=1` in the process env, a FRESH
  `MultiReferentWMOrgan()` that never touches `.referent_lexicon`, turn "the wolf watches the owl" ("owl" is in
  neither `_REFERENT_NOUNS` nor `HAND_NOUN_SEEDS`/`NONNOUN_SEEDS` -- genuinely held out of every hand list). FAILS
  if `judge()` returns None, or `n_referents != 2`, or "owl" is absent from `input_order` -- i.e. the env var does
  not actually reach the organ's decision.
- **R2 ROUTE-OFF, byte-identical contrast.** The SAME process, SAME already-built singleton, env var deleted
  (unset, the production default). FAILS if the SAME turn is still in-scope with "owl" recovered -- an OFF arm
  credited with the ON arm's effect is not a control.
- **R3 POPULATION RECOVERY, routed through the flag (not attribute injection).** Over >=12 held-out noun PAIRS
  (POS ground-truth fixture, excluded from every hand list and the training curriculum), flag ON, a fresh organ's
  "the A and the B walked in" -> "who are we talking about?" recovers both held-out nouns. GATED on the MINIMUM
  per-seed recovered-both rate across all 6 seeds >= 0.50 (a 10-point tolerance band under the already-banked S7
  direct-injection threshold of 0.60, for the extra indirection hop). FAILS if any seed's routed rate drops below
  0.50.
- **R4 LESION, ROUTED.** Flag ON AND `BRAIN_LEARNED_REFERENT_LESION=1`, same population battery. GATED on the
  MAXIMUM per-seed recovered-both rate across all 6 seeds <= 0.20 (S7's lesion ceiling) AND every seed's lever
  having MOVED (`r4_lever_moved`, `tools.lab.lever(..., required=False)`, factored into `_r4_gate` and
  unit-tested to never raise). FAILS if the lesion does not reduce recovery on any seed when reached through the
  env-var route, OR if a seed's intact and lesioned rates are numerically equal (the lesion never reached the
  route at all -- recorded as a failed gate, not a crash). The single-word HELD-phrase lesion outcome is
  explicitly NOT gated here either (S3b/S8 already documented it as a ~30%-residual coin flip at n=1); this is
  the 12-trial population arm instead.
- **R5 SINGLETON REUSE -- REPORT-ONLY, NOT GATED.** `get_lexicon()` is called exactly once across the whole
  R1-R4 sequence in-process. Reported in `report_only`, excluded from `evidence`/`passed`: R1 passing already
  implies R5, so it cannot independently fail and must not sit inside the verdict. A count of zero is impossible
  given R1 passed, so a human reading the report can still use it to catch an unreachable-import regression.

**INTEGRITY SMOKE (must hold, not evidence):** the organ's own `all_recovered`/`recovered` bookkeeping and the
detector's own weight-hash check (`decide()` raises on a lesion that does not hold at measurement) are reused
unmodified -- no new pass-by-construction check is added here.

`seed_go` (per seed) = R1 AND R2 AND R3 AND R4, all True (R5 reported, not required). `pooled_go` (the 6-seed
verdict `score()` computes) = R1_all AND R2_all AND R3_min>=0.50 AND (R4_max<=0.20 AND every seed's lever moved).

## Honest residuals

- **ONE shared lexicon across all 6 seeds, not 6 independent substrates.** `get_lexicon()` is called with no
  seed argument from `d6_multiref_wm_production_organ.py` (`lex = get_lexicon()`, defaulting to `seed=42`)
  regardless of which conversational seed the calling `MultiReferentWMOrgan` was built with. `run_seed` force-resets
  the module-level singleton (`L._LEXICON = None`) once per seed's OWN process, but each of those 6 processes then
  trains the identical detector at the identical seed=42. The R3/R4 6-seed result is therefore **1 lexicon/detector
  x 6 organ-seed population/pair draws**, not 6 independently-trained substrates -- it must never be written up as
  6-seed replication of the detector itself (that claim is already banked separately, and correctly, by the
  direct-injection gate's own 6-seed run in `_lexicon_spiking_referent_derisk.py`, which DOES vary the detector's
  training seed). What R3/R4 DOES replicate 6 times is the WIRING (env var -> singleton -> organ) and the
  population sampling (a fresh `cond_rng` per seed draws different held-out noun pairs), which is exactly the one
  new thing this file tests.
- NOUN-hood, not REFERENT-hood; teacher-supervised curriculum, not self-organized category discovery. See
  `lexicon_spiking_frame_category`'s own docstring for the full residual list -- not repeated here.
- Honesty boundary: nothing here asserts phenomenal experience or discourse-level referent tracking beyond noun
  recovery under a WM-binding readout; the readout (`judge()`'s `recovered` dict) is host bookkeeping over the
  organ's own spiking WM slots, declared as such in `d6_multiref_wm_production_organ.py`.

## Compute plan

Seed 42: LOCAL, under `bash tools/memcap.sh <measured_gb> --` after `tools/mem_ok.sh` passes (numpy backend, a
lexicon-training + small WM-organ workload; measured peak RSS recorded in the `--checked` pool line for the
remaining 5 seeds, below). Seeds 43/44/100/101/102: staged on the pool (`tools/pool_queue.sh add`, pool41 +
pool42, isolated revision, `--checked` line naming the measured mem_gb and this prereg's commit).

Run (per seed; numpy backend, real TinyStories corpus + fixture required):
```
SIM_BACKEND=numpy python -u -m research.runners._d6_learned_referent_env_flag_derisk --seed <SEED> \
    --json research/findings/raw/_d6_learned_referent_env_flag/s<SEED>.json
python -m research.runners._d6_learned_referent_env_flag_derisk --score research/findings/raw/_d6_learned_referent_env_flag
```

## AMENDMENT 1 (2026-09-24, before any seed other than 42 ran; gates and thresholds unchanged)

Review of the seed-42 run found that it read a 7,988,286-byte prefix of `data/corpus/tinystories.txt` (the builder
worktree's copy) while the primary checkout holds the 19,971,040-byte file. The runner reads 8,000,000 characters, so
the two give different vocabularies and lexicons: re-running the same commit at seed 42 on the full file read
R3 = 0.8333 instead of 0.9167 (R4 = 0.00 both times). Neither input was recorded. Changes, provenance only: <!--derived-->

- each per-seed JSON records the sha256 and size of both inputs (the frame-environment corpus and the corpus the
  learned lexicon reads, which ignores `--corpus`);
- `score()` returns `MIXED-INPUT` (never GO) unless all six seeds carry one identical recorded input pair;
- all six seeds, seed 42 included, run on the pool at the amended revision, provisioned from the primary checkout
  (full 19.97 MB corpus). The earlier seed-42 artifact (`research/findings/raw/_d6_learned_referent_env_flag/s42.json`,
  7.99 MB prefix, no input hash) is superseded: it has no input hash, so `score()` cannot pool it.

