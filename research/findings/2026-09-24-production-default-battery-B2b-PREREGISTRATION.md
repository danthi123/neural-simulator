---
type: preregistration
status: preregistered
date: 2026-09-24
lane: load-bearing
mechanism: the #1 metric (lesion-verified load-bearing fraction) on the full 50-row registry at F (origin/main
  a308f1e09babcc9ed096c3c8046d00040391368c), at the production default and again with the flip candidate
  BRAIN_LEARNED_REFERENT_LEXICON=1 in every arm of every shard (flip-rule criterion (c) for the learned referent lexicon)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any b2b0924-base or b2b0924-flipcand job or shard exists. F is the measured
  revision; the commit that adds this file is a documentation-only child of F.
---

# Battery B2b: the full registry at F, with and without the learned referent lexicon (pre-registration)

## Why

B2a (`research/findings/2026-09-24-production-default-battery-B2a-PREREGISTRATION.md`, tag b2a0924) measures revision M1
9db761329 with the 38-row registry of that revision. Since M1, main gained the row-registry hook (`research/runners/lbf_rows/`)
with three row modules, plus several default-OFF lanes. B2b measures the whole registry at F.

Its second arm answers criterion (c) of the owner's flip rule (2026-09-23) for the first flip candidate that already holds
criterion (a): the learned referent lexicon, `research/findings/2026-09-24-language-learned-referent-production-route-GO-6seed.md`.
The rule: a validated feature flips default-ON only with (a) a 6-seed GO finding, (b) a SOUND review, (c) no regression in a
combined battery with it ON, and (d) a production-default validation run.

## Frozen revision and registry

- F = `a308f1e09babcc9ed096c3c8046d00040391368c` (origin/main when this was filed). Every job runs from
  `~/derisk-pool/revisions/<F>`.
- The registry at F is `load_bearing_fraction.FACULTY_LESIONS` after the hook merge (`LBF_ROW_MERGE_REPORT`): 50 rows
  <!--derived-->. Kinds: 33 neural-lesion, 3 whether-disable, 6 thin, 1 mechanism-only, 4 in-process, 3 proposed.
- Row modules merged: learning, live_organs, reward_value_afferent, with no import error. Rows parked by their own modules stay
  out: self-schema, reasoning-transitive-chat, tom-false-belief. Eight proposed_lesions_conflict_kb rows collide with base rows
  and the hook drops them; the base rows stand.
- Coverable rows (neural-lesion and whether-disable; these enter the fraction): 36. Sharded rows (`lb_shard.py`
  MEASURABLE_KINDS, which adds thin and mechanism-only): 43 per seed, so 258 shards per arm and 516 in all <!--derived-->.
- Against M1: no row removed and no kind changed. Twelve rows added. Eight are coverable: affect-appraisal-interoceptive,
  affective-tom, causal-whatif, gnw-bus, multiref-competition and spiking-anaphor (neural-lesion); d5-consolidate and
  sleep-replay (whether-disable). Four are thin: learned-referent, onebrain-xedge, spiking-qroute and
  surprise-salience-snc-afferent (the last arrived with the A10 merge 11bfd7232). M1 had 28 coverable and 31 sharded rows.

## Arms (fixed)

Both arms: `--no-fixes` (no fix flag, so every mechanism runs at its production default), the adequate probe set (the
`lb_shard.py` default: nine LB_* probe flags), `--repeats 2`, numpy backend, one BLAS/OMP thread, seeds 42 43 44 100 101 102,
the same 50-row registry. Each job line keeps the B2a layout:
`cd <revision dir> && .venv/bin/python tools/assert_flipped_defaults.py && <the lb_shard.py line>`.

```
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2b0924-base     --no-fixes --probe-set adequate --root '~/derisk-pool/revisions/<F>'
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2b0924-flipcand --no-fixes --probe-set adequate --extra-env BRAIN_LEARNED_REFERENT_LEXICON=1 --root '~/derisk-pool/revisions/<F>'
python tools/lb_shard.py aggregate --tag b2b0924-base     --seeds 42 43 44 100 101 102
python tools/lb_shard.py aggregate --tag b2b0924-flipcand --seeds 42 43 44 100 101 102
```

For the same seed and row, a base line and a flipcand line differ in two places only: the tag in the output path, and one
token in the `env` prefix, `BRAIN_LEARNED_REFERENT_LEXICON=1`.

**How the flag reaches the intact and the lesion arm of every shard (read in the code at F).** `lb_shard.py jobs --extra-env`
puts the flag in the `env` prefix of the shard process. Every arm is a subprocess started by
`onebrain_regression_battery._spawn_arm` with `env=dict(os.environ)`. Its worker (`_collect_worker`) only overlays the row's
driving `base_env` and, in the lesion arm, the row's lesion flag. No row at F uses BRAIN_LEARNED_REFERENT_LEXICON as its lesion
flag or in a driving env; its only mention in `load_bearing_fraction.py` and `research/runners/lbf_rows/` is the note of the
learned-referent row.

So in flipcand the intact arm, the null-control rebuild, the lesion arm and its repeat all carry the flag, and in base none
does. A probe that returns before any `_spawn_arm` call (the open-ended distributional ruler) runs inside the shard process and
sees the same environment. The existing `--extra-env` path does the job; no new mechanism is needed.

**Guard.** `tools/assert_flipped_defaults.py` runs before the `env` prefix, in the node's own environment. It checks that F
contains the three 2026-09-23 flips and that no BRAIN_* variable is set on the node. It does not see the flipcand flag, by
design. Static check on the job text at generation: base lines carry no `BRAIN_*=` token, and flipcand lines carry exactly
one, `BRAIN_LEARNED_REFERENT_LEXICON=1`.

**Cost of the flag.** With the flag on, the first `extract_referents` call in each arm process builds the deployment lexicon
(`lexicon_spiking_frame_category.get_lexicon()`: an 8,000,000-character prefix of `data/corpus/tinystories.txt`, lexicon seed
42 in every arm whatever BRAIN_CHAT_SEED is). One standalone build at F on the local box took about 23 s and 0.39 GB peak RSS
(not an artifact). Jobs declare the same memory as B2a, `mem_gb=6`, the `tools/pool_runner_mem.tsv` value for
load_bearing_fraction.

**Integrity smoke (declared here, no weight).** Before dispatch, one flipcand line for a row whose organ calls
`extract_referents` is run at seed 7, which is not a battery seed, into a scratch directory outside both tags. It checks only
that arms build with the flag on and records peak memory. No criterion reads it.

## Validity of a cell (written before any shard exists)

A (faculty, seed) cell of an arm is DEFINED only if all of these hold:
- its `lb.json` exists, and its provenance sidecar records `git_sha` F, `git_dirty` false and `SIM_BACKEND=numpy`;
- in flipcand, the sidecar's env holds `BRAIN_LEARNED_REFERENT_LEXICON=1`; in base, the sidecar's env holds no BRAIN_* key;
- the report is not UNRELIABLE, and `null_control_clean` is not False;
- the verdict is a measured outcome: `regressed`, `pass` or `not-exercised`.

Any other verdict (`arm-build-failed`, `lesion-knob-missing`, `noisy`, `noisy-null-control`, `unmapped-*`, `artifact-*`,
`seed-missing-in-artifact`) makes the cell UNDEFINED. `lb_shard.py aggregate` lists an `arm-build-failed` cell as present and
not load-bearing, so this check reads the per-seed `verdicts` in `per_faculty` and the sidecars, not only
`incomplete_faculties`. An UNDEFINED cell is re-run once with its own unchanged job line at F. It is never scored as 0 and
never as a pass.

## Criteria (written before any shard exists)

- **R1 (no regression with the flag on):** for each of the 36 coverable faculties, n_load_bearing (the number of seeds, out of
  six, with `load_bearing` true) in b2b0924-flipcand is not lower than in b2b0924-base. numpy shards are deterministic, so
  there is no tolerance: one seed lower fails R1 for that faculty. The two arms differ only by the flag, so a drop is
  attributed to the flag. A drop is reported with the host of each cell; if the two hosts differ, both cells are re-run once on
  one host, and that pair decides R1 for the cell.
- **R2 (complete):** in both arms, each of the 36 coverable faculties has six DEFINED cells, with clean null controls. Missing
  or UNRELIABLE is UNDEFINED, never 0.
- **R3 (reported, not gated): the learned-referent row itself.** The row is kind `thin`. `measure_faculty` returns
  `not-covered:thin` for a thin row before it builds any arm, so R3 will read `not-covered:thin` in both arms by construction.
  B2b therefore carries no evidence on whether the lexicon is load-bearing in chat; that evidence is the 6-seed GO finding
  (lesion recovery 0.0 on every seed). Making the row measurable is a registry change (its own note: a single-flag row
  becomes expressible once the lexicon ships default-ON) and is not part of B2b.
- **Verdict, in this order:** FAIL if R1 fails for any faculty with six DEFINED cells in both arms (named). Otherwise
  INCOMPLETE if any faculty still has an UNDEFINED cell after its one re-run, in either arm (cells named); INCOMPLETE is not a
  PASS. Otherwise PASS.

## Reported (not gated)

- Per arm: robust core (members and count), union, mean fraction and SD over the six seeds (per seed, n_load_bearing over
  n_exercised, as `lb_shard.py aggregate` computes it), and the backend, host mix and ltm_mode each aggregate records.
- Per faculty: n_load_bearing in base against flipcand, including any rise.
- Manipulation check: for each shard, the harness's `compare()` over the row's decision fields between the base intact arm and
  the flipcand intact arm, plus whether any turn's `answer` differs. The rows most likely to differ are those whose organs call
  `extract_referents` (the D6 multi-referent WM organ and the activity-silent WM organ). If no intact arm differs anywhere,
  the flag was inert on this probe set, and a PASS says only that it broke nothing the probes reach.
- Comparison with B2a (the aggregate that `lb_shard.py aggregate --tag b2a0924` writes; it does not exist yet at filing). The denominator changes: B2a
  covers the 28 coverable rows at M1, and B2b covers 36. Reported side by side: (i) B2b-base over the same 28 rows, with
  per-faculty n_load_bearing against B2a (a change is attributed to code merged between M1 and F, and is not bisected here);
  (ii) B2b-base over all 36, with the eight added rows named. If B2a is not complete when B2b is scored, the comparison is
  reported as pending, never estimated.

## What a PASS licenses

Criterion (c) of the owner's flip rule for BRAIN_LEARNED_REFERENT_LEXICON: no regression in a combined battery with it ON.
Nothing more. It does not meet (b), the SOUND review, or (d), the production-default validation run on the flipped default.
It is not a claim that the lexicon is load-bearing in chat (R3 cannot measure that), and it licenses no other flag. A FAIL
keeps the flag default-OFF and names each faculty that dropped.

## Residuals

- The six brain seeds share one lexicon trained at seed 42, as in the GO finding: flipcand varies the brain seed, not the
  lexicon.
- Latency is not measured; the lexicon build adds about 23 s to each arm process in flipcand.
- The 2026-09-24 midnight plan (steps S26-S28) sketched B2b at the head of research/integration-0924 with tags b2b-rows and
  b2b-caps. This battery is frozen at origin/main instead, with tags b2b0924-base and b2b0924-flipcand, and holds one flip
  candidate.
