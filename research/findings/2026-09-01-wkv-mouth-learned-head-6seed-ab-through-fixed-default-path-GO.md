---
type: finding
status: positive
date: 2026-09-01
mechanism: 6-seed A/B of the e-prop LOCALLY-LEARNED WKV-mouth read-out head vs the checkpoint's NATIVE
  (host-trained "copied") head, through the PRODUCTION `webapp.wkv_mouth_generator.generate()` entry point, at
  the module's own DEFAULT (un-overridden) `BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH` template
verdict: GO, 6 seeds (42/43/44/100/101/102). The fixed default learned-head path resolves and APPLIES cleanly
  on every seed (no silent fail-safe fallback anywhere), and the learned head generates DECISIVELY coherent,
  in-vocabulary TinyStories prose on every seed. Beyond the regression pin, the quality result is stronger than
  the earlier single-seed check: the properly per-seed-templated heads (`sub_recov_ratio_mean=0.9273`,
  `sub_recov_ratio_min=0.8906`, `research/findings/raw/_persist_eprop_head_scope/eprop_learn_persist_6seed.json`)
  generate a lower (better) mean self-NLL than native on all 6 seeds. Still default-OFF, opt-in
  (`BRAIN_WKV_MOUTH_LEARNED_HEAD`) -- this finding does not flip that flag.
lane: e-mouth-fluency / mouth crutch-burndown rung-1 (GAP_CLOSURE_MISSION.md "ORDERED RUNGS")
artifacts:
  - research/findings/raw/_wkv_learned_vs_native_head_ab_6seed.json
  - research/findings/raw/_wkv_learned_vs_native_head_ab_6seed.json.prov.json
  - research/findings/raw/_wkv_learned_vs_native_head_ab.json
  - research/findings/raw/_persist_eprop_head_scope/eprop_learn_persist_6seed.json
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s42.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s43.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s44.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s100.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s101.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s102.npz
  - research/findings/2026-08-28-mouth-better-head-persist-6seed-GO-plus-wander-production-partial.md
  - research/findings/2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md
  - webapp/wkv_mouth_generator.py
  - tests/test_wkv_mouth_learned_head_path.py
runner: research/runners/_wkv_learned_vs_native_head_ab_6seed.py
---

# WKV mouth rung-1, closed: fixed default path verified 6/6, plus a stronger-than-parity 6-seed A/B

## 0. What this closes

`GAP_CLOSURE_MISSION.md`'s "ORDERED RUNGS (Plan agent)" names rung-1 as three parts: **(1)** fix the
learned-head default PATH mismatch (`webapp/wkv_mouth_generator.py:76-79` pointed at a nonexistent template),
**(2)** a real 6-seed learned-vs-native A/B through production `generate()`, **(3)** a committed pytest. This
finding closes all three, with one correction to the board's framing:

**Part (1), the path fix itself, was ALREADY LANDED before this session** -- commit `aa7c3a23c` (2026-08-28,
`fix(mouth crutch-burndown rung-1): point the WKV learned-head default path at the 6/6-GO persisted heads`),
already an ancestor of `main`/HEAD (`3ad5fa228`, verified via `git merge-base --is-ancestor aa7c3a23c HEAD`).
Its diff replaced the old default `research/findings/raw/_wkv_eprop_learned_head_seed{seed}.npz` (a location
that never existed) with `research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.
npz`, which DOES exist for all 6 seeds. **This session did not re-fix the path** -- it verified the existing
fix is real (SHA1-distinct per-seed files, `Path.exists()` for all 6 seeds) and then closed the two remaining
parts, (2) and (3), which were genuinely still open: no committed pytest existed for this path, and the only
prior A/B (`_wkv_learned_vs_native_head_ab.py`) is deliberately pinned to a SINGLE seed (102) against a
DIFFERENT, older learned-head artifact (see SS3 for why these are not interchangeable).

## 1. Anti-cheat: the 6 per-seed learned-head files are genuinely distinct

Before trusting any per-seed result, the runner SHA1-hashes each of the 6 resolved default paths and asserts
no two match -- a regression pin against the EARLIER bug this exact artifact family already had once
(`eprop_learn_persist_6seed.json`'s first run passed a literal, non-`{seed}`-templated `--save-w-hat`, so all
6 seeds silently overwrote one file; documented in the single-seed A/B's own SS1). Result: 6 distinct SHA1s
(`c7d44cd7...`, `90d10b80...`, `7030aec8...`, `aac8547b...`, `2dbd33c6...`, `212acffe...` for seeds
42/43/44/100/101/102) -- confirmed genuinely per-seed, not a repeat of that bug.

## 2. Method

Primary artifact: `research/findings/raw/_wkv_learned_vs_native_head_ab_6seed.json`.

`research/runners/_wkv_learned_vs_native_head_ab_6seed.py`, CPU/numpy,
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_learned_vs_native_head_ab_6seed`, 14.5s elapsed.
6 seeds (42, 43, 44, 100, 101, 102) x the SAME 8 in-vocab TinyStories-domain prompts used by the single-seed
A/B (imported verbatim, not re-typed) x 2 arms (native, learned) = 96 production `generate()` calls
(`max_new_tokens=50, read_window=40, pop=8, topk=64, gen_temp=0.8`). **Deliberately does NOT set
`BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH`** -- the whole point is to exercise the module's own default template,
un-overridden, for every seed. Per call: teacher-forced self-NLL (nats, vs chance `log(1000)=6.9078`),
distinct-1/2/3 n-gram ratios, longest consecutive-repeated-word run, `head_w` SHA1 (the lever), and the
loader's own `applied`/`reason` provenance dict. Host process-global numpy RNG state is checked byte-identical
before/after the entire 96-call run.

## 3. An honest disambiguation: this is a DIFFERENT artifact from the single-seed A/B's

<!--derived-->

`2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md` ran its A/B against
`_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz` (`research/findings/raw/_wkv_learned_vs_native_
head_ab.json`'s own `npz_meta.sub_recov_ratio=0.9132`) -- the file that, per its own SS1, actually holds only
seed=102's head from the FIRST (buggy, un-templated) persist run. That finding's headline numbers (native
1.4691 / learned 1.5138 nats at seed=102, learned somewhat WORSE) describe THAT specific artifact. This runner
instead uses the module's DEFAULT path, which resolves to `wkv_eprop_learned_head_0p94_s{seed}.npz` -- six
DIFFERENT files from a LATER, separately-run persist pass (`eprop_learn_persist_6seed.json`, `sub_recov_ratio_
mean=0.9273`, `sub_recov_ratio_min=0.8906` -- note seed=102's own individual ratio in THIS later persist pass
is the worst of the six at 0.8906, the opposite of its 0.9132 in the earlier single file). At seed=102
specifically, this run measures native=1.4691 (matches the earlier finding's own number exactly -- same
native head, same prompts) but learned=1.2151, notably better than the earlier finding's 1.5138 -- because it
is a genuinely different trained head for the same seed, not a re-measurement of the same one. **These two
findings are not in tension; they describe two different artifacts**, and this one is the artifact the fixed
production default actually loads today.

## 4. Results (6 seeds, all through the production `generate()` entry point)

<!--derived-->

| seed | native self-NLL | learned self-NLL | native max-repeat-run | learned max-repeat-run | learned wins (of 8) | applied | heads differ |
|---|---|---|---|---|---|---|---|
| 42 | 1.3824 | 1.1184 | 1.125 | 1.000 | 7/8 | True | True |
| 43 | 1.6436 | 1.4587 | 1.000 | 1.000 | 6/8 | True | True |
| 44 | 1.4830 | 0.9875 | 1.125 | 1.000 | 8/8 | True | True |
| 100 | 1.4365 | 1.1723 | 1.000 | 1.125 | 8/8 | True | True |
| 101 | 1.4923 | 1.2189 | 1.000 | 1.500 | 8/8 | True | True |
| 102 | 1.4691 | 1.2151 | 1.000 | 1.125 | 8/8 | True | True |
| **6-seed mean** | **1.4845** | **1.1951** | -- | -- | **45/48** | **6/6** | **6/6** |
| worst seed (43) | -- | 1.4587 | -- | -- | -- | -- | -- |

(All per-seed and aggregate numbers above are read directly from `research/findings/raw/_wkv_learned_vs_
native_head_ab_6seed.json`'s `per_seed`, `native_self_nll_6seed_mean`, `learned_self_nll_6seed_mean`,
`learned_self_nll_6seed_worst` and `total_learned_wins_of_48` fields, rounded to 4 decimals for display.)

Chance is `log(1000)=6.9078` nats; even the worst single learned seed (43, mean 1.4587) sits well below
chance. `all_learned_applied_6of6=true` (no silent fail-safe fallback on any of the 48 learned-arm calls --
the rung-1 regression, had it still been present, would show up here as `applied=False`), `all_heads_differ_
6of6=true` (the lever fired on every seed), `rng_untouched_across_run=true`. The runner's own `tools.verdict.
Verdict` (anti-cheat distinctness + 6/6 lever + 6/6 fail-safe + RNG + worst-seed separation-from-chance >2.0
nats) reads **GO**.

Repetition/looping -- the single-seed A/B's named residual (5/8 learned samples affected, 2 severe) -- is
NOT reproduced at this scale with this artifact: `max_repeat_run` for the learned arm is 1.000-1.125 on 5 of
6 seeds and 1.500 on one (seed 101), all close to native's 1.000-1.125 range. This is consistent with SS3: the
two A/Bs measure different trained heads, and the later persist pass (`eprop_learn_persist_6seed.json`, named
"the learn-runner's BETTER head" in its own finding) looks like a genuine quality improvement over the head
the single-seed A/B characterized, not merely a re-measurement noise difference.

## 5. Part (3): the committed pytest

`tests/test_wkv_mouth_learned_head_path.py`, 37 cases, 0.75s, all green
(`SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_wkv_mouth_learned_head_path.py -v`). Pins, per seed
(42/43/44/100/101/102) and as a group: the default path resolves to an EXISTING file; it is not the old broken
`_wkv_eprop_learned_head_seed{seed}.npz` pattern; it is genuinely seed-templated (not one literal file reused);
the 6 files are byte-distinct; `_apply_learned_head` reports `applied=True, reason=None` on every seed when
the flag is on; native and learned `head_w` differ (the lever); and with the flag off, no learned-head status
is ever recorded at all (byte-identical-off preserved). Verified the tests can actually FAIL in their failing
direction: pointing `BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH` at the OLD broken template resolves to a non-existent
path (`Path(...).exists() == False`), which `test_default_path_exists` would catch.

## 6. Verdict

**GO, 6/6 seeds.** Rung-1's path fix (landed prior to this session, `aa7c3a23c`) is verified genuinely correct
and load-bearing end to end -- every one of the 6 non-negotiable seeds resolves to an existing, seed-distinct
learned-head file and applies it with zero silent fallbacks. The 6-seed A/B through the exact production
`generate()` path is decisively positive on the primary self-NLL metric (learned better than native on every
seed, 45/48 individual comparisons), a stronger result than the earlier single-seed check because it measures
a later, separately-trained artifact (SS3) -- not a re-measurement of the same one, and not in tension with it.
The regression is pinned by a committed, fast (0.75s), CPU-only pytest.

**`BRAIN_WKV_MOUTH_LEARNED_HEAD` stays default-OFF.** This finding closes rung-1 exactly as scoped (path
verified + 6-seed A/B + pytest) and deliberately does NOT flip that flag -- byte-identical-off is preserved.
Given the 6/6-seed, 45/48-comparison margin measured here is materially stronger than the "roughly on par"
single-seed result that previously justified staying opt-in, **flipping `BRAIN_WKV_MOUTH_LEARNED_HEAD` default-
on is now a well-evidenced candidate next rung** -- named here as a lever for a future session/owner decision,
not enacted in this one (out of this task's explicit 3-part scope, and a default flip is its own decision this
repo's convention reserves separately from the A/B that evidences it).

**Honest residuals, not swept under the verdict:**
1. n=8 prompts per seed, one hyperparameter configuration -- sufficient to certify "the fixed path loads a
   genuinely different, coherent head on every seed" (the regression-pin claim) and "the later persist-pass
   head measurably beats native on self-NLL at this scale" (the quality claim), but a wider prompt set would
   sharpen the exact margin.
2. Self-NLL under-detects non-adjacent "A B A B" repetition loops (documented by the single-seed A/B's own
   SS3); this run's `max_repeat_run` numbers are reassuring but were not eyeballed per-sample the way the
   single-seed A/B's were -- a full eyeball pass on this artifact's 48 learned continuations is the honest next
   check before any default-on claim beyond "well-evidenced candidate."
3. The base checkpoint per seed (`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz`) and the learned head
   per seed are trained artifacts whose OWN provenance (which training method, which run) is inherited from
   `2026-08-28-mouth-better-head-persist-6seed-GO-plus-wander-production-partial.md` and not re-derived here.
