# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-27-local-sleep-01b-read.md

# Local sleep + SHY downscale prereg (2026-11-27): local rule, then a synaptic-homeostasis cut of the adapters

STATUS: REGISTERED 2026-11-27 after pre-data review (filename keeps DRAFT). Reviewed before any SHY data by a sonnet
reviewer (this review); fixes applied: verdict split GO-SHY / GO-PLAIN-LOCAL, matched control gated on recall AND drift, (a) raised,
g = 1 excluded from the pick, pick cap and no_sleep pinned to `dev_s7_04b/ref.json`, six-seed rows limited to g*/lr*, provenance
(COMMIT required, host + env HOST), off-path output unchanged in structure (tested), dev drops `local_shuffled`, M-scale caveat
and lowered expectation added. Written before any SHY data. Follows the
NEGATIVE 0.1B screen of `brain/research/2026-11-27-local-sleep-rule-prereg-DRAFT.md`; that verdict stands and is not re-opened.
Milestone: ARCHITECTURE section 4, 2026-12-31, "sleep learning by a local rule instead of gradients".

## Record brief
- **Record check.** `bash tools/before_you_build.sh "M-scale downscale adapters synaptic homeostasis sleep"` printed
  "STALE CORPUS: published main 7ccd8ef98 is not in local main (700a7dd12)" and was NOT recorded, so no search ran.
  The M-scale record below was read by hand from the prereg and result files; refresh the corpus (`tools/w_follow.sh`) before registering.
- **What M-scale showed** (`research/2026-11-19-life-muse-mscale-prereg.md`; `results/2026-11-20-life-mscale-{dev,sab,e-dev}-1.5b.md`).
  `PlasticCortex.downscale(g, "B"|"AB")` was used in a multi-night life (1.5B, Brain `sleep_downscale`), cut BEFORE the
  night (S, SAB) or AFTER it (SABE, SHY renormalisation by morning), gradient sleep. Recall held or rose: pooled H int
  66/96 (S) and 65/96 (SAB) vs 42 for B; SABE 53/96, drift of off-cue intrusions fell (37 -> 10). It failed on other
  guards: B-only cut let ||A|| compound (1.5-1.6x per 6 nights), the cut-before-replay left norm over the 2.2x bar, and the
  after-night AB cut (SABE) fails the whole-transcript read (identity and told names overwritten by interference at a fixed
  budget). Verdict: the M-scale family ended for the LIFE use (a fixed budget shared by many nights).
- **What is new.** A single night, one-shot, applied after a LOCAL-rule night, judged on the sleep_probe battery (12 facts
  hippocampus-off, KL drift on generic text), not a multi-night budget. Motivation: the binding constraint is drift. Seed-7
  0.4B dev (`results/2026-11-27-local-sleep/dev_s7_04b/`): transport lr 0.3 recall 4/12 (gradient 5) at drift KL 0.088
  vs gradient 0.032 (cap 1.25x = 0.040). KL under a small logit change grows roughly as (delta)^2, so a g cut should lower
  drift roughly as g^2 (g about 0.67 gives 0.040 if linear in B: an expectation, not a measurement), while recall may or may not
  survive the cut (greedy answer margins shrink). M-scale never measured recall after a single-night cut on the battery.
- **Not in scope / caveat.** The life-level M-scale verdict is not a verdict on this one-night use, but it is a warning:
  M-scale SABE (cut after the night, multi-night 1.5B lives) failed the transcript read on 4/4 seeds, with identity and
  told-name loss. A one-night screen GO here would therefore need a multi-night check (whole-transcript read, 1.5B) before any
  adoption or default flip; the expected outcome below is lowered accordingly.
- **Prerequisites.** As the local-rule prereg (no temporal credit; transposed-weight feedback is a declared scaffold). The cut
  itself is a host constant g (a declared scaffold like K in M-scale; the brain does not choose g). Biology: SHY (global
  synaptic downscaling in sleep) is the motivating prerequisite, here a uniform multiplicative scale on B.

## Method
`brain/experiments/local_sleep.py --downscale-grid "1,0.8,0.6,0.4,0.25" --downscale-factors B` (default off; with no
downscale flag the output has the same structure as before: same keys, no `rows`, only host/host_env/commit and per-arm `feedback` added, tested). One night per arm and seed; then adapters are copied, `PlasticCortex.downscale(g, "B")`, the
`sleep_probe.evaluate` battery run, adapters restored; one row per g (`rows` in the JSON, key `arm@g<g>`). Every JSON records
host (`socket.gethostname()`), host_env (env HOST, if set), commit (env COMMIT or `~/brain-lane/route_repo/COMMIT`; `shy_dev.sh`
exports it and exits if empty), g, factors and the preset each arm really used.
Same seeds, steps (240), replay and keep texts in every arm. g = 1 is the plain night (the screen's own number) and is a REFERENCE row only: it is never picked as g*.

## Arms (per seed)
1. `local@(lr*, g*)`: transport preset, decay 1e-3, lr* and g* from dev; also read at g = 1 (same night) as the unscaled reference.
2. `gradient@g*` (matched control): the gradient sleep cut by the SAME g*; also `gradient@1` (the screen's reference).
3. `local_shuffled@(lr*, g*)`: shuffled episodes, same cut (six-seed run only; not run in dev).
4. `no_sleep`. (`local_random` is not run: it diverged at 0.4B in dev, KL 13-14.)

## Dev (seed 7, 0.4B only; frozen before six seeds)
Grid: lr in {0.1, 0.3} x g in {1, 0.8, 0.6, 0.4, 0.25} evaluated for `local` (transport, decay 1e-3, as the screen's pick), plus the
gradient arm over the same g. Script `brain/experiments/results/2026-11-27-local-sleep/shy_dev.sh`: 3 sleeps (gradient, local
lr 0.3, local lr 0.1), each evaluated at 5 g (15 batteries). `local_shuffled` is NOT run in dev (the pick does not use it).
**Pinned references** (from `brain/experiments/results/2026-11-27-local-sleep/dev_s7_04b/ref.json`, not re-measured): gradient
KL 0.03234, so the pick drift cap = 1.25 x 0.03234 = **0.04043**; `no_sleep` recall = **0**, so the recall floor is 0 + 2 = 2.
**Pick rule:** g = 1 is excluded (reference only); candidates are (lr, g) with lr in {0.1, 0.3}, g in {0.8, 0.6, 0.4, 0.25}.
Among candidates with `local` drift <= 0.04043 and recall >= 2 on seed 7, pick max hippocampus-off recall; ties to the larger g
(least intervention), then lower drift. If no candidate qualifies, the six seeds are NOT run and the verdict is **NEGATIVE-DEV**
(this method killed on the dev seed; no grid extension, no other g, no other lr). One dev seed, discrete recall /12: noise is
large; accepted and disclosed.

## Six-seed rule (0.4B; seeds 42, 43, 44, 100, 101, 102; written before data)
Which rows count: only the `g*` and `lr*` rows (and the `g = 1` same-night rows named below) enter the verdict. Other g rows are
report-only and cannot change g*, lr* or the verdict. Means across seeds unless stated. G1 = mean `gradient@1` recall,
Gg = mean `gradient@g*` recall.
Clauses (all on the 0.4B six seeds):
- (a) mean `local@(lr*,g*)` recall >= max(0.5 x max(G1, Gg), no_sleep mean + 2).
- (b) `local@g*` recall > `local_shuffled@g*` recall on >= 5 of 6 seeds AND mean `local_shuffled@g*` <= 1.0.
- (c) mean `local@g*` drift <= 1.25 x mean `gradient@1` drift.
- (d) mean `local@g*` false recall <= mean `gradient@1` false recall + 0.5.
- (e) matched control, BOTH gated: (e1) mean `local@g*` recall >= 0.5 x Gg, AND (e2) mean `local@g*` drift <= 1.25 x mean
  `gradient@g*` drift. So the local rule is judged against the gradient sleep given the same cut on recall and on drift; a pass is
  not just "downscaling helps any sleep".
- (f) `local@1` (same night, no cut) drift > 1.25 x mean `gradient@1` drift, i.e. the cut is what brought drift under the bar.
Verdicts:
- **GO-SHY** = all of (a)-(f).
- **GO-PLAIN-LOCAL** = (a)-(e) hold and (f) fails: the earlier screen's drift problem was not present on these seeds, so the result
  reads as a plain local-rule GO and is NOT credited to SHY.
- **UNDEFINED** if G1 < 2 (no reference), or `local` adapter norm zero or non-finite on any seed, or mean `local_shuffled@g*`
  recall > 1.0 (the control moved; the recall measure is not specific to true episodes).
- **NEGATIVE** otherwise: a verdict on a local transport rule with a one-shot B downscale, not on local learning or SHY; judge
  against the missing temporal-credit prerequisite before the next method.
A GO at 0.4B does not pass the milestone: 0.1B must also be re-read at the frozen (lr*, g*) (recall bar +2 at 0.1B) and the 1.5B
step needs its own prereg. `AB` factors are not run (an allowed variant only if B-only g* is fixed first; any use needs its own line).
Report next to the verdict: recall and drift at each g per arm (all rows), `local@g*` vs `gradient@g*` drift, ||A||, ||B|| (logging only).

## Compute and memory (estimates; nothing measured here)
CPU fp32 only. A sleep is unchanged in cost (the cut and battery are cheap next to 240 steps, but each battery is a full
12 + 12 + 4 greedy-answer evaluation, so 5 g cost 5 batteries per night: dev has 15 batteries; time unmeasured).
Memory: the 0.4B dev ran under memcap 20 on pool40 (23 GB); no RSS was recorded and the logs are not available, so peak RSS
at 0.4B is UNMEASURED. The screen prereg guessed 8-10 GB (weights 1.6 GB fp32). Whether 0.4B fits pool41/42 (15 GB) under memcap 13
is therefore unknown: run dev on pool40 first; `shy_dev.sh` wraps the run in `/usr/bin/time -v` (when present) so "Maximum resident set
size" lands in each log; only then place six seeds (pool41/42 under memcap 13 if RSS < 12 GB, else pool40 / AWS lane).
Run: `ssh pool40` then `cd ~/brain-lane/route_repo && COMMIT=$(cat COMMIT) bash brain/experiments/results/2026-11-27-local-sleep/shy_dev.sh`
(after `tools/mem_ok.sh`).

## Expected (my estimate, not measured)
Dev: about 40% that some (lr, g) is under the drift cap with recall >= 2 on seed 7 (g cuts drift roughly as g^2 but may
erase the 4/12 margin); given dev passes, about 20% GO-SHY at six seeds (lowered from 30% after the M-scale SABE transcript
failures; (e) and (f) are the likely failing clauses, and a GO-SHY would still need the multi-night check before adoption).
