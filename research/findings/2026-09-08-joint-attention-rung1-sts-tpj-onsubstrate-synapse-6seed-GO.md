---
type: finding
status: contributing
date: 2026-09-08
lane: perception
mechanism: joint-attention-gaze-following-onsubstrate-sts
runner: research/runners/_joint_attention_derisk.py
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_joint_attention/summary_6seed_spiking_sts.json
  - research/findings/raw/_joint_attention/summary_6seed.json
---

# Joint attention rung-1: the STS-TPJ object-cell read moves onto REAL cross-region synapses (6-seed GO) — a population POOL read-out, not a single labeled neuron, is required to avoid a per-cell heterogeneity confound

## Verify-first (what this checked before building)

Per the perception-lane mandate: `.venv-rag/bin/python tools/rag/rag_search.py "joint attention STS-TPJ
on-substrate synapse rung population pool" 5 --corpus finding` and `git log --all --oneline --grep='joint.attention'`
both return only the 2026-08-26 base finding and its landing commits (`c8f330c52`/`abbb18b2b`/`9d75389c0`) — no
prior attempt at this rung exists. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` has no joint-attention row (the
capability is not reachable from `/api/brain-chat`; consistent with the base finding's own honesty section). Two
other perception-lane candidates were checked and found ALREADY WORKED PAST THE OBVIOUS NEXT STEP: board #88's
named "learned cross-position pooling" lever was already run and closed NO-GO-through-the-existing-pooler by
`2026-08-19-laneD-cross-position-OR-pool-opens-invariance-...` (the correct downstream is a spiking HMAX S->C
stack, i.e. board #72, which is blocked on production lacking any live visual input to `/api/brain-chat` — not a
same-session-sized task); board #173's named next mechanism (a Turrigiano homeostatic-scaling warm-up to harden
the BCM edge-detector PARTIAL) was already run to a decisive two-axis NO-GO
(`2026-08-27-b1-v1-selforg-bcm-warmup-dose-pilot-NOGO.md`) with its own next mechanism (a structural-plasticity
companion process, Overman & Clopath 2024) being a `sim/`-level build, not a same-session de-risk. Joint attention
(#153) was the genuinely ready item: de-risked GO, not yet closed on its own two named honesty-boundary shortcuts,
self-contained to one runner file, no `sim/` edit, no GPU.

## Claim

The 2026-08-26 joint-attention GO documented two host shortcuts under this project's brain-based-only standard:
"the STS-TPJ object-cell read `s[k] = W_obj[k,:] @ rates` is a HOST synaptic-sum over the gaze-ring spike rates"
and "the spotlight winner is a HOST argmax over accumulated spike counts," and named rung-1 as replacing the host
synaptic-sum with "an on-substrate direction-tuned pathway so the object-cell drive is computed by real synapses
on gaze-ring spikes." This finding builds and de-risks that rung: the SAME direction-tuned tuning
`W_obj[k,i] = relu(cos(theta_i - phi_k))**sharpness` is now written as REAL cross-region synapse weights
(`SimulationBridge.set_pathway_weights`, row=pre/col=post per `sim/bridge.py`'s own CSR-construction convention)
from the gaze ring onto a population of object-selective cells; the simulator's own per-step current integration
performs the weighted sum that host numpy previously computed. At 6 seeds the runner's own verdict is `GO`,
matching the base finding's four-way anti-cheat structure. Rung 2 (the spotlight's own host argmax read-out,
`fswta_drive`, a SHARED primitive across ~5 other lanes) and rung 3 (driving the schema from a live scene instead
of a synthetic per-trial layout) are NOT attempted here — named as the next rungs below, unchanged from the base
finding.

## Result — 6 seeds (42/43/44/100/101/102), chance 0.167, `--spiking-sts` (default off = byte-identical)

`research/runners/_joint_attention_derisk.py --seeds 42 43 44 100 101 102 --n-trials 60 --spiking-sts` (defaults:
`--sts-gain 6000 --sts-w-scale 20 --sts-settle 150 --sts-n-pool 48`), CPU numpy. Artifact
`research/findings/raw/_joint_attention/summary_6seed_spiking_sts.json` -> `go: true`, `checks` all true, band =
chance + 0.10 = 0.267.

Values below are rounded to 3dp from each cited artifact's `aggregate.*` fields (raw: host
`0.9778333333333333`/`0.0`/`0.17500000000000002`/`0.17766666666666667`; on-substrate
`0.9111666666666666`/`0.0`/`0.164`/`0.17766666666666667`). <!--derived-->

| arm | align_acc (GO >= 0.85) | align_acc_lesion (<= 0.267) | align_acc_scramble (<= 0.267) | align_acc_blind (<= 0.267) |
|---|---:|---:|---:|---:|
| host (`summary_6seed.json`, unchanged) | 0.978 | 0.000 | 0.175 | 0.178 | <!--derived-->
| on-substrate synapse (`--spiking-sts`, this finding) | 0.911 | 0.000 | 0.164 | 0.178 | <!--derived-->

Per-seed on-substrate `align_acc` / `lesion` / `scramble` / `blind`:
- seed 42: 0.950 / 0.000 / 0.217 / 0.100
- seed 43: 0.783 / 0.000 / 0.217 / 0.267
- seed 44: 0.967 / 0.000 / 0.133 / 0.183
- seed 100: 0.933 / 0.000 / 0.150 / 0.183
- seed 101: 0.917 / 0.000 / 0.167 / 0.133
- seed 102: 0.917 / 0.000 / 0.100 / 0.200

5 of 6 seeds individually clear the 0.85 align bar (seed 43 alone sits at 0.783); the runner's gate is on the
**6-seed aggregate** (0.911, i.e. `aggregate.align_acc` above <!--derived-->), which clears
it, and every anti-cheat holds in aggregate. `align_acc_blind` at seed 43 (0.267, exactly at the band ceiling) is
IDENTICAL to the host arm's own seed-43 value — expected, since the blind control is a pure function of the
trial's gaze/target draw (`np.random.RandomState(seed)`), which the `--spiking-sts` flag does not touch; it is not
a new borderline this rung introduced.

## The anti-cheats — unchanged interpretation, now measured on the on-substrate read

1. **Held bar (a).** Aggregate `align_acc` 0.911 <!--derived--> >= 0.85: the spotlight tracks the partner's actual
   gaze target through the new synaptic pathway, not just through the unchanged host pathway.
2. **Lesion (b).** `s_les = zeros(K)` is unchanged by this rung (severing the schema output is still "no
   gaze-derived drive at all," independent of whether the drive, when present, arrives via host arithmetic or real
   synapses) — 0.000 aggregate, `docs/TERMS.md`'s lesion condition holds (no plasticity is enabled anywhere in
   this bridge, so there is nothing for a severed weight to regrow within a trial).
3. **Scramble (c).** 0.164 aggregate: swapping in another trial's gaze angle still collapses the on-substrate
   read to chance, confirming the synaptic drive rides the ACTUAL gaze current, not a residual/cached state from
   a prior trial.
4. **Not-a-copy (d).** 0.178 aggregate <!--derived-->: layout-blind decoding stays at chance regardless of which
   read computes the intact score.

## The obstacle found and fixed: single-neuron heterogeneity is a non-averaging identity bias

The first working version used ONE neuron per object cell (direct port of the host's `K`-length score vector).
Direct argmax over its firing rate (bypassing the spotlight organ entirely) plateaued at ~0.70-0.78 regardless of
gain, weight scale, sharpness, or settle-window length (1-500 steps) — settle-length insensitivity past ~150
steps was the tell that this was not sampling noise. Inspecting individual failures showed a specific object
INDEX (not angle) winning far more often than its gaze-alignment justified (e.g. seed 42 trial 19: true target at
angular distance 0.037 rad from gaze scored LOWER than a distractor at 1.974 rad <!--derived--> — from an ad-hoc
inspection script run during this session's debugging, not saved as a repo artifact). Reading the built bridge's own
per-neuron parameters located the cause: `cfg.enable_brain_region_framework`'s heterogeneous Izhikevich draw gives
each of the 6 single object neurons a different firing threshold (`cp_neuron_firing_thresholds` ranged over
~18.4 mV across the 6 cells at seed 42) — a real, intentional biological-heterogeneity feature of this framework,
but because one fixed cell stands for one object identity across every trial of a seed, that cell's idiosyncratic
excitability becomes a PERSISTENT per-identity bias that does not average out and can outweigh the actual
gaze-driven signal.

**Fix:** replace the single labeled neuron per object with a POOL of `n_pool` independently-heterogeneous neurons
(`--sts-n-pool`, default 48) reading out by MEAN firing rate. This is the standard population-coding remedy for
independent single-unit noise/heterogeneity (Averbeck, Latham & Pouget, *Nat Rev Neurosci* 7:358-366, 2006, on how
population averaging over independent neuronal variability recovers a cleaner population signal than any one
unit carries) applied to the object side of this schema, mirroring what the gaze ring already does with its own
`n_dir`-neuron population code for direction. Sweeping ONLY pool size, with gain=6000, w_scale=20, sharpness=4.0
and settle=100 held fixed at seed 42, direct argmax accuracy (bypassing the spotlight organ) rose from ~0.70
(`n_pool=1`) to 0.87 (`n_pool=24`) to 0.90 (`n_pool=32`); `n_pool=48` (with settle raised to 150 through the full
spotlight pipeline) is the value that cleared the 6-seed GO bar above, not a further-optimized point on this same
sweep. This is a genuine mechanism finding, not a tuning footnote: a labeled-line
single-cell code for a small identity set is fragile to this framework's own per-neuron heterogeneity in a way a
population code is not.

## What this is NOT (honesty boundary; brain-based-only standard, unchanged categories from the base finding)

- **Still NOT fully spiking end-to-end** (`docs/TERMS.md` sense: "spiking with a host read-out"). The gaze ring,
  the new object-cell pools, and their connecting synapses are real neurons and real synapses; `fswta_drive`'s
  winner read (`np.argmax(acc)` over accumulated spike counts) remains a host argmax — rung 2, not attempted here.
  `fswta_drive` is a SHARED primitive (imported by the D3/event/reslm/mouth/joint-attention runners); converting
  it is a cross-lane change out of this lane's scope, named again below rather than touched.
- **NOT wired / integrated / closed.** Still a runner-only, default-off (`--spiking-sts`) de-risk; joint attention
  remains unreachable from `/api/brain-chat` (rung 3, blocked on the same "no live visual input to the chat loop"
  constraint noted on board #72/#153).
- **The pool's per-neuron heterogeneity draw is still host-configured** (via `cfg.enable_brain_region_framework`),
  as it is for every other bridge in this codebase that uses that framework — this finding does not touch or
  claim anything new about that.

## Non-claims

- Does not retract, reweigh, or re-run the base 2026-08-26 finding. Its host-path 6-seed artifact is cited
  unchanged in the table above. A stash-and-rerun control (this session, not saved as a repo artifact) found the
  DEFAULT (`--spiking-sts` off) path on both the pristine pre-edit file and this edited file reproduce each other
  exactly, but land ~0.3% below the committed artifact's own `align_acc` <!--derived--> — a small pre-existing
  run-to-run non-determinism in this environment unrelated to this rung's change (confirmed present on the
  unmodified file too), far inside every gate's margin, not investigated further here.
- Does not claim `n_pool=48` is the smallest pool size that would clear the GO bar — it is simply the first size
  tested that did, at 6 seeds; a smaller pool was not swept to a floor.
- Does not claim the population-pool fix generalizes to every small-K labeled-line construction in this codebase;
  it is verified here for this schema only.

## Next (no-defer, in order, unchanged priority from the base finding)

1. **Rung 2**: replace `fswta_drive`'s host argmax winner-read with a neural read-out. Cross-lane (shared
   primitive) — scope as its own arc rather than folding into this lane's runner.
2. **Rung 3**: drive the schema from a live partner cue (scene/conversation) instead of a synthetic per-trial
   layout, once board #72's visual-input-to-`/api/brain-chat` blocker is addressed (shared blocker with #153's own
   "cheap visual consumer" framing).
3. Apply the population-pool-vs-single-cell lesson prospectively: any future small-K labeled-line spiking
   construction in this codebase should default to a pool, not a single cell, unless a single-cell design is
   specifically verified against this same heterogeneity confound first.

## Sources

Kandel, Principles of Neural Science, Ch. 62, Fig. 62-4 (STS-TPJ gaze/biological-motion mentalizing component;
unchanged from the base finding, `research/biology/joint-attention-gaze-following.md`). Averbeck, B., Latham, P.
& Pouget, A. (2006). Neural correlations, population coding and computation. *Nat. Rev. Neurosci.* 7:358-366
(population averaging over independent single-neuron variability; web-verified this session <!--derived-->, DOI
10.1038/nrn1888 <!--derived-->). The base finding this extends:
[`2026-08-26-joint-attention-sts-tpj-other-attention-schema-spotlight-6seed-GO.md`](2026-08-26-joint-attention-sts-tpj-other-attention-schema-spotlight-6seed-GO.md).
The two prior perception-lane candidates checked and found already-advanced-past-the-obvious-next-step:
[`2026-08-19-laneD-cross-position-OR-pool-opens-invariance-trace-pooler-degrades-margin-instrument-underreads.md`](2026-08-19-laneD-cross-position-OR-pool-opens-invariance-trace-pooler-degrades-margin-instrument-underreads.md),
[`2026-08-27-b1-v1-selforg-bcm-warmup-dose-pilot-NOGO.md`](2026-08-27-b1-v1-selforg-bcm-warmup-dose-pilot-NOGO.md).
