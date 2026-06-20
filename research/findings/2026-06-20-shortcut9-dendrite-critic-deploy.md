# Shortcut #9 host-RETIRED — the dendrite graded-plateau VALUE drives the production nav-RL critic (2026-06-20)

**Item:** DEPLOY shortcut #9's validated mechanism. The graded dendritic-plateau VALUE read-out was validated
end-to-end on a critic bridge (`2026-06-20-dendrite-stage1-snc-calibration.md`: on-bridge SNc burst δ=1.33 =
host 1.30, 6/6 seeds, V graded, all anti-cheats green, NO `sim/` edit). This task wires it into the
**production navigation reinforcement-learning critic**, RETIRING the point-neuron value computation: the
critic's value-of-location V(s) is now computed by the validated graded dendritic plateau instead of the
point-neuron MSN-D1 firing rate. The downstream neural GABA_B→SNc value subtraction (the brain's r−V) is
unchanged.

## VERDICT: __PENDING multi-seed (smoke GREEN; faithful 3→6-seed in flight)__

(Filled below once the multi-seed nav table lands.)

---

## What "host-Gaussian RETIRED" means here (the precise scope)

The production neural critic (`g11_bg_runner.py --enable-neural-critic`, Stage B, 2026-06-08) already does the
r−V subtraction NEURALLY (the `striosome_value → snc` GABA_B/GIRK pathway) — that part was never a host
shortcut. The residual non-graded computation was the **value V itself**: a point-neuron MSN-D1 computing V(s)
through its firing RATE, which (per Mikulasch-Priesemann + the dendrite de-risk's anti-cheat (a)) provably
cannot express the graded analog value the way a dendritic plateau can (the LINEAR point-neuron control sits
flat at δ=1.00; only the graded plateau grades to δ=1.33). This deploy moves the **V computation** onto the
validated graded dendritic plateau. The "host-Gaussian" the de-risk benchmarked against is the Stage-0
`reward_ema`/Gaussian scaffold ceiling (δ≈1.30); the deployed point-neuron critic is the path the dendrite
value replaces, and it is the baseline arm here.

## Wiring summary (the insertion point + the flag)

- **Flag:** `--dendrite-critic` (runner-side, default OFF = byte-identical). CLI knobs:
  `--dendrite-critic-coincidence-k` (default 50, high → the all-or-none plateau stays OFF so the GRADED form
  is the active value read-out), `--dendrite-critic-graded-center/slope/strength` (validated 1.5 / 1.0 / 80).
- **Insertion point 1 — the value pathway** (`build_bg_brain_regions`, the `enable_neural_critic` branch,
  `g11_bg_runner.py:~1911`): the PLASTIC value-learner pathway `vs_place_context → striosome_value` is tagged
  `coincidence_detector=bool(dendrite_critic)`. ON → its synapses route into the per-synapse coincidence mask
  so the graded dendritic plateau reads the WEIGHTED coincident drive (the Poirazi-Mel analog read-out that
  grades with the LEARNED value weight). OFF → `coincidence_detector=False`, the same plastic rate-coded
  pathway as before (zero pathways carry the flag → structurally identical to the pre-edit build).
- **Insertion point 2 — the config** (`run_moving_goal_episode`, the cfg block after the
  `_neural_place_selforg` coincidence block, `g11_bg_runner.py:~4385`): when `dendrite_critic and
  enable_neural_critic and not _neural_place_selforg`, set `cfg.enable_coincidence_detection=True`,
  `cfg.enable_graded_dendritic_plateau=True`, the calibrated `cfg.graded_plateau_center/slope/strength`,
  `cfg.coincidence_weighted_drive=True`, `cfg.coincidence_plateau_strength=0.0` (all-or-none OFF), and
  `cfg.enable_nmda=True` (the per-region mask restricts the Mg²⁺-block kernel to the critic slice).
- **Unchanged:** the `striosome_value → snc` GABA_B subtraction, the reward_us → snc excitation (the r term),
  the actor, the SNc → policy three-factor loop. Only the V *computation* moved.
- **Ignored under `--neural-place-selforg`** (roadmap #5), which ALREADY routes the value through a
  `coincidence_detector` (line 1871-1877) — so dendrite-critic is the deploy of the SAME mechanism on the
  dense-`vs_place_context` critic path.

## Nav-score table (dendrite-critic vs production point-neuron critic vs lesion) — grid-32, multi-goal det, 1800 steps

(Lower sum = better; the agent spends more steps AT the goal. δsum vs baseline ≤ ~25% is the deploy GATE.)

| seed | dendrite-critic (sum) | point-neuron baseline (sum) | lesion: graded-strength=0 (sum) |
|---|---|---|---|
| 42 | _pending_ | _pending_ | _pending_ |
| 43 | _pending_ | _pending_ | _pending_ |
| 44 | _pending_ | _pending_ | _pending_ |
| **mean** | _pending_ | _pending_ | _pending_ |

Smoke (seed 42, 200 steps): dendrite-critic path BUILDS + RUNS; `enable_neural_critic=True`,
`critic_afferent=vs_place_context`, the value-learner weight grew 0.20 → 88.9 (the critic LEARNS V under the
dendrite routing), agent reaches distance 0-1 (at/near goal). No crash.

## Anti-cheat table

| anti-cheat | result | reading |
|---|---|---|
| **(OFF byte-identical)** dendrite-critic OFF == pre-edit | structurally CONFIRMED: 0 pathways carry `coincidence_detector` when OFF; no cfg dendrite block fires; the field default was already `False` | the new flag adds NOTHING to the OFF path |
| **(host/point-neuron baseline positive control)** the deploy ≈ baseline nav score | _pending table_ | the dendrite value drives the policy as well as the point-neuron critic |
| **(the dendrite δ is the value source)** lesion the graded plateau → critic degrades | _pending: `--dendrite-critic --dendrite-critic-graded-strength 0`_ | severing the plateau's value current degrades navigation → the dendrite V is load-bearing |
| **(moat untouched)** the nav critic is array-disjoint from any conversational regions | CONFIRMED by construction: this is the standalone nav bridge (no conversational regions); on the merged bridge the conversational slices are array-disjoint + `enable_graded_dendritic_plateau` is default-OFF for them | the no-confab moat is preserved |
| **(regime fidelity)** the faithful nav config (grid-32, 1800 steps, multi-goal deterministic) | the production benchmark, not a toy | replicates deployment |

## NO `sim/` edit

`git diff --stat -- sim/` is empty. The `cfg.enable_graded_dendritic_plateau` + `cfg.graded_plateau_*` params
already ship (`d69cc0ab` / `52dafaeb`, byte-reviewed). The deploy is runner-side only (one flag + a pathway
tag + a cfg block).

## Files
- Runner (wiring): `research/runners/g11_bg_runner.py` (`--dendrite-critic`; the pathway tag + cfg block)
- Validated mechanism: `2026-06-20-dendrite-stage1-snc-calibration.md` (δ=1.33 6/6),
  `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md` (the on-bridge V), the de-risk runner
  `research/runners/_dendrite_stage1_onbridge_graded_plateau.py`
- Raw nav JSONs: `research/findings/raw/dendrite_critic/`
