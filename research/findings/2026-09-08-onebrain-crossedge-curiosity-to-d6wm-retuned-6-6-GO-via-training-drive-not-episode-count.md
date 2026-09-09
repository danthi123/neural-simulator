---
type: finding
status: live
date: 2026-09-08
mechanism: onebrain-crossedge-curiosity-d6wm-retune
board: Curiosity lane — the loose end from the 2026-09-08 6-lane fan-out (workflow's Curiosity agent was still
  running a seed battery when the fan-out closed; no commits landed). Picked up fresh in an isolated worktree.
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_traindrivescale1.5_6seed.json
  - research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_nepisodes300_scout_negative.json
  - research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_traindrivescale1.25_scout_partial.json
runner: research/runners/_onebrain_crossedge_curiosity_to_d6wm.py
builds_on:
  - research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md
  - research/findings/2026-09-08-onebrain-curiosity-d6-semantic-drop-competitive-allocation-confound-CLOSED.md
---

# curiosity.ask -> d6.w0 cross-edge, re-tuned: the read-isolation-corrected NO-GO 3/6 is now GO 6/6 — the lever is the TRAINING-time induction drive, not episode count

**One-line:** the 2026-09-02 read-isolation fix left this cross-edge's own honest residual unresolved ("the
mechanism may need re-tuning against the now-trustworthy read ... not attempted here"). Two levers were scouted.
Raw episode count (3x, 100->300) is a **NEGATIVE** — the 3 marginal seeds' grown weight is genuinely PLATEAUING
near a seed-specific fixed point, not under-trained. Scaling ONLY the training-time `ask` induction current
(read-time currents at production strength, untouched) is a **POSITIVE**: at `train_drive_scale=1.5` all 6 seeds
clear the floor, GO 6/6, fully lesion-attributable, byte-identical-off. No `sim/` edit; no production default
touched (this remains a de-risk runner, not a flip).

## 1. Where this picks up

The 2026-09-08 6-lane roadmap fan-out landed 5 lanes on `main`; the Curiosity lane's own agent was still running
a seed battery when the workflow closed and committed nothing (`GAP_CLOSURE_MISSION.md`'s own CURRENT STATE names
this as "the one loose end — pick it up"). Verify-first found no recoverable partial battery (no uncommitted
artifact, no live process, no un-merged branch carrying curiosity-specific work) — the closest live thread was
Vikunja #197's own description, updated the SAME session by a *different* lane (language-wm, commit `60234408f`),
which explicitly names the untouched next step: **"[the semantic-drop fix] does NOT by itself unblock the parent
gate, whose own separate NO-GO 3/6 (a harder, distinct mechanism problem) is untouched."** That parent gate —
`onebrain_crossedge_curiosity_to_d6wm`'s base cross-edge, corrected from an inflated 6/6 to a real 3/6 by the
2026-09-02 read-isolation fix — is the genuinely-ready, highest-value, not-yet-attempted item this session
advances.

## 2. The starting wall (banked, unchanged by this finding)

`research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md`:
seeds 43/101/102 fail `clears_registered_floor` (`|delta_intact| < INTACT_FLOOR=0.008`) while 42/44/100 pass, on
every seed fully lesion-attributable (`frac_attributable=1.0`). That finding's own honest residual (§6) named the
untried next step verbatim: *"the mechanism may need re-tuning against the now-trustworthy read ... that
re-tuning is not attempted here."*

## 3. Lever 1 (SCOUTED, NEGATIVE) — raw episode count

The banked trajectory for 43/101/102 was still nominally rising at `ep=100` (not flat), so the first hypothesis
was plain under-training. Re-run at `N_EPISODES=300` (3x, `research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_nepisodes300_scout_negative.json`):

<!--derived-->
(rounded to 4dp for legibility; the "@100ep" column is the ALREADY-BANKED
`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json` values for these 3 seeds,
reproduced here for the side-by-side comparison; the "@300ep" column is this row's own cited artifact,
`_onebrain_crossedge_curiosity_to_d6wm_nepisodes300_scout_negative.json`, at full precision `1.6149...`,
`1.6478...`, `1.7605...` for grown and `-0.0035`/`-0.0055`/`-0.0030` for delta_intact.)

| seed | grown @100ep | grown @300ep | Δintact @100ep | Δintact @300ep |
|---|---|---|---|---|
| 43  | 1.589 | 1.615 | -0.0035 | -0.0035 |
| 101 | 1.622 | 1.648 | -0.0055 | -0.0055 |
| 102 | 1.661 | 1.761 | -0.0020 | -0.0030 |

Tripling the episode count moved grown weight by only 0.03-0.10 and left `delta_intact` essentially flat — these
3 seeds are genuinely converging to a seed-specific fixed point well under `HMAX=6.0`, not merely
under-trained. **Banked negative: raw episode-count scaling is not the lever for this wall.**

## 4. Lever 2 (SCOUTED, POSITIVE) — the training-time induction drive, decoupled from the read-time drive

The module's `ASK_DRIVE_PA=600` is used BOTH to induce the edge during `train()` (paired with `w0`'s tonic
co-drive) AND, unchanged, to drive `ask` alone during the scored `'novel'` read. Biologically this conflates two
different quantities: induction-protocol intensity and later readout-signal intensity (LTP induction strength
and the strength of the signal that later reads out the potentiated synapse are not the same knob in a real
synapse). `run_seed()`'s new `train_drive_scale` parameter multiplies ONLY the train-time `ask_drive_pa` fed into
`AskToW0Pool.train()`; `load_pa` and every read-time current are left at their original values.

A quick 3-seed scout (43/101/102) at `train_drive_scale` in {1.0, 1.5, 2.0}:

<!--derived-->
(this scout ran via a throwaway diagnostic script, not saved as a committed artifact — an honest gap, not a
withheld one. The `scale=1.0` column reproduces the ALREADY-BANKED
`_onebrain_crossedge_curiosity_to_d6wm_readfix_6seed.json` delta_intact values for these 3 seeds; the `scale=1.5`
column is reproduced, saved, and fully cited in §5's own artifact below; the `scale=2.0` column alone is
UNARCHIVED — used here only to show the dose-response is monotonic, not as a standalone claim.)

| seed | scale=1.0 Δintact | scale=1.5 Δintact | scale=2.0 Δintact |
|---|---|---|---|
| 43  | -0.0035 | -0.0130 | -0.0200 |
| 101 | -0.0055 | -0.0150 | -0.0190 |
| 102 | -0.0020 | -0.0095 | -0.0155 |

Monotonic, seed-consistent, and well clear of the floor by `scale=1.5` — a real dose-response, not overfit noise.

## 5. The full 6-seed confirmation (this finding's own artifact)

`--train-drive-scale 1.5` across the full battery
(`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_traindrivescale1.5_6seed.json`):

<!--derived-->
(grown weight rounded to 3dp for legibility from the cited artifact's full-precision values: seed 42
`3.490537...`, 43 `3.268718...`, 44 `3.000558...`, 100 `2.774332...`, 101 `3.297610...`, 102 `2.782245...`.
delta_intact/delta_lesion/frac_attrib/GO are the artifact's own `interaction.per_condition.novel` fields verbatim.)

| seed | grown | Δintact (novel) | Δlesion | frac_attrib | GO |
|---|---|---|---|---|---|
| 42  | 3.491 | -0.0175 | +0.0000 | 1.0 | GO |
| 43  | 3.269 | -0.0130 | +0.0000 | 1.0 | GO |
| 44  | 3.001 | -0.0140 | +0.0000 | 1.0 | GO |
| 100 | 2.774 | -0.0115 | +0.0000 | 1.0 | GO |
| 101 | 3.298 | -0.0150 | +0.0000 | 1.0 | GO |
| 102 | 2.782 | -0.0095 | +0.0000 | 1.0 | GO |

**6/6 GO** — the runner's own `Verdict` machinery (`Vd.decide`, not a metric lifted out of a negative run),
`n_go=6/6`, `lesion_removes_bias` and `byte_identical_off` both hold on every seed. `no_corruption` (frozen-synapse
drift `< 1e-6`) holds on all 6 — the stronger induction current does not leak onto the non-edge synapses
`freeze_rest` protects. Every grown weight (2.77-3.49) stays well under `HMAX=6.0` (46-58% of the bound) — this is
NOT the project's own previously-diagnosed "the clamp dominates the effect" trap; the mechanism has genuine room
to grow further before hitting the ceiling.

**Margin check (robustness, not cherry-picked):** `--train-drive-scale 1.25`
(`research/findings/raw/_onebrain_crossedge_curiosity_to_d6wm_traindrivescale1.25_scout_partial.json`) gives 2/3
on the marginal seeds (43 and 101 clear; 102 at -0.0055 does not) — confirming `1.5` sits past a real threshold
with margin (smallest margin at `1.5` is seed 102's 19% over-floor), not exactly at the boundary of failure.

## 6. What this does and does not change

**Changes:** `research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` gains two new, additive, default-preserving
parameters — `run_seed(..., n_episodes=N_EPISODES, train_drive_scale=1.0)` and CLI flags `--n-episodes` /
`--train-drive-scale` — plus a payload provenance field for each. **At their defaults the runner is byte-for-byte
unchanged** (verified: `--seeds 42` at default args reproduces the banked `grown=2.578103`, `delta_intact=-0.0125`
to full precision).

**Does NOT change:** any `sim/` file; `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`; `webapp/server.py`; the
`_XEDGE_CD6_DEFAULT_ON` flag (still `False`, per commit `afcb3ba7b`'s separate, owner-flagged decision); the
semantic-drop rung's own artifacts (§4 of the 2026-09-02 finding — un-reverified there, still un-reverified here,
since that rung imports this pair's mechanism but was not re-run against the retuned drive in this session). No
production flip, per this lane's explicit scope (rank-1/rank-6 flip files untouched).

## 7. Honest residuals / the next rung

- **The semantic-drop rung (Vikunja #197) is still blocked**, not because of a mechanism problem now, but because
  `_XEDGE_CD6_DEFAULT_ON=False` is a *separate* owner-flagged production decision (2026-09-01's auto-flip premise
  — "validated-GO" — was falsified by the 2026-09-02 correction; whether to re-flip given THIS session's genuine
  re-tuned 6/6 GO is an owner UX call, not made here).
- **If the parent gate is re-enabled at some point, the semantic-drop rung's own 6-seed self-test should be
  re-run against `train_drive_scale=1.5`** (it currently assumes the module's untouched defaults) — not attempted
  here, flagged for whoever next touches this pair.
- **Why these particular 3 seeds converge to a lower Hebbian fixed point at the original drive is not explained
  mechanistically here** (heterogeneous neuron/synapse parameters are the obvious candidate, per this project's
  own per-seed heterogeneity precedent elsewhere, but this finding fixes the CALIBRATION, not the reason for the
  heterogeneity's asymmetric effect).
- **Cupy is not run here** (numpy CPU only, per this lane's compute restriction — the local GPU is in use by
  another arc's latency confirmations). A guarded cupy 6-seed re-verify at `train_drive_scale=1.5` is a cheap
  follow-up once the GPU is free; numpy and cupy have diverged before on adjacent edges (§6 of the 2026-09-02
  finding), so this is not assumed to reproduce identically on cupy without that confirmation.

Functional read-outs only; no phenomenal-experience claim.
