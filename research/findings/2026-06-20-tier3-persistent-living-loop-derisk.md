# Tier-3 persistent living loop — cheap-first de-risk: GO, 6 seeds (the first persistent living primitive)

**Date:** 2026-06-20
**Status:** **GO, 6/6 seeds (rate-proxy / algorithm level, CPU).** The Tier-3 artificial-life capstone's sub-gap 1
(the recommended LOW-risk cheap-first piece from the scoping `2026-06-20-tier3-artificial-life-capstone-deep-research.md`,
commit `4d8ec213`) — **a continuous `live()` loop in which a self-generated homeostatic drive keeps the agent ALIVE
and the internal life-state PERSISTS across a reset** — is GO. The first living-loop primitive (survival +
persistence) is reachable on the existing substrate, decoupled from the deferred dendrite wall.
**Runner:** `research/runners/persistent_living_loop_derisk.py`. **No `sim/` edit** (reuse-by-import).

---

## 0. Top-line

The scoping verified that the motivational DRIVE is already de-risked GO-6-seed
(`2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`) and the limbic reward/value/dopamine core is co-resident on
the merged one-brain (`co_resident_limbic`), but that **the honest VERIFIED GAP is the assembly: there is no
continuous outer loop in which an interoceptive drive persists across resets and motivates the agent** (the
validated pieces are bounded per-episode function calls, not a life). This de-risk builds exactly that minimal
persistent living loop and falsifies it cheap-first. **Result: GO 6/6** — the agent keeps itself alive over a
continuous life from a self-generated intrinsic drive (never crashes) AND a reload resumes its EXACT internal
state (energy + drive + learned policy + position), not a blank slate; lesioning/yoking the drive starves it and a
no-persistence cold-start visibly re-warms. This converts the one-brain from "a battery of demos" into "a life" at
the two Tier-3 primitives (survival + persistence), at the validated rate-proxy level.

---

## 1. What was built (`persistent_living_loop_derisk.py`)

A continuous `live()` step-loop over a persistent `LivingState`, reusing the VALIDATED 2-pool drive
(`TwoPoolDrive` from `_homeostatic_drive_rl_cheap_first_probe.py`) and `BridgeLineage` (`sim/lineage.py`) verbatim
— host code is the body + environment only (the brain-based-only standard); the drive + intrinsic reward are the
"brain" parts, rate-proxied exactly as the 2026-06-17 GO:

1. **Drive wired to the agent (the `homeostatic_hook`-shaped reward).** A body deficit (energy `E∈[0,1]`) rises as
   energy depletes each step; the 2-pool push-pull hunger drive (AgRP/POMC reciprocal inhibition) tracks it; the
   drive biases action selection; reaching the food site reduces the deficit → drops the drive → an **INTRINSIC
   drive-reduction reward** `r = drive_before − drive_after` (Keramati-Gutkin). **No host distance/goal term
   anywhere** (verified by grep: the only `distance` references are comments asserting its absence).
2. **A continuous `live(state, n_steps)` loop.** The body/drive/policy state lives on a `LivingState` object that
   mutates IN PLACE — no per-episode reset of the internal life. The agent online-Q-learns a self-directed
   food-seeking policy from the intrinsic reward (the action→direction map is REMAPPED per seed — the validated
   load-bearing anti-cheat — so it cannot reach food by default and must LEARN from `r`).
3. **Persistence across a reset.** The full `LivingState` (body-energy + drive pools + learned Q + position + RNG)
   is saved through `BridgeLineage` (atomic save via a custom `save_fn`); the process "dies" (`del`); a reload
   reconstructs the EXACT state and the agent resumes its SAME life. (At the rate-proxy level the `LivingState`
   stands in for the spiking bridge's neuron/synapse state, which `BridgeLineage` already persists atomically for
   the production path.)

The survival dynamics (L=6 corridor, deplete 0.015/step, refill 0.3) match the validated 2026-06-17 sustained-
agency GO: a learned policy reliably survives while random wandering reliably crashes.

---

## 2. Result — GO, 6/6 seeds (42/43/44/100/101/102, CPU)

| seed | check 1 corr(deficit,drive) sweep | check 2 survival (ALIVE minE / LESION minE / YOKE minE) | check 3 persist | verdict |
|---|---|---|---|---|
| 42  | **+0.99** | **0.955** / 0.055 / 0.000 | exact ✓ | **GO** |
| 43  | **+0.99** | **0.865** / 0.000 / 0.000 | exact ✓ | **GO** |
| 44  | **+0.99** | **0.955** / 0.000 / 0.000 | exact ✓ | **GO** |
| 100 | **+0.99** | **0.925** / 0.000 / 0.000 | exact ✓ | **GO** |
| 101 | **+0.99** | **0.955** / 0.000 / 0.000 | exact ✓ | **GO** |
| 102 | **+0.99** | **0.715** / 0.000 / 0.000 | exact ✓ | **GO** |

- **Check 1 — the drive is neural + tracks the body deficit.** corr(deficit, drive) = **+0.99** on a controlled
  deficit sweep, all 6 seeds (the scoping's GO band ≥ +0.9). (Measured on a regulation-INDEPENDENT sweep, not the
  lived trace — a successfully-regulating agent stays so near setpoint that its lived deficit barely varies, which
  COMPRESSES the lived corr and would perversely penalize BETTER homeostasis; the lived corr is reported as a
  secondary, +0.83…+0.97, and the load-bearing proof the drive *tracks* the deficit is the lesion/yoke collapse in
  check 2.)
- **Check 2 — self-directed survival ("alive over time").** The drive agent keeps energy in the healthy band the
  whole second half (band occupancy 1.00), **NEVER crashes** (min-energy **0.715…0.955**, well above the 0.1 crash
  floor; 0% crash), by self-directed food-seeking with NO external goal — while the LESION (drive frozen → r≡0 →
  no learned policy) and YOKE (drive shuffled → reward uninformative) controls both **CRASH** (min-energy 0.00…0.055).
- **Check 3 — persistence across a reset ("self over time").** A reload resumes the EXACT internal life-state on
  every seed: body-energy, drive pools, learned Q-policy, position, and lifetime step-count all match the save
  point exactly (atol 1e-9). The agent resumes its life, not a cold start.

```
SIM_BACKEND=numpy python -m research.runners.persistent_living_loop_derisk --seeds 42 43 44 100 101 102
```

---

## 3. Anti-cheat table — ALL collapse / hold

| Anti-cheat | Mechanism | Result (6 seeds) |
|---|---|---|
| **Drive-lesion** (self-direction must collapse) | freeze the drive → `r≡0` → no learned policy | **STARVES** — min-energy 0.00…0.055 (vs alive 0.715…0.955); survival is the drive's doing, not perception/a leftover goal |
| **Yoked-random** (coupling to the deficit is load-bearing) | replace the drive with a shuffled signal of matched marginal stats, no relation to the body deficit | **STARVES** — min-energy 0.000; "any extra signal makes it move" is falsified — only the INFORMATIVE deficit-coupled drive sustains behaviour |
| **Reward-provenance** (brain-based-only bar) | `r = drive_before − drive_after`, computed from the drive pools | **intrinsic by construction** — grep confirms NO `r = f(distance_to_food)` host term; the only `distance` mentions are comments asserting its absence |
| **No-persistence** (persistence is load-bearing) | identical loop, but cold-start the internal state after the reset instead of reloading | **cold-start visibly re-warms** — the persisted resume never dips (min-energy ~0.985) while the cold-start dips during re-learning (min-energy 0.385…0.880), all 6 seeds; the persisted save carried a learned policy the cold reset lacks |
| **No-confab moat** (conversation safety) | no composer/parser in this loop | **untouched by construction** — the cross-modal "one animal" check + the moat-assertion are the spiking-merged-bridge follow-on (§5) |

A subtlety the no-persistence check correctly handles: because the loop is deterministic per-seed, a cold-start
agent eventually re-converges to the SAME policy — so a long-window comparison would FALSELY report "no
difference." The persistence pays off precisely in the **avoided re-warm transient**; the check measures the DEPTH
of the early-window energy dip (robust to its DURATION; fast-learning seeds re-warm in a few steps), not a mean
that the recovery would dilute.

---

## 4. `sim/` edit: NONE

As the scoping predicted (the prior was no-edit, additive-only, templated on the `co_resident_limbic` lift which
needed none), **no `sim/` edit was required.** The runner reuses `TwoPoolDrive` and `BridgeLineage` by import. The
only structural pieces are a new runner + this finding. `git status sim/` is empty.

---

## 5. Honest scope + the noted follow-ons

This is the **FIRST living-loop primitive (survival + persistence) at the validated rate-proxy / algorithm level**
— deliberately decoupled from the deferred dendrite wall, exactly as the scoping framed it: survival and
persistence are demonstrable WITHOUT a converged spatial policy (the discriminator is crash-avoidance + state-
resumption, not spatial-policy optimality — the 2026-06-17 rate-proxy already showed survival is GO without it).
It is the smallest thing that makes the merged one-brain *a life rather than a battery of demos*. It is NOT yet:

- **The spiking-bridge co-resident realization.** The cheap-first level (per the prompt + the 2026-06-17 GO that
  was itself rate-proxy) validates the LOOP and its persistence. The brain-based spiking realization — a
  `co_resident_drive` slice on `build_merged_nav_conv_bridge` (templated EXACTLY on the proven `co_resident_limbic`
  lift: append the 2-pool AgRP/POMC drive region after the existing slices, nav-inert, default-off byte-identical,
  per-region `enable_homeostasis` mask for the operating point) + `run_moving_goal_episode(homeostatic_hook=...)`
  wiring the neural drive-reduction reward into the validated BG-cascade learner — is the noted follow-on. The
  localized risk there is check-1's f-I operating point surviving co-residence (the systemic merged-config
  sensitivity the limbic lift hit and RESOLVED via the per-region homeostasis mask; the fix template exists).
- **The cross-modal "one animal" property (scoping check 4).** When hunger raises the shared `dopamine`, the
  already-built `enable_da_salience_gate` would tighten the conversational moat gate (moat-safe by construction —
  it can only tighten abstention) — the SAME drive touching BOTH halves of the one brain, with the moat ASSERTED
  byte-unchanged. A cheap follow-on once the drive is co-resident.
- **The learned spatial policy from intrinsic reward.** For the drive's reward to *carve a learned place→action
  navigation policy* (a hidden/relocating goal solved by reward, not perception), the actor-critic credit-
  assignment must work — and that hit a 3rd rigorous NEGATIVE on 2026-06-19 (the F-S-G water-maze de-risk,
  "resolves toward the DENDRITE"). **This stays the deferred Tier-4 dendrite wall** and is NOT on this cheap-first
  critical path; the living loop is demonstrated on the validated `sustain-your-energy-by-eating` behaviour where
  the reward is load-bearing for survival even with a simple spatial policy.

---

## 6. Verdict

**GO 6/6 (rate-proxy).** The first persistent living loop: a self-generated homeostatic drive keeps the merged
one-brain agent ALIVE over a continuous life (never crashes, NO external goal) and its life PERSISTS across a reset
(exact internal-state resumption). Drive-lesion → starves; yoked-random → starves; reward is intrinsic
drive-reduction (no host goal); no-persistence cold-start visibly re-warms; the no-confab moat is untouched by
construction. **No `sim/` edit.** ⇒ the two Tier-3 primitives (survival + persistence) are reachable on the
existing substrate; the spiking co-resident realization + the cross-modal demonstration are cheap follow-ons, and
the learned spatial policy stays the honest deferred dendrite wall.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners.persistent_living_loop_derisk --seeds 42 43 44 100 101 102
```
