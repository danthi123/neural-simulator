# Navigation spiking-superior-colliculus deploy — honest NO-GO + root cause localized (2026-06-19)

**Verdict: NO-GO for deploying the spiking superior colliculus (SC) + neural reward as the merged-bridge
navigation default.** The fully-neural SC-orienting + neural-reward closed loop underperforms the host shortcut
by ~58× on the full moving-goal task. The SC *mechanism* is sound in isolation; the closed *loop* is the failure.
This is the brain-based-only deliverable: a clean spiking organ that underperforms the host scaffold maps a real
substrate boundary. The spiking SC stays **opt-in** (`--spiking-sc`), not the merged default; the conversational
no-confab moat is untouched (the SC slice is synapse-disjoint from the parser/composer by construction).

## The deploy A/B (3 seeds, decisive; grid-32 / 1800 steps; metric = Σ per-phase final-quarter mean distance, lower=better)

| arm | seed 42 | seed 43 | seed 44 | mean | vs host |
|---|---|---|---|---|---|
| **host** (heuristic orient + sign-distance reward) | 2.0 | 2.0 | 2.0 | **2.0** | 1× (optimal floor) |
| **SC-on** (spiking SC orient + neural reward/critic/SNc) | 93.7 | 118.8 | 140.1 | **117.5** | **~59×** |
| **scramble** (scrambled-retinotopy SC + neural reward) | 128.1 | 98.0 | 123.9 | **116.7** | ~58× |

GO bar was SC-on ≤ 1.25× host. SC-on is ~47–70× per seed — unambiguous NO-GO at any seed count (the effect is
~50× the bar and mechanistic, not a marginal variable effect near a threshold; cf. the nav-2a byte-identity gate,
also called on 3 seeds for an exact/mechanistic effect — distinct from the 6-seed rule for *variable* effects).

## Root cause — the neural reward/actor-drive loop, NOT the SC orienting

- **Motor goes silent.** On SC-on seed 42 the actor fires in the warmup window (`motor_counts[0]=[10,10,10,9]`)
  then drops to ~zero; it reaches the goal **8/1800** steps (host: 822). The actor is not driven to navigate over
  the full task.
- **The scramble control localizes it.** scramble ≈ SC-on (116.7 vs 117.5, means within 1%). Scrambling the SC
  retinotopy — i.e. destroying the orienting signal's spatial meaning — does **not** change the navigation
  outcome. Therefore the orienting quality is irrelevant to the failure; the dominant failure is the
  **neural-reward → SNc → critic → actor-drive** closed loop (the actor stays silent regardless of orienting).
- **The SC mechanism itself is sound** (Step-0, 2026-06-19): the `sc_map` bump is a clean Mexican-hat (peak/mean
  35.7×), N1 orienting matches the host 8/8 (incl. diagonals), and the reward is monotone in distance
  (corr(eccentricity, reward_us) = −0.989). The organ works; the loop around it does not sustain navigation.

`--spiking-sc` couples BOTH the SC orienting AND the neural reward (`spiking_reward_us` + `enable_neural_critic`
+ `spiking_snc`, with the host reward write zeroed at `g11_bg_runner.py:7271`); the scramble result isolates the
failure to the reward/drive half.

## Anti-cheat / methodology note

The 1-seed scramble read (seed 42: scramble 128 > SC-on 94) suggested the orienting *did* contribute; the 3-seed
means show it does not (seeds 43/44 flip). The primary NO-GO was so large (50×) that one seed sufficed, but the
*localization* — a comparison of two near-equal-magnitude failing arms — genuinely needed the replication. Lesson
re-confirmed: effect size, not a fixed seed count, sets the rigor; large mechanistic effects resolve at 1–3 seeds,
near-comparable contrasts need the full replication.

## Follow-on (not on the critical path)

The honest negative localizes the next mechanism precisely: a brain-based reward/value/actor-drive loop that
*sustains* navigation over a long moving-goal task (the current neural SNc/critic drives the actor in warmup but
cannot hold it on-policy). This is the same family as the documented #5 value-train δ boundary (the self-org place
value underperforms the host Gaussian) — the neural reward/value substrate works in pieces but does not yet match
the host scaffold for sustained closed-loop control. The spiking SC organ is validated and available opt-in for
when that loop is built.

## Reproduce

```bash
# host baseline
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run --with-conv --seed 42 --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_host_seed42.json
# SC-on (the deploy arm)
SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run \
    --with-conv --spiking-sc --seed 42 --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_on_seed42.json
# scramble-retinotopy control
SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run \
    --with-conv --spiking-sc --scramble-sc --seed 42 --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_scramble_seed42.json
```

Step-0 GO (the organ is sound): `2026-06-19-nav-spiking-sc-deploy-prep.md`.
