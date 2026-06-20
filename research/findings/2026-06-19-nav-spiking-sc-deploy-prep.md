# Tier-2 nav SC deploy — DEPLOY-AND-VALIDATE prep: Step-0 op-point check GO, conversational moat disjoint by construction, Step-1 6-seed A/B handed to the controller (2026-06-19)

**Type:** deploy-and-validate prep (the cheap checks done; the slow 6-seed GPU A/B is the controller's to poll).
**Pre-registered by:** `research/findings/2026-06-19-tier2-nav-spikeification-scoping.md` (commit `b2a5564f`) §4 (the two-step ladder).
**Target:** deploy the already-6-seed-GO spiking superior colliculus (N1 orienting + N5 approach-reward) as the merged
"one brain" default, retiring the host Manhattan orienting heuristic AND the host `sign(distance)` reward in one move.
**Owner standard:** BRAIN-BASED-ONLY ([[feedback_brain_based_only_standard]]). ZERO new `sim/` edit (deploy-and-validate).

---

## Headline

The **single most likely spurious-NEGATIVE cause is CLEARED**: on the merged nav+conversation `SimulationBridge`,
the spiking SC's `sc_map` Mexican-hat WTA bump **FIRES co-resident** at the de-risked merged operating point — it is
**NOT starved** despite the het-off + homeostasis-off merged config (the documented "standalone organ fires ~6-10×
weaker co-resident" boundary, `2026-06-18-merged-limbic-core-lift.md`). Step-0 is a clean, strong GO; the
conversational no-confab moat is **disjoint from the SC slice by construction** (0 cross-edges in `cp_connections`);
and the Step-1 6-seed A/B is wired into one runner with the exact command + `--out` below for the controller to poll.

## Step 0 — co-residence op-point check: GO (peak/mean 35.7×, N1 8/8, corr(ecc,reward_us) −0.989)

Probe: `research/runners/_navsc_merged_opcheck.py` (CPU/numpy, seed 42). It builds the merged bridge with
`nav_critic_spiking_sc=True` (54 regions, 9468 neurons, the SC chain `sc_retina`/`sc_map`/`sc_fs`/`sc_rostral` +
`reward_us` co-resident with the parser+dlPFC), installs the SC wiring at the **merged-tuned op-point** (the de-risk's
promoted values: `w_ret_sc=160`, `w_sc_rec=12`, `sc_map→cortex_X=18`, `sc_retina drive=3500`, `sc_rostral→reward_us=40`),
and re-runs the `sc_map_orienting_probe` / `sc_n5_rpe_probe` falsifiers **on the merged bridge** (the standalone probes
build their own tiny bridge and cannot catch the co-residence starvation). On 8 hand-set (agent, goal) renders + 6 N5
proximity cases:

| Check | Result | Bar | Verdict |
|---|---|---|---|
| **(a) bump alive** | `sc_map` peak **130.8 Hz**, mean 3.7 Hz → **peak/mean 35.7×** | peak ≥ 20 Hz AND peak/mean ≥ 3× | **GO** (a clean single sharp bump, NOT starved) |
| **(b) N1 orienting** | winning `cortex_X` BY FIRING matches host `sc_orienting_cardinal_from_image` **8/8** (incl. diagonals) | ≥ 7/8 | **GO** |
| **(c) N5 reward** | `reward_us` max **109.4 Hz**, **corr(eccentricity, reward_us-rate) = −0.989** | reward_us ≥ 5 Hz AND corr < −0.6 | **GO** (closer goal → more reward_us firing, monotone) |

⇒ **the merged op-point genuinely fixes the starvation.** No op-point knob change was needed beyond the already-
documented merged values (160/12/3500/40); the env vars `SC_RET_SC` / `SC_REC` / `SC_RET_DRIVE` / `SC_ROS_US` carry
them. **Recommendation (for the eventual default flip):** promote 160/12/3500/40 from env-var-gated to the merged
builder default so the deploy is reproducible without the env (the ONE small additive runner change the scoping doc
flagged; default-off byte-identical to standalone). Until then the Step-1 A/B sets them via the env (below).

Reproduce: `SIM_BACKEND=numpy python -m research.runners._navsc_merged_opcheck --seed 42`
(add `--also-standalone-op` for the 80/6/2500/14 contrast).

## Step 2 — conversational-regression: the moat is DISJOINT from the SC slice BY CONSTRUCTION

A structural disjointness check on the same merged+SC bridge (CPU/numpy) walked the full `cp_connections` CSR:

```
SC slice neurons: 2340  (sc_retina + sc_map + sc_fs + sc_rostral)
conv slice neurons: 2166 (parse_conj + parse_role + cortex_ctx + dlpfc_wm)
cp_connections: SC->conv edges = 0   conv->SC edges = 0
ARRAY-DISJOINT (moat-by-construction): True
```

⇒ enabling the spiking-SC nav default **cannot** perturb the conversational comprehension — the nav read-out is
array-disjoint from the parser/composer, so the no-confab moat holds **by construction** (the same argument the
2026-06-19 spiking-decision default-on relied on). The full `is None` no-confab CI gates are GPU-gated (`skipif`
off-GPU); the controller runs them WITH the Step-1 A/B:

```bash
SIM_BACKEND=cupy python -m pytest tests/test_nav_conv_merged_agent.py tests/test_nav_conv_step2b_coresident.py -q
```

(8/8 + 7/7 expected, incl. the three `what_does`/`elaborate`/`describe` `is None` assertions — they exercise the
default merged agent, which is byte-unchanged by the SC arm; the disjointness above is why they pass.)

## Step 1 — the decisive 6-seed merged-bridge A/B (CONTROLLER-OWNED; do NOT run inline)

Wired into `research/runners/_nav_gate_merged_run.py` via an **opt-in `--spiking-sc` arm** (additive, default-off =
byte-identical to the STEP-2a gate). It forwards `enable_spiking_sc` + `enable_spiking_sc_approach` +
`spiking_reward_us` (the SC chain + the synaptic `reward_us`) + `enable_neural_critic` + `spiking_snc` (the SNc
encodes δ=r−V from the SYNAPTIC reward) + `heuristic_strength=0` (retire the host Manhattan orienting). The reward
becomes coord-free **synaptically**: with `enable_spiking_sc_approach`, the host reward write to `reward_us` is ZEROED
(`g11_bg_runner.py:7271`) and `reward_us` is driven by `sc_rostral → reward_us` (the SC bump's proximity).

**SC-on arm** (×6 seeds 42/43/44/100/101/102):
```bash
SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 \
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run \
    --with-conv --spiking-sc --seed <SEED> --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_on_seed<SEED>.json
```

**Host-control arm** (the current merged default: host heuristic orienting + host `sign(Manhattan)` reward, ×6 seeds):
```bash
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run \
    --with-conv --seed <SEED> --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_host_seed<SEED>.json
```

**Decisive anti-cheat — scrambled-retinotopy lesion** (≥3 seeds; MUST REGRESS vs SC-on):
```bash
SC_RET_SC=160 SC_REC=12 SC_RET_DRIVE=3500 SC_ROS_US=40 \
SIM_BACKEND=cupy python -m research.runners._nav_gate_merged_run \
    --with-conv --spiking-sc --scramble-sc --seed <SEED> --grid-size 32 --n-steps 1800 \
    --out research/findings/raw/nav_gate_2a/navsc_scramble_seed<SEED>.json
```

**Metric:** `nav_sum` = the per-run JSON's `gate_score` (= Σ `final_quarter_mean_distance` over the 4 moving-goal
phases; LOWER = better) — the established nav metric. Aggregate with the existing
`research/runners/nav_gate2a_aggregate.py` pattern (it reads `gate_score` from each arm's per-seed JSON).

### GO bar (the controller's call)

- **SC-on mean `nav_sum` ≤ 1.25 × host-control mean** (the project "within 25% of the host it replaces" deploy bar,
  consistent with the 2026-06-19 spiking-decision default-on GO at 1.16× host).
- **Scrambled-retinotopy lesion REGRESSES** (the decisive one — a scrambled `sc_retina→sc_map` map → orienting →
  chance; if nav survives a scrambled map the signal leaks from somewhere non-retinotopic → reject).
- **Conversational 8/8 + 7/7 green** (by construction, per the disjointness above).
- **Image-only-afferent provenance clean:** the SC reads only the egocentric render `render_egocentric_goal` (a
  legitimate ENVIRONMENT render of the agent's sensory input, channel-1 of the bar); `(x,y)`/`(gx,gy)`/Manhattan
  never enter the SC drive or the reward.

Per BRAIN-BASED-ONLY, a clean honest NEGATIVE here (the SC, alive at a non-starved op-point, still underperforms the
host heuristic on the merged bridge by > 25% even after a `SC_CORTEX_W` re-tune) **IS the deliverable** — it maps the
co-residence operating-point limit. But given the standalone 6-seed GO + the Step-0 merged bump GO, the expected
outcome is GO → flip the merged default to `nav_critic_spiking_sc=True`, retiring the host orienting heuristic AND the
host `sign(Manhattan)` reward on the one brain (N1 + N5 closed on the merged bridge).

### Two notes for the controller (verified, so the run isn't misread)

1. **The `--spiking-reward-us WITHOUT --perceived-approach-reward` WARN is STALE for this config — do NOT add
   `--perceived-approach-reward`.** The warn (`g11_bg_runner.py:3905`) only checks `spiking_reward_us and not
   perceived_approach_reward`; it does NOT account for the `enable_spiking_sc_approach` branch, which is exactly what
   ZEROES the host reward write and routes `reward_us` synaptically from `sc_rostral` (`:7260-7271`). In the
   `--spiking-sc` config the reward IS coord-free (SC-synaptic). Adding `--perceived-approach-reward` would route the
   reward through the HOST image-graded `sc_salience_offset_from_image` — MORE host, the opposite of the goal.
2. **The `sc_map→sc_rostral` strength is hard-coded 20.0 inside `install_spiking_sc_wiring` (`g11_bg_runner.py:294`),
   not env-tuned.** Step-0 confirmed the rostral pool + `reward_us` fire robustly at the merged op-point with this
   value, so no change is needed; flagged only for completeness if a re-tune is ever required.

### Pre-flight (already verified inline; the controller does NOT need to repeat)

- Both arms run end-to-end on GPU at grid-8/120-step smoke: the merged+SC bridge builds (58 regions, the SC chain +
  neural critic + parser + dlPFC on ONE bridge), navigates, and the scramble arm shows the expected directional
  regression (intact sum_finalQ 2.60 vs scrambled 4.50 at 40 steps; the 6-seed run makes it conclusive). The runner's
  `--help` parses the new flags; default-off byte-identity to the STEP-2a gate is by construction (the `kw.update` is
  gated on `args.spiking_sc`).

## Commits (all on `main`, PATHSPEC)

- `6c7652e1` — Step-0 probe `_navsc_merged_opcheck.py` + the bump-fires GO result.
- `09099a5b` — the `--spiking-sc` / `--scramble-sc` arm on `_nav_gate_merged_run.py` (default-off byte-identical).
- (this finding) — `2026-06-19-nav-spiking-sc-deploy-prep.md`.

## Honest scope

Step-0 + the disjointness check are the cheap, fully-diagnostic gates that the scoping doc identified as the most
likely spurious-NEGATIVE cause; both are GREEN. The decisive 6-seed merged A/B (the `nav_sum` head-to-head + the
scrambled-retinotopy lesion) is the controller's to run/poll — it is the test that produces the GO/NEGATIVE verdict
and the default-flip decision. This is a deploy-and-validate of an already-validated organ (the standalone spiking SC
is 6-seed GO); the science is largely done. NO `sim/` edit anywhere in this prep (reuse-by-import; the op-point is the
already-shipped env-var values).
