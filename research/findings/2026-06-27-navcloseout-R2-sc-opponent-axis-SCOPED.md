# Nav close-out R2 — SC opponent-axis margin-SNR remedy — SCOPED + BUILT + CPU-smoke GO (2026-06-27)

**Type:** scope + runner-level BUILD + CPU smoke. NO `sim/` edit. The long GPU nav eval is NOT run here (a
subagent cannot resume on background completion); the EXACT GPU command + de-risk criteria are flagged below
**FOR THE CONTROLLER TO RUN**.

**What R2 is.** R1-a (`2026-06-27-navcloseout-R1a-spiking-default-GO.md`) flipped the deployed nav to the
spiking decision + spiking SC-orienting (popvector + FIX1 stochastic tie-break + log-polar foveal render) at
an honest **1.91× the host-oracle cost**. R2 is the ranked margin-SNR remedy (the nav-loop research gate's
RANK 2, `2026-06-27-nav-loop-closure-research-gate.md`): **organize the SC orienting decode as an OPPONENT-AXIS
push-pull** — decode the orienting direction from the DIFFERENCE of opposing SC sub-populations (E-minus-W,
N-minus-S) rather than four independent half-plane pops — so the position-bearing direction gets a HIGHER
margin-SNR → fewer weak-margin ties for the tie-break to coin-flip → a tighter spiking-SC orienting → shrinks
the 1.91×.

**Runner-level vs `sim/`: RUNNER-LEVEL, NO `sim/` edit.** R2 is a weight-formula augmentation in
`install_spiking_sc_wiring` (`g11_bg_runner.py`) + opt-in plumbing through `run_moving_goal_episode` + the
de-risk rig + both CLIs. The opponent inhibition uses the project's EXISTING `cortex_FS` interneurons (the
cortex WTA pool), wired through `set_pathway_weights` — no protected-`sim/` change, no new neuron model, no
negative weights.

---

## The mechanism (the precise reframe that separates R2 from the FIX-3 NEGATIVE)

There are TWO distinct "opponent-axis" loci, and the prior NEGATIVE was at the WRONG one:

- **(L1) the DECISION read-out** — `sc_opponent_axis` (`_argmax_action` on `sel_X`/`commit_X` counts), the
  prior **FIX 3 = NEGATIVE** (`2026-06-20-shortcut6-FIX3-opponent-axis.md`). It re-biased to N because it read
  the counts AFTER the Wang-2002 sel accumulator AMPLIFIES a structural N-S common-mode ~9× (thal N−S +939 →
  sel N−S +10587). A hard signed-difference on those counts reads the amplified contamination as decisive.
- **(L2) the SC ORIENTING DECODE itself** — `sc_map → cortex_{N,E,S,W}` (the R2 BUILD). This is the
  **furthest-UPSTREAM** point — it shapes the SC's OWN contribution to `cortex_X` BEFORE the cascade amplifies
  anything. The FIX-3 finding's own decisive prescription (its §"the differential remedy", option b) was
  exactly this: *"the SC opponent push-pull realized at the integrator, not the read-out where it failed."* R2
  realizes it even one stage upstream of the integrator — at the SC decode.

**The realization (point-neuron, NO `sim/` edit, NO negative weights).** Inhibition in this substrate is set by
the PRESYNAPTIC neuron's type (`sim/bridge.py` `_cached_inhibitory_mask` routes a spike into `g_i`/`E_inh` iff
the source is inhibitory), so a negative weight from an excitatory `sc_map` neuron is NOT clean inhibition. The
biology-faithful route is the EXISTING `cortex_FS_{cardinal}` interneuron (all-inhibitory, `exc_fraction=0.0`,
the cortex-WTA pool). R2 wires the **opposing-half-plane sc_map sites → the cardinal's `cortex_FS`** at the
same cosine-projection magnitude: a W-bump fires `cortex_FS_E` → INHIBITS `cortex_E`. So `cortex_E`'s NET SC
drive = (E-site excitation via `sc_map_to_cortex_E`) − (W-site inhibition via `sc_map_opponent_to_cortex_FS_E`)
= the **E-minus-W opponent contrast** (and symmetrically each axis). A symmetric common-mode (equal E & W mass)
CANCELS; an eccentric bump yields a sharp single winner.

**Biology.** The superior-colliculus motor map's opponent/push-pull organization (catalog **H.25**;
opposing-direction populations with balanced E/I) + center-surround / ON-OFF push-pull opponency (catalog
**E.05**/**E.06** — "push-pull luminance"; the project's own E-cluster ON/OFF opponency) + the SC population
vector decode (**E.03**). The opponent contrast is the canonical sensory-motor sharpening motif, here applied
to the orienting decode.

**Complementary, not competing, with FIX B.** `enable_sel_opponent_pair` (FIX B) opponent-pairs the sel
ACCUMULATORS (the integrator stage). R2 shapes the SC's own signal at the DECODE (one stage further up). They
are complementary: R2 raises the orienting margin-SNR the SC injects; FIX B fights the accumulator-amplified
downstream common-mode. The GPU eval below tests R2 alone first (the mandate's target); FIX B / R2+FIXB stays a
ranked follow-on if R2 alone under-shrinks.

---

## The build (files; ALL runner-level, default-off byte-identical)

`research/runners/g11_bg_runner.py`:
- `install_spiking_sc_wiring(...)` — new params `sc_opponent_decode=False`, `sc_opponent_fs_weight=14.0`. When
  on (and the `cortex_FS_{N,E,S,W}` pools exist), installs `sc_map_opponent_to_cortex_FS_{a}` edges: the
  OPPOSING-cardinal half-plane sites (same cosine magnitude as the excitation) drive the cardinal's `cortex_FS`
  → opponent inhibition. Guarded: if `cortex_FS` pools are absent it SKIPS with a warning (so it can't
  silently do nothing-wrong). Default-off installs ZERO edges (byte-identical).
- `run_moving_goal_episode(...)` — new params `sc_opponent_decode=False`, `sc_opponent_fs_weight=14.0`; resolved
  via kwarg OR `SC_OPPONENT_DECODE=1`; passed into the `install_spiking_sc_wiring` call; surfaced in the result
  JSON (`sc_opponent_decode`, `sc_opponent_fs_weight`).
- CLI: `--sc-opponent-decode` + `--sc-opponent-fs-weight` (argparse-registered, default-off).

`research/runners/_nav_sc_popvector_readout_derisk.py` (the rig the FIX1/FIX3/log-polar findings used):
- `run_arm(..., fix_r2=False, opponent_fs_weight=14.0)` — when on, sets `sc_opponent_decode=True` +
  `sc_opponent_fs_weight` and AUTO-enables `enable_cortex_lateral_inhibition` (the opponent inhibition needs the
  `cortex_FS` pools). CLI `--fix-r2` / `--r2` + `--opponent-fs-weight`. Per-arm summary surfaces
  `r2_sc_opponent_decode` / `r2_opponent_fs_weight`. `SC_OPPONENT_DECODE` added to the per-run env reset.

`research/runners/_navcloseout_R2_opponent_decode_smoke.py` — the CPU smoke (below).

**Honest scope point (the controller must hold the anti-cheat for this):** the deployed R1-a / merged-gate path
does NOT enable cortex lateral inhibition (verified: no `cortex_wta` in `_nav_gate_merged_run.py`), so R2 ADDS
the `cortex_FS` pools. They are standard biology-grounded cortical PV-basket interneurons (already a validated
mechanism, `--cortex-wta`), but the lift must be attributed to the OPPONENT geometry, not merely to adding WTA.
⇒ the GPU eval includes a **cortex-WTA-only arm** (popvector + cortex_wta, NO opponent) as the matched control,
so R2's gain over THAT arm is the opponent contribution.

---

## CPU smoke — WELL-FORMED + HIGHER-MARGIN, PASS

`python -m research.runners._navcloseout_R2_opponent_decode_smoke` (pure numpy, no GPU, no bridge). It replays
the EXACT stage-3 decode geometry (`install_spiking_sc_wiring` cosine weights) on a synthetic SC bump and
compares the independent-popvector margin vs the opponent-difference margin:

```
case      truth | pv-win pv-nmarg pv-raw  | opp-win opp-nmarg opp-raw | nmarg-x better?
far-E     E     | E        0.4920   14.79 | E          0.8446   16.72 |    1.72 YES
far-W     W     | W        0.4920   14.79 | W          0.8446   16.72 |    1.72 YES
far-N     N     | N        0.4920   14.79 | N          0.8446   16.72 |    1.72 YES
far-S     S     | S        0.4920   14.79 | S          0.8446   16.72 |    1.72 YES
NE-diag   E     | E        0.0000    0.00 | E          0.0000    0.00 |     inf no   (45deg tie, both decodes)
SW-diag   W     | W        0.0000    0.00 | W          0.0000    0.00 |     inf no   (45deg tie, both decodes)
near-E    E     | E        0.2795    9.56 | E          0.6659   13.05 |    2.38 YES  (weak-margin: biggest gain)
near-N    N     | N        0.2795    9.56 | N          0.6659   13.05 |    2.38 YES

well-formed (decode == truth): pop-vec 8/8   opponent 8/8
opponent normalized-margin HIGHER than pop-vec: 6/8 cases  (mean nmarg ratio 1.94x)
common-mode (symmetric E+W bump): pop-vec E=22.43 W=22.43 (both large, |E-W|=0.000 = tiny margin)
                                  opponent E=-0.000 W=0.000 (net ~0 = common mode REJECTED on the E-W axis)
VERDICT: PASS  (well-formed=True, higher-margin=True)
```

**What it proves:** (1) the opponent decode is WELL-FORMED — returns the correct cardinal on every eccentric
case (8/8). (2) it is HIGHER margin-SNR — normalized margin +0.49→+0.84 on far goals (1.72×), +0.28→+0.67 on
the WEAK near goals (2.38×, the biggest gain exactly in the weak-margin regime where R1-a random-walks), mean
1.94× across the defined cases. (3) it REJECTS the common-mode — a symmetric E+W bump gives the popvector a
large drive on BOTH E and W (margin 0) but the opponent net cancels to ~0 (correctly "no E-W preference"). The
two non-improving cases are perfect 45° diagonals where both decodes correctly tie at 0. This is the precise
margin-SNR improvement R2 targets. (The smoke validates the decode GEOMETRY/SNR, NOT the absolute
`sc_opponent_fs_weight` scale or the nav score — those are the GPU eval below.)

---

## FOR THE CONTROLLER TO RUN — the GPU nav eval (does R2 shrink the 1.91× vs the R1-a baseline?)

GPU-only (`SIM_BACKEND=cupy`), grid-32 / 1800 / warmup-600 (grid-32 IS the verdict, NEVER grid-8). Use the
de-risk rig `_nav_sc_popvector_readout_derisk.py` — it reproduces the R1-a config exactly and adds `--fix-r2`.
Cheap-first seeds **42, 43**, then **6 seeds (42/43/44/100/101/102)** if seed-42/43 shows shrink (R2 is a
VARIABLE effect → the 6-seed rule applies; this is NOT a byte-identity gate).

### The arms (per seed) — the R1-a baseline, the WTA-only control, and R2

```bash
# (1) R1-a BASELINE — popvector + FIX1 + log-polar (the 1.91x deployed config), NO opponent, NO cortex-wta
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 --log-polar \
    --out research/findings/raw/nav_gate_2a/r2/scpv_BASELINE_s42.json

# (2) WTA-ONLY control — + cortex-wta, NO opponent (isolates "adding cortex_FS" from "the opponent geometry")
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 --log-polar --cortex-wta \
    --out research/findings/raw/nav_gate_2a/r2/scpv_WTAONLY_s42.json

# (3) R2 — the SC-decode OPPONENT-AXIS push-pull (auto-enables cortex-wta). THE BUILD.
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 --log-polar --fix-r2 \
    --out research/findings/raw/nav_gate_2a/r2/scpv_R2_s42.json

# (4) HOST ceiling (anchors the gap; once per seed)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms host --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --out research/findings/raw/nav_gate_2a/r2/scpv_HOST_s42.json

# (5) SCRAM(R2) — retinotopy lesion + R2 (the anti-cheat: the opponent decode MUST still be load-bearing)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector_scr --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --fix1 --log-polar --fix-r2 \
    --out research/findings/raw/nav_gate_2a/r2/scpv_SCRAM_R2_s42.json
```

Repeat (1)-(3)+(5) for seed 43 (and 44/100/101/102 if shrink confirmed); HOST once per seed. The metric is
`post_change_finalQ_sum` (phases 1-3, lower=better) + `per_phase_dominant_cardinal` (must TRACK, not re-bias).
If `sc_opponent_fs_weight=14` over- or under-inhibits (R2 dom collapses to a single cardinal, OR R2 ≈ WTA-only
with no shrink), do a SHORT calibration scan `--opponent-fs-weight {7, 10, 14, 20}` at seed 42 ONLY (the smoke
fixes the geometry; the scale is the one free GPU knob) — do NOT grind beyond that.

### De-risk criteria (the shrink test)

- **GO (R2 shrinks the 1.91×):** R2 `post_change_finalQ_sum` is materially LOWER than the R1-a BASELINE (and
  lower than the WTA-ONLY control → the gain is the OPPONENT geometry, not merely adding WTA), the dom-cardinal
  TRACKS the goal every phase (NOT re-biased to a single cardinal — the FIX-3 failure signature), and the
  host/R2 post-change ratio moves toward 1 from the 1.91×. Confirm on seed 42+43, then 6 seeds.
- **HONEST NEGATIVE (R2 does not shrink / re-biases):** report it crisply (does R2 ≈ WTA-only = the gain was
  just the WTA? does the opponent decode re-bias like FIX-3 did at the read-out, indicating the common-mode is
  downstream of the SC decode and only FIX B / R2+FIXB can touch it?). Do NOT loosen anything. A NEGATIVE here
  sharpens to the ranked follow-on (R2+FIX B, the integrator-stage opponent pairing), NOT a stop.

### Anti-cheat (the controller must assert these)

| anti-cheat | requirement |
|---|---|
| HOST positive control (the ceiling) | host re-orients, anchors the 1.91× gap |
| **WTA-ONLY matched control** (THE R2-specific anti-cheat) | R2's shrink must EXCEED the cortex-WTA-only arm → attribute the gain to the OPPONENT geometry, not to adding the `cortex_FS` pools |
| matched drive | all SC arms at `--sc-cortex-w 18` (the gain is the geometry, not a covert drive lift — the amplification screen already showed stronger drive only re-biases) |
| per-phase dom-cardinal TRACKS (THE discriminator) | dom must SHIFT to the goal's bearing every phase, NOT re-bias to one cardinal (the FIX-3 stuck-N signature) |
| SCRAM(R2) MUST collapse | the retinotopy-scramble lesion regresses → the opponent decode is genuinely carried by the RETINOTOPIC map, not a non-retinotopic leak |
| host-oracle UNCHANGED | `--readout-source motor --no-spiking-sc` reproduces the documented 2.86 benchmark unchanged (R2 is opt-in, default-off byte-identical) |
| no-confab MOAT untouched | the nav cascade `cp_*` state is array-disjoint from the composer's complex `cp_rf_w_*` synapses → preserved by construction; re-assert `tests/test_nav_conv_merged_agent.py` + `tests/test_nav_conv_step2b_coresident.py` (29 passed / 1 xfailed) |
| grid-32 (NEVER grid-8) | the verdict is grid-32/1800/warmup-600 |
| 6-seed for the effect | R2 is a variable effect → 6 seeds (42/43/44/100/101/102); seed-42/43 cheap-first gate first |

---

## Verdict (this session)

**R2 (SC opponent-axis margin-SNR remedy) is SCOPED, BUILT runner-level (NO `sim/` edit), and CPU-smoke GO.**
The mechanism is the biology-faithful SC opponent/push-pull decode (catalog H.25/E.05/E.06) realized at the SC
orienting DECODE via the existing `cortex_FS` interneurons — the UPSTREAM locus the FIX-3 read-out NEGATIVE
explicitly prescribed for the differential remedy (and one stage further up than FIX B's integrator pairing).
The CPU smoke proves the decode is well-formed (8/8 correct cardinals) and HIGHER margin-SNR (mean 1.94×, up to
2.38× in the weak-margin regime, common-mode rejected) — the precise lever to shrink the R1-a 1.91×. The GPU
nav eval (does it shrink the 1.91× vs the R1-a baseline AND beat the WTA-only control? — seeds 42/43 cheap-first
then 6) is flagged above FOR THE CONTROLLER. Default-off byte-identical; the host-oracle path unchanged; the
no-confab moat array-disjoint and untouched.

_NO `sim/` edit. grid-32 IS the verdict (never grid-8). The GPU eval is NOT run here (a subagent cannot resume
on background completion) — the exact command + de-risk criteria are flagged FOR THE CONTROLLER. Committed on
`main`, strict narrow `git add`, pushed to BOTH remotes (origin + gitea)._
