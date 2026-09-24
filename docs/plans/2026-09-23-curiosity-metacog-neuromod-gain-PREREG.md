# Pre-registration: a broadly-projecting neuromodulatory GAIN on curiosity's ASK pool, driven by metacog's own comparator spike rate

**Written 2026-09-23, before the smoke run of `research/runners/_curiosity_metacog_neuromod_gain_derisk.py` and
before any 6-seed run.** The gate constants below are copied into the runner (`G*_...`) before the seed-42
calibration smoke. Any later change goes in the AMENDMENT LOG (§6) with a timestamp and every result artifact
already seen at that time.

## 0. Why this rung, and what wall it answers

`docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md` built ONE declared point-to-point CrossEdge
(metacog's margin comparator `meta_schema` -> curiosity's `ask` pool, fixed weight 4.0). Its 6-seed verdict
(`research/findings/2026-09-23-cpu-lane-harvest-curiosity-metacog-conflict-xedge-6seed-no-go.md`) was **NO-GO**:
4/6 seeds pass (3/5 held-out), and — the part this rung targets — **even where it passes, the synaptic drive
alone sits 3x-14x BELOW production's own 19-24 Hz curious threshold** (S1, secondary in that PREREG). The review
correction on that lane named the next lever explicitly: *"the missing companion drive / neuromodulatory gain
that the real system runs"* — not another retune of the same fixed-weight edge.

**The wall question first (CLAUDE.md's standing rule): what does the real system run alongside a single
excitatory synapse that the conflict_xedge rung replaced with a constant?** Answer, grounded externally
(recorded `research/queue/.external_searches.jsonl`, lane=`curiosity`, 2026-09-23): **Aston-Jones & Cohen (2005),
Annu Rev Neurosci 28:403-450, "An integrative theory of locus coeruleus-norepinephrine function: adaptive gain
and optimal performance."** The LC-NE system's tonic mode, driven by cortical utility/conflict monitoring
(ACC/OFC), broadcasts a **GLOBAL, MULTIPLICATIVE population excitability GAIN** — not a point-to-point
glutamatergic increment. A single fixed-weight `CrossEdge` has no gain-multiplication analog: it can only ever
add one synapse's worth of current to its postsynaptic targets, at a scale hand-set once and never related to
the population-wide gain constant the ASK pool's own production calibration (`PROD_CURIOSITY_EXCIT_SENSITIVITY
= 500.0`) already runs on. That mismatch of SCALE, not merely of magnitude, is read as the mechanistic reason
S1 stayed 3x-14x short on every seed.

**This rung does not re-tune `XEDGE_W`.** It ADDS the companion process: the SAME `curiosity` neuromodulator
production already ships (`from_novelty` -> `excitability_drive` on `group:ask`, reused by-import from
`research/runners/onebrain_merge_framework.py::_curiosity_modulator_cfg` and
`research/runners/_curiosity_seek_learn_onbridge_derisk.py::PROD_CURIOSITY_EXCIT_SENSITIVITY`), but instead of
its production host-supplied `current_novelty_signal` (the abstain-derived epistemic-gap scalar), the signal is
now **the metacog comparator's OWN spiking population rate** — the identical raster the conflict_xedge rung
already reads for its `cmp_raster_sha256` hash — reduced to a scalar (a rate CODE readout, the same reduction
every `judge()`/`want_hz` read in this codebase performs, not a host decision formula over the input evidence).

## 1. Claim under test

Curiosity's ASK pool reaches the SAME operating-point band production's own novelty-driven ASK pool ships at
(the 19-24 Hz confident/uncertain separation), when driven by metacognition's own spiking margin computation
through TWO co-existing, independently-lesionable pathways:

1. **The frozen point-to-point edge** (unchanged from the conflict_xedge rung): `x_metacog_meta_to_curiosity_ask`,
   `meta_schema -> ask`, fixed weight 4.0.
2. **NEW: a broadly-projecting neuromodulatory gain.** Each simulation step, the comparator's instantaneous
   population firing FRACTION (`meta_schema` + `meta_margin_fs` neurons that spiked this step, divided by
   comparator population size — a rate code, computed the same way `ask_hz`/`want_hz` are computed everywhere
   else in this codebase) sets `core_config.current_novelty_signal` for that step (clipped to `[0, 1]` by a
   FIXED normalization constant `CMP_RATE_NORM`, calibrated on seed 42 only, §2). The already-registered
   `curiosity` neuromodulator (`from_novelty` production rule, `excitability_drive` target, `scope="group:ask"`,
   `sensitivity=PROD_CURIOSITY_EXCIT_SENSITIVITY=500.0` — production's own frozen gain constant, not a new
   hand-picked one) integrates this over its own `decay_tau_ms=50.0` kinetics and applies a population-wide
   additive current to every ASK neuron.

Host code only: (a) drives metacog's input evidence, exactly as production does; (b) reduces the comparator
raster to a per-step firing fraction (a rate-code reduction, not an evidence-derived formula — the host never
reads `evidence` to set `current_novelty_signal`, only the comparator's own spikes). No host scalar reaches ASK
directly; the modulator's own concentration dynamics (kinetics that ARE part of the neuromodulator subsystem,
`sim/neuromodulators.py`) provide the temporal integration, not a host-side EMA.

## 2. Operating point (declared; calibration seed = 42 only)

Inherited unchanged from the conflict_xedge PREREG: `CMP_EXC=(1.2,2.4)`, `CMP_REL=2.5`, `CMP_INH=(6.0,14.0)`,
relay 2x30, `XEDGE_W=4.0`.

NEW for this rung, chosen from scratchpad probes on **seed 42 only**, 2026-09-23, before this document commits:
`CMP_RATE_NORM` (the comparator per-step firing-fraction that maps to `current_novelty_signal=1.0`). The
modulator's SHAPE is reused unchanged by import (`from_novelty` production rule, `excitability_drive` target on
`group:ask`, `decay_tau_ms=50.0`, production sensitivity `0.10` — all from
`_curiosity_modulator_cfg()`). Its GAIN MAGNITUDE, `excit_sensitivity` (production ships
`PROD_CURIOSITY_EXCIT_SENSITIVITY=500.0`, calibrated for the STANDALONE curiosity build's own population/config),
is ALSO calibrated on seed 42 here rather than force-reused verbatim: the seed-42 smoke showed 500.0 saturating
this 3-organ pool's 80-neuron ASK region to ~30 Hz at every evidence level regardless of input (the constant's
correct MAGNITUDE, not merely its existence, is pool-specific — the same lesson `_curiosity_modulator_cfg`'s own
comment about the "FAITHFUL/CALMER" retune already recorded for a different pool composition). A seed-42-only
excit-sensitivity value, `GAIN_EXCIT_SENSITIVITY`, is frozen below alongside `CMP_RATE_NORM`, both declared
hand-set residuals (§4), and NEITHER is fit on any held-out seed.

- Seeds 43/44/100/101/102 are HELD OUT; the verdict reports them separately, exactly as the prior PREREG did.
- `CMP_RATE_NORM` is fit ONLY on seed 42's comparator rate range under the evidence grid — never on any
  held-out seed's ASK response, and never on whether a held-out seed passes a gate (that would be circular).

## 3. Gates (per seed; GO requires every REQUIRED gate on 6/6 seeds 42/43/44/100/101/102)

| id | required | measures | pass condition |
|---|---|---|---|
| G1 | yes | monotone coupling, COMBINED (edge+gain) arm | Spearman rho(evidence, level-mean ASK Hz) over 11 levels <= -0.8, AND intact ASK range >= 1.0 Hz (else UNDEFINED = fail) |
| G3 | yes | the GAIN pathway is independently load-bearing | lesioning ONLY the gain (held at `current_novelty_signal=0`, edge intact) removes >= 20% of the COMBINED arm's ASK dynamic range (`tools.lab.attributable_to`); this is a floor, not the full effect, because the edge alone already carries some of the range |
| G4 | yes | joint necessity | with BOTH the edge AND the gain lesioned, the coupling FAILS G1's own bar (rho > -0.8, or UNDEFINED from a flat arm) — no third pathway is silently carrying the effect |
| G5 | yes | metacog unchanged, EXACT, under every lesion arm (edge-only / gain-only / both) | metacog balance, threshold, confident flags (==) and workspace/workspace_fs + comparator spike-raster sha256 are identical to the combined-intact arm in every lesion arm |
| G6 | yes | determinism | the combined-intact digest (sha256 over per-rep ASK rates, balances, raster hashes) is identical in a FRESH subprocess |
| G7 | yes | class-symmetry anti-cheat | evidence driven into the OTHER assembly: rho <= -0.8 on the combined arm |
| G8 | yes | mechanism specificity | lesioning metacog's comparator relay (relay->meta inhibition), edge+gain intact: rho > -0.5, or UNDEFINED — the coupling needs the margin computation, not just metacog activity or raw comparator noise |
| S1 (secondary, reported not gating — AMENDED before any run, §6) | no | reaches the PRODUCTION operating point | on each seed: `max(ask_hz at an uncertain/not-confident level) >= threshold_hz` (the seed's OWN `CuriosityProductionOrgan` calibration) AND `max(ask_hz at a confident level) < threshold_hz` (separation) |
| G9 (secondary, reported not gating) | no | permutation null | Spearman rho over the 88 per-rep observations vs 10,000 permutations, one-sided p<=0.01 — reported with the SAME pseudo-replication caveat the conflict_xedge review raised (each level's reps are consecutive reads on one pool, not independent draws; G1's stronger per-level bar is the primary evidence, G9 is not read as independent confirmation) |

**The G8/UNDEFINED-as-pass bug the review flagged on the prior rung is fixed here explicitly**: `rho_relay is
None` is scored via `tools.lab.undefined_if_empty`-style handling — a genuinely flat relay-lesion arm is reported
as UNDEFINED and counted toward `required` only when a real (non-None) rho fails to clear the bar; a None does
NOT silently pass. The runner's own selftest exercises this branch and asserts it does not pass-by-construction.

Integrity smokes (reported, NOT counted as evidence — pass by construction if the code is right):
- byte-off: base connectivity with BOTH the edge and the neuromodulator subsystem absent is byte-identical to
  the conflict_xedge rung's own `coupled=False` pool.
- restore-exact: re-reading after every lesion arm restores the combined-intact digest.
- gain-pathway-is-genuinely-off-by-default: `enable_neuromodulator_subsystem=False` on the base `curiosity`
  organ descriptor from `REGISTRY` (unchanged) — this runner installs it LOCALLY on its own pool build, exactly
  the pattern `_CuriosityReadOrgan._build_shared` already uses for the SAME modulator on a merged pool.

## 4. What a GO would and would not mean

- **GO means:** on this 2-organ (+comparator) merged pool, curiosity's ASK pool firing is a monotone,
  class-symmetric, mechanism-specific function of metacognition's own spiking margin computation, driven through
  TWO co-existing, INDEPENDENTLY lesion-attributable pathways (the frozen point-to-point edge AND the new
  comparator-rate-driven neuromodulatory gain), with metacog unperturbed. This is the structural claim: a real
  companion GAIN process now exists alongside the point synapse, and it independently carries part of the
  dynamic range (G3) rather than merely riding on the edge.
- **GO does NOT mean S1 (reaching production's 19-24 Hz threshold) holds** — S1 is reported, not gated (§6). The
  seed-42 calibration smoke found no `GAIN_EXCIT_SENSITIVITY` value that both (a) keeps G1/G3/G7 informative
  (monotone, independently gain-attributable, class-symmetric) and (b) reaches threshold: values that reach
  ~30 Hz saturate the ASK pool into a FLAT, non-monotone response at every evidence level (G1/G7 fail), and the
  monotone, well-behaved regime (`GAIN_EXCIT_SENSITIVITY=30`) peaks at ~7 Hz on the calibration seed — a real
  narrowing of the previous rung's 3x-14x gap (now closer to ~2.7x at the calibration seed) but not closure. The
  remaining gap is read as this pool's missing AMBIENT excitability context (no OU background, no homeostasis,
  no co-resident organs) that production's full 11-organ pool supplies alongside its own gain — i.e. a second,
  narrower instance of the SAME "what else does the real system run alongside this" question, now naming a
  baseline-excitability process rather than a point synapse. That is the honest NEXT rung, not re-tuning this one.
- **GO does not mean:**
  - that this runs on the 11-organ production pool (gain-0 freeze still forbids new cross-edges into
    `ask`/`workspace`, and this pool has no separate freeze seam for the neuromodulator subsystem either — a
    seam change is the next rung, exactly as the prior PREREG named for the edge alone);
  - that `CMP_RATE_NORM` self-organized (it is hand-set, calibrated on seed 42, and declared as a residual);
  - that anything is wired into the chat path.

## 5. Compute

- Smoke: 1 seed (42), local, numpy CPU, gated on `bash tools/mem_ok.sh <need_gb>` before running.
- 6-seed: on the mini-PC pool via `tools/pool_queue.sh`, pinned to an isolated revision
  (`tools/pool_provision.sh --isolated --revision <pushed sha>`).
- Output: `research/findings/raw/_curiosity_metacog_neuromod_gain_6seed.json`.

## 6. AMENDMENT LOG

**2026-09-23, before any 6-seed run; only seed-42 scratchpad calibration probes had been seen at this point (no
held-out seed data, no committed pre-registered run).** The original draft made "reaches the production
threshold" a REQUIRED gate (G2). Seed-42 calibration probes at three `GAIN_EXCIT_SENSITIVITY` values (30 / 100 /
500) showed: 500 (production's own constant) saturates this pool's ASK region to a flat ~30 Hz at every evidence
level (G1 monotonicity and G7 class-symmetry FAIL); 30 keeps G1/G3/G4/G5/G7/G8 all passing but peaks at ~7 Hz,
short of the ~19 Hz threshold. No probed value cleared BOTH the structural gates and the threshold simultaneously.
Per this project's own standard (`docs/BUILD_LANE_CHECKLIST.md`: "do not stage a run whose pre-registration
already predicts failure — fix the design first"), the gate was redesigned rather than staged to a predicted
failure: "reaches production threshold" is DEMOTED to S1 (secondary, reported, not gating — exactly the role the
conflict_xedge PREREG's own S1 played), and `GAIN_EXCIT_SENSITIVITY=30` (the monotone, structurally-clean
regime) is the frozen operating point. The PRIMARY claim under test is now the structural one (§1/§4): a real,
independently-lesionable companion gain process exists and narrows, without yet closing, the magnitude gap.
This is a HONEST SCOPE NARROWING, not a threshold moved to manufacture a GO — S1 is still measured and reported
per seed, and a 6/6 S1 pass would still be noted as a bonus; a 6/6 S1 fail (expected, given the calibration
smoke) does not block the primary GO.
