# Pre-registration: curiosity x metacog v5 — a reference-placed G11 grid, and a commitment veto held at a set-point by inhibitory plasticity

Runner: `research/runners/_curiosity_commitment_veto_v5_derisk.py`, revision `9cef44164` (committed before this
document; the constants and thresholds below are its module constants, frozen). Substrate measurement:
`research/runners/_curiosity_ask_operating_point_measure.py`. Branch: `research/curiosity-ask-operating-point`.
Builds on: `docs/plans/2026-09-24-curiosity-lcne-phasic-gain-PREREG.md` (v4) and its 6-seed NO-GO 3/6,
`research/findings/2026-09-24-curiosity-metacog-lcne-phasic-gain-v4-6seed-NOGO-3of6-gain-holds-circuit-residual.md`.

**Evaluation set: seeds 42/43/44/100/101/102, ALL SIX held out.** Every constant below was chosen on DEV seeds
7-12 only. No evaluation seed has been run with this mechanism. v4's evaluation artifacts are public; §0 reads them
once, as a diagnosis of v4's circuit. No v5 constant was set from them.

## 0. Why this rung, and what the substrate measurement says

v4 failed 3/6. Its gain held on all six seeds. Seed 42's G11 was UNDEFINED: its reference response rises across only
two points of the coarse drive grid. Seeds 44 (G1) and 100 (G7) failed the same way with the gain lesioned. The v4
finding named an ASK operating-point problem as the likely residual and a homeostatic set-point on ASK as the
companion process. **This rung measured that hypothesis before choosing a mechanism**
(`bash tools/before_you_build.sh "curiosity ASK operating point homeostasis"` logged first).

Measurement: `_curiosity_ask_operating_point_measure.py` on dev seeds 7-12. Artifacts are
`research/findings/raw/_curiosity_ask_op_measure/dev_s{7..12}.json` and `dev_summary.json`. metacog's margin
comparator (`meta_schema`) is two halves, meta_0 and meta_1, and the point-edge sums both onto ASK. "Sum rho" is
Spearman(evidence, meta_0 + meta_1). "Favored rho" is the same statistic for the half the evidence drives. "Best
threshold rho" is the most negative rho that any threshold on the measured sum can produce; this is the ceiling of
the ASK-operating-point route, since any operating point of ASK is a monotone transfer of its input.

<!--derived from research/findings/raw/_curiosity_ask_op_measure/dev_summary.json and dev_s*.json-->
| seed | ASK ref. threshold drive / open-loop slope (Hz per unit) | arm | ASK rho | sum rho | favored rho | min(halves) rho | sum last/min | best threshold rho (levels silenced) |
|---|---|---|---|---|---|---|---|---|
| 7 | 0.80 / 17.1 | intact | -0.94 | -0.92 | +0.97 | -0.99 | 1.002 | -0.98 (4) |
| 7 | | swap | -0.96 | -0.91 | -0.88 | -0.94 | 1.017 | -0.98 (4) |
| 8 | 0.90 / 17.0 | intact | **-0.61** | -0.78 | +0.98 | -0.95 | 1.026 | -0.92 (6) |
| 8 | | swap | **-0.77** | -0.93 | +0.78 | -0.96 | 1.021 | -0.98 (4) |
| 9 | 1.05 / 14.5 | intact | **-0.67** | +0.63 | +1.00 | -0.60 | 1.082 | +0.40 (10) |
| 9 | | swap | -0.91 | -0.31 | +0.99 | -0.99 | 1.000 | -0.39 (8) |
| 10 | 0.95 / 18.0 | intact | -0.98 | -0.44 | +0.82 | -0.99 | 1.045 | -0.79 (8) |
| 10 | | swap | **-0.27** | +0.28 | +0.99 | -0.99 | 1.145 | +0.07 (9) |
| 11 | 0.80 / 17.1 | intact | -0.83 | -0.93 | +0.54 | -0.97 | 1.021 | -0.98 (4) |
| 11 | | swap | -1.00 | -0.96 | -0.17 | -1.00 | 1.019 | -0.98 (3) |
| 12 | 1.00 / 15.3 | intact | **+0.09** | +0.62 | +0.99 | -0.24 | 1.173 | +0.50 (10) |
| 12 | | swap | -0.97 | -0.97 | +0.62 | -0.98 | 1.007 | -0.98 (2) |

Reading it:

- **Every failing arm (bold) has a sum that rises again at the confident end** (last level over minimum 1.02-1.17).
  On each of them the favored half rises with evidence (+0.78 to +1.0) while the rival half falls to a floor. On the
  two arms where the favored half does not rise (7 swap, 11 swap), ASK passes.
- **The ASK operating-point route cannot repair that input.** The best threshold rho is +0.40 (9 intact), +0.07
  (10 swap) and +0.50 (12 intact). Where a threshold does reach -0.8 or below (8 intact, 10 intact), it silences 6-8
  of the 11 levels.
- **ASK's own operating point does vary by seed** (reference threshold drive 0.80-1.05). This is a separate and smaller
  residual: seeds 9 and 12 have a weak base drive (intact ASK 0.43 and 0.90 Hz at the most uncertain level).
- **The evaluation seeds show the same shape.** From v4's public artifacts, read through lc_add (v4's non-adapting
  readout of the same sum; its output was closed): seed 44 intact goes 76 -> 111 spikes over evidence 0.7 -> 1.0
  (the G1 tail), and seed 100 class swap goes 58 -> 102 (the G7 tail).

**So the residual is the comparator's summed read, not ASK's operating point.** Following the task's own branch
("if the measurement says the failure is NOT operating-point, target that instead"), v5 targets the comparator's
read.

What the real system runs alongside a two-channel comparator, which this circuit had no counterpart for:

- **An opponent commitment signal.** These are neurons that fire when one channel dominates the other. Decision
  confidence is carried by single-neuron firing rates, and confidence changes behaviour: Kepecs et al. 2008, Nature
  455:227-231, doi:10.1038/nature07200 (PubMed-verified). In the conflict-monitoring account, conflict is
  co-activation of competing channels, not their sum: Botvinick et al. 2001, Psychol Rev 108:624-652,
  doi:10.1037/0033-295x.108.3.624 (PubMed-verified).
- **Inhibitory plasticity that holds such a detector at a set-point.** Inhibitory synaptic plasticity balances
  excitation and inhibition and acts as a homeostatic mechanism on the postsynaptic rate: Vogels et al. 2011, Science
  334:1569-1573, doi:10.1126/science.1211095 (PubMed-verified). The set-point is needed because each seed's two halves
  sit at different rates with no evidence at all (dev: up to 19.1 vs 13.9 Hz). A fixed-threshold detector would fire
  on some seeds with no evidence.

## 1. Instrument change: G11 on a grid placed on each seed's own reference curve

v4 scored G11 at edge_drive 0 and 0.6-1.2 (step 0.1). A steep reference left two points on the rising limb.
The v5 rule is fixed here. It reads the lc-OFF arm only, never the lc-on or additive arm.

1. Placement scan at evidence 0: the reference arm (lc_ne -> ask_fb closed, additive control closed, veto open)
   at edge_drive 0.5, 0.6, ..., 1.6.
2. Onset = the first placement drive where the reference ASK >= 0.5 Hz. If there is none, G11 is UNDEFINED (fail).
   Peak = the FIRST LOCAL MAXIMUM at or after onset: the first placement drive whose successor reads lower, or the
   top of the scan if the reference never falls. (v1.1; v1 took the global maximum, see §8.)
3. Scored grid = 11 evenly spaced drives from (onset - 0.1) to (peak + 0.05), plus drive 0.
4. On/off/add are read at every scored drive. v4's G11 logic follows unchanged: a point is defined when off >= 0.25
   Hz; the rising limb runs up to the reference's peak ON THE FINE GRID; G11a offset <= 0.05 Hz; G11b gain at the top
   of the limb >= 1.15; G11c Spearman(drive, on/off) >= 0 with ratios quantized to 0.05. The limb now needs >= 4
   points (v4: 3), or G11 is UNDEFINED (fail).
5. Instrument validity, same seed, on the same fine grid: the additive control must FAIL G11c while its own max
   ratio is >= 1.15, or G11 is UNDEFINED (fail). The lc_ne raster must hash identical across every placement and
   scored read.

Selftest (`--selftest`, no simulation, in the committed revision): a seed-42-like steep reference (coarse limb = 2
points) is placed to onset 0.9 / peak 1.0 and a 0.8-1.05 grid. On that grid a pure response gain passes, a growing
gain passes, an additive shift FAILS G11c, and an additive control that passes G11c makes G11 UNDEFINED.

## 2. Mechanism under test (every step neurons + synapses)

Pool: `[metacog (production, param-het), curiosity, metacog_margin (comparator), lcne_gain_organ (v4, unchanged),
commitment_veto_organ]`, plus the gated point-edge `meta_schema -> ask` (weight 4.0, as v4). Host code drives only
metacog's input evidence, as production does, and reads spike rasters.

| synapse | type | weight | role |
|---|---|---|---|
| `meta_j -> cv_inh{j}` | dense E, per-post-neuron gain U(0.5, 1.5) | W_MI 12.0 | the rival half's relay (FS, 20 per channel) |
| `meta_k -> cv_veto{k}` | dense E, per-post-neuron gain U(0.5, 1.5) | W_MV 10.0 | drives the channel's commitment units (RS, inhibitory output, 20 per channel) |
| `cv_inh{1-k} -> cv_veto{k}` | GABA-A, **plastic** (Vogels inhibitory STDP), gate `cv_istdp` | init W_PV0 4.0, bound [0, 80] | veto_k reads "channel k over channel 1-k"; the set-point pathway |
| `cv_veto{k} -> ask` | GABA-A, transmission gate `cv_veto_out` | W_VA 1.5 | a committed channel suppresses ASK |

The per-post-neuron gain spread is the comparator's own reason, carried over. The dev probes measured it: with
uniform weights every v5 population was quasi-binary (§3).

**Set-point epoch (once, before any scored read).** Open `cv_istdp`. Drive metacog with NO evidence differential (its
own lesion drive, both assemblies at base) for 64 production reads (8 reps x 135 steps each). The Vogels rule
(target 1.0 Hz per veto neuron, eta 1.0, trace tau 20 ms) moves only the declared plastic rows. Then close `cv_istdp`
for the rest of the run. The rule sees only the veto and relay neurons' own spikes. It never sees ASK, the evidence
level or any gate. `enable_inhibitory_stdp` is on at build. The only synapses the engine's rule could touch are the
declared rows: this is checked at the open gate, and on dev seed 7 the unmodified v4 pool had zero other eligible
synapses. The calibration is checked to have changed nothing else.

**Built but OFF.** The ASK operating-point set-point (`cv_norm`: a workspace-driven FS pool -> ASK, inhibitory STDP to
a common ASK target) has its output gate closed (`ASK_SP=0`). It is a dev method that failed (§3). `--set` is refused
on evaluation seeds, so it cannot run there.

**Residuals (declared, not closed by this rung):**

- All fixed weights are hand-set on dev seeds.
- The set-point target is a constant, as every homeostat's is.
- The set-point epoch and the gate closing after it are protocol steps.
- An inhibition-only set-point cannot raise a veto neuron whose drive is below target. It also cannot fully silence
  one whose excitation shunts past the maximal inhibition. G13 is where that shows (§3, §6).
- The rule is the engine's trace-based Vogels rule, not a biophysical model of GABA-A plasticity.
- ASK's own operating point is not regulated.
- v4's sAHP-relay residuals stand. No `sim/` edit.

## 3. Operating point and how it was chosen (dev seeds 7-12 only)

**Criterion (fixed before choosing):** the worst case over dev seeds 7, 8, 10 and 11 on the gates this rung's
companion controls: G1 and G7 (with G1's 1.0 Hz range floor), and G13. Dev seeds 9 and 12 are excluded and reported.
With the veto closed, their intact ASK at the most uncertain level is 0.43 and 0.90 Hz, below G1's range floor. The
veto can only remove drive, and ASK's operating point is not regulated here.

**No configuration measured met the criterion on all four seeds.** The frozen point is the best worst case among
those measured. It is registered with that stated; §6 names the gates at risk.

Quick dev probes (`--quick`: set-point epoch plus the intact, swap, veto-lesion and gain-lesion arms). Outputs are
scratch, not committed. Every run is reproducible with the committed runner and `--dev --quick --set NAME=VALUE`.

| lever | measured | decision |
|---|---|---|
| veto form: rival relay, uniform weights | seeds 8/10: veto is quasi-binary. The rival relay drops 13-14 -> 7 Hz between no evidence and evidence 0, then floors near 6.5 Hz. W_VA 16-40 silences ASK at every level | rejected |
| veto form: one shared pool driven by the sum | seeds 8/10/12: step-like. The falling sum disinhibits the veto: 3 -> 7 Hz between evidence 0 and 0.1, then flat near 8 Hz | rejected |
| veto form: rival relay + per-neuron spread 0.5 | seeds 8/10/12: graded favored veto (5-9 Hz at evidence 0 -> 13-14 Hz at 1.0), rival veto 0 | **chosen** |
| calibration input: weakest evidence, alternating classes | seeds 8/10/12, 24-48 reads: set-point ratios 2.1-8.7, not converging (non-stationary input for a per-synapse rule) | rejected |
| calibration input: no evidence | stationary input. Pooled form, seed 8, 24 reads: both halves at 0.8-1.3 Hz. Rival form (frozen point, 64 reads): 8 of 12 dev channels in band | **chosen** |
| eta 2.0 / 0.5 / 1.0 | 2.0 (alternating input, seeds 8/10/12): weights pinned at the bounds (std ~30 at mean 20-30), set-points 1.9-8.7x. 0.5 (no evidence, 32 reads): 3 channels still at 3.2-3.5x | **1.0, 64 reads** |
| veto-drive spread 0.25-0.3 with w_max 160-400 | seed 7's strongest channel diverges late (set-point 4.2x-24x) | rejected (spread 0.5, w_max 80) |
| W_MV 6 | 6 of 8 channels stay below target with their inhibition driven to ~0 (0.0-0.09x) | rejected (W_MV 10) |
| W_VA 3.0 / 1.0 / 1.5 | 3.0: ASK range 0.84 / 0.29 Hz on seeds 8/10. 1.0: seed 10's swap tail returns (0.09 Hz at evidence 1.0) | **1.5** |
| ASK set-point on (edge 5-6, target 3 Hz) | the stronger edge it needs re-exposes the tails (seed 10 swap rho +0.05/+0.06, seed 7 intact -0.39) | **off; banked method** |

At the frozen point (the `M15` probe: the same constants, before the inert `cv_norm` population existed). "v4
circuit" is the veto-lesion arm on the same calibrated pool.

| dev seed | G1 rho (v4 circuit) | G7 rho (v4 circuit) | ASK range intact / swap (Hz) | G3 gain share | G13 set-point (veto0, veto1) x target |
|---|---|---|---|---|---|
| 7 | -0.96 (-0.94) | -0.99 (-0.96) | 4.27 / 4.12 | 0.47 | 0.60, **2.78** |
| 8 | -0.90 (**-0.61**) | -0.98 (**-0.77**) | 1.60 / 1.25 | 0.36 | 0.79, **0.28** |
| 10 | -0.92 (-0.98) | **-0.44** (**-0.27**) | **0.57** / 0.49 | **0.10** | 0.56, 0.51 |
| 11 | -0.95 (-0.83) | -1.00 (-1.00) | 5.83 / 5.90 | 0.76 | 1.06, 0.60 |
| 9 (excl.) | -0.50 (-0.67) | -0.91 (-0.91) | 0.05 / 0.71 | 0.00 | **2.96**, 0.51 |
| 12 (excl.) | -0.50 (+0.09) | -0.94 (-0.97) | 0.32 / 1.72 | 0.14 | **3.52**, 0.56 |

What the veto does and does not do, on dev:

- **Fixes the comparator-driven tails.** Seed 8 goes from failing both G1 and G7 to passing both. Seed 12 intact
  loses its evidence-1.0 tail. Seed 10 swap improves (-0.27 -> -0.44) but still fails: its swap response is small
  (0.49 Hz), so one or two stray spikes decide the rank order.
- **Compresses ASK at the most uncertain level** on weaker seeds (seed 8: 3.23 -> 1.60 Hz; seed 10: 1.38 -> 0.57 Hz).
- **G13 is missed in both directions** on seeds 7, 8, 9 and 12 (§2 residual).

A dev confirmation smoke from the committed revision follows as a separate artifact commit, labeled DEV-SMOKE.
It covers the full gate set, including the fine G11 grid on the real substrate, determinism in a fresh process and
the integrity checks.

## 4. Gates (per seed; GO requires every REQUIRED gate on 6/6 of 42/43/44/100/101/102)

| id | required | pass condition |
|---|---|---|
| G1 | yes | v3/v4's, unchanged: Spearman rho(evidence, level-mean ASK Hz) on the intact arm <= -0.8 AND range >= 1.0 Hz; None fails |
| G3 | yes | v4's: closing `lc_ne -> ask_fb` removes >= 0.2 of the intact ASK range (`tools.lab.attributable_to`) |
| G6 | yes | the intact-arm digest (v4's plus the v5 population rates) is identical in a fresh subprocess, set-point epoch included |
| G7 | yes | class swap: Spearman rho <= -0.8 (v3/v4's statistic) |
| G8 | yes | comparator-relay lesion (meta_margin_fs -> meta_schema zeroed) **with the veto output closed**: rho > -0.5; None fails |
| G10 | yes | v4's: rho(evidence, lc_ne Hz) <= -0.3 AND lc_ne range >= 0.3 Hz |
| G11 | yes | §1 (the fine, reference-placed grid) |
| G12 | yes | v4's phasic test, unchanged |
| G13 | yes | **set-point reached:** each veto half's rate on the calibration input (no evidence), read with every plasticity gate closed, is in [0.5, 1.5] x 1.0 Hz |
| G4 | integrity, reported | edge AND lc lesioned: rho > -0.8 or None |
| G5 | precondition | metacog balance/confident/raster and comparator raster EXACT across every lesion arm (veto lesion included) |
| reported | no | veto-lesion arms (the v4 circuit on the same pool, both classes); relay lesion with the veto OPEN; veto class selectivity; the confident-half ASK attributable to the veto; the two-init set-point check (plastic rows reset to 12.0, epoch re-run); S1; G9 |

**Why G8 closes the veto.** v4's G8 asks whether the EDGE's coupling needs the comparator's margin computation, not
raw metacog activity. The veto is a second, downstream opponent read of the same comparator. With it open, a relay
lesion tests two margin computations at once. The open-veto arm is reported, not scored. Its failing direction stays
available to a reader.

**Integrity preconditions (any failure makes the seed's verdict UNDEFINED, never a pass):**

- v4's full list: metacog balance varies; byte-off (the pool minus exactly the declared synapses of v4's and v5's
  organs plus the point-edge equals the conflict_xedge rung's `coupled=False` pool key for key, and the removed count
  equals the declared count); GIRK routing exactly the declared set; metacog's read EXACTLY the base pool's; G5;
  restore exact; every lesion read back at measurement; lc_ne acts only through the feedback loop; curiosity's
  non-ASK regions silent; no host novelty scalar or neuromodulator subsystem.
- Added in v5:
  - inhibitory STDP eligible ONLY on `cv_inh1 -> cv_veto0` and `cv_inh0 -> cv_veto1` at the open gate;
  - the epoch changed no weight outside the plastic rows, and did change the plastic rows;
  - the plastic rows hash identical from the end of the epoch to the end of the scored run.

A missed set-point is written into the artifact's `operating_point_ack` (`tools/gates/operating_point`) and scored as a
G13 failure. It is never passed silently.

## 5. What a GO would and would not mean

A 6/6 GO means three things. On this 5-organ merged pool, an opponent commitment signal removes the confident-end
rise of metacog's summed comparator output from curiosity's ASK pool, and that signal is held at a no-evidence
set-point by inhibitory plasticity that reaches it on every seed. v4's phasic LC-NE response gain still passes its
multiplicative test on a grid placed where each seed's ASK actually responds. And the coupling is monotone and
class-symmetric.

It does NOT mean:

- that any weight other than the declared plastic rows was set by a learning rule;
- that ASK's operating point is regulated;
- that ASK reaches production's threshold (S1);
- that the circuit runs on the production pool or the chat path;
- anything about felt states. Functional read-outs only.

## 6. Known risks, stated before any evaluation run

1. **G13 (set-point) is at risk on seeds whose comparator halves are strongly asymmetric with no evidence.** Dev seeds
   7, 8, 9 and 12 missed in both directions: a strongly driven channel stays above target at maximal inhibition, and
   a channel below target cannot be raised by inhibition. An inhibition-only set-point is the bound. The biology it
   lacks is an excitatory arm (multiplicative synaptic scaling of the drive). This is named as the next companion if
   G13 fails.
2. **G1's range floor and G7 on weak seeds.** The veto removes drive at the most uncertain level too. v4's weakest
   evaluation range was seed 101 (1.66 Hz). Dev seed 10 lost 58% of its range. Seed 100's G7 failure has the same
   form as dev seed 10's swap (a U-shaped sum). The veto fixes the tail there but may leave too few graded levels.
3. **G3 on weak seeds.** Dev seed 10's gain share fell to 0.10 (v4 had it at 0.29 on the same seed).
4. **The G11 fine grid on the real substrate** is verified in the dev smoke, not assumed. The additive control's
   failing direction is part of G11 on every seed.

## 7. Compute and the combined verdict

Local, numpy, one process per seed (`OMP_NUM_THREADS=1`; memcap 2 GB). Peak RSS and wall time are measured in the
dev smoke and recorded in the artifact (`peak_rss_mb`, `elapsed_s`).

Dev confirmation smoke (after this commit; separate artifact commit, labeled DEV-SMOKE, not evidence):

```
SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds <7|8|10|11> --dev --out research/findings/raw/_curiosity_commitment_veto_v5_dev_s<seed>.json
```

Evaluation, one command per seed, pinned to the pushed revision that contains this document and the runner:

```
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 42 --out research/findings/raw/_curiosity_commitment_veto_v5_s42.json
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 43 --out research/findings/raw/_curiosity_commitment_veto_v5_s43.json
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 44 --out research/findings/raw/_curiosity_commitment_veto_v5_s44.json
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 100 --out research/findings/raw/_curiosity_commitment_veto_v5_s100.json
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 101 --out research/findings/raw/_curiosity_commitment_veto_v5_s101.json
cd <worktree> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 102 --out research/findings/raw/_curiosity_commitment_veto_v5_s102.json
```

**The ONLY pre-registered verdict** is `--combine` over the six per-seed files. It refuses unless the union is exactly
{42,43,44,100,101,102} with no duplicate, one mechanism string, one operating point, and one runner blob across the
inputs' provenance SHAs:

```
.venv/bin/python -m research.runners._curiosity_commitment_veto_v5_derisk --combine research/findings/raw/_curiosity_commitment_veto_v5_s42.json research/findings/raw/_curiosity_commitment_veto_v5_s43.json research/findings/raw/_curiosity_commitment_veto_v5_s44.json research/findings/raw/_curiosity_commitment_veto_v5_s100.json research/findings/raw/_curiosity_commitment_veto_v5_s101.json research/findings/raw/_curiosity_commitment_veto_v5_s102.json --out research/findings/raw/_curiosity_commitment_veto_v5_6seed_combined.json
```

## 8. Amendment log

v1, 2026-09-24: initial registration. Written after the dev-seed substrate measurement and the quick calibration
probes, and before any run of this mechanism on seeds 42/43/44/100/101/102. It governs runner revision `9cef44164` on
`research/curiosity-ask-operating-point`.

v1.1, 2026-09-24, still before any evaluation-seed run of this mechanism. The dev confirmation smoke failed G11 on
all four seeds for an instrument reason. It ran on seeds 7/8/10/11 at revision `235d5bfee`, with artifacts
`research/findings/raw/_curiosity_commitment_veto_v5_dev_s{7,8,10,11}.json`.

- The reference (lc-off) ASK curve is rise-dip-rise on this substrate. Seed 7's placement scan reads 2.25 Hz at
  drive 1.0, 1.18 Hz at 1.2 and 4.17 Hz at 1.6. The v4 prereg's calibration noted the first fall (the strong drive
  recruits ASK's slow feedback on the onset transient). The second rise only appears above v4's grid.
- The v1 rule placed the peak at the GLOBAL maximum (1.6). The scored limb then spanned the dip and the second rise,
  where the lc-on arm collapses onto the reference. G11c read -0.16 to -0.58.
- On the first rising limb the lc-on/off ratio rises (seed 7: 1.07, 1.49, 1.87, 2.38).
- The additive control's failing direction held on the fine grid on all four seeds (trend -0.71 to -0.92, instrument
  valid).

The change: Peak becomes the FIRST local maximum at or after onset (§1 step 2). No other gate, threshold, constant or
statistic changes. The selftest now places dev seed 7's measured rise-dip-rise scan at its first peak (1.0, not 1.6).
The governed runner revision is `4cbedbaf8`, the commit that carries this code change, immediately before this amendment. The v1
dev artifacts stay committed as the record of the defect. A dev re-smoke from the amended runner follows
(`_curiosity_commitment_veto_v5_dev_v11_s<seed>.json`). The evaluation commands in §7 are unchanged and pin the
pushed revision that contains this amendment.
