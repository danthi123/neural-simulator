---
type: preregistration
status: preregistered
date: 2026-09-24
mechanism: HOMEOSTATIC REGULATION OF THE READOUT PORT'S OPERATING POINT (`--port-homeostasis ip`,
  `_learn_port_homeostasis` + `_ip_port_current` + `_attention_gated_soft_fbgain_ip_class_read` in
  `research/runners/_vision_lindiscrim_readout_derisk.py`). Every LIF unit of the class-population port learns
  its OWN multiplicative synaptic-scaling gain and intrinsic threshold from its own realized spike counts on
  TRAIN trials (Triesch-rule intrinsic plasticity toward an exponential rate distribution, mean 1/n_classes of
  the refractory ceiling), labels never read, then FROZEN for every read. Applied to the NEUTRAL finding's
  gain-only arm (`--readout attention-gated-soft-fbgain --attn-gain-exponent 1.0 --fb-strength 0.0`) at the
  satdiv-GO front-end operating point, everything else byte-identical.
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PREREGISTERED -- no evaluation-seed run exists at commit time. Dev seeds 7/8/9 (NOT evaluation
  seeds) were used to build and sanity-check the rule and are reported below as dev data carrying no
  pre-registered weight.
artifacts:
  - research/findings/raw/lanes/perception/dev_portip/portip_dev_s7_s8_s9.json (dev seeds 7/8/9, IP arm;
    committed in bc65929ee, after the code commit 9b5930361 and before this pre-registration)
  - research/findings/raw/lanes/perception/dev_portip/portip_dev_s7_neutral.json (dev seed 7, NEUTRAL
    gain-only arm, `--port-homeostasis none`; same commit bc65929ee)
  - research/findings/raw/lanes/perception/dev_portip/portip_dev_s7_lesion.json (dev seed 7, lesion arm
    `--ip-lesion-update`; same commit bc65929ee)
external: Turrigiano, Leslie, Desai, Rutherford & Nelson (1998), Nature 391:892-896, doi:10.1038/36103 <!--derived-->
  (PMID 9495341) -- synaptic scaling of ALL of a neuron's inputs as a function of its own activity; Desai,
  Rutherford & Turrigiano (1999), Learn Mem 6:284 (PMID 10492010) -- activity-dependent regulation of
  intrinsic excitability; Weber & Triesch (2008), Neural Comput 20:1261, doi:10.1162/neco.2007.02-07-472 <!--derived-->
  (PMID 18194109) -- intrinsic plasticity with a gain and a threshold parameter maintaining an exponentially
  distributed firing rate (the Triesch 2005 gradient rule). Read via PubMed, recorded lane-tagged with
  `tools/record_external_search.sh` this session. Local anchor: Kandel PNS-6e, "Neuronal Excitability Is
  Plastic" (research/biology/readout-port-homeostasis.md).
builds_on:
  - research/findings/2026-09-24-vision-configural-binding-spiking-feedback-divisive-gain-control-readout-NEUTRAL-6seed.md
    (the NEUTRAL result this targets: every arm, including RANDOM and TRAIN, reads exactly 0.25 on 12/12 runs)
  - research/biology/readout-port-homeostasis.md (new registry entry, this commit)
prereg-same-commit: this commit adds NO research/findings/raw artifact; the dev-seed artifacts cited above
  were committed in bc65929ee (before this one) and are declared dev data that no gate below reads.
---

# Readout-port homeostasis (intrinsic plasticity + synaptic scaling): pre-registration

**This is a pre-registration only.** It fixes the mechanism, the exact commands, the gates and the bands
before any evaluation seed (42/43/44/100/101/102) is run with `--port-homeostasis`. The mechanism, its
selftest (9b5930361) and the dev-seed artifacts (bc65929ee) were committed before this one.

## The wall question, asked first, and the measured answer

"What does the real system run alongside this readout that we replaced with a constant?" The NEUTRAL
finding left this as a code-reading hypothesis. It is now MEASURED on a non-evaluation dev seed (7), in the
committed dev artifact `portip_dev_s7_s8_s9.json` (block `port_homeostasis.learned.train_drive_*`) and the
matching NEUTRAL dev run: the pre-spike class drive the port's LIF units receive sits at trial-INDEPENDENT
per-class means of -833.0206 / 376.8297 / 547.0126 / -86.8217 current units, with a trial-to-trial SD of
only 1.9424-2.3884 (`train_drive_mean_per_class` / `train_drive_sd_per_class`). Two
class populations are rectified silent and two fire at the refractory ceiling (16 spikes per 48 ms, every
unit, every trial). The two saturated populations tie on every trial and the arg-max returns the same index:
a CONSTANT output. The RANDOM control's drive is orders of magnitude larger still (a scratch diagnostic on
dev seed 7, not a committed artifact) and saturates the same way, which is why RANDOM also read exactly
chance.

The port receives that drive through two fixed host constants (`read_gain` 2.5, `read_bias` 1.0). A real
neuron is not handed its operating point: synaptic scaling (Turrigiano et al. 1998) and intrinsic-
excitability regulation (Desai et al. 1999) run alongside every synaptic change and keep firing rates out
of saturation. That is the companion process this port lacked.

## The mechanism (commit 9b5930361; unchanged here)

Per unit i: `I_i = g_i * (x_i - theta_i)`, rectified, into the unchanged LIF stepper. `x_i` is the existing
pre-port drive (duplicated verbatim in `_fbgain_pre_port_drive`). Per epoch over the TRAIN trials, from the
unit's own spike fraction `y` (count / refractory ceiling) and its own current relative to threshold:

```
delta_n  = 1 - (2 + 1/mu) y_n + y_n^2 / mu
theta   <- theta - eta_theta * mean_n(delta_n) / g
log g   <- log g + clip(eta_gain * mean_n(1 + g (x_n - theta) delta_n), +-kappa)
```

Frozen constants (dev seeds only, never an evaluation seed): `mu = 0.25` (1/n_classes, fixed a priori),
`ip_epochs 400`, `eta_theta 0.05`, `eta_gain 0.01`, `kappa 0.5`. These are the first values tried; they
were not swept. The RANDOM and label-shuffle controls each learn their own homeostasis with the same rule
on the same train drive (an un-regulated random port would stay collapsed and inflate learned-minus-random).

**Host shortcuts declared.** (1) The parameter update is host numpy bookkeeping of a per-unit state, as is
every plasticity rule in this runner; its inputs are local to the unit. (2) The set point `mu` is a
constant. (3) Unchanged from every prior arm: the top-down template `A_c`, the gain multiply, the
cross-class mean-centering of `net`, `read_gain`/`read_bias`, and the arg-max over class-population spike
counts. The linear discriminant `V` is still fit in closed form (ridge). This lever closes none of those; it
replaces only the fixed operating point of the LIF port.

## Dev-seed data (NOT evaluation seeds; no pre-registered weight)

From `research/findings/raw/lanes/perception/dev_portip/portip_dev_s7_s8_s9.json` (HOMEO arm, dev seeds):

| dev seed | `LEARNED_spkwta_held` | `LEARNED_spkwta_train` | `RANDOM_spkwta_held` | `scramble_learned_held` | `train_pred_entropy_bits` | `lesion_permuted_held` | `position_decode_heldsplit` | `capability_go` |
|---|---|---|---|---|---|---|---|---|
| 7 | 0.5208 | 1.0 | 0.2708 | 0.2188 | 2.0 | 0.25 | 0.4375 | false |
| 8 | 0.4583 | 1.0 | 0.2396 | 0.2292 | 2.0 | 0.25 | 0.5 | false |
| 9 | 0.4375 | 1.0 | 0.25 | 0.2188 | 2.0 | 0.25 | 0.3333 | false |

The NEUTRAL arm at dev seed 7 (`portip_dev_s7_neutral.json`) reads 0.25 on held and train, as on every
evaluation seed. On all three dev seeds the port's train output is no longer constant, held-out accuracy is
above chance, the permuted-state lesion falls back to 0.25, and scramble stays near chance. `capability_go`
is false on all three, for two different reasons: on dev seeds 7 and 8 `position_pooled_out` fails
(position decodable from the class-population spike code above the 0.40 bar); on dev seed 9 held accuracy
is just under the 0.44 no-go-floor-plus-margin bar. The dev lesion run (`portip_dev_s7_lesion.json`) is the
runner-level identity check of the lesion arm against the dev NEUTRAL run: its `decode`, `reframe`,
`dissociation`, `verdicts` and `conj_select` dicts compare `==` to the NEUTRAL run's (checked by parsing both
JSON files), and its `max_abs_log_gain` and `max_abs_theta` are both 0.0.

## Commands (fixed)

Common flags (the NEUTRAL gain-only arm's exact flags):
`--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --s2-satdiv-n 2.0 --ridge 1.0 --n-glimpses 6
--n-s2 96 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 --conj-select-kwta-frac
0.1 --conj-n 1152 --conj-offset-max 4 --readout attention-gated-soft-fbgain --attn-gain-exponent 1.0
--fb-strength 0.0 --heldout-position --scramble-null`

- HOMEO arm: common + `--port-homeostasis ip --seeds {SEED} --out research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s{SEED}.json`
- LESION arm: common + `--port-homeostasis ip --ip-lesion-update --seeds {SEED} --out research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIPlesion_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s{SEED}.json`
- NEUTRAL arm: NOT re-run. The pinned artifacts
  `research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s{SEED}.json`
  as they exist at commit `9f4b35138` (read with `git show 9f4b35138:<path>`).

All 12 runs staged on the pool at this pre-registration's commit. Seed 42 is additionally run locally into a
`_local`-suffixed path as a determinism cross-check; the POOL artifact is the scored one.

## Gates -- one line each: "this gate FAILS if <realistic outcome>"

Every quantity is read from `by_code.count.per_seed[0]` of the named artifact.

- **G0 INTEGRITY (an integrity smoke, NOT evidence).** LESION's `decode`, `reframe`, `dissociation` and
  `verdicts` dicts equal the pinned NEUTRAL artifact's, exactly (`==` on parsed JSON), on 6/6 seeds, AND
  LESION's `port_homeostasis.learned.max_abs_log_gain == 0.0 and max_abs_theta == 0.0` (the lesion holds
  at measurement). FAILS IF the new current path perturbs any value, or an update leaks through the lesion.
  (It cannot detect a shift of the LIF read seed: the NEUTRAL port is saturated or silent, count SD exactly 0, so
  noise never changes a spike there. A read-seed shift would not favour any arm.) It passes by construction when the code is right, so it is not counted as evidence that
  the homeostasis is the cause; it IS the check that "the lesion of the update restores the collapse".
- **G1 TRAIN OUTPUT NOT CONSTANT.** HOMEO `port_homeostasis.train_pred_entropy_bits >= 1.0` AND
  `train_count_trial_var_mean > 0` on 6/6 seeds. FAILS IF on any seed the rule does not bring every class
  population into its dynamic range (a class left silent or at the ceiling, theta oscillating, the gain step
  bound too slow for that seed's offsets) -- the NEUTRAL arm has entropy 0 bits and variance exactly 0.
- **G2 HELD-OUT ABOVE THE NEUTRAL ARM (paired).** `d_s = HOMEO LEARNED_spkwta_held - NEUTRAL
  LEARNED_spkwta_held`; PASS iff `mean(d) > 0` and `t = mean(d) / (SD(d)/sqrt(6)) >= 2.571` <!--derived--> (df 5, two-sided
  0.05). `SD(d) == 0` is UNDEFINED and counts as FAIL. FAILS IF the regulated port reads held-out trials at
  or near chance -- realistic, because the homeostat re-standardizes each class's drive separately and the
  discriminant was never fit for that, and LIF noise is added on top.
- **G3 SPECIFIC LEARNED STATE (the evidential lesion).** `e_s = HOMEO LEARNED_spkwta_held -
  lesion_permuted_held` (the learned per-unit state rolled by one class population, same read seed); PASS
  iff `mean(e) > 0`, `t >= 2.571` <!--derived-->, AND `lesion_permuted_held <= 0.40` on >= 5/6. FAILS IF a generic
  shrink of the drive -- not each unit's own learned state -- does the rescuing.
- **G4 SCRAMBLE NULL.** HOMEO `scramble_learned_held <= 0.40` (`verdicts.scramble_null_pass`) on 6/6.
  FAILS IF the frozen homeostatic state lets a pixel-scramble-surviving statistic through.
- **G5 FAIR NULLS.** `RANDOM_spkwta_held <= 0.40` AND `label_shuffle_null <= 0.40` on >= 5/6 seeds, each null
  regulated by its own homeostasis under the same rule. FAILS IF the homeostat alone, without the learned
  discriminant or the true labels, lifts held-out reads above chance (dev seed 7's `label_shuffle_null` read 0.3438).
  G2 compares against the NEUTRAL arm, which is a constant 0.25, so G2 alone is only "held above chance".

Reported, not gating the port verdict (they gate the CAPABILITY claim): `verdicts.capability_go` count,
`learning_load_bearing` count (learned minus RANDOM >= 0.10, RANDOM now regulated too), `position_pooled_out`,
`label_shuffle_null`, and the component diagnostics `diag_threshold_only_held` / `diag_gain_only_held`.

## Bands (evaluated in this order; exhaustive)

1. **VOID** -- G0 fails or G4 fails. Nothing else is read.
2. **NO-GO (port not repaired)** -- G1 fails.
3. **NEGATIVE-HELD** -- G1 passes, G2 fails: the train output is regulated but carries no held-out signal.
4. **NON-SPECIFIC** -- G1 and G2 pass, and G3 or G5 fails.
5. **PORT-REPAIRED, CAPABILITY SHORT** -- G1-G5 pass and `capability_go < 5/6`. The finding names which
   capability sub-gates fail on which seeds as the residual.
6. **GO** -- G1-G5 pass and `capability_go >= 5/6`.

**Prediction written before any evaluation seed:** G0-G5 pass, and the most likely band is 5 (PORT-REPAIRED,
CAPABILITY SHORT): `capability_go` was false on all three dev seeds (`position_pooled_out` on two, the
no-go-floor bar on one), so `capability_go >= 5/6` is not expected. The run is staged anyway because its
registered question is the port gates G1-G3, which the dev data predict will pass and which can fail; the
capability residual it names (position leaking into the regulated class-population code) is the next lever's
input, not something this lever claims to fix.

## Amendment log

(none after registration. Before this file's first commit, the adversarial review's two non-blocking items were
adopted: G5 fair nulls added to bands 4-6, and G0 no longer claims to detect an LIF read-seed shift.)
