---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2e-directed-novelty-biased-exploration-equalise-action-frequency
backend: numpy
runner: research/runners/_vocal_gateb_stage2e_directed_novelty.py
builds-on: 2026-08-06-gateB-stage2d-uncertainty-gated-exploration-NO-GO.md
surpasses-method-wall: 2026-08-06-gateB-stage2d-uncertainty-gated-exploration-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2e_directed_novelty/numpy.json
  - research/findings/raw/gateb_stage2e_directed_novelty/numpy_confgated.json
---

# Gate B Stage 2e: directed novelty EQUALISES action sampling and removes the 2d yoked-lock killer, but the exploration/exploitation trade-off it creates still fails the per-seed steer gate at 4/6

## Verdict

**STAGE2E_NO_GO** (earned: preconditions hold; reward-OFF byte-identical, both
lesions, and reversal PASS; only the per-seed steer gate fails, as in 2d). A NEURAL
DIRECTED-novelty drive -- extra excitatory current into the LESS-sampled action's
SPIKING proposal population, scaled by the per-action count DEFICIT (a novelty/
habituation read-out; the counts come from the spiking motor read-out) -- does
EXACTLY its named job: it EQUALISES action sampling (yoked train-p0 -> ~0.50 on
every seed, `yoked_action_balance_err_mean` **0.018** <!--derived-->) and thereby REMOVES the 2d
per-seed killer (all six D_yoked now **<= 0**: -0.67/-0.74/0/-0.15/-0.75/0 vs 2d's
coincidental **+1.0** on 730605). But equalising sampling in ALL conditions fights
the EXPLOITATION the contingent condition needs, so `steer_seed_passes` is still
**4/6 (need >= 5)** -- the sole unmet gate, its FAILURE MODE moved from 2d's
yoked-lock variance to an exploration/exploitation trade-off. Artifacts:
`research/findings/raw/gateb_stage2e_directed_novelty/numpy.json` (ungated),
`research/findings/raw/gateb_stage2e_directed_novelty/numpy_confgated.json`
(confidence-gated).

## The directed-novelty drive is NEURAL, load-bearing, and equalises sampling as designed

- **Drive:** during the onset window, extra external current (peak
  `NOVELTY_DRIVE_MAX_PA=350` pA, scaled by `min(1, |c0-c1|/EQUALIZE_DEFICIT=2)`) is
  added to `proposal_{under-sampled}` -- the SAME spiking channel `_apply_afferents`
  drives with the practice-arousal/thalamic tonic current. It does NOT pick the
  action: the BG competition + motor argmax still select the winner; the drive only
  lets the under-sampled proposal compete. Applied ONLY in training, never in the
  frozen WTA test, never in the reward-OFF equivalence build -> byte-identical guard
  UNTOUCHED (weights + raster match). This is the Bogacz-Brown / Oudeyer-Schmidhuber
  under-sampled-action novelty bonus; amplitude-only OU (2b: 40..600 pA) is
  UNDIRECTED and cannot break a bias, this is DIRECTED to the deficit channel.
- **Load-bearing (measured, this arc).** The `novelty_lesion` control turns the
  directed drive OFF (back to the 2d undirected regime), all else identical, on the
  SAME lesion seed 730605: yoked train-p0 -> **0.975 / 0.0** (fully locked) and
  D_yoked -> **1.0** (the 2d failure reproduced). With the drive ON, yoked train-p0
  -> ~0.50 and D_yoked <= 0. The equalisation is caused by the neural drive.

## Two variants, one residual: contingent-commitment vs yoked-equalisation trade off

**Ungated** (`numpy.json`): perfect yoked equalisation (`balance_err` 0.018 <!--derived-->, all
D_yoked <= 0), but the always-on drive keeps forcing 50/50 sampling even in the
CONTINGENT condition -> on the two strongly-biased NON-exploring seeds it prevents
the exploitation lock: **730603 D_contingent 0.00** (conf_c0 0.24, never committed),
**730604 D_contingent 0.10**. Steer passes = {730601, 730602, 730605, 730606} = 4/6.

**Confidence-gated** (`numpy_confgated.json`, the committed default): the drive is
scaled by `(1 - conf)` using the SAME neural uncertainty read-out that gates the OU
sigma, so curiosity should yield to learned value. This RESTORES contingent
commitment on **730603 (D_contingent 0.00 -> 1.00)**, but a spurious conf-rise in a
lucky yoked run fades the drive early -> sampling de-equalises (`balance_err`
0.018 -> 0.11 <!--derived-->) -> the yoked lock RETURNS on **730605 (D_yoked +0.80**, conf_y0
spuriously 1.00). 730604 still fails (contingent target-1 never commits, conf_c1
0.48). Steer passes = {730601, 730602, 730603, 730606} = 4/6.

**The residual is now exactly located.** D_contingent_mean_exploring = **1.00** in
BOTH variants (the three exploring seeds steer perfectly). The two variants' per-seed
passes DIFFER; their UNION is **{601, 602, 603, 605, 606} = 5/6** -- only 730604 (the
maximally-biased seed, baseline p0 = 1.0) fails both. A gate that keeps the drive ON
in yoked (equalise) AND OFF in contingent (exploit) would clear >= 5/6. The blocker
is that the confidence signal (value-DIFFERENCE of the str_d1 rates under DECOUPLED
reward) cannot separate genuine action->reward contingency from a coincidental yoked
reward STREAK: a lucky streak transiently inflates the value difference, spuriously
gating off the equalising drive exactly where it must stay on.

## Frozen criteria (unchanged from the Stage-2 preregistration)

- Reward-OFF byte-identical to Stage-1 (weights + raster): PASS.
- Acquisition lesion: contingent 1.00 vs acq-lesion 0.30 (delta 0.70 >= 0.15) PASS.
  Expression lesion: vs 0.35 (delta 0.65 >= 0.15) PASS.
- Same-brain reversal: P(B) 0.00 -> 1.00 (>= 0.60, and 1.00 > 0.00) PASS.
- Contingency steer_seed_passes: **4/6 (need >= 5) FAIL** in BOTH variants --
  D_contingent_mean_exploring 1.00; the trade-off, not systematic yoked steering.

## Quantified residual + exact next mechanism

The named surpass (DIRECTED novelty that equalises action frequency) IS achieved
(balance_err 0.018 <!--derived-->, load-bearing) and it DOES remove the 2d yoked-lock killer (all
D_yoked <= 0). The NEW blocker is the exploration/exploitation trade-off: the drive
must stay ON in yoked and OFF in contingent, but the value-magnitude confidence
signal mis-fires because decoupled reward can transiently mimic contingency.

**Next mechanism (Stage 2f, biology-grounded, in-substrate): a CONTINGENCY-based
confidence gate.** Gate the directed drive (and the OU sigma) on an estimate of the
Hammond CONTINGENCY dP = P(reward|action) - P(reward|no-action), NOT on raw value
magnitude. In yoked, reward is DECOUPLED so dP ~ 0 EVEN under a lucky streak -> conf
stays low -> the equalising drive stays ON -> D_yoked ~ 0 low-variance. In
contingent, dP is high -> conf rises -> the drive fades -> the brain exploits the
target -> D_contingent stays high. The neural substrate already exists: the opponent
D2/indirect-pathway omission (negative-RPE) arm carries the "action taken, reward
OMITTED" evidence per action; route that into the confidence read-out (a D1-minus-D2
per-action contrast) so confidence reflects CONTINGENCY, not value. This targets the
5/6 the two current variants split between them. Closure is deferred to a METHOD (a
value-magnitude confidence signal cannot separate contingency from a coincidental
streak), not the CAPABILITY.
