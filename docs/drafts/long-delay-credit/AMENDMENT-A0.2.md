# Amendment A0.2 (DRAFT FOR CLAUDE'S REVIEW): soft bounds on the corticostriatal weight dependence (D2 floor saturation)

Amends `research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md` in the same way A0 does (recorded here; prereg text not edited). Input: RULING ldc-d2-floor (2026-10-09), steps (a)-(c) done on dev only. Nothing here is evidence; dev seeds only, no tuning against bars. Units: time ms, weights in the runner's unitless scale (`w_max` 90.0).

## Defect

The plastic corticostriatal weights into the D2 MSNs saturate at the hard lower bound. The learning rule (`sim/bridge.py` "4c. C2": `weight_updates = effective_reward_lr * effective_signal * cp_eligibility_trace`, then `cp_d1_d2_sign`, then `cp.clip(data, hebbian_min_weight, hebbian_max_weight)`) has no weight dependence: the step is the same size at weight 90 and at weight 0.5, and the clip then discards the part below `hebbian_min_weight = 0.0` (set in `credit_extension`, `research/runners/bg_corticostriatal_credit.py` lines ~140-150). At the floor the rule keeps issuing negative steps that the clip throws away — wasted credit — and the D2 pathway is functionally clamped: behaviour cannot move. This is the first weight-dependence lever against this defect (step (c): `bash tools/before_you_build.sh "D2 weight floor saturation"` ran rc=0, record in `corpus_checks_shared.jsonl`; prior ldc levers spent on credit/timing).

## Evidence (dev only; probe `research/runners/_ldc_dip_observability.py`, outputs `research/findings/raw/_long_delay_credit_bg/d2_floor_observability/{on_s7,on_s8,cut_s7,cut_s8}.json`, 400 trials each; params verified in-file: `lr` 5.0, `w_max` 90.0, `elig_tau_ms` 1000.0, `da_window_ms` 30.0, `host_reward_signal` 0.0)

**(a) The dip in the CUT arm is not the LHb relay.** Median-split correlate over the first 80 trials (dips = `da_min` below median-0.004; all dip trials unrewarded): on_s7 dip trials show lhb +91, rmtg +47, lhb_inh -62, snc -57 spikes vs non-dip (on_s8: +85/+48/-66/-55) — the LHb->RMTg->SNc relay explains the ON-arm dip. In cut_s7/cut_s8 the lhb/lhb_inh/rmtg differences are all within +/-5 spikes (the cut removes `reward_us->lhb_inh`, so the relay no longer tracks the dip) while snc still moves (-57/-50) alongside total striatal activity (cut_s7 +110: str_d2_0 +70, str_d2_1 +42, d1 ~0; cut_s8 +41: d1 +39/+42, str_d2_0 -46). READ: in the cut arm the SNc pause tracks striatal activity onto SNc (`w_strio_snc`, GABA), not the LHb route. CAVEAT: `spk_tot` is whole-trial, not window-resolved; a time-resolved probe is needed to confirm eligibility-window timing. `da_argmin` lands ~150-250 ms, inside the eligibility window at both onsets.

**(b) D2 trajectories hit the floor, arm-independently, seed-dependently.** on_s8: `ctx_a->str_d2_0` 23.34 -> exactly 0.000 at trial 329, <=1.0 for 224/400 trials, final 0.670; `mem_a->str_d2_0` 14.20 -> 0.000 at 288 (256 trials <=1), final 2.168; floor_frac (the probe's second per-trial element = fraction of the pathway's synapses at the bound) hits exactly 1.000 — 100% of the pathway parked at 0 — on 32/400 trials (ctx) / 50/400 (mem), with nonzero floor_frac on 346/400 / 304/400. cut_s8: `ctx_a->str_d2_0` min 0.023 at 336 (111 trials <=1); `mem_a->str_d2_0` 0.000 at 336 (193 trials <=1) — saturation is arm-independent, not the omission route. Seed 7 (both arms) never reaches the floor (minima 2-16): seed-dependent. Pilot aggregate (PILOT-4B): 16/32 ON cells `void_floor_ceiling` (the flag at `_ldc_omission_pilot.py` line ~471: mean final weight <= 0.05 x `w_init`), all 62 flags are floor, zero ceiling; 46/62 target `str_d2_*` (16/62 `str_d1_*`; 26 corticostriatal, 36 mem-to-striatal). Once parked, the mean stops moving for stretches while `lr*signal*elig` keeps being computed and discarded by the bounds clip — the wasted-credit claim rests on that clip step (4c adds, then clips to bounds; there is no +/-1 per-step clamp anywhere).

## Proposed weight dependence: trafficking-slot soft bounds

**Mechanism (biological precedent).** The expression of LTP and LTD at a glutamatergic synapse is limited by receptor-slot occupancy at the postsynaptic density: potentiation inserts AMPA receptors into free slots, so its magnitude scales with free slots (proportional to `w_max - w`); depression removes surface receptors, so its magnitude scales with receptors present (proportional to `w - w_min`) (Lisman & Raghavachari 2006 slot model; Huganir & Nicoll 2013 receptor trafficking; the standard soft-bound form of van Rossum et al. 2000 / Fusi et al. 2005 cascaded plasticity). This is a weight dependence inside the same three-factor rule — not a host-side clamp, not a wrapper, not a reward-shaped bound.

**Exact parameterization.** In `sim/bridge.py` "4c. C2", after the `cp_d1_d2_sign` and TAN-gate multipliers and before `self.cp_connections.data += weight_updates`, gated on a new flag (default 0 = byte-identical; the `hebbian_oja` precedent at `sim/config.py` line ~694):

```
# cfg.reward_soft_bounds == 1, plastic synapses only (cp_plasticity_rate_gain > 0), enable_stdp False
R = max(cfg.hebbian_max_weight - cfg.hebbian_min_weight, 1e-6)
w = self.cp_connections.data[:actual_nnz]
up   = clip((cfg.hebbian_max_weight - w) / R, 0.0, 1.0) ** cfg.reward_soft_bound_n_up
down = clip((w - cfg.hebbian_min_weight) / R, 0.0, 1.0) ** cfg.reward_soft_bound_n_down
weight_updates = weight_updates * where(weight_updates >= 0.0, up, down)
```

The factor multiplies the SIGNED update after the D2 sign flip, so a D2 synapse's potentiation (negative teaching term x sign -1 > 0) fades near the ceiling and its depression fades near the floor, per-synapse. The existing `cp.clip` stays as a safety rail (float error only). Defaults `n_up = n_down = 1.0` = the textbook linear soft bound — no free fitted constant.

**CreditParams additions** (`bg_corticostriatal_credit.py` CreditParams class ~lines 60-100, passed in `credit_extension`'s `updates` dict lines ~140-150, mirrored in `_ldc_credit_omission.py`):

```
soft_bounds: int = 0            # 0 OFF (byte-identical), 1 = trafficking-slot soft bounds
soft_bound_n_up: float = 1.0    # free-slot exponent on (w_max - w)/R
soft_bound_n_down: float = 1.0  # occupied-slot exponent on (w - w_min)/R
```

**CoreConfig additions** (`sim/config.py`, C2 block lines ~994-1005): `reward_soft_bounds: int = 0`, `reward_soft_bound_n_up: float = 1.0`, `reward_soft_bound_n_down: float = 1.0`. Applied to ALL plastic synapses (no D2 special-casing; the mechanism is universal biology and a reward/D2-shaped scope would be a host prior).

## What stays unchanged

The registered runner and organ, the prereg bars and gate seeds, Step 4's banked sweep, the frozen Step-4b omission-route constants, `lr` 5.0, `w_max` 90.0, `hebbian_min_weight` 0.0 (now the asymptote reference, still the safety clip), `restrict_clip_to_plastic` (lines ~173-182) as is, and the hard clip itself. Step 4b / this amendment are draft builds on dev seeds in pilot copies; a route or rule change enters the registered organ only through Claude's A1, with its own byte-identical-off test. No bar is touched or tuned here.

## Tests (dev seeds only)

1. **Byte-identical-off.** `soft_bounds 0`: rebuild the seed-7 credit_immediate on150 ON cell with `_ldc_dip_observability.py`; `weights_per_trial` and final `cp_connections` must match `d2_floor_observability/on_s7.json` bitwise (pattern: `tests/test_ldc_omission_identity.py`).
2. **Dev seeds 7/8, 400 trials, credit_immediate onset 150, arms on and cut, `soft_bounds` 0 vs 1** (the observability probe cells). Expect on seed 8 (saturating): `ctx_a->str_d2_0` no longer reaches exactly 0.000 at ~329; trials with mean str_d2 weight <= 5% of init collapse from 224/400 (on_s8 ctx) to <25/400; floor_frac stops saturating at 1.000 (soft bounds brake the approach before 100% of synapses are parked) and the per-trial mean-weight step shrinks monotonically with w (g_down). Seed 7 (non-saturating control): trajectories shift slightly, no floor flags either way. Also verify the rebound: a near-floor weight driven back up by rewarded trials moves at g_up ~ 1.
3. **Rerun the voiding PILOT-4B cells** with `soft_bounds 1`, same frozen constants and clear flag, `--n-train 400`: credit_on30 ON s9,s10,s12,s14,s15 and credit_on150 ON s8,s9,s11,s12,s14, each with its CUT pair (20 cells). Readout: `void_floor_ceiling` rate (today 10/16 credit ON cells, 46/62 flags str_d2->0) and the paired ON-minus-CUT success, reported as in PILOT-4B. No bar tuning; results are method diagnosis only.

## Bar-sanity line (run BEFORE any gate use)

```
python tools/bar_sanity.py docs/drafts/long-delay-credit/bar_sanity_spec_softbounds.py --n 1000 --seed 0 --out docs/drafts/long-delay-credit/bar_sanity_softbounds.json
```

Affected bar: the STEP4B section-4 credit read (on passes U/R/I, off fails U, cut fails R; sanity and credit_immediate >= 0.8). The spec's `decide` calls that real scorer with `void_floor_ceiling` cells UNDEFINED; `perfect(rng)` = a brain whose D2 weights stay in range and whose ON arm signs the outcome; `null(rng)` = a saturated-D2 brain (weights at floor, behaviour at chance). Only once that record exists does the amendment/prereg carry its bar-sanity record line pointing at `docs/drafts/long-delay-credit/bar_sanity_softbounds.json` with verdict OK (gate `tools/gates/bar_sanity_line.py`).

bar-sanity: N/A this draft registers no bar; the spec and its record are created after Claude's approval (bars are Claude-only)

## Ablation plan (six gate seeds 42/43/44/100/101/102; after Claude's approval, via A1)

2 x 2: `soft_bounds` {0,1} x arm {on, cut}, credit_immediate onset 150 (onset 30 secondary), 400 trials, frozen constants. Capability claim: with soft bounds the D2 corticostriatal weights stay in a learnable range AND behaviour can move. Mechanism-necessary criterion fixed before the run: (i) soft_bounds 1 cuts the str_d2 floor-void rate to <= 1/6 cells per arm AND (ii) paired ON-minus-CUT success deviates from the soft_bounds 0 arm beyond seed spread on >= 4/6 seeds. If (i) holds and (ii) does not, soft bounds fix the instrument, not the capability: report honestly and take the named contingency below as the next lever. If soft_bounds 0 behaves the same as 1, the mechanism is NOT necessary — the claim is voided, not softened.

## Risks and open questions

1. **Asymptote is not a setpoint.** Soft bounds brake the collapse and remove the dead zone/wasted clipped credit, but a persistent net-negative teaching term still drives w -> 0 asymptotically; D2 drive may still fall below functional range in 400 trials. Named contingency (not built here): a postsynaptic homeostat — Turrigiano-type synaptic scaling of the str_d2 population toward a target rate, or a BCM sliding threshold (Bienenstock et al. 1982) — as the next weight-dependence lever.
2. Weights already pinned at exactly 0.0 by the old rule revive only via the rebound path (g_up ~ 1 at w ~ 0); test 2 must show it.
3. RESOLVED (a02-conditions (1), 2026-10-10): the |delta|=1.000 pin was the RECORDER's `floor_frac` (fraction of synapses at the bound), not a weight delta — `info["wt"] = {pathway: [mean, floor_frac]}` in the probe header; n(floor_frac==1.0)=32/400 (ctx, on_s8), n>0=346/400 matches the earlier "nonzero delta" count exactly. Mean moves freely (|delta mean| max 3.452). The rule has NO +/-1 clamp: sim/bridge.py 4c adds `weight_updates` then clip-to-bounds. No recorder fix needed (documented behavior); the earlier wording above is corrected.
4. FACTS (a) caveat stands: `spk_tot` is whole-trial; the cut-arm striatal-drive read needs a time-resolved probe before claiming eligibility-window timing.
5. Softening also slows OFF-arm D1 ceiling hits (7 in route-off): check no OFF-arm regression in rerun 3.
6. Cost: one elementwise multiply per plastic synapse per step; `n = 1.0` compiles to a multiply. Speed is secondary.

## Needs Claude's approval

- [ ] CreditParams fields `soft_bounds: int = 0`, `soft_bound_n_up: float = 1.0`, `soft_bound_n_down: float = 1.0` (defaults OFF) and passthrough in `credit_extension` (lines ~140-150) + the `_ldc_credit_omission.py` copy.
- [ ] CoreConfig fields in `sim/config.py` C2 block (lines ~994-1005) and the update in `sim/bridge.py` "4c. C2" (the `weight_updates` assembly and the "Clip to bounds" region) gated on `reward_soft_bounds`. This is a `sim/` edit — the first in the ldc arc; the runner-side alternative (`cp_plasticity_rate_gain` re-derived per step from host weights) was rejected as a host-side wrapper, so sign-off is needed on the scope change.
- [ ] The byte-identical-off test (default 0 must be bitwise-identical; seed-7 artifacts as reference).
- [ ] The bar-sanity spec + committed OK record as a precondition for gate use.
- [ ] Exponents frozen at 1.0 (proposed) vs a dev-seed-only sweep before freezing.
- [ ] Whether this rides in A1 with the omission route or a separate A2, and the exact capability wording the gate prereg may carry (soft bounds alone are an instrument fix; the capability claim needs the six-seed ablation).

---
Drafted by one MiMo v2.6 Pro delegate (RULING models) on 2026-10-10, from the (a)/(b)/(c) results of RULING ldc-d2-floor; rationale quoted in `SPIKING-WORKLOG.md`.