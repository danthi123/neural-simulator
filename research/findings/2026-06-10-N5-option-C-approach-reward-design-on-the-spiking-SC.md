# N5 Option C — the neural approach-reward, building on the now-committed spiking superior colliculus

**Date:** 2026-06-10
**Type:** Build design (the last remaining navigation host-computation). Grounded in the deep-research GO (`2026-06-10-N1-N5-spiking-superior-colliculus-research.md` §Option C) + the de-risk (`2026-06-10-N1-N5-spiking-SC-derisk-RESULT.md`, which showed the SC bump *carries* approach but a static foveal read is SNR-limited).
**Prerequisite (now satisfied):** the spiking superior colliculus is built into the nav runner (`--enable-spiking-sc`: `sc_retina` egocentric eye → `sc_map` retinotopic sheet + Mexican-hat, forming a bump at the goal's retinal site each nav step). N5 Option C *reuses that bump*.

## What N5 still is (the host shortcut to replace)

The reward *value* is host arithmetic: `reward = sign(eccentricity_after − eccentricity_before)` via `sc_salience_offset_from_image` (`g11_bg_runner.py:6611–6635`) — the bar's named cheat "a reward computed by a distance formula." The spiking `reward_us` population only *delivers* this scalar to the SNc; the *content* ("did the goal get closer?") is host. Option C makes the content neural.

## The mechanism (slow-channel temporal-difference of the SC bump's rostral-ward motion)

The goal getting closer = its egocentric retinal eccentricity decreasing = the SC bump moving toward the foveal/rostral pole (the map centre). The de-risk confirmed this directly (a closer goal puts the bump nearer centre). Option C reads the *temporal change* of rostral activity in spikes — robust where the static foveal read (de-risk: 3–6/8) was SNR-limited, because in the nav the bump moves *continuously* as the agent steps.

**Regions/pathways to add (runner-side, gated by a new `enable_spiking_sc_approach`; the slow channels already exist + are runner-enabled — `RegionPathway.exc_receptor="nmda_slow"`, `receptor="gaba_b"`, verified in `sim/regions.py`):**

1. **`sc_rostral`** (~24 RS): pools the *central* `sc_map` sites with a Gaussian weight (the de-risk's foveal pool). Fires graded with how central the bump is (= how small the goal's eccentricity is). `sc_map → sc_rostral` explicit Gaussian pooling (reuse the de-risk's `wA`).
2. **`sc_rostral_slow`** (~24 RS): a *lagged* copy of `sc_rostral` via `sc_rostral → sc_rostral_slow` with `exc_receptor="nmda_slow"` (tau ~100 ms) — represents the *previous-frame* rostral activity (a neural memory trace).
3. **`approach`** (~24 RS): driven by `sc_rostral` (excitatory, AMPA) **minus** `sc_rostral_slow` (via `receptor="gaba_b"` GIRK inhibition). Fires when rostral activity *rose* vs its lagged trace = the bump moved toward centre = **the goal got closer**. This is the neural temporal-difference (Option C).
4. **`approach → reward_us`** (excitatory): the `approach` firing gates the reward burst, **replacing** the host `reward_us_drive_pa * max(0, reward)` write. The whole reward *value* is now `approach`-pool firing (neural), delivered by `reward_us` into the spiking SNc, where the striosome critic subtracts V (the already-spiking N9 δ = r − V loop).

## Nav-loop integration

When `enable_spiking_sc_approach`: the `sc_map` bump already updates each step (from `sc_retina`); `sc_rostral`/`sc_rostral_slow`/`approach` settle through their synapses; read `approach` firing over the readout window → drive `reward_us` proportional to it (replacing the host reward write at `g11_bg_runner.py:6611–6635` + the `reward_us` drive). Requires `--perceived-approach-reward` OFF (Option C *is* the perceived approach, now neural) — or compose so the host reward is dropped.

## Anti-cheats (must hold)

1. **Image-only:** `approach` is driven *solely* by the SC bump (which reads only the egocentric render); assert no `(x,y)`/`(gx,gy)`/Manhattan/`sc_salience_offset` enters the `reward_us` drive.
2. **Behavioural equivalence:** on a held-out trajectory, `sign(approach firing this step − baseline)` agrees with the host `sign(Δ ecc)` on ≥ 7/8 steps (the N5 label-agreement bar) — *measured in the nav* (continuous motion), where the de-risk's static probe was SNR-limited.
3. **Scrambled-retinotopy lesion** (`SC_SCRAMBLE=1`): the approach signal must break (the bump goes to wrong sites → no coherent rostral-ward motion) → nav reward becomes uninformative.
4. **`approach → reward_us` relay lesion:** zero that pathway → the reward must vanish (proves it is carried by synaptic transmission, not host arithmetic).

## De-risk-first (cheapest)

Before the full nav build, extend `sc_map_orienting_probe.py` with a **continuous-trajectory** mode: drive `sc_retina` with a sequence of egocentric renders (agent stepping toward/away from a fixed goal, NO reset between steps), wire `sc_rostral`/`sc_rostral_slow`/`approach`, and check the `approach` firing's `sign(now − prev)` agrees with the host reward sign ≥ 7/8 *across the continuous trajectory* (the regime the static probe couldn't test). If it passes → build into nav; if the TD is still noisy → tune the `nmda_slow` tau / the gaba_b strength / a dead-band, or accept N5 as the coordinate-free perceived-approach scaffold (an honest residual the owner can gate on).

## Honest framing

This gridworld's goal does NOT expand, so "approach" is foveation/eccentricity-decrease (the SC rostral-shift), **not** looming — Option C (rostral-ward TD) is the faithful model; a looming detector (Option D) is only right if the render is later changed to scale the goal with proximity. With Option C, the *entire* navigation δ = r − V loop is synaptic: **r** = `approach` (neural TD of the SC bump) → `reward_us` → SNc; **V** = the striosome critic's GABA_B subtraction. That would close N5 (the last host nav-computation) and, with N1 (the spiking SC orienting) + N6 (spiking commit) + N9 (spiking SNc RPE) + N2/N7 (defensible perception), make navigation fully brain-based by the strict bar — then the unification gate genuinely opens.
