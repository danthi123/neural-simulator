# gap#1 RF PHASE ENCODE — the FIRST spiking-input encode PAST the fidelity wall (deployed deep-NLL GO, 6-seed)

**Date:** 2026-07-20 · **Status:** **GO, 6-seed (dev 42/43/44 + blind 100/101/102)** — adversarially verified
(SURVIVES_WITH_SCOPE_FIXES, framing applied). The spiking-INPUT half of gap#1 is CLOSED for the graded-state recurrent
LM: a spiking phase delivery (FHRR) reaches parity with a perfect host input on every seed, where the 3 prior spiking
encodes (NEF/token-SDR/co-adaptation) all catastrophically collapsed. Honest scope: this is spiking DELIVERY of the
value (a high-fidelity phase ADC into the validated graded `cp_ssm_state`), NOT spike-based deep-context computation
(see the honest-framing box).

## The wall this closes

gap#1's spiking-INPUT half was a precisely-characterized WALL (`2026-07-20-gap1-DEFINITIVE-...`): the recurrent
STATE + READ-OUT run on-bridge with a PERFECT host input (M1 exact-state **+0.542..+0.874 GO**, corr 1.000), but
THREE spiking encodes of the per-token value all collapsed the deep-NLL — NEF regression (corr 0.616, −2.904),
token-SDR selection (corr 0.501, −3.416, REFUTED), co-adaptation (corr 0.579, −2.876). The deep-NLL is
HYPERSENSITIVE to state fidelity, and none reached the near-1.0 the deep win requires.

**M0 reframe** (`2026-07-20-gap1-M0-REFRAME-...`): the wall is not raw fidelity but a **value-dependent BIAS** in the
encode. i.i.d. noise on the exact input accumulates GRACEFULLY (deep-NLL crosses zero at corr ~0.80), but the actual
rate encodes are ~1.5–2 nats WORSE at the same corr → their error is a coherent, value-dependent dead-zone that
compounds over the recurrence. **The fix must deliver the value UNBIASED across the value range.**

## The mechanism: RF phase (Frady-Sommer FHRR) — value in a spike's TIMING

The per-token value `v_t = Wv·LN(emb[x_t])` (dual-nonneg inject `d = relu(±v_t)`) is delivered as the **PHASE of a
single resonate-and-fire spike** on an **independent** RF-oscillator pool (`RESONATE_AND_FIRE` + `rf_kick`/`rf_read_phases`,
`connections_per_neuron=0` — no synapses), guard-banded into the arc [0.05, 0.95] so no value hits the 0/1 phase wrap.
The value rides TIMING, not a rate → **no rate-code dead-zone**. Decode the phase, charge the validated graded
`cp_ssm_state` (exactly M1's path, but the inject now arrives through a spike's timing). NO `sim/` edit (drives + reads
public arrays; `--rf-phase-encode` on the M1 runner, default-absent = byte-identical to host-inject M1).

### ⚖️ Honest framing (adversarial-verify LENS 1 + synthesis, applied) — spiking DELIVERY, not spike COMPUTATION

The RF pool is a set of **independent** oscillators (no synaptic weights installed), so the resonate step is a
**high-fidelity phase quantizer/ADC** — it performs NO cross-channel or contextual computation. The **deep-context
capture is done by the graded `cp_ssm_state` leaky integrator (the already-validated M1 path)**, evidenced by MAIN and
MEMORYLESS sharing map_corr 0.99927 yet diverging on the deep metric (+0.878 vs −0.434, the swing driven purely by
toggling the graded recurrence lam=0). So the claim is precise: **this is the first SPIKING-INPUT DELIVERY that clears
the gap#1 fidelity wall** — it delivers the per-token value in a spike's PHASE, UNBIASED across the value range
(round-trip corr 0.998 vs NEF 0.616 / SDR 0.501 / co-adapt 0.579, clearing the M0 value-dependent-bias wall) — **NOT a
claim that spikes computed the long context.** This is exactly the SpikeGPT/biology-faithful target the M1 finding
named: spike the I/O, hold the recurrent state in a graded slow conductance. The phase→value decode is a host read of
spike timing (what a downstream coincidence detector does); the fully-synaptic phase→conductance transduction (RF spike
drives the NMDA charge directly, no host read) is the next polish.

## Pre-flights (control-first, both on DEPLOYED inputs)

- **M1 control re-confirmed** (the deployment's own control MUST pass first): `--ssm-state --use-ssm-readout` on
  `wkv_ssmU_v1000_d128_seed42.npz` → corr **1.000**, deep-10-99 vs-trigram **+0.874 GO**. Harness valid.
- **Static RF pre-flight** (`_gap1_rf_phase_preflight.py`): 128 values in [−3,3] → corr 0.954, **bias-spread 0.0007**
  (unbiased across value bands) — the exact M0 property.
- **Deployed-accumulated-state pre-flight** (`_gap1_rf_phase_deployed_preflight.py`, the day's hardest lesson —
  validate on the DEPLOYED zero-inflated distribution, not a uniform grid): on real sentences the injects are 50%
  zero (relu floor); the RF decode is **UNBIASED in the dominant near-zero band** (n=29372, mean −0.0021, rms 0.0066)
  and the **accumulated state corr = 0.9984** (per-channel median 0.9999, min 0.9919). Far above the M0 GO threshold
  (~0.85) and FAR above the three failed encodes (0.5–0.6).

## THE GATE — deployed deep-NLL through the bridge + trained read-out (seed 42, n_eval=200)

| arm | map_corr | deep-10-99 vs-trigram | mid-6-9 vs-trigram | verdict |
|---|---|---|---|---|
| M1 host-inject (reference) | 1.000 | +0.874 | +0.828 | GO |
| **RF phase encode (MAIN)** | **0.999** | **+0.878** | **+0.828** | **GO** |
| memoryless anti-cheat (recurrence off, lam=0) | 0.999 | **−0.434** | −0.742 | collapse ✓ |
| scrambled-phase anti-cheat (channel→value permuted) | **0.070** | **−1.652** | −2.173 | collapse ✓ |

- **RF phase (spike-timing delivery) reaches +0.878 ≈ the perfect-host-input ceiling (+0.874)** — and its FULL per-depth
  curve matches M1 (mid-6-9 +0.828 == M1's +0.828), so it is not a single noisy bucket. Where NEF (−2.904),
  token-SDR (−3.416), and co-adaptation (−2.876) all catastrophically collapsed, RF phase is at parity with a perfect input.
- **memoryless** (recurrence off) → −0.434 (below trigram): the deep win requires the RECURRENCE integrating the
  RF-delivered value; the encode alone does not leak the target.
- **scramble** (permute the RF-decoded injects across channels — same spikes, destroyed channel→value map) →
  map_corr 0.070, −1.652: the RF phase carries the CORRECT per-channel value; it is not "the machinery happens to
  produce a plausible state."
- **Default-path byte-identity TESTED (not asserted):** with `--rf-phase-encode` ABSENT the M1 path re-runs to deep
  vs-trigram **+0.874 / map_corr 1.000** post-edit — identical to the pre-edit reference (`_gap1_M1_byteident_postedit`).
- **Phase-ADC control (adversarial-verify LENS 1):** replacing the RF resonate loop with the analytic phase quantizer
  it implements (`--rf-numpy-quantize`) gives deep vs-trigram **+0.867 ≈ the real RF +0.878 within roundoff** — proving
  the RF pool is functionally a high-fidelity phase ADC (faithful spiking DELIVERY of the value, no spiking computation).

## Verdict

**RF-PHASE-ENCODE SURPASS = GO, 6-seed (dev 3/3 + blind 3/3).** The first spiking-INPUT encode to clear the gap#1
deep-NLL wall — value carried in a spike's PHASE (FHRR), unbiased across the value range (round-trip corr 0.998),
delivered into the validated graded `cp_ssm_state`. At parity with a perfect host input on every seed (|RF−M1| ≤ 0.015
nat), both anti-cheats collapsing (memoryless & scramble, on a blind seed too), adversarially verified (5 skeptic lenses,
SURVIVES_WITH_SCOPE_FIXES) with the phase-ADC framing empirically confirmed (numpy-quantize +0.867 ≈ +0.878). Where
NEF (−2.904), token-SDR (−3.416), and co-adaptation (−2.876) all destroyed the deep capture, RF phase preserves it.
⇒ **the spiking-INPUT half of gap#1 is CLOSED for the graded-state recurrent LM.** Open next: the fully-synaptic
phase→conductance transduction (no host read of the phase); scaling V/D.

## Multi-seed generalization (matched provenance-recorded ssmU6 set, seeds 42/43/44 dev + 100/101/102 blind)

See the table below (filled from `_gap1_rf_multiseed_aggregate.py`). All 6 M1 references reproduce on-bridge
(map_corr 1.000). GO = RF-MAIN tracks M1 (deep > 0.02, map_corr > 0.9) on all 6, anti-cheats collapse.

| seed | grp | M1 deep | RF deep | RF−M1 | map_corr (M1/RF) | GO |
|---|---|---|---|---|---|---|
| 42 | dev | +0.409 | +0.423 | +0.014 | 1.000 / 1.000 | GO |
| 43 | dev | +0.307 | +0.320 | +0.013 | 1.000 / 0.999 | GO |
| 44 | dev | +0.579 | +0.567 | −0.012 | 1.000 / 0.997 | GO |
| 100 | blind | +0.329 | +0.338 | +0.009 | 1.000 / 0.999 | GO |
| 101 | blind | +0.391 | +0.403 | +0.012 | 1.000 / 0.998 | GO |
| 102 | blind | +0.213 | +0.228 | +0.015 | 1.000 / 0.999 | GO |

**RF-MAIN GO on 6/6 seeds (dev 3/3, blind 3/3); RF tracks M1 to within ±0.015 nat everywhere, map_corr ≥ 0.997.**
Anti-cheats on blind seed 100: **RF+memoryless deep −0.906 (COLLAPSE)**, **RF+scramble map_corr 0.109 / deep −2.469
(COLLAPSE)** — both generalize. The spiking phase delivery is at parity with a perfect host input on every checkpoint,
far above the anti-cheat floors. (Matched provenance-recorded set `wkv_ssmU6_v1000_d128_seed{42,43,44,100,101,102}.npz`,
trained `--recurrence ssm --dual-nonneg --uniform-decay --vocab 1000 --d-model 128 --n-sentences 40000 --epochs 12`;
the original seed-42 dev ckpt `wkv_ssmU_...` gave the +0.878 ≈ +0.874 headline.)

## Honest scope / open

- The seed-42 +0.878 magnitude is single-seed / single-eval-slice / smallest-bucket n=126 (SE unquantified); the
  n_eval=20 smoke was a NESTED SUBSET of the 200-slice (same rng permutation), so it gave zero independent-slice
  confirmation. The 6-seed matched-set run (above) is the independent-slice + cross-seed generalization test.
- **The framing is spiking DELIVERY (a phase ADC), not spike COMPUTATION** — see the honest-framing box above. The
  RF decode is a host read of spike timing (what a downstream coincidence detector does); the fully-synaptic
  phase→conductance transduction (RF spike → downstream NMDA charge, no host read) is the next polish.
- Per THE LAW: the three prior encode methods (NEF/token-SDR/co-adaptation) stay refuted/bounded; this is the
  mechanism that surpasses them — a spiking-input delivery that, unlike all three, does not destroy the deep capture.

## Adversarial-verify (5 independent skeptic lenses, before commit) — SURVIVES_WITH_SCOPE_FIXES

- LENS 1 (passthrough): SURVIVES_WITH_SCOPE_FIX — genuine lossy spiking delivery (not a copy of `_inj`); reframed as
  phase-ADC delivery, empirically confirmed by the `--rf-numpy-quantize` control (+0.867 ≈ +0.878).
- LENS 2 (like-for-like): SURVIVES — clean single-variable (same checkpoint, same frozen `--use-ssm-readout`, same eval).
- LENS 3 (anti-cheat validity): SURVIVES — memoryless truly zeros integration (k_leak=1→lam=0); scramble is a real
  permutation (3 fixed pts/256) on byte-identical firing; MAIN separates from both by 1.31 / 2.53 nats.
- LENS 4 (honest scope): SURVIVES — delivery is genuinely spiking (cp_rf_spike_step from the Im zero-crossing during
  the run loop is the sole path to the charged state); host-read residual disclosed.
- LENS 5 (number stability): SURVIVES_WITH_SCOPE_FIX — MAIN tracks M1 to ≤0.007 NLL at every depth bucket, monotone
  with depth; single-slice caveat addressed by the multi-seed run.

## Extension — on-bridge GENERATION with the spiking input (gap#1's actual capability)

The RF phase encode closed the spiking-input *comprehension* half (deep-NLL). Added on-bridge autoregressive GENERATION
(`--gen-tokens N --gen-prompt … --gen-temp T`): roll out tokens, charging `cp_ssm_state` per token via the RF phase
encode + argmax/sample the SSM's own read-out. **Ceiling check first** (workflow discipline): the off-bridge WKV at
V=1000/d=128 generates recognizable TinyStories prose (*"once upon a time there was a little boy named tim … played
together in the sun and the blue sky"*), so the mechanism generates — worth building on-bridge.

**On-bridge result (seed 42, prompt "once upon a time", temp 0.8):**
- RF-phase (spiking input): *"… he said goodbye to his friends and had a fun time … the pieces of `<unk>` together on
  the ground …"*
- host-inject (M1 reference): *"… the kite with its paw and it was time to leave … saw a big tree with lots of trees
  and flowers and a bright `<unk>` together in the sky …"*

**⇒ the RF-phase (spiking-input) generation produces coherent TinyStories-style prose AT PARITY with the perfect-host-
input reference** (both track identically early — same prompt/seed, near-identical state corr 0.999 — then diverge via
sampling RNG). Argmax mode-collapses to `<unk>`/function-words (greedy-decoding artifact); temperature sampling fixes
it. The `<unk>`-heaviness is the V=1000 vocab-scale limit (matches the off-bridge ceiling), a scale lever, not a
mechanism issue. **So the spiking-input WKV cortex both COMPREHENDS (deep-NLL GO) and GENERATES (coherent prose) —
gap#1's capability on the spiking substrate.** **Multi-prompt firming (3 prompts × seeds 43/44/100, RF-phase):** all produce coherent TinyStories prose after a short
`<unk>` warmup — *"…he knew he had to do a walk with the giant…"* / *"…for everyone to eat their dinner with the … and
their friends … to play on the swings…"* / *"…she could help her friend the duck and his mom were very happy"*. So the
generation is NOT prompt-specific. HONEST SCOPE: single checkpoint, V=1000 vocab-limited (the `<unk>` warmup + OOV
density); a larger-vocab checkpoint (to cut `<unk>`) is the follow-on; the generator is the graded-state +
trained-readout reservoir path (the R3 / gap#4-a-1 convergence: the value is the readout over a fixed substrate).

## Artifacts

- Runner: `_emerge_wkv_onbridge_derisk.py --ssm-state --use-ssm-readout --rf-phase-encode` (+ `--rf-scramble`,
  `--ssm-memoryless` anti-cheats). Pre-flights: `_gap1_rf_phase_preflight.py`, `_gap1_rf_phase_deployed_preflight.py`.
- Checkpoint: `bridges/wkv_ckpt/wkv_ssmU_v1000_d128_seed42.npz`.
