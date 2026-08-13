---
type: finding
status: qualified
claim_check: synthesis
verdict: >
  GO (6-seed, 6/6). The DEEPEST mouth rung — the WKV RECURRENT leaky STATE integrator (step 2 of the mouth,
  ap,an = decay*ap+relu(v); the remaining Qwen/BPTT-core dependency the two graded reads consume) — is held ON THE
  ACTUAL Izhikevich bridge as a GRADED SLOW recurrent-NMDA CONDUCTANCE, with the per-token input delivered as a
  GRADED SYNAPTIC DRIVE (a driven carrier population, read LINEARLY off cp_conductance_g_nmda_recurrent) rather than
  the spike-count-of-a-saturating-plateau the July on-bridge arc was bounded by. The substrate recurrent state
  reconstructs the host WKV leaky state at state_corr 0.793 mean / 0.777 min (6/6) — DECISIVELY above the July
  on-bridge caps (self-NMDA firing-integral 0.55, dendritic-plateau full-port 0.67), the point-neuron rate-code wall
  the whole July arc characterised. Recurrence is load-bearing (memoryless-reset collapses state_corr 0.793 -> 0.450,
  -0.343, 6/6). Fed through the host Wo_sp/head read the substrate state REPRODUCES the host mouth's next-word decision
  0.797 mean (vs zero-input 0.086, scramble 0.071 — both anti-cheats collapse) and its DEEP-context decision 0.732
  mean; the substrate-state deep-NLL sits AT the exact-host-state ceiling (mean gap -0.04 nats). The carrier->state
  wiring is load-bearing (lesion freezes the state exactly, move 0.000 6/6). The operating point (long per-token
  window t_step=40 + graded drive) is the July arc's OWN named un-tried candidate ("read the GRADED SYNAPTIC DRIVE
  rather than the per-window spike count") realised with the 2026-08-13 graded-conductance-read instrument that
  post-dates it. Calibrated ONCE on seed 42, 5 UNSEEN seeds pass. NOT "fully spiking / mouth complete": the WEIGHTS
  (Wv, the scalar decay, Wo_sp, head) are still BPTT-trained (a tracked scaffold; the transport-free LOCAL rule for
  the diagonal store is the separate 2026-08-12 GO), and rate v=Wv@LN(emb) is host-computed to isolate the STATE
  realisation. A named-residual moved onto the substrate; a real partial step, honestly bounded. Runner-only,
  default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the WKV recurrent STATE integrator)
date: 2026-08-13
mechanism: >
  GRADED SLOW-CONDUCTANCE recurrent integrator. Two regions on the real bridge: carr (F*Cp carriers) and state (F=2D
  subthreshold channels, F=[ap(D),an(D)] the dual-nonneg code). Block-diagonal carr[c]--nmda_slow-->state[c]
  (exc_receptor="nmda_slow", enable_nmda_recurrent) so the state channel's slow recurrent-NMDA conductance
  cp_conductance_g_nmda_recurrent = a clean dual-exp leaky integral of its carrier's firing (NO saturating plateau
  sigmoid, NO firing-integral self-mismatch). Per token: rate[c]=relu(+v)|relu(-v); drive carr[c]=bias+gain*rate[c]
  over t_step=40 steps WITHOUT resetting the conductance -> g_state = decay_step*g + strength*carr_spikes, matched so
  decay_step**t_step == the checkpoint's SSM decay -> the leaky integral of relu(v). Read g_state LINEARLY (graded
  conductance, 0 host draws on the read path). A per-channel affine (scale/offset) fit ONCE on seed 42 maps g_state
  to the reference scale (the labelled-line read-out; the same one-time calibration the graded reads use), FIXED for
  the 5 unseen seeds. The long window is load-bearing: it lets the slow conductance accumulate a decay-weighted graded
  charge that tracks relu(v) (t_step=8 gives input_lin 0.02 = the July wall; t_step>=30 gives 0.7-0.85). Speed is
  secondary, so long windows are in scope. cfg.seed-controlled substrate; NO sim/ edit; runner-only, default-off.
artifacts:
  - research/runners/_wkv_graded_recurrent_state_derisk.py
  - research/findings/raw/_wkv_graded_recurrent_state_6seed.json
  - research/findings/raw/_wkv_graded_recurrent_state_smoke.json
---

# gap#1 / A1 — the DEEP frontier: the WKV recurrent STATE integrator held as a GRADED SLOW-CONDUCTANCE leaky integrator on the substrate, input delivered as graded synaptic drive (GO, 6/6)

## The lever, and why it is the tractable next rung — grounded in the arc's record

The mouth's per-token chain, with the substrate boundary as of 2026-08-13:

    (1) v      = Wv @ LN(emb[tid])                     # input projection      (host, BPTT weights)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)   # WKV leaky STATE       <<< THIS RUNNER realizes step (2)
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))            # receptance gate       (host)
    (4) h      = r_h * (Wo_sp @ [ap,an])               # OUTPUT PROJECTION     (SUBSTRATE graded read — projection-GO)
    (5) logits = head_w @ h + head_b                   # read-out              (SUBSTRATE graded read — read-GO)

Steps (4)+(5) are substrate graded-conductance reads at parity (`2026-08-13-fluid-mouth-graded-conductance-read-GO`,
`...-upstream-output-projection-GO`). Their honest residual named step (2) — the RECURRENT leaky integrator with the
BPTT-trained decay — as the remaining Qwen/BPTT-core dependency (the projection-GO's next-rung #3, "the deep frontier
… the gap#1<->gap#4 meeting"). This runner attacks it.

**Why this is NON-REDUNDANT with the exhaustive July on-bridge arc** (`2026-07-19-gap1-WKV-...-RUNG1a`). July realised
the leaky state via (a) a self-NMDA autapse (firing-integral mismatch -> corr ~0.55) and (b) a dendritic GRADED
PLATEAU (corr 0.98 for a CLEAN dense value, but the full multi-channel port capped ~0.67 and the deep-NLL was
NEGATIVE). Its precisely-characterised bound was the INPUT DELIVERY: the plateau's coincidence drive is carried by the
input-pool FIRING — a threshold/refractory/dead-zone NON-MONOTONE spike-count map of relu(v) — fed through the
plateau's SATURATING SIGMOID. July NAMED the un-tried fix verbatim (its line 391): *"make c_w read the GRADED SYNAPTIC
DRIVE (the smooth postsynaptic conductance the inp firing produces through the coincidence synapse) rather than the
per-window spike count."* The `2026-08-13` graded-conductance-domain read (reconstruct a projection off the net
synaptic conductance at corr 0.98) is EXACTLY that graded-synaptic-drive instrument, and it did NOT exist during the
July arc. This runner applies it to the RECURRENT integrator: the leaky state lives in the substrate's slow
recurrent-NMDA CONDUCTANCE (a clean dual-exp leaky integral of presynaptic firing — no plateau sigmoid, no
firing-integral self-mismatch), the per-token input is delivered by a driven carrier population, and the state is READ
as the graded conductance, LINEAR, never a spike count.

**The separate LEARNING-RULE question is already resolved** (`2026-08-12-gap1-A1-deep-context-credit-on-diagonal-WKV-store-local-rule`):
on the DIAGONAL WKV store a transport-free local e-prop rule ties BPTT at adequate capacity — no deep-context
credit-quality wall. This runner is orthogonal: it is the STATE REALISATION (holding + updating the recurrent leaky
state on the substrate), not the rule that trains it.

## The mechanism — the substrate's own slow recurrent-NMDA conductance IS the leaky integrator

The slow recurrent-NMDA conductance `cp_conductance_g_nmda_recurrent` (enabled by `enable_nmda_recurrent`, fed only by
`exc_receptor="nmda_slow"` synapses) is incremented per step by the routed presynaptic firing and decays with
`nmda_recurrent_tau_decay_ms` — a clean dual-exp leaky integral, read as a CONDUCTANCE (the Mg2+ block gates only the
CURRENT, so the conductance read is a faithful graded integral). Wire carr[c] --nmda_slow--> state[c] block-diagonal;
drive carr[c] = bias + gain*relu(v)[c] over a long per-token window (t_step=40, no reset) with tau matched so the
per-token decay equals the checkpoint's SSM decay. The state channel's conductance is then the leaky integral of the
carrier firing = the leaky integral of relu(v) = the WKV state ap/an. Read it graded. This is the graded analog read a
distributed code affords (Mikulasch-Priesemann; the 2026-06-20 graded-plateau template; the read-GO/projection-GO
results), realised for the RECURRENCE. NO `sim/` edit — drives + reads public bridge arrays; cfg.seed-controlled.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; D=128; F=256; t_step=40; carrier_pop=24; tau_rec≈144 ms; GPU)

<!--derived-->

| seed | state_corr | (worst-sent min) | mless | input_lin | argmax vs host | (zero / scramble) | deep_argmax (n) | deep-NLL gap→host | lesion_move | GO |
|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.7928 | 0.686 | 0.4435 | 0.583 | 0.7928 | 0.100 / 0.076 | 0.529 (17) | -0.102 | 0.000 | ✓ |
| 43  | 0.7965 | 0.710 | 0.4569 | 0.587 | 0.8261 | 0.092 / 0.047 | 0.636 (11) | +0.073 | 0.000 | ✓ |
| 44  | 0.7978 | 0.656 | 0.4671 | 0.587 | 0.7968 | 0.125 / 0.064 | 0.850 (20) | -0.105 | 0.000 | ✓ |
| 100 | 0.7915 | 0.578 | 0.4233 | 0.564 | 0.8320 | 0.067 / 0.080 | 0.765 (17) | +0.094 | 0.000 | ✓ |
| 101 | 0.7766 | 0.622 | 0.4329 | 0.548 | 0.7795 | 0.058 / 0.039 | 1.000 (16) | -0.246 | 0.000 | ✓ |
| 102 | 0.8021 | 0.716 | 0.4763 | 0.599 | 0.7549 | 0.075 / 0.121 | 0.609 (23) | +0.024 | 0.000 | ✓ |
| **mean** | **0.7929** (min 0.7766) | 0.661 | **0.4500** | 0.578 | **0.7970** | **0.086 / 0.071** | **0.732** | **-0.044** | **0.000** | **6/6** |

**Headline.** The substrate recurrent state reconstructs the host WKV leaky state at **state_corr 0.793** (6/6, min-seed
0.777) — decisively above the July on-bridge caps (self-NMDA 0.55, plateau full-port 0.67), the point-neuron rate-code
wall the whole July arc mapped. The CORE probe (clean synthetic input, isolating the state realisation) reads 0.902 at
this operating point; the drop to 0.793 on the real corpus is the sparse-signed relu(v) input delivery (input_lin
0.578; ~49% of channels are 0 each token — the per-token dead-zone the leaky integration then smooths away). **Recurrence
is load-bearing** (memoryless-reset collapses 0.793 -> 0.450, -0.343, 6/6). Fed through the exact host Wo_sp/head read,
the substrate state **reproduces the host mouth's next-word decision 0.797** (zero-input 0.086, scramble 0.071 — both
collapse) and its **DEEP-context decision 0.732** (small per-seed deep_n 11-23); the substrate-state deep-NLL sits **at
the exact-host ceiling** (mean gap -0.04 nats, range -0.25..+0.09 across the small deep samples). The carrier->state
wiring is load-bearing (lesion freezes the state exactly, move 0.000 6/6).

## The operating-point lever (calibrated ONCE on seed 42; a WIDE plateau; NOT overfit)

A pre-registered operating-point probe on seed 42 established the load-bearing lever: the **long per-token window**. At
t_step=8 the per-token charge does not track relu(v) at all (charge-vs-rate corr 0.017; charge_at_rate0 ≈ charge_at_
rate_hi — the July input-pool wall); at t_step>=30 the slow conductance accumulates a decay-weighted graded charge that
tracks relu(v) (input_lin 0.70-0.85), and the recurrent leaky integration lifts that to core state_corr 0.87-0.90. The
gain/bias/carrier-pop sit on a wide plateau. The single fixed operating point + the per-channel affine calibration are
fit ONCE on seed 42; seeds 43/44/100/101/102 are the UNSEEN generalisation test and pass 6/6 — not a per-seed-tuned
result.

## Anti-cheats (all 6 seeds)

- **Memoryless collapse (recurrence load-bearing):** resetting the conductance every token (no cross-token persistence)
  drops state_corr 0.793 -> 0.450 — the substrate is genuinely INTEGRATING across tokens, not re-reading a per-token
  transient.
- **Zero-input collapse (cache-immune):** driving the carriers with a ZERO rate drops the downstream host-decision
  agreement 0.797 -> 0.086 — the input drives the state; not a floor/frequency artifact.
- **Scramble:** a post-hoc state[c]->channel relabel drops the downstream agreement to 0.071 — the labelled-line
  channel map carries the signal.
- **Carrier lesion:** zeroing the carr->state weights freezes the state (it only decays), move 0.000 on all 6 — the
  wiring drives the integrator (a clean lesion here, unlike the read-side weight-lesion the graded reads documented as
  a no-op).
- **Provenance:** state read from `cp_conductance_g_nmda_recurrent`; 0 host draws on the read path.

## External grounding

According to PubMed, Wang XJ (1999), *J Neurosci* 19(21):9587-603
([DOI](https://doi.org/10.1523/JNEUROSCI.19-21-09587.1999)): sustaining persistent-activity (working-memory) integrator
state requires recurrent excitation **dominated by a SLOW component** — *"to achieve a stable persistent state,
recurrent excitatory synapses must be dominated by a slow component … slow NMDA receptor-mediated synaptic transmission
is likely required for sustaining persistent network activity."* This is the biological basis for holding the WKV leaky
integrator state in the substrate's slow recurrent-NMDA conductance rather than a spike rate. It composes with the
graded-read grounding (Holt & Koch 1997, the subthreshold membrane current at rest is a faithful LINEAR combination —
why the graded conductance read is faithful) and the SpikeGPT reframe the July arc surfaced (Zhu et al. 2023 keep the
WKV state graded in FP32; "state = mean firing RATE" is stricter than SOTA and than biology, which holds integrator
state in graded slow conductances) — the reframe that makes this graded-conductance state a legitimate spike-based
substrate mechanism, not a shortcut.

## Honest residuals — what moved onto the substrate, and what still leans on BPTT/Qwen

1. **ON the substrate now:** the WKV RECURRENT leaky STATE INTEGRATION (held in the slow recurrent-NMDA conductance,
   persisting + decaying across tokens on the real Izhikevich bridge), the per-token INPUT DELIVERY (graded synaptic
   drive from a carrier population), and the STATE READ (linear graded conductance) — all neurons + synapses + graded
   conductances. This is the named residual #2/#3 of the two graded-read findings ("the WKV store is BPTT-trained …
   the leaky ap/an integrator … the deep frontier") moved onto the substrate at state_corr 0.79, reproducing the host
   mouth's decisions at ~0.80 (deep 0.73) with the deep-NLL at the host ceiling.
2. **state_corr 0.79, not 1.0:** the input delivery is imperfect on the SPARSE signed relu(v) (input_lin 0.58 — a
   dead-zone for small positive values, where the carrier is sub-threshold). The leaky integration smooths it to 0.79
   but not to the 0.90 clean-input core. Named next lever: a MATCHED-PAIR common-mode-subtracting carrier (the
   projection-GO's signed E/I read, driven-current + bias then cancel) or a staggered-threshold carrier population, so
   even small relu(v) fires linearly — lift the dead-zone toward the 0.90 core.
3. **STILL host / BPTT:** the WEIGHTS — the input projection Wv, the scalar decay VALUE, Wo_sp, head — are BPTT-trained
   (a tracked scaffold; the transport-free LOCAL rule for the diagonal store is the separate 2026-08-12 GO). The rate
   v = Wv@LN(emb) is computed host-side HERE to isolate the STATE realisation (its own substrate read is the
   projection/read GOs; the two compose but the end-to-end host-free chain was not run in this rung). The read-out was
   applied host-side to isolate the state metric.
4. **NOT "fully spiking" / NOT "the mouth works":** a named residual (the recurrent STATE) moved onto the substrate as
   a graded conductance integrator — a real partial step on the deepest gap, honestly bounded. Functional read-outs
   only; no claim of phenomenal experience.

## Named next rungs (in tractability order)

1. **Lift the input-delivery dead-zone** (residual #2): a matched-pair common-mode-subtracting carrier (projection-GO
   signed read) so small relu(v) delivers linearly — target state_corr -> the 0.90 clean-input core, and deep-argmax
   agreement up.
2. **Compose the whole host-free chain end-to-end:** feed the substrate state (this rung) into the graded Wo_sp@state
   (projection-GO) into the graded head_w@h (read-GO), so `head_w@(Wo_sp@state)` from a SUBSTRATE recurrent state is
   entirely graded-conductance, leaving only v=Wv@emb + the r_h gate host. A wiring composition of three validated
   pieces.
3. **The scalar decay from the biophysical tau + a local rule for Wv** (the gap#1<->gap#4 meeting): the decay is
   already matched from `nmda_recurrent_tau_decay_ms` here (a biophysical tau, not a BPTT parameter); the remaining
   BPTT residual is the input projection Wv, for which the 2026-08-12 transport-free diagonal e-prop is the rule.

## Files
- Runner: `research/runners/_wkv_graded_recurrent_state_derisk.py`
- Raw: `research/findings/raw/_wkv_graded_recurrent_state_6seed.json` (+ `_smoke.json`)
- Builds on: `2026-08-13-fluid-mouth-graded-conductance-read-GO.md` + `...-upstream-output-projection-GO.md` (the
  graded-conductance-domain read instrument this applies to the recurrence), `2026-07-19-gap1-WKV-...-RUNG1a-...md`
  (the state-integrator frontier + the July input-delivery bound this breaks), `2026-08-12-gap1-A1-deep-context-credit-
  on-diagonal-WKV-store-local-rule-...md` (the orthogonal transport-free LEARNING rule).
