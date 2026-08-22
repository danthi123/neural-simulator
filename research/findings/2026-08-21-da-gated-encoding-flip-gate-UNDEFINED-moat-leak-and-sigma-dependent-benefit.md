---
type: finding
status: contributing
date: 2026-08-21
mechanism: da-gated-encoding
lane: integration
integration_faculty: da-gated-encoding
seeds: [42, 43, 44, 100, 101, 102]
verdict: UNDEFINED
instrument: the CONTROL-ARM decomposition is OFF-arm (encoding-gate disabled, write gain pinned 1.0) vs ON-arm
  (DA-gated per-fact write gain, Latin-square DA — 3 high g=2.48 / 3 low g=0.5-floor / 3 tonic g=1, orthogonal
  to agent+action) recall at MATCHED read-damage sigma. The soak's per_seed sweep + preconditions block carry
  the ON-minus-OFF attribution per sigma (recall_off/recall_on, n_regressed/n_improved, regressed_idx and the
  per-arm moat flags per fact), so every recall count is decomposed to WHICH arm and WHICH fact produced it —
  a SIZE with its SOURCE, not a bare score. Derivation cross-checked byte-equal (max|dw|=0.0, 576 conns) to a
  real per-fact encoding_gain_fn build.
runner: research/runners/_da_encoding_leansoak.py
artifacts:
  - research/findings/raw/_da_encoding_leansoak/soak.json
external: reuses the in-repo validated I-7-b gain map (g = clip(0.5, 3.0, 1 + 2.0*(DA−0.5)), k_DA=2.0) and the
  #76/#79 spiking DA-mode read; biology anchor for the coupling is the Lisman–Grace hippocampal–VTA loop /
  Kandel D.16 (dopamine gates entry into long-term memory). Anchor for the next-lever companion process is
  Turrigiano 2008 ("The Self-Tuning Neuron", Cell 135(3):422-435) — homeostatic synaptic scaling to a firing
  set-point that PRESERVES relative synaptic strengths (Turrigiano & Nelson 2004 Nat Rev Neurosci 5:97-107;
  Turrigiano 2012 CSH Perspect Biol a005736). This external-literature round is logged (lane integration) in
  the external-searches record.
supersedes: none — the WIRE-IN GO
  (research/findings/2026-08-21-da-gated-encoding-wired-into-chat-GO.md, default-OFF) named this soak as its
  next rung; this finding records that soak's verdict.
---
# DA-gated-encoding default-ON flip gate is UNDEFINED — a moat leak plus a sigma-dependent benefit (flip HELD, faculty stays default-OFF)

## Verdict

**UNDEFINED / UNDEFINED — this is a NEGATIVE outcome for the default-ON flip (the flip is HELD; the faculty
stays default-OFF). Two preconditions are unmet, so the no-regression soak yields no interpretable GO/NO-GO.**

Artifact: `research/findings/raw/_da_encoding_leansoak/soak.json` (6-seed soak, seeds 42/43/44/100/101/102,
D=64, k_da=2.0, da_baseline=0.5, 9 facts, sigma grid 0/0.5/0.75/1.0/1.5/2.0/3.0/4.0/6.0). Backend cupy/GPU
(recorded in the co-located `soak.json.prov.json`: sim_backend=cupy). Reproduce:
`SIM_BACKEND=cupy python -m research.runners._da_encoding_leansoak`.

This is the flip gate the wire-in GO (`2026-08-21-da-gated-encoding-wired-into-chat-GO.md`) explicitly deferred
to: the lean production magnitude-store no-regression soak — does the self-produced-DA write gain regress recall
of already-stored facts when read under stress, over the realistic DA distribution.

## The instrument bites (the no-regression test is not vacuous)

`instrument_bites=true`. The read-damage sweep genuinely degrades recall, so a no-regression test has something
to measure. Aggregate OFF-arm recall (of 54 = 9 facts × 6 seeds) falls monotonically across the sigma grid:

| sigma | 0 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |
|------|---|-----|------|-----|-----|-----|-----|-----|-----|
| OFF recall (of 54) | 54 | 52 | 43 | 33 | 13 | 8 | 4 | 2 | 2 |

Calibrated knee sigma = 0.5. Clean read (sigma=0): zero facts regress OFF→ON on every seed
(`go_clean_zero_regression=true`, `clean_regressions_total=0`) — the dominant production case for a modest fact
store (a phase read is magnitude-invariant), so the coupling is harmless where it will most often run.

## Unmet precondition 1 — the MOAT leaks (moat_fail_total=2)

The moat precondition ("an unstored cue abstains on BOTH arms at every sigma — encoding never manufactures a
fact") fails at 2 of the 54 (seed, sigma) points. Decomposed by arm — and this matters for the next step:

- **sigma 1.0, seed 44: the ON-arm moat flips** (`moat_on=false`). This is the candidate "encoding manufactures
  a fact" case — the DA-gated write gain is on this arm.
- **sigma 2.0, seed 44: the OFF-arm moat flips** (`moat_off=false`). The encoding gate is DISABLED on this arm
  (gain pinned 1.0), so this leak is NOT the encoding manufacturing a fact — it is a BASELINE read-floor
  artifact of the unstored-cue probe itself (the cleanup read spuriously "completes" the unstored cue under
  heavy read damage even with the coupling off).

So of the two leaks only one is attributable to the mechanism under test; the other is an instrument artifact on
the control arm. The instrument must be verified at each leak before either is read as a mechanism failure.

## Unmet precondition 2 — STRESS net regression (measured=14, expect=0), sigma-dependent

The stress-net precondition ("recall_ON ≥ recall_OFF at every swept sigma") fails at 14 of 54 (seed, sigma)
points. Decomposed by sigma, the sign of the ON−OFF difference FLIPS with read damage — the DA-gating REGRESSES
recall at low sigma and only IMPROVES it at high sigma:

| sigma | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|------|-----|------|-----|-----|-----|
| OFF recall (of 54) | 52 | 43 | 33 | 13 | 8 |
| ON recall (of 54) <!--derived--> | 46 | 36 | 32 | 24 | 18 |

<!--derived--> (ON aggregates are per-sigma sums of `recall_on` across the six seeds in the artifact's
`per_seed` block.) At low sigma the ON arm is WORSE (46/36/32 vs 52/43/33 at 0.5/0.75/1.0); at high sigma it is
BETTER (24/18 vs 13/8 at 1.5/2.0). first_regression_sigma=0.5.

The mechanism is salience REDISTRIBUTION, not a uniform lift: the Latin-square DA gives 3 facts a high gain
(g=2.48) and 3 a low gain (g=0.5 floor). The high-DA facts get a stronger, more read-robust trace (the win that
shows up at heavy damage); the low-DA facts get their magnitude HALVED, which costs recall under only-mild
damage. Over the realistic DA distribution the net is negative where reads are lightly damaged and positive only
once damage is heavy — so a flat default-ON flip trades away recall of the low-salience facts to buy robustness
for the high-salience ones. That trade is not net-neutral, so the flip is not earned.

## Disabled companion process (recorded, not hidden)

The spiking-cleanup read was disabled for speed (`enable_spiking_cleanup=False`). It is recorded as the SAME
KIND of stress the host read-damage sweep already applies (a further read-stress point on this sigma sweep),
not a distinct disabled companion mechanism — the magnitude sensitivity lives in the store's |w| plus the RF
read floor, which the sweep exercises directly. This is disclosed so the soak's scope is auditable, per the
"what else does the real system run alongside this" reframe.

## Next mechanism (no-defer, evidence-ordered)

The faculty stays default-OFF until a clean GO (0 moat leaks AND stress-net non-negative). In order of leverage:

1. **Verify the instrument at the 2 moat-leak points first** (real manufacture vs read-floor artifact). The
   sigma-2.0 OFF-arm leak is already a strong candidate for a baseline read-floor artifact (encoding disabled);
   confirming that reclassifies it out of "encoding manufactures a fact" and isolates the single ON-arm leak
   (sigma 1.0, seed 44) as the only mechanism-attributable one — a much smaller residual than moat_fail_total=2
   suggests. If the ON-arm leak is also a read-floor crossing, tighten the unstored-cue abstain threshold.
2. **A homeostatic encoding threshold co-adapting with the DA distribution.** The low-sigma regression is caused
   by the low-DA facts being written at the g=0.5 floor. A homeostatic set-point that normalizes the gain map to
   the running DA distribution (so the mean write gain tracks 1.0 and low-DA facts are not driven below the
   recall floor) is the companion process we replaced with a static clamp — the "what else does the real system
   run alongside this" reframe. The proven biological mechanism is homeostatic synaptic scaling (Turrigiano
   2008, "The Self-Tuning Neuron", Cell 135(3):422-435): a MULTIPLICATIVE scaling to a firing set-point that
   PRESERVES relative synaptic strengths. Applied here it removes the low-sigma regression (low-DA facts no
   longer driven below the recall floor) WITHOUT giving up the high-sigma robustness, because relative strengths
   — hence the salience ordering the coupling exists to create — are preserved.
3. **The consolidation interaction.** Once the write-time gain is homeostatically bounded, test the gain against
   the live consolidation/replay path (a stronger initial trace should consolidate more reliably), which is the
   behavioral payoff the flip is ultimately for.
