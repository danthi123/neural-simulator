# gap#5 (i) SWR generative-replay READOUT — precisely localized to a ca3→ca1 CONDUCTANCE CAP (NOT the completion, pathway, threshold, or latched-set). The completion WORKS (latched 116-320 cells) and the Schaffer pathway is abundant (61161 synapses), CA3 fires in phase-2, CA1 receives g_e — but ca1_g_e does NOT scale with schaffer_boost (tracks CA3 firing rate instead), so boosting weights is silently clipped and CA1 never depolarizes past rest. A precisely-characterized hard integration; the next lever is the effective-strength cap / CA1 excitability, NOT a bigger boost.

**2026-07-18.** After the user flagged the gap#4-vs-gap#5 order drift, I re-prioritized gap#5 and ran its "imaginative
replay" readout (`read_ca1`, the two-phase SWR ripple) — which was BUILT but never run. It does not fire CA1
(`ca1_fire=0` at every schaffer_boost 2→400). Four diagnostics localized why, precisely.

## The read (seed 42, n_ca3=2000, the validated completion GO config)
| schaffer_boost | latched (completed CA3) | CA3 phase-2 fire-rate | ca1_g_e | ca1_v | ca1_fire |
|---|---|---|---|---|---|
| 8   | 116-320 | 0.09-0.26 | 0.33-0.59 | -66 to -67 | 0.000 |
| 50  | 112-318 | 0.09-0.25 | 0.23-0.43 | -65.7 to -67.6 | 0.000 |
| 150 | 114-318 | 0.09-0.25 | 0.23-0.43 | -65.7 to -67.6 | 0.000 |

Schaffer ca3→ca1 synapses found = **61161**, mean_w(pre-boost) = 3.558, n_ca1 = 120.

## The precise localization (4 diagnostics — what it is NOT, and what it IS)
- **NOT the completion:** 116-320 CA3 cells latch/fire (far_max ~0.45) — the completion works (it's the CLOSED 5/6-GO).
- **NOT the latched-set threshold:** latched is far from empty (100-320 cells at thr 0.08; ~same at 0.02/0.04).
- **NOT a missing pathway:** 61161 ca3→ca1 (Schaffer) synapses exist, mean_w 3.56, excitatory (inhibitory ca3 excluded).
- **NOT CA3 failing to fire in phase-2:** the latched cells fire at rate 0.09-0.26 under the ripple.
- **NOT `_hard_silence` leaving CA1 clamped:** it is a clean reset (external current → bias 0, no residual).
- **IT IS a ca3→ca1 effective-CONDUCTANCE CAP:** `ca1_g_e` (~0.25-0.6) does NOT scale with schaffer_boost — it tracks
  the CA3 FIRING RATE instead (boost=8/fire=0.26→g_e=0.59; boost=150/fire=0.25→g_e=0.43). Multiplying the ca3→ca1
  weights (cc.data ×boost, up to 3.56×400≈1400) does NOT raise the effective conductance → the boost is SILENTLY
  CLIPPED (a known weight-clip / effective_synaptic_strength cap in this codebase, CLAUDE.md gotcha). With g_e ~0.5nS
  the drive is ~33 pA (g_e × ~66mV driving force) — ~20× too weak to fire an Izhikevich CA1 cell from -66mV, and it
  CANNOT be increased by boosting because the boost doesn't reach g_e. So `ca1_v` stays at rest (-66) and CA1 fires 0.

## BOTH diagnosis-aligned levers EXHAUSTED (2026-07-18) — it is a HARD conductance cap, not weight OR rate
- **Weight lever (schaffer_boost 8→400):** ca1_g_e ~0.25-0.6, unchanged → the boost is clipped.
- **Rate lever (ripple_pA 800→8000):** CA3 phase-2 fire-rate ROSE to 0.32-0.41 (from ~0.1), yet **ca1_g_e stayed
  0.11-0.46** (unchanged/lower), ca1_v -66 to -68 (rest), ca1_fire 0. So g_e does NOT scale with the CA3 firing rate
  EITHER. ⇒ ca1_g_e is HARD-CAPPED at ~0.5 regardless of weight or rate — a bridge-level ca1 synaptic
  conductance/effective-strength cap. Neither runner-side lever can overcome it; the fix is a bridge-level `sim/`
  investigation (the effective_synaptic_strength / g_e cap on the ca1 pathway), a focused future pass.

## Refinement (2026-07-18): the boost IS in the live weight matrix, so it's a deeper g_e-PATH puzzle
`base_synaptic_weights = self.cp_connections.data` (bridge.py:5970) directly — the schaffer_boost edits `cc.data`, so
the boosted weights ARE in the live matrix the g_e matmul uses (not a cached CSR, not a separate array). Yet ca1_g_e
doesn't scale with them. So it is NOT a simple weight-cache staleness; it is a deeper g_e-path issue (e.g. the ca1
region's synaptic-current path, a per-region conductance scaling, or the ca1 cell type's g_e→current conversion). A
genuine `sim/`-level investigation — the honest limit of runner-side diagnosis. Deferred to a focused future pass.

## 🎯 ROOT CAUSE FOUND (2026-07-19): STP DEPRESSION on the Schaffer (ca3→ca1) crushes g_e — the cap is not a mystery
With `enable_short_term_plasticity=False` (diagnostic): PEAK ca1_g_e jumps ~1 → **~3000** (the boost NOW reaches g_e) and
**ca1 FIRES** (0.34). ⇒ STP depression was the cap: `effective_synaptic_strength = cc.data × stp_u × stp_x`, and under
the ripple's sustained ca3 firing the resource `stp_x` depletes so the effective Schaffer strength is capped by the
resource, NOT the weight — boosting the weight is nullified. Two caveats (why the fix is TARGETED, not global STP-off):
(a) global STP-off makes the COMPLETION run away (latched 2000/2000, ca1_v → −30000 numerical blow-up — STP was also
bounding the ca3→ca3 recurrent); (b) g_e=3000 at boost=800 is wildly over-driven. **⇒ THE FIX = disable STP on the
Schaffer (ca3→ca1) pathway ONLY (keep it on ca3→ca3 for the completion) + a MODERATE boost → the completed assembly's
volley reaches ca1 cleanly.** A per-pathway STP config, not a global toggle. Turns the SWR readout from a "hard
integration" into a targeted fix. (Next: implement per-pathway/phase-2 Schaffer STP-off + moderate boost, verify ca1
fires WITH SPECIFICITY — ca1_match >> ca1_cross.)

## 🎯 SPECIFICITY BARRIER FOUND (2026-07-19): fixed-random DENSE Schaffer can't discriminate assemblies — needs LEARNED weights
With the STP fix (phase-2 STP-off → ca1 FIRES), the readout now has the OPPOSITE problem: **no specificity** (ca1_match =
ca1_cross = 1.000 at every boost 0.02→20 + every ca1_ff_inhib 20→150). Every completed assembly drives EVERY ca1 cell
to saturation (fire_sum ~40-50 of 120, ca1_v explodes negative from Izhikevich u-accumulation under the sustained
ripple). ROOT CAUSE (structural): the Schaffer ca3→ca1 projection is FIXED-RANDOM + DENSE (61161 synapses, ~510 inputs
per ca1). A large completed assembly (~300-400 cells) delivers NEAR-IDENTICAL drive (~76±9 inputs) to EVERY ca1 cell →
no cell is preferentially driven by ANY specific assembly → E%-max inhibition can't discriminate a near-tied drive, and
reducing the boost just moves between all-fire and all-silent (no specific-subset window). ⇒ **the SWR readout
specificity fundamentally needs LEARNED Schaffer weights** — the ca3→ca1 association POTENTIATED during encoding (the
CA3-assembly → CA1-target-pattern binding), so recall of an assembly drives ITS specific ca1 pattern. That is the
biologically-correct consolidation mechanism (Schaffer collateral LTP), NOT a fixed-random projection. **NEXT MECHANISM
(clear, biology-grounded): (a) LEARN the Schaffer ca3→ca1 during encoding** (Hebbian/BTSP potentiation when the assembly
co-fires with a target ca1 pattern) → recall gives the specific pattern; OR (b) a BRIEF single-volley sharp-wave read
(not the 60-step sustained ripple) so ca1 fires ONCE (the who-fires-most pattern) + sparse top-k. (a) is the real fix.
⇒ gap#5 (i) went from "hard integration, ca1_fire=0, mystery cap" → precisely: STP crushed g_e (fixed) + fixed-random
Schaffer blocks specificity (needs learned associations). A clear mechanism build, no longer a mystery.

## Status (per THE LAW — a precisely-characterized boundary that names the next lever)
- **The SWR readout is BLOCKED by the ca3→ca1 effective-conductance cap** — a real, precisely-localized hard
  integration (the documented "hard fresh-pass integration" snag, now root-caused). It is NOT closeable by the
  schaffer_boost lever the code provides (that's the wrong knob — it's clipped).
- **Next levers (NOT a bigger boost):** (a) find + raise the effective_synaptic_strength / conductance cap on the
  ca3→ca1 pathway (a bridge-level clip); (b) raise CA1 excitability (lower threshold / a controlled depolarizing bias)
  WHILE preserving specificity via a competitive CA1 mechanism (the completed assembly's volley → its OWN ca1 pattern,
  not a uniform bias that fires all CA1); (c) more/faster CA3 firing in phase-2 (higher ripple_pA / longer gamma) to
  raise g_e via the rate (the only thing g_e currently tracks). A focused future pass.
- **BOTH gap#5 extensions are hard, precisely-characterized integrations, NOT quick wins:** (i) this SWR readout
  (ca3→ca1 conductance cap), and (ii) emergent-DG (needs the layer-2 amplification wired in). The completion MECHANISM
  is CLOSED (5/6 GO); extending it in either direction is non-trivial. This is the honest gap#5 map.
- Infra: SWR_DEBUG-gated instrumentation in `_measure_ca1` + the schaffer-boost block (default-off → byte-identical).
