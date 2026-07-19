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
