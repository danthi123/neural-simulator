---
status: live
type: finding
lane: gateb-v14
date: 2026-08-05
---

# Stage B source-model transfer: no unmodified candidate

**Date:** 2026-08-05
**Decision:** NO_SOURCE_TRANSFER_CANDIDATE
**Stage 2 integration:** not authorized

## Question

Can an unmodified, source-backed sodium or Kv3 kinetic model reproduce the
previously sealed SNr voltage-clamp targets closely enough to replace the
failed hand-built fast-channel packet?

The prospective comparison kept the four sources separate:

- Khaliq/Raman 13-state sodium, using graph-stationary initialization because
  the author-supplied native initializer was independently shown to produce
  negative occupancy;
- Balbi Nav1.6 six-state sodium at the source package's 22 C;
- Labro Kv3.1b four-state activation/deactivation model at 20, 22.5, and 25 C;
- Desai Kv3.3 activation/availability model with no invented temperature term.

No transition, rate, conductance scale, or topology was fitted. The sealed Ding
SNr command protocols and original endpoint estimators were reused unchanged.

## Authenticated result

All 12 preregistered jobs completed: six NumPy CPU references and six CuPy
RTX 3090 parity runs. Every provenance-v2 receipt verifies. Pointwise CPU/GPU
parity passed for all six conditions across 3,860,582 compared numeric values.

None of the four source models passed its available SNr endpoints:

| Model/condition | Failed endpoints | Result |
|---|---:|---|
| Khaliq/Raman graph-stationary | 9 of 10 | SOURCE_TRANSFER_NO_GO |
| Balbi Nav1.6 at 22 C | 8 of 10 | SOURCE_TRANSFER_NO_GO |
| Labro Kv3.1b at 20 C | 6 of 6 available | SOURCE_TRANSFER_NO_GO |
| Labro Kv3.1b at 22.5 C | 6 of 6 available | SOURCE_TRANSFER_NO_GO |
| Labro Kv3.1b at 25 C | 6 of 6 available | SOURCE_TRANSFER_NO_GO |
| Desai Kv3.3 | 6 of 8 | SOURCE_TRANSFER_NO_GO |

Labro has no inactivation state, so its inactivation endpoint remained
structurally unavailable and was not borrowed from Desai. No endpoint passed
anywhere in Labro's sealed room-temperature envelope.

Representative mismatches show that this is not a near miss:

- Khaliq sodium deactivation at -40 mV was 1.650 ms versus an allowed
  0.0812-0.1168 ms.
- Balbi sodium inactivation decay at 0 mV was 1.834 ms versus
  0.167-0.215 ms.
- Desai Kv3 rise at +40 mV was 4.477 ms versus 0.35-0.47 ms.
- Labro Kv3 activation midpoint remained near 0 mV across the temperature
  envelope versus the allowed -11.7 to -5.3 mV.

## Interpretation

The negative result rules out direct cross-preparation transfer of these four
unmodified parameter sets. It does not rule out their kinetic graphs as useful
candidate structures, and it does not imply that an SNr-specific model is
impossible. The sources describe different channels, preparations, species,
and temperatures; the result shows that those differences matter at the
whole-waveform level.

The former host-built gate packet remains retired. Combining the best endpoint
from each model, grafting Desai availability onto Labro, fitting conductance to
hide kinetic errors, or entering compartment integration would all violate the
prospective stop rules.

## Exact next action

Preregister an automated, source-constrained SNr kinetic-identification
campaign before running it:

1. retain separate candidate graphs and source identities;
2. authorize only named kinetic parameters with biologically justified bounds
   from primary sources or explicit uncertainty;
3. fit complete training waveforms, not only scalar endpoints;
4. reserve independent command voltages or waveform regions for held-out
   confirmation;
5. require identifiability diagnostics, CPU authority, GPU parity, and
   no-compensation endpoint gates;
6. stop without integration if no candidate passes.

This is the next useful experiment-engine workload: automated proposal,
parallel execution, scoring, rejection, and resumable evidence handling should
replace manual parameter adjustment.

## Evidence

- Prospective contract:
  research/specs/v14_snr_stageB_source_model_transfer_v1.json
  (ddb054d8aebdf580c355670cb7082c07bf838310f1dc41d2db19a4bd0a33b5f8)
- Analysis:
  research/findings/raw/v14_snr_stageB_source_model_transfer_analysis_v1.json
  (semantic SHA-256
  0a6df25bd84f395d84ae1d885ab6e198de4d053a3098e5edb9d08bb304b13361)
- Consumption ledger:
  research/findings/raw/v14_snr_stageB_source_model_transfer_consumption_v1.json
  (semantic SHA-256
  f63debfff84470845b24860ba40cd64a9631fc92cf237d7749b0c82710ed7c57)
- Analysis receipt:
  research/findings/raw/v14_snr_stageB_source_model_transfer_analysis_v1.receipt.json
  (dc58827800fbb7256ee26445360651142044780c630aad2845349cd02081d460)

The analyzer initially produced the same byte-identical result without a
provenance sidecar, so that unreceipted attempt was discarded. The successful
execution initialized the already-manifested research.runners provenance hook
before invoking the unchanged, hash-bound analyzer.
