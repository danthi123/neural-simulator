# RUNG B-1c signed (±) synaptic read-out — **BOUNDARY** (honest negative; the signed relay fails + the objrel premise fails)

**Date:** 2026-07-04
**Runner:** `research/runners/_rungB1c_signed_readout_derisk.py` (`--seeds`, `--fixed-scale`)
**Raw:** `scratchpad/signed44.log` (ephemeral; re-run the runner to reproduce)

## The question

The B-1c.2 close-out (`2026-07-04-rungB1c-spiking-reservoir-synaptic-readout.md`) is GO 2/3 with a POSITIVE (Dale-offset)
synaptic read-out, and the adversarial verify located two residuals: (1) seed 44's sub-1% margin under-resolves; (2) the
reservoir's RECURRENCE is not load-bearing because canonical role ≈ content-word POSITION. The named surpass mechanism
was a **SIGNED (±) read-out** — `Ws+` as excitatory reservoir→ensemble synapses + `Ws−` through an **inhibitory relay**
(reservoir→relay[r] excitatory, relay[r]→ensemble[r] inhibitory) so the net ensemble drive ≈ `(Ws+ − Ws−) @ firing` = the
SIGNED `Ws @ firing` — to (a) resolve seed 44 and (b) read a NON-CANONICAL object-relative structurally where a
recurrence-lesion would finally collapse it. This de-risks that mechanism, isolated (reservoir + 3 ensembles + relay; no
WTA/composer/gate).

## Result — NEGATIVE on both counts (seed 44)

```
seed 44 (scale signed 14.94 / pos 21.97)
  sweep signed: 14.9:0/18  22.4:0/18  33.6:0/18  48.5:0/18  67.2:0/18  89.6:0/18  120:0/18
  sweep pos   : 14.6:10/18 22:12/18   32.9:5/18  47.6:9/18  65.9:8/18  87.9:4/18  117:6/18
  (a) CANONICAL host-agree /18: SIGNED 0/18   POSITIVE 11/18
  (b) OBJREL ["the","toad","that","the","dove","soars"]  host slot-winners [0,1,1] (slot0 should be THEME=2)
      SIGNED syn [1,2,0] (0/3)   POSITIVE syn [2,1,2] (1/3)   SIGNED+RECURRENCE-LESION syn [1,2,0] (0/3)
```

**Boundary 1 — the disynaptic inhibitory relay does NOT reproduce the linear signed argmax.** The signed read-out is
**0/18 at EVERY scale** (not a margin/resolution issue — a mechanism failure). A signed read-out with the full ± info
should be ≥ the positive (offset) read-out; instead it is far worse. Root (diagnosis, next step to confirm): the relay is
a NONLINEAR intermediary — `relay[r]` firing is an Izhikevich f-I function of `Ws− @ firing`, and `relay[r]→ens[r]`
inhibition is a fixed weight × that firing, so the subtraction is `RELAY_IE_W · f_I(Ws− @ firing)`, NOT the linear
`Ws− @ firing`. When the relay under-fires the negatives are simply dropped (degenerating to `Ws+`, which loses the
argmax); when it over-fires it swamps the ensembles. A clean linear negation via a single inhibitory relay pool is not
achievable at a single operating point — the honest hard part the argmax-preserving Dale OFFSET was invented to avoid.

**Boundary 2 — the reservoir does not structurally read the tested objrel.** The HOST `f@Ws` read-out (the ground truth
the synaptic read-out must reproduce) reads the object-relative slot-0 as **AGENT (0), not THEME (2)** — `[0,1,1]`. So the
reservoir's OWN learned read-out does not parse THIS construction structurally, which means the non-canonical test does
not exercise the reservoir's recurrence, and the recurrence-lesion is inconclusive (the signed read-out is already
broken). The premise "a non-canonical construction makes the reservoir's recurrence load-bearing" needs a construction
the reservoir provably reads structurally (a SPECIFIC EMERGE-78/79-validated relative clause + the head-role read-out),
not an arbitrary objrel fact.

## Implication for the close-out (honest state)

The FULL B-1c.2 close-out (3/3, a fixed non-host-calibrated scale, the reservoir's recurrence genuinely load-bearing) is
**NOT achieved via the signed inhibitory-relay read-out as approached** — both the mechanism and the premise hit real
boundaries. The honest committed state stands at **3 fully-clean rungs** (B-1 dict, B-1b argmax, B-1c.1 rate-reservoir all
removed) **+ B-1c.2 at GO 2/3** (the RUNTIME bind is synaptic; the per-seed `Ws'` scale is host-calibrated; canonical role
≈ position). This finding is the honest negative for the residual.

## The surpass path (undiscovered mechanisms — next session)

1. **A clean signed read-out that isn't a nonlinear relay:** a PAIRED ON/OFF split (each reservoir neuron gets an ON and
   an OFF copy driven by ±firing, like the composer's ±1 role bank), so the negation is at the SOURCE (two excitatory
   channels), not a downstream nonlinear inhibitory pool — OR a signed reservoir feature (rate ± around a baseline).
2. **A reservoir-VALIDATED non-canonical construction:** pick the exact EMERGE-78/79 relative-clause + head-role read-out
   the rate reservoir was proven to read structurally, port it to the spiking reservoir, confirm the HOST read-out reads
   it structurally FIRST, then test whether the synaptic read-out + recurrence-lesion follow.
3. Alternatively accept the honestly-scoped B-1c.2 (2/3, positive read-out) as the on-substrate result and characterize
   the per-seed-scale + canonical-position caveats as the documented limits.

## Files
- `research/runners/_rungB1c_signed_readout_derisk.py` — the isolated signed-vs-positive read-out de-risk (recurrence-
  lesion fixed to read the res→res edges from the CSR).
