# (B) in-network opponency — NEGATIVE-branch options (pre-staged prep) — 2026-06-05

Pre-staged for if the NEF signed-value de-risk (subagent a259253f) comes back NEGATIVE (the small-signal
common-mode-removal `onoff(bon−boff)` may be fundamentally lossy in rate-coded spikes). A parallel deep-research +
catalog/Kandel pass (subagent ad862b3421046aa0f) is mining the biological opponency angle. This doc is the
controller's own next-step analysis, ordered cheapest-first. (If NEF = GO, this is moot — integrate + full clear.)

## The problem (recap)
`bind_fact`: spiking binds give per-role `(o,f)`; numpy `bon+=o; boff+=f` (superposition — IN-NETWORK FAITHFUL,
per-channel cos 0.97) then numpy `onoff(bon−boff)` (opponency = common-mode removal). The opponency is the blocker:
`bon,boff` correlated (cos 0.89, large common mode), `bon−boff` small → spiking read-noise swamps the difference
(signed cos 0.41), AND conductance inhibition is divisive. Even perfect numpy opponency on the spiking-read channels
recovered only 0.64.

## Option 0 (CHEAP-FIRST — try BEFORE any new mechanism): is the opponency even NEEDED in-network?
The opponency is "common-mode removal BEFORE storage." But the downstream is the spiking UNBIND (coincidence of
role⊗bound) + the validated spiking NEF CLEANUP (A). Both may already be robust to the common mode (the unbind's
coincidence may cancel a uniform offset; the cleanup is a nearest-neighbor match). **Test: store the RAW superposed
`(bon,boff)` (skip `onoff(bon−boff)`), unbind, cleanup, check recovery vs the numpy opponency path.** If recovery
holds → the opponency is UNNECESSARY in-network: do the superposition in-network (already faithful) + store raw +
let the validated downstream absorb the common mode → the WHOLE opponency problem disappears, full clear achieved
cheaply. If recovery drops → the opponency is genuinely load-bearing, proceed to Options 1-3. **~15 min GPU test,
no new mechanism. DO THIS FIRST.** (Also test the intermediate: store raw + a CHEAP in-network common-mode subtract
only at the unbind drive, where the signal is re-amplified by the role drive.)

## Option 1: biological center-surround / PREDICTIVE common-mode removal (the strongest principled lead)
The retina removes a SMOOTH/low-pass common mode by SUBTRACTING A PREDICTION (the surround = a local average), not
the raw correlated channel (Srinivasan-Laughlin-Dubs 1982 predictive coding; center-surround). If the `bon,boff`
common mode is a SMOOTH/low-rank envelope (the fill-magnitude envelope — plausibly low-rank), subtracting a
low-pass/low-rank ESTIMATE of it (robust, averages noise) instead of the full per-component channel may preserve the
residual. In-network: a "surround" pool computes the running mean/low-rank common mode; subtract it (in the linear
regime) from each channel → the residual is the de-common-moded signal, robust. The research pass is detailing this.

## Option 2: two-cell PUSH-PULL ON/OFF opponent representation (compute the channels DIRECTLY)
Biology represents opponent (ON/OFF, color) signals as TWO half-wave-rectified cells (ON fires for +, OFF for −).
Instead of computing `bon−boff` then rectifying, build an opponent pair where each cell receives `bon` excitatory +
`boff` inhibitory (and vice versa) and the winner-take-all-ish antagonism yields the rectified difference directly.
Same divisive-inhibition risk, BUT operated as a normalized opponent process (Carandini-Heeger divisive normalization
removes the common mode multiplicatively — the A cleanup already validated divisive-norm FS pools on this bridge).
Worth testing a divisive-normalization common-mode removal (divide by the envelope) vs the subtractive one.

## Option 3: PHASOR / complex (FHRR resonate-and-fire) re-representation (avoid the common mode entirely)
The deepest fix: in a phasor/complex VSA the bound value is UNIT-MAGNITUDE (no large common mode; the structure is in
PHASE, not a small rate difference), so the opponency/small-signal problem doesn't arise. Frady-Sommer 2019 PNAS
resonate-and-fire spiking phasors; the repo HAS a numpy FHRR reference (`spiking_phasor_fhrr`). Honest tradeoff: this
is a REWORK of the bind (the composer is the ±1 Hadamard, not FHRR) — a bigger arc, but it would make the ENTIRE
compute path naturally spiking. Reserve unless Options 0-2 fail; the research pass is assessing the rework cost.

## Option 4: HONEST FUNDAMENTAL BOUNDARY (if 0-3 fail)
If no mechanism preserves the small-signal common-mode removal in rate-coded spikes, that is a real biology-translatable
result (an SNR/channel-capacity bound on reading a small difference of correlated rates). Document it; the two LINEAR
glue ops stay numpy DISCLOSED (n=111); BOTH DEEP shortcuts (A readout, B storage) remain cleared. Then pivot to the
fully-grounded run (spine item 3) — the higher-value goal — with the composer's nonlinear core fully spiking.

## Execution order (NEGATIVE branch)
0. Cheap-first: is opponency needed? (~15 min). If unnecessary → DONE (full clear cheaply).
1. The research synthesis (a259...'s parallel pass) → pick Option 1 (predictive common-mode) or 2 (divisive-norm
   opponent) → de-risk (the A/B pattern).
2. If 1-2 fail → assess Option 3 (phasor rework) cost vs Option 4 (honest boundary), surface to owner.
