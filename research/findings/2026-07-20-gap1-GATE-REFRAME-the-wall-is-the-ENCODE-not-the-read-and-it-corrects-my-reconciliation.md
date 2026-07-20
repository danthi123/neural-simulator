# gap#1 GATE REFRAME — the wall is the ENCODE (few-spike rate-code of a continuous v_t), NOT the read; and it CORRECTS my own reconciliation

The conductance-drive research gate returned a sharp reframe that **corrects the reconciliation I filed 2 commits
earlier** (`2026-07-20-gap1-RECONCILIATION-...`). Recording the correction, same discipline as the day's other
self-corrections.

## ⛔ WHAT MY RECONCILIATION GOT WRONG

I wrote: "M2 tried a NEF population that DECODES spikes to a scalar then re-injects; the lossy step is the decode;
route 2 (drive from a conductance) escapes it." **False.** The gate's repo lens establishes:

- **M2 ALREADY IS the conductance-drive.** It reads `cp_conductance_g_e = sum_i d_i * spikes_i` — the NEF decode
  happens IN THE SYNAPSES — and feeds that conductance to `cp_ssm_state`. The `cp_ssm_inject[:] = g_e` copy is
  cosmetic. So "conductance not rate-decode" does **not** distinguish a new arm from the burned M2 arm.
- **gap#4's "0.92 vs 0.000000" is a READ lever — already solved in gap#1** (the state IS a graded conductance;
  M1's read works). Applying "read as conductance" to the INPUT just re-charges a conductance from the input pool's
  few spikes = the burned feedforward/M2 arm. **My cross-gap connection was a category error.**
- **The wall is at the WRITE (encode), not the read.** Every gap#1 negative (self-NMDA 0.786, feedforward-NMDA
  0.55, M2 0.786, multi-channel plateau non-transfer) fails for ONE reason: the charging conductance is charged by
  a **few-spike rate code of a continuous v_t** over a ~6-step window. Quantizing a graded magnitude into a handful
  of spikes is the lossy step, and it is at the INPUT ENCODE.

## THE GATE'S ESCAPE — exploit that v_t is a function of a DISCRETE token

A candidate is genuinely new ONLY if it changes how `v_t` becomes the charging drive so that **spikes encode
something CLEAN, not a graded magnitude.** The structural fact that permits this: **`v_t = Wv * LN(emb[x_t])` is a
deterministic function of the DISCRETE token `x_t`** — one of V fixed vectors. That converts the encode from a
REGRESSION (M2's wall) into a **SELECTION + table-lookup**, which point neurons do cleanly (this project's own
concept-pool SDRs discriminate 320 concepts at 100%).

## RECOMMENDED de-risk (#1) — token-SDR selection → fixed Wv value-synapses → slow-NMDA conductance → inject

- **Operator:** token `x_t` drives a clean high-margin SDR token pool (deterministic spiking); its spikes charge a
  slow NMDA conductance through FIXED value-projection synapses `Wv` whose columns store the per-token value
  (LN baked in offline — it is a fixed function of the discrete token): `g_nmda = decay*g_nmda + Wv*s_token`, then
  `cp_ssm_inject <- g_nmda`. Because `s_token` is a clean deterministic selection, `Wv*s_token = v_t` as the EXACT
  sum of a fixed active-synapse set — **no few-spike quantization of a graded magnitude to lose.**
- **This is exactly the audit-gap fix:** M1's host matmul `Wv*LN(emb)` becomes spiking token-selection + real
  synaptic conductance. The leak stays the validated `g_i` shunt (RUNG4a 5/6), untouched.
- **Cheapest falsifier — a WRITE-FIDELITY gate, NO deep-NLL run:** over the ACTUAL deployed token sequence, measure
  `corr(v_t_true, conductance-derived v_t)` vs M1's exact host inject. Pre-registered bar **>= 0.95** (must clear
  M2's 0.786 by a wide margin; the -0.345 deep-NLL gap needs a near-exact input to recover M1's +0.126). Because
  A's input then approximately equals M1's exact input but spiking, passing the gate essentially PREDICTS the
  6-seed deep-NLL — so a cheap correlation measurement de-risks the whole thing.
- **THE BAKED-IN PRE-FLIGHT (the trap that refuted 3 mechanisms today):** run the concept-pool
  discrimination/determinism probe using the REAL token-drive the runner generates — real window length, real
  vocab, real correlated embeddings — NOT a synthetic one-hot. If deployed firing is not deterministic-per-token
  AND separable-across-tokens, the "selection" degrades to a rate code and #1 is dead pre-flight, cheaply.

## ⚠️ THE HONEST CAVEAT THE GATE FLAGS — for owner/controller judgement

If the gate can only be passed by making the token pool effectively a clean one-hot, **a skeptic will call this
"M1 with a spiking veneer" — the spikes carry only the token identity the world already supplied.** The gate's
position (which I find defensible but flag rather than assume): that IS the legitimate division of labor — the WORLD
supplies the discrete token (a legitimate sensory input under the brain-based-only standard), and the BRAIN's fixed
cortical `Wv` synapses do the value projection AS SPIKING CONDUCTANCE. It closes the exact audit gap (host matmul ->
spiking synapses) without the M2 decode. **Whether that satisfies the "fully spiking input" bar is a judgement call
worth surfacing** — it is recorded here so it is decided explicitly, not smuggled.

## Burned arms — DO NOT re-propose (gate + RAG)

M2 NEF regression decode (it IS the conductance-drive, 0.786/-0.345, all 9 levers exhausted); self-NMDA autapse
read (0.786); feedforward-NMDA-from-a-rate-pool (0.55); naive population/hetero-pop/scale/read-window/latency/
decay-match. **And the category error: "read the input as a conductance (gap#4)" is NOT an input fix** — gap#4 is a
read lever already solved in gap#1; a conductance-drive is new ONLY if the driving conductance escapes the
few-spike rate code AT THE CHARGE, which #1 does by discrete selection.

## Next

Build #1 as a minimal edit of the M2 runner (swap the NEF regression pool for a token-SDR pool + fixed `Wv`
synapses, keep `g_nmda -> inject`); run the WRITE-FIDELITY gate with the baked-in deployed-input pre-flight FIRST;
pre-register the >= 0.95 bar before running. NO deep-NLL until write-fidelity clears.
