# gap#1 RUNG M5 — PRE-REGISTRATION: does the token-SDR conductance input (write-fidelity 0.906) recover positive deep-NLL?

**Filed 2026-07-20 before the deep-NLL run.** This MEASURES the real quantity the write-fidelity gate was a proxy
for, replacing the gate's ESTIMATED 0.95 bar with the actual deep-vs-trigram.

## Why this is the right test, and why it is NOT goalpost-moving

The write-fidelity de-risk gave 0.906 (token-SDR selection, OU-off), beating M2's regression 0.786, with
quantization refuted (a K-sweep made it worse). The 0.95 bar was the GATE'S ESTIMATE. The M2 finding MEASURED two
points of the write-fidelity -> deep-NLL map: 0.786 -> -0.030, 1.000 -> +0.126; linear interpolation puts 0.906 ->
~+0.06 (positive). Whether that holds is EMPIRICAL. Running the deep-NLL ONCE at the honestly-achieved 0.906 tests
the mechanism's actual capability. (Tuning knobs to reach 0.95 would be chasing the proxy — that is the move I am
NOT making; the K-sweep already refuted the obvious knob.)

## Method — reuse M2's VALIDATED eval, swap ONLY the encode

Add `--input-mode tokensdr` to the M2 runner (`_emerge_wkv_m2_nef_onbridge_derisk.py`). It builds a token-SDR pool
(V groups x K neurons) + FIXED Wv value-synapses into the same 2D `chan` neurons, and in `onbridge_states` drives
token x_t's SDR with fixed current (identity selection) instead of the NEF pool's v-dependent drive. **Everything
else — the slow SSM state step (ssm_k_leak = 1-decay), the read-out `head_w @ (rh * (Wo_sp @ state))`, the trigram
fit + per-depth eval — is M2's exact code, unchanged.** M2 already runs OU-off, so the comparison is noise-free
on both sides. Default `--input-mode nef` is byte-identical to M2.

## PRE-REGISTERED PREDICTIONS

1. **P1 — verify-first:** corr(token-SDR SSM state, numpy WKV reference) POST-rescale >= 0.85 on held-out eval
   sentences (the write-fidelity 0.906 should propagate to a high state-correlation).
2. **P2 — THE TEST:** deep (d10-99) vs-trigram **> 0** (the token-SDR conductance input recovers a positive margin).
3. **P3 — memoryless collapse (anti-cheat):** with `--memoryless` (decay=0, state cannot hold context), deep
   vs-trigram drops by >= 0.3 vs the intact run (proving the RECURRENT STATE, not the read-out, carries the deep win).

**INTERPRETATION FIXED IN ADVANCE:**
- **P1 + P2 + P3 pass** ⇒ **gap#1's spiking-input path WORKS via token-SDR selection** — M1's host matmul is replaced
  by spiking token-selection + real synaptic conductance, and it recovers a positive deep-NLL. The gate's 0.95
  estimate was conservative. (The "spiking veneer" caveat still stands for owner judgement — see below.)
- **P1 passes, P2 fails** ⇒ the ~0.906 write-fidelity is genuinely insufficient; the structural conductance-readout
  ceiling is the wall, and the gate's 0.95 estimate was right.
- **P1 fails** ⇒ write-fidelity 0.906 did NOT propagate to the state (a mismatch between the write-fidelity metric
  and the deployed state) — an instrument problem to diagnose before any verdict.

## The unresolved caveat (recorded, for owner judgement — NOT decided here)

Even if P1-P3 pass: the token pool is one fixed SDR per token = a lookup, so a skeptic calls this "M1 with a spiking
veneer — the spikes carry only the token identity the world supplied." The gate's defense: the WORLD supplies the
discrete token (legitimate sensory input under the brain-based-only standard) and the BRAIN's fixed Wv synapses do
the value projection AS SPIKING CONDUCTANCE, closing the exact audit gap (host matmul -> spiking synapses) without
M2's decode. Whether that meets the "fully spiking input" bar is the owner's call; recorded so it is decided
explicitly, not smuggled by a GO.
