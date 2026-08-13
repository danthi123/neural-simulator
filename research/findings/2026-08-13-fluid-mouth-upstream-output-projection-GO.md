---
type: finding
status: qualified
claim_check: synthesis
verdict: GO (6-seed, 6/6). The mouth's UPSTREAM output projection `h_pre = Wo_sp @ state` — the WKV block's output matmul that PRODUCES the hidden feature `h` FROM the recurrent state, and residual #2's named host residual `h = r_h*(Wo_sp@state)` — is realized as a SIGNED GRADED-CONDUCTANCE synaptic read on the spiking substrate (the exact template the read-GO validated for the DOWNSTREAM `head_w @ h`, moved one matmul upstream toward the WKV state). The substrate graded margin reconstructs the reference projection at Pearson corr 0.9842 mean / 0.9601 min / cosine 0.9811 (6/6, calibrated ONCE on seed 42, 5 UNSEEN seeds pass); the inhibitory shadow (NEGATIVE `Wo_sp` weights, ~50% of the matrix) is LOAD-BEARING 6/6 (signed corr 0.984 vs positive-only 0.848, +0.14; the projection is ~46% negative so positive-only structurally caps ~0.85), generalising to the unseen seeds, NOT the 2026-07-04 decorative trap. Downstream the substrate projection carries the LM signal: fed through the (host) r_h gate + head_w read it agrees with the host next-word 0.769 (>76x chance) and free-gen stays coherent (self-NLL 0.85-1.34). Anti-cheats 6/6: scramble->corr ~0 (mean -0.0013), zero-STATE-input->downstream chance (cache-immune), 0 host draws on the projection read path. NOT "fully spiking": the WKV recurrent STATE (Wv input proj + the leaky integrator + BPTT-trained decay) is STILL host, the r_h gate is applied host-side, and the read-out was applied host-side here to ISOLATE the projection metric (it composes with the read-GO but the end-to-end chain was not run). A named upstream residual moved onto the substrate; a partial step, not the mouth's completion. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the UPSTREAM state->hidden output projection)
date: 2026-08-13
mechanism: SIGNED GRADED / CONDUCTANCE-DOMAIN output projection. `Wo_sp` [D,2D] (~50% negative) is Dale-split `Wo_sp = Wo_pos - Wo_neg` (both >=0). The WKV state `[ap,an]` [2D] is already NONNEG (the dual leaky ON/OFF code), so it rate-codes TWO matched carrier populations driven by the SAME current: EXCITATORY `stc_e` (wires Wo_pos as E_TO_E -> cp_conductance_g_e) and INHIBITORY `stc_i` (wires Wo_neg*ratio as I_TO_E -> cp_conductance_g_i). The D hidden pools are kept SUBTHRESHOLD (floor 0); each channel's hidden-feature value is read from the substrate's OWN net signed synaptic-current margin at rest, `hpre_k = (E_e-v_ref)*g_e[k] + (E_i-v_ref)*g_i[k] ~ (Wo_pos-Wo_neg)_k @ state = Wo_sp_k @ state` (v_ref=rest; the same combination fused_conductance_decay_and_current does at bridge.py:8375), integrated over the read window. The inhibitory:excitatory SYNAPTIC ratio is calibrated ONCE (seed 42; a WIDE plateau 0.3-0.7 -> corr 0.96-0.98, ratio=0.5 fixed) to balance the two driving-force terms. The read is a CONTINUOUS D-vector (the reconstructed hidden feature), not an argmax winner: 0 host draws on the projection read path. Reuse-by-import of WKVReadout + the graded-conductance read pattern; cfg.seed-controlled substrate; NO sim/ edit; runner-only, default-off.
artifacts:
  - research/runners/_wkv_graded_output_projection_derisk.py
  - research/findings/raw/_wkv_graded_output_projection_6seed.json
---

# gap#1 / A1 — biologizing the mouth's UPSTREAM output projection `Wo_sp @ state` as a signed graded-conductance synaptic read (GO, 6/6)

## The lever, and why it is the tractable next rung upstream

The read-out `head_w @ h` is now a substrate graded-conductance signed read at parity
(`2026-08-13-fluid-mouth-graded-conductance-read-GO`). That finding's honest residual #2 names what remains host:
"the hidden `h = r_h*(Wo_sp@state)` is a host residual; the WKV store is BPTT-trained; the read-out weights are
host-designed." This lane starts biologizing the UPSTREAM feature. The mouth's per-token chain is:

    (1) v      = Wv @ LN(emb)                     # input projection    (host, BPTT)
    (2) ap,an  = decay*ap+relu(v), ...            # WKV leaky STATE     (host, BPTT decay)   <- the upstream state
    (3) r_h    = sigmoid(Wr @ LN(emb))            # receptance gate     (host)
    (4) h      = r_h * (Wo_sp @ [ap,an])          # OUTPUT PROJECTION   (host, BPTT)   <- residual #2 (this runner: 4a)
    (5) logits = head_w @ h                       # read-out            (SUBSTRATE graded read, read-GO)

`Wo_sp @ [ap,an]` (step 4a) is the WKV block's output matmul — the DOMINANT compute in the hidden-feature
computation (D x 2D, vs the elementwise r_h gate) — and it is the op that PRODUCES the hidden feature `h` the read
consumes, directly FROM the recurrent state. Of the three named upstream levers (a spiking recurrent state-update,
biologizing `Wo_sp@state` as a synaptic read, a local rule for one weight matrix), the projection read is the most
tractable x highest-leverage: it is a DIRECT reuse of the just-validated read-GO template one matmul upstream, and
it is NON-REDUNDANT with the exhaustively-characterised on-bridge STATE arc (the parity-capped ~0.55 neural-integrator
frontier in `2026-07-19-gap1-WKV-...-RUNG1a-...`, which attacks step 2, not step 4). Biologizing step 4a moves the
substrate boundary from "the read consumes a HOST hidden feature" to "the hidden feature is a SUBSTRATE-computed
signed graded synaptic projection of the WKV state".

## The mechanism — the substrate computes `Wo_sp @ state` as its own net signed synaptic current

`Wo_sp` [128,256] is 50.1% negative — the sign is intrinsically load-bearing. Dale-split it `Wo_sp = Wo_pos - Wo_neg`
(both >=0). The WKV state `[ap,an]` is ALREADY nonneg (the dual leaky ON/OFF code the deployed mouth uses), so it
rate-codes two matched carrier populations driven by the SAME current: `stc_e` (excitatory) and `stc_i` (inhibitory).
`Wo_pos` wires `stc_e -> hpool` (E_TO_E, charging `cp_conductance_g_e`); `Wo_neg*ratio` wires `stc_i -> hpool`
(I_TO_E, charging `cp_conductance_g_i`). Keep the D hidden pools SUBTHRESHOLD and read each channel from the
substrate's own net signed synaptic-current margin at rest,
`hpre_k = df_e*g_e[k] + df_i*g_i[k] ~ (Wo_pos - Wo_neg)_k @ state = Wo_sp_k @ state`, integrated over the window
(the ~5-10 ms conductance taus average out the OU noise). This is the graded analog read a distributed code affords
(the read-GO's head_w@h result; the 2026-06-20 graded-plateau template) — a genuine conductance read, NOT a host
matmul, NOT a spike count. The output is a CONTINUOUS D-vector = the reconstructed hidden feature (not an argmax).
NO `sim/` edit; runner-only, default-off. The inhibitory:excitatory ratio is calibrated ONCE (seed 42, a WIDE
plateau 0.3-0.7 -> corr 0.96-0.98; ratio=0.5 fixed) — the 5 other seeds are the unseen generalisation test.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; D=128; GPU; 36 s)

| seed | hpre_corr_signed | (min) | corr_positive_only | cosine | scramble | downstream_argmax_agree | zero-state | GO |
|---|---|---|---|---|---|---|---|---|
| 42  | 0.9851 | 0.9652 | 0.8478 | 0.9833 | -0.010 | 0.775 | 0.000 | ✓ |
| 43  | 0.9835 | 0.9623 | 0.8347 | 0.9807 | -0.001 | 0.785 | 0.118 | ✓ |
| 44  | 0.9832 | 0.9601 | 0.8387 | 0.9794 | -0.004 | 0.765 | 0.028 | ✓ |
| 100 | 0.9830 | 0.9691 | 0.8579 | 0.9806 | +0.006 | 0.765 | 0.000 | ✓ |
| 101 | 0.9865 | 0.9756 | 0.8492 | 0.9818 | -0.004 | 0.770 | 0.044 | ✓ |
| 102 | 0.9837 | 0.9636 | 0.8616 | 0.9805 | +0.004 | 0.755 | 0.063 | ✓ |
| **mean** | **0.9842** | **0.9601** | **0.8483** | **0.9811** | **-0.0013** | **0.7692** | ~0.04 | **6/6** |

The substrate graded projection reconstructs the reference `Wo_sp @ state` at corr ~0.98 on all 6 seeds (5 unseen).
The inhibitory shadow (negative `Wo_sp`) is load-bearing 6/6 (signed corr 0.984 vs positive-only 0.848, +0.14): the
projection is ~46% negative, so a positive-only read (Wo_pos@state, all >=0) structurally caps ~0.85; the signed
graded margin recovers the negative half. Free generation with the SUBSTRATE projection in the loop stays coherent
(self-NLL 0.85-1.34), e.g. seed 100 *"once upon a time there was a little boy named tim found a big box of toys and a
lot of fun together they were all happy and played together every day"*.

## Anti-cheats (all 6 seeds)

- **Zero-state collapse (cache-immune):** driving the carriers with a ZERO state drops the downstream argmax-agreement
  from 0.769 to ~0.04 (vs 10x-chance 0.01) — the state input drives the read; it is not a floor/frequency artifact.
- **Scramble -> ~0:** a post-hoc `hpool -> channel` relabel collapses corr(hpre_sub, hpre_host) to mean -0.0013 (the
  labelled-line pool->channel map carries the projection; the high corr is per-channel structure, not a common scale).
- **Provenance:** hpre read from `cp_conductance_g_e/g_i`; `host_rng_draws_on_read_path = 0` on every seed.
- **(Diagnostic only, NOT gated) weight-lesion:** `hpre_corr_lesion` is recorded but DELIBERATELY not gated — a
  direct probe verified it is a KNOWN-UNRELIABLE instrument in this wiring (a NO-OP on cupy: `set_pathway_weights` on
  a fresh pathway name does not zero the existing proj synapses; hpre unchanged 13.0->13.0). This is exactly the
  contamination the read-GO parent documented ("zeroing the weights does not fully remove the conductance the pools
  already hold") and why it replaced the weight-lesion with the cache-immune zero-INPUT control used above.

## External grounding (reused)

Holt & Koch (1997), Neural Computation 9(5):1001-13 (the read-GO's binding): shunting inhibition is SUBTRACTIVE on
firing rate but the subthreshold membrane current at rest is a faithful LINEAR signed combination — the mechanistic
reason the graded SUBTHRESHOLD read at v_ref=rest recovers the signed projection as a clean continuous margin. The
inhibitory shadow here is `Wo_neg` (step 4) rather than the read-out's `-head_w` (step 5); the same low-floor,
rest-referenced signed-margin mechanism applies.

## Honest residuals — what moved onto the substrate, and what still leans on BPTT/Qwen

1. **ON the substrate now:** the `Wo_sp @ state` OUTPUT PROJECTION (D x 2D, the dominant matmul in the hidden-feature
   computation) is realized as a signed graded synaptic-current read (corr 0.984 to the reference, 6/6, sign
   load-bearing 6/6). This is a named host residual (#2's `Wo_sp@state`) moved onto neurons+synapses+conductances.
2. **STILL host / BPTT:** the WKV recurrent STATE itself — the input projection `Wv`, the leaky `ap/an` integrator,
   and the BPTT-trained `decay` (steps 1-2) — is unchanged host. The `r_h` receptance gate (step 3/4b) is applied
   host-side. The read-out `head_w` (step 5) was applied host-side HERE to ISOLATE the projection metric; it is
   biologized separately by the read-GO and the two graded reads COMPOSE, but the end-to-end substrate chain
   (state -> hpre graded -> logits graded) was not run in this rung.
3. **The downstream argmax-agree is 0.769, not 1.0:** the projection corr is 0.98, not 1.0; that ~2% reconstruction
   residual of a LINEAR graded read propagates through the argmax over V=1000 near-tied words. Same class of residual
   the read-GO named (recov 0.921, not 1.0). A facilitating accumulator or a learned/normalised read would close it.

## Named next rungs (in tractability order)

1. **Compose the two graded reads end-to-end** — feed the substrate `hpre` (this rung) into the read-GO's graded
   `head_w @ h` read, so the WHOLE `head_w @ (Wo_sp @ state)` state->logits chain is substrate graded-conductance,
   leaving only the WKV recurrent STATE + `r_h` host. A wiring composition, not new mechanism.
2. **Biologize the `r_h` receptance gate** as a per-channel shunting/divisive gain on the hidden pools (Holt & Koch
   shunting at a higher floor) — the remaining elementwise host op in step 4.
3. **The deep frontier: the WKV recurrent STATE itself** (step 2) — the leaky integrator + BPTT-trained decay/Wv.
   This is the neural-integrator problem the RUNG1a arc characterised (on-bridge parity-capped ~0.55; SpikeGPT keeps
   the WKV state graded in FP32), plus a local (non-BPTT) rule for `Wv`/`Wo_sp`/`decay` (the gap#1<->gap#4 meeting).
   The graded-conductance-domain insight validated here + in the read-GO is the lever for holding that state in a
   graded slow conductance rather than a spike rate.

## Files
- Runner: `research/runners/_wkv_graded_output_projection_derisk.py`
- Raw: `research/findings/raw/_wkv_graded_output_projection_6seed.json`
- Builds on: `2026-08-13-fluid-mouth-graded-conductance-read-GO.md` (the downstream read template + the graded-domain
  insight), `2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-6seed-GO-...md` (the WKV mouth + the state-integrator
  frontier), `2026-06-20-dendrite-derisk-A-graded-plateau-readout.md` (the graded analog read template).
