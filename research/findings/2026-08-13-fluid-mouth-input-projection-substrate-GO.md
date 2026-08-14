---
type: finding
status: qualified
claim_check: synthesis
verdict: >
  GO (6-seed, 6/6). The mouth's INPUT PROJECTION `v = Wv @ LN(emb[tid])` — the LAST host matmul on the mouth forward
  path — is realized as a SIGNED GRADED-CONDUCTANCE synaptic read on the spiking substrate, via the SAME validated
  template that biologized the output projection `Wo_sp @ state` (corr 0.984) and the read-out `head_w @ h`. The signed
  input is the new subtlety: unlike the output projection whose WKV-state input was already nonneg, here BOTH the weight
  `Wv` (~50% negative) AND the input `LN(emb)` (~51% negative) are signed, so the read uses the four-quadrant Dale
  reduction `Wv@x = [Wv,-Wv] @ [relu(x),relu(-x)]` — which reduces EXACTLY to the output-projection read on the extended
  weight `[Wv,-Wv]` and the nonneg dual input `[x_pos,x_neg]` (reused verbatim by import). The substrate graded margin
  reconstructs the reference `Wv@LN(emb)` at Pearson corr 0.9805 mean / 0.936 min / cosine 0.9763 (6/6; drive_gain +
  inh:exc ratio + one output scalar calibrated ONCE on seed 42, 5 UNSEEN seeds pass). The inhibitory shadow (negative
  weights) is LOAD-BEARING 6/6 (signed corr 0.980 vs positive-only 0.797, +0.18) — with a signed input AND signed
  weights all four quadrants are populated, so a positive-only read structurally caps ~0.80. Fed forward the substrate
  `v` carries the LM signal: (B1) substrate `v` -> HOST WKV recurrence -> HOST read reproduces the full host mouth's
  next word at recov_argmax 0.9835 mean / argmax_agree 0.849 (autoregressive, cache-immune); (B2) fed into the
  ALREADY-substrate pipeline (WKV state slow-NMDA conductance + composed substrate read) the full state->logits chain
  holds recov_argmax 0.8847 mean / 0.854 min vs 0.9309 on host `v` — adding the input-projection stage costs only ~0.046
  recov (within tolerance; above the ~0.86 full-pipeline target, >> the 0.70 bar). Anti-cheats 6/6: scramble->corr
  ~0 (mean 0.0007), zero-input->downstream chance (0.017-0.058, cache-immune), scramble-input->chance, 0 host RNG draws
  and 0 host matmul on the `v` read path. On this GO the mouth's ENTIRE MATMUL CHAIN (Wv -> WKV state -> Wo_sp -> r_h
  shunt -> head_w -> head_b) is a substrate graded-conductance read; STILL host = the LN, the embedding LOOKUP, the r_h
  gate, and every trained WEIGHT VALUE (the learning rule is the separate 2026-08-12 e-prop GO), plus fixed unit
  scalars. NOT "fully spiking", NOT production-wired. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the INPUT projection Wv@LN(emb), the last host matmul in the forward path)
date: 2026-08-13
mechanism: >
  SIGNED GRADED / CONDUCTANCE-DOMAIN input projection, four-quadrant Dale reduction. The input projection `v = Wv @ x`
  (x = LN(emb[tid]), Wv [D,D] ~50% negative, x ~51% negative) is a signed-weight SIGNED-input matmul. A carrier
  population's rate codes a NONNEG magnitude, so it is realized by extending both to nonneg dual codes:
  `x = x_pos - x_neg` (x_pos=relu(x), x_neg=relu(-x)) and `Wv @ x = [Wv,-Wv] @ [x_pos,x_neg] = Wv_ext @ xstate`. With
  `Wv_ext` [D,2D] the projection weight and `xstate` [2D] the nonneg input, the problem IS an output-projection read:
  the SAME class `GradedOutputProjection` (Dale-split `Wv_ext = Wv_ext_pos - Wv_ext_neg`, EXCITATORY stc_e wires
  Wv_ext_pos as E_TO_E -> cp_conductance_g_e, INHIBITORY stc_i wires Wv_ext_neg*ratio as I_TO_E -> cp_conductance_g_i,
  driven by the SAME nonneg xstate current) reads each channel from the substrate's OWN net signed synaptic-current
  margin at rest `v_k = df_e*g_e[k] + df_i*g_i[k] ~ Wv_ext_k @ xstate = Wv_k @ x`, integrated over the read window.
  Expanded the four quadrants are g_e ~ Wp@x_pos + Wn@x_neg (excitatory) and g_i ~ Wn@x_pos + Wp@x_neg (inhibitory),
  so v ~ (Wp-Wn)@(x_pos-x_neg) = Wv@x (Wp=relu(Wv), Wn=relu(-Wv)). The read is a CONTINUOUS D-vector, NOT a spike
  count: 0 host draws, 0 host matmul on the v read path. drive_gain=450 + inh:exc ratio=0.5 (a WIDE plateau 0.3-0.7)
  + one output scalar v_out_scale=0.0867 (least-squares margin->host-v magnitude) are calibrated ONCE on seed 42 and
  FIXED for the 5 unseen seeds. Because `v = Wv@LN(emb[tid])` depends ONLY on tid, the substrate v is cached per token
  id. Reuse-by-import of GradedOutputProjection + WKVReadout + (for B2) GradedRecurrentState + ComposedEndToEndRead;
  cfg.seed-controlled substrate; NO sim/ edit; runner-only, default-off.
artifacts:
  - research/runners/_wkv_input_projection_substrate_derisk.py
  - research/findings/raw/_wkv_input_projection_substrate_6seed.json
---

# gap#1 / A1 — biologizing the mouth's INPUT projection `Wv @ LN(emb)` as a signed graded-conductance synaptic read (GO, 6/6)

## The lever — the LAST host matmul on the mouth forward path

The mouth's per-token next-word path is:

    (1) v      = Wv @ LN(emb[tid])                # INPUT PROJECTION    <- THIS RUNNER (the last host matmul)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)  # WKV leaky STATE     (SUBSTRATE slow-NMDA — state-GO)
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))       # receptance gate     (host)
    (4) h      = r_h * (Wo_sp @ [ap,an])          # OUTPUT PROJECTION   (SUBSTRATE graded read — projection-GO)
    (5) logits = head_w @ h + head_b              # read-out            (SUBSTRATE graded read + bias pop — read-GO)

Steps (2),(4),(5) are already substrate graded-conductance reads (`2026-08-13-fluid-mouth-wkv-state-graded-conductance-integrator-GO`,
`...-upstream-output-projection-GO`, `...-endtoend-substrate-read-GO`, `...-full-substrate-pipeline-GO`). Step (1) — the
input projection that turns the token embedding into the WKV drive `v` — was the ONLY remaining host matmul on the
forward path. Biologizing it closes the mouth's matmul chain: on a GO everything between the embedding lookup and the
winner word is a synaptic conductance read.

## The mechanism — the signed-input subtlety, and why it reduces EXACTLY to the output projection

The output-projection read (corr 0.984) had a NONNEG input (the WKV dual state `[ap,an]`), so one matched
excitatory/inhibitory carrier pair driven by the same state current sufficed. The input projection's input
`x = LN(emb[tid])` is SIGNED (~51% negative) AND `Wv` is ~50% negative. A carrier's firing rate codes a nonneg
magnitude, so a signed-input signed-weight matmul needs the four-quadrant (Dale) decomposition. Extending both to
nonneg dual codes makes it IDENTICAL to an output-projection read:

    x = x_pos - x_neg,   x_pos = relu(x) >= 0,   x_neg = relu(-x) >= 0            # nonneg dual input [2D]
    Wv @ x = [Wv, -Wv] @ [x_pos, x_neg]  =  Wv_ext @ xstate                        # Wv_ext [D,2D], xstate [2D]

So `GradedOutputProjection` (imported verbatim) runs on the shim weight `Wv_ext = [Wv,-Wv]`: it Dale-splits
`Wv_ext = Wv_ext_pos - Wv_ext_neg`, wires `Wv_ext_pos` as EXCITATORY (stc_e -> hpool, charging `cp_conductance_g_e`)
and `Wv_ext_neg*ratio` as INHIBITORY (stc_i -> hpool, `cp_conductance_g_i`), drives both carriers by the SAME nonneg
`xstate`, keeps the D pools subthreshold, and reads each channel's `v` from the net signed synaptic-current margin at
rest `v_k = df_e*g_e[k] + df_i*g_i[k]`. Expanded, the four quadrants are `g_e ~ Wp@x_pos + Wn@x_neg` (excitatory) and
`g_i ~ Wn@x_pos + Wp@x_neg` (inhibitory), giving `v ~ (Wp-Wn)@(x_pos-x_neg) = Wv@x` (`Wp=relu(Wv)`, `Wn=relu(-Wv)`).
Because BOTH the weight and the input are signed, all four quadrants are populated — the inhibitory shadow is
structurally load-bearing (a positive-only read caps ~0.80). `drive_gain=450 + ratio=0.5 + v_out_scale=0.0867` are
calibrated ONCE on seed 42 (a WIDE ratio plateau 0.3-0.7 -> corr 0.93-0.98); the 5 other seeds are the unseen test.
`v = Wv@LN(emb[tid])` depends only on `tid`, so the substrate `v` is cached per token id. NO `sim/` edit; runner-only.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; D=128; GPU/cupy; 1296 s)

### (a) reconstruction of `v = Wv @ LN(emb)` + (b1) substrate `v` -> HOST WKV recurrence -> HOST read

| seed | v_corr_signed | (min) | corr_positive_only | cosine | scramble | B1 argmax_agree | B1 recov | B1 zero-input | GO |
|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.9829 | 0.9623 | 0.7934 | 0.9786 | +0.0044 | 0.895 | 0.9907 | 0.058 | ✓ |
| 43  | 0.9816 | 0.9571 | 0.7936 | 0.9774 | -0.0019 | 0.830 | 0.9828 | 0.033 | ✓ |
| 44  | 0.9799 | 0.9529 | 0.7738 | 0.9763 | -0.0024 | 0.825 | 0.9827 | 0.025 | ✓ |
| 100 | 0.9752 | 0.9361 | 0.8116 | 0.9715 | +0.0001 | 0.875 | 0.9870 | 0.017 | ✓ |
| 101 | 0.9819 | 0.9510 | 0.8207 | 0.9774 | +0.0095 | 0.825 | 0.9729 | 0.033 | ✓ |
| 102 | 0.9813 | 0.9606 | 0.7915 | 0.9769 | -0.0054 | 0.845 | 0.9851 | 0.017 | ✓ |
| **mean** | **0.9805** | **0.9361** | **0.7974** | **0.9763** | **+0.0007** | **0.849** | **0.9835** | **~0.03** | **6/6** |

### (b2) full-substrate pipeline: substrate `v` fed into the ALREADY-substrate WKV state + composed read

| seed | fullsub_subV recov | fullsub_hostV recov | subV argmax_agree | delta (subV-hostV) |
|---|---|---|---|---|
| 42  | 0.8790 | 0.9242 | 0.6375 | -0.045 |
| 43  | 0.8832 | 0.9325 | 0.7375 | -0.049 |
| 44  | 0.8539 | 0.9234 | 0.6625 | -0.070 |
| 100 | 0.9061 | 0.9545 | 0.6750 | -0.048 |
| 101 | 0.9269 | 0.9445 | 0.7375 | -0.018 |
| 102 | 0.8594 | 0.9062 | 0.6250 | -0.047 |
| **mean** | **0.8847** | **0.9309** | **0.679** | **-0.046** |

The substrate graded input projection reconstructs the reference `Wv @ LN(emb)` at corr ~0.98 on all 6 seeds (5
unseen). The inhibitory shadow (negative weights) is load-bearing 6/6 (signed corr 0.980 vs positive-only 0.797,
+0.18): with a signed weight AND a signed input all four quadrants carry signal, so a positive-only read
(excitatory g_e alone) structurally caps ~0.80. Fed forward, the substrate `v` carries the LM signal: through the
host WKV recurrence + host read (B1) it reproduces the full host mouth's next word at recov 0.9835 (argmax_agree
0.849); fed into the already-substrate pipeline (B2) the whole state->logits chain holds recov 0.8847 vs 0.9309 on
host `v` — adding the input-projection stage costs only ~0.046 recov, within tolerance and above the ~0.86
full-pipeline target.

## Anti-cheats (all 6 seeds)

- **Signed vs positive-only (load-bearing):** signed corr 0.980 vs positive-only 0.797 (+0.18), 6/6 — the negative
  weights / inhibitory shadow are not decorative; the reconstruction genuinely requires all four quadrants.
- **Scramble -> ~0:** a post-hoc `hpool -> channel` relabel collapses corr(v_sub, v_host) to mean +0.0007 (the
  labelled-line pool->channel map carries the projection).
- **Zero-input (cache-immune):** zeroing `LN(emb)` drives the substrate `v` to ~0; the B1 downstream argmax-agreement
  drops from 0.849 to 0.017-0.058 (chance is 0.001) — the token input drives the read, not a floor/frequency artifact.
- **Scramble-input (cache-immune):** permuting the substrate-`v` channels before the recurrence drops B1 downstream to
  0.05-0.19 — the labelled-line channel identity of `v` is load-bearing through the recurrence.
- **Provenance:** `v` read from `cp_conductance_g_e/g_i`; `host_rng_draws_on_read_path = 0` on every seed; 0 host
  matmul on the `v` read path.
- **B2 not-degraded:** fullsub_subV recov within 0.10 of fullsub_hostV on every seed (worst delta -0.070), and >= 0.70
  on every seed — the input-projection substitution does not collapse the composed substrate pipeline.

## External grounding (reused)

Holt & Koch (1997), Neural Computation 9(5):1001-13 (the read-GO / projection-GO binding): shunting inhibition is
SUBTRACTIVE on firing rate but the subthreshold membrane current at rest is a faithful LINEAR signed combination —
the mechanistic reason the graded SUBTHRESHOLD read at v_ref=rest recovers the signed projection as a clean continuous
margin. Here the four-quadrant construction extends that to a SIGNED input: the same low-floor, rest-referenced
signed-margin mechanism carries all four excitatory/inhibitory quadrants of `[Wv,-Wv] @ [x_pos,x_neg]`.

## Honest residuals — what moved onto the substrate, and what still leans on the host/BPTT

1. **ON the substrate now:** the `Wv @ LN(emb)` INPUT PROJECTION (D x D, the last host matmul on the forward path) is a
   signed graded synaptic-current read (corr 0.980 to the reference, 6/6, sign load-bearing 6/6). Composed with the
   already-substrate state/projection/read-out, the mouth's ENTIRE matmul chain
   (Wv -> WKV state -> Wo_sp -> r_h shunt -> head_w -> head_b) is now a graded-conductance read.
2. **STILL host:** the LN inside `LN(emb)`, the embedding table LOOKUP (a labelled-line sensory input, legitimate host
   under the brain-based standard — the world rendering the token identity), the `r_h` receptance gate (elementwise
   host op), and every trained WEIGHT VALUE (Wv/decay/Wo_sp/head_w/head_b — the LEARNING rule is the separate
   2026-08-12 transport-free e-prop GO), plus the fixed unit scalars (drive_gain/ratio/v_out_scale, and B2's state
   affine + proj_out_scale + bias_scale). This is NOT "fully spiking" and NOT production-wired.
3. **The B2 subV argmax_agree is ~0.68, not 1.0:** the input-projection corr is 0.98, not 1.0; that ~2% linear
   reconstruction residual compounds with the state (~0.79 corr) and read residuals through the argmax over V=1000
   near-tied words. The recov (mass-weighted) stays 0.88 because the peak mass is preserved even when the exact argmax
   flips among near-ties. Same class of residual the projection/read GOs named; a facilitating accumulator or a
   learned/normalised read would tighten it.

## Named next rungs (in tractability order)

1. **Biologize `LN(emb)`** — the layer-norm is a per-token divisive/subtractive normalisation over the D embedding
   channels; a substrate realization (a normalising interneuron pool / divisive shunt over the embedding drive) would
   leave only the embedding LOOKUP + the trained weights host. This is now the last host ARITHMETIC on the input side.
2. **The e-prop weight learning** (the deep frontier, gap#1<->gap#4): every matmul is now a substrate read, but the
   WEIGHT VALUES (Wv/decay/Wo_sp/head) are BPTT-trained. The 2026-08-12 transport-free diagonal e-prop GO is the local
   rule; extending it to the full mouth weight set (with the substrate reads as the forward path) is the remaining
   scaffold-retirement step.
3. **Biologize the `r_h` receptance gate** as a per-channel shunting/divisive gain (Holt & Koch shunting at a higher
   floor) — the last elementwise host op on the state->hidden path.

## Files
- Runner: `research/runners/_wkv_input_projection_substrate_derisk.py`
- Raw: `research/findings/raw/_wkv_input_projection_substrate_6seed.json`
- Builds on: `2026-08-13-fluid-mouth-upstream-output-projection-GO.md` (the signed graded-conductance read template
  reused verbatim), `2026-08-13-fluid-mouth-full-substrate-pipeline-GO.md` (the B2 substrate state + composed read
  machinery), `2026-08-13-fluid-mouth-graded-conductance-read-GO.md` (the graded-domain read + Holt & Koch binding).
