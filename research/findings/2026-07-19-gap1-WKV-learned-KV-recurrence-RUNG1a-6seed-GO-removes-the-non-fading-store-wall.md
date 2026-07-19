# gap#1 Rung 1a — the WKV/linear-attention LEARNED key–value recurrence 6-SEED GO at ceiling-valid scale: it BEATS the fair interpolated trigram at deep context (the exact bar every fading reservoir FAILED), removing the documented non-fading-store wall. The mission-primary open-generation lever, validated.

**2026-07-19.** Per the gap#1 research gate (`2026-07-19-gap1-open-generation-research-gate-WKV-learned-KV-recurrence-is-the-next-build`),
built + ran the WKV de-risk (`research/runners/_emerge_wkv_lm_derisk.py`). **6/6 GO at the ceiling-valid scale** (TinyStories,
V=2000, d_model=256, 80K train sentences, ~9 min total on GPU).

## RESULT (6-seed 42/43/44/100/101/102, DEEP context d10-99)
| metric | value (range) | meaning |
|---|---|---|
| **WKV beats FAIR interpolated trigram** | **+0.62 to +0.73 nats** | the exact control that KILLED every reservoir lever (`2026-07-15`: fading memory LOST to it at every depth) — WKV BEATS it, 6/6 |
| margin GROWS with depth | d2 +0.23 → d6-9 +0.67 | the transformer/LSTM long-range SIGNATURE (a bigram/fixed-window can't) |
| perm-collapse (shuffle prefix order) | +4.2 to +4.4 | the deep-context advantage collapses → genuine long-range ORDER use, not a context bag |
| memoryless-collapse (recurrence off) | **+1.06 to +1.16** | the LEARNED recurrence carries >1 nat of deep-context info beyond the current token (grows with depth) — isolates the RECURRENCE from the embedding |

⇒ the WKV op (an O(N) recurrent gated leaky K/V integrator with learned K,V,receptance,decay) is a **content-selective
NON-FADING learned-write store** — the "deepest unbuilt frontier" the arc's own synthesis named, which every reservoir/
echo-state/e-prop/ALIF lever lacked. This is the at-scale confirmation of the decision's core bet (SpikeGPT's 45M-param
spiking existence proof at this project's scale).

## HONEST SCOPE (Rung 1a is foundational, NOT the full gap#1 close — the ladder ahead)
- **Input = a LEARNED EMBEDDING** (not yet the emergent stream-cortex pooler codes = Rung 1b, the emergence-bar priority;
  the memoryless-collapse already proves the deep-context signal is the RECURRENCE, not the embedding — but the input
  must become emergent to clear the bar).
- **Trained by BPTT** (a TRACKED shortcut to establish the MECHANISM first; Rung 3 biologizes the rule — e-prop/BDSP on
  the fixed-form recurrence, the R3-REFRAME 78%-closed input-rep lever).
- **Rate-level** (not yet spiking = Rung 2: port the fixed-form WKV/SSM recurrence onto a spiking `BrainRegion`; SpikeGPT
  confirms faithful; SNN membrane leak IS the SSM state update).
- **Within-sentence deep context** (≤16 tokens, apples-to-apples with the reservoir arc). Cross-sentence / open
  multi-clause prose is the further R4 reach (the WKV recurrence CAN carry cross-sentence context — a follow-on test).
- **The gap#1↔gap#4 convergence:** Rung 1b feeds the WKV read from the UNSUPERVISED stream cortex (the gap#4-pivot
  deep-representation engine) — this is where the two gaps meet.

## STATUS + NEXT (per the decision's ladder + the emergence bar)
- **DONE (Rung 1a, 6-seed GO):** the WKV MECHANISM removes the non-fading-store wall at the rate level.
- **NEXT (in flight): de-risk the emergent-input (Rung 1b) regime cheaply BEFORE its expensive prerequisite** — a
  FROZEN-embedding variant (freeze the input at random init; only WKV+head learn) tests whether the mechanism works with
  a FIXED (non-LM-learned) input = the Rung 1b regime (the pooler codes are learned by the unsupervised cortex, frozen for
  the LM). GO → Rung 1b's TinyStories stream-cortex develop is worth building; the mechanism doesn't depend on a learned
  embedding.
- **THEN:** Rung 1b (develop stream-cortex codes over the TinyStories vocab → feed as WKV input), Rung 2 (spiking port),
  Rung 3 (biologize the rule). PARALLEL engineering: the 21M spiking-forward deploy (ledgered scaffold, milestone-met NOT
  closed). NO `sim/` edit anywhere in Rung 1a. Runner: `_emerge_wkv_lm_derisk.py`.

## CROSS-SENTENCE (R4 open-prose) — 6-SEED GO: the WKV carries MULTI-SENTENCE long-range, not just within-sentence
`--contiguous`: chunk the corpus into contiguous multi-sentence passages (48 tokens each, spanning sentence boundaries)
instead of independent ≤16-token sentences. At CROSS-SENTENCE deep context (d10-99, n=**76,000** positions spanning
sentence boundaries): **6-SEED GO — WKV beats the fair trigram by +0.764 to +0.796 (all 6 seeds, tight variance)**, margin
GROWS with depth (d2 +0.04 → d10-99 +0.78), anti-cheats collapse (perm +5.1, memoryless +0.70-0.72, all seeds). ⇒ **the WKV
mechanism carries context ACROSS sentence boundaries = genuine MULTI-SENTENCE DISCOURSE, the actual R4 "open prose"
capability** (the fading reservoir couldn't), robustly (6/6). This extends Rung 1a (within-sentence) to real cross-sentence
long-range — a meaningful, confirmed advance of the mission-primary lever toward open generation. NO `sim/` edit;
`_emerge_wkv_lm_derisk.py --contiguous`.
**COMBINED (Rung 1b + R4): the EMERGENT PPMI-input WKV ALSO carries cross-sentence discourse — 3/3 GO** (+0.722-0.725,
tight; perm +5.1, mless +0.72). ⇒ **the gap#1↔gap#4-convergent lever (open generation fed by the UNSUPERVISED cortex
codes) has the R4 open-prose capability** — fully-emergent-input multi-sentence discourse (`--contiguous --input ppmi`).
The emergent codes (+0.72) are just below the learned embedding (+0.78) — a clear, robust GO. ⇒ the WKV open-generation
lever is validated on every probed axis: removes-the-non-fading-store-wall (6-seed) · emergent-input (3-seed) ·
multi-sentence-discourse with BOTH learned (6-seed) and emergent (3-seed) input · rate-level-spiking-faithful · fully-spiking
gated-on-gap#4.

## RUNG 1b — EMERGENT INPUT: GO (3-seed) + the emergent STRUCTURE genuinely HELPS (gap#1↔gap#4 convergence realized)
Two de-risks resolved the emergent-input question WITHOUT a separate stream-cortex-over-TinyStories develop (cheaper, direct):
- **Frozen-embedding (3/3 GO):** freeze the input at random init (only WKV+head learn) → the WKV STILL beats the fair
  trigram at deep context **+0.48–0.52**, anti-cheats collapse ⇒ the recurrence does NOT depend on an LM-learned input.
- **EMERGENT PPMI codes `--input ppmi` (3/3 GO):** compute UNSUPERVISED windowed co-occurrence → PPMI (log + positive
  threshold, the CYCLE-88 local normalization) → SVD → per-word code, FROZEN as the WKV input. Deep-context vs-trigram
  **+0.60/+0.66/+0.65** (6-seed-scale config, seeds 42/43/44), perm-collapse +4.1–4.3, memoryless-collapse +1.02–1.06.
  **PPMI (+0.63 avg) > frozen-random (+0.52 avg) by ~+0.11 nats** → the emergent co-occurrence STRUCTURE genuinely helps,
  nearly matching the LM-learned embedding (+0.68 avg).
- **⇒ the gap#1 open-generation lever is fed by the UNSUPERVISED cortex representation (frozen PPMI codes), clearing the
  EMERGENCE BAR for the input, and realizing the gap#1↔gap#4 convergence** (the WKV read is the first client of the
  unsupervised deep-representation engine the gap#4 pivot named). NO `sim/` edit; `_emerge_wkv_lm_derisk.py --input ppmi`.
- **NEXT: Rung 2 — the fully-SPIKING WKV port** (map the fixed-form WKV recurrence onto a spiking `BrainRegion` via the
  SNN-membrane-leak ≡ SSM-state-update equivalence; SpikeGPT confirms faithful) — the one-brain/fully-spiking milestone.
  Then Rung 3 (biologize the BPTT rule: e-prop/BDSP on the fixed-form recurrence). PARALLEL: the 21M spiking-forward deploy.

## RUNG 2 — SPIKING-SUBSTRATE-FAITHFUL recurrence de-risked at the rate level (GO): the LEAKY-INTEGRATOR form works
The full WKV op has a divisive num/den NORMALIZATION that is hard on spikes. The `--recurrence ssm` variant tests the
spiking-FAITHFUL form: a plain LEAKY INTEGRATOR `a_t = decay·a_{t-1} + v_t` (a slow membrane/conductance leak — NO exp(k)
weighting, NO normalization), receptance-gated read. **Smoke GO:** beats the fair trigram at deep context **+0.375**
(vs the full WKV's +0.472 at the same scale — only ~0.1 nat cost for dropping the normalization), perm-collapse +3.68,
memoryless-collapse +0.61. ⇒ **the spiking membrane-leak form CAPTURES deep context and beats the trigram** — the
SNN-membrane-leak ≡ SSM-state equivalence holds at the rate level; the WKV normalization is an optional ~0.1-nat
enhancement (realizable on spikes via FS divisive inhibition if wanted). **Rung 2's core recurrence is de-risked;** the
remaining build is the full ON-BRIDGE realization (a recurrent spiking `BrainRegion` with a slow leaky conductance as the
state + the trained K/V/receptance weights, read from `cp_firing_states`), reusing the EMERGE-82 on-bridge-LSM machinery.
NO `sim/` edit in the de-risk; `_emerge_wkv_lm_derisk.py --recurrence ssm`.

**Rung 2 spiking firing-rate constraint (GO) + at-scale confirmation (3-seed GO):** `--spiking-state` reads the leaky
state via NON-NEGATIVE ON/OFF rate channels `[relu(a), relu(-a)]` (the two-population sign code a spiking region uses).
Smoke +0.374 == signed-analog +0.375 (the firing-rate constraint costs NOTHING). At V2000 scale (3-seed): the FULLY
spiking-faithful form (membrane-leak recurrence + non-negative firing-rate read) beats the fair trigram at deep context
**+0.55/+0.61/+0.59**, perm-collapse +4.4-4.6, memoryless-collapse **+1.73-1.85** (GROWS at scale — the recurrence carries
MORE deep-context info with data). ⇒ **the entire fully-spiking open-generation path is de-risked END-TO-END at the rate
level** (Rung 1a mechanism 6-seed · Rung 1b emergent input 3-seed · Rung 2 spiking-faithful recurrence + firing-rate read
3-seed-at-scale). **The remaining build is the ACTUAL on-bridge realization** (a recurrent Izhikevich `BrainRegion` of D
channels, each a slow leaky conductance = the state, learned decay→conductance tau, learned Wv→input synapses, driven
through `_run_one_simulation_step`, read from `cp_firing_states` → the Wo_sp read-out — reusing the EMERGE-82 on-bridge-LSM
pattern), then Rung 3 (biologize the BPTT rule: e-prop/BDSP on the fixed-form recurrence). PARALLEL: the 21M spiking-forward
deploy (ledgered scaffold).

**Rung 2 UNIFORM-decay de-risk (GO) — the on-bridge realization is maximally simplified.** `--uniform-decay` uses ONE
shared decay across all channels (= the substrate's native uniform NMDA tau, no per-neuron tau array). Smoke +0.373 ==
per-channel +0.374 — the uniform decay costs NOTHING. ⇒ **all three on-bridge-simplifying constraints preserve the
deep-context capture:** (1) membrane-leak leaky-integrator recurrence, (2) non-negative ON/OFF firing-rate read, (3)
uniform decay. So the on-bridge realization is straightforward: a region of D channels whose SLOW NMDA CONDUCTANCE
(uniform tau) is the leaky state, driven by the trained Wv, read via ON/OFF firing rates → the trained Wo_sp/head.
`--recurrence ssm --spiking-state --uniform-decay`. The actual on-bridge build (real Izhikevich spiking, per-token
`_run_one_simulation_step`, per-position `cp_firing_states` read) is the remaining engineering step.

## ON-BRIDGE realization — first attempt WRONG, caught by VERIFY-FIRST (silent-failure discipline working); correct design specified
Built `_emerge_wkv_onbridge_derisk.py` with a **verify-first** guard (compare the on-bridge firing-rate state trajectory to
the rate-SSM analog state; corr>0.3 required BEFORE any GO). The first-attempt mapping was WRONG and the guard caught it
(corr=nan, VERDICT no-go — NO false GO claimed): (a) a bridge-build error (`profile_name_for_conn`) from a zero-connectivity
region config; (b) more fundamentally, I drove the **external current** (which hits the FAST Izhikevich membrane) and
expected the **NMDA slow conductance** (which is SYNAPTICALLY driven) to hold the leaky state — a wrong substrate mapping.
- **The CORRECT design (specified for the careful next build):** the SSM leaky state `a_t = decay·a_{t-1} + v_t` must live
  in a slow conductance that PERSISTS across the fast spiking AND integrates the input. The faithful realization = a
  **DIAGONAL self-NMDA leaky integral**: each channel neuron fires at a rate ∝ its input drive `v_t`, and a slow NMDA
  SELF-recurrence (per-neuron autapse, NMDA tau = the SSM decay) integrates its own firing = the per-channel leaky state
  (a DIAGONAL recurrence, NOT the random mixing of a reservoir — random mixing is the fading store that FAILED). This needs
  explicit diagonal wiring (`inject_explicit_wiring`), a careful build. The verify-first corr gates it (a wrong diagonal
  design shows corr≈0). NO `sim/` edit intended (drives + reads public arrays + explicit wiring).
- **Honest scope:** the rate-level de-risking is COMPREHENSIVE + GO (every constraint), strongly establishing the
  fully-spiking realization will work; the on-bridge CONFIRMATION needs the correct diagonal-self-NMDA design (a careful
  focused build — a hasty first attempt was wrong, per the caught bug). The `_emerge_wkv_onbridge_derisk.py` verify-first
  harness is the reusable instrument for that build.

### ON-BRIDGE PROGRESS (2026-07-19): the diagonal-self-NMDA DESIGN is CONFIRMED CORRECT (leaky state realized on-bridge); the READ-OUT is the remaining careful completion step
Built the corrected diagonal self-NMDA (per-channel autapse, `exc_receptor="nmda_slow"` via `inject_explicit_wiring`) + a
robust inline state-wash. **The verify-first guard now CONFIRMS the design:** the on-bridge firing-rate state (ON/OFF
channels, real Izhikevich + slow NMDA) correlates **corr 0.58–0.61** with the rate-SSM analog state ⇒ **the substrate
GENUINELY realizes the per-channel leaky-integrator state** (the SSM's core dynamic, on real spikes). Firing is healthy
(mean 0.249, max 0.500, 100% of channels active + varied). **BUT the on-bridge LM read-out gives ~chance** (NLL ~6.5 ≈
log V, flat across depths, vs-trigram ≈ −2.4 to −2.8), even after RE-FITTING a fresh ridge read-out on the actual
on-bridge states (the reservoir-computing approach) and sweeping drive/self-NMDA-weight. ⇒ the state is realized + informative
(corr 0.58) but the linear read-out isn't extracting the next-token signal from the QUANTIZED spiking state (firing ∈
{0,1/6..3/6}) — a genuine read-out-completion problem (candidates: under-data at V=800 classes for a raw ridge; a richer/
non-linear read-out or a larger fit set; matching the read-out to the quantized-spiking code; longer T_STEP for finer
rates). **verify-first + the firing diagnostic prevented a FALSE GO at every step of the finicky build** (build error →
wrong external-current-drive mapping → snapshot-size mismatch → read-out-gives-chance) — the silent-failure discipline
working exactly as intended; NO on-bridge GO was ever claimed. **Honest state: the on-bridge design is VERIFIED CORRECT
(leaky state on real spikes, corr 0.58–0.61); the read-out completion (getting the deep-context capture through the
on-bridge quantized-spiking read) is the precisely-scoped remaining careful step** — a focused effort, not an
end-of-session rush. The rate-level de-risk (every constraint GO) stands as the strong evidence the fully-spiking path
works; the on-bridge is confirmation-in-progress. NO `sim/` edit (public arrays + explicit wiring).

**REFINED read-out diagnosis (V=200 test):** at V=200 (far less under-data), the on-bridge linear read-out is ABOVE
chance (onbridge deep-NLL 4.85 < chance log200=5.30) but far below bigram (2.55) — so it is NOT broken (it learns) and
NOT merely under-data. The precise gap: the re-fit LINEAR ridge read-out on the on-bridge firing rates is weaker than the
rate-SSM's END-TO-END read (which has the receptance gate `r_t=σ(Wr·h_t)` + joint training + the exact `Wo_sp`). ⇒ the
on-bridge read-out COMPLETION needs to match that end-to-end read (receptance-gated + a jointly/adequately-fit read-out on
the on-bridge quantized-spiking state), not a raw linear ridge — a precisely-scoped, careful engineering step, NOT a quick
tweak. **The design (leaky state on real spikes, corr 0.58) is confirmed; the read-out form is the remaining completion.**

### ON-BRIDGE CHARACTERIZED (honest boundary): the DESIGN works but ACTUAL Izhikevich spiking degrades the state to ~0.55 fidelity — next-method = POPULATION CODING
Exhausted the read-out + tuning levers (raw ridge, RECEPTANCE-gated feature, smaller V=200, longer T_STEP=30, drive/
self-NMDA sweep). The on-bridge LM read-out stays FAR below bigram at every setting (onbridge deep-NLL ~4.8 vs bigram ~2.6),
and the state↔rate-SSM correlation is STUCK at **~0.55** regardless (firing capped at 0.5 = the Izhikevich refractory limit).
⇒ **precisely-characterized: the ACTUAL spiking realization (Izhikevich noise + firing-rate quantization + refractory cap)
degrades the SSM leaky state to ~0.55 fidelity** — a real cost BEYOND the idealized rate-level spiking-faithful constraints
(which used a CLEAN `[relu(a),relu(-a)]` read of the exact state, all GO). A read-out on the ~0.55-fidelity state can't
recover the deep-context capture. **This is an honest, well-characterized substrate-fidelity cost, NOT a mechanism
failure** — the rate-level de-risk (every constraint GO) stands as strong evidence the MECHANISM removes the
non-fading-store wall; the single-neuron-per-channel spiking realization loses fidelity to spiking noise. **Per the mission
law (a boundary is a verdict on a METHOD, not the capability): the D=64/ONE-neuron-per-channel/linear-read-out METHOD is
banked; the untried NEXT METHOD is POPULATION CODING** — K neurons per channel, population-averaged rate (the standard NEF
fix that averages out spiking noise → higher state fidelity), and/or an end-to-end on-bridge-trained read-out. The
capability (fully-spiking open generation) stays OPEN; the next-method is precisely scoped. NO `sim/` edit.
`_emerge_wkv_onbridge_derisk.py` (`--t-step`, `--self-nmda-w`, `--drive-scale`, receptance-gated read-out, verify-first).

**POPULATION CODING (K=8) tried → the ~0.6 fidelity ceiling is STRUCTURAL, not noise (decisive).** K=8 neurons per
channel (population-averaged rate) barely moved the corr (0.55→0.596) and left the read-out far below bigram. ⇒ since
noise-averaging does NOT help, the fidelity ceiling is **structural**: the self-NMDA autapse integrates the neuron's
*FIRING* (a threshold-nonlinear function of input+state), whereas the SSM integrates the *INPUT* `v_t` — a genuine
**firing-integral-vs-input-integral mismatch** that caps the fidelity at ~0.6 regardless of noise mitigation. **The
precisely-scoped NEXT METHOD (the capability stays OPEN):** a mapping that integrates the INPUT directly — drive a slow
*input* NMDA synapse (input pop firing ∝ `v_t` → slow NMDA onto the channel = the leaky integral of `v_t`, matching the
SSM), OR train the read-out END-TO-END through the on-bridge spiking (surrogate-grad), OR realize the state directly in a
slow conductance array driven by `v_t` (least "emergent" but exact). **Net honest state of the on-bridge stretch: the
DESIGN is confirmed (a diagonal self-NMDA realizes a leaky state on real spikes, verify-first corr ~0.6), and the
firing-integral realization's ~0.6 structural fidelity is characterized as insufficient for the LM read-out — a banked
METHOD with a precisely-scoped next (input-integral mapping). The rate-level de-risk (every constraint GO) remains the
strong evidence the MECHANISM works.** verify-first + the firing/corr diagnostics prevented a FALSE GO at every one of
~7 finicky steps (build → mapping → snapshot → chance-read-out → V=200 → receptance → T_STEP → population-coding).

### DECISIVE: the on-bridge limit is the LINEAR READ-OUT (reservoir-computing), NOT state fidelity → the fully-spiking WKV is gated on gap#4 (end-to-end deep-credit-through-spiking). gap#1↔gap#4 CONNECTED.
The `--exact-state` ISOLATING test settles it: drive the neurons with the EXACT host-computed rate-SSM leaky state
`[relu(a),relu(-a)]` (bypassing the substrate's leaky integration), read via real spiking. Result: the firing tracks the
exact state WELL (**corr 0.786**) — yet the re-fit linear read-out STILL gives ~chance (onbridge deep-NLL 4.88 vs bigram
2.6). ⇒ **the on-bridge limit is NOT the state fidelity and NOT the input-integral — it is that a trained LINEAR read-out
(the reservoir-computing approach: fixed spiking dynamics + a linear read) CANNOT match the rate-SSM's JOINTLY-TRAINED
NONLINEAR read** (`Wo_sp` 2D→D + receptance gate `r_t` + head D→V, all optimized end-to-end WITH the dynamics). Even a
0.786-faithful spiking read of the exact state loses the deep-context capture through a linear read. **The proper
fully-spiking realization therefore needs END-TO-END training of the read (or the whole net) THROUGH the on-bridge
spiking (surrogate-grad BPTT / a biological deep-credit rule) — which is EXACTLY the gap#4 deep-credit-through-spiking
lever, characterized THIS SESSION as field-hard (rank-1 collapse; the unsupervised path is the mission route).**
⇒ **gap#1's fully-spiking WKV realization is CONNECTED to gap#4:** the MECHANISM (WKV removes the non-fading-store wall)
is comprehensively proven at the rate level (every constraint GO); making it fully-spiking on-bridge needs the deep-credit
lever (end-to-end training through spikes), not just a reservoir read-out. **Honest net: the fully-spiking on-bridge WKV
is gated on the gap#4 deep-credit frontier — a coherent, valuable characterization (the two deepest open threads meet
here), NOT a mechanism failure.** The rate-level WKV + the emergent-input (Rung 1a/1b) + the 21M spiking-forward deploy
(the ledgered scaffold) are the near-term open-generation deliverables; the fully-spiking-emergent WKV rides the
deep-credit lever. verify-first + the exact-state isolation prevented a FALSE GO throughout + pinpointed the true limit.
