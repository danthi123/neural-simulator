# Mood iteration 5: research notes (workflow wf_07a11b91-c66 readers, 11-18; unedited)

The mood iteration-4 NOT PASS at 1.5B points to three mechanisms that are biologically fair and map onto the RWKV7 cortex, and a short list of host-decisions to avoid. I read the repo files you named. The Bower, Aston-Jones and Cohen, Servan-Schreiber, Niv and Eldar-Niv citations come from the repo's own design doc and biology entries. I did not re-verify them.

## What the repo already records

- **Design doc section 1:** valence and arousal are separate things. NE from the locus coeruleus (LC) is "arousal and adaptive GAIN on cortical input-output functions". Dopamine tonic level tracks average reward and vigour. Serotonin tracks aversive prediction and patience. Mood biases perception of the next outcomes (Eldar and Niv). Each signal is low-dimensional and broadcast diffusely.
- **Design doc section 4, option B:** arousal as an inverse temperature on the output, for arousal only. Option C: arousal on per-head decay rates, "later; needs the cortex forward to be opened". Options D (logit bias toward affect words) and E (text in the prompt) are already rejected.
- **`locus-coeruleus-ne-adaptive-gain-swap.md`:** established, with a lesion and 6-seed GO. The LC has a tonic drive (the lesionable arousal state) and a phasic burst from the mismatch population (Bouret and Sara's "network reset"). Its rate sets a gain on an existing effector. The lesion pins the gain at a floor below the default, and a readout-zeroed control shows the gain, not the firing, is load-bearing. The repo says plainly that the rate-to-gain readout is host arithmetic, a disclosed residual.
- **`interoceptive-affect.md`:** body state enters only as an afferent current. Comfort and discomfort drive V+ and V-, and arousal drives the arousal pool, all through synapses. The body-to-affect map is never computed in host code.
- **`affect-active-clear.md`:** a basin-switch beats a standing brake. Durstewitz and Seamans 2008 is cited as the "same network holds or clears depending on modulatory gain" sibling.
- **`importance-tagging-at-encoding.md`:** the arousal mark (beta-adrenergic NE via the amygdala) acts locally. It amplifies the most active representation, suppresses weaker ones, and sets what is stored. It is status proposed.

## Why the current method is biologically thin

The additive `GAIN*mood*axis` is a valence push added to the state. It never acts as gain, so the arousal half of mood has no effect (the design doc says "arousal effects (not acting yet)"). It also asks one scalar to do the work of three systems. At 1.5B it also had two separate faults:

- **Appraisal fault:** held-out positive events are read right on only 2/6 seeds.
- **Fluency ceiling:** an additive push big enough to move tone costs fluency (NLL ratio 1.38 at GAIN 32).

The biology also says a push is the wrong shape. Valence is a bias on appraisal and on which memories are retrieved. Arousal is a multiplicative gain on processing. Neither works as a direct tone vector.

## Mechanisms that are biologically fair and map onto RWKV

The first three are the best fits.

1. **Arousal as a gain on the readout temperature or inverse temperature (option B).**
   - Fit: this is the closest match to Servan-Schreiber and Aston-Jones and Cohen. Sharper or flatter selectivity of the output is the standard NE-gain claim, and the gain slot already exists in `speak(temperature=, top_p=)`.
   - Lesion: gain pinned at 1, with the mood and arousal still computed and read out.
   - Limit: it moves how decisive and varied the reply is, not whether it is warm or dark. Test it as an arousal effect only, as the doc says.

2. **Arousal as a gain on per-head state decay `w` (option C).**
   - Fit: in `plastic_cortex.py`, `w = exp(-0.606531 * sigmoid(w0 + w_t))` is the forget rate. Scaling the sigmoid input moves integration versus reset, which matches the LC "network reset" and NE/ACh integration claims.
   - Phasic variant: a burst on a surprise event, which `modulators.py` was planned to supply, gives a reset when something salient happens.
   - Lesion: gain pinned at the floor, as in the LC entry.
   - Limit: this is not yet runnable on the pretrained 1.5B. The decay is applied inside the RWKV forward, and the modulation requires opening that forward for the real cortex. The 1.5B fluency risk is unknown and needs a dev check.
   - Why it fits the 1.5B failure: gain on dynamics acts on how the model weighs its own context. That matches the "gain-type, how strongly the cortex weighs its own state" branch already registered as next.

3. **Mood-congruent reinstatement (Bower), the strongest valence mechanism.**
   - Fit: mood does not add a tone vector. It scales the hippocampal reinstatement gain (`hippocampus.py`, currently a fixed 4.0, listed as a scaffold to replace) for episodes whose stored valence matches the current mood.
   - Why it is fair: the same slot is already scheduled to move to neuromodulatory control, and Bower is a retrieval and processing effect. Episodes already carry salience and strength, and the appraisal axis can tag valence at encoding, as the amygdala does.
   - Limit: it needs episodes to exist. It shapes replies only on turns where recall fires, so the lesion test must use prompts that cue memories. It is also the biologically intended channel for tone, which a bare push fakes.

4. **Valence as a bias on appraisal, not on output.**
   - Fit: Eldar and Niv's mood biases how the next outcomes are perceived. Mood can shift the appraisal zero (the neutral reference) so that positive and negative events are read relative to it.
   - Use: this addresses the appraisal fault and the fact that positive events read right on only 2/6. It is also an expectation-relative reference, which `SCAFFOLDS.md` already names as the biological target.
   - Caution: that is a change to the reader, so its lesion is not a reply lesion on its own. It also risks circularity if the same mood both sets the reference and reads the event.

5. **Mood or arousal gain on encoding strength.** This is the amygdala-to-hippocampus arousal tag (importance tagging). It affects which episodes are kept strongly, not the current reply. It is a milestone candidate for the later retrieval arm, not the shaping test.

6. **Keep the additive axis only as the amygdala-to-cortex valence projection.** It is fair if framed as that projection alone. Per the 1.5B result it is weak, so it should not carry the claim by itself.

## Host-decisions to avoid

- Logit bias toward affect words, or any marker-word or template insertion. Rejected options D and E, and the owner rule.
- Choosing the valence of the reply in host code: an `if mood < 0` branch that picks a register, or a host switch that selects sad or happy replies.
- A host-computed rate-to-gain map presented as spiking. The LC entry admits it is host arithmetic, so record it as a declared scaffold rather than claiming a brain-based gain.
- Hand-tuning GAIN, the temperature, or the decay gain on the very test questions that are then scored. Select on dev only, as iterations 3 and 4 did. `feedback_dont_optimize_own_tests` applies.
- Letting affect decide whether to answer or what fact is recalled. The composition law says that belongs to the hippocampus and the familiarity bar.
- Making valence a gain. Gain is the arousal channel. Using it for sign collapses the valence versus arousal split.
- Treating a standing brake as the lesion. The affect-active-clear entry shows a transient gate works and a brake fails.
- Pinning the lesion in a way that only changes a number the reply never sees. The lesion must change the reply.

## Suggested order

1. Re-select the appraisal read on 1.5B dev events. This is a precondition for any positive-mood claim.
2. Register arousal-as-gain on temperature (cheapest, no forward edit) and on state decay (needs the forward opened) as separate arms, with a gain pinned at 1 as the lesion.
3. Add mood-congruent reinstatement gain as the valence channel, tested on turns that cue stored episodes.
4. Pre-register before building, per the existing branch in design doc section 15.

Files read: `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/affect.py`, `brain/research/2026-11-03-mood-state-design.md`, `brain/experiments/results/2026-11-18-mood4-1.5b.md`, `brain/ARCHITECTURE.md`, `brain/PREREQUISITES.md`, `brain/SCAFFOLDS.md`, `brain/cortex.py`, `brain/plastic_cortex.py`, and under `research/biology/` the LC-NE, interoceptive-affect, affect-active-clear, importance-tagging and affective-marker-abstention-congruence-gate entries. I did not read `brain/experiments/mood_probe.py`, `brain/experiments/results/2026-11-03-mood3.md` or the iteration-4 result JSONs; the iteration-3 numbers I cite come from the design doc and `ARCHITECTURE.md`.

=====

Record check finished. The check itself did not run clean, and the answer to "what is reusable for a 1.5B RWKV cortex" is mostly nothing. Only ideas carry over. Nothing here changes `brain/PREREQUISITES.md` or `docs/RETRACTED.md`.

## 0. The check did not run clean
`bash tools/before_you_build.sh "mood or affect shaping language model replies: gain modulation, arousal, neuromodulator"` stopped at its first step with "STALE CORPUS: published main 0474195f5 is not in local main (cca07cc5a)". The tool says this check is NOT recorded, and none of its search sections ran. The fix is `bash tools/w_follow.sh`, then `systemctl --user start sim-rag-autoupdate.service`. This worktree was read only, so I did not do that.

I replaced it with a manual read of the same record:
- `git log --all` filtered by keyword.
- `research/findings/`.
- `research/biology/`.
- `docs/RETRACTED.md`.
- `docs/drafts/affect-results-audit/RESULTS.md`.
- `brain/research/2026-11-03-mood-state-design.md` (section 2 has the old-track port table; I read sections 4 and 16, and the 11-03 iteration-3 and 11-18 iteration-4 result write-ups).

The git log is mostly noise. About 50 of the lines are "Merge claude/gain-refit-1113" commits. The "gain" in the recent non-merge commits is the hippocampal reinstatement gain, which is not mood. The genuine mood commits are:
- 28089180a: prototype `brain/affect.py`.
- d15220f8e: 0.1B probe.
- 515cbeca6 and eb5788b9e: iteration 2.
- fc5fef539 and 241477398: iteration 3.
- 8f877debc, 827e1d631 and cc2fb3e7e: iteration 4.

## 1. RETRACTED.md affect items
- `2026-09-25-affect-learned-vocabulary-eval-NOGO` is UNMEASURED, not failed. The node lacked the valence map. Re-measure is pending (RULING affect-vocab-remeasure).
- Two findings are void in part under RULING affect-audit-rule:
  - `2026-09-23-onebrain-affect-ladder...-NOT-ALL-GO` (affect-pool verify verdicts only).
  - `2026-09-24-open-ended-gated-turn...` (Gate-B valence-sign claim only).
  - The cause is nodes with no valence map that still read valence.
- The audit draft flags 6680 pool results, all from the old `affect_production_organ` import closure. Both items are old-track instrument problems.
- Neither item touches `brain/affect.py`. No mood or arousal result that `brain/` relies on is retracted.

## 2. Old track: what was tried
**A. Affect state (spiking region)**
- P0.3 affect-state region (`F/2026-07-24-P0.3...`):
  - It was a slow-NMDA opponent attractor. Mood persistence was 0.62 of peak with NMDA on versus 0.00 with NMDA off.
  - Congruent-recall advantage was +0.067, and it vanished under an `affect_out` lesion. A yoked-random affect misdirected it.
  - The filename says GO but the verdict is QUALIFIED-GO / BOUNDARY. An earlier "graded circumplex" claim (r=0.68) was retracted the same day. The state is a bistable good/bad latch.
- Follow-ups: staggered bistable ladder (graded, GO), graded attractor (GO), and active-clear (a spiking quench is the only thing that evicts a saturated latch; every outward brake fails).
- Interoception (`BIO/interoceptive-affect`): body state (comfort and arousal) drives the affect pools through synapses. It gave a signed switch, and a lesion collapsed it to 0.000.
- Appraisal origin (DR-2 and D1): valence self-organizes from about 10 innate primary reinforcers, so no hand lexicon is needed. This is the origin of the current `affect.py` design.

**B. Arousal**
- `affect-arousal-channel` (BOUNDARY): arousal as the SUM of the opponent reinforcer channels predicts Warriner arousal at r=+0.265.
- `affect-lc-arousal-population` (GO, 6 seeds): an LC-like population integrating two salience afferents reached r=+0.311 and beat the sum channel on 6/6 seeds.
- `BIO/locus-coeruleus-ne-adaptive-gain-swap` (DE-RISK GO):
  - NE sets the gain on the existing eviction effector: `boost_gain = FLOOR + SPAN * ne_level`.
  - Lesion leaves the gain at a floor below the baseline, giving a slow, sticky swap.
  - It worked 6/6 on speed, and the readout-load-bearing control passed 6/6.
  - Caveat: it acts on a spiking workspace swap. Language never entered it.

**C. Affect acting on a language mouth (the closest analogue to the current job)**
- 08-19 `affect-drives-chat`: a graded "Wonderful — / Honestly —" lead marker. This is the marker-word, host-templating pattern the owner has since ruled out (design doc option D/E). Load-bearing, but it is not a reusable method.
- 09-03 `affect-wiring-into-wkv-mouth` (GO, 1 seed, waived):
  - A decode-time additive logit bias over a Warriner lexicon, wired into a WKV/SSM mouth.
  - Load-bearing on the SSM mouth on the small checkpoint. It was weaker on the linattn/BPE mouth.
- 09-04 linattn flip confirmation (live NO-GO):
  - The output was byte-identical under affect lesion on the stronger LM, because a sharper logit distribution swallowed the bias at live valence of about 0.04.
  - A fixed boost of 8 or more did register but collapsed into word-salad.
- 09-04 sharpness-aware GO: scaling the bias to the logit margin made it load-bearing on both mouths. Fluency held (salad fraction 0.16 or below, against a 0.09 to 0.10 baseline). It was seed 42 only, and still host arithmetic.
- 09-04 `affect-coupling-neural-not-host` (PARTIAL, 6 seeds):
  - The bias was replaced by a per-candidate neuromodulator concentration through `sim/neuromodulators` `excitability_drive`.
  - It was load-bearing on 18/18 combinations. The flag-off and lesioned arms were byte-identical.
  - The source half (which word is mood-congruent) is still a host Warriner scalar.
  - Negative mood undershot the lesion baseline on 7/18 combinations.
- 08-27: the tone-spiking-mouth fix (default ON).
- 09-05 gate-embodiment BOUNDARY series: experienced-opponent gate needs embodiment; concept codes must be grounded.

**D. Neuromodulation in general**
- 08-19 `neuromod-reconfiguration` and `neuromod-spiking-da-mode` (6/6 GO): one neuromodulator switches Go versus NoGo circuits on fixed wiring. These are basal-ganglia results, not language.

## 3. What worked and what did not
Worked:
- A persistent, lesionable valence state (P0.3), and appraisal from innate primaries instead of a lexicon (DR-2b).
- NE-as-gain with a lesion floor (LC finding), on a spiking workspace.
- Making a state load-bearing on a language mouth, but only through a host logit bias on small ssm/BPE mouths.

Failed or bounded:
- Latch versus graded mood, and the ratchet (a mood that cannot be evicted).
- A fixed-constant bias on a sharper LM, which went byte-identical. This is the same signature as the 1.5B additive failure: the push is too small to move the argmax unless fluency is paid for.
- The Warriner and marker-word routes, which the owner has rejected.

## 4. Reusable for the 1.5B RWKV cortex
Nothing in the old code ports directly. All of it is spiking-substrate or host-logit code. The design doc already recorded this, with option B (arousal as a multiplicative gain, lesion = gain pinned at 1) marked as the NEXT arm.

What carries over as design:
1. **Gain, not a direction.** The LC finding's form is `gain = FLOOR + SPAN * arousal`. The lesion floor is below baseline, and a readout-load-bearing control zeroes the readout coefficient at the same drive.
2. **Scale the bias to the model's own margin.** The sharpness-aware GO and the linattn NO-GO both show that a fixed-constant push gets swallowed by a sharper LM. The 1.5B additive result (negative push -1.42 vs -4.08 at 0.1B, GAIN capped at 16 by NLL) is the same effect. A gain applied to the cortex's own quantities (for example the temperature, or how strongly it weighs its own state) scales with it by construction.
3. **Arousal inputs.** The sum-channel and LC-population results suggest arousal as |appraisal| plus the cortex's own surprise. The surprise signal exists at AUROC 1.00 per the 11-18 novelty stage 0.
4. **Neuromodulators as a small set of scalars (a "diffuse bus").** The mood-to-word coupling used a per-candidate concentration. A per-layer or per-head scalar is the 1.5B equivalent, and the design doc's option C (per-head decay) is the same idea.
5. **Appraisal.** The 1.5B defect is the positive-pole read (2/6 seeds, with negative right on 6/6). The old DR-2b route (conditioning on about 10 primaries) is what `affect.py` already uses. Nothing in the old record fixes a linear read that fails on the positive side, so that fault is new ground.
6. **Controls and gates.** Reuse the lesion-vs-lesion repeat determinism check and the flag-off byte-identical check. A live-pipeline test on the deployed config is also worth copying, because the proxy-verified GO on the old mouth was a false positive.
7. **Rules.** Affect sits below honesty (it never manufactures an answer or overrides an abstain), and it must not become a refusal mode.

Do not port: the Warriner lexicon bias and the marker-word lead (both owner-rejected), the bistable latch, the spiking ladder (not needed with a float integrator), and any text injection.

## 5. Open gaps in the record
- The old track never tried gain on the output distribution or on state weighting inside a recurrent LM. The closest analogue is the logit bias.
- Nobody has measured whether the 1.5B cortex's own surprise or novelty can drive arousal. `brain/PREREQUISITES.md` lists it as "missing".
- The positive-pole appraisal read has no precedent beyond the lexicon.

Files read:
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/PREREQUISITES.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/docs/RETRACTED.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/docs/drafts/affect-results-audit/RESULTS.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/research/biology/locus-coeruleus-ne-adaptive-gain-swap.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/research/biology/interoceptive-affect.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/research/biology/affect-active-clear.md`
- The `research/findings/` files named in section 2, including `2026-07-24-P0.3-affect-state-region-6seed-GO.md`.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-03-mood-state-design.md`
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-18-mood4-1.5b.md`

I did not open the mood4 JSONs, `brain/affect.py` in full, or `mood_probe.py`. For the mood3 result I used only the section 14 summary in the design doc, not `2026-11-03-mood3.md`.

=====

# Literature review: steering tone in LMs without changing the input text, mapped onto `affect.py`

I read the repo files before searching: `affect.py`, the mood4 1.5B writeup, design doc sections 12-16, and `Cortex.add_delta`. `add_delta` clones the state and adds `alpha*delta` into the listed slots. `Affect` fills only the matrix-memory slot (index `3i+1`) of all layers, and `modulate` is called before the turn is read. I did not open `mood_probe.py`, the mood3 table or the JSONs. I did not verify the call order inside `mood_probe.py`.

Several source pages did not give the numbers asked for (noted per item). Everything marked "not in the source" is unverified.

## Per method

**ActAdd (Turner et al. 2023).**
- **Where the vector is taken:** the residual stream at a middle layer. The layer is grid-searched per vector: "Love" works best around layer 6 and the abstract "Conspiracy" vector around layer 23 (GPT-2-XL, 48 layers).
- **How it is built:** one contrast pair of prompts (for example Love minus Hate), with the difference of their layer activations added at the front positions of the user prompt. A counterbalanced negative prompt is better than the positive alone. No labelled data and no optimization.
- **Magnitude:** coefficient c is grid-searched. The paper says 3 to 15 is typical, with c=1 in the wedding demo. It cannot be raised indefinitely before general competence is lost.
- **Size dependence:** the paper only argues that its cost falls as model size grows (124M to 2.7B). It reports no sentiment-versus-size result.
- **Fluency cost:** always paid. Sentiment steering on OPT-6.7B raised a disfluency score from 8.95 to 24.2 (roughly 1.5x to 3x worse than unsteered).
- **Sentiment results (IMDb, n=1000):** negative to positive reached 0.56 (OPT) and 0.67 (LLaMA-3-8B), against 0.445 unsteered. Positive to negative was much weaker: 0.43 and 0.27. The authors attribute part of the LLaMA-3 gap to alignment post-training.
- **Source:** [arxiv 2308.10248](https://arxiv.org/html/2308.10248v4). I did not read the last 33k characters (appendices).

**CAA (Rimsky/Panickssery et al.).**
- **Where the vector is taken:** the residual stream, at the answer-letter token. It is added at every position after the prompt, including generated tokens.
- **How it is built:** mean difference over hundreds of contrast pairs (290 to 1000 per behavior).
- **Layer:** layer 13 is best for Llama-2-7B. It is usually 14 or 15 for 13B, so the best layer shifts slightly deeper with size. That is about 40% of depth.
- **Magnitude:** sweeps use plus or minus 1. Open-ended examples use plus or minus 2. Larger multipliers degraded text quality, so the authors capped them.
- **Size dependence:** the best layer shifts deeper with size. The snippets I got report no systematic scaling of effect.
- **Other:** MMLU is roughly preserved. Fluency falls at larger multipliers.
- **Source:** [arxiv 2312.06681](https://arxiv.org/html/2312.06681v4).

**Representation Engineering (Zou et al. 2023).**
- **What I could confirm:** reading vectors come from PCA over stimulus sets of 5 to 128. Emotion is one of the concepts extracted. Control operations include linear combination, piece-wise and projection. LoRRA low-rank adapters are an alternative.
- **What I could not confirm:** the emotion layer range, coefficient and model sizes. The fetched text did not reach the emotion section. My recollection (unverified) is that RepE controls over a mid-to-late band of layers rather than one layer.
- **Source:** [arxiv 2310.01405](https://arxiv.org/abs/2310.01405).

**Steering reliability and the number of pairs.**
- Tan et al. 2024 find steerability is mostly a dataset-level property. Per-sample steerability varies widely, and several datasets push the opposite way on about 50% of inputs ([arxiv 2407.12404](https://arxiv.org/html/2407.12404v2)).
- Hao et al. 2025 report diminishing returns in the number of contrast samples at about 80. Larger models are more resistant to steering-induced degradation. Steering hurts perplexity on almost all topics ([arxiv 2505.03189](https://arxiv.org/abs/2505.03189)). This is the only size-dependence claim I found, and it covers degradation, not effect size.
- The Qwen3 emotion-steering post scales strength as a fraction of the layer's residual norm (plus or minus 0.02 to 0.1), but only for 32B and 235B.

**RNN and state-space steering.**
- The one directly relevant paper is Paulo et al., "Does Transformer Interpretability Transfer to RNNs?" ([arxiv 2404.05971](https://arxiv.org/html/2404.05971v1)). It tests Mamba-2.8B and RWKV-v5-7B.
- **State steering:** CAA's contrastive mean difference, taken over the recurrent state, added once at the last token before the response. Vectors are not normalized.
- **Magnitude:** multipliers are usually between -3 and 3. The RWKV story steering used 0.5, 1 and 2, and it moved content toward French, hate or love.
- **Layers and vector type:** middle layers have the largest effect for activation steering. The paper does not say which layers were used for state steering, or whether the WKV matrix or the token-shift state was used.
- **Combining:** state plus residual steering is sub-additive, because residual steering already changes the state.
- **Gaps:** no size sweep, no fluency number, no sentiment-specific result. I found no other RWKV or Mamba steering work. Initial-state tuning (RWKV state tuning) exists, but it is gradient-trained, not a hand-built vector.

**Gain-type and multiplicative modulation.**
- Scaling attention weights is equivalent to adding a pre-softmax bias. Multiplicative attention steering exists: PASTA (Zhang 2023, from the search snippet only) and Guardieiro 2025. I did not open either.
- Activation Scaling (Stoehr et al. 2024) learns one scalar per component and location. It is comparable to an additive vector at far fewer parameters, but generalizes worse across templates ([arxiv 2410.04962](https://arxiv.org/html/2410.04962v1)). It is tested on GPT2-XL and Pythia-1.4B, but only on synthetic short tasks.
- Conceptor steering replaces addition with a soft projection ([arxiv 2410.16314](https://alphaxiv.org/abs/2410.16314)).
- I found no study of gain steering for sentiment, and none on RWKV.
- A pure scalar gain has no direction, so it cannot make tone positive or negative by itself. It can only amplify or attenuate what is already present.

## What this implies for our result

1. **Placement and timing are our biggest departure from the precedents.** `modulate` pushes the delta in before the turn is read. The reply is generated only after the question tokens have rewritten the state and the per-head decay has acted on the push. The state-steering paper injected at the last token before the response. CAA adds at every position through generation. A one-shot push before the question is the weakest placement in the literature. Check this in `mood_probe.py`, which I did not read.
2. **We add at all 24 layers.** Every precedent steers a middle band (ActAdd, CAA, RepE, the state-steering paper). Adding everywhere compounds both the effect and the fluency cost. This is a plausible reason the NLL rule capped GAIN at 16 while the tone effect stayed small. A band such as layers 8 to 16 should give more tone per unit of NLL.
3. **Our GAIN is in raw state units.** The CAA-style approach is to scale per layer, as a fraction of that layer's typical norm. A fixed raw 16 hits different layers very differently. The same applies to the appraisal read: "deep half" at 24 layers is layers 12 to 23, which is not the mid band the literature favors.
4. **Our axis comes from very few contrasts.** It uses 8 positive and 8 negative short phrases in "Body:" speaker framing, and they are topic-laden (hug, burn, cold). CAA uses 290 to 1000 pairs, and returns diminish near 80. ActAdd gets by on one pair, but it grid-searches the layer and coefficient on that pair. Heterogeneous primaries carry topic content, which fits "32 intrudes primary content". They also fit 2/6 positive appraisal at 1.5B. Tan et al. add that steerability is dataset-specific.
5. **Positive being harder than negative has a precedent.** In ActAdd's sentiment table, positive-to-negative is weaker than negative-to-positive.
6. **Non-monotonic behavior across 0.1B, 0.4B and 1.5B is expected.** The best layer shifts with size, so a layer, scale and read-out frozen at 0.1B should not be expected to transfer.

## Recommendation: what is most likely to move tone at 1.5B at low fluency cost

All of these are dev-only experiments, switched off by default.

1. **Move the injection point (cheapest change).** Apply `add_delta` after the user turn has been read and before the first reply token. Optionally re-apply every N reply tokens, as CAA does at each position. This is the closest match to the only RWKV precedent. It needs a small change to where `modulate` is called.
2. **Restrict the band and normalize per layer.** Fill only mid layers of `axis_delta` (try 8 to 16 of 24, a sweep over the band). Set the magnitude as a fraction of each layer's matrix-memory norm and sweep the fraction against the NLL ratio. The state-steering paper used 0.5 to 2 on unnormalized mean differences, but that scale does not carry over to our units. Tune the band and fraction on 1.5B dev only.
3. **Rebuild the axis from more, matched contrasts.** Use roughly 40 to 80 pairs in the conversation register with the same frame and flipped sentiment, mean-centered to remove topic. This stays within the declared body scaffold if the primaries list is extended and documented in `SCAFFOLDS.md`. Use the same axis for the appraisal read. Re-select the appraisal layers at mid depth on 1.5B dev events.
4. **Other state slots.** The token-shift slots (per layer, the previous token's layer input) are a residual-like vector for one position. A CAA-style residual difference vector for the last token could be added there. This is untested in RWKV, so treat it as an exploratory arm. The 2404.05971 paper suggests combining state and residual steering gives less than the sum, so do not expect gains from stacking.
5. **Gain-type variant (secondary, speculative).** A signed scalar mood cannot express direction by gain alone, so use a mood-congruent amplifier. Scale the component of the state along the valence axis by (1 + k·|mood|), as a projection-style operation, as conceptor steering does. This would not inject a push when the context has no valence, so the effect on the affect-free test questions would be small. Other gain levers in RWKV7, such as the decay `w` as a per-head forgetting gain, would affect arousal-like style and persistence, not valence. I found no literature support for any of these. They are hypotheses to pre-register, not expected wins.

Raising the sampling temperature is a host-side sampler change. It would move variability, not tone, and it would need to be declared as a scaffold.

Sources:
- [Activation Addition (Turner et al.)](https://arxiv.org/html/2308.10248v4)
- [CAA, Steering Llama 2](https://arxiv.org/html/2312.06681v4)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Does Transformer Interpretability Transfer to RNNs?](https://arxiv.org/html/2404.05971v1)
- [Generalization and Reliability of Steering Vectors](https://arxiv.org/html/2407.12404v2)
- [Patterns and Mechanisms of Contrastive Activation Engineering](https://arxiv.org/abs/2505.03189)
- [Activation Scaling for Steering](https://arxiv.org/html/2410.04962v1)
- [Conceptor steering](https://alphaxiv.org/abs/2410.16314)

=====

All numbers below come from the six 1.5B JSONs in `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/results/2026-11-18-mood4-1.5b/`. "Push" means GAIN (16) × mood. Lex is lexical sentiment per 100 words. My scripts are in the scratchpad (`a1.py`, `a2.py`).

## 1. Why held-out positive appraisal fails at 1.5B

The axis ranks events correctly. What fails is the zero point, and the zero point, not the ranking, is why C1 fails.

- **Ranking is mostly fine.** Positive beats negative with AUROC 1.00 on every seed, dev and held-out. Positive beats neutral with held-out AUROC 0.88–1.00. Dev AUROC is lower: 0.69 (s43), 0.72 (s102), 0.81 (s101), 0.84 (s44).
- **Neutral does not sit at 0.** It averages −0.06 to −0.13 per seed, even though the zero was set on 6 reference sentences. The mean over 12 dev and held-out neutral events is about −0.11.
- **Positive scores are compressed against that offset.** Seed-level positive means (dev / held-out) are:

  | Seed | Dev | Held-out |
  |---|---|---|
  | 42 | +0.29 | +0.23 |
  | 100 | +0.20 | +0.23 |
  | 43 | +0.03 | +0.10 |
  | 44 | +0.05 | +0.04 |
  | 101 | +0.04 | +0.04 |
  | 102 | +0.01 | +0.05 |

  A positive event therefore scores about 0 to +0.1, and a sign test at 0 gives 0.5–0.69 on those seeds (sign test at 0 over all 16 positive events: 0.56–0.69 on s43/s44/s101/s102).
- **The axis is dominated by the negative pole.** Negative means are −0.27 (s42), −0.41, −0.48, −0.45, −0.52 and −0.54 (s102). Seeds with a clear positive mean (s42, s100) are the only ones where the poles are roughly symmetric.
- **Moving the threshold to the pos/neg midpoint fixes the signs.** Midpoints run −0.12 to −0.26 on every seed except s42 (+0.01). A threshold at the dev midpoint gets held-out positive 8/8 and held-out negative 8/8 on all six seeds.
- **Recentred, mood magnitudes become comparable.** Computed from the stored induction lists, with the dev midpoint subtracted:
  - m_pos goes from +0.07, +0.05, +0.20, +0.10, −0.04 (s43, s44, s100, s101, s102) to +0.23, +0.23, +0.30, +0.29, +0.17. s42 is unchanged.
  - m_neg shrinks from −0.35 … −0.45 to about −0.20 … −0.25.
  - So the pos/neg mood asymmetry mostly disappears.
- **It is content-driven, not noise.** The per-event pattern correlates 0.77 across seeds (mean over seed pairs, 16 positive events), and seed-mean centring does not change that. The axis reads interpersonal warmth and affiliation, not generic positive:
  - High across seeds: "so happy to talk with you again" (+0.30), "grandmother sent cookies" (+0.27), "you made me laugh" (+0.28), "you did a wonderful job" (+0.21).
  - About zero: "storm passed and the garden survived" (+0.05), "gave me the job" (+0.01), "whole team clapped" (+0.01).
  - Negative: "passed the test with the highest score" (−0.17).
  - Achievement and relief events are not read as positive. Negative events (loss, threat, hostility) all score −0.33 to −0.59 on mean.
- **Social vs bodily:** I cannot separate this from the JSON. The appraisal events are all linguistic and social, and per-primary contributions are not stored. Primary composition does not explain it: s42 and s100 draw both bodily and social positives, and so do the weak seeds.
- **Norm:** also not testable here. Only `scale` is stored implicitly, and it is not in the JSON.
- **Likely cause:** the opponent-pole axis is P̂ − N̂ with each pole centred on neutral body states, and the 'Body' primaries differ in speaker and register from the 'User' events. The zero is also fitted on 6 sentences. The result is a roughly constant offset plus a negative pole that is far larger than the positive one.
- **Cheap test:** put the zero at the midpoint of the pos and neg primaries' own projections, or use per-pole scales. Both are legitimate scaffold quantities, and neither needs host text.

## 2. Does the positive push work where m_pos is clearly positive (s42 and s100)?

Weakly, yes.

Pooled s42 and s100, 48 paired replies, bootstrap CIs:

| Contrast | Mean | 95% CI |
|---|---|---|
| positive − off | +1.50 | [+0.43, +2.64] |
| positive − null | +1.13 | [+0.18, +2.13] |

For the weak seeds (96 pairs), positive − null is +0.65 [−0.05, +1.34] and positive − off is +0.49 [−0.01, +0.99].

Per seed, positive − null is +1.65 [+0.41, +3.02] on s42 and +0.61 [−0.86, +2.13] on s100. Only s42 clears the null CI on its own. s100 is positive in sign only. The two single nulls on s100 disagree: positive − null is −0.04 against one and +1.27 against the other.

Pooled over all six seeds, the positive minus negative lex contrast is +2.38 [+1.63, +3.14]. The matched random-direction contrast is about 0 (−0.05 and +0.33). The true axis therefore has a real directional signal. It is smaller than the sampling noise per reply: reply-level SD of the diffs is 1.8–3.9.

## 3. Effect size vs mood magnitude

The dose-response is clear for the negative arm and weak for the positive arm. Seed-level regression of lex shift on push, n=6:

| Arm | Slope | r |
|---|---|---|
| positive vs off | +0.40 per push unit | 0.88 |
| positive vs null | +0.24 | 0.59 |
| negative vs off | +0.39 | 0.66 |
| negative vs null | +0.66 | 0.94 |

Seed by seed (push = GAIN × m, lex shift vs the mean of the 2 null directions):

| Seed | Push + | Push − | Pos − null | Neg − null |
|---|---|---|---|---|
| 42 | +5.2 | −3.8 | +1.65 | +0.07 |
| 43 | +1.1 | −5.6 | +1.88 | −1.54 |
| 44 | +0.8 | −6.6 | +0.77 | −2.09 |
| 100 | +3.1 | −5.5 | +0.61 | −0.92 |
| 101 | +1.5 | −7.0 | +0.37 | −2.34 |
| 102 | −0.7 | −7.1 | −0.41 | −1.73 |

- **s42 is the cleanest dose-response case.** It has the largest positive push and the smallest negative one. Its negative arm is at the null (+0.07) and its positive arm clears it.
- **The negative "success" is mostly size, not skill.** Neg − null grows with |m_neg|. At |push| ≥ 5.5 the shift is −0.9 to −2.3 per 100 words.
- **The per-unit effect is small.** About 0.4–0.66 lex per push unit. GAIN 16 is capped by fluency, so reaching a ±3 lex shift would need a push of 8–10 or more.
- **Random pushes hurt fluency far more than the true axis.** Mean NLL, true axis vs random direction at equal norm:

  | Seed | Pos / Neg (true) | Rand pos / Rand neg |
  |---|---|---|
  | 43 | 0.75 / 0.94 | 0.80 / 1.22 |
  | 44 | 0.73 / 0.96 | 0.70 / 1.50 |
  | 101 | 0.74 / 1.09 | 0.67 / 1.59 |
  | 102 | 0.84 / 1.10 | 0.81 / 1.46 |

  Off sits at 0.72–0.80. The matched-norm null is not inert as a perturbation, so the "null does nothing" instrument claim holds for lexical mean only, not for fluency or reply-length variance. This also means a negative push of the same size on the true axis is cheaper than a random one.
- **A bigger gain is not simply available.** The dev table shows GAIN 16 gives NLL 1.16 and GAIN 32 gives 1.38.

## 4. Twenty paired replies

I read 20 replies from s42, s100, s102 and s43, all questions with k=0.

- **Everything diverges.** Word-sequence similarity to the off reply is 0.27–0.30 for positive and negative on s42, and 0.18–0.70 elsewhere. It is lowest for random-negative (0.14–0.19). Any perturbation re-rolls the sampled continuation. The 'mood' effect is a bias in what the re-roll picks, not a lesion that edits a fixed reply.
- **Negative push, on s42, s43 and s102.**
  - It reframes the same content toward difficulty or loneliness:
    - "It's not easy. I have to get up at 5:30 am every day to go to school."
    - s102: "It's not a good day. I'm feeling really isolated and anxious."
    - s100, winter: "like a giant, frozen monster that comes out of nowhere and scares everyone", against off's "cozy blanket".
  - It also produces some refusals and truncation: "I can't help you with that.", "You miss the bus.", "I'm sorry, but I don't have access…".
  - Replies are shorter and flatter on s42 and s102.
  - On s43 the push barely changes the off refusal ("I'm sorry, but I don't have the ability…") and shifts weather text only a little ("pretty much the same as any other time of year").
- **Positive push, on s42 and s100.**
  - It leaves the structure intact and moves one clause: "noisy and uncomfortable, but it's also a great way to see different parts of the city", and "It's a great way to relax and recharge."
  - It also yields terse or odd outputs: s100 gave "It's the weekend! 🎉" at NLL 1.34, and s102 gave "I can't wait for news."
  - The positive effect is a clause-level warmth shift, not a mood change in the whole reply.
- **Random negative pushes** produce fragment and non-sequitur outputs, such as "I can't answer that.", "What is it like to be a young child who has to spend the day at home alone?" and "The following is a list of common words and their meanings: -". Random positive pushes often produce the same text as off (s102) or a plain content change.
- **Net:** the true-axis push shifts tone mainly as a negative bias plus an occasional refusal or short reply. It does not give a graded, readable mood across the reply.

## What follows

- **Fix the zero or per-pole scale first.** Re-centre on the primaries' own midpoint (or give each pole its own scale), then rerun the C1 and C2 check offline. It needs one 1.5B appraisal pass per seed, not a full probe. Judge the positive arm only after that.
- **Compare modulation methods on equal pushes.** The additive push has a measured slope of about 0.4–0.66 lex per push unit and cannot exceed GAIN 16 without costing fluency. A gain-type action should be compared on a matched effective push, not matched GAIN.
- **Report the null with NLL and reply length as well as lex.** Random nulls change length and fluency much more than the true axis does.