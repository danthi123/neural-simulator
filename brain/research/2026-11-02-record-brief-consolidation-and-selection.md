# Record brief: sleep consolidation (A) and semantic episode selection (B)

Mined 2026-11-02 from the old track (`/home/dant123/Projects/sim`); paths are relative to that repo, `F/` =
`research/findings/`. `tools/before_you_build.sh` was run for A and B (nothing blocking). No cited finding is in
`docs/RETRACTED.md`. Old substrate = small spiking pools or numpy readouts; lessons are marked MECHANISM (transfers)
or SUBSTRATE (does not).

## A. Sleep consolidation into cortex weights

### A1. Prior results

| finding | tested | verdict | root cause / operating point |
|---|---|---|---|
| `F/2026-08-26-cls-interleaved-consolidation-6seed-GO-replay-defeats-catastrophic-interference.md` | fast store replayed interleaved with new set B into a slow over-capacity associator (60 items, dim 24, lr 0.02) | GO 6/6: A retention 0.939 vs 0.317 no replay, 0.133 shuffled replay, 0.278 extra-B steps | replay CONTENT is the cause; AC1-AC6 control set; test read by the slow store only |
| `F/2026-08-20-sleep-replay-consolidation-6seed-GO-on-the-principle-host-hippocampus-scaffold.md` | self-generated replay from stored engrams | GO 6/6: N=10 0.55 vs 0.13 (scramble 0.05) | host-scaffold store; per-seed spread 0.20-0.90 |
| `F/2026-08-09-teacher-loop-sleep-replay-budget-sweep-more-replay-does-not-reach-ceiling-NEGATIVE.md` then `F/2026-08-11-sleep-replay-bdsp-declamp-lifts-continual-retention-past-055-cap-clamp-artifact.md` and `F/2026-08-09-bound-trap-bdsp-wmax6-silences-teacher-loop-reservoir-declamp-recovers-75pct-of-N20-retention.md` | 64x more replay; then widening a weight clamp | more replay flat; the "0.55 cap" was a weight bound (N=20: 0.05 to 0.742 de-clamped, 6 seeds). The de-clamp replay result is a 2-seed smoke only | the lever was a saturation bound, not fidelity or quantity |
| `F/2026-08-09-weight-protection-does-not-stack-on-self-replay-honest-negative.md` | EWC/SI Fisher gate on top of replay | NEGATIVE, 6 seeds: protect 0.700, scramble 0.700, replay 0.683 | replay already reconsolidates the whole readout; the gate acted as a global brake |
| `F/2026-08-11-metaplastic-acquisition-continual-learning-6seed-NOGO-mechanism-real-but-subthreshold.md`, `F/2026-08-11-benna-fusi-multitimescale-chain-does-not-beat-single-var-metaplasticity-6seed-NEGATIVE.md`, `F/2026-08-11-continual-combine-disjoint-codes-plus-consolidation-does-NOT-compound-6seed.md` | per-synapse consolidation state; multi-timescale chain; adding disjoint codes | single variable 0.257 vs 0.120 (1/6 strict GO); chain 0.422 vs single 0.443; combined 0.854 vs 0.839 | residual named as fact-code interference, not timescale; oldest fact stays unprotected |
| `F/2026-08-11-continual-retention-interleaved-generative-replay-N50fact-NEGATIVE.md` | generator replays old facts, interleaved with new | N=50: 0.72 vs plain self-replay 0.90, but newest-fact acquisition 0.65 vs 0.05 | stability/plasticity trade |
| `F/2026-08-09-prioritized-replay-fixed-k-fails-at-scale-compute-half-NEGATIVE.md` | replay only k=6 by forgetting risk | N=50: 0.627 vs full 0.893, random-k 0.547 | signal real; fixed k below coverage floor |
| `F/2026-08-03-replay-cortical-consolidation-calibration-NO-GO.md` to `F/2026-08-06-replay-cortical-consolidation-v5-learned-CA1-to-cortex-reinstatement.md`, `F/2026-08-06-replay-cortical-consolidation-v5-sfa-eviction-closes-interference-wall-order-control-is-next.md`, `F/2026-08-06-replay-cortical-consolidation-v6-multiseed-NO-GO-operating-point-overfit.md` | spiking CA1 to cortex consolidation | v5 GO on one seed; v6 tuned on 2 seeds, fails 0/3 dev seeds (false recall 0.46-0.50 vs 0.09-0.12) | reinstatement must be memory-specific; interference control was an overfit operating point |
| `F/2026-09-09-order-consolidation-v10-indegree-law-scale-invariant-6seed-GO.md` | same, region doubled | 6/6 GO after an in-degree law | a fixed constant did not survive a scale change |
| `F/2026-09-25-sleep-replay-capture-rc-GO-6seed.md`, `F/2026-09-25-sleep-replay-capture-r2-NO-GO-6seed.md`, `F/2026-09-25-sleep-forgetting-interference-fi-6seed-NO-GO.md` | told-once fact captured by one night's replay; forgetting by constant downscaling (r2) or load-keyed renormalisation (fi) | rc GO 6/6; r2 0/6; fi 3/6 | constant downscaling fails; fi: weak fact lost before sleep, re-telling matched an older duplicate block |
| `F/2026-05-20-7th-arc-ABLATION-localised-OVER-CONSOLIDATION-is-primary-culprit-6th-arc-confirmed-LOCAL-OPTIMUM.md` | 20 vs 50 replays per tag | 50: 0.274 vs 0.458 | over-consolidation hurts |

Not found: no measured savings, gist or schema result (sleep transitive inference was UNDEFINED, `F/2026-10-01-sleep-transitive-inference-6seed-UNDEFINED.md`). Gist/schema are design only:
`F/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md` (gates WM1-WM11) and
`research/biology/gist-detail-graded-forgetting.md`. Treat both as new.

### A2. What transfers

MECHANISM, transfers:
- Interleaved replay of ALL stored items into the slow store, tested with the hippocampus off. This is your adapter plan and has the strongest record support (cls 0.939).
- Shuffled, extra-exposure and replay-off controls make retention attributable.
- Skip EWC/importance gating when every episode is replayed (0.700 = 0.683 = random-synapse control).
- Prioritise by strength, but budget as a fraction of stored items, not a constant k.
- Cap replays per episode (7th arc).
- Forgetting must be keyed to later learning, not a constant (r2 0/6).
- A re-telling should strengthen the same episode; the design doc calls appending a new block the RAG-like failure. `hippocampus.store` appends today.

SUBSTRATE: CA1/SFA/order-STDP details, in-degree numbers, DA tag-capture gains, the 0.55 cap.

My inference, not in the record: with a pretrained LM, forgetting also means losing general language. The old nets had none to lose, so add a generic-text perplexity control.

### A3. Traps already hit

1. A clamp posing as a method limit: 97% of dW was the clamp (`F/2026-07-31-gap5-tuned-point-is-INSIDE-the-bound-trap-97pct-of-dW-is-the-clamp.md`). Before blaming replay, compare adapter norm/rank/LR clip with the change needed.
2. A read that writes: recall overwrote the store while reading it (drift +1.28 vs 0.000 frozen; READ FIRST header of `F/2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md`). Several GOs flipped after read-isolation fixes (`docs/RETRACTED.md`, 2026-09-02 rows). Eval must never train, and `use` boosts must not leak into test arms.
3. Tuning on 2 seeds (v6). Fix on dev seeds, score held-out seeds, six before any claim.
4. Hippocampus still on at retest. The v5 signature was recovery with it off, and no-sleep, reinstatement-lesion and plasticity-off arms at exactly 0.000.
5. Measure false recall (wrong cue returns the fact); it killed v6.
6. Operating points do not survive a scale change (v10); derive gain and thresholds per model on dev data.
7. Generic capture: after consolidation one engram swallowed untaught inputs (Jaccard 0.90 vs 0.44), `F/2026-09-01-board91-memory-separator-readfix-popstate-reset-NOGO-relocalizes-to-consolidation-induced-generic-capture.md`. A consolidated name that answers every "who" question is the same failure; your permanent episodes (strength 3+) are at risk.

### A4. Recommended first experiment

"Distilled reinstatement": a low-rank adapter is trained so that cortex+adapter, with NO hippocampus and no reinstated state, gives on the cue the same next-token logits as base cortex WITH the top-1 delta reinstated at gain 4 (KL on the answer tokens). This needs no stored text, so it respects rule 1. (My design.)

Protocol, following `research/runners/_cls_interleaved_consolidation_derisk.py` (371 lines, numpy; copy structure, not code): set A = 8 facts (including a friend's name) consolidated first, then set B = 8 new facts. Arms: (1) interleaved replay A+B; (2) sequential B only; (3) shuffled (episode i's delta paired with cue j); (4) exposure-matched (B only); (5) hippocampus-on, adapter-off reference; (6) top-k by `uses`/strength vs random-k at 25% of stored. Measures: A retention with hippocampus lesioned; B acquisition; false recall on 8 unrelated and 8 confusable cues; generic-text perplexity change; `eval/weekly.py` fluency. Pre-register the cls thresholds (interleaved retention >= 0.80, lesion arms <= 0.60, new learning >= 0.80) plus a perplexity bound set on dev. Reusable: `tools/lab.py` (`attributable_to`, `shuffle_preserving_marginal`), `tools/verdict.py`. Run 0.1B first, then 0.4B; six seeds (42/43/44/100/101/102) varying adapter init, fact set, order.

Prerequisites: (a) a differentiable forward through RWKV-7 (the `rwkv` pipeline in `cortex.py` looks inference-only; confirm); (b) re-tellings merge into one episode (`uses`/strength already persist in `brain.py`); (c) a fact bank large enough for six seeds; (d) the generic-perplexity probe; (e) `sleep.py` does not exist yet.

## B. Semantic episode-selection key

### B1. Prior results

| finding | tested | verdict | root cause / operating point |
|---|---|---|---|
| `F/2026-07-11-content-addressable-retrieval-needs-LEARNED-keys-the-arc-converges-on-deep-credit-learned-representations.md` | retrieve by similarity of fixed reservoir states | content - shuffle = -0.015 nats (6 seeds): keys carry no context identity | a fading recurrent state is a bad key |
| `F/2026-07-11-LEARNED-keys-make-content-addressable-retrieval-load-bearing-first-positive-derisk-of-the-rung2-frontier.md` | same, keys from a trained reservoir | content - shuffle -0.10 to -0.42 (3 seeds), largest at deepest context | learning the key is the lever |
| `F/2026-07-15-discourse-marker-routing-is-semantic-nearest-intent-plus-novelty-threshold-NOT-deep-credit-dispatch.md` | held-out synonyms routed to intent | plain 1-NN over semantic codes: 1.000 every seed; deep net added nothing | synonyms generalise by proximity in a semantic code; a familiarity threshold is needed for out-of-set |
| `F/2026-08-19-replay-pattern-separation-DG-separator-keeps-similar-memories-discriminable-6seed-GO.md` | sparse-expansive separator, two memories at Jaccard 0.60 | GO 6/6; lesion gives identical codes; false recall halved (2:1), not removed | ~27% of units still shared |
| `F/2026-05-31-DG-biologization-FUNDAMENTAL-BOUNDARY-separation-vs-reliability-tradeoff-no-DG-size-threads-both.md` (read with its REFINEMENT header) | DG size sweep | separated 0.27 but unstable 0.24, or stable 0.59-0.82 but unseparated 0.66-0.76 | separation vs reliability tradeoff; the "fundamental" wording was retracted in the refinement. Instability was spiking k-WTA noise; deterministic top-k was stable |
| `F/2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md` | DG to CA3 cleanup, correlated codes | matched filter 1.000 on full and 40%-kept cues; spiking loop 0.083/0.100 (chance 0.062) | codes correlated (cos ~0.81); DG read not reproducible |
| `F/2026-07-08-riii-DEFINITIVE-ca3-partial-cue-completion-fails-across-param-space.md`, `F/2026-08-10-ca3-point-neuron-attractor-completion-trilemma-NEGATIVE-redirect-dendritic-plateau.md`, `F/2026-08-08-episodic-CA3-completion-CLOSED-within-assembly-attractor-potentiation-loadbearing-WTA-still-negative.md` | recurrent partial-cue completion | held-out cells fire 0.00 over the grid; strong completion is non-specific (permuted cue overtakes); within-assembly potentiation worked (seed-42 smoke) | magnitude vs specificity trilemma on point neurons |

### B2. What transfers

MECHANISM:
- Your layer-0 key failing at scale is the old "fixed state is a bad key" result (both 07-11 findings). The record-backed fix is a learned or task-shaped key, judged against a shuffled-key control.
- Cosine on mean-centred keys is a linear matched filter: perfect on correlated codes and 40%-kept cues where recurrent loops sat at chance (06-10). Keep it; test whitening (ZCA on generic text) as the cheap upgrade.
- Paraphrase tolerance and separation trade off; report one curve.
- Proximity in a semantic code plus a familiarity threshold for "none of these" is validated (07-15); that is your `FAMILIARITY` + `MARGIN` design, so fit them per model on dev data.
- Every completion test needs a wrong-cue control.
- `tanh(strength)` weighting lets consolidated episodes win everything (A3.7); cap it.

SUBSTRATE: spiking k-WTA instability, CA3 weight/density sweeps, DG granule counts.

### B3. Traps already hit

1. A partial cue overlapping the full set (cos ~0.5) read as "completion" (riii). Same-word is the easy case; paraphrase and partial need their own arms.
2. Two DG sizes read as a law, then partly retracted; sweep first.
3. Sigma-standardising blew up near-zero-variance units (`F/2026-08-08-teacher-loop-sparse-gated-readout-allocation-catastrophic-forgetting.md`: raw top-k Jaccard 1.00, centred deviation 0.20-0.33). Whiten with a floor; compare to centring alone.
4. Over-sparsifying kills the shared block paraphrase needs (`F/2026-07-02-emergent-structure-from-experience-research-gate.md`, last paragraph).
5. Lexical match passing as semantic: `F/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md` (90%) matched cue words to tags, as your same-word 12/12 does.
6. 12 items is too few; use at least 40.

### B4. Recommended first experiment

Selection benchmark, no consolidation. >= 40 facts in template pairs differing only in a name or attribute (confusables). Cues per fact: same-words, 3 paraphrases, partial; plus 40 unrelated cues (must abstain). Keys compared, each at 0.1B, 0.4B, 1.5B: (0) current layer-0 delta; (1) deltas pooled over layers; (2) last-token hidden or state summary at the end of the cue; (3) each of 0-2 after mean-centring and after ZCA; (4) a linear projection trained contrastively on held-out paraphrase pairs (declared scaffold; train and test templates disjoint). Controls: shuffled-key (07-11 form), permuted-cue, and a reference floor (1-NN over a plain sentence embedding, never in the brain path). Report top-1 per cue type, confusable-pair confusion, abstention on unrelated, and the paraphrase-recall vs confusion curve across `MARGIN`/`FAMILIARITY` sweeps. Fit thresholds on dev facts, score held-out facts, six seeds over fact sampling and template split. Bar: beats shuffled-key at all three scales and keeps same-word recall at least as high as layer-0.

Prerequisites: a paraphrase and confusable set (teacher-written is a declared scaffold); a dev/held-out split; hidden states exposed from `Cortex.read` (today only matrix deltas); `experiments/key_variants.py` is the natural home.

