---
type: finding
status: design
claim_check: synthesis
date: 2026-09-05
mechanism: Deep-research round (external literature + RAG cross-check) for the own-voice mouth's BROAD-DOMAIN fluency wall — a ranked next-rung ladder + a banked-exhausted list; the top new build lever (delta-rule error-corrective write) is in build and the decisive objective A/B is queued
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [n/a — a research synthesis; each named lever carries the 6-seed bar when run]
lane_wall: brain-native open-ended generation (own-voice mouth) — roadmap Wall #7 / R4
external: >
  Sources surfaced + read by the DR round's research agents (arXiv ids as they reported them; not independently
  re-verified digit-by-digit by the controller). Delta-rule / erase-before-write: "Gated Delta Networks"
  (arXiv:2412.06464); RWKV-7 "Goose" (arXiv:2503.14456); "Parallelizing Linear Transformers with the Delta Rule"
  (arXiv:2406.06484). Predictive / multi-token objective: Aynetdinov & Akbik BabyLM-2025 multi-token-prediction
  study (read from PDF); Gloeckle et al. multi-token prediction; ProphetNet; Rao & Ballard 1999 (predictive
  coding, Nat Neurosci 2:79-87); tPC-RTRL (arXiv:2602.18131). Same-regime small-data: BLaLM, BabyLM-2025
  (arXiv:2511.05560); LTG-BERT (Samuel et al.); data-constrained scaling (Muennighoff et al. arXiv:2305.16264);
  Chinchilla (Hoffmann et al. arXiv:2203.15556). Architecture: Gated Linear Attention; Mamba selectivity; Hedgehog
  learned feature map.
artifacts:
  - research/findings/2026-09-05-own-voice-fluency-reaim-objective-and-capacity-ROADMAP.md
  - research/findings/2026-09-05-hippokey-hippo-ssm-content-addressable-attention-NO-GO.md
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
  - research/runners/_emerge_wkv_lm_derisk.py
verdict: >
  Deep-research round (workflow: 3 domain research agents + 1 opus synthesis, RAG-cross-checked against our own
  record) for the mouth's broad-domain wall — linattn crosses a fair trigram on simplewiki but falls below it on
  wikitext-103, the domain that retires Qwen. The external literature CONFIRMS the roadmap's objective+capacity
  levers AND adds two new BUILD levers that are NOT the banked content-addressing family and target linattn's exact
  measured failure (interference + unbounded memory-norm as content diversity rises = narrow->broad): (1) the
  DELTA-RULE error-corrective write (erase-before-write on the SAME linattn fast-weight; convergent across three
  external groups, and independently our own cheap-first #1 on 2026-07-15, never built) and (2) a short causal CONV
  prefix (the single biggest perplexity drop in BLaLM's same 10-15M-token regime, with an honest caveat that our
  own 2026-07-11 ceiling found the local-copy signal thin here). The decisive cheapest next run is the training
  OBJECTIVE on the BROAD domain: a single-variable A/B flipping only the predictive-coding auxiliary on, against
  the byte-identical wt103 linattn baseline, at horizon k=2 (a further-ahead horizon of 4 is neutral-to-harmful
  for prose at small scale per the strongest source) — QUEUED on the GPU. The delta-rule build is IN FLIGHT. A
  strong banked-exhausted list (content-addressing incl. the pre-built learnkey arm; passive multi-timescale
  retention; naive complexity-ordering curriculum; frequency down-weighting; distillation/MLM/weight-tying host
  tricks) keeps future levers from re-deriving concluded work. NO-DEFER honoured: the wall now has a ranked ladder
  of named methods, not a stopping point.
---

# Mouth broad-domain fluency: a deep-research ladder (external + RAG-cross-checked)

## The wall
The own-voice spiking mouth (retire the Qwen scaffold) is the #1 goal-blocker. Its best deployable arm, `linattn`
(normalized Hebbian fast-weight linear attention, additive write `S += v·kᵀ`, depth-2, d_model=192, ~13.5M BPE
tokens), CROSSES a fair interpolated-trigram baseline on simplewiki but FALLS BELOW it on the BROAD wikitext-103
domain. Broad-domain fluency is the capability that retires Qwen. The content-addressing direction is banked
exhausted (hippokey NO-GO 6/6). This round asked: what concrete recipes cross a strong n-gram in our exact
~10-15M-token regime, and which are portable to a spiking/one-brain substrate?

## Method
A workflow fan-out: three domain research agents (BabyLM/small-data recipes; architecture/inductive-bias;
predictive-coding objective) + one opus synthesis agent that RAG-cross-checked every candidate against our own
findings and filtered for one-brain/spiking plausibility. This is the deep-research-at-wall step the record was due
(the mouth has hit the broad-domain floor with >5 levers).

## The ranked next-rung ladder
1. **Predictive-coding auxiliary OBJECTIVE** (`--pred-aux-weight`, already built). Further-ahead auxiliary
   read-out head(s) on the shared linattn hidden state predicting token t+k, strictly causal, discarded at
   generation. Tune k=2 only (drop the offset-4 head) and weight toward equal (~1.0). The roadmap's #1, now
   sharpened on the horizon/weight. Bio: Rao & Ballard multi-horizon cortical prediction; a predictive objective
   can even train online without BPTT (tPC-RTRL). **DECISIVE RUN — QUEUED** (see below).
2. **CAPACITY** (`--d-model` 384/512, `--n-layers`) + **weight-decay ~0.1** (a ~100x raise as an anti-overfit
   regularizer against narrow-domain memorization). wt103 was only ever run at d_model=192; capacity must be
   tested at 384+ to separate capacity-bound from architecture-bound. Weight-decay 0.1 is UNTESTED in our record.
3. **DELTA-RULE error-corrective write on linattn** (BUILD — **IN FLIGHT**). Replace the additive write with
   `S_t ← S_{t-1}·diag(w) + β·(v − S_{t-1}·φ(k))·φ(k)ᵀ` — retrieve the value bound to the incoming key, subtract
   it, write only the residual, with learned per-channel decay. Error-corrective (Widrow-Hoff), erase-before-write,
   bounded norm — directly attacks linattn's interference/unbounded-norm failure. **NOT** a new content-addressing
   key (a write-rule fix to the same fast-weight). Doubly-supported: convergent external evidence + our own
   2026-07-15 (scoped #1, never built) and 2026-07-13 (precondition "structured codes + real scale" now met by
   linattn's learned embeddings). Bio: local, weight-transport-free, short-term-plasticity-realizable.
4. **Short depthwise causal CONV prefix** (kernel 2-4) on linattn's q/k/v or embeddings (BUILD). Near-zero-param
   exact local n-gram/copy that a first-order recurrence is weak at; the single biggest perplexity drop in BLaLM's
   same regime. **Honest caveat:** our 2026-07-11 ceiling found the local induction/copy signal thin at this scale
   — so this ranks below delta-rule and must beat that prior thin-signal result to count.
5. **Data-dependent decay GATE on linattn** (BUILD; GLA/Mamba selectivity). Learned input-dependent per-channel
   state decay. PARTIALLY BANKED: generic input-magnitude gating already bought a small win (6/6) but the
   content-selective part did NOT transfer on one-hot codes — a data-dependent gate must beat that ceiling.
6. **Heterogeneous broad-domain corpus MIX** (data lever). Multi-source mix at fixed token budget. Tension with
   2026-09-01 (broad plateau = starvation) vs the wt103 run (more+harder data did not deliver fluency at d=192) —
   run only after capacity is raised.
7. **Hedgehog learned feature map** (BUILD). Trainable MLP feature map recovering softmax's spiky weighting.
   `bio_plausible_one_brain = FALSE` (a host-training trick, not spiking-portable) — lowest priority for us.

## Banked exhausted (do NOT re-propose — the refuted-mechanism guard applies)
- **Content-addressable attention / "richer key"** (assoc, assoc_t, hippokey) — all lose to the trigram at depth,
  the richest key the worst. **INCLUDES the pre-built `--recurrence learnkey` arm** (same family; do not run as a
  live lever despite being staged). The 2026-09-04 wt103 finding named learnkey as its next step; the 2026-09-05
  roadmap supersedes that and banks the whole direction.
- **Passive multi-timescale / heterogeneous fixed-τ RETENTION as a language lever** — 6-seed 0/6, "retention !=
  prediction" (2026-07-13). This down-ranks the external spiking-timescale recipes (SiLIF learnable log-τ,
  resonate-and-fire, forget-gate bias) AS FLUENCY LEVERS; they remain the spiking PORTING mechanism to apply once
  a host-side recipe is validated, not an independent win.
- **Naive sentence-complexity/length curriculum ordering** — external replicated negative + no win in our own
  scoping. (Objective-curricula / readability-ordering are the positive variants; the reverse-MTP schedule is
  folded into rung 1.)
- **Frequency down-weighting (devnorm-style) for generation** — 2026-07-12: common tokens are locally predictive,
  so down-weighting them HURTS next-token CE.
- **One-brain-LOW-VALUE host tricks** — same-corpus teacher-ensemble distillation (two networks = a scaffold);
  masked/hybrid-MLM + span-corruption (need bidirectional/encoder-decoder shape a causal mouth lacks — only their
  causal half ports, = rung 1); weight-tying + Muon optimizer (non-spiking host-training tricks); EMA mean-teacher
  (mixed-to-negative on grammaticality).

## Derived — numbers (external + our-record, all cited; none a new measurement)
<!--derived: every number below is a direct read of an EXTERNAL source named in `external:` or of one of our own
cited findings/artifacts; none is a new measurement of this document -->
- linattn broad-domain gap: `margin_vs_trigram` -0.29..-0.57 at depth>=2 (our `research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json`, the empirical wall this ladder addresses).
- Content-addressing deep-bucket means (our findings): assoc -0.347, assoc_t -0.147, hippokey -0.284; linattn +0.05 (simplewiki cross).
- Objective dominance (external): causal+masked hybrid BLiMP 0.794 vs tuned n-gram 0.633 vs causal LSTM 0.661 at a matched budget; MTP k=2 reverse-curriculum best at 10M words.
- Delta-rule (external): Gated DeltaNet Wiki ppl 16.42 vs Mamba2 16.56 vs worse plain-DeltaNet.
- Conv (external, same regime): BLaLM +conv perplexity 20.01 -> 12.37 (single biggest drop); our 2026-07-11 ceiling: local copy signal thin.
- Magnitude gating (our 2026-07-13): +0.09 (6/6) generic input-magnitude gate; content-selective did not transfer.

## What is in flight from this round
- **Decisive objective A/B — QUEUED (GPU):** the byte-identical wt103 linattn baseline with ONLY the predictive
  objective flipped on (`--pred-aux-weight 1.0 --pred-aux-offsets 2`, s43 direction-test vs
  `_emerge_wkv_lm_linattn_wt103_scale_s43.json`). If it lifts the broad-domain margin -> 6-seed + stack capacity;
  if flat -> delta-rule/conv move up.
- **Delta-rule build — IN FLIGHT** (rung 3, agent) — additive, default-off, de-risked on wt103.

## No-defer
A wall defers a METHOD, never the capability. The mouth's broad-domain fluency now has a ranked ladder of named,
evidence-backed methods and a banked-exhausted list; two new build levers (delta-rule, conv) are on the same
linattn substrate the roadmap said to push (not new content-addressing arms), and the cheapest decisive test of the
#1 lever is running. The own-voice mouth remains the #1 goal-blocker; its critical path is now well-stocked.
