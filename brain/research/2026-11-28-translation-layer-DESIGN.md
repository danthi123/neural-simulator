# DESIGN (not a prereg): a feature-code translation layer and a layered transmission model

Status: design only, 2026-11-28 (Fable), revised after critique (PROCEED-WITH-CHANGES). Nothing has run. Owner
idea 11-28. Tags: **[M]** our record; **[C]** cited at search-snippet depth; **[A]** citation and abstract checked
on PubMed, paper not read; **[U]** unread, from memory, fetch before any prereg cites it; **[E]** my estimate.

## 0. Record brief

- Record check 11-27: nothing built or refuted under this name. Adjacent: FHRR phasor algebra (worked; grounding
  failed at 0.45 pairwise overlap), DG separation (0.82 -> 0.18 overlap; near-orthogonality unmet) [M].
- Two 11-27 reads set the target. Forced reinstatement echoes the stored raw user line (7/7 user-voice answers are
  verbatim copies), and a right fetch adds only +17/217 [M]. One residual speaker direction at 1.5B lost knowing
  (43 -> 36) while moving voice: fact and frame entangled in a single direction; class closed [M].
- The voice arc is killed until a method's training data carries speaker contrast, after test2 [M, HANDOFF].
  Voice is report-only here.
- No SAE on RWKV is published [C].
- Missing prerequisite: binding inside a pooled code (section 4). A failure at 0.4B is a scale result until
  1.5B agrees.

## 1. The translation layer

**The idea in one line.** A sparse autoencoder is a one-layer network of threshold units. We use it as a **sparse overcomplete
interface** between cortex and hippocampus. This is an engineering analogy: entorhinal cortex is a convergence
zone, not an overcomplete sparse coder; the expansion is in DG.

**Read path.** Residual stream at layer 12 of 24 (0.4B and 1.5B, the 11-27 layer) -> encoder -> top-k (k = 64 of
16k-32k units) -> one sparse code per token. Each active unit drives one spiking assembly (signed-count L1
convention of the spiking-cortex route memo). At 0.2-0.4% density, codes of unrelated content overlap little;
learned features co-fire, so related content overlaps and separation remains DG's job. Low overlap is not
grounding and no grounding claim is made.

**Write path.** Assembly counts -> decoder directions -> added to the residual at the same layer, scaled by a
gain. The cortex never runs on the reconstruction, so only the dose can damage fluency.

**Episode format.** Today: a matrix-memory delta plus text. New: 3-7 ordered sparse frames pooled over spans of
the telling (a single bag loses which name belongs to which subject), each a list of (unit, count). No raw text.

**Frame versus content.** A unit is "frame" if a probe on the declared "User:" marking predicts its activity on
held-out chat. Reinstatement writes content units only.

**Scaffolds (new SCAFFOLDS rows, each with a retirement path).**
- Frame/content split: a host probe on a declared marking decides what is written. Retire: split read from the
  cortex's own provenance signal (corollary-discharge assembly; the 08-01 one-bit monitor was a six-seed GO [M]).
- Host-set gain and decay. Retire: gain from the surprise/arousal control plane (`affect.py`), decay from
  `Episode.strength`.
- Host-set pooling spans, k and layer. Retire: spans from the cortex's own surprise boundaries.
- The dictionary itself (gradient-trained beside the cortex). Retire: section 5.
Brain-side: what to reinstate is chosen by code overlap, never by a host rule on words.

**One-substrate cost.** A gradient-trained dictionary with versioned codebooks is a second learner beside the
cortex, refit whenever sleep changes it: a modular design joined by standards, the opposite direction from one
shared substrate. It is accepted only as a scaffold and a measuring tool; the target is a code that lives in the
shared population and is learned there (section 5).

**Drift from sleep.** Episodes are deltas from the pretrained base [M], so the dictionary is trained on the base.
Base-to-chat-fine-tune dictionary transfer is reported [U]. Check on the banked 14-night adapters; if variance
explained falls more than 0.05, refit the decoder during sleep; failing that, a crosscoder over base and slept
cortex.

## 2. A layered transmission model (audit instrument only)

This section is a bookkeeping standard for auditing seams: not an architecture, not a runtime bus, not a
biological claim. Nothing is built from it.

**Answer to the owner's question.** Partly yes. The field has event transport (AER, SpiNNaker routing) and model
description (NIR, ONNX), but no runtime standard between unlike neural modules [C]. The candidate narrow waist is the **addressed sparse event: (codebook, address, time,
magnitude)**: a spike, a feature activation and a token all fit. The difficulty is that a neural receiver has no
parser, its weights are the parser, so a standard must name a shared codebook. Cross-architecture feature
similarity (Mamba versus transformer) is reported [U]; whether shared codebooks exist is open.

| Layer | Carries | Biology | Here |
|---|---|---|---|
| L1 event | (address, time, signed count) | spikes | assembly counts |
| L2 frame and clock | micro-step, token, turn, night | gamma in theta | token step; turn; night |
| L3 code | codebook id and version | rate, phase, population codes | token, residual, dictionary, delta, DG index |
| L4 route and class | source, destination, driver or modulator | thalamic relay | content versus gain events |
| Control plane | scalar broadcasts, seconds | neuromodulators | surprise, arousal -> encoding gain |

**The rule.** Every header field is either realised as activity the receiver's synapses read, or it is
instrumentation and no decision may read it. A header a host `if` branches on is host templating.

**What the audit buys.** One seam type instead of one ad-hoc protocol per seam (old track: host-decision share
0.958 [M]), so that share is measurable per message class. Portability of feature-coded episodes between cortices
is an untested hypothesis, not a benefit.

## 3. Staged test plan

Order: S1a (no dictionary) -> S0 -> S1b -> S2 -> S3 -> S4. Nothing displaces test2, the 12-12 gate or the twin.

**Seeds and statistics (all stages).**
- Replies are greedy and forward-only, so a rerun is deterministic. What a seed varies: the life (which facts
  are told, order, filler chat) and, from S0 on, dictionary init and data order. 319/320/321 are the 11-27
  lives, kept only to reproduce banked rows. Any claim uses the six project seeds 42/43/44/100/101/102 as life
  and dictionary seeds.
- 67 told questions: 10% is 7 items, too few for a bootstrap over 5 gains x 7 arms. So: gain is fixed on the 36
  dev facts and never tuned on test; one primary contrast per stage; secondary contrasts in a fixed order under
  Holm correction; everything else report-only. Test: paired exact (McNemar) on items pooled over the six lives
  (about 400 item-lives), clustered by item. On the three banked lives alone, results are a lead only.

**S1a. Dictionary-free first experiment (0.4B; pools; no training; about 1 day [E])**
- Question: can any residual write carry a told fact?
- Hippocampus off, right episode forced. Arms: OFF; DELTA (today: delta plus stored text); DELTA-NOTEXT (matrix
  delta only, text withheld); DENSE (raw residual difference added); SPARSE-RAW (top-64 of the raw residual, and a
  random overcomplete projection with top-64; no learned dictionary); SHAM (another item's vector, each write
  arm); C (telling in context, ceiling).
- Measures: strict `correct_answer`; refusals; verbatim echo; neutral-text perplexity ratio per gain.
- Primary contrast: DENSE correct - OFF.
- Decision: if DENSE and SPARSE-RAW are about 0 correct over OFF (under 3 items per life, no better than SHAM)
  at every gain with perplexity ratio <= 1.05, the residual-write path is the barrier. S0 and dictionary
  training are premature and not run; go to the next method (section 4, item 4). Only headroom in a dense arm
  earns S0 its GPU hours.
- UNDEFINED if C is under 0.6 or DELTA does not reproduce its banked rows.

**S0. Are fact and frame separable in the units? (0.4B; 3090 about 3-6 h [E] between queued jobs; only after S1a)**
- Build: TopK dictionary, layer 12, about 200M streamed tokens (fineweb plus cortex-sampled two-speaker chat;
  test facts excluded), off-the-shelf trainer.
- SHUF, defined: same recipe on a cortex whose every weight tensor is permuted within itself, embeddings
  included. SHUF-E keeps embeddings and permutes the rest (isolates what token identity alone gives).
- Report-only (cannot fail, so not gates): variance explained (>= 0.70 is routine for TopK); dead units; speaker
  AUROC of 8 units ("User:" is a token, so this is lexically trivial; reported beside a token-identity baseline,
  marker positions excluded).
- **Main gate: matched ablation, measured in generation.** With the telling in context, generate answers with
  the frame units zeroed at layer 12, versus zeroing an equal number of random units matched on firing rate and
  decoder norm (20 draws). Reference: the 11-27 single direction cost 43 -> 36 knowing [M].
- Pass: knowing after frame ablation is within the random-ablation range (not below its 5th percentile) and
  loses under half the 11-27 loss (under 8% of intact); and the ablation moves the cortex's speaker-label logit
  more than every random draw (a manipulation check, not a voice claim); and REAL beats SHUF and SHUF-E on that
  check. Fail: fact and frame are entangled in units too; stop.

**S1b. Does a feature-coded memory answer correctly? (0.4B; pools; 2 days)**
- Arms: S1a arms plus FEAT-ALL; FEAT-CONTENT; FEAT-SHAM; RAND (norm-matched SHUF code).
- FEAT stores no raw user line, so "echo below DELTA" passes by construction. The fair baseline is DELTA-NOTEXT
  and the gate is correct answers and refusals.
- Primary contrast: FEAT-CONTENT correct - DELTA-NOTEXT correct.
- Pass, all of: primary contrast positive (paired exact test) and >= 10% of item-lives over OFF; FEAT-CONTENT
  beats FEAT-SHAM, RAND, DENSE and SPARSE-RAW (Holm, in that order); refusals not above DELTA-NOTEXT; knows >=
  0.9 of FEAT-ALL; perplexity ratio <= 1.05. Echo is reported, not gated.
- If SPARSE-RAW matches FEAT-CONTENT, the gain is sparsity, not the learned dictionary: drop the dictionary.

**S2. Scale and drift (1.5B; after the twin frees the 3090; about a day [E], or one g5 at about $25 [E])**.
Repeat S1a, S0, S1b; add the 14-night adapters. Pass: S1b rule holds; with adapters the effect keeps >= 0.8 of
its size and variance explained falls <= 0.05.

**S3. DG-CA3 on feature codes (pools)**. Gated on S1b: a right fetch buys little while replies echo [M]. Arms:
layer-0 key; `S_all`; FEAT key; FEAT + DG expansion; SHUF key; on the 11-03 meaning-selection real path. Pass:
heavy-reworded selection >= `S_all` + 0.05 on 6/6 seeds; told-versus-untold AUROC >= 0.75 (0.63 today [M]);
completion from half the units >= 0.8.

**S4. Spiking assemblies and interop (2027 Q1)**. Units as assemblies on the GPU spiking bridge; code decoded
from counts matches top-k >= 0.95; S1b replies unchanged.

## 4. Risks, odds, kill conditions

Odds [E]: S1a shows dense headroom 40%. Given that, S0 passes 50% and S1b passes at either size 35%. Whole
chain changes the weekly transcript by Q1: 10%. Section 2 useful as an internal audit 70%; as a field standard,
no claim.

Why S1 is hard:
1. Residual writing failed once at 1.5B, and local rule v3 matched the telling's activity without answering
   [M]. S1a tests this before anything is trained.
2. Binding: pooled units lose subject-to-name pairing; ordered frames are an untested mitigation.
3. Steering costs coherence as dose rises; dictionaries may miss rare names, which the told facts are [C].
4. A residual write does not reach matrix memory as a delta does. Next method if S1a fails: present the vector
   for one frame and let the cortex's own delta rule write it.

Kills: DENSE and SPARSE-RAW at zero in S1a at both sizes (ends residual writing); REAL does not beat SHUF;
FEAT-CONTENT no better than DELTA-NOTEXT, DENSE and SPARSE-RAW at both sizes; S0 matched ablation fails at both
sizes; working gain needs perplexity ratio above 1.2; drift needs over one GPU-hour of refit per night at 1.5B.

Two failed methods end the arc. A fail is a verdict on residual feature writing at this scale, not on the
capability.

## 5. Distributed population codes and richer neurons

Owner question 11-28: apply superposition, features-as-directions, polysemanticity, key-value memory, linear
relational embeddings and SAEs to the digital neurons themselves.

**(a) Research base.** All eight below: citation and abstract checked on PubMed 11-28, none read in full [A].
- Mixed selectivity: Rigotti et al., Nature 2013 (doi:10.1038/nature12160): PFC neurons respond to mixtures of
  task variables; population dimensionality predicts behaviour and collapses on error trials. Fusi, Miller &
  Rigotti, Curr Opin Neurobiol 2016 (doi:10.1016/j.conb.2016.01.010): mixing lets a linear readout produce many
  responses. This is the brain's polysemanticity.
- Neural manifolds: Gallego, Perich, Miller & Solla, Neuron 2017 (doi:10.1016/j.neuron.2017.05.025): population
  activity lies in low-dimensional modes. Features as directions.
- Representational geometry: Kriegeskorte & Kievit, Trends Cogn Sci 2013 (doi:10.1016/j.tics.2013.06.007).
- Sparse coding: Olshausen & Field, Nature 1996 (doi:10.1038/381607a0). The SAE's ancestor.
- Key-value memory: Gershman, Fiete & Irie, Neuron 2025 (doi:10.1016/j.neuron.2025.02.029): separate keys
  (retrieval, discriminable) from values (storage, faithful).
- Single neurons as deep networks: Beniaguev, Segev & London, Neuron 2021 (doi:10.1016/j.neuron.2021.07.002): a
  5-8 layer network is needed to fit an L5 pyramidal cell model; one hidden layer suffices without NMDA.
- Dendritic subunits: Poirazi, Brannon & Mel, Neuron 2003 (doi:10.1016/s0896-6273(03)00149-1): pyramidal neuron
  as a two-layer network.
- Dendritic XOR: Gidon et al., Science 2020 (doi:10.1126/science.aax6239): human L2/3 dendrites classify
  linearly non-separable inputs.
LLM side (superposition, linear relational embeddings, SAE literature): [U].

**(b) Diagnosis.** The pre-pivot design stored knowledge localist: a dedicated circuit per fact, 78,857 separate
129-neuron LTM bridges (about 5.4 GiB) for the imported store [M, pivot DECISION]. That is about 10 million
neurons, no shared structure, no generalisation, linear cost. Brains and LLMs both do the opposite: more
features than units, each unit in many features, each fact a pattern over a shared population.

**(c) Recommendation.** Population level first. Distributed superposed codes over one shared spiking population;
a learned sparse dictionary as the readout; hippocampal key-value indexing over that population (sparse
separated keys in DG-CA3, values as cortical patterns). This is also the retirement path for section 1's
side-car dictionary: the code lives in the shared population. Neuron level second, and only where a screen shows
behaviour changes. Extra per-neuron state must map to measurable biology (dendritic branch potentials,
plasticity tags, receptor states), never an unconstrained per-neuron vector.

**(d) A cheap decisive screen (CPU pools; about 1 day [E]; six project seeds; record check first; the old-track
circuit is run unchanged, harness in `brain/experiments`).**
- Fixed budget: N = 8,256 neurons (64 x 129). Same pre-pivot DG-CA3 circuit and plasticity in every arm.
- Arms: LOCALIST (one 129-neuron circuit per fact; ceiling 64 facts); SUPER (all facts as sparse distributed
  patterns over the one shared population, related facts sharing units, read out by a sparse dictionary);
  RAND-CODE (same sparsity, codes assigned at random, similarity structure destroyed); SHUF-DICT (SUPER with the
  dictionary's unit assignment permuted: readout floor).
- Load 32 to 2,048 facts. Measures: capacity (largest load with cued recall >= 0.9); recall from a half cue;
  interference (recall of the first 32 facts after full load; false recall on similar-key lures).
- Pass, 6/6 seeds: SUPER capacity >= 4 x LOCALIST (>= 256 facts); at that load early-fact recall >= 0.9 and
  lure false recall <= 0.10; SHUF-DICT at chance; SUPER >= RAND-CODE on half-cue recall (if RAND-CODE matches
  SUPER everywhere, the gain is distribution alone, still a pass, and learned structure is unproven).
- Fail: this circuit's plasticity cannot hold superposed codes at this size. A method verdict; the follow-up
  arm is the first neuron-level screen (two-layer branch subunits, equal parameter count; adopt only if capacity
  or interference moves).
