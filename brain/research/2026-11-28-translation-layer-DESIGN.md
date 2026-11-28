# DESIGN (not a prereg): a feature-code translation layer and a layered transmission model

Status: design only, 2026-11-28 (Fable). Nothing has run. Owner idea 11-28. Tags: **[M]** our record, **[C]** cited
at search-snippet depth (no full paper read), **[E]** my estimate. The key-value-memory and linear-relation papers
are from memory and must be fetched before a prereg cites them.

## 0. Record brief

- Record check 11-27: nothing built or refuted under this name. Adjacent: FHRR phasor algebra (worked; grounding
  failed at 0.45 pairwise overlap), DG separation (0.82 -> 0.18 overlap; near-orthogonality unmet) [M].
- Two 11-27 reads set the target. Forced reinstatement echoes the stored raw user line (7/7 user-voice answers are
  verbatim copies), and a right fetch adds only +17/217 [M]. One residual speaker direction at 1.5B lost knowing
  (43 -> 36) while moving voice: fact and frame are entangled in a single direction; class closed [M].
- The voice arc is killed until a method's training data carries speaker contrast, after test2 [M, HANDOFF]. A
  dictionary trained on both speakers' turns meets that wording, but a skeptic must accept this before any voice
  clause below is a gate. Until then voice is report-only.
- No SAE on RWKV is published [C].
- Capability: Episodic recall, Memory selection, Self and identity. Missing prerequisite: binding inside a pooled
  code (section 4). A failure at 0.4B is a scale result until 1.5B agrees.

## 1. The translation layer

**The idea in one line.** A sparse autoencoder is already a one-layer network of threshold units: the encoder is a
set of afferent synapses, JumpReLU is a firing threshold, top-k is lateral inhibition, the decoder is the efferent
projection. So we treat it as an entorhinal-like sheet between cortex and hippocampus, not as an analysis tool.
Its units are the "assemblies".

**Read path.** Residual stream at one middle layer (layer 12 of 24 at 0.4B and 1.5B, the layer the 11-27 read
used) -> encoder -> top-k (k = 64 of about 16k-32k units) -> one sparse code per token. Each active unit drives
one spiking assembly; activation becomes a spike count (the signed-count L1 convention of the spiking-cortex
route memo). A code at 0.2-0.4% density is near-orthogonal by construction for unrelated content, which is the
grounded symbol set the phasor work lacked. Related content still shares units; separating that is DG's job.

**Write path.** Assembly counts -> decoder directions -> added to the residual at the same layer, scaled by a
gain. The cortex is never run on the reconstruction: the layer is a side tap plus an additive write, so
reconstruction error cannot damage fluency. Only the write dose can.

**Episode format.** Today an episode is a matrix-memory delta plus text. The new format is a short sequence of
sparse frames (3-7 frames pooled over spans of the telling, in order), not one bag, because a bag loses which
name belongs to which subject. Each frame is a list of (unit, count).

**Frame versus content.** Units are split by a learned, data-driven test: a unit is "frame" if its activity is
predicted by speaker span (the declared "User:" marking) on held-out chat, "content" otherwise. Reinstatement
writes content units only; the speaker frame is left to the cortex's live state. Who said it is carried
separately (section 2).

**Learned versus declared.** New SCAFFOLDS rows: the dictionary (gradient-trained on the frozen cortex with an
off-the-shelf trainer; stands in for developmental learning of entorhinal codes); k, layer, gain and pooling spans
(hand-set on dev facts); the frame/content split (a learned probe on the "User:" marking); unit = assembly with
count coding. Brain-side: what to reinstate is chosen by code overlap, never by a host rule on words.

**Reconstruction error.** Report variance explained and dead units. Every claim is paired with a dictionary
trained on a weight-shuffled cortex [C, 2602.14111] and a dense arm that skips the dictionary.

**Drift from sleep.** Episodes are deltas from the pretrained base [M], so the dictionary is trained on the base.
Dictionaries trained on a base model transferred to its chat fine-tune [C]. Check ours on the banked 14-night
adapters. If variance explained falls by more than 0.05, refit only the decoder during sleep; if that fails, a
BatchTopK crosscoder over base and slept cortex. The same crosscoder can show what gradient sleep changed that the
local rules did not.

**Timescales.** Read per token; store per turn; write over the answer span with decaying gain; refit per night.

## 2. A minimal layered transmission model

**Answer to the owner's question.** Partly yes. The field has event transport (AER, AER over UDP, SpiNNaker
routing) and model description (NIR, ONNX), but no runtime standard between unlike neural modules and none for
clocking or modulatory control [C]. The Internet's gain was a narrow waist. The candidate waist here is the
**addressed sparse event: (codebook, address, time, magnitude)**. A spike is magnitude 1 at a neuron address. A
feature activation is a graded magnitude at a dictionary address. A token is an event at a vocabulary address. The
difficulty is not the format. Network packets are content-blind and a parser reads them; a neural receiver has no
parser, its weights are the parser. So the standard must name a shared codebook, and dictionaries are what makes
one possible. Similar features across Mamba and a transformer [C] suggest cross-model codebooks can exist.

| Layer | Carries | Biology | Here |
|---|---|---|---|
| L0 medium | buffers, dtype | ions, vesicles | tensors |
| L1 event | (address, time, signed count) | spikes | assembly counts |
| L2 frame and clock | micro-step, token, turn, night | gamma in theta, coherence windows | token step is the frame; turn; night |
| L3 code | codebook id and version | rate, phase, population codes | token, residual, feature dictionary, matrix delta, phasor, DG index |
| L4 route and class | source, destination, driver or modulator | thalamic relay, driver and modulator inputs | content events versus gain events |
| Control plane (out of band) | scalar broadcasts, seconds | dopamine, noradrenaline, acetylcholine | surprise and arousal -> encoding gain (`affect.py`) |
| Slow plane | tags, strength, replay, weight change, hours | tagging and capture, protein synthesis | `Episode.strength`, sleep, adapters |

**Message fields.** `src, dst, class (driver, modulator, plasticity), timescale (event, frame, turn, night),
code (codebook id, version), clock (token, turn, night), provenance (heard, self, replayed, reinstated; origin
episode), payload (address and count pairs), gain`.

**The rule that keeps it brain-based.** Every header field is either realised as activity the receiver's synapses
read (provenance is a small corollary-discharge assembly inside the payload; the 08-01 one-bit monitor was a
six-seed GO [M]) or it is instrumentation and no decision may read it. A header a host `if` branches on is host
templating.

**What it buys.**
1. One seam type. The old track had one ad-hoc protocol per seam and a host-decision share of 0.958 [M]. Typed
   messages make that share auditable.
2. Portability. A home refuses another cortex because deltas are cortex-specific [M]. Feature-coded episodes can
   move to the spiking twin or a larger cortex if a crosscoder aligns the codebooks. This is a test, not a given.
3. Provenance versus the voice wall. The stored code no longer has to contain the user's first-person frame.
4. Sleep can replay codes, retiring "sleep replays the stored text".

**Realism cost.** Brains have no headers, layering or versioned codebooks; routing is by content and timing.
State stays graded. The dictionary is gradient-trained. All of this is declared; the layering is engineering
discipline, not a biological claim.

**Extrapolation (not planned).** A transcoder on channel-mix would expose key -> value units; a told fact is then
one new pair and its update is pre-times-post local. A 12-12 candidate only after the credit map reads.

## 3. Staged test plan

Forward-only except dictionary training. Seeds 319/320/321 and the 67 told questions, to compare with the 11-27
reads; six seeds before any claim. Two days per stage. Nothing displaces test2, the 12-12 gate or the twin.

**S0. Do the features exist? (0.4B; 11-29 to 12-01; 3090 about 3-6 h [E] between queued jobs, probes on pools)**
- Build: TopK dictionary, layer 12, about 200M streamed tokens (fineweb plus cortex-sampled chat with both
  speakers; test facts excluded), off-the-shelf trainer.
- Arms: REAL; SHUF (same recipe on a weight-shuffled cortex); DENSE probe (ceiling).
- Measures: variance explained; speaker AUROC of the best 8 units on held-out chat; fact identity from units at
  the answer position with the telling in context; identity retained after removing frame units.
- Pass: variance explained >= 0.70; speaker AUROC >= 0.90 and >= SHUF + 0.10; identity retained >= 0.90 of
  intact. Fail on the last clause means fact and frame are entangled here too: stop before S1.

**S1. The decisive screen: does a feature-coded memory answer correctly, not echo? (0.4B; 12-02 to 12-03; pools)**
- Hippocampus off, right episode forced, so only the format is tested.
- Arms: OFF; DELTA (today's forced reinstatement); FEAT-ALL; FEAT-CONTENT; FEAT-SHAM (another item's code);
  RAND (norm-matched SHUF code); DENSE (raw residual difference, no dictionary); C (telling in context, ceiling).
- Gain: five levels chosen on the 36 dev facts; neutral-text perplexity reported at each (dose-response).
- Measures: strict scorer `correct_answer`; verbatim-echo rate; P(user voice | knows); refusals; perplexity ratio.
- Pass, all of: FEAT-CONTENT correct - OFF >= 2 x (DELTA - OFF) and >= 10% of rows; echo rate <= half of DELTA;
  FEAT-CONTENT beats FEAT-SHAM, RAND and DENSE (item-clustered bootstrap interval excludes 0); knows >= 0.9 of
  FEAT-ALL; perplexity ratio <= 1.05.
- UNDEFINED if C is under 0.6 or DELTA does not reproduce its banked rows.

**S2. Scale and drift (1.5B; 12-09 to 12-11, after the twin frees the 3090; about a day [E], or one g5 at about
$25 [E])**. Repeat S0 and S1; add the 14-night adapters. Pass: S1 rule holds, and with adapters on the effect
keeps >= 0.8 of its size and variance explained falls <= 0.05.

**S3. DG-CA3 on feature codes (12-11 to 12-18; pools)**. Gated on S1: the memory ceiling says a right fetch buys
little while replies echo [M], so selection work is worth it only if S1 raises that ceiling.
- Arms: layer-0 key; `S_all`; FEAT key (overlap of deterministic top-k codes); FEAT + DG expansion; SHUF key.
  Harness: the 11-03 meaning-selection real path.
- Pass: heavy-reworded selection >= `S_all` + 0.05 on 6/6 seeds; told-versus-untold AUROC >= 0.75 (0.63 today
  [M]); completion from half the units >= 0.8.

**S4. Spiking assemblies and interop (2027 Q1)**. Units as assemblies on the old GPU spiking bridge; code decoded
from counts over one frame matches top-k >= 0.95; S1 replies unchanged; the same messages drive the spiking twin
if E1 is a GO; one episode moved between 0.4B and 1.5B through a crosscoder.

Milestones: nothing here serves 11-30. S1 and S2 inform the 12-14 life; S4 is the first step onto the shared
spiking substrate.

## 4. Risks, odds, kill conditions

Odds [E]: S0 passes 45% (features exist 75%, disentangled enough 60%). S1 passes at either size 30%. S3 passes
given S1 40%. Whole chain changes the weekly transcript by Q1: 15%. The transmission model is useful as an
internal audit standard 70%; as a field standard, no claim.

Why S1 is hard:
1. The residual-write class already failed once at 1.5B [M]. The difference here is many learned sparse units
   and a frame split, not one mean direction. DENSE and RAND test whether that difference is real.
2. Local rule v3 matched the telling's activity and did not answer the question [M]. A code read from the telling
   may be a weak hint, not knowledge.
3. Binding. Pooled units lose subject-to-name pairing. Ordered frames are the mitigation, untested.
4. Steering costs coherence as dose rises, and dictionaries may miss rare names (the told facts) [C].
5. A residual write does not reach matrix memory as a delta does. Next method if S1 fails on dose: present the
   decoded vector for one frame and let the cortex's own delta rule write it.

Kills:
- REAL does not beat SHUF in S0 or S1.
- FEAT-CONTENT is no better than DENSE and DELTA at both sizes.
- Removing frame units costs more than 10% of knowing at both sizes.
- The working gain needs a perplexity ratio above 1.2.
- Drift needs more than one GPU-hour of refit per night at 1.5B.

Two failed methods end the arc. A fail is a verdict on residual feature writing at this scale, not on a
translation layer or a transmission standard.
