# Step-2a: spiking visual word recognition off V1_simple — honest characterization (2026-06-02)

**Context.** The input-side-fidelity insight (owner side-chat, validated 4 ways by cheap probes)
says language should be *transduced* as pixels through the existing visual pathway (earned,
shared-structure, data-efficient) instead of *tokenized* into given orthogonal codes. The
production build (`research/runners/text_visual_grounding.py`) realizes this on the GPU: a
region-framework bridge `retina -> V1_simple -> V1_complex -> V2 -> IT` with scaled Gabor V1
weights, reading rendered words as pixels.

This doc records step-2a: reading earned word recognition off the working V1_simple layer via a
plastic STDP pathway (`V1_simple -> word_pool`, teacher-supervised), and the honest limits found.

## What works (verified on GPU, RTX 3090)

- **Construction + transduction.** retina 64: 49,472 neurons, 13.7M synapses, builds ~80s.
- **Per-layer firing diagnostic** (retina 32, drive 2500 pA): retina 0.23, **V1_simple 0.03**.
  The retina -> V1_simple transduction *faithfully responds to rendered words*. This is the
  tokenizer-replacement: words enter as pixels through earned visual transduction, not given
  orthogonal codes. **The owner's input-side-fidelity fix is live on the GPU.**

## Diagnosed cascade gap (V1_complex)

The hierarchy does not propagate past V1_simple for text: V1_complex 0.005, V2/IT ~0. Root cause
is **V1_complex starvation**: text is *sparse* (thin letter strokes -> V1_simple fires 0.03)
whereas the g11 gridworld this pathway was tuned for shows *dense blocks* (many coincident V1s
spikes). The g11 random-density phase-pooling (weight 2.0) rarely gets coincident V1s spikes from
sparse text, so V1_complex stays silent and V2/IT are dead downstream. Strengthening the pooling
(weight 20, 4x density) lifted V1_complex to 0.022 for the strongest word but the full cascade to
IT still did not propagate -> this is multi-knob engineering (structured phase-pooling + V2/IT
inhibition + scale), not a one-line fix. Per the debugging iron law (reassess after 3 attempts),
stopped tuning and read recognition off the working V1_simple layer instead.

## Recognition off V1_simple — the ceiling

Teacher-supervised STDP (`train_word_to_pool` pattern reused from `concept_pool_demo`): drive
retina(word) -> V1_simple word-form fires; drive the target word-pool with teacher current;
STDP on the open-gated `V1_simple -> target-pool` pathway binds the word-form to the pool.
Interleaved events, one gate open at a time (isolated per-word training). Test: drive retina(word),
no teacher, the highest-firing pool is the recognition.

| Readout | Vocab | retina | result | chance |
|---|---|---|---|---|
| whole-word pools | dog,cat,run,sun | 32 | **1/4 = 0.25** | 0.25 |
| single-letter pools | a,e,o,t,x | 32 | **2/5 = 0.40** | 0.20 |

The single-letter 0.40 sits right at the **V1-simple-readout ceiling** the cheap scaled-Gabor probe
independently found (retina 64 = 0.37). Mechanism: V1 *simple* cells are position-specific and do
not build invariant object/word representations; a whole-glyph pool-argmax over sparse spike-counts
loses most of the structure. The cheap probe's **0.91** came specifically from *per-position letter*
readout (compositional — read each letter band, compose the word) on continuous Gabor features with
a trained per-position classifier — a fundamentally different, compositional readout.

## Conclusion + two clear paths

Reading recognition off the spiking V1_simple layer with simple pools is noise/ceiling-limited
(~0.40 single-letter, chance whole-word). This is consistent + honest: the faithful invariant
recognition is not in V1_simple. Two well-specified paths to a faithful spiking word recognizer:

1. **Full V1 -> V2 -> IT hierarchy** (the biologically faithful object-recognition route): fix the
   V1_complex propagation with *structured* phase-pooling complex cells (Hubel-Wiesel quadrature
   pairs, not random density) + bigger retina/bolder text (more V1s activity; owner: "no reason to
   limit retina to 32x32") + V2/IT inhibition tuning, so IT builds invariant word-form
   representations and the recognition reads off IT. Grounded in Riesenhuber-Poggio HMAX /
   DiCarlo IT object recognition.

2. **Per-position letter-composition pools** (the validated 0.91 architecture, in spiking): pools
   per (position, letter); read each letter band of V1_simple; compose into a word. Open-vocabulary
   + data-efficient (learn ~L letters -> read L^n words). Needs bigger retina (each letter band
   well-resolved) + temporal integration (denoise sparse spikes).

**Decisive cheap experiment (in flight):** does bigger retina (64) + long temporal-integration
window (200 steps) + reduced pool inhibition lift single-letter recognition above the 0.40 ceiling?
If yes -> path 2 (letter-composition) is viable without the full hierarchy. If still ~0.40 -> the
ceiling is fundamental to V1-simple-readout and path 1 (full hierarchy) is genuinely required.

### VERDICT (run landed): the V1-simple whole-glyph ceiling is FUNDAMENTAL

retina 64 + 200-step integration + reduced pool inhibition (iw 1.5) gave **1/5 = 0.20 = chance**,
*worse* than retina 32 (0.40), via **dominant-pool collapse**: every test letter predicted 'o'
(pool 'o' fired 0.7-0.8 for ALL inputs while the rest sat at 0.4-0.6). More signal + less
inhibition did not separate the glyphs — it let one pool's STDP weights grow to dominate every
input. This is the same winner-take-all collapse the concept-pool arc spent 14 iterations taming
(FS cross-inhibition + topographic prior + target-only gating).

Crucially, the cheap scaled-Gabor probe's ceiling (0.37) was measured with an **optimal linear
classifier** (no dominant-pool artifact) — so ~0.37-0.40 is the genuine **whole-glyph
V1-simple-readout ceiling**, not a WTA artifact. Adding WTA machinery would recover the collapse
back toward ~0.40 but cannot exceed it. **Conclusion: reading word/letter recognition off the
spiking V1_simple layer as a whole glyph is decisively insufficient (~0.40 ceiling).**

The faithful spiking word recognizer therefore needs structure beyond whole-glyph V1-simple:
- **Path 1 — full V1->V2->IT hierarchy** (DiCarlo: invariant object recognition is solved in IT,
  not V1). Requires fixing V1_complex propagation (structured phase-pooling) + V2/IT tuning +
  WTA pools. The biologically canonical route.
- **Path 2 — per-position letter-composition** (the cheap probe's 0.91, NOT YET tested in
  spiking): read each letter BAND of V1_simple separately (exploiting position structure), one
  letter pool per (position, letter) with FS cross-inhibition WTA, compose into a word. This is
  the data-efficient open-vocabulary route; the 0.91 came specifically from per-position reading,
  which my whole-glyph tests never exploited.

Next: test path 2 (per-position letter pools + WTA) cheap-first, since it directly exploits the
position structure that produced the validated 0.91 and is cheaper than fixing the full hierarchy.
If per-position spiking reading beats the 0.40 whole-glyph ceiling -> the data-efficient recognizer
is viable; if it also collapses -> path 1 (full hierarchy) is the only faithful route.

### Path-2 result + the likely missing mechanism (LATENCY CODING, literature-grounded)

Per-position letter reading on REAL spiking V1_simple (100-step rate readout, 80 novel 3-letter
words over an 8-letter alphabet): per-letter **0.09 / 0.115 / 0.192** at K=15/30/54 vs chance 0.125
-- essentially chance, nowhere near the continuous-feature probe's 0.91. The weak rise with K shows
a faint signal buried in spike noise. Mechanism: each V1 cell fires only ~3 times over the window
(0.03 rate), so the per-cell spike COUNT is a hopelessly noisy estimate of the graded Gabor response
the continuous features used.

**Literature check (owner: "brains have aspects we haven't implemented; use the scientific texts").**
The proven biologically-plausible spiking object/digit recognition models do NOT use rate/spike-count
readout. Kheradpisheh-Ganjtabesh-Thorpe-Masquelier 2018 ("STDP-based spiking deep convolutional
neural networks for object recognition", Neural Networks; arXiv 1611.01421) -- which matches/beats
deep CNNs on some tasks -- and Masquelier-Thorpe 2007 use **temporal latency / rank-order coding**:
the strongest-responding cell fires FIRST; recognition reads the spike-ORDER/latency pattern, not the
count. This is robust to sparsity AND directly preserves the Gabor-magnitude structure (strong
response = early spike) that the continuous features (0.91) exploited. Complementary mechanisms:
max-pooling convergence (RF ~2.5x/stage, ~10k inputs/neuron -> invariance; Rolls VisNet) and slow/
trace learning (temporal continuity -> invariance). My rate-count readout was simply the **wrong
neural code**.

=> Before concluding the full hierarchy is required, test the **latency code** cheap-first: read each
V1_simple cell's FIRST-SPIKE recency (earliest = strongest) instead of its spike count, same
per-position classifier. Implemented as `read_letters_test(code="latency")` / `--latency`. If latency
reading beats the rate ceiling -> the fix is the neural CODE (cheap), not a multi-week hierarchy build.
If latency also fails -> the structure genuinely needs the deep convergent hierarchy (path 1), now
grounded in the proven Thorpe/Masquelier convolutional-SNN design rather than a from-scratch build.

Sources: Kheradpisheh et al. 2018 (arXiv 1611.01421); Masquelier & Thorpe 2007; Rolls VisNet
(slow unsupervised invariance learning).

### LATENCY RESULT: the neural code matters (confirmed); full pipeline is the grounded path

Latency-coded per-position read (200-step window, retina 64, 120 novel words): per-letter
**0.167 / 0.192 / 0.342** at K=15/30/80 vs chance 0.125. Two things stand out vs the rate readout:
1. **Latency beats rate at matched training** (K=80: latency 0.342 vs 500-step rate 0.242, +10pp).
2. **Latency keeps CLIMBING with training data while rate PLATEAUS** (rate 0.233/0.242/0.242 flat;
   latency 0.167/0.192/0.342 rising). The first-spike-recency code carries more *learnable*
   structure -- exactly the literature's claim that latency/rank-order is the right code for
   sparse spiking vision.

So the reframe is validated: my earlier "spiking V1_simple readout is insufficient" was partly a
**wrong-code** artifact, not purely a substrate limit. Honest caveat: 0.34 per-letter (per-word
0.05) is still not *usable* -- but I changed ONLY the readout code on raw V1_simple. The proven
Thorpe/Masquelier/Kheradpisheh pipeline that reaches CNN-level is latency coding **+ max-pooling
convergence layers + STDP-learned feature hierarchy**. The +10pp-and-climbing from the code change
alone is the expected first-piece signal.

**Verdict: the faithful spiking visual recognizer is a GROUNDED, proven build** (Thorpe/Masquelier
convolutional-SNN: latency coding, which is now validated as the right first piece, + max-pooling
convergence + STDP feature learning + a final supervised/RSTDP readout). This is no longer a
from-scratch uncertain gamble -- it is a published architecture that matches CNNs, of which the
latency-code piece is now confirmed on our substrate. It is also the exact prerequisite for the
multimodal-grounding milestone (robust visual object/word representations to Hebbian-bind language
concepts to). Next: build the convergence + STDP-feature layers on top of the latency code (a real
but de-risked sub-arc), brainstorming/design-first per its size.

### kWTA BREAKTHROUGH: the recognizer is tractable CHEAPLY (verdict revised UP)

The Thorpe/Masquelier mechanism has TWO halves: latency coding (tested -> helps) AND **k-winners-
take-all lateral inhibition** (keep only the earliest/strongest responders per map, suppress the
rest -- the denoising my readout lacked). Adding per-band kWTA (keep top 10% earliest cells) to the
latency read off raw V1_simple:

| Readout (V1_simple, retina 64, novel words) | K=15 | K=30 | K=80 |
|---|---|---|---|
| rate (500-step integration) | -- | -- | 0.242 (plateaued) |
| latency only | 0.167 | 0.192 | 0.342 (noisy) |
| **latency + kWTA (0.1)** | **0.267** | **0.417** | **0.575** |

Per-letter **0.575 at K=80, 4.6x chance, and climbing STEEPLY with K** (per-word 0.025/0.10/0.15).
The lateral-inhibition denoising was the decisive missing piece. **This REVISES the earlier
"sparse-propagation wall / multi-week hierarchy required" framing DOWN**: the wall was the
**wrong neural code + missing lateral inhibition**, NOT a fundamental substrate limit. The core
recognition works on raw V1_simple with just the right *readout mechanism* (latency + kWTA) -- a
cheap readout-side change, no deep hierarchy needed to get a strong, climbing signal.

The full conv-SNN (conv feature layers + STDP-learned intermediate features) would push further and
add translation invariance, but the headline is: **the spiking substrate carries the word-form
structure fine; it just has to be read with the biologically-correct code (latency) + lateral
inhibition (kWTA)** -- exactly the Thorpe/Masquelier prescription, now confirmed on our bridge. This
is the cheap-first investigation paying off: the recognizer is a tractable build, not a multi-week
gamble. Next: push K + tune the kWTA fraction toward usable per-word recognition, then wire the
recognizer -> concept pools (the earned tokenizer replacement).

## Honest scope

The input-side-fidelity *science* is validated 4 ways (cheap probes) and the *transduction* is live
on the GPU. A faithful spiking *recognizer* is the well-specified next sub-arc (path 1 or 2 above),
not a one-session tune. No shortcuts; biology-faithful; both remotes.

## Where the kWTA must live: V1-LEVEL lateral inhibition (pool-level FS-WTA insufficient)

Two follow-ups sharpened the production architecture:
- kWTA fraction: 0.1 > 0.05 (tighter discards too much; 0.05 gave 0.43 vs 0.1's 0.575 per-letter).
- **Pool-level FS cross-inhibition (the validated Tier 1 motor-WTA recipe) on the spiking word pools
  = 0/5 chance.** Adding lateral inhibition at the POOL output did NOT fix the spiking recognizer.

The reason is precise and important: the breakthrough kWTA operated on the **V1 features** -- it
sparsified *which V1 cells* the readout reads (keep the earliest/strongest, drop the noisy rest). A
spiking pool instead reads *all* its V1 inputs weighted by STDP, so per-cell spike noise flows
straight in and STDP (local, unsupervised) can't learn the precise denoising a supervised readout
does. => the kWTA must live at the **V1 level** (lateral inhibition that sparsifies the V1 spiking
code *before* the pools read it) -- exactly the Thorpe/Masquelier per-layer-inhibition design (each
conv layer is followed by lateral inhibition). The discriminative information is provably there (0.575
readout proxy); a faithful spiking recognizer needs V1-level lateral inhibition to expose it, plus a
readout rule stronger than vanilla STDP (reward-modulated STDP / a learned readout layer -- both
biologically standard).

**Net scoping (honest, de-risked):** faithful spiking visual word recognizer = V1_simple (latency
code) + **V1-level lateral inhibition (kWTA sparsification)** + readout pools (R-STDP or learned
readout) + optional conv layers for translation invariance. The MECHANISM is found and the
representation is proven discriminative; the remaining work is the V1-level inhibition + readout rule
-- a focused build, not a gamble. Also the multimodal-grounding prerequisite. Next: build V1-level
lateral inhibition and test whether the sparsified V1 spiking code lets the pools/readout reach the
0.575 the offline proxy showed.
