---
type: plan
status: live
date: 2026-06-03
---

# Faithful spiking visual word recognizer — design (2026-06-03)

**Goal:** an EARNED, biology-faithful spiking recognizer that reads rendered words through the real
visual pathway (retina -> V1) and produces a discriminative word/letter representation that drives
the concept pools — replacing the orthogonal tokenizer (`set_token_drive`) and simultaneously
providing the visual->language **multimodal grounding** substrate.

## What is already done (this session)

- GPU transduction LIVE: `build_visual_text_bridge` (retina -> scaled-Gabor V1_simple) faithfully
  responds to rendered words (retina 0.23, V1_simple 0.03 firing).
- **Mechanism found + representation PROVEN discriminative:** reading V1_simple with the
  biologically-correct code — **latency** (first-spike recency) + **k-winners-take-all** (per-band
  lateral inhibition) — gives per-letter 0.575 on NOVEL words (4.6x chance, climbing with training),
  via a learned readout. Rate/spike-count readout = chance (the wrong code).
- Two in-substrate-kWTA placements RULED OUT: pool-level FS-WTA (chance — pools read all V1 inputs);
  V1-level GLOBAL feedback inhibition (no improvement — suppresses by total activity, not per-band).

## The remaining build (3 focused pieces)

### Piece 1 — per-band in-substrate kWTA (V1 lateral inhibition done right)

The 0.575 came from per-**band** kWTA (top ~10% earliest cells *within each spatial position band*).
Global inhibition fails because it does not implement per-position competition. Build per-band FS
lateral inhibition: one inhibitory FS pool per spatial band (or per orientation column), wired by
**band-restricted explicit weights** (like the Gabor install) so band-p FS receives from + inhibits
only band-p V1_simple cells. Feedback timing gives latency+kWTA for free (earliest/strongest fire
before inhibition suppresses the rest). Tune inhibition strength to ~10% surviving sparsity.

Validation: `read_letters_test(code="latency", kwta_frac=1.0)` off the inhibition-sparsified V1
should approach the 0.575 the readout-side kWTA showed. Bar: per-letter >> latency-only's 0.34.

### Piece 2 — learned readout (R-STDP, not vanilla STDP)

Vanilla STDP can't learn the precise denoised readout (pool-WTA test = chance). Use the project's
**reward-modulated STDP** (R-STDP, three-factor; used in the G-runners) for the V1->word-pool (or
V1->letter-pool) readout: teacher/reward signal when the correct pool fires for its word. This is
the biologically-standard supervised-ish rule that can match the logreg proxy. Pools keep per-pool
FS cross-inhibition for the final winner-take-all SELECTION (Piece 1 is the INPUT sparsification;
this is the OUTPUT selection — distinct roles).

Validation: spiking `--recognize` with Piece 1 + R-STDP readout should give discriminative
single-letter then word recognition (bar: clearly > chance, target toward the 0.575 proxy).

### Piece 3 — wire recognizer -> concept pools (the payoff)

Two wins at once:
- **Earned tokenizer replacement:** the recognized word drives its concept pool instead of
  `set_token_drive(orthogonal code)`. Word representations are now EARNED from pixels with shared
  orthographic structure (data-efficient, open-vocabulary — learn ~L letters, read L^n words).
- **Multimodal grounding:** the visual word-form pool binds to the language concept pool via Hebbian
  co-occurrence (STDP) — vision->concept binding. This is exactly the owner's multimodal-grounding
  direction; the cheap probe already showed grounding makes word-learning data-efficient. Then the
  EMNIST/CLEVR/CIFAR datasets (mapped in AUTONOMOUS_STATE) become usable for real grounding.

## Discipline

- Reuse-by-import: Gabor V1 (`sim.visual_cortex`), FS-WTA recipe (`text_minimal_isolation`), R-STDP
  (G-runner machinery), concept pools (`text_minimal_isolation.build_biological_brain_regions`).
- Cheap-first at each piece; the representation is already proven discriminative so each piece has a
  clear pass bar. Honest negatives propagated. Both remotes. No protected-module edits beyond the
  research runner. Biology-faithful; grounded in Thorpe/Masquelier/Kheradpisheh 2018 + Rolls VisNet.

## Scope honesty

Multi-session focused build (Piece 1 needs band-restricted wiring + inhibition tuning; Piece 2 is an
R-STDP readout; Piece 3 is the grounding wire-up). NOT a gamble — mechanism found, representation
proven discriminative, each piece de-risked. The latency+kWTA discovery (spiking layers were read
with the wrong neural code) is the session's scientific deliverable and likely generalizes beyond
vision (any spiking readout in this project).
