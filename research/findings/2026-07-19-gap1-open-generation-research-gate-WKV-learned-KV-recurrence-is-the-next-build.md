# gap#1 open-generation research gate — the next build is a SPIKING RWKV/linear-attention (SpikeGPT-family) LEARNED key–value recurrence over the emergent stream-cortex codes, decoded by the existing spiking-Broca producer. Run it NOW; deploy the 21M spiking-forward in PARALLEL as an honestly-ledgered scaffold (milestone-met, NOT closed).

**2026-07-19.** With gaps #2/#3/#5 CLOSED and gap#4-supervised root-caused as a boundary (pivot to the unsupervised stream
cortex), gap#1 (open-ended fluent generation) is the single genuinely-OPEN capability. A 4-lens research gate
(`wf_dd786412-527`, 1.10M subagent tokens: state/residual · emergent-biology path · scaffold/efficiency path ·
adjacent-field mechanisms · decision) delivered a decisive, de-risked next build.

## THE EXACT RESIDUAL (Lens 1 — R4 REFRAMED, not "4 orders of params")
- **NOT params:** the from-scratch surrogate-BPTT spiking LM at 25M/50M params overfits to token-soup (held-out ppl
  ~204K, WORSE with scale); a 6M transformer generalizes where 25M spiking does not — the gap is ARCHITECTURAL
  (`2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size`).
- **NOT recurrence:** a full-backprop LSTM reaches ~98% of the transformer's growing +1.8 long-range margin at scale
  (`2026-07-11-CEILING`) — attention buys ~nothing over recurrence.
- **The genuine irreducible residual = a high-capacity, content-selective, NON-FADING, LEARNED-WRITE associative store.**
  Every reservoir/echo-state/e-prop/ALIF lever has a FADING-AVERAGE memory that loses to a FAIR interpolated trigram at
  every depth incl. d≥10 (`2026-07-15-fluency-crossover-RESOLVED-NEGATIVE`). R4 quantified in the operative dims:
  ~1–2 orders too few TOKENS (5M→24–100M) + ~1 order too small a content VOCAB (V=300→2000–8000) — the "< bigram" verdict
  was measured at 5M-tok/V=300 where NO model (even a full transformer) beats a bigram (`2026-07-11-CEILING`), i.e. tested
  against noise. **Both "cheap levers refuted" were wrong-FRAMED**, not capability verdicts: propositional novelty is
  already GO (R-i recombination novelty 0.987, factor-recovery 1.000, `2026-07-08`); the residual is FLUENT PROSE over
  novel propositions.

## THE DECISION — build a LEARNED KEY–VALUE RECURRENCE (RWKV/WKV, SpikeGPT-family)
RWKV's WKV op **is** the missing store: an O(N) recurrent gated leaky K/V integrator (linear attention, not O(N²)) with
learned K, V, receptance — a content-selective NON-FADING learned-write memory, a plausible cortical recurrent microcircuit.
- **De-risked, not hoped:** SpikeGPT is a published SPIKING generative LM at 45M params (WikiText-103 ppl ≈ GPT-2-Medium)
  — an at-scale existence proof at THIS project's exact scale.
- **Composes the project's OWN positive footholds:** R3-REFRAME (frozen-form recurrence + LEARNED input beats BPTT — WKV
  is fixed-form, only K/V/receptance learned) + `2026-07-11-LEARNED-keys` (content-addressed read over learned keys is
  load-bearing at deep context).
- **Converges gap#1 with the gap#4 pivot:** the WKV read is the natural first CLIENT of the unsupervised stream-cortex
  deep-representation engine (input = the emergent pooler codes). Decoded each step by the existing self-organized
  spiking-Broca A→W producer (EMERGE-59..74). **Clears the emergence bar** (learned from the stream; zero per-construction
  branches — the inverse of whack-a-mole).

## THE CHEAP-FIRST DE-RISK (the immediate build)
`research/runners/_emerge_wkv_lm_derisk.py` — rate-level, reuse the `_emerge_reservoir_lm_*` / `_ssm_reservoir_lm` harness
(`Vocab`, `load_sentences`, deep-context CE bucketing, `fit_bigram`) for APPLES-TO-APPLES:
- A rate-level WKV/linear-attention head; **input = emergent stream-cortex/pooler codes** (primary; + a learned-embedding
  reference variant to separate "mechanism captures deep context" from "codes are good enough"). Trained by BPTT (a TRACKED
  shortcut to establish the MECHANISM first; the ladder biologizes the rule later).
- **Ceiling-VALID scale:** TinyStories ~24M words / V≈2000 (transformer margin over bigram GROWS +0.5→+1.9 with depth here;
  5M/V=300 is the refuted wrong scale). Corpus present: `data/corpus/tinystories_train.txt`.
- **Deep-context (d10–99) next-token NLL** metric (the reservoir arc's own harness).
- **GO gate:** WKV **beats the FAIR interpolated trigram** on held-out deep-context (d≥10) NLL (the exact control that
  killed every reservoir lever), AND its deep-context margin **grows with depth/scale**, AND all 4 anti-cheats collapse
  (fair-trigram, permuted-context, memoryless-bag, content-vs-shuffle-KV), 6-seed (42/43/44/100/101/102).
- **Rung 2 (if GO):** port the fixed-form WKV/SSM recurrence onto a spiking `BrainRegion` (SpikeGPT confirms faithful);
  the selective-SSM variant (`_ssm_reservoir_lm_derisk` lineage; SNN membrane leak IS the SSM state update) is the drop-in
  fallback if RWKV gating is awkward on-substrate.

## SEQUENCING — run BOTH
- **START NOW (research, the actual close):** the WKV de-risk above.
- **PARALLEL (engineering, buys the demo, does NOT close the gap):** deploy the ~21M TinyStories generator as a
  spiking-FORWARD on the RF substrate (validated == ANN: 88.6M ppl_ratio 1.0; 24-layer Qwen bit-exact on the live bridge
  at 14GB LOCAL), co-resident behind the gate-first moat, as the R4 open-prose renderer. The one named blocker = the
  KV-cache lever (generation ~4.4 tok/s → interactive). Ledger discipline (NON-NEGOTIABLE): record as **"gap#1
  milestone-met-by-tracked-scaffold; emergent open-prose capability STILL OPEN (R4)."** Never regress the EMERGE-59..74
  frame inventory back to the ANN. The emergence bar is held OVER the scaffold, not satisfied BY it.

## NOT worth doing (refuted / whack-a-mole traps)
Rebuild the from-scratch surrogate-BPTT spiking LM (dead-end); more pure reservoir/echo-state/e-prop/ALIF generation levers
(exhausted — fading memory loses to trigram); ask the VSA/FHRR composer to free-generate (0/16 is a correct retrieval
property); add more hand-built constructions (whack-a-mole); scale an O(N²)-attention spiking transformer (least
mission-aligned; fallback only); re-litigate "< bigram" at 5M-tok/V=300 (the wrong scale). Adjacent runner-up levers:
spiking selective-SSM (near-identical #2), hippocampal retrieval-augmented residual (buys open-domain BREADTH at fixed
small params, maps to the just-closed gap#5 CA3 completion).
