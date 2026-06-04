# Learned-code agent at 320 concepts — scale boundary (honest negative) — 2026-06-03

**One line:** The learned-code nesting agent's ~80% at 40 concepts does **not** cleanly extrapolate to 320.
With the defaults tuned for ~40 concepts, the 320-concept capstone gives a misleadingly-low headline; the
honest decomposition shows **simple capabilities scale (flat, one-attribute, retrieval) but complex
composition (two attributes, embedded clauses) collapses at the default dimension** — it needs the higher-D,
lower-correlation regime. The defaults don't transfer to 320.

This is the honest-negative-under-scale the project values. It corrects any naive reading that "the substrate
holds 320 (1.00 on constructed SVO) + the 40-concept agent works → the 320-concept learned-code agent works."
It does not, at fixed settings.

## The raw capstone (default n_input=256, D=2048, 320 concepts, 2 seeds)

`research/findings/raw/_scale320_learned_code_agent.py`:

| metric | seed 42 | seed 43 |
|---|---|---|
| memory recall (thresholded) | 0.00 | 0.00 |
| flat facts | 12/12 | 17/17 |
| one-attribute | 15/15 | 14/14 |
| two-attribute | 0/16 | 0/18 |
| embedded clause | 0/17 | 0/11 |
| total | 27/60 (0.45) | 31/60 (0.52) |

The headline (0.45–0.52, recall 0.00) looks like a broad collapse. The decomposition tells a more precise,
honest story.

## Decomposition — three distinct effects, diagnosed

**1. "Recall 0.00" is mostly the abstention threshold + an input-dimension overload — NOT retrieval failure.**
The memory's `recall` abstains below a confidence threshold (the no-confab guard), and at 320 concepts the
per-retrieval confidence drops, so it abstains — *correctly* (it is genuinely less certain). The underlying
argmax retrieval is fine **given enough input dimension**:

| memory input neurons | 320-concept argmax recall (no abstention) |
|---|---|
| 256 (default) | 0.72 |
| 1024 | **0.98** |

The default `n_input=256` overloads the associator (320 concepts cannot be independent patterns in a 256-dim
input space); at `n_input=1024` argmax retrieval is 0.98. The grounded *cue overlap* is fine (mean 0.10, max
0.38, no near-duplicates) — so it is **input dimensionality + the abstention threshold**, not cue collision.
Fix: `n_input ≥ vocabulary size`, and recalibrate the confidence threshold for the vocabulary scale.

**2. Flat and one-attribute facts scale to 320 (perfect).** The simple decode paths — a single argmax cleanup
over 320 codes (flat) and a single resonator factor (one attribute) — are robust to the 320-concept
correlated codes. The agent's threshold-free argmax cleanup over 320 works.

**3. Two-attribute and embedded-clause facts collapse at 320 — the real boundary.** The complex paths fail:
- **Two attributes (F=3 resonator):** the grounded-correlated adjective codes sit too close at D=2048; the
  measured lever is **D** (isolated two-attribute: D=2048→4096 lifts 0.83→0.96; restarts do nothing). At 320
  with D=2048 it is 0.
- **Embedded clause (recursive decode):** the clause is *correctly detected* (verb-detection confidence 0.288
  > 0.12 threshold) but the recursive inner cleanup returns garbage at 320 — the multi-hop decode (unbind →
  depth-detect → recurse → clean up over 320 correlated codes) accumulates crosstalk past the SNR floor. It
  needs higher D (and possibly scale-aware depth thresholds).

## Cause-isolation (correction): the 320 collapse is CAPACITY/DIMENSION, not correlation

A follow-up probe ran the *same* two-attribute and clause facts at 320 concepts with the agent's **default
well-separated random** phasor codes (not grounded). They collapse identically: **two-attribute 0/12, clause
0/12 (both seeds)** — the same as grounded codes (0/16, 0/17). So the 320 collapse is **not** caused by
grounded-code correlation; **it is dimension/decode capacity.** The specific limit: the resonator's factoring
capacity (~M=96-112 per codebook at D=2048, per the resonator-capacity finding) is **exceeded by the 200-noun
codebook** at 320 concepts; the recursive clause decode hits the analogous multi-hop cleanup-over-320 floor.

This **corrects** the milestone finding's earlier framing, which attributed the two-attribute weakness partly
to grounded correlation. Honest refinement: there are *two distinct effects* — (a) at moderate scale (~40
concepts), grounded correlation does cost the two-attribute resonator (0.56 grounded vs higher idealized);
(b) at large scale (320), capacity dominates and **both** random and grounded codes collapse. The 320 wall is
(b). The direct lever is D (resonator capacity scales with D — ~D=8192 would be needed for a 200-codebook,
4× the default, costly); a *smarter* fix (sparse codes / noise-injected resonator / hierarchical or gated
decode) is the subject of active deep research. Simple SVO still scales to 320 (the substrate finding's
1.00) because it needs no resonator — only the multi-factor and recursive paths hit the capacity wall.

## Honest verdict

**The learned-code agent partially scales to 320.** Simple conversational capabilities (factual recall with
adequate input dimension, flat facts, single-attribute entities) reach 320 concepts; **complex composition
(two attributes, embedded clauses) does not, at the dimension that worked for ~40 concepts.** The levers are
identified and partly measured — `n_input ≥ vocab` for retrieval, and substantially higher D for the resonator
and recursive paths — but they are real costs (D=4096+ for composition at 320, ~4× the memory of the 40-concept
setting), and the grounded-code correlation bites hardest exactly on the complex paths. This is the honest
boundary: the constructed-code *substrate* scales to 320 (SVO 1.00); the *learned-code agent's complex decode
paths* do not, at fixed dimension.

## What this means for the build

A 320-concept learned-code conversational agent is feasible but requires scale-aware settings (larger
`n_input`, larger D for composition, recalibrated thresholds) — not the 40-concept defaults. The
**production-integration arc must budget dimension for the complex paths**, and the two-attribute case in
particular may be better served by a different mechanism than the F=3 resonator at large vocab. The honest
negative narrows the design space before the big build, which is its value.

Conversational artifact `research/runners/phasor_chat.py` is validated at small vocabulary (where all paths
work); this finding documents where it breaks and why.
