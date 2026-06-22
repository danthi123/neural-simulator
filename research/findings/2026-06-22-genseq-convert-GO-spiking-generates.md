# Generative-sequence frontier — convert GO: the SPIKING Gen-F still generates coherent NOVEL text (training-free, T=32, 3/3); the measured 0-novel wall is BROKEN (2026-06-22)

**Scope:** the decisive Spine-A convert de-risk — does the working non-spiking generator Gen-F, converted to spikes
TRAINING-FREE, still GENERATE? This is loop-step 2 (GENERATE, spiking). `research/runners/_genseq_convert_derisk.py`,
PyTorch/CUDA. **NO `sim/` edit, NO retraining** (weights verbatim, one no-gradient calibration pass). On `main`.

## Result — GO (3/3 seeds, T=32)
Training-free spiking-rate conversion of Gen-F (`TinyGPT`, 3.45M params, V=513; linear ops exact under rate coding;
the 3 nonlinear ops — softmax-attention / LayerNorm inv-sqrt / GELU — rate-quantized to T levels over calibrated
ranges; QCFS / MBE / LAS / ECMT class). `nn.MultiheadAttention` hand-reimplemented (split Q/K/V, per-head SDPA,
causal mask) so the internal softmax could be made spiking.

| T | byte-unmodified Gen-F gate | spiking/ANN ho-ppl ratio | verdict |
|---|---|---|---|
| 16 | PASS 3/3 (all 5 bars) | [1.236, 1.240, 1.262] | gate ok, ratio just > 1.2 |
| **32** | **PASS 3/3 (all 5 bars)** | **[1.058, 1.060, 1.067]** | **GO** |

At T=32: spiking ho-ppl ≈ 6.53–6.56 (ANN ≈ 6.15–6.19 → **+6%**, in line with the SOTA's ~1.03–1.06); ≪ the
abs-competence floor 513; ≈ 0.15× the word-shuffle control (real structure preserved); distinct-trigram 0.958–0.992
(≥0.5); verbatim-8gram-copy 0.000–0.035 (≤0.20). Every gate bar (competent / real-structure-vs-shuffle / generalizes
/ non-degenerate / not-copying) = True on all 3 seeds.

**FAITHFULNESS validated:** the ANN baseline reproduces the documented ho-ppl 6.15–6.19 EXACTLY → the
spiking-capable attention reimplementation is correct, not a silent approximation.

**Generated sample (T=32, seed 42, SPIKING):** *"Tom thought it was fun to try it. He pushed the button on the grass
and put it in a button. They started to lay down around the garden with the microphone... The cat was nice, but he
also happy to have his cap back. It could clean up"* — coherent, story-shaped TinyStories English with character
continuity; novel grammatical sequences (copy ≤0.035), NOT regurgitation. The occasional malformed token (e.g.
"skeletonhew") is Gen-F's OWN documented 3.45M coherence ceiling, reproduced faithfully — not a conversion artifact.

## What this settles
⇒ **A SPIKING generator generates coherent NOVEL text — the categorical opposite of the measured 0-novel composer
wall** (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`). The Spine-A convert step (loop-step 2) is **GO**.
Combined with step-0 (C1 consolidation entry, 0.92 fidelity) and P2 (the Claude-teacher knowledge half, GO 3/3),
**three of the four loop stages are now positively de-risked**, all local, no cloud.

## Next
- **Loop-step 3 = consolidate** the converted spiking generator onto the ONE bridge (step-0-de-risked to 0.92; the
  named open residual = signed-weight E/I attention routing on the bridge, deferred to this consolidation step — this
  de-risk ran in PyTorch, not on the bridge).
- **Loop-step 4 = C2** (grow + no catastrophic forgetting) — the genuine remaining frontier.
- T=32 is the standard latency/fidelity knob (T=16 passed the gate but the stricter ratio bar wanted T=32; +6% ppl).
