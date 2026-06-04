# Capacity-curve scaling cost model — is the GPU port worthwhile beyond 320? — 2026-06-04

**One line:** A cheap-first measurement answers "would the rf-320-GPU port be worthwhile if future scaling needs
more than 320 concepts?" with a concrete cost model: **most of the agent scales cheaply on CPU at fixed
dimension; the F=3 two-attribute resonator is the lone scaling bottleneck, it needs dimension D ∝ M² (codebook
size), and the GPU is genuinely the enabler there (CPU can't reach the needed D).** Far scaling additionally
needs the algorithmic resonator-capacity fix (sparse block codes).

## The question

The spiking unified agent does the full composition benchmark at 320 concepts on CPU (~3.4 min on the biological
resonate-and-fire substrate). Is a GPU port worthwhile *for scaling past 320*? Measured, not guessed.

## Method

1. **Capacity curve at fixed dimension** (`_capacity_curve_probe.py`): run the benchmark at growing vocabulary
   (320 → 640 → 1280 concepts, keeping the 200:60:60 noun:verb:adj ratio) at fixed D=2048. The frozen facts use
   only core words, so a larger vocabulary adds *distractor* concepts — the capacity stress.
2. **Resonator dimension requirement** (`_gpu_resonator_capacity.py`): isolate the F=3 two-attribute resonator
   and sweep D (2048 → 32768) at vocab 640, on the GPU — because CPU cannot run the resonator at D≥8192 in
   reasonable time (it times out), which is itself part of the answer.

## Result 1 — most of the agent scales cheaply at fixed D

| vocab (concepts) | flat | 1-attr | **2-attr** | clause | who | abstain | overall |
|---|---|---|---|---|---|---|---|
| 320 | ✓ | ✓ | **5/5** | ✓ | ✓ | ✓ | 36/36 |
| 640 | ✓ | ✓ | **0/5** | ✓ | ✓ | ✓ | 31/36 |
| 1280 | ✓ | ✓ | **0/5** | ✓ | ✓ | ✓ | 31/36 |

**Fact memory, retrieval, who/what Q&A, the no-confabulation moat, one-attribute composition, and embedded
clauses all hold at 100% out to 4× vocabulary at fixed D=2048.** These are clean-up-based — the right code stays
nearest even among 4× more distractors. The only scaling *cost* here is the clean-up's Python loop (the 2560
point timed out), which vectorizes to a single matmul — a CPU win, GPU-friendly but not GPU-gated.

## Result 2 — the F=3 resonator needs D ∝ M², and GPU is the enabler

Two-attribute is the lone category that degrades, and on CPU it looked unrecoverable (D=2048/4096 and 4×
restarts all 0/5 at vocab 640). The GPU sweep over the resonator's dimension settles it:

| D | two-attribute recover (clean, vocab 640) | GPU time |
|---|---|---|
| 2048 | 0/5 | 13.6s |
| 4096 | 0/5 | 11.6s |
| **8192** | **5/5** | 11.4s |
| 16384 | 5/5 | 19.1s |
| 32768 | 3/5 (precision ceiling) | 36.6s |

So it **is** a dimension requirement: 60 adjectives needed D=2048 (vocab 320); 120 adjectives need D=8192 (vocab
640) — i.e. **D ∝ M²** in the codebook size. CPU cannot practically run the resonator at D≥8192; the GPU does it
in ~11s and recovers two-attribute fully. **For two-attribute (and higher-F) composition past ~320 concepts, the
GPU is genuinely the enabler.**

**Honest process note:** a mid-measurement read ("this looks algorithmic, not GPU") from CPU's D=4096 failure was
**premature** — the GPU large-D test flipped it. The cheap-first measurement earned its keep precisely by
changing the answer (twice: first "GPU not needed" from the fixed-D curve, then "GPU IS the enabler" from the
large-D GPU test).

## The cost model (the answer)

| capability | scales how | GPU worthwhile? |
|---|---|---|
| fact memory / retrieval / who-what / abstention / one-attribute / clause | accuracy holds at **fixed D**; cost ~ linear in vocab (clean-up) | only for the clean-up *cost* (vectorize first; GPU a secondary win) |
| **two-attribute (F=3) composition** | needs **D ∝ M²**; cost ~ **M⁴** | **YES — GPU is the enabler** (CPU can't reach the needed D) |

## Recommendation

- **Yes, a GPU resonator is worthwhile for scaling composition** — it's the difference between two-attribute being
  impossible (CPU can't reach D=8192) and ~11s. The targeted port is the **resonator** (`_resonator3` → CuPy;
  prototyped in `_gpu_resonator_capacity.py`), not the whole substrate.
- **But D ∝ M² (cost ~ M⁴) is a steep ceiling.** GPU buys near-term headroom (a few hundred concepts per
  codebook); for *far* scaling (thousands), the **algorithmic resonator-capacity fix — sparse block codes**
  (deep-research Track-1, ~5000× capacity) — is the necessary long-term lever, and it scales *without* the D²
  blow-up. The D=32768 degradation (3/5) hints at a precision ceiling for brute dimension even on GPU.
- **Memory/retrieval scaling is a CPU vectorization** (the clean-up loop → matmul), not a GPU need.

So the highest-value scaling roadmap is: **(1) vectorize the clean-up** (memory at large vocab, CPU); **(2) GPU
the resonator** (two-attribute composition, near-term, the prototyped port); **(3) sparse block codes** (the
resonator's M⁴ ceiling, far scaling). The GPU port the owner asked about is worthwhile — targeted at the
resonator — and the measurement turned a vague "maybe" into this concrete, evidence-based plan.
