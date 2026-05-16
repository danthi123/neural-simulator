# Generator Increment 3 — maxed-capacity scan: honest NEGATIVE (memorization, not structure)

## TL;DR

The Increment-2 negative left one honest open question: *was the
failure a student-**capacity** problem?* Increment 3 answered it
directly by training the largest char-level spiking student that
reasonably fits a single RTX 3090 — ~10x the Increment-1 tiny net
(3 hidden layers x 512 units, T=96 BPTT unroll, 1,000,000-char local
English corpus, 400 epochs) — with a kill-safe per-epoch
checkpoint so it could be paused for gaming and resumed.

**The pre-registered held-out gate FAILED.** Scaling capacity did
**not** produce generalizable sequential structure. The maxed
student *memorized* its training windows (training loss → 0.057) but
on held-out windows it scored **21.65** — roughly **5x worse than
uniform chance** (ln V = 4.57) and **418% worse than the tiny
Increment-1 baseline (4.18)**. The permuted-control student behaved
identically (held-out 23.22). More capacity bought more
memorization, not more learning.

This is a real, honest negative. The gate was **not** tuned, the
config was **not** cranked further, and the result is reported as it
landed.

## The pre-registered gate (held-out loss, fixed >= 10% bar)

| metric | REAL | PERMUTED control | uniform chance (ln V) | Inc-1 tiny baseline |
|---|---|---|---|---|
| final TRAINING loss | 0.0566 | 0.0677 | — | — |
| **HELD-OUT loss** | **21.65** | **23.22** | **4.57** | **4.18** |

- REAL held-out is only **6.75%** below PERMUTED (gate needed
  >= 10%) → **FAIL**.
- Both held-out losses are ~5x **above** uniform chance — the
  networks generalize *worse than guessing*. The tiny ~6.75% REAL/
  PERMUTED gap is noise between two catastrophically over-fit nets,
  not learned structure.
- Held-out windows are **provably disjoint** (zero character
  overlap) from the 2000 training windows: the trainer's sampler
  draws one deterministic `rng.integers` per window, so the exact
  training start-indices were reconstructed from the training seed
  and every held-out start within `seq_len` of any of them was
  rejected. Unit-tested (`tests/test_scaled_heldout_eval.py`).

## An honest mid-flight correction (anti-cheat discipline working)

The first version of the Task-4 gate compared **training** loss
(`loss_history[-1]`): REAL 0.0566 vs PERMUTED 0.0677 → it would have
printed **"PASS"**. That "PASS" was a pure **memorization artifact**
— a ~600K-parameter net trivially memorizes 2000 fixed windows, and
English windows are marginally more compressible than shuffled-char
windows, so REAL edges PERMUTED on *training* loss with zero learned
generalization. The Increment-3 plan pre-registered **held-out**
loss precisely to defeat this. Measuring held-out (the metric the
plan actually specified) flips the verdict to the honest FAIL. The
gate **bar** (>= 10%, the permuted-control logic) was never changed;
only the measured quantity was corrected to the pre-registered one.
This is the same class of error the 2026-05-03 permuted-label
control caught project-wide — surfaced and corrected here before any
result was claimed.

## A real bug was found and fixed en route (not papered over)

The scaled run initially produced a flat loss pinned at *exactly*
ln(V), identical for REAL and PERMUTED. Root cause (confirmed by a
diagnostic before any fix, not guessed): `cross_entropy_loss_np` /
`softmax_grad_np` in `sim/bptt_snn.py` computed `exp(logits)` with
no log-sum-exp stabilization. Inc-3 logits are rate-coded (sum of
output spikes over the unroll), so at T=96 they reach ~96;
`exp(96) ≈ 5e41` overflowed → `inf` loss → `nan` weights → dead
net. Phase 2.1/2.2 only escaped this because their validated T=32
kept `exp` in float range — the bug was latent and the larger Inc-3
unroll exposed it. Fixed at the root with the textbook log-sum-exp
trick (mathematically exact; softmax/CE are shift-invariant), one
DRY change benefiting Phase 2.1/2.2 and Inc-3. 17/17 BPTT tests
green incl. two new regression tests. Commit `f8af26a`.

## What this means (the honest, useful finding)

Increment 2 diagnosed "student-capacity-bound" and explicitly
deferred the capacity scan as legitimate next-increment science.
Increment 3 ran that scan honestly and **falsifies the optimistic
reading of Increment 2**: it is *not* the case that "just add
capacity and it will learn." A maxed local char-level surrogate-
gradient BPTT spiking student (Neftci, Mostafa & Zenke 2019), on the
largest config that reasonably fits a 3090, **overfits rather than
generalizes**. The bottleneck is not raw capacity — it is the
data/learning regime: 2000 fixed windows is a memorization regime
for a 600K-parameter net, and the architecture shows no inductive
bias that converts capacity into generalizable sequence structure at
this scale.

**Decision-relevant conclusion:** self-contained, locally-trained
fluent char-level generation via this BPTT-SNN approach is out of
reach on a single RTX 3090 under the no-cheating / local-only
constraints. This closes the "is it capacity?" question with an
honest no.

## One characterized limitation (transparent, not an excuse)

The training set was 2000 fixed sampled windows (the pre-registered
config). That is small enough that memorization dominates. A future
increment could test corpus *streaming* / vastly more windows so the
net cannot memorize. This is offered as a concrete, falsifiable next
hypothesis — **not** as a retroactive reinterpretation of this
FAIL. The pre-registered gate failed on the pre-registered metric;
the permuted control used the identical sample budget, so the
comparison is valid as run. No goalpost-moving.

## The robust, validated asset is unchanged

Generation remains unproven. The project's genuinely validated,
non-fragile contribution is still the **trustworthy continual
memory with no-confabulation abstention** (G.20 sparse distributed
ensemble: 160 concepts @ 100% / 320 @ 98.4% per-bridge, multi-seed;
no catastrophic forgetting per Marr/McClelland complementary-
learning-systems; refuses to make things up). That result is
anti-cheat-validated and untouched by this negative.

## Files

- `research/runners/scaled_generator_train.py` (kill-safe resumable
  trainer, `--permute-corpus` control)
- `research/runners/scaled_heldout_eval.py` (+ pure tests
  `tests/test_scaled_heldout_eval.py`) — pre-registered held-out
  metric, provably-disjoint windows
- `research/runners/scaled_capacity_gate.py` — held-out gate
  (fixed >= 10% bar)
- `sim/bptt_snn.py` — log-sum-exp fix (commit `f8af26a`,
  `tests/test_bptt_snn.py`)
- `research/findings/raw/g11_bg/scaled_gen_{real,perm}.log`,
  `*.heldout.json`, `scaled_capacity_gate.json`
- Plan: `docs/plans/2026-05-16-generator-increment3-scaled-capacity-scan.md`
- Prior: `research/findings/2026-05-16-generator-increment2-distillation-NEGATIVE.md`
