# Biologization sweep: the stream-cortex conversational pipeline is (nearly) all neurons now (2026-06-16, CYCLE 97)

## One-line

Owner directive — *biologize everything we can in this arc*. Result: **every cognitive operation in the
on-bridge stream-cortex conversational pipeline is now a validated neural mechanism (or de-risked as one)** —
the no-confab moat, the cleanup, the binding operation, and the read-out normalization. What remains idealized
is precisely localized: the binding *algebra* (a learned cortical bind = "step 3") and sentence generation.

## Context

CYCLE 95-97 realized a cortex that learns word meanings from the raw TinyStories stream on the real spiking
`SimulationBridge` (rate-Hebbian co-occurrence + population code), and carries the full who/what + no-confab
conversation end-to-end, multi-seed, at 64 AND 320 concepts. But several pipeline stages were still host-side
or idealized. This sweep converts them to brain-based mechanisms, on the stream-LEARNED (correlated) codes.

## The four pieces

| Piece | Was | Now | Result |
|---|---|---|---|
| **No-confab moat** | host confidence threshold | learned Bogacz-Brown anti-Hebbian familiarity gate (catalog D.04, perirhinal repetition suppression) | **GO** — 6 fact-sets, novelty margin **+0.87**, agrees with host 8/8, **0 false-accepts, 0 confabulations**, lesionable (anti-cheat). Works *because* the codes are correlated (the projector is high-capacity on correlated inputs). |
| **Cleanup** (snap to nearest concept) | host `argmax` | spiking NEF thresholded cleanup (Stewart-Tang-Eliasmith, the Spaun cleanup) | **GO** — **0.963** agreement with host argmax on the 320 stream codes (3 fact-sets). |
| **Binding operation** | numpy HRR algebra | on-substrate ±1 coincidence bind (AND-on-ON/OFF) on binarized codes, cleanup vs the graded codebook | **GO (lossy)** — who-Q&A recall **0.92** (6 fact-sets), moat mostly holds (2/48). Binarizing for the bind costs ~8% recall; the graded concept is recovered in cleanup. |
| **Read-out normalization** (double-centring) | host math | per-hub spike-freq **adaptation** + per-concept **feedforward inhibition**, POST-f-I | **de-risked GO** — **96% of host** (6 seeds) *with* realistic rate-coded-pool noise on the means; both ops load-bearing (anti-cheat). The on-bridge FS-feedforward-inhibition circuit is the low-risk realization. |

A notable convergence: the biologized moat (the learned familiarity gate) had **0** false-accepts on the same
320 codes where the host threshold had a seed-variable tail (1 at seed 43) — i.e. **biologizing the moat made
it cleaner** than the host check it replaces.

## What stays idealized / open (honest)

- **The binding *algebra*.** The ±1 coincidence makes the bind *spiking*, but the underlying scheme is a fixed,
  hand-designed, exactly-invertible vector algebra — not a *learned* cortical bind. This is the same status as
  the production composer; a learned (non-algebraic) bind is the genuine **"step 3"** frontier, an arc of its
  own.
- **Sentence generation.** Recall produces structured answers (a concept, a yes/no, a ranked list); turning
  them into grammatical sentences is still templated. Learned neural generation was a clear negative at our
  scale (~4 orders too small). The scale-bound open frontier.

## Verdict

Of the conversational pipeline, the **moat, cleanup, and binding operation are biologized** (validated on the
stream-learned codes) and the **read-out normalization is de-risked biologizable** (96%, pool-noise-robust).
The substantive biologization — every cognitive *operation* — is done. The residuals are the binding *algebra*
(learned-bind = step 3) and sentence generation, each a larger arc.

Runners: `research/runners/_phaseB_biologize_{moat,cleanup,binding_pm1,readout_norm}_*_derisk.py`.
Raw: `research/findings/raw/_phaseB_biologize_*.json`.
