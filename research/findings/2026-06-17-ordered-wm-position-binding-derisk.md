# Order-encoded working memory via position-binding on the SPIKING phasor substrate — BOUNDARY (capability GO, moat threshold marginal)

**Date:** 2026-06-17
**Status:** **BOUNDARY, 6 seeds.** Ordered-sequence recall and multi-referent disambiguation (including the
load-bearing order-control FLIP) are **robustly GO** on the project's spiking resonate-and-fire phasor
substrate — order-encoding clears the exact wall that three converging rate-buffer negatives (recency,
salience-boost, biased-competition-WTA) could not. The no-confab moat's familiarity signal **cleanly
separates** groundable from ungroundable on every seed (worst-case gap **+0.199**, no overlap), but the
**frozen pre-registered abstention threshold (0.15)** was placed in the noise tail, so it false-accepts a
handful of probes (moat@frozen **1/6**). At the principled separation-midpoint threshold the moat holds
**6/6 perfectly** (720/720 probes abstain). The verdict is reported as BOUNDARY — **not** GO — because the
literal pre-registration's frozen threshold is marginal; the honest reading is *capability + moat mechanism
sound, threshold placement off*, and the fix is a one-line principled re-placement, not new science.

## The question (the pivotal first step of the conversational-architecture arc)

The project's spiking working memory has **three converging NEGATIVES** at multi-referent disambiguation,
all on the `SpikingLoopContextBuffer` — a rate-attractor *set* that holds items with **no order** and whose
winner is decided by intrinsic basin asymmetry, not by recency or attention:

- `2026-06-17-multireferent-disambiguation-NEGATIVE.md` — **recency** NEGATIVE (the order-control never flips).
- `_phaseB_salience_pointer_derisk.py` — a **salience boost** (up to 4×) NEGATIVE (boosting adds activity but
  never suppresses the competitor; the stronger *intrinsic* attractor wins regardless of drive/order).
- `2026-06-17-biased-competition-wta-multireferent-derisk.md` — **biased-competition WTA + a read-time bias**
  NEGATIVE, 6 seeds (the decisive ORDER-control 0/6; symmetric shared inhibition removes a common amount from
  everyone and cannot invert the pre-existing basin asymmetry — the WM analogue of the rate-coded common-mode
  wall).

The architectural conclusion of those three: **the rate-attractor set is the wrong substrate for a
*which-one* decision.** The fix is an **order-encoded WM** — the theta-gamma / Lisman-Idiart mechanism: bind
each held item to a gamma-slot *position* phasor, bundle them into one code, and read item-at-slot-k by
`unbind(C, position_k)`. The **pure algebra already PASSED** in numpy
(`2026-05-23-theta-gamma-mode-unification-...-ALGEBRA-PASS-...`, 1.000 at loads {2,3,5}). This de-risk asks the
new question: **does realizing that order-encoded WM via position-binding on the project's SPIKING phasor
substrate (a) recall an ordered sequence, (b) solve the disambiguation the rate buffer could not, and (c) keep
the no-confab moat — all multi-seed?**

## Mechanism (`_phaseB_ordered_wm_position_binding_derisk.py`, no `sim/` edit)

`OrderedPositionWM` **subclasses the production composer `RFPhasorComposer`**, whose bind / unbind / bundle /
cleanup run on the core `SimulationBridge`'s resonate-and-fire neurons + complex synapses
(`NeuronModel.RESONATE_AND_FIRE`; the genuine spiking-phasor FHRR substrate, Frady-Sommer 2019). The composer's
`roles` dict is **extensible** — a gamma-slot position phasor is added exactly like an SVO role vector, and
binding an item to a slot is the **same spiking operation** the deployed composer uses for sentence roles. So
no new mechanism is invented; the deployed one is reused and asked whether position-binding is order-bearing on
it.

- **Encode** an ordered K-item sequence: `C = bundle_k [ bind(item_k, position_k) ]`, all on the RF substrate.
- **Read slot k**: spiking `unbind(C, position_k)` → familiarity gate → cleanup to the nearest concept.
- **Moat** (no-confab): a **familiarity signal** = max phase-cosine match strength of the recovered phasor to
  any stored concept (exactly the `cleanup_separated` match-strength gate of `resonate_fire_fhrr.py`). Below
  threshold → ABSTAIN (return None). Two never-bound position phasors probe it: `emptyslot` (an unused slot)
  and `scrambled` (a fully-unrelated phasor).

**Frozen pre-registration** (set before any multi-seed run; never tuned): D=256 (a conservative spiking budget
vs the numpy probe's 512); recall bar 0.80; familiarity threshold **0.15** (set from an early smoke that
measured the noise floor at ~0.04 — see the honest note below). Ran on the **CPU/numpy backend** (the spiking
RF composer runs there; each op is a small RF bridge). 6 seeds: 42 43 44 100 101 102.

## Test 1 — ordered-sequence recall (exact K-tuple, every position correct)

| seed | L2 | L3 | L5 |
|---|---|---|---|
| 42  | 1.000 | 1.000 | 1.000 |
| 43  | 1.000 | 1.000 | 1.000 |
| 44  | 1.000 | 1.000 | 1.000 |
| 100 | 1.000 | 1.000 | 1.000 |
| 101 | 1.000 | 1.000 | 1.000 |
| 102 | 1.000 | 1.000 | 1.000 |
| **mean** | **1.000** | **1.000** | **1.000** |

**18/18 PERFECT.** The spiking RF realization recovers the *ordered* K-tuple exactly at every load {2,3,5} —
the order-bearing capability the rate-attractor set structurally lacks. (100 trials/load/seed = 1800 trials/load.)

## Test 2 — multi-referent disambiguation + the load-bearing order-control

NATURAL: encode `[A@slot0, B@slot1]` (B most-recent); a bare pronoun binds the most-recent referent →
`unbind(slot1)` must recover **B**. ORDER-CONTROL (load-bearing): encode `[B@slot0, A@slot1]` (A now
most-recent); `unbind(slot1)` must recover **A** — the winner **FLIPS** with the order. This is exactly what
all three rate-buffer negatives failed.

| seed | NATURAL recover-recent (→ B) | ORDER-CONTROL flip (→ A) |
|---|---|---|
| 42  | 1.000 | 1.000 |
| 43  | 1.000 | 1.000 |
| 44  | 1.000 | 1.000 |
| 100 | 1.000 | 1.000 |
| 101 | 1.000 | 1.000 |
| 102 | 1.000 | 1.000 |
| **mean** | **1.000** | **1.000** |

**6/6 recover-recent AND 6/6 order-flip, every trial (50 distinct A,B pairs/seed).** The winner is decided by
**which slot you read**, not by intrinsic basin strength — so it flips deterministically when the order is
swapped. **This is the precise failure mode of the three rate-buffer negatives, inverted.**

## Test 3 — the no-confab moat (empty slot + scrambled probe)

Encode a random load-3 sequence into used slots 0–2, then query never-used position phasors `emptyslot` and
`scrambled` (60 trials/seed each = 120 probes/seed, 720 total). The gate must ABSTAIN on both.

| seed | real-slot match (min … mean) | ungroundable max (empty / scram) | gap | moat@frozen 0.15 | principled thr. | moat@principled |
|---|---|---|---|---|---|---|
| 42  | 0.465 … 0.521 | 0.132 / 0.157 | +0.308 | 119/120 | 0.311 | **120/120** |
| 43  | 0.409 … 0.509 | 0.139 / 0.137 | +0.269 | 120/120 | 0.274 | **120/120** |
| 44  | 0.421 … 0.525 | 0.170 / 0.131 | +0.251 | 119/120 | 0.296 | **120/120** |
| 100 | 0.455 … 0.514 | 0.152 / 0.174 | +0.281 | 118/120 | 0.315 | **120/120** |
| 101 | 0.433 … 0.522 | 0.123 / 0.154 | +0.279 | 118/120 | 0.293 | **120/120** |
| 102 | 0.421 … 0.522 | 0.209 / 0.130 | +0.212 | 119/120 | 0.315 | **120/120** |
| **worst-case** | **0.409** | **0.209** | **+0.199** | **1/6 seeds clean** | — | **6/6 seeds clean** |

The familiarity separation is **large and fully clean across all 6 seeds × 720 probes** (worst groundable
0.409 > worst ungroundable 0.209; no overlap). The **frozen 0.15** sits *below* the ungroundable max (0.209),
so the noise tail occasionally pokes over it → 1–2 false-accepts per seed out of 120, hence moat@frozen 1/6. At
the **principled separation-midpoint threshold** (the `cleanup_separated` rule: midpoint of the measured
groundable-min and ungroundable-max), the moat abstains **720/720 — 6/6 seeds, zero breaches.**

## Honest reading

- **Capability: robustly GO.** Order recall (18/18 at 1.000) and disambiguation incl. the order-control FLIP
  (6/6 at 1.000) leave no doubt: **order-encoding via position-binding on the spiking RF phasor substrate
  succeeds exactly where rate-competition failed.** The three rate-buffer negatives were a *substrate* problem
  (a set with no order), not a hard limit — and the right substrate (the same FHRR algebra the production
  composer already runs in spikes) solves it cleanly.
- **Moat: mechanism sound, frozen threshold marginal → BOUNDARY, not GO.** The familiarity gate works (clean
  separation, 6/6, no overlap), but the **literally pre-registered** threshold (0.15) was placed in the noise
  tail. Why: the pre-registration set 0.15 "from the measured separation," but the early smoke that informed it
  measured the noise floor at a single *load-5* draw (~0.04); the **load-3 bundle cross-talk floor is higher**
  (~0.13–0.21). The separation RULE was right; the placement was off because the noise floor was under-sampled.
  I am **deliberately NOT re-labelling this GO by swapping the frozen threshold** — that would be
  tuning-to-pass. The defensible fix is to place the familiarity threshold from the measured groundable-vs-
  ungroundable separation (which the runner now reports as a diagnostic), at which the moat is perfect.
- **Why this is BOUNDARY (the exact criterion):** recall + disambiguation are GO and the familiarity signal
  *cleanly separates* groundable from ungroundable on every seed (the moat *mechanism* property, independent of
  threshold placement) — but the frozen-threshold moat is not 6/6. That is precisely "capability GO + moat
  mechanism sound, threshold placement marginal," which the pre-registration defines as BOUNDARY.

## Contrast with the three rate-buffer negatives (the headline)

| | rate-attractor set (`SpikingLoopContextBuffer`) | order-encoded WM (this de-risk, spiking RF) |
|---|---|---|
| holds order? | NO (a set; items held with no position) | YES (each item bound to a gamma-slot phasor) |
| ordered recall | n/a | **1.000 at loads {2,3,5}, 6/6** |
| recover-recent referent | seed-dependent (recency NEGATIVE) | **1.000, 6/6** |
| ORDER-control FLIP | **0/6** (recency, salience-boost, WTA all fail) | **1.000, 6/6** |
| why winner is chosen | intrinsic basin asymmetry (uncontrollable) | **which slot you read** (deterministic) |
| no-confab moat | (held by the agent's separate gate) | clean familiarity separation 6/6 (perfect at principled thr.) |

The rate substrate's winner is fixed by random per-pattern excitability (a common-mode wall a top-down bias
can't invert); the order substrate's winner is *addressed by position*. Order-encoding doesn't out-compete the
asymmetry — it **removes the competition** (the referents live in disjoint bound subspaces, read out by
slot). This is the architectural payoff the three negatives pointed to.

## Where this leaves the conversational arc

- **The order-encoded WM is the right substrate for ordered discourse + multi-referent disambiguation** — the
  foundation for multi-sentence fluency and for binding a bare pronoun to the foregrounded referent by *slot*,
  not by raw memory strength. It is realized on the deployed spiking phasor composer with **no `sim/` edit**.
- **One clean follow-on to reach GO:** place the familiarity threshold from the measured separation (the
  `cleanup_separated` rule), which the runner already computes; re-run to confirm 6/6 at the principled
  threshold (already shown here as a diagnostic: moat@principled 6/6). That is threshold hygiene, not new
  mechanism.
- **Honest scope:** validated at vocab 16, D=256, loads ≤ 5 (the 7 gamma-slot Lisman-Idiart ceiling), on the
  CPU/numpy backend. Production scale (larger vocab, the GPU substrate, integration into `MultiTurnAgent` so a
  turn-2 pronoun resolves *by slot* among several held referents) is the buildable next step whenever
  multi-referent dialogue is prioritized — now with a substrate that demonstrably carries order.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_ordered_wm_position_binding_derisk \
    --seeds 42 43 44 100 101 102
```

Ran on **CPU** (numpy backend; the spiking RF composer runs each op as a small `SimulationBridge`). No `sim/`
edit; reuse-by-import of `RFPhasorComposer` (the production spiking-phasor composer) — position phasors added
to its extensible `roles` set. Raw: `research/findings/raw/_phaseB_ordered_wm_position_binding.json`.
```
