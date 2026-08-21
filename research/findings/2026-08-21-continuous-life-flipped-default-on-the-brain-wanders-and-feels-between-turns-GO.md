---
type: finding
status: positive
date: 2026-08-21
mechanism: continuous-state-engine-default-flip
lane: integration
seeds: [42]
seed-waiver: the flip's no-regression is BYTE-IDENTICAL BY CONSTRUCTION (the drive only prepends a lead when
  recent_wander() is non-None, which requires a prior idle tick) + a deterministic OFF-vs-ON panel through the real
  handler; the load-bearing evidence is the 0-diverged contrast + VRAM stability, not a stochastic effect size.
instrument: research/runners/_continuous_default_flip_soak_cupy.py — a sustained OFF-vs-ON soak through the REAL
  brain_chat handler on cupy: (S1) byte-identical core on ordinary turns, (S2) no GPU-memory pileup, (S3) drive still
  load-bearing, (S4) idle loop robust at scale, with tools.verdict.Verdict.
runner: research/runners/_continuous_default_flip_soak_cupy.py
artifacts:
  - research/findings/raw/_continuous_live_cupy/default_flip_soak.json
---
# The between-turn CONTINUOUS LIFE is FLIPPED DEFAULT-ON — the brain now WANDERS + FEELS between messages by default (GO, board Trunk-A)

Artifact: research/findings/raw/_continuous_live_cupy/default_flip_soak.json (VERDICT GO). Every number below is read
from that artifact.
<!--derived-->

**One line.** `BRAIN_CONTINUOUS` + `BRAIN_CONTINUOUS_DRIVES` are flipped **DEFAULT-ON** — the mission-defining flip.
The brain is no longer inert between your messages: an idle tick lets a THOUGHT wander (the self-initiation CA3
selects a concept off its own store) and RELAXES the felt mood (re-reading the spiking affect ladder), and the next
turn LEADS with what it was mulling. This is the substrate's ALIVENESS — the LLM-surpassing differentiator (owner
2026-08-19 reframe) — now live in production, not opt-in. Verified safe through the REAL brain_chat handler on cupy.

## The flip (default-ON with a byte-identical escape)
- `webapp/continuous_engine.py`: `continuous_enabled()` returns `_CONTINUOUS_DEFAULT_ON=True` when `BRAIN_CONTINUOUS`
  is unset; `BRAIN_CONTINUOUS=0` disarms the tick loop (byte-identical to the pre-flip behaviour).
- `webapp/server.py`: the `BRAIN_CONTINUOUS_DRIVES` wander-lead block now defaults `"1"`; `=0` is its escape.
Mirrors the affect-drives / gnw-multistep flip pattern.

## The verify (cupy, real handler) — GO
<!--derived-->
- **no-regression: 0/7 diverged** — ON is byte-identical (answer / recalled_svo / abstain) to the `=0` escape on every
  ordinary turn (recall / abstain / self / open-ended), through the REAL brain_chat handler.
- **no GPU-memory pileup: PASS** — VRAM drop 304 MiB across the whole soak (the ~S3 organ builds); the cupy pool went
  46.5 → 72.6 → 72.6 MB (a one-time first-use warm-up, then FLAT — settled, not a per-turn leak).
- **drive still load-bearing: PASS** (a recorded wander is handed to the next turn once then consumed; None under the
  `=0` escape) and **idle loop robust at scale: PASS** (20 concurrent idle sessions relax + re-read affect, no error).
- byte-identical BY CONSTRUCTION: an ordinary turn has no pending wander (`recent_wander()` is None) ⇒ no key, no lead
  ⇒ the reply path is untouched; the drive only ACTS on a turn that FOLLOWS an idle tick.

## Instrument correction (verify-go: fix the instrument, don't lift a metric from a negative run)
<!--derived-->
The FIRST verify returned NO-GO — but solely on a mis-calibrated `no_pileup` check: a raw 10% pool-%-growth threshold
false-alarmed on the 46.5 MB baseline (the 26 MB one-time warm-up reads as +56%). The raw data proved no leak (pool
flat at 72.6 MB after the turns; VRAM stable). The check was corrected to the REAL leak signal — VRAM stability
(drop <= a floor, since S3 legitimately builds the affect+selfinit organs) AND pool-settle after the turns — and the
re-run is a clean GO. A refutation needs its instrument verified exactly as much as a confirmation.

## What this unlocks + honest scope
The brain now has a continuous life BY DEFAULT: learn-through-use / feeling / trains-of-thought live in the substrate's
between-turn dynamics (the SEEDS are now an always-on ENGINE). Residual host scaffolds (unchanged): the idle-tick
SCHEDULER is a host timer ("WHEN to think" is host-clocked, not a neural default-mode oscillation); the mood-relax EMA
constants + the affect operating point are host-tuned; the wander rides the self-initiation organ's co-resident CA3
(one-brain merge #1). The load-bearing SPIKING parts are the CA3 wander selection + the affect-ladder re-read (both
lesion-proven in _continuous_drive_loadbearing_cupy.py). FUNCTIONAL correlate of an inner life, NOT a phenomenal claim.
Escape: `BRAIN_CONTINUOUS=0`. The soak is a 7-pair panel; the no-regression is byte-identical by construction +
empirical. NO `sim/` edit (a host anchor flip on already-wired, already-lesion-verified continuous machinery).
