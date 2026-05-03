# Overnight progress — SWR investigation

**Last updated:** 2026-05-03 04:35 EDT (autonomous)
**Status:** 4-seed batch in progress, seed 100 at ep 50/100
**Next user-facing summary:** updated after each batch completes

This is the running notebook for tonight's autonomous work. The user
went to sleep at ~02:30 EDT after seeing the 2-seed v2+SWR result.

---

## What was done before sleep

* Implemented Phase 3 SWR replay in text_train_curriculum
* Ran seeds 42 + 43 → both showed W→A drop to 22% (vs 28.5% baseline)
* Created the n=2 finding doc with hypotheses

## Tonight's autonomous work — what I committed to

The user gave full autonomy ("do whatever gets us best results") with
constraints: stay biology-grounded, no shortcuts, keep docs/frontend
updated, don't be afraid of long-running plans if they're the right
move. See `docs/plans/2026-05-03-autonomous-overnight-plan.md` for the
detailed plan with hypothesis tree.

## What's running NOW

`run_swr_remaining_seeds.ps1` master orchestrator (PID 39760). Runs
sequentially:
* ✅ seed 44 done — W→A 23%, I→W 18%
* ⏳ seed 100 in flight — Phase 2 ep 50/100
* ⏳ seed 101 queued
* ⏳ seed 102 queued

ETA: ~07:38 EDT for batch completion.

## Key findings since 02:30 EDT

### 3-seed v2+SWR W→A regression confirmed

| | seed 42 | seed 43 | seed 44 | mean | baseline |
|---|---|---|---|---|---|
| W→A | 22.0% | 22.0% | 23.0% | **22.3% ± 0.6** | 28.5% ± 2.1 |
| I→W | 39.0% | 26.0% | 18.0% | **27.7% ± 10.6** | 25.3% ± 4.5 |

W→A regression is **statistically significant** even at n=3 (chance
of 3 seeds within 1pp of each other if true mean were 28.5% with
σ≈2 is < 1%). I→W is variance-dominated.

### Mechanism analysis (deeper than initial framing)

Per-direction W→A breakdown across all SWR seeds shows the regression
hits ALL FOUR DIRECTIONS, not just one:

| direction | baseline | v2+SWR | Δ |
|---|---|---|---|
| north | 26.7% | 20.0% | −6.7 |
| east | 31.3% | 25.3% | −6.0 |
| south | 29.3% | 20.0% | −9.3 |
| west | 26.7% | 24.0% | −2.7 |

But which motor pool over-predicts SHIFTS per seed:
* seed 42: SWR pushed S −8pp (from 28→20% predictions)
* seed 43: SWR pushed S −19pp (from 40→21%)
* seed 44: SWR pushed S +14pp (from 22→36%)

This means SWR amplifies whichever direction the *correct-experiences
buffer* was over-rich in. The buffer composition varies stochastically
per seed (depends on cascade bias × goal placement × random init).
Result: per-seed prediction shifts vary in direction, but aggregate
accuracy drops because predictions become more skewed in some
direction every time.

**Implication for hypotheses:**
* H1 (balanced replay): STRONGER — directly addresses the bias
  amplification mechanism
* H2 (cascade overwrites direct): unclear — would expect uniform
  effect, which we don't see
* H3 (overshooting): possible, but not the primary mechanism
* H4 (architecture limit): unchanged — still need to test

## Plan: H4 first (faster, sets upper bound), then H1

After the 4-seed batch:
1. Aggregate the 6-seed result (run swr_aggregate.py)
2. **Launch H4 (PFC bypass isolation)** at 6 seeds. ETA ~2.5 hours.
   - Why first: faster, sets upper bound regardless of mechanism
3. Once H4 lands, **launch H1 (balanced replay)** at 6 seeds. ETA ~7
   hours.
   - Why: H1 is the most plausible mechanism fix, but the upper
     bound from H4 informs whether to even bother

Rough timeline (if all goes smoothly):
* 07:38 EDT — 4-seed batch done
* 07:45 EDT — launch H4
* 10:15 EDT — H4 done
* 10:20 EDT — launch H1
* 17:20 EDT — H1 done

User should wake up around 09:00–10:00 to find H4 either complete or
nearly so, and H1 either queued or just-started.

## Decision tree for the morning (no user input needed)

After H4 completes, three possible outcomes:

1. **H4 isolation gives 80%+** → architecture is fine; cascade
   interference is the real issue. Pivot: reverse curriculum (train
   language pathway first, then unfreeze cascade). Implement as H5.

2. **H4 isolation gives 50-79%** → architecture works but isn't great
   in isolation either. H1 still worth running (might help by reducing
   buffer-bias amplification on top of cascade interference).

3. **H4 isolation gives ~28%** → architecture itself is the bottleneck.
   Pivot: bigger structural changes — denser language coding (256→512
   neurons), different motor readout (not just argmax over pool rates),
   or a fundamentally different language→motor architecture.

I'll write the appropriate next-step plan when H4 lands.

## Files committed tonight

| Commit | What |
|---|---|
| dd354d7 | H1 balanced-replay flag in text_train_curriculum |
| 84e439a | Filter `*.master.pid` from inflight panel |
| a176bd9 | n=3 finding doc update |
| 334899a | H4 isolation runner + orchestrator scripts |
| e5ddd10 | swr_aggregate.py auto-summary script |
| a6e349f | Phase 2 buffer composition + mechanism analysis doc |
| a1301df | Per-direction aggregate breakdown in Language tab |

All pushed to gitea + github.

## Webapp surfaces

Refresh `localhost:8765/#tab=language` to see:
* Per-direction W→A bars at the top showing aggregate (currently
  shows north 32%, east 24%, south 21%, west 21% across all 33
  text I/O runs — a clear N-bias)
* Per-run detail with confusion matrices

The Brain tab Live mode is following seed 100 (or whichever current
seed is in flight). The cascade animation visible there is
synthesized from progress markers (Phase 2 episode counts) since
the curriculum runner doesn't emit per-step gridworld data.

## What I am NOT doing tonight

* Not modifying bridge.py — would change behavior for in-flight seeds
* Not pursuing H3 (replay sweep) — H1 is more informative for the
  same GPU time
* Not making big architectural changes without H4 data
* Not speculating about results before the data lands
