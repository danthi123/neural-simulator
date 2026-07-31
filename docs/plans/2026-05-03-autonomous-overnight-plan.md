---
type: plan
status: live
date: 2026-05-03
---

# Autonomous overnight plan — SWR investigation
**Started:** 2026-05-03 04:15 EDT
**Authority:** user explicitly granted full autonomy until told to stop.

---

## What we know (data so far)

| Seed | I→W | W→A | Phase 2 corr.move |
|---|---|---|---|
| 42 | 39.0% | 22.0% | 29.6% |
| 43 | 26.0% | 22.0% | 38.2% |
| 44 | 18.0% | 23.0% | 43.5% |
| 100, 101, 102 | (in flight) | | |

**Baseline (no SWR):** I→W 25.3%, W→A 28.5% (n=6, p=0.027)

**The headline:** W→A regresses ~6pp consistently across n=3 seeds when
Phase 3 SWR replay is added. I→W is dominated by seed-to-seed noise.

## Hypotheses (reasoned)

### H0: SWR is fundamentally incompatible with this architecture
The replay procedure drives motor pools coactively with language input,
strengthening language→motor pathways via R-STDP. But the same pathway
is also what's used at eval time, so over-amplifying motor pools that
got more replay events distorts the readout balance.

### H1: Replay distribution bias
The buffer reflects the cascade's natural per-direction frequency.
Dominant directions get more replay events → their language→motor
weights grow disproportionately → eval predictions skew toward those
directions, hurting per-direction discrimination.

**Prediction:** seed-44 W→A weak directions (north + west, both 16%)
should be the LEAST frequent in seed 44's training buffer.

**Test:** Run with `--phase3-balanced-directions` (already implemented
in commit dd354d7). If it rescues W→A to ≥ 28%, H1 is supported.

### H2: Replay overwrites direct PFC bypass
The direct `language_input → motor_X` pathway is short and precise.
Phase 3 replay drives the WHOLE cascade (cortex → str → gpi → thal
→ motor) which uses different intermediate pathways. Strengthening
those intermediate pathways might dilute the direct mapping.

**Test:** Run replay with `motor_replay_drive_pA = 0` so motor cortex
isn't coactively driven during replay — only language regions are.
The direct pathway gets the LTP without the cascade interference.

### H3: 500 replays is past optimum
Phase 2 has 100 episodes × ~22 steps = 2200 plastic events. Phase 3
adds 500 more = +23%. Maybe consolidation overshoots.

**Test:** Sweep `--phase3-replays = {50, 100, 200}` at seed 42.
Cheap (no code change). Curve shape will tell us if there's a
local optimum at smaller doses.

### H4: PFC bypass architecture has limited capacity
Even without ANY interference, can the language_input → motor_X
direct pathway learn the 4-way mapping? If H4 is right, the 28.5%
ceiling isn't a tuning issue — it's a fundamental architecture limit
(e.g. sparse-pattern coding capacity, or motor pool readout
discrimination).

**Test:** Synthetic isolation experiment — skip the cascade, just
do paired (token, action) stimulation training and eval. See what
the upper bound is.

This is the most informative single test because it sets the
maximum possible W→A under the current architecture.

---

## Plan

### Phase 1: 4-seed validation (in progress, ETA ~07:30 EDT)
Already running via `run_swr_remaining_seeds.ps1`. Outputs:
seed 44 (done), seed 100 (in flight), seeds 101 + 102 queued.

**Decision after Phase 1:**
- If 6-seed W→A mean stays at ~22% (consistent with n=3): regression confirmed
- If W→A varies wildly across 100/101/102: regression was a 3-seed coincidence

### Phase 2: H1 (balanced replay, 6 seeds) — IF regression confirmed
Launch via the same PowerShell orchestrator pattern but with
`--phase3-balanced-directions`. ETA ~07:30 → ~14:30.

**Decision after Phase 2:**
- If H1 rescues W→A to ≥ 28%: replay-distribution bias was the
  mechanism; balanced sampling is the fix
- If H1 partially rescues (say 25%): some bias contribution but not
  the whole story; H2 or H3 also at play
- If H1 doesn't help: the mechanism is elsewhere; pivot to H2 or H4

### Phase 3: H4 (PFC bypass isolation, 6 seeds) — runs in parallel design,
sequential GPU
While Phase 2 runs, implement and validate the PFC bypass isolation
runner. After Phase 2 finishes, launch H4 at 6 seeds.

**Why prioritize H4 over H2/H3:** H4 tells us the upper bound of the
architecture — most informative single experiment. H2 and H3 are
follow-up tuning tests that depend on knowing where the ceiling is.

### Phase 4: Synthesis + pivot decision
After H1 + H4 land, we have:
- Whether the regression is fixable by balanced replay
- The architecture's upper bound for W→A

**Possible pivots based on results:**

A. **PFC bypass isolation gives 50-80%** → cascade interference is
   real. Architectural change: train PFC bypass FIRST (before the
   cascade is plastic), THEN unfreeze cascade. Reverse curriculum.

B. **PFC bypass gives ~28%** → architecture limit. Architectural
   change: increase language_input neuron count (currently 256) or
   change pattern coding (currently sparse 0.1). Investigate readout
   discrimination — maybe the issue is motor pool noise, not the
   pathway itself.

C. **Balanced replay rescues regression** → SWR works fine when
   balanced. Add balancing as a default, declare SWR a small win,
   move on to bigger questions (multi-token language? compositional?).

---

## Internal debate: priorities

**Should I drop the SWR investigation and pivot to bigger architectural
changes?** No — we're n=3 into a real result. Completing the validation
gives us either a confirmed regression or a confirmed null. Either way
the data is worth ~7 hours of GPU.

**Should I run more of these one-off variations OR invest in something
bigger like Hodgkin-Huxley language regions?** Stick with the current
question. We have a specific hypothesis (W→A regression) with a
specific test (balanced replay). Closing this loop matters.

**Should I add more biology (theta rhythms, ACh, NE)?** Not right now.
These are stretch goals that don't address the current bottleneck (28.5%
W→A ceiling). After we know whether the bypass architecture itself can
learn 4-way (H4), THEN bigger biology might be the next step.

---

## Documentation hygiene

After each batch:
- Update `research/findings/2026-05-03-swr-multiseed-result.md` with new data
- Commit + push (gitea + github) so the work is durable
- Update `docs/CURRENT-STATE.md` if the bottom-line "what works today"
  changes (e.g. if H4 reveals a new architectural fact)
- Update `CLAUDE.md` if there are new gotchas (e.g. "never use SWR
  without balanced sampling" if H1 rescues)

## Frontend integrity

Webapp must stay usable for the user. Specifically:
- Brain tab Live mode must continue to work as runs come and go
- Inflight panel must show current run, not stale .pid files
- Findings tab must include the new analysis docs
- No console errors

I'll periodically curl `/api/inflight` and `/` to verify nothing has
crashed.

## What I will NOT do

- Take shortcuts in the science. No fudging numbers, no cherry-picking
  seeds. Report what the data shows.
- Make wholesale changes to bridge.py mid-batch (could disrupt running
  experiments).
- Pursue genuinely speculative ideas (e.g. "let's try a completely
  different language region architecture") without empirical
  motivation. Wait for data to point that direction.
- Skip documentation. Every batch gets its own finding entry.
