---
status: live
type: finding
lane: integration
date: 2026-08-27
mechanism: closes R1's two declared scaffold residuals on the SAME learned cross-region edge — the plasticity RULE upgraded from two-factor Hebbian to reward-DEFERRED (strict three-factor) STDP, and the candidate TOPOLOGY widened from a host-hardcoded pair to an unbiased 6-edge set whose winning wire self-selects and tracks a per-seed RANDOM ground truth
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_onebrain_integration_r2_threefactor_selforganized.py
artifacts:
  - research/findings/raw/_onebrain_integration_r2_threefactor_selforganized_6seed.json
builds_on:
  - research/findings/2026-08-27-onebrain-integration-R1-wm-to-comprehension.md
  - research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md
---

# One-brain INTEGRATION R2 — three-factor credit-gated plasticity + self-selecting topology (6/6 GO, two residuals precisely scoped)

R1 grew the FIRST learned cross-region edge (`d6_multiref_wm` referent → comprehension role competition) but
explicitly declared it **NOT strict `self-organized`** per `docs/TERMS.md`: the rule was pure two-factor
(pre×post) Hebbian, and the candidate edge set was a host-hardcoded pair (`w{0,1}→{sel_agent,sel_patient}`,
the ONLY plastic synapses the runner ever injected). This finding builds R2, the design's next rung
(`2026-08-27-onebrain-integration-phase-DESIGN.md` §4): **(R2-a)** upgrade the rule to genuine three-factor
credit-gated plasticity, and **(R2-b)** widen the candidate set to an unbiased topology whose winning wire is
discovered by experience, not pre-chosen by the runner. **6/6 GO on the SAME F1–F4 functional gate + migration
invariant as R1, plus both new emergence controls** — but the honest scope is narrower than a bare
"self-organized" headline, and this doc is deliberately explicit about exactly where each closure stops.

## Verdict — GO/PARTIAL per residual, and precisely what is still host-supplied

<!--derived-->

**R2-a (three-factor): GO, unqualified within its own scope.** The plasticity **RULE** is genuinely three-factor
— on the substrate's own reward-modulated STDP (`sim/bridge.py`'s existing "C2: Reward-Modulated Plasticity"
block), with `reward_defer_stdp_weight_update=True` so STDP alone **never** writes a weight; only a same-episode
`current_reward_signal` pulse converts the eligibility tag into an actual change. Credit is proven load-bearing
BOTH directions, 6/6 seeds: withholding it entirely produces **exactly zero growth** (every candidate edge stays
at W0=0.05 to six decimal places); decorrelating it from correctness collapses selectivity to 13–26% of the
intact value. **Honest residual, stated up front because it is the load-bearing caveat**: the credit
**signal's VALUE** (`current_reward_signal`) is still a **host-delivered scalar** — the runner sets it directly
from its own ground-truth bookkeeping of which episode is "correct," not from a spiking dopamine/value
population computed by the brain's own error or success detection. The RULE that consumes it is real substrate
machinery; the SOURCE of the third factor is not yet. This is exactly the design doc's own §2 note that a
spiking-pool-delivered modulator (`set_transmission_gate`/`couple_gate_to_pool`) is the fuller version, deferred
because it needs the neuromodulator-subsystem pool seam.

**R2-b (self-organized topology): PARTIAL — GO on a precisely-scoped claim, not the full `docs/TERMS.md` bar.**
`docs/TERMS.md`'s `self-organized` entry requires **four** things hold: both factors of the learning rule are
substrate-computed, the **target selection** is substrate-decided, and the **allocation of any slot/unit** is
substrate-decided. R2-b closes the first three (see F-gate below) but **not** the fourth: WHICH physical WM pool
holds a given referent on a given trial is still decided by the runner's host-directed `LOAD_PA` targeting,
unchanged from R1 — this is a different, harder capability (experience-driven allocation of the WM
representation itself) that R2 does not attempt. What IS closed and verified 6/6: the **cross-region WIRING
PAIR** — which of an UNBIASED 6-edge candidate set (`w{0,1,2}→{sel_agent,sel_patient}`, structurally identical,
no host-favored pair) is the functionally correct one — self-selects via the plasticity+credit rule and
correctly TRACKS a **per-seed RANDOM** ground-truth assignment (`_role_assignment`, computed before training,
never read by the wiring or verification code). Also unclosed: the candidate REGION PAIR itself (d6 slot pools →
comprehension sel pools) is still a host-chosen pair; R2-b closes WHICH of an unbiased set within that pair
wins, not whether that pair is the one tried at all.

## The F-gate + emergence result (6/6 on every arm)

<!--derived-->

`research/findings/raw/_onebrain_integration_r2_threefactor_selforganized_6seed.json`: **F1 6/6 · F2 6/6 · F3
6/6 · F4 6/6 · lesion-recovers-migration 6/6 · R2-a three-factor 6/6 · R2-b topology-tracks-assignment 6/6 ·
no-corruption 6/6.** The shuffled-credit control's topology-match rate is **0/6** (vs intact's 6/6) — a decisive
contrast, well inside the pre-registered `<=4/6` control bar.

## The mechanism (config-only; NO `sim/` edit)

<!--derived-->

ONE shared `merge_organs([d6_multiref_wm, comprehension], wire=True)` pool, exactly R1's substrate. An UNBIASED
candidate topology `w{0,1,2}→{sel_agent,sel_patient}` (6 structurally-identical edges, all seeded at W0=0.05) is
the SOLE plastic synapse set (`cp_plasticity_rate_gain` whitelist, R1's "one-line inversion" reused verbatim).
Config union (`dataclasses.replace` to strip `enable_stdp`/`enable_reward_modulation` from the two organs'
declared-False config, then a config-only extra descriptor re-supplies both True + `reward_defer_stdp_weight_
update=True`) turns on the bridge's existing reward-DEFERRED STDP: pre×post spike coincidence tags a LOCAL
eligibility trace (`cp_eligibility_trace`); the trace only becomes a real weight change when a same-episode
`current_reward_signal` pulse arrives (delivered ONLY in the tail 8 of each 30-step cue-drive window, after the
Wong-Wang WTA competition has settled — see engineering note 7 below). Training schedules 100 episode-PAIRS
(400 total episodes, BALANCED 1:1 correct:distractor — see note 8): a per-seed-RANDOM role assignment
(`_role_assignment`, a local RNG the wiring code never reads) designates which of the 3 candidate pools plays
agent, patient, and control-distractor; correct-agent and correct-patient episodes are credited, distractor
episodes (real co-activation on the control pool, deliberately never credited) are not.

## Engineering notes carried forward + newly earned (STDP-specific; read before extending this arc)

<!--derived-->

R1's four notes still apply verbatim (config-only descriptor pattern; the plasticity whitelist; clearing
`cp_conductance_g_nmda_recurrent`; migration invariant = decision-preservation, not bit-identity). R2 earned
three more, all STDP-specific (R1's pure Hebbian never hit them):

1. **`_run_one_simulation_step()` does not advance `runtime_state.current_time_ms`.** Hebbian's rate-window
   trace doesn't care about absolute time; STDP's `delta_t` computation does. Without a manual
   `bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms` after every step, every spike shares one
   timestamp, `delta_t≡0`, and `fused_stdp_weight_update` returns the unchanged weight — STDP is silently inert
   while every other liveness signal reads healthy. This is a documented, self-warning guard
   (`sim/bridge.py:9935`, "STDP IS INERT"); it fired harmlessly during this runner's own 40-step pre-training
   settle (before the clock-advance loop starts) and never again once training began.
2. **`cp_last_spike_time` must be reset to its −1000.0 init sentinel every episode.** A stale "last fired"
   timestamp on an otherwise-silent neuron becomes a spurious STDP-eligible pair the moment the OTHER endpoint
   fires in a LATER episode (adjacent episodes sit well within the default 100ms STDP window). `cp_eligibility_
   trace` must be zeroed too, or a reward pulse — a GLOBAL scalar applied to the WHOLE eligibility vector —
   credits a stale, unrelated synapse's leftover tag from a prior episode.
3. **Reward-timing + eligibility-tau interact with the WTA competition's loser.** Delivering reward for the
   FULL cue-drive window credits the Wong-Wang competition's LOSING pool's early transient firing too (before
   mutual inhibition suppresses it), which measurably hurt selectivity in early calibration. A SHORT eligibility
   tau (20ms, vs the 1000ms default) + a reward pulse confined to the LATE (post-settle) tail of the drive
   window fixes this — crediting only the RESOLVED state of the competition, which also reads as the more
   biologically apt choice (outcome-locked dopamine, not reward smeared across an undecided trial).

## Two calibration misses this build made and fixed (kept in, not smoothed over)

<!--derived-->

- **A Python default-argument trap cost real debugging time.** `def train(self, n_episode_pairs=N_EPISODE_
  PAIRS)` binds the module constant at FUNCTION-DEFINITION time; `R.N_EPISODE_PAIRS = 100` monkey-patched during
  interactive calibration silently had NO EFFECT on `run_seed()`'s call to `.train()` with no explicit argument
  — a training run that looked identical in every printed constant was actually still running the OLD episode
  count. Fixed by resolving the default at CALL time (`n_episode_pairs=None`, resolved inside the method body).
- **The first shuffle-control schedule was too weak an adversary.** A 2:1 correct:distractor schedule (matching
  R1's raw episode-count shape) left the SHUFFLED-credit control's topology-tracking collapse marginal: on a
  3-seed check, shuffled/intact selectivity sat at 0.44–0.51, straddling the pre-registered <0.5 floor, and
  seed 44 MISSED it (0.507). The reason is structural, not noise: with 2/3 of positions True, a random
  permutation of the credit vector still assigns True to roughly 2/3 of the originally-True positions BY
  CHANCE, so "shuffling" barely decorrelates a majority class from itself. Re-balancing the schedule to 1:1
  (two distractor episodes per pair instead of one) gave the permutation genuine 50/50 decorrelating power —
  re-measured collapse to ~0.13–0.26x intact, cleanly under a 0.35 floor on all 6 seeds, and the topology-match
  rate under shuffling dropped from a marginal handful of seeds to a clean 0/6. This is recorded as the correct
  fix (a properly-adversarial control), not as a loosened floor chasing a specific seed.

## The measured pattern (adversarial read, not just the aggregate)

<!--derived-->

Per-seed intact final weights cluster tightly (correct pair ≈ 11.6–12.0, second-place "wrong-role" pair ≈
4.1–4.5, the never-credited control pair pinned at exactly 0.05) regardless of WHICH physical pool the random
assignment picked for that seed — consistent with a stable underlying dynamical system whose OUTCOME (which
wire wins) is fully determined by which pool the training schedule happened to load, not by pool-specific
structural variance. `frozen_weight_maxdrift` is exactly 0.0 on all 6 seeds (the whitelist held perfectly); the
migration invariant's `read_maxerr` (0.002–0.017) sits well under R1's own FP-layout floor and far below the
decision gap. F2's lesion collapses the vary-then-lesion shift to `attributable_to` ≈ 1.0 (0.98–1.0 across 6
seeds) — the shift is the cross-edges', not a confound. The shuffled-credit final weights are NOT degenerate or
collapsed-to-zero (e.g. seed 42: `w1->A=8.46, w1->P=8.95` — a REAL, substantial mapping formed, just the WRONG
one, tracking the scrambled credit rather than the true assignment) — this is the qualitatively correct shape
for "forms wrong," not an instrument artifact reading as a false negative.

## Honest residual ledger (what R2 closed vs what is still scaffold)

<!--derived-->

**Closed by R2:**
- Two-factor Hebbian → three-factor (credit-gated) reward-DEFERRED STDP as the plasticity RULE (R2-a).
- A host-hardcoded 4-edge candidate set → an unbiased 6-edge candidate set whose winning WIRE self-selects and
  tracks a per-seed-random ground truth invisible to the wiring/verification code (R2-b, the "target selection"
  criterion of `docs/TERMS.md`'s `self-organized` entry).

**Still scaffold (declared, not hidden):**
- The credit signal's VALUE (`current_reward_signal`) is a host-delivered scalar reflecting the runner's own
  ground-truth bookkeeping, not a spiking dopamine/value population computed by the brain's own error/success
  detection — the design's own next rung, gated on the neuromodulator-subsystem pool seam.
- The ALLOCATION of which physical WM pool holds a given referent on a given trial remains host-directed
  (`LOAD_PA` targeting) — `docs/TERMS.md`'s "allocation of slot/unit" criterion, unclosed.
- The candidate REGION PAIR (d6 slot pools → comprehension sel pools) is still host-chosen; R2-b never tests
  whether THIS pair is the one that should be tried.
- The experience STREAM is still runner-scheduled (correct-vs-distractor episode structure, timing, and which
  trials are "successful"), carried from R1.
- The ambiguous item remains a balanced-cue competition (a substrate stand-in for full pronoun-resolution
  discourse), carried from R1.
- Not a production flip: this is an organ-level GO on the merge pool, per the design's roadmap (§4).

## Files

- `research/runners/_onebrain_integration_r2_threefactor_selforganized.py` — the R2 runner (F1–F4 gate + R2-a/
  R2-b emergence controls + lesion-recovers-migration; 6-seed; numpy CPU; NO `sim/` edit).
- `research/findings/raw/_onebrain_integration_r2_threefactor_selforganized_6seed.json` — the 6/6 GO artifact +
  preconditions + per-seed role assignments + full weight trajectories for all three credit regimes.
