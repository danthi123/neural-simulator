# Spiking action-decision cost-reduction PLAN — driving the fully-spiking nav read-out as close to the host argmax as we reasonably can, then deploying it default-on

**Date:** 2026-06-19
**Type:** DEEP-RESEARCH + RANKED, TESTABLE COST-REDUCTION PLAN (read-only — no build this phase; the controller drives the testing rounds from this).
**Owner priority:** "fully-spiking single brain as DEFAULT." The fully-spiking action-decision WORKS (genuine, moat-safe, host argmax retired) but navigates worse than the oracle argmax. GOAL: minimize the residual cost as far as is reasonable, then flip it on by default and eat whatever remains.

**Prior arc read in full:** `2026-06-18-merged-spiking-readout-SCOPE-GO.md` (§3-§5, the merged few-seed comparison),
`2026-06-06-action-selection-readout-deep-research.md` (the Wang/Lo-Wang/Stine/Rutishauser mechanism menu),
`2026-06-06-N6-accumulator-commit-readout-BOUNDARY.md` (the standalone boundary + the two named costs).

---

## 0. TL;DR for the controller

- **The gap is MOSTLY CLOSABLE, not fundamental.** The fully-spiking decision reaches the host/oracle floor in
  *stable* goal phases (per-phase final-quarter distance ≈ 0.50–0.60, == the host); the **entire** residual cost
  is concentrated in the **post-goal-change phases** and is two characterized, *mechanistically-fixable* spiking
  artifacts: **(1) cross-trial NMDA hysteresis** (the near-perfect integrator lingers on the previous winner
  when the goal switches) and **(2) silent-commit→fallback/random on weak-drive trials** (worst on seed 44). A
  small **irreducible** finite-size-noise floor (~1/√N) remains under it, but it is far below the current ~1.7×.
- **TOP 3 LEVERS (ranked, all runner-only, NO `sim/` edit, cheapest-first):**
  1. **A leak / forgetting on the `sel_X` accumulator (Usher-McClelland leak; urgency-gating low-pass).** Directly
     attacks the dominant cost (hysteresis). Cheapest test = lower the NMDA recurrence so the integrator forgets
     between trials (`--sel-recurrent-weight` sweep, already a flag), then per-trial loser-reset variants.
  2. **Urgency-shape tuning (peak + onset).** Already the biggest single win (~6×→~1.7×). One more cheap sweep of
     `--urgency-max-pa` (and an earlier ramp onset) to crush the residual silent-commit on weak-drive trials.
  3. **More neurons per pool (`--n-sel-per-action` / `--n-commit-per-action`).** Finite-size noise is 1/√N, so
     the deepest-seed catastrophe (seed 44) should shrink. Diminishing returns — a cheap 2-point N curve tells us
     whether it is worth the compute.
- **STOP/DEPLOY criterion:** stop tuning when **two consecutive levers each yield <0.15 absolute SUM improvement
  (≈ <5% of the host 2.0 floor) at 3 seeds**, OR the **6-seed mean spiking SUM is within 25% of the host floor
  (≤ ~2.5 on the gate-2a grid-32 / ≤ a documented small cost)**. Then flip default-on per §6.
- **DEPLOY step:** `MergedNavConvAgent(...)` build with `enable_spiking_wta_readout=True`; nav episode with
  `readout_source="spiking_wta"`, `enable_commit_burst=True`, `urgency_max_pA=<tuned>` + the tuned leak/N. Gate =
  the existing **nav-not-regressed** (`nav_gate2a_aggregate.py`) + **conversational answer-identity / no-confab**
  (`test_nav_conv_merged_agent.py`, `test_nav_conv_step2b_coresident.py`) — both must stay GREEN.

---

## 1. DIAGNOSIS — why a spiking decision race underperforms a perfect argmax (closable vs fundamental)

### 1.1 What the host argmax is, and why it is an *oracle*

The host argmax (`g11_bg_runner.py:6901`) reads the released-thalamus / motor counts at the end of the readout
window and returns `argmax`. It is **instantaneous, threshold-free, and zero-noise**: it compresses the brain's
two-stage decision (accumulate → commit) into one deterministic operation with no integration latency, no bound
to miss, and no trial-history. The deep-research finding (2026-06-06) already named this: *"the argmax compresses
both stages into one zero-latency operation"* — so it never pays the price a real spiking race pays.

### 1.2 What the spiking decision is (verified in code)

`readout_source="spiking_wta"` builds the **Wang-2002 accumulator → Lo-Wang/SC commit** layer
(`g11_bg_runner.py:2094-2203`):
- `sel_X` (20 neurons/action) with **NMDA-slow recurrent self-excitation** (`internal_density=0.5`,
  `exc_weight_mean=1.0`, `enable_nmda=True`, τ_decay≈100 ms) = the accumulator (Wang 2002; network
  τ = τ_syn/|1−w_rec|).
- `sel_FS_X` structured cross-inhibition (Rutishauser selective inhibition; gentle `sel_fs_to_sel_weight=5`).
- `commit_X` (20/action) burst pool driven by `sel_X` (`sel_to_commit_weight=22`) = the all-or-none termination
  (Lo-Wang 2006 SC; Stine-Shadlen 2023 LIP-accumulate/SC-commit).
- **Cisek urgency** (`g11_bg_runner.py:6808-6818`): a ramping action-INDEPENDENT current
  (`0 → urgency_max_pA` over `readout_start=30 → readout_end=100` substeps) injected into ALL `sel_X` so the
  effective bound collapses with time → a weak late winner still bursts in-window.
- **Decision = which `commit_X` bursts** (the threshold crossing). The host `max(...)` over `commit_counts`
  merely OBSERVES which pool fired (losers ≈ 0); fallback chain is commit-burst → `sel`-lean argmax → random.

### 1.3 The FAILING-DECISION PROFILE (derived from the raw 1800-step multi-goal runs)

This is the load-bearing diagnosis. From `research/findings/raw/_n6_refine_smoke8_urgency180_seed42.json`,
`_n6_val_urgency180_g8_seed{43,44}.json`, `_n6_val_urgency180_g32_seed42.json` (the cheat-5 SUM = sum over 4
phases of final-quarter mean distance, LOWER better; host/thal floor ≈ 2.0–2.34):

| run (urgency 180) | grid | SUM | per-phase [P0 P1 P2 P3] | decision-path {primary / fallback / random} |
|---|---|---|---|---|
| seed 42 | 8 | **4.08** | [0.60, **0.50**, 1.43, 1.55] | 1366 / 409 / 25  (76% / 23% / 1.4%) |
| seed 43 | 8 | **3.96** | [0.83, 0.62, 1.29, 1.21] | 1328 / 469 / 3 |
| seed 44 | 8 | **6.10** | [0.97, 1.89, 1.90, 1.34] | **680 / 805 / 315**  (38% / 45% / 17%) |
| seed 42 | 32 | **4.59** | [0.56, 0.60, 1.88, 1.55] | 1151 / 594 / 55 |

Two unambiguous signatures:

1. **The cost is the GOAL-CHANGE PHASES.** P0–P1 (stable goal) land at **0.50–0.62** = the host floor; P2–P3
   (after a goal switch) blow out to **1.3–1.9**. The `_n6_refine_analyze.py` "phases 0-1 vs 2-3" split confirms:
   stable ≈ host, the entire SUM excess is post-switch. **This is hysteresis** — the NMDA-slow accumulator
   (τ≈100 ms, ≈ one inter-trial) persists the previous winner across the goal change (named explicitly in the
   BOUNDARY finding). It is a property of a *near-perfect integrator*, and it is **closable** (add a leak).

2. **The catastrophe is WEAK-DRIVE SILENT-COMMIT.** Seed 44 is the warning: when the late-phase / off-policy
   thalamic drive is weak, the commit bound is not crossed → 45% fallback + 17% random. The fallback (sel-lean
   argmax) is ~80% correct but not the decisive burst; random is a coin-flip. Urgency 180 already cut random
   from 25%→1.4% on the easy seeds, but seed 44 shows the weak-drive tail is **not yet** closed. **Closable**
   (stronger/earlier urgency; more neurons to lift the sub-threshold winner above the finite-size floor).

### 1.4 The closable-vs-fundamental split (the honest answer)

- **CLOSABLE (the bulk, ~1.5 of the ~1.7×):** hysteresis (P2–P3 blow-out) and weak-drive silent-commit. Both are
  *mechanism-tuning* problems with named biological fixes (leak / urgency-gating / N-scaling), and the stable
  phases already PROVE the spiking decision can hit the host floor when the drive is clean and stationary.
- **FUNDAMENTAL (a small irreducible floor):** a real spiking race has **finite-size noise of order 1/√N** (the
  decision attractor's first-passage time fluctuates trial-to-trial; Deco-Rolls; Roxin-Ledberg). An oracle
  argmax has zero such noise. So even a perfectly-tuned spiking decision will sit *slightly* above the host
  floor — and that residual is exactly the "biology-faithful price for being a genuine accumulate-then-commit
  circuit" the project's BRAIN-BASED-ONLY standard ACCEPTS as the scientific deliverable. The plan's job is to
  get down to that floor, not below it. Estimated irreducible residual at N=20–40/pool: small (a few % of the
  2.0 SUM), well under the current 4.0–4.6.

**Verdict: the ~1.7× is ~85% closable, ~15% fundamental.** Tune the hysteresis + weak-drive levers to reach the
fundamental floor, document the small irreducible residual, deploy default-on.

---

## 2. RANKED LEVERS (each: mechanism · biology/catalog · expected nav-score effect · cheap-first test · risk)

All levers are RUNNER-ONLY (the entire layer is built additively in `build_bg_brain_regions` /
`run_moving_goal_episode`; **NO `sim/` edit**). Ordered by (expected effect on the *dominant* cost) × (cheapness).

### ★ LEVER 1 (RECOMMENDED FIRST) — Add a LEAK / forgetting to the `sel_X` accumulator (fix the hysteresis)

- **Mechanism.** The current accumulator is a *near-perfect integrator* (strong NMDA recurrence,
  `sel_recurrent_weight=1.0` at `density=0.5` → soft-WTA gain α near 1, τ≈100 ms). A perfect integrator carries
  stale evidence across the goal switch. A **leaky** accumulator (Usher-McClelland 2001 "leakage = the
  forgetting factor"; a low-pass filter that "rises when the stimulus is presented but falls back when it is
  removed") *discards* the previous winner so the new goal's drive wins promptly. The urgency-gating literature
  makes the same point: the policy that maximizes reward rate in **changing conditions** "emphasizes novel
  information" via a low-pass filter + time-dependent gain (Cisek-Thura urgency-gating; Carland-Thura-Cisek
  2019). Our task IS a changing-conditions task (multi-goal, 3 switches) — so a leak is the *normatively correct*
  knob, not a hack.
- **Biology / catalog.** Usher-McClelland 2001 LCA (leakage + lateral inhibition, the n-choice DDM with neural
  decay); Cisek-Thura urgency-gating (low-pass + collapsing bound); catalog **G.16** (drift-diffusion bound),
  **G.17** (LIP accumulator). Note the LCA explicitly motivates leak from *"neural excitatory input decays within
  5–10 ms"* — a perfect integrator is the biologically *less* faithful choice anyway.
- **Expected nav-score effect.** Targets the **P2–P3 blow-out directly** (the ~1.3–1.9 → toward 0.6). Largest
  expected single win, because P2–P3 IS essentially the whole gap. Risk that too much leak hurts the weak-drive
  ramp (the integrator's amplification helps the burst fire) — so sweep, don't max.
- **Cheap-first test (NO new code — knob already exists).**
  1. **Recurrence-as-leak sweep:** `run_moving_goal_episode(..., sel_recurrent_weight ∈ {1.0, 0.7, 0.5, 0.3})`
     (lower recurrence = shorter memory = more leak), grid-8 multi-goal, seeds 42/43/44. Report SUM + the
     phases-0-1-vs-2-3 split + decision-path counts via `_n6_refine_analyze.py`. WIN = P2-P3 drops with P0-P1 ≈
     unchanged. (The prior smoke #3 showed *stronger* recurrence WORSENS the full run via hysteresis — this is
     the same axis run the *helpful* direction, and is the single highest-value experiment.)
  2. **Loser-only NMDA reset at trial start** (`--reset-losers-only`, already a flag): clears the *losing* pools'
     carried drive without zeroing the eventual winner's lean — a targeted hysteresis clear. (Prior note: alone
     it added "no net lift combined with urgency"; re-test it *combined with the LEVER-1 lower-recurrence* — the
     two compose differently than reset+urgency did.)
- **Risk.** LOW. Both knobs exist; default-off / default-value preserves the current behavior exactly. The only
  failure mode is over-leak (weak-drive trials lose the ramp) — bounded by the sweep, and LEVER 2 (urgency)
  compensates the ramp.

### ★ LEVER 2 — Urgency SHAPE tuning (peak + earlier onset) — finish the weak-drive silent-commit

- **Mechanism.** Urgency is a **gain modulation** of the decision circuit (the title result of Niyogi-Wong-Wang
  "Gain Modulation by an Urgency Signal Controls the Speed-Accuracy Trade-Off in a Network Model of a Cortical
  Decision Circuit", PMC3042674): a growing action-independent drive raises every pool toward the bound so the
  *time-to-cross* shrinks. The project already injects this (`urgency_max_pA`, linear `30→100` substep ramp) and
  it was the biggest single win (standalone ~6× → ~1.7×; random fallback 25%→1.4% on easy seeds). The residual
  weak-drive silent-commit (seed 44: 45% fallback, 17% random) says the ramp is **not yet aggressive enough on
  the hard trials**.
- **Biology / catalog.** Cisek 2009; Thura-Cisek 2014/2016 (premotor/M1 urgency single-cells); Niyogi-Wong-Wang
  2011 (urgency = gain). Lo-Wang 2006 (the bound IS adjustable — the DA-modulated cortico-striatal weight). The
  urgency-gating literature's headline: urgency-gating *"yields a higher reward rate than any constant criterion
  model"* and "responds more quickly to changes in the environment" — exactly our metric.
- **Expected nav-score effect.** Targets the **random/fallback tail** (the seed-44 catastrophe + the residual on
  42/43). Moderate-to-large on the *worst* seed (which dominates a 6-seed mean), smaller on the easy seeds (near
  saturated already). Trades accuracy for speed — push only until the random-fallback is gone, not past it (too
  much urgency commits before the accumulator separates → wrong-commits).
- **Cheap-first test.**
  1. **Peak sweep:** `urgency_max_pA ∈ {180, 240, 300}`, seeds 42/43/44 grid-8. WIN = seed-44
     random+fallback collapses toward the easy-seed levels with SUM not regressing on 42/43 (no rise in
     wrong-commits — check thal-winner alignment stays ≥ ~90%).
  2. **Earlier onset (if peak alone saturates):** the ramp currently starts at `readout_start=30` substeps.
     Test an earlier urgency onset (parameterize the ramp start, a 2-line additive runner change) so the bound
     collapses sooner on weak trials. (Lowest priority sub-test; only if (1) plateaus.)
- **Risk.** LOW-MED. Over-urgency = premature/wrong commits (a speed-accuracy *over*-shoot). Guard: the
  thal-winner-alignment % in `_n6_refine_analyze.py` must not fall as urgency rises.

### ★ LEVER 3 — More neurons per pool (finite-size noise 1/√N) — shrink the seed-variance floor

- **Mechanism.** The decision attractor's trial-to-trial fluctuation (and thus wrong-/late-commit rate on
  marginal trials) is **finite-size noise of order 1/√N** ("as the number of neurons N increases, fluctuations
  of population activity decrease"; Deco-Rolls 2006; Roxin-Ledberg). The current pools are tiny (`n_sel=20`,
  `n_commit=20`). Bigger pools average the noise → the marginal/weak-drive trials (seed-44 tail) become more
  reliable, pulling the 6-seed *variance* (and worst-seed) down toward the easy-seed level.
- **Biology / catalog.** Wang 2002 spiking attractor (cortical decision pools are 100s–1000s of neurons, not 20);
  Deco-Rolls "Noise in Attractor Networks… Graded Firing Rate" (the 1/√N finite-size law). Catalog **G.16/G.17**
  (the accumulator is a population code).
- **Expected nav-score effect.** Improves the **worst seed / variance** more than the mean (the easy seeds are
  near the floor already). Because finite-size noise is 1/√N, the return **diminishes** — N=20→80 is a 2× noise
  cut, N=80→320 only another 2× for 4× the compute. So: a cheap 2-point curve to see the slope, then stop when
  it flattens.
- **Cheap-first test.** `n_sel_per_action ∈ {20, 40, 80}` and `n_commit_per_action` matched, seeds 42/43/**44**
  (44 is the discriminating seed), grid-8. WIN = seed-44 SUM + random-fallback drop materially at 40, with 80
  showing diminishing returns. Compute cost is the only downside (bigger bridge); grid-8 keeps it cheap.
- **Risk.** LOW (purely additive neurons; no instability — more neurons makes the soft-WTA *more* stable, not
  less). Only cost = GPU memory/time; bounded by the small N values and grid-8.

### LEVER 4 — Recurrence/inhibition attractor RE-TUNE (separation vs hysteresis trade) — secondary

- **Mechanism.** The soft-WTA gain (α via `sel_recurrent_weight × density`) and the structured inhibition
  (`sel_fs_to_sel_weight=5`) jointly set how *decisively* and *quickly* the winner separates (Rutishauser-
  Douglas-Slotine: α<1 stable soft-WTA; too symmetric/strong inhibition → the synchronized-oscillation
  instability the BOUNDARY finding already hit and fixed). LEVER 1 moves α *down* for leak; this lever co-tunes
  inhibition so separation stays crisp at the lower α (more inhibition can restore winner-separation lost to the
  leak).
- **Biology / catalog.** Rutishauser-Douglas-Slotine 2011 (contraction-stability, α<1, structured inhibition);
  catalog **B.04** (striatal WTA is feedforward-FSI, symmetric mutual inhibition is a weak selector — the
  empirical twin of the RDS result).
- **Expected effect.** Second-order; recovers winner-separation that LEVER 1's leak may erode. Only run if
  LEVER 1's lower-recurrence sweep shows separation degrading (winner/runner-up `sel` ratio dropping in the
  guard). Couples to LEVER 1 — do not sweep independently first.
- **Cheap-first test.** Only after LEVER 1 picks a recurrence: `sel_fs_to_sel_weight ∈ {5, 8, 12}` at the chosen
  recurrence, seed 42/44 grid-8. Watch the guard's winner/runner-up separation + SUM. **CAUTION:** the BOUNDARY
  finding documents symmetric over-inhibition → synchronized rebound bursting; stay gentle, watch for the
  all-pools-fire pathology.
- **Risk.** MED (the documented instability axis). Bounded by small steps + the guard.

### LEVER 5 — Read the cleanly-selective `thal` into the accumulator more strongly (feedforward evidence gain) — diagnostic / fallback

- **Mechanism.** The released `thal_X` is the *cleanest* signal (winner fires, losers exactly 0). The current
  `thal_to_sel_weight=30` is deliberately "modest evidence, not saturating." On weak-drive trials a higher
  feedforward gain delivers a stronger clean drive to the accumulator so the burst fires in-window — a cheap
  diagnostic for "is the weak-drive silent-commit just feedforward gain?". The deep-research finding ranked this
  #4 (the *engineering* version of the biological recurrent amplifier; amplifies noise too) — so it is a
  **fallback/diagnostic**, not a primary mechanism.
- **Biology / catalog.** Douglas-Martin canonical microcircuit (thalamic input amplified — but via *recurrence*,
  not a bigger feedforward weight, which is why this ranks below LEVER 1). Catalog A.04/A.05.
- **Expected effect.** Small; may help weak-drive trials but amplifies loser noise (the clean thal helps here
  since losers are 0, but plasticity-leaked late-phase thal is messier). Run as a one-shot diagnostic.
- **Cheap-first test.** `thal_to_sel_weight ∈ {30, 45, 60}`, seed 44 grid-8 (the weak-drive seed). WIN = seed-44
  silent-commit drops. If it does and LEVERS 1–3 didn't fully close it, fold in a modest bump.
- **Risk.** LOW-MED (saturation → the synchronized-oscillation pathology at very high feedforward + strong
  inhibition; the BOUNDARY's exact failure at `thal_to_sel=60 + sel_fs=28`). Keep inhibition gentle if raising
  this.

### NOT-RECOMMENDED (already tried / dominated)
- **Per-trial FULL accumulator reset** (`--reset-accumulator`): documented NEGATIVE — zeroes the winner's carried
  drive → MORE silent-commit (smoke #4: 55% silent, SUM 6.93). The *loser-only* reset (LEVER 1 sub-test) is the
  salvageable version.
- **`commit_OPN` tonic omnipause ON**: documented rate-coded synchronized-rebound instability (all-or-none across
  ALL pools); kept structurally faithful but OFF. Do not enable.
- **Stronger commit recurrence** (smoke #3): WORSENS the full run via commit hysteresis. Leave at 0.6.

---

## 3. RECOMMENDED TEST SEQUENCE (cheapest / highest-leverage first)

All on **grid-8 multi-goal, 1800 steps, seeds 42 / 43 / 44** (44 is the discriminating weak-drive seed) — the
cheap probe grid the prior arc used. Analyze every run with `research/findings/raw/_n6_refine_analyze.py`
(reports SUM, P0-1-vs-P2-3 split, and the primary/fallback/random decision-path counts). Baseline to beat:
**urgency-180 SUM ≈ 4.0 (easy seeds) / 6.1 (seed 44); host/thal floor ≈ 2.0.**

1. **ROUND 1 — LEVER 1 (leak):** `sel_recurrent_weight ∈ {1.0(base), 0.7, 0.5, 0.3}` at `urgency_max_pA=180`.
   Pick the recurrence that minimizes P2-P3 without hurting P0-P1. *(Highest expected win; the dominant cost.)*
2. **ROUND 2 — LEVER 1 combine:** best recurrence **+ `--reset-losers-only`**. Confirm the targeted hysteresis
   clear composes with the leak (it didn't with urgency alone; test the new combination).
3. **ROUND 3 — LEVER 2 (urgency peak):** at the ROUND-2 best, sweep `urgency_max_pA ∈ {180, 240, 300}`. Pick the
   peak that kills the seed-44 random+fallback tail WITHOUT dropping thal-winner alignment below ~90% on 42/43.
4. **ROUND 4 — LEVER 3 (N):** at the ROUND-3 best, `n_sel_per_action = n_commit_per_action ∈ {20, 40, 80}`,
   seeds 42/43/44. Stop at the N where the seed-44 improvement flattens (1/√N diminishing returns).
5. **ROUND 5 (only if needed) — LEVER 4 / 5:** if winner-separation degraded under the chosen leak (LEVER 4
   re-tune inhibition) OR seed-44 silent-commit persists (LEVER 5 feedforward-gain diagnostic). Gentle steps;
   watch the instability guard.
6. **VALIDATION — 6-seed + grid-32:** take the best config; run the **6-seed** `gate6_{standalone,merged}`
   campaign at grid-32 (the gate-2a scale) via `nav_gate2a_aggregate.py`. Report the merged spiking SUM vs the
   host floor as the deliverable.

Apply the STOP rule (§4) between every round — do not run a round whose predecessor already met "done."

---

## 4. THE "DONE ALL WE REASONABLY CAN" CRITERION (when to stop tuning + deploy)

Stop tuning and proceed to deploy when **either** holds (whichever first):

- **(A) Diminishing-returns stop:** **two consecutive rounds** each improve the 3-seed mean SUM by **< 0.15
  absolute** (≈ < 5% of the host 2.0 floor). The levers have flattened → we are at the practical floor.
- **(B) Good-enough stop:** the **6-seed mean spiking SUM is within 25% of the host floor** — i.e. **≤ ~2.5 on
  grid-32 gate-2a** (host/thal ≈ 2.0), with **decision-path primary ≥ 90%** (the decision is reliably the
  commit burst, not the argmax fallback) and **no seed catastrophically worse** (worst-seed SUM ≤ ~1.5× the
  mean). At that point the residual is dominated by the irreducible 1/√N finite-size floor and is an acceptable,
  documented biological cost.

If neither is reached after the §3 sequence (LEVERS 1–5 exhausted), **STOP at the best config and deploy it
anyway with the honest residual documented** — per the owner directive (eat whatever residual remains) and the
BRAIN-BASED-ONLY standard (a clean spiking decision with a characterized cost IS the deliverable). Do **not**
fabricate a pass, and do **not** keep grinding past diminishing returns.

**Anti-cheat throughout:** every round must keep **decision-path `primary` dominant** (the win must come from the
commit burst firing more reliably, NOT from the `sel`-lean argmax fallback quietly carrying more decisions — that
would re-introduce the host-argmax shortcut). `_n6_refine_analyze.py` already reports this; a config that lowers
SUM by raising fallback% is REJECTED.

---

## 5. WHAT IS *NOT* WORTH DOING (scope discipline)

- **Chasing 2.0 exactly.** The finite-size 1/√N floor means a spiking race cannot equal a zero-noise oracle
  argmax; aiming for parity is aiming below the biological floor. Target the floor + document the residual.
- **A `sim/` edit for a dedicated leak conductance.** Tempting (a true leak term), but the existing
  `sel_recurrent_weight` (lower recurrence = leak) + `--reset-losers-only` already realize the forgetting
  behavior runner-side. Only consider a `sim/` leak if LEVER 1's knobs prove too coarse (unlikely) — and then
  per the owner's "sim/ edits fine if justified + byte-diff review" standard, scoped as its own cycle.
- **Re-deriving the mechanism.** The accumulate-then-commit circuit is correct and validated; this is *tuning*,
  not a redesign. Do not rebuild the layer.

---

## 6. THE DEFAULT-ON DEPLOYMENT STEP (exact flags + the gate)

Once §4 is met, flip the fully-spiking read-out on by default for the merged "one brain":

### 6.1 The flags to flip (all already plumbed end-to-end — verified)

- **BUILD** (`MergedNavConvAgent.__init__` / `build_merged_nav_conv_bridge`): pass
  **`enable_spiking_wta_readout=True`** (already an additive pass-through kwarg →
  `build_bg_brain_regions`, `nav_conv_merged_bridge.py:450/581/603`).
- **EPISODE** (`run_moving_goal_episode`, the nav gate via `_nav_gate_merged_run.py`):
  **`readout_source="spiking_wta"`**, **`enable_commit_burst=True`** (default), **`urgency_max_pA=<ROUND-3>`**,
  plus the tuned **`sel_recurrent_weight=<ROUND-1>`**, **`n_sel_per_action=n_commit_per_action=<ROUND-4>`**, and
  (if used) **`reset_losers_only=True`**. All accepted today (`g11_bg_runner.py:3690/3713/3716/3752/3769`); the
  CLI mirrors them (`--readout-source spiking_wta --urgency-max-pa … --sel-recurrent-weight … --n-sel-per-action
  … --n-commit-per-action … --reset-losers-only`, `g11_bg_runner.py:7892-7982`).
- **DEFAULT-FLIP mechanics:** change the *defaults* of these kwargs on the merge path
  (`MergedNavConvAgent` / the merged nav-gate runner) to the tuned values, so the merged brain is fully-spiking
  out of the box. Keep `readout_source="motor"` (host argmax) available as an **opt-in oracle baseline**
  (`--readout-source motor`) for regression A/Bs — do NOT delete it; it is the test reference.

### 6.2 The deployment GATE (both must stay GREEN — these are the answer-identity / nav-not-regressed checks)

1. **Conversational answer-identity + no-confab moat UNCHANGED.** The spiking read-out is a NAV-side substrate
   (`sel_X`/`commit_X`, array-disjoint from the parser + RF composer). Run **`test_nav_conv_merged_agent.py`
   (8/8, incl. the three `is None` no-confab asserts)** and **`test_nav_conv_step2b_coresident.py` (7/7, incl.
   the co-residence anti-cheat)** — both must pass VERBATIM. (The SCOPE-GO finding already proved the moat holds
   with the WTA layer present; the gate re-asserts it post-tuning.)
2. **Nav-not-regressed-by-the-flip.** Run `nav_gate2a_aggregate.py` (the merged-vs-standalone scorer) so the
   merged spiking SUM == the standalone spiking SUM (the co-residence is not corrupting the decision), and report
   the spiking SUM vs the host floor as the documented cost. The pre-existing **byte-identity** check (the
   conversational slice stays frozen under the live nav reward-STDP + dopamine stressor) must still hold — the
   read-out change touches only `sel_X`/`commit_X`, which are array-disjoint from the conv slice.
3. **Decision-path `primary` ≥ 90%** on the deployed config (the decision IS the commit burst — the brain-based
   target — not the argmax fallback). Reported in the result dict (`decision_path_counts`).

If all three are GREEN, the merged "one brain" navigates via a genuine, fully-spiking action-decision by default,
with the host argmax retired to an opt-in test oracle — and the documented residual cost is the honest,
biology-faithful price (the BRAIN-BASED-ONLY deliverable).

---

## 7. KEY CITATIONS

**Project (read in full):** `2026-06-18-merged-spiking-readout-SCOPE-GO.md`,
`2026-06-06-action-selection-readout-deep-research.md`, `2026-06-06-N6-accumulator-commit-readout-BOUNDARY.md`.
Raw decision-profile data: `research/findings/raw/_n6_refine_smoke8_urgency180_seed42.json`,
`_n6_val_urgency180_g8_seed{43,44}.json`, `_n6_val_urgency180_g32_seed42.json` (+ `_n6_refine_analyze.py`).

**Catalog (`sim-catalog/references/feature-catalog.md`):** G.16 (drift-diffusion bound, speed-accuracy,
Kandel 6e Ch 56 pp 1399–1404), G.17 (LIP accumulator ramp-to-threshold, pp 1402–1404), A.04 (BG WTA at GPi/SNr,
selection emergent from the reentrant network), A.05 (reentrant loops latch), H.24/H.25 (SC saccade burst
generator / OPN), B.04 (striatal WTA is feedforward-FSI; symmetric mutual inhibition is a weak selector).

**Primary literature (levers):**
- Wang, X-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. *Neuron*
  36:955–968. — the NMDA-recurrent accumulator (the `sel_X` layer).
- Wong, K-F. & Wang, X-J. (2006). A recurrent network mechanism of time integration in perceptual decisions.
  *J. Neurosci.* 26(4):1314–1328. — reduced model; NMDA-slow recurrence sets the long integration τ; strong
  recurrence → winner persistence (the hysteresis LEVER 1 attacks).
- Usher, M. & McClelland, J.L. (2001). The time course of perceptual choice: the leaky, competing accumulator
  model. *Psychol. Rev.* 108:550–592. — **leakage = the forgetting factor; low-pass falls back when evidence is
  removed** (LEVER 1's biological charter).
- Cisek, P., Puskas, G.A., El-Murr, S. (2009). Decisions in changing conditions: the urgency-gating model.
  *J. Neurosci.* 29:11560–11571. + Carland, Thura, Cisek (2019), *Neuroscientist* — urgency-gating = low-pass +
  collapsing bound; maximizes reward rate under changing conditions; emphasizes novel information (LEVER 1 + 2).
- Niyogi, R.K. & Wong-Wang style "Gain Modulation by an Urgency Signal Controls the Speed-Accuracy Trade-Off in
  a Network Model of a Cortical Decision Circuit" (PMC3042674). — **urgency = gain modulation** of the decision
  circuit (LEVER 2's mechanism).
- Lo, C-C. & Wang, X-J. (2006). Cortico-basal ganglia circuit mechanism for a decision threshold in RT tasks.
  *Nat. Neurosci.* 9:956–963. — SC all-or-none commit burst; the bound is the (DA-modulated) cortico-striatal
  weight (the `commit_X` layer + the speed-accuracy knob).
- Stine, Trautmann, Jeurissen, Shadlen (2023). A neural mechanism for terminating decisions. *Neuron*
  (PMC10565788). — LIP accumulates, SC commits (rate + derivative threshold); SC inactivation removes commitment
  not accumulation (the two-stage division of labor the layer implements).
- Rutishauser, U., Douglas, R.J., Slotine, J-J. (2011). Collective stability of networks of winner-take-all
  circuits. *Neural Comp.* 23:735–773. — stable soft-WTA needs self-excitation α<1 + structured inhibition;
  symmetric naive WTA is unstable (LEVER 4's guardrail).
- Deco, G. & Rolls, E.T. (2006). Decision-making and Weber's law: a neurophysiological model. *Eur. J. Neurosci.*
  + Rolls-Deco *"Noise in Attractor Networks… Graded Firing Rate"* (PLoS ONE). — **finite-size noise ∝ 1/√N**;
  bigger pools average it (LEVER 3 + the irreducible floor).
- Roxin, A. & Ledberg, A. (2008). Accuracy and RT distributions for decision-making: linear perfect integrators
  vs nonlinear attractor circuits (PMC3825033). — the attractor decision's intrinsic stochasticity vs a linear
  integrator (the closable-vs-fundamental framing).

**Source URLs:**
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3042674/ (urgency = gain modulation)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6601812/ (urgency in the speed-accuracy tradeoff)
- https://www.jneurosci.org/content/34/49/16442 (context-dependent urgency)
- https://journals.sagepub.com/doi/10.1177/1073858419841553 (Carland-Thura-Cisek 2019 urgency review)
- https://pubmed.ncbi.nlm.nih.gov/19759303/ (Cisek 2009 urgency-gating, changing conditions)
- https://stanford.edu/~jlmcc/papers/UsherMcC01.pdf (Usher-McClelland LCA leak)
- https://www.jneurosci.org/content/26/4/1314 (Wong-Wang 2006 reduced model)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3825033/ (Roxin-Ledberg integrator vs attractor)
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0023630 (Rolls-Deco finite-size 1/√N noise)
