# Cascade North-bias / N-S axis lock — deep-research scoping (2026-06-20)

**Type:** READ-ONLY deep-research scoping (no code edits, no GPU). Per the project's standing
deep-research-first gate, this scopes the structural North-bias in the BG action-selection cascade — the
wall that the faithful grid-32 shortcut-#6 verdict (`2026-06-20-shortcut6-nav-orienting-CLOSE.md`) isolated
as the residual behind the neural-orienting closure.

**The owner's framing (load-bearing):** this is NOT a deferral. The #6 verdict pursued the orienting
*read-out* across seven mechanism variants (pop-vector geometry + divnorm + cortex-WTA at three strengths +
combo) and each was an honest negative *for re-orient* — but every variant pinned to the same residual: the
agent's **structural North-bias**. The verdict explicitly states (`…CLOSE.md:292-302`) that the next
direction is "correcting the cascade N-bias itself," scoped as a separate issue from the read-out geometry.
That is what this doc scopes — it is the next mechanism to pursue.

**Scope note (brain-based standard):** the North-bias is a SPIKING-substrate structural bias (it lives in the
neural cascade + its host read-out tie-break), not a host cognitive shortcut. But it BLOCKS the neural
orienting from steering the body on the N-S axis, so it is in scope. The no-confab moat is array-disjoint from
all of this (the nav cascade is `cp_*` nav state; the conversational composer's complex `cp_rf_w_*` synapses
are untouched), and nothing here weakens it.

---

## 1. Diagnosis — the structural source of the North-bias (with file:line)

**Summary (2-3 sentences):** The four cardinal pools are built byte-identically and the per-neuron
heterogeneity that would distinguish them is OFF (`cfg.enable_parameter_heterogeneity = False`,
`g11_bg_runner.py:4271`; the four `cortex_X` regions are constructed in a loop with identical `n_neurons`,
weights, and `weight_jitter=0` at `:924-943`). The dominant live North-bias is therefore a **degenerate
tie-break**: the host reads the spiking decision with `action_idx = max(range(N_ACTIONS), key=lambda i:
counts[ACTION_NAMES[i]])` (`:7073`, mirrored at the static probe `:2999`), and because Python's `max` returns
the FIRST argument on ties and `ACTION_NAMES = ["N","E","S","W"]` (`:79`), **every 4-way tie deterministically
resolves to N (index 0).** The #6 verdict's own trace confirms this is the operative regime: during re-orient
the four `sel_X`/`commit_X` accumulators "saturate together at the `n_commit_per_action=40` ceiling nearly
every step (`[40,40,40,40]` is the dominant pattern)" (`…CLOSE.md:97-99`) — i.e. the read-out is feeding the
argmax a tie, and the tie always picks N.

### 1.1 The four cardinal pools are symmetric by construction (rules out random-init asymmetry)

- `cortex_{N,E,S,W}` are built in one loop, `n_neurons=n_cortex_per_action`, `exc_weight_mean=0.0`,
  `weight_jitter=0.0`, `plastic_internal=False`, identical IZH preset (`:924-943`). The only per-action
  difference is `action_index` (a DA-targeting tag) and, under `enable_cluster_e_topography`, a
  `coordinate_center` corner (`:907-912, 926-927`) — neither changes the pool's intrinsic excitability.
- `cfg.enable_parameter_heterogeneity = False` (`:4271`) on the faithful nav path, and OU noise is OFF for the
  nav episode (`cfg.enable_ou_process = bool(build_with_ou)`, default False, `:4268`; the post-init hook turns
  it off). So there is no per-neuron jitter and no continuous stochastic drive to break symmetry between the
  four pools.
- **Implication:** the legacy comment "cortex_N fires 2x more at init" (`:2680`) and "cortex_N dominates from
  cluster A/E feedback" (`:2728`) describe an EARLIER (heterogeneity-ON / pre-merge) regime. Under the current
  het-OFF faithful path the pools are symmetric, so the structural asymmetry that the agent locks onto is NOT
  a random-init or biophysical asymmetry — it is the **deterministic tie-break degeneracy** plus whatever
  vanishingly-small numerical drift the symmetric dynamics produce, which the tie-break then amplifies to a
  hard N every step.

### 1.2 The tie-break is the amplifier — `max()` is N-first, and ties are the operative regime

- Decision read-out (`:7060-7080`): `_primary = commit_counts` (spiking_wta + commit). If `max(_primary) > 0`,
  `action_idx = max(range(N_ACTIONS), key=lambda i: _primary[ACTION_NAMES[i]])` (`:7073`). The deterministic
  RNG `action_rng` (`:2943`) is used ONLY in the `else` branch where ALL pools are silent (`:7079`) — it is
  **never** consulted for a tie among non-zero-but-equal counts.
- The same N-first argmax is used by the static probe path (`:2999-3000`) and by the legacy motor read-out.
- The #6 verdict's `decision_path_counts = {primary: 1800, fallback: 0}` with `[40,40,40,40]` commit
  saturation (`…CLOSE.md:96-99`) means: the commit layer fires for ALL FOUR actions every step (a tie), the
  argmax "primary" path always triggers, and the tie resolves to N. This is a textbook degenerate-WTA failure:
  the selection layer is not discriminating, and the read-out's deterministic ordering becomes the de-facto
  policy → **N every step regardless of the goal.**

### 1.3 Why only the N-S axis visibly locks (the verdict's sharper read)

The verdict's combo trace (`…CLOSE.md:270-275`) found x DOES drift west under the SC signal (combo phase-2
reached x=11) but y stays glued at row ~31 — a North-South axis lock, not a uniform freeze. Mechanistically
this is consistent with the tie-break diagnosis plus a small asymmetry in how much SC drive each cardinal
receives: the SC pop-vector margin is "position-correct but TINY" (`…CLOSE.md:225-228`), large enough to
occasionally tip the E↔W comparison (so x drifts) but never large enough to overcome the N-first tie ordering
on the N↔S comparison (N precedes S in `ACTION_NAMES`, so any N/S near-tie → N). In other words the
ordering-induced bias is sharpest on the very axis (N vs S) where the two competitors are adjacent in the
`max()` scan order. The HOST clears the bias on BOTH axes because its orienting reaches the actor at full
strength and produces a decisive (non-tie) margin (`…CLOSE.md:84, 293-294`).

### 1.4 What is NOT the cause (ruled out by the #6 sweep, recorded so it isn't re-litigated)

- **Not a divnorm calibration / SC-drive attenuation problem.** Both the `σ=5,gain=0.02` operating point AND
  the `gain=0` pure-cosine limit are stuck-N (`…CLOSE.md:107-128`); lowering gain improved static ACQUIRE but
  never RE-ORIENT. "More SC drive" is the wrong lever — the residual is competition/selection, not
  attenuation.
- **Not bump-attractor hysteresis.** The SC bump re-renders fresh every step, so Option-E (goal-change
  inhibition-of-return / fixation reset) is NOT indicated (`…CLOSE.md:104-105, 229-230`).
- **Not (only) too-weak cortex-WTA.** Cortex lateral inhibition at FS-weight 8 BROKE the phase-0 N-pinning
  (dom flipped to E, `…CLOSE.md:261`) — proving inter-cardinal competition is the RIGHT direction — but
  FS=16/40 over-quenched (all pools suppressed → N re-dominates via residual noise, the Rutishauser
  α-stability failure, `…CLOSE.md:243-247, 262-263`). So competition is necessary but the current
  cortex-WTA placement/strength alone did not close it — pointing at a combination of (a) removing the
  deterministic tie-break and (b) equalizing the pools' baseline so competition starts from parity.

---

## 2. Canonical biology — how a real brain avoids a fixed directional bias

Real action-selection circuits do not have a built-in "always pick the first option" degeneracy. Three
mechanisms keep the competition fair, all of which map onto our point-neuron cascade:

1. **Opponent / push-pull organization (the SC motor map).** During a direction decision, distinct populations
   in superior colliculus and frontal cortex encode OPPOSING movement directions and show push-pull dynamics —
   SC GABAergic neurons encode one side, glutamatergic neurons the other, and balanced E/I prevents a
   directional bias (the search synthesis below; Duan/Svoboda-lineage SC choice work, Nature Comms 2023;
   push-pull between colliculi, Comms Biol 2025). Catalog **H.25** (SC saccade topographic motor map; Kandel 6e
   Ch 35 p 875-882) and the opponent-channel entries (**E-cluster** ON/OFF + colour opponency, catalog
   :1408, :1468) are the project's reference for opponent coding. The four cardinals naturally form **two
   opponent axes** (N↔S, E↔W); organizing the competition as two balanced push-pull pairs (rather than four
   independent winner-take-all pools read by an ordered argmax) removes the privileged-first-option degeneracy.

2. **Divisive normalization equalizing the pool baselines (Carandini-Heeger).** A real cortical/collicular map
   normalizes each unit's drive by the summed activity of the local pool, so no pool sits at a structurally
   higher baseline. The project already has this primitive (`sim/bridge.py:6076-6080`,
   `input_divisive_norm` per-region, GUARDED). The #6 work used it on the SC→cortex drive but the residual was
   downstream (the tie at `sel_X`); applying the SAME normalization at the SELECTION stage (the `sel_X`
   accumulators), so all four start from an equal divisively-normalized baseline, is the canonical fix for a
   pool that wins by baseline rather than evidence.

3. **Homeostatic firing-rate balancing.** Real neurons regulate intrinsic excitability to a target rate
   (homeostatic plasticity of the AIS, catalog I.01:3237; Turrigiano scaling). The project already exposes
   **per-region homeostasis** (`BrainRegion.enable_homeostasis`, the per-region threshold-adapt mask built at
   `sim/bridge.py:1254-1259`, distinct from the global `cfg.enable_homeostasis` which is held OFF for the
   deterministic regime). Enabling per-region homeostasis on the four `sel_X` (or `cortex_X`) pools would
   drive their baseline firing toward a common target, equalizing the structural N-advantage at its source.

4. **Stochastic symmetry-breaking on genuine ties (decision noise).** Crucially, when evidence is genuinely
   equal, biological selection does not deterministically pick a fixed option — finite-size/Poisson noise
   breaks the tie randomly (Wang-2002 attractor decisions are noise-driven near threshold; the project's own
   `2026-06-19-spiking-decision-default-on-GO.md` documents finite-size-noise N-scaling as a lever). Our
   read-out does the OPPOSITE: it deterministically resolves ties to N. Replacing the N-first `max()` with a
   tie-aware random argmax (reuse the existing `action_rng`, `:2943`) restores the biological behavior on ties.

**Web-search synthesis (SC opponent/push-pull action selection):** "During decision-making, distinct
neuronal populations in both frontal cortex and superior colliculus encode opposing lick directions and
exhibit push-pull dynamics, with SC GABAergic neurons encoding ipsilateral choice and glutamatergic neurons
encoding contralateral choice." "In balanced activity changes, excitation and inhibition cancel out, which
helps prevent directional bias." "Appropriate actions are selected through competition between pools of
neurons representing competing choice options… a neural correlate of choice competition has been observed in
the superior colliculus where distinct populations encode potential choice options and exhibit push-pull
dynamics."

Sources:
- [Superior colliculus bidirectionally modulates choice activity in frontal cortex (Nature Communications, 2023)](https://www.nature.com/articles/s41467-023-43252-9)
- [Evidence for a push-pull interaction between superior colliculi in monocular dynamic vision mode (Communications Biology, 2025)](https://www.nature.com/articles/s42003-025-08081-0)
- [Spatial Representations in the Superior Colliculus Are Modulated by Competition among Targets (ResearchGate)](https://www.researchgate.net/publication/332366249_Spatial_Representations_in_the_Superior_Colliculus_Are_Modulated_by_Competition_among_Targets)

---

## 3. Ranked biologically-grounded fixes

Ordered cheapest-/highest-probability-first. All are point-neuron-achievable (this is a
balancing/normalization/symmetry-breaking problem, NOT a graded-read-out or dendritic one).

### FIX 1 (TOP) — Tie-aware stochastic argmax: break ties with noise, not with the N-first ordering
- **Mechanism:** when the top-K pools are within an epsilon count of the leader, resolve the choice by drawing
  from the tied set with the existing `action_rng` (`:2943`) instead of returning index 0. Biology: decision
  noise breaks genuine ties (Wang-2002; the project's own finite-size-noise lever,
  `2026-06-19-spiking-decision-default-on-GO.md`).
- **Reusable machinery:** `action_rng` already exists and is already used for the all-silent branch (`:3002`,
  `:7079`); the change is a few lines in the read-out tally (`:7060-7080`) and the static probe (`:2999`). NO
  `sim/` edit (runner read-out only).
- **Cost:** trivial (one runner function). **Point-neuron:** N/A (it is the host READ of the spiking decision;
  the spiking decision is genuinely a tie, so a fair read is the correct, non-cheating thing to do).
- **Caveat:** this alone makes the agent UNBIASED but does NOT make it SELECTIVE — under a true `[40,40,40,40]`
  tie it would random-walk. It is the necessary first step (removes the spurious N lock) and the discriminator
  for whether the SC margin is real: with a fair tie-break, if the SC signal carries ANY margin the dom
  cardinal should track the goal above chance. Combine with FIX 2/3 to convert "unbiased" into "selective."

### FIX 2 — Per-pool baseline equalization (divisive normalization OR per-region homeostasis at the selection stage)
- **Mechanism:** make the four `sel_X` (selection accumulators) — or `cortex_X` — start from a common baseline
  so the winner is decided by SC evidence, not by a structural advantage. Two interchangeable substrate
  primitives: (a) `input_divisive_norm=True` on the four selection pools (Carandini-Heeger,
  `sim/bridge.py:6076-6080`), normalizing each pool's drive by the four-pool mean; (b)
  `BrainRegion.enable_homeostasis=True` on the four pools (per-region threshold-adapt,
  `sim/bridge.py:1254-1259`), driving baseline rates to a common target.
- **Reusable machinery:** both primitives already exist and are GUARDED-off by default; flipping per-region
  flags is a builder change, NO `sim/` edit. The #6 work already proved the divnorm primitive composes on the
  SC→cortex drive; this applies it one stage downstream (at the competition).
- **Cost:** low (builder flags + a short calibration of the divnorm sigma/gain or homeostasis target at the
  selection stage). **Point-neuron:** yes (both are existing point-neuron primitives).

### FIX 3 — Opponent-axis selection: organize the four cardinals as two balanced push-pull pairs
- **Mechanism:** replace the four-independent-pool + ordered-argmax read with two opponent axes (N↔S, E↔W),
  each a balanced push-pull competition (the winner of each axis competes, then the two axis-winners compete).
  Biology: the SC/frontal opponent push-pull organization (search synthesis; catalog H.25). Removes the
  "first option wins ties" degeneracy structurally — there is no global ordered scan, only two symmetric
  pairwise comparisons.
- **Reusable machinery:** the existing per-action `cortex_FS_X` cortex-WTA microcircuit (`:945-961`) and the
  motor-FS lateral inhibition (`:2028-`) are the building blocks; wire the FS inhibition as opponent pairs
  rather than all-to-all. The `sel_X`/`commit_X` accumulator layer (`:2131-2224`) stays.
- **Cost:** medium (a wiring redesign of the cortex/sel FS topology + calibration). **Point-neuron:** yes
  (lateral-inhibition WTA is already on-substrate). Higher dev cost than FIX 1/2 but the most biologically
  faithful and the most robust against the over-quench failure the #6 cortex-WTA sweep hit (opponent pairs
  inhibit only their antagonist, avoiding the symmetric-over-inhibition α-instability).

### FIX 4 (reserve) — Stronger/recalibrated inter-cardinal competition at the cortex stage
- **Mechanism:** the #6 sweep showed cortex-WTA FS=8 broke phase-0 N-pinning but FS=16/40 over-quenched. With
  FIX 2 equalizing the baseline FIRST (so competition starts from parity), an intermediate cortex-WTA strength
  may now both break the bias AND remain stable. This is the #6 R1 remedy, retried on top of a debiased
  baseline rather than alone.
- **Reusable machinery:** `enable_cortex_lateral_inhibition` + the FS-weight knobs already exposed
  (`:3250-3255`). NO `sim/` edit.
- **Cost:** low (a strength sweep), but LISTED RESERVE because the #6 sweep already characterized it as
  insufficient ALONE — its value is conditional on FIX 1+2 first.

---

## 4. Recommended cheap-first de-risk

**The smallest test that the cascade can move SOUTH and re-orient on all 4 cardinals once the bias is
corrected, with the neural orienting signal driving it.**

**Arm to run first:** FIX 1 + FIX 2 stacked (tie-aware random argmax + per-pool baseline equalization on the
four `sel_X` via `input_divisive_norm` OR per-region homeostasis), on the EXACT faithful #6 NEURAL config
(`_nav_sc_popvector_readout_derisk.py`, grid-32, 1800, warmup-600, the merged-het-off SC op-point,
`enable_spiking_sc`, pop-vector read-out, the #4 WTA ring). One arm per invocation, commit each JSON as it
lands (anti-rest). This isolates whether removing the N-first degeneracy + equalizing baselines lets the
(already position-correct) SC margin steer the body — directly answering the #6 residual.

**Why this order:** FIX 1 is near-free and removes the deterministic N lock (the symptom); FIX 2 is low-cost
and reuses proven primitives to convert "unbiased random-walk" into "selective." If FIX 1+2 re-orients, #6
closes. If FIX 1+2 is unbiased-but-not-selective (random-walk, dom ≈ chance per phase), escalate to FIX 3
(opponent-axis) — the more faithful and robust structural fix. The #6 cortex-WTA FS=8 phase-0 break is the
positive prior that the competition mechanism direction is right.

**Calibration screen (grid-8, CALIBRATION ONLY, never a verdict):** if FIX 2 needs a divnorm sigma/gain or
homeostasis target at the selection stage, sweep it at grid-8 the way the #6 work calibrated the SC divnorm —
but the VERDICT is grid-32 (grid-8 is the documented false-GO scale: `…CLOSE.md:67, 92-94`).

---

## 5. Anti-cheats (the discriminators)

| anti-cheat | requirement | why it catches the cheat |
|---|---|---|
| **Per-phase per-cardinal action distribution** (THE discriminator) | the dominant cardinal must SHIFT across phases to track the moving goal — W-heavy in far-W phase, S-heavy in SW phase, etc. A fixed dom (N every phase) = the bias is NOT fixed. | the #6 verdict's NEGATIVE was exactly "N ~0.49-0.50 every phase, goal-invariant" (`…CLOSE.md:148`). A real fix moves this. |
| **4-cardinal symmetry check** | over a balanced goal schedule the four cardinals' total selection counts should be comparable (no cardinal structurally dominant); especially S must be reachable. | directly tests that the N-first degeneracy is gone, not just masked. |
| **Host ceiling** | the HOST orienting scaffold (centroid+argmax position decode) re-orients (Σ post-change ~1.74, dom tracks every phase, `…CLOSE.md:84`) — the fix's gap to host is the score. | anchors "how much of the orienting the spiking organ recovers." |
| **Regime fidelity = grid-32 (NOT grid-8)** | the verdict is grid-32/1800/warmup-600; grid-8 is the documented false-GO scale. | grid-8 hid the #6 negative (apparent tracking that didn't survive grid-32). |
| **Scramble / lesion control** | scramble the SC→cortex retinotopy (`SC_SCRAMBLE=1`, `install_spiking_sc_wiring(scramble=True)`, `:248-249`) → the re-orient MUST collapse. If debiased-NEURAL ≈ SCRAM, the SC decode is still not load-bearing (the bias was the only thing being read). | the #6 clincher: SCRAM ≈ NEURAL proved the decode wasn't load-bearing under the bias (`…CLOSE.md:130-138`). After debiasing, SCRAM must now clearly collapse for a real GO. |
| **Tie-break is not a covert random-walk win** | report the FRACTION of steps decided by the tie-break random draw. A GO needs the decision driven by the SC MARGIN (few ties), not by lucky random draws on `[40,40,40,40]` ties. | catches FIX 1 "passing" by random-walking onto the goal rather than steering. |
| **Multi-seed (only on a GO)** | a GO at seed 42 triggers the standing 6-seed confirmation (42/43/44/100/101/102). A robust NEGATIVE across mechanism variants does not (the #6 precedent). | the project's 6-seed rule for variable effects. |
| **No-confab moat untouched** | the nav cascade is `cp_*` nav state, array-disjoint from the composer's complex `cp_rf_w_*` synapses; no conversational regions in these runs. | moat unaffected by construction. |

---

## 6. Point-neuron-achievable? + Does it close #6? — verdicts

**Point-neuron-achievable: YES (high confidence).** Every ranked fix is a balancing / normalization /
symmetry-breaking operation realized with primitives the project ALREADY has on the point-neuron substrate:
the tie-break is a host READ of a genuine spiking tie (FIX 1, no substrate change); divisive normalization
(`sim/bridge.py:6076`) and per-region homeostasis (`sim/bridge.py:1254`) are existing point-neuron primitives
(FIX 2); lateral-inhibition opponent WTA is already on-substrate (FIX 3/4). This is explicitly NOT the
graded-read-out / divisive-normalization / point-neuron-limit family that the project's prior whitening /
opponency walls belonged to — it is a selection-fairness problem, which point neurons handle. No `sim/` edit
is required for FIX 1/2/4 (runner read-out + builder flags); FIX 3 is a runner-side wiring redesign.

**Does fixing the North-bias close #6? — LIKELY YES, with one honest caveat.** The #6 verdict established that
the SC pop-vector decode produces a position-CORRECT margin (the geometry is right) that is merely SWAMPED by
the N-bias before the WTA can amplify it (`…CLOSE.md:101-103, 225-228, 293-294`), and that the HOST — whose
orienting reaches the actor at full strength — clears this SAME bias on both axes and tracks every goal
(`…CLOSE.md:293-294`). So the orienting SIGNAL is present and correct; the only thing standing between it and
the body is the cascade's selection degeneracy. Removing that degeneracy (FIX 1) and equalizing the pools so
the small-but-real SC margin is decisive (FIX 2/3) is precisely the missing piece. **The honest caveat (the
deeper-coupling risk):** the SC margin at grid-32 is genuinely TINY (a far goal-blob is dim/small in the
16×16 `sc_map`, `…CLOSE.md:101`), so debiasing might yield UNBIASED-BUT-NOT-SELECTIVE behavior (a random-walk)
if the margin is below the selection noise floor even after equalization. In that case the residual is a
margin-magnitude (SNR) problem on top of the selection-fairness problem, and the next lever is amplifying the
SC margin (a larger/scaled `sc_map` so the far blob is brighter, OR FIX 3's opponent push-pull which extracts
a cleaner 1-D margin per axis). The per-phase-distribution + tie-fraction + scramble anti-cheats above are
exactly designed to tell "selective re-orient" (GO, close #6) from "unbiased random-walk" (margin-SNR
residual remains). Either way the bias correction is the correct, in-scope next mechanism — it is a
prerequisite for #6's closure whether or not it is sufficient alone.

---

## 7. Reusable project machinery (consolidated)

- **`action_rng`** deterministic RNG (`g11_bg_runner.py:2943`) — already used for the all-silent branch; reuse
  for tie-aware argmax (FIX 1).
- **`input_divisive_norm`** per-region Carandini-Heeger primitive (`sim/bridge.py:6076-6080`; flag set in the
  builder at `g11_bg_runner.py:4111-4112`) — reuse at the selection stage (FIX 2a).
- **`BrainRegion.enable_homeostasis`** per-region threshold-adapt (`sim/bridge.py:1254-1259`) — reuse to
  equalize `sel_X`/`cortex_X` baselines (FIX 2b). Distinct from the global `cfg.enable_homeostasis` held OFF
  for the deterministic regime.
- **`cortex_FS_X` cortex-WTA microcircuit** + the FS-weight knobs (`g11_bg_runner.py:945-961, 3250-3255`,
  `enable_cortex_lateral_inhibition`) — building blocks for opponent-axis WTA (FIX 3) and the reserve strength
  retry (FIX 4).
- **The `sel_X` / `commit_X` accumulate-then-commit selection layer** (`g11_bg_runner.py:2131-2224`,
  Wang-2002 + Lo-Wang) — stays; FIX 2/3 act ON it.
- **The #6 faithful harness** `_nav_sc_popvector_readout_derisk.py` (imports `run_moving_goal_episode`, sets
  the merged-het-off SC op-point) + the `SC_SCRAMBLE` lesion + the grid-32 goal schedule — the exact de-risk
  rig to reuse for the FIX 1+2 arm.

---

## Verdict

The cascade North-bias is a **degenerate-tie-break** structural bias: the four cardinal pools are symmetric by
construction (heterogeneity OFF, `:4271`), but the host reads the spiking decision with an N-first
`max()`-argmax (`:7073`) that deterministically resolves the `[40,40,40,40]` ties (the operative re-orient
regime, `…CLOSE.md:97-99`) to N (index 0, `:79`) every step. The fix is biologically standard and
point-neuron-achievable with existing primitives: (1) break ties with the existing decision RNG, (2) equalize
the four pools' baseline with the existing divisive-normalization or per-region homeostasis primitives, (3) —
if needed — organize the cardinals as two balanced opponent push-pull axes (SC motor-map biology, catalog
H.25). The cheap-first de-risk is FIX 1+2 on the faithful #6 NEURAL rig, discriminated by the per-phase
per-cardinal action distribution + the tie-fraction + the scramble lesion. This LIKELY closes #6 (the
orienting signal is present and position-correct; the bias is the only thing blocking it), with the honest
caveat that the SC margin's small magnitude at grid-32 may surface a residual SNR problem that the anti-cheats
are built to detect. **This is the next mechanism to pursue, not a deferral.**

_Read-only scoping. No code edited, no GPU run. The no-confab moat is untouched (the nav cascade is
array-disjoint from the composer's complex synapses)._
