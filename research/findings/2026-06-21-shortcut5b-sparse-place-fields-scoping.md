# Shortcut #5b genuine-close scoping — the self-org place-code value-δ (sparse-fields vs read-out vs validate-by-function)

**Read-only deep-research + catalog-review scoping** (the standing opening move at a characterized boundary;
owner rule: a boundary is a START of research, not an exit). Scope the genuine close for the **#5b** boundary —
the self-organized spiking place code's value-train δ (delta) is FLAT at nav scale, so the host-Gaussian
place-context scaffold stays as the residual shortcut.

**One-line conclusion up front (so the controller can triage fast):** the cheap-first close the prior NEGATIVE
named ("sparser self-org place fields → graded value → δ recovers") was **already run to ground 2026-06-19**
(`562c1055`) — sparsification FIXED value-learning (LEARNS-V 1.01→1.91×) but exposed a **deeper, distinct
read-out-regime wall**, so #5b is now a read-out + selectivity problem, not a field-density problem. **There is
one genuinely unexploited cheap close** (wire the already-`sim/`-shipped GRADED dendritic-plateau read-out onto
the sparse self-org afferent — the two have NEVER been combined), and **a stronger, owner-aligned reframe**: the
nav δ is the WRONG test (the #9 lesson — the nav value is not load-bearing on immediate-reward nav), and the
**just-validated #9 trace-conditioning harness IS the right test** — but the self-org place code is NOT what that
harness needs. The honest verdict is below in §5.

---

## Terms (defined once)

- **#5b** — the second half of TRUE-ONE-BRAIN roadmap item #5: replace the host-coded Gaussian place-context
  critic afferent (`vs_place_context`) with a self-ORGANIZED spiking `place` code, AND have it lift the value
  read-out's **δ (delta)** — the dopamine reward-prediction-error gap between a predicted (near-goal) and an
  unpredicted (far) location. Value `a` (the place code COMPOSES on the merged bridge) is **GO + committed**;
  value `b` (the δ-LIFT) is the boundary.
- **δ (delta)** — `r − V`: the SNc (substantia nigra dopamine) firing gap. The probe reads it as
  `pred(near) / unpred(far)`; a graded value gives δ ≈ 1.3, a flat value gives δ ≈ 1.0.
- **self-org place code** — a spiking `place` pool whose location-selective fields emerge from competitive
  threshold-WTA on the egocentric `place_sensors` render (Hartley-Burgess place-field formation), then frozen.
- **value-train (STEP-2)** — DA-gated STDP on `place → striosome_value` that potentiates the cells active at each
  goal, so V grades near≫far.
- **the coincidence-plateau read-out** — the `place → striosome_value` pathway is tagged `coincidence_detector`,
  so the FS-PING-synchronized spike volley fires an all-or-none dendritic plateau in the MSN-D1 critic (the
  sparse-async code can't fire it otherwise). This is the value READ-OUT.
- **FS-PING** — the `place_fs` fast-spiking interneuron pool that gamma-synchronizes the place volley
  (location-blind: it sets WHEN the active cells fire, not WHICH).
- **the host-Gaussian scaffold** — `vs_place_context`: a dense grid-32-tuned Gaussian place code injected each nav
  step, position-specific BY CONSTRUCTION. The residual shortcut #5b is trying to retire.

---

## 1. The residual + machinery map — and the precise place the residual NOW sits

### 1a. Where the dense/overlapping fields come from (the prior NEGATIVE's root cause)

The afferent `place_sensors → place` is built (`g11_bg_runner.py:1855`) with `place_sensors_to_place_weight=28`
× `density=0.5`. STEP-1 self-org (`_run_place_selforg`, `g11_bg_runner.py:5437`) drives the sensors at a sub-grid
of positions with the `landmark_to_place` plasticity gate OPEN and the `place_fs_gate` transmission gate held
CLOSED (clean threshold-WTA). The dense code is set by that afferent drive **overdriving ~46% of the place pool
past threshold** — two locations then share ~46% of the cells by density alone → near/far cosine 0.67. The pool
has **NO per-region homeostasis** by design (anti-cheat: it must fire from the learned current, not a threshold
collapse) and **NO feedforward-inhibition WTA** during self-org (the FS-PING is held closed). So the field
density is purely an afferent-drive-vs-threshold artifact — exactly the kind of thing a stronger inhibitory
WTA or a lower afferent weight sparsifies.

### 1b. The value-train + read-out pipeline

`place → striosome_value` is PLASTIC + DA-gated (`value_input` gate) + `coincidence_detector=True`
(`g11_bg_runner.py:1898`). At BUILD the plateau is the **strong all-or-none COUNT form**
(`coincidence_weighted_drive=False`, `g11_bg_runner.py:4494`) so it bootstraps the post-spike that drives
DA-gated LTP. The critic is the MSN-D1 `striosome_value`; its δ is delivered by a slow GABA_B/GIRK
`striosome_value → snc` subtraction. The SNc (`snc`) pool is **n_dopamine=10** (`g11_bg_runner.py:462`).

### 1c. THE RESIDUAL HAS MOVED — sparsification was already run to ground (2026-06-19, `562c1055`)

This is the single most important fact for the scoping. The prior NEGATIVE
(`2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md`) named the diagnosed close as "SPARSER,
less-overlapping self-org place fields (depth-tuning)." **That close was then built and exhaustively swept**
(`research/runners/_n5_place_sparsify_probe.py`; finding
`2026-06-19-place-code-sparsify-default-BOUNDARY.md`):

- **Sparsification WORKS and FIXES the named root cause.** Lowering `place_sensors_to_place_weight` 28→10
  sparsifies STEP-1 to **6.25% active** (cos 0.22), and the value-train LEARNS-V improves **1.01 → 1.91×**
  (w_near now ≫ w_far — the value gradient the NEGATIVE said was missing).
- **But the δ does NOT cross 1.3 — a deeper, distinct wall is exposed.** In the **FS-PING-OPEN operating regime
  the value-train and critic actually read in**, the surviving sparse cells are **NOT location-selective at nav
  scale** (a few dominant cells fire at MANY locations; the operative near/far cos = 0.42–0.78 regardless of the
  STEP-1 sparsity), and the **all-or-none coincidence-plateau read-out has only two reachable regimes**, neither
  graded:
  - **low weights** → critic in a physiological band (~5 Hz) but **cannot discriminate** (near ≈ far) → flat δ ≈ 1.04;
  - **high weights** → critic **over-fires** (98–238 Hz) → **over-clamps the SNc GABA_B to 0** → δ → 0.0.
- The single-goal clean-capability test is the decisive one: with only (6,6) trained, **w_far GREW ABOVE w_near
  (LEARNS-V 0.79, FALSE)** because the FS-open read fires the same dominant cells at near AND far, so the (6,6)
  pairing potentiates far's "exclusive" cells too — **the value-train cannot localize V to the trained location's
  cells at nav scale**.
- Exhaustively swept (all seed 42, GPU, deterministic self-org): afferent weight {8,9,10,11,12,18,28} ·
  `fs_to_place_weight` {8,16,20,24,40} · `N5_SPARSIFY_FS_DURING_SELFORG` {0,1} (FS-PING open during self-org =
  the canonical Hartley-Burgess mechanism — sparsity 0.06–0.10 but read cos still 0.74–0.78, "the FS-PING is a
  gamma synchronizer, NOT a WTA") · init V {0.2,0.3,0.5} · trials {40,60,80,150} · single-vs-multi-goal ·
  coincidence-k {4,6,8,12,15,20} · GIRK cap {0,10,12} · `critic_fs_weight` {16,40}. **None opened the narrow
  physiological+graded window.**

**⇒ The genuine residual is NO LONGER field density.** It is two stacked sub-residuals: **(R1) the read-out
regime is non-selective** (the FS-open competition leaves a few dominant cells firing everywhere — a
selectivity, not a sparsity, problem), and **(R2) the all-or-none plateau read-out is binary** (it either
under-discriminates or over-clamps; it has no graded middle). The 2026-06-19 finding itself names the two
specified next moves: (i) the dendritic substrate (per-cell nonlinear integration to carve selective fields), or
(ii) **a graded rate read-out that scales smoothly with V** (so a modest near>far weight → a modest near>far
critic rate → a graded δ, without the over-clamp).

---

## 2. Biology review (the standing opening move) — sparse, location-selective place fields + graded read-out

Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`; glossary:
`E:\Documents\Projects\sim\references\glossary.md`. Cluster D (hippocampus) + adjacent.

| ID | Mechanism | Relevance to #5b |
|---|---|---|
| **D.06** | Place cells (O'Keefe) — ~1–5% sparse, location-SELECTIVE allocentric fields | the target the self-org code falls short of (it is dense AND non-selective in the read regime) |
| **D.12** | Pattern separation (DG): EC inputs DIVERGE onto a larger pool + **strong feedforward inhibition** → ~2–5% sparse, orthogonalized | the canonical SPARSITY mechanism; **point-neuron-achievable** (FFI + WTA) |
| **D.13** | Pattern completion (CA3): recurrent autoassociator reconstructs from partial cue | not the bottleneck here (over-completion would WORSEN overlap) |
| **D.07** | Grid cells (Moser): periodic context-invariant code; grid→place competition selects single fields | a SELECTIVITY-from-structured-input route (the input, not the inhibition) |
| **B.06** | PV+ fast-spiking feedforward GABA inhibition (striatal template, already in project) | the reusable FFI/WTA motif for sparsifying the place layer |
| **I.01 / J.01** | AIS homeostasis / synaptic scaling — plastic spike threshold to a target rate | a point-neuron sparsity knob (raise threshold to hit ~5%) |
| **G.02** | Active dendrites: NMDA-spike PLATEAU potentials (Major-Larkum-Schiller), Poirazi-Mel two-layer dendritic computation, Larkum apical-basal coincidence gain | the GRADED read-out + the deep SELECTIVITY route; **the graded-plateau half is already shipped on-bridge** (see §3) |
| (CLAUDE.md) | Mikulasch-Priesemann point-neuron decorrelation/whitening limit | the deep wall: point neurons can't decorrelate correlated codes pre-spike; they spike on whatever structure the INPUT carries |

**The decisive biology insight (catalog review, confirmed by the 2026-06-19 finding's own data): SPARSITY ≠
SELECTIVITY.** Feedforward inhibition (D.12/B.06) handles sparsity cheaply on point neurons — and the sweep
confirms it (W=10 → 6% active). But **location-selectivity is upstream**: it requires location-STRUCTURED input.
The self-org place's input is the egocentric `place_sensors` bearing/distance render, which is **heavily
overlapping across nearby locations at nav scale** — so sparsifying the OUTPUT leaves a few dominant cells that
still fire at many locations (exactly R1). Getting genuine 1–5% AND selective fields from overlapping egocentric
sensors is the Mikulasch-Priesemann-flavoured limit: it plausibly needs either (a) a fundamentally
more-decorrelated INPUT (grid-cell-like or richer landmark sensors), or (b) per-cell dendritic nonlinear
integration to carve selective fields — the deep substrate route.

**Which mechanism most cheaply addresses the EXISTING residual?** Neither pure sparsity mechanism (FFI,
homeostasis) — those were swept and don't fix selectivity. The cheapest lever that touches the ACTUAL residual
(R2, the binary read-out) is the **graded dendritic plateau** (G.02) — and it is **already built and validated**
(§3). The deep selectivity residual (R1) is the genuine substrate-limit candidate.

---

## 3. Ranked cheap-first closes (each: machinery · de-risk · anti-cheat · sim/-edit-or-not)

### CLOSE A (RECOMMENDED, genuinely unexploited) — wire the already-shipped GRADED dendritic-plateau read-out onto the SPARSE self-org afferent

**The gap this fills.** The 2026-06-19 finding named "a graded rate read-out that scales smoothly with V" as a
specified next move and called it "out of this task's scope." **It is no longer out of scope: it was BUILT THE
NEXT DAY.** `enable_graded_dendritic_plateau` (the SMOOTH, non-saturating logistic sibling of the all-or-none
plateau) shipped 2026-06-20 as a guarded, default-OFF, byte-identical-when-off `sim/` edit (`d69cc0ab` +
`f941a39b`; finding `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`) and was **validated on-bridge to
produce a clean, monotone, location-selective graded V (3/3 seeds, V near 0.13 > mid 0.08 > far 0.014, ~9×
near/far)** — i.e. it solves R2 exactly. **But it was validated on the host-Gaussian `vs_place_context`
afferent, and the runner gates it OFF under the self-org path:** `g11_bg_runner.py:4496` reads
`if dendrite_critic and enable_neural_critic and **not _neural_place_selforg**`. **The graded read-out and the
self-org place code have NEVER been combined.** That combination — sparse self-org place (W=10) + the graded
plateau read-out (`coincidence_weighted_drive=True`, `enable_graded_dendritic_plateau=True`,
`coincidence_plateau_strength=0` so the all-or-none form is OFF) — is the genuinely unexploited cheap close.

- **Reusable machinery:** the shipped `enable_graded_dendritic_plateau` cfg block + `fused_graded_dendritic_plateau`
  kernel (reuses the EXISTING `coincidence_detector` routing mask — zero new wiring); the `--dendrite-critic`
  deploy block (`g11_bg_runner.py:4496-4521`) as the template; the W=10 sparse-afferent recipe + the
  `_n5_place_sparsify_probe.py` harness; the standalone-vs-merged array-disjoint argument (the controller-owned
  run pattern).
- **Cheap-first de-risk:** modify ONE runner-local gate so `--dendrite-critic` can co-exist with
  `--neural-place-selforg` (lift the `not _neural_place_selforg` exclusion, OR add a dedicated
  `--selforg-graded-readout` flag), then run `_n5_place_sparsify_probe.py --stage-b` at W=10 with the graded
  read-out ON. Read the STAGE-B VERDICT: does the critic now GRADE (near>far rate, physiological band, no
  over-clamp) → δ > 1.3? numpy-CPU smoke first (the harness runs deterministic on CPU), then GPU seed 42, then
  6-seed if seed 42 clears the 1.3 bar.
- **Anti-cheat:** (1) the field-sparsity + near/far-decorrelation metric (STEP-1 cos < 0.3 AND read cos
  reported) → the near/far VALUE separation (w_near/w_far ≥ 1.5×) → δ vs the host-Gaussian 1.3. (2) the
  **plateau-lesion** control (flag-off collapses δ → load-bearing, as in the dendrite Stage-1 finding). (3) a
  **no-sparsification** control (W=28 dense + graded read-out — should stay flat, isolating that the graded
  read-out needs the sparse-but-still-overlapping code). (4) the merged-build MOAT (`check_moat`: `place` +
  `striosome_value` present + array-disjoint from `parse_role`/`dlpfc_wm`; host `vs_place_context` absent) —
  preserved by construction (the graded plateau is additive on the same routed pathway).
- **sim/-edit-or-not:** **NO new `sim/` edit** — the graded-plateau `sim/` code already ships (byte-reviewed). The
  ONLY change is a runner-local gate relaxation in `g11_bg_runner.py` (default-preserving: keep the existing
  exclusion as the default, add an opt-in path). Reuse-by-import otherwise.
- **HONEST RISK (must be stated):** CLOSE A targets R2 (the binary read-out). It may NOT clear R1 (the
  read-regime non-selectivity): if the FS-open read still fires the same dominant cells at near AND far, even a
  perfectly-graded read-out grades the WRONG thing (it grades on an overlapping V). The dendrite Stage-1 finding
  got its clean 9× separation on `vs_place_context`, which is selective BY CONSTRUCTION; the self-org code is
  not. **So CLOSE A is the right FIRST experiment (cheap, high-value-if-it-works, and it isolates R1 from R2 —
  if it grades, #5b is closed; if it doesn't, the failure pinpoints R1 as the genuine residual), but its success
  is genuinely uncertain.** This is the honest "genuine-close-via-graded-read-out vs deeper-selectivity-limit"
  fork, and CLOSE A is the cheapest way to resolve which side it lands on.

### CLOSE B (point-neuron, addresses R1 directly but likely insufficient alone) — feedforward-inhibition WTA on the place layer + structured input

- **Reusable machinery:** the `N5_SPARSIFY_FS_DURING_SELFORG` lever (FS-PING open during self-org); the striatal
  PV-FSI template (B.06); per-region homeostasis (I.01); the topographic-prior / orthogonal-drive builders.
- **De-risk:** sweep a STRONGER, slower FS→place GABA_A WTA during BOTH self-org and read (not just gamma
  synchrony) + raise the place threshold via homeostasis to a 5% target; measure the READ-regime near/far cos.
- **Anti-cheat:** the read-cos metric (the operative cos, not just STEP-1) must drop below ~0.3; lesion the
  inhibitory pool → density collapses back to ~46%.
- **sim/-edit-or-not:** likely runner-local (reweighting existing pathways + flipping homeostasis), unless a
  dedicated inhibitory region is added (still framework-level, no `sim/` edit).
- **HONEST RISK:** the 2026-06-19 sweep ALREADY tried FS-during-self-org + stronger FS→place (up to weight 40)
  and the read cos stayed ≥ 0.42 ("the FS-PING is a gamma synchronizer, not a WTA"; weight 40 made it WORSE).
  **The catalog confirms why: sparsity ≠ selectivity** — FFI sparsifies the OUTPUT but selectivity is set by the
  INPUT (the overlapping egocentric sensors). So CLOSE B almost certainly needs the INPUT half (richer /
  decorrelated landmark sensors or a grid-cell-like front end) to matter — which is a larger build (a new
  sensory front end), edging toward the deep route. **Lower priority than CLOSE A.**

### CLOSE C (the deep substrate route — NOT cheap; the honest deferred fork) — dendritic per-cell nonlinear field carving

- The Major-Larkum-Schiller / Poirazi-Mel two-compartment substrate (G.02) to carve selective fields per cell
  from overlapping input (the NMDA-plateau "cluster-on-one-branch ≫ scattered" nonlinearity). This is the
  named months-scale deferred dendritic rewrite (the project's recurring Mikulasch-Priesemann wall). **Out of
  cheap scope; only justified if CLOSE A's failure proves R1 is the genuine irreducible residual AND the nav δ
  is confirmed to actually matter (see §4 — it does not).**

---

## 4 / §5. Validate-by-function — the #9 lesson, and the trace-harness connection (the decisive reframe)

This is the most important section, and it largely SUPERSEDES §1–3 for the controller's priority call.

**The nav δ is the WRONG test for the place-code value — by the project's own just-validated finding.** Shortcut
#9 (the dendrite-graded value) hit the IDENTICAL confound and resolved it TODAY
(`2026-06-21-shortcut9-trace-conditioning-value-derisk.md`, with scoping
`2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md`): **the moving-goal gridworld is
immediate-reward-solvable**, so the value's distinctive function (credit assignment over a temporal GAP) is
NEVER exercised — lesioning the value barely moves navigation (dendcritic 8.47 ≈ value-lesion 9.08, Δ7.2%), and
the whole gain over baseline came from the NMDA on the critic slice, not the value. **This is the same
`feedback_validate_signal_by_its_function` lesson as N5-reward.** A δ that's "flat on nav" may be flat because
the value is INERT on nav, not because the place code can't grade value. **So the #5b nav-δ probe is measuring a
quantity that doesn't matter on the task it's measured on.**

**The right test — already built and validated 6/6 — is trace conditioning.** #9's genuine close is a Pavlovian
trace-conditioning 2×2 factorial (TRACE vs DELAY × value-ON vs value-LESION) where the value IS provably
load-bearing: the **TRACE arm needs the value (lesion collapses the anticipatory CR, 6/6), the DELAY arm
(gap=0) does not (lesion survives, 6/6)**. The spiking lift runs on the **limbic-core topology**
(`cue → striosome_value → snc`, `_limbic_core_rpe_battery_derisk.build_limbic_core`), and the dendrite-graded
plateau's slow ~80 ms conductance is the gap-bridge.

**Does #5b's real close UNIFY with #9/B4 ("the sparser self-org place supports the trace-conditioning value")?
Honest verdict: NO — and stating it sharply is the deliverable.** The trace-conditioning harness uses an
**orthogonal SPARSE CUE** as the value's afferent (the B4 scoping is explicit: "Concept codes / cue drive are
the substrate's own (orthogonal sparse cue patterns)"). It deliberately does NOT require a place code, because
it deliberately **sidesteps spatial credit assignment** (the B4 scoping is explicit again: "The
trace-conditioning task DELIBERATELY SIDESTEPS this [the actor-critic SPATIAL credit-assignment wall, the 3×
NEGATIVE hidden-goal place→action arc]"). The value's function being tested there is "carry value across a
TEMPORAL gap from a clean cue," which the orthogonal cue provides BY CONSTRUCTION. **Feeding the self-org PLACE
code into the trace harness would re-import exactly the R1 problem (a non-selective, overlapping afferent) into
a task that was specifically designed to avoid afferent-quality confounds — it would CONTAMINATE the clean #9
close, not unify with it.** The two are the same broad family ("value over a gap") but distinct sub-problems
with distinct afferents: #9 = a clean cue + a temporal gap; #5b = a SPATIAL afferent (the place code) whose
quality is the bottleneck. The genuine spatial-credit test is the actor-critic hidden-goal arc — which is the
3×-NEGATIVE wall the dendrite is the named (deferred) unlocker for, NOT the trace task.

**⇒ The validate-by-function verdict for #5b has two honest layers:**
1. **The self-org place VALUE itself is already validated-by-function as a value** — via #9, the dendrite-graded
   value read-out (which the place code feeds) is load-bearing on trace conditioning. The place code's COMPOSE
   (#5 value `a`) is GO. So the "is this value ever useful?" question is answered YES, just not by the nav δ and
   not with the place code as the afferent.
2. **The specific #5b claim — "the self-org place code lifts the nav δ above the host Gaussian" — is testing a
   quantity that is INERT on the task** (the nav value is not load-bearing) AND is bottlenecked by an
   afferent-selectivity wall (R1) that the trace harness was built to avoid. **There is no clean
   validate-by-function task for "the place code grades SPATIAL value better than the host Gaussian" short of
   the actor-critic hidden-goal arc — which is the separate, deferred, 3×-NEGATIVE spatial-credit wall.**

---

## 6. Recommended cheap-first de-risk (the single next move)

**Run CLOSE A: combine the sparse self-org place afferent (W=10) with the already-shipped graded
dendritic-plateau read-out** — the one genuinely unexploited cheap experiment, NO new `sim/` edit (only a
runner-local gate relaxation in `g11_bg_runner.py:4496`). It is the cheapest way to resolve the
genuine-close-vs-deeper-limit fork:

- If it **GRADES** (critic near>far, physiological band, no over-clamp, δ>1.3, plateau-lesion collapses it,
  no-sparsification control stays flat, moat intact, 6/6): **#5b is genuinely closed** — the binary read-out was
  the residual, and the graded read-out (already validated on `vs_place_context`) transfers to the sparse
  self-org code.
- If it **does NOT grade** (the read still fires dominant cells at near AND far → the graded read-out grades an
  overlapping V): the failure **pinpoints R1 (afferent non-selectivity) as the genuine irreducible residual**,
  which — combined with §4's verdict that the nav δ is inert anyway — makes the host-Gaussian's retention the
  honest, characterized BRAIN-BASED-ONLY deliverable, with the deep dendritic/structured-input route (CLOSE
  B+C) as the named (deferred, owner-call) path and the actor-critic hidden-goal arc as the real spatial-credit
  test.

Recommended command shape (numpy CPU smoke → GPU seed 42 → 6-seed if it clears 1.3):
```bash
# after relaxing the `not _neural_place_selforg` gate (or adding --selforg-graded-readout):
SIM_BACKEND=cupy python -m research.runners._n5_place_sparsify_probe --seed 42 --stage-b \
  --overrides place_sensors_to_place_weight=10,coincidence_weighted_drive=1,\
enable_graded_dendritic_plateau=1,coincidence_plateau_strength=0
# read the STAGE-B VERDICT: LEARNS-V (expect ~1.91x, already fixed) + CRITIC FIRE+GRADE (the new question) + GABA_B gap (>1.3?) + LESION
```

---

## 7. Honest framing (genuine-close vs deeper-limit; the size; the #9 connection; downstream dependency)

- **Genuine-close-via-sparser-fields: ALREADY CLOSED OUT (negative).** The prior NEGATIVE's named close was run
  to ground — sparsification fixes value-LEARNING but not the value read-OUT. Field density is NOT the residual.
- **The genuine residual is the read-out (R2) + the afferent selectivity (R1).** R2 has a cheap, already-shipped
  candidate fix (CLOSE A, the graded plateau — never yet combined with the self-org code). R1 is the genuine
  Mikulasch-Priesemann-flavoured substrate-limit candidate (sparsity ≠ selectivity; the egocentric sensors are
  overlapping), with no cheap point-neuron fix (the sweep confirms FFI alone doesn't crack it).
- **The size of the genuinely-irreducible part is SMALL-to-MODERATE and unresolved until CLOSE A runs.** If
  CLOSE A grades, the irreducible part is zero (read-out was the whole thing). If it doesn't, the irreducible
  part is R1 — afferent selectivity from overlapping sensors — which is real but is ALSO (per §4) testing a
  quantity that's inert on nav.
- **The #9 trace-harness connection: a clean NO to unification, and that is itself the finding.** The trace
  harness validates the place code's downstream VALUE machinery (the dendrite-graded plateau) BY FUNCTION, but
  with an orthogonal cue — NOT the place code. Feeding the place code in would contaminate #9's clean close. The
  place code's spatial-value superiority over the host Gaussian has NO clean validate-by-function test short of
  the deferred actor-critic hidden-goal arc.
- **Downstream dependency:** #5b's δ-lift gates the **merged-default flip** (replacing the host-Gaussian
  `vs_place_context` with the self-org place code as the production merged critic afferent). The host Gaussian is
  retained as the better-δ scaffold ONLY because of this boundary. CLOSE A is the path to the flip; if it fails,
  the host Gaussian's retention is the honest BRAIN-BASED-ONLY deliverable (a neural-underperforms-host mapping),
  and #5 value `a` (the place code COMPOSES) remains the real, committed TRUE-ONE-BRAIN breadth win regardless.
- **Per the BRAIN-BASED-ONLY standard:** the place code is already a validated brain-based replacement (#5 value
  `a`, committed; and its value machinery is #9-validated). The δ-lift is a quality-superiority claim over a host
  scaffold; whether it closes (CLOSE A) or stays a characterized boundary, the outcome is a legitimate
  scientific deliverable.

---

## Files referenced (read-only)
- `research/runners/g11_bg_runner.py` — self-org build (1239–1295, 1847–1904), `_run_place_selforg` (5437–5491),
  the all-or-none read-out gate (4480–4495), the `--dendrite-critic` graded-read-out deploy gated OFF under
  self-org (4496–4521), `n_dopamine=10` (462).
- `research/runners/_n5_place_sparsify_probe.py` — the iteration harness (the CLOSE-A de-risk vehicle).
- `research/runners/nav_conv_merged_bridge.py` — the merged builder (`nav_critic_place_selforg`, 460/542–586/1334–1346).
- `research/runners/_limbic_core_rpe_battery_derisk.py` — the trace-harness topology (`cue→striosome_value→snc`).
- `sim/config.py` (180/197 coincidence + 197 `coincidence_weighted_drive`; the graded-plateau block), `sim/bridge.py`
  (6314–6378 the weighted/count read-out; the graded-plateau per-step block), `sim/kernels.py`
  (`fused_graded_dendritic_plateau`).
- Findings: `2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md` (the #5b boundary),
  `2026-06-19-place-code-sparsify-default-BOUNDARY.md` (sparsify run-to-ground),
  `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md` (the graded read-out, shipped + validated on `vs_place_context`),
  `2026-06-21-shortcut9-trace-conditioning-value-derisk.md` + `-B4-delayed-reward-value-task-scoping.md` (the validate-by-function close + the orthogonal-cue / spatial-credit-sidestep verdicts).
- Catalog `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (D.06/D.07/D.12/D.13/B.06/G.02/I.01/J.01);
  glossary `E:\Documents\Projects\sim\references\glossary.md`.
