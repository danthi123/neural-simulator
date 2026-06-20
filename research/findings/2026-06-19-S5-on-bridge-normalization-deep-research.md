# Deep research — can the S5 per-query NORMALIZATION run fully on-bridge on a point-neuron substrate, or is it the dendritic boundary? (2026-06-19)

**Standing practice** (deep research + reference-catalog review FIRST at a confirmed roadblock) run for the ONE
residual host operation in the integrated who/what conversational loop: the host read of the cleanup match score at
seam **S5** (`research/runners/rf_phasor_composer.py:270-272` — `re = to_host(...); peak = scores.max(); drive =
(scores/peak) * drive_pA`). READ-ONLY; no code, no experiments.

## TOP-LINE (the honest call the owner asked for)

**S5 is CLOSABLE on the point-neuron substrate. It is NOT the dendritic boundary.** The per-query normalization S5
needs is **divisive GAIN control on the DIAGONAL** (rescale a non-negative score vector so a placed threshold
separates winner from runner-up regardless of the query's absolute peak). That is a *canonical, point-neuron-feasible*
computation (Carandini-Heeger normalization circuit; catalog J.10 shunting "divides the EPSP by a factor"; E.05
center-surround "decorrelates output") — and **the project has ALREADY built it and validated it on-bridge at
production scale**: the NEF thresholded cleanup (`2026-06-05-composer-cleanup-NEF-GO.md`) performs its input
normalization with **a spiking inhibitory-trait FS pool that shunts the est input population so the matched-filter
drive is ~scale-invariant across seeds** — NO host peak-read — and reaches numpy parity 27/27 at D=2048. The residual
S5 host read in `rf_phasor_composer._spiking_cleanup` is a *cheaper host-peak shortcut* the FHRR composer kept; the
core-sim NEF cleanup already replaced it with an on-bridge circuit. **The fix for S5 is to route the FHRR composer's
S5 normalization through that same on-bridge NEF input-normalization FS pool** (the lever the S5 seam doc itself names,
`2026-06-19-phaseC-task1-S5-seam-derisk.md` lines 54-61).

The genuinely-dendritic Mikulasch-Priesemann boundary is a DIFFERENT operation — **OFF-diagonal decorrelation /
whitening** (mixing information across input dimensions to remove the *correlation structure*). S5 does not require
that. Do not conflate the two: the project's own arc has repeatedly separated them
(`2026-06-15-slow-perhub-mean-primitive-deep-research.md`: "Centering is the cheap, separable half of whitening; the
cross-neuron de-correlation is the expensive half").

**Caveat that sets the expectation honestly:** the project ALREADY ran a spiking-divisive-normalization cleanup to
ground (`2026-06-04-composer-cleanup-divisive-norm-NEGATIVE.md`) and it plateaued at worst-case **0.911** vs numpy
1.000 as a *deployable fixed-op 320-way argmax readout*. That NEGATIVE does NOT apply to S5, because (a) the NEF
follow-on subsequently CLEARED exactly that plateau to 0.978-1.000 with the input-norm FS pool already on-bridge, and
(b) S5 is a structurally easier ask (a placed-threshold pass/abstain that protects the moat, K small in the integrated
loop, the per-query DATA not 320-way numpy-exact parity). The de-risk below is scoped to falsify the S5-specific claim,
not to re-litigate the 320-way readout.

---

## 1. DIAGNOSIS

### The blocker, restated crisply

The integrated loop (`2026-06-19-phaseC-task2-wholeturn-loop.md`) runs the whole who/what turn — comprehend → store →
reconstruct/unbind/cleanup → match → answer/abstain — as ONE spiking loop, control flow on-substrate (the host `_scan`
orchestrator is GONE; the spiking BG WTA decides answer-vs-abstain). **One** host operation survives: S5 reads the
cleanup's graded match score to host to drive the sequencer's decoded word-line.

Task 1 (`2026-06-19-phaseC-task1-S5-seam-derisk.md`, commit `27c6422e`) proved a FIXED on-bridge synaptic projection
(option a) cannot carry it: the cleanup score is a graded matched-filter value `Re(c_k)` on `cp_membrane_potential_v`
(winner ≈ peak ~1.3e6, runner-up ≈ 0.4·peak), and the cleanup probe measured `both_suprathreshold_vs_izh_30mV: true`
— winner AND runner-up (AND every off-target with positive `Re`) are ALL far above the Izhikevich +30 mV firing
threshold. So a binary spike fires identically for winner and runner-up, the whole decoded row lights, the match
washes out, and the no-confab moat breaks. The peak-normalization that `scores_to_drive`/`_spiking_cleanup` perform on
host (the divide by `scores.max()`) is **per-query** (the peak varies query to query), and a *fixed* `cp_connections`
projection cannot express a per-query divisor.

### The two sub-problems — and WHICH is hard on point neurons

The S5 closure has two stages. They have **opposite** difficulty profiles; separating them is the whole analysis.

| sub-problem | what it does | on point neurons? | the relevant project finding |
|---|---|---|---|
| **(P1) per-query NORMALIZATION** | rescale the non-negative score vector by a per-query gain so its magnitude is query-invariant (≈ divide by peak / divide by `σ + Σ score`) | **FEASIBLE** (canonical divisive normalization; J.10 shunting + balanced E/I gain) | NEF-GO input-norm FS pool runs this on-bridge at D=2048 |
| **(P2) thresholded SELECTION** | a placed firing threshold so the off-target emits ZERO spikes and only the winner fires → argmax-by-firing | **FEASIBLE + already on-bridge** | NEF threshold placement; BG WTA (catalog A.04, `g11_bg_runner`) |

**The hard one on point neurons is NEITHER of these.** Both P1 and P2 are point-neuron-feasible and the project has a
working on-bridge realization of each. **The hard one is the *coupling* P1→P2 at a FIXED operating point** — i.e. the
placed threshold in P2 has to be robust to whatever scale P1 leaves, ACROSS queries with different raw peaks. That
coupling is exactly what the NEF input normalization buys ("here it makes the THRESHOLD seed-invariant — the
load-bearing role", `2026-06-05-composer-cleanup-NEF-GO.md` line 29). The whole question reduces to: **can a
point-neuron divisive pool make P1 good enough that the P2 threshold transfers across queries WITHOUT a host peak-read?**

### Why this is NOT the Mikulasch-Priesemann dendritic boundary

The dendritic boundary the project keeps hitting (whitening wall, rate-code wall, opponency wall) is about a
**different** operation: removing the COMMON MODE / correlation structure of a SIGNED, full-precision, balanced code —
the off-diagonal of the covariance, "a per-dimension, pre-spike, analog operation that mixes information *across*
dimensions (a matrix operation in dendrites)" (`2026-06-15-slow-perhub-mean-primitive-deep-research.md`). The
on-substrate face of it (`2026-06-06-graded-lgn-decorrelation-BOUNDARY.md`) is that a rectifying/saturating graded
read-out *degrades the gentle signed composing structure* (a known-100%-composing code drops to 72% just by passing
through spiking membranes).

S5 does NOT touch any of that. The S5 score is **already rectified and non-negative** (`scores = np.maximum(re, 0.0)`,
`rf_phasor_composer.py:271`) — off-target concepts are already at `Re~0`, the winner is strongly positive. There is no
small signed difference to lose, no common mode to remove, no gentle signed structure to preserve. P1 only has to
rescale a clean rank-ordered non-negative vector. **Divisive normalization of a non-negative magnitude is the
DIAGONAL/gain half** — the cheap, separable, point-neuron half (J.10 shunting literally "divides... by a factor").
Conflating S5's gain-normalization with whitening is the trap; they are categorically different.

---

## 2. RANKED, BIOLOGICALLY-GROUNDED OPTIONS

Each: mechanism · biology source · point-neuron-feasible vs needs-dendrites · expected failure mode.

### Option 1 (TOP) — route the FHRR composer's S5 normalization through the ALREADY-VALIDATED on-bridge NEF input-normalization FS pool

- **Mechanism.** The core-sim NEF cleanup (`research/findings/raw/_spiking_cleanup_nef.py`, operating point
  `NEF_CLEANUP_OP`) already does: (1) **input normalization** — a spiking inhibitory-trait FS pool pools the est
  ON/OFF input population and shunts it back, making the matched-filter drive ~scale-invariant (≈ cosine) across the
  per-query scale; (2) matched filter (codes as receptive fields); (3) **threshold placement** — a negative concept
  bias so off-target (cosine ~0) emits ZERO spikes; (4) per-concept firing-sum → argmax. The S5 fix is to give the
  FHRR composer's `_spiking_cleanup` the SAME stage-1 on-bridge input-norm FS pool *instead of* the host
  `peak = scores.max(); drive = scores/peak` shortcut. The decoded-line drive then comes from the normalized,
  threshold-placed firing — no host read.
- **Biology source.** Carandini-Heeger normalization-as-canonical-computation (Nature Rev Neurosci 2012,
  [nrn3136](https://www.nature.com/articles/nrn3136)); Stewart-Tang-Eliasmith 2011 NEF cleanup (Spaun); catalog **J.10**
  GABA_A shunting ("opening Cl⁻ channels increases membrane conductance and divides the EPSP voltage by a factor —
  works without hyperpolarization") — verified `implemented` in the sim; catalog **E.05** center-surround / **E.15**
  attention gain modulation ("multiplies firing rates").
- **Point-neuron-feasible: YES — already running on-bridge at production.** `2026-06-05-composer-cleanup-NEF-GO.md`:
  27/27 numpy parity at D=2048, V=320, seeds 42/43/44, NO `sim/` edits, the input normalization "makes the threshold
  scale-invariant... production parity held with no retuning." This is the strongest possible evidence — it is not a
  proposal, it is a deployed circuit.
- **Expected failure mode.** The input-norm FS pool is **sharply tuned** (the 2-stage divnorm arc found the input-FS
  weights need a "gentle" sweet spot, `2026-06-04-divnorm-NEGATIVE` UPDATE: `w_in_cfs=0.5`, over-shunts to 0.000 by
  4.0). The S5 risk is that the FHRR composer's per-query score range (1.3e5-3.3e5, possibly wider than the core-sim
  est) lands outside the tuned basin. Mitigation: the NEF op already shipped robustly at the FHRR-adjacent scale; the
  de-risk sweeps the input-norm gain to confirm the basin covers the S5 score range. Residual: if the FHRR composer's
  K (concepts compared per query) in the integrated loop is small (K=2 in Task 2), the problem is *easier* than the
  320-way NEF validation, not harder.

### Option 2 — conductance-based BALANCED-E/I divisive pool on the score layer (the canonical normalization circuit, done right)

- **Mechanism.** A dedicated inhibitory pool sums the score population and feeds back DIVISIVELY by modulating
  BOTH the excitatory drive and a shunting conductance in a balanced way, so `response_i ≈ drive_i / (σ + g·Σ_j drive_j)`.
  The crucial literature correction (below) is that **single-conductance shunting alone is SUBTRACTIVE, not divisive**;
  true divisive gain needs balanced E+I modulation. This is the "do it right" version of the project's prior FS-shunt
  pool.
- **Biology source.** Carandini-Heeger ([NECO pool-normalization model](https://dl.acm.org/doi/abs/10.1162/NECO_a_00675));
  **decisive 2024-2025 literature:** "shunting inhibition alone results in a SUBTRACTIVE, not divisive, modulation of
  firing rates... divisive gain modulation can be achieved by modulating both excitatory and inhibitory inputs in a
  balanced manner" ([Science Advances 2024, sciadv.adv9396](https://www.science.org/doi/10.1126/sciadv.adv9396)); E-I
  circuits "dynamically regulate neuronal activity through subtractive and divisive inhibition, which respectively
  control the activity and the gain of excitatory neurons" ([arXiv 2509.23253, Nov 2025](https://arxiv.org/html/2509.23253));
  "integration of excitation and feedforward inhibition leads to divisive normalization at SUB-THRESHOLD potentials...
  balanced inhibition progressively reduces output with increasing input" ([eLife 99808 multiple-interneuron-class
  gain](https://elifesciences.org/articles/99808)).
- **Point-neuron-feasible: YES (with the balanced-E/I caveat).** This is the textbook point-neuron divisive circuit —
  but the literature is explicit that a naive single-g shunt is subtractive. The project's 2026-06-04 FS-shunt pool
  reaching only 0.844-0.911 is CONSISTENT with the subtractive-not-divisive limitation; the balanced-E/I form is the
  documented fix. Note the literature's "divisive normalization at SUB-THRESHOLD potentials" is the GAIN on the
  membrane (point-neuron-fine), distinct from the analog-whitening sub-threshold computation the dendritic boundary
  forbids.
- **Expected failure mode.** Calibrating the balance (the E and I gains must track to give a clean divide) is fiddly;
  if mis-balanced it degrades to subtractive and the threshold doesn't transfer. Strictly inferior to Option 1 for
  S5 because Option 1 is already validated; Option 2 is the principled fallback if Option 1's basin is too narrow.

### Option 3 — recurrent soft-WTA on the score layer (self-normalizing, then the placed threshold reads the winner)

- **Mechanism.** A recurrent WTA / k-WTA over the score population self-normalizes (the winner suppresses the field;
  the steady-state firing reflects RANK, not absolute scale), then the placed threshold (P2) reads the winner. Soft-WTA
  "outputs reflect the rank of all inputs according to their size" — scale-invariant by construction.
- **Biology source.** Catalog **A.04** (BG output disinhibition is competitive WTA at GPi/SNr — IMPLEMENTED in
  `g11_bg_runner`, the integrated loop already uses it for answer/abstain); soft-WTA spiking literature
  ([self-stabilizing WTA, arXiv 1610.02084](https://arxiv.org/pdf/1610.02084); [WTA with spiking inputs, IEEE
  1399650](https://ieeexplore.ieee.org/document/1399650/)); Rutishauser WTA stability (α>1).
- **Point-neuron-feasible: YES.** WTA is a recurrent point-neuron motif; the project runs one (BG cascade). But — the
  project's hand-WTA cleanup attempt was a sharp NEGATIVE: "hand-tuned hard-WTA 0.13-0.16... violated Rutishauser α>1
  stability" (`2026-06-05-composer-cleanup-NEF-GO.md` line 20). The NEF GO explicitly chose **feed-forward, no
  recurrent WTA** because the recurrent version was unstable at this scale.
- **Expected failure mode.** Recurrent-WTA instability (oscillation / multi-winner / collapse) at the wrong α; the
  project has already been burned here. Lower-ranked than Options 1-2 for exactly this reason. (Note: the integrated
  loop's BG WTA at S6 is the *downstream* selection over the *already-decoded* lines — it is NOT a substitute for the
  S5 per-query normalization of the GRADED score; S6 operates on the post-S5 decoded lines.)

### Option 4 — repurpose the EXISTING on-bridge `input_divisive_norm` Carandini-Heeger primitive (MEAN-pool divisor)

- **Mechanism.** `sim/regions.py:235` + `bridge.py:6048-6057` already implement, opt-in and default-OFF, a per-region
  divisive normalization: for flagged neurons, `r_i = x_i / (σ + gain·MEAN_j x_j)`. Flag the cleanup score region with
  `input_divisive_norm=True`, set `cfg.enable_input_divisive_norm`, tune `input_divisive_sigma`/`input_divisive_gain`,
  and the pre-threshold score is divided by a per-query mean-pool divisor ON-BRIDGE — then the placed threshold reads
  the normalized firing. (The graded NEF/FS-pool versions in Options 1-2 are physically realized as a spiking pool;
  this one is the closed-form divisive op already in the step loop.)
- **Biology source.** Carandini-Heeger (the docstring cites it explicitly); built 2026-06-15 for the PPMI per-concept
  normalization.
- **Point-neuron-feasible: YES — it is literally in `sim/bridge.py` today, byte-identical when off.** This is the
  lowest-friction on-bridge knob.
- **Expected failure mode.** The divisor is the MEAN over the flagged set, not the MAX/peak. Mean-normalization is
  monotone-equivalent to peak-normalization for a fixed K and a fixed score-shape (both rescale by a per-query scalar
  that tracks the score magnitude), so a placed threshold CAN transfer — but if the number of above-zero scores varies
  query to query (e.g. abstain queries have NO winner → mean ~0 → divisor ~σ; answer queries have a sharp winner →
  mean dominated by it), the mean divisor behaves differently than peak across query TYPES, which could perturb the
  threshold's pass/abstain calibration. This is the precise thing the per-query-peak sweep (anti-cheat below) must
  test. Whether mean-pool suffices or the peak/NEF-FS version is needed is the empirical fork — and the cheap-first
  de-risk tests it directly, on a primitive that already exists, no new `sim/` code.

### Option 5 (the honest negative branch) — IF the de-risk shows mean/peak normalization cannot make the threshold transfer across query TYPES at a fixed op, S5 maps to the deferred dendritic substrate

- **Mechanism / verdict.** If, across a per-query-peak sweep AND the abstain-vs-answer query-type contrast, no fixed
  point-neuron divisive op (Options 1-4) lets the placed threshold cleanly separate winner from runner-up WHILE
  preserving the moat's abstain decision, then S5's normalization needs an analog/dendritic stage that the point-neuron
  rate readout cannot supply — the same family as the whitening / opponency / graded-readout boundaries.
- **Point-neuron-feasible: NO (this is the boundary branch).** Maps to the deferred dendritic substrate (the
  two-compartment neuron / learned graded cortex; D2 Phase 0-2 built, Phase 3 pending —
  `feedback_dendritic_substrate_fair_game.md`).
- **Why this is a VALID, valuable outcome.** It would tell the owner that the *fully-zero-host-read* loop needs the
  deferred dendritic substrate, and that the *qualified one-host-read* loop (Task 2's honest scope: all computation +
  all control in spikes, one DATA number read between cleanup and sequencer) is the point-neuron CEILING. Given the
  NEF-GO precedent, this is the LOW-probability branch — but the de-risk must be able to land here honestly.

**Ranking rationale:** Option 1 first because it is *already validated on-bridge at production* (lowest risk, highest
evidence). Option 4 is the cheapest *first probe* (the primitive exists in `sim/`, zero new code) and its result
selects between "S5 closes with the existing op" vs "needs the NEF-FS pool (Option 1) / balanced-E/I (Option 2)". So
the de-risk runs Option 4 as the falsification probe and escalates to Option 1 if mean-pool is insufficient.

---

## 3. REUSABLE PROJECT MACHINERY (what a cheap-first de-risk reuses, no/minimal new `sim/`)

On-bridge normalization primitives that ALREADY exist in `sim/` (all opt-in, default-OFF, byte-identical when off):

| primitive | location | what it does | bearing on S5 |
|---|---|---|---|
| **`input_divisive_norm`** (Carandini-Heeger) | `regions.py:235`; `bridge.py:6048-6057` | per-flagged-neuron `x_i/(σ+gain·mean_j x_j)`, pre-threshold | **Option 4 — the direct on-bridge per-query divisor** |
| **NEF input-normalization FS pool** | `research/findings/raw/_spiking_cleanup_nef.py`; wired in `core_sim_composition.py` (commit 18352657), op `NEF_CLEANUP_OP` | spiking inhibitory-trait FS pool shunts the input → scale-invariant matched-filter drive + placed threshold | **Option 1 — the validated on-bridge S5 normalizer (27/27 @ D=2048)** |
| **`input_mean_adapt`** (slow per-hub) | `regions.py:234`; `bridge.py:6076-6088` | per-neuron `raw − gain·m; m←(1−α)m+α·raw`, the DIAGONAL/DC centering | adjacent diagonal primitive (subtractive, not divisive — supporting, not the fix) |
| **`graded` analog pathway** | `regions.py:365`; `bridge.py` graded-synapse block | a pathway transmits on the SOURCE's continuous membrane, not spikes (the retina horizontal-cell mechanism) | the analog-channel route IF Option 5 (dendritic) is reached — carries graded magnitude without the spike binarization |
| **`graded_lateral`** (full-K×K pre-spike) | `regions.py:217`; `bridge.py:1841-1929`, `6098-6103` | learned pairwise lateral on sub-threshold `a` (the whitening attempt) | NOT for S5 (that's decorrelation/off-diagonal); cited to keep the two problems separate |
| **shunting inhibition** (J.10) | `E_inh=-75mV` + 0.7× scaling, implicit in `g_i` | conductance-increase divides the EPSP | the substrate-level divisive primitive Options 1-2 build on |
| **BG WTA** (A.04) | `g11_bg_runner`, `--bg-lateral-inhibition` | competitive selection at GPi/SNr | the S6 *downstream* selection (already in the loop), and the Option-3 soft-WTA reference |

Runners / harness to reuse: `rf_phasor_composer.py` (`_spiking_cleanup`, the S5 site), `core_sim_composition.py`
(`enable_spiking_cleanup`, the NEF integration), `_phaseC_task1_cleanup_probe.py` +
`_phaseC_task1_S5_seam_derisk.py` (the existing S5 probe + seam harness — extend, don't rebuild),
`_phaseC_task2_wholeturn_loop.py` (`LoopComposer` — the integrated loop the closed S5 plugs into), the divnorm
artifacts (`_spiking_cleanup_divnorm_probe.py`, `_spiking_cleanup_2stage.py`, `_spiking_cleanup_nef.py`).

The g_e/g_i trait-routing discovery (`2026-06-04-divnorm-NEGATIVE`): the bridge routes a synapse to `g_i` iff the
PREsynaptic neuron carries an inhibitory trait — the FS-pool normalizer MUST set `enable_inhibitory_neurons` and the
pool's trait, or "I_TO_E" weights add to `g_e` (lateral EXCITATION, the silent bug that made the early WTA "hurt").

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK (the smallest experiment that falsifies the top option)

**Goal:** falsify "a point-neuron divisive normalization on-bridge makes the placed threshold separate winner from
runner-up ACROSS queries with DIFFERENT peaks, with the moat intact and NO host read." CPU/numpy first.

**Probe (extend `_phaseC_task1_cleanup_probe.py` → `_phaseC_S5_normbridge_probe.py`):**
1. Build the real `OneBrainComposer` (D=64 cheap-first; the loop's K). For each of a battery of queries spanning a
   WIDE per-query peak range (engineer the score peaks to vary, e.g. by varying the number of stored facts and the
   cue match strength so raw peaks span ≥1 order of magnitude — the decisive control), run the cleanup to the GRADED
   `Re(c_k)` membrane.
2. **Option 4 first (cheapest, zero new `sim/` code):** flag the cleanup score region `input_divisive_norm=True`,
   `cfg.enable_input_divisive_norm=True`, sweep `input_divisive_sigma`/`input_divisive_gain`. After the on-bridge
   divisive op, apply the SAME placed firing threshold (P2) the NEF cleanup uses; read the per-concept firing-sum →
   argmax-by-firing. NO host peak-read anywhere in the score path.
3. If Option 4's mean-pool divisor does NOT give a single (σ, gain, threshold) op that works across the peak sweep AND
   the answer-vs-abstain query types, escalate to **Option 1** (the NEF input-norm FS pool driving the FHRR score)
   and/or **Option 2** (balanced-E/I), reusing `_spiking_cleanup_nef.py`.

**Decision metric (the decisive control = per-query-peak robustness):** the winner-vs-runner-up SEPARATION (winner
fires, runner-up silent) must hold at a FIXED op across the full peak sweep — NOT just one query. AND the abstain
queries (no true winner) must produce NO above-threshold firing (the moat). GO iff one fixed op separates across all
peaks AND the moat holds; NEGATIVE (→ Option 5 dendritic branch) iff no fixed op survives the peak sweep without a
moat breach.

**Why this is the right cheap-first cut:** it reuses an existing `sim/` primitive (Option 4) and an existing on-bridge
circuit (Option 1) — likely NO new `sim/` code at all. It is CPU-runnable at D=64. It tests the ONE thing the host
read does (per-query rescaling) under the ONE condition that matters (different peaks), with the moat as the gate.

---

## 5. ANTI-CHEAT CONTROLS (the de-risk's required guards)

1. **Moat = 0 false-accepts = HARD GATE.** Every absent-agent / absent-action / cross cue MUST abstain (no
   above-threshold firing). A single moat breach = NEGATIVE for that op, reported as the boundary, never relaxed to
   manufacture a pass — exactly as Task 1 did. (Per `feedback_moat_not_hard_lossy_memory_ok.md` the moat is a plus not
   a hard product gate generally, but for THIS de-risk it is the falsification gate: the failure mode of bad
   normalization IS a moat breach, so the moat is the measurement.)
2. **Host-normalization POSITIVE control.** Run the existing host-peak path (`scores/peak`, the current S5) on the
   SAME query battery — it must pass (winner/runner-up separated, moat held). This proves the battery is
   discriminable and the harness sound; the on-bridge version must match it, not beat a broken control.
3. **No-normalization NEGATIVE control.** Drive the placed threshold from the RAW un-normalized scores (no divisive
   op). This must FAIL the peak sweep (the whole row lights / moat breaks at some peak) — reproducing the Task-1
   `both_suprathreshold` failure, confirming normalization is load-bearing and the GO (if any) is attributable to it.
4. **Per-query-peak SWEEP (the decisive control).** The query battery MUST span a wide range of raw peaks at a FIXED
   operating point. A result that separates winner/runner-up on ONE peak but not across the sweep is NOT a closure —
   it is the host-read-replacement-that-only-works-for-one-query trap. (This is the exact control the S5 problem
   demands: "it must separate winner from runner-up across queries with DIFFERENT peaks, not just one.")
5. **Lesion-fails-safe.** Severing the on-bridge normalizer → decoded lines silent → abstain (as Task 1's
   `lesion-fails-safe=True`), so a GO is attributable to the live circuit, not a wiring artifact.
6. **OFF == byte-identical.** Any `sim/` touch (none expected for Option 4) keeps every existing run byte-identical
   when the flag is off (the project's standing `sim/`-edit discipline).

---

## 6. HONEST FRAMING (top-line, restated)

**S5 is closable on the point-neuron substrate — high confidence — because the project has ALREADY built and validated
the exact normalizer it needs on-bridge** (the NEF input-normalization FS pool, 27/27 numpy parity at production
D=2048, NO `sim/` edits; `2026-06-05-composer-cleanup-NEF-GO.md`). The residual S5 host read in
`rf_phasor_composer._spiking_cleanup` is a cheaper host-peak shortcut the FHRR composer kept; the fix is to route its
normalization through the validated on-bridge circuit (Option 1) — or, cheaper still, through the `input_divisive_norm`
Carandini-Heeger primitive already in `sim/bridge.py` (Option 4). The S5 seam doc itself names this exact lever
(`2026-06-19-phaseC-task1-S5-seam-derisk.md` lines 54-61): "if that normalization can be a point-neuron divisive pool,
S5 becomes fully on-substrate... if it needs analog/dendritic normalization, that maps the boundary to the deferred
dendritic substrate." **The biology + the project's own track record say it CAN be a point-neuron divisive pool.**

**The decisive distinction (do not lose it):** S5's per-query NORMALIZATION is the DIAGONAL/gain half of normalization
(divisive gain control on a clean non-negative magnitude — point-neuron-feasible, J.10 shunting, Carandini-Heeger). It
is categorically NOT the off-diagonal DECORRELATION / whitening that is the genuine Mikulasch-Priesemann dendritic
boundary (mixing across dimensions to remove a signed common mode). The project repeatedly conflates "graded magnitude
through a spike" with "the dendritic boundary"; for S5 that conflation is wrong — the score is already rectified, there
is no signed structure to preserve, only a rank-ordered magnitude to rescale.

**The honest caveat:** the project's prior spiking-divnorm cleanup plateaued at 0.911 (`2026-06-04-divnorm-NEGATIVE`)
as a *320-way numpy-exact argmax readout* — but that plateau was SUBSEQUENTLY CLEARED to 0.978-1.000 by the NEF
follow-on with on-bridge input normalization, and S5 is a structurally easier ask (small-K, placed-threshold
pass/abstain protecting the moat, per-query DATA not 320-way parity). The literature corrects the one trap to avoid:
**single-conductance shunting alone is SUBTRACTIVE, not divisive** ([Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adv9396))
— true divisive gain needs balanced E/I (Option 2) or the NEF-FS pool's input-shunt-then-place-threshold structure
(Option 1) — which is precisely why the naive 2026-06-04 FS-shunt pool capped at 0.91 and the NEF version did not.

**The recommendation:** run the Option-4 cheap-first probe (existing `input_divisive_norm` primitive, CPU, D=64,
per-query-peak sweep, moat as the gate). If it GOes, S5 closes with zero new `sim/` code and the loop has zero host
round-trips. If mean-pool is insufficient, escalate to Option 1 (the validated NEF input-norm FS pool). The dendritic
boundary (Option 5) is the low-probability branch, and if reached is itself a valid deliverable (it maps the
fully-zero-host-read loop to the deferred dendritic substrate, with the qualified one-host-read loop as the
point-neuron ceiling).

---

## Sources (verified against the actual text)

- **Catalog** `sim-catalog/references/feature-catalog.md`: **J.10** (GABA_A shunting "divides the EPSP voltage by a
  factor... implemented"), **E.05** (lateral inhibition / center-surround "decorrelates output", Kandel Ch 22
  p~588-593), **E.15** (top-down gain modulation "multiplies firing rates", Kandel Ch 25), **A.04** (BG output
  competitive WTA, IMPLEMENTED), **H.08** (Renshaw recurrent gain control on α-MN, missing).
- **Project findings:** `2026-06-19-phaseC-task1-S5-seam-derisk.md`, `2026-06-19-phaseC-task2-wholeturn-loop.md`,
  `2026-06-05-composer-cleanup-NEF-GO.md`, `2026-06-04-composer-cleanup-divisive-norm-NEGATIVE.md`,
  `2026-06-06-graded-lgn-decorrelation-BOUNDARY.md`, `2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`,
  `2026-06-15-slow-perhub-mean-primitive-deep-research.md`, `2026-06-15-phaseB-task3-centering-RESULT.md`.
- **`sim/` code:** `regions.py:217,234,235,365` (`graded_lateral`/`input_mean_adapt`/`input_divisive_norm`/`graded`);
  `bridge.py:6037-6103` (the divisive-norm + input-mean + graded-lateral step blocks); `rf_phasor_composer.py:242-297`
  (`_spiking_cleanup`, the S5 host read at 270-272).
- **Literature:** Carandini & Heeger, Normalization as a canonical neural computation, Nat Rev Neurosci 2012
  ([nrn3136](https://www.nature.com/articles/nrn3136)); shunting-is-subtractive / balanced-E/I-is-divisive
  ([Science Advances 2024, sciadv.adv9396](https://www.science.org/doi/10.1126/sciadv.adv9396)); E-I divisive/subtractive
  inhibition in deep SNNs ([arXiv 2509.23253, Nov 2025](https://arxiv.org/html/2509.23253)); multiple-interneuron-class
  gain & stability ([eLife 99808](https://elifesciences.org/articles/99808)); self-stabilizing WTA tradeoffs
  ([arXiv 1610.02084](https://arxiv.org/pdf/1610.02084)).

_Read-only deep-research deliverable. No code written, no experiments run. Load-bearing catalog/source claims verified
against the actual text._
