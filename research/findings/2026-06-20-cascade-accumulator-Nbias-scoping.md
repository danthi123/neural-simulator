# Cascade-accumulator North-bias — deep-research scoping (2026-06-20)

**Type:** READ-ONLY deep-research scoping (no code edits, no GPU). The gated next step after the FIX-3
NEGATIVE verdict (`2026-06-20-shortcut6-FIX3-opponent-axis.md`, `e4eaf2a6`) localized the residual #6 sub-blocker
to a **structural North-bias amplified by the Wang-2002 selection accumulators**. The host tie-break shortcut is
already CLOSED (converted to spikes via FIX 1); this scoping targets the remaining residual that keeps the neural
orienting above the host ceiling, so the host orienting heuristic can be RETIRED (closing #6).

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; grid-32 IS the verdict (never grid-8); a boundary is not an
exit. The no-confab moat is array-disjoint from the nav cascade (`cp_*` nav state vs the composer's complex
`cp_rf_w_*` synapses) and untouched. The research gate fires here under conditions (a) confirmed boundary + next
move is a mechanism to overcome it, (b) known family (the divisive-normalization / common-mode family), and (f)
≥2 distinct approaches to the same goal have failed (FIX 1 partial, FIX 2 ~null, FIX 3 + margin-amp NEGATIVE).

---

## 1. Diagnosis (two parts: SOURCE + AMPLIFICATION)

**One-paragraph statement.** A small, persistent, goal-invariant **thalamic North-over-South lead (`thal_counts`
N−S ≈ +1233, ~11%)** is **amplified ~9× by the Wang-2002 `sel_X` NMDA-recurrent accumulators (`sel_counts` N−S ≈
+10999, ~22%)** into a large selection-stage surplus that the Lo-Wang `commit_X` burst inherits (+10937) and that
no read-out reorganization can undo. The amplification is mechanistically expected — an NMDA-recurrent integrator
is *winner-amplifying by design* (positive feedback compounds a small persistent lead over the integration window;
Wang 2002). The source of the thalamic lead is a **geometric/topographic asymmetry**, not a read-out artifact: the
rig uses the symmetric pop-vector cosine decode (`sc_popvector_readout=True`, so the deployed half-plane-ramp's
S-suppression is NOT active here) and `enable_cluster_e_topography=True` places the N channel at the geometric TOP
of the unit square. The shared STN injects a COMMON (symmetric) baseline to all GPi, so it cannot be the
*differential* source — it raises the common mode the accumulator then amplifies, but the N-over-S *difference*
originates upstream of GPi.

### 1a. SOURCE of the small thalamic N-over-S lead (+1233, ~11%) — candidate localization

The thalamic relay `thal_X` is the disinhibition output of the per-cardinal `cortex_X → str_D1/D2_X → gpi_X →
thal_X` cascade, with a shared STN adding diffuse excitation to every `gpi_X` (`g11_bg_runner.py`):

- **`thal_X` is the GPi→thal disinhibition output** (`gpi_X → thal_X` inhibitory, `bridge.py:1780`). A higher
  `thal_N` count means `gpi_N` is more disinhibited than `gpi_S` — i.e. `str_D1_N` inhibits `gpi_N` slightly more,
  OR `cortex_N` drives `str_D1_N` slightly more, OR the shared-STN re-excitation of `gpi_S` is slightly stronger
  relative to its D1 inhibition.
- **The shared STN is a COMMON baseline, NOT a differential source** (`bridge.py:1767-1771`: a single `stn` pool
  projects diffusely to ALL `gpi_X` with the same weight). It cannot create an N-over-S *difference* by itself; it
  raises the common-mode drive into every GPi equally. (It does, however, set the *operating point* on which a
  small differential rides — see §1b.)
- **The cluster-E topographic prior is the most likely differential SOURCE.** `enable_cluster_e_topography=True`
  stamps 2D coordinates on `cortex_X` / `str_D{1,2}_X` at the cardinal corners of the unit square
  (`g11_bg_runner.py:919-924`: `N=(0.5,1.0)`, `E=(1.0,0.5)`, `S=(0.5,0.0)`, `W=(0.0,0.5)`), and the
  `cortex_X → str_X` weights are Gaussian-distance-weighted by `cluster_e_distance_sigma`
  (`g11_bg_runner.py:1618-1648`). The N corner `(0.5,1.0)` and S corner `(0.5,0.0)` are geometrically symmetric
  *about the centre*, so a perfectly centred Gaussian would be balanced — but the SC pop-vector read-out drive
  into `cortex_X` is NOT centred: it is the cosine projection `max(0, û_site · û_cardinal)` summed over the SC
  grid (`g11_bg_runner.py:287-296`). **At grid-32, the goal-bump sits in one SC quadrant; the cosine mass onto
  each cardinal is the projection of that bump's eccentricity vector, and any small top/bottom asymmetry in how
  the SC grid discretizes the bump (an even `SCN`, off-by-half centre `sc_center`, edge clipping at the top vs
  bottom row) biases the N-vs-S cosine sum** — a few-percent geometric offset, exactly the +1233/~11% observed.
- **Verified NOT the half-plane S-suppression** (the FIX-3 doc confirms `sc_popvector_readout=True`, the symmetric
  decode; the deployed ramp's `max(0,−sy_offset)` cortex_S darkening is inactive in this rig).

⇒ **The indicated source is a small geometric/topographic N-vs-S imbalance in the SC pop-vector → cortex drive
and/or the cluster-E corner geometry, riding on the shared-STN common baseline.** The cheap diagnostic (single-
region firing-rate probe down `cortex_X → str_D1_X → gpi_X → thal_X` per cardinal, GOAL-OFF / centred-bump) will
localize WHICH stage first shows the N−S offset; this is part of the de-risk (§4), not yet run here.

### 1b. AMPLIFICATION at the `sel_X` accumulators (~9×, the precise target)

The `sel_X` pools are Wang-2002 NMDA-recurrent accumulators with structured cross-pool WTA
(`g11_bg_runner.py:2148-2257`):

- **`thal_X → sel_X`** feed-forward, `density=1.0`, `weight_mean=thal_to_sel_weight`, **no baseline subtraction**
  (`g11_bg_runner.py:2217-2222`). Each `sel_X` integrates its OWN absolute thalamic drive.
- **`sel_X → sel_X` internal recurrence** (`internal_density=sel_recurrent_density=0.5`,
  `exc_weight_mean=sel_recurrent_weight`, `enable_nmda=True` → NMDA-slow τ≈100 ms; `g11_bg_runner.py:2161-2171`).
  This is the winner-amplifying positive feedback: a pool with a slightly higher input ramps faster, and its own
  recurrence re-excites it, compounding the lead (Wang 2002). With the 2026-06-19 default-on tuning
  (`sel_recurrent_weight=0.3`, the Usher-McClelland LEAK lever; `g11_bg_runner.py:3889`) the hysteresis is reduced
  but the within-trial amplification of a *persistent* lead remains.
- **`sel_FS_X → sel_Y` (Y≠X) cross-inhibition** (`g11_bg_runner.py:2231-2240`): structured lateral inhibition, but
  it is driven by the WINNER's own pool (`sel_X → sel_FS_X → suppress sel_Y`). **Crucially this is feedback
  inhibition that fires AFTER a pool ramps — it does NOT subtract the common-mode at the INPUT.** It sharpens the
  winner once a winner emerges, but a pool that is *already* ahead by the structural N-surplus is the one that
  recruits its FS first, so the lateral inhibition *reinforces* the structural lead rather than cancelling it.

**Why this amplifies a common-mode lead instead of a differential.** A free race of N absolute-evidence integrators
(each integrating its own drive, mutual feedback inhibition) is a "race / max" architecture, not a "diffusion /
difference" architecture (Bogacz et al. 2006, *Psychol Rev* 113:700 — the race and the diffusion model differ
precisely in whether each accumulator integrates its absolute input or the network integrates the *difference* of
inputs). A race over absolute inputs is sensitive to a persistent additive offset on one channel: the offset is
integrated like signal. The diffusion / opponent architecture integrates `x_N − x_S`, in which a *common*-mode
offset cancels and only the *differential* (position-bearing) evidence accumulates. **The cascade is a race; the
fix is to make the integration differential.**

**Why FIX 2 (per-pool homeostasis at `sel`) barely dented it (−9%).** Per-pool threshold-adaptation regulates
each pool's baseline *independently* toward a common target — it removes a *static per-pool* offset but NOT a
*persistent differential DRIVE* that is continuously re-supplied by the upstream thalamic lead every step. The
homeostatic EMA chases a moving target it can never catch (the lead is re-injected faster than the slow adapt
equalizes). This is the empirical signature that the remedy must be **differential/opponent at the input**, not
per-pool.

---

## 2. Canonical biology — how a real decision circuit avoids amplifying a small input bias

| principle | mechanism | source | maps to OUR cascade |
|---|---|---|---|
| **Integrate the DIFFERENCE, not each absolute side** | The optimal 2AFC accumulator integrates `x_A − x_B`; a common additive offset on both channels cancels (it is common-mode-rejected). The diffusion model = a single difference variable; the balanced mutual-inhibition LCA *approximates* it; the free race does NOT. | Bogacz et al. 2006 *Psychol Rev* 113(4):700–765 (the race↔diffusion equivalence requires balanced mutual inhibition + leak); Usher & McClelland 2001 *Psychol Rev* 108:550 (leaky competing accumulator) | The `sel_X` race integrates absolute `thal_X`; an opponent-PAIR (N↔S, E↔W) that integrates the difference would CANCEL the +1233 N-S common-mode lead before amplifying. |
| **Balanced mutual inhibition gives common-mode rejection** | When cross-inhibition between competing units is balanced with leak, the network's slow mode is the *difference* of activities; the *sum* (common mode) is a fast-decaying mode that is suppressed. Imbalanced/late inhibition does NOT reject the common mode. | Usher-McClelland 2001; Bogacz 2006 §"linear approximation" (the eigen-decomposition: difference mode = slow integrator, sum mode = leaky/decaying) | OUR `sel_FS` lateral inhibition is FEEDBACK (post-ramp), so the common (sum) mode is NOT suppressed — the structural N-surplus survives into the difference. Opponent-pairing at the INPUT fixes the mode structure. |
| **Divisive normalization removes the common drive** | Each unit's drive is divided by `σ + g·(mean of the pool)`; a uniform additive component is normalized away, leaving the relative (differential) pattern. Canonical cortical canonical-computation. | Carandini & Heeger 2012 *Nat Rev Neurosci* 13:51 (normalization as a canonical computation); catalog **E.05** (lateral inhibition & center-surround antagonism), **E.16/E.17** (opponent channels, L−M / S−(L+M) push-pull). | The bridge ALREADY has `input_divisive_norm` (`bridge.py:6079-6088`): divide a flagged region's input by `σ + g·mean`. Flagging `sel_X` (or `thal_X`) would divide out the common N+E+S+W drive, suppressing the common-mode lead BEFORE the accumulator integrates it. |
| **Subtractive feed-forward inhibition / predictive centering** | Subtract a slow running mean of the pool's own drive before threshold (the separable DC half of whitening; subtractive spike-frequency adaptation). | catalog E.05 / Mikulasch-Priesemann point-neuron predictive-coding; the project's own `2026-06-15-slow-perhub-mean-primitive-deep-research.md`. | The bridge ALREADY has `input_mean_adapt` (`bridge.py:6107-6119`): subtract a slow per-neuron EMA of its own input. A FAST per-pool variant of this on `sel_X` would subtract the persistent N-lead. (Caveat: per-NEURON/per-pool mean is exactly what FIX 2 was; the OPPONENT difference is the stronger form — see ranking.) |
| **SC motor map is an opponent push-pull** | Distinct SC populations encode OPPOSING movement directions with balanced E/I that prevents a directional bias; the omnipause (OPN) holds all burst pools until one crosses bound (Lo-Wang 2006 SC threshold). | catalog **A.07** (SNr→SC saccade release), **H.24/H.25** (SC omnipause / saccade-generator EBN / opponent push-pull — cited in the runner's `commit_X`/`commit_OPN` comments, `g11_bg_runner.py:2186-2214`); Nature Comms 2023 / Comms Biol 2025 (cited in the FIX-3 doc). | FIX 3 put the opponent push-pull at the READ-OUT (where it re-biased). Biology puts it at the INTEGRATOR (the SC bursters are mutually balanced BEFORE the OPN-gated commit). Realizing the opponent at the `sel`/`commit` stage (not the argmax) is the biology-faithful placement. |
| **MSN collaterals linearize, they do NOT WTA** | Wilson (PBR-160 ch 6): the functional role of MSN-MSN recurrent inhibition is *dendritic linearization* (normalization of the input range), NOT winner-take-all (which is weak: <0.5 mV IPSPs, ~14–25% connectivity). | catalog **B.04** + Wilson Fig 7/8; TK-2017 pp 160–163. | Reframes the striatal stage: the cascade's selection is NOT supposed to be a hard WTA at striatum; the differential competition belongs at the dedicated accumulator (`sel`), where an opponent organization is the right mechanism. |

**Net biology:** the cascade implements a *race over absolute drives* (sensitive to a common-mode offset); every
canonical decision circuit that avoids amplifying a small input bias does so by making the integration
**differential** — integrate the difference (diffusion/opponent), reject the common mode (balanced mutual
inhibition or divisive normalization), at the INPUT to / WITHIN the integrator, not at the read-out.

---

## 3. Ranked integrator/input-stage fixes (NOT the read-out)

> All four are point-neuron-achievable. Ranked by expected leverage × cheapness × reuse-of-existing-machinery,
> and by directness of common-mode rejection. The read-out is deliberately untouched (that is where FIX 3 failed).

### FIX A (RANK 1, RECOMMENDED) — Divisive normalization at the `sel_X` (or `thal_X`) input
**Mechanism:** flag `sel_X` (the four selection pools) with `BrainRegion.input_divisive_norm=True` and set
`cfg.enable_input_divisive_norm=True`, `input_divisive_sigma`, `input_divisive_gain`. Each `sel_X`'s input current
is divided by `σ + g·mean(sel-pool input)` (`bridge.py:6084-6088`) — the common N+E+S+W drive is divided out,
leaving the relative differential, so the accumulator amplifies the *position-bearing* pattern, not the structural
common-mode lead. **Biology:** Carandini-Heeger canonical normalization; catalog E.05/E.16/E.17. **Reuse:** the
`input_divisive_norm` primitive ALREADY EXISTS in `sim/bridge.py` (built + masked + guarded-no-op when off; it is
the very mechanism the `sc_popvector_readout` path already uses on `cortex_X`, `g11_bg_runner.py:4356-4361`).
**Cost:** ZERO `sim/` edit (the primitive exists) + one runner flag to set `input_divisive_norm=True` on the
`sel_X` BrainRegions + a σ/gain sweep. **Point-neuron:** yes (it is a per-region current-domain op already on the
substrate). **Risk:** a too-strong divisor could also normalize away the genuine differential (over-flatten) — the
σ/gain sweep + the anti-cheat (the N−S surplus at `sel` must SHRINK *while* the per-phase dom still tracks) gates
this.

### FIX B (RANK 2) — Opponent-PAIR the `sel` accumulators (N↔S, E↔W integrate the DIFFERENCE)
**Mechanism:** add DIRECT mutual inhibition between the opponent partners at the accumulator stage — `sel_N ↔
sel_S` and `sel_E ↔ sel_W` inhibit each other STRONGLY and symmetrically (a dedicated opponent FS pair, or direct
balanced inhibitory pathways), so each axis integrates `sel_N − sel_S` (the diffusion-model difference variable;
Bogacz 2006). A common-mode N-S offset cancels in the difference; only the position-bearing differential
accumulates. This is the SC opponent push-pull realized at the INTEGRATOR (where biology has it) rather than the
read-out (where FIX 3 had it). **Biology:** Bogacz 2006 race↔diffusion equivalence (balanced mutual inhibition →
difference mode is the slow integrator); Usher-McClelland 2001; SC opponent organization (H.24/H.25, A.07).
**Reuse:** the `sel_FS_X` per-pool FS machinery already exists; this re-wires the cross-inhibition from the current
"winner's-own-FS suppresses all losers" (feedback) to a balanced opponent-pair topology (the strong N↔S / E↔W
mutual inhibition). **Cost:** runner-only (re-declare the `sel_FS → sel` pathways as opponent pairs with balanced
weights); NO `sim/` edit. **Point-neuron:** yes. **Risk:** the *balance* is load-bearing — imbalanced opponent
inhibition re-creates a bias; the symmetry must be exact (same weight both directions) and the lateral inhibition
must be strong enough to make the difference-mode dominate. Slightly more wiring-design than FIX A.

### FIX C (RANK 3) — FAST per-pool subtractive centering at `sel_X` (input_mean_adapt, fast α)
**Mechanism:** flag `sel_X` with `input_mean_adapt=True` and a FAST `input_mean_adapt_alpha` (and gain) so each
pool subtracts a fast running mean of its own input before threshold (`bridge.py:6113-6119`). **Why it might work
where FIX 2 didn't:** FIX 2 was per-pool *threshold-adaptation* (a homeostatic EMA on firing rate); this is
per-pool *input-current* subtraction with a tunable-fast α directly on the drive. A faster α tracks the persistent
lead more tightly. **Biology:** subtractive spike-frequency adaptation / predictive centering (E.05; the project's
slow-per-hub-mean primitive). **Reuse:** `input_mean_adapt` ALREADY EXISTS (`bridge.py:6107`). **Cost:** ZERO
`sim/` edit + one runner flag + an α/gain sweep. **Point-neuron:** yes. **Risk:** it is the same *family* as FIX 2
(per-pool, not differential), so it shares FIX 2's structural weakness — subtracting each pool's OWN mean does not
remove a *differential* drive as cleanly as dividing/subtracting across the pool (FIX A) or pairing (FIX B). Listed
because it is the cheapest possible test (one flag) and a useful negative control: if a FAST per-pool subtraction
STILL barely dents the surplus, that confirms (per the §1b logic) the remedy must be differential (FIX A/B), not
per-pool — sharpening the diagnosis at near-zero cost.

### FIX D (RANK 4, fallback if the source-(1) lead is irreducible) — Equalize the SOURCE (SC far-blob SNR / topography)
**Mechanism:** if the §4 source-probe shows the N-S lead originates at the SC pop-vector → cortex drive geometry
(an even-`SCN`/off-centre discretization asymmetry), fix it at the source: re-centre the SC grid (odd `SCN` so a
true centre site exists), or symmetrize the cosine read-out so the N-vs-S cosine sums are exactly balanced for a
centred bump, or sharpen the SC RF / add SC neurons so the far-blob differential SNR exceeds the residual offset.
**Biology:** SC retinotopic map fidelity (catalog E cortical encoding; the foveation/RF-sharpening lever the
FIX-3 doc flagged). **Reuse:** the `install_spiking_sc_wiring` geometry (`g11_bg_runner.py:202-310`). **Cost:**
runner-only geometry change; possibly a `sim/` grid-size knob (small, additive). **Point-neuron:** yes. **Risk:**
treats the symptom-source rather than making the accumulator robust; only pursue if FIX A/B cannot suppress the
lead (i.e. the lead is too large for normalization to reject without over-flattening). This is the "make the input
clean" path vs FIX A/B's "make the integrator reject the common mode" path; A/B are preferred because robustness-
to-a-biased-input is the more general (and more biologically faithful) property.

---

## 4. Recommended cheap-first de-risk + anti-cheats

**The smallest test (do this first, before any wiring change):** a **two-step CPU/cheap probe** —

1. **SOURCE probe (localize the +1233).** Run the existing `_nav_sc_popvector_readout_derisk` rig with the SC
   bump held at a CENTRED / GOAL-OFF condition and log per-cardinal firing down the chain (`cortex_X`,
   `str_D1_X`, `gpi_X`, `thal_X` — the `*_counts` logs already exist, `g11_bg_runner.py:7149-7161`). Identify the
   FIRST stage that shows the N−S offset. (Read-only of existing instrumentation; localizes FIX D vs A/B.)
2. **FIX A cheapest-first (divisive-norm at `sel`).** Flag `sel_X.input_divisive_norm=True` +
   `enable_input_divisive_norm=True`, sweep `input_divisive_sigma` / `input_divisive_gain`, and at grid-32 measure
   BOTH (a) the `sel_counts` N−S surplus SHRINKS toward 0, AND (b) the per-phase per-cardinal dom now TRACKS the
   moving goal (the FIX-3 failure mode was dom = N,N,N,N). The success criterion is the conjunction: **surplus
   shrinks AND dom tracks AND post-change Σ drops toward HOST ~1.6.**

**Anti-cheats (every one required for a GO):**

| anti-cheat | requirement |
|---|---|
| **Per-phase per-cardinal dom (THE discriminator)** | dom must SHIFT to track the moving goal across phase0 NE / phase1 far-W / phase2 SW / phase3 SE — NOT re-bias to a new fixed cardinal (the exact FIX-3 failure). |
| **The N−S surplus must SHRINK at `sel`/`commit`** | the decisive quantitative gate: `sel_counts` N−S must drop from ~+10999 toward 0 (FIX 2 only dented it −9%; a real fix removes most of it). Report the per-phase N−S and E−W axis margins. |
| **Scramble / retinotopy lesion MUST collapse** | SCRAM(FIX) must collapse relative to intact (the SC decode must remain load-bearing; a fix that works by ignoring the SC drive is a cheat). |
| **grid-32, NOT grid-8** | the verdict is grid-32/1800/warmup-600. A grid-8 "pass" does not count. |
| **Host ceiling anchors the gap** | report HOST (post-change Σ ~1.57, dom tracks every phase) alongside; the GO bar is the neural Σ approaching HOST so the host heuristic can RETIRE. |
| **Multi-seed** | 6-seed before any GO claim (the standing rule); a single-seed indicator is a screen only. |
| **Default-off byte-identical** | the divisive-norm flag is guarded-no-op when off (`bridge.py:6079`); the default cascade must be byte-identical (regression on `test_nav_conv_merged_agent` 8/8 + `test_nav_conv_step2b_coresident` 7/7). |
| **No-confab moat untouched** | the nav cascade is `cp_*` nav state, array-disjoint from the composer's complex `cp_rf_w_*` synapses; no conversational regression. |
| **Over-flatten control (FIX A/B specific)** | confirm the fix is not just suppressing ALL selection (a degenerate "everything ties" state that the FIX-1 tie-break then random-walks); the dom must still be DECISIVE per phase (a clear winner that tracks), not a flat draw. |

---

## 5. Verdict

**Point-neuron-achievable? YES.** This is a **normalization / differential-integration problem**, not a
graded-read-out or dendritic problem. Common-mode rejection by divisive normalization (FIX A) and by balanced
opponent mutual inhibition (FIX B) are both standard point-neuron, current-domain operations, and the two
load-bearing primitives (`input_divisive_norm`, `input_mean_adapt`) ALREADY EXIST on the bridge as guarded,
default-off, per-region masks. The fix is at the INTEGRATOR/INPUT stage (where biology rejects the common mode),
explicitly NOT the read-out (where FIX 3 re-biased). No dendritic substrate, no `sim/` edit expected (the primitive
exists; the work is runner-side flagging + a σ/gain or opponent-balance sweep).

**Would fixing it close #6 (retire the host orienting heuristic)? PLAUSIBLY YES, conditional on the de-risk.** The
FIX-3 diagnosis is decisive that the residual is a count-level common-mode contamination upstream of the read-out;
removing the common mode at the accumulator input is the mechanistically-matched remedy. If FIX A (or B) shrinks
the `sel`/`commit` N−S surplus toward 0 AND the per-phase dom tracks AND post-change Σ approaches HOST ~1.6 at
grid-32 multi-seed, the neural orienting matches host and the host heuristic RETIRES — closing #6. The honest
residual risk: if the source-(1) lead is *irreducible* (too large for normalization to reject without
over-flattening the genuine differential), FIX D (equalize the SC source) is the fallback, and the over-flatten
anti-cheat will catch the degenerate case. The recommended order is **source-probe → FIX A → (FIX B if A
over-flattens or under-shrinks) → FIX D only if the source lead is irreducible.** This is the next mechanism to
pursue (the FIX is gated behind this scoping); it is NOT a deferral and NOT a closed boundary.

---

_READ-ONLY scoping (no code edits, no GPU). Load-bearing claims cited to `g11_bg_runner.py` / `bridge.py` line
numbers and to the catalog (A.07, B.04, E.05, E.16/E.17, H.24/H.25) + Bogacz 2006 / Usher-McClelland 2001 / Wang
2002 / Carandini-Heeger 2012. grid-32 IS the verdict (never grid-8). The no-confab moat is array-disjoint from the
nav cascade and untouched. (WebSearch was unavailable this session; the decision-circuit theory cited is canonical
and is anchored by the catalog entries pulled above.)_
