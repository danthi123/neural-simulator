# M1′ BUILD — the per-source ABSOLUTE windowed-spike-count gate on the BTSP write: **SHIPPED, VERIFIED, and NO-GO on the realized metric** — because the write's *instructive signal* is not exclusive (v_apical runs to ~1.9e5 mV; on/off ratio only 3.5:1), so there is no fact-specific write structure for any eligibility gate to shape (2026-07-25)

**Status:** the `sim/` edit is BUILT, default-off byte-identical (bit-equal golden), CI-guarded (4 new tests), and the gate
demonstrably ENGAGES on the real substrate at a sparse, non-degenerate fraction. **The GO metric did NOT move** — the
realized `CORE-GATED own/other` stays at ~1.0 at every θ, every seed. Per the task's instruction ("if it stays flat, that
is a real finding — report which of the two halves failed to compose and why"), this doc localises the failure precisely:
**neither half failed. A THIRD, previously unmeasured link failed — the one between the eligibility and the weight.**

---

## 1. The edit (additive, default-off, ~50 lines across 2 protected files)

| File | Lines | What |
|---|---|---|
| `sim/config.py` | **364-376** | `btsp_win_gate_theta: float = 0.0` (ABSOLUTE spike-count set-point; `<= 0.0` ⇒ gate OFF) + `btsp_win_gate_hill_n: float = 8.0` (CaMKII-like cooperativity), with the biology citation block. |
| `sim/bridge.py` | **307-309** | `self.cp_btsp_win_count = None` — the per-neuron box-car counter. **Never allocated** unless the gate is on. |
| `sim/bridge.py` | **3573-3582** | `def reset_btsp_window(self)` — the public **BOX-CAR RESET** (zeroes the counter; a no-op that allocates nothing when the gate is off). |
| `sim/bridge.py` | **8070-8093** | The gate itself, inside the `enable_btsp` block, immediately after `etilde_bt = self.cp_btsp_pre_elig[coo_bt.row]` and **before** `btsp_elig_exponent` / `btsp_elig_hard_thresh` / the summation: lazily allocate the counter, `+= fired_this_step`, form the **ABSOLUTE** Hill gate `g = c^n/(c^n + θ^n + 1e-30)` (**no** `.max()` normalisation anywhere), gather on `coo_bt.row`, and multiply into `etilde_bt`. Plus two observability scalars (`_btsp_win_gate_pass_n` / `_total_n`). |
| `research/runners/nmda_compositional_consolidation.py` | **371-377** | `build_substrate` plumbing: `comp_btsp_win_gate_theta` / `comp_btsp_win_gate_hill_n` (additive, default-off, mirroring `comp_btsp_elig_hard_thresh`). |
| `research/runners/_consol_twosided_generalize_probe.py` | — | `bridge.reset_btsp_window()` at the START of each fact's burst in `instrumented_write` (the box-car reset); `--btsp-win-theta` / `--btsp-win-hill-n`; per-fact gate-engagement stats; and the diagnostics of §4. |
| `tests/test_btsp_win_gate.py` | new | 4 CI tests (§2). |

All three qualifiers the research gate named as load-bearing are realised and none was expressible before:
**(i)** a box-car **COUNT** (the existing `Etilde` is an exponential low-pass); **(ii)** an explicit **RESET** (so it is a
*per-window* count); **(iii)** an **ABSOLUTE** θ — `btsp_elig_exponent` / `btsp_elig_hard_thresh` both normalise by
`etilde.max()` over **every synapse in the bridge**, which a spine has no access to (CaMKII Hill ≈ 8 against a fixed
molecular set-point — Bradshaw *PNAS* 2003; per-spine Ca²⁺ compartmentalisation — Kandel 6e Ch 13 pp 296-298;
prebound-glutamate frequency facilitation, "5-10 sustained afferents beat 20+ transient ones" — Polsky/Mel/Schiller
*J Neurosci* 2009).

## 2. Verification — byte-identical when off (a TEST, not a comment)

**Empirical pre-edit vs post-edit comparison** (`git stash` of `sim/bridge.py` + `sim/config.py`, identical harness,
**deterministic numpy backend**, 60-step held-plateau BTSP protocol):

| | md5(`cp_connections.data`) | dw (full precision) | `cp_btsp_win_count` |
|---|---|---|---|
| **PRE-EDIT** | `b14caf505f444290a992e71af87a3457` | `7.9346046447753906` | attribute absent |
| **POST-EDIT, θ=0.0 (default)** | `b14caf505f444290a992e71af87a3457` | `7.9346046447753906` | `None` (never allocated) |

**BIT-EQUAL.** That golden hash is pinned in `tests/test_btsp_win_gate.py::test_win_gate_off_is_byte_identical_to_pre_edit`,
which ALSO asserts `cp_btsp_win_count is None` — i.e. the guarded block provably never executes.

**Test suites:**
- `pytest tests/test_onbridge_btsp.py -q` — **numpy: 5 passed** pre-edit AND post-edit. **cupy: 2 failed, 3 passed**
  pre-edit AND post-edit — *identical failures* (`test_onbridge_btsp_behavioral_timescale_via_real_bistable_plateau`,
  `test_onbridge_btsp_stores_recurrent_assembly_specifically`); both are pre-existing, GPU-only, and **neither references
  any new parameter** (verified: they call `_gap4_btsp_onbridge_*_derisk.run`, which never touches `btsp_win_gate_*`).
- `pytest tests/test_btsp_win_gate.py -q` — **4 passed on numpy AND 4 passed on cupy** (byte-identical-off · box-car
  reset · absolute-threshold engagement · sustained-vs-transient discrimination). The golden-hash assertion is
  numpy-only by construction (cupy reductions are not bit-reproducible — see the noise floor below).
- `pytest tests/test_determinism.py -q` — 9 passed (no regression).

**GPU non-determinism floor (measured, important for reading §3).** The consolidation probe is *not* run-to-run
reproducible on cupy even at a fixed seed with an identical `thr_hash`: two PRE-EDIT runs of the winning protocol at
seed 42 gave `dw` 22.24 vs 18.31, `core_sizes` [20,17,14] vs [19,20,18], and `core_gated_own_over_other`
[0.921,0.962,0.943] vs [1.003,0.831,1.075]. **So ±0.15 on the ratio and ±3 on core sizes is noise.** A GO at 2.5 would
be far outside it; the observed ~1.0 is inside it.

## 3. Results — the gate ENGAGES; the GO metric does NOT move

Protocol = the M0-authorized winner, verbatim:
```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_twosided_generalize_probe --seed <S> \
  --commit-top-k 15 --elig-tau 30 --elig-hard-thresh 0.4 --cycles 1 --btsp-wmax 2000 --btsp-lr 0.000003 \
  --blocked --reset-elig --settle-steps 200 --btsp-win-theta <THETA>
```

### 3.1 Gate engagement (the "is the gate a void arm?" control) — PASSES
Window counts run 0-15 over the 30-step burst. The gate cuts a sparse, non-degenerate, uniform-across-facts fraction of
the 120 CA1 sources:

| θ | pass fraction (CA1), seed 42 | bridge-wide pass |
|---|---|---|
| 4 | 0.55 / 0.62 / 0.56 | 6114-6253 / 8860 |
| 6 | 0.31 / 0.42 / 0.48 | 4727-4842 / 8860 |
| **8** | **0.28 / 0.29 / 0.24** | 3476-3691 / 8860 |
| 10 | 0.13 / 0.13 / 0.12 | 2286-2743 / 8860 |
| 12 | 0.07 / 0.05 / 0.12 | 1568-1843 / 8860 |

θ=8-10 is the target regime the research gate specified (~10-25%), non-degenerate (≥14 cells), and **balanced across
facts** — so this is a real arm, not a gate that fails to cut.

### 3.2 The GO metric — FLAT at every θ, every seed (15/15 configs, 0/3 facts, 0/45 fact-passes)

θ=0 is the LESION arm (gate off, everything else identical). Every ratio is reported with its magnitude-free twin and
the raw per-slot masses, per the mass triad; per-fact, never a mean.

| seed | θ | **CORE-GATED own/other** | n_pass | slot-norm twin (mass-free) | slot mass max/min | gate pass frac | core sizes |
|---|---|---|---|---|---|---|---|
| 42 | 0 (LESION) | [0.99, 1.047, 1.08] | **0/3** | [1.027, 1.019, 1.071] | 1.044 | — | [20,18,18] |
| 42 | 6 | [0.982, 1.049, 1.091] | **0/3** | [0.961, 1.061, 1.103] | 1.022 | 0.31/0.42/0.48 | [19,29,24] |
| 42 | 8 | [0.965, 1.004, 1.060] | **0/3** | [0.956, 1.007, 1.067] | 1.010 | 0.28/0.29/0.24 | [28,23,22] |
| 42 | 10 | [1.049, 0.887, 0.890] | **0/3** | [1.023, 0.926, 0.873] | 1.046 | 0.12/0.12/0.12 | [22,25,19] |
| 42 | 12 | [1.047, 0.996, 0.962] | **0/3** | [1.041, 0.997, 0.967] | 1.008 | 0.07/0.07/0.07 | [21,17,28] |
| 43 | 0 (LESION) | [0.997, 1.110, 0.960] | **0/3** | [1.066, 1.038, 0.965] | 1.092 | — | [26,22,15] |
| 43 | 6 | [0.942, 1.250, 1.022] | **0/3** | [0.963, 1.214, 1.030] | 1.035 | 0.44/0.38/0.42 | [19,9,24] |
| 43 | 8 | [0.969, 1.047, 1.049] | **0/3** | [1.001, 1.002, 1.062] | 1.052 | 0.21/0.32/0.33 | [21,25,19] |
| 43 | 10 | [0.984, 1.108, 1.006] | **0/3** | [1.027, 1.064, 1.005] | 1.056 | 0.17/0.08/0.24 | [18,22,13] |
| 43 | 12 | [0.988, 0.762, 1.251] | **0/3** | [1.011, 0.765, 1.221] | 1.032 | 0.06/0.04/0.11 | [20,14,15] |
| 44 | 0 (LESION) | [1.010, 1.041, 0.922] | **0/3** | [1.002, 1.007, 0.963] | 1.054 | — | [18,18,18] |
| 44 | 6 | [0.873, 1.051, 0.896] | **0/3** | [0.872, 1.022, 0.921] | 1.038 | 0.38/0.39/0.48 | [16,22,18] |
| 44 | 8 | [0.886, 1.187, 0.879] | **0/3** | [0.936, 1.113, 0.889] | 1.082 | 0.31/0.17/0.25 | [19,29,21] |
| 44 | 10 | [0.898, 0.950, 1.018] | **0/3** | [0.893, 0.965, 1.007] | 1.018 | 0.15/0.12/0.13 | [21,23,24] |
| 44 | 12 | [0.993, 1.027, 0.993] | **0/3** | [1.076, 1.046, 0.906] | 1.120 | 0.04/0.10/0.07 | [15,20,16] |

**Mass triad, all 15 configs:** PERMUTED-CORE control 0.81-1.13 (collapsed, as required); RANDOM-CA1 control 0.79-1.22
(collapsed); `slot_mass_ratio` 1.008-1.120 (no winner-slot artifact anywhere — contrast the retracted lead's
`[24, 80, 24]`); the slot-normalised twin tracks the raw ratio everywhere (so nothing here is mass-driven, in either
direction); every core ≥ 9 cells (non-degenerate); `thr_hash` distinct per seed.

**GO-gate: realized core-gated own/other ≥ 2.5 AND own-is-max on ≥2/3 facts at ≥3 seeds → 0 of 45 fact-passes. NO-GO.**
The M0 prediction was 2.9-4.0; the realized value is 1.00 ± 0.10, i.e. indistinguishable from the lesion arm and from
the measured GPU noise floor.

**Critically, the M0 oracle REPRODUCES in every one of these runs** (`M0 VERDICT: 3/3` at all 15 configs, during-write
gated ceilings 2.6-8.3). So the harness is unchanged and M0's GO is not in question — the write simply does not realise
it.


## 4. WHY it stays flat — the localiser (this is the deliverable)

Three links must hold for the M0 oracle's algebra (`own/other = self/cross overlap of the code as seen through g`) to be
realised. The probe now measures each one directly.

| Link | Metric | Result | Verdict |
|---|---|---|---|
| **1. firing → windowed count → gate → eligibility** | `corr(window_elig, window_fire)` per fact | **0.831-0.907** — all 45 fact×config values, every seed, every θ | ✅ **INTACT.** The gate composes exactly as designed onto the eligibility. |
| **2. eligibility → WEIGHT** | `corr(w[·→slot_j], window_elig_j)` | **≈ 0** — 45 values spanning −0.126 … +0.356, median ≈ 0.04 | ❌ **BROKEN.** The realized `ca1→slot` weight vector carries essentially none of the eligibility structure the gate shaped. |
| **3. the write is slot-EXCLUSIVE** | cross-slot `corr(w[·→slot_i], w[·→slot_j])` | **0.945-0.983** — all 45 pairs | ❌ **BROKEN.** The three slots receive ~97 % the SAME weight vector. |
| **3b. the instructive signal is exclusive** | `IS[i,j]` diagonal ÷ off-diagonal | **3.45-3.48** at every one of the 15 configs (σ ≈ 0.01) | ❌ **BROKEN, and astonishingly stable** — the write drive into a NON-target slot is 29 % of the target's, every step. |

### The root cause, measured
The BTSP rule is `dw ∝ η · Ẽ_pre · IS_post · (w_max − w)` with `IS_post = max(v_apical − plateau_v_hold, 0)`. The write
protocol's whole premise is an **exclusive apical clamp**: the target slot is held at `v_teach = −25 mV`, every other slot
at `Er = −70 mV`, re-applied before every step, so only the target slot has `IS > 0`. Measuring `IS` **inside the step**
(i.e. at the moment the BTSP block actually reads `cp_v_apical`, after the bridge's own apical dynamics have run):

```
v_apical[window_i, slot_j] mean, mV          plateau_v_hold = -50.0
  [[185144,  51987,  53063],
   [ 55114, 189600,  54840],
   [ 54793,  54782, 189204]]
```

Two facts, both new:
1. **The apical compartment is at ~1.9e5 mV** — five orders of magnitude outside any physiological range. The apical
   update is `dv = −(v_apical − Er) + R·I_coincidence + g_c(v_soma − v_apical)` with `comp_apical_R = 50.0`, so its fixed
   point is `≈ Er + R·I_coincidence`; once the `ca1→slot` weights grow, `I_coincidence` is large and the "thin high-R
   apical" parks at ~10⁵ mV. **The teaching clamp we write each step (−25 vs −70, a 45 mV difference) is numerically
   irrelevant against it.**
2. **The instructive signal is therefore only 3.5 : 1 selective, not exclusive** (≈185 000 on the diagonal vs ≈53 000
   off-diagonal, i.e. each non-target slot receives **28 %** of the target's write drive, *every step of every window*).

That single number explains the whole boundary. Since `Ẽ` is a per-*presynaptic-cell* quantity shared by all posts, the
realized weight is `w[k→slot_j] = Σ_i Ẽ_i[k] · IS[i,j]`, i.e. `≈ 185·Ẽ_j[k] + 53·(Ẽ_a[k] + Ẽ_b[k])`. With the measured
cross-fact eligibility cosine (~0.65), that is ~97 % the same vector for every j — exactly the measured cross-slot
correlation — and pins own/other at ~1.0 **for any per-source write-side gate whatsoever**. The probe now computes this
prediction in-run: `predicted_own_over_other` from (measured gated eligibility × measured IS matrix) lands in the same
flat 0.75-1.72 band as the realized metric.

**A second, independent break sits underneath it:** `corr(w_realized, w_predicted)` is only ~0.1, while the per-cell
weight spread is large (`CV ≈ 1.5-1.75`) and ~97 % shared across slots. So even the (eligibility × IS) model does not
explain the realized per-cell weights — the `ca1→slot` weight vector is dominated by a **slot-independent structural
factor** (wiring / per-cell synapse count), not by the write at all. Both breaks are *upstream of and independent from*
the eligibility gate.

### What this does NOT say
- It does **not** refute M0. M0's two results stand: the absolute count gate amplifies a peaked code ~10× (isolated
  1.27 → 12.3), and the `--cycles 1 --settle-steps 200` protocol delivers a peaked, mass-balanced, uniformly
  fact-specific code (during-write gated ceiling 2.9-4.0, 3/3 facts, 3 seeds — **reproduced in every run here**, see the
  `M0 VERDICT` lines).
- It does **not** say the gate is wrong. Link 1 is intact at 0.84-0.91; the sustained-vs-transient discrimination is
  proven on real spikes in `tests/test_btsp_win_gate.py`.
- **It says M0's ceiling algebra assumed `w[k→slot_j] ∝ g(count_j[k])`, and the realized write violates that assumption
  by construction** — because the instructive signal it multiplies is a runaway, only-3.5:1-selective plateau current
  rather than the exclusive teaching clamp the protocol believes it is applying. That assumption had never been measured
  in ~25 probes of this arc; it is measured now.

## 5. Verdict + the next method

**M1′ (the `sim/` edit): SHIPPED and BANKED.** Byte-identical off (bit-equal golden), CI-guarded, engages sparsely,
composes correctly onto the eligibility. It is the primitive the research gate specified and it works as specified.

**M1′ (the consolidation GO metric): NO-GO.** `CORE-GATED own/other` ~1.0 at every θ ∈ {4,6,7,8,9,10,12} and every seed
{42,43,44}; 0/3 facts pass at every configuration; predicted 2.9-4.0 not approached. Reported with the full mass triad
throughout: permuted-core and random-CA1 controls collapse to ~1.0 (as they must — there is nothing to control for),
the slot-weight-normalised twin tracks the raw ratio (`slot_mass_ratio` ≤ 1.05, so no winner-slot artifact anywhere in
this arc), and every number is per-fact with a degenerate guard, never a mean.

**THE LAW: this is a verdict on the WRITE'S INSTRUCTIVE SIGNAL, not on the capability, and not on the gate.**

### The lever is IDENTIFIED and already MEASURED (a bonus probe, `--comp-apical-R`, no `sim/` edit)
The apical fixed point is `≈ Er + R·I_coincidence`, so `comp_apical_R` is the direct lever. Sweeping it at seed 42
(everything else identical):

| `comp_apical_R` | v_apical target / non-target (mV) | **IS diag ÷ off-diag** | `corr(w, window_elig)` (link 2) | CA1 core sizes |
|---|---|---|---|---|
| **50.0 (shipped)** | ~188 000 / ~54 000 | **3.47** | ≈ 0 (−0.13 … +0.36) | [15-29] |
| 5.0 | ~445 / ~66 | 4.2 | **0.39-0.68** (θ=0) | **[0, 0, 0]** |
| 1.0 | **+54 / −48 (PHYSIOLOGICAL)** | **53** | — (dw ≈ 0) | **[0, 0, 0]** |

Two things fall out, and both confirm the diagnosis:
- **Lowering R restores exclusivity monotonically** (3.5 → 4.2 → 53) and puts the apical back in a physiological band.
- **It simultaneously repairs link 2**: at R=5 `corr(w, window_elig)` jumps from ~0 to 0.39-0.68 and
  `corr(w_realized, w_predicted)` from ~0.1 to 0.61-0.86 — i.e. **once the apical is not runaway, the write DOES become
  eligibility-driven.** *(Caveat: at R=5 the total write is tiny — `dw ≈ 0.006` — so these correlations are measured on
  a near-silent write and are indicative, not a result.)*

**But it also exposes a deeper, previously invisible dependency:** at R=5 and R=1 the CA1 core **vanishes entirely**
(`core_sizes = [0,0,0]` — no CA1 cell reaches the >10-spikes-per-40-step criterion). `apical_g_couple_to_soma = 5.0`
against a ~1.9e5 mV apical injects ~10⁶ pA into every soma, so **the sustained CA1 code that M0 measured, and that this
entire two-sided-read program is built on, exists only in the runaway-apical operating point.** At a physiological
apical, under the current drives, there is no code at all.

### The next de-risk, precisely specified (cheap, no `sim/` edit)
1. **JOINT retune:** lower `comp_apical_R` into a physiological band AND raise the reinstatement/tag drive (or
   `ca3→ca1` weight) until the CA1 sustained core reappears at comparable sizes. **Two gates, in order, before any
   own/other is quoted: (a) `core_sizes ≥ 10` per fact with the M0 during-write gated ceiling back at ≥ 2.5 on 3/3;
   (b) the measured `IS[i,j]` off-diagonal ≤ 5 % of the diagonal (≥ 20:1).** Report the `IS` matrix and `v_apical`
   every time.
2. Only then re-run this θ sweep. M0 predicts 2.9-4.0 the moment links 2/3 hold; **M1′ is already built, verified and
   in place to test it** — no further `sim/` work is needed for that experiment.
3. If no operating point satisfies both gates, the write protocol itself is the failing method, and the next mechanism
   is a different instructive signal — e.g. a slot-selective plateau driven by a **gated pathway** rather than a
   host-written `cp_v_apical` clamp. That is also the brain-based-only-correct form: a host-written voltage clamp is a
   scaffold, not a neuron.

**Standing caveats added to this arc (both new, both invisible to every metric used before now):**
1. Any consolidation number from this write protocol must be read alongside the `IS[window_i, slot_j]` matrix. A
   protocol whose "exclusive" teaching signal is 3.5:1 cannot produce a selective write.
2. The substrate's CA1 activity in this config is sustained by a runaway apical (`v_apical ~ 1.9e5 mV`,
   `comp_apical_R = 50`). Any claim about "the CA1 code" here is a claim about that operating point.

## Provenance
Build + verification, 2026-07-25. `sim/` edit reviewed by diff (`git diff sim/`); byte-identity established by stashing
the edit and re-running an identical deterministic-numpy harness (md5 of the full weight vector). Raw artifacts:
`research/findings/raw/consol_opsweep_gpu/m1prime/twosided_m1p*_seed{42,43,44}.json` (15 runs = 3 seeds × θ∈{0,6,8,10,12})
and `twosided_m1pR{1,5}*_seed42.json` (the apical-R lever). Runner:
`research/runners/_consol_twosided_generalize_probe.py` (`--btsp-win-theta`, `--btsp-win-hill-n`, `--comp-apical-R`, plus
the write-fidelity / localiser / IS-matrix diagnostics added here). CI guard: `tests/test_btsp_win_gate.py`.
Anti-cheats per `.claude/skills/verify-go/SKILL.md` lens 7 (mass triad) — applied to every ratio reported here.
Committed, NOT pushed.
