# gap#4 keystone — the BTSP one-shot place-field TASK, run for the first time: pre-registered NO-GO, with the biological backward-shift signature present and every control collapsing

**2026-07-20.** The experiment the record named on 2026-07-18 (*"(b) a one-shot TASK (association/place-field) the
substrate LEARNS via BTSP"*) and never ran. Item (c) was pursued; (b) had no runner until now.

**Why it matters:** every BTSP result banked so far gates on a WEIGHT CHANGE ("held dw is 8.4× the transient dw").
That is not the gap#4 capability claim. This runner asks the capability question directly — **does the substrate
acquire a BEHAVIOUR from ONE experience?**

## Verdict — NO-GO on the pre-registered gate

**Pre-registered before any result existed:** `field_acc ≥ 0.80` (≥24/30 instances) AND blind seeds passing on their
own, where `hit` = peak firing bin within ±2 of the plateau bin `b`; chance = 5/20 = 0.25.

**Result (dev seeds 42/43/44, 5 instances each, 200 ms/bin, full timing):**

| arm | mean `field_acc` | what it ablates | reading |
|---|---|---|---|
| **MAIN** | **0.467** | — | ~1.9× chance, **below the 0.80 bar** |
| **C1 frozen** (`btsp_learning_rate=0`) | **0.000** | the learning | learning is **load-bearing** (not a reservoir artifact) |
| **C3 no-plateau moat** | **0.000** | the instructive signal | plateau **load-bearing**; `dw` exactly 0 |
| C2 mis-targeted plateau | 0.267 | *where* the plateau points | ≈chance — targeting matters |
| C2b random plateau bin | 0.200 | which pre-pattern it pairs with | ≈chance |
| **C10 transient** (`bistable=False`) | **0.200** | *duration*, rule fixed | **the behavioral-timescale property is load-bearing** |

**Every control behaves correctly.** The mechanism is real and each ingredient is necessary; what falls short is the
*task performance*, honestly, against a bar fixed in advance.

## The scientifically interesting part — the field is BACKWARD-SHIFTED (the biological signature)

Signed circular offset of the peak bin relative to the plateau bin `b`:

| arm | mean offset | median |
|---|---|---|
| **MAIN** | **−2.08** | −1.0 |
| C10 transient | −3.80 | −4.0 |

**The field forms BEHIND the plateau** — which is exactly Bittner & Magee's BTSP result (a plateau creates a field at
the location that *preceded* it, via the seconds-long eligibility of inputs already active when the plateau arrives).
The substrate reproduces the signature, and the transient arm shifts it further back, consistent with plateau duration
shaping the window.

## ⚠️ Honest handling of a mis-specified criterion (NOT re-scored here)

My `hit` window was centered **on** `b` (±2), while the mechanism's own prediction — and biology's — is a field
**backward** of `b`. That is a genuine mis-specification of the metric, evidenced by the offset distribution above.

**I am NOT re-scoring this run against a corrected window.** Changing the criterion after seeing results is
goalpost-moving, and this project has already been burned by post-hoc favourable readings. For the record, the
corrected backward window `−5 ≤ (peak−b) ≤ +1` would give ≈0.69 on these same dev seeds — **still below 0.80**, so the
NO-GO does not turn on the criterion either way. Any corrected criterion must be **pre-registered and validated on the
blind seeds**, which have not been run.

## Instrument caveats recorded (rather than left clean-looking)

1. **C12 caught a real bug in my own runner before any science ran.** `cp_bdsp_apical_drive` starts as `None` and must
   be **assigned**, not written in place; my `is not None` guard silently skipped every plateau (`dw=0`, apical stuck
   at rest). Without the flag-engagement smoke, every arm would have read chance and this would have been written up as
   "gap#4 BTSP one-shot task NEGATIVE" **from an experiment that never ran** — the exact `--soma-g` failure that cost
   this project 7 runs. After the fix: `dw` 4160 vs 0 off; apical −24.15 held vs −65.00 transient.
2. **The C12 bistability check is partly mis-framed.** It compares `v_apical` at *lap end*, which at full timing reads
   −77.28 (post-release hyperpolarization) vs −65.00 — a difference, but not the *latch* it claims to verify. The
   load-bearing half (`enable_btsp` engagement) is sound, and C10 is the real test of what bistability contributes.
3. **The scoping caught a design trap.** `bdsp_apical_bistable=True` latches the apical indefinitely; without an
   explicit release pulse the field spans the whole track, localization is untestable **by construction**, and every
   arm reads the same number. The release pulse is mandatory and is implemented.
4. **C9 respected:** `dw` is reported in a separate `mechanism` block and is explicitly NOT the gate. Note `dw` is
   ~4000 in MAIN, C2, C2b **and** C10 — i.e. **large weight change with chance-level behaviour in three of them**,
   which is precisely why a dw gate would have been misleading here.

## Where this leaves gap#4

Unchanged in substance: **local one-shot plateau-gated credit is real on the substrate** (and now shows the biological
backward-shift), but **the substrate does not yet learn this behaviour to a usable bar**. The capability is not
claimed. Per the standing law, the METHOD is banked, not the capability.

**Named next levers (in order):** (a) raise field reliability — the fields are ~1 bin wide and the CA1 is silent
without the plateau, so the read-out is near-binary; more CA1 neurons and a sub-threshold baseline drive would let a
graded field form; (b) pre-register the backward-window criterion and validate on blind seeds; (c) sweep
`plateau_hold_ms` against `btsp_elig_tau_ms` — the hold currently makes the field partly forward-dominant, opposing the
backward eligibility. **Do not** report any of these as a result without the full C1-C12 table and a separate blind-seed
block.

---

# ⛔ FINAL VERDICT (corrected pre-registered metric, dev + BLIND): decisive NO-GO — and it vindicates C9

After the first run, the metric was corrected and **RE-PRE-REGISTERED before the blind seeds were touched**:
score the **DELTA map** (post − pre; the baseline map is NOT flat — per-pool threshold heterogeneity makes some bins
drive CA1 harder for free, flatness 7-9) against the **BACKWARD window** (−5..+1; BTSP forms the field BEHIND the
plateau). Note this correction made the test **HARDER**, not easier: the window spans 7/20 bins, so chance rose
**0.25 → 0.35** and the 0.80 bar became 2.3× chance.

## The result

| arm | DEV (42/43/44 — metric developed here) | **BLIND (100/101/102 — never touched)** |
|---|---|---|
| **MAIN** | 0.80 / 0.40 / 0.60 → **0.600** | 0.00 / 0.00 / 0.40 → **0.133** |
| C1 frozen | 0.000 | 0.000 |
| C3 no-plateau moat | 0.000 | 0.000 |
| C2 mis-targeted | 0.20–0.40 | 0.000 |
| C2b random bin | 0.20–0.40 | 0.00 / 0.00 / **0.40** |
| C10 transient | 0.20–0.60 | 0.00 / 0.00 / 0.40 |

**VERDICT: NO-GO.** Blind mean **0.133 — BELOW chance (0.35)**.

**The dev result did not transfer at all** (0.600 → 0.133). This is the exact failure mode this arc has already been
burned by twice (the record's own example: dev `3.31±0.74` → blind `4.56±1.95`). Had I reported dev seeds as "6-seed"
— the documented past error — this would have entered the record as a GO.

Two further facts kill any salvage reading:
1. **On 2 of 3 blind seeds NO FIELD FORMS AT ALL** (`field_acc` 0.00 with `width` 0.0 — the delta map is empty).
2. **On the one blind seed with signal (102), MAIN 0.40 EQUALS its own random-plateau control C2b 0.40** — and its
   transient arm too. Indistinguishable from a plateau delivered at a random bin.

## 🔴 This is the decisive vindication of C9 (dw is NOT the gate)

`dw ≈ 3949 / 4049 / 4064` on the blind MAIN arms — **large, healthy weight change — while `field_acc = 0.00`.**

Every BTSP result banked in this project gates on a weight change ("held dw is 8.4× transient dw"). **This experiment
gated on BEHAVIOUR, and the same substrate that produces a big dw produces no learned behaviour.** A dw gate would
have called this a GO. That is not a hypothetical: it is what the prior results did.

## What this means for gap#4 — the honest picture is now worse, and clearer

- **Local one-shot plateau-gated credit MOVES WEIGHT on the real substrate** — that part of the 2026-07-18 6-seed GO
  stands, and its controls (frozen 0.000, moat 0.000) are clean here too.
- **It does NOT produce a reliable learned BEHAVIOUR.** The capability gap#4 actually claims — *a substrate that
  learns from experience* — is **not met at the one-shot-task level**, on blind seeds, below chance.
- ⇒ gap#4 remains **OPEN**, and the earlier board correction ("FULLY RESOLVED is a scope redefinition") is if anything
  understated: the one on-substrate result that looked closest to a learning capability does not survive a
  behaviour-level test on unseen seeds.

## Named next levers (none of them "tune it")

The failure is **field formation reliability**, not credit direction. On 2/3 blind seeds the delta map is EMPTY —
the CA1 never changes its firing at all. Before any further task work:
(a) **diagnose why no field forms** — is the potentiated weight insufficient to change firing (a read-out threshold
    problem), or is the eligibility not overlapping the plateau on those seeds (a timing problem)? These predict
    different fixes and are cheaply separable by probing the post-induction weight map directly.
(b) the CA1 is **silent at baseline** (W0 ≤ 2 gives zero firing) so the read-out is near-binary; a graded field needs
    a responsive baseline — but raising W0 introduces the per-pool inhomogeneity confound above, so this needs the
    delta metric (now in place) **plus** pool-level normalization.
(c) **Do not** report any future version without dev/blind separated. This run is the third instance in this project
    where dev-only numbers would have produced a false GO.

---

# 🔬 DIAGNOSTIC (run immediately after the NO-GO): the credit is CORRECT; the READ-OUT is the blocker

The NO-GO's failure mode was ambiguous between two hypotheses that predict **different fixes**, so rather than tune,
I probed the **post-induction WEIGHT MAP** directly (plateau at bin `b=12`, mean `pos_k → ca1` weight per bin):

| seed | weight-map peak bin | **offset from plateau** | `dw_max` | bins potentiated | FIRING Δ | reading |
|---|---|---|---|---|---|---|
| **100** (blind, scored 0.00) | 11 | **−1** | 4.164 | 20/20 | **0.00000** | **H1 read-out** |
| **101** (blind, scored 0.00) | 11 | **−1** | 4.238 | 20/20 | **0.00000** | **H1 read-out** |
| 42 (dev, scored 0.80) | 11 | **−1** | 4.117 | 20/20 | 0.01688 | field formed |

## What this establishes

**On EVERY seed — including both blind seeds that scored zero — BTSP potentiates the CORRECT BACKWARD BIN
(offset −1 from the plateau), with essentially identical `dw_max` (~4.1-4.2).** The credit assignment is working, is
correctly targeted, is biologically shaped (backward-shifted, Bittner-Magee), and is **stable across seeds**.

**H1 CONFIRMED, H2 REFUTED.** The failure is NOT that eligibility missed the plateau (timing) — the weights moved to
exactly the right place on all three seeds. The failure is that **the potentiated drive remains sub-threshold**, so
CA1 firing does not change and the behaviour cannot be expressed. On 2/3 seeds the neuron simply never crosses
threshold; on seed 42 it barely does (Δ = 0.017).

**This reframes the NO-GO.** The correct statement is **not** "the biological rule fails to assign credit" — it
demonstrably assigns credit correctly, on every seed. It is: **the read-out cannot express the credit that was
correctly assigned.** Those are very different verdicts with very different next steps, and the earlier write-up could
not distinguish them.

## Also revealed: the potentiation is NOT selective

`n_bins_potentiated = 20/20` on every seed — **all** bins potentiate, with a peak at −1. So the weight map has correct
*structure* (a backward-shifted maximum) but poor *contrast*. A near-uniform potentiation with a small peak is exactly
what a sub-threshold read-out cannot turn into a localized field. This is consistent with `Etilde_pre` being non-zero
for every pool (each pool fires during its own bin, and the 1000 ms eligibility spans several bins), so every synapse
gets some potentiation.

## Corrected next levers (now well-posed)

1. **Read-out sensitivity is the binding constraint** — the CA1 must be responsive enough to express a graded field.
   The obstacle already measured: `W0 ≤ 2.0` gives a *silent* cell, `W0 ≥ 3.0` gives a *non-flat baseline* (per-pool
   threshold heterogeneity, flatness 7-9). The delta metric (already implemented) neutralizes the baseline
   inhomogeneity; combined with a larger CA1 population this is the direct fix.
2. **Contrast, not just magnitude** — near-uniform potentiation needs either heterosynaptic depression
   (`btsp_hetero_dep`, currently 0.0 and previously burned once) or a sparser eligibility so non-coincident pools do
   not potentiate. This is the mechanism-level question worth a research gate, not a sweep.
3. The credit-assignment half of gap#4 **does not need further work at this scale** — it is correct and stable. That
   is a genuine, if narrow, positive result recovered from a decisive NO-GO.

---

# 🔬 LEVER-1 TEST (read-out sensitivity, DEV ONLY) — does NOT rescue it, and reveals the real structure

Delta scoring neutralizes the baseline-inhomogeneity confound, so W0 could finally be raised. Dev seeds 42/43/44,
plateau at `b=12`:

| W0 / CA1_N | `pre_max` | `delta_max` | peak offsets (per seed) | in window (−5..+1) |
|---|---|---|---|---|
| 0.6 / 8 (original) | 0.00000 | 0.02063 | **[−4, −6, −4]** | 2/3 |
| 3.0 / 32 | 0.03938 | 0.08458 | [3, −6, −4] | 1/3 |
| 4.0 / 32 | 0.10354 | 0.02604 | **[−4, −6, −4]** | 2/3 |
| **5.0 / 32** | 0.12646 | **0.00000** | — | **0/3 — no field at all** |
| **6.0 / 32** | 0.12958 | **0.00000** | — | **0/3 — no field at all** |
| 4.0 / 64 | 0.19417 | 0.04979 | **[−4, −6, −4]** | 2/3 |

**Lever 1 does not rescue the task**, and it has a hard ceiling: at `W0 ≥ 5` the delta **vanishes entirely** — once the
cell fires appreciably at baseline, potentiation no longer increases its firing (saturation). So the operating window
between "silent" and "saturated" is narrow, and inside it the hit rate stays ~2/3.

## The real structure: the field forms at a lag ≈ the eligibility time constant

The peak offsets are **strikingly consistent at −4 to −6 bins across every read-out configuration and seed**. At
200 ms/bin that is **800–1200 ms behind the plateau — i.e. ≈ `btsp_elig_tau_ms = 1000 ms` almost exactly.**

⇒ **The one-shot field forms at a reproducible backward LAG set by the eligibility time constant.** That is a real,
mechanistically coherent characterization of what this rule does on this substrate, and it is consistent across seeds
that scored 0.00 on the task.

## ⛔ I am NOT re-scoring against a lag-centered window, and the NO-GO STANDS

A window centered on −4..−6 would raise the score. **I am not doing that.** I have already mis-centered this metric
twice (first on `b`, then −5..+1), and re-centering a third time — now with full knowledge of where the peaks land —
is precisely the goalpost-moving this project has been burned by. The pre-registered verdict (**blind 0.133, NO-GO**)
stands as the task result.

**What a legitimate future test looks like:** derive the window from `btsp_elig_tau_ms` **a priori** (the lag is
predicted by the rule's own time constant, not fitted to the data), pre-register it, and validate on seeds never used
in any of the above — 42/43/44 and 100/101/102 are all now contaminated for this purpose.

## Net state of gap#4 after this arc

- **Credit assignment: WORKS.** Correct backward bin, stable magnitude, on every seed including task failures.
- **Field lag: CHARACTERIZED.** ≈ `τ_elig`, reproducible across read-outs and seeds.
- **Behaviour: NOT demonstrated.** Blind 0.133; the read-out expresses the credit only weakly and only in a narrow
  non-saturated band; potentiation is non-selective (20/20 bins).
- **The binding constraint is CONTRAST, not sensitivity** — lever 1 is now tested and bounded. The open mechanism
  question is how to make potentiation *selective* (heterosynaptic depression, sparser eligibility), which is a
  research-gate question, not a sweep.

---

# 📐 QUANTITATIVE: the weight-map CONTRAST is only ~1.6×, which analytically explains the silent/saturated bind

Measured the actual post-induction weight map (mean `pos_k → ca1` weight per bin, plateau at `b=12`):

| seed | `w_pre` | `w_post` mean | peak | min | **contrast (peak/mean)** |
|---|---|---|---|---|---|
| 42 (dev, passed) | 0.600 | 2.918 | 4.707 | 0.600 | **1.613** |
| 100 (blind, scored 0.00) | 0.600 | 2.861 | 4.755 | 0.600 | **1.662** |
| 101 (blind, scored 0.00) | 0.600 | 2.914 | 4.830 | 0.600 | **1.658** |

**BTSP raises the PEDESTAL ~5× (0.600 → ~2.9) while the peak sits only ~1.6× above that pedestal** — and this is
near-identical on the seeds that scored 0.00 and the one that scored 0.80, confirming again that the *weight* outcome
is stable and the *behaviour* difference is downstream noise.

## This analytically explains the empirically-measured silent/saturated bind

A threshold read-out must separate peak (~4.7) from mean (~2.9):
- put the threshold **above ~4.7** → nothing fires → the cell is **silent** (observed at `W0 ≤ 2`, delta ~0)
- put it **below ~2.9** → every bin drives the cell → **no localization**, and once the cell already fires at baseline,
  potentiation adds nothing → **delta vanishes** (observed at `W0 ≥ 5`, delta exactly 0.000)
- the usable band between them is narrow, which is precisely the ~2/3 hit rate the read-out sweep plateaued at.

⇒ **The read-out was never the fixable half.** With contrast 1.6× on a 5× pedestal, *no* threshold setting localizes
this map. Lever 1 is not merely bounded — it is **analytically excluded**.

## What this predicts the fix must be

Localization requires **lowering the pedestal**, i.e. DEPRESSION of the non-coincident inputs, not more read-out
sensitivity. Two candidates, both to be judged by the research gate rather than assumed:
- **Bidirectional BTSP** (Milstein et al. 2021): BTSP *depresses* already-strong/non-coincident synapses — this is the
  mechanism that would flatten the pedestal while preserving the peak, and it is the biologically-attested form.
- **Heterosynaptic depression** (`btsp_hetero_dep`, currently 0.0) — **but this project already burned a competition
  arm once** ("erodes within-assembly", 2026-07-18), so it must not be re-run naively.

Recorded as a prediction *before* the gate reports, so the gate's recommendation can be scored against it.

---

# ⛔ RETRACTION #2: the post-confound-fix "6-seed GO (1.00)" was **n=1 repeated six times**

The research gate identified the dominant confound (`cfg.num_traits=5` dealing five Izhikevich cell types, rheobase
42-306 pA, into CA1 and the position pools -> 2.0-2.5x drive spread -> ~7-10x rate spread, versus BTSP's own 1.5x
weight contrast). Setting `cfg.num_traits = 1` collapsed rheobase to a single 51.4 pA on every seed (**verified**), and
the re-run reported **`field_acc = 1.00` on all six seeds, VERDICT: GO**.

**That GO is withdrawn.** Three independent tells:

1. **The numbers were byte-identical across seeds** — `field_acc=1.00`, `width=8.4`, **`dw=4041`** on all six. Real
   seeds do not agree to four significant figures.
2. **Direct test: the seeds are FUNCTIONALLY IDENTICAL.** Threshold arrays differ
   (`b1e0ae470f` vs `7e7048f27f`) but the FIRING and WEIGHT hashes are bit-identical (`cb415e05b8` / `8e5090d9a3`,
   `w_sum` 4760.284 on both). `cp_neuron_firing_thresholds` varies but **the Izhikevich path never reads it**. With
   `num_traits=1` and every noise source disabled, there was **no functional seed variation left** — the six-seed
   validation was vacuous.
3. **C7 fails anyway:** MAIN `width = 8.4 > 8` (the spec's track-spanning bound), and **C10-transient also scored
   1.00**, i.e. the behavioral-timescale bistability was NOT load-bearing in that configuration — which is the one
   property that distinguishes BTSP from a coincidence rule.

**The runner's own verdict logic was at fault too:** C7 was specified in the design but never wired into the GO
condition, so the runner printed GO while its own width guard was failing. Fixing the *analysis* is not enough if the
*gate* does not enforce it.

## What the fix-of-the-fix is

| cause | fix | status |
|---|---|---|
| cell-type lottery (gate cause #1) | `cfg.num_traits = 1` | done, verified (rheobase spread 1.000) |
| **no functional seed variation** (introduced by that fix) | `weight_jitter = 0.15` | done — **the ONLY knob that works**: `enable_parameter_heterogeneity` is *also* gated behind `num_traits>1`, so it changes nothing (measured). The DELTA metric already cancels the baseline inhomogeneity jitter introduces |
| `w_max` saturation (gate cause #2) | dose `eta` 0.02 -> **0.005** | done — peak sits at 0.65 of the clip instead of 0.98; **contrast 1.647 -> 2.003** |
| no depression arm (gate cause #3) | thresholded heterosynaptic depression | **OPEN** — contrast 2.0 is still below Milstein's 2.5x-plus-sub-baseline-flanks |

Re-running the same pre-registered gate on the now-valid instrument (confound removed **and** seeds genuinely
differing **and** dose off the clip). **No result from before this point should be cited.**
