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
