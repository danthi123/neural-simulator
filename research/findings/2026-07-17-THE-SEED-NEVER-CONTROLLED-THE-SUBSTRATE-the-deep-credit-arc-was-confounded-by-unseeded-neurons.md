# ⛔ `--seeds` NEVER CONTROLLED THE SUBSTRATE — the deep-credit arc's FULL-vs-FROZEN comparisons were confounded by **unseeded neuron heterogeneity**, and the confound is LARGER than the effect

**2026-07-17. FOUND, MEASURED, FIXED, VERIFIED, and the sweep relaunched. NO `sim/` edit — the bug was in the runner.**
**This is the deepest defect found in the 2026-07-16/17 audit arc, and it was found by chasing a number that refused to reconcile.**

## TL;DR

The on-bridge net builder set **`cfg.actual_seed_used`** — a **reporting field the bridge never reads**. The bridge
seeds neuron heterogeneity from **`cfg.seed`**, which stayed at its default `-1`, so **`cp.random.seed()` was never
called** and the per-neuron firing thresholds came from the **unseeded global RNG**.

⇒ **Every run of this arc had different neurons, even at the same `--seeds`.** Every FULL-vs-FROZEN comparison varied
`train_layers` **AND the substrate**. **The confound (±0.33 in `deep_credit_share`) is ~3× the effect it was measuring
(+0.111).**

## The chain (exact, all verified in code)

```
runner  : cfg.actual_seed_used = int(seed)        # a REPORTING field; the bridge never reads it
bridge.py:2136 : het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed
                 if het_seed >= 0: cp.random.seed(het_seed)
config.py:110  : heterogeneity_seed: int = -1
config.py:34   : seed: int = -1
  => het_seed == -1  =>  the guard is FALSE  =>  cp.random.seed() NEVER CALLED
bridge.py:1508 : cp_neuron_firing_thresholds = cp.random.uniform(...)   # UNSEEDED GLOBAL RNG
```

**The guard is correct. The builder simply never set the field the guard reads.**

## The measurements (not inferences)

| test | result |
|---|---|
| two **fresh processes**, `seed=42` | thresholds **DIFFER** — md5 `55c612a7…` vs `cc6815c6…`; means **−44.48 vs −41.79** |
| four nets built **back-to-back in ONE process**, same seed | **DIFFER from each other, max 18.4 mV** (each `_mk()` advances the global RNG) |
| **after the fix** (`cfg.seed = int(seed)`), two fresh processes | **byte-identical**, md5 `6d44f9a7bd615770` both |
| **after the fix**, 1st vs 4th net in one process | **IDENTICAL** ⇒ `net` and `fnet` finally share the same neurons |

## What it explains — the number that would not reconcile

`deep_credit_share` on **the same seed 42, same task**:

| arm | FULL | FROZEN | share |
|---|---|---|---|
| BANKED (numpy, old code, separate `--freeze-hidden` proc) | 0.852 | 0.667 | **+0.333** |
| CUPY (live sweep, new code, internal `fnet`) | 0.889 | 0.889 | **0.000** |
| NUMPY (backend A/B, new code, internal `fnet`) | 0.778 | 0.926 | **−0.333** |

**A 0.67 swing on one seed.** I spent hours attributing this to the **backend** (numpy vs cupy) — a hypothesis the A/B
**refuted**: numpy-new (0.926) did not match banked-numpy (0.667) either, *both numpy*. The only remaining variable
was **which build position was frozen** — and that turned out to mean **different neurons**. In the `−0.333` run the
frozen reservoir *beat* the trained network, which is not a result; it is a coin flip on threshold draws.

## What this invalidates (stated plainly, including my own work from the same day)

1. **The live 4-arm sweep was STOPPED mid-run.** Its `deep_credit_share` was measuring threshold noise. **~7 h of GPU
   saved from producing a confident, worthless answer.** Its partials are archived as `*.PRE-SEEDFIX-CONFOUNDED.*` —
   **kept, not deleted: they are the evidence of the bug.**
2. **My own "the deep-credit GO is ~80% RESERVOIR" (2026-07-16, ADDENDUM 5) is CONFOUNDED.** It compared a FULL run
   and a FROZEN run in **two separate processes** — i.e. **two different sets of neurons**. The headline's *shape*
   (the frozen-hidden control was never run, and the gate could not distinguish deep credit from a random projection)
   **stands** — that critique is about a missing control, not about the number. **The specific 80/20 split does not.**
3. **The arc's "seed variance" is partly NOT seed variance.** `+0.185 / +0.037` across seeds 42/43 was read as the
   deep-credit effect being seed-dependent. Part of it is **unseeded threshold noise** — the same seed re-run would
   have moved too. **This is why the effect looked "smaller than its own spread."**
4. **The banked "6-seed GO, K=8 0.877" inherits the confound**, on top of already being 3 dev seeds with
   `SIGNAL=False`.
5. **ADDENDUM 6's fit/generalize argument** (FROZEN fits better, generalizes worse ⇒ the +0.111 is real learning) is
   **built on four numbers from confounded pairs and must be re-derived** on the fixed code. Its *reasoning* may
   survive; its *evidence* does not.

## What it does NOT invalidate

- **The Stage-0 task screen.** `stage0_depth_genuineness` is a **rate oracle** (`DendriticMLP`, numpy) — **it never
  touches the bridge**, so no unseeded thresholds. Confirmed empirically: the live GPU run's stage0 lines reproduced
  my CPU pre-registration **to the digit** (seed 42 → l1 0.444; seed 100 → l1 0.407). **The seed-101 degenerate
  exclusion and the 9-seed screen stand.**
- **The D5 settle** (LSTM vs bigram) — a different runner (`_recurrent_lm_ceiling.py`, PyTorch), no `SimulationBridge`.
- **The anchor-claim audit's 10 defects** — documentation defects, independent of this.

## The fix

```python
cfg.actual_seed_used = int(seed)
cfg.seed = int(seed)          # <- the field the bridge ACTUALLY reads (bridge.py:2136)
```

One line. Also seeds `ou_seed`'s fallback (`config.py:124`), so the OU noise is deterministic too. **Pinned by two
tests** (`tests/test_plasticity_inertness.py`, 10/10): the builder must set `cfg.seed`; and the bridge's `het_seed`
guard still requires a non-negative seed (so *"not setting `cfg.seed`" means "not seeding"*).

**Honest note on the fix's cost:** the fixed runs are **not comparable to any banked number** — the banked runs were
never reproducible, so they were never a valid baseline. This is not a regression; it is the first time the arc has
had one.

## The fix is VERIFIED END-TO-END, not just on the thresholds (2026-07-17)

Fixing the thresholds is necessary but not sufficient: **cupy's atomics can make GPU reductions non-deterministic
run-to-run even under perfect seeding**, which would leave residual noise in every comparison. Tested rather than
assumed — same seed, two fresh cupy processes, build → train 2 epochs → evaluate:

| quantity | run 1 | run 2 |
|---|---|---|
| thresholds md5 | `f151e39d1ec89ee6` | `f151e39d1ec89ee6` ✅ |
| post-train `ff_weight_norm` | `100385.3125000000` | **identical to 10 dp** ✅ |
| post-train inherit acc | `0.3333333333` | identical ✅ |

⇒ **the whole pipeline — bridge construction, spiking dynamics, e-prop training, evaluation — is now DETERMINISTIC
run-to-run on cupy.** The seed controls the experiment, end to end, for the first time in this arc. *(Accuracy sits at
chance because this is 2 epochs on 40 examples; the test is determinism, not learning.)*

**Honest scope:** determinism holds **WITHIN a backend, not ACROSS one.** The cupy thresholds (`f151e39d…`) differ
from numpy's (`6d44f9a7…`) at the same seed — different RNG implementations draw different numbers. So a numpy result
and a cupy result are still **not** byte-comparable; each is internally reproducible. **This is exactly why the
earlier cross-backend "reproducibility check" was void** (rule 14) — and it is now a *property of the design*, not an
unknown.

## How it was found — and why the process matters more than the bug

Not by looking for it. By **a number refusing to reconcile**, and refusing to let it go:

1. The sweep's FROZEN (0.889) missed my pre-registered reproducibility check vs banked FROZEN (0.667) — 6 of 27
   examples apart.
2. I hypothesised **backend** (numpy vs cupy) and — crucially — **ran the A/B instead of asserting it**. It **refuted**
   my hypothesis: numpy-new ≠ numpy-banked.
3. That left only **build position**. I had *claimed* two hours earlier that "the sweep's FULL-vs-FROZEN is a clean
   single-variable comparison" because `_mk()` passes `seed=seed` and every RNG in the *runner* is seed-derived.
   **I had checked the runner and never checked the bridge.**
4. A 60-second test — build four nets, hash the thresholds — showed 18.4 mV of drift.

**Every step where I checked, I learned something. Every step where I modelled instead, I was wrong** (the "80%
reservoir" split, the 2-wide/4-wide throughput claim, the backend hypothesis, "the comparison is clean"). **The bug
survived because `--seeds 42` is the most natural thing in the world to assume works.**

**⇒ THE STANDING RULE THIS EARNS: a seed is a HYPOTHESIS until you hash the state it is supposed to control.**
Reproducibility is not a property of passing a seed; it is a property you **measure** — run it twice, hash the
substrate, compare. It costs 60 seconds and it is the foundation every other number in an arc rests on.

---

## ✅ THE ENGINE IS EXONERATED — and this is why a green determinism suite never caught it

**`sim/` is CORRECT. `tests/test_determinism.py` is CORRECT (7/7 pass, 140s). The bug is entirely in how 8 runners
CONSTRUCT their config.**

The passing determinism test seeds the **constructor**:
```python
config = CoreSimConfig(..., seed=42, ...)     # -> cfg.seed = 42 -> guard fires -> cp.random.seed(42) -> deterministic
```
The affected runners build it **bare** and then set the wrong field:
```python
cfg = CoreSimConfig()                          # -> cfg.seed stays -1
cfg.actual_seed_used = int(seed)               # -> a REPORTING field; the guard never fires
```

**⇒ the engine seeds correctly the moment you pass `seed=`. Nothing in `sim/` needs changing.**

**Why the suite missed it — the night's pattern, one level deeper:**
1. **It tests the MAIN path, never a RESEARCH-RUNNER path.** Every `*_deterministic_spikes` test builds its own
   correctly-seeded `CoreSimConfig`. The runner path — 93 files, the entire research corpus — was **never covered**.
2. **`test_explicit_seed_tracked` asserts `runtime_state.actual_seed_used == 12345`** — i.e. it tests that the
   **REPORTING field is set**, which is exactly the field that **controls nothing**. **The test enshrines the very
   confusion that caused the bug**: it makes "the seed is handled" look verified when only the bookkeeping is.
3. **`CLAUDE.md:3653` is technically right and practically misleading:** *"`RuntimeState.actual_seed_used` **tracks**
   the seed used"* — **tracks**, correctly stated. But a runner author reading it naturally sets `actual_seed_used`
   and believes they have seeded. **The doc is accurate; the affordance is a trap.**

⇒ **a green test suite, an accurate doc, and a correct engine — and the arc still ran unseeded for months.** The gap
was not knowledge; it was that **nothing ever asserted the property end-to-end on the path the science actually uses.**
The new tests close it for the e-prop arc (source-guard: the builder must set `cfg.seed`); the other 8 runners need
the same (see BLAST RADIUS + the spawned task).

## Is there a WIDER class of dead config fields? **Audited: NO — and the negative result sharpens the lesson.**

Natural follow-up: if `actual_seed_used` was set-but-inert, how many other `CoreSimConfig` fields do runners set that
the engine ignores? **Audited all 227 fields against `sim/` + `experiment/` + the host: ZERO are set-by-a-runner and
never-read. The config surface is CLEAN.**

*(First pass reported **52** — all FALSE POSITIVES. My regex matched only attribute access (`cfg.field`) while the
engine reads config **heavily via `getattr(cfg, "field", default)`** — a STRING lookup. `bdsp_w_max` and
`enable_d1_d2_asymmetry` were both in that list and both are plainly read (`bridge.py:7300`, `:2738`). **I verified
the classifier before reporting, which is the only reason 52 false accusations did not enter the record.** Same
lesson, fourth time tonight: **the instrument needs checking before its output does.**)*

**⇒ THE NEGATIVE RESULT IS THE POINT, and it sharpens rule 15 considerably:**

> **`actual_seed_used` was never a "set but never read" field. It IS read — `bridge.py` records it, for REPORTING.
> It simply does not CONTROL the RNG.**

So the real class is **narrower and nastier than a dead field**: **a field that IS READ, but only for REPORTING.**

- A static *"does anything read this?"* audit — **the exact audit above** — returns **YES** and hands out **false
  comfort**. It would NOT have caught this bug.
- Grepping for the field's *name* finds hits (in the recorder, the checkpoint writer, the test that asserts it is set).
- **The ONLY thing that catches it is a BEHAVIOURAL PROPERTY TEST**: set the field, change nothing else, and assert
  the thing it claims to govern actually changed (or, for a seed: run twice and assert the substrate is IDENTICAL).

**⇒ rule 15's sharp form: do not ask "is this field read?" — ask "if I change it, does the thing it names change?"**
That is why the fix ships with `TestSubstrateActuallySeeded` (a discriminating pair) rather than a source-guard alone:
a source-guard pins *this* runner; only the property test catches *the next* one.

## BLAST RADIUS — scoped mechanically. **The bug is NOT project-wide: 85 of 93 runners seed correctly. EIGHT do not.**

**Method.** Every file that sets `actual_seed_used` AND constructs a `CoreSimConfig()`; for each, check whether the
**config object** it built ever gets `.seed` or `.heterogeneity_seed` assigned. *(My first pass used a loose regex
`\.seed\s*=\s*seed`, which also matches `self.seed = seed` — a net attribute, not the config — and would have handed
out false OKs. Re-done keyed on the actual `CoreSimConfig()`-bound variable names. **The classifier needed the same
verification as everything else tonight.**)*

**Result: 94 files set `actual_seed_used`; 85 also seed the config ✅; 9 flagged — of which `sim/bridge.py` is a FALSE
POSITIVE (it is the bridge, which READS the field). ⇒ 8 genuinely-unseeded runners.**

| runner | findings citing it |
|---|---|
| **`_gnw_d1_spiking_bdsp_derisk.py`** | **9** ← the D1/BDSP deep-credit arc |
| `snc_pavlovian_probe.py` | 6 |
| `_da_composer_salience_cleanup_derisk.py` | 5 |
| `_homeostatic_spiking_agent_integration.py` | 3 |
| `_homeostatic_spiking_drive_mechanism_derisk.py` | 3 |
| `_batched_onbridge_forward_derisk.py` | 2 |
| `_homeostatic_spiking_reward_plasticity_derisk.py` | 2 |
| `_d1_apical_soma_coupling_probe.py` | 1 |

**`_onbridge_eprop_port_derisk.py` builds ZERO `CoreSimConfig`s** — it inherits `OnBridgeBDSPNet`, so **the fix
already covers the whole e-prop/semantic-inheritance arc.** ✅

## What this does and does NOT mean — do not over-read it

**"Unseeded substrate" ≠ "invalid finding."** It means those runs are **not reproducible** and their same-seed
comparisons carry an **uncontrolled neuron-heterogeneity term**. Whether that **invalidates** a given result depends
entirely on **effect size vs the confound**:

- **FATAL** where the effect is **comparable to or smaller than** the threshold noise — exactly the deep-credit case
  (effect +0.111, confound ±0.33 ⇒ the confound is ~3× the signal, and the verdict flips between runs).
- **PROBABLY SURVIVES** where the effect is **large and structural** — e.g. a lesion that collapses a result to
  chance, or a 1.00-vs-0.24 separation. Threshold jitter does not manufacture a 0.76 gap.

⇒ **The honest action is TRIAGE BY EFFECT SIZE, not a blanket retraction.** The rule: **any claim whose margin is
within ~±0.2 on a bridge-based runner from this list is UNSAFE until re-run seeded.** `_gnw_d1_spiking_bdsp_derisk`
(9 findings, the deep-credit family — precisely the marginal-effect regime) is the priority; the homeostatic/SNc
probes mostly report large structural collapses and are likely safe, but that is a **hypothesis, not a clearance.**

**NOT fixed in this commit** — deliberately. Each of the 8 needs its own `cfg.seed` line placed against its own
variable names and then **verified by the hash test** (two fresh processes → identical thresholds). A blind `sed`
across 8 load-bearing runners at the tail of a long session is exactly how a "fix" becomes the next silent failure.
**Queued as its own task**, with the hash test as the acceptance gate for each.
