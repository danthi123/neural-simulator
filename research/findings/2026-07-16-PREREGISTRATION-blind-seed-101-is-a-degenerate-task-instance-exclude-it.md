# PRE-REGISTRATION (2026-07-16 22:41:23) — blind seed **101 is a degenerate task instance**; exclude it from the FULL-vs-FROZEN gate. Filed **before any sweep result existed.**

**Status: PRE-REGISTERED. At filing time `research/findings/raw/_eprop6_*.json` matched ZERO files** — the 4-arm
sweep was ~16 min into a ~3 h run and had produced no output. Nothing below is post-hoc.
**CPU/numpy throughout; the GPU was untouched (it is running the sweep).**

## Why file this

The in-flight sweep produces the **first blind-seed data** the deep-credit claim has ever had (dev 42/43/44 vs blind
100/101/102). If I inspect the results and *then* drop a seed, that is p-hacking — indistinguishable from motivated
reasoning no matter how good the reason. So the exclusion criterion is fixed **now**, from an independent measurement
of the **task** that never touches FULL or FROZEN.

## The instrument was validated FIRST (it reproduces the banked record exactly)

`stage0_depth_genuineness` is deterministic given the seed + task config, so it can be recomputed on CPU. Rebuilt with
the runner's true defaults (`n_super=12, n_members=8, held_per_super=3, n_prop=2, member_id_dim=3, n_obs=16,
noise=0.02, feature_seed=0`):

| seed | chance | **l0 LINEAR** | l1 shallow | deep_best | depth_separating | check |
|---|---|---|---|---|---|---|
| 42 | 0.333 | 0.370 | 0.444 | 1.000 | True | **reproduces banked** (l1 0.444, dsep True) |
| 43 | 0.333 | 0.259 | 0.370 | 1.000 | True | **reproduces banked** (l1 0.370, dsep True) |
| 44 | 0.333 | 0.333 | 0.111 | 1.000 | True | **reproduces banked** (l1 0.111, dsep True) |
| 100 | 0.333 | 0.296 | 0.407 | 1.000 | True | [BLIND — first ever] |
| **101** | 0.333 | 0.148 | 0.333 | **0.370** | **False** | **[BLIND — DEGENERATE]** |
| 102 | 0.333 | 0.185 | 0.370 | 1.000 | True | [BLIND — first ever] |

The three dev rows match `raw/_epropport/k8_s4{2,3,4}.json` **to the digit** on chance, `l1`, and `depth_separating`.
That is the instrument check that licenses the blind rows. *(A first attempt with GUESSED task defaults produced
`chance=0.111`, `depth_separating=0/6` — a plausible-looking table measuring a different task. It disagreed with the
banked `chance=0.333` and was discarded, not recorded. Rule 3: verify the instrument before trusting its output.)*

## The pre-registered criterion

> **Seed 101 is EXCLUDED from the FULL-vs-FROZEN deep-credit gate**, because its Stage-0 gate fails on its own terms:
> `depth_separating = False`, `deep_best = 0.370` against `chance = 0.333`. **A deep rate ORACLE cannot solve seed
> 101's task instance.** A task no oracle can learn cannot test whether a credit rule learns depth — any FULL-vs-FROZEN
> difference there is noise on an unsolvable instance, in either direction.
>
> **The blind arm is therefore seeds 100 and 102** (both `depth_separating=True`, `deep_best=1.000`).
> **Seed 101 will be REPORTED, never silently dropped**, with this document as its justification.
> **This cuts against me:** it shrinks the blind arm to n=2, which *weakens* whatever the sweep concludes. I am filing
> it anyway, before the data, precisely because that is what makes it legitimate.

## Second, independent result: the LINEAR explanation is **ruled out**

The ladder, now complete (`l0` is computed by `stage0_depth_genuineness:308/316` but **never recorded** by
`run_seed`, which stores only `l1` — so the linear floor had never reached any output file):

```
chance 0.333  ->  l0 LINEAR 0.265  ->  l1 trained-shallow 0.340  ->  FROZEN random-deep 0.778*  ->  FULL learned-deep 0.889*
                  (BELOW chance)       (AT chance)                   (*dev-seed; blind in flight)
```

**A linear readout on the raw input scores 0.265 — *below* chance. A trained one-hidden-layer net scores 0.340 ≈
chance.** Yet a **frozen, untrained** 2-layer random expansion (+ `--pool-k 8` population coding) reaches **0.778**.

⇒ The reservoir's 0.778 is **not** "a linear classifier on the input" — that hypothesis is dead. The random nonlinear
expansion is doing real work on a genuinely depth-required task (Cover's theorem; the 8× population widening is the
kernel). This *strengthens* the reservoir reading of the banked GO rather than explaining it away: the projection is
powerful, and learning added only +0.111 on dev seeds (seed-variable: +0.185 / +0.037).

## What the sweep now asks, exactly

Not "is the task deep?" (**yes**, 5/6 seeds), and not "is it linearly trivial?" (**no**, l0 < chance), but:

> **On seeds 100 and 102 — which nobody tuned against — does LEARNING the hidden layers beat RANDOMLY PROJECTING to
> them?** i.e. is `FULL − FROZEN` > 0 and meaningfully so, off the dev seeds.

**Pre-registered reading of the outcome** (fixed before data, to stop me from reading the result twice):
- `FULL − FROZEN` clearly > 0 on **both** 100 and 102 → the deep-credit share is real, if small; segment (b) proceeds
  with this learner.
- `FULL ≈ FROZEN` on either → the "deep credit" headline is **substantially a reservoir**, and segment (b) must not be
  built on it. The months-scale plan's deep-credit lever needs re-scoping, not re-running.

## Follow-on (cheap, additive, NOT applied to the in-flight run)

`run_seed` should record `stage0_l0` alongside `stage0_l1`. It is one dict key; the value is already computed and
thrown away. **Deliberately not edited while the sweep is in flight** — the running processes have the module loaded,
and I will not perturb a 3 h run for a logging nicety.

---

## ADDENDUM A (2026-07-16 22:47, still **before any sweep result existed**) — the blind arm is n=2, which is too thin. An EXTENDED blind arm, pre-screened on task-validity only.

Excluding 101 leaves the blind arm at **n=2** (100, 102) against a dev effect that was already seed-variable
(+0.185 / +0.037). n=2 will very likely be **inconclusive**, which would waste the whole 3 h run. Fixed cheaply, and
pre-registered here rather than after seeing the outcome.

**The screen (CPU, `research/findings/raw/_eprop6_blindseed_screen.json`).** Candidate seeds 103–116 were scored on
**task validity ONLY** — `depth_separating AND deep_best >= 0.95`. This criterion is computed from the **task and a
rate oracle**; it never touches FULL, FROZEN, e-prop, or the bridge, so **it cannot select for the effect under test.**
Reported in full, pass and fail:

| seed | deep_best | dsep | usable | | seed | deep_best | dsep | usable |
|---|---|---|---|---|---|---|---|---|
| 103 | 1.000 | True | **YES** | | 110 | 1.000 | True | **YES** |
| 104 | 0.963 | True | **YES** | | 111 | 0.704 | False | no |
| 105 | 1.000 | True | **YES** | | 112 | 0.815 | True | no (deep_best < 0.95) |
| 106 | 0.370 | False | no | | 113 | 0.926 | True | no (deep_best < 0.95) |
| 107 | 1.000 | True | **YES** | | 114 | 1.000 | True | **YES** |
| 108 | 1.000 | True | **YES** | | 115 | 1.000 | True | **YES** |
| 109 | 0.815 | True | no (deep_best < 0.95) | | 116 | 1.000 | True | **YES** |

**Usable new blind seeds: `103 104 105 107 108 110 114 115 116` (9/14).**

### The screen's failure rate is itself a result — and it has teeth

**5 of 14 (~36%) candidate seeds are degenerate at this exact task config.** Seed 101 was not bad luck; it is the
base rate. Consequences:

1. **Any unscreened "6-seed" claim at this config runs ~2 degenerate instances** — task instances whose ceiling is
   chance. Those seeds contribute pure noise to a mean and inflate its spread.
2. **Retrospective:** the banked dev effect's seed-variance (+0.185 vs +0.037) may be substantially **task-instance
   variance, not credit-rule variance.** This is a *hypothesis this pre-registration does not test* — but it means
   "the deep-credit share is seed-variable" and "the task is seed-variable" are currently **confounded**, and the
   screened arm below is the first data that separates them.
3. ⇒ **Screening task instances for validity should precede any expensive arm at this config**, exactly as one
   excludes a degenerate stimulus before running subjects.

### Pre-registered plan

> **Blind arm = `100, 102` (already in flight) + the 9 screened seeds = n=11**, all `depth_separating=True`,
> `deep_best >= 0.95`. Launch the extended arm **when the in-flight sweep completes** (it owns the GPU until ~01:30;
> a second concurrent sweep would contend for one 3090). Same config (`--pool-k 8`, `SIM_BACKEND=cupy`, no MPS,
> per-seed checkpointing), FULL and FROZEN.
>
> **The gate is unchanged and was fixed before any data:** deep credit is real iff `FULL − FROZEN` > 0 consistently
> across the screened blind seeds. n=11 makes that answerable; n=2 would not have been.
>
> **Seed 101 and the 5 screened-out seeds are reported, never silently dropped.**
