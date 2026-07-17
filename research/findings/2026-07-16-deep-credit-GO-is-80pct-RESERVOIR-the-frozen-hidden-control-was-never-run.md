# The banked "feedforward spiking deep credit is ALREADY GO (K=8 0.877)" is **80% a fixed random reservoir + a linear readout** — the frozen-hidden control existed in the code, unused, and the GO gate never included it

**Date:** 2026-07-16
**Runner:** `research/runners/_onbridge_eprop_port_derisk.py` (+ new `--freeze-hidden`). CuPy, banked config (`enable_bdsp=True`, `lr=0`, `--pool-k 8` = exactly what produced 0.877), seeds 42/43, ONE variable.
**Verdict:** the GO's NUMBER **reproduces** (FULL 0.889 vs banked 0.877). Its **mechanistic claim does not**: a fixed random spiking reservoir + a trained linear readout accounts for **80% of the margin above chance**; deep credit adds **+0.111**, and it is **seed-variable** (+0.185 / +0.037). The runner's **own aggregate gate returns `SIGNAL=False`** for both arms.

## The table

| seed | FULL | FROZEN (reservoir+readout) | deep-credit contribution | permuted | shuffle-DFA | chance |
|---|---|---|---|---|---|---|
| 42 | 0.852 | 0.667 | **+0.185** | 0.185 | 0.519 | 0.333 |
| 43 | 0.926 | 0.889 | **+0.037** | 0.370 | 0.556 | 0.333 |
| **mean** | **0.889** | **0.778** | **+0.111** | 0.278 | 0.537 | 0.333 |

**Above chance: reservoir +0.444 · deep credit +0.111 ⇒ the reservoir is 80% of the margin.**

- **FULL** = all FF pathways train — byte-identical to the config behind the banked 0.877 (and it reproduces it: 0.889).
- **FROZEN** = hidden FF pathways frozen at init; ONLY the last FF pathway (the host-side linear softmax readout, which `_accum_grad` already SKIPS from the e-prop/DFA rule, `:389 skip_output`) trains. Realized via the runner's OWN `train_layers` hook (`:153`).
- **The instrument was verified before the run** (the day's lesson): `default -> train_layers=None` (all pathways train, byte-identical) vs `--freeze-hidden -> train_layers={2}` (hidden pathways 0,1 skipped at `:361`). Had the default also frozen, both arms would be reservoirs and the verdict meaningless — while looking perfectly plausible.

## Why nobody saw it: the control existed, unused

`trains_the_task` (`:481`) gates on **chance / permuted / shuffle-DFA — NOT ONE is a frozen-hidden baseline**. So a pure reservoir+readout result passes the gate **unchanged**. And `train_layers` — documented in the file itself as *"None => update all FF pathways; a set => update only those (isolation)"* — appears **only** as its definition (`:153`) and its check (`:361`). **Someone built the isolation hook for exactly this purpose and it was never once invoked.** Today is the first time this control has ever been run.

This is the session's recurring shape: **the machinery to check the claim already existed; nothing invoked it.** (Cf. `_ensure_gate_capacity` guarding 7 sites but not the Hebbian one; a requirements file nothing audited; an install doc telling you to run a tool it never told you to install.)

## Two corroborating signals inside FULL's own numbers (independent of the frozen arm)

1. **shuffle-DFA sits at 0.537 against chance 0.333.** A large slice of performance is **credit-INDEPENDENT** — visible without any new control.
2. **The runner's own aggregate `SIGNAL=False`** for BOTH arms: *"HONEST NEGATIVE — the ported e-prop does NOT cleanly train the task on the bridge ... The exact residual: controls not clean."* Per-seed `trains_the_task` passes; the aggregate fails. **This contradicts the banked "6-seed GO" and needs running down.**

## What this does and does not say

**DOES:** the banked headline substantially **overstates** the mechanism. "Feedforward spiking deep credit is ALREADY GO / is NOT a blocker" should read: *the on-bridge e-prop port reaches ~0.89 held-out inheritance, but ~80% of that margin is a fixed random spiking reservoir + a linear readout; the deep-credit contribution is ~20%, seed-variable (+0.037..+0.185), and the runner's own aggregate control gate does not pass.*

**DOES NOT:** say deep credit is nothing. It is real and positive on both seeds. The 5-lens adversarial verify (wf_5473ce0f-8d5) concluded "the deep hidden credit contributes NOTHING (readout-only 0.630 BEATS full 0.556)" — **not supported here**; their arms used a **mixed config** (`pool_k=1` appears 17× alongside `pool_k=8`; epochs 40/120/150/200/250) and their FULL (0.556) does not reproduce the same-config FULL measured here, so their conclusion is scoped to their run. Their **structural** finding — that the gate has no reservoir control — is the one that mattered, and it is vindicated.

**HONEST SCOPE:** **n=2**, against the project's standing 6-seed rule ⇒ **INDICATIVE, not final**. The seed-variability is itself the story (+0.185 vs +0.037): a 6-seed FULL-vs-FROZEN is required before the record is rewritten. But n=2 is already enough to say the headline is **unverified as a deep-credit claim**, because the load-bearing control was never run at all.

## Consequences

- **Segment (b) of the longest pole** was to co-train "the stream cortex + the deep-credit learner". That learner is **80% reservoir**. Co-training it would mostly test co-residence of a *reservoir*, not of a second *learning rule* — which was the entire point of (b) (rule heterogeneity). **(b) is gated on this.**
- **`docs/plans/2026-07-15-months-scale-plan-...`** §4 opens the unification critical path with *"The learning rule (feedforward deep-credit / BDSP, GO)"*. Already corrected once today (BDSP→e-prop+population-coding); now the **GO itself** needs the 80/20 caveat.
- **The 2026-07-15 gate's** *"feedforward spiking deep credit is SOLVED (not a blocker) — the genuine open frontier is RECURRENT off-diagonal"* is **too strong**: the feedforward side is ~20% mechanism, 80% reservoir. The frontier is wider than the record says.

## Retractions this produced (all mine, all caught before entering the record)

1. **"The clamp SUPPRESSED the deep-credit GO (0.877→1.000)"** — DEAD. `--bdsp-wmax` is ONE config field but TWO functional variables: the clamp is global over `cp_connections.data`, which holds BOTH the spiking FF synapses AND the host-side linear readout (measured 864/1536 readout synapses crushed 500→≤6 per forward). Widening the clip freed a **linear classifier**.
2. **"shuffle-DFA refutes the reservoir hypothesis"** — VOID. `train_batch` shuffles the delta LIST (`:383-384`) and feeds the SAME shuffled `d` to BOTH the hidden DFA credit AND the readout's delta rule; it collapses even with the hidden pathways FROZEN. It is a second wrong-label control, a near-duplicate of permuted.
3. **"ff-moved 6.6M is churn / 798 is the genuine update"** — BACKWARDS. `ff_weight_norm` = `sum(|w|)`; `ff_moved` = `|L1_after − L1_before|` — a one-way norm difference, blind to direction. 798 is the ZERO-INIT readout growing to `sum|W|≈798` over 1536 synapses (mean ~0.5 = a textbook softmax solution): evidence **FOR** the hidden layers not learning.

## Next

1. **6-seed FULL-vs-FROZEN** on the banked config (42/43/44/100/101/102) — the standing rule; this is n=2.
2. **Add the frozen-hidden arm to `trains_the_task`** so a reservoir result can never pass the gate again — the durable fix, and the same shape as `tests/test_plasticity_inertness.py`.
3. **Run down the `SIGNAL=False` vs banked "6-seed GO" discrepancy** — same config, opposite aggregate verdict.
4. Then segment (b), on a learner whose deep-credit share is known.

## Artifacts

`research/findings/raw/_eprop_banked_{FULL,FROZEN}.{json,log}` · `--freeze-hidden` in `_onbridge_eprop_port_derisk.py` (default off = byte-identical) · verify workflow `wf_5473ce0f-8d5`.

---

## ⛔ ADDENDUM (same day, decisive) — the banked GO **contradicts its own raw data**. The runner said HONEST NEGATIVE on every seed; the finding recorded "anti-cheat-clean".

Tracing the 0.877 to its source (`research/findings/raw/_epropport/k8_s4{2,3,4}.json` — the runs the claim cites):

| file | SIGNAL | inherit | shuffle-DFA | `shuf_ok` (needs ≤ chance+0.10 = 0.433) |
|---|---|---|---|---|
| `k8_s42.json` | **False** | 0.889 | 0.556 | **False** |
| `k8_s43.json` | **False** | 0.926 | 0.593 | **False** |
| `k8_s44.json` | **False** | 0.815 | 0.630 | **False** |

Those per-seed values are **verbatim** the banked claim's *"inherit 0.877 (per-seed 0.889/0.926/0.815)"* — so this is unambiguously the source. And each run's OWN verdict string reads:

> `HONEST NEGATIVE -- the ported e-prop does NOT cleanly train the task on the bridge`

While `2026-07-15-deep-credit-fresh-class-gate-feedforward-SOLVED-...md` records:

> *"ports to the production Izhikevich bridge with population coding to the LIF ceiling (K=1 0.47 → K=8 0.877 ≈ LIF 0.89), **anti-cheat-clean**, NO `sim/` edit. "The parked 'spikes can't do deep credit / SNR wall' verdict is **COMPREHENSIVELY REFUTED**.""*

**⇒ The 0.877 headline was produced by averaging the `inherit` field out of three runs that each reported `SIGNAL=False` with the anti-cheat control FAILING on every seed. The claim "anti-cheat-clean" is contradicted by the very files it cites.**

### This REFRAMES today's reservoir result

My frozen-hidden control (~80% reservoir) is **not a new discovery** — it is the MECHANISM behind a failure **the runner had already flagged and nobody read**. `shuffle_dfa_chance=False` was saying, on all three seeds: *most of this performance survives destroying the credit signal*. That IS the reservoir, visible in the original JSON, in the original verdict string, from the start.

**My runs reproduce it exactly** — seed 43 = **0.926**, matching `k8_s43` to three decimals, with `SIGNAL=False` and shuffle-DFA 0.537 (vs their 0.556–0.630). So this is not a migration artifact, not a stack difference, not my `to_host` fix. **The numbers were always right; the READING was wrong.**

### Corrected scope of "the gate was missing a control"

The **per-seed** gate (`trains_the_task`) genuinely lacked a reservoir control — fixed today (default-on + CI-guarded). But the **aggregate** gate (`SIGNAL`) already required `shuffle_dfa <= chance + 0.10` and **correctly returned False**. So the tooling was *less* broken than I first said: **it caught this and was overridden by the write-up.** The failure was not primarily instrumentation — it was **reporting a number the instrument had already rejected.**

### What must change

1. **`2026-07-15-deep-credit-fresh-class-gate-feedforward-SOLVED-...md` must be corrected** — "anti-cheat-clean" and "COMPREHENSIVELY REFUTED" are not supported by `k8_s4{2,3,4}.json`.
2. **ROADMAP + `docs/plans/2026-07-15-months-scale-plan-...` §4** open the unification critical path with *"The learning rule (feedforward deep-credit / BDSP, GO)"*. Corrected once today (BDSP→e-prop+population-coding); the **GO itself is now unsupported as stated**.
3. **The 2026-07-15 gate's "feedforward spiking deep credit is SOLVED / not a blocker; the genuine frontier is RECURRENT off-diagonal"** is wrong at its root: the **feedforward** side never passed its own aggregate gate. The frontier is wider than the record says, and the off-diagonal arc was deprioritized partly *because* feedforward was believed solved.
4. **Never average a metric out of a run whose own SIGNAL is False.** A runner that prints `HONEST NEGATIVE` has already done the analysis; lifting its numbers past its verdict is the failure mode this addendum documents.

---

## ⛔ ADDENDUM 2 (2026-07-16) — "6-seed GO" was **THREE dev seeds, all `SIGNAL=False`**. The blind seeds were never run.

The `SIGNAL=False`-vs-banked-"6-seed GO" contradiction is now fully run down, and the claim fails on **four** counts, each independently checkable from the files it cites.

**THE CLAIM:** *"on the production Izhikevich bridge to the LIF ceiling, K=1 0.47 → K=8 0.877, **6-seed GO**"*.

**THE FILES:**

| file | seeds it contains | SIGNAL |
|---|---|---|
| `raw/_epropport/k8_s42.json` | `[42]` | **False** |
| `raw/_epropport/k8_s43.json` | `[43]` | **False** |
| `raw/_epropport/k8_s44.json` | `[44]` | **False** |
| blind seeds 100 / 101 / 102 | **NO FILE ANYWHERE** (grep for `"seed": 10[012]` across `_epropport/` + `_onbridge_eprop*` → empty) | — |

1. **"6-seed" is THREE seeds** — 42/43/44 only.
2. **The BLIND seeds were NEVER RUN.** `CLAUDE.md` states the standing rule verbatim: *"6-seed validation (42/43/44/100/101/102) before any generalization claim."* The dev/blind split is the whole point of the rule — dev seeds are the ones a config was tuned against. This claim generalized off dev seeds alone.
3. **"GO" is `SIGNAL=False`** on every one of the three, each printing *"HONEST NEGATIVE — the ported e-prop does NOT cleanly train the task on the bridge."*
4. **The number is ~80% RESERVOIR** (this doc's main body: FULL 0.889 vs FROZEN 0.778 vs chance 0.333).

**⇒ Four failures in one sentence: the seed COUNT, the seed SPLIT, the VERDICT, and the MECHANISM.** Every one is contradicted by the very files the claim cites — none required new compute to detect, only reading them.

**Note what this does NOT impugn:** the underlying runs are honest. The runner computed the right controls, applied its own gate correctly, and printed `HONEST NEGATIVE` three times. **Every guard worked.** The failure is entirely at the WRITE-UP layer — a negative was read as a positive, three files were called six seeds, and the result propagated into the ROADMAP, the months-scale plan's critical path, and the decision to deprioritize the recurrent off-diagonal arc "because feedforward was solved".

**IN FLIGHT:** the 6-seed FULL-vs-FROZEN (4-wide, cupy-pinned, `[GPU]`-confirmed) covers `FULL/FROZEN × (42,43,44)/(100,101,102)` — i.e. it produces **the first blind-seed data this claim has ever had**, alongside the reservoir control it never had. The gate now carries the reservoir arm DEFAULT-ON + CI-guarded (`tests/test_plasticity_inertness.py`).

**STANDING RULES this produced (both now in the skill's SILENT-FAILURE CLASS):**
- *Never average a metric out of a run whose own `SIGNAL` is False.*
- **NEW:** *"N-seed GO" is a claim about FILES — count them, and check the SPLIT.* "6-seed" that is three dev seeds is not a weaker result; it is a different claim entirely, and the blind half is the half that tests generalization.

---

## ADDENDUM 3 (2026-07-16) — HOW FAR DOES THE CONTAMINATION SPREAD? Audited mechanically: **it does not.** One write-up, not a systemic failure.

Having found the 07-15 headline lifted out of `SIGNAL=False` runs, the obvious next question is whether the arc is
riddled with the same defect. **Answered mechanically rather than by assertion** (rule 1 applied to itself):

**Method.** Scan every `research/findings/raw/**/*.json` for a self-reported verdict; map each negative back to every
`findings/*.md` that cites it; flag any citing finding whose header claims GO / BREAKTHROUGH / CONFIRMED / VALIDATED /
SOLVED. That is the rule-1 violation shape, made grep-able.

**Population.** 1259 raw results carry a verdict field: **`SIGNAL=True` 7, `SIGNAL=False` 65.** The negatives
outnumber the positives ~9:1 — the runners are, overwhelmingly, honest reporters. *That ratio is exactly why lifting
a field out of a negative is the live hazard: negatives are the common case, so a positive-looking sub-field is
almost always sitting inside one.*

**Result: 5 flagged → 5 explained. Zero new contraband.**

| flagged finding | disposition |
|---|---|
| `2026-07-15-...-SOLVED-...` (cites `k8_s42/43`) | **the known defect** — CORRECTION block already appended today |
| `2026-07-16-...-80pct-RESERVOIR...` (this doc) | **false positive by design** — cites the negatives precisely to expose them |
| `AUTONOMOUS_STATE.md` | the running log; cites everything |
| `2026-07-08-...-feedforward-arc-COMPLETE-...` (cites `_rolefiller_binding_seed42_smoke`) | **CLEAN.** Run: *"STAGE-0 BOUNDARY (honest) — role-filler binding … is NOT depth-separating."* Finding: *"single role-filler binding is NOT depth-required"*, verdict *"honest boundary"*. Claim == run. Flagged only because my regex hit "COMPLETE"/"validated but NARROW". |
| `2026-07-08-...-inheritance-is-linear-...` (cites `_semantic_inheritance_ppmi_deep_credit`) | **CLEAN.** Run: *"HONEST STAGE-0 FINDING — NOT depth-required on REAL-PPMI codes."* Finding verdict: *"honest boundary + a clarifying reframe (**NOT a GO** — Stage-1 correctly NOT run on a shallow task)."* Claim == run. |

**Conclusion — and it matters for what to do next.** The deep-credit arc is **not** systemically overclaimed. Its
sibling findings state their negatives plainly and even decline to run Stage-1 when Stage-0 says shallow. **The 07-15
write-up is the single point of failure**, and it failed at the *write-up layer*, downstream of instruments that were
telling the truth in three separate files.

⇒ The remedy is therefore the **mechanical guard, not a re-audit of the science**: rule 1 (never lift a metric out of
a negative run) + rule 10 (an absent flag means DEFAULT, not OFF) + the now-default-on reservoir control. The arc's
*measurements* stand; only the one headline built on top of them does not.

**Honest scope of this audit:** it catches the shape "cites a negative, claims a positive." It cannot catch a finding
that overclaims **without citing its raw file**, or one whose run reports no verdict field at all (1259 do; most
runners do not). So this bounds the *known* contamination — it does not prove the arc globally clean.

---

## ADDENDUM 4 (2026-07-16) — WHY A WELL-DESIGNED GATE STILL LET A RESERVOIR THROUGH: **"depth-required" ≠ "learned-depth-required."** The task is genuinely deep; a RANDOM expansion satisfies it without learning.

Zero-GPU read of the banked runs' own Stage-0 fields (they were recorded all along, at `:528-529`, nested in
`per_seed` — I first suspected they were discarded and **that suspicion was wrong; verified before recording**):

| seed | `stage0_depth_separating` | `stage0_deep_best` | `stage0_l1` (**trained** 1-hidden oracle) | chance |
|---|---|---|---|---|
| 42 | **True** | 1.000 | 0.444 | 0.333 |
| 43 | **True** | 1.000 | 0.370 | 0.333 |
| 44 | **True** | 1.000 | 0.111 | 0.333 |

**Good news first, and it is real: the task IS genuinely depth-required.** A deep rate oracle solves it outright
(1.000) while a **trained** single-hidden-layer oracle sits at chance (0.44 / 0.37 / 0.11 vs 0.333). The Stage-0 gate
was well-built and it was telling the truth. **The arc is not measuring nothing.**

**Now the tension that explains everything.** FROZEN — **two random, untrained hidden layers + a trained readout** —
scored **0.778**. That is a frozen reservoir *beating a trained depth-1 oracle by ~0.35*. Not a contradiction:
it is **Cover's theorem**. A sufficiently wide random nonlinear expansion (here amplified by `--pool-k 8` population
coding = an 8× widening) makes a depth-required task **linearly separable at the readout**. That is the whole
reservoir-computing thesis, and it is working exactly as advertised.

**⇒ THE LESSON, and it generalizes past this runner:** a depth gate of the form *"a deep oracle beats a shallow
oracle"* certifies **the TASK needs more than one layer**. It says **nothing** about whether the hidden layer must be
**LEARNED**. A random projection clears the depth bar by brute force, with zero credit assignment. So:

> **"Depth-required" and "learned-depth-required" are different properties, and only the second one is evidence for a
> credit-assignment rule.** A Stage-0 depth gate cannot separate them **by construction** — only a frozen-hidden arm can.

This is why the missing control was not a nicety. The gate could be perfectly designed, honest, and passing, and a
fixed random projection still walks through it — which is precisely what happened for months. It also retro-explains
the measured split: FULL 0.889 vs FROZEN 0.778 vs chance 0.333 ⇒ the random expansion buys ~0.445 of the ~0.556
margin (~80%), learning buys ~0.111.

**What the in-flight sweep therefore actually asks** — sharpened: not *"is the task deep?"* (answered: yes) but
**"does LEARNING the depth beat RANDOMLY PROJECTING to it, on seeds nobody tuned against?"** The blind arm
(100/101/102) is the whole experiment; on dev seeds the +0.111 was already seed-variable (+0.185 / +0.037).

**Follow-on this suggests (cheap, CPU, not yet run):** record `stage0_l0` — `stage0_depth_genuineness` **computes**
`linear_inherit_heldout` (a no-hidden linear floor, `_semantic_inheritance_deep_credit_derisk.py:308/316`) but
`run_seed` records only `l1`, so **the linear floor never reaches the output.** It costs nothing and completes the
ladder: chance → **l0 linear** → l1 trained-shallow → FROZEN random-deep → FULL learned-deep.
