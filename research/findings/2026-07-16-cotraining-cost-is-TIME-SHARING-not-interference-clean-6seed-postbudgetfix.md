# Co-training the stream cortex: the clean post-budget-fix 6-seed table (93.2% retained, 5/6 GO) — and the cost DECOMPOSED: ~97% TIME-SHARING, ~3% interference (sign-varying)

**Date:** 2026-07-16
**Runner:** `research/runners/_cotrain_stream_cortex_isolation_derisk.py` (+ `_cotrain_iso_aggregate.py`). 6-seed 42/43/44/100/101/102, `--max-windows 12000` PER LEARNER, CuPy. **NO `sim/` edit in the de-risk** (one unrelated `sim/` BUG FIX it exposed — see below).
**Verdict:** **GO 5/6, and the residual is now EXPLAINED.** Supersedes the banked `_cotrain_iso_6seed.json` table (confounded) and CORRECTS the banked "homeostatic threshold drift is the leading cause" attribution.

## 1. The clean table (retires the confounded one)

The banked 6-seed file was **pre-budget-fix**: it records `cotrain=16000` AND `sepA=sepB=16000`, arithmetically impossible under the per-learner cap (`e049aaec`) — it used the OLD TOTAL cap, so each co-trained learner got HALF its baseline's reinforcement. Re-run at a matched per-learner budget:

| seed | dA_deploy | dB_deploy | co A/B | sep A/B | shared A/B | GO |
|---|---|---|---|---|---|---|
| 42 | -0.0425 | -0.0280 | 0.659/0.650 | 0.702/0.678 | 0.543/0.524 | GO |
| 43 | **-0.0892** | -0.0442 | 0.570/0.665 | 0.660/0.709 | 0.507/0.521 | **no** |
| 44 | -0.0535 | -0.0479 | 0.616/0.636 | 0.670/0.684 | 0.521/0.486 | GO |
| 100 | -0.0649 | -0.0292 | 0.616/0.720 | 0.681/0.749 | 0.544/0.550 | GO |
| 101 | -0.0448 | -0.0175 | 0.669/0.650 | 0.714/0.667 | 0.460/0.526 | GO |
| 102 | -0.0523 | -0.0452 | 0.652/0.628 | 0.705/0.674 | 0.458/0.516 | GO |

**MEAN dA -0.0579 · MEAN dB -0.0353 · GO 5/6 · RETAINED FIDELITY 93.2% mean (86.5% min) · shared +control degrades 6/6.**

⇒ **the banked "~90-95% of separate-bridge fidelity" SURVIVES the budget fix** (93.2%). The confound did NOT inflate the headline. Seed 43 is the sole miss, on the strict -0.08 dA gate.

**Cross-stack note (a caveat largely DISCHARGED):** the pre-migration controls were computed on the old Windows box. The homeostasis-ON arm reproduces them closely on the new Linux/CUDA stack in BOTH deltas and absolutes — seed 42 dA -0.042 vs banked -0.0429; seed 43 -0.089 vs -0.0910; absolutes within ~0.5-2%. The banked numbers are firmer than "soft".

## 2. The decomposition — the actual scientific content

The apparent residual was chased through three hypotheses. The first two were the banked leads; the third came from READING the window loop (the skill's a0 step) after the second looked weak:

- **shared Hebbian decay — REFUTED** (banked; `--hebbian-decay 0` leaves the gap unchanged).
- **homeostatic threshold drift — PARTIAL, not the cause** (`--homeostasis 0` closes 14%/30%/18% on seeds 42/43/44; mean 21%). And it CANNOT be a cross-talk channel at all: `sim/kernels.py:352-358` is strictly elementwise per-neuron with no cross-neuron reduction, so it can only MEDIATE an effect, never couple B→A. The banked "leading deferred cause" attribution is **corrected**.
- **THE CONTROL WAS NOT TIMING-MATCHED — the actual mechanism.** In `separateA` the B-windows are SKIPPED ENTIRELY (no steps run), so A's windows are BACK-TO-BACK; in `cotrain` every A-window is separated by `window_steps` of B-window during which A decays with zero external input (`present()` zeroes the whole input array then steps the WHOLE bridge). A co-trained learner starts each window from a COLDER state than its own baseline ever does. This explains BOTH prior refutations at once: not weight decay (so `--hebbian-decay 0` couldn't touch it), not threshold drift (so `--homeostasis 0` can't either).

`--idle-match 1` runs the same idle steps in the separate arm (zero input, nothing presented/counted/budgeted), isolating *B's presence* from *B's time*:

| seed | dA_deploy | dA_leak | dB_deploy | dB_leak | leak as % of cost |
|---|---|---|---|---|---|
| 42 | -0.0425 | -0.0006 | -0.0280 | +0.0016 | 1.4% |
| 43 | -0.0892 | +0.0011 | -0.0442 | -0.0009 | 1.2% |
| 44 | -0.0535 | +0.0013 | -0.0479 | +0.0023 | 2.4% |

**mean |leak| 0.0013 vs mean |deploy cost| 0.0509 ⇒ TIME-SHARING ≈ 97%, interference ≈ 3% and SIGN-VARYING** (-0.0006, +0.0011, +0.0013, +0.0016, -0.0009, +0.0023).

### The honest framing (an EARLIER version of this finding was RETRACTED)

I first wrote "the residual is a CONTROL ARTIFACT / cross-talk is EXACTLY ZERO / it's 100% not 90-95%". A 5-lens adversarial-verify Workflow (wf_3a281795-dc7), run on my own claim BEFORE it hardened, refuted that framing:

> **"'TIMING ARTIFACT, NOT INTERFERENCE' IS A CATEGORY ERROR.** The gap between A-windows is not an artifact of the harness; it is a CONSTITUTIVE property of time-shared co-training... The -5..-9% is therefore a REAL cost of co-residence. idle-match does not eliminate it; it LOCALIZES its mechanism... **Localizing a cost is not removing it.**"

Adopted. **`--idle-match` is a MECHANISM DECOMPOSITION, not a baseline, and must NOT feed the GO gate** — the de-risk's own gate specifies the SEPARATE-BRIDGE baseline, and on its own bridge a learner never idles; the idle-matched arm is *"A time-shared with a ghost"*, a system nobody would deploy. The headline stays `dA_deploy`; `dA_leak` is reported beside it. The word "artifact" is deleted.

**Two of my own errors, recorded because they are the methodological content:**
1. **"exactly zero / bit-identical" was an INSTRUMENT artifact.** `corrA/corrB` are stored `round(...,4)` and the deltas were computed FROM the rounded values ⇒ dA/dB quantized to 1e-4 ⇒ the question was unfalsifiable by construction. FIXED (`corrA_raw`/`dA_raw`).
2. **A refutation needs its instrument verified exactly as much as a confirmation does.** I declared a structural-plasticity hypothesis REFUTED using that same broken readout ("TRUE-ZERO 0/6") — a void test that could not resolve a 5e-5 effect. Re-tested at full precision, it IS genuinely refuted (struct-plast OFF leaves the leak unchanged: -4.7e-5/+1.5e-5/+7.8e-5) — but the first verdict was luck, not evidence.

### Where the adversarial verify is itself WRONG (falsified by data)

The skeptics ruled exact-zero a **theorem** (disjoint regions + elementwise ops ⇒ coupling impossible). **The data refutes this.** The leak is reproducible, seed-specific, and **SCALES WITH STEPS**: ~5e-5 at the 800-step smoke → **~1.3e-3 at the 24000-step real config** (~30× steps, ~25× leak). Noise does not scale with step count; a per-step coupling channel does. The skeptics did not examine **`_prev_any`** (`sim/bridge.py:5963`: `bool(self.cp_prev_firing_states.any())` — a GLOBAL reduction over EVERY neuron, gating FIVE per-step blocks at 6013/6155/6421/6476/6575). When B fires those blocks RUN for the whole bridge; in the idle arm nothing fires and they are SKIPPED. That is a real global coupling channel between nominally disjoint slices, and the leading candidate for the measured leak. **⇒ "disjoint regions are perfectly isolated" is FALSE — but the leak is ~3% of the cost, so the practical conclusion is unchanged.**

## 3. Why this matters for the LONGEST POLE

`docs/plans/2026-07-15-months-scale-plan-...` line 46 names the longest pole: *"co-training the learning pieces (stream cortex + deep-credit + long-range learner) WITHOUT cross-talk at scale."* This segment answers it for the stream×stream case:

**Co-training's cost is a FIXED PRICE PER ADDED LEARNER (time-slicing), not COMPOUNDING INTERFERENCE.** Interference would worsen superlinearly with each learner added — a wall. A time-slicing cost is predictable and budgetable: each learner simply gets fewer contiguous windows. **⇒ co-training scales.**

## 4. HONEST SCOPE — what this does NOT show

- **The rule is held CONSTANT.** Both learners use `global rate-Hebbian ON (the shared rule)`. This tests SPATIAL isolation with ONE rule. It says **nothing** about two DIFFERENT learning rules co-residing — which is precisely what the longest pole's remaining segments require. That is segment (b).
- **The regions are DISJOINT.** Isolation is near-free largely *because* of that. A realistic one-brain has learners SHARING substrate — the `shared` positive control, which genuinely degrades (6/6).
- **Seed 43 fails the strict gate** (dA -0.0892). Reported, not smoothed.

## 5. A `sim/` BUG this de-risk exposed (fixed) — arguably worth more than the de-risk

`sim/bridge.py:7041` used the RAW `cp_plasticity_rate_gain` in the Hebbian weight-decay. Structural plasticity (`enable_structural_plasticity` defaults **True**, `config.py:677`) FORMS synapses (≥1 per update, activity-biased, sampled from the WHOLE bridge) growing `cp_connections.nnz` WITHOUT growing the gate arrays → observed live **4915837 vs 4915200 (+637 formed synapses)** → `operands could not be broadcast` EVERY step, **silently caught (10023 tracebacks)** → **the Hebbian decay silently stopped applying.**

This is a KNOWN bug class here: `_ensure_gate_capacity` (`bridge.py:916`, added 2026-06-08) exists for exactly it — *"so the reward-modulated weight update raised 'operands could not be broadcast' every step and was silently caught, dropping plasticity"* — and 7 sites route through it. The Hebbian block was missed. **IMPACT BEYOND THIS DE-RISK: any run with a plasticity_gate + structural growth has been silently losing Hebbian decay — and the merged nav+conversational work uses plasticity gates heavily.** Masked until now because an UNGATED config leaves the gain `None` → the else branch multiplies by a SCALAR, which broadcasts against any nnz. Fixed byte-identically (46/46 regression). It surfaced only because tagging pathways with a `plasticity_gate` flipped the code path — i.e. **it was found by accident, not by a test.**

## 6. Artifacts

- `research/findings/raw/_cotrain_iso_homeo1_s424344.json` + `_homeo1_s100101102.json` — the clean 6-seed table
- `research/findings/raw/_cotrain_iso_homeo0_s424344.json` — the homeostasis probe (partial, 21% mean)
- `research/findings/raw/_cotrain_iso_idlematch_s424344.json` — the real-config decomposition
- `research/runners/_cotrain_iso_aggregate.py` — reproduces both tables

## 7. NEXT

1. **Re-verify the e-prop deep-credit GO before segment (b) is built on it** — `_onbridge_eprop_port_derisk.py` keeps `enable_bdsp=True` and relies on `lr=0` for inertness, but the committed kernel's `cp.clip` is UNCONDITIONAL (`kernels.py:485`): measured 239/512 FF synapses crushed to |w|≤6 from mean 370 every forward, while the runner's own note requires ~2000-scale weights. Re-run with `enable_bdsp=False` (the kernel's own documented lever), single-variable, K=8.
2. **Assert byte-inertness in a test** — three separate "X is inert/fine" claims failed silently in one day (this Hebbian bug; the BDSP clamp; `is_initialized=False` → every step a silent no-op). *A claim of inertness is a HYPOTHESIS that needs an ASSERTION, not a comment.*
3. **Segment (b): stream cortex + the deep-credit learner in its GO regime** (e-prop + POPULATION CODING, **not** bare BDSP — see the plan's 2026-07-16 correction note). The first test of RULE HETEROGENEITY. Scoping (wf_97529ac7-7f1) found the flag conflict is NOT the blocker (Hebbian is gate-isolable; homeostasis is per-region isolable); the genuine blockers are that **STP + structural plasticity are GLOBAL with no per-pathway escape and the stream cortex's validated +0.705 was obtained with BOTH ON** (⇒ re-validate with them off), and that `inject_explicit_wiring` REPLACES all connectivity (⇒ merge into ONE region-framework plan).
