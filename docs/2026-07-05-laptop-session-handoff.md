# Laptop session handoff — 2026-07-05

A self-contained report of everything done in this laptop session (2026-07-04 evening → 2026-07-05 afternoon)
and the planned upcoming work, so an independent **desktop** session can pick up seamlessly. Read this top-to-bottom;
it links every artifact.

> **One-line state.** The biological learned role read-out (`--mode c3`) is DELIVERED + validated (host ridge shortcut
> removed, 6/6 seed, anti-cheats clean). An adversarial audit then revealed the read reads *word position*, not
> grammatical *structure* — so the real open frontier is the **object-relative (objrel) structural read**, which was
> ground to ground across four spiking-read mechanism families and then **research-gated**. The key finding: **dendrites
> are a RED HERRING for it** — the residual is a *rank-1 common-mode* (a role-independent additive pedestal), and the
> specified fix is a **somatic feedforward common-mode subtraction** that was never cleanly tried. That is the primary
> upcoming build.

## Environment notes (for the desktop)

- This laptop is the **GTX 1660 Ti (6 GB)**; the objrel work is all **CPU/numpy** (`SIM_BACKEND=numpy`), fits fine.
- **cupy is now installed + verified end-to-end** on the laptop (see §1). The desktop's **3090** is for scale/VRAM.
- Git: this repo is on `main`, remote `origin` = github.com/danthi123/neural-simulator. This session's work is committed +
  pushed (this doc is part of that commit).
- **The desktop has its own independent session** — point it here; do NOT try to resume this specific session.

---

## Work completed

### 1. cupy GPU stack — installed + verified (the enabling win)

Installed **cupy 14.1.1** + the full **CUDA 12.9** math stack (nvrtc / runtime / nvcc / curand / cublas / cusparse /
cusolver / cufft / nvjitlink). Verified end-to-end on the 1660 Ti: kernels compile, `SimulationBridge` builds + steps +
runs GPU RNG, the concept-pool A→W bridge (`build_concept_bridge`) builds + steps, and **`train_word_to_pool` (STDP
training) runs on cupy**. ⇒ the A→W word-spell arc is GPU-viable now.

- **Known gap (tracked, task #5):** `research/runners/unified_brain_bridge.py:254` does `bridge.cp_traits[:] = 0` but
  `cp_traits` is None on the cupy backend → the rungB1c/unified conversational bridge only runs on numpy. Same class as
  the documented `test_regions.py` cupy failures. Out of scope for the numpy-validated read-out work; fix on the desktop.

### 2. The biological learned read-out — `--mode c3`, DELIVERED + validated

`research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` now has **`--mode c3`**: the reservoir→role
read-out matrix is **biologically LEARNED** by a per-role **delta rule** (`_learn_Ws_spiking`) on the frozen spiking
reservoir, **removing the last host shortcut** (a ridge least-squares fit, `np.linalg.solve`). Additive — c1/c2
byte-identical.

- **Seed 101 CLOSED — genuinely, by TEMPORAL RESOLUTION.** The prior 5/6 (seed 101 = 14/18) was NOT under-training
  (E20 was worse). `READ_T=30` (the c2 CRUX window; c3 had used step6's speed-compromise 18) closes seed 101 → 18/18 at
  the DEFAULT reservoir position. `C3_READ_T_STEP=30` is now the default. **The P160/P240 "population lever" that looked
  like a fix was a CONFOUND** — `WTA_P_C2` is a def-time-frozen arg, so it only enlarged a slice and *shifted the
  reservoir onto a luckier heterogeneity draw*; caught by a control before baking in.
- **Two integration bugs fixed:** `_build_wired_bridge` routed c3 to c1's small **P=20** WTA (now shares the c2 **P=80**);
  and **D2** — `_source_learned_readout_clean()` was defined but **never called** in the verdict (a dead anti-cheat) → now
  AND-ed into the c3 source-clean check.
- **Anti-cheats CLEAN, 2-seed (42/44):** `LEARNED 18/18 | scramble 0 | global 6 | syn-lesion 6` (chance = 6) — genuine,
  per-role-local, synapse-load-bearing. Source-clean (both SELECT + LEARN paths).
- **Committed-runner VALIDATION:** runner c3 seed-101 at T=30 = clean **GO** (route 12/12 == host-dict; all anti-cheats
  hold). Slow CI `test_seed42_c3_learned_readout_GO` passed; **7 fast CI tests pass**
  (`tests/test_rungB1c_spiking_reservoir_synaptic_readout.py`).
- Finding: `research/findings/2026-07-04-biological-learned-readout-delta-rule.md`. State: AUTONOMOUS_STATE **CYCLE 921**.

### 3. The adversarial audit — the position-vs-role reframe (the pivotal moment)

A read-only adversarial audit (workflow) of the c3 arc caught two committed defects:
- **D1 (load-bearing):** the GO test set is **canonical SVO only**, where a content word's grammatical role is largely
  predictable from its **word position** — so "5/6 seeds at 18/18" does NOT prove the read-out reads grammatical
  **role**. The machinery to break it (`_objrel_test_fact`, the object-relative construction) existed but was disabled.
- **D2:** the dead source-check (fixed, see §2).

⇒ ran the decisive test.

### 4. The objrel structural read — honest negative, precisely isolated

Object-relative sentences ("**the ball that the dog chased**", where word order ≠ roles: slot-0 = THEME, not AGENT).
Scored per-slot vs TRUE roles, held-out, **3-seed (42/44/100)**:

- **Canonical 36/36 = 1.00** (probe faithful), **OBJREL 0/36 = 0.00**, objrel slot-0 (THEME) 0/12. ⇒ the biological
  read-out reads **position, not grammatical role**.
- **Isolated exactly:** the reservoir *feature* genuinely **encodes** objrel — a shift-invariant **linear argmax** over
  it reads slot-0 = THEME at **1.00 on every seed, even positive-shifted**. What the spiking read loses is the structure.
  ⇒ **not irreducible; the information is present + linearly separable.**

### 5. The surpass grind — four spiking-read mechanism families, all seed-fragile

Ground to ground, all multi-seed, all anti-cheated:
| mechanism | result |
|---|---|
| positive WTA (all floors, competition on/off) | sharp on canonical, **objrel 0.00** (common-mode pedestal swamps the margin) |
| signed conductance (Wp exc + Wn inh relay) | structural-capable but **WEAK** (no competition, canonical 0.3–0.6) + **op-point/seed FRAGILE** (seed 100 recovers objrel-slot0 0.67 at a calibrated op-point; seed 44 recovers at none) |
| learned delta THROUGH the deploy | collapses to **position**; heavy objrel oversampling reaches slot-0 only **0.33** with a canonical tradeoff |
| FHRR **phase** read (design-workflow-recommended) | **seed-fragile** (works on 44, fails 42/100) + mathematically the wrong tool (a *similarity/nearest-phasor* classifier, not the *linear* one objrel needs — phase is for binding) |

**Two overclaims caught by the multi-seed/anti-cheat discipline** (both corrected in the finding): the population-lever
reservoir-position confound, and a signed-conductance seed-42 result (objrel 0.92) that **failed multi-seed** (0.75/0.00/0.50).

### 6. The dendritic research — RED HERRING (the key redirect; owner-directed)

Owner flagged that the project has repeatedly assumed dendrites were needed and been wrong (the "decorrelation → needs
dendrites" fork was overturned by PPMI feedforward normalization). A deep-research gate (workflow) inventoried the
existing dendritic groundwork **and** pressure-tested the assumption. Verdict, all three angles converging:

- **Dendrites are a RED HERRING for objrel.** The residual, isolated to the byte, is **one role-independent additive
  scalar per example** — the uniform tonic pedestal `WS_ENS_FLOOR_C2 = 150 pA` (to all 3 role ensembles) + the Dale-shift
  `−min·Σf`. Both are **identical across the 3 roles** and example-dependent. This is a **rank-1 common-mode
  subtraction**, NOT the per-input cross-neuron **decorrelation** that Mikulasch-Priesemann actually bounds on point
  neurons (the info is "present + linearly separable" — the exact decorrelation-red-herring pattern PPMI overturned).
- **Why every attempt failed:** they placed the subtraction in the WRONG spot (a downstream floor / competition) or WRONG
  domain (a nonlinear inhibitory relay, `f_I(Wn·f) ≠` linear). **The simplest move — subtract a shared feedforward
  baseline from the drive BEFORE the read (a single summing inhibitory pool, PPMI-style) — was never actually tried.**
- **Op-point fragility (seed 42/100 recover, 44 doesn't) is the fingerprint of a MISSING per-draw normalization**, not of
  a substrate that structurally can't subtract a baseline. Dendrites are "at most the 3rd option, never the default."
- **Existing dendritic groundwork IS real + reusable** (not months-scale): a validated two-compartment dendritic-plateau
  (dAP) neuron that senses apical voltage separately (`sim/dendritic_neuron.py`, findings `2026-07-02-emerge10-
  stageAprime-two-compartment-dap-GO.md`), `sim/dendritic_{mlp,plasticity}.py`, on-bridge graded plateau — but it is the
  3rd option, only if the somatic fix and a homeostatic op-point both fail.

### 7. Docs + diagram updates (this session)

- `docs/diagrams/brain_architecture_current.md` — a dated **currency banner** + a new **Diagram 1b (comprehension
  read-out)** showing the reservoir → biological read-out → roles → composer flow with the removed shortcut ✓ and the
  objrel frontier ⚠. (The exhaustive hand-authored **SVGs** are a desktop task — they need the `review-diagrams` render
  + tile-review + commit loop.)
- `docs/plans/2026-07-05-objrel-structural-read-surpass-plan.md` — the surpass ladder + decision tree (note: its ladder
  is now **updated by §6** — the somatic FF subtraction is the primary, dendrites deprioritized to 3rd).

---

## THE PRECISE DIAGNOSIS (the load-bearing carry-forward)

The objrel read needs to remove **one role-independent additive pedestal** (`WS_ENS_FLOOR` + Dale-shift `−min·Σf`) so a
subtle **role-dependent** structural margin survives the spiking nonlinearity. The shift-invariant *linear* argmax cancels
it (→ 1.00 every seed); the positive spiking WTA's ignition-order winner, sitting *on* the pedestal, cannot. **This is a
rank-1 common-mode, removable by a somatic feedforward inhibitory pool — not a dendritic computation.**

---

## Planned upcoming work

### PRIMARY — the somatic feedforward common-mode subtraction (the objrel surpass)

Implement + de-risk the specified fix. **Do it in the SYNAPTIC deploy, not a host shortcut** (my host-logit-injection
probes, `scratchpad/step11_centered_drive.py`, were buggy — a constant injected drive makes the WTA settle into an
index-bias; canonical 0.00, objrel a bias artifact — do NOT trust them).

1. **Reproduce the boundary faithfully.** `research/findings/raw/signed_conductance/step7_ridge_spiking.py` deploys the
   ridge Ws through the real spiking WTA and gives **objrel 0.00 / canonical 0.97** — that is the boundary harness to
   start from.
2. **Add a shared feedforward inhibitory pool** that pools the reservoir drive and delivers the **mean of the 3 role-ens
   drives** as inhibition to each ensemble **before** they fire, so the ensembles compete on the **centered** drive
   (removing the rank-1 common-mode). This is FF inhibition (Kandel), point-neuron/somatic — likely a runner-side wiring
   (a new inh region + a `role-ens → inh → role-ens` route), ideally **no `sim/` edit** (or a small additive one).
3. **Gate on multi-seed + anti-cheat:** objrel slot-0 high on ALL of 42/43/44/100/101/102, canonical held, scramble →
   chance, the FF pool load-bearing (lesion → collapse back to the pedestal boundary). **No single-seed or partial win.**
4. **If it's fragile too** → the **homeostatic self-calibrating operating point** (per-draw, a firing set-point — the
   step9 data showed a *calibrated* op-point already recovered seed 100). **Only if that also fails** → the
   two-compartment dAP (basal = read-out drive, apical = common-mode, soma fires on basal − apical), reusing the existing
   dAP groundwork (§6).

### Parallel arcs (independent of objrel — ready to run)

- **A→W BRIDGE-C fix** (fully-spiking word production for transitive/PP constructions). The EMERGE-75 boundary is a
  **co-training** issue (3 high-freq prepositions {to,on,is} + 13 object nouns on one 16-pool bridge — the Goldilocks
  signature), so the fix is the named **EMERGE-75b pool-reassignment**: move {to,on,is} onto BRIDGE-F's free filler pools
  (it has 11), leave BRIDGE-C with just the 13 nouns, **retrain both on GPU** (now unblocked), update the
  `UnifiedNeuralSpell` dispatch. Requires editing the validated EMERGE-68/75 runners **additively** → a focused arc.
- **RUNG B synaptic comprehension→composition hand-off** (the functional one brain): project the reservoir's role output
  → a role-ensemble region via the learned read-out as a fixed `RegionPathway` → WTA-select → feed the composer's
  parser-firing route, replacing the host `{role:word}` dict. CPU-feasible.
- **cupy-path `cp_traits=None`** (task #5) — unblocks the conversational bridge on the 3090.

---

## Artifacts / pointers

| What | Where |
|---|---|
| The read-out runner (`--mode c3`) | `research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py` |
| CI tests | `tests/test_rungB1c_spiking_reservoir_synaptic_readout.py` |
| The delta-rule + objrel finding | `research/findings/2026-07-04-biological-learned-readout-delta-rule.md` |
| The surpass plan + decision tree | `docs/plans/2026-07-05-objrel-structural-read-surpass-plan.md` |
| Current-state diagram (+ Diagram 1b) | `docs/diagrams/brain_architecture_current.md` |
| Session state | `research/findings/AUTONOMOUS_STATE.md` **CYCLE 921** |
| Objrel probes (isolation + surpass) | `research/findings/raw/signed_conductance/step7_*.py`, `step8_learned_signed.py`, `step9_signed_floor_sweep.py`, `step10_phase_concept.py`, `step11_centered_drive.py` (the last two/three: honest NEGATIVES + the buggy host-logit test — see the PRIMARY note) |
| Signed-readout machinery (dAP-adjacent) | `research/runners/_rungB1c_signed_readout_derisk.py`; dendritic: `sim/dendritic_{neuron,mlp,plasticity}.py` |

## Task-list state (for reference)

- ✅ #1 seed-101 (T=30) · ✅ #2 anti-cheats · ✅ #3 c3 promotion · ✅ #6 objrel test · ✅ #7 fixed-read surpass (characterized)
- ⏳ #4 next planned work (the parallel arcs above) · ⏳ #5 cupy cp_traits bug · ⏳ #8 objrel surpass → **now the somatic FF
  common-mode subtraction** (dendrites deprioritized per §6)

## Discipline reminder (carried forward)

Multi-seed + anti-cheat gate **every** claim — this session caught **two** of my own overclaims that way (the
population-lever confound; the signed-conductance seed-42 result). A partial/single-seed/lucky-draw result is not a win.
