# N9 place-code reproducibility + robustness research — the non-determinism is a TRANSPOSE-SpMV artifact (fixable runner/sim-side), and the right primary fix is to robustify the critic so the draw stops mattering

**Date:** 2026-06-10
**Type:** read-only deep-research + catalog/Kandel/literature review + LIVE engine probes. NO code edited.
**Backend probed:** CuPy **13.6.0**, CUDA **12.9** (runtime 12090), RTX 3090, the deterministic-nav regime (OU/cond-noise/global-homeostasis OFF; `CUBLAS_WORKSPACE_CONFIG=:4096:8` pinned at the top of `g11_bg_runner.py:62`).
**Boundary being root-caused:** the N9 value critic transfers to the full nav bridge and passes all four Stage-B gates on a *strong* place-code draw (`2026-06-10-N9-nav-deployment-stageB-PASS-seed42.md`), but the **place-code self-org is CuPy-non-deterministic run-to-run** (same seed, same config → different place code → `w_near` 3.628 vs 1.916 → different gate outcomes), so multi-seed robustness is blocked.

> **Scope note vs the sibling doc.** `2026-06-09-N9-robust-value-learning-diagnosis.md` already root-caused the *training-protocol* fragility (Yagishita DA-after-pairing timing + MSN up-state hold) and recommended Option 1 = phase-separate the trial. **This doc is the complementary pair:** (A) WHY the self-org is non-deterministic and how to make it reproducible, and (B) how to make the critic learn V strongly *regardless* of draw strength so non-determinism stops mattering. The two are synergistic, not competing — §7 stacks them.

---

## 0. TL;DR / recommendation

1. **Axis A is now SOLVED at the diagnosis level, and the fix is cheap.** I traced the non-determinism to one line and **proved it on this exact engine**: the per-step synaptic drive is a **transpose** sparse mat-vec, `effective_connections_matrix.T @ fired` (`sim/bridge.py:5613` and `:5620`). On CuPy 13.6.0 / CUDA 12.9, `csr.T` returns a **`csc_matrix`**, and `csc @ vec` routes through `cupyx.cusparse.spmv(..., transa=True, alg=CUSPARSE_MV_ALG_DEFAULT)` — the **transpose, default-algorithm** cuSPARSE path, which uses an atomic scatter and is bit-non-reproducible. **Live probe: `A.T @ v` gave 200/200 DISTINCT results over 200 runs; `A @ v` (non-transpose) gave 1/1.** Two drop-in deterministic replacements both gave **1/1, numerically identical** to `A.T @ v`: (i) pre-materialize the transpose as a CSR once (`A.T.tocsr() @ v`, turns the per-step op into a *non-transpose* SpMV); (ii) a tiny dense matmul (`A.toarray().T @ v`, only **2.56 MB** at 800×800). The NumPy/scipy backend is deterministic 1/100 (option A3 viable).
2. **But pinning the draw does NOT fix the science** — a *reproducibly weak* draw is still weak. **The owner-stated primary lever is correct: robustify the critic (axis B).** The cheapest, most biological, already-shipped lever is **per-region homeostasis** (the committed `89b8d909` `BrainRegion.enable_homeostasis` edit, exposed as `--enable-critic-homeostasis`), which is the project's Turrigiano/Desai mechanism and **already de-risk-PASSED 3/3** on the isolated critic by firing the afferent+critic into a useful range *independent of raw drive*. It is the direct answer to "make the readout robust to variable presynaptic drive."
3. **The "re-roll the place map until the goal volley fires ≥K" (goal-field-adequacy) gate is biologically real (Hollup 2001 goal over-representation) BUT is a banking/selection cheat as a *bootstrap loop* and I recommend against shipping it as the fix.** It selects the RNG draw on a downstream success criterion — the literature shows over-representation *emerges from reward-gated plasticity at a stable goal*, it is not a map-selection oracle. Use reward-gated strengthening of the goal field (D.16 D1/D5 attention-stabilized place fields) if you want the biology; do not use "keep-trying-until-it-works."
4. **Recommended cheap-first de-risk (one experiment, runner-only):** make the place-code self-org deterministic with the **pre-transposed-CSR trick scoped to the place pathway** (or, even cheaper to prototype, run the *self-org phase only* on the NumPy backend and transfer the frozen weights), then run the **already-shipped `--enable-critic-homeostasis`** at nav scale across seeds 42/43/44. **Success = same seed → byte-identical place code across two invocations AND 3/3 Stage-B PASS including the documented weak draw.** This isolates "is robustness from homeostasis, or from luck-of-the-draw?" — which the current non-determinism makes impossible to tell.

---

## 1. Diagnosis (crisp): the non-determinism is a transpose-SpMV atomic-scatter, and it is the *only* run-to-run source in the deterministic regime

### 1.1 The exact line

The per-step excitatory/inhibitory conductance increment in `_run_one_simulation_step` is (`sim/bridge.py:5613`, E/I-split branch; `:5620`, single branch):

```python
g_increase_2col = effective_connections_matrix.T @ fired_2col      # :5613  (.T == TRANSPOSE)
...
g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * cfg.propagation_strength   # :5620
```

`effective_connections_matrix` is a `cupyx.scipy.sparse.csr_matrix` (built at `:5515/:5569`). The simulator stores synapses **row = presynaptic source**, so propagating "source fired → drive its targets" needs the *transpose* (`Wᵀ @ fired`). That transpose is the non-deterministic operation.

### 1.2 Why the transpose is non-deterministic on THIS engine (proven, not inferred)

CuPy 13.6.0 source path, confirmed by inspecting the installed library:
- `csr_matrix.__mul__` (`cupyx/scipy/sparse/_csr.py`) only handles the **non-transpose** vector product (`self * vec`). On CUDA ≥ 11000 the CUB path is **disabled** (`is_cub_safe &= cub_build < 11000`), so non-transpose `A @ v` goes through `cusparse.csrmvEx`/`spmv` in a **deterministic** layout (one thread per output row, no cross-row atomics).
- `csr.T` returns a **`csc_matrix`** (verified live). `csc @ vec` enters `cupyx.cusparse.spmv`, which contains:
  ```python
  if isinstance(a, csc_matrix):
      a = a.T; transa = not transa          # CSC treated as transposed-CSR
  ...
  op_a = _transpose_flag(transa)            # -> CUSPARSE_OPERATION_(NON_)TRANSPOSE
  alg  = _cusparse.CUSPARSE_MV_ALG_DEFAULT  # <-- HARD-CODED default algorithm
  ```
  So `csr.T @ v` ⇒ `spmv(op=TRANSPOSE, alg=DEFAULT)`. A transpose SpMV on a CSR cannot assign one thread per *output* element without scattering, so cuSPARSE's default kernel does an **atomic add into the output**, whose floating-point summation order is non-deterministic.

**Live engine probe (n=800, density 0.05, float32, 200 trials each):**

| op | distinct fp results / 200 | deterministic? |
|---|---|---|
| `A.T @ v` (the bridge's per-step op) | **200** | NO |
| `A @ v` (non-transpose) | **1** | YES |

This is the run-to-run variance the Stage-B doc observed (STEP-1 diff-cos 0.031 vs 0.086). It accumulates across the ~2000 self-org steps and across the STDP that rides on those spikes, so by the end of self-org the *place code itself* differs (sparsity 0.041 vs 0.063) → `w_near` differs → gates differ. **`CUBLAS_WORKSPACE_CONFIG=:4096:8` pins cuBLAS GEMM only; it does nothing for this cuSPARSE atomic-scatter** — exactly as the Stage-B doc stated, now with the mechanism nailed.

### 1.3 The deterministic-regime claim is otherwise sound

In this regime OU noise, conductance noise, and global homeostasis are OFF (the runner asserts it). The neuron/STDP kernels are `@fuse`d elementwise ops (deterministic). The *only* cross-thread reduction in the hot loop is this transpose SpMV. So fixing it should make the self-org reproducible **without** touching anything else. (Caveat to verify in the de-risk: confirm no *other* `.T @` / `@` reduction enters the self-org path — the coincidence/NMDA/GABA_B restricted matvecs at `:5705/:5768/:5812` are also transpose SpMVs and would need the same treatment if they run during self-org; during STEP-1 the value arm is gated off, so they likely don't, but assert it.)

---

## 2. Axis A — make the place-code self-org reproducible (ranked)

> All of A1–A3 produce a **numerically identical** result to the current transpose SpMV (verified for A1/A2) — they change *only* the FP summation order, not the math. None is a "cheat": determinism is a tooling property, not a biological shortcut.

### A1 (RECOMMENDED, cheapest correct) — pre-materialize the transpose ONCE as a CSR, do non-transpose SpMV per step. **Runner-side IF scoped to the place pathway; `sim/` edit if applied to the global matrix.**
**Mechanism.** Replace `M.T @ fired` with `M_T @ fired` where `M_T = M.T.tocsr()` is built **once** (frozen weights after self-org freeze, so it never changes) — the per-step op is now a **non-transpose** CSR SpMV, which probed **1/1 deterministic** and `allclose` to the transpose product.
- **Live proof:** `(A.T.tocsr()) @ v` → 1/1 distinct over 200 runs, `matches A.T@v == True`.
- **Scoping (the load-bearing decision):** the bridge's SpMV is over the **whole** `cp_connections` (all regions), so a *global* swap to a pre-transposed CSR is a **protected `sim/` edit** touching every dynamics step (must be byte-reviewed + byte-identity-proven, mirroring the `89b8d909`/per-region-NMDA pattern). **However**, the self-org only needs the **place pathway** to be reproducible. The cleanest runner-side form is to run the self-org with a *place-only* sub-bridge or to do the place matvec in the runner against a pre-transposed place CSR during STEP-1 (the runner already drives `place_sensors → place` in `_run_place_selforg`). Assess which is feasible without a `sim/` edit; if the global swap is unavoidable, it is **the cleanest possible `sim/` change** (a one-time storage transpose, default-on, byte-identical math) and likely worth proposing because it makes the **entire engine** deterministic in the deterministic regime — a project-wide reproducibility win, not just N9.
**Cost:** minutes (runner-scoped) to ~1 hr (global `sim/` edit + byte review).
**Anti-cheat:** identical math (allclose), so no behavioral cheat; the only risk is forgetting that `M_T` must be rebuilt if weights change (it can't here — frozen after self-org).

### A2 — dense matmul for the (small) place pool self-org only. **Runner-side.**
**Mechanism.** The place pool is ~800 neurons. `M.toarray().T @ v` is **2.56 MB** and probed **1/1 deterministic**, identical to the sparse transpose. For the *self-org phase only*, build a dense place-pathway weight matrix and step the place dynamics densely in the runner (the place fields are the only thing that must be reproducible; the rest of the bridge isn't driven during STEP-1).
**Cost:** runner-side; small. **Trade-off:** duplicates the place dynamics in the runner (maintenance), and is only "cheap" because the pool is tiny — does NOT generalize to the full bridge. Prefer A1 if a clean scoping exists.
**Anti-cheat:** identical math; dense vs sparse is a storage choice, not a biological change.

### A3 — run the self-org phase on the NumPy backend, transfer the frozen weights to the GPU bridge. **Runner-side; uses existing `SIM_BACKEND=numpy` + checkpoint machinery.**
**Mechanism.** NumPy/scipy SpMV is deterministic (probed **1/100**). Build the place sub-bridge under `SIM_BACKEND=numpy`, run STEP-1 self-org, freeze, export the `landmark_to_place` weights (the project already has `save_checkpoint` + `BridgeLineage.export_shards` + `set_pathway_weights`), then load them into the CuPy nav bridge for STEP-2/Stage-B. Because the place fields are FROZEN after self-org, the GPU side never re-runs the non-deterministic learning — it only *reads* the place code (the read-out volley is a forward pass; if the read SpMV is also `.T @`, combine with A1 for the read, or accept that the read-volley jitter is exactly the FS-PING/coincidence robustness the arc already validated).
**Cost:** runner-side; ~1–2 hr (wire the backend switch + weight transfer). The project explicitly supports "NumPy self-org → transfer to GPU" as a pattern (`sim/backend.py`, the lineage shard export).
**Trade-off:** CPU self-org is slower (~minutes for 800 neurons × 2000 steps is fine), and you must verify the transferred CSR is byte-identical CuPy-vs-NumPy (the 16 sparse-pattern tests pin a similar regen invariant for G.20). **Best "no-`sim/`-edit, provably deterministic" option** if A1's scoping forces a `sim/` change.

### A4 (do NOT bother first) — chase a deterministic cuSPARSE transpose algorithm.
cuSPARSE *does* expose deterministic SpMV — but **only `CUSPARSE_SPMV_CSR_ALG2` and only for `opA == NON_TRANSPOSE`** (NVIDIA cuSPARSE 12.x docs). There is **no deterministic transpose SpMV**. And CuPy 13.6.0 **hard-codes `CUSPARSE_MV_ALG_DEFAULT`** with no pass-through, so reaching ALG2 needs monkeypatching `cupyx.cusparse.spmv` — strictly worse than A1 (which already converts the op to non-transpose, the regime where determinism *is* available). Mentioned only to close the "can we just flip a cusparse flag?" question: **no.**

**Axis-A recommendation:** **A1 scoped to the place pathway** if a clean runner-side scoping exists; otherwise **A3 (NumPy self-org + transfer)** as the provably-deterministic, no-`sim/`-edit path. A2 is a fine quick prototype. A global A1 `sim/` edit is attractive as a *project-wide* deterministic-regime win but is out of N9's minimal scope — flag it separately.

---

## 3. Axis B — robustify the critic so it learns V strongly REGARDLESS of draw (ranked, biology-grounded)

> This is the owner's stated primary lever and I agree it should lead: even a *reproducible* draw can be weak, and a value system that only works on strong draws is not a value system. The goal is a downstream readout that normalizes out variable presynaptic drive.

### B1 (RECOMMENDED, already shipped + already 3/3-de-risked) — per-region INTRINSIC-EXCITABILITY homeostasis on the afferent + critic. **`sim/` edit ALREADY COMMITTED (`89b8d909`), exposed runner-side as `--enable-critic-homeostasis`.**
**Mechanism / biology.** Intrinsic homeostatic plasticity (Desai 1999; Turrigiano 2008): a neuron adjusts its own excitability (threshold) to defend a target firing rate regardless of input scale. The committed edit adds `BrainRegion.enable_homeostasis` + `cp_homeostasis_neuron_mask`, running the EMA-threshold update (`fused_homeostasis_update`, target_rate 0.02, ema_alpha tau≈5 s, adapt 0.0005) on JUST the `vs_place_context` afferent and the `striosome_value` MSN-D1 — even though global `cfg.enable_homeostasis` stays OFF (deterministic regime). It is the **direct** "make the readout reach a target rate independent of input strength" mechanism.
**Status — this is not theoretical:** the commit message records the de-risk **PASS 3/3** with afferent+critic homeostasis — "critic fires ~1.3–1.5 Hz, place code stays sharply graded (afferent ~59 Hz near vs 0 Hz far — no place-blindness), GABA_B value subtraction opens (gaps 3.19/2.35/1.39), anti-cheats hold." Findings: `2026-06-08-navfaithful-derisk-FAIL-homeostasis-confound.md` (the critic-only FAIL + 5 confounds) + the 6th-iteration afferent+critic PASS.
**Reuses:** `--enable-critic-homeostasis` (`g11_bg_runner.py:7814`), `--critic-homeostasis-*` knobs (`:7830-7838`), the committed `cp_homeostasis_neuron_mask` (`sim/bridge.py:5863, 6660`). **No new code.**
**Timescale concern (the real caveat):** homeostatic threshold adaptation is SLOW (adapt_rate 0.0005 → ~0.5 mV/s; ema tau ~5 s). The de-risk PASSED, so over the value-train window it suffices, but it works by *slowly lowering the critic's threshold so any draw eventually fires it* — verify it does NOT collapse the threshold so far that it fires place-blind (the de-risk reports it stays graded; re-assert at nav scale). This is the BOTH-cells version; the commit notes critic-only FAILED (the afferent must also be lifted). **Use afferent+critic, as de-risked.**
**Anti-cheat:** the homeostasis acts on the *cell's own threshold* (intrinsic, neural), not on a host value; the place-shuffle control must still break V (it does in the de-risk); confirm `striosome_value` stays GRADED near-vs-far after threshold adapt (place-shuffle + far-cell readout).

### B2 — homeostatic SYNAPTIC SCALING on the place→value afferent (Turrigiano 2008). **Partially shipped (global flag), per-region scaling is NOT yet wired.**
**Mechanism / biology.** Multiplicative synaptic scaling: when the postsynaptic cell is under-active, *all* its excitatory afferents are scaled UP by a common factor (preserves relative weights → preserves selectivity), normalizing post firing across input scales. This is the canonical "normalize a readout to variable presynaptic drive" mechanism and is exactly catalog-described (`feature-catalog.md:956` "Synaptic scaling (homeostasis) provides a coarse functional analog… sets activity setpoints"; `:3587` "Synaptic scaling implicitly models AMPA-receptor density adjustment").
**Status in the project:** `enable_synaptic_scaling` EXISTS (`config.py:229`, rate 0.001, clip ±5%/step) and runs at `sim/bridge.py:6677` — `scale = 1 + rate·(target − ema)` per postsynaptic neuron. **BUT it is gated by the GLOBAL `cfg.enable_synaptic_scaling`, not by the per-region mask** (the `89b8d909` edit gated *threshold* homeostasis per-region; the scaling block still keys off the global flag). So turning it on scales **every** weight in the bridge, which in the nav bridge would perturb the actor — NOT cleanly scoped to place→value. **To use it cleanly you'd need a small per-region extension of the scaling guard** (mirror the threshold mask) — a `sim/` edit.
**Reuses:** the scaling kernel + EMA. **Gap:** per-region scoping (small `sim/` edit, same pattern as `89b8d909`).
**Timescale concern:** rate 0.001 with ±5%/step clip → reaches a 2× up-scale in ~hundreds of steps; comparable to the value-train window. Faster than threshold homeostasis to a *weight* target, but acts on weights (could fight the DA-gated LTP — order them: scale to set the operating point, then DA-LTP grades on top).
**Verdict:** biologically the *best-matched* mechanism ("normalize the afferent"), but **B1 already achieves the de-risked PASS with zero new code**, so B2 is the fallback if B1's threshold-collapse risk bites. If pursued, it's a clean, byte-reviewable per-region extension.
**Anti-cheat:** multiplicative scaling preserves *relative* weights → must NOT create place-blindness (the near>far grading must survive; test). It scales by the cell's OWN activity EMA (neural), not a host value.

### B3 — developmental GOAL-FIELD-ADEQUACY gate ("re-self-org until the goal volley fires the critic ≥K"). **Runner-side — but I recommend AGAINST it as the fix (banking/selection cheat). Argued both sides below.**
**The biology it appeals to (real):** hippocampal place cells **over-represent goal/reward locations** — Hollup et al. 2001 (annular water maze: excess fields accumulate at the hidden platform), Dupret et al. 2010 (fields remap to represent new goal configs when goals aren't cue-marked), Gauthier & Tank 2018 (a fixed sub-population of "reward cells"). Catalog **D.16** (`feature-catalog.md:1272`): "attended/goal-directed running → fields stable for days" (D1/D5-gated). So *more representational mass at the goal* is biologically attested.
**Why it is nonetheless a CHEAT as a bootstrap loop (the case against — decisive):**
- **It selects the RNG draw on a downstream success metric.** "Re-roll `self_org_rng` until the goal ensemble fires ≥K" is the classic *keep-trying-until-it-works* / banking pattern the project repeatedly flags (e.g. AUTONOMOUS_STATE's "deterministic-copies" and "drive-echo" retractions). The *brain* does not draw 50 random place maps and keep the one where the goal happens to fire hard — that's a host search over seeds, exactly the kind of host computation the BRAIN-BASED-ONLY standard bans (the *selection* is done by Python `if crit<K: reseed`, not by neurons).
- **The literature does NOT support map-selection.** Over-representation **emerges from reward-gated plasticity at a STABLE goal over experience** (Hollup: it builds *as the rat learns*; the JNeurosci 2022 follow-up: excess firing at a moved goal *appears then vanishes as the new goal is learned*). And it is **task-dependent** — present in the annular water maze, ABSENT in a place-preference task and when goals move unpredictably (Cell Reports 2020; JNeurosci 2022 "without place-field accumulation" when goals moved between sessions). So the faithful mechanism is "reward strengthens the goal field *in place*," not "pick a map where the goal is already strong."
**The case for (steelman):** a developmental "critical-period adequacy check" is not unheard of — if you framed it as "the organism keeps exploring/encoding until a usable spatial code for the goal region exists," and the re-roll were a stand-in for *more developmental experience* rather than a seed-search, it could be a defensible scaffold (the innate-reflex-teaches pattern). **But** the honest version of that is B4 (reward-gated strengthening of the existing goal field over more self-org steps), which doesn't reseed — it's the *same* map, learned harder. The reseed version's tell is that it discards the map wholesale, which has no neural correlate.
**Verdict:** **do not ship the reseed gate as the fix.** If you want the goal-over-representation biology, do it as **B4** (strengthen, don't select). The reseed gate is acceptable ONLY as a throwaway *diagnostic* to measure "what fraction of draws are adequate?" — and even then, report it as a diagnostic, never as the mechanism.

### B4 — reward-gated strengthening of the GOAL place field (the honest version of B3; the D.16 mechanism). **Runner-side (reuses DA + plasticity gate).**
**Mechanism / biology.** Instead of reseeding, give the goal location *more* DA-gated self-org so its field is reliably dense — D.16 (D1/D5 + attention stabilize goal-directed fields) + Hollup over-representation *as learned*. Concretely: during STEP-1, weight the agent-placement sweep toward the goal region (more exposure where it matters — a legitimate *environment/curriculum* choice, like a rat spending more time near the platform), or open a brief DA-gated potentiation of `landmark_to_place` specifically when the agent is at the goal (reward → strengthen the co-active goal field). The over-representation then **emerges from plasticity at a stable goal**, exactly as the literature shows — no map selection.
**Reuses:** `_n9_selforg_positions` (bias the sweep — environment-side), the DA modulator + `landmark_to_place` gate. **No `sim/` edit.**
**Trade-off:** must keep the goal *stable* during the strengthening (the literature: over-representation vanishes for unpredictably-moving goals) — fine for single-goal Stage-B; for multi-goal it would need per-goal fields (and the far=(1,1) being itself a trained goal is the very confound the Stage-B doc hit). Slower than B1; more moving parts.
**Anti-cheat:** the strengthening is DA-gated synaptic plasticity at the co-active goal field (neural), not a host pick; the place-shuffle control still applies (shuffling which cells track value must break V even though the goal field is denser); and biasing the *placement sweep* is environment-side (legitimate), but it must not become "only ever show the goal" (that would make the code goal-blind elsewhere — keep broad coverage, just over-weight the goal, as Hollup's rats did).

### B5 — divisive normalization on the critic's afferent (Carandini-Heeger). **`sim/` edit; defer.**
**Mechanism / biology.** Divisive normalization (Carandini & Heeger 2012, "a canonical neural computation") makes a readout invariant to input *magnitude* by dividing each input by a pooled-activity denominator — would make the critic's drive depend on the *pattern* (which place cells) not the *gain* (how many spikes), directly neutralizing draw-strength variance. Biologically it's implemented by pooled feedforward/lateral inhibition (the FS-PING pool is already a normalization substrate!).
**Status:** not a named mechanism in the engine; would be a `sim/` kernel (or an approximation via the existing FS inhibition tuned to divisive rather than subtractive). **Defer** — B1 already passes; B5 is the heaviest.
**Anti-cheat:** if done via real pooled inhibition (the FS pool), it's neural; if done as a host `x / (σ + Σx)` it's a host shortcut — must be synaptic.

**Axis-B recommendation:** **B1 first** (shipped, de-risked 3/3, zero new code, directly normalizes the readout), with **B2 (per-region synaptic scaling)** as the byte-reviewable fallback if B1's slow threshold-adapt collapses selectivity, and **B4** if you specifically want the goal-over-representation biology (the honest, non-selecting form of B3). **Reject B3 (reseed gate) as the shipped fix.**

---

## 4. What existing project machinery is reusable (named)

| Need | Reuse | Where |
|---|---|---|
| Deterministic non-transpose SpMV (A1) | swap `M.T @ v` → `M.T.tocsr() @ v` (frozen) | `sim/bridge.py:5613, 5620` (the math is `allclose`) |
| Deterministic CPU self-org (A3) | `SIM_BACKEND=numpy` + `get_backend()` + checkpoint/shard export + `set_pathway_weights` | `sim/backend.py`, `sim/lineage.py` (`export_shards`), bridge `set_pathway_weights` |
| Determinism env pin (cuBLAS only — insufficient alone) | `CUBLAS_WORKSPACE_CONFIG=:4096:8` already set | `g11_bg_runner.py:62`, `--deterministic` |
| **Intrinsic-excitability homeostasis on afferent+critic (B1)** | **`--enable-critic-homeostasis`** + `--critic-homeostasis-*` | `g11_bg_runner.py:7814-7838, 1076, 1107`; `cp_homeostasis_neuron_mask` `sim/bridge.py:5863, 6660` (committed `89b8d909`) |
| Synaptic scaling (B2) | `cfg.enable_synaptic_scaling` (global; per-region needs a small extension) | `config.py:229`, `sim/bridge.py:6677-6699` |
| Reward-gated goal-field strengthening (B4) | DA modulator + `landmark_to_place` plasticity gate + `_n9_selforg_positions` (bias the sweep) | `g11_bg_runner.py:4657, 4690`; `sim/neuromodulators.py` |
| The place self-org itself (target of the fix) | `_run_place_selforg`, `_n9_place_ensemble`, `_run_stage_a_smoke` | `g11_bg_runner.py:4682, 4665, 4721` |
| Anti-cheat place-shuffle (must survive any fix) | `--shuffle` permuted place→value control | the de-risk + Stage-B harness |
| Per-region edit PATTERN (if A1-global or B2 needs `sim/`) | `89b8d909` (per-region homeostasis), per-region NMDA mask | the committed, byte-identity-proven template |

---

## 5. Recommended CHEAP-FIRST de-risk (one experiment, runner-only, tells us the most)

**The single experiment:** make the self-org **reproducible** and run the **already-shipped robustifier**, then measure 3/3.

1. **Determinism (pick the cheapest scoping):**
   - *Fastest to prototype:* **A3** — run `_run_place_selforg` under `SIM_BACKEND=numpy` (deterministic, probed 1/100), freeze, transfer the `landmark_to_place` weights into the CuPy nav bridge via the existing checkpoint/`set_pathway_weights` path. No `sim/` edit.
   - *If you'd rather stay on GPU:* **A1 scoped to the place matvec** during STEP-1 (pre-transpose the place CSR once; `1/1` deterministic, `allclose`).
2. **Robustify:** run nav-scale Stage-B with **`--enable-critic-homeostasis`** (afferent+critic, the de-risked config) at the validated operating point (`--n-place 800 --selforg-steps 2000 --value-train-trials 40`, single goal, cap 40, clean reset).
3. **(Synergy, optional) add the sibling doc's Option 1** (`--pair-then-reward` Yagishita timing) if homeostasis alone leaves a residual on the weakest draw.

**Pass criteria (must hold to call it robust):**
- **(R-A) Reproducibility:** the SAME seed, run TWICE, yields a **byte-identical place code** (STEP-1 diff-cos and sparsity identical to all printed digits across two invocations). This *closes the non-determinism gap directly* and is the thing the current engine cannot do.
- **(R-B) Robustness 3/3:** seeds 42/43/44 ALL pass Stage-B gates 2a (critic ≥5 Hz near) + 2b (near ≥3× far) + 2c (`w_near ≥ 2× w_far`) — **including the documented weak draw** — with `w_near` reaching a fireable value on every seed.
- **(R-C) Mechanism, not banking:** report that the win comes from the critic firing into range on *every* draw (homeostasis), NOT from selecting draws — i.e. you did NOT reseed.

**Why this is the highest-information single experiment:** it disentangles the two confounded questions the current non-determinism fuses. Today, "3/3" could be luck-of-the-draw per invocation; with the place code pinned, a 3/3 is *attributable* — either homeostasis robustifies the readout (B1 works) or it doesn't (and you fall to B2/B4). You cannot learn this without first removing the non-determinism, which is why A and B must be done **together** in the de-risk.

---

## 6. Anti-cheat controls the de-risk needs (so "robustification" isn't a host shortcut or a banking trick)

1. **(NO reseed / NO banking) — the place map must be the FIRST draw, never selected.** Assert the self-org RNG is consumed once per seed; forbid any `while crit < K: reseed` loop in the shipped path. If a draw is weak, the *critic* (homeostasis/scaling) must rescue it — not a Python re-roll. This is the decisive control distinguishing B1/B2/B4 (legit) from B3 (cheat).
2. **(Place-shuffle still breaks V)** — the existing `--shuffle` permuted place→value control MUST drop gate-2c below 2× under the new protocol. Homeostasis/scaling lift *firing*; they must NOT let the critic learn V from "fired-on-any-drive." The value must ride on weights learned at the *rewarded* location.
3. **(Grading survives the robustifier)** — after homeostasis threshold-adapt (or synaptic scaling), the critic must still be GRADED near≫far (no place-blindness from a collapsed threshold or uniform up-scale). Test the far-cell readout explicitly; the de-risk reported afferent ~59 Hz near vs 0 Hz far — re-assert at nav scale.
4. **(No host V anywhere)** — V is read from `cp_firing_states` of `striosome_value`; δ is the SNc firing via `from_region_firing_signed`; `current_reward_signal=0.0` preserved. No Python value table, distance formula, or argmax-over-place-cells. (The sibling doc's §6 controls apply verbatim.)
5. **(Determinism asserted, not assumed)** — gate R-A (byte-identical place code across two invocations of the same seed) is itself an anti-cheat: it proves the fix removed the non-determinism rather than masking it with a lucky seed. For A3, additionally assert the NumPy-self-org → CuPy-transfer weights are byte-identical to what a CuPy forward pass reads (mirror the G.20 16-test pattern that pins pattern-regen byte-identity).
6. **(Actor untouched)** — confirm the homeostasis mask covers ONLY `vs_place_context`/`place` + `striosome_value` (it does, `:1076/:1107`), and that turning on global `enable_synaptic_scaling` (if B2 is used) is NOT what's scoped (it would scale the actor) — i.e. B2 requires the per-region scaling extension before it's anti-cheat-clean.
7. **(Timescale honesty)** — report the homeostasis adapt over the value-train window; if the critic only fires because the threshold drifted to a place-blind floor, that's control #3 failing — say so. Homeostasis must fire it *into a graded range*, not *into firing-on-everything*.

---

## 7. The synthesis (how A + B + the sibling doc compose)

- **A (determinism)** makes the experiment *legible* — a seed maps to one place code, so any robustness claim is attributable, and the anti-cheat R-A can be asserted.
- **B1 (per-region homeostasis)** makes the critic *fire into a graded range on any draw* — the readout normalizes out presynaptic-drive variance (the owner's stated lever), and it's already de-risked 3/3 with zero new code.
- **The sibling doc's Option 1 (Yagishita DA-after-pairing + up-state hold)** makes the *learning* deposit large, robust `w_near` steps once the cell fires — fixing the *credit-assignment* fragility that homeostasis alone doesn't address.

The cheap-first de-risk runs **A3 (or A1-scoped) + B1** and measures R-A/R-B/R-C. If that's 3/3, the blocker is resolved runner-side (A3+B1 are no-`sim/`-edit; A1-global and B2 are the only paths that touch `sim/`, and only if needed). If B1's threshold-adapt collapses selectivity (control #3), fall to **B2 (per-region synaptic scaling, byte-reviewable)**; if you want the goal-over-representation biology explicitly, add **B4 (strengthen, don't select)** — never **B3 (reseed)**.

---

## Appendix — load-bearing citations

| Claim | Source |
|---|---|
| Per-step drive is a TRANSPOSE SpMV `M.T @ fired` | `sim/bridge.py:5613, 5620` |
| `csr.T` → `csc_matrix`; `csc @ v` → `cusparse.spmv(transa=True, alg=CUSPARSE_MV_ALG_DEFAULT)` | live inspection of CuPy 13.6.0 `cupyx/scipy/sparse/_csr.py` + `cupyx/cusparse.py` (`spmv`) |
| `A.T @ v` 200/200 distinct; `A @ v` 1/1; `A.T.tocsr() @ v` 1/1 `allclose`; dense `A.toarray().T @ v` 1/1; scipy `A.T @ v` 1/100 | live engine probes, this session (n=800, density 0.05, fp32, RTX 3090, CuPy 13.6.0 / CUDA 12.9) |
| cuSPARSE deterministic SpMV exists ONLY for `CSR_ALG2`/`COO_ALG2` and ONLY `opA=NON_TRANSPOSE` | NVIDIA cuSPARSE 12.x generic-API docs |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` pins cuBLAS GEMM only (not cuSPARSE) | `g11_bg_runner.py:55-62`; the determinism is empirically still broken (probe above) |
| Per-region homeostasis edit (B1), byte-identity-proven, de-risk PASS 3/3 | commit `89b8d909`; `2026-06-08-navfaithful-derisk-FAIL-homeostasis-confound.md` + the afferent+critic PASS |
| Intrinsic homeostatic plasticity (excitability/threshold) | Desai et al. 1999; Turrigiano 2008 (Cell 135:422) |
| Synaptic scaling (multiplicative, preserves selectivity) | Turrigiano & Nelson 2004; Turrigiano 2008; catalog `feature-catalog.md:956, 3587` |
| Place cells over-represent goal/reward locations (B3/B4 biology) | Hollup et al. 2001 (J Neurosci 21:1635, annular water maze); Dupret et al. 2010 (Nat Neurosci 13:995); Gauthier & Tank 2018 (Neuron 99:179, "reward cells") |
| Over-representation is EXPERIENCE-DEPENDENT + TASK/STABILITY-DEPENDENT (why reseed is wrong, strengthen is right) | Hollup 2001 (builds as learned); JNeurosci 2022 PMC9097771 ("rapid goal representation… without place-field accumulation" when goals move); Cell Reports 2020 S2211-1247(20)30845-7 (distinct landmark vs reward over-representation); "Goal-Related Activity in Hippocampal Place Cells" PMC6672791 (annular-maze accumulation, absent in place-preference task) |
| Goal-directed/attended place fields stabilized by D1/D5 (catalog D.16) | `feature-catalog.md:1272`; Kandel 6e Ch 54 pp 1366–1367 |
| Divisive normalization (B5) as a canonical computation | Carandini & Heeger 2012 (Nat Rev Neurosci 13:51) |
| BRAIN-BASED-ONLY standard (host selection/banking = cheat) | CLAUDE.md "Standing standard: BRAIN-BASED ONLY"; AUTONOMOUS_STATE retractions (deterministic-copies/drive-echo) |
| Sibling protocol diagnosis (Yagishita timing; compose with this doc) | `2026-06-09-N9-robust-value-learning-diagnosis.md`; Yagishita-Kasai 2014 (Science 345:1616) |

**Sources (web):**
- [cuSPARSE Generic API — deterministic SpMV (CSR_ALG2 / COO_ALG2, NON_TRANSPOSE only)](https://docs.nvidia.com/cuda/cusparse/generic-api/generic-api-functions.html)
- [cupyx.scipy.sparse.csr_matrix — CuPy 13.6.0 docs](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupyx.scipy.sparse.csr_matrix.html)
- [Distinct Mechanisms of Over-Representation of Landmarks and Rewards in the Hippocampus — Cell Reports 2020](https://www.cell.com/cell-reports/fulltext/S2211-1247(20)30845-7)
- [Goal-Related Activity in Hippocampal Place Cells — PMC6672791 (Hollup annular-maze accumulation; task-dependence)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6672791/)
- [Spatial Learning Drives Rapid Goal Representation in Hippocampal Ripples without Place Field Accumulation — JNeurosci 2022, PMC9097771 (moving-goal caveat)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9097771/)
