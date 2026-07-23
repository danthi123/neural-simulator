# gap#4 deep-credit — AWS GPU re-validation SPEC (READ-ONLY scoping, 2026-07-23)

**Bottom line up front: the premise is stale, and the honest recommendation is DO NOT spend AWS GPU on this.**
The seed bug is already **fixed + verified**; the one genuinely-confounded arc already **re-ran clean → NOT-GO**;
and every current gap#4 deep-credit *result* (both the positives and the negative) is **numpy CPU**, not GPU-bound.
There is **no ready-to-launch, GPU-bound, seed-confounded gap#4 result waiting to be re-run.** Detail + the one
genuine GPU candidate (with its strong caveats) below.

---

## 0. What the record actually says (read the runners' own logic, not the headlines)

Three facts, each verified in code / git, overturn the task's framing:

1. **The seed bug is FIXED (commit `9471908a`, after the 2026-07-17 finding) and VERIFIED.**
   `_gnw_d1_spiking_bdsp_derisk.py` now sets `cfg.seed = seed` at all three bridge-construction sites
   (lines 652 / 706 / 762). `tests/test_determinism.py::TestSubstrateActuallySeeded` (line 334) pins it with a
   two-process threshold hash. The board (`GAP_CLOSURE_MISSION.md:344`) records: *"seed-fix VERIFIED … audit found
   NO runner still carrying the buggy `actual_seed_used=` pattern."*

2. **The genuinely-confounded arc already re-ran → NOT-GO.** Commit `9471908a`'s own message: *"the
   genuinely-confounded arc (`_semantic_inheritance` via `_onbridge_eprop_port`) was already fixed + re-run clean
   (NOT-GO)."* And the D1/BDSP **headline** numbers (held-out 0.664 vs oracle 0.958) *"come from a numpy Stage-B
   reference path seeded via `np.random.default_rng(seed)`, so they were NOT confounded — only three small Stage-A
   bridge smokes were."* So even the headline that looked at-risk was never the confounded number.

3. **Every current gap#4 deep-credit result is numpy CPU (`sim.dendritic_mlp.DendriticMLP`), not a cupy
   `SimulationBridge`.** Grep of all `*gap4*/*credit*/*bdsp*/*eprop*` runners: the accuracy runners
   (`_gap4_credit_vs_reservoir_mnist_derisk`, `_gap4_sparse_hidden_credit_derisk`, `_gap4_bdsp_faithful_credit_derisk`,
   `_onbridge_eprop_port_derisk`, `_deep_eprop_binder_bundling_derisk`) build **zero** `SimulationBridge`s and default
   `SIM_BACKEND=numpy`. Their own logs say *"CPU/numpy, coexisting with the fluency training."* The word "on-bridge"
   in several names is a **misnomer** — they emulate the sparse spiking op-point with a rate `DendriticMLP`.

**The board's own gap#4 verdict** (`GAP_CLOSURE_MISSION.md:758`): on-bridge deep-credit **learn-to-accuracy** is a
**CONFIRMED clean NEGATIVE** (BDSP 0.55/0.52/0.50 ≈ lesion ≪ credit-independent reservoir 0.765), a **DEPRIORITIZED
parallel frontier**. The *positive* reframe (`:70-71`) — the credit RULE builds deep accuracy, beats a reservoir —
is explicitly the **rate `sim.dendritic_mlp`** result (MNIST 6/6, faithful-BDSP 3-seed at spiking sparsity), i.e.
CPU.

---

## 1. WHICH single de-risk is THE key deep-credit result — and its true substrate

There are two candidates; **neither is a GPU-bound-confounded result the way the task assumed.**

### (A) The current highest-value POSITIVE, currently BELOW the 6-seed bar — but CPU, not AWS
`research/runners/_gap4_bdsp_faithful_credit_derisk.py` — a faithful numpy replica of the exact on-bridge BDSP rule
(`sim/kernels.py::fused_bdsp_update` M1.2: coincidence gate `Ẽ_pre·E_post` + sigmoid-baseline credit
`sigmoid(β·apical)−P̄`) on MNIST, showing **FA/BDSP-credit BEATS a frozen reservoir even at 2–5 % spiking
sparsity**. This is the load-bearing evidence that *the on-bridge negative is an op-point/LR issue, not the rule*.
It is at **3 seeds (42/43/44)** and the mission bar is 6. **But it is numpy `DendriticMLP` — it runs on CPU in
minutes and does not need AWS at all.**

Command to firm it to 6-seed (LOCAL CPU, or a cheap CPU box — NOT a GPU job):
```bash
SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=8 python -u -m research.runners._gap4_bdsp_faithful_credit_derisk \
    --seeds 42 43 44 100 101 102 --hidden 256 --depth 2 --fracs 1.0 0.1 0.05 \
    --p0 0.30 --beta 1.0 --n-train 8000 --n-test 2000 --epochs 15 --batch 64 --lr 0.03 \
    --out research/findings/raw/gap4/bdsp_faithful_6seed.json
```
(Note: `--lr 0.3` is the runner default and produced a dense→chance artifact; the board's own fix is **`--lr 0.03`**,
which recovered dense to 0.810. Use 0.03.) Seeding here is via `np.random.default_rng(seed)` + `DendriticMLP(seed=)`;
it **never touches the bridge**, so the 2026-07-17 seed bug never applied to it.

### (B) The only genuinely GPU-bound deep-credit piece — but it is NOT a packaged CLI, and it is a deprioritized negative
`research/runners/_gnw_d1_spiking_bdsp_derisk.py` — the D1/BDSP arc (the "9-findings" runner on the unseeded list,
now seed-fixed). It is the one deep-credit runner that genuinely constructs a `SimulationBridge` (3 sites, 25 BDSP
kernel refs). **However, read its own docstring:** Stage A = CPU multiplexing check + small bridge smokes; Stage B's
PRIMARY arm is *"a numpy REFERENCE of the EXACT `sim/` rule … the fast CPU smoke the builder validates."* The
genuinely GPU-bound part — *"the full 384-width GPU multi-seed … the fully-on-bridge net training"* — is named
**three times** as *"the CONTROLLER's GPU run"* (docstring lines 32–33, 63–64, and code line 872). **It is not wired
as a single reproducible command.** The flags `--hidden 384 --backend cupy` widen the Stage-B numpy reference and
run the Stage-A bridge smokes; a fully-on-bridge spiking net trained to accuracy would need to be **built** first.

If — against the recommendation — a GPU job is still wanted, the closest existing invocation is:
```bash
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m research.runners._gnw_d1_spiking_bdsp_derisk \
    --seeds 42 43 44 100 101 102 --hidden 384 --rule microcircuit --depth 2 \
    --epochs 300 --lr 0.5 --batch 128 --beta 1.0 --p0 0.30 --backend cupy \
    --json research/findings/raw/gap4/gnw_d1_bdsp_6seed_cupy.json
```
Use `--rule microcircuit` (the SST-cancellation "clean-error" variant the board names as the intended fix) rather
than the default `burstprop`. **Expected outcome: it CONFIRMS the characterized result** (held-out ≈ 0.66 < oracle
≈ 0.96, credit ≈ lesion ≪ reservoir on the accuracy sub-thread; mechanism-forms-but-doesn't-reach-accuracy) — it
does **not** produce a GO. Its value is "put the last on-bridge confound question to bed at the 6-seed bar," not
"unlock the enabler."

**Recommended single de-risk if you insist on one:** neither as an AWS GPU job. Firm **(A)** to 6-seed on CPU
(the actual enabler evidence), and treat **(B)** as an optional, low-value, local-GPU-when-free confirmation.

---

## 2. Is the seed fix APPLIED?

- **(B) `_gnw_d1_spiking_bdsp_derisk.py`: YES, applied + verified.** `cfg.seed = seed` at lines 652 / 706 / 762
  (alongside `cfg.actual_seed_used = seed`), committed in `9471908a`, guarded by
  `tests/test_determinism.py::TestSubstrateActuallySeeded`. No one-line fix needed.
- **(A) `_gap4_bdsp_faithful_credit_derisk.py`: N/A — it has no bridge.** It seeds via `np.random.default_rng(seed)`
  (data) and `BdspNet(sizes, seed=seed)` → `DendriticMLP(seed=seed)` (weights). The unseeded-substrate bug only
  affected `cp_neuron_firing_thresholds` on a `SimulationBridge`, which this runner never builds. Nothing to fix.

So for **both** candidates the seed question is already closed. There is no pending one-line fix for the controller
to apply before launch.

## 3. GO gate + load-bearing anti-cheats (how to tell real credit from a confound)

For **(A) faithful-BDSP** (the enabler evidence): the GO comparison is **`bdsp`/`fa_linear` accuracy > `reservoir`
accuracy by > 0.01 at each sparsity, all 6 seeds** (the runner prints `bdsp>RES:{...}` per row and a mean-over-seeds
SUMMARY). The **RESERVOIR arm (frozen random hidden + trained readout) is the load-bearing control** — it is the
credit-independent baseline; if credit merely matched it, the "value is the trainable readout, not credit-training
the hidden" (the project's R3 reservoir reframe) would stand. The 3-seed result: at 5 % sparsity FA 0.753 /
RES 0.360 (+0.267); the gap must hold and (per the trend) GROW with sparsity across the 6 seeds.

For **(B) `_gnw_d1`** (the on-bridge mechanism): its 7 pre-registered anti-cheats (docstring lines ~35–45) are the
right gate, and they are exactly the controls the seed-finding flagged:
1. **fixed-vs-learned feedback** — Y fixed-random, asserted never written / never == a forward `W`/`Wᵀ` (no weight
   transport).
2. **permuted-error/label** — shuffle `y` → held-out ~chance (rules out leakage/memorization).
3. **wrong-sign apical** — negate the burst deviation → held-out ≤ chance+0.05 (anti-learns; proves the apical sets
   the credit sign).
4. **apical-lesion** (Y=0 → P≡P0 → no credit) — collapses to the no-credit floor; linear probe ~0.5.
5. **the RESERVOIR / freeze-hidden control** — the decisive one the deep-credit gate *"never had"* until commits
   `92b7c507`/`b8c7f36f` added `--freeze-hidden` as a first-class DEFAULT-ON arm + CI guard, *"so a fixed random
   reservoir + logistic regression can never again pass as deep credit."* **Deep credit is real only if FULL (credit
   trains the hidden) BEATS FROZEN/RESERVOIR (hidden random, only readout trains) by a real margin, at the same
   seed** — the exact FULL-vs-FROZEN comparison the unseeded-neuron bug confounded (`±0.33` swing ≈ 3× the `+0.111`
   effect). Post-seed-fix, FULL and FROZEN finally share the same neurons, so the comparison is single-variable.
   **The `train_layers` isolation the seed finding flagged as "never invoked"** is the same idea: FROZEN must
   actually stop training the hidden (only the readout) — verify the frozen arm's hidden weights are byte-unchanged
   across training, else "FROZEN" secretly trained and the control is void.

**The verdict logic to trust:** a genuine GO = FULL/credit **>** reservoir/frozen by a margin **larger than the
seed-to-seed spread**, with permute→chance and lesion→floor. Per the board, on the learn-to-**accuracy** sub-thread
this gate is **not** met (credit ≈ lesion ≪ reservoir) — that is the confirmed clean negative. On the **mechanism**
(directed credit reaches the hidden, apical-load-bearing, no transport, P0 moat holds) it **is** met — but that is
"the rule ports to spikes," not "it trains a classifier to accuracy."

## 4. Is it GPU-bound? Runtime + AWS cost

- **(A) faithful-BDSP / MNIST / sparsity: NOT GPU-bound.** Pure numpy `DendriticMLP`, MNIST 8000×15 epochs,
  hidden 256. The 3-seed run completed on local CPU alongside the fluency training. 6-seed firming ≈ **~1 CPU-hour
  total** (order minutes/seed). **AWS cost: $0 needed** — run locally on the 20-core box, thread-limited so it never
  starves the training. If offloaded, a `c7i.2xlarge` (CPU, ~$0.36/hr) for ~1 hr ≈ **$0.40**. A GPU instance would
  be *wasted* — the code paths are numpy and won't touch the GPU.
- **(B) `_gnw_d1` fully-on-bridge 384-width: genuinely GPU-bound IF built**, but unpackaged. The task is tiny
  (10-bit boolean, small dataset) so per-seed wall-clock is likely **minutes to low-hours** even on the spiking
  bridge; 6 seeds ≈ **a few GPU-hours** (SMOKE-TIME ONE SEED FIRST — this is an estimate, the fully-on-bridge net
  isn't a run of record). On `g5.xlarge` (A10G, ~$1/hr) that is **~$3–10**; on `g4dn.xlarge` (T4, ~$0.53/hr)
  **~$2–6**. Cheap in dollars — but see §6 for why the dollars are not the issue.

## 5. Dependency to install on the AWS DL-Base AMI

Only relevant for candidate **(B)** (the GPU path). On an AWS Deep Learning Base GPU AMI (Ubuntu, CUDA 12.x
preinstalled):
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install cupy-cuda12x            # matches the local .venv (project memory: cupy-cuda12x[ctk], py3.11)
pip install -r requirements.txt     # h5py, numpy, scipy, etc. (the sim engine deps)
# Data/assets needed on-box: none for _gnw_d1 (task is generated); for (A) MNIST it needs data/mnist.npz.
export SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import cupy; cupy.zeros(1)+1; print('cupy OK', cupy.cuda.runtime.getDeviceCount())"
# then the §1(B) command; SMOKE ONE SEED and hash cp_neuron_firing_thresholds twice to confirm determinism first.
```
`SimulationBridge` runs on-device once `SIM_BACKEND=cupy` and CuPy sees a GPU (`sim/backend.py` auto-detect). No
`sim/` edit required. The repo is the sole source of truth (project memory: E: drive is gone) — `git clone` from
origin, do not expect any external asset store.

## 6. Reasons it should NOT be the AWS job (the load-bearing flags)

1. **The confounded result you'd re-run doesn't exist as a pending GPU job.** Seed bug fixed+verified; the one
   confounded arc already re-ran NOT-GO; the D1/BDSP headline was a seeded numpy reference, never confounded.
2. **The enabler evidence (the credit RULE builds deep accuracy / beats a reservoir) is CPU numpy.** Firming it to
   the 6-seed bar needs **no GPU** — it already coexists with the fluency training on local CPU. Sending it to an
   AWS **GPU** instance would run the same numpy code paths with the GPU idle (the runners `setdefault
   SIM_BACKEND=numpy`).
3. **The genuinely GPU-bound piece is a deprioritized, characterized NEGATIVE**, and is not even packaged as a
   run-of-record (it's "the controller's GPU run"). Spending GPU-hours to re-confirm a negative — that the board has
   already decided is a *parallel* frontier, with the emergence engine proceeding on the reservoir/shallow-readout +
   learned-input path — is low leverage.
4. **The real GPU contention is gap#1 fluency training** (owns the local GPU ~3–4 days). If the goal of going to AWS
   is to keep the roadmap moving while the local GPU is busy, the **highest-value AWS GPU job is a second gap#1
   fluency-training shard** (genuinely GPU-bound, on the critical path), *not* the gap#4 re-derisk. That is the
   recommendation to put to the owner.

**Net recommendation:** (i) firm faithful-BDSP to 6-seed on **local CPU** now (the enabler evidence, ~1 hr, $0);
(ii) do the `_gnw_d1` fully-on-bridge confirmation on the **local GPU when it frees** (or skip — it's a deprioritized
negative); (iii) reserve any **AWS GPU** budget for a gap#1 training shard, which is what is actually GPU-bound and
on the critical path.
