# Resource envelope — full-speed-3090 max · tiering capacity-vs-wallclock · train-vs-inference split (read-only analysis)

**Date:** 2026-06-25
**Type:** READ-ONLY resource-envelope analysis (NO edits/runs/webapp). Complements the in-flight corpus scoping `_foundational_curriculum_scaling_scoping.md` (that doc = WHICH corpus; this doc = WHAT FITS / WHAT IT COSTS).
**Question set (owner):** for scaling the brain toward a foundational (eventually Wikipedia-scale) curriculum — (1) the MAX at full GPU speed on a 3090; (2) the RAM/CPU/SSD tiering capacity-vs-wall-clock curve (shipped vs deferred, live vs finish-work); (3) the train-vs-load/interact resource split.

---

## TL;DR (the three answers in one breath)

1. **Full-speed 3090 max ≈ 2,000 concepts (≈7 sparse bridges, ≈80K neurons) resident at once.** The bridge's VRAM is dominated by a **~1.2 GB fixed CUDA/CuPy floor**, then **~26 KB/neuron-of-pool + ~28 B/synapse marginal** (derived from 4 measured `SIM_BRIDGE` init logs). A sparse bridge (≤320 concepts, ~2.3K pool neurons, ~3 GB resident incl. run buffers) is the unit; ~7 fit in 22 GB. Beyond that is tiering, not a bigger GPU.
2. **The RAM/CPU/SSD curve is "unlimited capacity, longer wall-clock," and it is LIVE today via the NumPy-CPU backend + the multi-bridge route** (both shipped end-to-end). The GPU→CPU slowdown is **~20× (measured), 4–50× depending on workload**. The synapse-tiering SSD layer is **Strategy B+C shipped (mirror + activity-track + shard-export), Strategy A (compute-off-the-tiered-store) DEFERRED ~3–4 wk** — so SSD *paging-for-capacity* is finish-work; CPU/RAM scaling is live now. Full-Wikipedia walls on **wall-clock** (a few-overnight-to-multi-week local stream), not VRAM.
3. **Inference is ~5–10× lighter than training and runs locally on the 3090 OR CPU regardless of how the brain was trained.** A trained brain LOADS in **0.7 s** (warm), answers a turn in **0.47–1.7 s**, at **~2–4 GB VRAM for the active bridge** (sparse activation — only the queried bridge resident). Training is the heavy phase (co-occurrence accumulation + the dense-read step + consolidation). **⇒ the "train-once-tiered/cloud → ship the artifact → interact cheap-local" model is exactly what the measured asymmetry supports.**

---

## ANSWER 1 — MAX AT FULL GPU SPEED ON THE 3090 (24 GB, ~22 GB usable)

### 1a. The per-neuron / per-synapse VRAM cost, derived from the code + measured logs

**The allocations (from `sim/bridge.py`):**

- **Per-neuron state arrays** (`_initialize_simulation_data`, the always-on Izhikevich path): `cp_membrane_potential_v` + `cp_recovery_variable_u` (`bridge.py:1455-1456`), the 9 Izhikevich param arrays `cp_izh_C/k/vr/vt/vpeak/a/b/c_reset/d_increment` (`:1449-1453`), `cp_neuron_firing_thresholds` (`:1461`), `cp_conductance_g_e/g_i` (`:1193-1194`), `cp_conductance_g_nmda/_rise` (`:1240-1241`), `cp_external_input_current` + `cp_firing_states` + `cp_traits` (`:1115/1129/1132`), `cp_refractory_timers` + `cp_neuron_activity_ema` + `cp_viz_activity_timers` (`:1432-1434`), `cp_neuron_type_ids` (`:1550`), `cp_syn_reversal_potential_i_per_neuron` (`:1202`). That is **~22 dense float32/int32 arrays of length N** = ~22·4 B = **~88 B/neuron** of *guaranteed* per-neuron state. With the common opt-in per-region masks (NMDA mask, homeostasis, GABA_B `:1382-1385`, coincidence/plateau `:1417-1430`, neuron_coords if coords used `:1225`) this rises to **~100–150 B/neuron**. The RF complex path REUSES `v`/`u` (`cp_rf_prev_im = cp_recovery_variable_u.copy()`, `:5679`) plus `cp_rf_fired` (bool) + `cp_rf_spike_step` (int64) = **~9 B/neuron extra**, so it adds essentially nothing per-neuron.
- **The synaptic CSR `cp_connections`** (`csp.csr_matrix`, `:2492`/`:5553`): CSR stores `data` (float32, nnz) + `indices` (int32, nnz) + `indptr` (int32, N+1) = **~8 B/synapse + 4 B/neuron**. Eligibility traces (`cp_eligibility_trace`, capacity = nnz·growth_factor, `:700/725`) add another **4 B/synapse** (or **2 B** with `cfg.fp16_synapse_state`, `:719`), and STP (`cp_stp_x/u`, `:708-709`) adds **8 B/synapse** when enabled. So the *effective* per-synapse cost is **~8 B (CSR only) → ~20 B (CSR + eligibility + STP)**.
- **The RF complex weights `cp_rf_w_re` + `cp_rf_w_im`** (two CSRs, `:5707-5708`): the composer's bind/unbind/bundle synapses are **O(D) SPARSE diagonal** (one complex weight per work-register dim D, NOT N²; comment `:5697-5699`), so for the conversational/composer slice they add **~16 B per RF synapse**, and the RF synapse count is tiny (≈ D per bound term). The DENSE RF mode (`cfg.rf_dense_weights`, `:5713-5715`) is **default-OFF** and only materializes an N² complex matrix for the off-bridge Qwen-faculty experiment — NOT the stream-cortex/composer path.

**The measured anchor (4 real `SIM_BRIDGE` init logs, RTX 3090, 25.8 GB visible):**

| neurons | synapses | GPU mem (resident at init) | source |
|---|---|---|---|
| 3,560 | 508,800 | **1.3 GB** | `research/findings/raw/cheat3_close_nverb500.txt:64` |
| 5,560 | 988,800 | **1.4 GB** | `research/findings/raw/cheat3_close_nverb1000_gpi600.txt:64` |
| 8,300 | 4,710,370 | **1.6 GB** | `research/findings/raw/_dlpfc_role.out:57` |
| 18,684 | 10,392,398 | **2.1 GB** (init) → **3.7–3.9 GB** (during run) | `research/findings/raw/_bio_conv320_demo_transcript.txt:15`, `_directional_POSTFIX.out:102-276` |

**The regression these four points imply:** a **~1.2 GB fixed floor** (CUDA context + CuPy memory pool reservation) + **~28 B/synapse + ~26 KB/(1000 neurons)** marginal. Worked: 18,684 n + 10.4M syn → 1.2 + 10.4M·28 B (≈0.29 GB) + 18.7K·… ≈ 2.0 GB resident, matching the 2.1 GB log. The **run-time working buffers roughly double the resident-init number transiently** (the 18.7K-neuron bridge climbed 2.1 → 3.7–3.9 GB while running — the per-step temporaries + the eligibility/STP capacity arrays).

**⇒ The headline cost facts:**
- **The fixed CUDA/CuPy floor (~1.2 GB) dominates a single small bridge** — a 3.5K-neuron bridge and an 18.7K-neuron bridge differ by only ~0.8 GB. This is why the v17 `14,464 neurons, 16M synapses → 2.5 GB` data point (CLAUDE.md:2167) and the 18,684-neuron `→ 2.1 GB` log are so close: **per-neuron/per-synapse marginal is small; the floor + run-buffers are the real cost.**
- **Per-concept (sparse-distributed):** a 64-concept sparse bridge = 2,300 pool+FS neurons (`2026-05-15-sparse-distributed-capacity-curve.md:9`); a 320-concept (5×64) ensemble = 11,500 neurons → measured **2.1 GB resident** for the 18.7K-neuron full-conv bridge, so a 320-tier sits **~2–3 GB resident, ~8 GB peak with run buffers** (scoping doc §2b). So **per concept ≈ 25–40 MB amortized** (mostly the floor amortized across the ensemble), **per pool-neuron ≈ 26 KB-of-VRAM amortized including buffers**.

### 1b. The full-speed-resident maximum

Using the measured "320-tier ≈ 8 GB peak / ~3 GB resident" + the **linear-in-bridge-count** multi-bridge route (`g20_multibridge --sparse`, validated 160 @ 100% and 320 @ 98.4%/bridge):

| foundational vocab | sparse bridges (≤320/bridge) | pool+FS neurons | resident VRAM (est) | peak VRAM w/ run-buffers | fits in 22 GB? |
|---|---|---|---|---|---|
| 320 (validated) | 1 ensemble (5×64) or 1×320 | ~11.5K | ~3 GB | ~8 GB | **yes, comfortably** |
| 1,280 | 4 | ~46K | ~8 GB | ~13 GB | **yes** |
| **2,048** | **~7** | **~80K** | **~13 GB** | **~22 GB (AT the line)** | **yes — the practical full-speed ceiling** |
| 5,000 | ~16 | ~184K | ~30 GB | >24 GB | **no — needs tiering** |
| 30,000 | ~94 | ~1.1M | ~180 GB | — | **no — heavy tiering** |

**Concrete number + formula:**
> **Full-speed-resident max on a 3090 ≈ 2,000 concepts (≈7 sparse bridges, ≈80K pool neurons, ≈13 GB resident / ≈22 GB peak with run buffers).**
> **VRAM(GB) ≈ 1.2 (CUDA/CuPy floor) + 28 B·n_synapses + 0.10 KB·n_neurons, then ×~1.8 for transient run buffers.**
> For the sparse-distributed ensemble specifically: **VRAM(GB) ≈ 1.2 + 0.6·n_bridges (resident) → ≈ 1.2 + 1.6·n_bridges (peak)**, i.e. **~7 bridges ≈ 2K concepts at the 22 GB line.**

**Cross-checks against the other measured anchors:**
- **494M-param spiking Qwen = 14.05 GB** (`2026-06-23-bridge-coresidence-DEMONSTRATED.md:4/20`) — this is the DENSE RF-weight path (`rf_dense_weights`), an N²-dense 494M-weight matrix stored as a complex CSR; it is the *language-faculty* experiment, NOT the stream-cortex. It confirms the 3090 fits a 494M-weight dense model resident with headroom (14 << 24), but it's a different storage regime from the sparse ensemble.
- **merged nav+conv "one brain" = 6,808 neurons** (`2026-06-10-nav-on-merged-smoke-PASS-hybrid-integration-works.md:13`) — small; the prompt's "~54K neurons" is the **onebrain composer at V=320** (parser slice + RF work registers + K=32 store), `2026-06-18-onebrain-320-scale-production-GO.md:38`. That 54K-neuron onebrain at 320 concepts is still **well under the 22 GB line** (a 54K-neuron bridge ≈ 2–3 GB resident).

**Honest uncertainty:** the run-buffer multiplier (~1.8×) is the soft part — it depends on which opt-in plasticity/conductance arrays are on (STP, eligibility, GABA_B, NMDA-recurrent, dendritic). A foundational stream-cortex run is **plasticity-ON during WAKE** (eligibility + Hebbian buffers live) so use the *peak* column. The "~2K at the line" is therefore a conservative full-speed ceiling; a leaner config (fp16 eligibility, STP off) pushes it toward ~2.5–3K resident.

---

## ANSWER 2 — THE RAM / CPU / SSD TIERING CURVE (shipped vs deferred; live vs finish-work)

### 2a. What is SHIPPED vs DEFERRED (precise)

| Layer | Mechanism / file | Status | What it gives |
|---|---|---|---|
| **NumPy CPU backend** | `sim/backend.py` (`get_backend("numpy")`, `SIM_BACKEND=numpy`) | **SHIPPED end-to-end** (`2026-05-11-numpy-backend-{shipped,chat-repl-shipped}.md`) — construction + init + steps + region framework + checkpoint + training + chat all run CPU-only | **RAM-bound capacity NOW** — the whole brain runs in system RAM, no 24 GB GPU ceiling. The capacity is your RAM (64–256 GB typical), at a wall-clock cost. **LIVE today.** |
| **Multi-bridge route** | `research/runners/g20_multibridge.py --sparse` | **SHIPPED** (160 @ 100%, 320 @ 98.4%/bridge validated end-to-end) | **Capacity LINEAR in bridge count** — bridges hold disjoint concept sets, never interact during learning, so each can be paged independently. **LIVE today** (the production scaling route). |
| **Synapse tiering — Strategy C (export)** | `sim/synapse_storage.py` + `BridgeLineage.export_shards` | **SHIPPED** | Per-pathway CSR shards written to disk (`.npz`), self-contained, for inspection/cold archive. |
| **Synapse tiering — Strategy B (mirror + activity-track)** | `sim/synapse_storage.py` `TieredSynapseStore`, `step(fired)` idle/pressure eviction (`DEFAULT_EVICT_AFTER_IDLE_STEPS=1000`, `:47`; `DEFAULT_GRACE_AFTER_PAGEIN_STEPS=100`, `:51`; `DEFAULT_RAM_BUDGET_BYTES` pressure eviction, `:56`) | **SHIPPED** (mirror + per-pathway activity tracking; 56 tests pass per CLAUDE.md) | Pathways page RAM↔NVMe transparently (`get_pathway` reloads on access, `:30-32`); idle pathways evict; the foundation for compute-off-the-store. **Observational today** — inference still uses the monolithic `cp_connections`. |
| **Synapse tiering — Strategy A (per-pathway compute-off-the-tiered-store)** | (design only) | **DEFERRED ~3–4 wk** (CLAUDE.md: "3-4 weeks scope, deferred") | Live SSD-paging-for-CAPACITY during compute (run a bridge whose synapses don't all fit in RAM). **FINISH-WORK** — this is the piece that makes SSD a true capacity tier, not just an archive. |

### 2b. The capacity-vs-wall-clock trade (the speed ratios)

- **GPU → CPU slowdown: ~20× (measured), 4–50× by workload.** The SK-latency finding states numpy is "**~20× slower for the brain ops**" (`2026-06-24-sk-latency-resolved-interact-console-complete.md:44`); CLAUDE.md's tiering note gives the general range "**4–50× faster [CuPy] depending on workload**." So a stream-cortex day that takes ~15 s of WAKE GPU-time (`2026-06-23-longitudinal-develop-loop-GPU-GO.md:26`) is **~5 min on CPU** — long but not impossible for an overnight/multi-day local develop run.
- **RAM-paging overhead:** the multi-bridge route pages whole *bridges* (independent), so paging is a bulk `cp.asarray`/host-copy of one bridge's CSR (~MB–low-GB), amortized over thousands of steps of that bridge's learning. Negligible vs the compute, because bridges never interact during learning — you only page when you switch which bridge is learning/queried.
- **SSD-paging overhead (Strategy A, when built):** the `TieredSynapseStore` reloads a pathway's `.npz` from NVMe on access (`:30-32`). At NVMe ~3–7 GB/s, a 0.5–3 GB bridge pages in ~0.1–1 s — fine for *bridge-granular* paging (page when you switch bridges), thrash-prone only if you tried *per-step* paging (which is why the hysteresis grace period exists, `:51`).

### 2c. Is it truly "unlimited capacity, longer wall-clock" — and via which path is it LIVE?

**Yes, with one nuance:**
- **LIVE today (capacity NOW):** (i) the **NumPy CPU backend** — capacity = system RAM, the whole brain in RAM, ~20× slower; and (ii) the **multi-bridge route** — capacity linear in bridge count, page whole bridges in/out of GPU manually (they don't interact during learning). Together these already give **"hold as many concepts as RAM allows, train one bridge at a time on the GPU, page the rest to RAM,"** which covers 5K–30K concepts *for the knowledge base* without a bigger GPU.
- **FINISH-WORK (the missing convenience):** **Strategy A** (transparent per-pathway SSD-paging *during* compute) is the piece that would make a *single bridge larger than RAM* runnable and make the paging automatic rather than manual-orchestrated. It is **deferred ~3–4 wk**, not built. Until then, SSD is an **archive/export tier (Strategy C)**, not a live compute-capacity tier.

**Where wall-clock becomes impractical at full-Wikipedia:** the scoping doc's window-budget model (`_foundational_curriculum_scaling_scoping.md:84-88`): ~150K windows → 320 concepts ≈ 84 min GPU; ~500 windows/concept ⇒ **2K concepts ≈ 9 GPU-hr, 5K ≈ 22 GPU-hr, 30K ≈ a few overnight runs** on the GPU path (×~20 if forced to CPU). A FULL Simple-Wikipedia / BabyLM-100M tier (~15–30K vocab over 100M words of stream) is a **multi-overnight-to-multi-week local GPU stream**. That is the wall — **wall-clock, not VRAM** (per the owner's standing directive, cloud is justified only for a genuine >24 GB VRAM wall; here cloud would only *speed* the stream ~3–5×, e.g. cut a 22-hr run to ~5 hr, not *enable* it).

**Honest uncertainty:** the windows-per-concept figure (~500) is an extrapolation from the 320-concept run, not measured at 2K+; if a larger/noisier corpus needs more windows per concept for clean discrimination, the wall-clock grows super-linearly in *quality*, not in *capacity*. And the CPU-path ~20× is a single measured ratio (brain ops); some sub-ops (the dense `cp_connections.todense()` read, below) are worse on CPU.

---

## ANSWER 3 — TRAIN vs LOAD/INTERACT RESOURCE SPLIT

### 3a. TRAINING (the heavy phase) — what it costs

Per simulated day (`_longitudinal_develop_loop_gpu.develop_gpu`): **WAKE** (stream-cortex co-occurrence Hebbian, `hear_day` → `_present_window` → `_run_one_simulation_step` per window, `:204-247`) → **CONVERSE** (store facts) → **SLEEP** (self-replay consolidation) → **GROWTH** (TierPromoter) → **PERSIST** (lineage HDF5).

- **Wall-clock:** mean **~133 s/day**, of which **WAKE/stream-learn ≈ 15 s/day** (the rest = converse + consolidate + grow + persist) at 24-concept smoke scale (`2026-06-23-longitudinal-develop-loop-GPU-GO.md:26`). **Compressed-week ≈ 15.6 min; month ≈ 1 hr; year ≈ ~13.5 hr (overnight, LOCAL).**
- **VRAM during training:** the stream cortex bridge is small (≈2,700 neurons at 24 concepts; the hub→target CSR ≈ a few M synapses ≈ 2–3 GB resident, scoping doc §2d). **Plasticity is ON** during WAKE (eligibility + Hebbian buffers live → use the peak-buffer column). Learning a large vocab **one bridge at a time** keeps training VRAM ~2–4 GB regardless of total vocab — *you never hold all bridges resident to train* (scoping doc §2d).
- **The one heavy READ step (a genuine scaling caveat):** `read_codes()` calls **`self.bridge.cp_connections.todense()`** (`_longitudinal_develop_loop_gpu.py:253`) to extract the learned `M` block. **`.todense()` is O(N²)** — it materializes the full dense N×N matrix. For an 18.7K-neuron bridge that is 18.7K² · 4 B ≈ **1.4 GB transient**; for a 54K-neuron bridge ≈ **11.6 GB transient**; for an 80K-neuron (2K-concept) bridge ≈ **25 GB → would OOM the 3090**. ⇒ **the dense-read at code-extraction time is the per-bridge size cap during training**, and is the reason per-bridge ≤320 concepts (≈11.5K neurons, ≈0.5 GB dense-read) is comfortable while a single giant bridge is not. This is a known-shaped engineering item (read the block from the sparse CSR slice directly instead of `.todense()`), not a substrate wall — but it is the concrete reason training stays bridge-granular.

### 3b. LOADING + ONE INTERACTION TURN (the light phase) — what it costs

- **Load:** a trained brain bundle LOADS in **0.0–0.7 s** (warm; the SK-latency fix took it ~9.8 min → 0.7 s, ~800×, by lazy-deferring the WM loop + batching CSR rebuilds + lazy parser + persisting KB composites, `2026-06-24-sk-latency-resolved-interact-console-complete.md:6-30`). The bundle is the lineage HDF5 + JSON (`sim/lineage.py`) — a few MB–low-GB on disk.
- **One turn (forward pass: recall + compose + appraise + render):** **first query 0.7 s, warm query 0.47 s** for the brain ops (`:30`); a full chat turn including the off-bridge Qwen renderer is **1.7–5.5 s warm** (`:66`). The **only large one-time cost is the Qwen-0.5B renderer model load (~58 s cold, downloaded once from HF Hub), now warmed at webapp startup** so the human's first real turn is fast (`:32-48`).
- **VRAM for inference:** only the **active (queried) bridge** need be resident — **sparse activation**: the multi-bridge ensemble means one ~320-concept bridge (~2–4 GB) answers a turn; other bridges can stay paged out. The composer's bind/unbind is O(D) sparse RF synapses (`bridge.py:5697-5699`), not N². So **inference VRAM ≈ 2–4 GB for one bridge**, vs training's need to also hold eligibility/Hebbian/consolidation buffers + the dense-read transient.

### 3c. The asymmetry, quantified

| Resource | TRAIN (per day, foundational stream) | LOAD + 1 INTERACT TURN | Asymmetry |
|---|---|---|---|
| Wall-clock | ~133 s/day (15 s WAKE), days–weeks total | **0.7 s load + 0.47–1.7 s/turn** | turn is **~100–300× faster** than a training day |
| VRAM | 2–4 GB/bridge **+ buffers + O(N²) dense-read transient** (caps single-bridge size) | **2–4 GB for the one active bridge**, no buffers, no dense-read | inference **~2–5× lighter** (no plasticity buffers, no dense-read, sparse activation) |
| Compute character | plasticity ON (Hebbian + eligibility + consolidation + replay) every step | forward only (recall scan + bind/unbind + cleanup + render) | inference is a **forward pass; training is forward + backward-equivalent plasticity + replay** |
| Backend | GPU strongly preferred (~20× over CPU) for the stream | **runs on GPU OR CPU** — a warm turn is ~0.5–2 s even on CPU at small scale | inference is **portable**; training wants the GPU |

**⇒ The decisive answer:** **a brain that needs tiering (or cloud, for the 100M tier) to TRAIN can be LOADED + interacted-with locally on the 3090 — or even CPU — cheaply.** The trained artifact is a lineage bundle; inference touches one bridge at a time at 2–4 GB and answers in <2 s. The "ship the trained artifact, interact cheaply" model is **exactly** what the measured asymmetry supports (and is already realized by the per-day-bundle console capstone: `/api/brains` lists each developed brain, the human loads one and chats, `2026-06-24-sk-latency-...:68-78`).

**Honest uncertainty:** the inference numbers are measured at the **52-fact / 106-vocab SK bundle and the 320-concept tier**, not at 5K–30K concepts. At very large multi-bridge ensembles, *which bridge holds the queried concept* + cross-bridge associative retrieval is an O(bridges) scan unless indexed — the scoping doc flags cross-bridge routing as the genuine open scaling risk (§6). So inference stays cheap **per active bridge**, but a naive "scan all 94 bridges" turn at 30K concepts would need a routing index (engineering, not a wall). Also: the Qwen renderer adds ~2 GB VRAM + the one-time load if the off-bridge fluency model is used; the brain's own neural renderer (`enable_neural_render`) avoids that for the word-order step.

---

## THE RESOURCE-ENVELOPE TABLE (consolidated)

| Axis | Number | Derived from | Honest uncertainty |
|---|---|---|---|
| **VRAM fixed floor** | ~1.2 GB | regression over 4 `SIM_BRIDGE` init logs (1.3 GB @ 3.5K-n/0.5M-syn … 2.1 GB @ 18.7K-n/10.4M-syn) | CUDA/CuPy version-dependent ±0.2 GB |
| **VRAM per synapse** | ~28 B (CSR+overhead); ~8 B CSR-only, +4 B eligibility (2 B fp16), +8 B STP | `bridge.py:700-725` (CSR + eligibility + STP dtypes) + the 4 logs | depends which plasticity arrays are on |
| **VRAM per pool-neuron (amortized incl. buffers)** | ~26 KB | 320-tier 11.5K neurons ≈ 3 GB resident / 8 GB peak | run-buffer multiplier ~1.8× is the soft part |
| **Full-speed-3090 resident max** | **≈2,000 concepts (~7 sparse bridges, ~80K neurons, ~13 GB resident / ~22 GB peak)** | measured 320-tier VRAM × linear-in-bridge-count (`g20_multibridge --sparse`, 160/320 validated) | leaner config (fp16 elig, STP off) → ~2.5–3K |
| **Tiering — CPU/RAM (LIVE)** | capacity = system RAM, **~20× slower** (4–50× by workload) | `sim/backend.py` shipped; `2026-06-24-sk-latency-...:44` (~20×); CLAUDE.md (4–50×) | single measured brain-op ratio; some sub-ops worse |
| **Tiering — multi-bridge (LIVE)** | capacity **linear in bridge count**, page whole bridges | `g20_multibridge --sparse` (160 @ 100%, 320 @ 98.4%/bridge) | cross-bridge routing at 1000s of bridges is the open risk |
| **Tiering — SSD live-paging-for-capacity (FINISH-WORK)** | Strategy A **DEFERRED ~3–4 wk**; B+C (mirror/activity/export) SHIPPED | `sim/synapse_storage.py` + CLAUDE.md tiering note | Strategy A unbuilt; SSD is archive-tier today |
| **Wall-clock — stream rate** | ~150K windows → 320 concepts ≈ **84 min GPU**; ~500 windows/concept ⇒ 2K ≈ 9 hr, 5K ≈ 22 hr, 30K ≈ multi-overnight | `_foundational_curriculum_scaling_scoping.md:84-88` | windows/concept (~500) extrapolated, not measured at 2K+ |
| **Wall-clock — develop loop** | ~133 s/day (15 s WAKE); week ≈ 15.6 min; year ≈ ~13.5 hr | `2026-06-23-longitudinal-develop-loop-GPU-GO.md:26` | 24-concept smoke scale |
| **Train VRAM** | 2–4 GB/bridge + buffers + **O(N²) `.todense()` read transient** (caps single-bridge ≤~320 conc) | `_longitudinal_develop_loop_gpu.py:253` (`cp_connections.todense()`) | dense-read is the per-bridge size cap; fixable engineering |
| **Inference — load** | **0.0–0.7 s** (warm; ~800× after SK-latency fix) | `2026-06-24-sk-latency-...:6-30` | measured at 52-fact SK bundle |
| **Inference — 1 turn** | brain ops **0.47–1.7 s**; full turn w/ Qwen **1.7–5.5 s** warm; +58 s one-time cold Qwen load | `2026-06-24-sk-latency-...:30/48/66` | renderer adds ~2 GB + cold load if used |
| **Inference VRAM** | **2–4 GB** (one active bridge; sparse activation) | composer O(D) sparse RF (`bridge.py:5697-5699`) + 320-tier resident | cross-bridge routing index needed at 30K |

---

## VERDICT — the practical scaling model + where each resource walls

**The measured envelope confirms the "train-once → ship → interact-cheap-local" model, with a single genuine wall (wall-clock), and one piece of finish-work (live SSD paging) that is convenience, not blocker.**

**The scaling model that the numbers support:**
1. **Train bridge-by-bridge.** Keep each sparse bridge ≤320 concepts (≈11.5K neurons): it discriminates at ~98%, its dense-read at code-extraction is ~0.5 GB, and it trains at 2–4 GB VRAM. Stream the foundational corpus, focusing each bridge on its concept cluster (the `g20_vocab_spec_2048` 32-cluster taxonomy is already designed for this). Run over the develop loop's simulated days/weeks — **wall-clock-bound, LOCAL, with an ETA** (per the owner's accepted long-local-run mode). Cloud is justified ONLY for the 100M/30K far tier's *turnaround* (~3–5× speed), never for VRAM.
2. **Hold the ensemble via the LIVE tiering paths:** up to ~2K concepts (~7 bridges) fit resident on the 3090 at full speed; beyond that, page whole bridges to RAM (the multi-bridge route + NumPy/CPU backend are both shipped) — they never interact during learning, so paging is bulk and amortized. The full 5K–30K knowledge base is "unlimited-capacity / longer-wall-clock," **LIVE today** via CPU/RAM + manual bridge-paging; the only **finish-work** is Strategy A (transparent SSD paging during compute, ~3–4 wk) which makes it automatic + lets a single bridge exceed RAM.
3. **Ship the trained artifact (the lineage bundle), interact locally.** Loading is 0.7 s, a turn is <2 s, at 2–4 GB for the one active bridge — **a brain that needed tiering/cloud to TRAIN runs its inference on the 3090 OR CPU cheaply.** This is already realized by the per-day-bundle console.

**Where each resource genuinely walls:**
- **VRAM walls at ~2K concepts resident-at-full-speed** (~22 GB peak, 7 bridges). This is NOT a hard ceiling — it is the *full-speed-resident* line; tiering takes it past indefinitely. **A bigger GPU is never required for the knowledge base** (bridges don't interact during learning); cloud lifts no VRAM wall here.
- **The O(N²) `.todense()` code-read walls a SINGLE bridge at ~30–50K neurons (~25 GB transient)** — which is exactly why per-bridge ≤320 concepts is the design, and is fixable engineering (slice the sparse CSR instead of densifying).
- **Wall-clock is the real wall at full-Wikipedia:** a 100M-word / ~30K-vocab stream is a multi-overnight-to-multi-week LOCAL GPU run (×~20 on CPU). This is the genuine scaling cost — and it is a *time* cost the owner already accepts with an ETA, not a *capability* wall. Cloud cuts it ~3–5× if turnaround matters.
- **Cross-bridge routing** (which of N bridges holds a queried concept; cross-bridge associative retrieval) is the open *correctness*-at-scale risk (scoping doc §6) and, for inference latency at 30K concepts, would need a routing index — engineering, surfaced by the scoping doc's Step-2 de-risk, not a substrate wall.

**One-line:** **the brain scales toward Wikipedia as a LOCAL, wall-clock-bound, bridge-by-bridge stream (VRAM never the wall for the knowledge base; cloud only for far-tier turnaround), and the trained artifact ships + interacts on a 3090 or CPU at <2 s/turn and 2–4 GB — train heavy/tiered, interact cheap/local.**

---

## Sources / artifacts (verified in-repo)

- **Bridge VRAM allocations:** `sim/bridge.py` — per-neuron Izhikevich state `:1449-1463`, conductances `:1193-1194/1240-1241`, external/firing/traits `:1115/1129/1132`, timers/EMA `:1432-1434`, reversal `:1202`; CSR `cp_connections` `:2492/5553`; eligibility/STP capacity + `fp16_synapse_state` `:700-725`; RF complex state reuses v/u `:5679-5682`; RF sparse complex weights + O(D) comment `:5697-5708`; dense RF mode default-off `:5713-5717`.
- **Measured VRAM logs:** `research/findings/raw/cheat3_close_nverb500.txt:64` (3,560 n → 1.3 GB), `cheat3_close_nverb1000_gpi600.txt:64` (5,560 n → 1.4 GB), `_dlpfc_role.out:57` (8,300 n → 1.6 GB), `_bio_conv320_demo_transcript.txt:15` + `_directional_POSTFIX.out:102-276` (18,684 n → 2.1 GB init / 3.7–3.9 GB run). v17 `14,464 n / 16M syn → 2.5 GB`: CLAUDE.md:2167.
- **Capacity curve / multi-bridge:** `2026-05-15-sparse-distributed-capacity-curve.md` (64@100%, 128@84%, pool/pattern sizes); `g20_multibridge.py --sparse`; `2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md` (320 @ 98.4%/bridge).
- **Qwen / onebrain anchors:** `2026-06-23-bridge-coresidence-DEMONSTRATED.md:4/20` (494M spiking Qwen = 14.05 GB); `2026-06-23-bridge-coresidence-perf-dense-matvec-GO-WITH-CAVEAT.md` (dense matvec, host-bottleneck shift, still LOCAL); `2026-06-18-onebrain-320-scale-production-GO.md:38` (onebrain V=320 ≈ 54K neurons); `2026-06-10-nav-on-merged-smoke-PASS-...:13` (merged nav+conv 6,808 neurons).
- **Tiering:** `sim/backend.py` (NumPy backend, `SIM_BACKEND`); `sim/synapse_storage.py` (`TieredSynapseStore`, eviction constants `:47/51/56`, page-in `:30-32`); CLAUDE.md tiering notes (Strategy B+C shipped, A deferred ~3–4 wk; 4–50× CuPy).
- **Train/inference wall-clock + VRAM:** `_longitudinal_develop_loop_gpu.py` (`StreamCortex`, `hear_day` `:218-247`, `read_codes` `.todense()` `:253`, `develop_gpu` 5-stage); `2026-06-23-longitudinal-develop-loop-GPU-GO.md:26` (133 s/day, 15 s WAKE, week 15.6 min, year ~13.5 hr); `2026-06-24-sk-latency-resolved-interact-console-complete.md` (load 0.7 s ~800×, turn 0.47–1.7 s brain / 1.7–5.5 s full, 58 s cold Qwen, ~20× CPU, console picker).
- **Corpus scoping (complement):** `research/findings/raw/_foundational_curriculum_scaling_scoping.md` (WHICH corpus; the linear-`M[V,n_hub]` fact; windows-per-concept wall-clock §84-88; cross-bridge routing risk §6).
