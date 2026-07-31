---
type: finding
status: contributing
date: 2026-07-18
mechanism: emergent-dg
---

# gap#5 emergent-DG arc - deep-research scoping (workflow synthesis, 2026-07-18)

# Emergent-DG CA3 Assembly — Buildable Scoping (gap#5 follow-on)

## 1. DIAGNOSIS — what "emergent-DG" actually requires

The current gap#5 completion win uses a **hand-assigned** CA3 assembly: `rng.choice(ca3_idx, n_assy)` at `_riii_ca3_synchronous_assembly_derisk.py:89-94`, driven directly (bypassing EC→DG→CA3 during encode). "Emergent-DG" means the assembly must instead be **SELECTED by the network from the input** — a cortical pattern flows `language_input→ec→dg→(mossy)→ca3`, and *which* CA3 cells fire is determined by wiring + dynamics, not by a Python draw. Different inputs must select **different, well-separated** CA3 assemblies; the same input must reselect the **same** assembly.

**The ONE key mechanism (mossy detonator selection + DG sparsity):**
- Separation happens **in DG** (expansion recoding + PV-basket feedforward inhibition → ~1–2% sparse GC code; input cos 0.80 → DG cos ~0.22, already validated in-project).
- The GC→CA3 map is a **FIXED sparse strong binary matrix** (1 GC→~15 CA3; 1 CA3←~50 GC). "Detonation" = the low-density/high-weight mossy projection: a *bursting* GC drives its ~15 targets over threshold near-deterministically (Vyleta-Jonas 2016: single-EPSP p≈0.12, 3×50 Hz burst p≈0.82, PTP→p≈0.71). So the CA3 assembly ≈ **trim(⋃ target-sets of the active bursting GCs)**, trimmed to sparse by CA3 feedforward inhibition.
- **Digitization crux:** you do NOT need a learned or graded GC→CA3 projection. Assembly *identity* = fixed wiring read by detonation; *reliability* = fixed wiring + FFI + CA3 completion; only the *DG code* must be reproducible per input. This is exactly why the earlier "graded k-WTA DG→CA3" hit the separation-vs-reliability wall (dense graded sum re-mixed the code); the sparse-strong-binary relay resolves it.

## 2. THE MECHANISM to build — what changes vs. the pre-assigned version

Reuse the EC→DG→CA3 wiring, the `encode_btsp` store, and the bistable completion **verbatim**. The *only* substitution: replace the random assembly draw with a **DG-selection read pass** run BEFORE encode.

Concretely, per input pattern m:
1. Freeze plasticity (`_set_gates(bridge, 0.0)` — these gate weight updates only; the feedforward path always conducts).
2. Drive `language_input` with a sparse code `pat_m`, run `lang→ec→dg` (with `dg_pv_basket` FFi giving the 2–5% DG code) → `dg→ca3` mossy → `ca3_pv_basket` feedback inhibition caps the winners.
3. Record CA3 firing over the drive window; `A_m = {ca3 cells firing in ≥30% of steps}` (the top-K logic already at `_riii_ca3_coincidence_completion_derisk.py:243-249`).
4. Feed `A_m` into the existing `run(..., assemblies=A_m, encode_btsp=True, bistable=True, ...)`.

**Everything downstream is already parameterized purely by the `assemblies` list** (`_apply_competition`, `structural_sep`, `selective_inhib`, BTSP co-fire + plateau placement, bistable recall, all anti-cheats) and needs **no change**. The de-risk's own in-code comments already name this as "the emergent DG-selected follow-on" (lines 219, 253).

**Mandatory levers** (all already in `_build`): `ca3_fb_inhib` (CA3 `internal_density=0.0` has NO wired feedback inhibition → without it CA3 spreads to 35–47%, the CYCLE-1072 boundary), `dg_ffi_weight` (DG sparsity), `mossy_weight`/`mossy_density` (CA3 selection sparsity).

**Honest scope split (decoupled, cheap-first):** Rung 1 makes SELECTION emergent (DG-read) but keeps the store *drive* direct-on-read-cells + runner-placed plateau (reuses `encode_btsp` verbatim). The **fully-input-driven store** — mossy detonators co-fire A_m during BTSP and the plateau is triggered by mossy detonation, not runner-placed — is the named **Rung 3** follow-on, biologically faithful but higher-risk on co-fire density.

## 3. THE CHEAP-FIRST DE-RISK

**R0 (risk-first, no BTSP / no recall / no training):** build once, present K≥3 distinct sparse inputs, read CA3 rate, sweep a two-knob grid `dg_ffi_weight × {ca3_fb_inhib, mossy_weight}`, OU off then on. Measure ONLY:

| Metric | GO target |
|---|---|
| DG sparsity | 0.02–0.08 |
| CA3 assembly sparsity `\|A_m\|/n_ca3` (natural, NOT top-k truncated) | 0.005–0.05 AND `\|A_m\|` in the validated size window (≈8–24 cells at n_ca3 1000–2000) |
| Stability (re-present from fresh reset) `Jaccard(A_m,A_m')` | ≥ 0.6 |
| Separation (m≠m′) `cos(r_m,r_m′)` and `Jaccard` | cos < 0.4, Jaccard < 0.2 |

Report both binary-assembly and rate vectors. Do **not** top-k truncate for the emergence claim (that launders the sparsity); top-k-among-θ_sel is a flagged fallback only.

**R1 store+complete on the DG-selected assembly:** pass `A_m` into the gap#5-GO `encode_btsp + bistable` config; recall drives `drive_region="ca3"` with a partial cue of A_m (never `language_input`, so completion is the recurrent attractor).

**GO gate (6-seed: 42,43,44,100,101,102 — all seeds pass BOTH stages):**
- *Selection:* the R0 table (sparsity/size/stability/separation) + no-input assembly ≤ 0.2·mean|A_m|.
- *Store+complete:* `held_cue ≥ 0.20` AND `≥ 3·held_nocue` AND `held_nocue ≤ 0.10` AND `≥ 3·held_perm` AND no-encode `held_cue < 0.10` AND cross-assembly `held_cross ≤ 0.10` AND DG-lesion→no assembly. Emergent `held_cue` within ~20% of the hand-mask like-for-like reference on the same config.

**Anti-cheats:**
| Anti-cheat | Test | Pass |
|---|---|---|
| Input-driven, not hand-assigned | permute/scramble input → `A_perm` | cos(r_m,r_perm)<0.4, Jaccard<0.2 (identity tracks input) |
| Deterministic selection | re-read same input | Jaccard ≥ 0.6 |
| Pattern separation | all pairs of K inputs | all cos < 0.4 |
| No-input → no assembly (encode moat) | zero input drive | `\|A_noinput\| ≤ 0.2·mean\|A_m\|` |
| DG/mossy lesion (provenance) | zero `dg→ca3` before read | no assembly selected |
| Cross-assembly specificity | cue A_m must NOT complete A_{m′} | held_cross ≤ 0.10 |
| No-cue (self-sustaining detector) [inherited] | silence → read held members | held_nocue ≤ 0.10 |
| Permuted-recall (the retraction gate) [inherited] | cue random non-assembly set | held_cue ≥ 3·held_perm |
| No-encode [inherited] | skip BTSP, then cue | held_cue < 0.10 |

The **no-cue + permuted-recall** pair is MANDATORY: git `b7be09d5` retracted the prior "6-seed GO" precisely because held members fired the same (~50) whether cued correctly or with a random set — an always-on attractor artifact, not cue-gated completion.

## 4. REUSE (file:line)

- **DG-selection read:** `_riii_ca3_coincidence_completion_derisk.py:230-249` (drive `language_input` → EC→DG→CA3, record CA3 firing, take top set); mirror at `validate_trisynaptic_loop.py:308-348`.
- **Bridge build + gates:** `_riii_ca3_coincidence_completion_derisk.py` `_build(...)` (:39-135, appends `ca3_pv_basket` when `ca3_fb_inhib` set) + `_set_gates` (:194-202, toggles `ca3_swr_burst/dg_to_ca3/ec_to_dg/lang_to_ec`).
- **BTSP store:** `_riii_ca3_synchronous_assembly_derisk.py:114-154` (`encode_btsp`); underlying kernels `sim/bridge.py:7370-7400` (BTSP), `:7222-7288` (BDSP bistable apical); flags `sim/config.py:302-306, 285, 351, 209-210, 237`.
- **Bistable completion + retraction-hardened gate:** `_riii_ca3_synchronous_assembly_derisk.py:270-440` (`_hard_silence` :291-312, `_measure` :314-331, gate :373-374, `rate_homeo` :333-352, SWR/CA1 read `read_ca1` :378-439).
- **Completion mechanisms (runner-side pathway flips, no sim/ edit):** coincidence/two-compartment `_riii_ca3_coincidence_completion_derisk.py:44-50, 162-186` + `sim/bridge.py:6448-6560`; Wang nmda_slow attractor `_build` :51-59 → `sim/regions.py:341` (`exc_receptor`), `:353` (`coincidence_detector`).
- **Selection/measurement helpers:** `validate_trisynaptic_loop.py` `measure_region_response` (:80-119), `build_drive_pattern` (:122-128), `overlap_drive_patterns` (:131-162), `cosine_similarity` (:69-77).
- **Competition kernel:** `sim.kernels.fused_htm_winner_inactive_depression` (imported `_riii_ca3_synchronous_assembly_derisk.py:70`, applied :105-110).
- **EC→DG→CA3 wiring:** `text_minimal_isolation.build_biological_brain_regions` (:173, gate :683; regions :684-728; pathways :731-1138 — mossy `dg→ca3` at :1112-1116 density 0.10 weight 8.0).
- **New code:** `_riii_ca3_emergent_dg_selection_derisk.py` (driver) + a ~3-line `assemblies=None` guard in `run(...)` at `_riii_ca3_synchronous_assembly_derisk.py:~93`. **NO `sim/` edit** (reuse-by-import; the whole arc stays that way).

## 5. BIGGEST RISK + cheapest FIRST check

**Biggest risk:** the reused feedforward DG→mossy→CA3 path does not produce a sparse-enough / stable-enough / separated-enough CA3 selection — the **joint** failure of three modes: too *dense* (CA3→35–47% without adequate FFI, the CYCLE-1072 boundary), too *sparse/empty* (mossy too weak → nothing fires, or `ca3_fb_inhib` too high → 1–2 uncompletable cells), or *unstable* (weak detonation + OU noise → different cells each presentation). Every downstream step is worthless if selection fails, and it is orthogonal to (and cheaper than) BTSP.

**Cheapest check FIRST = R0** (Section 3): a pure feedforward read, no BTSP/recall/training (~minutes/seed on GPU), sweeping `dg_ffi_weight × {ca3_fb_inhib, mossy_weight}` (OU off then on) to find the operating point where `|A_m|` is *simultaneously* sparse (~1–2%) AND large enough (≥~10–16 cells) AND stable AND separated — the exact window the hand-mask occupied. Only if R0 finds that joint window do R1–R2 spend BTSP GPU time. The knob is monotone but non-linear, so R0's success criterion is the joint window, not just "sparse."

## 6. Ranked build order (cheap → hardest); 6-seed, parallelizable across seeds; NO `sim/` edit

- **R0 — feedforward selection** sparse/stable/separated (risk-first, no learning): the two-knob sweep. *Gate: the R0 table.*
- **R1 — swap the mask:** feed R0 assemblies into the validated BTSP-store + bistable-complete; run the **hand-mask like-for-like reference** on identical bridge/config (attributes any gap to selection source, not config drift). *Gate: emergent held_cue within ~20% of reference.*
- **R2 — full anti-cheat battery:** permute-input, no-input moat, DG/mossy-lesion, cross-assembly + inherited nocue/perm/no-encode. *This is where a real (not artifact) emergence claim is earned.*
- **R3 (named follow-on) — fully-input-driven store:** mossy detonators co-fire the assembly during BTSP and trigger the plateau (removes the last runner-placed piece). Higher-risk on co-fire density; biologically faithful (mossy = detonator triggering the dendritic plateau).

---

**Honest flags / report uncertainties:**
- DG separation (D.12) is 3/3 multi-seed robust, but **completion (D.13) is seed-fragile** (1/3 on strict absolute cos>0.7). The emergent selector *inherits* DG's per-seed reproducibility — hence the stability anti-cheat (Jaccard≥0.6 over the drive window, top-K over the last few events not one) is load-bearing, not optional.
- DG separation was validated **single-seed** (seed 42, cos 0.218, ~1% sparsity — slightly *over*-sparse vs the 2–5% target); R0 must re-confirm multi-seed and may need `dg_ffi_weight` *down* if ~1% is too sparse to relay a completable assembly.
- The dg-biology report's detonator conditionality (single-spike p≈0.12 vs burst p≈0.82, PTP latch) and adult neurogenesis are **not** in the current wiring — the mossy path is a plain sparse-strong AMPA projection; detonation is emergent from low-density/high-weight, not a modeled conditional/PTP synapse. Adding conditional detonation/PTP is an *optional* sharpener, explicitly flagged non-essential for a first digitization; do not scope it into R0–R2.
- The reports agree the change is a ~3-line substitution; the residual honesty is that R1's store is still direct-drive + runner-placed plateau (R3 closes that). Don't overclaim "fully emergent end-to-end" until R3.