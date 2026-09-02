Both load-bearing facts confirmed. The report follows.

---

# READ-ISOLATION AUDIT — RANKED ACTION REPORT (C2 missing-4-array leak, 14 runners)

**Bottom line:** the bug class is real and present in **all 14** runners (every one omits the same 4 arrays: `cp_refractory_timers`, `cp_prev_firing_states`, `cp_neuron_activity_ema`, `cp_neuron_firing_thresholds`). But it **changed a banked verdict in only 5 of 14**: 4 suppressed-GO false walls + **1 inflated GO that is live in production**. The other 9 are robust (leak inert, or leak << margin). This is a high-value result: one integrity fire (inflated production GO) and a cluster of false walls whose fixes would clean the record — not a "nothing moved" outcome.

**Honest scope caveat up front:** none of the 4 false walls is a *fully-blocked* capability — each already has a primary, sibling, or superseding GO (noted per item). Their value is (a) scientific-record integrity, (b) the false-wall-is-a-measurement-artifact deliverable the mission explicitly rewards, and (c) hardening the sibling/superseding GO that shares the identical leak. The one item with true capability-integrity stakes is the **inflated GO in §2**.

---

## 1. FALSE WALLS — leak SUPPRESSED a real result (`yes_suppressed_GO`)

Ranked by (flip-confidence × mission value). All are clean NO-GO/UNDEFINED/boundary today that a read-isolation fix plausibly moves toward GO.

**FW-1 · `_spiking_expectation_rpe_derisk.py` (secondary config, `--cue-to-expected-weight 0.4`) — severity HIGH, flip DIRECTLY DEMONSTRATED.**
- Today: BOUNDARY/UNDEFINED, 3/6 GO, narrated in the finding as a genuine *"precision/homeostatic-companion-process wall."*
- Under per-fact isolation the audit already showed seed 100 (2.807x→4.372x) and seed 102 (2.794x→5.758x) **flip FAIL→PASS**; seed 44 stays a real fail (1.9x→2.1x). ⇒ ~**5/6, not 3/6.**
- **Unblocks / reframes:** the low-prior expectation-violation-surprise regime. Critically, it **demoted a narrated biological wall to a measurement artifact** — exactly the deliverable the "what did we replace with a constant?" law rewards. (Here the constant is the unreset homeostatic threshold/EMA — `enable_homeostasis` defaults **True** and is never disabled, so this runner's leak vector is all 4 arrays, not just 2.) Primary gain=0.8 GO is unaffected (leak works *against* it), but seed 44's thin 3.4x margin there is not leak-robust — hardened by the same fix.

**FW-2 · `_onebrain_integration_surprise_episodic_crossedge.py` — severity HIGH, on the one-brain spine.**
- Today: **UNDEFINED** (not NO-GO) — F2 precondition `f2_lesion_removes_shift` fails 5/6; `frac_attributable` 0.30–0.73 (only seed 101 clears 0.34).
- The leak's magnitude (order-dependence ~0.0008; repeat-read Δ ~0.0014) is the **same order as the entire `delta_lesion` (0.0018–0.0035)** it corrupts — the diagnosed direct cause of the lesion control not holding, which is *why* the verdict is UNDEFINED rather than GO.
- **Unblocks:** the direct surprise→provgen episodic cross-edge. (A sibling, `encode_decision`, already gives a clean GO 6/6 for a *variant* mechanism — so the capability is partly routed-around, but this exact edge's verdict is recoverable.)

**FW-3 · `_onebrain_integration_r3v2_noncorrupting_dopamine_credit.py` — severity HIGH, but flip-confidence MEDIUM (sign unpinned).**
- Today: **NO-GO 0/6**, with F2 the **sole** blocker (every other arm 6/6). `delta_agent_intact` 0.0027–0.0060, all under `F2_INTACT_FLOOR=0.008` — best seed misses by only **0.00198**.
- Confirmed reproducible confound of comparable-or-larger magnitude (read-to-read swing 0.0007–0.0038; a "read" moves the candidate weight >0.4 because global STDP/reward are never gated to `.train()`). **But the audit could not pin the confound's SIGN at the true trained operating point** — so a flip to GO is *plausible, not certain*.
- **Unblocks:** spiking-dopamine three-factor credit cross-edge — **already delivered by the superseding `r3v3` GO 6/6 (status: live).** So FW-3's value is integrity + confirming R3v3 survives isolation, not opening a blocked capability. R3v3 inherits the identical leak byte-for-byte and its own `read_isolation_verified` check does **not** test repeat-read identity → **R3v3 must be re-verified too (see §3/§6).**

**FW-4 · `_onebrain_integration_r3_spiking_dopamine_credit.py` — severity MEDIUM, historical/confirmatory only.**
- PARTIAL finding, F2 0/6. Same root cause as FW-3 (it's R3v2's parent; R3v2 inherits `_f1–_f4`/`amb_read` verbatim). **Already superseded by R3v3 GO.** Lowest urgency of the four — fix rides along with FW-3's fix but needs no independent capability claim.

---

## 2. INFLATED GO — GO that DEPENDS on the leak (`yes_inflated_GO`) — INTEGRITY FIRE 🔴

**IG-1 · `_onebrain_crossedge_curiosity_to_d6wm.py` (class `AskToW0Pool`) — severity HIGH. THE most time-sensitive item in this audit.**
- Today: **GO 6/6**, and **already wired default-on into production** (`/api/brain-chat` hold-query reply text, `2026-09-01-…-production-wire-GO.md`).
- The audit ran the **full real pipeline with a corrected `_hard_reset`** (not a toy probe): the buggy run reproduces the banked table to 6 decimals; **the fixed run flips seed 43 GO→NO-GO** (`delta_intact` 0.0105 buggy → 0.007375 fixed, below the 0.008 floor — the leak contributed **+0.003125, ~42% of the reported effect**). Because the runner's gate requires `n_go==6`, **the banked "GO 6/6" becomes "NO-GO 5/6" once reads are isolated.**
- Training is *also* contaminated (same leaky reset runs every episode → grown weights differ, e.g. seed 43 grown 1.89 buggy vs 1.58 fixed).
- **Action:** this is a live over-claim behind a default-on production faculty. Fix + re-verify **immediately**; if 5/6 confirms, retract/re-scope the finding and the production-wire GO, and decide whether the default-on stays. Do not let this wait behind the false walls.

---

## 3. HARDENING — leaks but GO is ROBUST (`understated_attributability_only`)

Fix strengthens the science (tightens `frac_attributable` toward its true value); verdict does not change. Ranked by one-brain-spine load.

**H-1 · `_onebrain_integration_r4_selfschema_provenance.py` — ON THE SPINE, highest hardening priority.** Leak real (exact sign-inversion under order reversal on fresh pools), but in the real trained/read-many-times context the tightest seed (100) shifted only 0.0001 under order reversal and F2 PASSED under both read orders on both seeds tested. **Caveat:** only seeds 42 & 100 were dynamically re-checked — a full 6-seed isolated re-verify is the honest close. This GO is **cited as a foundation by two downstream findings** (`…declarative-crossedge-migration-GO`, `…provenance-to-selfschema-reciprocal-GO`), so hardening it protects a dependency subtree.

**H-2 · `_neural_wta_word_decode_derisk.py` — SHARED-PRIMITIVE leak, fixes multiple call sites.** The leak lives in `fswta_drive()` inside `_d3_spiking_attractor_derisk.py`, **imported by other unaudited runners**. Directly demonstrated 2/16 per-word winner flips → most likely explanation for center-mode's single seed-101 soft miss. Verdict does **not** move (gate tolerates one miss; softmax holds 1.000 across all 96 decisions; anti-cheat `shuffle_parity` sits at/below chance, ruling out systematic inflation). Fixing the shared primitive is the highest-leverage hardening edit here.

**H-3 · `_wkv_graded_recurrent_state_derisk.py` — low severity.** Leak inert at the actual production operating point (high drive_bias saturates the carrier); reappears only ~0.1–0.5% relative at D=16 scale vs a GO passed with ~12% headroom. (Note the separate, expected OU-noise variance is out of scope — `reset_state` doesn't reset OU by design.) Safe hardening, low urgency.

**Also hardening (shares the leak, GO stands but needs the isolation re-verify it never got):**
**H-4 · `_onebrain_integration_r3v3_functional_drive.py`** (not in the 14, but inherits R3v2's `_hard_reset` byte-identically and is the *live* GO for the dopamine-credit capability). Its `read_isolation_verified` check only proves a read doesn't move weights, **not** repeat-read bitwise identity. Fold into the §6 re-verify with FW-3.

---

## 4. CLEAN — leak present but verdict provably unaffected (`no`)

No action beyond the §5 hardening port (defense-in-depth). All GO verdicts stand as banked.

| Runner | Why clean |
|---|---|
| `_onebrain_integration_r1_wm_comprehension.py` | Dynamic reads bitwise-identical (2 seeds × 2 shapes); LOAD_STEPS=30 forced-spike warmup erases sub-mV residue before recording. 2 of 4 arrays config-inert (`enable_homeostasis=False`). |
| `_onebrain_integration_r2_threefactor_selforganized.py` | Refractory(=2 steps)/prev-firing residue absorbed by the 30-step settle before every scored read; margins bitwise-identical; 0/6-vs-6/6 shuffle-topology control gap. |
| `_onebrain_surprise_episodic_encode_decision.py` | Lesion arm architecturally pinned to exactly 0.0 (`episodic_encode_gate` has zero afferents besides the lesioned edge). Intact ~0.3% read noise is >25× below the 7.6–8.1× floor clearance. |
| `_onebrain_crossedge_provenance_to_selfschema.py` | Leaks on gen/perc, but the `author` scalar (the delta_lesion numerator) has no afferent but the cross-edge → structurally pinned to 0.0 under lesion; intact 2.1–2.4× margin >> leak noise. |
| `_causal_forward_model_derisk.py` | Output bitwise-identical across 3×3×14 zeroed-mechanism combos and all trained-order permutations; 2 arrays config-dead, other 2 leak but never cross a spike-count boundary (1200pA drive, refractory=2). |
| `_affective_world_model_derisk.py` | Leak ~0.4% of baseline, ~240× below `intact_separation_hz=41.53`; balanced ±valence design forces lesion aggregates equal → cancels in the difference (lesion_sep exactly 0.0). |

**Could-not-run:** none — all 14 were dynamically probed. One residual unknown flagged honestly: FW-3's confound **sign** at the fully-trained operating point (would need a full 400-episode `.train()` to pin, out of read-only-audit scope) — the §6 cupy re-verify resolves it directly.

---

## 5. THE FIX RECIPE — minimal & safe, two ports

Both ports restore the **same 4 arrays** from a **true-rest snapshot** taken once at build (after the initial settle, before any read/train), on **every** `_hard_reset`. The landed C2 fix (`_crossedge_surprise_metacog_derisk.py`, `_EXTRA_RESET_ARRAYS` lines 195–196, restore at 322–328) is the exact template. Two delivery paths:

**Port A — route to the framework snapshot (preferred for onebrain_* runners).**
`onebrain_merge_framework.py::MergedPool` **already** carries `read_isolation(active)` (L540), `sequence_isolation()` (L574), and a `_PER_NEURON_STATE` tuple (L246–250) that **already lists all 4 arrays** (`cp_prev_firing_states`, `cp_refractory_timers`, `cp_neuron_firing_thresholds`, `cp_neuron_activity_ema`, + `cp_refractory`). Fix = make each runner's bespoke `_hard_reset` snapshot+restore `_PER_NEURON_STATE` (or wrap scored reads in `read_isolation`/`sequence_isolation`) instead of its hand-rolled partial list. Lowest-risk: reuses an already-correct, already-tested primitive.
- **Apply Port A to:** `surprise_episodic_crossedge` (FW-2), `r3` (FW-4), `r3v2` (FW-3), `r3v3` (H-4), `crossedge_curiosity_to_d6wm` (IG-1), `r4` (H-1) — and, as defense-in-depth (no verdict change), `r1`, `r2`, `encode_decision`, `crossedge_provenance_to_selfschema`.

**Port B — inline `_EXTRA_RESET_ARRAYS` (for standalone runners with no merge framework).**
Copy the C2 block verbatim. **Watch the typo:** `_spiking_expectation_rpe`, `_causal_forward_model`, `_affective_world_model` reset a **nonexistent** `cp_refractory` (`getattr`→None, dead no-op) — the fix must use `cp_refractory_timers` **and** add the other 3.
- **Apply Port B to:** `_spiking_expectation_rpe_derisk.py` (FW-1), `_wkv_graded_recurrent_state_derisk.py` (H-3), and the **shared primitive** `fswta_drive()` in `_d3_spiking_attractor_derisk.py` (fixes H-2 `neural_wta` + any other importer). `_causal_forward_model` / `_affective_world_model` get the same port as hygiene (verdicts already clean).

**Safety guarantee both ports share:** for a runner where the 4 arrays are provably inert (config-dead homeostasis, or forced-spike warmup), restoring them is a **no-op** — so applying the port to the §4 clean runners cannot change their verdicts, only harden them. Add a `selftest()`-style assertion (repeat-read bitwise identity on a zeroed-mechanism pool) so the fix, like every gate here, can fail in its failing direction.

---

## 6. RE-VERIFY PLAN — fresh 6-seed **cupy** verify, `gpu_queue` (sequential)

Only runners whose **verdict could move** need the decisive cupy re-verify. Order below puts false walls first per protocol, but **IG-1 is interleaved at the top** because it is a live production over-claim — it should not wait behind confirmatory work.

| # | Runner (config) | Why re-verify | Expected outcome |
|---|---|---|---|
| **1** | **IG-1 `crossedge_curiosity_to_d6wm`** | Live default-on production GO that fix flips (seed 43) | Confirm **NO-GO 5/6** → retract/re-scope finding + production-wire; decide default-on |
| **2** | **FW-1 `spiking_expectation_rpe` (gain=0.4)** | Directly-shown 3/6→~5/6; reframes a narrated biological wall | ~**5/6 GO**; update finding's "wall discipline" section (artifact was measurement, not precision limit). Re-run gain=0.8 in the same job to harden seed 44. |
| **3** | **FW-2 `surprise_episodic_crossedge`** | UNDEFINED, leak is diagnosed root cause of lesion-control failure | Lesion control holds → **UNDEFINED→GO** (or a clean NO-GO if not) |
| **4** | **FW-3 `r3v2` + H-4 `r3v3`** (batch) | r3v2 NO-GO sign unpinned; r3v3 is the live GO sharing the leak, never isolation-tested | r3v2: resolves GO-vs-true-NO-GO; r3v3: confirm GO **survives** isolation (protects the shipped capability) |
| **5** | **H-1 `r4`** (seeds 43/44/101/102 — 42/100 already done) | Foundation cited by 2 downstream findings; only 2 seeds dynamically checked | Confirm GO 6/6 under isolated reads; tightens `frac_attributable` |

**FW-4 `r3` needs no independent cupy run** (superseded by r3v3; its fix rides FW-3's edit and its PARTIAL status is already historical).

**H-2 / H-3 and the §4 clean runners:** **no cupy re-verify required** — verdicts don't move. Land the ports, rely on the added repeat-read bitwise-identity `selftest` as the regression guard, and (H-2) note the shared-primitive fix in `_d3_spiking_attractor_derisk.py` covers unaudited importers.

**Cost routing:** the fixes are mechanical (2 small ports + typo) → sonnet/haiku build agent, not Opus. The 5 cupy re-verifies → `tools/gpu_queue.sh` (`--seeds 42,43,44,100,101,102`), sequential, VRAM-safe. Do **not** spend Opus tokens on the re-verify runs themselves.

---

**Files referenced (all absolute under `/home/dant123/Projects/sim/`):** the 14 audited runners under `research/runners/`; the fix template `research/runners/_crossedge_surprise_metacog_derisk.py` (`_EXTRA_RESET_ARRAYS`, L195–196/305/322–328); the framework snapshot `research/runners/onebrain_merge_framework.py` (`_PER_NEURON_STATE` L246–250, `read_isolation` L540, `sequence_isolation` L574); the shared leaky primitive `research/runners/_d3_spiking_attractor_derisk.py::fswta_drive` (L55–79); the live superseding GO `research/runners/_onebrain_integration_r3v3_functional_drive.py`.