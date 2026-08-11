---
type: finding
status: corrected
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: the 14-turn conversation eval `research/runners/_conversation_turing_test_derisk.py` (Stage-A FULL one brain; SIM_BACKEND=numpy substrate, generator mouth on cpu) with the turn-7 episodic recall wired to a NEW on-substrate module `research/runners/_episodic_dap_dialogue_memory.py::EpisodicDapMemory`, which reuses-by-import the standing 6/6-GO gap#5 dendritic-dAP READOUT completion (`research/runners/_gap5_dendritic_dap_readout_completion_derisk.py`, ab9f7dbe): emergent-DG membership + `_build_dap_readout` (two-compartment apical dAP) + `_form_one_assembly` (BTSP one-shot store) + `_apical_up_read` (held-cell apical UP-state completion) + `_held_cue_perm` (cue/held/perm geometry). Recall GATE = which CA3 assemblies COMPLETE from the referential cue; fact CONTENT = host `episode_mem` oracle. Load-bearing control: the lesioned read restores the UNFORMED baseline recurrent weights before the apical read (`attributable_to` intact-vs-lesioned). Backend comparison: the identical GO config is read on numpy (the eval substrate) vs its cupy GO (ab9f7dbe).
---

## ⛔ CORRECTION (2026-08-10, direct 6-seed test in FRESH isolated builds) — the numpy "backend-block" was the WRONG kthresh (=30), NOT forward-Euler; at kthresh=8 the module FIRES cue-specifically on BOTH backends, 6/6

**What was wrong.** The honest-negative below was measured at **`kthresh=30`** — the value the module's `GO_DEFAULTS` shipped with. A direct test (2026-08-10) shows `EpisodicDapMemory` fires on NEITHER backend for ANY assembly size at kthresh=30, so the numpy `0.000` was a **non-firing operating point, not a forward-Euler apical-integration limit**. The "cupy is the 6/6 GO backend [for this module] / numpy is backend-blocked" split was an UNTESTED INFERENCE from the standalone dapB GO; it is FALSE. TWO errors compounded: **(1) Wrong kthresh** — the standalone dapB runner SWEEPS `kthresh ∈ {15,30}` and picks the point maximising the MEAN apical-UP over ALL patterns; **kt=15 wins there, never kt=30** (`research/findings/raw/_gap5_dapB/dapB_6seed.json`: `best k_thresh=15.0` on 6/6). The module hard-coded kt=30 — the losing arm. **(2) The MEAN masked a per-topic SIZE failure** — this module reads ONE topic per recall, and even at kt=15 the SMALL (~13-cell) emergent assembly is SILENT (0.00) while the large one fires (0.94); the mean hid it.

**The apical dAP UP-fraction read has a NARROW per-assembly operating window in kthresh**, and finding it required the FRESH isolated builds (the production path: build once at a kthresh, recall once), because the reuse-heavy per-topic sweep is unreliable (below). The window: **kt≥10 SILENCES the smallest emergent assemblies** (~13 cells: 0.57 @kt8 → 0.0–0.43 @kt10, on the cliff); **kt≤6 lets some emergent memberships SELF-IGNITE** — the `nocue` read (no cue driven) goes UP anyway, a specificity failure (a fresh-build s102 dog: `nocue=1.000` @kt6). **kt=8 threads the window.**

**Corrected cupy verdict — FRESH isolated builds, module's OWN `in_memory` cue-specific gate, per topic, 6 seeds** (raw: `research/findings/raw/_episodic_dap_kthresh/clean_verify_kt8.json`). **At kthresh=8 BOTH toy topics fire cue-specifically on 6/6 seeds**, including the smallest 13/14-cell emergent assemblies, with perm=nocue=lesion(baseline)=0 (load-bearing teeth hold):

| seed | cat apical_cue (size) | dog apical_cue (size) | perm | nocue | lesion (baseline) | both fire |
|---|---|---|---|---|---|---|
| 42 | 0.588 (33) | 0.571 (**13**) | 0.000 | 0.000 | 0.000 | ✓ |
| 43 | 0.909 (21) | 1.000 (28) | 0.000 | 0.000 | 0.000 | ✓ |
| 44 | 0.588 (34) | 0.857 (**14**) | 0.000 | 0.000 | 0.000 | ✓ |
| 100 | 1.000 (19) | 0.769 (25) | 0.000 | 0.000 | 0.000 | ✓ |
| 101 | 0.929 (27) | 0.917 (24) | 0.000 | 0.000 | 0.000 | ✓ |
| 102 | 0.812 (31) | 1.000 (23) | 0.000 | 0.000 | 0.000 | ✓ |

**The reuse-heavy per-topic SWEEP (`sixseed_kt_sweep.json`) is UNRELIABLE and produced TWO opposite artifacts, both disproven by the fresh builds:** it MISSED kt=6's self-ignition (its first-read s102 happened not to self-ignite → it reported a false "kt6 6/6") AND it FABRICATED a kt=8 teeth-fail (s102 cat `lesion=1.0` after ~12 prior live-mutated reads on the reused bridge — a state-contamination artifact; a fresh s102@kt8 build reads `lesion=0.0`, clean BOTH-PASS). The sweep's `{kt6:6/6, kt8:5/6}` is therefore an artifact; the fresh-build truth is **kt8: 6/6, kt6: 5/6** (s102 self-ignites at kt6). Emergent assembly membership is non-deterministic at the firing threshold (FMA/summation reorder, `sim/kernels.py`), so exact per-seed reads vary build-to-build — kt=8 passes across builds where kt=6 does not.

**Corrected numpy verdict — the biggest win: the module FIRES on numpy at the corrected kthresh, so the LIVE CHAT (numpy substrate) gets a genuinely-spiking recall with NO cupy needed.** Seed 42, `research/findings/raw/_episodic_dap_kthresh/numpy_kt_sweep_s42.json`: at **kt=8** `cat(27)=0.929`, `dog(22)=0.909`; at **kt=6** `cat 0.786`, `dog 0.909` — both fire strongly, cue-specific, perm=nocue=lesion=**0.000**. Contrast the SAME numpy substrate at kt=15 (the old measurement's neighbour): `cat 0.143`, `dog 0.091` — under-reads, below the 0.20 gate; at kt=30 (the value the finding actually used) it is 0.000. **The apical dAP plateau reaches the UP state on numpy fine; the whole "backend-block" was the operating point.**

**FIX APPLIED.** `GO_DEFAULTS` `kthresh 30 → 8` in `research/runners/_episodic_dap_dialogue_memory.py`; the conversation eval passes no kthresh override, so the corrected point flows into the `--spiking-episodic` path (numpy) automatically — turn-7 recall is now carried by the SPIKING completion, not the host-oracle fallback. This is the gap#5 assembly-SIZE residual (a small emergent assembly's weaker within-member cue volley needs a lower dendritic coincidence threshold, but too low self-ignites — kt=8 is the operating point that fires the small assembly without self-igniting) corrected by the operating point — **per THE LAW, the negative launched the fix; it was never a wall.** The original body below is retained for the record; **its `kthresh=30` numbers and every "numpy backend-blocked / forward-Euler" claim are SUPERSEDED by this section** (registered in `docs/RETRACTED.md`).

---

**Coordinator independent re-verification (2026-08-10, before merge).** Reproduced the fire in FRESH isolated builds outside the fix agent's process: 4 cupy builds (seeds 42/43 × both topics) → stored apical **0.57–1.0**, lesion 0; numpy seed-42 dog independently **0.909**, clean. **One ADDITIONAL control the agent's perm/nocue battery did not run** — recalling an UNSTORED cross-topic slot (its assembly never BTSP-formed) — reads **0.0 on seed 42** but **up to 0.154 on seed 43**: still BELOW the `COMPLETE_MIN=0.20` `in_memory` gate (so correctly classified NOT-in-memory), but a specificity MARGIN that, combined with the flagged build-to-build size non-determinism, is worth watching. **Both residuals are safety-netted by the eval's host-oracle self-consistency guard** (a spiking mis-fire → host-oracle fallback, never a confabulation), so the live chat's honesty is protected regardless. VERDICT UPHELD: kt=8 genuinely fires cue-specifically on both backends; the correction stands.

# INTEGRATION #4 — the turn-7 episodic-dialogue recall is WIRED to the on-substrate gap#5 dendritic-dAP readout completion (a spiking pattern-completion, load-bearing on cupy); on the numpy conversation-eval substrate the dendritic apical read does not fire — a quantified honest-negative — so the eval keeps the host oracle with the spiking gate wired + a no-regression fallback  [⛔ SUPERSEDED — see the CORRECTION section above: the numpy non-firing was kthresh=30, not forward-Euler; at kthresh=8 the module fires cue-specifically on numpy (0.93/0.91) AND cupy 6/6]

INTEGRATION #2 (main HEAD `77587122`) gave the chat episodic dialogue memory at turn 7 via a HOST per-turn buffer
(`episode_mem`, a DECLARED SCAFFOLD — the brain did no memory; the recall was a Python dict scan). The brain-based-only
standard requires the recall to be a SPIKING memory. This integration converts the recall's GATE — *which topics were
discussed* + *is the referent in memory* — from the host list scan to an on-substrate spiking pattern-completion, while
keeping the host buffer as the fact-content oracle + a fallback.

## What was built (additive, guarded, NO `sim/` edit)

<!--derived-->

- **`research/runners/_episodic_dap_dialogue_memory.py`** (NEW) — `EpisodicDapMemory`: each toy-world topic is
  pre-allocated a CA3 assembly SLOT (emergent-DG membership at the GO scale n_ca3=2000). `store(topic)` BTSP-forms ONLY
  that slot's assembly on the dedicated dAP readout bridge (episodic WRITE; the weight is the rule's output, not a
  hand-set constant). `recall(topic)` drives the slot's partial cue and reads the fraction of held-out cells whose
  `cp_v_apical` is in the UP state (the two-compartment dendritic dAP completion), with perm/nocue controls and the GO
  cue-specific criterion. `discussed_topics()` DECODES the discussed set from which assemblies complete. `lesion=True`
  restores the UNFORMED baseline weights before the read — the load-bearing teeth.
- **`research/runners/_conversation_turing_test_derisk.py`** (edited, additive) — a `--spiking-episodic` flag
  (default OFF ⇒ byte-identical host-oracle path). When ON: the topic branch calls `episodic_mem.store(topic)` (the
  spiking WRITE) alongside the host `episode_mem.append`; the referential branch DECODES `discussed` +
  `referent_in_memory` from `episodic_mem.discussed_topics()` (intact + lesioned) and uses it as the recall GATE, with
  the host oracle supplying fact content. A SELF-CONSISTENCY guard falls back to the host oracle (no regression) if the
  spiking read fails to complete a topic the store says was written, recording the quantified gap.

The mechanism is faithful to the standing GO: a BTSP-formed CA3 assembly per dialogue-topic, completed cue-specifically
from the referential cue via the two-compartment dendritic dAP readout.

## The honest-negative: the dendritic dAP apical read does not fire on the numpy substrate (the eval runs on numpy)

<!--derived-->

The conversation eval's spiking substrate is `SIM_BACKEND=numpy` (`build_one_brain` + the co-resident bridge). Running
the gap#5 dAP readout mechanism on numpy at its exact GO config (seed 42, n_ca3=2000, density 0.5, wmax 100, kthresh 30,
apical_R 0.15, self_regen 2.0, up_thresh −20 mV), with GENUINE emergent membership (assembly sizes **27, 22** cells)
and GENUINE BTSP formation (**w_within = 84.1**, grown from the 1.5 baseline by the rule) gives:

| read | formed slot (cue) | perm | nocue | unformed slot (cue) | lesion (baseline weights) |
|---|---|---|---|---|---|
| apical UP fraction (numpy) | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |

The formed assembly does NOT complete — the apical UP-state is never reached on numpy. Contrast the standing GO
(`ab9f7dbe`, SIM_BACKEND=cupy): held_cue ≥ 0.20, cue-specific, 6/6 seeds. **Formation is genuine on numpy (w_within
84.1); it is the dendritic READOUT — the two-compartment apical plateau (forward-Euler under numpy) reaching the UP
state — that is backend-blocked.** Per-op cost on numpy@2000: emergent selection ~37 s, one apical read ~21 s, one
slot formation ~373 s (~6 min) — so the mechanism is also impractically slow to rebuild live per-eval on the CPU.

Because the dAP read reads 0 for the stored topic on numpy, the eval's SELF-CONSISTENCY guard fires: the spiking gate
is recorded as an honest-negative on this backend and the reply falls back to the host oracle, so the eval does NOT
regress. (On cupy the gate would carry the recall and be load-bearing — the lesioned-vs-intact `attributable_to` is
built into the record.)

**Live end-to-end run** (`--spiking-episodic`, seed 42, numpy, cpu mouth,
`research/findings/raw/lanes/stageA/turing/conv_spiking_episodic_s42.json`): the spiking WRITE fires (`[episodic-dap] STORE topic='dog' slot=1
w_within=82.0`); at turn 7 the transcript SELF-DOCUMENTS the honest-negative — the `dog` read record is
`{"formed": true, "apical_cue": 0.0, "in_memory": false}` (stored, but the dAP read is 0 on numpy), `cat` is
`{"formed": false, "apical_cue": 0.0, "in_memory": false}` (never stored). `episodic_spiking_consistent_with_store =
False` ⇒ gate falls back to `host-oracle (spiking dAP readout did not fire on this backend -- honest-negative)`, turn 7
recalls `dog`, **`n_confabulations = 0`**, no regression. The `formed:true` / `apical_cue:0.0` split is the honest
signature: the STORE works on numpy, the dendritic READ does not.

## No regression on the default (host-oracle) path

<!--derived-->

Default eval (no `--spiking-episodic`, SIM_BACKEND=numpy, cpu mouth, seed 42): **`n_confabulations = 0`**; turn 7 =
false-premise recall of `dog` (gate=host-oracle, discussed=`['dog']`, referent `cat` correctly NOT in memory, confab
False, reply *"A dog gos to the east. A dog looks at river. A dog runs north."*). Turn-3/4/5 grounded prose intact,
turn-5 affect read-out intact (valence +0.07, level 3), turn-6 curiosity forward-model intact (predicts `south`,
margin 0.09, flagged not-observed), turn-13 self-model intact (honest structural self-description). The additive edits
leave the host-oracle path unchanged (the new `episodic_gate` field records provenance; behavior is identical).

## Per THE LAW — the next mechanism (never a wall)

<!--derived-->

The dAP readout is a verdict on a METHOD *on the numpy backend*, not on the CAPABILITY. The spiking episodic recall
capability has three live routes, in priority order:

<!--derived-->


1. **Run the wired eval on cupy** (the dAP GO backend) — the wire-in is complete; the spiking gate becomes
   load-bearing the moment the substrate is cupy. Blocked now only by the owner GAMING (GPU at 99%, Palworld holds
   9.4 GB); deferred with this clear note, not abandoned.
2. **The numpy-compatible SIBLING readout — the somatic slow-NMDA REVERBERATORY completion**
   (`research/runners/_gap5_emergent_end_to_end_episodic_loop_derisk.py`, also a 6-seed GO), whose read is SOMA firing
   (which numpy computes natively). Numpy@2000 test (seed 42, density 0.35, wmax 9000, same emergent 27/22-cell
   assemblies): the SOMA read **FIRES** — formed held_cue **0.489**, LESION (baseline weights) held_cue **0.000**
   (the formed attractor is load-bearing), unformed slot held_cue 0.045. Unlike the dAP apical read, **the somatic
   readout is numpy-viable.** BUT at this single working point it is NOT cue-specific (held_perm **0.488** ≈ cue,
   held_nocue **0.488** = self-ignition) — the point-neuron specificity horn the 2026-08-10 redirect finding named. The
   e2e GO reaches cue-specificity at SOME (density, wmax, ou) working point in its 6-seed sweep; this one config is not
   it. **⇒ The immediate numpy-actionable route: sweep the reverberatory working point on numpy to find the
   cue-specific window, then wire the SOMA-read completion as the live numpy episodic gate** (the module generalizes to
   a `readout=reverberatory` mode). The recall capability is NOT blocked on cupy — the somatic read runs on the CPU.
3. **Diagnose + fix the numpy two-compartment apical integration** so the dAP plateau reaches the UP state on the CPU
   (a real engineering/biology surpass — the forward-Euler apical ODE evidently integrates differently on numpy).

## Scope / honesty

- The recall GATE is spiking (pattern-completion); the fact CONTENT is still the host oracle — a NAMED residual
  scaffold, the next conversion after the gate fires live.
- Emergent membership + the ~23-cell assembly SCALE are load-bearing, not tunable away: at n_ca3=400 with a 72-cell
  pre-assigned assembly the apical read is also 0 (the small emergent assembly is what frees the per-cell read on
  cupy — the GO runner's own claim).
- No `sim/` edit; the eval default is byte-identical (host oracle) until `--spiking-episodic` is passed.
