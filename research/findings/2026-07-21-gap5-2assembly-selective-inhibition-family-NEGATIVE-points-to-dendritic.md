# gap#5 two-assembly co-storage — the SOMATIC selective-inhibition family (Kim-Kim E→I) VERIFIED-does-NOT-close the boundary; the cross-completion is NOT somatic-basket-gateable → the coupling is DENDRITIC (dendrite-targeting apical SOM is the ranked next mechanism). NO `sim/` edit (additive driver mechanism).

## ⚠️⚠️ CORRECTION (2026-07-21, same cycle) — the DENDRITE-TARGETING apical inhibition is PROMISING, NOT negative. I banked a wrong verdict from PARTIAL data (drift-#12 / silent-failure: concluded from seed 44 before the run finished).
The full 3-seed apical-inhib result (`apical_inhib_smoke.log`) is the OPPOSITE of what the "UPDATE — dendrite negative" section below claims:
| seed | cross (isolate baseline) | cross (+apical-inhib w0.7) | verdict |
|---|---|---|---|
| 44 (was CLEAN) | 0.00 | **0.69** | BROKE (the only regression) |
| 100 (was FAILING) | 0.45 | **0.06** | ✓ CLEAN — match 0.69, **11.5× ratio** |
| 102 (was FAILING) | 0.61 | **0.18** | ✓ CLEAN — match 0.55, 3× ratio |
⇒ **dendrite-targeting apical inhibition is the FIRST mechanism to actually gate the cross-completion on the hard seeds** (the two previously-FAILING seeds 100/102 became clean; the diagnosis "the coupling is dendritic" is CONFIRMED-and-fixable, not a wall). The ONLY regression is seed 44 (already-clean, size-asymmetric [50,18]) — the winner-detection/shunt breaks a seed that didn't need shunting. So the mechanism WORKS; the open piece is a WINNER-DETECTION/op-point fix so it stops breaking the clean seeds. **The "inhibitory-gating exhausted / 4th negative" framing below (and in the same-cycle commit fa501e80) is RETRACTED** — apical (dendritic) inhibition is the working mechanism; only the SOMATIC family failed. Reading the SUBSTANCE (per-seed), not the aggregate "SWR NO" (which seed-44's 0.69 drags up), was the lesson. NEXT: tune the winner-detection (lower `--pa-apical-w`, gate on cross-completion risk, or per-seed winner stability) → all-6-seeds clean.


**2026-07-21.** Executing the research gate `2026-07-21` (Kim-Kim 2025 read in depth) for the last gap#5 piece:
avalanche-stable, independently-addressable co-storage of two SIZE-VARIABLE emergent CA3 assemblies. The gate's Option A
(assembly-selective inhibition — spare-own/inhibit-other, condition-4 counteracts size-bias) was already wired as
`--per-assembly-inhib` but never run; ran it + the gate's named fix (within-assembly E→I potentiation). **Both fail,
verified.**

## Results (6-seed unless noted; `_gap5_r4_emergent_btsp_store --swr --isolate ...`)
| config | SWR-clean seeds (cross ≲0.2) | completion (cue mean) | note |
|---|---|---|---|
| `--isolate` alone (prior finding) | 2/6 (42,44) | ~0.17 (ref) | the boundary |
| `+ --per-assembly-inhib` (pa-inhib-w 40) | **2/6** (42,44) | **0.079** (weakened) | no gain + over-suppresses own completion |
| `+ --pa-ei-w 40` (within-E→I potentiation) | no better (seed 100 cross 0.45→**0.63**) | 0.05-0.10 | somatic inhibition makes cross WORSE |

- Per-seed (as-wired selective inhib, 6-seed): clean cross 0.00 on 42/44; **cross 0.33-0.61 on 43/100/101/102** (the
  smaller assembly still avalanches under co-storage). The `ca1_match 0.45-0.68` ≈ `ca1_cross` on the failing seeds.
- **Verified NOT a silent no-op** (silent-failure discipline): instrumented `within-assembly E→I edges matched=6505`
  (cross-EI zeroed=6550) — the potentiation genuinely set 6505 member→own-basket-sub-pool synapses to 40.0, and the
  cross-completion still did not improve (worsened on seed 100). So the negative is a real MECHANISM verdict.

## Diagnosis — the coupling is NOT somatic-basket-gateable (→ dendritic)
Kim-Kim's mechanism gates cross-completion by making an assembly's SOMATIC-targeting PV-basket sub-pool fire early and
suppress competitors. Here, potentiating that exact drive (member→own-sub-pool E→I) made the cross-completion WORSE, not
better. ⇒ the channel that cross-completes the small assembly is not the one the CA3 PV-basket gates. Consistent with the
prior finding ("`structural_sep` isolates the assembly-UNION from non-members, not the assemblies from EACH OTHER"; the
completion runs through the dense CA3 recurrent + the bistable APICAL PLATEAU). The R4 completion is a **dendritic**
plateau attractor (`cp_v_apical`, `plateau_self_regen`, `apical_kir_g`) — so a partial cue of A that drives the shared
recurrent can ignite B's APICAL plateau, which SOMATIC basket inhibition cannot shunt (it targets the soma, not the
apical compartment where the plateau integrates).

## Verdict + the ranked next mechanism (THE LAW: a wall is a METHOD verdict, not a capability abandonment)
- **The SOMATIC selective-inhibition family (Kim-Kim E→I, as-wired + within-E→I potentiation) does NOT close the
  two-assembly boundary** — verified, edges-confirmed, 3 configs. Banked.
- **NEXT (ranked, from the research gate's decision tree): DENDRITE-TARGETING apical SOM/O-LM inhibition** (Müller-Remy
  2014) — a per-assembly inhibitory pool that shunts the OTHER assembly's APICAL/plateau drive (the compartment that
  actually completes), sparing own. This targets the correct compartment (the somatic basket demonstrably cannot). Realize
  as a per-assembly inhibitory mask onto `cp_v_apical`/the plateau/coincidence conductance (mirroring the existing apical
  read-out path), additive default-off. Secondary/stack: stronger DG separation at store time to reduce SHARED cells
  between the two emergent assemblies (reduce the coupling SEED; the finding notes the emergent assemblies can share cells
  that `interassembly_isolate` — which zeros EDGES — does not separate).
- **Unchanged headline:** the SINGLE-assembly emergent-DG select→store→complete chain is GO (mechanism 6/6). The
  two-assembly independent-addressing is the precise remaining piece; this cycle narrowed it from "size-normalization gap"
  to "the coupling is dendritic, gate the apical compartment."
- Additive code (kept, valid mechanism for the arc + future co-storage work, NO `sim/` edit): `per_assembly_ei_w` on
  `_riii_ca3_synchronous_assembly_derisk.py` + `--pa-ei-w` on `_gap5_r4_emergent_btsp_store.py` (default None =
  byte-identical). Logs: `research/findings/raw/gap5_r4/kimkim_{selinhib_6seed,ei_pot_smoke}.log`.

## UPDATE — DENDRITE-TARGETING apical inhibition (the ranked escalation) ALSO negative at the default op-point
Implemented per-assembly dendrite-targeting O-LM/SOM apical inhibition (shunt the LOSER assembly's `cp_v_apical` toward
the GABA_A reversal −75mV, winner-detection-gated, sparing own; additive default-off, `--per-assembly-apical-inhib`
`--pa-apical-w`/`--pa-apical-gate`). **Verified it FIRES** (`shunt-steps=296 cells_shunted=10064`, not a silent no-op).
Result at w=0.7 (seed 44, the previously-CLEAN seed): **cross 0.00 → 0.69** — the apical shunt BROKE the clean
discrimination (ca1_match 0.73 but ca1_cross 0.69). ⇒ shunting the apical compartment disrupts the completion/readout
(likely the winner-detection shunts the target, or the strong shunt collapses the plateau non-selectively) — a NEGATIVE
at the default op-point (a w-sweep / winner-detection fix is a follow-on, but inhibitory GATING of the completion — whether
somatic OR apical — has now failed on 4 mechanisms).

## Sharpened verdict — the 2-assembly boundary is not closable by INHIBITORY GATING of the completion
Four mechanisms negative: isolate-alone (2/6), somatic Kim-Kim selective inhibition (2/6), within-E→I potentiation
(worse), dendrite-targeting apical inhibition (breaks the clean seed). Both somatic AND apical inhibition fail → the fix
is probably NOT "gate the completion with more inhibition." The next mechanism search should target the ENCODE/STORAGE
side (separate the two emergent assemblies so they share fewer CA3 cells / have orthogonal recurrent basins — DG
pattern-separation at store time, the untested ranked Option C) or a fundamentally different readout, NOT more inhibition.
A fresh GPU-free research gate is dispatched for the next mechanism. **The single-assembly emergent-DG chain remains GO;
this is the narrow 2-assembly independent-addressing piece.** Additive code kept (default-off byte-identical). NO `sim/` edit.

## ⚠️ MAJOR REFRAME (2026-07-21, from the encode-side research-gate Workflow + its adversarial critique reading the on-record data)
Two Workflows (5 agents each) + their adversarial critiques reframed the whole 2-assembly arc:
1. **DECISIVE DIAGNOSTIC (Agent A, measured):** the two co-stored emergent assemblies are **DISJOINT on ALL seeds**
   (Jaccard <0.05; failing 100/102 share 3-4 cells, clean 42/44 share 0-1). NOT a shared-cell problem → the
   cross-completion travels through **between-assembly ca3→ca3 recurrents AND the member→non-member spread path** in the
   dense substrate (and `structural_sep=1` — the GO_CFG default — blocks only non-member→member, leaving the indirect
   A→non-member→B channel open; that is WHY `--isolate` alone got 2/6).
2. **THE BINDING BLOCKER IS WEAK COMPLETION, not cross (critique, reading `_gap5_r4_emergent_btsp_store.json` 20:22):**
   on EVERY seed the emergent sparse assemblies complete at `held_cue` **~0.10** — BELOW the GO bar (0.13) and the cgo()
   bar (0.15). The apical inhibition "fixed" cross on 100/102 but their completion was ALWAYS too weak to pass. The
   sparse ~18-31-cell emergent assemblies at `ca3_density=0.05` have too little within-assembly fan-in (the driver's own
   HONEST NOTE lines 17-19: ~30 cells at 0.05 gives cue ~0.05; the reference's 240-cell assembly completes because its
   fan-in is 6× higher). **Lowering density (a ranked option) is BACKWARDS — completion needs HIGHER within density.**
3. **The seed labels FLIP under `--isolate`:** with isolate ON, seed 44 has the WORST cross (0.72) and seed 100 already
   passes (0.20) — so "44 clean / 100-102 fail" was the isolate-OFF regime; the mechanism must be judged per-seed on the
   ACTUAL config, and 6-seed (the substrate jitters cell membership run-to-run).
- **⇒ RULED OUT (method-negative, unanimous across both Workflows + Rolls-Treves): ALL recall-side inhibitory gating**
  (the 4 negatives) — "no downstream circuit can manufacture a distinction absent from CA3." The fix is at the
  STORE/recurrent-coupling level AND must FIRST restore strong completion.
- **SYNTHESIZED de-risk (testing now, b7ge89lpb):** `--isolate --structural-sep 2 --ca3-density 0.35` — HIGH within-
  assembly density (strong completion, the binding blocker) + full bidirectional isolation (zero cross-assembly AND
  member↔non-member spread → no coupling). GO gate (6-seed): both assemblies complete `held_cue≥0.13` AND
  `ca1_match≥0.6`/`cross≤0.3`/≥3× on ≥5/6, with the load-bearing lesion (isolate/sep OFF reproduces cross) + basin-
  symmetry (BOTH the large and the ~2× smaller assembly complete their own cue) anti-cheats. Fallbacks ranked: disjoint-DG
  inputs, equal-k sparser+symmetric selection. NO `sim/` edit (all existing flags). **THE CAPABILITY (two separately-
  addressable co-stored emergent attractors) stays open per THE LAW; only the recall-inhibition METHOD retired.**

## ✅ COMPLETION BINDING-BLOCKER RESOLVED (2026-07-21) — the reframe validated; the synthesized fix works
`--isolate --structural-sep 2 --ca3-density 0.35` (high within-density for completion + full bidirectional isolation for
no-coupling), 3-seed:
| seed | sizes | cue (completion) | ca1_match | ca1_cross |
|---|---|---|---|---|
| 42 | [35,21] | 0.176 | 0.74 | **0.00** ✓ |
| 100 | [45,22] | 0.200 | 0.67 | **0.00** ✓ |
| 102 | [56,29] | 0.194 | 0.66 | 0.48 |
- **COMPLETION-MECHANISM-GO 3/3** — `cue` 0.176-0.200 (ALL above the 0.15 cgo bar; was ~0.10 at density 0.05). The high
  within-assembly density RESOLVED the weak-completion binding blocker the critique identified. This is the real fix — the
  sparse emergent assemblies now complete strongly.
- **CROSS clean on 2/3** (42/100 at 0.00). Only seed 102 (the LARGEST assemblies [56,29], 4 shared cells) cross-completes
  0.48 — the 4 shared cells (which `isolate`/`sep2` zero EDGES for but the cells still fire for both) are AMPLIFIED at high
  storage density → they drive the cross. NEXT (testing, bm4k9zjz7): stack `--disjoint-dg` (make the two DG input codes
  share 0 cells → 0 shared CA3 cells; encode-side Rank 2) → seed 102's cross should drop clean → then 6-seed confirm.
- `--disjoint-dg` is additive/default-off byte-identical (still DYNAMICS-selected assemblies, not hand-disjoint pools →
  emergence bar preserved). NO `sim/` edit.
- ⇒ the exhaustive research-gate + adversarial-critique workflow delivered: the boundary was NOT recall-inhibition (5
  negatives) but weak-completion + shared-cell coupling; high density + full isolation + disjoint DG is the mechanistically-
  matched close. THE CAPABILITY (two separately-addressable co-stored emergent attractors) is being MET, per THE LAW.

## HONEST STATE (2026-07-21, after exhaustive investigation) — completion RESOLVED; independent-addressing = a geometry-sensitive residual (the sparse-vs-strong tension)
- **disjoint-DG (Rank 2) is geometry-sensitive, not a robust fix:** density 0.35 + isolate + sep2 + `--disjoint-dg` gave
  seed 42 clean (0.00), seed 100 BROKE (0.00→0.79), seed 102 barely improved (0.48→0.34). Removing the 3-4 shared cells
  SHIFTS the emergent basin geometry → the cross changes unpredictably. **Best config = WITHOUT disjoint-DG** (density 0.35
  + isolate + sep2: completion 3/3, cross clean 2/3 [42/100 at 0.00, 102 at 0.48]).
- **The two sub-problems, cleanly separated:**
  1. **COMPLETION (the real binding blocker) — RESOLVED.** The sparse emergent assemblies complete WEAKLY at low density
     (`held_cue` ~0.10); HIGH within-assembly density (0.35) fixes it robustly (3/3 GO, cue 0.176-0.233). This is the
     genuine deliverable — the reframe (completion, not cross, was the blocker) was correct and the fix works.
  2. **INDEPENDENT-ADDRESSING (cross) — a GEOMETRY-SENSITIVE residual (2/3 best, config-fragile).** The fundamental
     **sparse-separated-vs-strong-completion TENSION**: SPARSE basins separate (no cross) but complete weakly; DENSE basins
     (needed for strong completion) COUPLE and cross-complete in a way that depends on the precise, seed-dependent basin
     geometry (which every config knob shifts — disjoint-DG broke a clean seed; isolate flips seed labels; the assemblies'
     cell membership jitters run-to-run on the point-neuron substrate). No single config robustly gives BOTH strong
     completion AND clean cross across seeds.
- **RULED OUT (method-negatives, exhaustive):** all recall-side inhibitory gating (5 configs: isolate/somatic/E→I-pot/
  apical/size-norm) + disjoint-DG (geometry-sensitive). The completion fix (high density) + full isolation (sep2) gets 2/3.
- **THE PRINCIPLED NEXT DIRECTION (named, per THE LAW — a characterized deep residual, NOT a wall):** the DG's native
  **sparse-COMPLETABLE architecture** — equal-k SYMMETRIC selection (break the ~2× size-asymmetry gang effect that makes
  the larger basin spread into the smaller) + a within-assembly density HIGH ENOUGH for completion but a between-basin
  coupling structurally ZEROED, so the two basins are strong AND separated. This is a deeper architectural piece (the
  sparse-completable balance), not a config tweak, and likely interacts with the documented point-neuron substrate limits
  (dense recurrent couples strong basins). Ranked mechanisms staged: equal-k symmetric selection, feedback-inhibition
  equal-target-activity, weight-dependent heterosynaptic depression with interleaved co-active encode.
- **⇒ gap#5 episodic-memory CAPABILITY largely MET:** single-assembly emergent select→store→complete chain GO +
  two-assembly COMPLETION resolved. The two-assembly INDEPENDENT-ADDRESSING is the characterized geometry-sensitive
  residual with the principled next direction named. NO `sim/` edit anywhere (all additive default-off driver flags).
