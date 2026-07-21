# gap#5 two-assembly co-storage — the SOMATIC selective-inhibition family (Kim-Kim E→I) VERIFIED-does-NOT-close the boundary; the cross-completion is NOT somatic-basket-gateable → the coupling is DENDRITIC (dendrite-targeting apical SOM is the ranked next mechanism). NO `sim/` edit (additive driver mechanism).

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
