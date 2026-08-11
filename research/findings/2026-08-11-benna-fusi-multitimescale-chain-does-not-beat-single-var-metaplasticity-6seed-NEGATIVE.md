---
type: finding
status: contributing
date: 2026-08-11
mechanism: Benna-Fusi multi-timescale consolidation CHAIN (cascade of coupled variables at increasing timescales) vs single-variable metaplasticity, against continual acquisition-at-scale forgetting
lane: H-memory / continual-learning
seeds: [42, 43, 44, 100, 101, 102]
verdict: NEGATIVE — the multi-timescale chain does NOT beat single-variable metaplasticity (chain <= single-var on every seed) and does NOT protect the oldest fact; the acquisition-at-scale residual is fact-CODE interference, not consolidation timescale
runner: research/runners/_teacher_loop_bennafusi_chain_derisk.py
artifacts:
  - research/findings/raw/bennafusi_chain_s42.json
  - research/findings/raw/bennafusi_chain_s43.json
  - research/findings/raw/bennafusi_chain_s44.json
  - research/findings/raw/bennafusi_chain_s100.json
  - research/findings/raw/bennafusi_chain_s101.json
  - research/findings/raw/bennafusi_chain_s102.json
instrument: N-sweep {8,16,32} facts acquired by e-prop weight change; frac_recalled over the acquired set. Arms: vanilla (no metaplasticity), single_var (one per-synapse consolidation variable — the prior sub-threshold GO mechanism), chain (a Benna-Fusi cascade of coupled variables at geometric timescales), chain_lesion (chain frozen), chain_permute (chain on wrong synapses). de-clamp held constant across arms. SIM_BACKEND=numpy.
---

# Benna-Fusi multi-timescale CHAIN does NOT beat single-variable metaplasticity (6-seed NEGATIVE) — the continual acquisition-at-scale residual is fact-CODE interference, not consolidation timescale

The metaplastic de-risk (`2026-08-11-metaplastic-acquisition-continual-learning-6seed-NOGO...`) found a single per-synapse
consolidation variable moves acquisition-at-scale forgetting the right way but SUB-THRESHOLD, and — crucially — never
protects the very-OLDEST fact. It named a true multi-timescale **Benna-Fusi chain** (slow variables protect old memories
the single variable cannot reach) as the next mechanism. This de-risk builds + 6-seed-tests it. It does not help.

## Result — 6 seeds, N∈{8,16,32} (`research/findings/raw/bennafusi_chain_s*.json`)

<!--derived-->
Cross-seed mean frac_recalled (derived over the 6 per-seed artifacts): **chain 0.422 vs single_var 0.443 vs vanilla
0.271**. The chain is ≤ the single variable on EVERY seed (42: 0.375 vs 0.406; 43: 0.344 vs 0.375; 44: 0.500 vs 0.500;
100: 0.375 vs 0.406; 101: 0.5625 vs 0.5625; 102: 0.375 vs 0.406) — adding timescales adds NOTHING over one variable,
and is marginally worse. Both metaplastic arms beat vanilla (0.271), so metaplasticity helps in general, but the
timescale COUNT does not. The **oldest fact stays unprotected**: `oldest_fact_acc[chain]` = 0.0 on 5/6 seeds (0.083 on
seed 101) — the chain's slow variables do NOT reach fact 0, exactly the gap the single variable also had.

## Scope / honesty + the redirected next mechanism (per THE LAW — the capability stays OPEN)

<!--derived-->
NO-EXTERNAL-NEEDED: this is a quantitative comparison of two consolidation implementations, not a fundamental-limit
claim — the biology (Benna & Fusi 2016) is cited and the negative REDIRECTS to a different mechanism class, so no new
external read is required to bank it.

- **What it rules out:** more consolidation TIMESCALES is not the lever. The single-variable metaplasticity already
  captured whatever consolidation buys here; the cascade adds no protection for the oldest fact.
- **Why (the redirect):** the oldest fact is overwritten because later facts REUSE its synapses — an interference
  problem in the fact CODE, not a timescale problem in the consolidation. No amount of "protect these synapses more
  slowly" helps when the SAME synapses must encode the new fact. **Named next mechanism: orthogonalized / SPARSE fact
  codes** (so facts occupy disjoint synapse subsets and do not compete) — the same "keep codes disjoint" biology that
  worked for source-monitoring competitive-encoding and emergence-engine hetero-competition allocation (a THIRD lane
  where disjoint-codes-under-pressure is the crux). Secondary: neurogenesis / capacity growth as N scales.
- Load-bearing check held (chain_lesion collapses toward vanilla; chain_permute ≤ chance). Runner-side, reuse-by-import
  of `MetaplasticEpropNet` / `OnBridgeEpropNet`. NO `sim/` edit. de-clamp held constant across arms.
- Provenance note: the build agent DEFERRED before reporting (backgrounded its smoke + stopped); the coordinator
  recovered the uncommitted runner from the agent worktree, ran the 6-seed, and authored this finding.
