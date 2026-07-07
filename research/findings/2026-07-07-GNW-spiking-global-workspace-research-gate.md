# DEEP-RESEARCH GATE — a spiking Global Neuronal Workspace by UNIFYING the project's proto-workspace pieces (verdict: cheaply buildable; Rank-1 = the ignition curve)

**Date:** 2026-07-07
**Trigger:** the owner brought the Anthropic paper *Verbalizable Representations Form a Global Workspace in Language Models* (Transformer Circuits, 2026-07-06) and asked whether it lends useful insight. Gate = read-only, three-part (paper + primary GNW biology + ML global-workspace architectures), Workflow-orchestrated (`aa60e13ffffc3c202`), controller-synthesized.
**Verdict:** YES — a spiking Global Neuronal Workspace (GNW) is cheaply buildable by unifying the project's already-validated proto-pieces, and it is the *right* build because the spiking substrate can supply the one thing the paper says a transformer structurally LACKS: recurrent, all-or-none **ignition** + true long-range **broadcast**. The genuine residual is exactly ONE thing: a dedicated small pool of "workspace" neurons with an ignition threshold that all pieces read+write.

## What the paper establishes (the load-bearing claims to replicate)
1. **Membership test** (§1.2/§3.1): a representation is "in the workspace" iff **(report)** the model names it via its output projection AND **(causal swap)** swapping the vector changes the answer. Directly reproducible here.
2. **The dissociation** (§3.5.2): ablate the workspace → the model still parses, recalls facts, speaks fluently, but **fails multi-hop reasoning** → "the representations used for verbal report are the same ones that govern how the model silently reasons" (report-reps == reasoning-reps).
3. **Structural signatures** (§4): small selective capacity (occupancy plateau ≈ 25 vectors, ≤10% variance); broadcast (composes ~10× more broadly); an ignition-like sharpening at workspace onset.
4. **What transformers LACK** (§1.3/§9.3): the broadcast "occurs within a single feedforward pass rather than through recurrent loops"; "unclear whether this mirrors the sharp, competitive **ignition** … in the brain." **This is the project's opening.**

## (a) ISOLATE + QUANTIFY — how much GNW the project ALREADY has, and the tiny residual
| GNW property | Existing proto-piece | Coverage |
|---|---|---|
| Broadcast gating | `transmission_gate` / `cp_transmission_gain` (Logiaco-Abbott-Escola thalamocortical gating) | Strong |
| Capacity-limited hold | theta-gamma WM buffer (EMERGE-85, Lisman-Idiart N.15) | Strong |
| Sustained reverberation | NMDA WM latch (`enable_nmda`, `fused_nmda_update_and_current`, Wang 2002) | Strong |
| Verbalizability projection (the J-lens analog) | A→W concept→word read-out (`concept_speak_demo`, `UnifiedNeuralSpell`, EMERGE-67/68) | Strong — a *literal* J-lens |
| Access/ignition threshold | no-confab moat / Bogacz-Brown familiarity gate | **Partial** — a scalar gate on the read-out, NOT a regenerative all-or-none ignition of an assembly |
| Top-down selection | dlPFC content-selection / `elaborate()` (Hagoort Control) | Partial |
| Input modules → workspace | fronto-striatal reservoir comprehension (EMERGE-78/82) | Strong |
| One shared substrate | merged one-brain bridge (`nav_conv_merged_bridge`, `SimulationBridge.xp`) | Strong |

**The genuine residual (quantified):** the project has broadcast-gating, capacity-limited hold, reverberation, a verbalizability projection, top-down selection, and one shared substrate — but there is **no single explicit pool of "workspace neurons" that (i) is written by the modules through competitive access, (ii) undergoes a regenerative all-or-none IGNITION, and (iii) is broadcast back to all modules AND to the A→W read-out, so the SAME ignited assembly is simultaneously what the brain SAYS and what it REASONS with.** That is a composition-of-proven-pieces task, not a new mechanism class.

## (b) REFRAME — the real GNW circuit (why this is the right build)
- **Two spaces** (Dehaene-Kerszberg-Changeux 1998, *PNAS* 95:14529): a global workspace of long-range-axon neurons (L2/3 + L5 pyramidal, densest in PFC/parietal/cingulate — Dehaene-Changeux 2011, *Neuron* 70:200) + modular processors; workspace neurons "selectively mobilize or suppress, through descending connections," the processors.
- **IGNITION is the load-bearing nonlinearity** (Dehaene-Changeux 2011): sub-threshold stimuli produce feed-forward waves that "died out without triggering late global activation because insufficient self-sustaining reverberant activity was generated"; above threshold, **NMDA-dependent recurrent excitation** produces a self-amplifying, all-or-none, sustained assembly (a Hopf-bifurcation-like switch). Signatures: late ~300 ms amplification, P3b, long-range gamma/beta synchrony.
- **Competition + capacity**: mutual inhibition → only one coherent content ignites; metastable, reward/vigilance-modulated.
- **ML convergence on the SAME minimal design**: Goyal et al. 2022 (ICLR) "Shared Global Workspace" (bandwidth-limited shared workspace, write-competition, broadcast-back); Blum & Blum 2022 (PNAS) Conscious Turing Machine (single-slot STM, Up-Tree competition, Down-Tree broadcast).

**Reframe verdict:** do NOT emulate the transformer's depth-broadcast — build the brain's actual mechanism the transformer lacks: a small pool of spiking workspace neurons with recurrent NMDA reverberation + an all-or-none ignition threshold + lateral-inhibition capacity, that the comprehension reservoir/composer/dlPFC WRITE to via competitive `transmission_gate`-gated routes and that is BROADCAST BACK to them AND to the A→W read-out — making report-reps == reasoning-reps **true by construction**, with a genuine ignition the paper says LLMs only approximate.

## (c) RANKED cheap-first mechanisms (each with a single-variable de-risk + the report+causal-swap membership test)
- **RANK 1 (cheapest, FIRST) — workspace pool with ignition.** ~200-500-neuron `workspace` region, recurrent NMDA (reuse `enable_nmda` + `DLPFC_ATTRACTOR_WEIGHT=30`) + FS lateral inhibition; drive from one concept assembly through a `transmission_gate` route; read with A→W. **Single variable = input drive amplitude → show all-or-none ignition** (sub-threshold die-out vs sustained reverberation, the Dehaene-Changeux die-out-vs-ignite curve). Membership test: ignited assembly speakable via A→W; NMDA-lesion kills BOTH report + downstream use; sub-threshold drive un-reportable. NO `sim/` edit.
- **RANK 2 — competitive access (only one ignites).** ≥2 candidate assemblies compete via `transmission_gate` + mutual inhibition → sharp WTA crossover; causal salience swap flips the reported concept.
- **RANK 3 — broadcast-back drives reasoning (report==reasoning).** Route the ignited pool back into the multi-hop `query_chain` intermediate; accuracy holds with the pool, collapses when lesioned (the §3.5.2 dissociation); swapping the ignited intermediate redirects the conclusion.
- **RANK 4 — selectivity/capacity + eviction.** Sequence of concepts → holds a few; a new ignition evicts the old (occupancy plateau).
- **RANK 5 (defer) — unify the moat INTO ignition.** Unfamiliar cue fails to ignite → abstain; familiar ignites → speaks (touches the validated moat, so after 1-3 prove the pool).

## (d) VERDICT + cheapest FIRST de-risk
A spiking GNW is cheaply buildable by unifying the existing proto-pieces and is the scientifically correct next direction — it lets the project SURPASS the paper by implementing the genuine recurrent ignition + long-range broadcast the paper says a feedforward transformer only emulates over depth. **Cheapest FIRST de-risk = RANK 1:** build the ~300-neuron `workspace` pool (recurrent NMDA + FS lateral inhibition), drive it from one concept assembly, and demonstrate the all-or-none ignition curve (sub-threshold die-out vs above-threshold sustained reverberation) with the NMDA-lesion control (kills the sustained branch). Single variable: input drive amplitude. Reuses only validated primitives; no new mechanism class; likely no `sim/` edit.

## Sources
Gurnee/Sofroniew/Lindsey et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits, 2026-07-06 (full text read). Dehaene-Kerszberg-Changeux 1998 *PNAS* 95:14529. Dehaene-Changeux 2011 *Neuron* 70:200-227. Baars 1988. Goyal et al. 2022 ICLR (arXiv 2103.01197). Blum & Blum 2022 *PNAS*. Project primitives verified: `sim/bridge.py` (`cp_transmission_gain`, `cp_nmda_neuron_mask` per-region mask L1259/L6355), `sim/kernels.py` (`fused_nmda_update_and_current`), `nav_conv_merged_bridge.py` (`DLPFC_ATTRACTOR_WEIGHT=30`, `_build_dlpfc_loop_population`), `concept_speak_demo.py` (A→W read-out).

## Next concrete action
BUILD the Rank-1 ignition-curve de-risk (`research/runners/_gnw_rung1_ignition_curve_derisk.py`): the workspace pool + the drive-amplitude sweep + the NMDA-lesion anti-cheat; 1-seed CPU smoke → controller fans the 6-seed(-blind) sweep across cores → adversarially verify → commit both remotes.
