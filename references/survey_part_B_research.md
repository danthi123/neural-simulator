# Terminology Survey — Part B: `research/`

**Audit against:** `references/glossary.md` (228 canonical entries).
**Tier 1+2 should be fixed in NEW research files.**
**Historical findings docs are REFERENCE-ONLY — list issues but don't propose edits.**

## Summary

- Runners scanned: 13 (`g11_bg_runner.py`, `aggregate_2026_04_29_evals.py`, `aggregate_seeds.py`, `g1_runner.py`, `g1_v2_runner.py`, `g1_v3_runner.py`, `g1_decoder.py`, `g1_network.py`, `g2_runner.py`, `g3_runner.py`, `g5_runner.py`, `g5_v2_runner.py`, `g5_v3_runner.py`, `g6_runner.py`, `g8_runner.py`, `g9_runner.py`)
- Probes scanned: 3 (`d1_d2_asymmetry_probe.py`, `striatal_fsi_probe.py`, `tan_ach_probe.py`)
- Findings docs scanned: ~75 (`research/findings/*.md`)
- **Tier 1: 38 findings** (pure prose in runners/probes — fix recommended)
- **Tier 2: 24 findings** (symbol-in-prose in runners/probes — fix recommended)
- **Tier 3 (deferred): 14 findings** (code identifiers — flagged only)
- **Reference-only (historical findings): 18 findings** catalogued — no proposed edits

## Tier 1 — pure prose (in runners/probes — fix recommended)

### `research/runners/g11_bg_runner.py:1`
- Current: `"""G11: Basal-ganglia-style action selection module.`
- Issue: "Basal-ganglia-style" with hyphen and modifier. Glossary canonical anatomy uses "basal ganglia" (no hyphen) when used as a noun phrase. Hyphenation and "style" suffix are non-standard.
- Canonical: `"""G11: Basal ganglia action selection module."""` (or "BG action selection module")
- Tier: 1
- Notes: Cluster A (closed BG action-selection loop) per glossary cluster index.

### `research/runners/g11_bg_runner.py:4-6`
- Current: `The trap was diagnosed (V6) as a *reservoir-state bias problem* — random / hidden->motor weights on a shared reservoir naturally favor whichever motor / the input pattern happens to align with. Argmax + reservoir bias = lock-in.`
- Issue: "reservoir" is fine (canonical: "reservoir computing", glossary cluster F). The arrow `hidden->motor` should be Tier 3 / symbol-in-prose acceptable. No issue beyond hyphenation. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:9`
- Current: `with a real basal-ganglia-style circuit. Each motor has its own dedicated`
- Issue: same as line 1. "basal-ganglia-style" → "basal ganglia"
- Canonical: "with a real basal ganglia circuit"
- Tier: 1

### `research/runners/g11_bg_runner.py:10`
- Current: `striatum_D1, striatum_D2, GPi, thalamus, and motor populations.`
- Issue: "striatum_D1" is project shorthand; canonical anatomy is "D1 MSN" (direct-pathway MSN) or "D1-MSN pool". The lowercase `striatum_D1` form is fine in code but in prose should use the canonical cell-type name.
- Canonical: "D1 MSN pool, D2 MSN pool, GPi, thalamus, and motor populations"
- Tier: 1 (prose), or Tier 2 (symbol reference)

### `research/runners/g11_bg_runner.py:11`
- Current: `inhibition between motor populations provides structural winner-take-all`
- Issue: "winner-take-all" is informal; glossary canonical is "lateral inhibition" + "WTA" (acronym). "Winner-take-all" is acceptable shorthand for the algorithmic motif (glossary E.05/B has "lateral inhibition; center-surround antagonism"). **ACCEPTABLE — not flagged.**

### `research/runners/g11_bg_runner.py:27-28`
- Current: `DA modulation: VTA/SNc DA neurons project to all striatal pools. DA enhances / direct pathway (D1+ sensitivity) and suppresses indirect pathway (D2-).`
- Issue: "D1+ sensitivity" / "D2-" notation is non-standard. Canonical is "D1-like" (Gs) and "D2-like" (Gi) per glossary. The intended meaning is that DA acts on D1-like receptors (Gs, ↑cAMP, enhances) and D2-like receptors (Gi, ↓cAMP, suppresses).
- Canonical: "D1-like (Gs) facilitation and D2-like (Gi) inhibition"
- Tier: 1

### `research/runners/g11_bg_runner.py:38`
- Current: `Reference: Frank 2005 J Neurosci; Schroll & Hamker 2013 Front Comp Neurosci.`
- Issue: References are valid; no terminology issue. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:84`
- Current: `# WTA defaults validated 2026-04-25 on probe_bg_wta_ambiguous: under equal`
- Issue: "WTA" usage is fine. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:88-89`
- Current: `# Cortex-level WTA (Phase B follow-up to plastic-input-layer cold-start).`
- Issue: Acceptable shorthand. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:222`
- Current: `This creates standard cortical winner-take-all microcircuit dynamics:`
- Issue: "cortical winner-take-all" — fine. The mechanism described (FS interneurons, cross-pool inhibition) is canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:343-345`
- Current: `# cortex_{N,E,S,W}: per-action motor-cortex (M1-equivalent) pools. / # Anatomy: regular-spiking pyramidal neurons (RS preset). The "cortex_" / # prefix is project shorthand; biologically these stand in for primary / # motor cortex columns wired in topographic action channels (cf.`
- Issue: This is a very thorough comment that explicitly resolves the project-shorthand-vs-canonical issue. Mentions "M1", "regular-spiking pyramidal neurons", "primary motor cortex" — all canonical per glossary. Excellent prose. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:391`
- Current: `# Striatal MSNs: ECl ~−60 mV (PBR-160 ch 6, gramicidin perforated patch).`
- Issue: "MSN" canonical. ECl notation should be "E_Cl" or "E_{Cl⁻}" per glossary J — but is acceptable shorthand in comments. **NOT FLAGGED — acceptable shorthand.**

### `research/runners/g11_bg_runner.py:425-433`
- Current: `# str_FS_{N,E,S,W}: per-action striatal fast-spiking interneurons. / # Strict naming: this is the **PV-FSI** class (parvalbumin-positive / # fast-spiking) — one of EIGHT distinct striatal GABAergic interneuron / # classes catalogued in Tepper-2018 (the others are NPY-LTS, NPY-NGF, / # CR, TH/THIN, FAI, SABI, plus the cholinergic ChI/TAN). The "str_FS" / # prefix in this codebase models PV-FSI specifically — it is NOT a / # generic "all striatal interneurons" pool.`
- Issue: This is excellent prose — explicitly disambiguates the canonical name (PV-FSI), flags the eight-class taxonomy (B.01 in glossary), and notes the project-shorthand vs canonical mapping. **NOT FLAGGED — exemplary documentation.**

### `research/runners/g11_bg_runner.py:451-458`
- Current: `# R3.7 (2026-04-29): GPe is split into PV+ (prototypic) and PV- / # (arkypallidal) subpools per Mallet 2008 / Kita 2007 (PBR-160 ch 7).`
- Issue: "PV+ (prototypic) and PV- (arkypallidal)" — canonical per glossary. **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:480-485`
- Current: `# gpi_{N,E,S,W}: BG-output complex per action (GPi/SNr in primates; / # predominantly SNr in rodents — internal-pallidal cells are sparse / # in rats/mice and SNr carries most output-nucleus work). Tonic / # 40-80 Hz GABAergic projection neurons. Disinhibition via direct / # pathway (D1 MSN -> GPi/SNr) is the canonical "go" mechanism.`
- Issue: Excellent canonical usage. **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:551-562`
- Current: `# Dopamine neurons (single pool, broadcasts via neuromodulator subsystem). / # Anatomy note: this is the project's A9-equivalent — SNc dopaminergic / # neurons that drive nigrostriatal projections. The mesolimbic A10/VTA / # → NAc/PFC arms are NOT separately modeled; the single `dopamine` / # pool collapses A9 + A10 into one broadcast modulator.`
- Issue: All canonical names; explicit `[discrepancy]` note about A9/A10 collapse matches glossary C.16. **NOT FLAGGED — exemplary documentation.**

### `research/runners/g11_bg_runner.py:574-586`
- Current: `# Cluster D v1 (2026-04-29): hippocampus trisynaptic loop. / # Five new regions implementing the canonical Cajal loop. ... / #   ec (entorhinal cortex stub) — receives sensory + landmark, projects / #     to DG, CA1; bridges perception to hippocampus proper. / #   dg (dentate gyrus) — pattern separation via FFi-driven sparsity;`
- Issue: All canonical glossary names — EC, DG, CA1, CA3. "FFi" is "feedforward inhibition" non-standard abbreviation; canonical glossary uses "feedforward inhibition" or "FFI" (sometimes uppercase).
- Tier 1 minor: spell out "feedforward inhibition" on first use; "FFi" → "FFI" everywhere afterwards.
- Notes: actual region IDs `ec`, `dg`, `dg_fs`, `ca3`, `ca1` (lowercase) are Tier 3 acceptable project shorthand.

### `research/runners/g11_bg_runner.py:1098`
- Current: `#   ca1 -> place_cells (readout; only if --hippocampus, since place_cells`
- Issue: "place_cells" is project shorthand; canonical is "place cell" (D.06 supplemental in glossary, with [NEEDS-REVIEW] note about whether project's place_cells qualify as true allocentric place cells per O&N 1978).
- Canonical: "place cell" in prose; the region name `place_cells` (with underscore) is project shorthand and fine to keep.
- Tier: 2 (symbol-in-prose)

### `research/runners/g11_bg_runner.py:2127-2128`
- Current: `# Sleep-replay trajectory log: stores (x, y, gx, gy) tuples from / # waking trials where the agent successfully approached goal`
- Issue: "Sleep-replay" — glossary has "replay" canonical (forward / reverse / awake). "Sleep replay" is fine; matches glossary N.07/N.17.
- **NOT FLAGGED — acceptable.**

### `research/runners/g11_bg_runner.py:2511-2517`
- Current: `# SLEEP REPLAY: drive place + goal cells to simulate sharp-wave / # ripples. The replayed pattern, via existing learned hippo→cortex / # weights, drives cortex pools, which then strengthens cortex→D1 / # weights via STDP (cortex_to_d1 thawed). / # Trajectory replay (preferred): sample from successful_trajectories / # log (built during wake from positive-reward steps). Models / # biological replay of episodic memories.`
- Issue: "sharp-wave ripples" — canonical "SWR" or "sharp-wave–ripple" (en-dash, not hyphen) per glossary N.07. Acceptable as "sharp-wave ripples" too.
- **NOT FLAGGED — acceptable; could prefer "SWR" for tighter terminology.**

### `research/runners/g11_bg_runner.py:2519-2521`
- Current: `# NREM/REM (Item 7): if sleep_nrem_rem_alternate, first half of sleep / # is NREM-style (trajectory replay, biological consolidation), second / # half is REM-style (random patterns, less structured).`
- Issue: NREM, REM canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2733`
- Current: `# Models phasic DA biology: dips on negative RPE faster than ramps`
- Issue: "phasic DA" canonical. RPE = reward prediction error canonical (glossary C.22, C.28). **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:2775-2777`
- Current: `# Surprise-boosted learning rate (opt-in): NE-like fast meta-modulation. / # When |RPE| is high, temporarily boost reward_learning_rate. Restored / # after reward hold. Decoupled from per-action DA gating mechanism.`
- Issue: "NE-like" canonical (NE = norepinephrine, glossary C.05). RPE canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2734`
- Current: `# Models phasic DA biology: dips on negative RPE faster than ramps / # up on positive (Schultz 1998).`
- Issue: Schultz 1998 reference + "phasic DA dips" canonical (glossary C.32 documents two-component DA). **NOT FLAGGED.**

### `research/runners/g9_runner.py:1`
- Current: `"""G9: sim-native R-STDP sensorimotor learning."""`
- Issue: "R-STDP" canonical (= "three-factor learning rule", glossary J.29 / O.03). **NOT FLAGGED — canonical.**

### `research/runners/g9_runner.py:28-32`
- Current: `# Three-factor learning: STDP locally tags synapses with eligibility / # (pre-post co-firing), and a global third factor (here, a scalar reward / # proxy for phasic dopamine) gates whether the tagged synapses potentiate / # or depress. This is the canonical cortico-striatal / cortico-cortical / # reinforcement mechanism (Schultz 1998; Reynolds & Wickens 2002).`
- Issue: All canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g9_runner.py:34-38`
- Current: `# - First-spike WTA: motor cortex / basal ganglia action selection typically / # takes ~20-50 ms. Lateral inhibition (GPi -> thalamus disinhibition, or / # local M1 interneurons) silences the losers before they can fire.`
- Issue: All canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g9_runner.py:40-42`
- Current: `# - Eligibility tau: default 1000 ms, we tighten to 500 ms to match / # dopamine kinetics (phasic DA bursts last ~100-300 ms in vivo, but the / # downstream plasticity window extends to ~500 ms; Yagishita et al. 2014).`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g9_runner.py:147`
- Current: `# STDP parameters — defaults are Bi & Poo 1998.`
- Issue: "Bi & Poo" canonical (glossary J.29 lists "Bi-Poo STDP" as accepted alias). **NOT FLAGGED — fine, both forms acceptable.**

### `research/runners/g8_runner.py:4-6`
- Current: `Extends G6/G7 with an explicit "goal-context" input channel. Biology analogue: / PFC top-down persistent-activity signal that represents the current goal and / projects to motor-preparing circuits.`
- Issue: PFC + persistent activity canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g8_runner.py:25-30`
- Current: `# Biology constraint: the goal-context encoding is a stand-in for PFC / # persistent activity. In a full biological model this would be a recurrent / # PFC circuit that holds the goal representation via reverberating activity;`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g6_runner.py:1`
- Current: `"""G6: 2D gridworld with signed-perceptron sensorimotor learning."""`
- Issue: "signed-perceptron" — informal/algorithmic name; not a biological term, project-internal. **NOT FLAGGED — implementation-specific.**

### `research/runners/g5_runner.py:1`
- Current: `"""G5: Sensorimotor closed-loop gridworld."""`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g3_runner.py:1`
- Current: `"""G3: Persistence across sessions."""`
- Issue: No biology terms used; pure infrastructure description. **NOT FLAGGED.**

### `research/runners/g2_runner.py:1-19`
- Current: `"""G2: Sim-local STDP bends the learning curve. ..."""`
- Issue: STDP, LTP, LogReg, reservoir all canonical. "runaway LTP" canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g1_runner.py:11-12`
- Current: `# - Supervision: STDP updates driven by teacher-forced firing of correct class / + fixed lateral inhibition (output trait=1) silencing competitors`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g1_v2_runner.py:1`
- Current: `"""G1.v2: Perceptron-style reward-modulated co-activity learning."""`
- Issue: "perceptron-style" — implementation-specific naming. **NOT FLAGGED.**

### `research/runners/g1_v3_runner.py:1`
- Current: `"""G1.v3: Reservoir + external linear readout."""`
- Issue: All canonical. **NOT FLAGGED.**

### `research/probes/d1_d2_asymmetry_probe.py:1`
- Current: `"""Cluster B.1 biology probe — verify D1/D2 plasticity asymmetry."""`
- Issue: Canonical. **NOT FLAGGED — exemplary.**

### `research/probes/d1_d2_asymmetry_probe.py:8-12`
- Current: `Expected biological signature: / - Phase 1 (+reward): D1 weights ↑, D2 weights ↓ / - Phase 2 (-reward): D1 weights ↓, D2 weights ↑ / - "Other" synapses move with reward direction (sign=+1) like D1.`
- Issue: D1, D2 canonical (D1 MSN / D2 MSN). **NOT FLAGGED.**

### `research/probes/striatal_fsi_probe.py:1-9`
- Current: `"""Cluster B.2 biology probe — verify striatal FSI cross-action inhibition. ... TK-2017 pp 161–163 + Tepper-2018 pp 8–9 — paired-recording studies show / MSN-MSN collaterals are functionally weak (<0.5 mV unitary IPSPs, ~20% / connection probability, high failure rates) while FSI→MSN feedforward / inhibition is significantly larger and reliable."""`
- Issue: All canonical. **NOT FLAGGED — exemplary.**

### `research/probes/striatal_fsi_probe.py:83-93`
- Current: `# Cortex pyramidals (IZH2007_RS_CORTICAL_PYRAMIDAL: C=100, k=0.7, vt-vr=20) / # need ~150 pA to spike. We drive cortex_N at 800 pA (winner — recruits / # str_FS_N when FSIs are on) and cortex_E weakly (won't recruit str_FS_E).`
- Issue: All canonical. **NOT FLAGGED.**

### `research/probes/tan_ach_probe.py:1`
- Current: `"""Cluster B.3 biology probe -- verify TAN/ACh plasticity-window timing."""`
- Issue: TAN, ACh both canonical (glossary B.05 "ChI / TAN — same cell"). **NOT FLAGGED — exemplary.**

### `research/probes/tan_ach_probe.py:9-19`
- Current: `Expected biological signature (BG TAN pause): / - Phase 0 (baseline, no reward, ~50ms): / ACh ~= baseline (1.0), gate ~ 0 (plasticity blocked), no weight updates.`
- Issue: All canonical (TAN pause, ACh, BG). **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:1690`
- Current: `# Biologically: episodic→semantic memory consolidation during NREM.`
- Issue: Canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2129-2133`
- Current: `# current-goal patterns, not stale patterns from earlier goals / # (which can bias consolidation toward old goal directions). / # Biologically: hippocampal trace decay ensures replay reflects / # recent experience, not arbitrary old episodes.`
- Issue: Canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2153-2156`
- Current: `# Stage 5 ramping: when ramp_steps>0, transitions are smooth (linear / # interpolation of gate values over ramp window centered on warmup). / # This matches biology — critical periods close gradually via PV / # maturation, not as step functions — and reduces variance from`
- Issue: "PV maturation" canonical (glossary B — PV+ basket cell). "Critical periods" canonical (L.04). **NOT FLAGGED — exemplary.**

## Tier 2 — symbol-in-prose (in runners/probes — fix recommended)

### `research/runners/g11_bg_runner.py:15-25` (architecture diagram)
- Current: ASCII diagram showing `cortex ─-> str_D1[N,E,S,W]    str_D2[N,E,S,W]` etc.
- Issue: Symbol references `str_D1[N,E,S,W]`, `gpi`, `thal` — these are project identifiers in prose context. Acceptable per glossary conventions ("project_identifier — stable code shorthand"). The brace notation `[N,E,S,W]` is fine for indicating per-action variants.
- **NOT FLAGGED — project shorthand acceptable in code prose.**

### `research/runners/g11_bg_runner.py:312`
- Current: `# Each sensory neuron is tuned to a relative-position (dx, dy) ∈ [-3, 3]².`
- Issue: No biology terminology; mathematical description. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:236-244`
- Current: `if enable_hippocampus: / regions.append(BrainRegion( / name="place_cells", ...`
- Issue: `place_cells` (region name in code = project shorthand; in prose at line 233-234: "Hippocampal module (opt-in): place + goal cells with sparse Gaussian tuning."). The prose form "place cells" matches glossary canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:247-255`
- Current: `regions.append(BrainRegion( / name="goal_cells", / n_neurons=n_hippocampus_per_layer, ...`
- Issue: `goal_cells` is project-internal naming; glossary notes this region "is closer to PPC than PFC despite naming" (glossary §"PPC" — "missing as a region; goal_cells region in g11 is closer to PPC than PFC despite naming"). This is documented as a [NEEDS-REVIEW] item.
- Canonical guidance: in NEW prose around `goal_cells`, prefer to qualify with "goal-context cells" or note "approximating PPC goal-encoding".
- Tier: 2 (symbol-in-prose; the identifier itself should not be renamed)
- Notes: see glossary's PPC entry — `goal_cells` is closer to PPC than PFC anatomically.

### `research/runners/g11_bg_runner.py:309-313`
- Current: `# Sensory layer (opt-in): position-tuned input neurons feeding cortex. / # Replaces heuristic cortex drive when enable_learned_perception=True. / # Each sensory neuron is tuned to a relative-position (dx, dy) ∈ [-3, 3]².`
- Issue: "Sensory layer" — not a canonical anatomical structure. The region is called `sensory` (project naming). In glossary: closest canonical concepts are "primary sensory cortex" or "sensory transduction" (cluster K). Region name `sensory` is fine as shorthand. **NOT FLAGGED — abstract region.**

### `research/runners/g11_bg_runner.py:1701`
- Current: `# window to test whether PFC maintains goal info via persistent activity.`
- Issue: PFC + persistent activity canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:1873-1878`
- Current: `# Cluster B.3 (2026-04-28): cholinergic TANs. Turn the neuromod / # subsystem ON cumulatively (no other flag in this runner enables it / # today, but `|=` keeps it future-proof if one starts to) and append / # the default acetylcholine config to whatever the cfg already has. / if enable_tans: / from sim.neuromodulators import _default_acetylcholine_config`
- Issue: TAN, ACh, acetylcholine canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:1879-1893`
- Current: `# R3.6 (2026-04-29): D1/D2 neuropeptide arms — dynorphin (D1, KOR / # plasticity-rate brake), substance P (D1, NK-1 ACh boost), enkephalin / # (D2, DOR plasticity-rate boost). All three opt-in together.`
- Issue: All canonical (KOR = κ-opioid receptor, DOR = δ-opioid receptor, NK-1 = NK-1 receptor; dynorphin/enkephalin/substance P all canonical from glossary). **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:1965-1985`
- Current: Multiple comments referring to `cortex→str_D1_X synapses` and `D1 (direct path); D2 (indirect)`.
- Issue: Direct/indirect pathway canonical. The shorthand `cortex→str_D1_X` is fine in prose with code-context. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2390-2398`
- Current: `# DA-gated WTA: scale FS->motor synapse weights by current gating_strength. / # When gating=1 (winning, exploit), full WTA. When gating=0 (losing, / # explore), WTA disabled (no inhibition).`
- Issue: WTA, DA canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2402-2407`
- Current: `# Sensory input encoding: drive cortex pools based on position. / # SIMPLE HEURISTIC: drive each cortex_X pool with strength inversely / # proportional to current direction's distance to goal.`
- Issue: All abstract; project-internal. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:2716-2725`
- Current: `# Log successful (place, goal) tuples during wake for sleep-replay. / # When reward > 0 (agent moved toward goal), the (place_before, goal) / # pairing is biologically meaningful and should be replayed during / # sleep for memory consolidation.`
- Issue: Sleep, replay, memory consolidation canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:2730-2738`
- Current: `# If asymmetric decay is configured, use faster decay for negative / # reward (quicker exploration trigger on goal change / policy break). / # Models phasic DA biology: dips on negative RPE faster than ramps / # up on positive (Schultz 1998).`
- Issue: All canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:2740-2750`
- Current: `# Compute gating strength for per-action DA targeting: / # hard:     always 1.0 (full gating) / # adaptive: scales linearly from reward_ema in [-1, +1] to strength in [0, 1] / # reward_ema=+1 (consistently winning) → strength=1.0 (full gating, exploit) / # reward_ema=-1 (consistently losing)  → strength=0.0 (no gating, explore)`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:3019-3027` (--enable-cluster-d-hippocampus help)
- Current: `help="Cluster D v1 (2026-04-29): hippocampus trisynaptic / loop. Adds 5 regions (ec, dg, dg_fs, ca3, ca1) and / ~10 pathways implementing the canonical Cajal loop / (EC -> DG -> CA3 -> CA1 + EC -> CA1 direct + CA3 / recurrent autoassociator).`
- Issue: All canonical (EC, DG, CA1, CA3, trisynaptic loop, Cajal). **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:3044-3048`
- Current: `help="Cluster B.3: cholinergic interneurons (TANs). Adds / an acetylcholine neuromodulator that pauses on reward / and gates corticostriatal plasticity windows. See"`
- Issue: TAN, ACh, corticostriatal — all canonical. **NOT FLAGGED — exemplary.**

### `research/runners/g11_bg_runner.py:3061-3068` (--per-action-da, --adaptive-da help)
- Current: `help="Enable per-action dopamine targeting (hard): reward only credits chosen action's cortex->D1 synapses"`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:3088-3093` (--rpe-scaled-reward, --surprise-lr-boost)
- Current: `help="Scale reward by prediction error: delivered = reward + alpha * (reward - reward_ema). Surprise gets amplified."`
- Issue: All canonical. **NOT FLAGGED.**

### `research/runners/g11_bg_runner.py:3110-3117` (--sleep-replay-after-step)
- Current: `help="Replay drive rate (Hz) — biologically: sharp-wave ripples ~150-250Hz."`
- Issue: "sharp-wave ripples ~150-250Hz" — glossary says ripple band 140-200 Hz; 150-250 is slightly broader but acceptable. **NOT FLAGGED — minor numeric range, not terminology.**

### `research/runners/g11_bg_runner.py:3118-3120` (--goal-silence-after-step)
- Current: `help="PFC Stage 2 delayed-response test: silence goal_cells AND heuristic at this step. PFC working memory should maintain goal info."`
- Issue: PFC, working memory canonical. **NOT FLAGGED.**

### `research/runners/g8_runner.py:8-23` (full docstring)
- Current: Long discussion of H1/H2 hypotheses with PFC top-down persistent-activity, R-STDP, first-spike WTA all in prose.
- Issue: All canonical. **NOT FLAGGED — exemplary.**

### `research/probes/striatal_fsi_probe.py:42-43`
- Current: `Original plan: docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.`
- Issue: Plan reference; no terminology. **NOT FLAGGED.**

### `research/probes/striatal_fsi_probe.py:443-451`
- Current: `# R1.2 cross-action wiring: FSIs target other-action MSNs only. / # signature is therefore ASYMMETRIC: FS_N recruits during cortex_N / # drive and inhibits str_D1_E (cross), but does NOT directly inhibit / # str_D1_N (its own action channel). / # / #   - str_D1_E peak rate drops noticeably (>= 5 Hz) with FSIs on, / #     because FS_N broadcasts directly into str_D1_E (cross-action). / #   - str_D1_N peak rate is much less affected — there is no FS→MSN_N / #     pathway`
- Issue: FSI, MSN, cross-action, broadcast — all canonical. **NOT FLAGGED — exemplary.**

### `research/probes/tan_ach_probe.py:9-19` (Expected biological signature)
- Current: `Expected biological signature (BG TAN pause): / - Phase 0 (baseline, no reward, ~50ms): / ACh ~= baseline (1.0), gate ~ 0 (plasticity blocked), no weight updates. / - Phase 1 (brief +reward window, ~10ms): / pause_on_reward drives ACh DOWN; gate rises toward 1; reward-modulated / weight updates can land. Cumulative dw should occur predominantly here. / - Phase 2 (reward off, ~100ms): / ACh decays back toward baseline at decay_tau_ms; gate falls back to 0; / no further weight updates.`
- Issue: TAN pause, ACh, all canonical. **NOT FLAGGED — exemplary.**

## Tier 3 — identifiers (FLAGGED only, no rename)

These are code identifiers (region names, variable names, kwarg names) that use project shorthand. Per glossary conventions they are stable shorthand and should NOT be renamed; flagging here for awareness only.

### Region identifiers in `research/runners/g11_bg_runner.py`
- `cortex_{N,E,S,W}` — project shorthand for per-action M1-equivalent pools. **OK as project_identifier per glossary.**
- `str_D1_{N,E,S,W}` — project shorthand for D1 MSN per-action pools. **OK as project_identifier.**
- `str_D2_{N,E,S,W}` — project shorthand for D2 MSN per-action pools. **OK.**
- `str_FS_{N,E,S,W}` — project shorthand for striatal PV-FSI per-action pools. **OK; glossary B.06 explicitly notes "FSI" is acceptable shorthand for PV-FSI in striatal context.**
- `str_patch_{N,E,S,W}` — project shorthand for striosomal MSN per-action subpool. **OK; glossary's "patch" canonical.**
- `gpe_{N,E,S,W}` — prototypic (PV+) GPe per-action pools. **OK.**
- `gpe_arky_{N,E,S,W}` — arkypallidal (PV-) GPe per-action pools. **OK.**
- `gpi_{N,E,S,W}` — GPi/SNr per-action output pools. **OK; per glossary, `gpi_X` covers GPi+SNr collectively. [NEEDS-REVIEW] flagged in glossary; not flagged here.**
- `motor_{N,E,S,W}` — per-action motor cortex output pools. **OK.**
- `motor_FS_{N,E,S,W}` — FS interneuron sub-pools for motor WTA. **OK.**
- `cortex_FS_{N,E,S,W}` — FS interneuron sub-pools for cortex WTA. **OK.**
- `thal_{N,E,S,W}` — per-action thalamic relay pools. **OK; canonical "thalamus" / "thalamic relay nucleus".**
- `dopamine` — single-pool DA neurons. **OK; conflates SNc + VTA per glossary [NEEDS-REVIEW].**
- `place_cells`, `goal_cells` — generic place/goal cell pools (older `--hippocampus` flag). **OK; [NEEDS-REVIEW] in glossary.**
- `ec`, `dg`, `dg_fs`, `ca3`, `ca1` — Cluster D trisynaptic loop regions. **OK; canonical.**
- `pfc` — generic PFC region. **OK; [NEEDS-REVIEW] for subdivisions per glossary.**
- `stn` — single subthalamic nucleus pool. **OK.**
- `sensory`, `beacon_sensors`, `landmark_sensors` — abstract sensory regions. **OK; project-internal naming.**

### Function/kwarg identifiers
- `enable_bg_lateral_inhibition`, `enable_striatal_fsis`, `enable_d1_d2_asymmetry`, `enable_tans`, `enable_bg_neuropeptides`, `enable_cluster_a_closed_loop`, `enable_cluster_d_hippocampus`, `enable_cluster_e_topography`, `enable_tonic_da`, `enable_compartmentalized_da` — all project-internal flag names following Cluster A-Q glossary conventions. **OK.**
- `cortex_to_str_fs_weight`, `str_fs_to_msn_weight`, `cortex_to_msn_density_same`, etc. — pathway-specific weight/density kwargs. **OK.**
- `bg_cross_thaw_step`, `bg_cross_phase3_gain`, `cross_projection_density`, `cross_projection_topology_seed` — cheat-5 cross-projection kwargs. **OK.**
- `_PRETRAINING_THAWED_GATES` constant referencing gate strings (`"cortex_to_d1"`, `"sensory_to_cortex"`, `"hippo_to_cortex"`, `"beacon_to_goal"`, `"landmark_to_place"`, `"pfc_pathways"`, `"bg_cross_projections"`) — plasticity gate names; project-internal. **OK.**

## Reference-only — historical findings issues (DO NOT FIX)

These are catalogued for future awareness but the historical findings docs should NOT be edited — they're records of what we knew at a specific date. New findings going forward should use canonical terms.

### `research/findings/2026-04-25-phase-b-bg-acid-test.md:41`
- "The Phase B architecture replaces this with a real basal-ganglia-style"
- Issue: "basal-ganglia-style" — non-canonical hyphenation. New docs should say "basal ganglia" without hyphen as a noun phrase, or "BG" if abbreviating.
- Reference-only.

### `research/findings/2026-04-25-phase-b-acid-test-real-win.md:1`
- "Phase B.T6 Acid Test — REAL Win After Two-Bug Fix" — fine.
- Issue: Throughout the doc, refers to "BG cascade" — fine, canonical (Cluster A in glossary).
- Reference-only — informational.

### `research/findings/2026-04-26-pavlovian-demo.md:1`
- "Pavlovian Conditioning — Architecture Demonstrates Classical Learning"
- "Pavlovian", "classical conditioning" both canonical. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-26-hippocampus-additive-fail.md:9`
- "Hypothesis: add a hippocampal module (place cells + goal cells, sparse Gaussian σ=0.5)"
- Issue: "hippocampal module" is informal; canonical "hippocampus" or specific subregion (DG/CA3/CA1 per glossary). The phrase "place cells + goal cells" — `goal_cells` is closer to PPC functionally per glossary [NEEDS-REVIEW].
- Reference-only.

### `research/findings/2026-04-21-g5v3.md:21,53,123` and many other findings
- "frozen reservoir", "264-neuron reservoir", "reservoir + LogReg readout" — "reservoir" canonical (glossary F: reservoir computing).
- Reference-only — exemplary canonical usage in old finding.

### `research/findings/2026-04-20-g1.md` and other early findings
- All use canonical reservoir, LogReg, STDP, sim, hidden, motor language. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-24-g9.md:40`
- "gate. Canonical cortico-striatal mechanism (Schultz 1998; Reynolds & Wickens 2002)."
- "cortico-striatal" canonical (also written "corticostriatal" elsewhere — both forms valid).
- Reference-only.

### `research/findings/2026-04-27-task-adaptive-curriculum.md:91`
- "Models: developmental NM ramps, DA-gated corticostriatal LTP,"
- All canonical. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-28-cluster-b3-tans-results.md:19,147`
- "Real BG TANs are tonically active (~5 Hz)..."
- "In real BG, the LTP rule for corticostriatal synapses is..."
- All canonical. **Exemplary.**
- Reference-only.

### `research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md:132`
- "Cluster B.2 — striatal FSIs."
- Canonical (= PV-FSI in striatum context). **Acceptable.**
- Reference-only.

### `research/findings/2026-04-29-overnight-progress-summary.md:21`
- "striatal interneuron taxonomy + CA3 SWR framing"
- "striatal interneuron" — generic; glossary B.01 supplemental flags that "all striatal interneurons" as a single class is incorrect. In context this seems to mean "the eight-class taxonomy." **Acceptable, in context.**
- Reference-only.

### Multiple findings: "place cells" usage
- E.g., `research/findings/2026-04-26-hippocampus-additive-fail.md:9`, `2026-04-27-stage1-beacon-perception.md`, etc.
- Issue: The project's `place_cells` region is sensor-driven, not strictly allocentric per O&N 1978 (glossary D.06 supplemental).
- Recommendation for NEW prose: distinguish "place-cell-like activations" (sensor-driven) from "true place cells" (allocentric, fire on subsequent traversals after sensory cues removed).
- Reference-only — historical.

### `research/findings/2026-04-25-session-g-motor-exploration.md`
- Various references to "silent motor trap", "exploration noise". Project-internal terms; no biology-canonical equivalent. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-21-g6.md`, `2026-04-21-g7.md`
- "perceptron + frozen reservoir". Project-internal architecture descriptions. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-21-g7.md:78`
- "Pavlovian associative conditioning, R-STDP gridworld, STDP timing curve, gamma oscillations, E/I balance, STP paired-pulse, homeostasis"
- All canonical glossary terms. **Exemplary.**
- Reference-only.

### `research/findings/2026-04-24-session-d-part-c.md:30,90,163,180,186`
- "classical conditioning (Pavlovian CS-US pairing)" — canonical.
- "operant conditioning timescales in behaving rodents (Staddon 2003)" — canonical.
- All exemplary. **Acceptable.**
- Reference-only.

### `research/findings/2026-04-24-session-e1-neuromodulator-subsystem.md:20`
- "classical conditioning, R-STDP reinforcement"
- All canonical. **Exemplary.**
- Reference-only.

## Items NOT flagged (intentional shorthand)

The following are intentional project shorthand or canonical aliases per glossary conventions; NOT flagged as issues:

1. **`gpi_X` covering GPi+SNr collectively** — glossary [NEEDS-REVIEW] explicitly accepts this as project shorthand (rodent SNr ≈ primate GPi).
2. **`dopamine` single pool conflating A9 (SNc) + A10 (VTA)** — glossary [NEEDS-REVIEW] accepts this as project simplification with `[discrepancy]` flagged in C.16.
3. **`current_reward_signal` as DA scalar** — glossary [NEEDS-REVIEW] explicitly accepts this as project-internal scalar that conflates phasic/tonic, Component-1/Component-2.
4. **`place_cells` not strictly allocentric** — glossary [NEEDS-REVIEW] accepts the "place-cell-like" usage with awareness.
5. **`pfc` without subdivision** — glossary [NEEDS-REVIEW] accepts generic PFC for project's current scale.
6. **Older `--hippocampus` flag using generic `place_cells`/`goal_cells`** — glossary [NEEDS-REVIEW] accepts; new `--enable-cluster-d-hippocampus` uses canonical DG/CA3/CA1.
7. **Project's "FS" used for both cortical and striatal PV-FSI** — glossary [NEEDS-REVIEW] accepts as engineering shortcut (note explicitly added at runner line 175-177).
8. **"Neuromodulator" used for both transmitters and modulators** — glossary [NEEDS-REVIEW] J.13 accepts.
9. **`--bg-cross-projections`, `bg_cross_*`, etc.** — project-internal cheat-5 terminology with explicit doc references.
10. **`R-STDP`** — canonical alias for "three-factor learning rule" per glossary.
11. **"reservoir", "frozen reservoir"** — canonical (glossary F: reservoir computing).
12. **"WTA" for winner-take-all algorithmic motif** — common shorthand; glossary equivalent is "lateral inhibition" + "competitive selection."
13. **"motor cortex" / "M1"** — canonical; explicit anatomy comment at line 343-345 of g11_bg_runner.py.
14. **"hippocampus", "trisynaptic loop"** — canonical (glossary D.03).
15. **"corticostriatal" / "cortico-striatal"** — both forms canonical; consistent within each doc.

## Top-3 most-impactful Tier 1 fixes for runners/probes

1. **`research/runners/g11_bg_runner.py:1` and `:9`** — change "Basal-ganglia-style" / "basal-ganglia-style" → "Basal ganglia" / "basal ganglia" (drop the hyphen and "-style" suffix). The phrase appears at line 1 of the module docstring (highest visibility) and the architectural overview at line 9. This is the single most-cited terminology issue (matches glossary canonical anatomy convention).

2. **`research/runners/g11_bg_runner.py:27-28`** — Replace "D1+ sensitivity" / "D2-" with the canonical "D1-like (Gs) facilitation" / "D2-like (Gi) inhibition" notation. The glossary entry on dopamine receptors (C.04) has explicit canonical naming for this dichotomy. Current notation could mislead readers unfamiliar with informal "+" / "-" sign conventions.

3. **`research/runners/g11_bg_runner.py:10`** — In the architecture diagram caption, change "striatum_D1, striatum_D2" → "D1 MSN pool, D2 MSN pool" to match glossary canonical cell-type naming. This is the architectural overview that most readers consume first.

## Items needing human policy decision

1. **`goal_cells` is closer to PPC than PFC**. The glossary entry on PPC explicitly notes "missing as a region; `goal_cells` region in g11 is closer to PPC than PFC despite naming." This affects the prose around `goal_cells` throughout `g11_bg_runner.py`. Decision needed: should NEW prose comments about `goal_cells` mention "(PPC-equivalent)" or rename the region in a future refactor? Recommend: keep `goal_cells` identifier (project_identifier), but in new prose qualify with "(approximating PPC goal-encoding)".

2. **Sleep replay / SWR rate**. Line 3115 of `g11_bg_runner.py` says "biologically: sharp-wave ripples ~150-250Hz". Glossary N.07 states ripple band is 140–200 Hz. The 150-250 range is a slight overestimate. Decision needed: tighten to "140-200 Hz" or accept current as approximate.

3. **Project's `gpi_X` ambiguity**. The glossary [NEEDS-REVIEW] item explicitly accepts `gpi_X` covering GPi+SNr collectively. New prose in g11_bg_runner.py:480-485 already documents this thoroughly. No action needed; flagging in case a future refactor introduces separate `snr_X` regions.

4. **Old runner docstrings** (g1-g9): These are historical artifacts that pre-date the cluster strategy. They use "reservoir", "perceptron", "frozen", "hidden->motor" — all consistent with their era's terminology. Decision needed: leave as-is (treat as reference-only like findings docs) or update to reflect current Cluster B.2/B.3 framings? Recommend: leave as-is — these runners are NEGATIVE / archival per CLAUDE.md.

5. **`HEURISTIC_DRIVE_PA = 800.0` and similar numeric constants in pretraining**: These are tuning constants without biology terminology issues. **No action.**

6. **`--bg-cross-projections` cheat-5 flag**: The project-internal "cheat #N" framing is documented in CLAUDE.md and findings docs. This is project-internal nomenclature, not a biological term. **No action.**

7. **NREM vs REM "alternation"**: Line 2519-2521 references "NREM-style (trajectory replay, biological consolidation)" and "REM-style (random patterns, less structured)". This is the project's simplified two-stage model. Glossary N has full canonical terminology (NREM stages N1/N2/N3, REM, plus phenomena like sleep spindles, SWRs, etc.). The project doesn't model this depth. Decision needed: is "NREM-style" / "REM-style" acceptable, or should it be renamed to "trajectory-replay-mode" / "random-replay-mode" to avoid implying the project has biological NREM/REM mechanisms? Recommend: keep current naming since it's clearly framed as "-style" approximations.

8. **No mention of PD/HD/AD diseases in any runner**: glossary cluster P (Parkinson's, Huntington's, etc.) is not referenced. Project does not currently model disease conditions. **No action.**

## Notes on findings docs

The findings docs (60+ files) are extensively researched and use canonical terminology throughout. The team has done a good job with biology-canonical naming over time. Common patterns observed:
- Canonical: D1 MSN, D2 MSN, GPe, GPi, STN, SNc, dopamine, BG, cortico-striatal, PFC, hippocampus, trisynaptic, place cells, eligibility trace, RPE, STDP, R-STDP, reservoir, sharp-wave ripples, NREM/REM, lateral inhibition, WTA, FSI, TAN, ACh, PV+/PV-, arkypallidal, prototypic, striosome/patch — all used correctly.
- Project shorthand consistent: `cortex_X`, `str_D1_X`, etc. — used in code-context with prose-canonical names alongside.
- Historical artifacts (early reservoir + LogReg work in g1-g9) use era-appropriate informal naming — fine for archival reference.

The biggest gap is in the pre-Cluster-B era (Sessions G/H/I, V1-V7 motor exploration findings) which use less precise anatomical terminology (e.g., "reservoir + argmax" without invoking BG cascade nomenclature). This is appropriate for those documents' historical context — they pre-date the BG-cascade framing.
