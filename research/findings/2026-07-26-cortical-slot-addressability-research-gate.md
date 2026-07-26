# Cortical slot addressability — RESEARCH GATE. The `pool→slot` residual is NOT a plasticity-rule tension: it is an **unmeasured somatic SELECTION failure** of a WTA pool that is built but wired as *global* inhibition, plus **two confounds in the arms that closed the alternatives** (a hard-coded `5.0` clip; a mechanism attributed to a Hebbian branch that is not on the executed path). Selective addressability is reachable with **config-only + ~15 additive runner lines**, no `sim/` edit and no architectural redesign.

**Date:** 2026-07-26
**Type:** READ-ONLY deep-research gate. No build, no run, no `sim/` edit, no bar moved. Local corpus (RAG + direct source
reads of our own code/findings) → substrate reads at file:line → catalog/Kandel → external literature (2 papers read,
1 in full text).
**Fires because (research-gate conditions a, b, d, e):** the cortical store closed on a *soft* verdict —
"**CHARACTERIZED LIMIT: ~5% is the ceiling of what BTSP's plateau-gated write can express**… ⇒ NEXT IS STRUCTURAL"
(`2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md`, final section). Per the
SURPASS sharpening that is a DISGUISED boundary and mandates this round before any structural build.
**Predecessor gate (read, built on, not duplicated):** `2026-07-25-ca1-sparsification-research-gate-scope.md`. Its own
closing scope note already flagged THIS as the separate, harder problem: *"the DOWNSTREAM one-of-N attractor/WTA
selectivity … the slot WTA collapses to a single dominant winner, seed-variable ~chance"*. This gate is that problem.

---

## EXECUTIVE SUMMARY

1. **The stated residual is not established.** The record's mechanism is "a broad *coactivity* rule and a selective
   plateau rule compete on the same synapses, and the broad one wins", cited to `sim/bridge.py:838`. **That line is not
   on the executed path.** `:838` is the `enable_branchless_plasticity` branch (default `False`, `sim/config.py:196`);
   `_consol_cortical_store_probe.py` sets neither `enable_branchless_plasticity` nor `hebbian_rate_window`
   (`sim/config.py:554`, default `False`), so the Hebbian rule actually running is `sim/bridge.py:7710-7717` — the
   **causal spike-coincidence** form `delta = lr·(w_max − w)` gated on `pre_fired[t−1] AND post_fired[t]`. It is
   **post-SPIKE gated**. Therefore, *if slot selection worked, Hebbian would already be selective by construction.*
   The uniform per-slot mass is evidence that **every slot's neurons SPIKE in every write window** — and that quantity
   **has never been measured**, although the probe already collects it (§1.3).
2. **The genuine structural residual, restated:** the `comp_attr` slots have a dedicated inhibitory pool
   (`comp_attr_inh`, `nmda_compositional_consolidation.py:278-289`) whose stated design intent is *"WTA between them via
   a shared inhibitory pool = one-of-N selective ignition"* (`:250-252`) — but it is wired as a **single shared pool that
   every slot drives and that inhibits every slot including the winner**. That is **global/symmetric inhibition**, which
   this project has independently shown to be **non-selective or causally inert for selection three separate times**
   (EMERGE-41: winner set identical FS-on vs FS-lesion, overlap 1.00; riii CA3: sparsifies 0.43→0.21 but ratio
   unchanged 1.16×; EMERGE-11: over-suppression across the whole drive sweep). The project's **validated** selective
   motif is per-competitor FS pools with **cross**-inhibition (`sel_FS_X → sel_Y≠X`), shipped and multi-seed GO in
   `research/runners/biased_competition_buffer.py:169-176`.
3. **Two of the arms that closed the alternatives are confounded** (both code-verified, §1.4):
   - **The "synaptic scaling is a hard 5.0 ceiling" refutation is an artifact of an unrelated hard-coded literal.**
     `sim/bridge.py:8704`: `_hw_max = cfg.hebbian_max_weight if cfg.enable_hebbian_learning else 5.0`. That arm ran
     `--no-hebbian --syn-scaling`, so the homeostasis clip nested inside the synaptic-scaling block fell back to the
     **literal 5.0** and clipped every weight every step. "Mass pins at **exactly** 5.0 regardless of scaling rate" is
     that constant, not synaptic scaling's dynamics. Synaptic scaling as a non-coactivity bound is **untested**.
   - **A third plastic rule was running the whole time and is absent from the 7-hypothesis ledger.** `enable_stdp`
     defaults `True` (`sim/config.py:598`) and `concept_to_comp_attr` is `plastic=True`, so STDP
     (`stdp_a_plus=0.012`, `w_max=8.0` from `BASE`) was also writing to `pool→slot` in every arm, including the
     "BTSP alone" arms.
4. **The cheapest surpass mechanisms are already in the engine, default-off, and were never tried on this pathway:**
   `btsp_mean_subtract` (Miller-MacKay 1994 **subtractive** normalization, `sim/config.py:396`, implemented
   `sim/bridge.py:8153-8194`) and `fused_htm_winner_inactive_depression` (`sim/kernels.py:497`, the competitive-learning
   term whose ablation moved on-substrate held-out selectivity **0.20 → 0.96** at EMERGE-39, 6/6 seeds).
5. **VERDICT: reachable cheaply — NO architectural redesign.** Rank 0 is a *free* instrumentation read that decides
   between the two live diagnoses. Rank 1 is ~15 additive default-off runner lines at an operating point already
   validated in this repo. Ranks 2–3 are config-only / one committed kernel. The honest residual risk is a
   **tuning knife-edge** (the recurrent/inhibitory operating point), not a substrate limit.

---

## (MOVE a) ISOLATE + QUANTIFY THE GENUINE RESIDUAL

### a.1 — The pathway, read at file:line

| element | where | value |
|---|---|---|
| slot excitatory assembly | `nmda_compositional_consolidation.py:257-263` | `comp_attr_{s}`, `n_per=120`, `exc_fraction=1.0`, internal density 0.20, `plastic_internal=False` |
| slot self-loop (Wang-2002 hold) | `:265-269` | `comp_attr_s → comp_attr_s`, density 0.20, `weight_mean=comp_self_weight` (**12.0**), `nmda_slow`, gate `nmda_attractor` |
| **shared** inhibitory pool | `:278-283` | `comp_attr_inh`, `n = n_per*0.5 = 60`, `exc_fraction=0.0`, `internal_density=0.0` |
| slot → inh | `:285-286` | density 0.30, weight 3.0, `plastic=False` |
| inh → slot (**all** slots, incl. the driver) | `:287-289` | density 0.30, weight `comp_wta_weight` (**5.0**), `plastic=False` |
| `pool→slot` cortical store | `:291-302` | **every** noun+adj pool → **every** slot, density **0.15**, `weight_mean=comp_pool_slot_weight` (1.5), `plastic=True`, gate `concept_to_comp_attr` |
| ca1 → slot (the validated route) | `:271-276` | `plastic=True`, gate `ca1_to_comp_attr`, `coincidence_detector=comp_dendritic` |

`N = 3` facts / 3 slots (`:89`, `CONSOLIDATED_FACTS = FACTS_ALL[:3]`).

### a.2 — Which plasticity rules actually touch `pool→slot` (all defaults verified)

| rule | enabled? | executed code | gated on |
|---|---|---|---|
| Hebbian | `enable_hebbian=True` in `BASE` (`_consol_direct_weight_probe.py:33`) | **`sim/bridge.py:7710-7717`** — `pre_mask = cp_prev_firing_states[row]`, `post_mask = fired_this_step[col]`, `delta = lr·(hebbian_max_weight − w)` | **postsynaptic SPIKE** |
| STDP | **`enable_stdp: bool = True`** (`sim/config.py:598`), never disabled by the probe | fused STDP, `stdp_w_max=8.0` (`BASE:32`) | spike timing |
| BTSP | `comp_btsp=True` (probe `:44`) | `sim/bridge.py:8060-8200` | **apical plateau** `max(v_apical − v_hold, 0)` |
| homeostasis (threshold adapt) | `enable_homeostasis: bool = True` (`sim/config.py:567`) | `sim/bridge.py:~8654` | per-neuron rate EMA |
| Hebbian decay + gated clip | with Hebbian on | `sim/bridge.py:7735-7760` | every step |

**Two of the three plastic rules writing this pathway are postsynaptic-SPIKE-gated.** The third (BTSP) is
plateau-gated and is *already verified exclusive* (measured `v_apical` during the clamped write: target **−9.3 / −9.6 /
−9.8 mV** vs non-targets **−66 mV**, `v_hold = −50`). So the whole selectivity question reduces to one unmeasured
quantity: **do the non-target slots SPIKE during a write window?**

### a.3 — That quantity is ALREADY COLLECTED by the probe and simply not reported (FREE)

`_consol_cortical_store_probe.py:115` allocates `pool_fire[i]` as a **full-network** vector
(`np.zeros(int(b.cp_membrane_potential_v.shape[0]))`) and `:137` accumulates `to_host(b.cp_firing_states)` over every
step of fact *i*'s 30-step write window. It is then used only as a presynaptic weighting (`:178-181`). But
`pool_fire[i][slot[j]].mean()` is **slot j's somatic firing during fact i's write window** — the decisive number — and
it is thrown away. Reporting it costs **one line** and zero extra compute.

The same applies on the read side: the recall block (`:193-210`) already accumulates a full-network `acc`, so the
per-slot **and** `comp_attr_inh` firing during recall is free.

⇒ **The record's central mechanistic claim ("the broad rule wins") and the alternative ("selection never happened, so
both rules see the same postsynaptic set") make opposite predictions about a number the harness already computes.**

### a.4 — Two arms that closed alternatives are CONFOUNDED (code-verified)

**(i) "Synaptic scaling → mass pins at exactly 5.0, a hard ceiling, selectivity gone (0/3)" — CONFOUNDED.**
The clip that produced the 5.0 lives *inside* the synaptic-scaling block and reads:
```
sim/bridge.py:8703-8704
    _hw_min = cfg.hebbian_min_weight if cfg.enable_hebbian_learning else 0.01
    _hw_max = cfg.hebbian_max_weight if cfg.enable_hebbian_learning else 5.0
```
That arm was run with `--no-hebbian` (probe `:242`), so `_hw_max` took the **hard-coded literal 5.0** and clipped
`cp_connections.data` **every step**. The observation that motivated the "hard ceiling / headroom" theory —
*"pins at exactly 5.0 regardless of scaling rate — so the rate is not even the operative variable"* — is the signature
of a constant clip, not of scaling. **Synaptic scaling as a non-coactivity bound has not actually been tested**, and
the "unifying explanation" (BTSP drives to whatever ceiling exists; selectivity requires headroom) rests on it.
*Runner-side fix, no `sim/` edit:* keep `enable_hebbian_learning=True` (so `_hw_max` is settable) and set
`hebbian_learning_rate=0.0` — the Hebbian potentiation delta is then exactly zero while the bound comes from
`hebbian_max_weight`.

**(ii) The 7-hypothesis ledger omits STDP.** `enable_stdp` is `True` by default and `concept_to_comp_attr` is plastic.
Every arm labelled "BTSP alone" or "Hebbian off" still had a broad, spike-timing-gated, soft-bound rule
(`stdp_w_max=8.0`) writing the same synapses. `BASE` never sets `no_stdp` (the flag exists — `:340-341` sets
`cfg.enable_stdp=False` when `args.no_stdp`).

**(iii) A citation slip worth recording** (not a confound, but it mis-states the mechanism): the finding's
`bridge.py:838` (`delta = lr·coact·(w_max − w)`, "a SOFT bound driven by COACTIVITY") is inside
`_apply_branchless_hebbian`, reachable only when `enable_branchless_plasticity` is `True` (`sim/config.py:196`,
default `False`). The executed rule (`:7710-7717`) has a **constant** delta and no coactivity term at all.

### a.5 — The read side is also a suspect, and its operating point was never swept

The recall window is `read_steps=60` at `dt=0.5 ms` = **30 ms** (probe `:40`, `:203`), with `nmda_attractor` re-opened
(`:196`) so each slot's `nmda_slow` self-loop at `weight_mean=12.0` / density 0.20 over 120 neurons is live. Compare the
project's own validated accumulator-WTA operating point: `sel_recurrent_weight=0.35` with an explicit
**α<1, "never self-ignites"** constraint (`biased_competition_buffer.py:114-115`, and `:169` *"symmetric
over-inhibition is unstable"*). A self-igniting recurrent pool resolves to a **seed-dependent intrinsic winner
independent of its input** — which is exactly the previously-recorded behaviour of this same slot WTA
(*"collapses to a single dominant winner, seed-variable ~chance"*,
`2026-07-25-ca1-sparsification-research-gate-scope.md` scope note). Neither `comp_self_weight`, `comp_wta_weight`, nor
`read_steps` has been swept on the cortical-store probe.

### a.6 — The genuine residual, in one sentence

> **A fact's pools must make ONE slot's neurons spike and the other slots' neurons not spike — during the write (so the
> two spike-gated rules become selective) and during the read (so a small weight bias becomes a categorical answer).
> The machinery for that (a dedicated inhibitory pool + a recurrent attractor) is BUILT but wired as GLOBAL symmetric
> inhibition with an un-calibrated recurrent gain, and whether it selects has never been measured.**

The all-to-all `pool→slot` density-0.15 broadcast is a **contributing** structural fact (it makes the *feedforward*
drive to all slots near-identical, giving the competition nothing to amplify), but it is not by itself the blocker:
a working one-of-N competition converts a small feedforward asymmetry — which the exclusive apical plateau supplies
via `apical_g_couple_to_soma` — into a categorical winner. Sparsifying the connectivity is Rank 4, not Rank 1.

---

## (MOVE b) REFRAME — how real cortex makes a memory slot ADDRESSABLE

Sources below were **read**, not rerank-skimmed. Where I only read an abstract/summary I say so.

### b.1 — Allocation is a COMPETITION won by transient excitability, not an address decoded by connectivity
**Josselyn & Silva / Han et al., CREB-based memory allocation** (read: *Nature Neuroscience* 12(11) summary +
*J. Neurosci.* 44(21) e0846232024 "Intrinsic Neural Excitability Biases Allocation and Overlap of Memory Engrams", via
search-result summaries — full texts not fetched). Within a region, **eligible neurons compete** for inclusion in an
engram; those with relatively **increased intrinsic excitability at the time of the event win**. Raising excitability
(CREB overexpression, or direct depolarization) *funnels* the memory into those neurons.
⇒ Biologically, "which slot stores this fact" is decided by a **transient excitability bias + a competition**, not by a
private wire. This is exactly the shape the substrate already has: the **apical teaching clamp is the excitability
bias** (host-supplied, and therefore a documented scaffold), and `comp_attr_inh` is meant to be the competition. The
missing half is the competition, not the bias.

### b.2 — The competition is enforced by LATERAL INHIBITION, and it is DENDRITE-targeting
**Stefanelli, Bertollini, Lüscher, Muller & Mendez, *Neuron* 89:1074 (2016), "Hippocampal Somatostatin Interneurons
Control the Size of Neuronal Memory Ensembles"** (read: PubMed 26875623 abstract + Cell/Neuron summary; full text not
fetched). Active granule cells recruit **SOM+ dendrite-targeting interneurons that inhibit the DENDRITES of
surrounding granule cells**, suppressing their recruitment (lower cFos in neighbours) — a microcircuit that sets the
**size** of the engram.
⇒ Two functionally distinct inhibitions are needed and they act on different compartments:
- **perisomatic (PV/basket)** inhibition → makes the **somatic spike** one-of-N → makes the spike-gated rules
  (Hebbian `:7710`, STDP) selective;
- **dendrite-targeting (SOM)** inhibition → makes the **apical plateau** one-of-N → makes BTSP selective.
The arc solved the *dendritic* half **by host clamp** and never built the *somatic* half. This is corroborated inside
this repo from the other direction: `2026-07-21-gap5-2assembly-selective-inhibition-family-NEGATIVE-...md` found
**somatic** selective inhibition insufficient for its (harder, recall-time) problem while **apical-targeting** worked.
Here the compartments are the other way round — the apical half is already exclusive, the somatic half is missing.

### b.3 — Cortex does NOT let every input contact every assembly (the structural rule)
**Catalog A.12 — Kincaid, Zheng & Wilson 1998 (via Bolam-2000 p.529)** (read, catalog `feature-catalog.md:224-233`):
each cortical axon contacts a striatal MSN with only **1–2 boutons**, and **close-neighbour MSNs do not share common
cortical inputs**. The catalog calls this a **"decorrelation rule"** and states its function directly: *with shared
inputs, MSNs would co-fire and lose discriminative power*. Sim status recorded there: **missing** — the project's
existing pools use dense shared wiring. `pool→slot` at all-to-all density 0.15 is precisely the "shared inputs" case
the rule forbids.
**Catalog L.02 — synapse elimination by activity competition** (read, `feature-catalog.md:4193-4200`; Kandel 6e Ch 48
pp.1198-1205): where several axons innervate one target, those firing **coincidently with the strongest** are
stabilized and the poorly-correlated ones are **eliminated** (NMJ → 1:1, climbing fibre → 1:1, ~50% cortical pruning).
⇒ Biology reaches sparse selective convergence by **competitive elimination**, not by being wired sparse at birth.

### b.4 — The formal reason a *bounded* Hebbian rule cannot sharpen, and a *subtractive* one can
**Miller & MacKay, "The Role of Constraints in Hebbian Learning", *Neural Computation* 6(1):100-126 (1994)** —
**read in full text** (PDF fetched and converted). The load-bearing results:
- Correlation-based rules are intrinsically unstable — *"either all synapses grow until each reaches the maximum
  allowed strength, or all synapses decay to zero"* (p.100). **This is exactly the arc's measured "tension"**: bound it
  and it saturates; unbind it and it runs away (3×10⁷).
- **Multiplicative** enforcement converges to the principal eigenvector and yields a **"graded"** receptive field in
  which most mutually correlated inputs are represented. **Subtractive** enforcement *"typically leads to a final state
  in which almost all synaptic strengths reach either the maximum or minimum allowed value"* and yields a field
  **"sharpened" to a subset of maximally correlated inputs** (p.100).
- With two equivalent input populations onto a common target, **multiplicative enforcement PREVENTS their segregation**
  when the populations are weakly correlated; **subtractive enforcement allows segregation** (p.100). Slots 0/1/2
  competing for the same pool population is structurally the ocular-dominance-segregation problem.
- §3 (p.115-116) gives the direction rule: *constraints on **output** cells affect the form of the individual
  **receptive fields**; constraints on **input** cells affect the form of the individual **projective fields**.*

**Every bound the arc tried is multiplicative.** `hebbian_max_weight` soft bound `(w_max − w)` — multiplicative.
BTSP `(w_max − w)` — multiplicative. Synaptic scaling `w ← w·(1 + rate·err)` (`sim/bridge.py:8683`) — multiplicative
(and, per §a.4, it never actually ran cleanly). Miller-MacKay says this class **cannot** sharpen to a subset; it
preserves the graded pattern. That is a *theoretical prediction of the exact ~5%-graded outcome measured*, and it names
the fix: a **subtractive** constraint.

Direction caveat, stated honestly: the recall metric is a **projective-field** question ("fact *i*'s pools — which slot
do they drive?"), so Miller-MacKay §3 says the strictly-matched constraint is over **input (presynaptic)** cells. The
engine's `btsp_mean_subtract` is per-**POST**synaptic (`sim/bridge.py:8156-8158`, "subtract, per POSTsynaptic cell, the
MEAN increment across that cell's afferents"), i.e. an RF constraint. Because the teaching clamp makes the
slot↔fact assignment a **permutation** (slot *i* is taught only with fact *i*'s pools), sharpening every column also
sharpens every row here — so the shipped RF form should work **for this 3×3 case**, but it is not the general
mechanism, and a per-presynaptic (PF) constraint is **not implemented**.

### b.5 — Competitive learning in practice: the winner depresses its INACTIVE inputs
This is the standard algorithmic realization of b.1-b.4 (von der Malsburg 1973 / Rumelhart-Zipser; HTM Spatial Pooler,
Cui-Ahmad-Hawkins 2017; Diehl & Cook 2015). It has **two** parts, and the project has **only ever built the first**:
(1) lateral inhibition so one unit wins; (2) on the winner, **potentiate active inputs AND depress inactive inputs**
(+ homeostatic boosting so no unit monopolizes). The second part is a **committed kernel in this repo**
(`sim/kernels.py:497`, `fused_htm_winner_inactive_depression`) whose own docstring records the ablation:
**held-out 0.20 → 0.96** on 6 overlapping categories at EMERGE-39 — the closest prior on exactly this class of problem
(make an input→column mapping selective when the categories overlap).

---

## (MOVE c) RANKED CHEAP-FIRST MECHANISMS

All ranks: **no `sim/` edit required**. All GO gates below inherit the mandatory controls in §c.6.

### RANK 0 — MEASURE THE SELECTION (free; do this before anything else)
**What.** Report, from data the probe already collects: (a) per-slot **somatic** firing during each fact's write window
(`pool_fire[i][slot[j]].mean()`, `_consol_cortical_store_probe.py:115/137`); (b) `comp_attr_inh` firing during the
write; (c) per-slot + `comp_attr_inh` firing during the recall read (from `acc`, `:202-206`).
**Minimal change.** ~4 lines in the probe's return dict. Zero extra compute, one seed.
**Reused machinery.** Entirely existing arrays.
**Decides.** If non-target slots fire ≈ target slot ⇒ **selection never happened**; the spike-gated rules were
structurally unable to be selective and Rank 1 is the fix. If the target slot already dominates somatically ⇒ the
record's "broad rule wins" stands as stated and Rank 2/3 (plasticity-side) become primary. **These are the only two
live diagnoses and this read separates them for free.**
**Also free, same run:** print the executed Hebbian branch and `cfg.enable_stdp` / `cfg.enable_synaptic_scaling` /
`_hw_max`, so §a.4's confounds cannot recur silently.

### RANK 1 — SELECTIVE CROSS-INHIBITION between slots (replace global with per-slot FS + cross-suppression)
**Biology.** b.1 (allocation is a competition) + b.2 (SOM/PV lateral inhibition enforces it) + Kandel 6e Ch 48 pp.1198-1205.
**What.** In `build_substrate`, behind an additive default-`False` flag (e.g. `comp_wta_selective`), replace the single
`comp_attr_inh` with per-slot `comp_attr_FS_{s}` (`exc_fraction=0.0`) and wire
`comp_attr_{s} → comp_attr_FS_{s}` (exc) and `comp_attr_FS_{s} → comp_attr_{t≠s}` (inh) — **no self-inhibition**.
**Minimal change.** ~15 lines in `nmda_compositional_consolidation.py:278-289`, default-off ⇒ byte-identical.
**Reused machinery — the operating point is already validated in this repo**
(`research/runners/biased_competition_buffer.py:114-115, 164-176`, multi-seed GO): `sel_to_fs_weight=20.0`,
`fs_to_sel_weight=5.0` (*"gentle — symmetric over-inhibition is unstable"*, `:169`), `sel_recurrent_weight=0.35` with
**α<1 so the pool never self-ignites**, `sel_recurrent_density=0.5`. Also reusable as the pattern: `ca1_ffi_kwta`
(`nmda_compositional_consolidation.py:209-217`) shows the append-an-FS-region idiom in this exact builder.
**Paired sweep (cheap, same runner):** `comp_self_weight` 12.0 → the α<1 band, and `read_steps` 60 → 200-400
(30 ms is far too short for an attractor to resolve a few-% bias; Wong-Wang decisions take hundreds of ms).
**Why it should fix BOTH rules at once.** The executed Hebbian (`:7710-7717`) and STDP are **post-spike gated**. Make
somatic ignition one-of-N and they become selective *without touching the plasticity rules at all* — which dissolves
the "broad rule vs selective rule" tension rather than trying to out-tune it.
**Risk (honest).** This is a known knife-edge: EMERGE-11 over-suppressed the whole column at
`col_fs_weight=60 / fs_col_weight=400`; EMERGE-41's working point was `40 / 90`; `biased_competition_buffer`'s is
`20 / 5`. Expect an operating-point sweep, and require the Rank-0 per-slot firing read as the *in-band* check
(target fires, non-targets ~0, **and the target is not itself silenced**).
**Prior negatives this must be tested against (do not re-derive):** a **single shared/global** FS pool is
selection-inert (EMERGE-41: FS-on vs FS-lesion winner overlap **1.00**) and non-selective
(`2026-07-09-riii-ca3-feedback-inhibition-sparsifies-but-nonselective.md`: sparsity 0.43→0.21, ratio unchanged 1.16×,
byte-identical at inhibition 120 vs 250). **The claim here is specifically that cross-inhibition ≠ global inhibition**,
and the run must include a **global-arm control** (the current shipped `comp_attr_inh`) to show the topology, not the
weight, is what moved it.

### RANK 2 — SUBTRACTIVE (Miller-MacKay) normalization of the write, with the confounds removed
**Biology/theory.** b.4, read in full: multiplicative constraints yield graded fields and *prevent* segregation of
weakly-correlated populations; subtractive constraints sharpen to a subset. Zero-sum by construction ⇒ **it bounds the
pathway without a ceiling**, dissolving the recorded tension (bound→saturate vs unbound→runaway).
**What.** `cfg.btsp_mean_subtract > 0` (`sim/config.py:396`; implemented `sim/bridge.py:8153-8194`, active-set-projected
and Dale-compliant), with the arm cleaned up:
- `cfg.btsp_hetero_dep > 0` — **required for the mechanism to have anything to subtract.** Verified at
  `sim/bridge.py:8125-8129`: with `_hdep <= 0` the active set is `etilde>1e-6 AND is>1e-6`, i.e. **only the target
  fact's own pool synapses**, so the per-postsynaptic mean ≈ the increment and the update nets to ~0. With `_hdep > 0`
  the set widens to *all* afferents whose post plateaus, so silent pools can be depressed. (The branch **order** at
  `:8153` means `mean_subtract` still selects the subtractive update, **not** the separately-refuted
  `fused_btsp_hetero_update`.)
- `cfg.enable_stdp = False` (remove the third, unaccounted rule — §a.4(ii)).
- `cfg.hebbian_learning_rate = 0.0` with `enable_hebbian_learning=True` and a wide `hebbian_max_weight` — removes broad
  Hebbian potentiation **while avoiding the hard-coded `5.0`** of `sim/bridge.py:8704` (§a.4(i)).
- re-test `enable_synaptic_scaling` cleanly under the same fix (it has never actually been tested).
**Minimal change.** **Zero** runner or `sim/` edit — the probe already sets `b.core_config.*` after `build_substrate`
(`_consol_cortical_store_probe.py:68-84`); add the same pattern for these four fields.
**Honest caveats.** (1) direction mismatch (RF vs PF) — works here only because the teaching clamp makes the mapping a
permutation (§b.4); (2) `btsp_hetero_dep` on its own was **REFUTED** for a *different* purpose in
`2026-07-18-gap4-gap5-unification-competition-arm-REFUTED-...md` (it eroded within-assembly weight 72→28) — here it is
used only to widen the active set, not as the update rule, and the run must report within-slot weight mass to check the
same erosion does not occur.

### RANK 3 — COMPETITIVE LEARNING PROPER: winner-inactive depression on `pool→slot`
**Biology.** b.5 (competitive learning), b.3/L.02 (activity competition eliminates the poorly-correlated inputs).
**What.** After each write window, call the committed kernel `fused_htm_winner_inactive_depression(w, pre_active,
post_win, lam_dep_wi, w_min, w_max)` (`sim/kernels.py:497`) on the `concept_to_comp_attr` synapses, with
`post_win = 1` for the taught slot and `pre_active` from that window's pool firing.
**Reused machinery.** The kernel is committed and the host-drive idiom is established: `apply_kernel_update`
(`_emerge14_stageC_onbridge_learning_derisk.py:100`), `_winner_inactive_depress`
(`_emerge39_onsubstrate_competitive_pooler_derisk.py:155-164`), and the on-substrate variant that keeps permanences
**in `bridge.cp_connections.data`** (EMERGE-39/40). Starting constants from those GOs: `POOL_LP=0.05`, `POOL_LD=0.02`.
**Evidence strength.** The single strongest prior on this exact class: EMERGE-39 on-substrate held-out **0.96** with the
selectivity term vs **0.20** without (**+0.76**), permuted 0.15, lesion 0.00 — 6/6 seeds.
**Caveat.** `2026-07-02-emerge48-soft-l2-pooling-BOUNDARY.md` warns a **high** `lam_dep_wi` over-selectivizes and kills
held-out generalization; sweep it low-first.
**Cost.** Larger than Ranks 1-2 (a per-window host call in the probe, ~25 lines), but no `sim/` edit.

### RANK 4 — SPARSE / COMPETITIVELY-ELIMINATED `pool→slot` connectivity (the structural option the record proposed)
**Biology.** A.12 Kincaid sparse decorrelated convergence (`feature-catalog.md:224-233`); L.02 synapse elimination by
activity competition (`:4193-4200`).
**What.** (a) cheap static version: drop `pool→slot` density 0.15 → ~0.03-0.05 and/or restrict each pool to a subset of
slots; (b) the developmental version: `enable_structural_pruning` (`sim/config.py:841`) +
`bridge.update_pruning(eligibility_trace, reward_signal, prunable_indices=...)` (`sim/bridge.py:4008`) — the
`prunable_indices` argument exists precisely to **restrict pruning to one projection**, and
`struct_plast_activity_bias` (`sim/config.py:834`) gives activity-biased formation.
**Why RANK 4, not RANK 1.** Static sparsening is a *lottery*: with 3 slots it changes *which* slots a pool happens to
reach without making the mapping *learned*, and it risks disconnecting the correct slot. The developmental version is
the biologically right answer but is the most machinery, the slowest to converge, and the hardest to control. Both are
strictly better *after* Rank 0 shows whether feedforward symmetry is even the binding constraint.

### RANK 5 — soft competition primitives (cheap adjuncts, not standalone fixes)
`enable_input_divisive_norm` per-region (`sim/config.py:783`; `BrainRegion.input_divisive_norm`, `sim/regions.py:240`)
— Carandini-Heeger normalization by total pool drive on the slot region; and `enable_graded_lateral`
(`sim/config.py:729`; `BrainRegion.graded_lateral`, `sim/regions.py:217`) — learned anti-Hebbian **graded, pre-spike**
pairwise lateral inhibition, the closest built-in to a learned decorrelating lateral stage. Both are one-flag,
default-off. They shape competition but do not by themselves produce one-of-N ignition.

### c.6 — MANDATORY anti-cheat controls for every rank (per `.claude/skills/verify-go/SKILL.md`)

| control | why (which failure it catches) |
|---|---|
| **Permuted-target control on the SAME read that is claimed** | lens 7(a). The record's own history: the firing-weighted 3/3 was only credible once its *own* permuted control was run (true 1.042 vs permuted 0.988). A ratio without its matched control is not evidence. |
| **Raw per-target magnitudes** (per-slot weight mass, per-slot spike counts, both arms) | lens 7(b) + lens 3. Catches the winner-slot mass artifact (`[24,80,24]` → own/other 3.67 unearned) **and** catches two arms both pinned at floor/ceiling (a null between saturated arms is VOID). |
| **Per-fact passes, never a mean** (`own_is_max` as a 3-vector; N=3 is a small-number regime) | lens 7(c). A mean printed "GO 3/3 seeds" when 1 of 3 facts passed. |
| **Substrate validity DURING the write**: `v_apical ∈ [−90, +50]` asserted over every write step, arm marked VOID otherwise | the 333× lesson + the probe's own instrument gap. The probe's `write_phase_physiological` (`:220-221`) already does this — **keep it gated and add somatic `v` range too**, since Rank 1 changes inhibition and Rank 2 removes bounds. |
| **Mechanism-lesion arm** (Rank 1: revert to the shipped shared `comp_attr_inh`; Rank 2: `btsp_mean_subtract=0`; Rank 3: `lam_dep_wi=0`) | lens 2/3. Must collapse. For Rank 1 the lesion is the **global-inhibition** arm specifically, so the claim tested is *topology*, not weight. |
| **Selection read reported in every arm** (Rank 0's per-slot write-window + recall firing) | the in-band check: distinguishes "selective" from "everything silenced" (the EMERGE-11 over-suppression failure) and from "one slot always wins" (the seed-dependent intrinsic-attractor failure). |
| **Seed verification**: `cfg.seed` set (it is — `:337`) and `thr_hash` of `cp_neuron_firing_thresholds` printed (it is — `:85`) | lens 5. |
| **Rule inventory printed**: `enable_stdp`, `enable_hebbian_learning`, `hebbian_learning_rate`, `enable_synaptic_scaling`, the effective `_hw_max` | §a.4. The two confounds above were invisible because the arms never printed which rules were live. |
| **Control-outperforms guard** | if any control matches/beats the treatment, the arm is void, not "noisy". |
| **6 seeds (42/43/44/100/101/102) only after the magnitude is real** | lens 1. The record's own standard: *"a 5% effect is not worth 6 seeds yet."* |

**Proposed GO gate (cortical store, hippo-lesioned):** firing-weighted `own/other ≥ 2.5` with `own_is_max` **3/3 per
fact**, permuted-target control ≤ ~1.2, per-slot mass balanced, hippo-lesioned recall **3/3** with separating slot rates
(winner ≥ 3× runner-up, and runner-up > 0 so the win is not "the others are exactly zero" — the trap that produced the
earlier fake "2/3" at a rate of 0.008), substrate physiological throughout the write, mechanism-lesion collapses,
6 seeds.

---

## (MOVE d) RECOMMENDED FIRST BUILD

**Do Rank 0 and Rank 1 in one cycle; hold Ranks 2-3 as the pre-built next lane.**

1. **Rank 0 (minutes, 1 seed, free).** Add the per-slot write-window firing read + the recall-phase per-slot/inh firing
   read + the rule-inventory print to `_consol_cortical_store_probe.py`. Run the *existing* `--teaching-clamp` config
   unchanged. **This is a pure instrumentation change to a configuration whose numbers are already recorded**, so it is
   also a reproduction check.
2. **Branch on the result.**
   - non-target slots fire ⇒ **Rank 1** (selective cross-inhibition + the α<1 recurrent band + longer `read_steps`),
     with the shipped shared-pool arm as the lesion control.
   - target slot already dominates somatically ⇒ **Rank 2** first (subtractive normalization with STDP off,
     `hebbian_lr=0`, wide `hebbian_max_weight`), since the residual really would be plasticity-side.
3. **Either way, run the two confound-repair arms** (§a.4) as cheap singles alongside: clean synaptic scaling, and
   `enable_stdp=False`. They cost one run each and they repair two entries in the hypothesis ledger.
4. **Build Rank 2's config-only arm now** (it is four `b.core_config` assignments) so it is queued and ready the moment
   a lane frees — per the standing build-de-risks-ahead practice.

---

## (MOVE e) VERDICT

**Selective slot addressability is REACHABLE on this substrate without an architectural redesign — and the
"~5% is a CHARACTERIZED CEILING / next is structural" verdict should be treated as OPEN, not as a boundary.** Three
independent reasons, each traceable to a file:line rather than to an interpretation:

1. **The claimed mechanism is not the executed one.** The rule cited (`bridge.py:838`, coactivity-driven) is behind a
   default-`False` flag; the rule that ran (`:7710-7717`) is **post-spike gated**, so its broadness is a *symptom of a
   selection failure*, not an independent competing force. The decisive measurement — do non-target slots spike? — is
   **unmade**, and is **free**.
2. **Two of the arms that eliminated the alternatives are confounded.** The synaptic-scaling refutation measured a
   hard-coded literal `5.0` (`bridge.py:8704`), not scaling; and a third broad plastic rule (STDP,
   `config.py:598` default `True`) was writing the pathway in every arm, including those labelled "BTSP alone".
   The "unifying explanation" (BTSP saturates whatever ceiling exists; selectivity needs headroom) rests on the first
   of these.
3. **The mechanisms biology actually uses for this are BUILT and UNTRIED here.** Cross-inhibitory selective WTA at a
   multi-seed-GO operating point (`biased_competition_buffer.py:164-176`); Miller-MacKay **subtractive** normalization
   (`config.py:396` / `bridge.py:8153-8194`) — whose 1994 result *predicts the graded ~5% outcome* for every
   multiplicative bound the arc tried, and names the fix; and the competitive-learning winner-inactive depression
   kernel (`kernels.py:497`) with a **0.20 → 0.96** on-substrate ablation on the same class of problem.

**What is genuinely hard, stated precisely (not a wall, a tuning risk):** producing one-of-N ignition among 3×120-neuron
recurrent assemblies on point neurons at `dt=0.5 ms` is an operating-point knife-edge, and this project has hit both
failure ends — global inhibition that is selection-inert (EMERGE-41, riii CA3) and inhibition strong enough to silence
the whole column (EMERGE-11). Mitigations are in-repo: the `20 / 5` gentle-cross-inhibition point with an explicit α<1
recurrent constraint, plus the in-band per-slot firing read so an over-suppressed arm is caught immediately instead of
being read as "no signal".

**What would genuinely require a redesign, and is NOT yet demonstrated:** if Rank 0 shows the target slot already wins
somatically **and** Ranks 2-3 leave `own/other` at ~1.05, then the all-to-all `pool→slot` broadcast is the binding
constraint and Rank 4's competitive-elimination build (L.02 / A.12) becomes the real work. Nothing measured so far
licenses that conclusion — the record's own refutation of the broadcast hypothesis (`v_apical` exclusive, −9 vs −66 mV)
argues against it, and the connectivity change was proposed only *after* the plasticity levers were exhausted, without
a measurement pointing at connectivity.

**Scope honesty.** Nothing in this document was run. Every substrate claim is cited to file:line and was read directly;
every project result quoted is the project's own recorded multi-seed number; the two confounds in §a.4 are read from the
engine source and are **predictions about what a re-run will show**, not measurements. The `~5%` selectivity is real
(own-is-max 3/3, permuted control collapses) and none of this weakens that; what is disputed is only the claim that it
is a **ceiling**.

---

## Files & citations

**Substrate (read at file:line).**
`research/runners/nmda_compositional_consolidation.py`: slot regions `:257-263`; slot self-loop `:265-269`;
`ca1→slot` `:271-276`; **shared `comp_attr_inh`** `:278-289` (design intent `:250-252`); **`pool→slot` broadcast +
its "write-selectivity killer" comment** `:291-302`; `ca1_ffi_kwta` FS-append idiom `:209-217`; apical/BTSP config
`:349-370`; `no_stdp` `:340-341`; `cfg.seed` `:337`.
`research/runners/_consol_cortical_store_probe.py`: `BASE` overrides `:41-61`; post-build `core_config` overrides
`:68-84`; `thr_hash` `:85`; **`pool_fire` full-network array** `:115`; teaching clamp `:127-141`; firing-weighted read
+ permuted control `:172-191`; recall `:193-210`; write-phase validity gate `:216-221`.
`research/runners/_consol_direct_weight_probe.py`: `BASE` `:31-34`; `btsp_hetero_dep` usage `:73-75`.
`research/runners/biased_competition_buffer.py`: ctor op-point `:114-115`; `sel_X→sel_FS_X` `:164-167`;
**`sel_FS_X→sel_Y≠X`** `:169-176`.
`sim/bridge.py`: branchless (**not executed**) Hebbian `:834-849`; **executed Hebbian** `:7710-7717`; Hebbian decay +
gated clip `:7735-7760`; BTSP block `:8060-8200`, **active-set widening by `btsp_hetero_dep`** `:8125-8129`,
**Miller-MacKay subtractive update** `:8153-8194`; synaptic scaling `:8671-8710`, **hard-coded `5.0` fallback**
`:8703-8704`; `update_pruning(..., prunable_indices=)` `:4008`.
`sim/config.py`: `enable_branchless_plasticity:196`; `btsp_hetero_dep:351`, `btsp_hetero_theta:355`,
`btsp_elig_hard_thresh:359`, **`btsp_mean_subtract:396`**; `hebbian_learning_rate:532`, `hebbian_weight_decay:533`,
`hebbian_max_weight:535`, `hebbian_symmetric:544`, `hebbian_rate_window:554`; `enable_homeostasis:567`,
`enable_synaptic_scaling:575`; **`enable_stdp:598`**, `stdp_a_plus:599`, `stdp_w_max:604`;
`enable_graded_lateral:729`; `enable_input_divisive_norm:783`; `enable_structural_plasticity:826`,
`struct_plast_activity_bias:834`, `enable_structural_pruning:841`.
`sim/kernels.py`: `fused_htm_permanence_update:472`; **`fused_htm_winner_inactive_depression:497`**.
`sim/regions.py`: `BrainRegion.graded_lateral:217`, `BrainRegion.input_divisive_norm:240`.

**Project findings (read).**
`2026-07-25-CRITICAL-apical-R-333x-miscalibration-...md` (the source problem; cortical-store sections at the tail).
`2026-07-25-ca1-sparsification-research-gate-scope.md` (predecessor gate; its scope note names this problem).
`2026-07-09-riii-ca3-feedback-inhibition-sparsifies-but-nonselective.md` (global feedback inhibition: sparsifies
0.43→0.21, ratio unchanged 1.16×, saturating).
`2026-07-02-emerge41-fs-wta-kwinners-GO.md` (a single global FS pool is **causally inert** for selection — FS-on vs
FS-lesion winner overlap 1.00; it only sparsifies the loser pool 0.57→0.28).
`2026-07-02-emerge11-stageB1-reframing-dap-subsumes-selection-wta-is-burst-sparsification.md` (FS-WTA over-suppression
at `60/400`).
`2026-06-19-multireferent-biased-competition-derisk.md` + `2026-07-21-gap3-biased-competition-multireferent-6seed-GO.md`
(the cross-inhibition motif's GO + operating point).
`2026-07-21-gap5-2assembly-selective-inhibition-family-NEGATIVE-...md` (somatic selective inhibition insufficient;
apical-targeting works — the compartment distinction of §b.2).
`2026-07-18-gap4-gap5-unification-competition-arm-REFUTED-...md` (`btsp_hetero_dep` as an update rule eroded
within-assembly weight 72→28 — the caveat on Rank 2).
`2026-07-02-emerge39/40-...-GO.md` + `2026-07-02-emerge48-soft-l2-pooling-BOUNDARY.md` (winner-inactive depression:
0.20→0.96 on-substrate 6/6; and the over-selectivity caveat).

**Biology / external.**
Miller K.D. & MacKay D.J.C. (1994), *Neural Computation* 6(1):100-126, "The Role of Constraints in Hebbian Learning" —
**read in full text** (multiplicative→graded RF / prevents segregation; subtractive→weights at max or min, RF sharpened
to a subset; §3 p.115-116 output-cell constraints shape RFs, input-cell constraints shape PFs).
[MIT Press](https://direct.mit.edu/neco/article/6/1/100/5780/The-Role-of-Constraints-in-Hebbian-Learning)
Han J.-H. et al. / Josselyn & Silva — CREB regulates excitability and the **allocation** of memory to subsets of
neurons; neurons with relatively higher excitability *win the competition* to become engram neurons (abstract/summary
read, full texts not fetched).
[Nat Neurosci 12(11)](http://www.nature.com/neuro/journal/v12/n11/full/nn.2405.html) ·
[J Neurosci 44(21) e0846232024](https://www.jneurosci.org/content/44/21/e0846232024)
Stefanelli T. et al. (2016) *Neuron* 89:1074 — **SOM+ dendrite-targeting** interneurons laterally inhibit neighbouring
granule cells and set engram **size** (abstract/summary read, full text not fetched).
[PubMed 26875623](https://pubmed.ncbi.nlm.nih.gov/26875623/) ·
[Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(16)00049-0)
Catalog (read directly, `~/Projects/sim-catalog/references/feature-catalog.md`): **A.12** Kincaid/Bolam sparse
decorrelated cortico-striatal convergence, "decorrelation rule", sim status **missing** (`:224-233`); **L.02** synapse
elimination by activity competition, Kandel 6e Ch 48 pp.1198-1205 (`:4193-4200`); **D.14** Tonegawa engram cells,
Kandel 6e Ch 54 pp.1357-1359 (`:1248-1260`); **E.05** lateral inhibition / center-surround (`:1391-1400`).

## Provenance
Read-only research gate, 2026-07-26. Local corpus first (RAG over findings/catalog/Kandel, then direct source reads),
then substrate reads at file:line, then 3 external searches + 1 full-text paper fetch. One read-only subagent surveyed
the repo's competitive/WTA prior work; its load-bearing claims (`kernels.py:497`, `config.py:729/783/841`,
`bridge.py:4008`, `biased_competition_buffer.py:169-176`) were independently re-read by the controller before being
cited. **NO `sim/` edit, no build, no run, no bar moved.**
