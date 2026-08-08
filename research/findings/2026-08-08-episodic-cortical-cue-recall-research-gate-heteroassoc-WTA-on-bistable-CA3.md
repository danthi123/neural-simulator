---
type: research-gate
status: active
date: 2026-08-08
mechanism: episodic-cortical-cue-recall
lane: EPISODIC
---

# Research gate — a genuinely-NEW mechanism for NEURAL cortical episodic cue-recall: a learned CA3→cortex heteroassociative projection read out by a neural WTA cortical microcircuit, riding on the already-CLOSED bistable+specific CA3 attractor, with a recurrent-zero isolation control and a real-vs-sham teeth lesion. NOT a build — a build-ready design + honest feasibility.

**Verdict: DESIGN READY / buildable_now = YES (no `sim/` edit; all pieces committed default-off, reuse-by-import).** This is DIFFERENT IN KIND from the Wave-1 failed method (fixed feedforward who→ca3 + host-argmax over CA3 spike overlap, no recurrent attractor, tautological lesion). The honest risk is the OUTCOME, not the build: the cortical heteroassociative readout may be weak or non-specific (the same magnitude/specificity trilemma the CA3 arc fought, now at the cortical stage) — and an honest negative there is a first-class deliverable that maps exactly what the cortical readout stage needs.

## What the record already establishes (build ON this, do not re-derive)

The `before_you_build.sh` corpus check + `rag_search --corpus all` surface a long, converged CA3 arc. Read in depth:

- **`2026-07-08-riii-DEFINITIVE-...` + `-CORRECTION-...`**: the ca3→ca3 recurrents TRANSMIT (targets reach −48 mV under strong presyn drive) but a partial cue completed NOTHING across weight 5–200 × density 0.3–0.9 × drive 200–3000 pA. Root cause: no SPECIFIC within-ensemble attractor (a held-out member got the same recurrent drive, 2.99, as a random non-member, 2.98). STDP is silent at the Δt≈0 of synchronous co-firing — rate-Hebbian co-activity is required (CYCLE 95/96).
- **`2026-07-14-ca3-competitive-hebbian-formation-6seed-GO`**: competitive-Hebbian forms a strong SPECIFIC learned attractor.
- **`2026-07-18-gap5-CA3-completion-CLOSED-intrinsic-dendritic-bistability-resolves-the-trilemma`** (the load-bearing one): genuine cue-gated **BISTABLE + SPECIFIC** completion CLOSED — a partial cue reactivates the held-out members, a permuted cue does NOT, rest is silent. **5/6 GO, 6/6 mechanism** (perfect specificity + bistability on all seeds), FROZEN-recall + OU-off + no-encoding anti-cheat verified. The keystone was a committed `sim/` change (`fused_coincidence_plateau` v-gated self-regen SUSTAIN + apical KIR down-state, Sanders 2013) plus an asymmetric apical→soma read. **This mechanism is the completion engine the Wave-1 episodic attempt never used.**
- **Two RETRACTIONS on the same page** are the discipline this design inherits: a self-sustaining-attractor artifact (caught by permuted-recall + absolute-rate instrumentation) and a plasticity+noise confound at recall (caught by a plasticity-freeze). ⇒ MANDATORY gates baked in below: FROZEN plasticity at recall, OU-off, no-cue, permuted-cue, no-encoding, and a bistable-gate that RESETS `cp_v_apical`/`cp_conductance_g_coincidence` (the instrument fix that unlocked the close).

**Committed-and-verified substrate this session (assertions, not comments):**
- `sim/config.py` carries the bistability keystone default-off: `coincidence_plateau_self_regen` (L253), `apical_kir_g/E_K/vhalf/k` (L281-284), `apical_g_couple_to_soma` (L275); `sim/bridge.py` L7358/7374 consumes them, guarded byte-identical when 0. ⇒ the completion engine is reachable by CONFIG, no `sim/` edit.
- `hebbian_rate_window` (L574, default False) is the committed rate-window co-activity Hebbian; `bridge.py` L851/7866 apply it to plastic synapses via a per-neuron decaying trace gated by `hebbian_coactivity_thresh`. ⇒ the Δt≈0 heteroassociative encoding rule exists (STDP would be silent here — the documented failure).
- `RegionPathway` (`sim/regions.py:251`) supports `plastic`, `exc_receptor`, inhibitory `receptor`, `transmission_gate`, `plasticity_gate`. ⇒ a plastic cortex↔CA3 heteroassociative pathway and per-pool inhibitory (WTA) pathways are pure config.
- `cfg.seed` seeds the substrate: `bridge.py:2317` `het_seed = cfg.heterogeneity_seed if >=0 else cfg.seed`, L2320 `cp.random.seed(het_seed)`, thresholds at L1508. Build-twice-hash-`cp_neuron_firing_thresholds` is the required determinism check (NOT `actual_seed_used`).

## Why the Wave-1 method failed, restated as four missing pieces

The failed method: `end-to-end neural cortical recall 0/6; only a host-argmax-over-CA3-spike-overlap proxy passed (fragile 3/6); pattern completion not isolated from the fixed feedforward who→ca3 encoder (no recurrent ca3→ca3 zero control); anti-cheat lesion tautological (CA3 hyperpolarization zeroed the very overlap metric).` Decomposed:

1. **No recurrent attractor** — it read CA3 spike overlap driven purely by the fixed feedforward who→ca3 projection; the ca3→ca3 recurrence (the thing that DOES pattern completion) was never load-bearing.
2. **Host-argmax readout** — the "identity" was a Python argmax over CA3 spike counts. Per the BRAIN-BASED-ONLY standard this is a SHORTCUT (an argmax standing in for a neural decision), so even the "3/6 pass" is an honest-negative, not a mechanism.
3. **No isolation** — with no recurrent-zero control, "completion" was indistinguishable from the feedforward encoder simply re-driving the cue.
4. **Tautological lesion** — hyperpolarizing CA3 zeroes the CA3-overlap metric BY CONSTRUCTION; it cannot flip in the failing direction against anything, so it has no teeth.

## The genuinely-NEW mechanism (different in kind, not a parameter tweak)

**Episode = a conjunction of three attribute pools: WHO (identity) · WHAT (content) · WHEN (temporal-context).** Episodic recall = cue ONE attribute, reconstruct the other two. The design routes that through a real recurrent attractor and a real neural decision:

```
 WHO ─┐                                        ┌─► WHO pool  (feedback-inhib WTA)
 WHAT ┼─(plastic, hebbian_rate_window)─► CA3 ──┤   WHAT pool (feedback-inhib WTA)   ← readout = which neurons FIRE
 WHEN ┘   heteroassociative encode      (bistable  └─► WHEN pool (feedback-inhib WTA)    (measurement; NO argmax)
                                        +specific        ▲
                                        completion)      │  ca3→cortex heteroassociative
                                         ca3→ca3 recurrent (the CLOSED attractor)
```

**Encode (plasticity ON):** drive all three attribute pools with an episode's sparse patterns; they project into CA3 (plastic, `hebbian_rate_window` co-activity — potentiates the synchronous cortex↔CA3 co-firing that STDP cannot). CA3 forms the competitive-Hebbian specific attractor (2026-07-14) over the conjunction; the reciprocal CA3→cortex synapses heteroassociate the completed CA3 assembly back onto each bound attribute-neuron.

**Recall (plasticity FROZEN, OU off, dendritic state RESET):** cue ONE pool (e.g. WHEN). Its feedforward projection gives CA3 a PARTIAL cue → the CLOSED bistable+specific attractor completes the full assembly (or, on a permuted cue, does NOT — it rests silent). The completed CA3 assembly drives the CA3→cortex heteroassociative pathway, which reactivates the bound WHO and WHAT neurons. Each cortical pool runs a **feedback-inhibition WTA microcircuit** (an inhibitory `RegionPathway`, the exact `ca3_fb_inhib` pattern used in the CLOSED work, or the committed `_graded_lateral_inhibition_pA` per pool): the strongest-driven attribute-neuron fires and NEURALLY SILENCES its competitors. **The decision is the lateral inhibition; the host only reads which neuron crossed threshold — the same legitimacy as reading which motor pool fired.** No argmax anywhere in the loop.

### The two controls the failed method lacked, with teeth

- **Recurrent-zero ISOLATION control** (isolates completion from feedforward): zero the ca3→ca3 recurrent weights only. Prediction if completion is real: the CUED attribute still reads out (feedforward path intact) but the HELD-OUT WHO/WHAT readout COLLAPSES (no attractor to reconstruct them from a partial cue). If the held-out readout survives recurrent-zero, the "recall" was feedforward re-drive — exactly the Wave-1 confound, now made falsifiable.
- **Real-vs-SHAM teeth lesion** (non-tautological): REAL = ablate the CA3→cortex(WHO) heteroassociative synapses (a targeted pool). Prediction: WHO readout fails, WHAT/WHEN intact. SHAM = ablate an equal-size set of UNRELATED synapses (matched count, different targets). Prediction: all three readouts intact. This lesions the READOUT PATHWAY, not the scoring metric — so it can flip in the failing direction (unlike CA3 hyperpolarization, which zeroed its own metric by construction).

### The full anti-cheat panel (inherited from the two retractions)
no-cue (drive nothing → cortical readout SILENT) · permuted-cue (wrong cue → no held-out readout) · no-encoding (`encode_drive=0` → readout collapses, attractor load-bearing) · plasticity-FROZEN at recall + OU-off (kills the self-sustaining + noise-driven confounds) · bistable-gate that RESETS `cp_v_apical`/`cp_conductance_g_coincidence` before each condition (kills the encoding-leak silence artifact). Bar per attribute: cued→held-out readout ≥ threshold AND ≥3× the permuted/sham/recurrent-zero readout AND no-cue ≈ 0, across 6 seeds.

## Buildable-now assessment (honest)

**buildable_now = YES.** Every piece is committed, default-off, byte-identical-when-off, and reachable by CONFIG + reuse-by-import of the real bridge — no `sim/` edit:
- completion engine: the CLOSED bistable+specific CA3 config (self_regen + KIR + asymmetric read + recall_k_thresh + selective_inhib);
- heteroassociative encoding: `hebbian_rate_window` on plastic cortex↔CA3 `RegionPathway`s;
- neural readout: per-pool inhibitory `RegionPathway` feedback-inhibition WTA (the committed `ca3_fb_inhib` pattern), or `_graded_lateral_inhibition_pA` run per pool.

**Two design notes (do NOT silently paper over):**
- (a) `_graded_lateral_inhibition_pA` is currently flagged for ONE region. Three pools ⇒ use three per-pool inhibitory `RegionPathway`s (fully supported, the CLOSED-work pattern) rather than the single-region graded lateral; only if that under-performs would per-pool graded-lateral need a `sim/` change — a design note to raise THEN, not now.
- (b) The CA3 assembly is still PRE-ASSIGNED (a fixed sparse mask), inherited from the CLOSED completion — the emergent DG/mossy-selected assembly is a SEPARATE downstream piece. This gate tests the READOUT given a working attractor; it does not claim emergent episode selection. State that scope in the finding.

## Honest risk

The build is low-risk; the OUTCOME is genuinely uncertain and that is the point. The CLOSED CA3 completion is only magnitude ~0.2–0.33 and 5/6 — a weak upstream drive into the heteroassociative pathway. The cortical stage re-poses the SAME trilemma (magnitude vs specificity vs a silent rest) one synapse downstream: the CA3→cortex projection may (i) be too weak to cross the WTA threshold (readout silent → NO-GO on magnitude), or (ii) over-drive the whole pool so the WTA picks a winner that is not the bound member (non-specific → fails the permuted/sham teeth). Either is an HONEST NEGATIVE that maps precisely what the cortical readout needs (competitive heteroassociation / a cortical-stage bistability / WTA calibration) — a first-class deliverable, not a wall. What this design DOES guarantee regardless of outcome: a NEURAL decision (no argmax), a recurrent-zero isolation that the Wave-1 method could not perform, and a lesion with real teeth.

## Files / next step (build phase, separate from this gate)
New runner (to build): `research/runners/_episodic_cortical_cue_recall_derisk.py` — instantiate the real bridge; 3 attribute pools + CA3 (CLOSED bistable config); encode N episodes (plastic, rate-window); recall FROZEN with the full anti-cheat panel + recurrent-zero + real/sham lesion; 6 seeds via `cfg.seed`; build-twice threshold-hash determinism check first. Writes `research/findings/raw/_episodic_cortical_cue_recall/`. Derived-number tables must carry a `<!--derived-->` line. Corpus check already logged this session.
