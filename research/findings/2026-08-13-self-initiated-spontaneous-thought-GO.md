---
type: finding
status: live
date: 2026-08-13
mechanism: self-initiated-spontaneous-thought
runner: research/runners/_self_initiated_spontaneous_thought_derisk.py
artifacts:
  - research/findings/raw/_self_initiated_spontaneous_thought_derisk.json
---

# Self-initiated / spontaneous thought (a DMN correlate) is 6-seed GO: a stored bistable CA3 store reactivates coherent content with NO prompt, and a curiosity neuromodulatory gain steers WHICH thought surfaces

**2026-08-13 (autonomous, GPU/cupy, n_ca3=2000).** A genuinely-conversing brain does not only REACT to prompts —
it has internally-generated thought (mind-wandering / default-mode / replay-driven ideation) that seeds
curiosity-driven questions and self-initiated conversation. Today the sim is prompt-driven. This de-risks the FIRST
SPIKING self-initiation correlate: an internally-generated (noise-seeded) attractor reactivation that, with NO
external cue, lands on real stored content and is STEERED by the brain's own curiosity signal so the interesting
thought preferentially surfaces. **Functional correlate only — no claim of phenomenal experience.**

## Why this ran (RAG-grounded, not reinvented)

`rag_search "default mode network spontaneous thought internally generated replay curiosity ..."` surfaced our own
read-only cluster review [`2026-06-27-conv-thinking-research-reasoning-thinking.md`](2026-06-27-conv-thinking-research-reasoning-thinking.md)
§2.6 (the self-generated "train of thought"): the DMN *generates* the sequence, the frontoparietal control /
salience system STEERS it (Christoff et al. 2016 *Nat Rev Neurosci* "Mind-wandering as spontaneous thought"; Buckner
2008 "internal train of thought"). That review flagged this faculty as `missing` and its **wall #5** — "default-mode
mind-wandering has no clean behavioral validation gate … treat any DMN claim with maximum skepticism" — as the
hardest to anti-cheat. This runner answers that skepticism with four verified anti-cheats. NO `sim/` edit;
reuse-by-import.

## The mechanism (two validated organs composed + one neuromodulatory projection)

_Numbers in this section are config values + readouts from the diagnostic tuning sweeps, not the 6-seed artifact._

<!--derived-->

1. **Internally-generated + coherent** = the gap#5 **RANK-1 spontaneous-reactivation** 6-seed GO substrate
   ([`2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md`](2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md)),
   reused-by-import (`_gap5_spontaneous_reactivation_derisk._prepare/_hard_silence/_detect_events`). A CLOSED
   bistable CA3 store (dendritic-plateau two-compartment neurons + committed BTSP one-shot encode) holds the stored
   concepts as attractor basins. Under weak NON-SPECIFIC Poisson background (rate 0.015, 1500 pA, dur 10) — no cue,
   no recall_drive, **0 external CONTENT drive** — a stored assembly spontaneously, basin-selectively reactivates as
   a discrete event; the net rests silent between events. That reactivated assembly IS a coherent internally-driven
   "thought" (it lands on real stored content, not noise).
2. **Curiosity steering** = the production **CURIOSITY organ** (`curiosity_production_organ.CuriosityProductionOrgan`,
   reused-by-import), whose spiking ASK-pool WANT is read off `cp_firing_states` and tracks a concept's
   epistemic-gap/interest (**novel → 117.5 Hz, familiar → 8.3 Hz**). Each concept's want tags its CA3 engram with a
   proportional **neuromodulatory recurrent gain** (a transient multiplicative scaling of that assembly's
   within-assembly recurrent synapses; plasticity byte-frozen during rest) — the biology of DA/ACh tagging salient
   memories for preferential offline reactivation (McNamara et al. 2014 *Nat Neurosci* "Dopaminergic neurons promote
   hippocampal reactivation"; Ambrose/Pfeiffer/Foster 2016 *Neuron* "Reward enhances reverse replay"; Mattar & Daw
   2018 need × gain). A stronger recurrent basin completes from a smaller coincidental noise volley → the tagged
   thought reactivates more. The gain amplifies RECURRENCE only, so it is verified subthreshold (see anti-cheat a).

**Method banked per THE LAW.** The first steering method — a tonic per-cell excitability BIAS — was MEASURED too
weak/non-monotone on this substrate (fresh-bridge events 5/2/4 across bias 0/60/120 pA; only self-igniting at 200
pA). Banked; the recurrent-gain method is the surpass (member 0.29→0.35→0.38→0.41 across gain ×1/1.5/2/3, all
gain-only-silent).

## Result — 6/6 GO (seeds 42, 43, 44, 100, 101, 102), gain-scale 2 (novel ×3.0, familiar ×1.1)

_Per-seed values are rounded from, and the means/ratios aggregated over, the cited committed artifact_
_`research/findings/raw/_self_initiated_spontaneous_thought_derisk.json` — verify against the raw JSON._

<!--derived-->

Identity-controlled: the SAME stored thought is tested NOVEL-tagged vs FAMILIAR-tagged vs NO-tag, so intrinsic basin
strength is cancelled (each thought is its own control). Reactivation MASS = mean assembly-active fraction over the
whole rest, net of a random non-member floor.

| seed | mass novel | mass familiar | mass baseline | novel/fam | coherence memb vs rand | dwell nov/fam/base | NO-NOISE | GAIN-ONLY | STORE-LESION memb |
|------|-----------|---------------|---------------|-----------|------------------------|--------------------|----------|-----------|-------------------|
| 42   | 0.262 | 0.111 | 0.090 | 2.37 | 0.41 vs 0.04 | 492 / 143 / 105 | 0 ✓ | 0 ✓ | 0.00 ✓ |
| 43   | 0.249 | 0.113 | 0.090 | 2.20 | 0.39 vs 0.04 | 340 / 134 / 137 | 0 ✓ | 0 ✓ | 0.00 ✓ |
| 44   | 0.249 | 0.109 | 0.082 | 2.30 | 0.39 vs 0.04 | 272 / 155 / 124 | 0 ✓ | 0 ✓ | 0.00 ✓ |
| 100  | 0.243 | 0.095 | 0.071 | 2.57 | 0.38 vs 0.05 | 239 / 129 / 52  | 0 ✓ | 0 ✓ | 0.00 ✓ |
| 101  | 0.214 | 0.082 | 0.072 | 2.60 | 0.36 vs 0.04 | 139 / 89 / 54   | 0 ✓ | 0 ✓ | 0.00 ✓ |
| 102  | 0.221 | 0.100 | 0.090 | 2.21 | 0.37 vs 0.04 | 174 / 125 / 107 | 0 ✓ | 0 ✓ | 0.00 ✓ |

(Values are GPU/cupy; floating-point summation order is not byte-deterministic, so per-seed masses jitter ~3%
run-to-run — the GO and every anti-cheat hold with margin regardless. `attributable_to` = **65.6%** of the
novel-tag surfacing mass is owned by the curiosity gain, 34.4% is the intrinsic baseline present in both arms.)

Every seed → GO on all four anti-cheats (each VERIFIED, not asserted):

- **(a) INTERNALLY-GENERATED.** 0 external CONTENT drive (only non-specific Poisson to random CA3-exc cells).
  **NO-NOISE (acid) → 0 surfacing** every seed; **GAIN-ONLY (curiosity gain on, noise off) → 0 surfacing** every seed
  (the gain amplifies recurrence, it cannot manufacture a thought — the ignition is genuinely noise-seeded, only
  steered). Plasticity byte-frozen during rest; dendritic state reset verified.
- **(b) COHERENT.** The surfaced reactivation overlaps a STORED assembly at **member 0.36–0.41 vs random 0.04–0.05
  (~8× above chance)**. **STORE-LESION (NO-ENCODE, same noise+gain) → member 0.00** every seed — the content is the
  learned store, not the noise.
- **(c) CURIOSITY-STEERED (identity-controlled).** The SAME thought reactivates **~2.4× more mass** (steer ratio
  2.20–2.60) when tagged NOVEL (curiosity gain ×3.0) than when tagged FAMILIAR (×1.1) — mean mass novel 0.240 vs
  familiar 0.102 — and dwell roughly doubles/triples. The tag is set by the curiosity organ's SPIKING want, so
  surfacing tracks the curiosity VALUE, not the content identity (the familiar tag is the mismatched-value control
  on the same content).
- **(d) STEERING LESION-LOAD-BEARING.** Removing the curiosity tag (no-gain baseline) drops surfacing mass to 0.082
  (mean) — below the novel-tag 0.240 — so the curiosity gain is what boosts WHICH internally-driven thought surfaces
  (attribution: 66% of the surfacing mass is owned by the gain, not the intrinsic baseline).

## What is SUBSTRATE vs HOST (the honesty boundary is a deliverable)

- **SPIKING (load-bearing):** the reactivation itself (CA3 dendritic-plateau attractor completion), the silence
  between events, AND the steering VALUE (the ASK-pool want is read off `cp_firing_states` in the curiosity organ).
- **HOST (declared, rides existing burn-downs):** (i) the per-concept NOVELTY levels are the ENVIRONMENT (concepts
  genuinely differ in how novel they are — the same class of host boundary the curiosity organ declares for its
  novelty derivation and the surprise organ for its sensory encoding); (ii) the PROJECTION of the spiking want onto
  the CA3 engram as a recurrent-gain factor is a host-parameterised neuromodulatory projection scaling.

## Honest scoping (what this does NOT yet show)

<!--derived-->

- The n_mem=2 store reliably reactivates ONE dominant basin per seed; the second assembly is weakly ignitable
  (member ≈ chance, dwell 0). So this de-risks **steering of the reactivating thought's surfacing** (identity-
  controlled novel-vs-familiar on the same thought), which is the CLEAN way to show curiosity steers surfacing
  without the intrinsic-basin-dominance confound. It does NOT yet show **selection AMONG several equally-reactivatable
  concepts** — that needs a store with multiple balanced basins (the RANK-1 substrate this builds on only validated
  single/dominant reactivation).
- The steering VALUE here is curiosity/novelty. AFFECT (valence/arousal, via the existing affect organ) is a parallel
  salience source that the same recurrent-gain mechanism can carry — a named extension, not claimed here.

## Named next rungs (no defer — the capability continues)

1. **Multiple balanced basins** so the wander SELECTS among concepts (curiosity biasing which of several
   equally-storable thoughts wins the noise-seeded race) — pattern-separated encoding to equalise basin strength.
2. **One-brain merge:** release the `curiosity` neuromodulator directly onto the CA3 store on ONE bridge, so the gain
   is set BY the spiking modulator instead of a host scalar — the co-resident-merge rung the affect / surprise /
   episodic organs each carry.
3. **Seed → utterance:** route the surfaced thought vector into the composer/mouth so a spontaneous thought becomes a
   self-initiated question or remark (closes the loop to the DMN's conversational role).
4. **Affect steering** in parallel with curiosity (high-valence/high-arousal content preferentially surfaces).

Runner: `research/runners/_self_initiated_spontaneous_thought_derisk.py`. NO `sim/` edit; reuse-by-import of the
gap#5 RANK-1 reactivation substrate + the production curiosity organ.
