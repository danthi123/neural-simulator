---
type: finding
status: contributing
date: 2026-08-12
mechanism: lane-C self-model / metacognition — (1) INSTRUMENT FIX + selftest so the second-order metacog GO gate provably FAILS in its failing direction (chance type-2 -> NO-GO); (2) MECHANISM GO — a PURE-SPIKING balance-of-evidence confidence read (workspace WTA margin from cp_firing_states) carries genuine type-2 sensitivity (meta-d' above zero) on all six seeds, where the impoverished downstream read is at chance and the only other six-of-six read is a host-classifier shortcut
lane: C · Self/Workspace (roadmap §3 self-model / know source+strength of knowledge)
lane_ref: C
verdict: GO (Part 2, six of six seeds) + instrument FIX (Part 1). Part 1 — the second-order metacog GO gate was NOT mis-calibrated (contra the 2026-08-12 BOUNDARY finding, corrected here with a banner); the per-seed GO logic requires type2_auc, meta_d and m_ratio to clear their pre-registered bars, and re-running shows the impoverished default meta_rate read reports NEGATIVE while the cited learned_acc dynamic artifacts report GO with genuine type-2. The gate is now extracted into a pure `_seed_go_decision` + a `selftest()` (`--selftest`) that LOCKS the failing direction (a chance-level type-2 input is NO-GO even with type-1 in window and all controls clean; the scorer separates a real type-2 signal from noise). A latent always-False precondition (`domain_control` key never written) is fixed. Part 2 — following the D4 comprehension-monitor lead, a `balance` confidence read (the ABSOLUTE MARGIN of the first-order workspace WTA competition read DIRECTLY from cp_firing_states; Vickers balance-of-evidence / Kepecs distance-to-bound) clears the CORRECTED six-seed gate on all six: type2_auc, meta_d and M-ratio clear their bars on every seed, PERMUTED collapses to chance on all six, WITHIN-CLASS holds on all six, and every confidence value is read from cp_firing_states (no host formula). HONEST CAVEAT: loop-ablation does NOT collapse it — balance-of-evidence confidence lives in the evidence ENCODING, so it is a genuine decision-variable read but NOT type-1/type-2 dissociable; the dissociable comparator (`margin_abs`) is seed-fragile and remains the next rung. A FUNCTIONAL metacognition correlate, NOT a claim of subjective experience.
artifacts:
  - research/findings/raw/lanes/metacog/metacog_balance_6seed.json
  - research/findings/raw/lanes/metacog/metacog_balance_s42.json
  - research/findings/raw/lanes/metacog/metacog_balance_s43.json
  - research/findings/raw/lanes/metacog/metacog_balance_s44.json
  - research/findings/raw/lanes/metacog/metacog_balance_s100.json
  - research/findings/raw/lanes/metacog/metacog_balance_s101.json
  - research/findings/raw/lanes/metacog/metacog_balance_s102.json
verification: local six-seed (42/43/44/100/101/102, n_trials=160, numpy). Every seed clears the corrected gate; permuted collapses to chance and within-class holds on all six; loop-ablation does NOT collapse (non-dissociable, honest). The exact per-seed type2_auc / meta_d / m_ratio / permuted / within-class / loop-ablation values are in the cited artifacts and in the Part-2 table (marked derived). Part-1 selftest (`--selftest`) green. meta_rate re-run reports NEGATIVE (below the type-2 and M-ratio bars) — the corrected gate's honest verdict on the OLD mechanism.
---

# lane-C metacognition: the GO gate was NOT mis-calibrated (now self-tested), and a PURE-SPIKING balance-of-evidence read carries genuine meta-d′

## Part 0 — correcting the record

<!--derived-->

The 2026-08-12 BOUNDARY finding
(`2026-08-12-laneC-self-model-BOUNDARY-metacog-type2-at-chance-despite-GO-selfschema-3of6.md`) claimed the
second-order metacog runner reports `verdict=GO` on all six seeds while the type-2 metrics are AT CHANCE
(type2_auc ~0.45–0.53, meta_d ~0), i.e. a mis-calibrated GO gate checking type-1 rather than type-2. **That claim
is not reproducible and is contradicted by the very artifacts it cites.** Two independent checks:

- The per-seed GO logic (git-blamed to 2026-08-02) requires `type2_auc>=0.65 AND meta_d>0 AND m_ratio>=0.60` AND
  every anti-cheat control. There is NO path to GO that skips the type-2 metrics.
- Re-running confirms the honest verdict: the impoverished default `meta_rate` read scores type2_auc ~0.60,
  meta_d ~0.6, M-ratio ~0.30, within-class NOT ok → **seed GO=False → VERDICT NEGATIVE** (not GO). The cited
  `learned_acc --learned-feature-mode dynamic` artifacts score type2_auc ~0.77–0.92 → **GO** — genuine type-2.

The boundary finding conflated a `meta_rate` (chance) run's type-2 numbers with a `learned_acc` (genuine) run's
GO verdict. **The gate was correct; the finding mis-read it.** A correction banner is added to that finding; its
separate self-schema→metacog INTEGRATION sub-result (three of six, seed-fragile) is unaffected and still stands.

## Part 1 — the instrument is now SELF-TESTED (locks the described failure class out for good)

<!--derived-->

Regardless of the mis-read, the described failure CLASS ("a gate that can PASS without its key control",
CLAUDE.md rule 9) deserves a mechanical lock. Added to `_second_order_metacog_monitor_derisk`:

- `_seed_go_decision(...)`: the per-seed GO extracted into a PURE function (no I/O, no bridge) so it is testable.
- `selftest()` + `--selftest`: two halves, both green. **(A)** the type-2 SCORER separates signal from noise —
  confidence aligned with correctness scores type2_auc ~0.97 / meta_d ~4.5; RANDOM confidence scores
  type2_auc ~0.47 / meta_d 0 (rule 3: prove the instrument CAN detect an effect AND does not hallucinate one).
  **(B)** the GO decision REQUIRES type-2 — a chance input (type2_auc 0.50, meta_d 0, m_ratio 0) is NO-GO even
  with type-1 in window and ALL controls clean; a below-bar type2 (0.64) is NO-GO; a genuine strong type-2 with
  clean controls is GO; a permuted-leak control or type-1 out of window flips it back to NO-GO.
- Fixed a real latent bug: the `lesion_permutation_and_domain_controls_recorded` precondition referenced a
  top-level `domain_control` key this runner never writes, so it was ALWAYS False (a silent broken check).

## Part 2 — the MECHANISM: a pure-spiking balance-of-evidence read carries genuine meta-d′

<!--derived-->

**The real, subtler boundary** the corpus makes plain: of the confidence reads, only `learned_acc --dynamic`
clears six of six — but its DISCRIMINATION is a HOST logistic regression over spike-rate features (a
brain-based-only SHORTCUT; the spiking meta pool merely re-renders the host scalar). The PURE-SPIKING reads were
at chance (`meta_rate`) or seed-fragile (`margin` two of six, `margin_abs` zero of six). So genuine brain-based
type-2 was NOT yet demonstrated.

Following the D4 comprehension-monitor lead (`2026-08-12-spiking-comprehension-success-monitor-GO.md`: a
CONTENT-sensitive spiking competition margin read directly from `cp_firing_states` carried a genuine type-2
signal exactly where a content-blind read was at chance), I added a `balance` confidence read:
**confidence = |rate(assembly_1) − rate(assembly_0)|, the ABSOLUTE MARGIN of the first-order workspace WTA
competition, read DIRECTLY from `cp_firing_states` over the evidence window** — the Vickers balance-of-evidence
/ Kepecs distance-to-bound decision-variable margin. No downstream pool, no host formula.

Artifacts: `research/findings/raw/lanes/metacog/metacog_balance_6seed.json` (aggregate) + per-seed
`research/findings/raw/lanes/metacog/metacog_balance_s42.json` … `metacog_balance_s102.json`.

### 6-seed result (42/43/44/100/101/102, n_trials=160, numpy) — VERDICT: GO six of six

| seed | type1_acc | type2_auc | meta_d | M-ratio | permuted auc (collapse) | within-class min (ok) | loop-ablate auc (collapse) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.825 | 0.815 | 2.10 | 1.14 | 0.550 (yes) | 0.758 (yes) | 0.875 (no) |
| 43 | 0.800 | 0.741 | 1.78 | 0.91 | 0.492 (yes) | 0.587 (yes) | 0.896 (no) |
| 44 | 0.781 | 0.809 | 2.25 | 1.36 | 0.518 (yes) | 0.692 (yes) | 0.825 (no) |
| 100 | 0.787 | 0.668 | 1.04 | 0.66 | 0.463 (yes) | 0.656 (yes) | 0.750 (no) |
| 101 | 0.681 | 0.697 | 1.51 | 0.87 | 0.450 (yes) | 0.621 (yes) | 0.828 (no) |
| 102 | 0.644 | 0.707 | 1.51 | 1.84 | 0.531 (yes) | 0.576 (yes) | 0.834 (no) |
| mean | 0.753 | 0.740 | 1.70 | 1.13 | 0.501 (all yes) | all yes | 0.835 (all no) |

Every seed clears the CORRECTED gate: type1 in the operating window, type2_auc >= 0.65, meta_d > 0,
M-ratio >= 0.60, permuted collapses, within-class ok.

### Controls

<!--derived-->

- **PERMUTED (load-bearing):** decorrelating the margin from the trial collapses type-2 to chance on all six
  (mean ~0.50, meta_d → 0). This is the "type-2 back to chance when the confidence's informative content is
  destroyed" control — it passes cleanly.
- **WITHIN-CLASS:** the margin predicts correct/error WITHIN a fixed stimulus class on all six (min ~0.58–0.76).
  It reports how-sure-I-was, not which-stimulus.
- **LOOP-ABLATION (diagnostic):** zeroing the recurrent accumulator loop does NOT collapse the balance type-2
  (mean ~0.84, slightly HIGHER than intact). This is the honest signature that balance-of-evidence confidence
  lives in the evidence ENCODING (input→rate margin), not the recurrent amplification: it is a genuine
  decision-variable read but NOT type-1/type-2 dissociable (a theoretical property of balance-of-evidence
  confidence). The dissociable-but-seed-fragile comparator (`margin_abs`, zero of six) remains the next rung for
  a DISSOCIABLE spiking monitor.
- **Brain-based:** every confidence value is read from `cp_firing_states`; no host confidence formula is called
  (asserted at runtime; `read_from_firing_states` True and `host_confidence_formula_used` False on all seeds).

## Honest verdict

<!--derived-->

**Part 1 (instrument):** the second-order metacog GO gate was already correct (the boundary finding mis-read
it); it is now extracted + `--selftest`-locked so a chance-level type-2 input can never pass, and the corrected
gate's honest verdict on the OLD default (`meta_rate`) mechanism is NO-GO / NEGATIVE.

**Part 2 (mechanism): GO, six of six seeds.** A pure-spiking, content/evidence-sensitive balance-of-evidence
read produces genuine metacognitive sensitivity (meta-d′ above zero on every seed; type2_auc above the 0.65 bar
on every seed) where the impoverished downstream read is at chance and the only other clean sweep is a
host-classifier shortcut. This is the D4 lesson replicated in the metacognition setting: the confidence must be
a RICH, content/evidence-sensitive read of the settled first-order dynamics, read from spikes. This is a genuine
BRAIN-BASED metacognition correlate and unblocks wiring an honest functional confidence read-out onto the turn
("my decision-margin reads this as low-confidence, so I'm uncertain").

**The honest limit (not deferred, mapped):** balance-of-evidence confidence is not type-1/type-2 dissociable
(loop-ablation does not collapse it). A fully architecturally-separable second-order monitor (a slow-NMDA
comparator whose access can be lesioned while sparing the first-order decision — `margin_abs`) is still
seed-fragile and is the named next rung; the D4 insight (read the SETTLED, content-sensitive competition margin
directly from spikes) is the lead to make that comparator robust. This is a FUNCTIONAL correlate, NOT a claim of
subjective experience (phenomenal consciousness is OPEN, arguably untestable).

External grounding (Author, YEAR cites; page numbers are bibliographic, not measurements):

<!--derived-->

Kepecs, Uchida, Zariwala & Mainen (2008, Nature) — decision confidence as distance-to-category-boundary /
balance-of-evidence, orbitofrontal firing scaling with confidence; Fleming & Lau (2014, Front Hum Neurosci) —
meta-d′ / M-ratio; Maniscalco & Lau (2012, Conscious Cogn) — meta-d′ type-2 SDT.
