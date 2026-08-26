---
type: biology
id: metacog-three-factor-confidence
mechanism: Type-2 (second-order) metacognitive confidence as a reward-gated THREE-FACTOR Hebbian read of the brain's own first-order decision competition — an ACC/aPFC opponent (V+ "was correct" / V- "was error") confidence channel whose feature→correctness mapping is learned by a dopamine/RPE-gated local rule, NOT a host optimizer
status: established
last_verified: 2026-08-26
current_finding: research/findings/2026-08-18-self-organized-metacog-monitor-GO.md
current_status: "BRAIN-BASED mission GO (plastic_acc 6/6, mission_go=True; NO sim/ edit; runner-level de-risk). The type-2 confidence→correctness MAPPING is learned by a LOCAL, reward-gated THREE-FACTOR HEBBIAN rule over the brain's OWN first-order 2AFC competition — closing the host-logistic residual of the wave-1 learned_acc GO. pre = standardized ACC/aPFC evidence-&-conflict read (winner/runner rate, margin, balance, conflict, late-conflict/persistence); post = opponent aPFC V+/V- meta_schema subpools; third factor = dopamine/RPE δ=correct−V (V a TD-tracked reward baseline). w_plus/w_minus = Σ|δ|·z over each reward-gated subset, each normalized by its dopamine mass (Rescorla-Wagner asymptote). Single-pass, no loss minimisation, no matrix solve, no weight transport. Scored in Maniscalco-Lau type-2 SDT / meta-d'. 6/6 seeds: mean type2_AUC=0.825 (parity ratio 0.982 vs host learned_acc 0.841; AUC-parity 6/6 ≥0.85×host_mean=0.715), mean meta-d'=2.49, all in the [0.60,0.90] type-1 window. ALL controls collapse 6/6: meta-lesion (zero the monitor's drive → type2_AUC→0.500, meta-d'→0 while d'/accuracy UNCHANGED — the type-1/type-2 dissociation), permuted-confidence (200-draw permutation test, perm-p=0.005), self-read-lesion (re-apply the SAME three-factor rule on SHUFFLED correctness feedback → chance), within-class type2_AUC>0.55 (orthogonal to raw stimulus difficulty). Anti-cheat: host_logistic_fit_calls_on_path==0 asserted; weight_source=reward_gated_three_factor_hebbian. Independent 2026-08-26 seed-42 rerun reproduces the banked GO byte-identically (AUC 0.855, meta-d' 2.49). REMAINING RESIDUAL: the presynaptic ACC features are host RATE READS of the brain's own competition (not yet a fully-spiking presynaptic population read) — the next rung, NOT the mission bar. FUNCTIONAL metacognition correlate only; no phenomenal claim."
sources:
  - path: "doi:10.1016/j.neuron.2017.06.005 (Fleming & Daw 2017, Neuron — 'Self-Evaluation of Decision-Making: A General Bayesian Framework for Metacognitive Computation')"
    anchor: "second-order model in which confidence reflects a computation over the first-order decision variable"
    note: "EXTERNAL. The theory the mission bar is scored against: metacognitive confidence is a SECOND-ORDER read that predicts first-order correctness from the decision variable, dissociable from first-order sensitivity (type-2 vs type-1 / meta-d' vs d'). Our monitor is the second-order read; meta-lesion is the dissociation manipulation."
  - path: "doi:10.1037/0033-295X.109.4.679 (Holroyd & Coles 2002, Psychol. Review — 'The neural basis of human error processing: reinforcement learning, dopamine, and the error-related negativity')"
    anchor: "the error-related negativity is generated when the dopamine system conveys a negative reinforcement-learning signal to the anterior cingulate cortex"
    note: "EXTERNAL. The ACC error/conflict monitor is driven by a dopaminergic reinforcement-learning (RPE) signal — the third factor of our rule: a DA dip (δ<0) on an error writes the V- ('was error') channel, a DA burst (δ>0) writes V+. The ACC feature read is the pre; the RPE is the gate."
  - path: "doi:10.1126/science.275.5306.1593 (Schultz, Dayan & Montague 1997, Science — 'A Neural Substrate of Prediction and Reward')"
    anchor: "dopamine neurons report an error in the prediction of reward"
    note: "EXTERNAL. Grounds the third factor δ=correct−V as a dopamine reward-prediction-error against a running (TD-tracked) baseline V, gating which opponent channel is written — the Rescorla-Wagner/TD reference our _three_factor_weights computes before each update."
  - path: "doi:10.1038/nature14366 (Namburi, Tye et al. 2015, Nature — 'A circuit mechanism for differentiating positive and negative associations')"
    anchor: "positive and negative valence are encoded by distinct, opposing populations"
    note: "EXTERNAL. Motivates the OPPONENT post: two distinct subpopulations (V+ 'correct', V- 'error') rather than one graded channel — confidence = rate(V+) − rate(V-). Opponent coding is what makes the read symmetric across the two decision classes (the within-class control)."
implemented_by:
  - research/runners/_second_order_metacog_monitor_derisk.py
findings:
  - research/findings/2026-08-18-self-organized-metacog-monitor-GO.md
  - research/findings/2026-08-17-wave1-second-order-metacog-6-GO.md
  - research/findings/2026-08-02-laneC-metacog-margin-comparator-PARTIAL-real-signal-not-robust-next-is-symmetric-or-learned-error-monitor.md
---

# Type-2 metacognitive confidence — a reward-gated three-factor Hebbian ACC/aPFC opponent read

**What is measured.** A downstream aPFC/ACC monitor reads the brain's OWN first-order 2AFC decision
competition (winner/runner accumulator rates, the winning margin, the balance/conflict between competitors,
plus dynamic late-conflict and response-persistence terms) and emits a graded spiking confidence that predicts
whether the first-order decision was CORRECT — the type-2 SDT / meta-d' currency of Maniscalco & Lau. The
confidence→correctness MAPPING is not a host logistic fit: it is learned by a local, reward-gated three-factor
Hebbian rule and read out through spiking opponent (V+/V-) meta_schema subpools. Brain-based (no `sim/` edit),
mission GO 6/6 seeds, parity with the host-logistic ceiling.

## The three factors (why it is a plausibility, not a curve-fit)

- **pre** — the standardized ACC/aPFC feature vector: the same evidence-and-conflict reads of the first-order
  competition that the `balance`/`learned_acc` modes use (a "cleanup/winner score", the accumulator margin, the
  number/strength of surviving competitors via balance & conflict).
- **post** — the OPPONENT aPFC confidence channel: two spiking subpopulations, V+ ("this evidence pattern was
  right") and V- ("... was wrong"), after Namburi-Tye valence opponency. Confidence = rate(V+) − rate(V-).
- **third factor** — a dopamine/RPE gate δ = correct − V, V a TD-tracked reward baseline (Schultz RPE;
  Holroyd-Coles ACC error monitoring; Fleming-Daw confidence learning). A DA burst on positive surprise writes
  V+, a DA dip on an error writes V-; the presynaptic feature is the eligibility. Each channel is normalized by
  its total dopamine mass (Rescorla-Wagner asymptote), making it frequency-robust. Single-pass, local — no loss
  minimisation, no matrix solve, no weight transport.

## Why it is load-bearing (the dissociations that DEFINE metacognition)

The claim is not "a number correlates with correctness"; it is that the correlate is a genuine second-order read.
**Meta-lesion** severs the monitor's access and collapses type2_AUC→0.500 / meta-d'→0 while d' and accuracy are
UNCHANGED — the type-1/type-2 dissociation (Fleming-Daw). **Self-read-lesion** re-applies the SAME Hebbian rule
on SHUFFLED correctness feedback (a 200-draw permutation test) and falls to chance — the CONTINGENT reward-feature
pairing, not the architecture, organizes the mapping. **Permuted-confidence** decorrelates confidence from the
trial and falls to chance. **Within-class** type2_AUC>0.55 shows the read tracks correctness WITHIN a fixed
stimulus class — orthogonal to raw difficulty. All four collapse on all 6 seeds.

## No `constraints_config` bound

The plastic operating point (td_alpha 0.1, gain 1.2, conf range 150–750 pA, 96 calibration trials, 40 report
steps, dynamic features) is an EMPIRICAL calibration on this substrate, not a biology-REQUIRED constant. Binding
it as a hard config requirement would over-claim — the same discipline as the `affect-active-clear` and
`deep-credit-on-spikes` entries. The biology-REQUIRED invariants are structural, not scalar: the third factor
must be an RPE against a running baseline (δ = correct − V, not raw correctness), the post must be OPPONENT (two
channels, not one), and the mapping must be reward-GATED (no host optimizer) — all enforced in code by the
`host_logistic_fit_calls_on_path==0` tripwire and the self-read-lesion permutation control, not by a numeric gate.
