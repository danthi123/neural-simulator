# objrel emergent-learning — per-role graded reward-modulated Dale-legal-spiking plasticity GENUINELY REACHES objrel on MOST seeds (BPTT 0/6 → 9/10 with salience), correctly a BOUNDARY: the residual is per-seed STOCHASTIC INIT-BASIN REACHABILITY of the 7:1 minority (not a representational wall)

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_dopamine_plasticity_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_dopamine_plasticity.json`
**Verdict:** BOUNDARY (correctly) — but a GENUINE emergent advance, adversarially verified (Workflow `wkokh77e0`, 2/3 skeptics non-refute + 1/3 refute; MISLABELED_BOUNDARY = the runner mislabeled WHY, the boundary itself is correct).
**Builds on:** `2026-07-06-objrel-DANN-emergent-learning-BOUNDARY.md` (BPTT-from-scratch 0/6) + `2026-07-06-objrel-analytic-reference-adversarially-VERIFIED-*.md` (the Dale-legal spike-native read EXISTS in weight space, 6/6) + `2026-07-06-objrel-freq-vs-geometry-isolation-DEEPER-RESIDUAL.md` (neither frequency nor margin recovers it — a reachability residual).

## The mechanism (genuinely works; NOT confounded — rules out overclaim)
Per-ROLE independent binary **Dale-legal spiking** detectors (the see-saw killer: each role its own excitatory E-path + its own inhibitory interneuron carrying that role's NEGATIVE rows → one output LIF; the per-slot decision argmaxes the roles' OUTPUT-LIF spike counts). Each detector's signed direction is learned from a RANDOM Dale-init by a **GRADED reward-modulated three-factor delta rule** `Δw = lr·salience·(target−act)·feature` (Schultz graded RPE; a bang-bang reward is degenerate under the 7:1 imbalance — dev-probed — the graded margin-reward reaches the discriminant). Deployed Dale-legal-spiking. NO `sim/` edit; reuse-by-import; CPU/numpy.

**Adversarially verified GENUINE (2 high-confidence non-refuting skeptics, independent re-runs reproducing the JSON exactly):**
- **EMERGENT, not warm-started:** on the recovering seeds objrel-slot0 rises 0.00 (pre-learning, random init in the majority-AGENT basin) → 1.00; the THEME detector's learned direction grows init-norm ~0.18 → ~13.5 with **cos(init,final) ≈ 0** — real iterative work carving the minority signed direction, decisively NOT the retracted inert-BPTT/warm-start pattern.
- **Genuinely SPIKING:** decode = argmax over output-LIF summed spike counts (independent re-impl == runner every example); no-output-spike lesion → exactly 0 spikes / chance; the graded 1-2-spike THEME>AGENT margin FLIPS with the true role (not a saturated tie, not a host f@W argmax).
- **Dale-legal** all seeds (W_e≥0, W_fi≥0, W_io≤0; no mixed-sign neuron). **Held-out** (0/12 objrel test/train overlap; the subject-word pool is SHARED across THEME-slot0 and AGENT-slot1 → per-word memorization is impossible; train-only == held-out). **Reward-load-bearing** (no-reward → 0.00, shuffled-reward → 0.00 all 6 vs reward-on 1.00).

## The result — REACHES objrel on most seeds, but correctly a BOUNDARY (no config 6/6)
- **Official 6 seeds (42/43/44/100/101/102):** objrel-slot0 is BINARY per seed (12/12 or 0/12 — a single reachability Bernoulli). Salience 5/6 (fails 101), uniform 5/6 (fails 43) — they TRADE seed 43↔101.
- **Extended 10 seeds (42-46, 100-104), the decisive sample:** **SALIENCE recovers 9/10 (90%), UNIFORM 6/10 (60%)**; paired, salience > uniform on 5 seeds, uniform > salience on 1 (only 101). Uniform FAILS on {43,45,46,103}.
- vs the fixed spiking WTA baseline 0.50 and BPTT-from-scratch **0/6** — the emergent reward-modulated per-role rule is a REAL advance (reaches objrel 9/10). The analytic target is 1.00 (6/6, exists in weight space).

## The two HONEST CORRECTIONS the adversarial-verify forced (of an earlier over-read — the discipline working)
An earlier state note (CYCLE 931f) and the closure subagent's own draft claimed (a) "UNIFORM meets the strict GO bar" and (b) "salience is a red herring." Both are **REFUTED** by the extended-seed evidence:
1. **"Uniform meets the strict bar" is a LUCKY-PARTITION ARTIFACT (p-hacking), not principled.** The official blind trio {100,101,102} happens to contain ZERO of uniform's failures {43,45,46,103}, so "3/3 blind" is an artifact of WHICH seeds are blind, not a robust property. Worse, uniform's blind 3/3 leans on seed **100, which is INIT-LUCKY** (pre-learning already 1.00, no-reward already 1.00 → the plasticity does NO work there) — so uniform's genuinely-EMERGENT blind recovery is only **2/3, identical to salience**. Re-designating uniform-as-main to pass the gate = selecting the config that happens to pass THIS partition.
2. **Salience is NOT a red herring — it HELPS** (9/10 vs 6/10, paired 5:1). The "equal 5/6" appearance is confined to the exact official 6-seed window (salience's win on 43 and uniform's win on 101 cancel because 101 is in the blind partition). Salience is load-bearing; dropping it is not honest.

## The runner's self-verdict TEXT was factually WRONG (a gate-logic bug, corrected)
The runner self-stamped BOUNDARY with a verdict TEXT claiming "plasticity inert (pre≈learned)" and "reward not load-bearing" — BOTH FALSE (per-seed pre 0.00→1.00; reward-on 0.83 vs no-reward 0.17 vs shuffled 0.00). The cause: `all()`-ANDed per-seed flags that only seed 100 (init-lucky) breaks, plus the mean-collapse (mean pre 0.167 = five 0.00 + one 1.00). The BOUNDARY CONCLUSION is correct (no config 6/6); the WHY was mislabeled.

## The precise residual + the NEXT mechanism (a boundary = the next mechanism, not a wall)
The residual is **genuine emergent-reachability of the 7:1-minority signed-THEME direction**: a per-seed STOCHASTIC-reachability Bernoulli (the random init lands in the majority-AGENT basin ~10-40% of the time depending on config), NOT a representational/substrate/Dale wall (the analytic Dale reference reads 1.00 on all seeds; a host linear argmax generalizes held-out ~100%). Salience helps (biases toward the minority) but does not eliminate the basin-miss.

**Next mechanism (attack REACHABILITY directly, not the reward-weighting — per the adversarial-verify's guidance):**
1. **Basin-escape / init:** a THEME-detector init biased OFF the majority-AGENT basin, OR **multiple random restarts per role selected by the reward critic** (biologically: exploratory sampling + reward-critic selection — the RL exploration the emergent-learning research gate emphasized; Miconi node-perturbation / Legenstein reward-modulated exploration). The failure is a per-seed Bernoulli, so K restarts drive the miss-rate to ~(miss)^K.
2. **Pre-register a ≥10-seed FIXED dev/blind split BEFORE the run**, report the per-config recovery distribution (salience AND uniform) so no partition is chosen post-hoc.
3. **Require genuinely-emergent recovery (pre < 0.85) on every counted-blind seed** (exclude init-lucky seeds like 100 from the GO tally).
4. Fix the runner's `all()`-ANDed per-seed flag logic so the verdict TEXT stops falsely reporting inert/reward-not-load-bearing.

## Files
- `research/runners/_rungB1c_objrel_dopamine_plasticity_derisk.py` — the closure de-risk (per-role Dale-legal spiking + graded reward-modulated delta; NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_dopamine_plasticity.json` — 6-seed-blind record + anti-cheats.
- Adversarial-verify Workflow `wkokh77e0` (2/3 non-refute + 1/3 config-honesty refute; MISLABELED_BOUNDARY).
