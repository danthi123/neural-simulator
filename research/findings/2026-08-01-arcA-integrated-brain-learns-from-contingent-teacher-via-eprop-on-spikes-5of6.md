---
type: finding
status: live
date: 2026-08-01
mechanism: teacher-contingent-development
artifacts:
  - research/findings/raw/_a1_teacher_contingent_eprop_6seed.json
  - research/findings/raw/gap4_6seed_shuffleDFA/arcA_seed46_evalorder.json
---

# Arc A first de-risk: the integrated brain LEARNS from a CONTINGENT teacher via e-prop on the production spiking substrate — 5/6 (near-GO), contingency load-bearing

> ## ⚠️ CORRECTION (2026-08-01 PM): the "5/6" is largely a MEASUREMENT ARTIFACT, not a robust per-seed property
> The lone failing seed (46) is not a robust failure. Under **identical** config its accuracy ranges **0.200–1.000**,
> decided by an **uncontrolled bridge-noise offset** (the OU/background stream is drawn from the process-global RNG
> and is not isolated from incidental RNG consumption). A read-only `accuracy()` call before training — the runner's
> `test_before` record — flips seed 46 **1.000 → 0.200 (Δ+0.800)** while barely moving a robust seed (seed 42
> Δ+0.017). Seed 46 learns **6/6** across small noise offsets and **1.0/0.983/0.950** across epochs 120/240/360; it
> fails only at the specific offset the canonical runner path lands on. So neither "5/6" nor "6/6" is a well-posed
> measurement until the noise stream is pinned. **The correct next step is an INSTRUMENT fix — isolate the bridge's
> stochastic stream so measurement calls don't perturb training — NOT config-tuning to force 6/6 (that would be
> p-hacking over an artifact).** Evidence: `research/findings/raw/gap4_6seed_shuffleDFA/arcA_seed46_evalorder.json`.
> This does NOT touch the load-bearing result below (contingency lesion clean, forward learning real); it corrects
> only the seed-46 failure interpretation.

**One-line verdict:** the north-star developmental atom works. A Kuhl-style *contingent* teacher names K=5 noisy
referents; the brain (`OnBridgeEpropNet`, a real Izhikevich `SimulationBridge`) learns cue→label from the
teacher's corrections and generalizes to fresh held-out draws. **5/6 seeds learn** (mean main 0.742 vs chance
0.200), and the teacher's **contingency is the load-bearing anti-cheat** — a non-contingent (random-label)
teacher collapses every seed (mean 0.139; margin 0.603). Reuse-by-import, no `sim/` edit. This is a de-risk
(report-only, not verify-go'd), one seed short of its own strict 6/6 gate — reported as such.

Artifact: `research/findings/raw/_a1_teacher_contingent_eprop_6seed.json`.

## Result — 6 seeds {42–47}, firm config (noise 0.08, 120 epochs, 30 settle)

| seed | main_test | non-contingent | learns |
|---|---|---|---|
| 42 | 0.967 | 0.067 | ✅ |
| 43 | 1.000 | 0.017 | ✅ |
| 44 | 0.550 | 0.217 | ✅ |
| 45 | 0.850 | 0.000 | ✅ |
| 46 | **0.200** | 0.167 | ❌ (at chance) |
| 47 | 0.883 | 0.367 | ✅ |

`n_learn` = **5/6**; mean main **0.742**, mean non-contingent **0.139**, contingency margin **0.603**. Weights
genuinely moved on the bridge (ff-moved ~75k–127k per seed). The result is **seed-variable**: three strong
(0.85–1.0), one moderate (0.55), one failure (0.20). Honest read: the mechanism is real and holds in 5/6, but
it is not the clean 6/6 the strict gate wants — seed 46 does not learn.

## Why this is the north-star atom

The develop-loop today does NO error-driven learning (WAKE = Hebbian co-occurrence; CONVERSE stores a binding
via a host VSA composer; SLEEP = a re-hear proxy). e-prop is the missing piece that turns a teacher's
*contingent* correction into weight changes on the shared spiking substrate. Crucially the teacher signal is
delivered as an **error** `L = softmax(logits) − onehot`, distributed by fixed feedback — it **vanishes at
match**, so it cannot become the "supervised clamp becomes a crutch" the prior research gate warned against.
This atom is the smallest proof that a co-resident spiking brain can be *taught* through the loop.

## Anti-cheats + honest scope
- **Contingency lesion (the load-bearing one): clean** — main 0.742 vs non-contingent 0.139 (a random-label
  teacher only memorizes noise and fails to generalize). This is what makes it *learning from teaching*, not
  leakage.
- **Shuffle-DFA is reported, NOT gated here** — at 1 hidden layer the spiking reservoir + trained readout
  carry this shallow associative task, so scrambling the hidden DFA credit does not collapse it. That control
  belongs with the depth-2 semantic-inheritance task (the SECOND de-risk), where deep credit is actually
  required.
- **Scope:** 5/6 not 6/6 (seed 46 fails); a *shallow associative* atom (the deep-credit claim needs the
  depth-2 task at its validated ~62 min/seed config with a frozen-hidden reservoir control); CPU-rate; a
  de-risk, not verify-go'd.

## Next
(1) The depth-2 semantic-inheritance contingent-teacher de-risk with the frozen-hidden reservoir control — to
earn the *deep-credit* claim. (2) **[CORRECTED — see the banner above]** Fix the INSTRUMENT: isolate the bridge's
stochastic (OU/background) stream so incidental measurement calls do not perturb training, then re-measure the
true per-seed distribution. Do NOT config-tune to force 6/6 — seed 46's failure is an uncontrolled-noise-offset
artifact (ranges 0.2–1.0), not a robust per-seed property. (3) Wire the e-prop cortex slice + teacher hook into `develop_gpu`'s CONVERSE seam and the gap#5
wake/sleep phase-switch into its SLEEP seam — the integration build proper.
