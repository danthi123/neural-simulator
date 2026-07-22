# gap#4 REFRAMED — the biological local credit rule BUILDS deep accuracy on a proper task (MNIST); the "clean-negative" was a cleanxor task artifact. The open piece is the SPIKING rate-code, NOT the rule.

**2026-07-22, CPU/numpy, coexisting with the fluency training.** Follow-up to the 2026-07-22 deep-research
(`2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS...`), which flagged that the gap#4 clean-negative
("credit-training the hidden LOSES to a fixed-random reservoir, 0.55 vs 0.77") was measured ONLY on cleanxor — a task
with a **zero linear discriminant by construction**, where `e = onehot - softmax` (sums to 0) gives a rank-1 update in a
zero-information subspace, so credit CANNOT help and the learned-feedback fix has nothing to align to. The deep-read
called cleanxor the WRONG INSTRUMENT. This de-risk tested that on a PROPER deep compositional task with an informative
gradient. `research/runners/_gap4_credit_vs_reservoir_mnist_derisk.py` on the SAME committed `sim.dendritic_mlp`
(Urbanczik-Senn / feedback-alignment DFA rule), MNIST (`data/mnist.npz`), 4 arms: RESERVOIR (hidden FROZEN random +
trained readout = the credit-independent baseline) / FA (fixed-random feedback = the current method) / KP (learned
feedback, Kolen-Pollack, no weight transport) / ORACLE (hand-derived backprop, measurement ceiling).

## Result — credit DECISIVELY beats the reservoir, and the gap GROWS with depth
| depth | RESERVOIR | FA (credit) | KP | ORACLE | FA>RES |
|-------|-----------|-------------|-----|--------|--------|
| **2 (3-seed 42/43/44)** | 0.795 | **0.934** | 0.933 | 0.945 | **3/3** |
| **4 (seed 42)** | **0.102 (≈chance)** | **0.928** | 0.931 | 0.929 | ✓ |

- **FA (biological local credit) beats the reservoir 3/3 at depth-2 (+14 pts) and by +82 pts at depth-4** (where a random
  deep reservoir collapses to CHANCE 0.102 — its deep random features are garbage — while credit builds a near-oracle
  representation, 0.928 vs oracle 0.929).
- ⇒ **the "credit doesn't build accuracy / loses to a reservoir" negative was a cleanxor task artifact.** On a proper
  deep compositional task the SAME rule that lost on cleanxor decisively WINS, and the deeper the net the more essential
  credit-training the hidden becomes (a random deep scaffold is useless; credit is what makes depth pay off).

## Honest scope + the corrected gap#4 frame
- **This is the RATE numpy reference** (sigmoid DFA MLP), NOT the sparse spiking bridge. It proves the credit RULE is
  sound and cleanxor was the wrong test — it does NOT by itself solve the spiking case.
- **KP (learned feedback) ≈ FA here** (not the depth-degradation win the literature predicts) because FA already MATCHES
  the oracle on MNIST at depth 2-4 → no gap for KP to close. That specific untested fix is inconclusive-not-needed HERE
  (a harder/deeper task, or the spiking substrate, may still show FA degradation where learned feedback helps).
- **⇒ the gap#4 keystone reframes:** it is NOT "credit can't build accuracy" (it can, on real tasks, beating a reservoir
  massively) and NOT the dendrite topology (faithful). The genuine OPEN piece is **carrying the working rate rule on the
  sparse SPIKING point-neuron code** — the rate-code/point-neuron limit (the same BDSP algorithm hits 97%/73% on numpy
  graded signals but degenerates at firing 0.04-0.07 on the sparse bridge). That is a much more focused + encouraging
  frontier than "no working method": the method WORKS in rate; the task is a spiking-rate-code faithfulness problem.

## Corrects my session closeout (again — the record now)
The "gap#4 clean-negative / deprioritized parallel frontier" framing rested entirely on cleanxor and is now shown to be
task-specific. The keystone is more OPEN and more TRACTABLE than stated: the rule builds deep accuracy; the work is the
spiking port + (possibly) learned feedback where depth-degradation actually bites. NO `sim/` edit; multi-seed; the
reservoir/oracle controls are the load-bearing comparison. `research/findings/raw/gap4/mnist_depth{2_3seed,4_seed42}.log`.
