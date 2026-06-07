# N6 action-selection DECISION biologized — host argmax → spiking accumulate-then-commit (+ Cisek urgency); REALISTIC performance under the owner-relaxed gate; the residual is biology's faithful goal-change decision-memory — 2026-06-06

**Status:** N6 (the action-selection readout) — the DECISION is now made in SPIKES, not by a host argmax. GO under the
owner-relaxed gate ("realistic, not unreasonably slow" — NOT "match the memoryless argmax's 2.34"). Multi-seed +
grid-32 validation IN FLIGHT (appended below). NO `sim/` edits; all flags additive + default-off; 68/68 runner tests
pass. This concludes the high-priority N6-decision sub-arc the owner steered.

## The one-line result

The host-side argmax over thalamic pool rates (cheat N6's *decision* mechanism — the action was chosen by a Python
`argmax`, off-brain) is REPLACED by a genuine spiking decision: a recurrent NMDA **accumulator** integrates the clean
thalamic evidence to a bound (Wang 2002) and a downstream **commit burst** fires all-or-none at threshold (Lo-Wang
2006 / Stine-Shadlen 2023), with a Cisek collapsing-urgency bound so weak late-phase evidence still commits. The
decision EMERGES from spiking dynamics. It scores **4.08** (grid-8, seed 42) — **beats the original tonic+motor cheat
(5.0)**; the residual gap to the optimized argmax (2.34) is concentrated entirely in the post-goal-change phases and
is the *biologically faithful* cost of a decision circuit that carries memory (a real brain does not instantly
abandon a commitment when the target jumps).

## The arc (how we got here — the owner steered "fully biologize the argmax, exhaust the methods")

| step | mechanism | result | why |
|---|---|---|---|
| baseline cheat | host argmax over thalamus rates | 2.34 | the decision is computed off-brain (the cheat) |
| naive spiking WTA | motor-pool WTA / TRN WTA | 14.7 / 20.0 | NEGATIVE — a passive instantaneous comparator |
| **deep research** | catalog + Kandel + literature | diagnosis | the WTA had NO recurrent self-excitation (passive comparator) → provably can't make a winner from a weak signal; the brain ACCUMULATES then COMMITS |
| accumulate-then-commit | Wang-2002 NMDA accumulator → Lo-Wang commit burst | 4.71 (BOUNDARY) | decision GENUINELY spiking, but goal-change hysteresis + weak-drive silent-commit |
| **+ Cisek urgency** | collapsing commit-bound (urgency 180) | **4.08** | the urgency fixes the silent-commit; the decision stays spiking |

(The loser-only reset and the combined config were tried — 5.58 / 4.35 — and were worse than urgency-180 alone; the
naive all-trial reset was 6.93. Urgency-180 is the production config.)

## The mechanism (biologized, grounded)

- **ACCUMULATE:** each `sel_X` selection pool has NMDA-slow recurrent self-excitation (soft-WTA gain α<1 — a stable
  Wang-2002 attractor, NOT the unstable α>1; structured rather than blanket cross-inhibition per
  Rutishauser-Douglas-Slotine 2011) and joins the bridge's per-region NMDA mask, so it amplifies + integrates the weak
  clean thalamic drive to a bound over the readout window. (The thalamus is cleanly selective under the genuine
  GPi→thal disinhibition shipped earlier — N8.)
- **COMMIT:** a downstream tonically-inhibited `commit_X` burst pool (a superior-colliculus / burst-generator
  analogue) fires all-or-none when its `sel_X` crosses threshold. The selected action = which `commit_X` bursts.
- **URGENCY:** a ramping action-independent urgency current into `sel_X` over the readout window collapses the
  effective commit bound with elapsed time (Cisek-Kalaska / Thura-Cisek), so a weak late-phase winner still commits
  within the 100 ms window — eliminating the silent-commit fallback.

## Guards — the decision is REAL (not a hidden argmax)

- commit-burst winner **15.3 vs runner-up 0.0** (≈500× separation) — a decisive spiking winner.
- the winner's `sel_X` accumulator visibly RAMPS (cumulative 20→42→70→103→144) while losers stay at 0 — genuine
  integration, not an instantaneous comparison.
- the committed action matches the clean thalamic winner ~80% overall, **90–96% at stable goals** (where the host
  argmax is fully gone).
- 68/68 existing runner-flag tests pass; the readout is off by default (backward compatible); NO `sim/` edits.

## The relaxed gate (owner, 2026-06-06) + why 4.08 clears it

The owner relaxed the target: "real brains don't instantly snap to new targets either, so the goal-change delay is
biologically faithful; optimize reasonably but don't pour immense resources into matching the memoryless argmax."
So the gate is REALISTIC performance, not 2.34. 4.08 clears it:
- it **beats the original cheat** (tonic+motor argmax, 5.0);
- the agent re-acquires every goal — per-phase [0.6, 0.5, 1.42, 1.55]: phases 0–1 ≈ the argmax reference (~0.585),
  the cost is the goal-change phases and is a *delay*, not a failure to navigate;
- the residual is the biologically-correct decision-memory: a real accumulator carries its commitment briefly when
  the target jumps. The argmax "wins" goal-changes only by being memoryless — which is *less* biological.

## Honest scope

- The host argmax (the off-brain decision) is REMOVED — the decision is genuinely made in spikes. This is the win.
- The residual goal-change cost is accepted as biologically faithful (owner-endorsed), not ground away.
- Per the owner's reasonable-budget guidance, the further menu options (explicit race-to-threshold integrators,
  DA-modulated bound) were NOT pursued — urgency-180 is a realistic, grounded production config.
- Validation: grid-8 seed 42 here; **multi-seed (43/44) + grid-32 IN FLIGHT** (appended below).

## Production config
```
... --genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300 --genuine-thal-tonic-pa 750 \
    --readout-source spiking_wta --urgency-max-pa 180 ...
```
Flags (all additive, default-off): `--readout-source spiking_wta`, `--urgency-max-pa`, `--reset-losers-only` (opt-in,
not used in production). NO `sim/` edit.

## Cross-references
- `2026-06-06-action-selection-readout-deep-research.md` (the diagnosis + the accumulate-then-commit prescription)
- `2026-06-06-N6-accumulator-commit-readout-BOUNDARY.md` (the accumulate-then-commit, decision real, 4.71)
- `2026-06-06-N8N6-combined-readout-GO.md` (N8 disinhibition + the thal-source readout this builds on)
- `2026-06-06-navigation-cheat-audit-and-conversion-plan.md` (the cheat inventory)
