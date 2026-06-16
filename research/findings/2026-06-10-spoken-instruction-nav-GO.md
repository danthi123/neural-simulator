# Spoken-instruction navigation — GO: the FIRST functional integration of the navigation and conversational brains

> **Result: GO, multi-seed (42/43/44).** The agent follows a **parsed spoken command** to navigate — the
> conversational parser's *firing* opens a synaptic gate that lets the learned word→action route steer the
> navigation body, with the command as the **only** goal signal. Every control is decisive. This is the first
> time the two brains, which roadmap step 2 merged onto one substrate *without* interaction, **functionally
> interact** — and it closes the standing "the navigation reward isn't behaviorally load-bearing" residual (here
> the command *is* the only goal, so the cross-region route is load-bearing and lesion-confirmable).

## Why this matters

Roadmap step 2 put navigation and conversation on one `SimulationBridge`, but as disjoint slices with **zero
synapses between them** — they co-resided, they did not interact (the owner's challenge: "isn't the whole point
it being one brain?"). This result is the first genuine **functional** integration: the conversational
comprehension network's spikes drive the navigation action cascade through a single synaptic gate. One brain, two
systems, actually talking.

## The mechanism (fully brain-based; no `sim/` edit — reuse-by-import)

- **The body / cascade:** the basal-ganglia action-selection circuit `cortex_{N,E,S,W} → str_D1 → gpi → thal →
  motor → sel_{N,E,S,W}` (a spiking winner-take-all); the agent steps in the winning direction (per-direction
  disinhibition).
- **The learned word→action route** (NOT hand-wired): `language_input → cortex_X` is a real plastic pathway whose
  direction selectivity ("north" → cortex_N) is **grown by brain-based co-firing** — the Pulvermüller action-word
  somatotopy the project validated. (This replaces the de-risk's labeled-line stand-in.)
- **The gate, opened by the parser's firing:** one `command_route` transmission gate on `language_input →
  cortex_{N,E,S,W}`, held **closed** at rest, **coupled to the parser's action-role ensemble firing** via the
  in-substrate `couple_gate_to_indices` — when the parser comprehends the verb ("go") its action ensemble fires
  and opens the route. **No Python reads a value across the halves:** the parser supplies the *when* (a 0/1 gate
  from spikes), the word supplies the *which* (its learned route). Python is legitimate only for the environment
  (the grid) and the command schedule (the world's event timing).

## The task

A commanded-goal gridworld: the goal direction is **not** rendered to any retina and **not** given as
coordinates. Each phase a 3-word instruction ("dog go <direction>") is presented to the conversational channel;
the only way the body can know which way to move is parser-comprehends → gate-opens → the learned route biases the
cascade → the agent steps. The commanded direction changes across a multi-phase schedule. Metric =
command-following accuracy (chance = 0.25 for 4 directions).

## Results — 3-seed A/B (42/43/44)

| Condition | acc-vs-commanded (42 / 43 / 44) | reading |
|---|---|---|
| **COUPLED** (gate on, learned route on) | **1.000 / 1.000 / 1.000** | follows every command, every direction |
| ISOLATED-NAV (gate held closed) | 0.062 / 0.188 / 0.125 | at chance — no goal signal reaches the body |
| **LESION** (`command_route` cut, ~5,000 synapses zeroed, parser still firing) | 0.062 / 0.156 / 0.094 | at chance — cutting the synaptic route collapses it *even with the parser firing* |
| SCRAMBLE (the spoken word permuted) | vs-commanded 0.00 / 0.00 / 0.03; **vs-spoken 1.00 / 1.00 / 0.97** | the body tracks the *actual* spoken word, not a fixed mapping |
| ISOLATED-CONV (the conv brain alone) | parses correctly, **has no body — cannot move** | comprehension without action |

**Verdict = GO** (chance 0.25): COUPLED follows commands (≥0.5) all seeds; COUPLED ≫ ISOLATED-NAV & LESION
(margin ≥ 0.20); ISOLATED-NAV and LESION at chance (≤ 0.40) all seeds; ISOLATED-CONV parses but has no body;
SCRAMBLE collapses vs-commanded while vs-spoken stays ~1.0 — all seeds.

## Why the controls are decisive (it is the synapse, not a back-channel)

- **LESION** is the load-bearing control: with the parser still firing (the gate-open *signal* present) but the
  `command_route` synapses cut, command-following collapses to chance — so the behavior rides the **synaptic
  route**, not any ambient or Python path.
- **SCRAMBLE** proves the body follows the *content* of the spoken word (it tracks the permuted word), not a
  hard-wired schedule.
- **ISOLATED-CONV** confirms the conversational brain alone comprehends but cannot act — the integration is what
  produces behavior.
- Provenance: no Python copies the parsed direction into the navigation drive (the gate is opened by firing).

## 6-seed confirmation (42–47) — the flagged follow-on, now banked

The 3-seed A/B above (42/43/44) extends to **6 seeds**; seeds 45/46/47 (`research/findings/raw/spoken_instruction_nav_seed454647.json`, GPU) pass the gate decisively:

| seed | COUPLED (acc-vs-commanded) | ISOLATED-NAV | LESION | SCRAMBLE (vs-commanded) |
|---|---|---|---|---|
| 45 | **1.000** | 0.281 | 0.000 | 0.00 |
| 46 | **1.000** | 0.094 | 0.031 | 0.00 |
| 47 | **1.000** | 0.281 | 0.000 | 0.00 |

All 6 seeds (42–47): COUPLED follows every command (1.000), ISOLATED-NAV at chance (≤0.40), LESION collapses to ≤0.03 (the `command_route` synapses are load-bearing — cutting them kills the behavior even with the parser firing), SCRAMBLE 0.00 vs-commanded (the body tracks the actual spoken word). **Language→action functional integration is 6-seed GO.**

## Honest scope + what's next

Scope: 6-seed confirmed (42–47); the gating + learned route are both
project-validated primitives composed here for the first time. This is **interaction by gating a nav-native
channel** — it routes *around* the central cross-code problem (the nav perception is a rate code, the composer a
phasor code; they're not commensurable). The deeper interaction — **perception→memory** (what the agent perceives
while navigating writing into conversational memory) — hits that rate-vs-phasor wall head-on, and that wall is
precisely the deliverable that motivates **step 3 (the learned spiking cortex)**, whose job is to dissolve it.

## Trail

- Gating de-risk (GO): `2026-06-10-funcint-lang-to-action-cheap-first-GO.md`,
  `research/runners/funcint_lang_to_action_probe.py`.
- Design: `docs/plans/2026-06-10-functional-integration-one-brain-design.md`.
- Runner: `research/runners/spoken_instruction_nav.py`. Raw: `research/findings/raw/spoken_instruction_nav.json`
  (3-seed) + `..._smoke.json`.
