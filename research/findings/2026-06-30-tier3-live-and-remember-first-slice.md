# Tier-3 'live-and-remember' — the FIRST persistent living agent (perceive → remember → converse about lived experience)

**2026-06-30 (CYCLE 733-735, autonomous loop; owner-picked Option 1 of the Tier-3 capstone scoping).**
The first artificial-life-capstone SYNTHESIS slice: a merged one-brain that LIVES a drive-biased life, PERCEIVES +
GROUNDS + STORES the objects it encounters *during its own behaviour*, can be QUERIED about what it lived (the
no-confab moat intact), and PERSISTS across a reset. **Seed-42 GO (5/5 gates) on the real merged bridge; the full
6-seed is [PENDING — bg buy0f6vhn, ETA ~48 min].** Runner: `research/runners/_tier3_live_and_remember_derisk.py`.
**NO `sim/` edit** (one additive default-off `co_resident_drive` passthrough on `MergedNavConvAgent`, forwarding a
`build_merged_nav_conv_bridge` param that already existed).

## Why this is the genuine Tier-3 residual (the SURPASS re-frame)
The scoping (`2026-06-30-tier3-artificial-life-capstone-scoping.md`) found the capstone **largely done in PIECES** —
three artifacts each with two of the three living-agent axes (`develop_gpu` = continuous+converses but listen-only+
scripted; `persistent_living_loop` = continuous+body but no-converse; `navigate_to_compose_then_answer` = body+
converses but bounded+scripted). The genuine residual is the **intersection cell none fills**: *continuous life ∧ a
perceiving/composing body ∧ conversation about what it LIVED ∧ open-ended (self-chosen) ∧ persistent.* This runner is
the runner-only JOIN of those validated seams.

## The loop (host code legitimate ONLY for the world + body)
The agent lives in a corridor (food at cell 0; landmark OBJECTS at other cells; one object held OUT of the world =
the moat cue). Each step: the drive biases a validated survival policy (the rate-proxy Q — the motor-system stand-in;
the *learned* spatial policy on the substrate is the deferred Tier-4 dendrite wall); energy depletes; eating refills →
an INTRINSIC drive-reduction reward (Keramati-Gutkin; NO host distance term). On FIRST arrival at an object cell the
object is rendered into the perception slice and `agent.perceive_and_ground(obj)` grounds the LIVE spiking percept
into the co-resident composer; the agent STORES a lived fact linking it to the previously-encountered object
(`(prev, "near", cur)`). WHICH objects it knows is thus a consequence of its own drive-biased trajectory (open-ended,
not a scripted perceive-list). After the life the agent answers "what did you encounter near X?" from its lived,
grounded memory — or ABSTAINS on a never-encountered object. The life PERSISTS (body + lived facts + grounded codes
via `BridgeLineage`); a reset → reload resumes the SAME memory.

## The five gates (seed-42, on the real merged bridge, GPU)
| gate | result | evidence |
|---|---|---|
| **survival** (drive load-bearing) | GO | intact minE **0.95** / crash **0%**; LESION minE 0.00 / crash 12%; YOKE minE 0.00 / crash 30% |
| **drive-is-spiking** | GO | corr(deficit, `drive_agrp` firing rate) **+0.98** (controlled sweep, ≥0.9) |
| **lived, open-ended memory** | GO | grounded 3 encountered objects, stored 2 lived facts, recall **2/2**; **corrupting the grounded codes collapses recall to 0.00** (load-bearing) |
| **converse + no-confab MOAT** | GO | who/what recall correct AND abstains on the never-encountered `river` (**1/1**); conversational synapses **byte-identical** across the live run |
| **persistence across reset** | GO | reload resumes **2/2**; a no-persistence cold-start is **empty (0/2)** |

## The debug arc (honest — two bugs found + fixed via the smoke/1-seed ladder)
1. **KeyError `'near'`** (smoke-1): `RFPhasorComposer` builds its concept codebook only from the declared vocab;
   `vocab=None` fell back to a `DEFAULT_VOCAB` lacking the link verb (grounding *adds* objects, not the verb). Fixed
   by passing the exact `OBJECT_WORDS + ACTIONS` vocab the validated `navigate_to_compose.build_compose_bridge` uses.
2. **lived-memory anti-cheat mis-designed** (seed-42, 4/5 → the only failure): lesioning the grounding *before* the
   life stores self-consistent garbage codes and queries with the *same* garbage codebook → recall survives (it tests
   code *consistency*, not *correctness*). Fixed to `navigate_to_compose`'s validated **lesion-recompose** pattern —
   corrupt the grounded codes *after* storing → the codebook shifts *under* the stored composites → recall collapses
   → the recall provably depends on the SPECIFIC live-grounded codes = the percept. (Validate-by-function: the
   anti-cheat now matches what the grounding *computes* — correctly identifying the perceived object.)

## Tractability
Seed-42 first ran in **25 min** (6 gen-convergence-training bridge builds/seed dominate). Since `rate_proxy` survival
is pure host (the validated `persistent_living_loop` mechanism, NO bridge), the lesion/yoke controls now run
**host-only** (0 builds) via `live(perceive=False)`; the bridge is built only for the intact merged life + one
persistence agent (cold + resumed share it) = **2 builds/seed → 8 min/seed** (6-seed ETA ~48 min).

## Honest scope (deferred; flagged in the runner docstring)
- The **learned spatial policy** on the substrate = the deferred Tier-4 dendrite wall (survival uses the validated
  rate-proxy stand-in; survival, not spatial optimality, is the discriminator).
- **Persistence** is JSON re-instate (body + lived facts + grounded codes), NOT the raw `cp_connections` synaptic
  tensor — the `develop_gpu`/`LivingState` cheap-first stand-in (§1f); true synaptic persistence is a follow-on.
- **Open-endedness** is encounter-driven on a corridor (the agent grounds what its foraging brings it to, not a
  scripted route/layout — the discriminator vs `navigate_to_compose`); the richer 2D path-dependent-order is a follow-on.
- **Pure-spiking-reward survival** (`--drive-reward spiking`, the spiking hunger shaping the intrinsic reward) is
  smoke-validated (corr +0.92) but expensive; the 6-seed uses the tractable `rate_proxy` survival with the spiking
  drive corr-gated. A fully-spiking-reward 6-seed is the follow-on.

## Anti-cheat standing rules (all honored)
No-confab moat NEVER weakened (byte-frozen in vivo; the composer slice is array-disjoint from the nav read-out — moat
holds by construction). Validate-by-function (drive-lesion/yoke = survival; grounding-corruption = lived-memory).
6-seed for the robustness claim (pending). Honest negatives are the deliverable (the seed-42 anti-cheat mis-design
was caught, root-caused, and fixed — not hidden).

## Verdict
Seed-42 demonstrates the FIRST persistent living agent that perceives, remembers, and can be talked to about what it
lived — on the merged one brain, moat intact, NO `sim/` edit. **[6-seed robustness PENDING — buy0f6vhn.]** On a
6/6 GO this closes the first genuine Tier-3 synthesis slice; the ranked follow-ons (Option 2 develop-with-a-body,
Option 3 cross-modal one-animal, Option 4 lived consolidation) remain.
