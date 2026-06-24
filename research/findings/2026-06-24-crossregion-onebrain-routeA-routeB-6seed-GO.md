# Cross-region one-brain: Route A default-ON + Route B host-`M` CLOSED, 6-seed GO (2026-06-24)

The TRUE one brain advances: two cross-region hand-offs in the merged nav+conv bridge
become spiking-by-default, closing the **last cross-region host shortcut on the compose
route**. Reuse-by-import, **no `sim/` edit**. Commit `9a9975d6`.

## How we got here (the composer-internal loop was already done; the cross-region gaps were the real work)

- The persistent-loop de-risk (CYCLE 526, `_persistent_loop_flat_derisk.json`) found the
  `OneBrainComposer`'s flat + clause recall was **already** a persistent on-bridge spiking
  loop — it carries the live register phasor across the unbind→cleanup hand-off on
  substrate; the only `to_host` is the final cleanup-membrane body-read. So the remaining
  one-brain work was the **cross-region** hand-offs, not the composer internals.
- The cross-region scoping (CYCLE 527, `_crossregion_persistent_loop_next_build_scoping.md`)
  found the cross-region wiring was **already built + GO**, just default-OFF, with **one**
  host shortcut left — a host projection `M` on the perception→compose route.

## Route A (language→action) — DEFAULT-ON

The spoken-command route — the parser's firing opens a synaptic command gate, and the
learned word→action route steers the nav body — is flipped default-ON
(`nav_conv_merged_bridge.py:1470/:1659`, following the `co_resident_nav_critic=None`
sentinel precedent). Nav-neutral by construction (edges held CLOSED at rest), moat-safe
(array-disjoint from composer/parser). Validated:
- **Gates 1+2:** `test_nav_conv_merged_agent` 8 + `test_nav_conv_step2b_coresident` 7 =
  **15/15 PASS** (Route-A-default-ON agent, moat intact).
- **Gate 4** (`spoken_instruction_nav`, seed 42): **GO** — COUPLED follows commands 1.0,
  isolated-nav + lesion collapse to chance (0.094), scramble collapses, provenance
  **LEARNED** (no Python value copy).

## Route B (perception→compose) — host-`M` CLOSED with spikes-only grounding

The host projection `composer.concepts[o] = angle(M @ cortex_it_rate)` is **RETIRED from
the default**. The default `gen_spikes` mode grounds the perceived object's spiking
`gen_concept` response — produced by the LEARNED `gen_perception → gen_concept`
convergence — into the co-resident composer's codebook: **SPIKES-ONLY, no host quantity
crossing regions**, with a spikes-only provenance assertion and an anti-smuggle guard. The
legacy host-`M` is kept ONLY as the `--grounding host_m` A/B comparison (never a default
fallback). Validated **6-seed**:

| seed | grounded | compose | mem-floor | lesion | moat | iso-perc | byte-id |
|---|---|---|---|---|---|---|---|
| 42  | 3 | 1.000 | 0.500 | 0.167 | 1/1 | 0 | ✓ |
| 43  | 3 | 1.000 | 0.500 | 0.333 | 1/1 | 0 | ✓ |
| 44  | 3 | 1.000 | 0.333 | 0.000 | 1/1 | 0 | ✓ |
| 100 | 3 | 1.000 | 0.500 | 0.500 | 1/1 | 0 | ✓ |
| 101 | 3 | 1.000 | 0.500 | 0.000 | 1/1 | 0 | ✓ |
| 102 | 3 | 1.000 | 0.333 | 0.333 | 1/1 | 0 | ✓ |

held-out compose **1.000 across all 6 seeds** (≫ mem-floor 0.444, chance 0.250); LESION
(sever the convergence) collapses the compose; MOAT **1/1** (0 false-accepts) every seed;
ISO-PERC 0 (no body → nothing grounded); byte-identity holds every seed.

## ⇒ The BRAIN-BASED-ONLY closure

The agent navigates the merged nav+conv bridge (the BG cascade selecting each move
neurally), grounds perceived objects in-episode **via spikes** (the learned convergence,
not a host projection), and composes held-out perceived-object facts that recover the
object — all on ONE brain, the no-confab moat intact, validated 6-seed. The perceived-object
grounding between sensation and action is now spikes-only.

## Honest scope + the held fork

- The agent-constructor `co_resident_perception` default stays host_m-tagged (flipping it
  on the constructor would crash `test_nav_conv_merged_agent` — `co_resident_perception
  requires co_resident_composer` — and force the composer+GPU onto every agent). Route B's
  default-flip is at the **behavioral-runner** surface (`build_compose_bridge` default
  `gen_spikes`). Exposing `gen_spikes` on the agent default is the natural follow-on (it
  needs the gen stack + composer in the agent default — not a pure flip).
- **The bigger one-brain follow-ons (HELD for owner steer):** (1) consolidate
  `MergedRFComposer` onto the co-resident `OneBrainComposer` (brings the proven
  composer-internal persistent loop onto the merge AND unlocks the limbic write-side);
  (2) Route D — synaptic comprehension (the parser→composer role assignment) for the RF
  composer; vs (3) pivoting to the artificial-life horizon (a richer multi-hundred-word
  corpus, since "run the develop loop longer" plateaus at the ~25-word corpus).
