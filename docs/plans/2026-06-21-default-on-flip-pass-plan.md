---
type: plan
status: live
date: 2026-06-21
---

# Default-on flip pass — implementation plan

> The dominant remaining shortcut-closure work, per the definitive inventory
> (`research/findings/2026-06-21-shortcut-inventory-definitive.md`, `ddc3b8db`). The spiking /
> hardware-portable versions of most cognitive ops are **already built and validated but default OFF**, so
> the production one-brain DEFAULT still runs host paths. This pass flips the validated defaults (plus a few
> tiny builds) so the one brain is fully spiking end-to-end by default. Sequenced behind the in-flight
> closure fronts (#6 FIX-A, #9, #3 fold) by file-ownership — concurrent edits to the same file are edit
> conflicts, not parallelism. Captured 2026-06-21.

## The motivating finding

The inventory established **closed ≠ default**: a validated spiking version *existing* was being counted as
"closed," but many are opt-in. The two-criteria bar (runtime-spiking AND on-substrate structure, for the
neuromorphic-hardware port) is only met by the *default* path once these flips land. The B1 (V1 receptive
fields) and A13 (dialogue association graph) scopings both confirmed the same shape: the on-substrate version
is built + validated, defaults OFF.

## The overarching HARD gate (applies to every flip)

1. **No-confab moat preserved** (0-breach) — non-negotiable; do not flip anything that breaches it.
2. **Parity vs the host oracle** — the flipped path equals the host path on the validated matrix, OR a
   documented, owner-acceptable delta.
3. **No regression** — the existing CI tests + production demos pass.
4. **CPU-portability preserved** — GPU-only flags stay opt-in (numpy-CPU path + the rf test-oracle intact).
5. **Default-OFF escape retained** — every flip is revertible (flip back = byte-identical to today).

After all chunks: a **combined-config validation** (all flips on together — moat + no-regression under the
combined config, mirroring the CYCLE 269–271 conversational-capability consolidation).

## The chunks (file-scoped; each runs after its in-flight front frees the file)

### Chunk C — Composer (`one_brain_composer.py`, `rf_phasor_composer.py`) — behind the #3 fold build
- **A4/A5 — flip `local_reciprocal_unbind=True` default.** The FHRR-B unbind+cleanup conjugate structure
  (closed this session via the local reciprocal-wiring rule; byte-identical) currently ships OFF, so the
  production composer still calls host `np.conj`. Flipping makes the bind+cleanup structure host-free by
  default (np.conj=0 at build) = the neuromorphic-port close.
- **A6 — confirm/flip the spiking cleanup-select default** (the NEF/WTA cleanup vs host argmax).
- **A8/A9/A12 — flip `integrated_loop=True` default** *after* the #3 fold build validates (the spiking K-way
  sequencer replacing the host first-match `_scan`).
- Gate: `tests/test_one_brain_composer_agent.py` (11) + the 320 demo verbatim + moat + answer-identity.

### Chunk A — Agent + production demos (`brain_conversational_agent.py`, the demos) — behind the #3 fold
- **A13 — flip `enable_learned_assoc=True` default + plumb the two demos** (`consolidated_320_conversation_demo.py`,
  `MultiTurnAgent`). The spiking `LearnedAssocGraph` (Hebbian CA3 recurrent, validated 24/24 edges / 9/9 top
  associate, no `sim/` edit) replaces the host co-occurrence dict.
- Verify the CYCLE 269–271 conversational flips (attributed / multiframe / neural_render / biased_competition)
  are still default-on.
- Gate: `elaborate` parity vs the host-dict oracle + moat + lesion/no-learning collapse + the demos plumbed.
- Keep a `False` numpy-CPU / test-oracle escape (the learned path runs a ~1800-neuron GPU bridge per `hear()`).

### Chunk N — Navigation (`g11_bg_runner.py`, `nav_conv_merged_bridge.py`) — behind #6 FIX-A + #9
- **B1 — drop the `apply_v1_gabor_weights` overwrite** on the already-`plastic=True` `retina→V1` pathway →
  let the local rule develop the RFs, OR inject a developmentally-random oriented bank (B1 de-risked GO; the
  host Gabor formula becomes the scoring reference only).
- **B2/B3/B4 — flip the nav limbic core** (spiking reward / value / dopamine conversions) to default-on. Note
  the GREEN_INERT caveat: they are validated but inert in the ceiling-bound nav; flipping makes them
  spiking-by-default (the brain-based goal) even where nav-inert. Document, don't hide, the inertness.
- **The #9 surpass** (dendrite-value SNc-subtraction calibration port + validate-by-function on a
  delayed-reward task) — folds in here after the #9 verdict.
- Gate: nav not regressed + the conversational moat (array-disjoint from nav) unchanged + each flip validated.

## Order of execution

1. As each in-flight front lands → run its file's chunk (one writer per file, TDD, commit each green step).
2. After all three chunks → the combined-config validation (all flips on; moat + no-regression).
3. Update the definitive inventory: move the flipped items to CLOSED-fully (both criteria, default).

## Deliberately deferred (not part of this pass)

- **Deeper host-free end-states** that the flips approach but don't fully reach: the one-bridge association
  fold (A13's deeper close — learned graph + spread on ONE bridge, no dict hand-off), and the
  fully-host-free composer (the structure is host-free after A4/A5, but some setup remains host-orchestrated).
  These are follow-ons, not blockers for "fully spiking by default."
- **GPU-only flags stay opt-in** (`enable_rf_cudagraph` megakernel, `composer_kind=onebrain`) — deliberately
  GPU-if-available-guarded to preserve numpy-CPU portability and the rf test-oracle.
- **The characterized boundaries** (#4 spiking decision +1.46, #6 orienting if FIX-A doesn't close it, #5b
  place-δ, B4 TD cue-shift) are research items under their own surpass rounds, not flips.

## Why this is the right shape

Most of the spiking work is already done and validated; the gap to "fully spiking end-to-end by default" is
overwhelmingly configuration (flip + plumb), not invention. That is the cheapest, lowest-risk path to the
owner's bar, and it is gated entirely on the no-confab moat plus non-regression — no new mechanisms, every
step revertible.
