# Full FHRR-on-bridge — layer (c): production-scale validation + optimization — 2026-06-05

Layer (c) of the owner-greenlit full FHRR-on-bridge feature, after layer (b) complete
(`2026-06-05-fhrr-layer-b-rf-composer-GO.md`). The owner said "Proceed" → continue the optimization + a fuller
production-scale validation; the (c2) production switch of `BrainConversationalAgent` is the remaining
**owner-sign-off-gated** step (autonomous work does not flip the production agent without explicit sign-off).

## (c1) capability parity — GO multi-seed
Added generation (`render_fact` — "river look big apple", decoded from the RF unbind, no-confab moat preserved).
The RF composer now reproduces the rate-coded composer's CORE capability matrix: who/what Q&A, abstention,
negation/yes-no, one-attribute, recursive clauses, dialogue, generation. Full matrix at a 5-fact set, multi-seed 3/3.

## (c-scale) production-scale validation — GO multi-seed
The key KB-growth risk is spurious matches (a query falsely matching the wrong fact as the KB grows). At **10 facts**
(D=128): who/what retrieval picks the right fact via the (action,patient)/(agent,action) cue, and the no-confab moat
does **not** false-match — abstention cues correctly return None. Multi-seed 3/3, ~3.5 s/seed.

## (c-opt) optimization
- **Bridge reuse** (commit e81d6914): the composer caches RF bridges by neuron count, avoiding
  `_initialize_simulation_data` per op. The big win is a long conversation (one composer, many turns reuse the same
  2D/(L+1)D bridges).
- **Shorter resonate period:** the full capability matrix holds down to **period=80** (3/3 seeds) — latency scales
  linearly with the period, so period=80 is 4.5× faster than 400. Adopted a safe-margin **period=200** default
  (2.7× faster, comfortable phase resolution 0.5%), verified to hold for ALL cases (b.1–b.4 + one-attribute +
  clauses at D=128 + 10-fact scale + dialogue).
- Remaining (incremental, deferred): SPARSE complex weights (layer-a used dense N×N — the diagonal/unit synapses are
  O(D) sparse, relevant only at large D), batched per-fact ops. The GPU complex matvec is already backend-agnostic.

## Status: the RF composer is switch-ready (pending owner sign-off on the switch itself)
Correctness-complete (full capability matrix), production-scale-validated (10 facts, moat holds), optimized
(bridge-reuse + period=200), and ZERO regression on the existing agent/composer/models from the 6 protected `sim/`
edits. **The remaining step is (c2): switch `BrainConversationalAgent` to the RF composer** — a real production
change. Per the standing rule (owner steers major milestones; the production switch needs explicit sign-off), this
is surfaced for the owner. On switch: the opponency is cleared on the production path, and the F=3 two-attribute
resonator (which the ±1 scheme provably can't do) becomes available to lift the documented 2-attribute K=5 boundary.
The rate-coded composer stays production until the switch lands; no capability regression ships silently.

## Artifacts
- `research/runners/rf_phasor_composer.py` (generation + bridge-reuse + period=200); `tests/test_rf_phasor_composer.py`
  (b.1–b.4 + c1 + c-scale, period=200); `research/findings/raw/_rf_period_sweep.py` (the period sweep).
- commits: c1 b212c1ec, c-opt-reuse e81d6914, c-scale eda0ae09, period (this). Frozen bars / no-confab moat untouched.
