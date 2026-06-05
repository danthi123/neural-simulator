# (B) memory shortcut — the STORAGE is CLEARED (substrate weight-store == numpy at production D=2048) — 2026-06-05

**The composer's fact memory now lives in the substrate, not numpy.** Following the de-risk GO
(`2026-06-05-B-substrate-store-fidelity-GO.md`), the substrate weight-store was integrated into the composer
(opt-in `enable_spiking_memory`, commit 0304abf1) and the production validation confirms it: the spiking-memory
composer answers the capability matrix IDENTICALLY to numpy `self.kb` storage at **D=2048, V=320, seeds 42/43/44 →
27/27 match → GO**. The default numpy path is byte-unchanged; 15 on-brain tests green; NO `sim/` edits.

## The arc (research-first, the A pattern)
Owner steered "ground in the science." The (B) literature synthesis
(`2026-06-05-substrate-held-memory-literature-synthesis.md`) gave the Crawford-Eliasmith two-store NEF design and the
decisive verdict that **engrams BINARIZE** (so an engram-per-fact would corrupt the graded bound vector — engram is
out for graded storage). De-risk GO: a bound vector imprinted in connection WEIGHTS, retrieved IN SPIKES, unbinds at
numpy parity (12/12 per seed, recon cosine ~0.975, genuine spiking read confirmed by a 145× collapse on zeroing the
trigger). Integration → production validation 27/27. The crux (graded-pattern fidelity of substrate-held bound
vectors) — which the literature said was solved at this exact shape (Crawford held 117,659 facts at D=512) — holds on
this substrate at production parity.

## The mechanism (NO sim/ edits)
`core_sim_composition.py`, opt-in `enable_spiking_memory`:
- `store(fact)`: build the bound (ON,OFF) vector via `bind_fact` (the numpy superposition/opponency stay numpy — the
  linear glue follow-on), then IMPRINT it into a persistent per-fact substrate store: a trigger population (n_trig=40)
  whose OUTPUT weights onto two D-neuron ON/OFF readout banks (n_per=4) ARE the bound vector. `self.kb` holds
  `(fact, handle)` instead of `(fact, numpy_vector)`.
- query (`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`): `_get_bound(handle)` fires the trigger and reads
  the reconstructed (ON,OFF) IN SPIKES, then `_unbind_onoff` + cleanup as today.
- `enable_spiking_memory=False` (default) → `self.kb` numpy storage byte-unchanged.
- Reuse-by-import from the de-risk probe (`build_store_bridge`/`retrieve_bound`); per-fact stores
  (~6,440 neurons/fact at D=800; a shared-readout-bank consolidation is a possible optimization).

## Status of (B) + what remains
**CLEARED:** the STORAGE — the fact memory is held in the substrate (weight-store), retrieved in spikes, at numpy
parity multi-seed. This was the deep part of the (B) shortcut (the audit's "MEMORY shortcut").

**Remaining for the FULL (B) clear (the linear glue follow-on):** the two LINEAR inter-phase ops in `bind_fact` —
the superposition sum `bon += o` and the ON/OFF opponency `onoff(bon − boff)` — are still numpy (disclosed linear ops
per pillar n=111). In-network: superposition = rate-summation of the per-role binds on a shared accumulator bank;
opponency = ON/OFF lateral inhibition. This is the architectural piece (the spiking binds must drive a shared
accumulator on the bridge) and the last numpy in the composer's compute path.

## Where the composer stands now (the compute path)
bind (spiking, n=111) → [superposition + opponency: numpy linear glue — REMAINING] → store (substrate weight-store,
B CLEARED) → unbind (spiking) → cleanup (spiking NEF, A CLEARED). The two deep shortcuts the owner named — the
readout (A) and the memory storage (B) — are both replaced by validated, fully-spiking, biology-grounded mechanisms
at production parity. Only the linear inter-phase glue remains.

## Artifacts
- Integration: `core_sim_composition.py` (commit 0304abf1); test `tests/test_core_sim_composition.py` (spiking_memory).
- Production validation: `research/findings/raw/_b_memory_composer_validate.py` + `_b_memory_composer_validate.json`
  (27/27 D=2048 multi-seed).
- De-risk: `2026-06-05-B-substrate-store-fidelity-GO.md`; synthesis: `2026-06-05-substrate-held-memory-literature-synthesis.md`.
- Backend: CuPy / RTX 3090.
