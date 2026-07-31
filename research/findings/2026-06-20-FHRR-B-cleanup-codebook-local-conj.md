---
type: finding
status: live
date: 2026-06-20
mechanism: fhrr
---

# FHRR-B cleanup codebook — the last host `np.conj` residual, ELIMINATED by the SAME local conjugate wiring rule (bind+cleanup structure now fully host-free)

**Type:** implementation + multi-seed CPU/numpy de-risk + CI guard. NO GPU. Stayed on `main`. Reuse-by-import + the SAME guarded, default-OFF, reversible `local_reciprocal_unbind` flag Mechanism 1 added. The no-confab moat is NOT weakened (abstentions byte-identical).

## Verdict

**GO — the cleanup codebook's host `np.conj` (the matched-filter nearest-concept weights) is ELIMINATED by the same one-time LOCAL reciprocal-conjugate wiring rule already built for the unbind, and the result is BYTE-IDENTICAL to the host-`conj` path across 4 seeds × 3 dims, on BOTH the rf composer AND the production `OneBrainComposer`. With the flag ON, a FULL store+query build issues `np.conj` calls = 0 TOTAL → the WHOLE bind+cleanup structure is now host-free (random role codes + learned/developmental concept codes + local conjugate rules = a complete one-time device configuration, the neuromorphic-port property end-to-end).**

The genuine residual Mechanism 1 deliberately left untouched (`2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md` §"Honest scope": the cleanup codebook `np.conj` over the concept codes — `rf_phasor_composer.py:315/403`, the 7 `one_brain_composer.py` `comp.concepts[...]`/`comp.pol_words[...]` sites) is closed. The cleanup is a matched filter: it correlates the recovered phasor against each concept's CONJUGATE (the matched filter IS the transpose/reciprocal of the encoder). Because `conj` is per-component, each cleanup synapse is the per-component quadrature-flip of its concept synapse — the SAME `_local_conj` rule, a purely-local function of each single synapse's own weight, with NO host `np.conj`.

This does NOT add a capability and does NOT make the conjugate *learned* — it makes it a *local developmental/feedforward wiring rule* instead of a *host computation*, the brain-based-purity + hardware-portability goal. It is the cleanup-side twin of Mechanism 1's unbind reduction.

## The residual and the fix

| | the residual (legacy) | the fix |
|---|---|---|
| rf composer | `cc = np.conj(self._to_phasor(self.concepts[w]))` (`_spiking_cleanup`:315); `sims = (rec_z @ np.conj(cb).T).real` (`_cleanup_all`:403) | route the codebook conjugate through `_cleanup_conj` → `_local_conj` (the per-component quadrature flip) when the flag is ON |
| one-brain (production) | 7 sites `cc = np.conj(comp._to_phasor(comp.concepts[...]/pol_words[...]))` (`_read_block` ×2, `_build_batched_unbind_clean` ×2, `_read_all_blocks` stock ×2, `_decode_clause` ×1) | a single `_cleanup_conj(word)` helper (delegates to the inner composer's `_cleanup_conj`) routes all 7 sites |
| status | the substrate re-derived the concept-code conjugate host-side per cleanup build (`np.conj` over the concept vector) | the cleanup codebook is derived LOCALLY from each concept synapse by the same quadrature-flip wiring rule — no host `np.conj`, bit-for-bit `== conj` for a unit phasor |

**The rule:** for a unit-magnitude phasor the conjugate is the quadrature (imaginary-component) sign flip `re + i·im → re − i·im`, a purely-local function of each single synapse's own weight (the existing `_local_conj` primitive Mechanism 1 built and shared). Biologically the reciprocal/transpose of the concept-code synapse (the matched filter = the encoder transpose), the ubiquitous reciprocal motif. `_cleanup_conj(z) = _local_conj(z) if local_reciprocal_unbind else np.conj(z)` — flag-OFF (default) is the legacy host-conj path, byte-for-byte unchanged.

## Byte-equivalence table (the gate — local rule vs host-conj)

`research/runners/_fhrr_b_cleanup_codebook_local_conj_derisk.py` (SIM_BACKEND=numpy), seeds 42/43/44/45 × D 64/96/128:

| sub-check | what is compared | result |
|---|---|---|
| **cleanup codebook** | the local-rule cleanup-codebook weights (`_cleanup_conj`) vs `np.conj(concept_phasor)` reconstructed inline, per concept (full vocab + AFFIRM/NEGATE polarity tags), exact complex equality | **12/12 cells identical** |
| **batched cleanup codebook** | `_cleanup_all`'s `conj(cb)` over the whole codebook, local rule vs host conj, exact complex equality | **12/12 cells identical** |
| **full who/what matrix + abstentions** | 5 stored facts; who/what Q&A, one-attribute ("big apple"), generation, yes/no, AND the 4 no-confab abstentions (`None`/`unknown`) | **12/12 cells identical** |
| **spiking-cleanup path** | `enable_spiking_cleanup=True` (the `_spiking_cleanup` matched filter installs the cleanup-codebook synapses on the bridge) — who/what answers + the moat, OFF vs ON | **4/4 seeds identical** |
| **OneBrainComposer (production default)** | store 2 facts; `query_agent`/`query_patient` ×3 + the moat (`None`), OFF vs ON | **3/3 seeds identical** |

Sample matrix answers (seed 42, D=64) — correct, not identical-but-broken: `who go north → dog`, `what river look → big apple`, `render river → river look big apple`, `yesno cat look west → no` (stored NEGATE), and all four MOAT cues abstain (`None`/`None`/`unknown`/`None`).

## Substrate-purity assertion (the headline)

Instrumenting `np.conj` to count calls at build, seed 42:

| scope | flag OFF (host conj) | flag ON (local rule) |
|---|---|---|
| rf cleanup-codebook build only (`_spiking_cleanup` + `_cleanup_all`) | **18** | **0** |
| rf FULL store+query build (unbind STRUCTURE + cleanup CODEBOOK) | **14** | **0** |
| OneBrainComposer FULL store+query build (3 queries incl. the moat) | **342** | **0** |

**Assertion holds:** with the flag ON, a FULL store+query build issues **0** `np.conj` calls TOTAL. Mechanism 1 drove the unbind-structure sites to 0; this drives the cleanup-codebook sites to 0 too. So the ENTIRE bind+cleanup structure is host-free at build: the random role codes (a fixed config-loadable table), the learned/developmental concept codes (a learned table), the local reciprocal-conjugate unbind rule, and now the local-conj cleanup codebook are all that remain — a complete one-time device configuration with no host in the loop per op. (Out of scope, as before: the role codes' `rng.uniform(seed)` draw + the learned concept codes are accepted developmental/learned per the scoping; only the host `conj` *computation* was the residual.)

## Anti-cheats

| control | expectation | result |
|---|---|---|
| **byte-identity to host-conj** (the gate) | flag-ON answers == flag-OFF answers on the full matrix + abstentions, both composers | **PASS** (12/12 rf cells, 4/4 spiking, 3/3 one-brain) |
| **moat untouched** | every abstention (`None`/`unknown`) byte-identical | **PASS** (part of the full-matrix identity) |
| **permuted-codebook collapses** | correlating the recovered phasor against a DERANGED cleanup codebook (each concept's filter pointed at a DIFFERENT concept's local-conj code) does NOT recover the true filler | **PASS** 4/4 seeds (true → `north`; deranged → ≠ `north`) |
| **lesion destroys selection** | a zeroed (lesioned) cleanup codebook makes the selection content-free (scores flat → no longer depends on the recovered phasor) | **PASS** 4/4 seeds (true cleanup recovers `north`; the lesioned codebook's pick is content-free) |
| **flag-OFF reversible** | the flag-OFF cleanup codebook is EXACTLY `conj(concept)` (the pre-edit logic) | **PASS** (default-OFF byte-identical; 37/37 `test_rf_phasor_composer.py` pass, 4 skip) |

## The rule / flag summary (+ sim/ edit status)

- **Flag:** the SAME `local_reciprocal_unbind=False` flag (default OFF = the legacy host-conj path, byte-identical to today; ON = the local rule). No new flag — the cleanup codebook joins the unbind structure under the one rule, so "the whole bind+cleanup structure is host-free" is a single switch.
- **Helpers added:** `RFPhasorComposer._cleanup_conj(z)` (routes the codebook conjugate through `_local_conj` when ON, else `np.conj`) + `OneBrainComposer._cleanup_conj(word)` (delegates to the inner composer's `_cleanup_conj`). The 2 rf sites + 7 one-brain sites route through them.
- **NO `sim/` edit.** The whole change is runner-side reuse-by-import of the existing `_local_conj` primitive + the RF complex-synapse substrate (`rf_set_complex_weights` consumes the connection list either way). The bridge is untouched (`git diff -- sim/` = 0 lines).
- **CI guard:** `tests/test_rf_phasor_composer.py::test_rf_phasor_composer_cleanup_codebook_local_conj_byte_identical` (3 seeds, CPU) pins the cleanup-codebook identity, the full-matrix + abstention identity, and the **total-zero-`np.conj`** purity assertion on a full store+query build. `tests/test_one_brain_composer_agent.py::test_onebrain_cleanup_codebook_local_conj_byte_identical` (3 seeds, GPU-gated) pins the same on the production one-brain path.

## Hardware-port implication

With the flag ON, the entire bind+cleanup structure is a **one-time device configuration** computed once at construction with no runtime host call: install the developmental role phasor as the bind synapses; install its per-component quadrature-flip (a local rule) as the reciprocal unbind synapses; install the per-component quadrature-flip of each concept synapse as the matched-filter cleanup codebook (the encoder transpose). Combined with the random role codes and the learned concept codes (both config-loadable weight tables), the whole structure is host-free at runtime — portable to a memristor-crossbar / Loihi-2-synapse-table style one-time configuration. The residual that previously required "a host in the loop per op" (the runtime `np.conj` for both unbind and cleanup) is now entirely a one-time wiring rule the configuration step applies.

## Honest scope / boundaries

- **No capability added** — this closes a brain-based-purity + hardware-portability residual only; the strategic prize (generalization across similar concepts) lives on the *codes* axis (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`).
- **Not "learned," but "developmental/feedforward"** — the cleanup codebook is now a *local wiring rule* run at construction (the encoder transpose), not a *host computation*. Under the project's own standard (`dendritic_neuron.py:25`, catalog F.12/D.18/L.01) a fixed local wiring rule IS developmental self-organization. Making it *activity-learned* (a learned cleanup/familiarity readout) remains the separate, deeper learned-cortex frontier.
- **The bind operation stays the exact-inverse FHRR algebra** — this is a STRUCTURE-wiring residual (how the conjugate synapses are configured), not the binding-algebra idealization (the genuine learned-cortex bind = the separate step-3 frontier).

## Files

- `research/runners/rf_phasor_composer.py` — `_cleanup_conj(z)` helper + the 2 cleanup-codebook sites (`_spiking_cleanup`, `_cleanup_all`) routed through it.
- `research/runners/one_brain_composer.py` — `_cleanup_conj(word)` helper + the 7 cleanup-codebook sites routed through it.
- `research/runners/_fhrr_b_cleanup_codebook_local_conj_derisk.py` — the de-risk runner (byte-equivalence gate, total-purity, anti-cheats, hardware-port note).
- `research/findings/raw/_fhrr_b_cleanup_codebook_local_conj.json` — the raw result.
- `tests/test_rf_phasor_composer.py` (+ `tests/test_one_brain_composer_agent.py`) — the CI guards.
