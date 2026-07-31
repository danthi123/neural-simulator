---
type: finding
status: live
date: 2026-06-20
mechanism: fhrr
---

# FHRR-B Mechanism 1 — the unbind synapse as a one-time LOCAL reciprocal-wiring rule (the last host residual in the bind STRUCTURE, ELIMINATED)

**Type:** implementation + multi-seed CPU/numpy de-risk + CI guard. NO GPU. Stayed on `main`. Reuse-by-import + ONE guarded, default-OFF, reversible runner-side flag on `RFPhasorComposer`. The no-confab moat is NOT weakened (abstentions byte-identical).

## Verdict

**GO — the host `np.conj` computation of the unbind structure is ELIMINATED by a one-time LOCAL reciprocal-conjugate wiring rule, and the result is BYTE-IDENTICAL to the host-conj path across 4 seeds × 3 dims.** The genuine last residual in the FHRR-B bind structure (controller-verified: `rf_phasor_composer.py:204-209`, `_unbind_phases` — `zr_conj = np.conj(self._to_phasor(self.roles[role]))`, the host computing `conj` of the developmental role code and injecting it) is closed: with the flag ON, the unbind synapses are derived from the BIND synapses by a per-synapse quadrature(imaginary)-component flip applied ONCE at construction, with **zero** `np.conj` calls and **no** re-derivation from the role code. The bind structure is now a host-free device configuration (the property a neuromorphic hardware port needs).

This does NOT add a capability and does NOT make the conjugate *learned* — it makes it a *local developmental wiring rule* instead of a *host computation*, which is the brain-based-purity + hardware-portability goal the scoping (`2026-06-20-binding-structure-self-organization-scoping.md`, `e0cd6cf6`, Mechanism 1) named as the expected honest end-state.

## The residual (controller-verified) and the fix

| | the BIND synapse | the UNBIND synapse |
|---|---|---|
| code (legacy) | `conns=[(D+k, k, zr[k])]`, `zr` = the role phasor | `zr_conj = np.conj(self._to_phasor(self.roles[role]))`; `conns=[(D+k, k, zr_conj[k])]` |
| status | **developmentally-cheap RANDOM** — `self.roles[r] = rng.uniform(0,1,D)` per seed (a genome-style draw, accepted as self-organized like `sim/dendritic_neuron.py:25`, catalog F.12/D.18) | **the genuine host residual** — the host KNOWS "unbind = conj(bind)" and computes it; the substrate is never told the unbind synapse is the reciprocal of the bind synapse |

**The fix (Mechanism 1 — a one-time LOCAL reciprocal-wiring rule):** for a unit-magnitude phasor the conjugate is the quadrature (imaginary-component) sign flip `re + i·im → re − i·im`, a **purely-local function of each single synapse's own weight**. So instead of the host re-deriving `conj(role)` from `self.roles[role]` per op, at construction:

1. `_bind_conns(role_phases)` builds the BIND connectivity `[(D+k, k, zr[k])]` — the developmental role phasor installed directly (the bind synapse).
2. `_reciprocal_conjugate(bind_conns)` applies the LOCAL rule — for each bind synapse `(post, pre, w)`, emit the reciprocal synapse carrying `complex(w.real, -w.imag)` — computed locally from that synapse's OWN weight, with **no** read of `self.roles` and **no** `np.conj` over the role vector. Biologically a reciprocal/feedback connection with a quadrature-sign flip (the ubiquitous cortical/thalamocortical reciprocal motif).

Both `_unbind_phases` (single) and `_unbind_all_phases` (the batched production query path) branch on `self.local_reciprocal_unbind`: ON = the local rule; OFF (default) = the legacy host-conj path, byte-for-byte unchanged.

## Byte-equivalence table (the gate — local rule vs host-conj)

`research/runners/_fhrr_b_mechanism1_local_reciprocal_unbind_derisk.py` (SIM_BACKEND=numpy), seeds 42/43/44/45 × D 64/96/128:

| sub-check | what is compared | result |
|---|---|---|
| **unbind connectivity** | the local-rule weights vs `np.conj(role)` reconstructed inline, per role (agent/action/patient/polarity/attribute/attribute2), exact complex equality | **12/12 cells identical** (all roles) |
| **held-out bundle recovery** | bundle 3 role-filler bindings → unbind each role → the recovered phases out of the RF resonate | **12/12 cells byte-identical** |
| **full who/what matrix + abstentions** | 5 stored facts; who/what Q&A, one-attribute ("big apple"), generation, yes/no, AND the 4 no-confab abstentions (`None`/`unknown`) | **12/12 cells identical** |
| **batched query path** | `_unbind_all_phases` over the whole store (per role) + the actual batched query answers incl. the moat | **4/4 seeds identical** |

Sample matrix answers (seed 42, D=64) — correct, not identical-but-broken: `who go north → dog`, `what river look → big apple`, `render river → river look big apple`, `yesno dog stop east → yes`, and all four MOAT cues abstain (`None`/`None`/`unknown`/`None`).

## Substrate-purity assertion

Instrumenting `np.conj` to count calls during ONLY the unbind-structure build (single + batched), seed 42 D=64:

| flag | `np.conj` calls in the unbind-structure build |
|---|---|
| OFF (host-conj, legacy) | **2** (one for `_unbind_phases`, one for `_unbind_all_phases`) |
| ON (local rule) | **0** |

**Assertion holds:** with the flag ON, NO `np.conj` runs at unbind build, and `self.roles[role]` is read only to install the BIND synapses (the developmental role phasor) — the unbind connectivity comes **solely** from the local rule over the bind connectivity. (In scope is ONLY removing the host `conj` computation; the role codes' `rng.uniform(seed)` draw + the learned concept codes are accepted as developmental/learned per the scoping.)

## Anti-cheats

| control | expectation | result |
|---|---|---|
| **byte-identity to host-conj** (the gate) | flag-ON answers == flag-OFF answers on the full matrix + abstentions | **PASS** (12/12 cells, 4/4 batched) |
| **moat untouched** | every abstention (`None`/`unknown`) byte-identical | **PASS** (part of the full-matrix identity) |
| **permuted-role collapses** | unbinding with the WRONG role's local rule does NOT recover the filler | **PASS** 4/4 seeds (correct role → `north`; permuted role → ≠ `north`) |
| **lesion destroys recovery** | zeroing the local-rule unbind synapse weights → cleanup no longer recovers the filler | **PASS** 4/4 seeds |
| **flag-OFF reversible** | the flag-OFF path's unbind connectivity is EXACTLY the pre-edit logic | **PASS** (default-OFF byte-identical; 30/30 existing `test_rf_phasor_composer.py` pass unchanged) |

## The rule / flag summary (+ sim/ edit status)

- **Flag:** `RFPhasorComposer(local_reciprocal_unbind=False)` — default OFF = the legacy host-conj path, byte-identical to today; ON = the local rule. Reversible, guarded, additive.
- **Helpers added** (`research/runners/rf_phasor_composer.py`): `_bind_conns(role_phases, lo, hi)` (the bind connectivity) + `_reciprocal_conjugate(bind_conns)` (the local per-synapse quadrature-flip rule). `_unbind_phases` + `_unbind_all_phases` branch on the flag.
- **NO `sim/` edit.** The whole change is runner-side reuse-by-import of the existing RF complex-synapse substrate (`rf_set_complex_weights` consumes the connection list either way). The bridge is untouched.
- **CI guard:** `tests/test_rf_phasor_composer.py::test_rf_phasor_composer_local_reciprocal_unbind_byte_identical` (3 seeds) pins the connectivity-identity, the full-matrix + abstention identity, and the zero-`np.conj` purity assertion.

## Hardware-port implication

With the flag ON, the entire bind structure for a role is a **one-time device configuration** computed once at construction with no runtime host call: install the developmental role phasor as the bind synapses, install its per-component quadrature-flip (a local rule) as the reciprocal unbind synapses. The runner confirms `unbind_weight == conj(bind_weight)` exactly via the local rule. Combined with the random role codes (a fixed config-loadable weight table) and the learned concept codes (a learned weight table), the whole bind+unbind structure is host-free at runtime — portable to a memristor-crossbar / Loihi-2-synapse-table style one-time configuration. The residual that previously required "a host in the loop per op" (the runtime `np.conj`) is now a one-time wiring rule the configuration step applies.

## Honest scope / boundaries

- **No capability added** — the strategic prize (generalization across similar concepts) is already delivered on the *codes* axis (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`). This closes a brain-based-purity + hardware-portability residual only.
- **Not "learned," but "developmental"** — the conjugate is now a *local wiring rule* run at construction, not a *host computation*. Under the project's own standard (`dendritic_neuron.py:25`, catalog F.12/D.18/L.01) a fixed local wiring rule IS developmental self-organization — exactly how the genome specifies the retina's center-surround. Making the conjugate *activity-learned* (Mechanism 2: reciprocal-STDP refinement) remains the weeks-scale, research-grade option, worth attempting only if the local-rule reduction is judged insufficiently "emergent."
- **Out of scope (a separate residual, NOT touched):** the cleanup codebook `np.conj` over the concept codes (`rf_phasor_composer.py:255/335`; `one_brain_composer.py` 7 `comp.concepts[...]` sites) — the scoping marks it "reducible to LEARNED" (Option-1 GO), a different residual.

## Extension — the PRODUCTION-default one-brain path (`--composer onebrain`) closed too

The controller-verified residual for THIS scope was `rf_phasor_composer.py`'s `_unbind_phases`, closed above. But `OneBrainComposer` (the flagship production default at V=320, `--composer onebrain`) carries the SAME unbind residual at **6 sites** (`comp.roles[...]`: lines ~360/396/465/556/567/654 — block decode, batched unbind ×2, clause decode ×2, reconsolidation). Since the owner's goal is hardware-portability of the *production* bind structure, the same rule was threaded through it (default-OFF):

- `OneBrainComposer(local_reciprocal_unbind=False)` (default) propagates the flag to the inner `RFPhasorComposer`; a single `_unbind_conj(role)` helper returns `comp._local_conj(role_phasor)` (the local rule) when ON, else `np.conj(role_phasor)`. All 6 unbind-structure sites route through it; the 7 cleanup-codebook conj sites (`comp.concepts[...]`) are left untouched (the separate out-of-scope residual).
- `_local_conj(z) == np.conj(z)` **bit-for-bit** for a unit phasor (max|diff| = 0.0, verified directly) — so the answer-identity is backend-independent (the masked `rf_kick` is GPU; the math is not).
- **CPU smoke, 3 seeds:** `OneBrainComposer` store+query OFF vs ON byte-identical — `query_agent('go','north')='dog'`, `query_patient('cat','run')='south'`, `query_patient('dog','go')='north'`, and the moat `query_agent('go','south')=None` — all three seeds identical. The shared `_local_conj` primitive makes the GPU production path identical by the same bit-for-bit argument.

⇒ the host `np.conj` computation of the unbind structure is eliminated on BOTH the `rf` reference composer and the `onebrain` production default; the cleanup-codebook conj is the only remaining (separate, reducible-to-learned) conj residual.

## Files

- `research/runners/rf_phasor_composer.py` — `local_reciprocal_unbind` flag + `_local_conj` (the shared local-conj primitive) + `_bind_conns` + `_reciprocal_conjugate` + the branch in `_unbind_phases` / `_unbind_all_phases`.
- `research/runners/one_brain_composer.py` — the same `local_reciprocal_unbind` flag (propagated to the inner composer) + `_unbind_conj` routing all 6 unbind-structure sites through the local rule (cleanup conj untouched).
- `research/runners/_fhrr_b_mechanism1_local_reciprocal_unbind_derisk.py` — the de-risk runner (byte-equivalence gate, substrate-purity, anti-cheats, hardware-port note).
- `research/findings/raw/_fhrr_b_mechanism1_local_reciprocal_unbind.json` — the raw result.
- `tests/test_rf_phasor_composer.py` — the CI guard (+30 existing tests pass unchanged).
