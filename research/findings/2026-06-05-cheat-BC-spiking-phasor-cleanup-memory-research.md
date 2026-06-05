# Cheats B (cleanup) + C (memory store): biology-grounded spiking-phasor resolutions — research — 2026-06-05

**Scope.** The RF phasor composer (`research/runners/rf_phasor_composer.py`) runs bind/unbind/bundle in spikes on a
`SimulationBridge` of resonate-and-fire (RF) neurons with **complex synapses** (`cp_rf_w_re` + `cp_rf_w_im`, the
sparse complex matvec `u = W z` in `bridge._rf_advance_one`). But it REINTRODUCED two non-biological shortcuts that
the older **rate** composer (`core_sim_composition.py`, ±1 Hadamard codes) had already cleared — because the RF
codes are now FHRR **phasors** (phases in `[0,1)^D`), not the ±1/ON-OFF reals the cleared mechanisms assumed:

- **CHEAT B (cleanup):** `_cleanup` (lines 148–151) is a numpy phase-cosine argmax over the codebook
  (`words[argmax(mean(cos(2π(rec − code))))]`).
- **CHEAT C (memory store):** the fact KB is a Python list `self.kb = [(fact_dict, composite_phases), …]` (lines 74,
  172) — the bound composite is a numpy array + a Python label dict, not in the substrate.

This document finds, for EACH cheat, the **biology mechanism** (cited), the **on-bridge spiking realization** (the
rate composer's cleared mechanism, adapted to phasors), the **smallest de-risk test**, and an **honest difficulty**.

**The single most important finding:** the phasor versions of BOTH cleared mechanisms are the SAME object — a
**complex-valued (phasor) Hopfield / Threshold Phasor Associative Memory (TPAM)** whose weight matrix `W = S S*`
holds the phasor vocabulary (cleanup) or the phasor facts (store) — and that object is **already implemented in
numpy in this repo** (`research/runners/resonate_fire_fhrr.py::ResonateFireTPAM`, Frady & Sommer 2019) AND maps
**directly onto the bridge's existing complex synapse matvec** (`rf_set_complex_weights` installs exactly a complex
`W`; `_rf_advance_one` computes exactly `W z`). The phasor adaptation is therefore *more* aligned with the substrate
than the rate composer's ON/OFF mechanisms were — the substrate is complex-native.

---

## What the rate composer did (the two cleared mechanisms), and why phasors break the direct reuse

| | Rate composer (CLEARED) | Why it doesn't port verbatim |
|---|---|---|
| **B cleanup** | NEF thresholded cleanup (Stewart-Tang-Eliasmith 2011, the Spaun cleanup): `n_per` neurons/concept, encoders = the **real ±1 code** as ON/OFF receptive fields, a placed negative bias so off-target (cos≈0) emits **0 spikes**, true (cos≈0.31) fires; argmax over per-concept firing. `_spiking_cleanup_nef.py`; `2026-06-05-composer-cleanup-NEF-GO.md`. | The matched filter is `code · est` with **real** codes split into ON=max(code,0)/OFF=max(−code,0) currents. A **phasor** has no sign — `est` and codes are phases in `[0,1)`; the similarity is `mean cos(2π(est−code))`, a **circular** quantity, not a real dot product. The ON/OFF current split is undefined for a phase. |
| **C store** | Crawford-Eliasmith substrate weight-store: the bound **(ON,OFF) real vector** lives in the static OUTPUT weights `trigger_i → readout_ON[k] = bon[k]·w_gain`; fire the trigger → readout banks reconstruct `(bon′,boff′)` in spikes. `_b_substrate_weight_store_probe.py`; `2026-06-05-B-store-CLEARED.md` (117,659 facts at D=512, Crawford 2016). | The stored value is a **real** `(ON,OFF)` vector driving real readout-neuron firing **rates**. A phasor composite is a **complex/phase** pattern; "firing rate ∝ weight" loses the phase. The readout must reconstruct **phase**, which is spike-**timing**, not rate — a different decode (the RF phase readout, not a rate average). |

The common cure for both: **stop treating the code as a real vector; treat it as a complex phasor and use a
complex-weighted recurrent (Hopfield/TPAM) circuit**, which the RF substrate already supports.

---

# CHEAT B — phasor cleanup in spikes

## B.1 Biology mechanism

The cleanup is **nearest-stored-pattern recall = associative-memory pattern completion + winner-take-all**, two
biological mechanisms the project already grounds:

1. **CA3 pattern completion (the autoassociator).** Kandel 6e Ch 54 p.1360 ("The CA3 Region Is Important for Pattern
   Completion"): "Marr suggested in a second landmark paper in 1971 that the **recurrent excitatory connections of
   CA3 pyramidal cells** might underlie this phenomenon… reactivation of a subset of this stored cell assembly would
   be sufficient to activate the entire original neural ensemble… because of the **strong recurrent connections**."
   NMDA-dependent LTP at the CA3→CA3 recurrents is required (p.1360, the CA3-NMDA-knockout cue-reduction deficit).
   This is the project's catalog D.13 (validated single-seed, `validate_trisynaptic_loop.py`) and biology.md
   §"Memory: hippocampus and replay" ("CA3 — pattern completer… attractor states; partial cues retrieve full
   memories"; Marr 1971, McClelland 1995). A cleanup IS pattern completion: a noisy unbind `rec` is the partial cue;
   the nearest stored concept is the completed attractor.

2. **Winner-take-all action/category selection.** biology.md §"How the brain decides (action selection)" (Kandel 6e
   Ch 38 pp.932–960): the BG cascade + **striatal PV-FSI fast feedforward inhibition** (Tepper-2018; Bolam-2000)
   select ONE winner and suppress the others. The cleanup's "pick the single best concept, zero the rest" is exactly
   this WTA. The project's own NEF cleanup realizes the WTA as the neuron **rectification threshold** (off-target →
   0 spikes) — a feedforward threshold-WTA, not lateral; both are biological (Carandini-Heeger normalization /
   Rutishauser α>1 lateral WTA, both cited in `2026-06-05-spiking-cleanup-memory-literature-synthesis.md`).

## B.2 The phasor realization (literature-grounded)

**Threshold Phasor Associative Memory (TPAM)** — Frady & Sommer 2019, PNAS 116:18050–18059, "Robust computation
with rhythmic spike patterns" (arXiv:1901.07718). TPAM is the **phasor-native** version of the Hopfield/NEF cleanup:

- **Vocabulary in a complex weight matrix:** `W = S S*` where `S` is the `N×M` matrix of the `M` stored phasor
  codes as columns (here `M`=vocab, `N`=D). Componentwise `W_ij = Σ_m S_im^r S_jm^r e^{i(S_im^φ − S_jm^φ)}`, diagonal
  zeroed, normalized by N. (Frady-Sommer eq. for W; `ResonateFireTPAM.__init__`: `self.w = (s @ s.conj().T)/n`.)
- **Magnitude-gated phase-preserving threshold transfer (the cleanup nonlinearity):**
  `z_i(t+1) = [u_i/|u_i|] · H(|u_i| − Θ)` with `u = W z`. The phase `u/|u|` is kept; the Heaviside magnitude gate
  `H(|u|−Θ)` is the **WTA** — units whose recurrent drive is below threshold go silent. This is the *exact phasor
  analogue* of the NEF cleanup's "rectification = argmax discretizer": there the bias placed the real intercept so
  off-target→0 spikes; here the magnitude gate `Θ` placed between the groundable drive (≈unit) and the ungroundable
  drive (≪unit) does the same. **A graded phase readout would be a linear reconstructor and leak (the 0.91 cap the
  rate arc hit); the magnitude gate is the discretizer.**
- **The RF neuron IS this transfer in spikes** (Frady-Sommer §"phase-to-timing"): the RF neuron kicked by `u`
  spikes at the phase of `u` (timing within the cycle), and its spike/no-spike is the magnitude gate (a unit below
  `|Z|`-floor never crosses). **Magnitude-invariant phase readout** — the project's RF substrate docstring and
  `_primitive_check` (robustness_max_phase_err on a magnitude-noised kick) already validate this.
- **Capacity:** Frady-Sommer report TPAM **high capacity for sparse phasor patterns** (sparsity ~10%, N=400, M=100
  retrieved at high cosine; bits/synapse exceeds dense Hopfield, Fig 3B). At the composer's regime (V=16–320,
  D=64–2048, the *dense* codes are not sparse) capacity is comfortable: V≪N so the M=V attractors are well separated
  — this is the **easy** corner of TPAM (few, well-spread patterns), where the literature's harder sparse-capacity
  results are not the binding constraint.

Older phasor-store roots (capacity theory for the same `W=SS*`): **Noest 1988** ("Associative Memory in Sparse
Phasor Neural Networks", EPL 6(5):469; "Discrete-state phasor neural networks", Phys. Rev. A 38:2196) and
**Jankowski-Lozowski-Zurada 1996** ("Complex-valued multistate neural associative memory", IEEE TNN 7(6):1491). The
modern complex-Hopfield capacity (Q-state) scales toward Hopfield for small phase resolution Q and ∝ Q^−2 for large
Q — irrelevant here (continuous phase, few patterns).

## B.3 On-bridge spiking realization (reuse the cleared mechanism, adapted)

The bridge **already** supports the complex matvec `u = W z` (`_rf_advance_one`: `re += W_re@re − W_im@im`,
`im += W_re@im + W_im@re`). So the on-bridge TPAM cleanup is:

1. **Build once per codebook** (mirroring `enable_spiking_cleanup`'s persistent NEF bridge): compute the dense
   complex `W = S S*/D` from `self.concepts` (S columns = `exp(2πi·concept_phases)`), install via
   `bridge.rf_set_complex_weights([(i, j, W[i,j]) for nonzero])` — D=64..512 dense is `D²` ≈ 4k..262k entries, well
   within the sparse-matvec budget the substrate was built for.
2. **One cleanup pass = one TPAM iteration:** `rf_kick(z=exp(2πi·rec_phases))`, `rf_resonate_steps(period+8)`, read.
   The single matvec `W z` already concentrates onto the nearest stored phasor (TPAM's basin step). Read out the
   winner by the largest **overlap** `|S* z|` — itself a complex matvec, computable as a SECOND tiny RF read or, in
   the honest first build, a numpy argmax over `|S* z|` (this argmax is over a SUBSTRATE-COMPUTED `z`, not over the
   raw codebook — the cleanup compute has moved into spikes; the final index-selection argmax is the same residual
   the NEF cleanup keeps as `np.argmax(per_concept_firing)`).
3. **Multi-iteration settle (if one pass is not enough at high V):** loop {read phases → re-`rf_kick` → resonate} for
   k iterations — exactly what numpy `ResonateFireTPAM.settle` does (`z ← rf_resonate(W z)`), just driven through the
   bridge. The **annealed-threshold** and **separated familiarity-gate** variants (already in
   `resonate_fire_fhrr.py`, validated to clear the frozen 0.80 bar + abstention separation at loads {2,3,5}) carry
   over: the magnitude gate `Θ` is the bridge's `_rf_floor`; abstention (the no-confab moat) is the settle
   collapsing / familiarity `max phase_similarity < threshold` — biologically a novelty/familiarity signal, NOT a
   basin property (the numpy TPAM arc proved a pure attractor confabulates).

**Reuse map:** numpy `ResonateFireTPAM` (the algebra ceiling) ⇒ on-bridge by installing its `W` through
`rf_set_complex_weights` and running its settle loop through `rf_kick`/`rf_resonate_steps`. NO new sim/ code for the
single-pass version; the multi-pass settle is a Python loop over existing bridge calls.

## B.4 Smallest de-risk test

**`research/findings/raw/_bc_phasor_tpam_cleanup_probe.py`** (mirror the A de-risk methodology):
- Build an `RFPhasorComposer` (D=64, V=16, seed 42/43/44; the composer's default scale). Capture REAL noisy unbind
  `rec` phases from stored facts (drive the composer's actual `_unbind_phases`, the genuine noisy spiking est).
- **Oracle:** the current numpy `_cleanup` (phase-cosine argmax) — held constant as the deterministic reference.
- **Arm:** on-bridge TPAM — install `W=SS*/D`, `rf_kick(rec)`, `rf_resonate_steps`, winner = `argmax|S* z|`.
- **GATE (de-risk):** TPAM-cleanup winner == numpy-cleanup winner for every (fact, role) across seeds 42/43/44
  (parity, the A-arc bar). Report per-seed recovery + the settle's groundable-vs-ungroundable separation (abstention
  still separates). Smell-test it's from spikes: zeroing the kick / silencing the RF state collapses the readout.

This is **one composer, one bridge op already exercised by the composer's bind/unbind** — the cheapest possible
de-risk. Run on **GPU/CuPy** (the composer's bind is GPU-only at this op point, per the B-store de-risk note).

## B.5 Honest difficulty: **TRACTABLE.**

Strongest of the four reasons: (a) the mechanism (TPAM) is already coded in-repo and pre-registered-validated at the
frozen bar; (b) the substrate is **complex-native** — `rf_set_complex_weights` + `_rf_advance_one` ARE the TPAM
matvec, no sim/ edit for one pass; (c) V is tiny vs D (the easy TPAM corner); (d) the A arc already proved a placed
**magnitude/threshold** gate is the load-bearing discretizer, and TPAM's `H(|u|−Θ)` is that gate, phase-native.
Residual risks, all minor: the winner-selection `argmax|S*z|` is a small residual numpy step (identical in kind to
the NEF cleanup's retained `argmax(firing)`); multi-iteration settle, if needed, is a Python loop (no sim/ change but
more wall-clock); and the magnitude gate `Θ`/`_rf_floor` may need a one-line operating-point sweep (the numpy TPAM's
`theta` annealing schedule transfers). None of these is a re-architecture.

---

# CHEAT C — store the bound phasor composite in the substrate, retrieve in spikes

## C.1 Biology mechanism

**Memory is held in synaptic weights of a cell assembly** — the most directly-cited mechanism in the textbook:

- Kandel 6e Ch 54 p.1357 ("Memory Is Stored in Cell Assemblies"): Hebb's cell assembly = "a network of neurons…
  activated whenever a function is executed… Cells within an assembly are **bound together by excitatory synaptic
  connections strengthened at the time the memory was formed**." The memory IS the weights. Tonegawa/Liu 2012 (Fig
  54-11, p.1357–1359): reactivation of the tagged assembly is **sufficient** to recall the memory; the engram is the
  potentiated subset. This is the project's catalog D.14 engram API + the validated multitag retrieval.
- Kandel 6e Ch 54 p.1360 + Marr 1971: CA3 stores memories "as **changes in connections between active CA3 cells**";
  partial cue → full ensemble via the recurrent weights. A stored fact = an attractor in the recurrent weight matrix.
- biology.md §"Working memory needs special wiring" (Wang 1999/2002 NMDA bistability; Kandel Ch 60 pp.1330–1336): an
  alternative — hold the pattern in **persistent recurrent activity** (a latch) rather than static weights.

So three biological stores, all grounded: **(i) static feedforward weights** (Hebb cell-assembly = the Crawford
weight-store), **(ii) recurrent attractor weights** (Marr CA3 = the TPAM), **(iii) persistent activity** (Wang NMDA
latch). The first two store in WEIGHTS (the project's continual-learning premise); the third in ACTIVITY.

## C.2 The phasor realization

The rate composer cleared C with **(i)**: the bound real `(ON,OFF)` vector imprinted in static OUTPUT weights,
retrieved by firing a trigger (`2026-06-05-B-store-CLEARED.md`). The phasor adaptation holds a **complex/phase**
pattern instead of a real vector. Two routes, both biology-grounded and both fitting the RF substrate:

### Route C-A (recommended): complex weight-store (Crawford-store, phasor-typed)

The bound composite is a **phasor** `c = exp(2πi·composite_phases) ∈ ℂ^D`. Hold it in the **complex** output
synapses the substrate already has:
- A per-fact trigger population (the engram address, `n_trig` neurons). Each trigger neuron `i` projects to a
  D-neuron readout bank with **complex weight** `trigger_i → readout[k] = c[k]·w_gain` — installed via
  `rf_set_complex_weights` (the bridge stores `cp_rf_w_re[readout,trig]=Re(c)·w_gain`, `cp_rf_w_im=Im(c)·w_gain`).
  **The phasor composite IS the complex synaptic weight matrix** — the literal phasor analogue of the rate store
  (which held `bon[k]` in a real weight).
- **Retrieve in spikes:** `rf_kick` the trigger phasors (unit, phase 0), `rf_resonate_steps` → the complex matvec
  `u_readout = Σ_i W[readout,trig]·z_trig = (n_trig·w_gain)·c` drives each readout RF neuron at the **phase of c[k]**;
  the readout's spike timing reconstructs `composite_phases` (magnitude-invariant RF phase readout — the
  reconstruction does NOT depend on the `n_trig·w_gain` scale, which is the key advantage over the rate store, whose
  f-I rate reconstruction was only cosine ~0.975). Then unbind+cleanup on the retrieved `c′` as today.
- `self.kb` holds `(fact_dict, handle)` — `handle` = the trigger index — instead of `(fact_dict, numpy_phases)`.
  **The composite is in the substrate.**

Biology: the Hebb cell-assembly weight-store (Kandel p.1357), typed complex because the substrate's synapses are
complex (the RF/FHRR substrate, Frady-Sommer eq. [2] "the phase arithmetic is the complex synaptic integration
u = Σ W z" — weights are where operands live).

### Route C-B (alternative): phasor attractor store (TPAM as the KB)

Reuse the **same `W = S S*`** as cheat B, but `S` columns are the stored **composites** (one per fact), not the
vocabulary. Storing a fact = a Hebbian rank-1 update `W += c c*` (Marr CA3 recurrent imprint, Kandel p.1360). Recall
a fact = settle the TPAM from a partial cue (e.g. the agent role) → the full composite attractor → unbind. Biology:
Marr 1971 CA3 pattern completion; capacity Treves-Rolls / Frady-Sommer. **Honest caveat:** composites are DENSE
superpositions (M roles bundled), and TPAM's high capacity is for SPARSE phasors (Frady-Sommer Fig 3) — dense facts
are the harder corner; this route also requires a **fast-weight Hebbian path** (rank-1 `W` update at store time)
which the RF substrate's `rf_set_complex_weights` (full-rebuild, not incremental) does not yet expose. The project's
own (B)-options doc flagged exactly this ("dense-vs-sparse… reserve for partial-cue"). **Route C-A is simpler and
faithful; C-B is the partial-cue/pattern-completion upgrade for later.**

### The shared residual (same as the rate composer's, unchanged by phasors): the LINEAR glue

The rate composer's C clear left two numpy LINEAR ops (`bon += o` superposition; `onoff(bon−boff)` opponency) — the
"linear glue follow-on" (`2026-06-05-B-store-CLEARED.md`). The phasor composer's analogue is the **`_bundle`** (the
complex sum `Σ_l z_l`, lines 107–115) — but **this is ALREADY on the bridge** in the RF composer (unit complex
synapses summing the binds in the RF state; no opponency because the phasor algebra has no common mode — that was the
whole point of the RF recode). So the phasor composer is **ahead** of the rate composer here: there is no residual
numpy superposition/opponency. C is purely the **storage** question.

## C.3 On-bridge spiking realization

Route C-A, mirroring `_b_substrate_weight_store_probe.build_store_bridge`/`retrieve_bound`, phasor-typed:
1. `store(fact)`: compute `c = exp(2πi·_encode(fact))` (the bridge already produces `_encode` in spikes). Imprint:
   `rf_set_complex_weights([(readout_k, trig_i, c[k]·w_gain) for all i,k])` on a per-fact store bridge (or a shared
   bank addressed by a discrete trigger block — engram-style address per the synthesis). `self.kb.append((fact, handle))`.
2. query (`query_agent`/`query_patient`/`ask_yes_no`/`render_fact`): `_get_bound(handle)` → `rf_kick(trigger ON)`,
   `rf_resonate_steps`, `rf_read_phases(readout)` = `c′` in spikes → `_unbind_phases(c′,role)` + cleanup as today.
3. Opt-in flag `enable_spiking_memory` (exactly like the rate composer's), numpy `self.kb` path byte-unchanged when
   False. NO sim/ edits (`rf_set_complex_weights` is the existing API; per-fact store bridges, ~`n_trig + D·n_per`
   neurons/fact).

## C.4 Smallest de-risk test

**`research/findings/raw/_bc_phasor_weight_store_probe.py`** (mirror `_b_substrate_weight_store_probe.py`):
- `RFPhasorComposer` (D=64, V=16, seeds 42/43/44). Store 4–8 SVO facts → each fact's numpy `c = _encode(fact)`.
- For each fact: imprint `c` in the complex weight-store; RETRIEVE `c′` in spikes (trigger-driven readout phase
  readout); for each role compare `cleanup(_unbind_phases(c′,role))` vs `cleanup(_unbind_phases(c,role))`.
- **Cleanup held constant** (the deterministic numpy `_cleanup`, per the B-store methodology — so the STORE is what's
  tested, not the cleanup re-rolling). **GATE:** per-role recovery == 1.000 multi-seed; report the **phase
  reconstruction error** `max|angle(c′·conj(c))|/2π` (the phasor analogue of the rate store's recon cosine — expect
  it SMALL and, unlike the rate store's 0.975 cosine, scale-INDEPENDENT because RF phase readout is
  magnitude-invariant). Smell-test: silencing the trigger collapses `c′` to the OU/floor (a numpy passthrough would
  survive — the rate store's 145× collapse check, ported).

## C.5 Honest difficulty: **TRACTABLE for the store (C-A); the harder partial-cue (C-B) is PARTIAL.**

Route C-A is **tractable** and arguably *easier* than the rate composer's C clear: (a) the mechanism is the same
Crawford weight-store the project already cleared, only the weight TYPE changes real→complex, and the substrate's
weights are ALREADY complex (`rf_set_complex_weights`); (b) the phasor retrieval is **magnitude-invariant** (RF phase
readout), removing the rate store's only blemish (the f-I rate reconstruction's 0.975 cosine — the phasor read
should reconstruct phase essentially exactly, scale-free); (c) the linear-glue residual that the rate composer
deferred **does not exist** for the phasor composer (bundle is already on-bridge). The honest boundaries: (i) **scale**
— per-fact complex store bridges are `~n_trig + D·n_per` neurons/fact (D=2048 production → ~ tens of k neurons/fact,
a ~30-fact KB ~ 0.5M neurons; Crawford ran 2.5M for 117k facts, so in budget but not free); a shared-bank
consolidation (one readout bank, engram-addressed) is the optimization; (ii) **incremental store** — C-A rebuilds
the store-bridge weights per fact (fine; each fact is its own bridge); a single shared `W` with rank-1 updates (C-B)
needs a fast-weight path `rf_set_complex_weights` doesn't expose yet; (iii) **partial-cue / pattern-completion
recall** (query by agent alone, complete the fact) is route C-B = the TPAM-as-KB, which inherits the **dense-phasor
capacity** caveat (Frady-Sommer's high capacity is for sparse patterns; dense bundled facts are the harder corner) —
this is genuinely **PARTIAL/hard** and should be deferred, exactly as the project's own (B)-options doc concluded.

---

## Summary table

| | mechanism (biology) | on-bridge spiking realization | de-risk | difficulty |
|---|---|---|---|---|
| **B cleanup** | CA3 pattern completion (Kandel p.1360, Marr 1971) + striatal/BG WTA (biology.md, Kandel pp.932–960); phasor = **TPAM** (Frady-Sommer 2019; `W=SS*`, gate `z=u/\|u\|·H(\|u\|−Θ)`) | install `W=SS*/D` via `rf_set_complex_weights`; `rf_kick(rec)`+`rf_resonate_steps`; winner `argmax\|S*z\|`. **Already numpy-coded** (`ResonateFireTPAM`); substrate is complex-native. | `_bc_phasor_tpam_cleanup_probe.py`: TPAM winner == numpy-argmax winner, real noisy unbind, seeds 42/43/44, parity GATE | **TRACTABLE** |
| **C store** | Hebb cell assembly = memory in synaptic weights (Kandel p.1357, Tonegawa/Liu 2012); Marr CA3 recurrent store (p.1360); Wang NMDA latch (Ch 60) | **C-A:** bound phasor `c` in the **complex** output weights `trig→readout = c·w_gain` (`rf_set_complex_weights`); retrieve by firing trigger → readout **phase** readout (magnitude-invariant). `kb=(fact,handle)`. Bundle already on-bridge (no linear-glue residual). | `_bc_phasor_weight_store_probe.py`: per-role substrate-store unbind == numpy-store unbind, cleanup held constant, phase-recon error, trigger-silence collapse, seeds 42/43/44 | **TRACTABLE (C-A store)**; partial-cue C-B (TPAM-as-KB, dense-phasor capacity) **PARTIAL/deferred** |

## Strategic note (honest)

The phasor recode did NOT make these cheats harder — it made them **more substrate-native**. The rate composer fought
the substrate (real ON/OFF currents, an f-I rate reconstruction with a 0.975-cosine blemish, a deferred numpy
superposition/opponency). The phasor composer's cleared versions are **a single complex object** (`W=SS*`, the
Frady-Sommer TPAM) that (a) is already implemented in-repo and pre-registered-validated, (b) IS the bridge's existing
complex synapse matvec, (c) reconstructs magnitude-invariantly (no f-I blemish), and (d) has no linear-glue residual
(bundle is already spiking). The recommended sequencing mirrors the A arc that worked: **de-risk B first** (one
composer, one already-exercised bridge op, parity GATE), then **C-A** (the same complex-weight machinery), and leave
**C-B partial-cue completion** as the explicitly-deferred hard piece (dense-phasor capacity is the real wall, per
both Frady-Sommer and the project's prior (B) options conclusion).

## References

**Internal (cleared mechanisms + substrate):**
- `research/findings/2026-06-05-composer-cleanup-NEF-GO.md`, `research/findings/raw/_spiking_cleanup_nef.py` (rate B cleanup, NEF).
- `research/findings/2026-06-05-B-store-CLEARED.md`, `-B-substrate-store-fidelity-GO.md`,
  `research/findings/raw/_b_substrate_weight_store_probe.py`, `sim/bridge_memory.py` (rate C store, Crawford).
- `research/runners/resonate_fire_fhrr.py` (**`ResonateFireTPAM`** — the phasor TPAM, numpy reference, ALREADY VALIDATED
  at the frozen bar + abstention), `research/runners/spiking_phasor_fhrr.py` (phase_similarity, abstention cleanup).
- `sim/bridge.py` lines 4887–4975 (`rf_kick`/`rf_read_phases`/`rf_set_complex_weights`/`_rf_advance_one`/
  `rf_resonate_steps` — the **complex synapse matvec** `u=Wz` the substrate already runs).
- `docs/plans/2026-06-05-composer-B-substrate-held-memory-options.md` (the prior (B) options + dense-vs-sparse flag).

**Textbook (Kandel 6e, `references/textbooks/kandel-pns-6e/full-book.pdf`):**
- Ch 54 p.1357 "Memory Is Stored in Cell Assemblies" (Hebb; memory in strengthened excitatory synapses; Tonegawa/Liu).
- Ch 54 p.1360 "The CA3 Region Is Important for Pattern Completion" (Marr 1971; recurrent CA3 store; NMDA-LTP).
- Ch 54 pp.1359–1360 "The Dentate Gyrus Is Important for Pattern Separation" (Marr expansion recoding).
- Ch 38 pp.932–960 "The Basal Ganglia" (action-selection WTA) — via biology.md §"How the brain decides".
- Ch 60 pp.1330–1336 working memory (Wang NMDA bistability) — via biology.md §"Working memory needs special wiring".

**Literature:**
- Frady, E.P. & Sommer, F.T. (2019). "Robust computation with rhythmic spike patterns." *PNAS* 116(36):18050–18059.
  (arXiv:1901.07718; PMC6731666). **TPAM**: `W=SS*`, threshold transfer `z_i=[u_i/|u_i|]·H(|u_i|−Θ)`, resonate-and-fire
  phase-to-timing readout, high capacity for sparse phasors.
- Stewart, T.C., Tang, Y. & Eliasmith, C. (2011). "A biologically realistic cleanup memory: Autoassociation in
  spiking neurons." *Neural Networks* 24(7) (the Spaun cleanup; per-item population with placed firing threshold).
- Crawford, E., Gingerich, M. & Eliasmith, C. (2016). "Biologically Plausible, Human-Scale Knowledge Representation."
  *Cognitive Science* 40(4):782–821. (117,659 facts at D=512 in spiking associative memory.)
- Noest, A.J. (1988). "Associative Memory in Sparse Phasor Neural Networks." *EPL* 6(5):469; "Discrete-state phasor
  neural networks." *Phys. Rev. A* 38(4):2196. (Original sparse-phasor associative memory `W=SS*` capacity theory.)
- Jankowski, S., Lozowski, A. & Zurada, J.M. (1996). "Complex-valued multistate neural associative memory."
  *IEEE Trans. Neural Networks* 7(6):1491–1496. (Complex/phasor Hopfield, multistate capacity.)
- Frady, E.P., Kent, S.J., Olshausen, B.A. & Sommer, F.T. (2020). "Resonator Networks 1/2." *Neural Computation*
  32(12). (Phasor VSA factoring by interleaved bind + pattern completion — the broader phasor-cleanup context.)
- Marr, D. (1971). "Simple memory: a theory for archicortex." *Phil. Trans. R. Soc. Lond. B* 262(841):23–81.
  (CA3 recurrent autoassociator — the store-in-recurrent-weights mechanism.)
