# De-risk (B) — in-network SUPERPOSITION + ON/OFF OPPONENCY does NOT unbind at numpy parity → NEGATIVE — 2026-06-05

**Verdict: NEGATIVE.** The bound-fact **superposition** (`bon += o; boff += f` summed across roles) + **ON/OFF
opponency** (`onoff(bon − boff)`) — the last two numpy ops in `CoreSimComposer.bind_fact`'s compute path — do
**not** reach numpy parity when computed IN-NETWORK (spiking) via a shared accumulator with lateral-inhibition
opponency. Across seeds 42/43/44 at the production proj_dim=800 regime, the in-network-built bound vector unbinds
to the same filler as the numpy-built bound vector in only **46.2% / 69.2% / 69.2%** of roles (min **0.462**, mean
**0.615**) — far below the **1.000** GATE. The root cause is fully diagnosed and **fundamental to conductance-based
spiking opponency**, not a tuning miss: the **signed difference `bon − boff` that the unbind consumes is destroyed**
(signed cosine **0.41**). The documented next idea — a **gated NEF integrator** — is now the path forward, exactly as
this de-risk's NEGATIVE branch anticipated.

## The crux question (the GATE)
(A) cleared the READOUT shortcut (numpy `argmax` cleanup → spiking NEF cleanup,
`2026-06-05-composer-cleanup-NEF-GO.md`); (B) cleared the STORE shortcut (numpy `kb` list → Crawford-style
spiking weight-store, `2026-06-05-B-substrate-store-fidelity-GO.md`). Those two GOs left exactly **two linear
numpy ops** in the compute path (the audit's "linear inter-phase ops", n=111):
```
for each role:  o, f = self._op(role, fon, foff)   # SPIKING coincidence bind: o=rates(A)+rates(B), f=rates(C)+rates(D)
                bon += o; boff += f                 # numpy SUPERPOSITION (rate-sum across roles)
return onoff(bon - boff)                            # numpy OPPONENCY (ON/OFF rectified difference)
```
The host read of `(o, f)` per role is the numpy boundary. **Can the superposition + opponency be done in-network
(spiking) so the resulting bound vector unbinds at numpy parity?**

## Result (the de-risk GATE) — NEGATIVE
`research/findings/raw/_b_innetwork_superposition_probe.py`, proj_dim=800 (the production regime), numpy
deterministic cleanup held constant (to isolate the SUPERPOSITION), CuPy / RTX 3090. Best tuned-bank operating
point found by the sweep (mutual ON/OFF lateral inhibition, `w_acc=500, w_opp=200, E_inh=−80`):

| seed | recovery (in-network == numpy) | recon cosine `(bon′,boff′)·(bon,boff)` | **signed cosine** `(bon′−boff′)·(bon−boff)` | upper bound (in-net superposition + **numpy** opponency) |
|---|---|---|---|---|
| 42 | 6/13 = **0.462** | 0.603 | 0.373 | 0.462 |
| 43 | 9/13 = **0.692** | 0.629 | 0.430 | 0.692 |
| 44 | 9/13 = **0.692** | 0.630 | 0.430 | 0.769 |
| **mean** | **0.615** (min 0.462) | 0.620 | **0.411** | **0.641** |

The GATE is per-seed recovery == 1.000. **It fails decisively** (min 0.462, mean 0.615).

## Root cause — error amplification in subtracting two strongly-correlated channels
The diagnosis is unambiguous and is the load-bearing finding. The accumulator reconstructs each channel
**faithfully** — the in-network superposition `acc_on ≈ bon`, `acc_off ≈ boff` at **per-channel cosine 0.97**. But
the unbind does not consume the channels separately; it consumes their **signed difference** (`_scale_to_current`
then `_op` reads `(A+B)−(C+D)` driven by the scaled `bon′/boff′`). And `bon` and `boff` are **strongly correlated**
(`cos(o,f) ≈ 0.89` — the fill-magnitude envelope is a large common mode that cancels in `o − f`). Subtracting two
cos-0.97 reconstructions of correlated vectors **amplifies the 3% per-channel read noise** relative to the small
true difference:

| quantity | in-network vs numpy cosine |
|---|---|
| `acc_on` vs `bon` (ON channel) | **0.97** |
| `acc_off` vs `boff` (OFF channel) | **0.97** |
| **`acc_on − acc_off` vs `bon − boff` (SIGNED)** | **0.40** |

So the signed vector the unbind needs is recovered at only **cos 0.40** even though each channel is at 0.97. This
is why even **PERFECT numpy opponency** applied to the in-network superposition (the "upper bound" column above)
recovers only **0.64** — the superposition read's small-signal subtraction is the **root blocker**, before any
spiking opponency is even attempted.

### Why the spiking opponency cannot fix it
The opponency must orthogonalize the channels (rectify the signed difference) **before** the lossy f-I read.
Conductance-based inhibition cannot do the required precise small-signal subtraction, because `g_i·(E_inh − V)`
is **divisive/shunting** (voltage-dependent driving force), not a clean linear `g_e − g_i`. Every opponency
mechanism tried plateaus or collapses (max signed cosine ≈ 0.5, vs the ≈ 0.99 per-channel fidelity needed to
preserve a 0.89-common-mode difference):

| opponency mechanism | best signed cosine | parity |
|---|---|---|
| mutual ON/OFF lateral inhibition (the task's specified mechanism) | 0.50 | 6/9 |
| opponent inhibitory-relay interneurons (`A,B→relay→acc_off`, etc.) | 0.52 | 2/3 |
| per-dimension **pooled** relay (n_per=8, within-dim all-to-all) | negative (over-inhibits) | — |
| **direct** inhibitory-bank conductance subtraction (`C,D` inhibitory → `acc_on`) — the most faithful (common-mode cancels in the neuron's conductance) | 0.46 (vs `max(bon−boff,0)`) | — |
| late-window steady-state read (discard transient) | worse | — |

~30 configurations across `w_acc / w_opp / E_inh / n_per / read-window`: the signed-difference fidelity never
clears ≈ 0.5. More inhibition drives it toward 0 or negative (both populations silenced / anti-correlated). The
wall is the conductance-shunting nonlinearity meeting a 0.89-common-mode signal, not a missing gain setting.

## What DOES work — the superposition (spike-sum) is genuine
The accumulator genuinely **sums across roles in spikes** (the SUPERPOSITION half of the task is real). Smell-test
(pinned as `tests/test_b_innetwork_superposition.py::test_innetwork_accumulator_sums_in_spikes_not_passthrough`):
a 2-role fact's accumulated `acc` read is **larger than either single role's read** (neither role dominates) and
**tracks the sum** of the two single-role reads (within a spiking tolerance band). It is not a numpy passthrough
of one role. The accumulator is wired `A[k],B[k]→acc_on[k]`, `C[k],D[k]→acc_off[k]` (identity weights), and across
consecutive per-role bind windows (the accumulator is NOT reset between roles) it integrates the superposition.
**It is the OPPONENCY (the signed subtraction), not the superposition, that fails.**

## The mechanism built (NO `sim/` edits; the BIND machinery REUSED BY IMPORT)
A standalone 10·D-neuron `SimulationBridge` (`build_bind_accumulator_bridge`):
- the **8·D coincidence circuit** (role_ON/OFF + fill_ON/OFF → A/B/C/D) is wired by **reusing**
  `core_sim_composition.build_bind_bridge(shared_bridge=…)` — the ±1 Hadamard is **not re-implemented** (the
  shared-bridge path accumulates the `"bind"` population onto the probe bridge and returns the bank index arrays;
  on a Hebbian-OFF bridge its gate-zero is a harmless no-op). **`build_bind_bridge` was REUSED, not replicated.**
- two accumulator regions `acc_on[D] @ [8D,9D)`, `acc_off[D] @ [9D,10D)`, marked **inhibitory-trait** (so their
  outgoing synapses route through `g_i` — the routing keys on the **presynaptic inhibitory trait**, per the A
  divnorm finding `_spiking_cleanup_divnorm_probe.py` / bridge.py 5046–5070, **not** the conn_type string).
- accumulator wiring (fixed): `A,B → acc_on`, `C,D → acc_off` (identity, `w_acc`); opponency `acc_on −| acc_off`,
  `acc_off −| acc_on` (mutual lateral inhibition, `w_opp`). The accumulator wiring is added to the bridge's union
  via `merge_population_into_shared_bridge`.

### Saturation handling
2–4 superposed role⊗filler patterns saturate the accumulator Izhikevich f-I if driven too hard (a single bank
spike at very high `w_acc` fires `acc` regardless of rate → binarization, the relative magnitudes lost). The
sweep tuned `w_acc` to keep the accumulated sum in the **responsive (sub-saturation) band**: at `w_acc=500` the
`acc` rates are graded (max ≈ 0.03–0.11, not pinned at the f-I ceiling) and the per-channel superposition stays at
cos 0.97. Saturation was therefore **handled** (it is not the cause of the NEGATIVE) — the cause is the
post-read subtraction, which no drive-gain tuning addresses. (The research note flagged a gated NEF integrator as
the heavier alternative if a tuned bank did not suffice; it did not suffice.)

## Methodology — the cleanup held constant + the right comparison reference
- The cleanup is the **deterministic numpy argmax** oracle, held constant across both arms, so the **superposition**
  is what is tested (same rationale as the B-store de-risk: the stochastic NEF cleanup does not even agree with
  itself and would mask the store/superposition signal).
- The de-risk reports **three** numbers per seed: the recon cosine (per-channel fidelity, high, 0.62 here because
  it is over the post-opponency vector); the **signed cosine** (the diagnostic that exposes the wall, 0.41); and
  the **upper bound** (numpy opponency on the in-network superposition, 0.64) — which proves the superposition
  read is the root blocker, isolating it from the spiking opponency.

## Honest scope / boundaries
- **The two prior B pieces (cleanup, store) remain GO and unaffected.** This is the *third* piece (the in-network
  superposition+opponency); it is the one that does NOT clear. The composer's `bind_fact` superposition/opponency
  therefore **stay numpy for now** (the disclosed boundary is unchanged).
- **GPU-only.** On the numpy backend the composer's spiking bind is degenerate (all-zero bound vector), so the
  de-risk is a CuPy/GPU result (per the GPU-for-real-runs mandate); the regression test skips on a degenerate bind.
- **The accumulator's SUPERPOSITION is genuine and reusable** as a substrate piece — a future fix only needs to
  replace the OPPONENCY (the signed subtraction), keeping the spike-summing accumulator.

## Next (the documented next idea after this NEGATIVE)
A **gated NEF integrator** for the opponency: represent the signed value `s = bon − boff` in a recurrent linear
integrator population whose drive is `(A+B) − (C+D)`, so the **subtraction happens in the represented value before
the lossy f-I read** (the NEF decode is linear in the represented quantity, sidestepping the divisive-shunting
wall). The accumulator built here (the spike-summing superposition) is the front half; the NEF integrator replaces
the lateral-inhibition opponency. This is a heavier build (recurrent integrator + NEF encode/decode) than the
tuned bank, consistent with the research note.

## Artifacts
- Probe: `research/findings/raw/_b_innetwork_superposition_probe.py` (reuses `build_bind_bridge` by import; adds
  the accumulator + opponency; `bind_fact_in_network`; the signed-cosine + upper-bound diagnostics)
- Results: `research/findings/raw/_b_innetwork_superposition.json` (canonical 3-seed, proj_dim=800)
- Test: `tests/test_b_innetwork_superposition.py` (pins the NEGATIVE boundary + the genuine spike-sum smell-test;
  the boundary assertion fires if a future mechanism reaches parity → the signal to flip this finding to GO)
- Backend: CuPy / RTX 3090. **NO `sim/` edits.** `build_bind_bridge` **reused by import** (not replicated).
