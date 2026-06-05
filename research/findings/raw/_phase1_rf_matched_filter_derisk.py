"""Phase 1 (cheat B) FULL on-bridge de-risk, stage 1: the matched FILTER computed by the bridge's complex synapse
(the SAME op as unbind), NOT numpy. Install conj(codebook) complex synapses (rec -> concept), kick the rec phasor,
ONE matvec step -> each concept neuron's complex state = c_k = (S* rec); read its magnitude |c_k| =
sqrt(re^2+im^2) off cp_membrane_potential_v / cp_recovery_variable_u. GATE: argmax|c_k| (the substrate matched
filter) == the numpy argmax (Re-based cos cleanup) on the composer's REAL noisy unbinds, multi-seed. If GO, the
matched filter is on the substrate (only a membrane-magnitude readout is numpy); combine with the already-validated
Izhikevich WTA (the selection in spikes) for a fully-on-bridge cleanup.
"""
import numpy as np

from sim.backend import to_host
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge


def rf_matched_filter_winner(rec_phases, codebook, period, bridge):
    """|c_k| via the bridge complex-synapse matvec. concept k (index D+k) <- rec (0..D-1) weighted by conj(code_k)."""
    words = list(codebook)
    D = len(rec_phases)
    V = len(words)
    conj_codes = {w: np.conj(np.exp(2j * np.pi * codebook[w])) for w in words}
    conns = [(D + k, d, conj_codes[words[k]][d]) for k in range(V) for d in range(D)]
    bridge.rf_set_complex_weights(conns)
    kick = np.zeros(D + V, dtype=np.complex128)
    kick[:D] = np.exp(2j * np.pi * np.asarray(rec_phases))
    bridge.rf_kick(kick, period=period, lam=0.0)
    bridge.rf_resonate_steps(1)                      # ONE matvec -> concept[D+k] = c_k
    re = np.asarray(to_host(bridge.cp_membrane_potential_v)).astype(float)[D:D + V]
    im = np.asarray(to_host(bridge.cp_recovery_variable_u)).astype(float)[D:D + V]
    mag = np.sqrt(re * re + im * im)
    return words[int(np.argmax(mag))]


def numpy_argmax_cleanup(rec_phases, codebook):
    words = list(codebook)
    sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - codebook[w])))) for w in words]
    return words[int(np.argmax(sims))]


def run(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, period=200)
    comp.store("dog", "go", "north"); comp.store("cat", "run", "south"); comp.store("river", "look", "apple")
    V = len(comp.concepts)
    bridge = _build_rf_bridge(D + V, seed)
    n = n_match = 0
    for cph in [c for _, c in comp.kb]:
        for role in ("agent", "action", "patient"):
            rec = comp._unbind_phases(cph, role)
            w_np = numpy_argmax_cleanup(rec, comp.concepts)
            w_rf = rf_matched_filter_winner(rec, comp.concepts, comp.period, bridge)
            n += 1
            n_match += int(w_rf == w_np)
    return n_match, n


if __name__ == "__main__":
    for D in (256, 512):
        rows = []
        for seed in (42, 43, 44):
            m, nn = run(seed, D)
            rows.append((seed, m, nn))
        tot_m = sum(m for _, m, _ in rows); tot_n = sum(nn for _, _, nn in rows)
        print(f"D={D}: RF-matched-filter == numpy argmax {tot_m}/{tot_n}  "
              + "  ".join(f"s{s}:{m}/{nn}" for s, m, nn in rows), flush=True)
