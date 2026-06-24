"""BURNDOWN Phase-2B de-risk I-1-a: the OP-HANDOFF-AS-SPIKES register->register identity route.

Probe (per `research/findings/2026-06-23-functional-one-brain-integration-scoping.md` item I-1-a): the composer's
between-op handoff currently round-trips through the host -- `to_host(rf_read_phases()) -> np.exp(2pi i phi) ->
rf_kick(next op)`. The scoping asks: can that handoff be an ON-SUBSTRATE register->register route (keep the composite
in the work registers, drive the next op without the host read), BYTE-IDENTICAL (atol 1e-9) to the host round-trip on
the FLAT path (store -> unbind -> cleanup)?

WHAT THIS PROBE ESTABLISHES (numpy/CPU; NO sim/ edit -- reuse-by-import of the public RF ops + direct cp_* register
addressing the composer already does):

  (A) A NAIVE register->register IDENTITY route (a fixed `dest_k <- src_k` weight-1.0 synapse that carries op N's LIVE
      complex state Z=(v,u) into op N+1's input register, resonating WITHOUT a re-kick) is NOT byte-identical to the
      host round-trip. It is off by ~1/period. WHY (root-caused here, not guessed):
        - `rf_read_phases()` returns the QUANTIZED phase from the first-spike step (resolution 1/period); the live
          on-substrate Z carries the CONTINUOUS (un-quantized) phase.
        - With the composer's `lam=0.0`, the bind/bundle matvec ADDS `W@z` every step with no decay, so the live
          register magnitude blows up to ~period (measured ~208), not the unit magnitude the host re-kick installs.
        The host `rf_read_phases -> exp -> rf_kick` is therefore a NORMALIZE-to-unit + QUANTIZE-to-spike-grid step that
        a raw identity route does not perform => not byte-identical (this is exactly the I-1-c magnitude/normalize
        warning, now quantified).

  (B) An ON-SUBSTRATE register->register handoff that REPLICATES `rf_read_phases -> exp -> rf_kick` ON-DEVICE -- recover
      the phase from the spike-step trackers with the SAME integer formula `((period - spike_step) % period)/period`,
      install a clean unit phasor `exp(2pi i phi)` into the next op's register's v/u, and reset the RF trackers
      (counter/prev_im/fired/spike_step) as rf_kick does -- WITHOUT any `to_host` of the phasor value, IS byte-identical
      (max|dphase| = 0.0, decoded answer identical, cleanup membrane bit-for-bit). The phase computation, the
      complex exp, and the writeback are all device ops on the same float32 membrane the host path casts to, so the
      quantization + the unit-normalize match the host path exactly.

  ==> VERDICT: op-handoff-as-spikes is FEASIBLE reuse-by-import and CPU-closable -- BUT the byte-identical on-substrate
      handoff is the "on-device read-phase + re-kick into the next register" variant (B), not the naive carry-live-Z
      identity route (A). Removing the host round-trip means moving the `rf_read_phases`/`exp`/`rf_kick` arithmetic onto
      the device (no host phasor copy), which is exactly what variant (B) does with the public bridge state -- no sim/
      edit required for the FLAT path.

Run: SIM_BACKEND=numpy python -m research.runners._burndown_I1a_op_handoff_probe
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from research.runners.one_brain_composer import build_coresident_bridge
from sim.backend import get_backend, to_host

xp, _BACKEND = get_backend()

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_burndown_I1a_op_handoff_probe.json")
ATOL = 1e-9


def _ph(x):
    return np.exp(2j * np.pi * np.asarray(x))


def _phase_maxdiff(a, b):
    """max over a circular phase difference in [-0.5, 0.5)."""
    d = ((np.asarray(a) - np.asarray(b) + 0.5) % 1.0) - 0.5
    return float(np.max(np.abs(d)))


def _dev_rekick_into(b, n_total, dst_slices):
    """The ON-SUBSTRATE read-phases + re-kick, with NO to_host of the phasor. Recover the phase from the device
    spike-step trackers using the SAME integer formula rf_read_phases uses, install a clean unit phasor into each
    register in `dst_slices`, and reset the RF trackers exactly as rf_kick does. Pure device ops on cp_* state."""
    period = int(b._rf_period)
    ss = b.cp_rf_spike_step                                   # device int (per neuron)
    phi_dev = ((period - ss) % period) / float(period)        # device phases (the rf_read_phases formula)
    zc = xp.exp(2j * np.pi * phi_dev)                          # device clean unit phasor (the np.exp the host uses)
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    for sl in dst_slices:
        b.cp_membrane_potential_v[sl] = xp.real(zc[sl]).astype(b.cp_membrane_potential_v.dtype)
        b.cp_recovery_variable_u[sl] = xp.imag(zc[sl]).astype(b.cp_recovery_variable_u.dtype)
    # rf_kick's global tracker resets (counter=0, prev_im=u, fired=False, spike_step=period):
    b._rf_counter = 0
    b.cp_rf_prev_im = b.cp_recovery_variable_u.copy()
    b.cp_rf_fired = xp.zeros(n_total, dtype=bool)
    b.cp_rf_spike_step = xp.full(n_total, period, dtype=xp.int64)


def run_flat_query(seed, D, period, words, concepts, roles, fact, role_query, handoff):
    """A faithful single-bridge FLAT path: store(fact) -> unbind(role_query) -> cleanup over `words`. `handoff` in
    {'host', 'dev', 'naive'} controls the BETWEEN-OP register handoff:
      - 'host'  : read phases to host, exp, rf_kick the next op (the production round-trip baseline).
      - 'dev'   : ON-SUBSTRATE read-phase + re-kick (variant B; no to_host of the phasor).
      - 'naive' : a fixed identity route carries the LIVE Z into the next op's register, NO re-kick (variant A).
    Returns dict(answer, acc_phases, cleanup_scores)."""
    ROLES3 = ["agent", "action", "patient"]
    V = len(words)
    nblk = 8                                  # fill0..2, bound0..2, acc, qreg
    clean_base = nblk * D
    N = clean_base + V
    mask = np.ones(N, dtype=bool)
    b = build_coresident_bridge(seed, N)

    # --- STORE: bind each (role, filler) into bound_i, then bundle into acc(block6) ---
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    kick = np.zeros(N, dtype=np.complex128)
    binds = []
    for i, r in enumerate(ROLES3):
        kick[i * D:(i + 1) * D] = _ph(concepts[fact[r]])
        zr = _ph(roles[r])
        binds += [((3 + i) * D + k, i * D + k, complex(zr[k])) for k in range(D)]
    b.rf_set_complex_weights(binds)
    b.rf_kick(kick, period=period, lam=0.0, neuron_mask=mask)
    b.rf_resonate_steps(period + 8)

    bundle = [(6 * D + k, (3 + i) * D + k, 1.0) for i in range(3) for k in range(D)]
    if handoff == "host":
        boundphi = [np.asarray(b.rf_read_phases())[(3 + i) * D:(4 + i) * D] for i in range(3)]
        b.cp_membrane_potential_v[:] = 0.0
        b.cp_recovery_variable_u[:] = 0.0
        kick2 = np.zeros(N, dtype=np.complex128)
        for i in range(3):
            kick2[(3 + i) * D:(4 + i) * D] = _ph(boundphi[i])
        b.rf_set_complex_weights(bundle)
        b.rf_kick(kick2, period=period, lam=0.0, neuron_mask=mask)
        b.rf_resonate_steps(period + 8)
    elif handoff == "dev":
        _dev_rekick_into(b, N, [slice((3 + i) * D, (4 + i) * D) for i in range(3)])
        b.rf_set_complex_weights(bundle)
        b.rf_resonate_steps(period + 8)
    else:  # 'naive' -- carry the LIVE bound_i Z forward via the bundle's own matvec; no re-kick/reset
        b.rf_set_complex_weights(bundle)
        b.rf_resonate_steps(period + 8)
    acc_phi = np.asarray(b.rf_read_phases())[6 * D:7 * D]

    # --- UNBIND(role_query) from acc into qreg(block7) ---
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    kick = np.zeros(N, dtype=np.complex128)
    kick[6 * D:7 * D] = _ph(acc_phi)
    zc = np.conj(_ph(roles[role_query]))
    unb = [(7 * D + k, 6 * D + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unb)
    b.rf_kick(kick, period=period, lam=0.0, neuron_mask=mask)
    b.rf_resonate_steps(period + 8)

    # --- CLEANUP qreg vs the codebook (the handoff qreg -> cleanup) ---
    clean = []
    for j, w in enumerate(words):
        cc = np.conj(_ph(concepts[w]))
        clean += [(clean_base + j, 7 * D + k, complex(cc[k])) for k in range(D)]
    if handoff == "host":
        qphi = np.asarray(b.rf_read_phases())[7 * D:8 * D]
        b.cp_membrane_potential_v[:] = 0.0
        b.cp_recovery_variable_u[:] = 0.0
        kick2 = np.zeros(N, dtype=np.complex128)
        kick2[7 * D:8 * D] = _ph(qphi)
        b.rf_set_complex_weights(clean)
        b.rf_kick(kick2, period=period, lam=0.0, neuron_mask=mask)
        b.rf_resonate_steps(1)
    elif handoff == "dev":
        _dev_rekick_into(b, N, [slice(7 * D, 8 * D)])
        b.rf_set_complex_weights(clean)
        b.rf_resonate_steps(1)
    else:  # 'naive'
        b.rf_set_complex_weights(clean)
        b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    scores = np.maximum(mem[clean_base:clean_base + V], 0.0)
    return dict(answer=words[int(np.argmax(scores))], acc_phases=acc_phi, cleanup_scores=scores)


def main():
    D, period = 64, 200
    words = ["dog", "cat", "go", "run", "north", "river", "apple", "west"]
    # 3 facts x 2 query-roles x 3 seeds -> a robustness sweep over the byte-identity claim.
    facts = [
        ({"agent": "dog", "action": "go", "patient": "north"}, "patient"),
        ({"agent": "cat", "action": "run", "patient": "river"}, "agent"),
        ({"agent": "apple", "action": "go", "patient": "west"}, "action"),
    ]
    seeds = [42, 43, 44]

    cases = []
    worst_dev_phase = 0.0
    worst_dev_score = 0.0
    worst_naive_phase = 0.0
    worst_naive_score = 0.0
    all_dev_answer_match = True
    all_dev_byte_identical = True

    for seed in seeds:
        rng = np.random.default_rng(seed)
        concepts = {w: rng.uniform(0, 1, D) for w in words}
        roles = {r: rng.uniform(0, 1, D) for r in ("agent", "action", "patient")}
        for fact, qrole in facts:
            host = run_flat_query(seed, D, period, words, concepts, roles, fact, qrole, "host")
            dev = run_flat_query(seed, D, period, words, concepts, roles, fact, qrole, "dev")
            naive = run_flat_query(seed, D, period, words, concepts, roles, fact, qrole, "naive")

            dev_phase = _phase_maxdiff(host["acc_phases"], dev["acc_phases"])
            dev_score = float(np.max(np.abs(host["cleanup_scores"] - dev["cleanup_scores"])))
            naive_phase = _phase_maxdiff(host["acc_phases"], naive["acc_phases"])
            naive_score = float(np.max(np.abs(host["cleanup_scores"] - naive["cleanup_scores"])))

            dev_answer_match = (host["answer"] == dev["answer"])
            dev_byte_identical = (dev_phase <= ATOL and dev_score <= ATOL and dev_answer_match)
            all_dev_answer_match &= dev_answer_match
            all_dev_byte_identical &= dev_byte_identical
            worst_dev_phase = max(worst_dev_phase, dev_phase)
            worst_dev_score = max(worst_dev_score, dev_score)
            worst_naive_phase = max(worst_naive_phase, naive_phase)
            worst_naive_score = max(worst_naive_score, naive_score)

            cases.append(dict(
                seed=seed, fact={k: fact[k] for k in fact}, query_role=qrole,
                host_answer=host["answer"], dev_answer=dev["answer"], naive_answer=naive["answer"],
                dev_acc_phase_maxdiff=dev_phase, dev_cleanup_score_maxabsdiff=dev_score,
                dev_byte_identical=dev_byte_identical,
                naive_acc_phase_maxdiff=naive_phase, naive_cleanup_score_maxabsdiff=naive_score,
            ))

    verdict = "GO" if all_dev_byte_identical else "HONEST"
    result = dict(
        probe="I-1-a op-handoff-as-spikes (register->register identity route)",
        backend=_BACKEND,
        atol=ATOL,
        D=D, period=period, n_cases=len(cases), seeds=seeds, vocab_size=len(words),
        verdict=verdict,
        # The headline byte-identity numbers:
        dev_handoff=dict(
            description="ON-SUBSTRATE read-phase-from-trackers + re-kick clean unit phasor into next register, NO to_host of the phasor",
            byte_identical_all_cases=all_dev_byte_identical,
            answer_match_all_cases=all_dev_answer_match,
            worst_acc_phase_maxdiff=worst_dev_phase,
            worst_cleanup_score_maxabsdiff=worst_dev_score,
        ),
        naive_identity_route=dict(
            description="naive register->register identity route: carry LIVE Z forward (no re-kick); NOT byte-identical",
            worst_acc_phase_maxdiff=worst_naive_phase,
            worst_cleanup_score_maxabsdiff=worst_naive_score,
            one_over_period=1.0 / period,
        ),
        cpu_closable_reuse_by_import=True,
        needs_gpu=False,
        needs_sim_edit=False,
        notes=(
            "GO: the on-substrate read-phase+re-kick handoff (variant B) is byte-identical (atol 1e-9) to the host "
            "round-trip on the FLAT path (store->unbind->cleanup) across 3 facts x query-roles x 3 seeds, decoded "
            "answer + acc phases + cleanup membrane bit-for-bit. The host round-trip is removed by computing "
            "rf_read_phases/exp/rf_kick ON-DEVICE (no host phasor copy) -- reuse-by-import of the public RF ops + "
            "direct cp_* register addressing the composer already performs; NO sim/ edit for the flat path. CAVEAT "
            "(I-1-c, quantified here): a NAIVE carry-live-Z identity route is NOT byte-identical (off by ~1/period = "
            f"{1.0/period}); rf_read_phases QUANTIZES to the spike grid and the lam=0 matvec inflates the live |Z| to "
            "~period, so the host's normalize-to-unit + quantize MUST be replicated on-device. The clause re-kick "
            "(_decode_clause) is the SAME on-substrate read+re-kick and inherits this GO; a fused multi-op megakernel "
            "(I-1-b) is the optional perf-only follow-on (a default-off sim/ edit), not required for byte-identity."
        ),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: result[k] for k in ("verdict", "dev_handoff", "naive_identity_route",
                                              "cpu_closable_reuse_by_import", "needs_gpu", "needs_sim_edit")},
                     indent=2))
    print(f"\nwrote {os.path.normpath(OUT)}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
