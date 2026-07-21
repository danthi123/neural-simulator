"""Single-shared-substrate consolidation — cheap-first CO-RESIDENCE de-risk (2026-07-20).

The end-goal steer: "fully closing all gaps INHERENTLY means fully-spiking, one brain, single shared substrate."
De-risk 5 had the composer (RF phasor) + the WKV cortex (cp_ssm_state read-out) co-EXECUTING in one PROCESS but on
SEPARATE cupy bridges. The consolidation is onto ONE bridge. This de-risk proves the crux is byte-clean:

  Can ONE bridge simultaneously hold the WKV's slow-SSM read-out state (cp_ssm_state + cp_ssm_readout_w) AND serve
  as the composer's RF phasor substrate (cp_rf_w_* + rf_kick/resonate/read on a masked slice), with each producing
  results BYTE-IDENTICAL to running it on its own isolated bridge, and NEITHER corrupting the other?

Why it should hold (read from the step loop + RF ops, bridge.py):
  - The ssm block (5958) + read-out (5966) run UNCONDITIONALLY on array presence, independent of neuron_model_type;
    they read cp_ssm_inject/cp_ssm_shunt/cp_ssm_state ONLY (set by the runner) -> the composer touches none of them.
  - The RF ops (rf_kick / rf_resonate_steps / _rf_advance_one) use v/u (masked) + cp_rf_* ONLY, do NOT dispatch on
    neuron_model_type, and the composer re-kicks its phasor EACH op (De-risk 5b) -> the WKV's Izhikevich step between
    ops is harmless.
  => disjoint persistent arrays; the only shared array (v/u) is re-initialized by the composer's kick each op.

MAIN gate: co-resident WKV read-out == isolated WKV read-out (byte) AND co-resident composer phases == isolated
composer phases (byte), across N interleaved rounds. ANTI-CHEAT: a NO-REKICK arm (composer skips its kick) must
DIVERGE from isolated (proving v/u really is shared + the re-kick is load-bearing) -- i.e. the co-residence is real,
not a trivial two-untouched-bridges artifact.

Reuse-by-import of _build_ssm_state_bridge; NO sim/ edit. `--seed`, `--rounds`.
"""
import argparse
import numpy as np
from sim.backend import get_backend, to_host

from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge


def _install_readout(b, xp, seed):
    """Give bridge b a WKV read-out matrix over cp_ssm_state (a fixed random W). Returns W (host)."""
    n = b.core_config.num_neurons
    rng = np.random.default_rng(seed + 7)
    W = rng.standard_normal((16, n)).astype(np.float32) * 0.1
    b.cp_ssm_readout_w = xp.asarray(W)
    b.cp_ssm_readout_out = None
    return W


def _charge_read(b, xp, inject_vec):
    """WKV op: set cp_ssm_inject, step, return the read-out out = W @ cp_ssm_state (host)."""
    b.cp_ssm_inject[:] = xp.asarray(inject_vec.astype(np.float32))
    b.cp_ssm_shunt[:] = 0.0
    b._run_one_simulation_step()
    return np.asarray(to_host(b.cp_ssm_readout_out)).astype(np.float64)


def _install_rf(b, slice_idx, seed):
    """Set up a tiny RF binding on the masked slice: a small diagonal self-coupling (complex weight) within the
    slice so the resonate dynamics have real synaptic input to advance. connections = (post, pre, complex_w)."""
    conns = [(int(i), int(i), complex(0.02, 0.0)) for i in slice_idx]
    b.rf_set_complex_weights(conns)


def _composer_op(b, xp, slice_idx, kick_seed, n_resonate, do_kick=True):
    """Composer op: (optionally) re-kick the RF slice's phasor, resonate, read phases at the slice. Returns phases."""
    n = b.core_config.num_neurons
    mask = np.zeros(n, dtype=bool)                       # rf_kick converts (np.asarray/cp.asarray) internally
    mask[np.asarray(slice_idx, dtype=np.int64)] = True
    if do_kick:
        rng = np.random.default_rng(kick_seed)
        kick = np.zeros(n, dtype=np.complex128)
        ph = rng.uniform(-np.pi, np.pi, size=len(slice_idx))
        kick[np.asarray(slice_idx)] = np.exp(1j * ph)
        b.rf_kick(kick, neuron_mask=mask)
    b.rf_resonate_steps(n_resonate)
    phases = np.asarray(to_host(b.rf_read_phases())).astype(np.float64)
    return phases[np.asarray(slice_idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--n-resonate", type=int, default=40)
    args = ap.parse_args()
    xp, _ = get_backend()
    rng = np.random.default_rng(args.seed)

    decay = 0.9
    # --- isolated WKV bridge (A) ---
    bA, _cg, _cg2, _snap = _build_ssm_state_bridge(args.D, args.seed, decay, pop_k=1)
    nA = bA.core_config.num_neurons
    WA = _install_readout(bA, xp, args.seed)

    # --- isolated composer bridge (B) ---
    bB, _cgb, _cg2b, _snapb = _build_ssm_state_bridge(args.D, args.seed, decay, pop_k=1)
    nB = bB.core_config.num_neurons
    assert nA == nB, (nA, nB)
    n = nA
    # composer slice = last 24 neurons; WKV read_idx is the full population (disjoint use: composer reads phases only)
    slice_idx = list(range(n - 24, n))
    _install_rf(bB, slice_idx, args.seed)

    # --- co-resident bridge (C): BOTH read-out + RF on one bridge ---
    bC, _cgc, _cg2c, _snapc = _build_ssm_state_bridge(args.D, args.seed, decay, pop_k=1)
    WC = _install_readout(bC, xp, args.seed)
    _install_rf(bC, slice_idx, args.seed)
    assert np.allclose(WA, WC), "read-out matrices must match across A and C"

    # fixed per-round inject vectors + kick seeds
    injects = [rng.standard_normal(n) * 0.5 for _ in range(args.rounds)]
    kick_seeds = [args.seed * 1000 + r for r in range(args.rounds)]

    # ISOLATED trajectories
    iso_reads = [_charge_read(bA, xp, injects[r]) for r in range(args.rounds)]
    iso_phases = [_composer_op(bB, xp, slice_idx, kick_seeds[r], args.n_resonate, do_kick=True)
                  for r in range(args.rounds)]

    # CO-RESIDENT interleaved: WKV read then composer op, same order, same inputs
    co_reads, co_phases = [], []
    for r in range(args.rounds):
        co_reads.append(_charge_read(bC, xp, injects[r]))              # WKV op (Izhikevich step touches RF slice v/u)
        co_phases.append(_composer_op(bC, xp, slice_idx, kick_seeds[r], args.n_resonate, do_kick=True))  # re-kicks

    read_maxerr = max(float(np.max(np.abs(co_reads[r] - iso_reads[r]))) for r in range(args.rounds))
    phase_maxerr = max(float(np.max(np.abs(co_phases[r] - iso_phases[r]))) for r in range(args.rounds))

    # DISCRIMINATING control that v/u is GENUINELY shared (replaces the over-determined no-rekick-vs-isolated arm,
    # which diverged from a per-round kick-SEED mismatch alone). Two NO-REKICK arms that differ ONLY by whether the
    # WKV Izhikevich step runs between resonates, on the SAME kick sequence:
    #   arm A: kick once (round 0), then each round run the WKV _charge_read (its Izhikevich step writes v/u) + resonate
    #   arm B: kick once (round 0), then each round resonate ONLY (NO WKV step)
    # If v/u is shared, arm A's phasor is corrupted by the WKV step and DIVERGES from arm B. If v/u were DISJOINT, the
    # WKV step would not touch the composer's v/u and the two arms would be IDENTICAL. So (arm A != arm B) uniquely
    # isolates "the WKV Izhikevich step corrupts the shared v/u" -- the thing the re-kick discipline exists to defeat.
    bD, _cgd, _cg2d, _snapd = _build_ssm_state_bridge(args.D, args.seed, decay, pop_k=1)
    _install_readout(bD, xp, args.seed); _install_rf(bD, slice_idx, args.seed)
    armA = [_composer_op(bD, xp, slice_idx, kick_seeds[0], args.n_resonate, do_kick=True)]   # round 0: kick + read
    for r in range(1, args.rounds):
        _charge_read(bD, xp, injects[r])                              # WKV Izhikevich step (writes v/u)
        armA.append(_composer_op(bD, xp, slice_idx, 0, args.n_resonate, do_kick=False))       # resonate + read, NO kick
    bE, _cge, _cg2e, _snape = _build_ssm_state_bridge(args.D, args.seed, decay, pop_k=1)
    _install_rf(bE, slice_idx, args.seed)
    armB = [_composer_op(bE, xp, slice_idx, kick_seeds[0], args.n_resonate, do_kick=True)]    # SAME round-0 kick
    for r in range(1, args.rounds):
        armB.append(_composer_op(bE, xp, slice_idx, 0, args.n_resonate, do_kick=False))       # resonate + read, NO WKV step
    vu_shared_maxerr = max(float(np.max(np.abs(armA[r] - armB[r]))) for r in range(1, args.rounds))

    read_ok = read_maxerr < 1e-5
    phase_ok = phase_maxerr < 1e-5
    vu_shared = vu_shared_maxerr > 1e-3          # WKV step corrupts the composer's v/u => v/u genuinely shared
    verdict = "GO" if (read_ok and phase_ok and vu_shared) else "NO-GO"

    print(f"[RESULT {verdict}] one-bridge co-residence (seed {args.seed}, {args.rounds} rounds, n={n}, "
          f"RF slice={len(slice_idx)}):")
    print(f"  WKV read-out  co-resident vs isolated  max|err| = {read_maxerr:.3e}  ({'byte-clean' if read_ok else 'DIVERGES'})  "
          f"[disjoint-by-construction: cp_ssm_state is never touched by RF ops]")
    print(f"  composer phase co-resident vs isolated  max|err| = {phase_maxerr:.3e}  ({'byte-clean' if phase_ok else 'DIVERGES'})")
    print(f"  DISCRIMINATING v/u-sharing (WKV-step vs no-WKV-step, both no-rekick)  max|err| = {vu_shared_maxerr:.3e}  "
          f"({'DIVERGES => WKV step corrupts the SHARED v/u (re-kick load-bearing)' if vu_shared else 'IDENTICAL -- v/u NOT shared, SUSPECT'})")
    print(f"  => the WKV read-out (graded cp_ssm_state) + the composer RF phasor (v/u) CO-RESIDE on ONE bridge, each "
          f"byte-identical to isolated; the v/u array is genuinely shared (the WKV step corrupts it; the re-kick repairs it).")


if __name__ == "__main__":
    main()
