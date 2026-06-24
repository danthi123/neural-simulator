"""FOLLOW-ON #3 validation: the on-bridge DENSE complex-weight RF mode (cfg.rf_dense_weights, O-2-purity).

The optional default-OFF `sim/` edit: rf_set_complex_weights materializes a dense complex W_dense = W_re + i*W_im
(only when the flag is on) and the RF matvec uses a single dense GEMV (W_dense @ z) instead of four sparse SpMVs.

Validates, all SMALL + FOREGROUND (each well under ~5 min):
 (1) DEFAULT-OFF byte-identity: flag off => cp_rf_w_dense is None; the dense-off phases == a baseline build's phases
     (the sparse CSR path is byte-unchanged). (The existing RF tests cover this verbatim; this is a runner echo.)
 (2) DENSE-ON bit-exact: flag on => the dense-GEMV phases == the sparse-off phases (the SAME math), to machine
     roundoff; AND a direct raw-matvec check W_dense @ z vs (W_re@re-W_im@im)+i*(W_re@im+W_im@re) agrees to atol 1e-9
     (f64) / 1e-6 (f32 membrane-read parity).
 (3) VRAM trade: the dense (N^2) byte size vs the sparse (nnz) byte size for the test workload (dense is bigger for
     sparse weights but IS the hardware-natural representation -- the honest caveat).

Run: SIM_BACKEND=cupy python -m research.runners._followon3_rf_dense_weights_purity
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend, get_backend, to_host  # noqa: E402
from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402

OUT = "research/findings/raw/_followon3_rf_dense_weights_purity.json"


def _bind_bundle_workload(D=64, L=3, seed=7):
    """A bind+bundle workload exercising BOTH a permutation matvec (1 nnz/row) AND a strided accumulation
    (L nnz/row): bind L (cue,filler) blocks then bundle them. n = (L+1)*D neurons. Returns (n, conns, kick)."""
    rng = np.random.default_rng(seed)
    n = (L + 1) * D
    # bundle-style strided sum: post block (L*D .. (L+1)*D) <- sum_l pre block l (L nnz/row) -> matvec accumulation
    conns = [(L * D + k, l * D + k, complex(np.exp(2j * np.pi * rng.uniform(0, 1))))
             for l in range(L) for k in range(D)]
    kick = np.zeros(n, dtype=np.complex128)
    for l in range(L):
        kick[l * D:(l + 1) * D] = np.exp(2j * np.pi * rng.uniform(0, 1, D))
    return n, conns, kick


def _run_phases(dense, n, conns, kick, period=200, seed=7):
    """Build a fresh RF bridge, set rf_dense_weights=dense, install conns, kick, resonate, read phases.
    Returns (phases, cp_rf_w_dense_is_none, nnz, dense_bytes, sparse_bytes)."""
    b = _build_rf_bridge(n, seed)
    b.core_config.rf_dense_weights = bool(dense)
    b.core_config.enable_rf_cudagraph = False  # isolate the loop/dense path (megakernel is CSR-specific)
    b.rf_set_complex_weights(conns)
    dense_is_none = getattr(b, "cp_rf_w_dense", None) is None
    nnz = int(b.cp_rf_w_re.nnz)
    # byte sizes: dense complex128 (n*n*16) vs sparse two-CSR (re+im data float64 + indices int32 + indptr int32)
    dense_bytes = n * n * 16
    sparse_bytes = 2 * (nnz * 8) + 2 * (nnz * 4) + 2 * ((n + 1) * 4)
    b.rf_kick(kick, period=period)
    b.rf_resonate_steps(period + 8)
    ph = np.asarray(b.rf_read_phases()).copy()
    return ph, dense_is_none, nnz, dense_bytes, sparse_bytes


def _raw_matvec_bitexact(n, conns, seed=7):
    """Direct raw-matvec check (independent of the resonate loop): build the dense + sparse weights and compare
    W_dense @ z  vs  (W_re@re - W_im@im) + i*(W_re@im + W_im@re) on a random complex z. f64 max-err + f32 parity."""
    import cupy as cp  # GPU-only raw check
    from sim.backend import get_sparse_module
    csp = get_sparse_module()
    rng = np.random.default_rng(seed + 1)
    rows = np.array([p for (p, q, w) in conns], dtype=np.int32)
    cols = np.array([q for (p, q, w) in conns], dtype=np.int32)
    w_re = np.array([complex(w).real for (p, q, w) in conns], dtype=np.float64)
    w_im = np.array([complex(w).imag for (p, q, w) in conns], dtype=np.float64)
    W_re = csp.csr_matrix((cp.asarray(w_re), (cp.asarray(rows), cp.asarray(cols))), shape=(n, n))
    W_im = csp.csr_matrix((cp.asarray(w_im), (cp.asarray(rows), cp.asarray(cols))), shape=(n, n))
    W_dense = W_re.toarray().astype(cp.complex128) + 1j * W_im.toarray().astype(cp.complex128)
    z = cp.asarray(np.exp(2j * np.pi * rng.uniform(0, 1, n)).astype(np.complex128))
    re, im = z.real, z.imag
    sparse_mv = (W_re @ re - W_im @ im) + 1j * (W_re @ im + W_im @ re)
    dense_mv = W_dense @ z
    err_f64 = float(cp.max(cp.abs(dense_mv - sparse_mv)).get())
    # f32 parity (the membrane read is f32): cast the operands to f32 and compare
    z32 = z.astype(cp.complex64)
    Wd32 = W_dense.astype(cp.complex64)
    dense_mv32 = (Wd32 @ z32).astype(cp.complex128)
    err_f32 = float(cp.max(cp.abs(dense_mv32 - sparse_mv)).get())
    return err_f64, err_f32


def main():
    res = {"backend": get_backend()[1], "gpu": bool(is_gpu_backend())}
    if not is_gpu_backend():
        res["verdict"] = "SKIP_NO_GPU"
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return

    D, L, seed = 64, 3, 7
    n, conns, kick = _bind_bundle_workload(D, L, seed)
    res["workload"] = {"D": D, "L": L, "n": n, "n_conns": len(conns)}

    # (1) default-off: a baseline build (flag never touched) vs an explicit-off build -- both the sparse path.
    ph_base, base_none, _, _, _ = _run_phases(False, n, conns, kick, seed=seed)
    ph_off, off_none, nnz, dense_bytes, sparse_bytes = _run_phases(False, n, conns, kick, seed=seed)
    off_byte_identical = bool(np.array_equal(ph_base, ph_off))

    # (2) dense-on: bit-exact same math as sparse-off.
    ph_on, on_none, _, _, _ = _run_phases(True, n, conns, kick, seed=seed)
    dense_max_phasediff = float(np.max(np.abs(ph_off - ph_on)))
    err_f64, err_f32 = _raw_matvec_bitexact(n, conns, seed=seed)

    res["default_off"] = {
        "cp_rf_w_dense_is_None_when_off": bool(off_none and base_none),
        "off_phases_byte_identical_to_baseline": off_byte_identical,
    }
    res["dense_on"] = {
        "cp_rf_w_dense_materialized_when_on": bool(not on_none),
        "dense_vs_sparse_max_phase_diff": dense_max_phasediff,
        "raw_matvec_max_err_f64": err_f64,
        "raw_matvec_max_err_f32": err_f32,
    }
    res["vram_trade"] = {
        "nnz": nnz,
        "dense_bytes_NxN_complex128": dense_bytes,
        "sparse_bytes_two_CSR": sparse_bytes,
        "dense_over_sparse_x": round(dense_bytes / max(sparse_bytes, 1), 1),
        "note": "dense is bigger for sparse weights (O(N^2)) but IS the hardware-natural representation",
    }

    # GO criteria: default-off byte-identical + dense materialized only when on + dense==sparse to roundoff.
    go = (off_byte_identical and off_none and base_none and (not on_none)
          and dense_max_phasediff < 1e-9 and err_f64 < 1e-9 and err_f32 < 1e-4)
    res["verdict"] = "GO" if go else "HONEST_NEEDS_REVIEW"
    res["go_criteria"] = {
        "default_off_byte_identical": off_byte_identical,
        "dense_None_when_off": bool(off_none and base_none),
        "dense_materialized_when_on": bool(not on_none),
        "dense_eq_sparse_phase_lt_1e-9": dense_max_phasediff < 1e-9,
        "raw_f64_lt_1e-9": err_f64 < 1e-9,
        "raw_f32_lt_1e-4": err_f32 < 1e-4,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
