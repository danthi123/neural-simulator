"""CI guard for the R-iii EMERGENT CA3 pattern completion (CYCLE 1076): a partial cue of a SELF-ORGANIZED sparse
CA3 assembly (formed by the Kopsick-2024 direct-synchronous protocol + the rate-window co-activity Hebbian, learning
a strong ~12.6x recurrent attractor) completes the held-out members SPECIFICALLY via the two-compartment dendritic
dAP plateau, where a linear read-out, an untrained net, and a random cue all fail = pattern completion LEARNED FROM
EXPERIENCE. GPU-gated (the formation + plateau-ignition regime is cupy-validated). Winning config: hebb_max=120
(within/cross ~12.6x), k_thresh=25 (between the cross-drive floor and the held-out within-drive), cue 1000, pres 60."""
import os

import pytest


def _gpu():
    try:
        from sim.backend import get_backend, is_gpu_backend
        get_backend()
        return is_gpu_backend()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="emergent CA3 completion is cupy-validated (formation + plateau, GPU-gated)")
def test_emergent_ca3_completion_seed42():
    from research.runners._riii_ca3_emergent_completion_derisk import run_seed
    # winning 6-seed-GO config lives in run_seed's defaults (hebb_max=120, k_thresh=25, cue 1000, pres 60)
    on = run_seed(42, do_train=True, coincidence=True)
    linear = run_seed(42, do_train=True, coincidence=False)                 # dendritic plateau OFF, same learned attractor
    notrain = run_seed(42, do_train=False, coincidence=True)                # fresh net, no learned attractor
    perm = run_seed(42, do_train=True, coincidence=True, permuted_cue=True)  # drive a RANDOM half (not the assembly)
    assert on["heldout"] > 0.30, on                                        # the partial cue COMPLETES the held-out members
    assert on["nonassembly"] < 0.20, on                                    # SPECIFIC -- non-assembly cells stay silent
    assert linear["heldout"] < 0.20, linear                                # the dendritic plateau is LOAD-BEARING
    assert on["heldout"] - linear["heldout"] > 0.20, (on, linear)
    assert notrain["heldout"] < 0.20, notrain                              # completion needs the LEARNED attractor
    assert perm["heldout"] < 0.20, perm                                    # completion rides the trained ASSEMBLY, not the drive
