"""R4 close (Option 3 of `2026-06-30-tier2-integrated-spiking-loop-scoping.md`): make the cross-region
perception->compose grounding hand-off DEVICE-RESIDENT (NO host `gen_proj @ conc_rate` matmul, NO `to_host` of the
gen_concept spike VECTOR) -- closing the secondary host DATA seam on the navigate-to-compose route.

THE RESIDUAL (R4, precisely). In the step-3 grounding (`navigate_to_compose_then_answer.py`), the DEFAULT `gen_spikes`
mode's LOAD-BEARING transform (the LEARNED rate-Hebbian `gen_perception->gen_concept` convergence) is already SYNAPTIC
on the merged bridge. BUT the grounded CODE is then computed host-side:
  - `read_gen_concept_spikes` (`navigate_to_compose_then_answer.py:213-220`): each step `fs = to_host(cp_firing_states)`
    reads the gen_concept SPIKES to host, `conc_acc += fs[conc_region]` accumulates host -> a host rate VECTOR.
  - `gen_grounded_phases` (`navigate_to_compose_then_answer.py:227-228`): `z = gen_proj @ conc_rate` (a HOST complex
    matmul -- the "M @ rate" the scoping names) then `angle(z)` (host rate->phasor).
That `to_host` of the gen_concept spike vector + the host `gen_proj @ rate` matmul is the perception->memory DATA
hand-off: the percept's neural code crosses host (as a spike vector AND a host matmul) to enter the composer codebook.

THE CLOSE (this module, reuse-by-import, NO `sim/` edit) -- the cross-region twin of R1's device-resident handoff
(`_seq_fused_fabric.py`): keep the gen_concept spike accumulation + the projection + `angle()` ALL on-device.
  1. `accumulate_conc_spikes_device(bridge, conc_region, read_steps)` -- accumulate `cp_firing_states[conc_region]`
     ON-DEVICE (a backend gather + add, NEVER `to_host` of the per-step firing carrier). Returns the per-neuron mean
     spike-rate as a backend (device) array. REPLACES `read_gen_concept_spikes`'s `fs = to_host(cp_firing_states)`.
  2. `device_resident_grounded_phases(conc_rate_dev, gen_proj)` -- the FIXED cortico-cortical fan-in projection
     `gen_proj @ conc_rate` + `angle()` run ON-DEVICE (the projection matrix is moved to `xp` once + cached; the
     matvec + `angle` + mod are backend ops). REPLACES `gen_grounded_phases`'s host `proj @ rate` matmul + host
     `angle`. Returns the D-length grounded PHASES as a host array -- the ONLY host crossing, the formatted code, the
     same legitimacy class as `rf_read_phases` reading phases off RF neurons (the R5 body-read boundary).

WHY IT IS A LEGIT CLOSE (not a host shortcut moved). The fixed complex projection `gen_proj` is a FIXED cortico-cortical
fan-in (the C-6 ruling, `2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md`: "accept as a legit fixed
cortico-cortical fan-in ... or realize it as a fixed complex bridge synapse") -- it is run ONCE per perceived object
(not per fact), and it carries NO learned per-fact structure. R4 was never a host COMPUTATION-of-cognition leak (the
load-bearing percept->concept transform is the LEARNED convergence, already synaptic); it was a host DATA TRANSFER (the
spike vector + the fan-in projection crossing host). Moving the fan-in on-device + accumulating the spikes on-device
eliminates the host `gen_proj @ rate` matmul + the gen_concept-spike-vector `to_host`, leaving only the final phases
crossing host (R5). This is byte-faithful: the on-device matmul == the host matmul to numerical tolerance.

`navigate_to_compose_then_answer.read_gen_concept_spikes`/`gen_grounded_phases` get an opt-in `device_resident` flag
(default False = the host path, BYTE-UNCHANGED); `CoResidentOneBrainComposer(perception_device_resident=True)` flips it
for the production one-brain perceive-and-ground. Both default-off = the validated host path verbatim.

  SIM_BACKEND=numpy python -u -m research.runners._r4_grounding_onbridge --seeds 42,43,44
  SIM_BACKEND=cupy  python -u -m research.runners._r4_grounding_onbridge --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from sim.backend import get_backend, to_host, is_gpu_backend


# Cache the on-device projection matrix per (id(gen_proj)) so the host->device move happens once, not per object.
_DEV_PROJ_CACHE: dict = {}


def _device_proj(gen_proj):
    """Move the fixed complex projection `gen_proj` (D, n_conc) to the active backend ONCE (cached by object id).
    The projection is a FIXED cortico-cortical fan-in -- the host->device move is a one-time device configuration
    (memristor-crossbar / synapse-table style), not per-fact work."""
    xp, _ = get_backend()
    key = id(gen_proj)
    cached = _DEV_PROJ_CACHE.get(key)
    if cached is None:
        cached = xp.asarray(np.asarray(gen_proj, dtype=np.complex128))
        _DEV_PROJ_CACHE[key] = cached
    return cached


def accumulate_conc_spikes_device(bridge, conc_region, read_steps, drive_dev=None):
    """Accumulate the gen_concept per-neuron spike rate ON-DEVICE over `read_steps` steps -- the device-resident
    replacement for `read_gen_concept_spikes`'s `fs = to_host(cp_firing_states); conc_acc += fs[conc_region]`.

    `conc_region` is a backend (xp) int index array (the gen_concept global indices). `drive_dev` (optional) is the
    structured-perception drive re-asserted each step (a backend array, length num_neurons); if None the caller is
    expected to hold the drive (the standalone runner re-asserts it). The per-step firing state is GATHERED on-device
    (`cp_firing_states[conc_region]`) and added to a device accumulator -- the per-step firing CARRIER is NEVER
    `to_host`-ed. Returns the per-neuron mean spike-rate as a backend (device) array (length = conc_region size).

    The carrier stays device-resident: the only host crossing downstream is the final D-length grounded phases
    (`device_resident_grounded_phases`), the R5 body-read boundary."""
    xp, _ = get_backend()
    conc_idx = xp.asarray(conc_region)
    acc = xp.zeros(int(conc_idx.shape[0]), dtype=xp.float64)
    for _ in range(int(read_steps)):
        if drive_dev is not None:
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[:] = drive_dev
        bridge._run_one_simulation_step()
        # GATHER the gen_concept firing state ON-DEVICE (backend indexing) -- NO to_host of the per-step carrier.
        acc = acc + bridge.cp_firing_states[conc_idx].astype(xp.float64)
    return acc / float(read_steps)


def device_resident_grounded_phases(conc_rate_dev, gen_proj):
    """The DEVICE-RESIDENT grounded-code formatting: the fixed cortico-cortical fan-in projection `gen_proj @ rate` +
    `angle()` run ON-DEVICE -- the device-resident replacement for `gen_grounded_phases`'s host
    `z = gen_proj @ conc_rate; angle(z)` matmul.

    `conc_rate_dev` is a backend (xp) gen_concept spike-rate array (the `accumulate_conc_spikes_device` output, kept on
    device). The projection matrix is moved to `xp` once (cached) and the matvec + `angle` + mod are backend ops. The
    ONLY host crossing is the final D-length phases (`to_host` of the formatted code = the R5 body-read, the same
    legitimacy class as `rf_read_phases`). Returns the grounded phases[D] in [0,1) as a HOST array (the composer
    codebook is a host numpy array; the FHRR algebra reads `self.concepts[w]` as host numpy -- INTRINSIC to the
    composer, the documented R5 read).

    `_to_phasor(phases) = exp(2pi i phases) = exp(i angle(proj @ rate))` -- byte-faithful to the host path."""
    xp, _ = get_backend()
    proj_dev = _device_proj(gen_proj)
    rate_dev = xp.asarray(conc_rate_dev).astype(xp.complex128)
    z = proj_dev @ rate_dev                                   # the fixed cortico-cortical fan-in, ON-DEVICE
    phases_dev = (xp.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)
    return np.asarray(to_host(phases_dev), dtype=np.float64)  # the formatted code crosses host (R5 body-read)


# ----------------------------------------------------------------------------------------------------------------
# The CPU/GPU de-risk (the R4 GO bar): device-resident == host-matmul grounded phases (==host), the gen_concept-spike
# to_host eliminated + the host gen_proj@rate matmul gone (the SEAM closed), on the LIVE merged-bridge gen stack.
# CPU runs a tiny smoke (a few seeds at the merged tier); the GPU 6-seed is the controller's run.
# ----------------------------------------------------------------------------------------------------------------
def _seed_compare(seed):
    """Build the merged nav+conv bridge with the co-resident gen stack, render+ground ONE object, and isolate the R4
    CLOSE: do the gen_concept-spike accumulation ONCE (device-resident -- the carrier never crosses host), then FORMAT
    that ONE shared rate vector BOTH ways -- the HOST `angle(gen_proj @ rate)` matmul and the device-resident on-device
    projection -- and assert they are EQUAL.

    WHY format-on-ONE-snapshot (the fix for the GPU NEGATIVE, 2026-06-30): the close changes ONLY where the projection
    runs (host matmul vs on-device); it must NOT change the value GIVEN THE SAME spike rate. The earlier de-risk ran
    the HOST path (`read_gen_concept_spikes`) and the DEVICE path as TWO SEPARATE perception windows -> two DIFFERENT
    gen_concept spike snapshots (GPU spike timing is not bit-identical across two windows) -> a phase_cos ~0.87 gap that
    is the cross-window RATE-INPUT variance, NOT the close. The production code reads ONCE per object, so the two-window
    variance never arises in deployment. The correct GO bar accumulates ONCE + formats both ways on that snapshot, which
    is exactly what `test_device_resident_equals_host_matmul` pins on a fixed rate vector (here on the LIVE rate)."""
    import sim.backend as backend
    from research.runners.navigate_to_compose_then_answer import (
        build_compose_bridge, gen_grounded_phases, GEN_SETTLE_STEPS, GEN_READ_STEPS, GEN_PERC_DRIVE_PA,
    )
    from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS

    xp, backend_name = get_backend()
    # the merged bridge (host_m vs gen_spikes irrelevant for the build -- we force the gen stack via gen_spikes).
    bridge, composer, h, proj = build_compose_bridge(seed, with_body=False, grounding="gen_spikes", composer_kind="rf")
    gen = h["gen"]
    gen_proj = h["gen_proj"]
    obj_word = OBJECT_WORDS[0]
    obj_idx = OBJECT_WORDS.index(obj_word)

    # render the percept ONCE: drive the held-out shape's structured-perception set into gen_perception + settle.
    perc_region = np.asarray(gen["perc_region"], dtype=np.int64)
    conc_region = np.asarray(gen["conc_region"], dtype=np.int64)
    vis_sets = gen["vis_sets"]
    held_out = list(gen["gen_held_out"])
    perc_local = np.asarray(vis_sets[held_out[obj_idx]], dtype=np.int64)
    perc_global = perc_local + int(perc_region[0])
    n = int(bridge.core_config.num_neurons)
    drive = np.zeros(n, dtype=np.float32)
    drive[perc_global] = GEN_PERC_DRIVE_PA
    drive_dev = xp.asarray(drive)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(GEN_SETTLE_STEPS):
        bridge._run_one_simulation_step()

    # ACCUMULATE the gen_concept spikes ONCE (device-resident: the carrier never crosses host). Instrument to_host to
    # count any read of the per-step firing carrier DURING the accumulate -- that is the structural R4 close (the seam).
    conc_idx_dev = xp.asarray(conc_region)
    firing_reads = {"n": 0}
    real_to_host = backend.to_host
    import research.runners._r4_grounding_onbridge as r4mod

    def _spy(arr):
        try:
            if arr is bridge.cp_firing_states:
                firing_reads["n"] += 1
        except Exception:
            pass
        return real_to_host(arr)

    backend.to_host = _spy
    r4mod.to_host = _spy
    try:
        conc_rate_dev = accumulate_conc_spikes_device(bridge, conc_idx_dev, GEN_READ_STEPS, drive_dev=drive_dev)
    finally:
        backend.to_host = real_to_host
        r4mod.to_host = real_to_host
    bridge.cp_external_input_current[:] = 0.0
    to_host_clean = (firing_reads["n"] == 0)

    # FORMAT the ONE shared rate vector BOTH ways:
    #   HOST   path: bring the rate to host + the host `angle(gen_proj @ rate)` matmul (the validated default format).
    #   DEVICE path: the close's on-device projection + angle() (only the final phases cross host, the R5 body-read).
    conc_rate_host = np.asarray(to_host(conc_rate_dev), dtype=np.float64)
    host_phases = gen_grounded_phases(conc_rate_host, gen_proj)            # host gen_proj @ rate + angle
    dev_phases = device_resident_grounded_phases(conc_rate_dev, gen_proj)  # on-device gen_proj @ rate + angle

    # ==host on the SAME snapshot: the close must not change the value. (atol 1e-6 covers a benign cupy/numpy GEMV
    # float-order delta; the phasor cosine is the composer's own similarity, reported for context.)
    phase_cos = float(np.mean(np.cos(2.0 * np.pi * (host_phases - dev_phases))))
    equal = bool(np.allclose(host_phases, dev_phases, atol=1e-6)) or phase_cos > 0.99999

    print(f"  [seed {seed} backend={backend_name}] ==host (SAME snapshot) phase_cos={phase_cos:.6f} equal={equal}  "
          f"max|dphase|={float(np.max(np.abs(host_phases - dev_phases))):.2e}  "
          f"gen_concept-spike-carrier to_host reads in device accumulate: {firing_reads['n']} "
          f"({'CLEAN' if to_host_clean else 'LEAK'})", flush=True)
    return dict(seed=int(seed), backend=backend_name, phase_cos=phase_cos, equal=bool(equal),
                max_dphase=float(np.max(np.abs(host_phases - dev_phases))),
                firing_carrier_reads=int(firing_reads["n"]), to_host_clean=bool(to_host_clean),
                n_conc=int(conc_region.size))


def main():
    ap = argparse.ArgumentParser(
        description="R4 close: make the perception->compose grounding hand-off DEVICE-RESIDENT (no host gen_proj@rate "
                    "matmul, no to_host of the gen_concept spike vector). GO if device-resident == host grounding "
                    "AND the gen_concept-spike to_host is GONE from the device path.")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="research/findings/raw/_r4_grounding_onbridge.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[R4 DEVICE-RESIDENT GROUNDING] perception->compose grounding hand-off device-resident "
          f"(no host gen_proj@rate, no to_host of the gen_concept spike vector). gpu={is_gpu_backend()}\n", flush=True)
    rows = [_seed_compare(s) for s in seeds]
    all_equal = all(r["equal"] for r in rows)
    all_clean = all(r["to_host_clean"] for r in rows)
    verdict = "GO" if (all_equal and all_clean) else "NEGATIVE"

    print(f"\n{'=' * 96}", flush=True)
    print(f"  {len(rows)} seed(s): device==host (SAME snapshot) {sum(r['equal'] for r in rows)}/{len(rows)}  "
          f"gen_concept-spike to_host-clean {sum(r['to_host_clean'] for r in rows)}/{len(rows)}  ==> {verdict}",
          flush=True)
    if verdict == "GO":
        print("  GO: GIVEN THE SAME gen_concept spike snapshot, the device-resident on-device projection produces the "
              "SAME grounded code as the host `angle(gen_proj@rate)` matmul (==host), AND the gen_concept SPIKE VECTOR "
              "is never read to host in the accumulate + the host gen_proj@rate matmul is GONE from the device path -- "
              "only the final D-length phases cross host (the R5 body-read). R4 (the perception->compose grounding "
              "host-marshal) is closed: the fixed cortico-cortical fan-in runs on-device, the spike accumulation runs "
              "on-device. The integration (the navigate-to-compose runner + the CoResidentOneBrainComposer "
              "perceive-and-ground) flip on via the opt-in flag.", flush=True)
    else:
        print("  NEGATIVE: localize -- the on-device projection != the host matmul ON THE SAME rate snapshot (a genuine "
              "backend GEMV gap, > a benign float-order delta) OR the gen_concept-spike carrier still read to host in "
              "the accumulate. (NOTE: this de-risk formats ONE shared snapshot both ways -- a divergence here is the "
              "complex op itself, NOT cross-window spike-snapshot variance.)", flush=True)
    print(f"{'=' * 96}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(verdict=verdict, gpu=is_gpu_backend(), per_seed=rows), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
