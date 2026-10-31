"""Selftest for the #203 follow-up: the PRODUCTION comprehension organ, read 30 times, with its frozen synapses flat.

The #203 merge (main `dc6abf989`, `cfg.enforce_plastic_mask_beyond_hebbian`, umbrella env `BRAIN_ENFORCE_PLASTIC_MASK`,
DEFAULT OFF) gates six write sites, and its reply arm found the flag changes the pooled comprehension organ's reply
ZERO times. The 2026-09-02 read-isolation audit (research/findings/2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md)
had measured the opposite on the organ it actually ran: max weight drift 13.8 after ONE read, 56.1 after 30 reads of
one sentence, on the 1710 internal recurrent `sel_* <-> sel_FS_*` synapses declared `plastic_internal=False`.

This test closes the gap on the SAME protocol, DEV seeds only (7, 8 -- the generalization seeds 42/43/44/100/101/102
are untouched by this instrument), and pins down both directions:

  (1) frozen_flat_on   -- 30 successive reads of one sentence, flag ON: every synapse the plastic mask calls
                          frozen is BIT-UNCHANGED (max|dw| == 0.0 exactly), and no write to cp_connections.data
                          touches a frozen entry at all.
  (2) drift_off_is_real -- the identical protocol with the flag OFF still drifts (> 0). Without this the ON-arm
                          zero would be vacuous: a run that never fires also produces a zero.
  (3) instrument_is_inert -- the write tap (research/runners/pi203_write_path_tap.py) is an observer: tapped and
                          untapped runs of the same cell produce byte-identical reply digests. A measurement tool
                          that changes what it measures is not a measurement.

Runs on numpy (forced): no GPU, deterministic, ~25 s total.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

READS = 30
SENTENCE = "the monkey grab the box"
TEACHER_FLAGS = {
    "BRAIN_LEARNED_ANIMACY_CUE": "1",
    "BRAIN_LEARNED_VERB_SELECTS": "1",
    "BRAIN_LEARNED_ANIMACY_LESION": "0",
    "BRAIN_LEARNED_VERB_SELECTS_LESION": "0",
}


def _set_env(enforce: bool) -> None:
    for k, v in TEACHER_FLAGS.items():
        os.environ[k] = v
    os.environ["BRAIN_ENFORCE_PLASTIC_MASK"] = "1" if enforce else "0"


def _coords(bridge) -> dict:
    """(pre, post) coordinate -> weight. Identity-based, so a CSR re-ordering cannot fake a zero."""
    coo = bridge.cp_connections.tocoo()
    return {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}


def _frozen_coords(bridge) -> set:
    mask = getattr(bridge, "cp_synapse_plastic_mask", None)
    if mask is None:
        return set()
    mask = np.asarray(mask)
    coo = bridge.cp_connections.tocoo()
    return {(int(r), int(c)) for k, (r, c) in enumerate(zip(coo.row, coo.col))
            if k < mask.size and not mask[k]}


def _arm(enforce: bool, seed: int, tap: bool) -> dict:
    """Run READS successive reads of one sentence on the organ the audit measured, and report the drift."""
    _set_env(enforce)
    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan
    from research.runners import pi203_write_path_tap as tapmod

    organ = ComprehensionProductionOrgan(seed=seed)
    organ.ensure_built()
    bridge = organ.comp.bridge
    if tap:
        tapmod.install_tap(bridge)

    coords_before = _coords(bridge)
    frozen = _frozen_coords(bridge)
    labels_before = tapmod._mask_label_by_coord(bridge) if tap else {}
    for _ in range(READS):
        organ.judge(SENTENCE)
    coords_after = _coords(bridge)

    common = set(coords_before) & set(coords_after)
    frozen_drift = {cc: abs(coords_after[cc] - coords_before[cc]) for cc in common if cc in frozen}
    plastic_drift = {cc: abs(coords_after[cc] - coords_before[cc]) for cc in common if cc not in frozen}
    sites = tapmod._aggregate(tapmod._LOG) if tap else []
    return {
        "tap_blind": tapmod.tap_is_blind(bridge) if tap else False,
        "n_frozen": len(frozen),
        "frozen_max_abs_drift": max(frozen_drift.values(), default=0.0),
        "frozen_synapses_drifted": sum(1 for d in frozen_drift.values() if d > 1e-9),
        "plastic_max_abs_drift": max(plastic_drift.values(), default=0.0),
        "plastic_synapses_drifted": sum(1 for d in plastic_drift.values() if d > 1e-9),
        "sites_touching_frozen": [s for s in sites if s["frozen_sum_abs_delta"] > 0],
        "csr_rebuilds": getattr(tapmod, "_REBUILDS", [0])[0] if tap else 0,
        "mask_labels_changed": (sum(1 for cc, lab in tapmod._mask_label_by_coord(bridge).items()
                                    if cc in labels_before and lab != labels_before[cc]) if tap else 0),
        "signed_residual": (sum(s["frozen_potentiation"] - s["frozen_depression"] for s in sites)
                            - sum(coords_after[cc] - coords_before[cc] for cc in common if cc in frozen)),
    }


@pytest.mark.parametrize("seed", [7, 8])
def test_frozen_synapses_flat_over_30_reads_when_flag_on(seed):
    """The audit's protocol, flag ON: 30 reads, every declared-frozen synapse bit-unchanged."""
    r = _arm(enforce=True, seed=seed, tap=True)
    assert r["n_frozen"] > 0, "the organ has no frozen synapses at all -- the test would be vacuous"
    assert r["frozen_max_abs_drift"] == 0.0, (
        "flag ON but %d frozen synapses still moved (max |dw| %s): a write path escapes the mask"
        % (r["frozen_synapses_drifted"], r["frozen_max_abs_drift"]))
    if r["tap_blind"]:
        pytest.skip("the weight buffer is not a numpy array (cupy path): the per-write tap cannot hook it, so the "
                    "per-site and residual assertions are unmeasurable here. The coordinate assertion above is the "
                    "real one; per-site write attribution on cupy needs a cupy-native tap, not a silent zero.")
    assert not r["sites_touching_frozen"], (
        "flag ON but a write still landed on a frozen synapse: %s" % r["sites_touching_frozen"])
    # The tap must account for every change it sees; a nonzero signed residual means a write escaped the hooks.
    assert abs(r["signed_residual"]) < 1e-6, "the tap cannot account for the drift it did not record: %s" % r["signed_residual"]


@pytest.mark.parametrize("seed", [7, 8])
def test_the_drift_is_real_when_the_flag_is_off(seed):
    """Non-vacuity guard: the same protocol without the flag DOES move the frozen synapses."""
    r = _arm(enforce=False, seed=seed, tap=True)
    assert r["frozen_synapses_drifted"] > 0, (
        "the flag-OFF arm shows no frozen drift: either the defect is fixed by something else, or this probe "
        "is not reaching the pathway the 2026-09-02 audit measured -- do not read the ON-arm zero as a fix")
    if r["tap_blind"]:
        pytest.skip("tap blind on this backend (non-numpy weight buffer): the signed residual cannot be computed; "
                    "the drift assertion above is the measured part")
    # A nonzero residual in the OFF arm is not an instrument bug: it is the mask-misalignment defect (v) --
    # a CSR rebuild re-orders synapses and, with the flag OFF, the mask is never remapped, so the position-based
    # frozen split disagrees with the coordinate-based drift. The ON arm above has no rebuild-driven misalignment.
    if abs(r["signed_residual"]) > 1e-6:
        assert r["csr_rebuilds"] > 0, (
            "the tap missed writes worth %s with NO CSR rebuild to explain it: the per-site table is wrong"
            % r["signed_residual"])


def test_the_tap_does_not_change_what_it_measures():
    """Instrument inertness: tapped and untapped runs of the same cell give byte-identical replies."""
    from research.runners._comprehension_learned_cues_joint_arm import all_items
    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, extract_transitive
    from research.runners import pi203_write_path_tap as tapmod

    def replies(tap: bool) -> list:
        _set_env(False)
        organ = ComprehensionProductionOrgan(seed=7)
        organ.ensure_built()
        if tap:
            tapmod.install_tap(organ.comp.bridge)
        out = []
        for label, n0, v, n1 in all_items():
            text = f"the {n0} {v} the {n1}"
            tr = extract_transitive(text)
            out.append((label, organ.judge(text), organ.competent(*tr) if tr else None))
        return out

    a, b = replies(True), replies(False)
    # digest, not structural float equality: "byte-identical" has to mean the same thing it means in the
    # orchestrator (research/runners/pi203_write_path_attribution.py, instrument_control)
    da, db = tapmod._digest(a), tapmod._digest(b)
    assert da == db, ("the write tap changed the organ's replies: tapped %s vs untapped %s "
                      "(first three items differ: %s)" % (da[:16], db[:16], list(zip(a[:3], b[:3]))))
