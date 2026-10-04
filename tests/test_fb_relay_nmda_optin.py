"""tom_spiking_witness_organ: the 2026-10-04 relay_nmda mode and its receptor constants are opt-in. The registered
GRID (prereg 2026-10-01) must build exactly as before: no slow-NMDA conductance, no receptor overrides. The relay
mode's diagnosis is docs/drafts/fb-spiking-witness-relay2/DIAGNOSIS.md."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import research.runners.tom_spiking_witness_organ as O


def test_new_constants_default_to_the_registered_behavior():
    assert (O.EVICT_RECEPTOR, O.RELAY_BELIEF_RECEPTOR, O.BELIEF_FS_RECEPTOR) == ("gaba_a", "ampa", "gaba_a")
    assert all(g.get("MODE") in ("relay", "readout") for g in O.GRID)
    assert len(O.GRID) == 8


def test_registered_ops_build_without_slow_nmda_and_relay_nmda_builds_with_it():
    O.set_operating_point(0)
    b, _xp, _idx, _snap = O.build_witness_bridge(7)
    assert not getattr(b.core_config, "enable_nmda_recurrent", False)
    old = O.MODE
    try:
        O.MODE = "relay_nmda"
        b2, _xp, _idx, _snap = O.build_witness_bridge(7)
        assert b2.core_config.enable_nmda_recurrent is True
        assert b2.cp_nmda_recurrent_synapse_mask is not None and bool(b2.cp_nmda_recurrent_synapse_mask.any())
    finally:
        O.MODE = old
        O.set_operating_point(0)
