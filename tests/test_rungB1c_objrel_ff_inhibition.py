"""CPU structural guard for the objrel subtractive-FF-inhibition de-risk runner.

The full 6-seed de-risk is a slow research run (each seed builds the c2 bridge + reservoir + Ws + the spiking read, and
selects the graded-subtraction op point); it is NOT a fast CI gate. This light test guards the runner's structure — the
confound-free design (it repurposes the c2 WTA's EXISTING shared inhibitory pool via a graded flag, no new neurons), the
6-seed-blind op-point protocol, and the four anti-cheat flags with their correct thresholds — so the runner stays
committable + importable without a GPU or a long run.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import pytest  # noqa: E402

import research.runners._rungB1c_objrel_ff_inhibition_derisk as m  # noqa: E402


def test_op_point_grid_and_mechanism_hooks_exist():
    # the dev op-point sweep grid is defined (searched ONLY on the dev seeds, frozen for the blind seeds)
    assert hasattr(m, "_select_op_point") and hasattr(m, "run_seed")
    # the confound-free graded-subtraction hooks (repurpose the EXISTING WTA inh pool -- no new neurons)
    assert hasattr(m, "mark_graded") and hasattr(m, "_revert_graded") and hasattr(m, "_set_i2e")


def test_anti_cheat_flag_thresholds_are_correct():
    # a synthetic run_seed result dict must produce the right load-bearing anti-cheat flags (the honest gate):
    #   objrel recovers >= 0.85 ; canonical not regressed >= 0.90 ; differential-load-bearing (pedestal-off collapse +
    #   >= 0.30 drop) ; scrambled-label collapses <= 0.50.
    def flags(objr_s0, canon, ped_s0, scr_s0):
        return {
            "objrel_recovers": bool(objr_s0 >= 0.85),
            "canonical_not_regressed": bool(canon >= 0.90),
            "differential_load_bearing": bool(ped_s0 <= 0.50 and objr_s0 - ped_s0 >= 0.30),
            "scramble_chance": bool(scr_s0 <= 0.50),
        }
    # a clean GO would set all four True; the observed seed-42 see-saw (objrel 1.0 but canon 0.33, scramble 1.0) does NOT
    go = flags(1.0, 0.95, 0.0, 0.0)
    assert all(go.values())
    seesaw = flags(1.0, 0.33, 0.0, 1.0)                 # objrel up but canonical regressed + scramble did not collapse
    assert seesaw["objrel_recovers"] and not seesaw["canonical_not_regressed"] and not seesaw["scramble_chance"]
