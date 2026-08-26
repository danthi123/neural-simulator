"""Prints the DA read (da_level, snc_firing) for a FIXED afferent on `SIM_BACKEND=numpy`, seed=42, at full
float precision (repr) -- run BEFORE and AFTER the cupy-interop fix (via `git stash` / `git stash pop`) to
confirm the numpy path is byte-identical (the board-#76 6/6-seed-GO runner must not regress).

Usage:
    SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python \
        research/findings/raw/da_axis_cupy_interop/numpy_unchanged_check.py
"""
import os
import sys

assert os.environ.get("SIM_BACKEND") == "numpy", "run with SIM_BACKEND=numpy"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from webapp.da_mode_drives_chat import DaModeDrivesWorkspace  # noqa: E402


def main():
    # Three fixed afferents spanning rest / focus / arousal (per the module's calibration comment).
    for afferent in (0.0, 800.0, 1300.0):
        ws = DaModeDrivesWorkspace(seed=42)
        info = ws.observe("irrelevant -- afferent_override drives the SNc directly", afferent_override=afferent)
        print(f"afferent={afferent!r} acted={info['acted']!r} da_level={info['da_level']!r} "
              f"snc_firing={info['snc_firing']!r} mode={info['mode']!r} reason={info['reason']!r}")

    # Also the direct runner-level call (bypassing the workspace, matching the board-#76 GO invocation shape).
    import research.runners._perturb_and_measure_derisk as PM
    import research.runners._neuromod_spiking_da_mode_derisk as DA
    sb, regions, _ = PM.build(42)
    nbt = PM.names_by_type(regions)
    mgr = DA.make_manager(sb)
    rates, conc, sncf = DA.measure_self_driven(sb, mgr, nbt, DA.BASELINE, DA.APP_SNC)
    print(f"runner-level: conc={conc!r} sncf={sncf!r}")


if __name__ == "__main__":
    main()
