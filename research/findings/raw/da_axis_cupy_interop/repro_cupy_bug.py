"""Repro script for the 2026-08-25 DA-axis cupy-interop silent failure (board #76 wired into
`webapp/da_mode_drives_chat.py`). Run BEFORE the fix, on `SIM_BACKEND=cupy`, to print the FULL traceback that
`DaModeDrivesWorkspace.observe()`'s bare `except Exception` swallows into `da_drives.reason = "error:..."`.

Usage:
    SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python \
        research/findings/raw/da_axis_cupy_interop/repro_cupy_bug.py

Two things are shown:
  1. `observe()` AS THE PRODUCTION PATH RUNS IT -- confirms the swallowed `reason` string matches the
     diagnostic-observed `"error:ValueError: non-scalar numpy.ndarray cannot be used for fill"`.
  2. The SAME call chain (`_ensure` -> `_read_da_level` -> `_DA.measure_self_driven`) run WITHOUT the
     try/except, so the real traceback (throw site + line number) prints in full.
"""
import os
import sys
import traceback

assert os.environ.get("SIM_BACKEND") == "cupy", "run with SIM_BACKEND=cupy (see module docstring)"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from webapp.da_mode_drives_chat import DaModeDrivesWorkspace  # noqa: E402


def main():
    print("=== SIM_BACKEND =", os.environ.get("SIM_BACKEND"), "===")

    print("\n--- (1) production path: observe() (swallows the exception into `reason`) ---")
    ws1 = DaModeDrivesWorkspace(seed=42)
    info = ws1.observe("tell me something surprising and new about the ocean")
    print("info =", info)
    assert info["acted"] is False, "expected the cupy interop bug to make acted=False"
    assert info["reason"] is not None and info["reason"].startswith("error:"), (
        f"expected a swallowed error reason, got {info['reason']!r}")
    print(f"CONFIRMED: da_drives.reason = {info['reason']!r}")

    print("\n--- (2) same call chain, UNCAUGHT, for the full traceback ---")
    ws2 = DaModeDrivesWorkspace(seed=42)
    try:
        ws2._isolated(ws2._ensure)
        conc, sncf = ws2._isolated(lambda: ws2._read_da_level(500.0, False))
        print("UNEXPECTED: no exception raised; conc=", conc, "sncf=", sncf)
    except Exception:
        print("FULL TRACEBACK (the real throw site):\n")
        traceback.print_exc()
        exc_type, exc_val, _ = sys.exc_info()
        print(f"\nException: {exc_type.__name__}: {exc_val}")


if __name__ == "__main__":
    main()
