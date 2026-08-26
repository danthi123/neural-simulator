"""Verifies the SECOND latent bug named in the task: `_ensure`'s post-build snapshot used
`isinstance(x, np.ndarray)`, which is FALSE for every `cp_*` attr on a cupy-backed substrate -> the snapshot
dict came back EMPTY -> `_restore()` was a silent no-op -> repeated reads on the SAME persistent workspace are
NOT history-independent (the substrate's membrane potentials / adaptation variables drift turn-to-turn even at
a FIXED afferent, violating the module's own documented contract: "every read is a deterministic function of
THIS turn's afferent, history-independent").

Proof: on ONE persistent workspace, run the SAME fixed afferent TWICE in a row (via `afferent_override`, which
bypasses the EMA so the input is identical both times) and compare da_level. Restore working => identical.
Restore broken (silent no-op) => the two reads differ (the substrate's dynamic state carried over).

Usage:
    SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python \
        research/findings/raw/da_axis_cupy_interop/verify_ensure_restore_cupy.py
"""
import os
import sys

assert os.environ.get("SIM_BACKEND") == "cupy", "run with SIM_BACKEND=cupy"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from webapp.da_mode_drives_chat import DaModeDrivesWorkspace  # noqa: E402


def main():
    ws = DaModeDrivesWorkspace(seed=42)
    snapshot_keys_after_ensure = None

    # Peek at the snapshot dict size right after the (lazy) build, to directly confirm it's non-empty.
    ws._isolated(ws._ensure)
    snapshot_keys_after_ensure = len(ws._snapshot)
    print(f"post-build snapshot dict size: {snapshot_keys_after_ensure} cp_* entries captured")
    assert snapshot_keys_after_ensure > 0, "BUG STILL PRESENT: snapshot is EMPTY on cupy (isinstance(x, np.ndarray) filter)"

    r1 = ws.observe("first turn -- some rich novel content here", afferent_override=900.0)
    r2 = ws.observe("second turn -- totally different words entirely", afferent_override=900.0)
    r3 = ws.observe("third turn -- yet more different content again", afferent_override=900.0)
    print("read 1 (afferent=900):", r1["da_level"], r1["snc_firing"])
    print("read 2 (afferent=900):", r2["da_level"], r2["snc_firing"])
    print("read 3 (afferent=900):", r3["da_level"], r3["snc_firing"])

    assert r1["da_level"] == r2["da_level"] == r3["da_level"], (
        f"HISTORY-DEPENDENT (restore is a no-op): da_level drifted across identical-afferent reads: "
        f"{r1['da_level']} vs {r2['da_level']} vs {r3['da_level']}")
    assert r1["snc_firing"] == r2["snc_firing"] == r3["snc_firing"]
    print("\nHISTORY-INDEPENDENCE CONFIRMED: repeated reads at the SAME afferent on the SAME persistent "
          "workspace are byte-identical -- _restore() is correctly resetting the cupy substrate state each turn.")


if __name__ == "__main__":
    main()
