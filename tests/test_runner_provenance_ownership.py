import os
import subprocess
import sys
import textwrap


def test_provenance_stamps_only_the_declared_concurrent_output(tmp_path):
    script = textwrap.dedent(
        """
        import os
        import sys
        import time

        import research.runners as provenance_door

        root = sys.argv[1]
        raw = os.path.join(root, "research", "findings", "raw")
        os.makedirs(raw)
        own = os.path.join(raw, "seed43.json")
        peer = os.path.join(raw, "seed44.json")
        for path, seed in ((own, 43), (peer, 44)):
            with open(path, "w") as fh:
                fh.write('{"seed": %d}' % seed)

        provenance_door._RAW_DIR = raw
        provenance_door._START = time.time() - 5.0
        argv = ["/repo/research/runners/probe.py", "--out", own]
        sys.argv = argv
        rec = {
            "run_id": "run-seed43",
            "runner": "research/runners/probe.py",
            "argv": argv,
            "cwd": root,
            "git_sha": "deadbeef",
            "git_dirty": True,
            "started": "2026-08-03T00:00:00",
            "env": {"SIM_BACKEND": "numpy"},
        }

        made = provenance_door._stamp_outputs(rec)

        assert made == [own]
        assert os.path.exists(own + ".prov.json")
        assert not os.path.exists(peer + ".prov.json")
        """
    )
    env = os.environ.copy()
    env["SIM_NO_PROVENANCE"] = "1"
    env["SIM_BACKEND"] = "numpy"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
