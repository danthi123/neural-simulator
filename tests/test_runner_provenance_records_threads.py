import json
import os
import subprocess
import sys
import textwrap


def test_provenance_records_thread_counts_cpu_count_and_brain_flags(tmp_path):
    # 2026-09-24: a math-library thread-count change moved a ridge decode on one seed (perception G0: 12 threads read
    # 0.6458, 1 or 4 threads 0.625), and provenance recorded no thread count; BRAIN_* flags were recorded only for
    # BRAIN_MULTIREF_*. The run record now carries both, plus cpu_count (the effective count when unset).
    script = textwrap.dedent(
        """
        import json
        import sys
        import research.runners as provenance_door
        provenance_door._PROV_DIR = sys.argv[1]
        rec = provenance_door._record_start()
        print(json.dumps({"env": rec["env"], "cpu_count": rec.get("cpu_count")}))
        """
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SIM_PROVENANCE", "BRAIN_"))}
    env.update({"OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "BRAIN_TEST_FLAG_XYZ": "1",
                "UNRELATED_VAR_XYZ": "1", "SIM_NO_PROVENANCE": "1"})
    r = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=root, env=env, text=True,
                       capture_output=True, check=True)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["env"]["OMP_NUM_THREADS"] == "3" and out["env"]["OPENBLAS_NUM_THREADS"] == "3"
    assert out["env"]["BRAIN_TEST_FLAG_XYZ"] == "1"
    assert "UNRELATED_VAR_XYZ" not in out["env"]
    assert out["cpu_count"] == os.cpu_count()
