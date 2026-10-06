"""tools/where_ran.py: host / start / git sha / argv[0] per result file from the .prov.json sidecar, runs.jsonl as the
fallback, host classes (pool, workstation, aws, server container, other, unknown) and the `--expect` exit code."""
import importlib.util
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("where_ran", os.path.join(_ROOT, "tools", "where_ran.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def _result(d, name, prov=None):
    p = os.path.join(str(d), name)
    json.dump({"x": 1}, open(p, "w"))
    if prov is not None:
        json.dump(prov, open(p + ".prov.json", "w"))
    return p


def _runs(tmp_path, recs):
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return str(p)


def test_classify_host():
    c = W.classify_host
    assert c("node-dl4g243") == "pool" and c("node-mxl1462drl") == "pool"
    assert c("dant123-wk") == "workstation"
    assert c("ip-172-31-47-57") == "aws" and c("AWS r7i.4xlarge us-east-1") == "aws"
    assert c("pool2 (AWS r7i.4xlarge, research/queue/.pool_ssh_config)") == "aws"
    assert c("83db10ba1ddb") == "server"                       # the hermes container's hex hostname
    assert c("vm") == "other"
    assert c(None) == "unknown" and c("") == "unknown"


def test_reads_the_sidecar_per_file_and_summarises(tmp_path, capsys):
    a = _result(tmp_path, "a.json", {"host": "node-dl4g243", "started": "2026-10-04T05:14:37", "git_sha": "049d4bb0be",
                                      "git_dirty": False, "argv": ["/home/node/derisk-pool/revisions/049d/x.py", "--seed", "1"]})
    b = _result(tmp_path, "b.json", {"host": "dant123-wk", "started": "2026-10-03T10:00:00", "git_sha": "abc1234",
                                      "git_dirty": True, "argv": ["research/runners/y.py"]})
    rows = [W.where(p, W.RunsIndex(str(tmp_path / "none"))) for p in (a, b)]
    assert [r["class"] for r in rows] == ["pool", "workstation"]
    assert rows[0]["argv0"].endswith("x.py") and rows[0]["git_sha"].startswith("049d4bb") and rows[1]["git_dirty"] is True
    s = W.summarize(rows)
    assert s["by_class"] == {"pool": 1, "workstation": 1}
    out = W.render(rows, s)
    assert "node-dl4g243" in out and "more than one host class" in out and "+dirty" in out
    assert W.main([a, b, "--runs", str(tmp_path / "none")]) == 0


def test_runs_jsonl_fallback_by_run_id_and_by_artifact(tmp_path):
    runs = _runs(tmp_path, [{"run_id": "r1", "host": "ip-172-31-47-37", "started": "2026-09-01T00:00:00"},
                            {"run_id": "r2", "host": "dant123-wk", "artifact": "research/findings/raw/z.json",
                             "argv": ["q.py"], "git_sha": "ff00ff0"}])
    idx = W.RunsIndex(runs)
    nohost = _result(tmp_path, "n.json", {"run_id": "r1", "argv": ["n.py"], "git_sha": "1111111"})   # sidecar without host
    row = W.where(nohost, idx)
    assert row["host"] == "ip-172-31-47-37" and row["class"] == "aws" and row["source"] == "sidecar+runs.jsonl"
    assert row["argv0"] == "n.py" and row["git_sha"] == "1111111"
    nosidecar = os.path.join(W.ROOT, "research/findings/raw/z.json")
    assert W.where(nosidecar, idx)["class"] == "workstation"        # no sidecar at all: runs.jsonl by artifact path


def test_no_record_anywhere_is_unknown_not_fine(tmp_path):
    p = _result(tmp_path, "u.json")
    row = W.where(p, W.RunsIndex(str(tmp_path / "none")))
    assert row["class"] == "unknown" and row["host"] is None
    out = W.render([row], W.summarize([row]))
    assert "UNKNOWN" in out


def test_expand_dirs_globs_and_prov_paths(tmp_path):
    a = _result(tmp_path, "a.json", {"host": "node-x"})
    sub = tmp_path / "s42"
    sub.mkdir()
    b = _result(sub, "b.json", {"host": "node-y"})
    assert set(W.expand([str(tmp_path)])) == {a, b}
    assert W.expand([a + ".prov.json"]) == [a]
    assert set(W.expand([str(tmp_path / "*.json")])) == {a}
    assert W.expand([str(tmp_path / "nothing*")]) == []


def test_expect_exit_codes(tmp_path, capsys):
    a = _result(tmp_path, "a.json", {"host": "node-x"})
    b = _result(tmp_path, "b.json", {"host": "dant123-wk"})
    none = str(tmp_path / "none")
    assert W.main([a, "--expect", "pool", "--runs", none]) == 0
    assert W.main([a, b, "--expect", "pool", "--runs", none]) == 1
    assert W.main([str(tmp_path / "missing*.json"), "--runs", none]) == 2
    capsys.readouterr()
    assert W.main([a, "--json", "--runs", none]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["summary"]["by_class"] == {"pool": 1} and doc["files"][0]["host"] == "node-x"
