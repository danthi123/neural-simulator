"""brain/provenance.py: stamp adds only res["provenance"] (host, git sha or brain/ source hash, resolved model file);
the source hash is the same in a copy without .git (AWS `git archive`, the pool's synced copy), where git_sha is None."""
import json
import os
import shutil
import socket

from brain import provenance as P


def test_stamp_adds_only_provenance_and_resolves_the_model(tmp_path, monkeypatch):
    ckpt = tmp_path / "rwkv7-g1x-1.5b-test.pth"
    ckpt.write_bytes(b"x" * 10)
    monkeypatch.setattr("brain.cortex.MODEL_DIR", str(tmp_path))
    res = {"seed": 42, "rows": [{"ok": True}]}
    out = P.stamp(res, "1.5b")
    assert out is res and set(res) == {"seed", "rows", "provenance"} and res["rows"] == [{"ok": True}]
    p = res["provenance"]
    assert p["host"] == socket.gethostname()
    assert p["model_arg"] == "1.5b" and p["model_path"] == str(ckpt) and p["model_bytes"] == 10
    assert len(p["src_hash"]) == 64 and p["src_hash"] == P.src_hash()
    assert "git_sha" in p and "git_dirty" in p
    json.dumps(res)                                     # serialisable as written


def test_stamp_never_fails_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr("brain.cortex.MODEL_DIR", str(tmp_path))   # no checkpoint of that size
    p = P.stamp({}, "9.9b")["provenance"]
    assert p["model_path"] is None and "model_error" in p


def test_src_hash_matches_in_a_copy_without_git(tmp_path):
    copy = tmp_path / "brain"
    shutil.copytree(P.BRAIN_DIR, copy, ignore=shutil.ignore_patterns("__pycache__", "results", "transcripts"))
    v = P.code_version.__wrapped__(str(copy))
    assert v["git_sha"] is None and v["git_dirty"] is None
    assert v["src_hash"] == P.src_hash()
    (copy / "brain.py").write_text((copy / "brain.py").read_text() + "\n# edit\n")
    assert P.src_hash(str(copy)) != P.src_hash()
    os.remove(copy / "provenance.py")
    assert P.src_hash(str(copy)) != P.src_hash()
