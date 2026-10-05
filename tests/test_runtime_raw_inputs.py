"""tools/pool/runtime_raw_inputs.txt: the run-time INPUTS that live under research/findings/raw/ (2026-10-04).

Every provisioner excludes research/findings/raw/ as run outputs. A production reader whose default path sits there
falls back silently on a node without the file: the affect organ read the raw Warriner norm and the Stage-1 exit
gate's C4 arms were void (FAILURE_LOG 2026-10-04). These tests pin three things: every listed input is git-tracked,
every provisioner ships the list, and every known production reader's default path is covered by it."""
import os
import re
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIST = os.path.join(_ROOT, "tools", "pool", "runtime_raw_inputs.txt")
_RAW = "research/findings/raw/"

# Production readers of a run-time input under research/findings/raw/: the file, and a regex for its default path.
_READERS = [
    ("research/runners/affect_production_organ.py", r'_LEARNED_MAP_PATH\s*=\s*os\.path\.join\(_REPO,\s*(.+)\)'),
    ("research/runners/affect_learned_vocabulary.py", r'DEFAULT_WEIGHTS_DIR\s*=\s*os\.path\.join\(_REPO,\s*(.+)\)'),
    ("webapp/wkv_mouth_generator.py", r'(research/findings/raw/_persist_eprop_head_scope/[^"]+)'),
    ("research/runners/load_bearing_fraction.py", r'"(research/findings/raw/_affect_tone_open_output/[^"]+)"'),
]


def _entries():
    out = []
    for line in open(_LIST, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _as_path(expr):
    """A path from a regex capture: either a literal path or the arguments of os.path.join("a", "b", ...)."""
    parts = re.findall(r'"([^"]+)"', expr)
    return "/".join(parts) if len(parts) > 1 else (parts[0] if parts else expr)


def _covered(path, entries):
    return any(path == e or path.startswith(e.rstrip("/") + "/") for e in entries)


def test_every_listed_input_is_under_raw_and_git_tracked():
    entries = _entries()
    assert entries, "the runtime-input list is empty"
    for e in entries:
        assert e.startswith(_RAW), e
        tracked = subprocess.run(["git", "ls-files", "--", e], cwd=_ROOT, capture_output=True, text=True).stdout
        assert tracked.strip(), "%s is listed but not tracked in git, so no provisioner can ship it" % e


def test_every_provisioner_ships_the_list():
    for script in ("tools/pool_provision.sh", "tools/aws_provision.sh", "tools/aws_cpu_provision.sh"):
        text = open(os.path.join(_ROOT, script), encoding="utf-8").read()
        assert "tools/pool/runtime_raw_inputs.txt" in text, "%s does not ship the runtime-input list" % script


def test_every_production_reader_default_path_is_covered():
    entries = _entries()
    for rel, pattern in _READERS:
        text = open(os.path.join(_ROOT, rel), encoding="utf-8").read()
        m = re.search(pattern, text)
        assert m, "%s: the default-path pattern no longer matches; update _READERS" % rel
        path = _as_path(m.group(1))
        assert path.startswith(_RAW), (rel, path)
        assert _covered(path, entries), "%s reads %s, which tools/pool/runtime_raw_inputs.txt does not ship" % (rel, path)


def test_coverage_check_fails_in_its_failing_direction():
    assert not _covered("research/findings/raw/_new_map.json", ["research/findings/raw/_affect_learned_valence_map.json"])
    assert _covered("research/findings/raw/_a/run1/w.npz", ["research/findings/raw/_a/run1"])
    assert not _covered("research/findings/raw/_a/run10/w.npz", ["research/findings/raw/_a/run1"])
