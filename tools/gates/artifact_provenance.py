"""CLASS P — ARTIFACT PROVENANCE: a result JSON that cannot say what produced it.

THE FAILURE CLASS. 645 of 1330 runners take `--out`, so the destination lives only in argv and is never written
into the artifact. MEASURED on this tree (7130 result JSONs under `research/findings/raw/`): only **393 (5.5%)
have a sibling `.cmd.json`**; **50.7%** carry a recognised provenance key at the TOP level. ("Carries some key
at any depth" reads 94% — a per-trial `seed` buried in a results array — which is exactly the kind of figure
that flatters.) CONSEQUENCE, both real: ~94 GPU-hours re-deriving an already-banked NO-GO, and a runner writing
UNCONDITIONALLY to the path of a banked 6-seed GPU GO, where a CPU smoke test would have clobbered it silently.

THE GATE. A **newly added** `*.json` under `research/findings/raw/` must carry provenance, satisfied by ANY of:
  · a sibling `<name>.cmd.json` / `<name>.provenance.json`;
  · a STRONG key within 3 nesting levels: cmd·argv·command·cmdline·provenance·runner·run_id·preset·script;
  · a WEAK key (seed·seeds·config·cfg·params·args) at the top level or ONE level in. Weak keys are depth-capped
    on purpose: `{"arms":{"lesion":{"trial":{"seed":1}}}}` records a seed and still cannot name its producer;
  · an explicit top-level `"provenance_exempt": "<reason>"` for the rare non-run artifact.

CALIBRATION. Fires only on paths git reports as ADDED, so the 12455 files already here never fire and a
*modified* artifact never fires. With no paths it returns NO problems — the corpus rate is INFORMATION
(`corpus_rate()`, or `python -m tools.gates.artifact_provenance`); emitting 2800 problems on an unrelated commit
is how a gate gets switched off. Spot-checked against the real corpus: rejected files look like `{"runs": ...}`,
`{"rows":…, "agg":…}` — genuinely producer-less, not noise.

WHAT THIS GATE CANNOT CATCH.
  · Whether the recorded provenance is TRUE. `"runner": "foo"` written by bar passes. Presence, not honesty.
  · The overwrite itself — it cannot stop a runner clobbering a banked GO, only ensure the survivor names its
    producer. A write-guard on `--out` is the complementary fix and is NOT implemented here.
  · Artifacts outside `research/findings/raw/`, non-JSON artifacts (`.npz`/`.log`/`.csv`), and anything never
    staged — a pre-commit gate cannot see an unstaged file by construction.
  · Retroactive repair of the existing corpus: deliberately out of scope, reported by `corpus_rate()` instead.
"""
from __future__ import annotations

import json
import os
import subprocess

NAME = "artifact-provenance"
CLASS_ID = "P"
BLOCKING = True

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_MARK = "research/findings/raw/"
_STRONG = {
    "cmd", "argv", "command", "command_envelope", "cmdline", "execution_receipt",
    "provenance", "runner", "run_id", "preset", "script",
}
_WEAK = {"seed", "seeds", "config", "cfg", "params", "args"}
_WEAK_MAX_DEPTH = 1
_STRONG_MAX_DEPTH = 3
# `.prov.json` is what research/runners/__init__.py -- the automatic provenance door -- actually writes. The
# gate was authored without knowledge of that door and accepted only the two hand-written forms, so EVERY
# artifact produced by the automatic capture was reported as unprovenanced. Two halves of one system that did
# not agree on a filename; caught when the gate blocked three pool artifacts that DID have sidecars.
_SIDECARS = (".cmd.json", ".provenance.json", ".prov.json")


def _is_artifact(path: str) -> bool:
    p = path.replace("\\", "/")
    return p.endswith(".json") and _RAW_MARK in p and not any(p.endswith(s) for s in _SIDECARS)


def _scan(obj, depth: int = 0) -> bool:
    """True if a recognised provenance key sits within its allowed nesting depth."""
    if depth > _STRONG_MAX_DEPTH:
        return False
    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj}
        if (_STRONG & keys) or (depth <= _WEAK_MAX_DEPTH and (_WEAK & keys)):
            return True
        return any(_scan(v, depth + 1) for v in list(obj.values())[:200])
    if isinstance(obj, list):
        return any(_scan(v, depth + 1) for v in obj[:50])
    return False


def _has_provenance(path: str):
    """(ok, why_not). An unparseable file reports WHY rather than silently passing or silently failing."""
    # TWO NAMING CONVENTIONS, both legitimate: the automatic door (research/runners/__init__.py) appends to
    # the FULL path -> "x.json.prov.json"; hand-written sidecars replace the extension -> "x.cmd.json".
    # Accepting only one silently reported every artifact of the other kind as unprovenanced. This gate and
    # that door are two halves of one system, written by different authors, and they disagreed twice --
    # first on the suffix, then on how it attaches.
    _stem = path[: -len(".json")]
    if any(os.path.exists(_stem + sfx) or os.path.exists(path + sfx) for sfx in _SIDECARS):
        return True, ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(65536)
        if any('"%s"' % k in head for k in sorted(_STRONG)):
            return True, ""
        return False, "unparseable JSON (%s) with no provenance key in its first 64KB" % type(e).__name__
    if isinstance(data, dict) and isinstance(data.get("provenance_exempt"), str) and data["provenance_exempt"]:
        return True, ""
    if _scan(data):
        return True, ""
    return False, ("no sibling .cmd.json/.prov.json and no %s key (weak keys count only at depth<=%d)"
                   % ("/".join(sorted(_STRONG | _WEAK)), _WEAK_MAX_DEPTH))


def _not_added() -> set:
    """Staged paths whose git status is NOT 'A'. Unknown paths are CHECKED, so a git failure cannot mute us."""
    try:
        r = subprocess.run(["git", "diff", "--cached", "--name-status", "-z"], cwd=_REPO_ROOT,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    if r.returncode != 0:
        return set()
    fields, skip, i = [f for f in r.stdout.split("\0") if f], set(), 0
    while i + 1 < len(fields):
        status, i = fields[i], i + 1
        n = 2 if status[:1] in ("R", "C") else 1                  # rename/copy carry <old> <new>
        if status[:1] != "A":
            skip.update(p.replace("\\", "/") for p in fields[i:i + n])
        i += n
    return skip


def check(paths) -> list:
    if not paths:
        return []                                                 # the corpus rate is information, not 2800 problems
    skip, problems = _not_added(), []
    for path in paths:
        if not _is_artifact(path) or not os.path.exists(path):
            continue
        rel = os.path.relpath(os.path.abspath(path), _REPO_ROOT).replace("\\", "/")
        if rel in skip or path.replace("\\", "/") in skip:
            continue
        ok, why = _has_provenance(path)
        if not ok:
            problems.append("CLASS P no provenance: %s — %s. Fix: write sys.argv (or a {runner, seed, config} "
                            "block) INTO the artifact, or emit a sibling <name>.cmd.json." % (path, why))
    return problems


def corpus_rate(limit: int = 1200) -> dict:
    """Informational: what fraction of the EXISTING raw corpus can name its producer. Never a problem."""
    files = []
    for root, _d, names in os.walk(os.path.join(_REPO_ROOT, "research", "findings", "raw")):
        files += [os.path.join(root, n) for n in names if _is_artifact(os.path.join(root, n))]
    files.sort()
    sample = files[:: max(1, len(files) // limit)] if files else []
    ok = sum(1 for f in sample if _has_provenance(f)[0])
    return {"corpus": len(files), "sampled": len(sample),
            "with_provenance_pct": round(100.0 * ok / len(sample), 1) if sample else 0.0}


def selftest() -> list:
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "research", "findings", "raw")
        os.makedirs(raw)

        def w(name, text):
            p = os.path.join(raw, name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            return p

        # --- THE FAILING DIRECTION FIRST: cases the gate MUST catch ---
        must_catch = {"bare result": w("a.json", '{"final_score": 0.91, "n": 6}'),
                      "deep weak key": w("b.json", '{"arms": {"lesion": {"trial": {"seed": 1}}}}'),
                      "empty exemption": w("c.json", '{"provenance_exempt": "", "score": 1}'),
                      "unparseable, no key": w("d.json", '{"score": 1, ')}
        for label, p in must_catch.items():
            if not check([p]):
                bad.append("GATE CANNOT FAIL: %s (%s) produced no problem" % (label, os.path.basename(p)))

        # --- only then the passing direction: cases it must NOT cry wolf on ---
        w("e.cmd.json", '{"cmd": ["python", "-m", "r"]}')
        outside = os.path.join(td, "elsewhere.json")
        with open(outside, "w", encoding="utf-8") as fh:
            fh.write('{"score": 1}')
        must_pass = {"top-level argv": w("f.json", '{"argv": ["--seed", "42"], "score": 1}'),
                     "sibling cmd.json": w("e.json", '{"score": 1}'),
                     "nested runner": w("g.json", '{"meta": {"runner": "g11_bg_runner"}, "score": 1}'),
                     "top-level seed": w("h.json", '{"seed": 42, "score": 1}'),
                     "explicit exemption": w("i.json", '{"provenance_exempt": "hand-written fixture"}'),
                     "not an artifact (.md)": w("j.md", "no provenance here"),
                     "the sidecar itself": os.path.join(raw, "e.cmd.json"),
                     "json outside raw/": outside}
        must_pass["evidence manifest references command"] = w(
            "manifest.json", '{"command_envelope": {"path": "commands/run.json", "sha256": "abc"}}'
        )
        for label, p in must_pass.items():
            probs = check([p])
            if probs:
                bad.append("FALSE POSITIVE: %s flagged — %s" % (label, probs[0][:90]))

        mixed = check([must_catch["bare result"], must_pass["top-level argv"], must_pass["not an artifact (.md)"]])
        if len(mixed) != 1:
            bad.append("batch check returned %d problems, expected exactly 1" % len(mixed))
        if check([]):
            bad.append("check([]) returned problems; the corpus rate must be information, not a blocker")
    return bad


if __name__ == "__main__":                                        # the informational view, never a blocker
    print("class P artifact provenance — %s" % corpus_rate())
