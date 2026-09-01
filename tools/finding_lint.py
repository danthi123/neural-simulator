#!/usr/bin/env python3
"""tools/finding_lint.py — AUTHOR-TIME preview of the commit gates for a findings doc.

THE PROBLEM (measured 2026-08-01). Nearly every findings commit was blocked 2-4 times in a
row by tools/gates/ — claim_check (a number cited to a DIRECTORY, not the file that holds it),
doc_type (artifact path only in frontmatter, not body prose), device_and_cost (artifact records
no backend), below_chance, verdict_preconditions, stated_value_mismatch, closure_names_mechanism.
Every block was CORRECT; the first drafts genuinely lacked provenance. This tool surfaces all of
it at authoring time, so a finding is fixed in ONE pass instead of a round-trip per gate.

    tools/finding_lint.py <finding.md> [more.md ...] [extra/artifact.json ...] [--fix] [--quiet]

Exit 0 if the finding would PASS the commit hook; 1 if it would BLOCK. So it doubles as a CI check.

REUSE, NOT REIMPLEMENTATION — this is the load-bearing property, and it is enforced by construction.
This file contains NO copy of any gate's logic. It IMPORTS and CALLS the real modules the hook uses:

  * GATE 5 — the failure-class registry: `tools.gates.run_all` (+ `discover`), the SAME entry point
    tools/githooks/pre-commit invokes. run_all's verdict is authoritative here; a per-gate pass is
    derived from the same `discover()` modules purely to attach fix scaffolding, and is cross-checked
    against run_all every run (a DRIFT line prints if they ever disagree).
  * GATE 2 — claims: `tools.claim_check.check`, the SAME module the hook shells out to. Its stdout is
    captured to name the specific unsupported numbers; the verdict is the module's own return code.
  * GATE 4 — new-finding status: the hook's 2-line inline test (first line `---`, a `status:` field).
    It is not a module, so it is reproduced faithfully and labelled as such.

Because the checks ARE the gate modules, they cannot drift from the gates: change a gate and this
tool changes with it. GATE 1 (document-structure W1/W2) and GATE 3 (biology bindings) are repo-wide
and orthogonal to a single new finding — they check governed docs / biology config, never the
finding itself — so they are intentionally out of scope here (noted, not run).
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.gates import discover, run_all          # the SAME registry the pre-commit hook calls
import tools.claim_check as claim_check            # the SAME GATE 2 module the hook shells out to

# A cited artifact path in prose/frontmatter: must contain a "/" and end in .json/.jsonl (claim_check's rule).
_ART_FILE_RE = re.compile(r"[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:jsonl|json)")
# A bare path token (may be a directory) — used only for frontmatter `artifacts:` items and existence-tested.
_PATH_TOKEN_RE = re.compile(r"[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+/?")

# parsed claim_check stdout
_CC_LINE_RE = re.compile(r"⛔ line\s+(\d+)\s+([-\d.eE+]+)\s+not in any cited artifact\s*\|\s*(.*)")
_CC_MISS_RE = re.compile(r"⛔ MISSING\s+(.*)")

# provenance sources to mine for a backend value (device_and_cost scaffolding)
_BACKEND_IN_TEXT = re.compile(r"SIM_BACKEND\s*[=:]\s*['\"]?(cupy|numpy|cuda|gpu|cpu)['\"]?", re.I)
_BACKEND_KEYS = ("sim_backend", "backend", "device")


# ---------------------------------------------------------------------------------------------------
# artifact resolution: the finding + every artifact it CITES, directories resolved to the files inside
# ---------------------------------------------------------------------------------------------------
def _rel(path):
    """Repo-relative for in-repo paths; ABSOLUTE for anything outside the repo (a tempdir test fixture,
    say) — the gates accept both (`os.path.isabs(p)`), and a `../..` relative path would be fragile."""
    ap = os.path.abspath(path if os.path.isabs(path) else os.path.join(_ROOT, path))
    rel = os.path.relpath(ap, _ROOT).replace("\\", "/")
    return ap if rel.startswith("..") else rel


# ---------------------------------------------------------------------------------------------------
# hook scoping: the pre-commit checks only git diff --cached --diff-filter=A (staged-ADDED) files, so an
# artifact that is already committed, or merely sits untracked in the working tree, is NOT gated. This
# tool reproduces that: the finding is always linted (it is the subject); a cited artifact is gated by
# the registry only if the hook would see it — staged-as-added, or (opt-in) untracked-and-new, or simply
# outside the repo (a fixture the hook could never reach but a caller clearly wants checked).
# ---------------------------------------------------------------------------------------------------
def _git_lines(args):
    try:
        r = subprocess.run(["git"] + args, cwd=_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    if r.returncode != 0:
        return set()
    return {ln.strip().replace("\\", "/") for ln in r.stdout.split("\n") if ln.strip()}


def _git_new_sets():
    return (_git_lines(["diff", "--cached", "--name-only", "--diff-filter=A"]),   # staged-added
            _git_lines(["ls-files", "--others", "--exclude-standard"]))            # untracked


def _scope_to_hook(cited, include_untracked):
    """Keep only the cited artifacts the pre-commit hook would actually check at commit."""
    added, untracked = _git_new_sets()
    gated, skipped = [], []
    for a in cited:
        if os.path.isabs(a):                      # outside the repo -> the hook can't reach it; caller wants it
            gated.append(a)
        elif a in added:                          # staged as ADDED -> exactly what --diff-filter=A sees
            gated.append(a)
        elif include_untracked and a in untracked:
            gated.append(a)
        else:                                     # committed (unmodified/modified) or untracked-not-opted-in
            skipped.append(a)
    return gated, skipped


def _frontmatter(text):
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end < 0:
        return "", text
    return text[3:end], text[end + 4:]


def _frontmatter_artifacts(fm):
    """The `artifacts:` YAML list items — verbatim, dirs and files both."""
    out, in_block = [], False
    for ln in fm.split("\n"):
        if re.match(r"^\s*artifacts:\s*$", ln):
            in_block = True
            continue
        if in_block:
            m = re.match(r"^\s*-\s*(\S+)\s*$", ln)
            if m:
                out.append(m.group(1))
                continue
            if ln.strip() and not ln.startswith((" ", "\t")):
                in_block = False           # next top-level key ends the list
    return out


def _expand(token):
    """A cited token -> the concrete .json/.jsonl files it names (glob-expanded, dirs walked)."""
    token = token.strip().strip("`").rstrip("/")
    if not token:
        return []
    full = token if os.path.isabs(token) else os.path.join(_ROOT, token)
    if any(c in token for c in "*?["):
        return [_rel(h) for h in sorted(glob.glob(full)) if h.endswith((".json", ".jsonl"))]
    if os.path.isdir(full):
        hits = glob.glob(os.path.join(full, "**", "*.json"), recursive=True) \
            + glob.glob(os.path.join(full, "**", "*.jsonl"), recursive=True)
        return [_rel(h) for h in sorted(hits)]
    if full.endswith((".json", ".jsonl")):
        return [_rel(full)]
    return []


def cited_artifacts(finding_path):
    """Every artifact the finding cites: frontmatter `artifacts:` list AND body `.json/.jsonl` paths,
    directories resolved to the files inside. Deduped, repo-relative."""
    text = open(finding_path, errors="ignore").read()
    fm, body = _frontmatter(text)
    tokens = list(_frontmatter_artifacts(fm)) + _ART_FILE_RE.findall(body)
    seen, out = set(), []
    for tok in tokens:
        for f in _expand(tok):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


# ---------------------------------------------------------------------------------------------------
# GATE 2 — claim_check (reuse the module; capture its stdout for the specific unsupported numbers)
# ---------------------------------------------------------------------------------------------------
def run_claim_check(finding_path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = claim_check.check(finding_path, verbose=True)
    text = buf.getvalue()
    unsupported, missing = [], []
    for ln in text.split("\n"):
        m = _CC_LINE_RE.search(ln)
        if m:
            unsupported.append((int(m.group(1)), m.group(2), m.group(3).strip()))
            continue
        m = _CC_MISS_RE.search(ln)
        if m:
            missing.append(m.group(1).strip())
    return rc, unsupported, missing, text


# ---------------------------------------------------------------------------------------------------
# GATE 4 — new-finding status frontmatter (the hook's inline 2-line test, reproduced faithfully)
# ---------------------------------------------------------------------------------------------------
def status_present(finding_path):
    try:
        with open(finding_path, errors="ignore") as fh:
            first = fh.readline().rstrip("\n")
            rest = fh.read()
    except OSError:
        return False
    return first == "---" and bool(re.search(r"^status:[ \t]*[a-z-]+", rest, re.M))


# ---------------------------------------------------------------------------------------------------
# GATE 5 — the registry. run_all() is authoritative; a per-gate pass gives grouping + scaffolding.
# ---------------------------------------------------------------------------------------------------
def run_registry(paths):
    """Per-gate results, mirroring tools.gates.run_all's selftest/crash guard so the pass is faithful.
    run_all() below is the AUTHORITATIVE verdict; this pass exists to group + scaffold, and is
    cross-checked against it."""
    results = []
    for name, mod, err in discover():
        if err:
            results.append({"name": name, "class_id": "!", "blocking": True,
                            "status": "broken", "problems": [err]})
            continue
        st = mod.selftest()
        if st:
            results.append({"name": name, "class_id": getattr(mod, "CLASS_ID", "?"), "blocking": True,
                            "status": "selftest_failed", "problems": ["selftest FAILED: " + "; ".join(st)]})
            continue
        try:
            probs = mod.check(paths)
        except Exception as e:                         # a crashing gate is loud, never silently absent
            results.append({"name": name, "class_id": getattr(mod, "CLASS_ID", "?"), "blocking": True,
                            "status": "crashed", "problems": ["CRASHED: %s: %s" % (type(e).__name__, e)]})
            continue
        results.append({"name": name, "class_id": getattr(mod, "CLASS_ID", "?"),
                        "blocking": bool(getattr(mod, "BLOCKING", True)),
                        "status": "problems" if probs else "ok", "problems": list(probs)})
    return results


def _blocking_gate_status(blocking_gate):
    """Read status from either a direct gate result or its report wrapper."""
    return blocking_gate.get("status", blocking_gate.get("gate", {}).get("status"))


# ---------------------------------------------------------------------------------------------------
# scaffolding — concrete fixes, printed; the safe mechanical ones applied only under --fix
# ---------------------------------------------------------------------------------------------------
def _load_json(path):
    full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    try:
        return json.load(open(full, errors="ignore"))
    except (OSError, ValueError):
        return None


def _top_numeric(obj):
    """Top-level numeric scalars of a dict artifact (ints/floats, not bools)."""
    if not isinstance(obj, dict):
        return {}
    return {k: float(v) for k, v in obj.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _seed_means(artifact_paths):
    """mean over cited per-seed files, per shared top-level numeric key: {key: (mean, [values])}."""
    per_key = {}
    for p in artifact_paths:
        vals = _top_numeric(_load_json(p))
        for k, v in vals.items():
            per_key.setdefault(k, []).append(v)
    return {k: (sum(v) / len(v), v) for k, v in per_key.items() if len(v) >= 2}


def scaffold_claim_check(finding_path, unsupported, missing, artifact_paths, do_fix, emit):
    """A number in no cited artifact -> cite the file that holds it, mark <!--derived-->, or (the single
    most common miss) it is a MEAN over per-seed files that lives in no per-seed file."""
    if missing:
        emit("    FIX: these cited paths do not exist on disk — correct the path or add the artifact:")
        for m in missing[:8]:
            emit("         - %s" % m)
    if not unsupported:
        return
    means = _seed_means(artifact_paths)
    mean_hits = []
    for lineno, valstr, ctx in unsupported:
        try:
            val = float(valstr)
        except ValueError:
            val = None
        matched_key = None
        if val is not None:
            for k, (mean, vals) in means.items():
                if abs(val - mean) <= max(5e-6, 1e-4 * abs(val)):
                    matched_key = (k, mean, vals)
                    break
        if matched_key:
            k, mean, vals = matched_key
            mean_hits.append((lineno, val, k, mean, vals))
            emit("    line %-4s %-12s looks like the MEAN of `%s` over the cited per-seed files "
                 "(%s) — which lives in NO per-seed file." % (lineno, valstr, k,
                 "/".join("%.4g" % x for x in vals)))
        else:
            emit("    line %-4s %-12s is in no cited artifact. Either cite the artifact FILE that holds "
                 "it (a path with a '/'), or mark it <!--derived--> inline on the same line." % (lineno, valstr))
    if mean_hits:
        agg_rel = _suggest_aggregate_path(finding_path)
        emit("    FIX (the aggregate miss): means over seeds belong in an aggregate JSON you cite.")
        emit("         suggested path : %s" % agg_rel)
        emit("         then add a body line, e.g.:  Aggregate: `%s`" % agg_rel)
        if do_fix:
            written = _write_aggregate(agg_rel, artifact_paths, means)
            if written:
                emit("    --fix: wrote aggregate  %s  (mean + per-seed for %d keys, from %d cited files)"
                     % (agg_rel, len(means), len(artifact_paths)))
                emit("           NOW cite it in the finding body (line above) and re-run finding_lint.")
            else:
                emit("    --fix: could not write the aggregate (no cited per-seed files parsed).")


def _suggest_aggregate_path(finding_path):
    base = os.path.basename(finding_path)
    stem = re.sub(r"\.md$", "", base)
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)[:48]
    return "research/findings/raw/%s/_finding_lint_aggregate.json" % stem


def _write_aggregate(agg_rel, artifact_paths, means):
    if not means:
        return False
    full = os.path.join(_ROOT, agg_rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    doc = {
        "_note": "aggregate GENERATED by tools/finding_lint.py from the cited per-seed files; "
                 "means/per-seed computed, not measured anew. Cite THIS file for the means.",
        "source_files": list(artifact_paths),
        "n_files": len(artifact_paths),
        "means": {k: round(mean, 6) for k, (mean, _v) in means.items()},
        "per_seed": {k: v for k, (_m, v) in means.items()},
        "provenance_exempt": "derived aggregate over cited artifacts (finding_lint)",
    }
    # An aggregate OVER runs inherits its runs' device — so the aggregate does not itself trip
    # device_and_cost. Recorded ONLY if the sources UNANIMOUSLY name one (in the artifact or a sibling
    # log); never guessed, never invented when they disagree.
    backends = set()
    for p in artifact_paths:
        b = _search_backend_obj(_load_json(p) or {}) or _find_backend_for(p)[0]
        if b:
            backends.add(str(b).lower())
    if len(backends) == 1:
        doc["sim_backend"] = next(iter(backends))
        doc["_sim_backend_source"] = "inherited by finding_lint from the aggregated per-seed files (unanimous)"
    json.dump(doc, open(full, "w"), indent=2)
    return True


def _find_backend_for(artifact_rel):
    """Look for a backend value in THIS artifact's own provenance neighbours — never a global file, never
    guessed. A shared dispatch log is trusted ONLY on a line that also names this artifact, so one run's
    backend is never misattributed to another. Returns (value, source_rel) or (None, None)."""
    full = artifact_rel if os.path.isabs(artifact_rel) else os.path.join(_ROOT, artifact_rel)
    stem = re.sub(r"\.json$", "", full)
    base = os.path.splitext(os.path.basename(full))[0]
    # 1. per-artifact sidecars (JSON) with a backend key — same stem, unambiguously this run's
    for sfx in (".prov.json", ".cmd.json", ".provenance.json"):
        for cand in (full + sfx, stem + sfx):
            obj = _load_json(cand)
            if isinstance(obj, dict):
                v = _search_backend_obj(obj)
                if v:
                    return v, _rel(cand)
    # 2. a SIBLING log with the same stem — unambiguously this run's
    for cand in (full + ".log", stem + ".log"):
        if os.path.exists(cand):
            try:
                m = _BACKEND_IN_TEXT.search(open(cand, errors="ignore").read(65536))
            except OSError:
                m = None
            if m:
                return m.group(1).lower(), _rel(cand)
    # 3. a shared dispatch log — ONLY a line that also NAMES this artifact (else it is someone else's run)
    for cand in (os.path.join(os.path.dirname(full), "dispatch.log"),
                 os.path.join(_ROOT, "research", "queue", "dispatch.log")):
        if not os.path.exists(cand):
            continue
        try:
            for ln in open(cand, errors="ignore"):
                if base in ln:
                    m = _BACKEND_IN_TEXT.search(ln)
                    if m:
                        return m.group(1).lower(), _rel(cand) + " (line naming %s)" % base
        except OSError:
            continue
    return None, None


def _search_backend_obj(obj, depth=0):
    if not isinstance(obj, dict) or depth > 3:
        return None
    for k, v in obj.items():
        if str(k).lower() in _BACKEND_KEYS and isinstance(v, str) and v:
            return v
        if str(k).upper() == "SIM_BACKEND" and isinstance(v, str) and v:
            return v
    for v in obj.values():
        if isinstance(v, dict):
            got = _search_backend_obj(v, depth + 1)
            if got:
                return got
    return None


def scaffold_device_and_cost(problems, do_fix, emit):
    for prob in problems:
        m = re.match(r"([^:]+):\s*records NO backend", prob)
        if not m:
            continue
        art = m.group(1).strip()
        val, src = _find_backend_for(art)
        if val:
            emit("    FIX: %s records no backend, but provenance names one: `%s` (from %s)." % (art, val, src))
            if do_fix:
                if _stamp_backend(art, val, src):
                    emit("    --fix: stamped  sim_backend=\"%s\"  into %s (RECORDED from provenance, not guessed)."
                         % (val, art))
                else:
                    emit("    --fix: could not stamp %s (unreadable or not a JSON object)." % art)
            else:
                emit("         run with --fix to stamp it (recorded-from-provenance, never guessed).")
        else:
            emit("    FIX: %s records no backend and no sibling .prov/.cmd/.log supplies one. Re-run under "
                 "tools.lab.assert_backend, or let research/runners/__init__ emit a .prov.json sidecar. "
                 "Do NOT hand-set a backend you cannot source." % art)


def _stamp_backend(artifact_rel, value, source_rel):
    full = artifact_rel if os.path.isabs(artifact_rel) else os.path.join(_ROOT, artifact_rel)
    obj = _load_json(full)
    if not isinstance(obj, dict):
        return False
    obj["sim_backend"] = value
    obj["_sim_backend_source"] = "recorded by finding_lint from %s (not guessed)" % source_rel
    json.dump(obj, open(full, "w"), indent=2)
    return True


def scaffold_doc_type(finding_path, problems, artifact_paths, emit):
    for prob in problems:
        if "cites no artifact path" in prob:
            example = artifact_paths[0] if artifact_paths else "research/findings/raw/<run>/<seed>.json"
            emit("    FIX: the artifact is only in frontmatter — doc_type needs a path in the BODY prose.")
            emit("         add a line to the body, e.g.:  Artifact: `%s`" % example)


# ---------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------
def lint_one(finding_path, extra_paths, do_fix, quiet, include_untracked):
    """Return True if this finding WOULD BLOCK the commit hook."""
    finding_rel = _rel(finding_path)
    out = []
    emit = out.append
    emit("=" * 96)
    emit("finding_lint: %s" % finding_rel)

    arts_all = cited_artifacts(finding_path)               # every cited artifact (dirs resolved)
    arts_gated, arts_skipped = _scope_to_hook(arts_all, include_untracked)
    extras = [_rel(p) for p in extra_paths]
    paths = [finding_rel] + arts_gated + extras
    emit("  cited artifacts: %d resolved | %d gated now (staged-added%s) | %d skipped (committed/untracked, "
         "as the hook does)%s"
         % (len(arts_all), len(arts_gated), " + untracked" if include_untracked else "",
            len(arts_skipped), ("  +%d extra" % len(extras)) if extras else ""))
    if arts_skipped and not include_untracked:
        emit("  note: %d cited artifact(s) are committed or untracked-unstaged — the hook skips them "
             "(--diff-filter=A). Pass --include-untracked to gate the untracked ones as a pre-staging preview."
             % len(arts_skipped))

    # GATE 2 — claims
    cc_rc, cc_unsupported, cc_missing, _cc_text = run_claim_check(finding_path)
    # GATE 4 — status
    g4_ok = status_present(finding_path)
    # GATE 5 — registry (authoritative verdict) + per-gate pass (grouping/scaffolding)
    auth_blocking, _auth_report = run_all(paths, verbose=False)
    per_gate = run_registry(paths)

    # cross-check: the per-gate pass must agree with run_all on the would-block verdict (drift guard)
    my_g5_block = any(g["blocking"] and g["problems"] for g in per_gate)
    if bool(auth_blocking) != my_g5_block:
        emit("  ⚠️  DRIFT: run_all=%s vs per-gate=%s — the grouping pass diverged from the registry entry "
             "point; trust run_all (authoritative)." % (bool(auth_blocking), my_g5_block))

    # assemble blocking gates, most-blocking first
    blocking_gates = []
    if cc_rc != 0:
        probs = ["line %s: %s  (%s)" % (n, v, c[:60]) for n, v, c in cc_unsupported] \
            + ["MISSING artifact: %s" % m for m in cc_missing] \
            or ["a measurement is unsupported by the cited artifacts (see claim_check)"]
        blocking_gates.append({"name": "claim-check", "class_id": "G2", "problems": probs, "kind": "claim"})
    if not g4_ok:
        blocking_gates.append({"name": "status-frontmatter", "class_id": "G4",
                               "problems": ["new finding declares no `status:` (first line `---`, then "
                                            "`status: live|qualified|corrected|superseded|retracted`)"],
                               "kind": "status"})
    for g in per_gate:
        if g["blocking"] and g["problems"]:
            blocking_gates.append({"name": g["name"], "class_id": g["class_id"],
                                   "problems": g["problems"], "kind": "registry", "gate": g})
    blocking_gates.sort(key=lambda g: -len(g["problems"]))

    warn_gates = [g for g in per_gate if not g["blocking"] and g["problems"]]

    would_block = bool(blocking_gates)

    if would_block:
        emit("  VERDICT: ⛔ WOULD BLOCK at commit — %d blocking gate(s):" % len(blocking_gates))
    else:
        emit("  VERDICT: ✔ would pass the commit hook (GATE 2 + GATE 4 + GATE 5 registry).")

    for g in blocking_gates:
        emit("")
        emit("  ⛔ %-24s [%s]  %d problem(s)" % (g["name"], g["class_id"], len(g["problems"])))
        for p in g["problems"][:10]:
            emit("      - %s" % p)
        if len(g["problems"]) > 10:
            emit("      ... and %d more" % (len(g["problems"]) - 10))
        # scaffolding
        if g["kind"] == "claim":
            scaffold_claim_check(finding_path, cc_unsupported, cc_missing, arts_all, do_fix, emit)
        elif g["name"] == "stated-value-mismatch":
            emit("    FIX: a NAMED quantity disagrees with the artifact's own value — quote the artifact's "
                 "number, or fix the prose. (Existence is not agreement; claim_check cannot catch this.)")
        elif g["name"] == "device-and-cost":
            scaffold_device_and_cost(g["problems"], do_fix, emit)
        elif g["name"] == "doc-type":
            scaffold_doc_type(finding_path, g["problems"], arts_all, emit)
        elif g["name"] == "closure-names-mechanism":
            emit("    FIX: add `mechanism: <id>` to the frontmatter (a closure claim needs a mechanism so "
                 "biology_check can adjudicate it against other live claims).")
        elif g["name"] == "verdict-preconditions":
            emit("    FIX: the artifact asserts a verdict with no/failed/unmeasured `preconditions` block — "
                 "emit one via tools.verdict.Verdict; a failed precondition means the verdict is UNDEFINED.")
        elif g["name"] == "below-chance":
            emit("    FIX: a cited cell is at/below chance — do not read a verdict off it; mark it or drop "
                 "the claim. (If the collapse IS the result, say so and flag the cell, as the reference does.)")
        elif _blocking_gate_status(g) in ("broken", "selftest_failed", "crashed"):
            emit("    NOTE: this gate is itself broken — its verdict is not trusted until its selftest passes.")

    if warn_gates and not quiet:
        emit("")
        emit("  ⚠️  non-blocking warnings (reported, will NOT block the commit):")
        for g in warn_gates:
            emit("      %-24s [%s]  %d — %s" % (g["name"], g["class_id"], len(g["problems"]),
                                                g["problems"][0][:80]))

    if not quiet or would_block:
        print("\n".join(out))
    return would_block


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Author-time preview of the commit gates for a findings doc (reuses tools/gates + "
                    "tools/claim_check). Exit 1 if the finding would block at commit.")
    ap.add_argument("paths", nargs="+", help="finding .md file(s); any non-.md path is added as an extra artifact")
    ap.add_argument("--fix", action="store_true",
                    help="apply the SAFE mechanical fixes: generate a cited aggregate JSON for mean-misses, "
                         "and stamp a backend recorded in provenance. Never edits finding prose; never guesses.")
    ap.add_argument("--quiet", action="store_true", help="print only findings that would block")
    ap.add_argument("--include-untracked", action="store_true",
                    help="also gate cited artifacts that are UNTRACKED (a pre-staging preview: shows what "
                         "would block once you `git add` them). Default matches the hook: staged-added only.")
    args = ap.parse_args(argv)

    findings = [p for p in args.paths if p.endswith(".md")]
    extras = [p for p in args.paths if not p.endswith(".md")]
    if not findings:
        ap.error("no .md finding given (positional args ending in .md are the findings to lint)")

    any_block = False
    for f in findings:
        if not os.path.exists(f if os.path.isabs(f) else os.path.join(_ROOT, f)):
            print("finding_lint: no such file: %s" % f, file=sys.stderr)
            any_block = True
            continue
        any_block |= lint_one(f, extras, args.fix, args.quiet, args.include_untracked)
    return 1 if any_block else 0


if __name__ == "__main__":
    sys.exit(main())
