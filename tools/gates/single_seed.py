"""FAILURE CLASS 9 — single-seed headline (4 recorded incidents).

EVIDENCE. "The headline was reported all day from seed 42 alone"; the 6-seed replication then gave **+0.234**
against the quoted **+0.282**, per-seed spread **0.094–0.375** — the quote was the top of the range, not the
effect. Separately, "0.7050 (n=6)" was a favourable-seed artifact. Project standard: **6 seeds
(42/43/44/100/101/102)** before any generalisation (memory: `feedback_6seed_validation`).

THE GATE. A finding whose HEADLINE asserts a positive verdict (a bare `GO`, or SOLVED/WORKS/CONFIRMED/
VALIDATED/BREAKTHROUGH/SURPASSED/CLOSED/SUCCESS in caps) must evidence >= 6 seeds, by any of: a textual
`n=6` / `6 seeds` / `18-seed` claim; >= 6 distinct seed ids in prose (`--seeds 42 43 44 100 101 102`); or a
CITED `research/findings/raw/**` artifact (globs expanded) whose JSON carries `n_seeds >= 6` or >= 6 distinct
`seed` values. A labelled single-seed probe is legitimate, so a `seed-waiver: <reason>` line passes. If the
headline zone states its OWN seed count and it is < 6, that self-declaration wins over any artifact.

SCOPE LIMIT — DELIBERATE. Only findings carrying YAML frontmatter are checked. The ~1841 legacy findings have
none and are NOT scanned: retro-firing on a thousand historical documents is the cry-wolf failure that gets a
gate ignored. Frontmatter `status` retracted/superseded/corrected/void, and headlines already marked
⛔/RETRACT/VOID/NO-GO/NEGATIVE/REFUTED, are skipped — that claim is withdrawn or negative already.

WHAT THIS GATE CANNOT CATCH.
  * A headline that states NO seed count of its own while the document cites some >= 6-seed artifact for an
    unrelated point — seed evidence is not bound to the specific claim. Deliberately lenient: binding
    claim→artifact is `claim_check`'s job, and firing on every cross-reference would be noise.
  * Six seeds that are not INDEPENDENT (a seed reused, or `actual_seed_used` set instead of `cfg.seed`, which
    seeds nothing — see CLAUDE.md). Counting six values proves reporting, not six distinct substrates.
  * A GO stated only in a commit message, the board, or chat — this reads findings files only.
  * Cherry-picking WITHIN a reported 6: it counts seeds, it does not look at the spread.
  * A positive verdict phrased in lower case or in words outside the marker list.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re

NAME = "single-seed"
CLASS_ID = "9"
BLOCKING = True

_STD_SEEDS = "42/43/44/100/101/102"
_POSITIVE = re.compile(r"\b(SOLVED|WORKS|CONFIRMED|VALIDATED|BREAKTHROUGH|SURPASSED|CLOSED|SUCCESS)\b")
_NEGATIVE = re.compile(r"⛔|\b(RETRACT\w*|VOID|NO-GO|NEGATIVE|REFUTED|CONFOUNDED|WITHDRAWN|FALSE)\b")
_GO_FALSE_FRIENDS = re.compile(r"NO[-/ ]GO|GO[-/ ]NO[-/ ]GO|GO[- ]gates?", re.I)
_WAIVER = re.compile(r"seed[-_ ]waiver\s*:", re.I)
_N_SEEDS = re.compile(r"(\d+)\s*[-–]?\s*seeds?\b", re.I)
_N_EQUALS = re.compile(r"\bn\s*=\s*(\d+)")
_SEED_IDS = re.compile(r"seeds?[\s=:_-]+((?:\d+[\s,]+)*\d+)", re.I)
_ARTIFACT = re.compile(r"research/findings/raw/[^\s`\"')\]]+\.json")


def _frontmatter(text):
    """The YAML block, as a flat lowercase dict, or None when the file carries no frontmatter."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    fm = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            fm[k.strip().lower()] = v.strip().strip('"').strip("'")
    return fm


def _headline_zone(text):
    """Titles and Status/Verdict lines from the head of the document — the CLAIM, not the body prose."""
    out = []
    for line in text.splitlines()[:80]:
        s = line.strip()
        # `search`, not `match`: these docs write "**Date:** ... · **Status:** MEASURED, 3 seeds ..." on one
        # line, so anchoring at column 0 silently dropped every status line in the real corpus.
        if s.startswith("#") or re.search(r"\*\*(Status|Verdict|Result|Headline)\b", s, re.I):
            out.append(s)
    return "\n".join(out)


def _asserts_positive(zone):
    if _NEGATIVE.search(zone):
        return None
    m = _POSITIVE.search(zone)
    if m:
        return m.group(1)
    if re.search(r"(?<![A-Za-z-])GO(?![-A-Za-z])", _GO_FALSE_FRIENDS.sub("", zone)):
        return "GO"
    return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _walk_seeds(obj, ids, best, depth=0):
    """Collect `seed` values and the largest `n_seeds` anywhere in an artifact's JSON."""
    if depth > 6:
        return best
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl == "seed" and _is_int(v):
                ids.add(v)
            elif kl in ("seeds", "seed_list") and isinstance(v, list):
                ids.update(e for e in v if _is_int(e))
            elif kl in ("n_seeds", "num_seeds") and _is_int(v):
                best = max(best, v)
            best = _walk_seeds(v, ids, best, depth + 1)
    elif isinstance(obj, list):
        for e in obj[:400]:
            best = _walk_seeds(e, ids, best, depth + 1)
    return best


def _seed_evidence(text, root):
    """(largest stated seed count, distinct seed ids) from the prose AND every cited artifact."""
    counts = [int(m) for m in _N_SEEDS.findall(text)]
    counts += [int(m) for m in _N_EQUALS.findall(text) if 1 <= int(m) <= 24]  # n=400 is neurons, not seeds
    ids = set()
    for grp in _SEED_IDS.findall(text):
        ids.update(int(t) for t in re.findall(r"\d+", grp))
    files = []
    for rel in set(_ARTIFACT.findall(text)):
        files += sorted(_glob.glob(os.path.join(root, rel)))[:60]
    for path in files[:60]:
        try:
            if os.path.getsize(path) > 8_000_000:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            # An unreadable/corrupt artifact contributes NO seed evidence, so the gate fires rather than
            # passing on a file it could not read. Artifact integrity itself is claim_check's class, not ours.
            continue
        counts.append(_walk_seeds(data, ids, 0))
    return (max(counts) if counts else 0), ids


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _label(path):
    p = path.replace(os.sep, "/")
    return p.split("research/findings/")[-1] if "research/findings/" in p else p


def check(paths):
    problems = []
    cands = [p for p in (paths or []) if p.endswith(".md") and "research/findings/" in p.replace(os.sep, "/")
             and "/raw/" not in p.replace(os.sep, "/")]
    if not paths:
        cands = sorted(_glob.glob(os.path.join(_repo_root(), "research", "findings", "*.md")))
    for path in cands:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue                                    # a staged path may be a deletion; not this gate's class
        fm = _frontmatter(text)
        if fm is None or fm.get("status", "").lower() in ("retracted", "superseded", "corrected", "void"):
            continue
        if _WAIVER.search(text) or any(k in fm for k in ("seed-waiver", "seed_waiver")):
            continue
        zone = _headline_zone(text)
        verdict = _asserts_positive(zone)
        if not verdict:
            continue
        # The document's OWN declared evidence base wins. A Status line reading "3 seeds" under a SOLVED
        # headline is a self-declared violation, and a 6-seed artifact cited elsewhere for another point does
        # not repair it. Only fires when EVERY seed count stated in the headline zone is < 6.
        declared = [int(m) for m in _N_SEEDS.findall(zone)]
        if declared and max(declared) < 6:
            n, how = max(declared), "on its own declared"
        else:
            root = path.replace(os.sep, "/").split("research/findings/")[0] or "."
            cnt, ids = _seed_evidence(text, root)
            n, how = max(cnt, len(ids)), "but the document and its cited artifacts evidence only"
            if n >= 6:
                continue
        problems.append(
            "%s: headline asserts %s %s %d seed(s) — project standard is 6 (%s). Replicate at 6 seeds, "
            "cite a >=6-seed artifact, or add a 'seed-waiver: <reason>' line."
            % (_label(path), verdict, how, n, _STD_SEEDS))
    return problems


def selftest():
    """FAILING DIRECTION FIRST: a one-seed GO headline the gate MUST catch, then the calibration cases."""
    import tempfile

    bad = ("---\nstatus: live\nlane: gap#4\n---\n\n# gap#4 SOLVED: the readout WORKS\n\n"
           "**Status:** MEASURED, seed 42 · a 6-fold improvement over the n=400 baseline\n")
    fixtures = {
        # name                  body                                            expect a problem?
        "a_violating.md":      (bad,                                            True),
        "b_six_seeds.md":      (bad + "\nReplicated over 6 seeds (%s).\n" % _STD_SEEDS,   False),
        "c_seed_ids.md":       (bad + "\nRun with --seeds 42 43 44 100 101 102\n",        False),
        "d_waiver.md":         (bad + "\nseed-waiver: labelled single-seed pilot probe\n", False),
        "e_negative.md":       (bad.replace("SOLVED", "⛔ NOT SOLVED"),                    False),
        "f_retracted.md":      (bad.replace("status: live", "status: retracted"),         False),
        "g_no_frontmatter.md": (bad.split("---\n\n", 1)[1],                               False),
        "h_artifact.md":       (bad + "\nEvidence: `research/findings/raw/t/agg.json`\n", False),
        # the declared-base rule: a 3-seed Status line is NOT repaired by a 6-seed artifact cited elsewhere
        "i_declared_3.md":     (bad.replace("seed 42", "3 seeds × 6 cells")
                                + "\nCross-ref: `research/findings/raw/t/agg.json`\n",   True),
        "j_declared_6.md":     (bad.replace("seed 42", "6 seeds"),                       False),
    }
    probs = []
    with tempfile.TemporaryDirectory() as td:
        fdir = os.path.join(td, "research", "findings")
        os.makedirs(os.path.join(fdir, "raw", "t"))
        with open(os.path.join(fdir, "raw", "t", "agg.json"), "w") as fh:
            json.dump({"seeds": [{"seed": s, "v": 1.0} for s in (42, 43, 44, 100, 101, 102)]}, fh)
        for name, (body, _) in fixtures.items():
            with open(os.path.join(fdir, name), "w") as fh:
                fh.write(body)
        for name, (_, should_fire) in fixtures.items():
            got = bool(check([os.path.join(fdir, name)]))
            if got != should_fire:
                probs.append("fixture %s: expected %s, gate returned %s"
                             % (name, "A PROBLEM" if should_fire else "no problem", "a problem" if got else "none"))
    return probs
