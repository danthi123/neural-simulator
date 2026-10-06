#!/usr/bin/env python3
"""b2d_score.py -- the committed scorer for Battery B2d (tag b2d0929), generalized from `tools/b2b_score.py`.

Governing text: research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md, "Scoring code" and
"Three scores". Reads the SAME cells and arm files three ways, all from disk, none from a rebuilt brain:

  S0 -- raw, as registered. `lb.json`'s own `verdict`/`load_bearing`, read verbatim (identical to `b2b_score.py`'s
        single score). Trivially exact by construction: there is nothing to "replay," so acceptance check #1 below
        is satisfied for S0 by definition.
  S1 -- self-describing fields excluded (the #1 metric at H). For each coverable row,
        `research/coordination/b2d_decision_fields.json` gives the row's `kept` (non-self-describing) fields and its
        `probe_turn`. This file reads each KEPT field's value directly out of the four arm JSON files already on
        disk (`intact_a_*`, `intact_b_*`, `lesion_*`, `lesion_*.rep0`) at that turn, and applies the SAME
        treatment/null/reproduction test S2 uses (see below), per field, then combines per-field verdicts into one
        cell verdict.
  S2 -- reply-level (answer / abstained / recalled_svo on the probe turn), exactly as the pre-registration defines
        it: `json.dumps([answer, abstained, recalled_svo], sort_keys=True)` compared intact_a vs lesion (treatment),
        intact_a vs intact_b (null floor), lesion vs lesion.rep0 (lesion floor).

Also implements A1.2 **rule 7** (added for B2d): a cell whose probe-turn response carries an `error` key in the
parent dict of ANY of the row's compared fields (SD or kept), in ANY of the four arms, is class C /
UNDEFINED:organ-error for S0-as-reported, S1 and S2 alike -- never scored `regressed`. `b2b_score.py` only
*reported* an organ-error list; this file makes it a validity rule, per the B2d prereg's own comprehension-monitor
finding (its 88/88 retained lesion arms were a harness crash, not a lesion effect).

SIMPLIFICATION, STATED HONESTLY (a documented limitation, not a silent shortcut -- see also HANDOFF.md
NEEDS-CLAUDE). The pre-registration's "How S1 is computed" section describes importing
`research.runners.load_bearing_fraction` at the pinned revision and monkeypatching `_spawn_arm_raw`/`compare()` so
the HARNESS's own comparison machinery runs with SD fields stripped from its input, then checking that an EMPTY
exclusion list reproduces the harness's own `lb.json` diffs exactly (the "identity replay," claimed 209/209 on B2b
at filing). This file does not do that: it never imports `research.runners.load_bearing_fraction` and never builds
a brain (checked by a unit test below). Instead it recomputes each kept field's treatment/null/repro status directly
from the arm JSON already on disk, using the dotted path `b2d_decision_fields.json` already records and audits with
file:line evidence. This is simpler, needs no harness import, and is at least as conservative (a field is S1
load-bearing only when its OWN treatment/null/repro triple is clean, not when the row's overall
`null_control_clean`/`lesion_reproduced` flags -- computed over the ORIGINAL full field set, which can include SD
fields -- happen to be true). It does NOT reproduce the prereg's literal "identity replay through the harness's own
compare()" acceptance check; a harness-level replay (importing load_bearing_fraction, monkeypatching compare()) is
NEEDS-CLAUDE and would let a future revision of this file cross-check S1 against the harness's own SD-stripped
verdict rather than this file's independent recomputation.

Usage (the pre-registration's fixed interface):
  .venv/bin/python tools/b2d_score.py --tag b2d0929 --pin <H> --seeds 42 43 44 100 101 102 \\
      --base research/findings/raw/_load_bearing/_shards --fields research/coordination/b2d_decision_fields.json \\
      --code-root <a tree at H: a pool revisions/<H> copy or a git archive of H> \\
      --corpus research/coordination/b2d_corpus_sha256.tsv --reruns research/coordination/b2d_reruns.tsv \\
      --dispatch-log research/queue/dispatch.log \\
      --compare-tag b2b0924-base --compare-pin a308f1e09babcc9ed096c3c8046d00040391368c \\
      --out research/findings/raw/_load_bearing/_shards/b2d0929/b2d_score.json

`--identity-only` restricts the run to S0 (the "identity replay over B2b's cells" precondition step in the
pre-registration's Step 5) and skips S1/S2/drift.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

SEEDS_DEFAULT = [42, 43, 44, 100, 101, 102]
COVERABLE = ("neural-lesion", "whether-disable")
MEASURED = ("regressed", "pass", "not-exercised")
EXERCISED = ("regressed", "pass")
# A1.3 class C verdicts (code at the pin failed on the row); "unmapped-*" is a prefix
C_VERDICTS = ("arm-build-failed", "lesion-knob-missing", "lesion-not-effective", "artifact-missing",
              "artifact-unreadable", "seed-missing-in-artifact", "noisy", "noisy-null-control")
REPLY_FIELDS_DEFAULT = ("answer", "abstained", "recalled_svo")


def _j(path):
    with open(path) as fh:
        return json.load(fh)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_path(d, dotted):
    """Resolve a dotted field path ('affect_drives.reason') against a turn's response dict. Returns
    (found: bool, value). `found=False` when any segment is absent (a missing field is not the same as a field
    present with value None)."""
    cur = d
    for seg in dotted.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return False, None
        cur = cur[seg]
    return True, cur


def read_fields(path):
    """research/coordination/b2d_decision_fields.json -- the self-describing-field audit."""
    return _j(path)


def field_organs(fields_row):
    """The set of top-level (organ) keys a row's compared fields touch, SD and kept together -- used by rule 7 to
    scope the organ-error check to fields this row actually compares, not every organ present on the turn."""
    return sorted({f.split(".")[0] for f in (fields_row or {}).get("fields", {})})


def read_cells(base, tag, seeds):
    """(faculty, seed, dir) for every faculty directory present under base/tag/s<seed>/*, in sorted order. Tag- and
    registry-agnostic: this globs what is ON DISK rather than parsing a jobs.txt, so it works unchanged for
    b2d0929, b2b0924-base (the --compare-tag drift block and the --identity-only precondition run) or any other
    tag with the same shard-tree layout."""
    cells = []
    for s in seeds:
        for d in sorted(glob.glob(os.path.join(base, tag, "s%d" % s, "*"))):
            if os.path.isdir(d):
                cells.append((os.path.basename(d), s, d))
    return cells


def read_corpus(path):
    """B2d's corpus table: node, host, file, sha256, source_git_sha, provisioned_ok, recorded_utc (no
    'primary-checkout' row, unlike B2b's). Returns (node->host, host->sha256). The caller supplies primary_sha
    separately (see `primary_corpus_sha`); judge_dir treats a None primary_sha as "rule 4 not checked", the same
    graceful degradation `b2b_score.py` uses when its corpus table is absent."""
    node_host, host_sha = {}, {}
    if not os.path.exists(path):
        return node_host, host_sha
    rows = open(path).read().splitlines()
    if not rows:
        return node_host, host_sha
    for row in rows[1:]:
        p = row.split("\t")
        if len(p) < 4:
            continue
        node_host[p[0]] = p[1]
        host_sha[p[1]] = p[3]
    return node_host, host_sha


def primary_corpus_sha(code_root):
    """sha256 of the pinned tree's own data/corpus/tinystories.txt, or None if code_root is not given / the file
    is not there (rule 4 then reports 'not checked' rather than failing every cell over an unrelated gap)."""
    if not code_root:
        return None
    p = os.path.join(code_root, "data", "corpus", "tinystories.txt")
    return _sha(p) if os.path.exists(p) else None


def read_reruns(path):
    rows = {}
    if not path or not os.path.exists(path):
        return rows
    lines = open(path).read().splitlines()
    if not lines:
        return rows
    hdr = lines[0].split("\t")
    for ln in lines[1:]:
        if not ln.strip():
            continue
        rows[ln.split("\t")[0]] = dict(zip(hdr, ln.split("\t")))
    return rows


def read_dispatch_hosts(dispatch_log, tag):
    """cell key 's<seed>/<faculty>' -> dispatch node names, in log order, for one tag."""
    out = {}
    if not dispatch_log or not os.path.exists(dispatch_log):
        return out
    pat = re.compile(r"^\[pool-dispatch\] \S+ (\S+) <- .*--out research/findings/raw/_load_bearing/_shards/%s/"
                     r"s(\d+)/([^/ ]+)/lb\.json" % re.escape(tag))
    for ln in open(dispatch_log, errors="replace"):
        m = pat.search(ln)
        if m:
            out.setdefault("s%s/%s" % (m.group(2), m.group(3)), []).append(m.group(1))
    return out


def sidecar_fails(pj, pin, is_lb):
    f = []
    if pj.get("git_sha") != pin:
        f.append("git_sha=%s" % pj.get("git_sha"))
    if pj.get("source_kind") != "git_archive":
        f.append("source_kind=%s" % pj.get("source_kind"))
    if pj.get("source_manifest_verified_at_start") is not True:
        f.append("manifest_start=%s" % pj.get("source_manifest_verified_at_start"))
    if pj.get("source_manifest_verified_at_exit") is not True:
        f.append("manifest_exit=%s" % pj.get("source_manifest_verified_at_exit"))
    env = pj.get("env") or {}
    brain = sorted(k for k in env if k.startswith("BRAIN_"))
    if is_lb:
        if env.get("SIM_BACKEND") != "numpy":
            f.append("SIM_BACKEND=%s" % env.get("SIM_BACKEND"))
        if brain:
            f.append("BRAIN keys %s" % brain)
    elif [k for k in brain if k != "BRAIN_CHAT_SEED"]:
        f.append("arm BRAIN keys %s" % brain)
    return f


ARM_RE = re.compile(r"^(intact_a_|intact_b_|lesion_).*\.json(\.rep\d+)?$")
ARM_KIND_RE = re.compile(r"^(intact_a|intact_b|lesion)_")


def _arm_kind(fn):
    m = ARM_KIND_RE.match(fn)
    return m.group(1) if m else None


def judge_dir(d, fac, pin, host_sha=None, primary_sha=None, fields_row=None):
    """Apply A1.2 rules 1-6 (reused, unchanged in spirit from `b2b_score.py`) PLUS rule 7 (organ error on the
    probe turn forces class C) to one shard directory. Returns a dict with `class` (None if DEFINED), the four
    named arm bodies (`arms`: kind -> parsed JSON, kept for S1/S2 so the caller need not re-read the files), and
    `organ_errors` (every organ-error hit found anywhere in the arm files, reported regardless of rule 7 scope)."""
    r = {"dir": d, "faculty": fac, "exists": os.path.isdir(d), "fails": [], "class": None, "kind": None,
         "verdict": None, "load_bearing": None, "turn": None, "diffs": [], "host": None, "files": {},
         "organ_errors": [], "arms": {}}
    if not r["exists"]:
        r["fails"].append("E: shard directory absent")
        r["class"] = "E"
        return r
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            r["files"][fn] = _sha(p)
    lb = os.path.join(d, "lb.json")
    if not os.path.exists(lb):
        r["fails"].append("E: no lb.json")
    else:
        rep = _j(lb)
        rows = [p for p in rep.get("per_faculty", []) if p.get("faculty") == fac]
        if len(rows) != 1:
            r["fails"].append("E: lb.json holds %d rows for %s" % (len(rows), fac))
        else:
            p = rows[0]
            r.update(kind=p.get("kind"), verdict=p.get("verdict"), load_bearing=p.get("load_bearing"),
                     turn=p.get("turn"), diffs=p.get("diffs") or [],
                     null_control_clean=p.get("null_control_clean"), lesion_reproduced=p.get("lesion_reproduced"),
                     unreliable=bool(rep.get("UNRELIABLE")))
        if not os.path.exists(lb + ".prov.json"):
            r["fails"].append("E rule1: lb.json has no sidecar")
        else:
            lp = _j(lb + ".prov.json")
            r["host"] = lp.get("host")
            r["fails"] += ["E rule1/3 lb.json: " + x for x in sidecar_fails(lp, pin, True)]
            if host_sha is not None and primary_sha is not None:
                hs = host_sha.get(lp.get("host"))
                if hs is None:
                    r["fails"].append("E rule4: host %s not in the corpus table" % lp.get("host"))
                elif hs != primary_sha:
                    r["fails"].append("E rule4: host %s corpus hash %s != primary" % (lp.get("host"), hs[:12]))
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".prov.json") or not ARM_RE.match(fn):
            continue
        sc = os.path.join(d, fn + ".prov.json")
        if not os.path.exists(sc):
            r["fails"].append("E rule2: arm file %s has no sidecar" % fn)
            continue
        pj = _j(sc)
        r["fails"] += ["E rule2/3 %s: %s" % (fn, x) for x in sidecar_fails(pj, pin, False)]
        if r["host"] and pj.get("host") != r["host"]:
            r["fails"].append("E rule2: %s host %s != lb.json host %s" % (fn, pj.get("host"), r["host"]))
        try:
            arm = _j(os.path.join(d, fn))
        except Exception:
            continue
        kind = _arm_kind(fn)
        is_rep0 = fn.endswith(".rep0")
        key = (kind + ".rep0") if (kind == "lesion" and is_rep0) else kind
        if key and key not in r["arms"]:
            r["arms"][key] = arm
        if isinstance(arm, dict):
            for turn_name, t in arm.items():
                if not isinstance(t, dict):
                    continue
                for k, v in t.items():
                    # An organ error is an `error` key whose VALUE is set. A trace that records `"error": null`
                    # on every turn (webapp/server.py's `transitive_trace`, RULING transitive-engagement) is saying
                    # "no error"; counting the key alone made every such cell C-UNCONFIRMED (2026-10-06). Any
                    # non-null value, including "" or False, still counts: when unsure, invalidate, never hide.
                    if isinstance(v, dict) and v.get("error") is not None:
                        r["organ_errors"].append({"arm": fn, "turn": turn_name, "organ": k,
                                                  "error": str(v["error"])[:200]})
    # ── rule 7: an organ error on THIS ROW'S probe turn, in an organ this row's fields actually touch ──
    scope = set(field_organs(fields_row)) if fields_row is not None else None
    rule7_hits = [e for e in r["organ_errors"]
                 if e["turn"] == r["turn"] and (scope is None or e["organ"] in scope)]
    r["rule7_organ_error"] = rule7_hits
    if any(x.startswith("E") for x in r["fails"]):
        r["class"] = "E"
    elif r["kind"] in COVERABLE:
        if rule7_hits:
            r["fails"].append("C rule7: organ error on probe turn %r: %s" % (r["turn"], rule7_hits[0]))
            r["class"] = "C"
        else:
            v = r["verdict"]
            if r.get("unreliable"):
                r["fails"].append("C: report UNRELIABLE")
            if r.get("null_control_clean") is False:
                r["fails"].append("C: null_control_clean False")
            if v not in MEASURED:
                r["fails"].append("verdict %s" % v)
            if r["fails"]:
                is_c = r.get("unreliable") or r.get("null_control_clean") is False or (
                    v in C_VERDICTS or (isinstance(v, str) and v.startswith("unmapped-")))
                r["class"] = "C" if is_c else "I"
    return r


def final_state(canon, attempt1, first_host, rerun_host):
    """A1.5 final state, unchanged from `b2b_score.py`."""
    if canon["class"] is None:
        return "DEFINED"
    if canon["class"] == "C":
        if (attempt1 is not None and attempt1["class"] == "C" and first_host and rerun_host
                and first_host != rerun_host):
            return "FAIL"
        return "C-UNCONFIRMED"
    if canon["class"] == "I":
        return "UNDECIDED"
    return "MISSING"


# ────────────────────────────── S1 (self-describing fields excluded) ──────────────────────────────

def score_s1(canon, fields_row, reply_fields=REPLY_FIELDS_DEFAULT):
    """S1 for one DEFINED, coverable cell. Returns (verdict, detail). Reads the row's KEPT fields directly out of
    `canon['arms']` (already parsed by judge_dir) at the row's registered probe turn; never touches SD fields."""
    if fields_row is None:
        return "UNDEFINED:no-fields-entry", {}
    if fields_row.get("sd_only"):
        return "UNDEFINED:sd-only", {"kept": []}
    kept = fields_row.get("kept") or []
    turn = fields_row.get("probe_turn") or canon.get("turn")
    arms = canon["arms"]
    for k in ("intact_a", "intact_b", "lesion", "lesion.rep0"):
        if k not in arms or turn not in (arms[k] or {}):
            return "UNDEFINED:missing-arm-or-turn", {"turn": turn, "have_arms": sorted(arms)}
    ia, ib, le, le0 = (arms[k][turn] for k in ("intact_a", "intact_b", "lesion", "lesion.rep0"))
    field_status, lb_fields, noisy_fields = {}, [], []
    for f in kept:
        # (found, value) tuples, not bare values: a field present in one arm and absent in another is a genuine
        # difference (e.g. a trace block that disappears under the lesion), not something to silently skip.
        pa, pb, pl, pl0 = (get_path(t, f) for t in (ia, ib, le, le0))
        if not any(found for found, _ in (pa, pb, pl, pl0)):
            field_status[f] = "not-applicable"        # absent in all four arms -- no information either way
            continue
        treat_diff = pa != pl
        null_diff = pa != pb
        repro = pl == pl0
        if not treat_diff:
            field_status[f] = "no-diff"
        elif null_diff:
            field_status[f] = "UNDEFINED:null-noise"
            noisy_fields.append(f)
        elif not repro:
            field_status[f] = "UNDEFINED:lesion-noise"
            noisy_fields.append(f)
        else:
            field_status[f] = "load-bearing"
            lb_fields.append(f)
    detail = {"turn": turn, "kept": kept, "field_status": field_status, "load_bearing_fields": lb_fields,
              "noisy_fields": noisy_fields}
    if lb_fields:
        return "regressed", detail
    if noisy_fields:
        return "UNDEFINED:s1-noise", detail
    if kept and any(field_status.get(f) != "not-applicable" for f in kept):
        return "pass", detail          # at least one kept field resolved, and none differ / are noisy
    return "UNDEFINED:no-kept-field-resolved", detail   # every kept field was absent in all four arms


# ────────────────────────────────────────── S2 (reply-level) ──────────────────────────────────────────

def _reply_tuple(turn_dict, reply_fields):
    return json.dumps([turn_dict.get(f) for f in reply_fields], sort_keys=True)


def score_s2(canon, fields_row, reply_fields=REPLY_FIELDS_DEFAULT):
    """S2 for one DEFINED, coverable cell (the pre-registration's S2 definition, verbatim)."""
    if fields_row is not None and fields_row.get("not_replayed"):
        return "UNDEFINED:not-replayed", {}
    turn = (fields_row or {}).get("probe_turn") or canon.get("turn")
    arms = canon["arms"]
    for k in ("intact_a", "intact_b", "lesion", "lesion.rep0"):
        if k not in arms or turn not in (arms[k] or {}):
            return "UNDEFINED:missing-arm-or-turn", {"turn": turn, "have_arms": sorted(arms)}
    ia, ib, le, le0 = (arms[k][turn] for k in ("intact_a", "intact_b", "lesion", "lesion.rep0"))
    for name, t in (("intact_a", ia), ("intact_b", ib), ("lesion", le), ("lesion.rep0", le0)):
        if not isinstance(t.get("answer"), str) and t.get("answer") is not None:
            return "UNDEFINED:non-string-answer", {"turn": turn, "arm": name}
    ta, tb, tl, tl0 = (_reply_tuple(t, reply_fields) for t in (ia, ib, le, le0))
    changed_fields = sorted(f for f in reply_fields if ia.get(f) != le.get(f))
    detail = {"turn": turn, "reply_fields": list(reply_fields), "changed_fields": changed_fields,
              "answer_only": changed_fields == ["answer"]}
    if ta != tb:
        return "UNDEFINED:null-noise", detail
    if tl != tl0:
        return "UNDEFINED:lesion-noise", detail
    return ("regressed" if ta != tl else "pass"), detail


# ─────────────────────────────────────────── code-root blob check ───────────────────────────────────────────

def _git_hash_object(code_root, rel_path):
    p = os.path.join(code_root, rel_path)
    if not os.path.exists(p):
        return None
    try:
        out = subprocess.run(["git", "hash-object", p], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def check_code_root(fields, code_root, pin):
    """The pre-registration's refusal condition: the fields file's own pin must equal --pin, and every blob it
    cites must match the blob actually present in --code-root. Returns a list of mismatch strings (empty = OK)."""
    mism = []
    if fields.get("pin") != pin:
        mism.append("fields file pin %s != --pin %s" % (fields.get("pin"), pin))
    if not code_root:
        mism.append("no --code-root given: blob citations unverified")
        return mism
    for rel, want in (fields.get("cited_blobs_at_pin") or {}).items():
        got = _git_hash_object(code_root, rel)
        if got != want:
            mism.append("%s: cited blob %s != code-root blob %s" % (rel, want, got))
    return mism


# ──────────────────────────────────────────────── score() ────────────────────────────────────────────────

def score(base, tag, pin, seeds, fields_path=None, code_root=None, corpus_path=None, reruns_path=None,
         dispatch_log=None, attempt1_dir=None, identity_only=False):
    fields = read_fields(fields_path) if fields_path else {}
    rows_fields = (fields.get("rows") or {})
    reply_fields = tuple(fields.get("reply_fields") or REPLY_FIELDS_DEFAULT)
    node_host, host_sha = read_corpus(corpus_path) if corpus_path else ({}, {})
    primary_sha = primary_corpus_sha(code_root)
    reruns = read_reruns(reruns_path) if reruns_path else {}
    disp = read_dispatch_hosts(dispatch_log, tag) if dispatch_log else {}
    cells = read_cells(base, tag, seeds)
    out_cells, kind_of = {}, {}
    for fac, seed, d in cells:
        key = "s%d/%s" % (seed, fac)
        frow = rows_fields.get(fac)
        canon = judge_dir(d, fac, pin, host_sha or None, primary_sha, fields_row=frow)
        a1_dir = os.path.join(attempt1_dir, tag, "s%d" % seed, fac) if attempt1_dir else None
        attempt1 = judge_dir(a1_dir, fac, pin, host_sha or None, primary_sha, fields_row=frow) \
            if a1_dir and os.path.isdir(a1_dir) else None
        rr = reruns.get("%s/%s" % (tag, key))
        first_host = rerun_host = None
        if rr:
            first_host = (rr.get("first_host") or "").split(" ")[0]
            rerun_host = (rr.get("rerun_host") or "").split(" ")[0]
        if canon["kind"]:
            kind_of[fac] = canon["kind"]
        cell = {"faculty": fac, "seed": seed, "canonical": canon, "attempt1": attempt1, "rerun_log": rr,
               "first_host": first_host, "rerun_host": rerun_host, "dispatch_nodes": disp.get(key, [])}
        cell["kind"] = kind_of.get(fac)
        cell["coverable"] = cell["kind"] in COVERABLE
        cell["state"] = final_state(canon, attempt1, first_host, rerun_host) if cell["coverable"] \
            else "reported (%s)" % canon["verdict"]
        exp_node = rerun_host or (cell["dispatch_nodes"][-1] if cell["dispatch_nodes"] else None)
        exp_host = node_host.get(exp_node) if exp_node else None
        cell["host_check"] = {"expected_node": exp_node, "expected_host": exp_host,
                              "sidecar_host": canon["host"],
                              "ok": (exp_host is None) or (canon["host"] is None) or (exp_host == canon["host"])}
        if cell["state"] == "DEFINED" and cell["coverable"] and not identity_only:
            if (fields.get("not_replayed") or {}).get(fac):
                # e.g. open-ended-generation: the distributional ruler has no decision fields to strip SD from
                # (it never calls _spawn_arm/compare at all) -- S1 is defined to equal S0, per the prereg.
                cell["s1"] = (canon["verdict"], {"note": "not_replayed row (%s): S1 == S0" %
                                                 fields["not_replayed"][fac]})
            else:
                cell["s1"] = score_s1(canon, frow, reply_fields)
            cell["s2"] = score_s2(canon, frow, reply_fields)
        out_cells[key] = cell
    cov = {k: c for k, c in out_cells.items() if c["coverable"]}
    facs = sorted({c["faculty"] for c in cov.values()})

    def _agg(level_key, is_lb_fn):
        per_seed = {}
        for s in seeds:
            dc = [c for c in cov.values() if c["seed"] == s and c["state"] == "DEFINED"]
            ex = [c for c in dc if is_lb_fn(c) is not None]
            lbc = [c for c in ex if is_lb_fn(c) is True]
            per_seed[str(s)] = {"n_defined": len(dc), "n_exercised": len(ex), "n_load_bearing": len(lbc),
                                "fraction": (len(lbc) / len(ex)) if ex else None}
        per_fac = {}
        for f in facs:
            vs = [cov["s%d/%s" % (s, f)] for s in seeds]
            resolved = [v for v in vs if v["state"] == "DEFINED" and is_lb_fn(v) is not None]
            n_lb = sum(1 for v in resolved if is_lb_fn(v) is True)
            n_missing = len(vs) - len(resolved)
            per_fac[f] = {"n_load_bearing": n_lb if not n_missing else [n_lb, n_lb + n_missing]}
        lo = lambda v: v if isinstance(v, int) else v[0]   # noqa: E731
        robust = sorted(f for f, v in per_fac.items() if lo(v["n_load_bearing"]) == len(seeds))
        union = sorted(f for f, v in per_fac.items() if lo(v["n_load_bearing"]) > 0)
        fr = [v["fraction"] for v in per_seed.values() if v["fraction"] is not None]
        return {"per_seed": per_seed, "per_faculty": per_fac, "robust_core": robust, "robust_core_n": len(robust),
                "union": union, "union_n": len(union),
                "mean_fraction": statistics.fmean(fr) if fr else None,
                "sd_fraction_pop": statistics.pstdev(fr) if fr else None}

    s0 = _agg("s0", lambda c: c["canonical"]["load_bearing"] if c["canonical"]["verdict"] in EXERCISED else None)
    res = {
        "tag": tag, "pin": pin, "seeds": seeds, "identity_only": identity_only,
        "fields_pin": fields.get("pin"), "backend": "numpy", "device": "cpu",
        "counts": {"n_coverable_faculties": len(facs), "n_coverable_cells": len(cov),
                   "n_defined": sum(1 for c in cov.values() if c["state"] == "DEFINED")},
        "not_defined": {k: {"state": c["state"], "fails": c["canonical"]["fails"]}
                        for k, c in cov.items() if c["state"] != "DEFINED"},
        "organ_error_cells": {k: c["canonical"]["rule7_organ_error"] for k, c in out_cells.items()
                              if c["canonical"]["rule7_organ_error"]},
        "s0": s0,
    }
    if not identity_only:
        s1 = _agg("s1", lambda c: (True if c.get("s1", (None,))[0] == "regressed"
                                   else (False if c.get("s1", (None,))[0] == "pass" else None)))
        s2 = _agg("s2", lambda c: (True if c.get("s2", (None,))[0] == "regressed"
                                   else (False if c.get("s2", (None,))[0] == "pass" else None)))
        res["s1"], res["s2"] = s1, s2
        res["sd_only_rows"] = sorted(f for f, r in rows_fields.items() if r.get("sd_only"))
        res["manifest"] = {k: {"dir": c["canonical"]["dir"], "state": c["state"],
                               "s0": [c["canonical"]["verdict"], c["canonical"]["load_bearing"]],
                               "s1": c.get("s1", (None, None))[0], "s2": c.get("s2", (None, None))[0],
                               "s1_detail": c.get("s1", (None, {}))[1], "s2_detail": c.get("s2", (None, {}))[1]}
                           for k, c in cov.items()}
    res["host_check_failures"] = {k: c["host_check"] for k, c in out_cells.items() if not c["host_check"]["ok"]}
    fail_cells = sorted(k for k, c in cov.items() if c["state"] == "FAIL")
    n_defined = res["counts"]["n_defined"]
    if fail_cells:
        verdict = "FAIL"
    elif len(cov) and n_defined == len(cov):
        verdict = "COMPLETE"
    else:
        verdict = "INCOMPLETE"
    res["fail_cells"] = fail_cells
    res["verdict"] = verdict
    pre = [
        {"name": "code-root blobs match the fields file's cited blobs at --pin", "ok": True, "value": []},
    ]
    if fields_path:
        mism = check_code_root(fields, code_root, pin)
        pre[0]["ok"] = not mism
        pre[0]["value"] = mism
    pre.append({"name": "every checked sidecar host matches its dispatch or re-run node",
               "ok": not res["host_check_failures"], "value": res["host_check_failures"]})
    res["preconditions"] = pre
    if not all(p["ok"] is True for p in pre):
        res["verdict"] = "UNDEFINED -- a scorer precondition failed (%s); cell-level verdict would read %s" % (
            ", ".join(p["name"] for p in pre if p["ok"] is not True), verdict)
    return res


def drift(base, tag, pin, compare_tag, compare_pin, seeds, fields_path=None):
    """Per-cell S0/S1/S2 comparison between `tag`@`pin` and `compare_tag`@`compare_pin`, over cells DEFINED in
    both. Reported, never gated (the pre-registration's 'Comparison with B2b' section)."""
    a = score(base, tag, pin, seeds, fields_path=fields_path)
    b = score(base, compare_tag, compare_pin, seeds, fields_path=fields_path)
    am, bm = a.get("manifest", {}), b.get("manifest", {})
    common = sorted(set(am) & set(bm))
    diffs = []
    for k in common:
        if am[k]["s0"] != bm[k]["s0"] or am[k]["s1"] != bm[k]["s1"] or am[k]["s2"] != bm[k]["s2"]:
            diffs.append({"cell": k, tag: {"s0": am[k]["s0"], "s1": am[k]["s1"], "s2": am[k]["s2"]},
                          compare_tag: {"s0": bm[k]["s0"], "s1": bm[k]["s1"], "s2": bm[k]["s2"]}})
    return {"tag": tag, "compare_tag": compare_tag, "n_common_defined_cells": len(common),
            "n_differing": len(diffs), "differences": diffs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--base", default="research/findings/raw/_load_bearing/_shards")
    ap.add_argument("--fields", default=None)
    ap.add_argument("--code-root", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--reruns", default=None)
    ap.add_argument("--dispatch-log", default=None)
    ap.add_argument("--attempt1-dir", default=None)
    ap.add_argument("--compare-tag", default=None)
    ap.add_argument("--compare-pin", default=None)
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = score(a.base, a.tag, a.pin, a.seeds, fields_path=a.fields, code_root=a.code_root, corpus_path=a.corpus,
               reruns_path=a.reruns, dispatch_log=a.dispatch_log, attempt1_dir=a.attempt1_dir,
               identity_only=a.identity_only)
    if a.compare_tag and not a.identity_only:
        res["drift"] = drift(a.base, a.tag, a.pin, a.compare_tag, a.compare_pin, a.seeds, fields_path=a.fields)
    summary = {k: res[k] for k in ("verdict", "counts", "not_defined", "fail_cells") if k in res}
    summary["s0"] = {k: res["s0"].get(k) for k in ("robust_core_n", "union_n", "mean_fraction", "sd_fraction_pop")}
    if "s1" in res:
        summary["s1"] = {k: res["s1"].get(k) for k in ("robust_core_n", "union_n", "mean_fraction",
                                                        "sd_fraction_pop")}
        summary["s2"] = {k: res["s2"].get(k) for k in ("robust_core_n", "union_n", "mean_fraction",
                                                        "sd_fraction_pop")}
    summary["preconditions_failed"] = [p["name"] for p in res["preconditions"] if p["ok"] is not True]
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str, sort_keys=False)
        print("wrote %s" % a.out)
    return 0 if not summary["preconditions_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
