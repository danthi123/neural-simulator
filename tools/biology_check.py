#!/usr/bin/env python3
"""Bind CODE to the BIOLOGY it implements, and verify the binding still resolves.

WHY (2026-07-31, owner directive). The biology is STABLE. A pipeline's interactions with other regions grow and
change, but the neuroscience it was based on does not. Yet this project keeps re-researching what it already
established, and -- worse -- keeps running configurations that contradict the biology it claims to implement.

The case that earned this, today: a 21-agent research round established that Bittner et al. 2017 reports BTSP as
a ONE-SHOT mechanism -- a SINGLE plateau creates a place field. The gap#5 runner's `laps` default was 5. Five
traversals re-potentiate every position and destroy the field; measured, place-specificity decays 4.40x -> 2.57x
-> 1.11x as laps go 1 -> 2 -> 5. The wrong default survived every sweep of density, w_max, lr and eligibility tau,
because nothing connected the runner to the paper. The biology was knowable from the first day of the arc.

WHAT THIS ENFORCES, three things a human cannot be relied on to keep doing:

  1. SOURCE POINTERS MUST RESOLVE. Each claim cites a local corpus file plus a distinctive `anchor` quote, and
     the checker OPENS the file and confirms the anchor is still present. A rotted pointer fails loudly instead of
     silently becoming folklore. (The project's local corpus paths had already rotted to dead `E:` drive paths
     once, and the RAG silently fell back to online search for weeks.)
  2. CONFIG MUST NOT CONTRADICT THE BIOLOGY. `constraints_config` maps a config/CLI key to the value the biology
     REQUIRES, with the reason. `laps: 1` for a one-shot mechanism is checkable, and would have caught the defect
     above on day one.
  2b. THE PROTOCOL MUST NOT CONTRADICT THE BIOLOGY EITHER -- see `protocol:` below.
  3. IMPLEMENTED-BY MUST EXIST. An entry claiming a runner implements it is checked against the filesystem, so
     the map cannot quietly describe code that was renamed or deleted.

    .venv/bin/python tools/biology_check.py                 # check every entry
    .venv/bin/python tools/biology_check.py --config research/runners/_gap5_btsp_place_field_derisk.py
    .venv/bin/python tools/biology_check.py --selftest      # prove the checker still FAILS on cases it must catch

Entries live in research/biology/<id>.md with YAML-ish frontmatter. Deliberately a flat directory of small files:
the point is that opening ONE file tells you what the code is supposed to be doing and where that came from.


THE `protocol:` BLOCK (2026-07-31) -- because THE PROTOCOL IS PART OF THE MECHANISM, AND NOBODY WRITES IT DOWN.

That is one of four measured causes of this project's walls: "papers give the MECHANISM; the OPERATING POINT and
the PROTOCOL are implicit in the animal." Bittner never says "run one lap" -- an animal ran the track once, so the
one-shot-ness is in the METHODS, not the result. `constraints_config` caught that case only because the protocol
happened to reduce to a single scalar equality (`laps == 1`). Most do not:

  * `bout_len > 1` for a trace rule -- an INEQUALITY. The `invariance-from-temporal-continuity` entry says outright
    it refuses to declare a `constraints_config` for exactly this reason: "the biology constrains them only as an
    inequality ... and the checker compares by equality. Pinning 0.8/12 would fire on a legitimate re-tuning, and a
    gate that cries wolf gets switched off."
  * "the recall read must not write" -- a BOOLEAN, `--freeze-read`, `action="store_true"`. The
    `systems-consolidation` entry records that this is the single highest-value constraint it has and that it
    CANNOT declare it, because the legacy matcher only sees numeric defaults. A live read cost that arc a
    retraction (store drift +1.28-1.41 live vs +0.000000 frozen).
  * "presentation must be grouped, not shuffled" -- a MEMBERSHIP over a string mode.

So a protocol rule is a list of `- rule: / why: / check:`, where `check` is `<config_key> <op> <operand>`:

    protocol:
      - rule: "Induction is ONE-SHOT -- a single plateau creates the field"
        why: "Repeated traversals re-potentiate every position and ERASE it: 4.40x -> 2.57x -> 1.11x at laps 1/2/5"
        check: laps eq 1
      - rule: "The eligibility trace must outlast a single presentation"
        why: "A bout of one makes the trace a no-op; the rule then has nothing to bind across"
        check: bout_len gte 2
      - rule: "The recall read must not write"
        why: "Hebbian overwrote the store WHILE it was being read; drift +1.28-1.41 live vs +0.000000 frozen"
        check: freeze_read eq true
      - rule: "Members of a category must be presented contiguously"
        why: "Shuffled temporal order collapses held-out super-acc 0.958 -> 0.556"
        check: presentation in grouped,contiguous

`op` is one of eq · ne · lt · lte · gt · gte · in · not_in. `in`/`not_in` take a comma-separated operand list
(brackets optional). Ordered ops require both sides to be numeric; asking `lte` of a string is a SCHEMA error that
fails loudly rather than quietly evaluating to nothing.

WHY THE RESOLVER IS SEPARATE FROM `constraints_config`'s. The legacy matcher (`default=<number>` or a bare
`key = <number>`) is LIVE and blocking a real defect, so it is untouched, byte for byte. The protocol resolver is
a superset written alongside it: it walks each `add_argument(...)` call with a paren/quote-aware scanner, honours
`dest=`, accepts the `--dashed-form` of an underscored key, reads `action="store_true"` / `"store_false"` as the
booleans they are, and reads string and boolean defaults as well as numbers. It resolves in tiers -- argparse,
then a module-level assignment, then a `"key": value` mapping entry -- and if one tier yields two DIFFERENT values
it reports AMBIGUOUS rather than picking one, because a resolver that guesses is a check that cannot be trusted.

WHAT THIS STILL CANNOT CATCH (state it, so nobody mistakes green for verified):
  * A default is not a run. A runner can default to `laps=1` and be launched with `--laps 5`; the value that
    reaches the experiment lives in the command line and the artifact, not here. `tools/gates/artifact_provenance`
    is the layer for that. If a key has NO default, this reports the constraint as UNENFORCEABLE rather than
    passing it.
  * Protocol properties that are not a config value at all -- "the update uses only locally available terms", "the
    stimulus set was grouped by construction inside the generator" -- are invisible. `coincidence-binding` and
    `dendritic-plateau-coincidence-burst` both say so in their own words; the control that catches those lives in
    the runner (a permuted-role arm reading 0.000), not here.
  * It reads SOURCE, not behaviour. A default overwritten at import time, or plumbed through **kwargs, resolves to
    whatever the literal says.
  * String comparison is case-sensitive and exact, matching the legacy behaviour.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIO_DIR = os.path.join(ROOT, "research", "biology")


def _parse_frontmatter(text):
    """Minimal YAML subset: scalars, lists of scalars, and lists of `- key: value` blocks. No dependency."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    body, out, cur_key, cur_list = text[3:end], {}, None, None
    for raw in body.split("\n"):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.startswith("  - ") or raw.startswith("- "):
            item = raw.split("- ", 1)[1].strip()
            if ":" in item and not item.startswith(("http", '"')):
                k, v = item.split(":", 1)
                cur_list.append({k.strip(): v.strip().strip('"')})
            else:
                cur_list.append(item.strip('"'))
            continue
        if raw.startswith("    ") and cur_list and isinstance(cur_list[-1], dict):
            if ":" in raw:
                k, v = raw.strip().split(":", 1)
                cur_list[-1][k.strip()] = v.strip().strip('"')
            continue
        if ":" in raw:
            k, v = raw.split(":", 1)
            cur_key, v = k.strip(), v.strip().strip('"')
            if v == "":
                cur_list = []
                out[cur_key] = cur_list
            else:
                out[cur_key] = v
                cur_list = None
    return out


def _expand(p):
    return os.path.expanduser(p) if p.startswith("~") else (p if os.path.isabs(p) else os.path.join(ROOT, p))


# --------------------------------------------------------------------------------------------------------------
# protocol: the comparison layer.  Everything below is NEW and is used ONLY by `protocol:` rules.  The
# `constraints_config` path further down is deliberately untouched -- it is live and blocking a real defect.
# --------------------------------------------------------------------------------------------------------------

_ORDERED_OPS = ("lt", "lte", "gt", "gte")
_SET_OPS = ("in", "not_in")
_OPS = ("eq", "ne") + _ORDERED_OPS + _SET_OPS

_UNPARSED = object()          # a source literal we refuse to guess at (an identifier, a call, an f-string)
_LIT = r"""(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|True|False|None|"[^"\n]*"|'[^'\n]*')"""


def _norm_yaml(s):
    """A scalar written in the entry (`check: order in grouped,contiguous`) -> a comparable value.

    LENIENT: a bare word is a string, because that is how a human writes a mode name in YAML.
    """
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    return f if _finite(f) else s


def _norm_source(s):
    """A literal read out of PYTHON SOURCE -> a comparable value, or `_UNPARSED`.

    STRICT, and that is the point: `default=DEFAULT_LAPS` must NOT silently become the string "DEFAULT_LAPS" and
    then compare unequal to everything.  An unresolvable default is reported, never quietly failed or passed.
    """
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s in ("True", "False"):
        return s == "True"
    if s in ("None", "null"):
        return None
    try:
        f = float(s)
    except ValueError:
        return _UNPARSED
    return f if _finite(f) else _UNPARSED


def _finite(f):
    return f == f and f not in (float("inf"), float("-inf"))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _values_equal(a, b):
    """Equality that does NOT let Python's `True == 1` quietly satisfy a boolean protocol rule.

    The bool branch is deliberately written against raw `isinstance(..., (int, float))` -- which INCLUDES bool --
    and not against `_is_num`, which excludes it.  Written the other way the branch is dead code: `_is_num` would
    already have rejected the pair, so deleting the guard would change no answer.  A guard whose removal changes
    nothing is not a guard, it is decoration, and a mutation test walks straight through it.  (Caught exactly that
    way on 2026-07-31: the first version of this function survived having its bool guard deleted.)
    """
    a_bool, b_bool = isinstance(a, bool), isinstance(b, bool)
    if a_bool or b_bool:
        return a_bool and b_bool and (a is b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    if a is None or b is None:
        return a is None and b is None
    return str(a) == str(b)


def _split_top_level(s, sep=","):
    """Split on `sep` at bracket depth 0 and outside quotes.  Used for argparse call arguments."""
    out, buf, depth, q = [], [], 0, ""
    i = 0
    while i < len(s):
        ch = s[i]
        if q:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1]); i += 2; continue
            if ch == q:
                q = ""
        elif ch in "\"'":
            q = ch; buf.append(ch)
        elif ch in "([{":
            depth += 1; buf.append(ch)
        elif ch in ")]}":
            depth -= 1; buf.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [x.strip() for x in out]


def _iter_add_argument_calls(text):
    """Yield the argument text of every `add_argument(...)`, with balanced parens and quote awareness.

    The legacy matcher used `[^)]*?`, which stops at the first `)` -- so `type=lambda s: tuple(s)` truncates the
    call and the default is never seen.  This scanner does not have that failure mode.
    """
    for m in re.finditer(r"add_argument\s*\(", text):
        i, depth, q = m.end() - 1, 0, ""
        j, n = i, len(text)
        while j < n:
            ch = text[j]
            if q:
                if ch == "\\":
                    j += 2; continue
                if ch == q:
                    q = ""
            elif ch in "\"'":
                q = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    yield text[i + 1:j]
                    break
            j += 1


def _argparse_defaults(text, key):
    """[(raw_literal_or_None, evidence)] for every add_argument whose option strings or dest name `key`."""
    found = []
    for args in _iter_add_argument_calls(text):
        parts = _split_top_level(args)
        opts, kw = [], {}
        for p in parts:
            mk = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)$", p, re.S)
            if mk:
                kw[mk.group(1)] = mk.group(2).strip()
            elif re.match(r"""^["'][^"']*["']$""", p):
                opts.append(p.strip("\"'"))
        names = {o.lstrip("-") for o in opts}
        dest = kw.get("dest", "").strip("\"'")
        if dest:
            names.add(dest)
        if key not in names:
            continue
        flag = next((o for o in opts if o.startswith("--")), opts[0] if opts else key)
        action = kw.get("action", "").strip("\"'")
        if "default" in kw:
            found.append((kw["default"], "add_argument(%s, default=%s)" % (flag, kw["default"][:40])))
        elif action == "store_true":
            found.append(("False", "add_argument(%s, action=\"store_true\") => default False" % flag))
        elif action == "store_false":
            found.append(("True", "add_argument(%s, action=\"store_false\") => default True" % flag))
        else:
            found.append((None, "add_argument(%s) has NO default" % flag))
    return found


def _assign_defaults(text, key):
    """[(raw, evidence)] for a module/statement-level `key = <literal>`."""
    out = []
    for m in re.finditer(r"^[ \t]*%s\s*=\s*%s\s*(?:#.*)?$" % (re.escape(key), _LIT), text, re.M):
        out.append((m.group(1), "%s = %s" % (key, m.group(1))))
    return out


def _mapping_defaults(text, key):
    """[(raw, evidence)] for a `"key": <literal>` mapping entry -- the config-dict shape."""
    out = []
    for m in re.finditer(r"""["']%s["']\s*:\s*%s""" % (re.escape(key), _LIT), text):
        out.append((m.group(1), '"%s": %s' % (key, m.group(1))))
    return out


def resolve_config_default(text, key):
    """Resolve the runner's DEFAULT for `key`.  -> (status, value, evidence).

    status: "ok" | "missing" | "nodefault" | "unparsed" | "ambiguous"

    Tiers are tried in order and the first tier with ANY hit wins.  Two different values inside one tier is
    AMBIGUOUS, not a coin flip: a resolver that guesses turns a gate into decoration.
    """
    aliases = []
    for a in (key, key.replace("_", "-"), key.replace("-", "_")):
        if a not in aliases:
            aliases.append(a)
    for tier, fn in (("argparse", _argparse_defaults), ("assignment", _assign_defaults),
                     ("mapping", _mapping_defaults)):
        cands = []
        for a in aliases:
            cands += fn(text, a)
        if not cands:
            continue
        raws = sorted({("<none>" if c[0] is None else c[0]) for c in cands})
        ev = "%s: %s" % (tier, " | ".join(sorted({c[1] for c in cands}))[:160])
        if len(raws) > 1:
            return ("ambiguous", raws, ev)
        raw = cands[0][0]
        if raw is None:
            return ("nodefault", None, ev)
        val = _norm_source(raw)
        if val is _UNPARSED:
            return ("unparsed", raw, ev)
        return ("ok", val, ev)
    return ("missing", None, "")


def parse_check(expr):
    """`"laps eq 1"` -> (key, op, operand, error).  Operand is a LIST for in/not_in, a scalar otherwise."""
    toks = str(expr).strip().split(None, 2)
    if len(toks) < 3:
        return (None, None, None, "check must be `<config_key> <op> <value>`, got %r" % expr)
    key, op, rhs = toks[0], toks[1].lower(), toks[2].strip()
    if op not in _OPS:
        return (None, None, None, "unknown comparison %r in %r (expected one of: %s)" % (op, expr, " ".join(_OPS)))
    if not re.match(r"^[A-Za-z_][A-Za-z_0-9-]*$", key):
        return (None, None, None, "%r is not a usable config key in %r" % (key, expr))
    if op in _SET_OPS:
        items = [x for x in (i.strip() for i in rhs.strip("[]()").split(",")) if x != ""]
        if not items:
            return (None, None, None, "%s needs a non-empty comma-separated list, got %r" % (op, rhs))
        return (key, op, [_norm_yaml(i) for i in items], None)
    return (key, op, _norm_yaml(rhs), None)


def evaluate_check(actual, op, operand):
    """-> (satisfied|None, detail).  None means NOT EVALUABLE, which is a problem, never a silent pass."""
    if op in _SET_OPS:
        hit = any(_values_equal(actual, o) for o in operand)
        return ((hit if op == "in" else not hit), "one of %s" % ", ".join(_fmt(o) for o in operand))
    if op == "eq":
        return (_values_equal(actual, operand), "== %s" % _fmt(operand))
    if op == "ne":
        return (not _values_equal(actual, operand), "!= %s" % _fmt(operand))
    if not (_is_num(actual) and _is_num(operand)):
        return (None, "'%s' compares numbers, but got %s and %s" % (op, _fmt(actual), _fmt(operand)))
    a, b = float(actual), float(operand)
    ok = {"lt": a < b, "lte": a <= b + 1e-9, "gt": a > b, "gte": a >= b - 1e-9}[op]
    return (ok, "%s %s" % ({"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[op], _fmt(operand)))


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "None"
    if _is_num(v):
        return ("%g" % v)
    return "'%s'" % v


def check_entry(path, verbose=True):
    text = open(path).read()
    fm = _parse_frontmatter(text)
    name = os.path.basename(path)
    problems = []
    if not fm:
        return ["%s: no frontmatter — an entry with no machine-readable head cannot be checked or indexed" % name]

    for req in ("id", "mechanism"):
        if not fm.get(req):
            problems.append("%s: missing required field '%s'" % (name, req))

    srcs = [s for s in (fm.get("sources") or []) if isinstance(s, dict)]
    if not srcs:
        problems.append("%s: NO SOURCE — a biology entry without a citation is folklore" % name)
    for s in srcs:
        p, anchor = s.get("path", ""), s.get("anchor", "")
        if not p:
            continue
        if p.startswith(("http", "doi:", "DOI:", "PMC", "PMID")):
            if not anchor:
                problems.append("%s: external source %s has no anchor quote to verify against" % (name, p[:60]))
            continue                                     # cannot resolve an external source offline
        full = _expand(p)
        if not os.path.exists(full):
            problems.append("%s: SOURCE PATH DOES NOT RESOLVE: %s" % (name, p))
            continue
        if anchor:
            try:
                with open(full, errors="ignore") as fh:
                    if anchor.lower() not in fh.read().lower():
                        problems.append("%s: ANCHOR NOT FOUND in %s — the pointer has rotted: %r"
                                        % (name, os.path.basename(full), anchor[:60]))
            except Exception as e:
                problems.append("%s: source unreadable (%s): %s" % (name, type(e).__name__, p))

    for impl in (fm.get("implemented_by") or []):
        ip = impl if isinstance(impl, str) else str(impl)
        if not os.path.exists(_expand(ip)):
            problems.append("%s: implemented_by path missing: %s" % (name, ip))

    problems += check_protocol_schema(fm, name)

    if verbose and not problems:
        n_c = len([c for c in (fm.get("constraints_config") or [])])
        n_p = len([p for p in (fm.get("protocol") or []) if isinstance(p, dict)])
        extra = (", %d protocol rule(s)" % n_p) if n_p else ""
        print("  ✔ %-34s %d source(s) resolve, %d config constraint(s)%s"
              % (fm.get("id", name), len(srcs), n_c, extra))
    return problems


def check_protocol_schema(fm, name):
    """Validate every `protocol:` rule as SCHEMA, independent of any runner.

    This runs even when no implementing runner exists, because the alternative is the failure class where the
    anti-drift mechanism is itself the defect: a typo'd operator (`lte` -> `leq`) in a rule nobody ever evaluates
    is a rule that can never fire, and it would look exactly like a passing one.
    """
    problems = []
    for i, rule in enumerate(fm.get("protocol") or []):
        where = "%s: protocol[%d]" % (name, i)
        if not isinstance(rule, dict):
            problems.append("%s is not a `- rule:/why:/check:` block: %r" % (where, rule))
            continue
        label = rule.get("rule") or "(unnamed)"
        for req in ("rule", "why", "check"):
            if not rule.get(req):
                problems.append("%s (%s): missing '%s' — a protocol rule without a %s is %s"
                                % (where, label[:48], req, req,
                                   {"rule": "unreadable", "why": "folklore", "check": "unenforceable"}[req]))
        if not rule.get("check"):
            continue
        _k, _op, _operand, err = parse_check(rule["check"])
        if err:
            problems.append("%s (%s): UNPARSEABLE check — %s" % (where, label[:48], err))
    return problems


def check_protocol(runner_path, verbose=True):
    """Evaluate every `protocol:` rule of every biology entry that claims this runner implements it.

    Separate from `check_config` on purpose.  The `constraints_config` matcher is live and blocking a real defect,
    so it is not refactored underneath itself; this walks the same entries with the richer resolver documented at
    the top of the file.
    """
    text = open(runner_path).read()
    rel = os.path.relpath(os.path.abspath(runner_path), ROOT)
    problems, checked = [], 0
    for f in sorted(glob.glob(os.path.join(BIO_DIR, "*.md"))):
        fm = _parse_frontmatter(open(f).read()) or {}
        impls = [str(i) for i in (fm.get("implemented_by") or [])]
        if not any(rel in i or os.path.basename(runner_path) in i for i in impls):
            continue
        bid = fm.get("id", os.path.basename(f))
        for rule in (fm.get("protocol") or []):
            if not isinstance(rule, dict) or not rule.get("check"):
                continue                                  # schema problems are reported by check_protocol_schema
            key, op, operand, err = parse_check(rule["check"])
            if err:
                continue                                  # ditto -- do not report the same defect twice
            checked += 1
            label, why = rule.get("rule", "(unnamed)"), rule.get("why", "")
            tail = "\n        RULE: %s\n        WHY: %s" % (label, why)
            status, val, ev = resolve_config_default(text, key)
            if status == "missing":
                problems.append("%s: ⛔ PROTOCOL UNCHECKABLE — %s constrains '%s' but no default for it was found "
                                "in the runner (looked for an add_argument, an assignment, and a mapping entry)."
                                "%s" % (rel, bid, key, tail))
                continue
            if status == "nodefault":
                problems.append("%s: ⛔ PROTOCOL UNENFORCEABLE — %s constrains '%s' but the runner gives it NO "
                                "default [%s], so the value comes from the command line and this checker cannot "
                                "hold it.  Give it the biological default.%s" % (rel, bid, key, ev, tail))
                continue
            if status == "unparsed":
                problems.append("%s: ⛔ PROTOCOL NOT EVALUABLE — '%s' resolves to %s [%s], which is not a literal. "
                                "UNDEFINED is not a pass: make the default a literal, or drop the rule.%s"
                                % (rel, key, val, ev, tail))
                continue
            if status == "ambiguous":
                problems.append("%s: ⛔ PROTOCOL AMBIGUOUS — '%s' resolves to %d different values in the runner "
                                "(%s) [%s]. A resolver that picks one is a check that cannot be trusted.%s"
                                % (rel, key, len(val), ", ".join(str(v) for v in val), ev, tail))
                continue
            ok, detail = evaluate_check(val, op, operand)
            if ok is None:
                problems.append("%s: ⛔ PROTOCOL NOT EVALUABLE — %s: %s%s" % (rel, bid, detail, tail))
            elif not ok:
                problems.append("%s: ⛔ PROTOCOL CONTRADICTS BIOLOGY — '%s' is %s but %s requires %s [%s].%s"
                                % (rel, key, _fmt(val), bid, detail, ev, tail))
    if verbose and checked:
        print("  %s: %d protocol rule(s) checked" % (rel, checked))
    return problems


def check_mechanism_status(verbose=True):
    """ONE mechanism -> ONE current status. Flags conflicts instead of leaving them to be re-derived.

    THE FAILURE THIS CLOSES (owner, 2026-07-31): "we run so many experiments touching adjacent regions/runners
    that we end up with results and memories citing the SAME flags/code/biology but reaching DIFFERENT
    conclusions, so determining the latest status lands on something irrelevant or outdated."

    Measured: the string "btsp" appears in 96 findings; 41 of them mention GO and 35 mention
    NO-GO/NEGATIVE/REFUTED/BOUNDARY. Asking "what is the status of BTSP?" returns ninety-six documents that
    disagree. Nothing says which one is CURRENT.

    So a mechanism entry names `current_finding:`, and the rule is enforced from BOTH ends:
      * `current_finding` must exist and be declared `status: live`;
      * any OTHER finding declaring `mechanism: <id>` with `status: live` is an UNRESOLVED CONFLICT -- exactly
        two live answers to one question -- and must be superseded or the current one updated.
    Supersession then has to be recorded where retrieval will see it, not merely known.
    """
    import glob as _g
    problems = []
    # every finding's declared (mechanism, status)
    declared = {}
    for f in _g.glob(os.path.join(ROOT, "research", "findings", "*.md")):
        try:
            with open(f, errors="ignore") as fh:
                head = "".join(next(fh, "") for _ in range(15))
        except Exception:
            continue
        if not head.startswith("---"):
            continue
        fmz = head.split("\n---", 1)[0]
        mm = re.search(r"^mechanism:\s*([A-Za-z0-9_.-]+)\s*$", fmz, re.M)
        st = re.search(r"^status:\s*([a-z-]+)\s*$", fmz, re.M)
        if mm:
            declared.setdefault(mm.group(1), []).append(
                (os.path.relpath(f, ROOT), st.group(1) if st else "undeclared"))

    for bf in sorted(_g.glob(os.path.join(BIO_DIR, "*.md"))):
        fm = _parse_frontmatter(open(bf).read()) or {}
        mid = fm.get("id")
        cur = fm.get("current_finding")
        if not mid:
            continue
        if cur:
            cp = _expand(str(cur))
            if not os.path.exists(cp):
                problems.append("%s: current_finding does not exist: %s" % (mid, cur))
            else:
                st = dict(declared.get(mid, [])).get(os.path.relpath(cp, ROOT))
                if st and st != "live":
                    problems.append("%s: current_finding is declared '%s', not 'live': %s" % (mid, st, cur))
        lives = [p for p, st in declared.get(mid, []) if st == "live"]
        extra = [p for p in lives if not cur or os.path.relpath(_expand(str(cur)), ROOT) != p]
        if len(lives) > 1 and extra:
            problems.append(
                "%s: UNRESOLVED CONFLICT — %d findings declare this mechanism LIVE. Exactly one can be current; "
                "supersede the others or update current_finding:\n        %s"
                % (mid, len(lives), "\n        ".join(lives)))
        if verbose and mid:
            print("  %-34s current=%s  live-declared=%d"
                  % (mid, os.path.basename(str(cur)) if cur else "(none)", len(lives)))
    return problems


def check_config(runner_path, verbose=True):
    """Compare a runner's argparse defaults against every biology entry that claims to be implemented by it."""
    text = open(runner_path).read()
    rel = os.path.relpath(os.path.abspath(runner_path), ROOT)
    problems, checked = [], 0
    for f in sorted(glob.glob(os.path.join(BIO_DIR, "*.md"))):
        fm = _parse_frontmatter(open(f).read()) or {}
        impls = [str(i) for i in (fm.get("implemented_by") or [])]
        if not any(rel in i or os.path.basename(runner_path) in i for i in impls):
            continue
        for c in (fm.get("constraints_config") or []):
            if not isinstance(c, dict):
                continue
            key, want, why = c.get("key"), c.get("value"), c.get("why", "")
            if not key or want is None:
                continue
            checked += 1
            m = re.search(r'add_argument\(\s*["\']--%s["\'][^)]*?default\s*=\s*([^,)\s]+)' % re.escape(key), text)
            if not m:
                m = re.search(r'\b%s\s*=\s*([0-9.]+)' % re.escape(key), text)
            if not m:
                problems.append("%s: biology '%s' constrains '%s' but it was not found in the runner"
                                % (rel, fm.get("id"), key))
                continue
            actual = m.group(1).strip()
            try:
                same = abs(float(actual) - float(want)) < 1e-9
            except ValueError:
                same = actual.strip("\"'") == str(want)
            if not same:
                problems.append(
                    "%s: ⛔ CONFIG CONTRADICTS BIOLOGY — '%s' default is %s but %s requires %s.\n        WHY: %s"
                    % (rel, key, actual, fm.get("id"), want, why))
    if verbose:
        print("  %s: %d biology constraint(s) checked" % (rel, checked))
    return problems


_ST_ENTRY = """---
type: biology
id: %(bid)s
mechanism: selftest fixture
implemented_by:
  - %(runner)s
%(protocol)s---
selftest fixture
"""


def _st_case(td, i, protocol_yaml, runner_src, in_scope=True):
    """Build a throwaway entry+runner pair in `td` and return check_protocol's problems for it."""
    global BIO_DIR
    bio = os.path.join(td, "case%02d" % i, "bio")
    rdir = os.path.join(td, "case%02d" % i, "runners")
    os.makedirs(bio); os.makedirs(rdir)
    rp = os.path.join(rdir, "st_runner_%02d.py" % i)
    open(rp, "w").write(runner_src)
    decl = os.path.relpath(rp, ROOT) if in_scope else "research/runners/_st_not_this_one.py"
    block = ""
    if protocol_yaml:
        block = "protocol:\n" + "".join("  " + ln + "\n" for ln in protocol_yaml.strip("\n").split("\n"))
    open(os.path.join(bio, "st%02d.md" % i), "w").write(
        _ST_ENTRY % {"bid": "st-%02d" % i, "runner": decl, "protocol": block})
    BIO_DIR = bio
    return check_protocol(rp, verbose=False)


def selftest():
    """Prove this checker still FAILS on the cases it exists to catch.  -> list of failures ([] == healthy).

    A selftest that only shows good input passing is vacuous, and this repo has shipped four gates that looked
    healthy while checking nothing.  So every CATCH case below is a config the biology forbids, asserted to
    produce a specific problem; every PASS case is the anti-cry-wolf direction (a gate that fires on legitimate
    configs gets switched off, which is worse than no gate); and the last block PINS the legacy
    `constraints_config` strings byte for byte, because that path is live and blocking a real defect and this
    change must not have moved it.
    """
    import tempfile
    global BIO_DIR
    saved_bio, fails, n_catch, n_pass = BIO_DIR, [], 0, 0

    # (expect_marker, protocol_yaml, runner_src).  expect_marker None => must produce NO problems.
    CATCH = [
        # 1. THE EARNED DEFECT: BTSP is one-shot and the runner default was 5 (4.40x -> 2.57x -> 1.11x).
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=int, default=5)'),
        # 2. INEQUALITY, low side -- the shape `invariance-from-temporal-continuity` refuses to write as equality.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "the trace must outlast one presentation"\n  why: "a bout of one is a no-op"\n'
         '  check: bout_len gte 2',
         'p.add_argument("--bout-len", type=int, default=1)'),
        # 3. INEQUALITY, high side.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "dwell within the plateau window"\n  why: "past it the kernel has decayed"\n'
         '  check: dwell_ms lte 30',
         'p.add_argument("--dwell-ms", type=float, default=60.0)'),
        # 4. BOOLEAN store_true -- the systems-consolidation case the legacy matcher cannot see AT ALL.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "the recall read must not write"\n  why: "drift +1.28-1.41 live vs +0.000000 frozen"\n'
         '  check: freeze_read eq true',
         'p.add_argument("--freeze-read", action="store_true")'),
        # 5. MEMBERSHIP over a string mode.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "present a category contiguously"\n  why: "shuffled order collapses 0.958 -> 0.556"\n'
         '  check: presentation in grouped,contiguous',
         'p.add_argument("--presentation", default="shuffled")'),
        # 6. DASHED CLI FORM of an underscored key -- measured blind spot of the legacy matcher.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "the AdEx integrator must not blow up"\n  why: "at dt=0.5 V sticks at +45.9 mV"\n'
         '  check: dt_ms eq 0.1',
         'p.add_argument("--dt-ms", type=float, default=0.5)'),
        # 7. dest= aliasing.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "the recall read must not write"\n  why: "a read that writes is not a read"\n'
         '  check: freeze_read eq true',
         'p.add_argument("--no-write", dest="freeze_read", action="store_true")'),
        # 8. NESTED PARENS before the default -- the legacy `[^)]*?` truncates here and sees nothing.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=lambda s: int(float(s)), default=5)'),
        # 9. True == 1 must NOT satisfy a boolean rule.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "the recall read must not write"\n  why: "a read that writes is not a read"\n'
         '  check: freeze_read eq true',
         'p.add_argument("--freeze-read", type=int, default=1)'),
        # 10. bare module-level assignment, string valued (legacy sees numbers only).
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "present a category contiguously"\n  why: "shuffled order collapses 0.958 -> 0.556"\n'
         '  check: presentation eq grouped',
         'presentation = "shuffled"'),
        # 11. mapping-entry form.
        ("PROTOCOL CONTRADICTS BIOLOGY",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'CFG = {"laps": 5, "seed": 42}'),
        # 12. the key is simply absent.
        ("PROTOCOL UNCHECKABLE",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--unrelated", type=int, default=3)'),
        # 13. present but with NO default -- unenforceable, and that must be SAID, not passed.
        ("PROTOCOL UNENFORCEABLE",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=int)'),
        # 14. a non-literal default: UNDEFINED is not a pass.
        ("PROTOCOL NOT EVALUABLE",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=int, default=DEFAULT_LAPS)'),
        # 15. two different resolutions -> say so, do not pick one.
        ("PROTOCOL AMBIGUOUS",
         '- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=int, default=1)\nq.add_argument("--laps", type=int, default=5)'),
        # 16. an ordered op against a string is a mistake, not a silent nothing.
        ("PROTOCOL NOT EVALUABLE",
         '- rule: "mis-typed rule"\n  why: "author used lte on a mode name"\n  check: presentation lte 5',
         'p.add_argument("--presentation", default="grouped")'),
        # 17. an ordered op against a BOOLEAN is a category error too -- "is the read frozen" has no >=.
        ("PROTOCOL NOT EVALUABLE",
         '- rule: "mis-typed rule"\n  why: "author used gte on a boolean flag"\n  check: freeze_read gte 1',
         'p.add_argument("--freeze-read", action="store_true")'),
    ]
    PASS = [
        # the compliant version of every catch above must be SILENT, or the gate cries wolf and gets disabled.
        ('- rule: "one-shot induction"\n  why: "five laps erase the field"\n  check: laps eq 1',
         'p.add_argument("--laps", type=int, default=1)'),
        ('- rule: "the trace must outlast one presentation"\n  why: "a bout of one is a no-op"\n'
         '  check: bout_len gte 2',
         'p.add_argument("--bout-len", type=int, default=12)'),
        ('- rule: "dwell within the plateau window"\n  why: "past it the kernel has decayed"\n'
         '  check: dwell_ms lte 30',
         'p.add_argument("--dwell-ms", type=float, default=30.0)'),
        ('- rule: "the recall read must not write"\n  why: "drift +1.28-1.41 live vs +0.000000 frozen"\n'
         '  check: freeze_read eq true',
         'p.add_argument("--live-read", dest="freeze_read", action="store_false")'),
        ('- rule: "present a category contiguously"\n  why: "shuffled order collapses 0.958 -> 0.556"\n'
         '  check: presentation in grouped,contiguous',
         'p.add_argument("--presentation", default="grouped")'),
        ('- rule: "the AdEx integrator must not blow up"\n  why: "at dt=0.5 V sticks at +45.9 mV"\n'
         '  check: dt_ms eq 0.1',
         'p.add_argument("--dt-ms", type=float, default=0.1)'),
        ('- rule: "not a one-lap protocol"\n  why: "the inverse direction must also be quiet"\n'
         '  check: laps ne 5',
         'p.add_argument("--laps", type=int, default=1)'),
        ('- rule: "excluded modes"\n  why: "not_in must pass when the value is absent from the set"\n'
         '  check: presentation not_in shuffled,random',
         'p.add_argument("--presentation", default="grouped")'),
    ]

    try:
        with tempfile.TemporaryDirectory() as td:
            for i, (marker, proto, src) in enumerate(CATCH):
                n_catch += 1
                probs = _st_case(td, i, proto, src)
                if not any(marker in p for p in probs):
                    fails.append("CATCH case %d did NOT fire (%s): expected %r, got %r"
                                 % (i + 1, src.strip()[:56], marker, probs))
            for j, (proto, src) in enumerate(PASS):
                n_pass += 1
                probs = _st_case(td, 100 + j, proto, src)
                if probs:
                    fails.append("PASS case %d cried wolf (%s): %r" % (j + 1, src.strip()[:56], probs))
            # scoping: an entry that does not name this runner must not be evaluated against it.
            if _st_case(td, 200, '- rule: "one-shot"\n  why: "w"\n  check: laps eq 1',
                        'p.add_argument("--laps", type=int, default=5)', in_scope=False):
                fails.append("SCOPE: a biology entry whose implemented_by omits the runner was still evaluated "
                             "against it — this gate would flag unrelated files")

            # ---- schema: a rule that can never be evaluated must fail LOUDLY, not sit there looking green ----
            SCHEMA = [
                ("UNPARSEABLE", '- rule: "r"\n  why: "w"\n  check: laps leq 1'),        # typo'd operator
                ("UNPARSEABLE", '- rule: "r"\n  why: "w"\n  check: laps'),              # not a triple
                ("UNPARSEABLE", '- rule: "r"\n  why: "w"\n  check: presentation in ,'),  # empty set
                ("missing 'why'", '- rule: "r"\n  check: laps eq 1'),                    # undocumented rule
                ("missing 'rule'", '- why: "w"\n  check: laps eq 1'),
                ("missing 'check'", '- rule: "r"\n  why: "w"'),
            ]
            for marker, proto in SCHEMA:
                n_catch += 1
                fm = _parse_frontmatter(_ST_ENTRY % {"bid": "st-schema", "runner": "x.py",
                                                     "protocol": "protocol:\n" + "".join(
                                                         "  " + ln + "\n" for ln in proto.split("\n"))})
                got = check_protocol_schema(fm or {}, "st-schema.md")
                if not any(marker in g for g in got):
                    fails.append("SCHEMA case %r did NOT fire: got %r" % (marker, got))
            fm_ok = _parse_frontmatter(_ST_ENTRY % {"bid": "st-schema", "runner": "x.py",
                                                    "protocol": 'protocol:\n  - rule: "r"\n    why: "w"\n'
                                                                '    check: laps eq 1\n'})
            if check_protocol_schema(fm_ok or {}, "st-schema.md"):
                fails.append("SCHEMA: a well-formed protocol rule was rejected: %r"
                             % check_protocol_schema(fm_ok or {}, "st-schema.md"))

            # ---- the legacy path is LIVE and BLOCKING: pin its exact strings, not merely its behaviour ----
            n_catch += 1
            lbio = os.path.join(td, "legacy", "bio"); os.makedirs(lbio)
            lrun = os.path.join(td, "legacy", "r_legacy.py")
            open(lrun, "w").write('p.add_argument("--laps", type=int, default=5)\n'
                                  'p.add_argument("--freeze-read", action="store_true")\n')
            rel = os.path.relpath(lrun, ROOT)
            open(os.path.join(lbio, "legacy.md"), "w").write(
                "---\ntype: biology\nid: bi-pin\nmechanism: m\nimplemented_by:\n  - %s\n"
                "constraints_config:\n  - key: laps\n    value: 1\n    why: \"one-shot\"\n"
                "  - key: freeze_read\n    value: 1\n    why: \"unseeable by design\"\n---\nb\n" % rel)
            BIO_DIR = lbio
            legacy = check_config(lrun, verbose=False)
            expect = [
                "%s: ⛔ CONFIG CONTRADICTS BIOLOGY — 'laps' default is 5 but bi-pin requires 1."
                "\n        WHY: one-shot" % rel,
                "%s: biology 'bi-pin' constrains 'freeze_read' but it was not found in the runner" % rel,
            ]
            if sorted(legacy) != sorted(expect):
                fails.append("LEGACY constraints_config DRIFTED — it is live and blocking a real defect.\n"
                             "          expected: %r\n          got:      %r" % (expect, legacy))
    finally:
        BIO_DIR = saved_bio

    if n_catch < 20 or n_pass < 6:
        fails.append("the selftest itself thinned out (%d catch / %d pass cases) — a shrinking selftest is how a "
                     "gate quietly stops checking" % (n_catch, n_pass))
    return fails


def main():
    args = sys.argv[1:]
    problems = []
    if "--selftest" in args:
        fails = selftest()
        for f in fails:
            print("  ⛔ SELFTEST: %s" % f)
        print("  => %s" % ("⛔ %d selftest failure(s) — this checker cannot be trusted until they are fixed"
                           % len(fails) if fails
                           else "✔ selftest: the checker FAILS on every case it is supposed to catch"))
        return 1 if fails else 0
    if "--config" in args:
        for p in args[args.index("--config") + 1:]:
            problems += check_config(p)
            problems += check_protocol(p)
    else:
        entries = sorted(glob.glob(os.path.join(BIO_DIR, "*.md")))
        if not entries:
            print("biology_check: no entries in research/biology/ — nothing to check.")
            return 0
        print("biology_check: %d entr(ies)" % len(entries))
        for e in entries:
            problems += check_entry(e)
        problems += check_mechanism_status()
        for e in entries:
            fm = _parse_frontmatter(open(e).read()) or {}
            for impl in (fm.get("implemented_by") or []):
                ip = _expand(str(impl))
                if os.path.exists(ip) and ip.endswith(".py"):
                    problems += check_config(ip, verbose=False)
                    problems += check_protocol(ip, verbose=False)
    for p in problems:
        print("  ⛔ %s" % p)
    print("  => %s" % ("⛔ %d problem(s)" % len(problems) if problems
                       else "✔ every source resolves and no config contradicts its biology"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
