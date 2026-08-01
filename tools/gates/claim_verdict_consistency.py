"""CLASS CVV — A FINDING CLAIMS A GO / SURPASS / CLOSURE WHILE CITING AN ARTIFACT WHOSE OWN VERDICT IS NEGATIVE.

THE FAILURE, recorded 2026-08-01. The gap#4 "deep-credit-on-spikes closure" was banked as a session headline
AND a roadmap surpass banner ("closure ... reproduced-with-provenance ... K=16 above the LIF ceiling"), while the
e-prop artifacts it cited each printed `"SIGNAL": false` / a HONEST-NEGATIVE verdict and computed
`deep_credit_share` ≈ 0.005. The claim read the `inherit` field and never read the verdict the SAME run emitted.
This is silent-failure rule #1 ("never lift a metric out of a run whose own verdict is negative") + #7 ("a control
existed and was never read") — and it survived every existing gate:

  * `verdict_preconditions` checks an ARTIFACT's INTERNAL consistency (its verdict vs its own preconditions),
    not a doc's CLAIM against the artifact.
  * `claim_check` traces a finding's NUMBERS to cited artifacts, not the finding's VERDICT.
  * `terminology` / `closure_names_mechanism` check the WORD "closure", not whether the run it rests on agreed.

Nothing connected the DOC's positive claim to the cited run's NEGATIVE self-verdict. This module is that link.

WHAT IT CHECKS. For each staged `.md` under `research/findings/` whose frontmatter `status:` is `live` (a
correction/retraction/contributing doc is self-hedged and OUT of scope — that is the escape, not a loophole):
  claim   <- a line asserting a POSITIVE result — a word-boundary token in {GO, surpass(ed), closure, closed,
             reproduced, unlocked, SOLVED, "beats the reservoir"} that does NOT also carry a negation on the
             same line ({NOT, NO-GO, ⛔, retract, reservoir, fail, cannot, "does not", corrected, mirage,
             superseded, not established, ~0}). The negation guard is what lets a CORRECTION say "the closure was
             a reservoir" without tripping.
  artifact <- every cited `*.json` (frontmatter `artifacts:` + inline `research/.../*.json` paths) that exists.
  NEGATIVE <- that artifact's top-level `SIGNAL`/`signal` is false, OR its `verdict`/`status`/`VERDICT` string
             carries a negative token (HONEST NEGATIVE, NO-GO, "SIGNAL=False", "NEGATIVE", "not clean").
  FIRE when a live finding makes >=1 positive claim AND cites >=1 negative-verdict artifact.

The finding clears itself by (a) setting `status:` to superseded/contributing/retracted, (b) marking the claim
line as a correction (⛔ / RETRACT / "corrected"), or (c) not resting a positive claim on a run that failed —
which is the point. Owning the negative is exactly what the class asks for.

WHAT IT CANNOT CATCH: a claim that rests on a POSITIVE-verdict artifact that is itself wrong (that is
verify-go's job), or a claim citing no artifact (claim_check's job). It enforces one relationship: a live
GO-claim may not rest on a run that declared itself a NO-GO.
"""
from __future__ import annotations
import json
import os
import re

NAME = "claim-verdict-consistency"
CLASS_ID = "CVV"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAX_BYTES = 512 * 1024

_POS = re.compile(r"(?<![\w-])(GO|surpass(?:ed|es)?|closure|closed|reproduc(?:ed|es)|unlocked|SOLVED|"
                  r"beats?\s+(?:the\s+)?reservoir)(?![\w-])", re.I)
_NEG_LINE = re.compile(r"(?<![\w-])(not|no[- ]?go|retract|reservoir|fail|cannot|can'?t|mirage|superseded|"
                       r"contributing|does\s+not|doesn'?t|isn'?t|un-?converged|not\s+established|~\s*0|"
                       r"below|corrected)(?![\w-])|⛔", re.I)
# A verdict STRING is negative only if it OPENS with a negative verdict token, or explicitly records SIGNAL false.
# It must NOT match a POSITIVE verdict that merely mentions a negative word while describing its controls
# (a genuine "GO ... the NEGATIVE/permuted control collapses ... does NOT confabulate" is positive) — measured:
# that loose match false-flagged a real 6-seed GO (`_affect_distributional_tag_6seed.json`).
_NEG_ARTIFACT_START = re.compile(r"^\W*(HONEST[ _-]?NEGATIVE|NO[- ]?GO|NEGATIVE|UNDEFINED|POWERED\s+NO-?GO|"
                                 r"REFUTED|VOID)\b", re.I)
_NEG_ARTIFACT_ANY = re.compile(r"SIGNAL\s*[=:]\s*False", re.I)
_ART_INLINE = re.compile(r"research/[A-Za-z0-9_./-]+\.json")


def _frontmatter(text):
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end < 0:
        return "", text
    fm, body = text[3:end], text[end + 4:]
    return fm, body


def _fm_status(fm):
    m = re.search(r"^status:\s*([A-Za-z_-]+)", fm, re.M)
    return (m.group(1).lower() if m else "")


def _fm_artifacts(fm):
    out = []
    m = re.search(r"^artifacts:\s*$", fm, re.M)
    if m:
        for line in fm[m.end():].splitlines():
            am = re.match(r"\s*-\s*(\S+)", line)
            if am and am.group(1).endswith(".json"):
                out.append(am.group(1))
            elif line.strip() and not line.startswith((" ", "\t", "-")):
                break
    return out


def _artifact_is_negative(rel):
    full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
    if not os.path.isfile(full):
        return False
    try:
        if os.path.getsize(full) > _MAX_BYTES:
            return False
        d = json.loads(open(full, encoding="utf-8", errors="replace").read())
    except (OSError, ValueError):
        return False
    if not isinstance(d, dict):
        return False
    for k in ("SIGNAL", "signal", "GO", "go"):
        if k in d and d[k] is False:
            return True
    for k in ("verdict", "status", "VERDICT"):
        v = d.get(k)
        if isinstance(v, str) and (_NEG_ARTIFACT_START.search(v) or _NEG_ARTIFACT_ANY.search(v)):
            return True
    return False


def _check_doc(rel):
    full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
    try:
        text = open(full, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    fm, body = _frontmatter(text)
    if _fm_status(fm) != "live":               # corrections/contributing/superseded are self-hedged: out of scope
        return None
    # The claim locus is the finding's HEADLINE VERDICT, not any body line: a negative/correction finding's
    # TITLE says so ("...shuffle-DFA LEAKS", "...NEGATIVE"), so its title carries a negation and is cleared.
    # Scanning every body line flagged corrections that merely DISCUSS a positive word (measured: 4 hits, ≥1 a
    # false positive on this arc's own correction doc). The title + an explicit `One-line verdict:` line are the
    # only spots that ASSERT the finding's result.
    title = next((ln for ln in body.splitlines() if ln.startswith("# ")), "")
    claims = [title.strip()[:120]] if (_POS.search(title) and not _NEG_LINE.search(title)) else []
    if not claims:
        return None
    arts = _fm_artifacts(fm) + _ART_INLINE.findall(body)
    neg = [a for a in dict.fromkeys(arts) if _artifact_is_negative(a)]
    if not neg:
        return None
    return ("%s: a `status: live` finding asserts a POSITIVE result (\"%s\") while citing an artifact whose OWN "
            "verdict is NEGATIVE (%s: SIGNAL=false / HONEST-NEGATIVE). A GO/surpass/closure may not rest on a "
            "run that declared itself a NO-GO (silent-failure rule #1). Fix: read the run's verdict, not just a "
            "metric field; or if this IS the correction, set status: superseded/contributing or mark the claim "
            "line ⛔/RETRACT." % (rel, claims[0], os.path.basename(neg[0])))


def _staged(paths):
    return [p for p in (paths or []) if p.endswith(".md")
            and "research/findings/" in p.replace(os.sep, "/")]


def _audit():
    import glob
    return [os.path.relpath(p, _ROOT) for p in glob.glob(os.path.join(_ROOT, "research/findings/*.md"))]


def check(paths):
    if paths is not None and len(paths) == 0:
        return []
    files = _staged(paths) if paths else _audit()
    out = []
    for rel in files:
        r = _check_doc(rel)
        if r:
            out.append(r)
    return out


def selftest():
    """Must demonstrate the gate FIRING on a known-bad case (live GO-claim citing a SIGNAL=false artifact),
    AND staying silent on the escape (the same doc marked as a correction)."""
    import tempfile
    problems = []
    d = tempfile.mkdtemp(prefix="cvv_")
    art = os.path.join(d, "a.json")
    open(art, "w").write(json.dumps({"SIGNAL": False, "verdict": "HONEST NEGATIVE -- controls not clean"}))
    bad = os.path.join(d, "bad.md")
    open(bad, "w").write("---\nstatus: live\nartifacts:\n  - %s\n---\n\n# The deep-credit CLOSURE is reproduced "
                         "with provenance — GO\n\nfull result.\n" % art)
    hit = _check_doc(bad)
    if not hit:
        problems.append("SELFTEST BROKEN: gate did NOT fire on a live GO-claim citing a SIGNAL=false artifact")
    # escape: a correction must NOT fire
    good = os.path.join(d, "good.md")
    open(good, "w").write("---\nstatus: superseded\nartifacts:\n  - %s\n---\n\n# ⛔ RETRACT: the closure was a "
                          "reservoir\n" % art)
    if _check_doc(good):
        problems.append("SELFTEST BROKEN: gate FIRED on a correction (status: superseded) — escape failed")
    # a live claim citing a CLEAN artifact must NOT fire
    art2 = os.path.join(d, "b.json")
    open(art2, "w").write(json.dumps({"SIGNAL": True, "verdict": "GO"}))
    ok = os.path.join(d, "ok.md")
    open(ok, "w").write("---\nstatus: live\nartifacts:\n  - %s\n---\n\n# Deep credit GO\n" % art2)
    if _check_doc(ok):
        problems.append("SELFTEST BROKEN: gate FIRED on a claim citing a clean (SIGNAL=true) artifact")
    # Registry contract: EMPTY == the gate behaved correctly on every constructed case (fired on bad, silent on
    # the escapes). Non-empty means the gate is BROKEN. `hit` is asserted above (fires on bad) but not returned.
    assert hit, "gate must fire on the canonical bad case"
    return problems


if __name__ == "__main__":
    hits = check(None)
    print("CVV corpus audit: %d live finding(s) claim a positive result while citing a negative-verdict artifact"
          % len(hits))
    for h in hits[:40]:
        print("  ⛔", h[:160])
