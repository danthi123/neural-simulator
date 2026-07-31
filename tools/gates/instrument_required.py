"""CLASS I — INSTRUMENT-REQUIRED: a GO that reports a SIZE without a SOURCE.

THE FAILURE CLASS (cause 4 of `2026-07-31-why-we-hit-walls-the-missing-companion-process.md`): *"usually we
cannot tell WHICH of the above is happening."* Half of 2026-07-31 went into building decomposition instruments,
and **every one relocated the problem**:

  * raw ON/OFF weights          — lane D was recorded as weight COLLAPSE; it is common-mode convergence
                                  (`on_mean` 9.1544 vs `off_mean` 9.17, `AGG_norm_arms.json`).
  * the sign budget             — gap#5's tuned point was recorded as the best field; **97% of the measured dW
                                  was the CLAMP**, identical in the `lr=0` control (`AGG_clamp_budget.json`).
  * the position-shuffled null  — the tuned `circ_dW` 0.7050 was increment CONCENTRATION, not place-specificity
                                  (shuffle ratio 1.01, p=0.42).

In all three the magnitude was measured correctly and the SOURCE was assumed. So: a mechanism does not get a GO
until its DECOMPOSITION MEASUREMENT exists — the instrument that says WHERE the effect came from, not merely how
big it is. **The instrument is part of the emulation.**

THE GATE. A finding whose HEADLINE asserts a positive verdict (a bare `GO`, or SOLVED/WORKS/CONFIRMED/VALIDATED/
BREAKTHROUGH/SURPASSED/CLOSED/SUCCESS in caps) must carry decomposition evidence, by ANY of:

  1. an explicit `instrument:` line — frontmatter, a body line, or `**Instrument:**` — naming HOW the effect was
     attributed (>= 12 chars of reason; `instrument: yes` does not pass);
  2. a CITED `research/findings/raw/**.json` artifact carrying a decomposition FIELD — a key whose tokens name a
     null/shuffle/permutation, a budget/attribution/share, or a lesion/control/arm comparison;
  3. the decomposition REPORTED in the prose — a term from the same three families with a digit within 160 chars,
     which is the cheap separator between a decomposition that was MEASURED and one that was merely mentioned
     ("future work: run a lesion control" carries no number and does not pass).

Families 1-3 mirror the three instruments above: null/shuffle, budget/attribution, control-comparison.

CALIBRATION, against this tree (`python -m tools.gates.instrument_required`). 1845 findings; **283 carry
frontmatter**; of those **87 assert a live positive headline** (retracted/superseded/void skipped); **12 fire —
13.8%**, none later than 2026-07-20, and every one a GO resting on a bare score, a test count, or a runnable
demo. The other 1562 legacy findings are NEVER scanned: frontmatter is the opt-in, exactly as in `single_seed`,
because retro-firing on a thousand historical documents is the cry-wolf failure that gets a gate switched off.
In staged mode an EMPTY path list returns nothing; only `paths=None` audits the corpus.

ROUTE COVERAGE, measured independently rather than assumed: prose 75/87, artifact 29/87, declared 0/87 — and
the 29 artifact-passers are a strict SUBSET of the 75. **On today's corpus the artifact route adds ZERO
independent coverage**, and the declared route none at all because `instrument:` is new here. Stated because a
three-route gate that is really a one-route gate is the shape of a check that cannot fail; routes 1 and 2 are
kept for the terse finding that cites an aggregate instead of tabulating it, not because they are pulling
weight today.

Which VOCABULARY carries it, by deletion (marginal fires if the term were removed): `byte-identical`/`vs
isolated` **+7** — the one-brain and additive-default-off findings, where the reference comparison IS the
instrument; `oracle` **+1** (a `| numpy oracle | 1.000 |` results row); `anti-cheat` **+1** (this project's own
name for a control); `N arms` and `lr=0` **+0** — carried entirely by other terms today and kept only against
a future terse write-up. The loosest two were audited by hand at the single document each carries, and both are
real reference comparisons rather than passing mentions.

DIVERGENCE FROM `single_seed`, deliberate and named: that gate also skips `status: corrected`. This one does
NOT — a corrected finding still asserts its revised GO, and a correction is itself evidence that the original
instrument was inadequate, so it is the last population to exempt. **Measured effect today: none.** Skipping
`corrected` would drop the candidate pool 87 → 77 and leave the fire count at 12; all 10 corrected candidates
already pass. The divergence is a forward-looking choice, not a difference this tree demonstrates.

WHAT THIS GATE CANNOT CATCH.
  * Whether the decomposition is CORRECT, or bound to the claim it is quoted for. A permutation null computed
    over the wrong axis passes; so does a lesion arm cited for a different result in the same document. This is
    presence, not validity — the same limit `artifact_provenance` states for provenance.
  * A number near the term that has nothing to do with it. A digit is a weak separator; a section heading
    "## 3. Lesion plans" passes. It rejects the plainly-hypothetical, not the artfully-worded.
  * An `instrument:` line that names an instrument nobody ran. The escape hatch is a DECLARATION, and its value
    is that it forces the author to name something falsifiable — not that it verifies it.
  * A GO asserted only in a commit message, on the board, or in chat; and any finding without frontmatter.
  * Findings in SUBDIRECTORIES of `research/findings/`. The audit globs the flat directory only (1 such file
    exists today); staged mode does check them, so the two modes disagree by one file. Same asymmetry as
    `single_seed`, carried deliberately rather than diverging from the sibling gate on a 1-file difference.
  * The reverse error — a NEGATIVE reported without decomposition. Lane D's "weight collapse" was a NO-GO
    misattributed for exactly this reason, and this gate would not have fired on it. Scoping to positive
    headlines is what keeps the fire rate at 15% instead of 60%; the negative half is genuinely uncovered.
  * Existence / wire-up / byte-identical GOs that claim no effect size at all. `byte-identical` and `vs isolated`
    are accepted as control-comparisons so these mostly pass, but a demo GO with neither still fires and its
    honest remedy is the `instrument:` line, not a control it never needed.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re

NAME = "instrument-required"
CLASS_ID = "I"
BLOCKING = True

_POSITIVE = re.compile(r"\b(SOLVED|WORKS|CONFIRMED|VALIDATED|BREAKTHROUGH|SURPASSED|CLOSED|SUCCESS)\b")
_NEGATIVE = re.compile(r"⛔|\b(RETRACT\w*|VOID|NO-GO|NEGATIVE|REFUTED|CONFOUNDED|WITHDRAWN|FALSE)\b")
_GO_FALSE_FRIENDS = re.compile(r"NO[-/ ]GO|GO[-/ ]NO[-/ ]GO|GO[- ]gates?", re.I)
_WITHDRAWN = ("retracted", "superseded", "void")

# The declaration route. Accepts the frontmatter key, a body line, and the `**Instrument:**` bold form these
# documents actually use for `**Status:**` / `**Verdict:**`.
_INSTRUMENT = re.compile(r"^\s*(?:\*\*|__)?instrument(?:\*\*|__)?\s*:\s*(.+)$", re.I | re.M)
_MIN_REASON = 12

# ---------------------------------------------------------------------------------------------------------
# The decomposition vocabulary — three families, one per instrument built on 2026-07-31.
#
# `attribution`, NOT `attribut\w+`: this project writes "multi-attribute binding" constantly, and the loose form
# scored 8 free passes (incl. one on the string "AttributeError") before the corpus calibration caught it. A
# vocabulary that matches the project's own jargon is a gate that cannot fail.
# A bare `control` is likewise excluded — "gain control", "divisive normalisation control" and "version control"
# are mechanisms and tooling, not instruments. A NAMED control carries a modifier, so a hyphenated one qualifies
# ("no-learning controls", "noise-input control").
# ---------------------------------------------------------------------------------------------------------
_TERM = re.compile(
    r"("
    # -- null / shuffle / permutation ------------------------------------------------------------------
    r"shuffl\w+|permut\w+|perm[_ ]null|surrogate|scrambl\w+|null\s+(?:model|distribution|hypothesis)|"
    r"(?:vs\.?|versus|above|below|at)\s+chance|chance[- ]level|ratio_vs_chance|"
    # -- budget / attribution --------------------------------------------------------------------------
    r"budget|attribution|attributed to|attributable|decompos\w+|variance explained|"
    # -- control comparison ----------------------------------------------------------------------------
    r"lesion\w*|ablat\w*|knock[- ]?out|counterfactual|anti[- ]cheat|"
    r"control\s+(?:arm|condition|group|run|comparison)|"
    r"(?:[a-z]+-[a-z]+|positive|negative|matched|frozen|untrained|shuffled|scrambled|lesioned)[- ]controls?\b|"
    r"lr\s*=\s*0|\d\s*arms?\b|\barms?\s*[:=]|byte[- ]identical|vs\.?\s+isolated|oracle"
    r")", re.I)
_NUM = re.compile(r"\d")
_WINDOW = 160

_ARTIFACT = re.compile(r"research/findings/raw/[^\s`\"')\]]+\.json")
_KEY_TOKENS = {
    "null", "nulls", "shuffle", "shuffled", "shuffles", "perm", "permuted", "permutation", "permutations",
    "surrogate", "scramble", "scrambled", "chance", "budget", "attribution", "decomposition", "decomp",
    "lesion", "lesioned", "ablation", "ablated", "knockout", "counterfactual", "control", "controls",
    "arm", "arms", "oracle", "share", "pval", "pvalue", "anticheat",
}
_KEY_PREFIXES = ("perm", "shuffl", "permut", "lesion", "ablat", "scrambl", "decomp", "attribut", "counterfact")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frontmatter(text):
    """The YAML block as a flat lowercase dict, or None when the file carries no frontmatter."""
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


def _declared_instrument(text, fm):
    """(True, how) when the document NAMES how the effect was attributed, with a real reason."""
    for key in ("instrument", "instruments"):
        v = (fm or {}).get(key, "")
        if len(v.strip()) >= _MIN_REASON:
            return True, "frontmatter instrument:"
    for m in _INSTRUMENT.finditer(text):
        if len(m.group(1).strip().strip("*_`").strip()) >= _MIN_REASON:
            return True, "declared instrument: line"
    return False, ""


def _key_is_decomposition(key):
    toks = [t for t in re.split(r"[^a-z0-9]+", str(key).lower()) if t]
    for t in toks:
        if t in _KEY_TOKENS or t.startswith(_KEY_PREFIXES):
            return True
    return False


def _walk_keys(obj, out, depth=0):
    if depth > 4 or len(out) > 4000:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _walk_keys(v, out, depth + 1)
    elif isinstance(obj, list):
        for e in obj[:30]:
            _walk_keys(e, out, depth + 1)


def _artifact_decomposition(text, root):
    """(True, how) when a CITED raw artifact carries a decomposition field."""
    files = []
    for rel in set(_ARTIFACT.findall(text)):
        files += sorted(_glob.glob(os.path.join(root, rel)))[:40]
    for path in files[:60]:
        try:
            if os.path.getsize(path) > 8_000_000:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue                     # an unreadable artifact contributes NO evidence, so the gate still fires
        keys = set()
        _walk_keys(data, keys)
        for k in sorted(keys):
            if _key_is_decomposition(k):
                return True, "artifact field %r in %s" % (k, os.path.basename(path))
    return False, ""


def _prose_decomposition(text):
    """(True, how) when a decomposition term is REPORTED — i.e. sits within 160 chars of a digit."""
    for m in _TERM.finditer(text):
        window = text[max(0, m.start() - _WINDOW):m.end() + _WINDOW]
        if _NUM.search(window):
            return True, "reported %r" % m.group(0).strip().lower()[:40]
    return False, ""


def _label(path):
    p = path.replace(os.sep, "/")
    return p.split("research/findings/")[-1] if "research/findings/" in p else p


def _evidence(path, text, fm):
    """(ok, how). The three acceptance routes, cheapest first."""
    ok, how = _declared_instrument(text, fm)
    if ok:
        return True, how
    ok, how = _prose_decomposition(text)
    if ok:
        return True, how
    root = path.replace(os.sep, "/").split("research/findings/")[0] or "."
    return _artifact_decomposition(text, root)


def _candidates(paths):
    if paths is None:
        return sorted(_glob.glob(os.path.join(_repo_root(), "research", "findings", "*.md")))
    return [p for p in paths
            if p.endswith(".md")
            and "research/findings/" in p.replace(os.sep, "/")
            and "/raw/" not in p.replace(os.sep, "/")]


def check(paths):
    # An EMPTY list means "staged mode, nothing of my kind staged". Only paths=None means "audit the corpus";
    # without this split the pre-commit driver's scoping is undone by the gate's own corpus fallback.
    if paths is not None and len(paths) == 0:
        return []
    problems = []
    for path in _candidates(paths):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue                                  # a staged path may be a deletion; not this gate's class
        fm = _frontmatter(text)
        if fm is None or fm.get("status", "").lower() in _WITHDRAWN:
            continue
        verdict = _asserts_positive(_headline_zone(text))
        if not verdict:
            continue
        ok, _how = _evidence(path, text, fm)
        if ok:
            continue
        problems.append(
            "CLASS I %s asserts %s with NO decomposition — a SIZE without a SOURCE: nothing in the document, "
            "its frontmatter, or its cited artifacts says WHERE the effect came from (no null/shuffle, no "
            "budget/attribution, no lesion/control-arm comparison). This is the shape that made lane D read as "
            "'weight collapse' when it was common-mode convergence, and made 97%% of gap#5's dW the clamp. Fix: "
            "report the null/control number, cite an artifact carrying the decomposition field, or add an "
            "'instrument: <how the effect was attributed>' line." % (_label(path), verdict))
    return problems


def audit():
    """Informational: how the WHOLE corpus scores. Never a blocker — see `artifact_provenance.corpus_rate`."""
    files = sorted(_glob.glob(os.path.join(_repo_root(), "research", "findings", "*.md")))
    total = len(files)
    fm_n = pos_n = 0
    routes = {"declared": 0, "prose": 0, "artifact": 0}
    firing = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        fm = _frontmatter(text)
        if fm is None:
            continue
        fm_n += 1
        if fm.get("status", "").lower() in _WITHDRAWN:
            continue
        if not _asserts_positive(_headline_zone(text)):
            continue
        pos_n += 1
        ok, how = _evidence(path, text, fm)
        if not ok:
            firing.append(os.path.basename(path))
        elif how.startswith("frontmatter") or how.startswith("declared"):
            routes["declared"] += 1
        elif how.startswith("reported"):
            routes["prose"] += 1
        else:
            routes["artifact"] += 1
    return {"findings": total, "with_frontmatter": fm_n, "live_positive_headlines": pos_n,
            "fire": len(firing), "fire_pct": round(100.0 * len(firing) / pos_n, 1) if pos_n else 0.0,
            "passed_via": routes, "firing": firing}


def selftest():
    """FAILING DIRECTION FIRST: a GO with a bare effect size and no instrument MUST be caught."""
    import tempfile

    head = "---\ntype: finding\nstatus: live\ndate: 2026-07-31\n%s---\n\n"
    # A GO reporting a SIZE and nothing else. This is the case the gate exists for.
    bare = (head % "") + ("# gap#7 SOLVED: the plateau-gated rule WORKS\n\n"
                          "**Status:** GO, 6 seeds 42/43/44/100/101/102 · effect 0.9100 against 0.4200.\n\n"
                          "The mechanism was enabled and the score rose. Committing.\n")
    # A term with NO number anywhere near it: MENTIONED, not measured. Padding keeps every digit >160 chars away.
    pad = ("\n\nSome further prose that carries no measurements at all and exists only to separate the mention "
           "below from any numeral appearing earlier in this document, so that the window test is exercised on "
           "a genuinely bare mention rather than on an accident of layout in the fixture itself. " * 2)
    fixtures = {
        # name                        body                                                      expect a fire?
        "a_bare_go.md":              (bare,                                                      True),
        "b_shuffle_null.md":         (bare + "Position-shuffled null 0.0324, ratio 4.40, median p 0.0050.\n",
                                                                                                 False),
        "c_lesion_arm.md":           (bare + "The lesion arm scores 0/6; the intact arm 3/3.\n",  False),
        "d_sign_budget.md":          (bare + "Sign budget: 97% of the measured dW was the clamp.\n",
                                                                                                 False),
        "e_instrument_fm.md":        ((head % "instrument: position-shuffled permutation null over 200 draws\n")
                                      + bare.split("---\n\n", 1)[1],                             False),
        "f_instrument_body.md":      (bare + "\ninstrument: lr=0 control arm isolates the clamp contribution\n",
                                                                                                 False),
        "g_instrument_bold.md":      (bare + "\n**Instrument:** ON/OFF raw weight decomposition per seed\n",
                                                                                                 False),
        "h_artifact_field.md":       (bare + "\nEvidence: `research/findings/raw/t/agg.json`\n",  False),
        # --- the cases that MUST still fire ---
        "i_artifact_no_decomp.md":   (bare + "\nEvidence: `research/findings/raw/t/plain.json`\n", True),
        "q_broken_citation.md":      (bare + "\nEvidence: `research/findings/raw/t/missing.json`\n", True),
        "j_mentioned_not_measured.md": (bare + pad + "Future work: run a lesion control." + pad,  True),
        "k_trivial_instrument.md":   (bare + "\ninstrument: yes\n",                               True),
        "l_attribute_false_friend.md": (bare + "\nMulti-attribute binding held at 0.9100 across seeds.\n", True),
        # --- calibration: these must NOT fire ---
        "m_no_frontmatter.md":       (bare.split("---\n\n", 1)[1],                                False),
        "n_retracted.md":            (bare.replace("status: live", "status: retracted"),          False),
        "o_negative_headline.md":    (bare.replace("SOLVED", "⛔ NOT SOLVED"),                     False),
        "p_corrected_still_checked.md": (bare.replace("status: live", "status: corrected"),       True),
    }
    bad = []
    with tempfile.TemporaryDirectory() as td:
        fdir = os.path.join(td, "research", "findings")
        os.makedirs(os.path.join(fdir, "raw", "t"))
        with open(os.path.join(fdir, "raw", "t", "agg.json"), "w") as fh:
            json.dump({"perm_null": 0.0324, "circ_dW": 0.1426}, fh)
        with open(os.path.join(fdir, "raw", "t", "plain.json"), "w") as fh:
            json.dump({"score": 0.91, "n": 6, "elapsed_seconds": 12}, fh)
        for name, (body, _) in fixtures.items():
            with open(os.path.join(fdir, name), "w") as fh:
                fh.write(body)
        for name, (_, should_fire) in fixtures.items():
            got = bool(check([os.path.join(fdir, name)]))
            if got != should_fire:
                bad.append("fixture %s: expected %s, gate returned %s"
                           % (name, "A FIRE" if should_fire else "no problem", "a fire" if got else "none"))
        # a mixed batch must report exactly the violators, not one verdict for the batch
        mixed = check([os.path.join(fdir, n) for n in
                       ("a_bare_go.md", "b_shuffle_null.md", "k_trivial_instrument.md", "m_no_frontmatter.md")])
        if len(mixed) != 2:
            bad.append("batch check returned %d problems, expected exactly 2" % len(mixed))
        if check([]):
            bad.append("check([]) returned problems; staged-mode-with-nothing-staged must be silent")
        # a deleted/nonexistent staged path must be skipped, never crash the registry
        if check([os.path.join(fdir, "deleted_by_this_commit.md")]):
            bad.append("a nonexistent staged path produced a problem")
        # THE ANTI-VACUITY CHECK. Every must-fire fixture above is a fixture I wrote; if the vocabulary were
        # widened until nothing could fire, the passing half would still be green. Assert the gate is capable
        # of firing on a document that differs from a PASSING one by exactly the decomposition sentence.
        near_miss = fixtures["b_shuffle_null.md"][0].replace(
            "Position-shuffled null 0.0324, ratio 4.40, median p 0.0050.", "The number went up.")
        p = os.path.join(fdir, "z_near_miss.md")
        with open(p, "w") as fh:
            fh.write(near_miss)
        if not check([p]):
            bad.append("GATE CANNOT FAIL: removing ONLY the null sentence from a passing document did not "
                       "make it fire — the pass was not caused by the decomposition evidence")
    return bad


if __name__ == "__main__":                                        # the audit view, never a blocker
    a = audit()
    print("class I instrument-required — %d findings, %d with frontmatter, %d live positive headlines, "
          "%d FIRE (%.1f%%)" % (a["findings"], a["with_frontmatter"], a["live_positive_headlines"],
                                a["fire"], a["fire_pct"]))
    print("  passed via: %s" % a["passed_via"])
    for f in a["firing"]:
        print("    FIRE  %s" % f)
