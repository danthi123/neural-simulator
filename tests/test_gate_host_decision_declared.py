"""Host decision declared (CLASS HD, tools/gates/host_decision_declared.py): every prereg, packet or lane plan a commit
touches says up front whether it adds host code that decides content (CLAUDE.md non-negotiable 1, brain-based only),
and a scaffold cannot take another build past its cap -- in the same file, in a successor file, or by rewriting the
file's history -- without a Claude-signed grant on main. Owner 2026-10-21, after the moat allowlist's eight builds of
English-reading host rules (2026-10-20 prereg, Amendments 1-7) repeated the 2026-10-16 English-grammar-template plan.
Pure judge() cases, real `git diff --cached` in scratch repos, and real hooks for a clean merge; no brain, no GPU."""
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import smoke_first as SF  # noqa: E402
from tools.gates import host_decision_declared as GATE  # noqa: E402
from tools.gates import prereg_amendment_order as PRA  # noqa: E402

RUNBOOK = SF.RUNBOOK_REL
NEW = "research/findings/2026-10-22-newplan-PREREGISTRATION.md"
NEWB = os.path.basename(NEW)
OLD = "research/findings/2026-10-01-oldplan-PREREGISTRATION.md"
PLANS_PREREG = "docs/plans/2026-10-22-plan-PREREG.md"
PACKET_DRAFT = "docs/drafts/some-lane/PACKET.md"
PACKET_COORD = "research/coordination/some-lane-PACKET.md"
SPEC = "docs/drafts/some-lane/SPEC.md"
MOAT = "research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md"
TODAY = "2026-10-22"
RULINGS = ("# runbook\n\n## 5. Idle\n\n- idle-key: a section-5 bullet; cap-raise: %(n)s +1 builds (Claude)\n\n"
           "## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`)\n\n"
           "- cap-ok-1022: one more build of %(n)s, then a neural method\n"
           "  cap-raise: %(n)s +1 builds\n  (Claude 2026-10-22)\n"
           "- 2026-10-22 RULING dated-cap-key (Claude; owner-approved): cap-raise: %(n)s +1 builds\n"
           "- answers-key: cap-raise: %(n)s +1 builds (answers [lane] ASK 06:10; Claude)\n"
           "- two-more-1022: cap-raise: %(n)s +2 builds (Claude 2026-10-22)\n"
           "- hold-1022: HOLD. %(n)s stays default OFF and NO further build is approved (Claude 2026-10-22)\n"
           "- nogo-1022: build 3 of %(n)s is NO-GO on bar 3 (Claude 2026-10-22)\n"
           "- relay-1022: cap-raise: %(n)s +5 builds (Hermes, relaying what Claude said in chat)\n"
           "- unsigned-key: cap-raise: %(n)s +1 builds (Hermes 2026-10-22)\n"
           "- other-key: cap-raise: another-plan-PREREGISTRATION.md +1 builds (Claude 2026-10-22)\n"
           "- host-filter-ok-1022: a host filter may run as a teaching scaffold (Claude 2026-10-22)\n"
           "- triage: the test triage lane runs first (Claude 2026-10-02)\n\n"
           "## NEEDS CLAUDE\n\n- needs-key: cap-raise: %(n)s +1 builds (Claude)\n") % {"n": NEWB}
NONE = "host-decision: none it registers bars for a spiking organ; no host code chooses content\n"
FIELDS = {"decides": "decides: which mouth sentences are kept, by hand-written English rules\n",
          "allowed-by": "allowed-by: CLAUDE.md non-negotiable 1 (a teaching scaffold for its neural replacement)\n",
          "replaced-by": "replaced-by: board task #271, the deixis organ on the shared substrate\n",
          "cap": "cap: 3 builds\n"}


def scaffold(cap="3 builds", drop=(), **over):
    vals = dict(FIELDS, cap="cap: %s\n" % cap)
    for k, v in over.items():
        vals[k.replace("_", "-")] = "%s: %s\n" % (k.replace("_", "-"), v)
    return "host-decision: scaffold\n" + "".join(vals[k] for k in GATE._FIELDS if k not in drop)


def prereg(header="", body="# a plan\n\nThe bars.\n"):
    return "---\ntype: preregistration\nflip: no it registers bars only and flips no default at all\n%s---\n\n%s" % (
        header, body)


def amend(n, start=1):
    return "".join("\n## AMENDMENT %d (2026-10-22, Claude; filed before build %d)\nmore\n" % (k, k + 1)
                   for k in range(start, start + n))


def judge(text, rel=NEW, kind="prereg", runbook=RULINGS, parents=(), peers=None):
    return GATE.judge(rel, text, kind, lambda: runbook, parents=parents,
                      peers_fn=(lambda: peers) if peers is not None else None, today=TODAY)


# ---------------------------------------------------------------- the header, pure
def test_a_prereg_without_the_header_blocks_and_the_message_quotes_the_fields():
    p = judge(prereg())
    assert len(p) == 1 and "no `host-decision:` header" in p[0]
    for frag in ("host-decision: none", "host-decision: scaffold", "decides:", "allowed-by:", "replaced-by:",
                 "cap: <1-5> builds"):
        assert frag in p[0], frag


def test_none_with_a_short_reason_blocks():
    p = judge(prereg("host-decision: none spiking only\n"))
    assert len(p) == 1 and "at least 20 characters" in p[0] and "got 12" in p[0]
    p = judge(prereg("host-decision: none\n"))
    assert len(p) == 1 and "at least 20 characters" in p[0]
    p = judge(prereg("host-decision: none 0123456789012345678\n"))                    # 19 characters
    assert len(p) == 1 and "got 19" in p[0]


def test_none_with_a_reason_of_exactly_20_characters_passes():
    assert judge(prereg("host-decision: none 01234567890123456789\n")) == []


def test_none_with_the_template_placeholder_as_its_reason_blocks():
    p = judge(prereg("host-decision: none <why nothing it builds decides content on the host, at least 20 chars>\n"))
    assert len(p) == 1 and "at least 20 characters" in p[0]


def test_none_with_a_reason_passes_in_any_line_style():
    assert judge(prereg(NONE)) == []
    assert judge(prereg(body="# a plan\n\n- **host-decision:** `none` -- it measures a spiking organ only\n")) == []
    assert judge(prereg(body="# a plan\n\n> host-decision: none: it measures a spiking organ only, no host rule\n")) == []


@pytest.mark.parametrize("val", ["None", "NONE"])
def test_the_value_is_read_case_blind_none(val):
    assert judge(prereg(NONE.replace("none", val, 1))) == []


@pytest.mark.parametrize("val", ["Scaffold", "SCAFFOLD"])
def test_the_value_is_read_case_blind_scaffold(val):
    assert judge(prereg(scaffold().replace("scaffold", val, 1))) == []


def test_none_that_also_carries_scaffold_fields_blocks():
    p = judge(prereg(NONE + FIELDS["decides"]))
    assert len(p) == 1 and "a plan is `none` OR a scaffold" in p[0] and "`decides:`" in p[0]


def test_two_header_lines_block():
    p = judge(prereg(NONE, body="# a plan\n\n" + NONE))
    assert len(p) == 1 and "2 `host-decision:` lines" in p[0]


def test_a_header_only_inside_a_code_fence_does_not_count():
    p = judge(prereg(body="# a plan\n\n```\n" + NONE + "```\n"))
    assert len(p) == 1 and "no `host-decision:` header" in p[0]
    assert judge(prereg(NONE, body="# a plan\n\n```\n" + NONE + "```\n")) == []        # a quoted example is no 2nd line


def test_a_header_only_inside_an_html_comment_does_not_count():
    p = judge("# a plan\n\n<!--\n" + NONE + "-->\n\nbody\n")
    assert len(p) == 1 and "no `host-decision:` header" in p[0] and "HTML comment" in p[0]
    assert judge("# a plan\n\n" + NONE + "\n<!--\n" + NONE + "-->\n") == []              # a commented copy is no 2nd line


def test_a_header_only_in_an_indented_code_block_does_not_count():
    p = judge("# a plan\n\n    " + NONE)
    assert len(p) == 1 and "no `host-decision:` header" in p[0]
    p = judge("# a plan\n\n\t" + NONE)
    assert len(p) == 1 and "no `host-decision:` header" in p[0]


def test_the_header_must_be_up_front():
    p = judge("# a plan\n\n## 1. Bars\n\nbars\n\n" + NONE)
    assert len(p) == 1 and "after the first `##` heading (line 3)" in p[0]
    p = judge("# a plan\n" + "body\n" * 900 + "\n" + NONE)                               # no section at all
    assert len(p) == 1 and "within the first 60 lines" in p[0] and "on line 903" in p[0]
    assert judge("# a plan\n\n" + NONE + "\n## 1. Bars\n\nbars\n") == []


def test_an_unparsable_header_blocks_never_skips():
    p = judge(prereg("host-decision: maybe, after the next build lands\n"))
    assert len(p) == 1 and "cannot be parsed" in p[0] and "never skipped" in p[0]


def test_a_complete_scaffold_passes():
    assert judge(prereg(scaffold())) == []
    assert judge(prereg(body="# a plan\n\n" + "".join("- " + ln + "\n" for ln in scaffold().splitlines()))) == []


@pytest.mark.parametrize("field", GATE._FIELDS)
def test_a_scaffold_missing_any_field_blocks(field):
    p = judge(prereg(scaffold(drop=(field,))))
    assert len(p) == 1 and "lacks `%s:`" % field in p[0] and GATE._FIELD_HINT[field] in p[0]


@pytest.mark.parametrize("field", GATE._FIELDS)
def test_a_scaffold_field_separated_by_a_blank_line_is_outside_the_header_block(field):
    text = prereg(body="# a plan\n\n" + scaffold(drop=(field,)) + "\n" + FIELDS[field])     # only a blank line between
    p = judge(text)
    assert len(p) == 1 and "lacks `%s:`" % field in p[0] and "outside the block does not count" in p[0]
    text = prereg(scaffold(drop=(field,)), body="# a plan\n\n" + FIELDS[field])             # the frontmatter's `---`
    p = judge(text)
    assert len(p) == 1 and "lacks `%s:`" % field in p[0] and "outside the block does not count" in p[0]


@pytest.mark.parametrize("sep", ["# Heading as the only separator\n", "---\n", "***\n"])
def test_a_heading_or_a_rule_alone_ends_the_header_block(sep):
    """No blank line anywhere around the separator: the separator alone must end the block."""
    text = "# a plan\n" + scaffold(drop=("cap",)) + sep + FIELDS["cap"]
    p = judge(text)
    assert len(p) == 1 and "lacks `cap:`" in p[0] and "outside the block does not count" in p[0]


@pytest.mark.parametrize("field", GATE._FIELDS)
def test_an_empty_or_template_field_blocks(field):
    for val in ("", "<%s>" % field):
        p = judge(prereg(scaffold(**{field.replace("-", "_"): val})))
        assert any("`%s:` is empty or still the template placeholder" % field in x for x in p), (field, val, p)


def test_a_duplicated_field_blocks():
    p = judge(prereg(scaffold() + FIELDS["replaced-by"]))
    assert len(p) == 1 and "`replaced-by:` appears 2 times" in p[0]


@pytest.mark.parametrize("val", ["TBD", "tbd - later", "n/a", "N/A.", "none", "none yet", "?", "unknown",
                                 "(TBD)", "[TBD]", "(TBD -- see board)", "_TBD_", "T.B.D.", "to be decided",
                                 "To be determined", "later", "pending review", "N / A", "-", "—", "...", "todo",
                                 "not yet named"])
def test_replaced_by_placeholders_are_refused_bracketed_or_not(val):
    p = judge(prereg(scaffold(replaced_by=val)))
    assert len(p) == 1 and "`replaced-by: %s` is refused" % GATE._clean(val) in p[0], p


@pytest.mark.parametrize("val", ["the learned wh-frame organ (spiking, on the shared substrate)",
                                 "board task #271, the deixis organ", "Native speech self-monitor organ",
                                 "a laterality organ that learns which reading to keep"])
def test_replaced_by_that_names_a_mechanism_passes(val):
    assert judge(prereg(scaffold(replaced_by=val))) == []


@pytest.mark.parametrize("val", ["none, it is only a filter", "(none)", "[n/a]", "nothing", "TBD"])
def test_decides_that_names_nothing_blocks(val):
    p = judge(prereg(scaffold(decides=val)))
    assert len(p) == 1 and "names nothing" in p[0]


@pytest.mark.parametrize("val,ok", [
    ("CLAUDE.md non-negotiable 1", True), ("non-negotiable #1, a teaching scaffold", True), ("NN1", True),
    ("CLAUDE.md non-negotiable 12", True), ("owner directive 2026-10-16", True), ("owner 2026-10-22", True),
    ("RULING host-filter-ok-1022", True), ("host-filter-ok-1022", True), ("host-filter-ok-1022 (owner asked)", True),
    ("RULING answers-key", True),
    ("my own judgement", False), ("RULING unsigned-key", False), ("RULING needs-key", False),
    ("RULING idle-key", False), ("the owner", False), ("RULING relay-1022", False),
    ("non-negotiable 99", False), ("non-negotiable 13", False), ("NN 0", False), ("2099-01-01", False),
    ("owner directive 2026-10-23", False), ("2026-13-45", False), ("NOT allowed by non-negotiable 1", False),
    ("no directive; non-negotiable 1 forbids it", False), ("my triage of the residue", False),
    ("triage", True)])
def test_allowed_by_must_name_a_directive(val, ok):
    p = judge(prereg(scaffold(allowed_by=val)))
    assert (p == []) == ok, p
    if not ok:
        assert len(p) == 1 and "names no directive" in p[0]


def test_the_non_negotiable_bound_is_claude_md_s_count():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    section = text.split("## Non-negotiables", 1)[1].split("\n## ", 1)[0]
    assert len(re.findall(r"^\d+\. ", section, re.M)) == GATE.NN_COUNT


@pytest.mark.parametrize("cap", ["0 builds", "6 builds", "0", "6", "-1 builds", "three builds", "2.5 builds", "10",
                                 "3", "3 weeks", "3 W", "3x", "3 buildings", "3 builds, then a ruling"])
def test_a_cap_outside_1_to_5_or_not_an_integer_followed_by_builds_blocks(cap):
    p = judge(prereg(scaffold(cap=cap)))
    assert len(p) == 1 and "must be an integer 1-5 followed by `builds`" in p[0]


@pytest.mark.parametrize("cap", ["1 builds", "1 build", "5 builds", "3builds", "2 builds (the prereg and one amendment)",
                                 "2 builds."])
def test_a_cap_1_to_5_builds_is_accepted(cap):
    assert judge(prereg(scaffold(cap=cap))) == []


# ---------------------------------------------------------------- the cap, pure
def test_a_scaffold_prereg_whose_amendments_reach_its_cap_blocks():
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2))) == []    # builds 1-3: within the cap
    p = judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3)))
    assert len(p) == 1 and "carries 3 amendment(s)" in p[0] and "build 4 exceeds the cap" in p[0]
    assert "cap-ruling: <KEY>" in p[0] and "cap-raise: <prereg file name> +N builds" in p[0] and NEWB in p[0]


def test_amendments_in_other_forms_count_and_containers_do_not():
    other = ("\n**AMENDMENT 1 (2026-10-22, Claude)** a\n\n### Amendment 2 (2026-10-22)\nb\n"
             "\n## AMENDMENT 3 (2026-10-22)\nc\n")
    assert GATE.amendments(other) == ["1", "2", "3"]
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + other))
    log = "\n## AMENDMENT LOG\n\n(none at filing)\n" + amend(2)
    assert GATE.amendments(log) == ["1", "2"] and GATE.build_count(log) == 2
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + log)) == []


@pytest.mark.parametrize("label,extra", [
    ("## AMENDMENT #2", "\n## AMENDMENT #2 (2026-10-22)\nbuild\n"),
    ("## AMENDMENT [2]", "\n## AMENDMENT [2] (2026-10-22)\nbuild\n"),
    ("emoji prefix", "\n## \u26a0 AMENDMENT 2 (2026-10-22)\nbuild\n"),
    ("roman", "\n## AMENDMENT II (2026-10-22)\nbuild\n"),
    ("a number word", "\n## AMENDMENT two (2026-10-22)\nbuild\n"),
    ("a reused ID", "\n## AMENDMENT 1 (continued)\nnew rules\n"),
    ("a reused ID naming its build", "\n## AMENDMENT 1 (continued: build 3)\nnew rules\n"),
    ("a record subsection under a reused ID", "\n### AMENDMENT 1, record (appended after the run)\nnew rules\n"),
    ("AMENDMENT 1b", "\n## AMENDMENT 1b (2026-10-22)\nnew rules\n"),
    ("an erratum", "\n## ERRATUM 1 (2026-10-22): new rules for build 3\nnew rules\n"),
    ("an addendum", "\n## ADDENDUM 2 (2026-10-22)\nnew rules\n"),
    ("## Revision 2", "\n## Revision 2 (2026-10-22)\nnew rules\n"),
    ("## Build 3", "\n## Build 3 (filed before measurement)\nnew rules\n"),
    ("## Changes for build 3", "\n## Changes for build 3\nnew rules\n"),
    ("a plain line", "\nAmendment 2 (2026-10-22): new rules for build 3\n"),
    ("a glued bold line, no date", "\nsome prose line\n**Amendment 2** new rules for build 3\n"),
    ("a summary", "\n<details><summary>AMENDMENT 2 (2026-10-22)</summary>\n\nbuild\n</details>\n"),
    ("a dated list item", "\n- Amendment 2 (2026-10-22): new rules\n"),
])
def test_a_second_build_in_any_amendment_form_reaches_cap_2(label, extra):
    assert judge(prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(1))) == []
    p = judge(prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(1) + extra))
    assert len(p) == 1 and "build 3 exceeds the cap" in p[0], (label, p)


def test_back_to_back_amendment_headings_each_count():
    bare = "## AMENDMENT 1\n## AMENDMENT 2\n## AMENDMENT 3\n"
    assert GATE.build_count(bare) == 3
    wrapped = "## AMENDMENT 1 (2026-09-24) -- the gates below were WRONG before this\n## amendment and must not be used\n"
    assert GATE.build_count(wrapped) == 1                       # a W2-wrapped heading line is the same heading


def test_a_prose_line_that_wraps_onto_an_amendment_word_is_not_an_amendment():
    text = "# a plan\n\nThe earlier review (see\nAMENDMENT 9) found the bars too loose; nothing changes here.\n"
    assert GATE.build_count(text) == 0


def test_a_scaffold_prereg_with_an_unclosed_fence_blocks():
    p = judge(prereg(scaffold(), body="# a plan\n\n```text\nexample\n\n## AMENDMENT 2 (2026-10-22)\nbuild\n"))
    assert len(p) == 1 and "never closed" in p[0]
    assert judge("# packet\n\n" + scaffold() + "\n```\nnot closed\n", rel=PACKET_DRAFT, kind="packet") == []


@pytest.mark.parametrize("key", ["cap-ok-1022", "dated-cap-key", "answers-key"])
def test_a_cap_ruling_with_a_grant_from_claude_in_section_6_passes(key):
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3) + "\ncap-ruling: %s\n" % key)) == []


def test_a_grant_gives_exactly_its_builds():
    body = "# a plan\n" + amend(4) + "\ncap-ruling: cap-ok-1022\n"                      # 5 builds, cap 3 + 1
    p = judge(prereg(scaffold(cap="3 builds"), body=body))
    assert len(p) == 1 and "build 5 exceeds the cap" in p[0] and "plus 1 granted" in p[0]
    assert judge(prereg(scaffold(cap="3 builds"), body=body + "cap-ruling: dated-cap-key\n")) == []
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(4) + "\ncap-ruling: two-more-1022\n")) == []
    twice = "# a plan\n" + amend(4) + "\ncap-ruling: cap-ok-1022\ncap-ruling: cap-ok-1022\n"     # one ruling, once
    assert judge(prereg(scaffold(cap="3 builds"), body=twice))
    many = "# a plan\n" + amend(20) + "\ncap-ruling: two-more-1022\n"                    # one ruling, not 19 builds
    assert judge(prereg(scaffold(cap="2 builds"), body=many))


@pytest.mark.parametrize("key,why", [
    ("made-up-key", "not a Claude-signed RULING"), ("unsigned-key", "not a Claude-signed RULING"),
    ("idle-key", "not a Claude-signed RULING"), ("needs-key", "not a Claude-signed RULING"),
    ("relay-1022", "not a Claude-signed RULING"), ("other-key", "does not cite this prereg"),
    ("hold-1022", "grants no builds"), ("nogo-1022", "grants no builds")])
def test_a_cap_ruling_that_is_missing_unsigned_unrelated_or_grants_nothing_blocks(key, why):
    p = judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3) + "\ncap-ruling: %s\n" % key))
    assert len(p) == 1 and "`cap-ruling: %s`" % key in p[0] and why in p[0], p


def test_every_cited_cap_ruling_must_carry_a_grant_once_grants_are_needed():
    body = "# a plan\n" + amend(3) + "\ncap-ruling: cap-ok-1022\ncap-ruling: hold-1022\n"
    p = judge(prereg(scaffold(cap="3 builds"), body=body))
    assert len(p) == 1 and "hold-1022" in p[0] and "grants no builds" in p[0]
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(1) + "\ncap-ruling: hold-1022\n")) == []


def test_the_real_hold_ruling_on_main_unlocks_nothing():
    """wrc-amendG-final-1018 HELD the word-role flip; under the first build of this gate it validated as a cap-ruling."""
    rel = "research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md"
    text = (ROOT / rel).read_text(encoding="utf-8")
    rb = SF.read_runbook(str(ROOT))
    if "wrc-amendG-final-1018" not in SF.ruling_entries(rb):
        pytest.skip("the ruling is not in this checkout's main runbook")
    hdr = scaffold(cap="1 builds") + "cap-ruling: wrc-amendG-final-1018\n"
    p = GATE.judge(rel, text.replace("---\n", "---\n" + hdr, 1), "prereg", lambda: rb, today=TODAY)
    assert any("wrc-amendG-final-1018" in x and "grants no builds" in x for x in p), p


def test_a_packet_carries_the_header_but_counts_no_builds():
    assert judge("# packet\n\n" + scaffold(cap="1 builds") + amend(4), rel=PACKET_DRAFT, kind="packet") == []
    p = judge("# packet\n", rel=PACKET_DRAFT, kind="packet")
    assert len(p) == 1 and "(packet)" in p[0] and "no `host-decision:` header" in p[0]
    p = judge("# spec\n", rel=SPEC, kind="plan")
    assert len(p) == 1 and "(plan)" in p[0]


def test_the_real_moat_prereg_has_no_header_so_its_next_amendment_must_declare_itself():
    text = (ROOT / MOAT).read_text(encoding="utf-8")
    if "host-decision:" in text:
        pytest.skip("the moat prereg has declared itself since this test was written")
    assert GATE.amendments(text)[:7] == ["1", "2", "3", "4", "5", "6", "7"] and GATE.build_count(text) == 7
    assert any("no `host-decision:` header" in p for p in judge(text, rel=MOAT))
    p = judge(text.replace("seeds: [42,", scaffold(cap="5 builds") + "seeds: [42,", 1), rel=MOAT)
    assert len(p) == 1 and "build 8 exceeds the cap" in p[0]
    p = judge(text.replace("seeds: [42,", "host-decision: none it only registers measurement bars\nseeds: [42,", 1),
              rel=MOAT)
    assert len(p) == 1 and "known host scaffold" in p[0]


def test_none_is_refused_on_every_known_scaffold():
    assert GATE.KNOWN_SCAFFOLDS == {MOAT, "docs/drafts/moat-allowlist/SPEC.md"}
    for rel in GATE.KNOWN_SCAFFOLDS:
        p = judge("# x\n\n" + NONE, rel=rel, kind=GATE.kind_of(rel))
        assert len(p) == 1 and "known host scaffold" in p[0], rel
        assert judge("# x\n\n" + scaffold(), rel=rel, kind=GATE.kind_of(rel)) == []


# ---------------------------------------------------------------- history, pure (the parent versions)
def test_a_scaffold_turned_none_blocks():
    p = judge(prereg(NONE, body="# a plan\n" + amend(3)), parents=[prereg(scaffold(), body="# a plan\n" + amend(2))])
    assert len(p) == 1 and "stays a scaffold" in p[0]
    assert judge(prereg(NONE), parents=[prereg(body="# an old plan\n")]) == []            # undeclared -> none is fine


def test_a_cap_never_goes_up():
    p = judge(prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(3)),
              parents=[prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2))])
    assert len(p) == 1 and "goes up from 3 to 5" in p[0]
    assert judge(prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(1)),
                 parents=[prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(1))]) == []


@pytest.mark.parametrize("label,body", [
    ("folded into one heading", "\n## AMENDMENT 1 (2026-10-22; builds 2-4, consolidated)\nall rules\n"),
    ("moved under a log as bullets", "\n## AMENDMENT LOG\n\n- build 2 rules\n- build 3 rules\n- build 4 rules\n"),
    ("dropped", amend(1)),
])
def test_amendments_never_shrink(label, body):
    p = judge(prereg(scaffold(cap="5 builds"), body="# a plan\n" + body),
              parents=[prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(3))])
    assert any("never removed, folded" in x or "are gone" in x or "is gone" in x for x in p), (label, p)


def test_an_amendment_relabelled_while_another_is_added_blocks():
    p = judge(prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(1) + amend(1, start=3)),
              parents=[prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(2))])
    assert len(p) == 1 and "amendment entry 2" in p[0] and "is gone" in p[0]
    assert judge(prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(3)),
                 parents=[prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(2))]) == []


def test_an_old_undeclared_prereg_cannot_fold_its_amendments_while_declaring():
    p = judge(prereg(scaffold(cap="5 builds"), body="# a plan\n\n## AMENDMENT 1 (2026-10-22; all of 1-7)\nx\n"),
              parents=[prereg(body="# a plan\n" + amend(7))])
    assert any("never removed, folded" in x for x in p) and any("are gone" in x for x in p)


def test_a_prose_rewrap_does_not_trip_the_history_check():
    before = prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(1) + "\nAmendment 7 (2026-10-01) of another "
                                                                                "plan set this bar.\n")
    after = prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(1) + "\nThe bar comes from Amendment 7\n"
                                                                              "(2026-10-01) of another plan.\n")
    assert judge(after, parents=[before]) == []


# ---------------------------------------------------------------- the group: one scaffold across files, pure
P2 = "research/findings/2026-10-23-newplan-build4-PREREGISTRATION.md"


def test_a_successor_prereg_for_the_same_scaffold_joins_its_group():
    peers = {NEW: ("prereg", prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2)))}
    p = judge(prereg(scaffold(cap="5 builds"), body="# build 4\n"), rel=P2, peers=peers)
    assert len(p) == 1 and "shared" in p[0] and NEW in p[0] and "build 4 exceeds the cap" in p[0]
    reworded = scaffold(cap="5 builds", decides="which replies the mouth keeps, by host rules")
    p = judge(prereg(reworded, body="# build 4\n"), rel=P2, peers=peers)                  # replaced-by still links
    assert len(p) == 1 and "shared" in p[0]


def test_amendment_files_for_the_same_scaffold_join_its_group():
    peers = {NEW: ("prereg", prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(1)))}
    files = ["research/findings/2026-10-2%d-newplan-PREREG-AMENDMENT-%d.md" % (k + 2, k) for k in (1, 2)]
    peers[files[0]] = ("prereg", prereg(scaffold(cap="1 builds")))
    p = judge(prereg(scaffold(cap="1 builds")), rel=files[1], peers=peers)
    assert len(p) == 1 and "shared" in p[0]


def test_a_grant_for_any_file_of_the_group_counts_once():
    peers = {NEW: ("prereg", prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2) + "\ncap-ruling: cap-ok-1022\n"))}
    assert judge(prereg(scaffold(cap="5 builds"), body="# build 4\n"), rel=P2, peers=peers) == []


def test_a_packet_of_an_over_cap_group_blocks_and_an_unrelated_scaffold_does_not():
    peers = {NEW: ("prereg", prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2)))}
    p = judge("# packet\n\n" + scaffold(cap="2 builds"), rel=PACKET_DRAFT, kind="packet", peers=peers)
    assert len(p) == 1 and "shared" in p[0]
    other = scaffold(cap="2 builds", decides="which heard words are stored as names",
                     replaced_by="the learned hearing organ (word roles on statements)")
    assert judge("# packet\n\n" + other, rel=PACKET_DRAFT, kind="packet", peers=peers) == []


# ---------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,want", [
    (NEW, "prereg"), (PLANS_PREREG, "prereg"), ("research/findings/2026-10-22-x-prereg-amendment.md", "prereg"),
    (PACKET_DRAFT, "packet"), ("docs/drafts/PACKET.md", "packet"), ("docs/drafts/a/b/PACKET.md", "packet"),
    ("docs/drafts/one-brain-m2/PACKET-m2-fixes.md", "packet"), ("docs/drafts/x/moat-PACKET.md", "packet"),
    (PACKET_COORD, "packet"), ("research/coordination/-PACKET.md", "packet"),
    ("research/coordination/moat-PACKET-v2.md", "packet"), ("research/coordination/PACKET-moat.md", "packet"),
    (SPEC, "plan"), ("docs/drafts/some-lane/PLAN.md", "plan"), ("docs/drafts/some-lane/HANDOFF.md", "plan"),
    ("docs/drafts/some-lane/DESIGN.md", "plan"), ("docs/drafts/x/toy-DESIGN-NOT-A-PREREG.md", "plan"),
    ("docs/drafts/long-delay-credit/AMENDMENT-A0.2.md", "plan"),
    ("docs/drafts/some-lane/WORKLOG.md", None), ("docs/drafts/some-lane/SPECIAL.md", None),
    ("research/findings/2026-10-22-x-RESULT.md", None), ("research/findings/raw/x/2026-10-22-PREREG.md", None),
    ("research/coordination/LOCAL_LLM_RUNBOOK.md", None), ("research/coordination/x/y-PACKET.md", None),
    ("docs/PACKET.md", None), ("docs/plans/2026-10-22-x-PLAN.md", None)])
def test_scope_is_the_smk_prereg_set_the_packets_and_the_lane_plans(rel, want):
    assert GATE.kind_of(rel) == want


def test_the_prereg_scope_is_read_from_smk_not_copied(monkeypatch):
    assert GATE.kind_of(PLANS_PREREG) == "prereg"
    monkeypatch.setattr(SF, "PREREG_RE", re.compile(r"^nowhere$"))            # SMK's file set moves: so does this one
    assert GATE.kind_of(PLANS_PREREG) is None


def test_every_in_scope_plan_on_this_tree_is_matched_by_the_git_pathspecs():
    """kind_of() and the pathspecs that feed `git diff --cached` must agree, or a plan is never judged."""
    r = subprocess.run(["git", "ls-files", "-z", "--"] + list(GATE._specs()), cwd=str(ROOT), capture_output=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    listed = set(r.stdout.decode().split("\0")) - {""}
    every = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT), capture_output=True).stdout.decode().split("\0")
    in_scope = {f for f in every if f and GATE.kind_of(f)}
    assert in_scope and in_scope <= listed, sorted(in_scope - listed)[:5]


# ---------------------------------------------------------------- commit time: real git, scratch repos
def _git(td, env, *a, check=True):
    r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *a], cwd=td, env=env,
                       capture_output=True, text=True, timeout=60)
    if check:
        assert r.returncode == 0, r.stderr
    return r


@pytest.fixture()
def repo(tmp_path):
    env = PRA._stripped_env()
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")

    class R:
        td = str(tmp_path)

        def w(self, rel, text):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

        def rm(self, rel):
            (tmp_path / rel).unlink()

        def git(self, *a, date=None, check=True):
            e = dict(env)
            if date:
                e["GIT_COMMITTER_DATE"] = e["GIT_AUTHOR_DATE"] = date + "T12:00:00+00:00"
            return _git(self.td, e, *a, check=check).stdout.strip()

        def commit(self, msg, date=None):
            self.git("add", "-A")
            self.git("commit", "-q", "-m", msg, date=date)

        def staged_check(self, kind="commit", add=True):
            if add:
                self.git("add", "-A")
            return GATE.check([], root=self.td, invocation=kind)

    r = R()
    r.git("init", "-q", "-b", "main")
    r.w(RUNBOOK, RULINGS)
    r.w(OLD, "# an old plan, filed before the gate, no header\n")
    r.w("docs/drafts/old-lane/PACKET.md", "# an old packet, no header\n")
    r.commit("base", date="2026-10-01")
    return r


def test_gate_blocks_a_new_prereg_without_the_header(repo):
    repo.w(NEW, prereg())
    p = repo.staged_check()
    assert len(p) == 1 and NEW in p[0] and "no `host-decision:` header" in p[0]
    assert "new: first added on or after 2026-10-21" in p[0]
    repo.w(NEW, prereg(NONE))
    assert repo.staged_check() == []


def test_gate_blocks_an_old_prereg_edited_without_the_header(repo):
    repo.w(OLD, "# an old plan, filed before the gate, no header\n" + amend(1))
    p = repo.staged_check()
    assert len(p) == 1 and OLD in p[0] and "no `host-decision:` header" in p[0]
    assert "touch it, declare it" in p[0] and "modifies" in p[0]
    repo.w(OLD, "# an old plan\n\n" + NONE + amend(1))
    assert repo.staged_check() == []


def test_new_or_old_follows_smks_date_rule_the_later_of_git_date_and_file_name_date(repo):
    """Both must declare when touched; the date words the message (new since the cutoff vs touch it, declare it)."""
    late = "research/findings/2026-10-01-backdated-PREREGISTRATION.md"        # an old file-name date, added after
    lane = "docs/drafts/new-lane/PACKET.md"                                    # no file-name date, added after
    early = "research/findings/2026-10-25-forward-PREREGISTRATION.md"         # a later file-name date, git date before
    repo.w(late, prereg())
    repo.w(lane, "# a packet\n")
    repo.commit("added after the cutoff", date="2026-10-22")
    repo.w(early, prereg())
    repo.commit("committer date before the cutoff", date="2026-10-02")
    for rel in (late, lane, early):
        repo.w(rel, prereg(body="# edited\n"))
        p = repo.staged_check()
        assert len(p) == 1 and "new: first added on or after 2026-10-21" in p[0], (rel, p)
        repo.git("reset", "-q")
        repo.git("checkout", "--", rel)
    repo.w(OLD, prereg(body="# edited\n"))
    p = repo.staged_check()
    assert len(p) == 1 and "touch it, declare it" in p[0] and "new:" not in p[0]


def test_an_old_prereg_untouched_passes_and_a_clean_tree_passes(repo):
    assert repo.staged_check() == []
    repo.w("research/findings/2026-10-22-unrelated-RESULT.md", "# a result, no header\n")
    assert repo.staged_check() == []                                           # an unrelated commit leaves OLD alone


def test_gate_judges_a_renamed_old_prereg(repo):
    repo.git("mv", OLD, "research/findings/2026-10-02-oldplan-moved-PREREGISTRATION.md")
    p = repo.staged_check(add=False)
    assert len(p) == 1 and "renames" in p[0] and "touch it, declare it" in p[0]


def test_an_undeclared_or_scaffold_plan_is_never_deleted_and_a_none_plan_may_be(repo):
    repo.git("rm", "-q", OLD)
    p = repo.staged_check(add=False)
    assert len(p) == 1 and "deletes %s" % OLD in p[0] and "never declared" in p[0]
    repo.git("reset", "-q", "--hard")
    repo.w(OLD, "# an old plan\n\n" + NONE)
    repo.commit("declare first", date="2026-10-22")
    repo.git("rm", "-q", OLD)
    assert repo.staged_check(add=False) == []
    repo.git("reset", "-q", "--hard")
    repo.w(NEW, prereg(scaffold()))
    repo.commit("a scaffold", date="2026-10-22")
    repo.git("rm", "-q", NEW)
    p = repo.staged_check(add=False)
    assert len(p) == 1 and "declares `host-decision: scaffold`" in p[0]


@pytest.mark.parametrize("dest,seen_as", [
    ("research/findings/2026-10-20-oldplan-PRE-REGISTRATION.md", "deletes"),       # outside every pathspec: a D
    ("docs/plans/oldplan-notes.md", "deletes"),
    ("docs/drafts/old-lane/BARS.md", "moves"),                                    # inside a pathspec: an R out of scope
    ("research/findings/raw/x/oldplan-prereg-copy.md", "moves")])
@pytest.mark.parametrize("edit", [False, True])
def test_moving_a_plan_out_of_scope_blocks(repo, dest, seen_as, edit):
    (Path(repo.td) / dest).parent.mkdir(parents=True, exist_ok=True)
    repo.git("mv", OLD, dest)
    if edit:                                     # a rewrite past rename detection is a delete plus an add: still caught
        repo.w(dest, "# an old plan, filed before the gate, no header\n" + amend(8))
    p = repo.staged_check()
    assert len(p) == 1 and "never declared" in p[0], p
    if not edit:
        want = ("moves %s out of the plan scope (to %s)" % (OLD, dest)) if seen_as == "moves" else ("deletes %s" % OLD)
        assert want in p[0], p


@pytest.mark.parametrize("rel", [PACKET_DRAFT, PACKET_COORD, "docs/drafts/a/b/PACKET.md",
                                 "docs/drafts/one-brain-m2/PACKET-m2-fixes.md", SPEC, "docs/drafts/x/HANDOFF.md"])
def test_a_plan_under_each_plan_path_is_covered(repo, rel):
    repo.w(rel, "# a packet\n")
    p = repo.staged_check()
    assert len(p) == 1 and rel in p[0] and "(%s" % GATE.kind_of(rel) in p[0]
    repo.w(rel, "# a packet\n\n" + scaffold(cap="1 builds"))
    assert repo.staged_check() == []


def test_an_old_packet_edited_must_declare(repo):
    repo.w("docs/drafts/old-lane/PACKET.md", "# an old packet, no header\n\nstep 2 added\n")
    p = repo.staged_check()
    assert len(p) == 1 and "touch it, declare it" in p[0]


@pytest.mark.parametrize("rel", ["docs/drafts/some-lane/WORKLOG.md", "docs/drafts/some-lane/RESULTS.md",
                                 "research/findings/2026-10-22-x-RESULT.md", "research/coordination/NOTES.md",
                                 "research/findings/raw/x/2026-10-22-PREREG.md", "docs/plans/2026-10-22-x-PLAN.md"])
def test_a_non_plan_markdown_file_is_untouched(repo, rel):
    repo.w(rel, "# no header here\n")
    assert repo.staged_check() == []


def test_gate_blocks_a_scaffold_prereg_at_its_cap_and_passes_with_a_cap_ruling_on_main(repo):
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(1)))
    repo.commit("the plan and its first amendment", date="2026-10-22")
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2)))
    p = repo.staged_check()
    assert len(p) == 1 and "build 3 exceeds the cap" in p[0]
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2) + "\ncap-ruling: cap-ok-1022\n"))
    assert repo.staged_check() == []
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2) + "\ncap-ruling: unsigned-key\n"))
    assert any("not a Claude-signed RULING" in x for x in repo.staged_check())


def test_a_cap_ruling_written_in_the_same_commit_does_not_count_only_main_does(repo):
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2) + "\ncap-ruling: self-made\n"))
    repo.w(RUNBOOK, RULINGS.replace("## NEEDS CLAUDE", "- self-made: cap-raise: %s +1 builds (Claude 2026-10-22)\n\n"
                                    "## NEEDS CLAUDE" % NEWB))
    p = repo.staged_check()
    assert len(p) == 1 and "`cap-ruling: self-made`" in p[0] and "not a Claude-signed RULING" in p[0]
    repo.git("reset", "-q")
    repo.git("add", RUNBOOK)
    repo.git("commit", "-q", "-m", "Claude's ruling lands on main", date="2026-10-22")
    assert repo.staged_check() == []


def _at_cap(repo):
    repo.w(NEW, prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2)))
    repo.commit("a scaffold prereg at its cap", date="2026-10-22")


def test_history_r1_the_commit_that_adds_the_build_past_the_cap_cannot_switch_to_none(repo):
    _at_cap(repo)
    repo.w(NEW, prereg(NONE, body="# a plan\n" + amend(3)))
    p = repo.staged_check()
    assert len(p) == 1 and "stays a scaffold" in p[0]


def test_history_r2_the_commit_that_adds_the_build_past_the_cap_cannot_raise_it(repo):
    _at_cap(repo)
    repo.w(NEW, prereg(scaffold(cap="5 builds"), body="# a plan\n" + amend(3)))
    p = repo.staged_check()
    assert len(p) == 1 and "goes up from 3 to 5" in p[0]


@pytest.mark.parametrize("body", ["\n## AMENDMENT 1 (2026-10-22; builds 2-4, consolidated)\nall rules\n",
                                  "\n## AMENDMENT LOG\n\n- build 2 rules\n- build 3 rules\n- build 4 rules\n"])
def test_history_r9_amendments_cannot_be_folded_or_moved_under_a_log(repo, body):
    _at_cap(repo)
    repo.w(NEW, prereg(scaffold(cap="3 builds"), body="# a plan\n" + body))
    assert repo.staged_check()


def test_group_r3_a_successor_prereg_does_not_reset_the_cap(repo):
    _at_cap(repo)
    repo.w(P2, prereg(scaffold(cap="5 builds"), body="# build 4 of the same host filter\n"))
    p = repo.staged_check()
    assert len(p) == 1 and P2 in p[0] and "shared" in p[0] and "build 4 exceeds the cap" in p[0]


def test_group_r4_amendment_files_do_not_reset_the_cap(repo):
    _at_cap(repo)
    for n in range(4, 6):
        repo.w("research/findings/2026-10-2%d-newplan-PREREG-AMENDMENT-%d.md" % (n, n), prereg(scaffold(cap="1 builds")))
    p = repo.staged_check()
    assert len(p) == 2 and all("shared" in x for x in p)


def test_a_real_two_parent_merge_judges_only_what_differs_from_every_parent(repo):
    """MERGE_HEAD present (a conflicted merge, or --no-commit): OLD changed on the side branch only, so it equals the
    second parent and is not judged; an edit made in the merge itself is."""
    repo.git("checkout", "-q", "-b", "side")
    repo.w(OLD, "# an old plan\n\n" + NONE + "\nside edit\n")
    repo.commit("side declares and edits", date="2026-10-22")
    repo.git("checkout", "-q", "main")
    repo.w("research/findings/2026-10-22-main-RESULT.md", "# main moves on\n")
    repo.commit("main", date="2026-10-22")
    repo.git("merge", "-q", "--no-ff", "--no-commit", "side")
    assert repo.staged_check() == []
    repo.w("docs/drafts/old-lane/PACKET.md", "# an old packet, no header\n\nedited in the merge\n")
    p = repo.staged_check()
    assert len(p) == 1 and "docs/drafts/old-lane/PACKET.md" in p[0]


def test_a_clean_auto_merge_is_judged_for_the_cap_only_and_a_git_failure_fails_closed(repo, monkeypatch):
    repo.w(NEW, prereg())
    assert repo.staged_check("commit") and repo.staged_check("merge") == []              # no header: cap only
    repo.w(NEW, prereg(scaffold(cap="2 builds"), body="# a plan\n" + amend(2)))
    p = repo.staged_check("merge")
    assert len(p) == 1 and "build 3 exceeds the cap" in p[0]

    def boom(*a, **k):
        raise PRA._GitReadError("simulated")
    monkeypatch.setattr(PRA, "_parents", boom)
    p = GATE.check([], root=repo.td, invocation="commit")
    assert len(p) == 1 and "failing CLOSED" in p[0]


# ---------------------------------------------------------------- real hooks: a clean `git merge` (finding HD-10)
_HOOK = """#!/bin/sh
exec "%s" - <<'PY'
import os, sys
sys.path.insert(0, %r)
from tools.gates import host_decision_declared as m
problems = m.check([], root=os.getcwd())
print("\\n".join(problems))
sys.exit(1 if problems else 0)
PY
"""


def _hooked_repo(tmp_path):
    root = os.path.realpath(str(tmp_path / "hooked"))
    os.makedirs(root)
    env = PRA._stripped_env()
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")

    def git(*a, check=True):
        r = subprocess.run(["git", "-c", "commit.gpgsign=false", *a], cwd=root, env=env, capture_output=True, text=True,
                           timeout=60)
        if check:
            assert r.returncode == 0, r.stdout + r.stderr
        return r

    def w(rel, text):
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    git("init", "-q", "-b", "main")
    hooks = os.path.join(root, "tools", "githooks")
    os.makedirs(hooks)
    with open(os.path.join(hooks, "pre-commit"), "w") as fh:
        fh.write(_HOOK % (sys.executable, str(ROOT)))
    shutil.copy(str(ROOT / "tools" / "githooks" / "pre-merge-commit"), os.path.join(hooks, "pre-merge-commit"))
    for h in ("pre-commit", "pre-merge-commit"):
        os.chmod(os.path.join(hooks, h), 0o755)
    with open(os.path.join(root, ".git", "info", "exclude"), "a") as fh:
        fh.write("/tools/\n")
    git("config", "core.hooksPath", hooks)
    w(RUNBOOK, RULINGS)
    base = prereg(scaffold(cap="3 builds"), body="# a plan\n\n## 1. bars\n\nbars\n\n## 2. more\n\nmore\n" + amend(1))
    w(NEW, base)
    git("add", "-A")
    git("commit", "-q", "-m", "base: a scaffold prereg, cap 3, one amendment")
    return git, w, base


def test_real_hooks_a_clean_merge_that_takes_a_scaffold_past_its_cap_is_refused(tmp_path):
    git, w, base = _hooked_repo(tmp_path)
    git("checkout", "-q", "-b", "side")
    w(NEW, base.replace("## 2. more\n\nmore\n", "## 2. more\n\nmore\n\n## AMENDMENT 2 (2026-10-22, Claude)\nside\n"))
    git("commit", "-q", "-am", "side: amendment 2 mid-file (3 builds, within cap 3)")
    git("checkout", "-q", "main")
    w(NEW, base + "\n## AMENDMENT 3 (2026-10-22, Claude)\nmain build\n")
    git("commit", "-q", "-am", "main: amendment 3 appended (3 builds, within cap 3)")
    r = git("merge", "--no-ff", "--no-edit", "side", check=False)
    assert r.returncode != 0 and "build 4 exceeds the cap" in r.stdout + r.stderr, r.stdout + r.stderr


def test_real_hooks_a_clean_merge_within_the_cap_passes(tmp_path):
    git, w, base = _hooked_repo(tmp_path)
    git("checkout", "-q", "-b", "side")
    w("research/findings/2026-10-22-side-RESULT.md", "# side\n")
    git("add", "-A")
    git("commit", "-q", "-m", "side")
    git("checkout", "-q", "main")
    w(NEW, base + "\n## AMENDMENT 2 (2026-10-22, Claude)\nmain build\n")
    git("commit", "-q", "-am", "main: amendment 2")
    r = git("merge", "--no-ff", "--no-edit", "side", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- the registry's trust signal
def test_the_registry_discovers_it_as_a_blocking_gate_with_a_clean_selftest():
    from tools.gates import discover
    found = {name: mod for name, mod, err in discover() if not err}
    assert found["host_decision_declared"].CLASS_ID == "HD" and found["host_decision_declared"].BLOCKING is True
    assert GATE.selftest() == []
    matrix = (ROOT / "docs" / "FAILURE_GATE_MATRIX.md").read_text(encoding="utf-8")
    assert "| HD |" in matrix and "`gates/host_decision_declared`" in matrix


def test_the_docstring_and_matrix_say_what_review_must_check_for_a_dishonest_none():
    doc = GATE.__doc__
    assert "WHAT IT CANNOT CATCH, AND WHAT REVIEW MUST CHECK" in doc and "dishonest `none`" in doc
    row = next(ln for ln in (ROOT / "docs" / "FAILURE_GATE_MATRIX.md").read_text(encoding="utf-8").splitlines()
               if ln.startswith("| HD |"))
    assert "review" in row and "dishonest" in row


def test_the_tree_test_is_not_a_time_bomb():
    """Nothing staged -> nothing judged, on the real tree: untouched plans without the header are grandfathered."""
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT), capture_output=True, text=True)
    if staged.returncode != 0:
        pytest.skip("not a git checkout")
    if staged.stdout.strip():
        pytest.skip("something is staged in this checkout; the gate would rightly judge it")
    assert GATE.check([], root=str(ROOT), invocation="commit") == []


_GATE_SRC = str(ROOT / "tools" / "gates" / "host_decision_declared.py")
MUTANTS = {
    "no-header check removed": ('    if h["state"] == "missing":\n        return ["%s has no',
                                '    if False:\n        return ["%s has no'),
    "two header lines read as one": ("    if len(head) != 1:\n        return h", "    if not head:\n        return h"),
    "fenced lines read": ("        if fence:\n            continue\n        if first is None",
                          "        if first is None"),
    "HTML comments read": ("            comment = True\n", "            comment = False\n"),
    "header anywhere": ('    if h["line"] >= min(h["first"], HEAD_MAX_LINE):', "    if False:"),
    "blank line no block boundary": (r'_BOUNDARY_RE = re.compile(r"^\s*(?:$|', r'_BOUNDARY_RE = re.compile(r"^\s*(?:(?!)|'),
    "allowed-by unchecked": ('if "allowed-by" in vals and not _allowed_by_ok', 'if False and not _allowed_by_ok'),
    "allowed-by any date": ('if "2026-01-01" <= d <= today:', "if True:"),
    "runbook read from the index": ('cache["rb"] = sf.read_runbook(root)',
                                    'cache["rb"] = pra._git(["cat-file", "blob", ":" + sf.RUNBOOK_REL], root, env)'
                                    '.stdout.decode()'),
    "reason length unchecked": ("len(reason) < MIN_REASON", "len(reason) < 0"),
    "fields not required": ("    missing = [k for k in _FIELDS if k not in fields]", "    missing = []"),
    "replaced-by placeholder allowed": ('if "replaced-by" in vals and _names_nothing', 'if False and _names_nothing'),
    "brackets hide a placeholder": ("    s = _norm(v)\n", "    s = v.strip().lower()\n"),
    "cap range unchecked": ("cm and CAP_MIN <= int(cm.group(1)) <= CAP_MAX", "cm"),
    "cap rule off": ("    if builds <= cap:\n        return []", "    if True:\n        return []"),
    "any cap-ruling accepted": ("    if not n:\n        return 0, (", "    if False:\n        return 0, ("),
    "Hermes may sign": ("(_SIGNER_RE.match(segs[0]) or _SIGNER_RE.match(segs[-1]))", "True"),
    "packets out of scope": ('    if _PACKET_RE.match(rel):\n        return "packet"', '    if False:\n        return "packet"'),
    "old edits grandfathered": ('            prev = [blob(',
                                '            if st != "A":\n                continue\n            prev = [blob('),
    "fail open on git errors": ('        return ["CLASS HD could not read', '        return []\n        return ["CLASS HD could not read'),
    "scaffold may turn none": ('        if ph["state"] == "scaffold" and h["state"] == "none":', "        if False:"),
    "cap may rise": ('if ph["cap"] is not None and h["cap"] is not None and h["cap"] > ph["cap"]:', "if False:"),
    "amendments may shrink": ("            if after < before:", "            if False:"),
    "amendments may be relabelled": ("            if gone:", "            if False:"),
    "no group": ("            if o not in group and ks & ident[r]:", "            if False:"),
    "deletion allowed": ('    what = ("deletes %s" % rel)', '    return []\n    what = ("deletes %s" % rel)'),
    "move out of scope allowed": ('                if st == "R" and kind_of(old):', "                if False:"),
    "known scaffolds may say none": ("        if rel in KNOWN_SCAFFOLDS:", "        if False:"),
    "unclosed fence ignored": ('    if kind == "prereg" and h["open_fence"] is not None:', "    if False:"),
    "clean merge unchecked": ("            probs = []\n            for p in sorted(ch):",
                              "            return []\n            for p in sorted(ch):"),
    "literal headings not counted": ("                literal += 1\n", "                pass\n"),
    "build N headings ignored": ("                top = max([top] + [int(n) for n in re.findall(", "                top = 0 * max([top] + [int(n) for n in re.findall("),
    "line-form IDs ignored": ("                if k and k not in ids:\n", "                if False:\n"),
}


@pytest.mark.parametrize("label", sorted(MUTANTS))
def test_selftest_kills_mutants(label):
    """The registry trusts a gate only through its selftest: each mutant must make selftest() FAIL."""
    src = Path(_GATE_SRC).read_text(encoding="utf-8")
    old, new = MUTANTS[label]
    assert src.count(old) == 1, "mutant %r no longer applies: update MUTANTS" % label
    mod = types.ModuleType("hd_mutant")
    mod.__file__ = _GATE_SRC
    exec(compile(src.replace(old, new), _GATE_SRC, "exec"), mod.__dict__)
    assert mod.selftest(), "selftest() still PASSES with the mutant %r -- the registry would trust a broken gate" % label


def test_key_line_re_reads_smk_lines_as_before():
    """smoke_first.key_line_re() is shared with this gate; SMK's own line grammar is unchanged by the refactor."""
    rx = SF.key_line_re(SF._KEYS)
    for ln in ("flip: yes", "- **flip:** no it is a battery", "> smoke-record: `a.json`", "  * smoke-hurts-ruling: K"):
        assert rx.match(ln) and SF._LINE_RE.match(ln).groupdict() == rx.match(ln).groupdict()
    assert not rx.match("flipper: yes") and not rx.match("| flip: yes |")
