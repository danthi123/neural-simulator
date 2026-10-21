"""Host decision declared (CLASS HD, tools/gates/host_decision_declared.py): every prereg or packet a commit touches says
up front whether it adds host code that decides content (CLAUDE.md non-negotiable 1, brain-based only), and a scaffold
prereg cannot take another build past its cap without a Claude-signed ruling on main. Owner 2026-10-21, after the moat
allowlist's eight builds of English-reading host rules (2026-10-20 prereg, Amendments 1-7) repeated the 2026-10-16
English-grammar-template plan. Pure judge() cases plus real `git diff --cached` in scratch repos; no brain, no GPU."""
import os
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
OLD = "research/findings/2026-10-01-oldplan-PREREGISTRATION.md"
PLANS_PREREG = "docs/plans/2026-10-22-plan-PREREG.md"
PACKET_DRAFT = "docs/drafts/some-lane/PACKET.md"
PACKET_COORD = "research/coordination/some-lane-PACKET.md"
RULINGS = ("# runbook\n\n## 5. Idle\n\n- idle-key: a section-5 bullet about 2026-10-22-newplan-PREREGISTRATION.md "
           "(Claude)\n\n## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`)\n\n"
           "- cap-ok-1022: one more build of 2026-10-22-newplan-PREREGISTRATION.md, then a neural method\n"
           "  (Claude 2026-10-22)\n"
           "- 2026-10-22 RULING dated-cap-key (Claude; owner-approved): 2026-10-22-newplan-PREREGISTRATION.md may "
           "take one more build\n"
           "- unsigned-key: allows 2026-10-22-newplan-PREREGISTRATION.md (Hermes 2026-10-22)\n"
           "- other-key: allows another-plan-PREREGISTRATION.md (Claude 2026-10-22)\n"
           "- host-filter-ok-1022: a host filter may run as a teaching scaffold (Claude 2026-10-22)\n\n"
           "## NEEDS CLAUDE\n\n- needs-key: 2026-10-22-newplan-PREREGISTRATION.md (Claude)\n")
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


def judge(text, rel=NEW, kind="prereg", runbook=RULINGS):
    return GATE.judge(rel, text, kind, lambda: runbook)


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


def test_none_with_the_template_placeholder_as_its_reason_blocks():
    p = judge(prereg("host-decision: none <why nothing it builds decides content on the host, at least 20 chars>\n"))
    assert len(p) == 1 and "at least 20 characters" in p[0]


def test_none_with_a_reason_passes_in_any_line_style():
    assert judge(prereg(NONE)) == []
    assert judge(prereg(body="# a plan\n\n- **host-decision:** `none` -- it measures a spiking organ only\n")) == []
    assert judge(prereg(body="# a plan\n\n> host-decision: none: it measures a spiking organ only, no host rule\n")) == []


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
    text = prereg(scaffold(drop=(field,)), body="# a plan\n\n" + FIELDS[field])
    p = judge(text)
    assert len(p) == 1 and "lacks `%s:`" % field in p[0] and "outside the block does not count" in p[0]


@pytest.mark.parametrize("field", GATE._FIELDS)
def test_an_empty_or_template_field_blocks(field):
    for val in ("", "<%s>" % field):
        p = judge(prereg(scaffold(**{field.replace("-", "_"): val})))
        assert any("`%s:` is empty or still the template placeholder" % field in x for x in p), (field, val, p)


def test_a_duplicated_field_blocks():
    p = judge(prereg(scaffold() + FIELDS["replaced-by"]))
    assert len(p) == 1 and "`replaced-by:` appears 2 times" in p[0]


@pytest.mark.parametrize("val", ["TBD", "tbd - later", "n/a", "N/A.", "none", "none yet", "?", "unknown"])
def test_replaced_by_placeholders_are_refused(val):
    p = judge(prereg(scaffold(replaced_by=val)))
    assert len(p) == 1 and "`replaced-by: %s` is refused" % val in p[0]


def test_decides_that_names_nothing_blocks():
    p = judge(prereg(scaffold(decides="none, it is only a filter")))
    assert len(p) == 1 and "names nothing" in p[0]


@pytest.mark.parametrize("val,ok", [
    ("CLAUDE.md non-negotiable 1", True), ("non-negotiable #1, a teaching scaffold", True), ("NN1", True),
    ("owner directive 2026-10-16", True), ("RULING host-filter-ok-1022", True), ("host-filter-ok-1022", True),
    ("my own judgement", False), ("RULING unsigned-key", False), ("RULING needs-key", False),
    ("RULING idle-key", False), ("the owner", False)])
def test_allowed_by_must_name_a_directive(val, ok):
    p = judge(prereg(scaffold(allowed_by=val)))
    assert (p == []) == ok, p
    if not ok:
        assert len(p) == 1 and "names no directive" in p[0]


@pytest.mark.parametrize("cap", ["0 builds", "6 builds", "0", "6", "-1 builds", "three builds", "2.5 builds", "10"])
def test_a_cap_outside_1_to_5_or_not_an_integer_blocks(cap):
    p = judge(prereg(scaffold(cap=cap)))
    assert len(p) == 1 and "must be an integer 1-5" in p[0]


@pytest.mark.parametrize("cap", ["1 builds", "1 build", "5 builds", "3", "2 builds (the prereg and one amendment)"])
def test_a_cap_1_to_5_is_accepted(cap):
    assert judge(prereg(scaffold(cap=cap))) == []


# ---------------------------------------------------------------- the cap, pure
def test_a_scaffold_prereg_whose_amendments_reach_its_cap_blocks():
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(2))) == []    # builds 1-3: within the cap
    p = judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3)))
    assert len(p) == 1 and "carries 3 amendment(s)" in p[0] and "build 4 exceeds the cap" in p[0]
    assert "cap-ruling: <KEY>" in p[0] and "2026-10-22-newplan-PREREGISTRATION.md" in p[0]


def test_amendments_in_other_forms_count_and_containers_do_not():
    other = ("\n**AMENDMENT 1 (2026-10-22, Claude)** a\n\n### Amendment 2 (2026-10-22)\nb\n"
             "\n## AMENDMENT 3 (2026-10-22)\nc\n### AMENDMENT 3, record (appended after the run)\nd\n")
    assert GATE.amendments(other) == ["1", "2", "3"]
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + other))
    log = "\n## AMENDMENT LOG\n\n(none at filing)\n" + amend(2) + "\n## Erratum 1 (2026-10-22)\nx\n"
    assert GATE.amendments(log) == ["1", "2"]
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + log)) == []


@pytest.mark.parametrize("key", ["cap-ok-1022", "dated-cap-key"])
def test_a_cap_ruling_that_is_a_claude_signed_section_6_ruling_citing_the_prereg_passes(key):
    assert judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3) + "\ncap-ruling: %s\n" % key)) == []


@pytest.mark.parametrize("key,why", [
    ("made-up-key", "not a Claude-signed RULING"), ("unsigned-key", "not a Claude-signed RULING"),
    ("idle-key", "not a Claude-signed RULING"), ("needs-key", "not a Claude-signed RULING"),
    ("other-key", "does not cite this prereg")])
def test_a_cap_ruling_naming_a_missing_unsigned_or_unrelated_ruling_blocks(key, why):
    p = judge(prereg(scaffold(cap="3 builds"), body="# a plan\n" + amend(3) + "\ncap-ruling: %s\n" % key))
    assert len(p) == 1 and "`cap-ruling: %s`" % key in p[0] and why in p[0]


def test_every_cited_cap_ruling_must_validate():
    body = "# a plan\n" + amend(3) + "\ncap-ruling: cap-ok-1022\ncap-ruling: made-up-key\n"
    p = judge(prereg(scaffold(cap="3 builds"), body=body))
    assert len(p) == 1 and "made-up-key" in p[0]


def test_a_packet_carries_the_header_but_no_cap():
    assert judge("# packet\n\n" + scaffold(cap="1 builds") + amend(4), rel=PACKET_DRAFT, kind="packet") == []
    p = judge("# packet\n", rel=PACKET_DRAFT, kind="packet")
    assert len(p) == 1 and "(packet)" in p[0] and "no `host-decision:` header" in p[0]


def test_the_real_moat_prereg_has_no_header_so_its_next_amendment_must_declare_itself():
    rel = "research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md"
    text = (ROOT / rel).read_text(encoding="utf-8")
    if "host-decision:" in text:
        pytest.skip("the moat prereg has declared itself since this test was written")
    assert GATE.amendments(text)[:7] == ["1", "2", "3", "4", "5", "6", "7"]
    assert any("no `host-decision:` header" in p for p in judge(text, rel=rel))
    hdr = scaffold(cap="5 builds")
    p = judge(text.replace("seeds: [42,", hdr + "seeds: [42,", 1), rel=rel)
    assert len(p) == 1 and "build 8 exceeds the cap" in p[0]


# ---------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,want", [
    (NEW, "prereg"), (PLANS_PREREG, "prereg"), ("research/findings/2026-10-22-x-prereg-amendment.md", "prereg"),
    (PACKET_DRAFT, "packet"), ("docs/drafts/PACKET.md", "packet"), ("docs/drafts/a/b/PACKET.md", "packet"),
    (PACKET_COORD, "packet"), ("research/coordination/-PACKET.md", "packet"),
    ("docs/drafts/some-lane/PLAN.md", None), ("docs/drafts/some-lane/SPEC.md", None),
    ("research/findings/2026-10-22-x-RESULT.md", None), ("research/findings/raw/x/2026-10-22-PREREG.md", None),
    ("research/coordination/LOCAL_LLM_RUNBOOK.md", None), ("research/coordination/x/y-PACKET.md", None),
    ("docs/PACKET.md", None)])
def test_scope_is_the_smk_prereg_set_plus_the_two_packet_paths(rel, want):
    assert GATE.kind_of(rel) == want


def test_the_prereg_scope_is_read_from_smk_not_copied(monkeypatch):
    import re
    assert GATE.kind_of(PLANS_PREREG) == "prereg"
    monkeypatch.setattr(SF, "PREREG_RE", re.compile(r"^nowhere$"))            # SMK's file set moves: so does this one
    assert GATE.kind_of(PLANS_PREREG) is None


# ---------------------------------------------------------------- commit time: real git, scratch repos
def _git(td, env, *a):
    r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *a], cwd=td, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


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

        def git(self, *a, date=None):
            e = dict(env)
            if date:
                e["GIT_COMMITTER_DATE"] = e["GIT_AUTHOR_DATE"] = date + "T12:00:00+00:00"
            return _git(self.td, e, *a)

        def commit(self, msg, date=None):
            self.git("add", "-A")
            self.git("commit", "-q", "-m", msg, date=date)

        def staged_check(self, kind="commit"):
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


def test_an_old_prereg_untouched_passes_and_a_clean_tree_passes(repo):
    assert repo.staged_check() == []
    repo.w("research/findings/2026-10-22-unrelated-RESULT.md", "# a result, no header\n")
    assert repo.staged_check() == []                                           # an unrelated commit leaves OLD alone


def test_gate_judges_a_renamed_old_prereg_and_ignores_a_deleted_one(repo):
    repo.git("mv", OLD, "research/findings/2026-10-02-oldplan-moved-PREREGISTRATION.md")
    p = GATE.check([], root=repo.td, invocation="commit")
    assert len(p) == 1 and "renames" in p[0] and "touch it, declare it" in p[0]
    repo.git("reset", "-q", "--hard")
    repo.git("rm", "-q", OLD)
    assert GATE.check([], root=repo.td, invocation="commit") == []


@pytest.mark.parametrize("rel", [PACKET_DRAFT, PACKET_COORD, "docs/drafts/a/b/PACKET.md"])
def test_a_packet_under_each_packet_path_is_covered(repo, rel):
    repo.w(rel, "# a packet\n")
    p = repo.staged_check()
    assert len(p) == 1 and rel in p[0] and "(packet" in p[0]
    repo.w(rel, "# a packet\n\n" + scaffold(cap="1 builds"))
    assert repo.staged_check() == []


def test_an_old_packet_edited_must_declare(repo):
    repo.w("docs/drafts/old-lane/PACKET.md", "# an old packet, no header\n\nstep 2 added\n")
    p = repo.staged_check()
    assert len(p) == 1 and "touch it, declare it" in p[0]


@pytest.mark.parametrize("rel", ["docs/drafts/some-lane/PLAN.md", "docs/drafts/some-lane/SPEC.md",
                                 "research/findings/2026-10-22-x-RESULT.md", "research/coordination/NOTES.md",
                                 "research/findings/raw/x/2026-10-22-PREREG.md"])
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
    repo.w(RUNBOOK, RULINGS.replace("## NEEDS CLAUDE", "- self-made: 2026-10-22-newplan-PREREGISTRATION.md may "
                                    "build again (Claude 2026-10-22)\n\n## NEEDS CLAUDE"))
    p = repo.staged_check()
    assert len(p) == 1 and "`cap-ruling: self-made`" in p[0] and "not a Claude-signed RULING" in p[0]
    repo.git("reset", "-q")
    repo.git("add", RUNBOOK)
    repo.git("commit", "-q", "-m", "Claude's ruling lands on main", date="2026-10-22")
    assert repo.staged_check() == []


def test_a_clean_auto_merge_is_not_checked_and_a_git_failure_fails_closed(repo, monkeypatch):
    repo.w(NEW, prereg())
    assert repo.staged_check("commit") and repo.staged_check("merge") == []

    def boom(*a, **k):
        raise PRA._GitReadError("simulated")
    monkeypatch.setattr(PRA, "_parents", boom)
    p = GATE.check([], root=repo.td, invocation="commit")
    assert len(p) == 1 and "failing CLOSED" in p[0]


# ---------------------------------------------------------------- the registry's trust signal
def test_the_registry_discovers_it_as_a_blocking_gate_with_a_clean_selftest():
    from tools.gates import discover
    found = {name: mod for name, mod, err in discover() if not err}
    assert found["host_decision_declared"].CLASS_ID == "HD" and found["host_decision_declared"].BLOCKING is True
    assert GATE.selftest() == []
    matrix = (ROOT / "docs" / "FAILURE_GATE_MATRIX.md").read_text(encoding="utf-8")
    assert "| HD |" in matrix and "`gates/host_decision_declared`" in matrix


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
    "no-header check removed": ('    if not head:\n        return ["%s has no', '    if False:\n        return ["%s has no'),
    "reason length unchecked": ("len(reason) < MIN_REASON", "len(reason) < 0"),
    "fields not required": ("    missing = [k for k in _FIELDS if k not in fields]", "    missing = []"),
    "replaced-by placeholder allowed": ('if "replaced-by" in vals and _first_word', 'if False and _first_word'),
    "cap range unchecked": ("not CAP_MIN <= int(cm.group(1)) <= CAP_MAX", "False"),
    "cap rule off": ("        if len(ams) >= cap:", "        if False:"),
    "any cap-ruling accepted": ("                    if not ok:", "                    if False:"),
    "packets out of scope": ('    if _PACKET_RE.match(rel):\n        return "packet"', '    if False:\n        return "packet"'),
    "old edits grandfathered": ('if kind_of(p) and all(c[p][0] != "D" for c in ch)]',
                                'if kind_of(p) and ch[0][p][0] == "A" and all(c[p][0] != "D" for c in ch)]'),
    "fail open on git errors": ('        return ["CLASS HD could not read', '        return []\n        return ["CLASS HD could not read'),
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
