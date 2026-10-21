"""The RAG corpus covers the coordination record (rulings, packets, RESULT files) and draft plans (2026-10-21).

Before this, research/coordination/ and docs/drafts/ were outside the index, so a search could not find the 10-16 and
10-20 rulings that bind #271. These tests pin the per-ruling split of the runbook and the source lists that must agree.
"""
from __future__ import annotations

import re
from pathlib import Path

from tools.rag import index_status
from tools.rag.coord_split import RUNBOOK_BASENAME, split_runbook

REPO = Path(__file__).resolve().parents[1]

RUNBOOK = """# Runbook
intro line

## 1. Every session
- read the board
- run the checks

## 2. Hard rules
never force-push

## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`; you follow it)

- push-credential: there is no push credential on S.
  continuation of the push ruling
- 2026-10-16 RULING c1-learned-1016 (Claude): constructions are learned, never English cue rules.
- 2026-10-20 RULING wrc-frame-go-1020b (Claude): no English list decides a role.
- [deixis-271-s0] RESULT census built
- [deixis-271-s0] RESULT census built
- spiking-commit-owner-calls: (Claude) first-come role assignment is acceptable (see RULING word-role-p4-checks).

## NEEDS CLAUDE
- 2026-10-21 14:28 EDT [cs-race step 1] wave consumed
"""


def _by_label(parts):
    return {p["label"]: p for p in parts}


def test_sections_before_rulings_stay_whole_and_rulings_split_per_bullet():
    parts = split_runbook(RUNBOOK)
    labels = [p["label"] for p in parts]
    assert labels[:3] == ["preamble", "1. Every session", "2. Hard rules"]
    one = _by_label(parts)["1. Every session"]
    assert "read the board" in one["text"] and "run the checks" in one["text"]
    assert "push-credential" in labels
    assert "RULING c1-learned-1016" in labels and "RULING wrc-frame-go-1020b" in labels
    assert "[cs-race step 1]" in labels                     # NEEDS CLAUDE bullets split too
    assert "spiking-commit-owner-calls" in labels           # a key mentioned later in the line names another ruling
    assert "RULING word-role-p4-checks" not in labels
    push = _by_label(parts)["push-credential"]
    assert "continuation of the push ruling" in push["text"]
    assert "RULING c1-learned-1016" not in push["text"]
    for p in parts:
        assert p["text"].startswith(f"{RUNBOOK_BASENAME}, section '")


def test_ids_are_unique_and_survive_edits_elsewhere():
    parts = split_runbook(RUNBOOK)
    fragments = [p["fragment"] for p in parts]
    assert len(set(fragments)) == len(fragments)            # the repeated RESULT line gets a suffix
    edited = RUNBOOK.replace("no English list decides a role.", "no English list decides a role (edited).")
    edited = edited.replace("never force-push", "never force-push or reset")
    before = {p["label"]: p["fragment"] for p in parts}
    after = {p["label"]: p["fragment"] for p in split_runbook(edited)}
    assert after["RULING c1-learned-1016"] == before["RULING c1-learned-1016"]
    assert after["push-credential"] == before["push-credential"]
    assert after["2. Hard rules"] == before["2. Hard rules"]


def test_real_runbook_loses_no_line_and_holds_the_binding_rulings_as_their_own_documents():
    text = (REPO / "research" / "coordination" / RUNBOOK_BASENAME).read_text(encoding="utf-8")
    parts = split_runbook(text)
    joined = "\n".join(p["text"] for p in parts)
    split_heads = {line for line in text.splitlines() if line.startswith("## ")}
    for line in text.splitlines():
        if line.strip() and line not in split_heads:
            assert line in joined, line[:120]
    labels = {p["label"] for p in parts}
    keys = set(re.findall(r"^- (?:\S+ )?RULING ([A-Za-z0-9][\w.\-]*)", text, flags=re.M))
    assert keys, "the runbook names no RULING key"
    assert {"RULING " + k for k in keys} <= labels


def _source_block(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_source_list_names_the_coordination_and_draft_corpora():
    builder = _source_block(REPO / "tools" / "rag" / "build_llamaindex_full.py")
    assert '("coord",   [os.path.join(SIM, "research", "coordination", "*.md")])' in builder
    assert '("draft",   [os.path.join(SIM, "docs", "drafts", "**", "*.md")])' in builder
    assert '"live_state.md"' in builder
    status = dict(index_status._patterns(REPO, REPO / "catalog"))
    assert status["coord"] == [str(REPO / "research/coordination/*.md")]
    assert status["draft"] == [str(REPO / "docs/drafts/**/*.md")]
    assert "live_state.md" in index_status.EXCLUDE_BASENAMES
    updater = _source_block(REPO / "tools" / "rag" / "update_indexes.py")
    assert '":(glob)research/coordination/*.md"' in updater and '":(glob)docs/drafts/**/*.md"' in updater
    search = _source_block(REPO / "tools" / "rag" / "rag_search.py")
    assert '"coord", "draft"}' in search


def test_builder_and_status_list_the_same_source_types():
    builder = _source_block(REPO / "tools" / "rag" / "build_llamaindex_full.py")
    start = builder.index("SOURCES = [")
    block = builder[start:builder.index("\n]\n", start)]
    builder_types = re.findall(r'^\s*\("(\w+)",', block, flags=re.M)
    status_types = [t for t, _ in index_status._patterns(REPO, REPO / "catalog")]
    # the static Kandel text is outside both freshness hashes (update_indexes.evolving_files skips it too)
    assert [t for t in builder_types if t != "kandel"] == status_types


def test_a_bulk_addition_rebuilds_instead_of_refreshing_one_document_at_a_time():
    from tools.rag.rag_paths import BULK_REBUILD_MIN_NEW, bulk_change_needs_rebuild

    indexed = {f"sim:research/findings/f{i}.md" for i in range(100)}
    few = indexed | {f"sim:research/coordination/c{i}.md" for i in range(BULK_REBUILD_MIN_NEW - 1)}
    many = indexed | {f"sim:research/coordination/c{i}.md" for i in range(BULK_REBUILD_MIN_NEW)}
    assert not bulk_change_needs_rebuild(few, indexed)
    assert bulk_change_needs_rebuild(many, indexed)
    assert not bulk_change_needs_rebuild(indexed - {"sim:research/findings/f0.md"}, indexed)   # deletions only
    updater = (REPO / "tools" / "rag" / "update_indexes.py").read_text(encoding="utf-8")
    assert "if bulk_change_needs_rebuild(cur_ids, idx.ref_doc_info.keys()):" in updater


def _hit(source, source_type):
    from types import SimpleNamespace

    return SimpleNamespace(node=SimpleNamespace(metadata={"source": source, "source_type": source_type},
                                                ref_doc_id=source), score=0.0)


def test_a_ruling_counts_as_named_only_when_the_query_spells_its_whole_key():
    from tools.rag.retrieval import _source_intent

    q = "may an English word list, strip rule or position rule decide a role on the word-role frame path?"
    assert _source_intent(q, _hit("LOCAL_LLM_RUNBOOK.md § word-role-credit-path", "coord")) == 0
    assert _source_intent(q, _hit("LOCAL_LLM_RUNBOOK.md § RULING wrc-frame-go-1020b", "coord")) == 0
    named = "what did RULING wrc-frame-go-1020b decide about English lists?"
    assert _source_intent(named, _hit("LOCAL_LLM_RUNBOOK.md § RULING wrc-frame-go-1020b", "coord")) == 1
    assert _source_intent(named, _hit("LOCAL_LLM_RUNBOOK.md § [deixis-271-s0]", "coord")) == 0
    # other corpora keep the word-level named-source prior
    assert _source_intent("the word-role parse finding", _hit("2026-10-06-word-role-parse-GO.md", "finding")) >= 2
