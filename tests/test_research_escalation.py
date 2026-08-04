from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import research_escalation as escalation
from tools.rag import index_status, source_intake


def _root(tmp_path: Path) -> Path:
    for relative in ("tools", "research/findings", "research/queue"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _start_args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        slug="tonic-output-wall",
        title="Tonic output research gate",
        blocked_experiment="GPi output remains silent without host current.",
        wall_reason="Two parameter sweeps did not produce autonomous tonic firing.",
        failed_attempt=[
            "Raised synaptic drive; firing remained trial-bound.",
            "Retuned reset; zero-input cells stayed silent.",
        ],
        parameter_question=["What autonomous GPi firing-rate range is measured without fast synaptic input?"],
        wiring_question=["Which GPe cell classes project to GPi output neurons, and where do they terminate?"],
        query="GPi autonomous tonic firing GPe projection",
        output=str(output),
    )


class FakeCommands:
    def __init__(
        self,
        *,
        index_state: str = "current",
        local_source: bool = False,
        malformed_search: bool = False,
        malformed_gate: bool = False,
        update_rc: int = 0,
    ) -> None:
        self.index_state = index_state
        self.local_source = local_source
        self.malformed_search = malformed_search
        self.malformed_gate = malformed_gate
        self.update_rc = update_rc
        self.calls: list[list[str]] = []

    def __call__(self, command, cwd, text, stdout, stderr, check):
        self.calls.append(command)
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, "abc123\n")
        if any(item.endswith("index_status.py") for item in command):
            rc = 0 if self.index_state == "current" else 1
            return subprocess.CompletedProcess(
                command,
                rc,
                json.dumps({"status": self.index_state, "reason": f"{self.index_state} test index"}) + "\n",
            )
        if any(item.endswith("before_you_build.sh") for item in command):
            return subprocess.CompletedProcess(command, 0, "prior work checked\n")
        if any(item.endswith("research_gate.sh") for item in command):
            if self.malformed_gate:
                return subprocess.CompletedProcess(command, 0, "unexpected gate output\n")
            if self.local_source:
                return subprocess.CompletedProcess(command, 0, "a primary source is represented\n")
            return subprocess.CompletedProcess(command, 1, "NO RELEVANT PRIMARY SOURCE\n")
        if any(item.endswith("record_external_search.sh") for item in command):
            return subprocess.CompletedProcess(command, 0, "external search recorded\n")
        if any(item.endswith("update_indexes.py") for item in command):
            return subprocess.CompletedProcess(command, self.update_rc, "index refresh\n")
        if any(item.endswith("finding_lint.py") for item in command):
            return subprocess.CompletedProcess(command, 0, "lint passed\n")
        if any(item.endswith("rag/search.sh") for item in command):
            if self.malformed_search:
                return subprocess.CompletedProcess(command, 0, "not parseable\n")
            query = command[2]
            return subprocess.CompletedProcess(
                command,
                0,
                f"Q: {query} (0.01s, top 5, corpus=test, index=/tmp/index)\n"
                f"  [1] 1.0 (catalog) {query}.md\n      at line 1\n      {query}\n",
            )
        raise AssertionError(f"unexpected command: {command}")


def _install_fake(monkeypatch, fake: FakeCommands) -> None:
    monkeypatch.setattr(escalation.subprocess, "run", fake)
    monkeypatch.setattr(
        escalation,
        "resolve_paths",
        lambda root: SimpleNamespace(rag_python=Path("/fake/rag-python")),
    )


def _start(tmp_path: Path, monkeypatch, fake: FakeCommands) -> tuple[Path, Path]:
    root = _root(tmp_path)
    output = root / "research/findings/gate.md"
    _install_fake(monkeypatch, fake)
    escalation.start(_start_args(output), root)
    return root, output


def _absence_args(gate: Path, question: str) -> argparse.Namespace:
    return argparse.Namespace(
        gate=str(gate),
        questions=question,
        database=["PubMed", "Crossref citation graph"],
        query=[f"{question} exact measurement", f"{question} quantitative wiring"],
        outcome="No primary report supplied the requested numerical value.",
        url=["https://pubmed.ncbi.nlm.nih.gov/?term=x", "https://search.crossref.org/?q=x"],
        date_from="1900-01-01",
        date_to="2026-08-04",
        claim_absence=True,
    )


def test_start_distinguishes_local_reading_from_no_local_evidence(tmp_path, monkeypatch):
    _, absent_gate = _start(tmp_path / "absent", monkeypatch, FakeCommands())
    absent = escalation._load(absent_gate)
    assert absent["status"] == "no-relevant-local-evidence"
    assert absent["revision"] == 1
    assert [record["command"][1] for record in absent["retrieval"]] == [
        "tools/before_you_build.sh",
        "tools/rag/index_status.py",
        "tools/rag/search.sh",
        "tools/research_gate.sh",
    ]

    _, local_gate = _start(tmp_path / "local", monkeypatch, FakeCommands(local_source=True))
    assert escalation._load(local_gate)["status"] == "local-reading-required"


@pytest.mark.parametrize(
    ("fake", "reason"),
    [
        (FakeCommands(index_state="unavailable"), "unavailable"),
        (FakeCommands(index_state="stale"), "stale"),
        (FakeCommands(malformed_search=True), "malformed"),
        (FakeCommands(malformed_gate=True), "malformed"),
    ],
)
def test_unavailable_stale_or_malformed_retrieval_fails_closed(tmp_path, monkeypatch, fake, reason):
    _, gate = _start(tmp_path, monkeypatch, fake)
    state = escalation._load(gate)
    assert state["status"] == "retrieval-unavailable"
    assert state["retrieval_blocked"] is True
    assert reason in state["retrieval_attempts"][-1]["reason"]


def test_failed_retrieval_blocks_finalization_until_successful_retry(tmp_path, monkeypatch):
    fake = FakeCommands(index_state="unavailable")
    root, gate = _start(tmp_path, monkeypatch, fake)
    fake.index_state = "current"
    for question in ("P1", "W1"):
        escalation.record_search(_absence_args(gate, question), root)
        search_id = escalation._load(gate)["searches"][-1]["id"]
        escalation.answer(
            argparse.Namespace(
                gate=str(gate), question=question, status="not-found",
                answer="The missing value must be bounded experimentally.", references=search_id,
            ),
            root,
        )
    finish = argparse.Namespace(
        gate=str(gate), decision="Use a bounded scan.", next_experiment="Run the preregistered lesion matrix."
    )
    with pytest.raises(escalation.EscalationError, match="retry-retrieval"):
        escalation.finalize(finish, root)
    escalation.retry_retrieval(argparse.Namespace(gate=str(gate)), root)
    escalation.finalize(finish, root)
    assert escalation._load(gate)["evidence_gate"] == "passed"


def test_not_found_rejects_incomplete_or_shared_absence_searches(tmp_path, monkeypatch):
    root, gate = _start(tmp_path, monkeypatch, FakeCommands())
    incomplete = _absence_args(gate, "P1")
    incomplete.database = ["PubMed"]
    incomplete.url = ["https://pubmed.ncbi.nlm.nih.gov/?term=x"]
    with pytest.raises(escalation.EscalationError, match="at least two databases"):
        escalation.record_search(incomplete, root)

    invalid_date = _absence_args(gate, "P1")
    invalid_date.date_from = "not-a-date"
    with pytest.raises(escalation.EscalationError, match="date-from"):
        escalation.record_search(invalid_date, root)

    shared = _absence_args(gate, "P1,W1")
    with pytest.raises(escalation.EscalationError, match="exactly one question"):
        escalation.record_search(shared, root)

    discovery = _absence_args(gate, "P1")
    discovery.claim_absence = False
    discovery.database = ["PubMed"]
    discovery.query = ["one query"]
    discovery.url = []
    escalation.record_search(discovery, root)
    with pytest.raises(escalation.EscalationError, match="question-specific absence"):
        escalation.answer(
            argparse.Namespace(
                gate=str(gate), question="P1", status="not-found",
                answer="No value found.", references="X1",
            ),
            root,
        )


def test_new_source_is_durably_intaken_indexed_and_retrievable(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    monkeypatch.setenv("SIM_CATALOG", str(catalog))
    fake = FakeCommands(local_source=True)
    root, gate = _start(tmp_path / "repo", monkeypatch, fake)
    args = argparse.Namespace(
        gate=str(gate),
        questions="P1",
        kind="peer-reviewed-primary",
        citation="Example et al. (2026), autonomous output measurements",
        url="https://doi.org/10.0000/example",
        query="autonomous output measurements",
        locator="Results, Figure 3",
        evidence="Cell-attached recordings report autonomous firing under receptor blockade.",
        license_status="metadata-only",
        local_file=None,
    )
    escalation.record_source(args, root)
    state = escalation._load(gate)
    intake = state["sources"][0]["intake"]
    assert intake["retrievable"] is True
    assert Path(intake["record_path"]).is_file()
    ledger = [json.loads(line) for line in (catalog / "source-intake.jsonl").read_text().splitlines()]
    assert ledger[0]["intake_id"] == intake["intake_id"]
    update_index = next(i for i, call in enumerate(fake.calls) if any(x.endswith("update_indexes.py") for x in call))
    retrieval_index = next(
        i for i, call in enumerate(fake.calls)
        if any(x.endswith("rag/search.sh") for x in call) and intake["intake_id"] in call
    )
    assert update_index < retrieval_index

    escalation.answer(
        argparse.Namespace(
            gate=str(gate), question="P1", status="resolved",
            answer="Use the reported preparation-specific range.", references="S1",
        ),
        root,
    )
    assert escalation._load(gate)["questions"][0]["status"] == "resolved"


def test_source_intake_failure_is_durable_and_reopens_retrieval_block(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    monkeypatch.setenv("SIM_CATALOG", str(catalog))
    root, gate = _start(tmp_path / "repo", monkeypatch, FakeCommands(local_source=True))

    def unavailable(_root):
        raise OSError("canonical RAG path is offline")

    monkeypatch.setattr(escalation, "resolve_paths", unavailable)
    escalation.record_source(
        argparse.Namespace(
            gate=str(gate), questions="P1", kind="peer-reviewed-primary",
            citation="Example et al. (2026)", url="https://doi.org/10.0000/offline",
            query="offline source", locator="Figure 1", evidence="A measured value.",
            license_status="metadata-only", local_file=None,
        ),
        root,
    )
    state = escalation._load(gate)
    assert Path(state["sources"][0]["intake"]["record_path"]).is_file()
    assert state["sources"][0]["intake"]["retrievable"] is False
    assert state["status"] == "retrieval-unavailable"
    assert state["retrieval_blocked"] is True


def test_local_copy_requires_permission_and_is_archived_when_permitted(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    local = tmp_path / "paper.txt"
    local.write_text("full licensed source text", encoding="utf-8")
    monkeypatch.setenv("SIM_CATALOG", str(catalog))
    with pytest.raises(source_intake.SourceIntakeError, match="local copy requires"):
        source_intake.register_source(
            tmp_path,
            citation="Example", url="https://example.org/paper", kind="peer-reviewed-primary",
            license_status="metadata-only", accessed_at="2026-08-04T00:00:00+00:00",
            questions=["P1"], query="query", locator="page 1", evidence="evidence", local_file=str(local),
        )
    record = source_intake.register_source(
        tmp_path,
        citation="Example", url="https://example.org/paper", kind="peer-reviewed-primary",
        license_status="open-access", accessed_at="2026-08-04T00:00:00+00:00",
        questions=["P1"], query="query", locator="page 1", evidence="evidence", local_file=str(local),
    )
    assert Path(record["archived_path"]).read_text(encoding="utf-8") == "full licensed source text"


def test_concurrent_writers_keep_every_update_and_unique_ids(tmp_path, monkeypatch):
    fake = FakeCommands()
    root, gate = _start(tmp_path, monkeypatch, fake)

    def add(index: int) -> None:
        escalation.record_search(
            argparse.Namespace(
                gate=str(gate), questions="P1", database=[f"Database {index}"],
                query=[f"query {index}"], outcome=f"outcome {index}", url=[],
                date_from=None, date_to=None, claim_absence=False,
            ),
            root,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(24)))
    state = escalation._load(gate)
    assert len(state["searches"]) == 24
    assert {item["id"] for item in state["searches"]} == {f"X{i}" for i in range(1, 25)}
    assert state["revision"] == 25


def test_index_status_detects_catalog_staleness(tmp_path, monkeypatch):
    repo = tmp_path / "sim"
    catalog = tmp_path / "catalog"
    rag_root = tmp_path / "rag"
    (repo / "docs").mkdir(parents=True)
    catalog.mkdir()
    (rag_root / "llamaindex_full").mkdir(parents=True)
    (repo / "docs/note.md").write_text("current", encoding="utf-8")
    (catalog / "feature.md").write_text("catalog", encoding="utf-8")
    expected = index_status.corpus_manifest_hash(repo, catalog)
    (rag_root / ".rag_manifest.json").write_text(json.dumps({"hash": expected}), encoding="utf-8")
    (rag_root / ".rag_schema.json").write_text(
        json.dumps({"document_id_schema": "repo-relative-v1", "source_repo": str(repo)}), encoding="utf-8"
    )
    paths = SimpleNamespace(rag_root=rag_root, full_index=rag_root / "llamaindex_full", catalog=catalog)
    monkeypatch.setattr(index_status, "resolve_paths", lambda _repo: paths)
    assert index_status.inspect_index(repo)["status"] == "current"
    (catalog / "new-source.md").write_text("new evidence", encoding="utf-8")
    assert index_status.inspect_index(repo)["status"] == "stale"
