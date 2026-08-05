#!/usr/bin/env python3
"""Create and maintain a bounded research escalation gate.

This tool orchestrates the repository's existing retrieval and evidence tools. It
does not search literature or judge experimental results itself.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Iterable

try:
    from tools import research_packet
    from tools.rag import source_intake
    from tools.rag.rag_paths import resolve_paths
except ModuleNotFoundError:  # direct script execution
    import research_packet
    from rag import source_intake
    from rag.rag_paths import resolve_paths


ROOT = Path(__file__).resolve().parents[1]
STATE_PREFIX = "<!-- research-escalation-state:v1"
STATE_SUFFIX = "-->"
MAX_CAPTURE_CHARS = 30_000
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SOURCE_QUALITY = {
    "peer-reviewed-primary": (
        "high",
        "Peer-reviewed primary measurements or methods; may resolve a question.",
    ),
    "primary-preprint": (
        "medium",
        "Primary measurements without completed peer review; may resolve with an explicit caveat.",
    ),
    "review": (
        "context-only",
        "Useful for discovery and synthesis, but not sufficient to resolve a parameter or wiring question.",
    ),
    "secondary": (
        "discovery-only",
        "Useful only as a pointer to primary evidence.",
    ),
}
RESOLVING_SOURCE_KINDS = {"peer-reviewed-primary", "primary-preprint"}
ABSENCE_MIN_DATABASES = 2
ABSENCE_MIN_QUERY_VARIANTS = 2


class EscalationError(RuntimeError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _run(args: list[str], root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    truncated = len(output) > MAX_CAPTURE_CHARS
    if truncated:
        output = output[:MAX_CAPTURE_CHARS] + "\n[output truncated by research_escalation.py]\n"
    return {
        "command": args,
        "returncode": completed.returncode,
        "captured_at": _utc_now(),
        "output": output,
        "truncated": truncated,
    }


def _git_sha(root: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], root)
    return result["output"].strip() if result["returncode"] == 0 else "unknown"


def _encode_state(state: dict[str, Any]) -> str:
    raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, width=100))


def _decode_state(text: str) -> dict[str, Any]:
    start = text.find(STATE_PREFIX)
    if start < 0:
        raise EscalationError("not a research-escalation artifact (state marker missing)")
    payload_start = text.find("\n", start)
    end = text.find(STATE_SUFFIX, payload_start)
    if payload_start < 0 or end < 0:
        raise EscalationError("research-escalation state marker is malformed")
    payload = "".join(text[payload_start:end].split())
    try:
        return json.loads(base64.b64decode(payload).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EscalationError(f"research-escalation state is unreadable: {exc}") from exc


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EscalationError(f"gate artifact does not exist: {path}")
    return _decode_state(path.read_text(encoding="utf-8"))


def _safe_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _command_text(record: dict[str, Any]) -> str:
    return " ".join(record["command"])


def _packet_review_status(packet: dict[str, Any]) -> str:
    statuses = {claim.get("status", "pending_review") for claim in packet.get("claims", [])}
    if "rejected" in statuses:
        return "rejected"
    if statuses and statuses == {"accepted"}:
        return "accepted"
    return "pending_review"


def _packet_is_promotable(packet: dict[str, Any]) -> bool:
    claims = packet.get("claims", [])
    if not claims or not all(claim.get("status") == "accepted" for claim in claims):
        return False
    # Packet review is necessary but does not replace the repository's existing source-intake gate. The
    # external packet may carry a URL and a locator, but those are not evidence that the cited source is present
    # in the maintained catalog/RAG path. A later intake step must attach a durable, retrievable intake record
    # before an RP record can resolve a research question.
    sources = packet.get("sources", [])
    return bool(sources) and all(
        isinstance(source.get("intake"), dict)
        and isinstance(source["intake"].get("intake_id"), str)
        and bool(source["intake"].get("retrievable"))
        for source in sources
    )


def _validated_packet_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    try:
        packet = research_packet.validate_packet(record.get("packet"))
    except research_packet.ResearchPacketError as exc:
        raise EscalationError(
            f"stored research packet {record.get('id', 'unknown')} failed validation: {exc}"
        ) from exc
    if packet["question"]["id"] != record.get("question_id"):
        raise EscalationError(
            f"stored research packet {record.get('id', 'unknown')} is bound to the wrong question"
        )
    return packet, _packet_is_promotable(packet)


def _render(state: dict[str, Any]) -> str:
    questions = state["questions"]
    searches = state.get("searches", [])
    sources = state.get("sources", [])
    packets = state.get("packets", [])
    retrieval_attempts = state.get("retrieval_attempts") or [
        {"id": "R1", "status": state.get("status", "unknown"), "records": state.get("retrieval", [])}
    ]
    retrieval = retrieval_attempts[-1]["records"]
    lines = [
        "---",
        "type: research-gate",
        f"status: {state['status']}",
        f"date: {state['date']}",
        "lane: research",
        f"mechanism: {_safe_cell(state['slug'])}",
        f"revision: {state.get('revision', 0)}",
        "---",
        "",
        f"# {state['title']}",
        "",
        "## Wall that triggered this gate",
        "",
        state["blocked_experiment"].strip(),
        "",
        f"**Why work stopped:** {state['wall_reason'].strip()}",
        "",
        "### Failed attempts already counted",
        "",
    ]
    lines.extend(f"- {attempt}" for attempt in state["failed_attempts"])
    lines.extend([
        "",
        "No further implementation or tuning against this wall is authorized until every open question below is",
        "resolved from primary evidence or explicitly recorded as absent after an external search.",
        "",
        "## Retrieval performed first",
        "",
        f"Query: `{state['query']}`",
        "",
        "| Existing repository tool | Result | Time (UTC) |",
        "|---|---:|---|",
    ])
    for record in retrieval:
        lines.append(
            f"| `{_safe_cell(_command_text(record))}` | rc={record['returncode']} | {record['captured_at']} |"
        )
    latest_attempt = retrieval_attempts[-1]
    lines.extend([
        "",
        f"Retrieval state: `{latest_attempt['status']}`. {latest_attempt.get('reason', '')}",
        f"Attempt: `{latest_attempt['id']}` of {len(retrieval_attempts)}.",
    ])
    lines.extend([
        "",
        "The exact captured output is retained in the embedded machine-readable state. Use",
        "`python tools/research_escalation.py inspect --gate <path>` to review it without rerunning retrieval.",
        "",
        "## Unresolved questions",
        "",
        "| ID | Kind | Status | Precise question |",
        "|---|---|---|---|",
    ])
    for question in questions:
        lines.append(
            f"| {question['id']} | {question['kind']} | {question['status']} | {_safe_cell(question['text'])} |"
        )

    lines.extend(["", "## External searches", ""])
    if searches:
        lines.extend([
            "| ID | Questions | Databases | Query variants | Date range | URLs | Outcome |",
            "|---|---|---|---|---|---|---|",
        ])
        for search in searches:
            lines.append(
                "| {id} | {questions} | {databases} | {queries} | {date_range} | {urls} | {outcome} |".format(
                    id=search["id"],
                    questions=", ".join(search["questions"]),
                    databases=_safe_cell("; ".join(search.get("databases", [search.get("database", "")]))),
                    queries=_safe_cell("; ".join(search.get("query_variants", [search.get("query", "")]))),
                    date_range=_safe_cell(
                        f"{search.get('date_from', 'unspecified')} to {search.get('date_to', 'unspecified')}"
                    ),
                    urls=_safe_cell("; ".join(search.get("urls", [search.get("url", "")]))),
                    outcome=_safe_cell(search["outcome"]),
                )
            )
    else:
        lines.append("No external search has been recorded yet.")

    lines.extend(["", "## Source-quality record", ""])
    if sources:
        for source in sources:
            lines.extend([
                f"### {source['id']}: {source['citation']}",
                "",
                f"- Questions: {', '.join(source['questions'])}",
                f"- Source kind: `{source['kind']}`; quality: `{source['quality']}`",
                f"- Provenance: [{source['url']}]({source['url']}) (accessed {source['accessed_at']})",
                f"- Search query: {source['query']}",
                f"- Exact locator: {source['locator']}",
                f"- Relevant evidence: {source['evidence']}",
                f"- Quality note: {source['quality_note']}",
                f"- Catalog intake: `{source.get('intake', {}).get('intake_id', 'not-recorded')}`; "
                f"retrievable: `{source.get('intake', {}).get('retrievable', False)}`",
                "",
            ])
    else:
        lines.append("No source has been recorded yet.")

    lines.extend(["", "## External research packets", ""])
    if packets:
        for handoff in packets:
            packet = handoff["packet"]
            lines.extend([
                f"### {handoff['id']}: question {handoff['question_id']}",
                "",
                f"- Packet file: `{_safe_cell(handoff['packet_path'])}`",
                f"- Received: {handoff['received_at']}",
                f"- Review status: `{handoff['status']}`; promotable as resolved evidence: `{handoff['promotable']}`",
                "- Prior-work matches:",
            ])
            for prior in packet["prior_work_matches"]:
                lines.append(
                    f"  - `{prior['id']}` `{prior['status']}`: {_safe_cell(prior['reference'])}; "
                    f"{_safe_cell(prior['relationship'])}; {_safe_cell(prior['summary'])}"
                )
            lines.append("- Online searches:")
            for search in packet["online_searches"]:
                lines.append(
                    f"  - `{search['id']}` {_safe_cell('; '.join(search['databases']))}; "
                    f"queries: {_safe_cell('; '.join(search['query_variants']))}; "
                    f"URLs: {_safe_cell('; '.join(search['urls']))}; {_safe_cell(search['outcome'])}"
                )
            lines.append("- Sources and provenance claims:")
            for source in packet["sources"]:
                lines.append(
                    f"  - `{source['id']}` `{source['kind']}`: {_safe_cell(source['citation'])}; "
                    f"{_safe_cell(source['url'])}; locator: {_safe_cell(source['locator'])}; "
                    f"evidence: {_safe_cell(source['evidence'])}; license: `{source['license_status']}`"
                )
            lines.append("- Structured claims:")
            for claim in packet["claims"]:
                value = claim.get("value")
                value_text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
                review = claim.get("review") or {}
                review_text = review.get("decision", "not reviewed")
                if review.get("reviewer"):
                    review_text += f" by {review['reviewer']}"
                lines.append(
                    f"  - `{claim['id']}` `{claim.get('status', 'pending_review')}`; "
                    f"value: {_safe_cell(value_text)} {_safe_cell(claim['units'])}; "
                    f"condition: {_safe_cell(claim['condition'])}; sources: {', '.join(claim['source_ids'])}; "
                    f"locator: {_safe_cell(claim['locator'])}; review: {_safe_cell(review_text)}; "
                    f"limitations: {_safe_cell(claim['limitations'])}"
                )
            lines.append("")
    else:
        lines.append("No external research packet has been handed off yet.")

    lines.extend(["", "## Question dispositions", ""])
    dispositions = [q for q in questions if q["status"] != "open"]
    if dispositions:
        for question in dispositions:
            refs = question.get("references", [])
            lines.extend([
                f"### {question['id']}: {question['status']}",
                "",
                question["answer"],
                "",
                f"Evidence records: {', '.join(refs)}",
                "",
            ])
    else:
        lines.append("All questions remain open.")

    lines.extend(["", "## Gate decision", ""])
    if state.get("decision"):
        lines.extend([
            state["decision"],
            "",
            f"**Next bounded experiment:** {state['next_experiment']}",
            "",
            f"Existing evidence gate: `{state.get('evidence_gate', 'pending')}`",
        ])
    else:
        lines.append("Pending. Implementation remains stopped at this wall.")

    lines.extend([
        "",
        "## Reuse rule",
        "",
        "Future work on the same mechanism must retrieve this artifact, cite the relevant question IDs, and",
        "explain what new evidence changes the decision before repeating a failed attempt.",
        "",
        STATE_PREFIX,
        _encode_state(state),
        STATE_SUFFIX,
        "",
    ])
    return "\n".join(lines)


@contextmanager
def _gate_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write(path: Path, state: dict[str, Any], *, increment_revision: bool = True) -> None:
    if increment_revision:
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_at"] = _utc_now()
    _atomic_text_write(path, _render(state))


def _parse_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _question_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {question["id"]: question for question in state["questions"]}


def _validate_question_ids(state: dict[str, Any], ids: Iterable[str]) -> list[str]:
    result = list(ids)
    known = _question_map(state)
    unknown = sorted(set(result) - set(known))
    if unknown:
        raise EscalationError(f"unknown question ID(s): {', '.join(unknown)}")
    if not result:
        raise EscalationError("at least one question ID is required")
    return result


def _artifact_path(root: Path, slug: str, output: str | None) -> Path:
    if output:
        path = Path(output)
        return path if path.is_absolute() else root / path
    date = dt.date.today().isoformat()
    return root / "research" / "findings" / f"{date}-{slug}-research-escalation-gate.md"


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _valid_date_range(date_from: str | None, date_to: str | None) -> bool:
    if not date_from or not date_to:
        return False
    try:
        start = dt.date.fromisoformat(date_from)
        end = dt.date.fromisoformat(date_to)
    except ValueError:
        return False
    return start <= end


def _valid_rag_search(record: dict[str, Any]) -> bool:
    output = record.get("output", "")
    return (
        record.get("returncode") == 0
        and "Q:" in output
        and "index=" in output
        and (re.search(r"\n\s*\[1\]", output) is not None or "(no hits" in output)
    )


def _index_record_status(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("returncode") not in (0, 1):
        return "unavailable", "index freshness command failed"
    try:
        payload = json.loads(record.get("output", ""))
    except json.JSONDecodeError:
        return "unavailable", "index freshness output was malformed"
    status = payload.get("status")
    if status not in {"current", "stale", "unavailable"}:
        return "unavailable", "index freshness result was malformed"
    return status, str(payload.get("reason", ""))


def _run_retrieval(query: str, root: Path) -> dict[str, Any]:
    records = [
        _run(["bash", "tools/before_you_build.sh", query], root),
        _run([sys.executable, "tools/rag/index_status.py", "--json"], root),
        _run(["bash", "tools/rag/search.sh", query, "5", "--corpus", "plan"], root),
        _run(["bash", "tools/research_gate.sh", query], root),
    ]
    index_status, index_reason = _index_record_status(records[1])
    reason = ""
    if records[0]["returncode"] != 0:
        status = "retrieval-unavailable"
        reason = "prior-work retrieval failed"
    elif index_status != "current":
        status = "retrieval-unavailable"
        reason = f"RAG index is {index_status}: {index_reason}".strip()
    elif not _valid_rag_search(records[2]):
        status = "retrieval-unavailable"
        reason = "RAG search failed or returned malformed output"
    elif records[3]["returncode"] == 0 and "primary source is represented" in records[3]["output"]:
        status = "local-reading-required"
        reason = "relevant local primary material was surfaced and must be read"
    elif records[3]["returncode"] == 1 and "NO RELEVANT PRIMARY SOURCE" in records[3]["output"]:
        status = "no-relevant-local-evidence"
        reason = "healthy local retrieval found no relevant primary evidence"
    else:
        status = "retrieval-unavailable"
        reason = "primary-source retrieval failed or returned malformed output"
    return {
        "id": "",
        "status": status,
        "reason": reason,
        "completed_at": _utc_now(),
        "records": records,
    }


def _append_retrieval_attempt(state: dict[str, Any], attempt: dict[str, Any]) -> None:
    attempts = state.setdefault("retrieval_attempts", [])
    attempt["id"] = f"R{len(attempts) + 1}"
    attempts.append(attempt)
    state["retrieval"] = attempt["records"]
    state["status"] = attempt["status"]
    state["retrieval_blocked"] = attempt["status"] == "retrieval-unavailable"
    if not state["retrieval_blocked"]:
        state["retrieval_recovered_at"] = attempt["completed_at"]


def _refresh_rag(root: Path) -> dict[str, Any]:
    try:
        paths = resolve_paths(root)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": ["resolve-rag-paths"],
            "returncode": 1,
            "captured_at": _utc_now(),
            "output": str(exc),
            "truncated": False,
        }
    return _run(
        [str(paths.rag_python), "tools/rag/update_indexes.py", "--force"], root
    )


def _refresh_and_verify_intake(
    root: Path,
    intake: dict[str, Any],
    *,
    shared_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    update = shared_update if shared_update is not None else _refresh_rag(root)
    if update["command"] == ["resolve-rag-paths"]:
        return {
            "retrievable": False,
            "reason": f"RAG paths unavailable: {update['output']}",
            "update": update,
            "verification": update,
        }
    verify = _run(
        [
            "bash",
            "tools/rag/search.sh",
            intake["intake_id"],
            "5",
            "--corpus",
            "catalog",
        ],
        root,
    )
    record_name = Path(intake["record_path"]).name
    retrievable = (
        update["returncode"] == 0
        and _valid_rag_search(verify)
        and (intake["intake_id"] in verify["output"] or record_name in verify["output"])
    )
    return {
        "retrievable": retrievable,
        "verified_at": _utc_now(),
        "update": update,
        "verification": verify,
        "reason": "" if retrievable else "incremental update or source-specific retrieval failed",
    }


def start(args: argparse.Namespace, root: Path) -> Path:
    if not SLUG_RE.fullmatch(args.slug):
        raise EscalationError("--slug must contain lowercase words separated by single hyphens")
    if len(args.failed_attempt) < 2:
        raise EscalationError("the wall trigger requires at least two --failed-attempt entries")
    if not args.parameter_question or not args.wiring_question:
        raise EscalationError("provide at least one --parameter-question and one --wiring-question")
    path = _artifact_path(root, args.slug, args.output)
    with _gate_lock(path):
        if path.exists():
            raise EscalationError(f"refusing to overwrite existing gate: {path}")
        query = args.query or args.blocked_experiment
        attempt = _run_retrieval(query, root)
        questions: list[dict[str, Any]] = []
        for index, text in enumerate(args.parameter_question, start=1):
            questions.append({"id": f"P{index}", "kind": "parameter", "status": "open", "text": text})
        for index, text in enumerate(args.wiring_question, start=1):
            questions.append({"id": f"W{index}", "kind": "wiring", "status": "open", "text": text})

        state = {
            "schema": 2,
            "revision": 0,
            "date": dt.date.today().isoformat(),
            "created_at": _utc_now(),
            "git_sha": _git_sha(root),
            "slug": args.slug,
            "title": args.title or f"Research escalation: {args.slug.replace('-', ' ')}",
            "blocked_experiment": args.blocked_experiment,
            "wall_reason": args.wall_reason,
            "failed_attempts": args.failed_attempt,
            "query": query,
            "retrieval_attempts": [],
            "questions": questions,
            "searches": [],
            "sources": [],
            "packets": [],
            "decision": None,
            "next_experiment": None,
            "evidence_gate": "pending",
        }
        _append_retrieval_attempt(state, attempt)
        _write(path, state)
    return path


def retry_retrieval(args: argparse.Namespace, root: Path) -> Path:
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        state = _load(path)
        pending_intakes = [
            source["intake"]
            for source in state.get("sources", [])
            if not source.get("intake", {}).get("retrievable")
        ]
        for record in state.get("packets", []):
            packet = record.get("packet", {})
            if record.get("status") != "accepted":
                continue
            for source in packet.get("sources", []):
                intake = source.get("intake")
                if isinstance(intake, dict) and not intake.get("retrievable"):
                    pending_intakes.append(intake)
        shared_update = _refresh_rag(root) if pending_intakes else None
        for intake in pending_intakes:
            intake.update(
                _refresh_and_verify_intake(root, intake, shared_update=shared_update)
            )
        for record in state.get("packets", []):
            packet = record.get("packet", {})
            if record.get("status") == "accepted":
                record["promotable"] = _packet_is_promotable(packet)
        attempt = _run_retrieval(state["query"], root)
        unresolved_sources = any(
            not source.get("intake", {}).get("retrievable") for source in state.get("sources", [])
        )
        unresolved_packets = any(
            record.get("status") == "accepted" and not record.get("promotable")
            for record in state.get("packets", [])
        )
        if unresolved_sources or unresolved_packets:
            attempt["status"] = "retrieval-unavailable"
            attempt["reason"] = (
                "one or more source or reviewed-packet intake records are still absent from RAG retrieval"
            )
        _append_retrieval_attempt(state, attempt)
        _write(path, state)
    return path


def record_search(args: argparse.Namespace, root: Path) -> Path:
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        state = _load(path)
        question_ids = _validate_question_ids(state, _parse_ids(args.questions))
        databases = list(dict.fromkeys(_as_list(args.database)))
        queries = list(dict.fromkeys(_as_list(args.query)))
        urls = list(dict.fromkeys(_as_list(getattr(args, "url", []))))
        claim_absence = bool(getattr(args, "claim_absence", False))
        date_from = getattr(args, "date_from", None)
        date_to = getattr(args, "date_to", None)
        if not databases or not queries:
            raise EscalationError("record-search requires at least one database and query")
        complete_absence = (
            len(question_ids) == 1
            and len(databases) >= ABSENCE_MIN_DATABASES
            and len(queries) >= ABSENCE_MIN_QUERY_VARIANTS
            and _valid_date_range(date_from, date_to)
            and len(urls) >= len(databases)
            and all(re.match(r"^https?://", url) for url in urls)
        )
        if claim_absence and not complete_absence:
            raise EscalationError(
                "an absence search requires exactly one question, at least two databases, "
                "two query variants, an ordered --date-from/--date-to range, and an http(s) URL per database"
            )
        summary = f"{'; '.join(databases)}: {args.outcome}"
        result = _run(["bash", "tools/record_external_search.sh", queries[0], summary], root)
        if result["returncode"] != 0:
            raise EscalationError(f"record_external_search.sh failed:\n{result['output']}")
        state["searches"].append({
            "id": f"X{len(state['searches']) + 1}",
            "questions": question_ids,
            "databases": databases,
            "query_variants": queries,
            "date_from": date_from,
            "date_to": date_to,
            "urls": urls,
            "claim_absence": claim_absence,
            "absence_protocol_complete": complete_absence,
            "outcome": args.outcome,
            "recorded_at": _utc_now(),
            "recorder": result,
        })
        _write(path, state)
    return path


def record_source(args: argparse.Namespace, root: Path) -> Path:
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        state = _load(path)
        question_ids = _validate_question_ids(state, _parse_ids(args.questions))
        if not re.match(r"^https?://", args.url):
            raise EscalationError("--url must be an http(s) primary-source location")
        quality, quality_note = SOURCE_QUALITY[args.kind]
        summary = f"{args.citation}; {args.url}; {args.evidence}"
        result = _run(["bash", "tools/record_external_search.sh", args.query, summary], root)
        if result["returncode"] != 0:
            raise EscalationError(f"record_external_search.sh failed:\n{result['output']}")
        accessed_at = _utc_now()
        try:
            intake = source_intake.register_source(
                root,
                citation=args.citation,
                url=args.url,
                kind=args.kind,
                license_status=getattr(args, "license_status", "metadata-only"),
                accessed_at=accessed_at,
                questions=question_ids,
                query=args.query,
                locator=args.locator,
                evidence=args.evidence,
                local_file=getattr(args, "local_file", None),
            )
        except source_intake.SourceIntakeError as exc:
            raise EscalationError(str(exc)) from exc
        verification = _refresh_and_verify_intake(root, intake)
        intake.update(verification)
        state["sources"].append({
            "id": f"S{len(state['sources']) + 1}",
            "questions": question_ids,
            "kind": args.kind,
            "quality": quality,
            "quality_note": quality_note,
            "citation": args.citation,
            "url": args.url,
            "query": args.query,
            "locator": args.locator,
            "evidence": args.evidence,
            "accessed_at": accessed_at,
            "recorder": result,
            "intake": intake,
        })
        if not intake["retrievable"]:
            attempt = {
                "id": "",
                "status": "retrieval-unavailable",
                "reason": "new source intake was durable but did not become retrievable",
                "completed_at": _utc_now(),
                "records": [verification["update"], verification["verification"]],
            }
            _append_retrieval_attempt(state, attempt)
        _write(path, state)
    return path


def handoff_packet(args: argparse.Namespace, root: Path) -> Path:
    packet_path = Path(args.packet)
    if not packet_path.is_absolute():
        packet_path = root / packet_path
    try:
        packet = research_packet.load_packet(packet_path)
    except research_packet.ResearchPacketError as exc:
        raise EscalationError(f"research packet validation failed: {exc}") from exc

    packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        state = _load(path)
        question_id = packet["question"]["id"]
        if question_id not in _question_map(state):
            raise EscalationError(
                f"research packet question ID is not present in this gate: {question_id}"
            )
        review_status = _packet_review_status(packet)
        if review_status == "accepted":
            source_claims = {
                source["id"]: [
                    claim for claim in packet["claims"] if source["id"] in claim["source_ids"]
                ]
                for source in packet["sources"]
            }
            provenance = {
                "packet_path": str(packet_path.resolve()),
                "packet_sha256": packet_sha256,
                "packet_version": packet["packet_version"],
                "question_id": question_id,
            }
            for source in packet["sources"]:
                search = next(
                    item for item in packet["online_searches"] if item["id"] == source["search_id"]
                )
                try:
                    intake = source_intake.register_source(
                        root,
                        citation=source["citation"],
                        url=source["url"],
                        kind=source["kind"],
                        license_status=source["license_status"],
                        accessed_at=packet["created_at"],
                        questions=[question_id],
                        query="; ".join(search["query_variants"]),
                        locator=source["locator"],
                        evidence=source["evidence"],
                        parameter_claims=source_claims[source["id"]],
                        packet_provenance=provenance,
                    )
                except source_intake.SourceIntakeError as exc:
                    raise EscalationError(f"reviewed packet source intake failed: {exc}") from exc
                intake.update(_refresh_and_verify_intake(root, intake))
                source["intake"] = intake

        packets = state.setdefault("packets", [])
        packets.append({
            "id": f"RP{len(packets) + 1}",
            "questions": [question_id],
            "question_id": question_id,
            "packet_path": str(packet_path.resolve()),
            "received_at": _utc_now(),
            "packet_sha256": packet_sha256,
            "status": review_status,
            "promotable": _packet_is_promotable(packet),
            "packet": packet,
        })
        if review_status == "accepted" and not _packet_is_promotable(packet):
            failed = [
                source["intake"]
                for source in packet["sources"]
                if not source["intake"]["retrievable"]
            ]
            attempt = {
                "id": "",
                "status": "retrieval-unavailable",
                "reason": "reviewed packet intake was durable but did not become retrievable",
                "completed_at": _utc_now(),
                "records": [
                    record
                    for intake in failed
                    for record in (intake["update"], intake["verification"])
                ],
            }
            _append_retrieval_attempt(state, attempt)
        _write(path, state)
    return path


def answer(args: argparse.Namespace, root: Path) -> Path:
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        state = _load(path)
        question = _question_map(state).get(args.question)
        if question is None:
            raise EscalationError(f"unknown question ID: {args.question}")
        references = _parse_ids(args.references)
        source_map = {source["id"]: source for source in state["sources"]}
        search_map = {search["id"]: search for search in state["searches"]}
        packet_map = {packet["id"]: packet for packet in state.get("packets", [])}
        unknown = sorted(set(references) - set(source_map) - set(search_map) - set(packet_map))
        if unknown:
            raise EscalationError(f"unknown evidence record(s): {', '.join(unknown)}")
        if not references:
            raise EscalationError("an answer requires at least one source/search reference")
        unrelated = [
            ref for ref in references
            if args.question not in (
                source_map.get(ref) or search_map.get(ref) or packet_map.get(ref, {})
            ).get("questions", [])
        ]
        if unrelated:
            raise EscalationError(f"evidence record(s) do not cover {args.question}: {', '.join(unrelated)}")
        if args.status == "resolved":
            packet_resolution = []
            for record in (packet_map[ref] for ref in references if ref in packet_map):
                _, promotable = _validated_packet_record(record)
                if not promotable:
                    packet_resolution.append(record["id"])
            if packet_resolution:
                raise EscalationError(
                    "unreviewed or unresolved research packet(s) cannot be promoted as resolved evidence: "
                    + ", ".join(packet_resolution)
                )
            resolving = [
                source_map[ref] for ref in references
                if ref in source_map
                and source_map[ref]["kind"] in RESOLVING_SOURCE_KINDS
                and source_map[ref].get("intake", {}).get("retrievable")
            ]
            resolving.extend(packet_map[ref] for ref in references if ref in packet_map)
            if not resolving:
                raise EscalationError(
                    "resolved questions require a retrievable peer-reviewed primary source or primary preprint"
                )
        if args.status == "not-found":
            absence_records = [
                search_map[ref] for ref in references
                if ref in search_map
                and search_map[ref].get("claim_absence")
                and search_map[ref].get("absence_protocol_complete")
                and search_map[ref].get("questions") == [args.question]
            ]
            if not absence_records:
                raise EscalationError(
                    "not-found requires a complete, question-specific absence search protocol"
                )
        question.update({"status": args.status, "answer": args.answer, "references": references})
        _write(path, state)
    return path


def finalize(args: argparse.Namespace, root: Path) -> Path:
    path = Path(args.gate).resolve()
    with _gate_lock(path):
        original = path.read_text(encoding="utf-8")
        state = _decode_state(original)
        open_ids = [question["id"] for question in state["questions"] if question["status"] == "open"]
        if open_ids:
            raise EscalationError(f"cannot finalize with open questions: {', '.join(open_ids)}")
        if not state["sources"] and not state["searches"] and not state.get("packets"):
            raise EscalationError("cannot finalize without recorded external evidence work")
        latest = state.get("retrieval_attempts", [{}])[-1]
        if state.get("retrieval_blocked") or latest.get("status") == "retrieval-unavailable":
            raise EscalationError(
                "cannot finalize after a RAG/index failure until retry-retrieval records a successful recovery"
            )
        missing_intakes = [
            source["id"] for source in state.get("sources", [])
            if not source.get("intake", {}).get("retrievable")
        ]
        if missing_intakes:
            raise EscalationError(
                "cannot finalize while source intake is not retrievable: " + ", ".join(missing_intakes)
            )

        state.update({
            "status": "selected",
            "decision": args.decision,
            "next_experiment": args.next_experiment,
            "evidence_gate": "pending",
            "finalized_at": _utc_now(),
        })
        _write(path, state)
        lint = _run([sys.executable, "tools/finding_lint.py", str(path), "--include-untracked"], root)
        if lint["returncode"] != 0:
            _atomic_text_write(path, original)
            raise EscalationError(f"existing finding evidence gates blocked finalization:\n{lint['output']}")
        state["evidence_gate"] = "passed"
        state["evidence_gate_record"] = lint
        _write(path, state)
        verify = _run([sys.executable, "tools/finding_lint.py", str(path), "--include-untracked"], root)
        if verify["returncode"] != 0:
            _atomic_text_write(path, original)
            raise EscalationError(f"final artifact failed evidence-gate verification:\n{verify['output']}")
    return path


def inspect(args: argparse.Namespace, _root: Path) -> None:
    state = _load(Path(args.gate).resolve())
    json.dump(state, sys.stdout, indent=2, sort_keys=True)
    print()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("start", help="stop at a wall, retrieve priors, and create a research gate")
    create.add_argument("--slug", required=True)
    create.add_argument("--title")
    create.add_argument("--blocked-experiment", required=True)
    create.add_argument("--wall-reason", required=True)
    create.add_argument("--failed-attempt", action="append", default=[], required=True)
    create.add_argument("--parameter-question", action="append", default=[], required=True)
    create.add_argument("--wiring-question", action="append", default=[], required=True)
    create.add_argument("--query", help="retrieval query; defaults to --blocked-experiment")
    create.add_argument("--output", help="output path; defaults under research/findings/")
    create.set_defaults(handler=start)

    retry = sub.add_parser("retry-retrieval", help="retry failed or stale local retrieval and clear the hard block")
    retry.add_argument("--gate", required=True)
    retry.set_defaults(handler=retry_retrieval)

    search = sub.add_parser("record-search", help="record an external search, including a no-result search")
    search.add_argument("--gate", required=True)
    search.add_argument("--questions", required=True, help="comma-separated question IDs")
    search.add_argument("--database", action="append", required=True)
    search.add_argument("--query", action="append", required=True, help="repeat for each query variant")
    search.add_argument("--outcome", required=True)
    search.add_argument("--url", action="append", default=[], help="repeat for database result/search URLs")
    search.add_argument("--date-from", help="earliest publication date searched, YYYY-MM-DD")
    search.add_argument("--date-to", help="latest publication date searched, YYYY-MM-DD")
    search.add_argument("--claim-absence", action="store_true", help="assert this complete protocol supports not-found")
    search.set_defaults(handler=record_search)

    source = sub.add_parser("record-source", help="record an external source and classify its evidence strength")
    source.add_argument("--gate", required=True)
    source.add_argument("--questions", required=True, help="comma-separated question IDs")
    source.add_argument("--kind", choices=sorted(SOURCE_QUALITY), required=True)
    source.add_argument("--citation", required=True)
    source.add_argument("--url", required=True)
    source.add_argument("--query", required=True)
    source.add_argument("--locator", required=True, help="page, figure, table, or methods subsection")
    source.add_argument("--evidence", required=True, help="precise parameter/wiring evidence, not a paper summary")
    source.add_argument(
        "--license-status",
        choices=sorted(source_intake.LICENSE_STATUSES),
        default="metadata-only",
        help="controls whether a local source copy may be archived",
    )
    source.add_argument("--local-file", help="licensed local copy to archive in the source catalog")
    source.set_defaults(handler=record_source)

    packet = sub.add_parser(
        "handoff-packet",
        help="validate and attach an external research packet without accepting its claims",
    )
    packet.add_argument("--gate", required=True)
    packet.add_argument("--packet", required=True, help="validated research-packet JSON path")
    packet.set_defaults(handler=handoff_packet)

    resolve = sub.add_parser("answer", help="resolve a question or record that the searched evidence is absent")
    resolve.add_argument("--gate", required=True)
    resolve.add_argument("--question", required=True)
    resolve.add_argument("--status", choices=("resolved", "not-found"), required=True)
    resolve.add_argument("--answer", required=True)
    resolve.add_argument("--references", required=True, help="comma-separated S*/X*/RP* evidence IDs")
    resolve.set_defaults(handler=answer)

    finish = sub.add_parser("finalize", help="select a bounded next experiment through existing evidence gates")
    finish.add_argument("--gate", required=True)
    finish.add_argument("--decision", required=True)
    finish.add_argument("--next-experiment", required=True)
    finish.set_defaults(handler=finalize)

    show = sub.add_parser("inspect", help="print the embedded machine-readable state and captured retrieval")
    show.add_argument("--gate", required=True)
    show.set_defaults(handler=inspect)
    return parser


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args, root)
    except EscalationError as exc:
        parser.error(str(exc))
    if isinstance(result, Path):
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
