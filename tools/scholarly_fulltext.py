#!/usr/bin/env python3
"""Safely retrieve scholarly full text and locate candidate parameter passages.

This module deliberately stops before evidence review.  Retrieved passages are
candidate locators only: they are not accepted claims, interpreted values, or
permission to bypass a publisher's access controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = "scholarly-fulltext-v1"
SUPPORTED_MIME = {
    "application/pdf": "pdf",
    "application/xhtml+xml": "html",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/html": "html",
    "text/plain": "plain",
}
CONTENT_SUFFIX = {"pdf": "pdf", "html": "html", "xml": "xml", "plain": "txt"}
RECEIPT_FIELDS = {
    "schema_version", "request_sha256", "source", "retrieval",
    "candidate_locators", "evidence_boundary", "sha256",
}
SOURCE_FIELDS = {"doi", "canonical_url", "title"}
RETRIEVAL_FIELDS = {
    "declared_url", "final_url", "mime_type", "document_kind",
    "byte_count", "content_sha256", "content_path",
}
LOCATOR_FIELDS = {
    "record_kind", "locator", "page", "line_start", "line_end",
    "matched_parameter_terms", "matched_unit_patterns", "passage",
    "claim_status", "review_status",
}
FORBIDDEN_MIME_PARTS = (
    "archive",
    "compressed",
    "executable",
    "java-archive",
    "msdownload",
    "shockwave",
    "x-7z",
    "x-bzip",
    "x-dosexec",
    "x-gzip",
    "x-rar",
    "x-tar",
    "zip",
)
FORBIDDEN_MAGIC = (
    b"MZ",
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"\x7fELF",
)


class ScholarlyFulltextError(RuntimeError):
    """Raised when retrieval or persisted evidence cannot be trusted."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str


Transport = Callable[[str, Mapping[str, str], float, int], HttpResponse]
Converter = Callable[[bytes, str, float], str]


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return value


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((str(value) for key, value in headers.items() if key.lower() == name.lower()), None)


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_bytes: int,
) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            body = response.read(max_bytes + 1)
            return HttpResponse(response.status, dict(response.headers.items()), body, final_url)
    except HTTPError as exc:
        final_url = exc.geturl() or url
        return HttpResponse(exc.code, dict(exc.headers.items()), exc.read(max_bytes + 1), final_url)
    except (URLError, TimeoutError, OSError) as exc:
        raise ScholarlyFulltextError(f"request failed for {url}: {exc}") from exc


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ScholarlyFulltextError("full text could not be decoded")


def default_converter(content: bytes, kind: str, timeout: float) -> str:
    """Convert supported source bytes to searchable text.

    PDF conversion uses the installed ``pdftotext`` command without a shell.
    HTML/XML/plain conversion remains in-process and stdlib-only.
    """
    if kind == "pdf":
        try:
            completed = subprocess.run(
                ["pdftotext", "-layout", "-", "-"],
                input=content,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ScholarlyFulltextError(f"PDF conversion failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace")[:300].strip()
            raise ScholarlyFulltextError(f"PDF conversion failed with exit {completed.returncode}: {detail}")
        return completed.stdout.decode("utf-8", "replace")
    decoded = _decode_text(content)
    if kind in {"html", "xml"}:
        parser = _VisibleTextParser()
        parser.feed(decoded)
        parser.close()
        return "".join(parser.parts)
    if kind == "plain":
        return decoded
    raise ScholarlyFulltextError(f"unsupported conversion kind: {kind}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _receipt_sha256(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json({
        key: item for key, item in value.items() if key != "sha256"
    })).hexdigest()


def validate_receipt(receipt: Mapping[str, Any], *, store: Path | str | None = None) -> dict[str, Any]:
    """Validate a retrieval receipt and, when supplied, its stored content."""
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != SCHEMA_VERSION:
        raise ScholarlyFulltextError("full-text receipt has an unsupported schema")
    try:
        result = json.loads(json.dumps(receipt))
    except (TypeError, ValueError) as exc:
        raise ScholarlyFulltextError("full-text receipt must be JSON-serializable") from exc
    if set(result) != RECEIPT_FIELDS:
        raise ScholarlyFulltextError("full-text receipt has unexpected or missing fields")
    request_sha = result.get("request_sha256")
    if not isinstance(request_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", request_sha):
        raise ScholarlyFulltextError("full-text receipt has an invalid request digest")
    if result.get("sha256") != _receipt_sha256(result):
        raise ScholarlyFulltextError("full-text receipt digest does not match its contents")
    source = result.get("source")
    retrieval = result.get("retrieval")
    locators = result.get("candidate_locators")
    boundary = result.get("evidence_boundary")
    if not isinstance(source, dict) or not isinstance(retrieval, dict) or not isinstance(locators, list):
        raise ScholarlyFulltextError("full-text receipt is missing source, retrieval, or locator records")
    if set(source) != SOURCE_FIELDS:
        raise ScholarlyFulltextError("full-text receipt source has unexpected or missing fields")
    if not isinstance(source["title"], str) or not source["title"].strip():
        raise ScholarlyFulltextError("full-text receipt source has an invalid title")
    if source["doi"] is not None and (
        not isinstance(source["doi"], str) or not source["doi"].strip()
    ):
        raise ScholarlyFulltextError("full-text receipt source has an invalid DOI")
    if source["canonical_url"] is not None and _http_url(source["canonical_url"]) != source["canonical_url"]:
        raise ScholarlyFulltextError("full-text receipt source has an invalid canonical URL")
    if set(retrieval) != RETRIEVAL_FIELDS:
        raise ScholarlyFulltextError("full-text receipt retrieval has unexpected or missing fields")
    declared_url = retrieval["declared_url"]
    final_url = retrieval["final_url"]
    if _http_url(declared_url) != declared_url or _http_url(final_url) != final_url:
        raise ScholarlyFulltextError("full-text receipt retrieval has an invalid URL")
    mime = retrieval["mime_type"]
    kind = retrieval["document_kind"]
    if mime not in SUPPORTED_MIME or SUPPORTED_MIME[mime] != kind:
        raise ScholarlyFulltextError("full-text receipt retrieval has inconsistent MIME and document kind")
    byte_count = retrieval["byte_count"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ScholarlyFulltextError("full-text receipt retrieval has an invalid byte count")
    if boundary != {
        "content_retrieved": True,
        "locators_are_candidates_only": True,
        "accepted_claims": False,
        "interpreted_parameter_values": False,
        "review_status": "pending_review",
    }:
        raise ScholarlyFulltextError("full-text receipt violates the pending-review evidence boundary")
    for locator in locators:
        if not isinstance(locator, dict) or set(locator) != LOCATOR_FIELDS:
            raise ScholarlyFulltextError("full-text receipt contains a nonconforming candidate locator")
        if locator.get("record_kind") != "candidate_parameter_locator":
            raise ScholarlyFulltextError("full-text receipt contains an invalid candidate locator")
        if locator.get("claim_status") != "not_a_claim" or locator.get("review_status") != "pending_review":
            raise ScholarlyFulltextError("full-text locator violates the pending-review evidence boundary")
        if not isinstance(locator["passage"], str) or not locator["passage"].strip():
            raise ScholarlyFulltextError("full-text locator has an invalid passage")
        line_start = locator["line_start"]
        line_end = locator["line_end"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (line_start, line_end)):
            raise ScholarlyFulltextError("full-text locator has invalid line bounds")
        if line_start < 1 or line_end < line_start:
            raise ScholarlyFulltextError("full-text locator has invalid line bounds")
        page = locator["page"]
        if kind == "pdf":
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ScholarlyFulltextError("PDF full-text locator has an invalid page")
            expected_locator = f"page {page}, lines {line_start}-{line_end}"
        else:
            if page is not None:
                raise ScholarlyFulltextError("non-PDF full-text locator cannot specify a page")
            expected_locator = f"lines {line_start}-{line_end}"
        if locator["locator"] != expected_locator:
            raise ScholarlyFulltextError("full-text locator text does not match its coordinates")
        for field in ("matched_parameter_terms", "matched_unit_patterns"):
            if not isinstance(locator.get(field), list) or not locator[field] or not all(
                isinstance(item, str) and item.strip() for item in locator[field]
            ):
                raise ScholarlyFulltextError(f"full-text locator has invalid {field}")
            if len(locator[field]) != len(set(locator[field])):
                raise ScholarlyFulltextError(f"full-text locator has duplicate {field}")
    relative = retrieval.get("content_path")
    digest = retrieval.get("content_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ScholarlyFulltextError("full-text receipt has invalid content identity")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ScholarlyFulltextError("full-text receipt content path escapes store")
    expected_relative = f"content/{digest}.{CONTENT_SUFFIX[kind]}"
    if relative != expected_relative:
        raise ScholarlyFulltextError("full-text receipt content path is not content-addressed")
    if store is not None:
        root = Path(store).expanduser().resolve()
        content_path = (root / relative).resolve()
        try:
            content_path.relative_to(root)
        except ValueError as exc:
            raise ScholarlyFulltextError("full-text receipt content path escapes store") from exc
        if not content_path.is_file():
            raise ScholarlyFulltextError("full-text receipt content is missing or corrupt")
        content = content_path.read_bytes()
        if len(content) != byte_count or sha256(content).hexdigest() != digest:
            raise ScholarlyFulltextError("full-text receipt content is missing or corrupt")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_content_once(path: Path, content: bytes, expected_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ScholarlyFulltextError(f"content-addressed object is corrupt: {path.name}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".content.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256(Path(temporary).read_bytes()).hexdigest() != expected_sha:
            raise ScholarlyFulltextError("temporary content digest changed while writing")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if sha256(path.read_bytes()).hexdigest() != expected_sha:
                raise ScholarlyFulltextError(f"content-addressed object collision: {path.name}")
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _classify(content_type: str | None, body: bytes) -> tuple[str, str]:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if any(part in mime for part in FORBIDDEN_MIME_PARTS) or any(body.startswith(magic) for magic in FORBIDDEN_MAGIC):
        raise ScholarlyFulltextError("archive or executable content is forbidden")
    kind = SUPPORTED_MIME.get(mime)
    if kind is None:
        raise ScholarlyFulltextError(f"unsupported MIME type: {mime or '<missing>'}")
    if kind == "pdf" and not body.startswith(b"%PDF-"):
        raise ScholarlyFulltextError("PDF MIME does not match content signature")
    if kind != "pdf" and body.startswith(b"%PDF-"):
        raise ScholarlyFulltextError("content signature does not match declared MIME")
    return mime, kind


def _clean_terms(values: Sequence[str], *, label: str, max_items: int = 64) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence")
    cleaned = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or len(value) > 160:
            raise ValueError(f"{label} contains an invalid item")
        cleaned.append(value.strip())
    cleaned = sorted(set(cleaned), key=lambda item: (item.casefold(), item))
    if not cleaned or len(cleaned) > max_items:
        raise ValueError(f"{label} must contain between 1 and {max_items} items")
    return cleaned


def _compile_units(patterns: Sequence[str]) -> list[tuple[str, re.Pattern[str]]]:
    cleaned = _clean_terms(patterns, label="unit_patterns")
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for pattern in cleaned:
        try:
            compiled.append((pattern, re.compile(pattern, re.IGNORECASE)))
        except re.error as exc:
            raise ValueError(f"invalid unit pattern {pattern!r}: {exc}") from exc
    return compiled


def locate_parameter_passages(
    text: str,
    *,
    kind: str,
    parameter_terms: Sequence[str],
    unit_patterns: Sequence[str],
    context_lines: int = 2,
    max_passages: int = 25,
    max_passage_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Return bounded candidate passages containing a term and unit pattern."""
    if kind not in {"pdf", "html", "xml", "plain"}:
        raise ValueError("kind is unsupported")
    if not 0 <= context_lines <= 20 or not 1 <= max_passages <= 200 or not 100 <= max_passage_chars <= 20000:
        raise ValueError("invalid passage bounds")
    terms = _clean_terms(parameter_terms, label="parameter_terms")
    units = _compile_units(unit_patterns)
    term_needles = [(term, term.casefold()) for term in terms]
    pages = text.split("\f") if kind == "pdf" else [text]
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    for page_number, page in enumerate(pages, 1):
        lines = page.splitlines()
        for line_index, line in enumerate(lines):
            folded = line.casefold()
            matched_terms = [term for term, needle in term_needles if needle in folded]
            if not matched_terms:
                continue
            start = max(0, line_index - context_lines)
            end = min(len(lines), line_index + context_lines + 1)
            passage = "\n".join(lines[start:end]).strip()
            matched_units = [pattern for pattern, regex in units if regex.search(passage)]
            if not matched_units:
                continue
            identity = (page_number, start, end)
            if identity in seen:
                continue
            seen.add(identity)
            passage = passage[:max_passage_chars]
            locator = (
                f"page {page_number}, lines {start + 1}-{end}"
                if kind == "pdf"
                else f"lines {start + 1}-{end}"
            )
            results.append({
                "record_kind": "candidate_parameter_locator",
                "locator": locator,
                "page": page_number if kind == "pdf" else None,
                "line_start": start + 1,
                "line_end": end,
                "matched_parameter_terms": matched_terms,
                "matched_unit_patterns": matched_units,
                "passage": passage,
                "claim_status": "not_a_claim",
                "review_status": "pending_review",
            })
            if len(results) >= max_passages:
                return results
    return results


class ScholarlyFulltextRetriever:
    """Retrieve declared public full-text leads into a content-addressed store."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 20.0,
        converter_timeout: float = 30.0,
        max_bytes: int = 32 * 1024 * 1024,
        max_text_chars: int = 16 * 1024 * 1024,
        transport: Transport = _default_transport,
        converter: Converter = default_converter,
    ) -> None:
        if not isinstance(user_agent, str) or "/" not in user_agent or not user_agent.strip():
            raise ValueError("user_agent must identify the application")
        if timeout <= 0 or converter_timeout <= 0 or not 1 <= max_bytes <= 512 * 1024 * 1024:
            raise ValueError("invalid timeout or byte limit")
        if not 1 <= max_text_chars <= 256 * 1024 * 1024:
            raise ValueError("invalid searchable-text limit")
        self.user_agent = user_agent.strip()
        self.timeout = float(timeout)
        self.converter_timeout = float(converter_timeout)
        self.max_bytes = int(max_bytes)
        self.max_text_chars = int(max_text_chars)
        self.transport = transport
        self.converter = converter

    def retrieve(
        self,
        metadata_lead: Mapping[str, Any],
        *,
        store: Path | str,
        parameter_terms: Sequence[str],
        unit_patterns: Sequence[str],
        context_lines: int = 2,
        max_passages: int = 25,
        max_passage_chars: int = 4000,
    ) -> dict[str, Any]:
        if not isinstance(metadata_lead, Mapping) or metadata_lead.get("record_kind") != "metadata_candidate":
            raise ScholarlyFulltextError("input must be one metadata_candidate lead")
        raw_leads = metadata_lead.get("full_text_url_leads")
        if isinstance(raw_leads, (str, bytes)) or not isinstance(raw_leads, Sequence):
            raise ScholarlyFulltextError("metadata lead has no declared full_text_url_leads")
        urls = []
        for value in raw_leads:
            url = _http_url(value)
            if url is None:
                raise ScholarlyFulltextError("declared full-text lead is not a safe HTTP(S) URL")
            if url not in urls:
                urls.append(url)
        if not urls:
            raise ScholarlyFulltextError("metadata lead has no declared full_text_url_leads")

        terms = _clean_terms(parameter_terms, label="parameter_terms")
        patterns = _clean_terms(unit_patterns, label="unit_patterns")
        # Compile before network or filesystem side effects.
        _compile_units(patterns)
        request_basis = {
            "schema_version": SCHEMA_VERSION,
            "metadata_lead": metadata_lead,
            "parameter_terms": terms,
            "unit_patterns": patterns,
            "context_lines": context_lines,
            "max_passages": max_passages,
            "max_passage_chars": max_passage_chars,
        }
        request_sha = sha256(_canonical_json(request_basis)).hexdigest()
        root = Path(store).expanduser().resolve()
        receipt_path = root / "receipts" / f"{request_sha}.json"
        if receipt_path.exists():
            return self._resume(receipt_path, root, request_sha)

        failures: list[str] = []
        for url in urls:
            try:
                response = self.transport(
                    url,
                    {
                        "Accept": "application/pdf, application/xml, text/xml, text/html, text/plain",
                        "User-Agent": self.user_agent,
                    },
                    self.timeout,
                    self.max_bytes,
                )
                final_url = _http_url(response.final_url)
                if final_url is None:
                    raise ScholarlyFulltextError("redirect target is not a safe HTTP(S) URL")
                if not 200 <= response.status < 300:
                    raise ScholarlyFulltextError(f"HTTP {response.status}")
                if len(response.body) > self.max_bytes:
                    raise ScholarlyFulltextError(f"response exceeds max_bytes={self.max_bytes}")
                mime, kind = _classify(_header(response.headers, "Content-Type"), response.body)
                digest = sha256(response.body).hexdigest()
                suffix = CONTENT_SUFFIX[kind]
                relative_content = Path("content") / f"{digest}.{suffix}"
                content_path = root / relative_content
                _write_content_once(content_path, response.body, digest)
                searchable = self.converter(response.body, kind, self.converter_timeout)
                if not isinstance(searchable, str):
                    raise ScholarlyFulltextError("converter must return text")
                if len(searchable) > self.max_text_chars:
                    raise ScholarlyFulltextError("converted text exceeds configured character limit")
                passages = locate_parameter_passages(
                    searchable,
                    kind=kind,
                    parameter_terms=terms,
                    unit_patterns=patterns,
                    context_lines=context_lines,
                    max_passages=max_passages,
                    max_passage_chars=max_passage_chars,
                )
                receipt = {
                    "schema_version": SCHEMA_VERSION,
                    "request_sha256": request_sha,
                    "source": {
                        "doi": metadata_lead.get("doi"),
                        "canonical_url": metadata_lead.get("canonical_url"),
                        "title": metadata_lead.get("title"),
                    },
                    "retrieval": {
                        "declared_url": url,
                        "final_url": final_url,
                        "mime_type": mime,
                        "document_kind": kind,
                        "byte_count": len(response.body),
                        "content_sha256": digest,
                        "content_path": relative_content.as_posix(),
                    },
                    "candidate_locators": passages,
                    "evidence_boundary": {
                        "content_retrieved": True,
                        "locators_are_candidates_only": True,
                        "accepted_claims": False,
                        "interpreted_parameter_values": False,
                        "review_status": "pending_review",
                    },
                }
                receipt["sha256"] = _receipt_sha256(receipt)
                _atomic_json(receipt_path, receipt)
                return receipt
            except ScholarlyFulltextError as exc:
                failures.append(f"{url}: {exc}")
        raise ScholarlyFulltextError("all declared full-text leads failed: " + "; ".join(failures))

    @staticmethod
    def _resume(receipt_path: Path, root: Path, request_sha: str) -> dict[str, Any]:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScholarlyFulltextError(f"persisted receipt is unreadable: {receipt_path.name}") from exc
        if not isinstance(receipt, dict) or receipt.get("request_sha256") != request_sha:
            raise ScholarlyFulltextError("persisted receipt does not match request")
        return validate_receipt(receipt, store=root)
