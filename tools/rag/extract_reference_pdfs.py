"""Create missing searchable text siblings for reference PDFs.

Existing text files are preserved. New text is published atomically only when
the PDF yields enough readable content to be useful to retrieval.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_llamaindex_full as B  # noqa: E402


MIN_YIELD = 500


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        from pypdf import PdfReader
    except ImportError:
        print(
            "pypdf missing; install it in the isolated RAG environment",
            file=sys.stderr,
        )
        return 2

    root = Path(B.CAT) / "textbooks"
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs under {root}; check SIM_CATALOG", file=sys.stderr)
        return 1

    extracted = skipped = failed = low_yield = 0
    for pdf in pdfs:
        target = pdf.with_suffix(".txt")
        if target.exists() and not args.force:
            skipped += 1
            continue
        try:
            reader = PdfReader(pdf)
            body = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if len(body.strip()) < MIN_YIELD:
                print(
                    f"LOW-YIELD ({len(body.strip())} chars): {pdf}",
                    file=sys.stderr,
                )
                low_yield += 1
                continue
            temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(target)
            extracted += 1
            if not args.quiet:
                print(
                    f"extracted {len(body):,} chars from {len(reader.pages)} "
                    f"pages: {pdf.name}"
                )
        except Exception as exc:
            print(f"FAILED {pdf}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1

    if not args.quiet or extracted or failed or low_yield:
        print(
            f"extracted={extracted} skipped={skipped} "
            f"failed={failed} low_yield={low_yield}"
        )
    return 1 if failed or low_yield else 0


if __name__ == "__main__":
    raise SystemExit(main())
