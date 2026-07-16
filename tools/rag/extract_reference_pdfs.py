"""Extract a .txt sibling for every reference PDF under sim-catalog/references/textbooks/, so the specialty
TEXTS/PAPERS/BOOKS enter the RAG corpus as source_type="paper" (build_llamaindex_full.SOURCES).

Why: the research gate cites these by name (Marr 1969, Albus 1971, Buzsaki "Rhythms of the Brain", O'Keefe-Nadel
"The Hippocampus as a Cognitive Map", Schultz, Sutton-Barto, the Tepper/Bolam BG reviews) and the skill's step (a)
says READ THE ORIGINAL SOURCE IN DEPTH -- but only Kandel's full-book.txt was ever indexed, so a-1 could not LOCATE
a passage in any of the others. Extraction makes them locatable; the discipline is still to open the cited PDF and
read the load-bearing passage (a rerank hit is a pointer, not a paraphrase).

Idempotent: a PDF whose .txt sibling already exists is skipped, so this is safe to re-run when new PDFs are added
(several already shipped with hand-made .txt siblings -- those are preserved, never overwritten).

Run with the isolated RAG venv (has pypdf):
    .venv-rag/bin/python tools/rag/extract_reference_pdfs.py [--force]

Then rebuild/refresh the index:  .venv-rag/bin/python tools/rag/update_indexes.py --rebuild
"""
import os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_llamaindex_full as B   # CAT (portable, env-resolved)

MIN_YIELD = 500   # chars; below this a PDF is almost certainly scanned images with no text layer -> flag it loudly


def main():
    force = "--force" in sys.argv
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf missing. Install into the RAG venv:  .venv-rag/bin/pip install pypdf")
        raise SystemExit(2)

    pdfs = sorted(glob.glob(os.path.join(B.CAT, "textbooks", "*", "*.pdf")))
    if not pdfs:
        print(f"no PDFs under {B.CAT}/textbooks/ -- is sim-catalog present? (see $SIM_CATALOG)")
        raise SystemExit(1)

    done = skipped = failed = low = 0
    for pdf in pdfs:
        txt = pdf[:-4] + ".txt"
        base = os.path.basename(pdf)
        if os.path.exists(txt) and not force:
            print(f"  skip (txt exists): {base}"); skipped += 1; continue
        try:
            reader = PdfReader(pdf)
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")           # one unparseable page must not lose the rest of the book
            body = "\n".join(parts)
            if len(body.strip()) < MIN_YIELD:
                print(f"  !! LOW-YIELD ({len(body.strip())} chars) -- likely scanned, no text layer: {base}")
                low += 1
            open(txt, "w", encoding="utf-8").write(body)
            print(f"  OK {len(body):>9,} chars  {len(reader.pages):>4} pages  {base}")
            done += 1
        except Exception as e:
            print(f"  FAIL {base}: {type(e).__name__}: {e}"); failed += 1

    print(f"\nextracted {done}, skipped {skipped}, failed {failed}, low-yield {low}")
    if low:
        print("LOW-YIELD files need OCR before they carry any retrievable content -- they are NOT searchable as-is.")


if __name__ == "__main__":
    main()
