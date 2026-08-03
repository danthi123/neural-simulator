"""Build the BROADENED LlamaIndex over the project's whole PROSE knowledge base (not just findings), with source_type
metadata so the a-1 "check our own memory" gate can target the right corpus.

Corpora (prose only — code/tests are for Grep, deliberately excluded):
  finding : research/findings/*.md          -> "have we already CONCLUDED / tried X?"
  plan    : docs/plans/*.md                 -> "did we already DESIGN X?"
  doc     : docs/*.md + CLAUDE/ROADMAP/README -> project state / architecture
  catalog : sim-catalog/references/*.md      -> "is there a CATALOG ENTRY for X?" (feature-catalog, glossary, roadmap)
  kandel  : sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt -> "how does the BIOLOGY do X?"

Each Document gets metadata {source_type, source} so rag_search can filter by --corpus and display what each hit is.
Same embedder (all-MiniLM-L6-v2) + retrieval-only as the findings-only index. Run with the canonical RAG interpreter.
Persists to the canonical rag_index/llamaindex_full directory."""
import os, time, glob
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from .rag_paths import resolve_paths, stable_document_id
except ImportError:  # direct script execution
    from rag_paths import resolve_paths, stable_document_id

# Paths resolve from this file's location (repo root = two dirs up) with env overrides, so the tooling is portable:
#   $SIM_REPO      -> the sim repo            (default: two dirs above this file)
#   $SIM_CATALOG   -> sim-catalog/references  (default: <parent-of-repo>/sim-catalog/references)
#   $SIM_RAG_ROOT  -> the index root          (default: <parent-of-repo>/rag_index)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PATHS = resolve_paths(os.path.dirname(os.path.dirname(_HERE)))
SIM = str(_PATHS.repo)
CAT = str(_PATHS.catalog)
RAG_ROOT = str(_PATHS.rag_root)
PERSIST = os.path.join(RAG_ROOT, "llamaindex_full")

# (source_type, list of file globs / explicit files)
SOURCES = [
    # RECURSIVE (2026-07-31). A flat `*.md` matched 1845 files while `**/*.md` matches 1887: 42 findings —
    # 24 `_*_scoping.md`, 6 production-reviewer verdicts, 2 iteration plans — sat one directory down in
    # `research/findings/raw/` and were therefore ABSENT FROM THE RECORD'S OWN INDEX. A document invisible to
    # the corpus query is a document that gets re-derived, which is the failure mode that cost 94 GPU-hours.
    ("finding", [os.path.join(SIM, "research", "findings", "**", "*.md")]),
    ("plan",    [os.path.join(SIM, "docs", "plans", "*.md")]),
    ("doc",     [os.path.join(SIM, "docs", "*.md"),
                 os.path.join(SIM, "CLAUDE.md"), os.path.join(SIM, "ROADMAP.md"), os.path.join(SIM, "README.md"),
                 os.path.join(SIM, "GAP_CLOSURE_MISSION.md"),
                 os.path.join(SIM, "docs", "FAILURE_GATE_MATRIX.md")]),
    ("catalog", [os.path.join(CAT, "*.md")]),
    ("kandel",  [os.path.join(CAT, "textbooks", "kandel-pns-6e", "full-book.txt")]),
    # the specialty TEXTS/PAPERS/BOOKS the workflow already cites by name (Marr 1969, Albus 1971, Buzsaki Rhythms,
    # O'Keefe-Nadel, Schultz, Sutton-Barto, the Tepper/Bolam BG reviews). Text is extracted alongside each PDF as a
    # .txt sibling (tools/rag/extract_reference_pdfs.py). MUST come AFTER "kandel" so the dedupe in load_docs()
    # leaves full-book.txt as source_type=kandel rather than re-claiming it here (same path => same Document id_).
    ("paper",   [os.path.join(CAT, "textbooks", "*", "*.txt")]),
]


# Excluded from the corpus: huge, constantly-edited running scratchpads whose content is duplicated in the individual
# findings (indexing them causes big-doc dominance + a rebuild on nearly every commit; grep them directly for the latest).
EXCLUDE_BASENAMES = {"AUTONOMOUS_STATE.md", "AUTONOMOUS_STATE_ARCHIVE.md"}


def document_id(source_type, path):
    """Stable IDs let the canonical index refresh from any checkout of the same repo."""
    return stable_document_id(source_type, path, SIM, CAT)


def load_docs():
    docs = []
    seen = set()   # An earlier source type wins; duplicate matches would otherwise emit duplicate stable IDs.
    for stype, patterns in SOURCES:
        files = []
        for p in patterns:
            # recursive=True is REQUIRED for the finding source's `**` to mean anything: without it glob treats
            # `**` as a single `*` and the 42 nested findings stay invisible. It is a no-op for the patterns
            # that carry no `**`, so it is safe to apply to all of them.
            files.extend(sorted(glob.glob(p, recursive=True)))
        files = [f for f in files if os.path.basename(f) not in EXCLUDE_BASENAMES]
        files = [f for f in files if not (f in seen or seen.add(f))]
        n = 0
        for f in files:
            try:
                text = open(f, encoding="utf-8", errors="replace").read()
            except Exception as e:
                print(f"  [skip] {f}: {e}", flush=True); continue
            if not text.strip():
                continue
            docs.append(Document(text=text, id_=document_id(stype, f),
                                 metadata={"source_type": stype, "source": os.path.basename(f), "path": f},
                                 excluded_embed_metadata_keys=["path"], excluded_llm_metadata_keys=["path"]))
            n += 1
        print(f"  [{stype}] {n} docs", flush=True)
    return docs


def main():
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    # ~1024-token chunks (matches the findings-only index); big files (Kandel/AUTONOMOUS_STATE) split into many nodes.
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    t0 = time.time()
    docs = load_docs()
    print(f"[build] loaded {len(docs)} docs ({time.time()-t0:.0f}s)", flush=True)
    t1 = time.time()
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST)
    print(f"[build] indexed {len(index.docstore.docs)} nodes -> {PERSIST} "
          f"(embed {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    print("BUILD_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
