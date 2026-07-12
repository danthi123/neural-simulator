"""Keep BOTH RAG indexes (LlamaIndex + SOMA) fresh as new docs are added. Idempotent, lock-guarded, manifest-gated:
runs only when the evolving prose actually changed since the last successful update, and never two at once (bursts of
commits collapse to one update). Wired to fire automatically via the git post-commit hook (tools/git-hooks/post-commit).

- LlamaIndex (rag_compare/llamaindex_full): INCREMENTAL via refresh_ref_docs -> only new/changed docs are re-embedded
  (the static 8.7MB Kandel is skipped when unchanged), so a typical update is seconds. Needs the index built with
  path-based ids (build_llamaindex_full.py, id_=path) -- if the persisted index predates that, pass --rebuild once.
- SOMA (soma_bundles/sim_kb): full rebuild (soma _ingest is fresh-MemoryLayer only) over the EVOLVING corpus mirrored
  into soma_bundles/kb_md/ (findings/plans/docs/catalog; Kandel excluded -- static biology lives in LlamaIndex --corpus
  kandel and needs no re-chunking each commit).

Run with the rag_compare_env python (has BOTH llama-index and soma):
  E:/Documents/Projects/rag_compare_env/Scripts/python.exe tools/rag/update_indexes.py [--rebuild] [--force]
"""
import os, sys, time, json, hashlib, shutil, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_llamaindex_full as B   # SOURCES + load_docs (+ PERSIST)

RAG = r"E:\Documents\Projects\rag_compare"
PERSIST = B.PERSIST
LOCK = os.path.join(RAG, ".update.lock")
MANIFEST = os.path.join(RAG, ".rag_manifest.json")
SOMA_BUNDLE = r"E:\Documents\Projects\soma_bundles\sim_kb"
SOMA_MANIFEST = r"E:\Documents\Projects\soma_bundles\.soma_kb_manifest.json"
# evolving source types SOMA covers (Kandel excluded — static)
SOMA_TYPES = {"finding", "plan", "doc", "catalog"}


def evolving_files():
    """(source_type, path) for the EVOLVING prose (everything B.SOURCES lists EXCEPT the static kandel textbook and the
    excluded running scratchpads — AUTONOMOUS_STATE etc., which would else force a rebuild on nearly every commit)."""
    for stype, patterns in B.SOURCES:
        if stype == "kandel":
            continue
        for p in patterns:
            for f in sorted(glob.glob(p)):
                if os.path.basename(f) in B.EXCLUDE_BASENAMES:
                    continue
                yield stype, f


def manifest_hash():
    h = hashlib.sha256()
    for stype, f in evolving_files():
        try:
            st = os.stat(f)
            h.update(f.encode("utf-8", "replace")); h.update(str(int(st.st_mtime)).encode()); h.update(str(st.st_size).encode())
        except OSError:
            pass
    return h.hexdigest()


def acquire_lock():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        try:                                                 # stale-lock reclaim (>30 min = a crashed run)
            if time.time() - os.path.getmtime(LOCK) > 1800:
                os.remove(LOCK); return acquire_lock()
        except OSError:
            pass
        return False


def refresh_llamaindex(rebuild=False):
    from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    docs = B.load_docs()
    if rebuild or not os.path.isdir(PERSIST):
        idx = VectorStoreIndex.from_documents(docs, show_progress=False)
        idx.storage_context.persist(persist_dir=PERSIST)
        return f"llamaindex REBUILT ({len(idx.docstore.docs)} nodes)"
    idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST))
    res = idx.refresh_ref_docs(docs)                         # insert new + update changed (by hash); skip unchanged
    idx.storage_context.persist(persist_dir=PERSIST)
    return f"llamaindex refreshed ({sum(bool(x) for x in res)}/{len(res)} docs new-or-changed)"


def _soma_evolving():
    """{path: int(mtime)} for the SOMA-covered evolving prose (source read directly; no mirror)."""
    return {f: int(os.stat(f).st_mtime) for stype, f in evolving_files() if stype in SOMA_TYPES and os.path.exists(f)}


def _store_file(mem, chunk_markdown, load_text, f):
    from pathlib import Path
    text = load_text(Path(f))                                # _load_doc_text expects a Path (uses .suffix)
    if text is None:
        return 0
    n = 0
    for ch in chunk_markdown(text, path=os.path.basename(f)):
        mem.store(ch.text, metadata={"path": ch.path, "heading": getattr(ch, "heading", ""), "src": f})
        n += 1
    return n


def rebuild_soma(rebuild=False):
    """INCREMENTAL when only NEW files were added (load bundle -> store the new files -> save = fast, no re-embed of the
    existing corpus). FULL rebuild (fresh MemoryLayer) when a file was EDITED or DELETED, or the bundle is missing, or
    --rebuild (correctness: SOMA has no per-doc delete-by-path, so an edit needs a clean rebuild)."""
    from soma.memory.api import MemoryLayer
    from soma._cli_commands.wiki_chat import chunk_markdown, _load_doc_text
    cur = _soma_evolving()
    prev = {}
    if os.path.exists(SOMA_MANIFEST):
        try:
            prev = json.load(open(SOMA_MANIFEST))
        except Exception:
            prev = {}
    changed = [f for f in cur if f in prev and cur[f] != prev[f]]
    deleted = [f for f in prev if f not in cur]
    new = [f for f in cur if f not in prev]
    full = rebuild or (not os.path.isdir(SOMA_BUNDLE)) or bool(changed) or bool(deleted)

    if full:
        mem = MemoryLayer.with_sbert()
        cc = sum(_store_file(mem, chunk_markdown, _load_doc_text, f) for f in cur)
        mem.save(SOMA_BUNDLE)
        json.dump(cur, open(SOMA_MANIFEST, "w"))
        return f"soma FULL rebuild ({len(cur)} files / {cc} chunks; {len(changed)} changed, {len(deleted)} deleted)"
    if not new:
        return "soma up to date (no new files)"
    mem = MemoryLayer.load_with_sbert(SOMA_BUNDLE)
    cc = sum(_store_file(mem, chunk_markdown, _load_doc_text, f) for f in new)
    mem.save(SOMA_BUNDLE)
    json.dump(cur, open(SOMA_MANIFEST, "w"))
    return f"soma incremental (+{len(new)} new files / {cc} chunks appended)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="full LlamaIndex rebuild (needed once to install path-ids)")
    ap.add_argument("--force", action="store_true", help="update even if the manifest is unchanged")
    args = ap.parse_args()

    os.makedirs(RAG, exist_ok=True)
    cur = manifest_hash()
    prev = None
    if os.path.exists(MANIFEST):
        try:
            prev = json.load(open(MANIFEST)).get("hash")
        except Exception:
            prev = None
    if (not args.force) and (not args.rebuild) and prev == cur:
        print("[update-indexes] no evolving-doc change since last update; skip.", flush=True); return

    if not acquire_lock():
        print("[update-indexes] another update is running; skip (it will pick up these changes).", flush=True); return
    try:
        t0 = time.time()
        print(refresh_llamaindex(rebuild=args.rebuild), flush=True)
        try:
            print(rebuild_soma(), flush=True)
        except Exception as e:
            print(f"[update-indexes] SOMA rebuild failed (LlamaIndex still updated): {e}", flush=True)
        json.dump({"hash": manifest_hash(), "at": int(time.time())}, open(MANIFEST, "w"))
        print(f"[update-indexes] done in {time.time()-t0:.0f}s.", flush=True)
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
