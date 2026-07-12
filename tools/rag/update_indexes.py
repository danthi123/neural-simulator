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
KB_MD = r"E:\Documents\Projects\soma_bundles\kb_md"          # evolving-corpus mirror for SOMA
SOMA_BUNDLE = r"E:\Documents\Projects\soma_bundles\sim_kb"
# evolving source types SOMA mirrors (Kandel excluded — static)
SOMA_TYPES = {"finding", "plan", "doc", "catalog"}


def evolving_files():
    """(source_type, path) for the EVOLVING prose (everything B.SOURCES lists EXCEPT the static kandel textbook)."""
    for stype, patterns in B.SOURCES:
        if stype == "kandel":
            continue
        for p in patterns:
            for f in sorted(glob.glob(p)):
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


def sync_kb_mirror():
    """Copy-if-newer the evolving prose into KB_MD/<source_type>/ (a flat mirror SOMA _ingest walks)."""
    n = 0
    for stype, f in evolving_files():
        if stype not in SOMA_TYPES:
            continue
        dst_dir = os.path.join(KB_MD, stype); os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(f))
        if (not os.path.exists(dst)) or os.path.getmtime(f) > os.path.getmtime(dst):
            shutil.copy2(f, dst); n += 1
    return n


def rebuild_soma():
    from pathlib import Path
    from soma._cli_commands.wiki_chat import _ingest
    synced = sync_kb_mirror()
    fc, cc = _ingest(Path(KB_MD), Path(SOMA_BUNDLE), include_pdf=False, verbose=False)
    return f"soma rebuilt ({fc} files / {cc} chunks; {synced} synced)"


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
