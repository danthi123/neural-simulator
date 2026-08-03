"""Keep BOTH RAG indexes (LlamaIndex + SOMA) fresh as new docs are added. Idempotent, lock-guarded, manifest-gated:
runs only when the evolving prose actually changed since the last successful update, and never two at once (bursts of
commits collapse to one update). Wired to fire automatically via the git post-commit hook (tools/githooks/post-commit).

- LlamaIndex (rag_index/llamaindex_full): INCREMENTAL via refresh_ref_docs (new/changed re-embedded) + explicit
  delete_ref_doc for vanished docs; the static 8.7MB Kandel is skipped when unchanged, so a typical update is seconds.
  Uses repository-relative ids so one canonical index can be refreshed from the main linked worktree. If the
  persisted index predates that schema, pass --rebuild once.
- SOMA (soma_bundles/sim_kb): INCREMENTAL for new/edited/deleted files -- an edited/deleted file's old chunks are
  forgotten by their manifest-recorded node_ids and (for an edit) the new chunks stored, so a single edit no longer
  re-embeds the whole corpus. Covers the EVOLVING prose (findings/plans/docs/catalog; Kandel excluded -- static biology
  lives in LlamaIndex --corpus kandel). A FULL rebuild happens only on --rebuild, a missing bundle, or a one-time
  migration of a pre-node-id manifest.

Run with the canonical checkout's RAG interpreter:
  CANONICAL=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  "$CANONICAL/.venv-rag/bin/python" tools/rag/update_indexes.py [--rebuild] [--force]
"""
import os, sys, time, json, hashlib, shutil, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_llamaindex_full as B   # SOURCES + load_docs (+ PERSIST)

RAG = B.RAG_ROOT                      # portable: see build_llamaindex_full ($SIM_RAG_ROOT / <parent-of-repo>/rag_index)
PERSIST = B.PERSIST
LOCK = os.path.join(RAG, ".update.lock")
MANIFEST = os.path.join(RAG, ".rag_manifest.json")
SCHEMA = os.path.join(RAG, ".rag_schema.json")
DOCUMENT_ID_SCHEMA = "repo-relative-v1"
_SOMA_ROOT = os.environ.get("SIM_SOMA_ROOT") or os.path.join(os.path.dirname(B.SIM), "soma_bundles")
SOMA_BUNDLE = os.path.join(_SOMA_ROOT, "sim_kb")
SOMA_MANIFEST = os.path.join(_SOMA_ROOT, ".soma_kb_manifest.json")
# evolving source types SOMA covers (Kandel excluded — static)
SOMA_TYPES = {"finding", "plan", "doc", "catalog"}


def persist_atomically(index):
    """Publish a complete index directory without exposing half-written JSON."""
    staging = f"{PERSIST}.staging-{os.getpid()}"
    previous = f"{PERSIST}.previous-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    index.storage_context.persist(persist_dir=staging)
    moved_previous = False
    try:
        if os.path.isdir(PERSIST):
            os.replace(PERSIST, previous)
            moved_previous = True
        os.replace(staging, PERSIST)
        if moved_previous:
            shutil.rmtree(previous)
    except Exception:
        if moved_previous and not os.path.exists(PERSIST) and os.path.isdir(previous):
            os.replace(previous, PERSIST)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def evolving_files():
    """(source_type, path) for the EVOLVING prose (everything B.SOURCES lists EXCEPT the static kandel textbook and the
    excluded running scratchpads — AUTONOMOUS_STATE etc., which would else force a rebuild on nearly every commit)."""
    for stype, patterns in B.SOURCES:
        if stype == "kandel":
            continue
        for p in patterns:
            for f in sorted(glob.glob(p, recursive=True)):
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
        persist_atomically(idx)
        json.dump(
            {"document_id_schema": DOCUMENT_ID_SCHEMA, "source_repo": B.SIM},
            open(SCHEMA, "w"),
            indent=2,
        )
        return f"llamaindex REBUILT ({len(idx.docstore.docs)} nodes)"
    try:
        schema = json.load(open(SCHEMA)).get("document_id_schema")
    except Exception:
        schema = None
    if schema != DOCUMENT_ID_SCHEMA:
        raise RuntimeError(
            "RAG index uses legacy checkout-specific document IDs; run "
            "tools/rag/update_indexes.py --rebuild once before incremental refresh"
        )
    idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST))
    # refresh_ref_docs inserts NEW + updates CHANGED (by content hash) but does NOT remove docs that vanished, so a
    # deleted findings/doc would linger as stale hits -> explicitly delete the ref docs no longer in the corpus.
    cur_ids = {d.id_ for d in docs}
    ndel = 0
    for rid in list(idx.ref_doc_info.keys()):
        if rid not in cur_ids:
            try:
                idx.delete_ref_doc(rid, delete_from_docstore=True); ndel += 1
            except Exception:
                pass
    res = idx.refresh_ref_docs(docs)                         # insert new + update changed (by hash); skip unchanged
    persist_atomically(idx)
    return f"llamaindex refreshed ({sum(bool(x) for x in res)}/{len(res)} new-or-changed, {ndel} deleted)"


def _store_file(mem, chunk_markdown, load_text, f):
    """Store all chunks of file f into the memory layer; return the list of node_ids (empty if unreadable/empty).
    The ids are recorded in the manifest so a later EDIT/DELETE can forget exactly this file's chunks."""
    from pathlib import Path
    text = load_text(Path(f))                                # _load_doc_text expects a Path (uses .suffix)
    if text is None:
        return []
    ids = []
    for ch in chunk_markdown(text, path=os.path.basename(f)):
        ids.append(mem.store(ch.text, metadata={"path": ch.path, "heading": getattr(ch, "heading", ""), "src": f}))
    return ids


def _load_soma_manifest():
    if not os.path.exists(SOMA_MANIFEST):
        return {}
    try:
        return json.load(open(SOMA_MANIFEST))
    except Exception:
        return {}


def refresh_soma(rebuild=False):
    """Keep the SOMA bundle in sync with the evolving corpus, FULLY INCREMENTAL for new/edited/deleted files: a
    changed/deleted file's old chunks are forgotten by their recorded node_ids and (for an edit) the new chunks are
    stored -- so a single edit no longer forces a full re-embed of the whole corpus. Manifest schema is
    {path: {"mtime": int, "ids": [node_id, ...]}}. FULL rebuild (fresh MemoryLayer) only when: --rebuild, the bundle
    is missing, or the manifest predates the node-id schema (a one-time migration to record ids)."""
    from soma.memory.api import MemoryLayer
    from soma._cli_commands.wiki_chat import chunk_markdown, _load_doc_text
    cur = {f: int(os.stat(f).st_mtime) for stype, f in evolving_files() if stype in SOMA_TYPES and os.path.exists(f)}
    prev = _load_soma_manifest()
    old_schema = any(not isinstance(v, dict) for v in prev.values())   # pre-node-id manifest -> one-time migrate
    full = rebuild or (not os.path.isdir(SOMA_BUNDLE)) or old_schema or (not prev)

    if full:
        mem = MemoryLayer.with_sbert()
        man, cc = {}, 0
        for f in cur:
            ids = _store_file(mem, chunk_markdown, _load_doc_text, f)
            man[f] = {"mtime": cur[f], "ids": ids}; cc += len(ids)
        mem.save(SOMA_BUNDLE)
        json.dump(man, open(SOMA_MANIFEST, "w"))
        return f"soma FULL rebuild ({len(cur)} files / {cc} chunks)"

    new     = [f for f in cur if f not in prev]
    changed = [f for f in cur if f in prev and cur[f] != prev[f]["mtime"]]
    deleted = [f for f in prev if f not in cur]
    if not (new or changed or deleted):
        return "soma up to date (no changes)"

    mem = MemoryLayer.load_with_sbert(SOMA_BUNDLE)
    man = {f: dict(v) for f, v in prev.items()}
    n_forgot = 0
    for f in changed + deleted:                              # drop the file's old chunks by their recorded node_ids
        for nid in prev[f].get("ids", []):
            if mem.forget(nid):
                n_forgot += 1
        man.pop(f, None)
    cc = 0
    for f in new + changed:                                  # (re)store new/edited files, recording fresh node_ids
        ids = _store_file(mem, chunk_markdown, _load_doc_text, f)
        man[f] = {"mtime": cur[f], "ids": ids}; cc += len(ids)
    mem.save(SOMA_BUNDLE)
    json.dump(man, open(SOMA_MANIFEST, "w"))
    return (f"soma incremental (+{len(new)} new, ~{len(changed)} changed, -{len(deleted)} deleted; "
            f"{cc} chunks stored, {n_forgot} forgotten)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="full LlamaIndex rebuild (needed once per ID schema)")
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
            print(refresh_soma(rebuild=args.rebuild), flush=True)
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
