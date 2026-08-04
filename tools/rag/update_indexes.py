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
import os, sys, time, json, hashlib, shutil, glob, argparse, subprocess
from pathlib import Path

# RAG refreshes run as a background maintenance lane. Keep them off the
# experiment GPU unless an operator explicitly opts in with CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
SOMA_BATCH_SIZE = 256
# evolving source types SOMA covers (Kandel excluded — static)
SOMA_TYPES = {"finding", "plan", "doc", "catalog"}
PROJECT_PROSE_PATHS = [
    ":(glob)research/findings/*.md",
    ":(glob)research/findings/**/*.md",
    ":(glob)docs/*.md",
    ":(glob)docs/plans/*.md",
    "CLAUDE.md",
    "ROADMAP.md",
    "README.md",
    "GAP_CLOSURE_MISSION.md",
]


def persist_candidate(index):
    """Write a complete candidate index beside the live index for validation."""
    candidate_root = os.path.join(RAG, f".candidate-{os.getpid()}")
    staging = os.path.join(candidate_root, "llamaindex_full")
    shutil.rmtree(candidate_root, ignore_errors=True)
    os.makedirs(candidate_root)
    index.storage_context.persist(persist_dir=staging)
    return candidate_root, staging


def publish_candidate(staging):
    """Atomically promote a validated candidate while retaining rollback safety."""
    previous = f"{PERSIST}.previous-{os.getpid()}"
    moved_previous = False
    try:
        if os.path.isdir(PERSIST):
            os.replace(PERSIST, previous)
            moved_previous = True
        os.replace(staging, PERSIST)
        if moved_previous:
            shutil.rmtree(previous, ignore_errors=True)
    except Exception:
        if moved_previous and not os.path.exists(PERSIST) and os.path.isdir(previous):
            os.replace(previous, PERSIST)
        raise
    finally:
        shutil.rmtree(os.path.dirname(staging), ignore_errors=True)


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


def project_prose_dirty():
    """True when indexing would expose project claims that are not committed."""
    commands = [
        ["git", "diff", "--quiet", "--", *PROJECT_PROSE_PATHS],
        ["git", "diff", "--cached", "--quiet", "--", *PROJECT_PROSE_PATHS],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=B.SIM)
        if result.returncode == 1:
            return True
        if result.returncode != 0:
            raise RuntimeError("could not verify committed RAG project sources")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *PROJECT_PROSE_PATHS],
        cwd=B.SIM,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(untracked.stdout.strip())


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
        candidate_root, candidate = persist_candidate(idx)
        return f"llamaindex candidate REBUILT ({len(idx.docstore.docs)} nodes)", candidate_root, candidate
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
    candidate_root, candidate = persist_candidate(idx)
    return (f"llamaindex candidate refreshed ({sum(bool(x) for x in res)}/{len(res)} new-or-changed, "
            f"{ndel} deleted)", candidate_root, candidate)


def _file_records(chunk_markdown, load_text, f):
    """Return SOMA text/metadata records for one source file."""
    text = load_text(Path(f))                                # _load_doc_text expects a Path (uses .suffix)
    if text is None:
        return []
    return [
        (
            ch.text,
            {"path": ch.path, "heading": getattr(ch, "heading", ""), "src": f},
        )
        for ch in chunk_markdown(text, path=os.path.basename(f))
    ]


def _store_file(mem, chunk_markdown, load_text, f):
    """Store one file in bounded batches and return its node_ids.

    Batch writes avoid one filesystem lock/WAL fsync per chunk. Keeping a
    bounded batch size avoids holding all corpus embeddings in one temporary
    encode allocation during a rebuild.
    """
    records = _file_records(chunk_markdown, load_text, f)
    ids = []
    for start in range(0, len(records), SOMA_BATCH_SIZE):
        batch = records[start:start + SOMA_BATCH_SIZE]
        ids.extend(
            mem.store_batch(
                [text for text, _ in batch],
                metadatas=[metadata for _, metadata in batch],
            )
        )
    return ids


def _load_soma_manifest():
    if not os.path.exists(SOMA_MANIFEST):
        return {}
    try:
        with open(SOMA_MANIFEST, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _soma_manifest_needs_rebuild(manifest):
    """Reject legacy absolute-path manifests before incremental work.

    The first SOMA index was produced on another machine and used absolute
    Windows paths. A stable document key plus the recorded source path is the
    current schema; anything else is rebuilt into a clean candidate bundle.
    """
    if not isinstance(manifest, dict) or not manifest:
        return True
    for key, record in manifest.items():
        if not isinstance(key, str) or key.split(":", 1)[0] not in {"sim", "catalog"}:
            return True
        if not isinstance(record, dict):
            return True
        if not isinstance(record.get("path"), str) or not os.path.isabs(record["path"]):
            return True
        if not isinstance(record.get("mtime"), int) or not isinstance(record.get("ids"), list):
            return True
    return False


def _soma_corpus():
    """Return stable SOMA document keys mapped to current source metadata."""
    current = {}
    for stype, f in evolving_files():
        if stype not in SOMA_TYPES or not os.path.exists(f):
            continue
        current[B.document_id(stype, f)] = {
            "path": f,
            "mtime": int(os.stat(f).st_mtime),
        }
    return current


def _write_json_atomic(path, value):
    """Publish a JSON sidecar only after its complete contents are written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _publish_soma_candidate(candidate):
    """Atomically replace the live SOMA bundle with a complete candidate."""
    previous = f"{SOMA_BUNDLE}.previous-{os.getpid()}"
    moved_previous = False
    try:
        if os.path.isdir(SOMA_BUNDLE):
            os.replace(SOMA_BUNDLE, previous)
            moved_previous = True
        os.replace(candidate, SOMA_BUNDLE)
        if moved_previous:
            shutil.rmtree(previous, ignore_errors=True)
    except Exception:
        if moved_previous and not os.path.exists(SOMA_BUNDLE) and os.path.isdir(previous):
            os.replace(previous, SOMA_BUNDLE)
        raise
    finally:
        shutil.rmtree(candidate, ignore_errors=True)


def refresh_soma(rebuild=False):
    """Keep SOMA current with a worktree-safe manifest and clean rebuild path.

    Incremental updates forget only changed/deleted files. A legacy manifest
    (including the old absolute Windows-path format) triggers a fresh
    in-memory build, which is saved to a candidate directory and atomically
    promoted so stale WAL sidecars cannot be replayed into the new index.
    """
    from soma.memory.api import MemoryLayer
    from soma._cli_commands.wiki_chat import chunk_markdown, _load_doc_text
    cur = _soma_corpus()
    prev = _load_soma_manifest()
    full = rebuild or (not os.path.isdir(SOMA_BUNDLE)) or _soma_manifest_needs_rebuild(prev)

    if full:
        mem = MemoryLayer.with_sbert(device="cpu")
        man, cc = {}, 0
        for key, source in cur.items():
            ids = _store_file(mem, chunk_markdown, _load_doc_text, source["path"])
            man[key] = {"path": source["path"], "mtime": source["mtime"], "ids": ids}
            cc += len(ids)
        candidate = f"{SOMA_BUNDLE}.candidate-{os.getpid()}"
        shutil.rmtree(candidate, ignore_errors=True)
        mem.save(candidate)
        _publish_soma_candidate(candidate)
        _write_json_atomic(SOMA_MANIFEST, man)
        return f"soma FULL rebuild ({len(cur)} files / {cc} chunks)"

    new     = [key for key in cur if key not in prev]
    changed = [key for key in cur if key in prev and cur[key]["mtime"] != prev[key]["mtime"]]
    deleted = [key for key in prev if key not in cur]
    if not (new or changed or deleted):
        return "soma up to date (no changes)"

    mem = MemoryLayer.load_with_sbert(SOMA_BUNDLE, device="cpu")
    man = {key: dict(value) for key, value in prev.items()}
    for key, source in cur.items():
        if key in man:
            man[key]["path"] = source["path"]
            man[key]["mtime"] = source["mtime"]
    n_forgot = 0
    for key in changed + deleted:                             # drop old chunks by their recorded node_ids
        for nid in prev[key].get("ids", []):
            if mem.forget(nid):
                n_forgot += 1
        man.pop(key, None)
    cc = 0
    for key in new + changed:                                 # (re)store new/edited files
        source = cur[key]
        ids = _store_file(mem, chunk_markdown, _load_doc_text, source["path"])
        man[key] = {"path": source["path"], "mtime": source["mtime"], "ids": ids}
        cc += len(ids)
    mem.save(SOMA_BUNDLE)
    _write_json_atomic(SOMA_MANIFEST, man)
    return (f"soma incremental (+{len(new)} new, ~{len(changed)} changed, -{len(deleted)} deleted; "
            f"{cc} chunks stored, {n_forgot} forgotten)")


def check_retrieval_quality(candidate_root):
    """Run the labeled quality floor against a candidate before publication."""
    evaluator = os.path.join(B.SIM, "tools", "rag", "rag_eval.py")
    env = dict(os.environ)
    env["SIM_RAG_ROOT"] = candidate_root
    min_mrr = max(0.90, float(os.environ.get("SIM_RAG_MIN_MRR", "0.90")))
    result = subprocess.run(
        [sys.executable, evaluator, "--no-write", "--min-mrr", str(min_mrr)],
        cwd=B.SIM,
        env=env,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"RAG retrieval quality check failed with status {result.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="full LlamaIndex rebuild (needed once per ID schema)")
    ap.add_argument("--force", action="store_true", help="update even if the manifest is unchanged")
    args = ap.parse_args()

    if project_prose_dirty():
        print(
            "[update-indexes] BLOCKED: indexed project prose has uncommitted changes; "
            "commit or restore it before refreshing.",
            flush=True,
        )
        return 2

    os.makedirs(RAG, exist_ok=True)
    cur = manifest_hash()
    prev = None
    if os.path.exists(MANIFEST):
        try:
            prev = json.load(open(MANIFEST)).get("hash")
        except Exception:
            prev = None
    if (not args.force) and (not args.rebuild) and prev == cur:
        print("[update-indexes] no evolving-doc change since last update; skip.", flush=True); return 0

    if not acquire_lock():
        print("[update-indexes] another update is running; skip (it will pick up these changes).", flush=True); return 0
    try:
        t0 = time.time()
        while True:
            indexed_hash = manifest_hash()
            message, candidate_root, candidate = refresh_llamaindex(rebuild=args.rebuild)
            print(message, flush=True)
            try:
                check_retrieval_quality(candidate_root)
                publish_candidate(candidate)
            except Exception:
                shutil.rmtree(candidate_root, ignore_errors=True)
                raise
            schema_tmp = f"{SCHEMA}.tmp-{os.getpid()}"
            with open(schema_tmp, "w", encoding="utf-8") as handle:
                json.dump(
                    {"document_id_schema": DOCUMENT_ID_SCHEMA, "source_repo": B.SIM},
                    handle,
                    indent=2,
                )
            os.replace(schema_tmp, SCHEMA)
            print("llamaindex candidate PASSED and published", flush=True)
            try:
                print(refresh_soma(rebuild=args.rebuild), flush=True)
            except Exception as e:
                print(f"[update-indexes] SOMA rebuild failed (LlamaIndex still updated): {e}", flush=True)
            latest_hash = manifest_hash()
            if latest_hash == indexed_hash:
                json.dump({"hash": indexed_hash, "at": int(time.time())}, open(MANIFEST, "w"))
                break
            print("[update-indexes] corpus changed during refresh; repeating before marking current.", flush=True)
        print(f"[update-indexes] done in {time.time()-t0:.0f}s.", flush=True)
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
