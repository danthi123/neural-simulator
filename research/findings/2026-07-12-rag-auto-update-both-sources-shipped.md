# RAG auto-update for new/edited/deleted docs — shipped, both sources (2026-07-12)

**Owner request:** "Can you ensure the RAG is automatically updated for new docs? Same for both RAG sources."

**Status: DONE + verified end-to-end for both sources across new / edit / delete.**

## What the RAG is for

The `(a-1)` deep-research-gate step ("check our own findings FIRST") queries two local indexes over the
project's prose knowledge base (findings / plans / docs / catalog / Kandel), so we stop re-deriving conclusions
the record already holds. Two indexes are kept in sync:

- **LlamaIndex** (`rag_compare/llamaindex_full`) — vector + BM25 fusion + rerank; the primary, queried by
  `tools/rag/rag_search.py`.
- **SOMA** (`soma_bundles/sim_kb`) — the owner's sbert memory layer, used as an imported library.

`AUTONOMOUS_STATE*.md` is **excluded** from the corpus (a huge running scratchpad whose content is duplicated
in the individual findings; indexing it would thrash the auto-update and dominate results — grep it directly).

## The automatic path

A tracked git **`post-commit` hook** (`tools/git-hooks/post-commit`, installed by the idempotent
`tools/rag/install_git_hook.sh`, re-installable per clone/worktree) fires whenever a commit touches
`research/findings/|docs/|CLAUDE.md|ROADMAP.md|README.md`. It launches a backgrounded, 45s-debounced,
lock+manifest-gated `tools/rag/update_indexes.py` — non-blocking (the commit returns immediately), and a burst
of commits collapses to one update (the lock + a sha256 manifest of the evolving files).

## Both indexes are fully incremental (new / edit / delete)

A single edit no longer forces a full re-embed. Typical update ~30s, not the ~15min baseline rebuild:

- **LlamaIndex**: `refresh_ref_docs` (only new/changed docs re-embed, keyed by content hash on path-based ids)
  **plus** an explicit `delete_ref_doc` for docs that vanished from the corpus — `refresh_ref_docs` only
  inserts/updates, so without this a deleted doc would linger as stale hits.
- **SOMA**: the manifest schema is `{path: {mtime, ids}}`, recording each file's node_ids at store time. An
  edited or deleted file's old chunks are forgotten by those ids (`mem.forget`) and, for an edit, the new
  chunks stored. A full rebuild happens only on `--rebuild`, a missing bundle, or a one-time migration of a
  pre-node-id manifest.

## Verification

- **Add → edit → delete cycle** (controlled, clean bundle): add → v1 searchable in both; edit → v1 forgotten
  from the bundle + v2 stored + LlamaIndex swapped (SOMA `~1 changed; 1 stored, 1 forgotten`); delete → purged
  from both (LlamaIndex `1 deleted`, SOMA `-1 deleted; 1 forgotten`). Each step ~30s.
- **SOMA library confirmed correct** (not modified — used as a library): node_ids are stable across
  save/reload, cross-process `forget`+save persists the removal, and a full-rebuild `save()` **overwrites**
  (no stale-content merge). An earlier "stale v1" observation was a bootstrap artifact of mixing an old-code
  append (pre-node-id manifest) with a delete-test job killed mid-rebuild — not reproducible from a clean state.
- **Hook fires on real doc commits** (confirmed from `_autoupdate.log`), backgrounded and non-blocking; the
  lock correctly serializes concurrent fires and the manifest gate makes an unchanged corpus a no-op.

## Repo hygiene

All of this lives in the **sim repo** (`tools/rag/`, `tools/git-hooks/`) and is committed + pushed to both sim
remotes (origin + gitea). The **SOMA repo is untouched** by this work (SOMA is an imported library); the only
SOMA-side change in the broader effort is the earlier CLI sbert-load fix on its own SOMA-repo branch.

## Open (separate, science-side)

The deep-credit GPU runner probe surfaced a real infra note: `_semantic_inheritance_onbridge_spiking_derisk`
is not cleanly GPU-runnable via `SIM_BACKEND=cupy` (a numpy-oracle / cupy-bridge seam) and its ~500-neuron
on-bridge net is launch-bound, so the GPU escalation the deep-credit verdicts demand needs a **batched
on-bridge forward**, not naive per-example cupy — a scoped science-infra rung, tracked in the frontier map.
