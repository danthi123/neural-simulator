"""Shared production retrieval path for RAG search and quality evaluation."""
from __future__ import annotations

import io
import os
import re
import time
from contextlib import redirect_stderr
try:
    from .rag_paths import RagPaths, choose_index
except ImportError:  # direct script execution
    from rag_paths import RagPaths, choose_index


def candidate_count(top_k: int) -> int:
    """Keep enough lexical and semantic candidates for the cross-encoder."""
    return max(top_k * 6, 30)


def node_source(node_with_score) -> str:
    node = node_with_score.node
    metadata = node.metadata or {}
    return metadata.get("source") or os.path.basename(str(node.ref_doc_id or ""))


def _source_intent(query: str, node_with_score) -> int:
    """Prefer a named source when the user explicitly includes its distinctive name."""
    query_terms = set(re.findall(r"[a-z]{4,}", query.lower()))
    source_terms = set(re.findall(r"[a-z]{4,}", node_source(node_with_score).lower()))
    generic = {"book", "full", "paper", "review", "finding", "gate", "smoke"}
    return len(query_terms & (source_terms - generic))


def node_locator(node_with_score, cache: dict[str, str] | None = None) -> str:
    """Return an actionable source path and best-effort line for a retrieved chunk."""
    metadata = node_with_score.node.metadata or {}
    path = metadata.get("path")
    if not path:
        return node_source(node_with_score)
    contents = cache if cache is not None else {}
    if path not in contents:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                contents[path] = handle.read()
        except OSError:
            contents[path] = ""
    source_text = contents[path]
    chunk = node_with_score.node.get_content() or ""
    offset = -1
    for probe in (line.strip() for line in chunk.splitlines()):
        if len(probe) >= 32:
            offset = source_text.find(probe)
            if offset >= 0:
                break
    if offset < 0 and len(chunk) >= 32:
        offset = source_text.find(chunk[: min(120, len(chunk))])
    if offset < 0:
        return path
    return f"{path}:{source_text.count(chr(10), 0, offset) + 1}"


class RagRetriever:
    """Reusable hybrid vector+BM25 retriever with corpus filtering before fusion."""

    def __init__(self, paths: RagPaths, *, corpus: str, top_k: int):
        self.corpus = corpus
        self.top_k = top_k
        self.persist = choose_index(paths, corpus)
        self._candidate_count = candidate_count(top_k)

        with redirect_stderr(io.StringIO()):
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.core.postprocessor import SentenceTransformerRerank
            from llama_index.core.retrievers import QueryFusionRetriever
            from llama_index.core.vector_stores import (
                FilterOperator,
                MetadataFilter,
                MetadataFilters,
            )
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.retrievers.bm25 import BM25Retriever

            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            Settings.llm = None
            self.index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(self.persist))
            )
            self._MetadataFilter = MetadataFilter
            self._MetadataFilters = MetadataFilters
            self._FilterOperator = FilterOperator
            self._BM25Retriever = BM25Retriever
            self._QueryFusionRetriever = QueryFusionRetriever
            self._fusion_by_corpus = {}
            self._reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=self._candidate_count,
            )

    @property
    def node_count(self) -> int:
        return len(self.index.docstore.docs)

    def _fusion_for(self, corpus: str):
        if corpus in self._fusion_by_corpus:
            return self._fusion_by_corpus[corpus]
        filters = None
        if corpus != "all":
            filters = self._MetadataFilters(
                filters=[
                    self._MetadataFilter(
                        key="source_type",
                        value=corpus,
                        operator=self._FilterOperator.EQ,
                    )
                ]
            )
        vector = self.index.as_retriever(
            similarity_top_k=self._candidate_count,
            filters=filters,
        )
        lexical = self._BM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=self._candidate_count,
            filters=filters,
        )
        fusion = self._QueryFusionRetriever(
            [vector, lexical],
            num_queries=1,
            mode="reciprocal_rerank",
            similarity_top_k=self._candidate_count,
            use_async=False,
        )
        self._fusion_by_corpus[corpus] = fusion
        return fusion

    def retrieve(self, query: str, *, corpus: str | None = None):
        selected_corpus = corpus or self.corpus
        with redirect_stderr(io.StringIO()):
            started = time.time()
            candidates = self._fusion_for(selected_corpus).retrieve(query)
            nodes = self._reranker.postprocess_nodes(
                candidates,
                query_str=query,
            )
            nodes.sort(
                key=lambda node: (
                    _source_intent(query, node),
                    float(node.score if node.score is not None else float("-inf")),
                ),
                reverse=True,
            )
            nodes = nodes[: self.top_k]
        return nodes, (time.time() - started) * 1000.0
