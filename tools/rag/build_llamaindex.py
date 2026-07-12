"""Build a LlamaIndex vector index over the SAME findings corpus SOMA indexed, for a head-to-head RAG comparison.
Config mirrors SOMA as closely as possible: the SAME embedder (all-MiniLM-L6-v2); retrieval-only (no LLM).
Run with the isolated venv python (rag_compare_env)."""
import os, sys, time
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

CORPUS = r"E:\Documents\Projects\soma_bundles\_findings_md"   # the SAME staged findings SOMA indexed
PERSIST = r"E:\Documents\Projects\rag_compare\llamaindex_findings"

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = None                                          # retrieval only, no generation

t0 = time.time()
docs = SimpleDirectoryReader(input_dir=CORPUS, required_exts=[".md"], filename_as_id=True).load_data()
print(f"[build] loaded {len(docs)} docs ({time.time()-t0:.0f}s)", flush=True)

t1 = time.time()
index = VectorStoreIndex.from_documents(docs, show_progress=True)   # default SentenceSplitter chunking
index.storage_context.persist(persist_dir=PERSIST)
n_nodes = len(index.docstore.docs)
print(f"[build] indexed {n_nodes} nodes -> {PERSIST}  (embed {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
print("BUILD_DONE", flush=True)
