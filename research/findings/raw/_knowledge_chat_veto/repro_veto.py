"""Light-path repro for Vikunja #142: 'what country is chelsea fc from' -> I don't know,
despite the shipped wikidata_core_15k LTM holding (chelsea_fc, country, united_kingom).

SIM_BACKEND=numpy, tiny-demo ChatBrain (the light path), NOT the GPU server.
"""
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-a407bb2153c57ee03"
sys.path.insert(0, REPO)

t0 = time.time()
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
print(f"[{time.time()-t0:.1f}s] imported brain_chat_tui", flush=True)

agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
print(f"[{time.time()-t0:.1f}s] built tiny-demo agent (onebrain composer)", flush=True)

# --- attach the shipped LTM exactly like server.py's _build_chat_brain does ---
LTM_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"
from research.runners.developed_brain_io import _inner_agent
from research.runners.tiered_fact_store import TieredFactStore
from research.runners.sharded_phasor_store import ShardedPhasorStore
import json
from pathlib import Path

manifest_path = Path(LTM_BUNDLE) / "manifest.json"
mani = json.loads(manifest_path.read_text(encoding="utf-8"))
assert "n_shards" in mani
ltm = ShardedPhasorStore.load(LTM_BUNDLE)
print(f"[{time.time()-t0:.1f}s] loaded ShardedPhasorStore LTM ({mani.get('n_shards')} shards)", flush=True)

inner = _inner_agent(agent)
inner.composer = TieredFactStore(inner.composer, ltm)
print(f"[{time.time()-t0:.1f}s] attached TieredFactStore(buffer, ltm)", flush=True)

chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
print(f"[{time.time()-t0:.1f}s] built ChatBrain", flush=True)

Q = "what country is chelsea fc from"

# === STAGE 0: direct store query (bypassing the live-turn layers entirely) ===
direct = chat.inner.composer.query_patient("chelsea_fc", "country")
print(f"\n=== STAGE 0: direct composer.query_patient('chelsea_fc','country') = {direct!r}")

# === STAGE 1: comprehension-only extraction (_extract_route) ===
route = chat._extract_route(Q)
print(f"=== STAGE 1: chat._extract_route({Q!r}) = {route!r}")

# === STAGE 2: full substrate recall (_substrate_recall) ===
sub = chat._substrate_recall(Q)
print(f"=== STAGE 2: chat._substrate_recall({Q!r}) = {sub!r}")

# === STAGE 3: full gate() (the exact call /api/brain-chat makes: chat.gate(msg)) ===
gate_result = chat.gate(Q)
print(f"=== STAGE 3: chat.gate({Q!r}) = {gate_result!r}")

# === Also probe the raw tokenization / grounding manually ===
_STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
         "to", "it", "that", "this", "they", "them", "of", "about"}
toks = [t.lower().strip(".,!?") for t in Q.split()]
content_raw = [t for t in toks if t and t not in _STOP]
print(f"\n=== raw tokens: {toks}")
print(f"=== content after _extract_route's LOCAL _STOP strip (note: 'from' NOT in this stopset): {content_raw}")

from research.runners.brain_chat_tui import _ground_content_words, _knowledge_grounding_enabled
known = chat.agents_set | chat.actions_set | chat.patients_set
print(f"=== knowledge_grounding_enabled = {_knowledge_grounding_enabled()}")
print(f"=== 'chelsea_fc' in known_words (agents_set) = {'chelsea_fc' in known}")
grounded = _ground_content_words(chat.inner.composer, content_raw, known_words=known)
print(f"=== content after _ground_content_words: {grounded}")

if len(grounded) >= 2:
    padded = [grounded[0], grounded[1], "__q__"]
    print(f"=== _neural_question_parse padded SVO = {padded}  (content[0]->pos0, content[1]->pos1)")
    nq = chat._neural_question_parse(grounded)
    print(f"=== _neural_question_parse(grounded) = {nq!r}  (agent, action) as comprehended")
