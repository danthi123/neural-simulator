"""Board #99/#112 open question 1 (of two named): does the from-scratch WKV/SSM spiking mouth's own
vocabulary cover TYPICAL CHAT TOPICS -- as opposed to its own TinyStories training-corpus text (already
measured in `research/findings/2026-08-31-wkv-mouth-rung4-vocab-coverage.md`) or Wikidata FACT triples
specifically (already measured in `research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md` Part
1/2)? Both prior measurements are self-referential to the checkpoint's own training domain or to the
knowledge STORE's fact structure; neither used a probe drawn from what a real user would actually TYPE in
open-ended chat. This runner closes that gap with a genuinely different probe corpus.

THE PROBE CORPUS (three groups, all REUSED VERBATIM from material already in this repo -- nothing invented
here, nothing downloaded):

  1. conversational_register -- the 14 VERBATIM human turns from `_conversation_turing_test_derisk.
     HUMAN_TURNS`, the project's own Turing-style stress-test battery (greeting, small talk, in-domain
     entry, emotion, forward-model curiosity, referential/episodic, out-of-domain fact, arithmetic,
     self/experiential, humor, abstract opinion, meta self-awareness, social closing) -- topic-agnostic
     EVERYDAY chat register, independent of any specific knowledge domain.
  2. everyday_real_world_topics -- `_open_ended_state_driven_generation_derisk._QWEN_KNOWN_STORE_UNKNOWN`,
     the project's own canonical list of famous, ordinary-conversation entities (paris, python,
     shakespeare, coffee, jupiter, beethoven, tokyo, everest, photosynthesis, gravity), each wrapped in a
     natural "Tell me about X." query using the SAME lead-in template `webapp.open_ended_chat._LEADINS`
     already accepts.
  3. wikidata_known_agents -- a seeded sample (seed=42, n=100) of real agents from the LIVE production
     knowledge store (`sim-data/knowledge_bundles/wikidata_core_15k/facts.json`, the store
     `webapp.open_ended_chat.build_index` actually serves), drawn with
     `_open_ended_bundle_moat_safety_soak._sample_known_topics` (reused unchanged -- the SAME sampling this
     project's own moat-safety soak already uses), each slug rendered as a natural noun phrase via
     `_wkv_fact_to_sentence_lexicon_lever.slug_to_np` and wrapped in the same "Tell me about X." template --
     representative of "ask the brain about a specific real-world topic it might actually know."

THE VOCAB + THE GATE. Vocab is read via `webapp.wkv_mouth_generator._get_readout(seed)` (pure `np.load`,
no RNG effect, no SimulationBridge -- memory-light) against the PRODUCTION-DEFAULT checkpoint
(`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz`, V=1000, word-level, closed vocabulary -- confirmed
by reading `WKVReadout.__init__`: `self.words = list(W["words"])`, a flat word list, no subword/BPE merge
table anywhere in the checkpoint). The production ACCEPT/REJECT gate is measured with the REAL, CURRENT
`webapp.wkv_mouth_generator.in_vocab_scope` (imported, never re-implemented -- the 2026-08-31 measurement's
own `_measure_wkv_ckpt_coverage.py` re-implemented an OLDER version of this gate missing the 2026-09-01
`_LEADIN_WORDS` fix; this runner does not repeat that drift).

MEMORY DISCIPLINE. Reads ONLY `facts.json` (~2 MB) from the shipped bundle + the ~1.4 MB V=1000 checkpoint
npz (+ the small persisted learned-head npz `_get_readout` now also loads by production default, board
#191) -- no `SimulationBridge`, no GPU, no torch. CPU/numpy only.

Run: `SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._wkv_mouth_chat_topic_vocab_coverage_derisk \\
    --out research/findings/raw/_wkv_mouth_chat_topic_vocab_coverage.json`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._conversation_turing_test_derisk import HUMAN_TURNS  # noqa: E402
from research.runners._open_ended_state_driven_generation_derisk import _QWEN_KNOWN_STORE_UNKNOWN  # noqa: E402
from research.runners._open_ended_bundle_moat_safety_soak import _sample_known_topics  # noqa: E402
from research.runners._wkv_fact_to_sentence_lexicon_lever import slug_to_np  # noqa: E402

FUNC = WKV._FUNCTION_WORDS
WORD_RE = WKV._WORD_RE


def _build_probe_corpus(seed: int, n_wikidata: int) -> dict:
    groups: dict[str, list[dict]] = {}

    # ── Group 1: conversational register (topic-agnostic everyday chat) ────────────────────────────────
    groups["conversational_register"] = [
        {"text": text, "subtype": kind} for text, kind in HUMAN_TURNS
    ]

    # ── Group 2: famous everyday real-world topics (canonical project list) ────────────────────────────
    groups["everyday_real_world_topics"] = [
        {"text": f"Tell me about {topic}.", "subtype": topic} for topic in _QWEN_KNOWN_STORE_UNKNOWN
    ]

    # ── Group 3: real production knowledge-store agents (seeded sample of the LIVE store) ──────────────
    agents = _sample_known_topics(n_wikidata, seed=seed)
    if agents:
        groups["wikidata_known_agents"] = [
            {"text": f"Tell me about {slug_to_np(a)}.", "subtype": a} for a in agents
        ]
    else:
        groups["wikidata_known_agents"] = []

    return groups


def _analyze_utterance(text: str, vocab: set, seed: int) -> dict:
    tokens = [w.lower() for w in WORD_RE.findall(text)]
    content_tokens = [t for t in tokens if t not in FUNC]
    oov_tokens = [t for t in tokens if t not in vocab]
    oov_content = [t for t in content_tokens if t not in vocab]
    return {
        "text": text,
        "n_tokens": len(tokens),
        "n_content_tokens": len(content_tokens),
        "n_oov_tokens": len(oov_tokens),
        "n_oov_content": len(oov_content),
        "oov_content_words": sorted(set(oov_content)),
        "fully_in_vocab_content": len(oov_content) == 0 and len(content_tokens) > 0,
        "gate_pass": bool(WKV.in_vocab_scope(text, seed=seed)),
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_utterances": 0, "note": "UNDEFINED (empty group) -- not a 0% score"}
    tot_tokens = sum(r["n_tokens"] for r in rows)
    tot_content = sum(r["n_content_tokens"] for r in rows)
    tot_oov_tokens = sum(r["n_oov_tokens"] for r in rows)
    tot_oov_content = sum(r["n_oov_content"] for r in rows)
    n_fully_in_vocab = sum(1 for r in rows if r["fully_in_vocab_content"])
    n_gate_pass = sum(1 for r in rows if r["gate_pass"])
    missing_counts: dict[str, int] = {}
    for r in rows:
        for w in r["oov_content_words"]:
            missing_counts[w] = missing_counts.get(w, 0) + 1
    top_missing = sorted(missing_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    return {
        "n_utterances": n,
        "token_oov_rate": round(tot_oov_tokens / tot_tokens, 5) if tot_tokens else None,
        "content_word_oov_rate": round(tot_oov_content / tot_content, 5) if tot_content else None,
        "n_content_tokens_total": tot_content,
        "n_oov_content_total": tot_oov_content,
        "fully_in_vocab_content_pct": round(100.0 * n_fully_in_vocab / n, 2),
        "fully_in_vocab_content_frac": f"{n_fully_in_vocab}/{n}",
        "gate_pass_pct": round(100.0 * n_gate_pass / n, 2),
        "gate_pass_frac": f"{n_gate_pass}/{n}",
        "top_missing_content_words": top_missing,
    }


def main(seed: int = 42, n_wikidata: int = 100, out: str | None = None) -> dict:
    out_data: dict = {"runner": "_wkv_mouth_chat_topic_vocab_coverage_derisk", "seed": seed}

    _, vocab, word_to_id = WKV._get_readout(seed)
    out_data["checkpoint_path"] = WKV._ckpt_path(seed)
    out_data["checkpoint_vocab_size"] = len(vocab)
    out_data["tokenization"] = "word-level, closed vocabulary (V=%d incl. <unk> sentinel), no subword/BPE fallback" % len(vocab)
    out_data["learned_head_enabled"] = WKV.learned_head_enabled()

    groups = _build_probe_corpus(seed, n_wikidata)
    out_data["groups"] = {}
    all_rows: list[dict] = []
    for gname, items in groups.items():
        rows = [dict(_analyze_utterance(it["text"], vocab, seed), subtype=it["subtype"]) for it in items]
        agg = _aggregate(rows)
        out_data["groups"][gname] = {"aggregate": agg, "rows": rows}
        all_rows.extend(rows)

    out_data["overall"] = _aggregate(all_rows)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(json.dumps(out_data, indent=2))
    return out_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-wikidata", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(seed=args.seed, n_wikidata=args.n_wikidata, out=args.out)
