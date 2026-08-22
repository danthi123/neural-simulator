"""KNOWLEDGE CORE CURATE + BUILD — the curated default-on knowledge bundle for BRAIN_LTM_BUNDLE (Task 2, board #133).

The knowledge-scale INFRASTRUCTURE is done + GREEN on main: a TieredFactStore (small conversation BUFFER + a routed
ShardedPhasorStore cortical LTM), wired opt-in as `BRAIN_LTM_BUNDLE`, byte-identical to an unsharded store, scales to
100k (finding 2026-08-21-knowledge-scale-flip-soak...). The only open piece is the OWNER-UX decision: *which* bundle
ships as the default. Owner guard (verbatim): "depth not breadth, a brain you communicate with, not a fancy plastic RAG."

So this builds a CURATED CORE — not the raw 5M dump — from wikidata5m by FREQUENCY/CONNECTIVITY: keep facts whose
subject AND object are among the most-connected entities and whose relation is among the most-frequent relations. That
yields a dense, SHARED-VOCAB core (~10-20k facts over a bounded ~8k vocab) which stays in the snappy sub-second regime
(latency is O(V*D) in the codebook cleanup, V=vocab; the flip-soak finding shows >~20k distinct entities lifts latency
above 1 s). Entities/relations are mapped to their canonical (first) alias, sanitized to a single atomic token so each
concept is one vocab word (the store encodes by WORD).

The persisted store is built with the GENUINE neural resonate bind (fast=False) — the biologically faithful bind, not
the closed-form encode_fast shortcut (faithfulness > speed is the standing non-negotiable). ~52 ms/fact => ~13 min for
15k, ~1.5 h for 100k, so RUN HEADLESS. Output is a ShardedPhasorStore.save() bundle dir (manifest.json + facts.json +
composites.npz) that `load_developed_brain(ltm_bundle=<dir>)` / `BRAIN_LTM_BUNDLE=<dir>` loads directly.

The learn-through-use consolidation (BRAIN_D5_CONSOLIDATE, default-on this session) stays in the live path — this
bundle is the bulk cortical LTM the buffer falls through to; consolidation is what makes it integrated biological
memory rather than static RAG. The default-on FLIP itself is deferred to the owner/Tuesday harvest, gated on the
companion soak (_knowledge_core_bundle_soak.py) — this runner only BUILDS + PERSISTS the artifact.

Run (headless, CPU/numpy, LOCAL — the wikidata5m source is on this box):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_core_curate \
    --out-bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k \
    --n-facts 15000 --top-entities 8000 --top-relations 40 --seed 42
  # --smoke  : tiny (top-entities 400, n-facts 300) end-to-end sanity in seconds
  # --fast   : closed-form bulk bind instead of the genuine resonate bind (NOT for the shipped bundle)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

DEFAULT_DATA = "/home/dant123/Projects/sim-data/wikidata5m"
DEFAULT_OUT = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"

_TOK_RE = re.compile(r"[^a-z0-9]+")
# wikidata5m aliases are crowd-sourced: the FIRST is often a typo/variant ("united stated", "califnornia") and the
# clean canonical label sits later. Reject Wikipedia cruft, then prefer a properly-capitalized proper-noun form.
_CRUFT = ("list of", "wikiproject", "reference", "selected picture", "candidate", "disambiguation",
          "template", "category", "/selected", "index of")


def sanitize(alias: str, maxlen: int = 40) -> str:
    """Canonical single atomic token: lowercase, non-alnum -> '_', collapse, strip, cap length."""
    t = _TOK_RE.sub("_", alias.strip().lower()).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip("_")
    return t


def pick_clean_alias(aliases) -> str:
    """Choose the cleanest canonical label among the first ~20 aliases (a big quality win over 'take the first')."""
    best, best_score = None, -1e18
    for i, al in enumerate(aliases[:20]):
        a = al.strip()
        low = a.lower()
        if len(a) < 2:
            continue
        if any(c in a for c in "/()+|:0123456789"):   # parens/slashes/digits => wiki cruft or an id
            continue
        if any(k in low for k in _CRUFT):
            continue
        w = a.split()
        if len(w) > 5:
            continue
        score = 0.0
        if 2 <= len(w) <= 4:            # proper multi-word names beat 1-word typos/abbreviations
            score += 2
        if a[:1].isupper():             # canonical labels are capitalized; typo variants are usually all-lower
            score += 1.5
        if a.islower():
            score -= 1
        score -= 0.15 * i               # mild preference for earlier (still lets a later clean name win)
        if score > best_score:
            best_score, best = score, a
    if best is None:
        best = aliases[0] if aliases else ""
    return sanitize(best)


def _alias_map(path, wanted: set, clean: bool) -> dict:
    """id -> token, only for ids in `wanted`. clean=True runs the cruft-rejecting proper-noun picker (noisy ENTITY
    aliases); clean=False takes the first alias (RELATION labels are already canonical + clean)."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eid = parts[0]
            if eid not in wanted:
                continue
            tok = pick_clean_alias(parts[1:]) if clean else sanitize(parts[1])
            if tok:
                out[eid] = tok
    return out


def curate(data_dir, n_facts, top_entities, top_relations, seed):
    train = os.path.join(data_dir, "wikidata5m_transductive_train.txt")
    ent_txt = os.path.join(data_dir, "wikidata5m_entity.txt")
    rel_txt = os.path.join(data_dir, "wikidata5m_relation.txt")
    for p in (train, ent_txt, rel_txt):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # --- Pass 1: entity degree (head+tail) + relation frequency ---
    t0 = time.time()
    ent_freq = Counter()
    rel_freq = Counter()
    n_triples = 0
    with open(train, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            h, r, t = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
            if not (h and r and t):
                continue
            ent_freq[h] += 1
            ent_freq[t] += 1
            rel_freq[r] += 1
            n_triples += 1
    print(f"[curate] pass1: {n_triples:,} triples, {len(ent_freq):,} entities, "
          f"{len(rel_freq):,} relations ({time.time()-t0:.0f}s)", flush=True)

    top_ent = {e for e, _ in ent_freq.most_common(top_entities)}
    top_rel = {r for r, _ in rel_freq.most_common(top_relations)}

    # --- aliases for the selected ids only ---
    ent_tok = _alias_map(ent_txt, top_ent, clean=True)     # entities: noisy -> cruft-rejecting picker
    rel_tok = _alias_map(rel_txt, top_rel, clean=False)    # relations: first alias is already canonical
    print(f"[curate] aliases: {len(ent_tok):,}/{len(top_ent):,} entities, "
          f"{len(rel_tok):,}/{len(top_rel):,} relations have a name ({time.time()-t0:.0f}s)", flush=True)

    # --- Pass 2: collect qualifying facts, dedup (subj,rel)->best-scored obj ---
    best = {}   # (subj_tok, rel_tok) -> (obj_tok, score)
    with open(train, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            h, r, t = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
            if h not in top_ent or t not in top_ent or r not in top_rel:
                continue
            sh, sr, st = ent_tok.get(h), rel_tok.get(r), ent_tok.get(t)
            if not (sh and sr and st) or sh == st:
                continue
            key = (sh, sr)
            score = ent_freq[h] + ent_freq[t]
            cur = best.get(key)
            if cur is None or score > cur[1]:
                best[key] = (st, score)
    print(f"[curate] pass2: {len(best):,} distinct (subject,relation) candidate facts ({time.time()-t0:.0f}s)",
          flush=True)

    # rank by score, cap at n_facts -> the highest-value dense core
    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[:n_facts]
    facts = [{"agent": s, "action": rel, "patient": obj, "polarity": "AFFIRM"}
             for (s, rel), (obj, _sc) in ranked]
    vocab = sorted({w for f in facts for w in (f["agent"], f["action"], f["patient"])})
    rel_used = Counter(f["action"] for f in facts)
    meta = {
        "n_triples_scanned": n_triples, "n_entities_total": len(ent_freq), "n_relations_total": len(rel_freq),
        "top_entities": top_entities, "top_relations": top_relations,
        "n_facts": len(facts), "vocab_size": len(vocab),
        "relations_used": rel_used.most_common(40),
        "sample_facts": [[f["agent"], f["action"], f["patient"]] for f in facts[:25]],
    }
    return facts, vocab, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-bundle", default=DEFAULT_OUT)
    ap.add_argument("--n-facts", type=int, default=15000)
    ap.add_argument("--top-entities", type=int, default=8000)
    ap.add_argument("--top-relations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--fast", action="store_true", help="closed-form bulk bind (NOT for the shipped bundle)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", default="research/findings/raw/_knowledge_core/curate_report.json")
    a = ap.parse_args()

    if a.smoke:
        a.top_entities, a.top_relations, a.n_facts = 400, 20, 300
        a.out_bundle = a.out_bundle + "_smoke"

    t0 = time.time()
    facts, vocab, meta = curate(a.data_dir, a.n_facts, a.top_entities, a.top_relations, a.seed)
    if not facts:
        print("[curate] NO FACTS produced — check thresholds/data", flush=True)
        return 1

    from research.runners.tiered_fact_store import build_ltm_from_facts
    print(f"[build] building ShardedPhasorStore: {len(facts):,} facts, vocab={len(vocab):,}, "
          f"seed={a.seed}, D={a.D}, fast={a.fast} (genuine resonate bind if False) ...", flush=True)
    tb = time.time()
    ltm = build_ltm_from_facts(facts, vocab=vocab, seed=a.seed, D=a.D, fast=a.fast)
    build_s = time.time() - tb
    os.makedirs(a.out_bundle, exist_ok=True)
    n_saved = ltm.save(a.out_bundle)
    print(f"[build] built + saved {n_saved:,} facts to {a.out_bundle} in {build_s:.0f}s "
          f"({1000*build_s/max(1,len(facts)):.1f} ms/fact); shards={ltm.n_shards}", flush=True)

    report = dict(meta)
    report.update({
        "out_bundle": a.out_bundle, "seed": a.seed, "D": a.D, "fast": a.fast, "n_shards": int(ltm.n_shards),
        "n_saved": int(n_saved), "build_seconds": round(build_s, 1), "total_seconds": round(time.time() - t0, 1),
        "ship_ready": bool(n_saved == len(facts) and not a.fast),
    })
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(report, fh, indent=2)
    # also drop the report INSIDE the bundle for provenance-at-rest
    with open(os.path.join(a.out_bundle, "curation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[curate] wrote {a.report}  ship_ready={report['ship_ready']}  ({report['total_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
