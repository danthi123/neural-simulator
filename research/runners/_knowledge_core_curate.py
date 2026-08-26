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


def _alias_quality(raw_alias: str):
    """Score a RAW (unsanitized) alias for natural-language usefulness, or None to reject outright. Deliberately
    mirrors `pick_clean_alias`'s own reject/score rules (cruft, digit/paren/slash noise, proper-noun shape) so
    the alias net applies the SAME quality bar the canonical-label picker already trusts, WITHOUT touching that
    already-verified function -- this is an independent scorer over the FULL alias list (not a pick-one-best),
    used to rank+cap every OTHER alias, below."""
    a = raw_alias.strip()
    low = a.lower()
    if len(a) < 2:
        return None
    if any(c in a for c in "/()+|:0123456789"):
        return None
    if any(k in low for k in _CRUFT):
        return None
    w = a.split()
    if len(w) > 5:
        return None
    score = 0.0
    if 2 <= len(w) <= 4:
        score += 2
    if a[:1].isupper():
        score += 1.5
    if a.islower():
        score -= 1
    return score


_MAX_ALIASES_PER_ID = 6   # caps vocab growth (raw Wikidata aliases run ~30/concept; most are low-quality noise)


def _all_other_aliases(path, id_to_canon: dict, max_per_id: int = _MAX_ALIASES_PER_ID) -> dict:
    """For every id in `id_to_canon`, take the TOP `max_per_id` OTHER raw aliases on that id's line (ranked by
    `_alias_quality`, rejects dropped outright) and return {id: [sanitized_token, ...]}, excluding the canonical
    token itself and de-duplicated. This is the SAME file, re-scanned (aliases were discarded after `_alias_map`
    picked one) -- a second pass is the simplest correct way to recover them without changing `_alias_map`'s
    existing (and already-verified) contract. Only ids present in `id_to_canon` are scanned/kept (callers
    restrict this to ids whose canonical token actually made it into the final curated fact vocab, so alias
    volume tracks the SHIPPED core, not the full top_entities/top_relations candidate pool). The cap keeps
    vocab growth bounded (the store's O(V*D) cleanup latency scales with vocab size -- see the module
    docstring's ">~20k distinct entities lifts latency above 1s" flip-soak finding) and, as a side effect,
    improves precision (a raw Wikidata alias list runs ~30 entries/concept, most low-quality: redirects, year
    tags, typos -- ranking by the SAME quality bar `pick_clean_alias` trusts keeps the natural multi-word
    phrasings a real question would use)."""
    wanted = set(id_to_canon)
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eid = parts[0]
            if eid not in wanted:
                continue
            canon = id_to_canon[eid]
            seen = {canon}
            scored = []
            for i, al in enumerate(parts[1:]):
                q = _alias_quality(al)
                if q is None:
                    continue
                t = sanitize(al)
                if not t or t in seen:
                    continue
                seen.add(t)
                scored.append((q - 0.05 * i, t))
            scored.sort(key=lambda x: -x[0])
            out[eid] = [t for _q, t in scored[:max_per_id]]
    return out


def build_alias_facts(ent_tok: dict, rel_tok: dict, ent_txt: str, rel_txt: str, used_vocab: set):
    """Build `{agent: alias_token, action: 'alias_of', patient: canonical_token}` facts from every OTHER raw
    Wikidata alias of each entity/relation already in the shipped core (restricted to `used_vocab`, the
    canonical tokens that actually appear in the final curated `facts` list -- so alias volume scales with
    what shipped, not the larger candidate pool).

    AMBIGUITY POLICY (honest, matches the module's dense-shared-vocab philosophy): an alias token is emitted
    ONLY if (a) it names exactly ONE distinct canonical concept across the whole scanned set, AND (b) it does
    not collide with any EXISTING canonical vocab word (an alias that happens to spell a different concept's
    own name is a genuine ambiguity -- 'is this word itself, or the thing it aliases?' -- and is dropped, never
    guessed). This mirrors `compositional_chain_route.py`'s own multi-valued-hop abstain: an unresolvable
    surface form must abstain, not silently pick one reading.

    Returns (alias_facts, n_collisions_dropped) for curation_report.json honesty.
    """
    ent_wanted = {eid: tok for eid, tok in ent_tok.items() if tok in used_vocab}
    rel_wanted = {rid: tok for rid, tok in rel_tok.items() if tok in used_vocab}
    ent_aliases = _all_other_aliases(ent_txt, ent_wanted)
    rel_aliases = _all_other_aliases(rel_txt, rel_wanted)
    all_canon = set(ent_wanted.values()) | set(rel_wanted.values())

    rev = {}   # alias_token -> set of canonical tokens it was seen pointing at
    for id_to_tok, alias_lists in ((ent_wanted, ent_aliases), (rel_wanted, rel_aliases)):
        for eid, toks in alias_lists.items():
            canon = id_to_tok[eid]
            for t in toks:
                rev.setdefault(t, set()).add(canon)

    alias_facts = []
    n_collisions = 0
    for tok, canons in sorted(rev.items()):
        # ambiguous if the alias names >=2 distinct concepts, OR collides with an EXISTING canonical word
        # (itself already means something directly -> a second, aliased meaning is an unresolvable homonym).
        if len(canons) != 1 or tok in all_canon:
            n_collisions += 1
            continue
        canon = next(iter(canons))
        if canon == tok:
            continue
        alias_facts.append({"agent": tok, "action": "alias_of", "patient": canon, "polarity": "AFFIRM"})
    return alias_facts, n_collisions


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

    # --- NATURAL-LANGUAGE GROUNDING (2026-08-26, board #65/#66 knowledge-grounding frontier A) ---
    # The canonical tokens above are Wikidata-alias-derived (e.g. 'chelsea_fc', 'instance_of') and are NOT the
    # words a natural question uses ('chelsea fc', 'is a'). Emit 'alias_of' facts from every OTHER raw alias of
    # each entity/relation ALREADY in `vocab`, so a query-time alias-hop (brain_chat_tui.py) can reach the
    # canonical token via a genuine spiking `query_patient` read. See `build_alias_facts` docstring for the
    # ambiguity/drop policy. `vocab` set alone (not full `used_vocab`) already IS "what shipped".
    alias_facts, n_alias_collisions = build_alias_facts(ent_tok, rel_tok, ent_txt, rel_txt, set(vocab))
    print(f"[curate] aliases: {len(alias_facts):,} alias_of facts, {n_alias_collisions:,} ambiguous "
          f"aliases dropped ({time.time()-t0:.0f}s)", flush=True)
    facts_with_aliases = facts + alias_facts
    # CRITICAL: include EVERY word an alias fact touches (agent, action, AND patient -- the literal 'alias_of'
    # relation token included) in the vocab passed to `build_ltm_from_facts`/`ShardedPhasorStore`, so NONE of
    # them is ever dynamically GROWN (an out-of-vocabulary word hit during the fast-path bulk encode). A grown
    # word's code comes from a SEPARATE runtime `_growth_rng`, inserted into the shared codebook at its
    # alphabetical position; `ShardedPhasorStore.save()`/`.load()` persists+reconstructs the codebook by a FRESH
    # BATCH generation over the (by-then-larger) vocab list, which does not reproduce that mixed batch+growth
    # history -- discovered by this arc's own end-to-end verification (a missing 'alias_of' from this set
    # corrupted decode for the ENTIRE reloaded bundle, alias facts and plain facts alike, not just the new
    # ones): every plain fact ALSO failed to recall after a save/reload round-trip once one word was grown.
    # Pre-existing store fragility (`ShardedPhasorStore`'s fast-path save/load does not preserve a runtime-grown
    # word's code); the fix here is to never trigger it -- the curated bundle is BUILT ONCE, not runtime-grown.
    vocab_with_aliases = sorted(set(vocab) | {w for f in alias_facts for w in (f["agent"], f["action"], f["patient"])})

    meta = {
        "n_triples_scanned": n_triples, "n_entities_total": len(ent_freq), "n_relations_total": len(rel_freq),
        "top_entities": top_entities, "top_relations": top_relations,
        "n_facts": len(facts), "vocab_size": len(vocab),
        "n_alias_facts": len(alias_facts), "n_alias_collisions_dropped": n_alias_collisions,
        "n_facts_with_aliases": len(facts_with_aliases), "vocab_size_with_aliases": len(vocab_with_aliases),
        "relations_used": rel_used.most_common(40),
        "sample_facts": [[f["agent"], f["action"], f["patient"]] for f in facts[:25]],
        "sample_alias_facts": [[f["agent"], f["action"], f["patient"]] for f in alias_facts[:25]],
    }
    return facts_with_aliases, vocab_with_aliases, meta


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
