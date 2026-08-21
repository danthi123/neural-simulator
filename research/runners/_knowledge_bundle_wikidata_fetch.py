"""KNOWLEDGE-BUNDLE fetch: pull a general-knowledge SVO triple set from Wikidata for the standalone-composer
knowledge-rich chat test scaffold (owner ask, 2026-08-20). Reuse-by-import of the validated `_sparql`/`_head_token`
primitives from `_fluidconv_phase15_wikidata_breadth_derisk.py` (the same endpoint, retry/backoff, and single-
head-token label simplification already de-risked there) -- this script only widens the SEED SET and adds a
label->QID resolver (`wbsearchentities`) so we are not hand-memorizing QIDs. NO sim/ edit; this is host-side data
prep (declared a TEST SCAFFOLD -- the composer's recall + no-confab moat over the loaded facts are the genuine
read, not this fetch).

Relations fetched (task-specified + the pipeline's existing clean set):
  P279 subclass-of   -> "isa"   (dog isa mammal)      -- taxonomic breadth, 2 levels (root -> child -> grandchild)
  P31  instance-of    -> "isa"   (france isa country)  -- named-entity class membership (countries/planets/elements)
  P527 has-part       -> "has"  (car has wheel)        -- part-whole, curated small set of classes with clean parts
  P36  capital         -> "has"  (france has paris)     -- one more clean, high-value general-knowledge relation

Simplification: each Wikidata value label -> its single lowercase head token (`_head_token`, reused verbatim) --
the SAME lossy-but-clean simplification the validated phase-15 pipeline uses. Multi-word labels ("United States")
collapse to their last word ("states"); this is an honest, disclosed limitation of the bulk loader, not the
composer (see the bundle's README / the runner's own docstring HONEST_CEILING field).

Run: SIM_BACKEND=numpy python -m research.runners._knowledge_bundle_wikidata_fetch [--refetch] [--out PATH]
First run fetches + caches to `--out`; later runs read the cache unless --refetch.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._fluidconv_phase15_wikidata_breadth_derisk import _sparql, _head_token, _UA  # noqa: E402

OUT_DEFAULT = _REPO / "research" / "findings" / "raw" / "_knowledge_bundle_wikidata" / "facts_raw.json"
_SEARCH_EP = "https://www.wikidata.org/w/api.php"

# ---------------------------------------------------------------------------------------------------------------
# Seed set: broad general-knowledge root categories across animals / plants / objects / places / abstract kinds.
# Resolved to QIDs at fetch time via wbsearchentities (first hit = Wikidata's own relevance ranking, i.e. the
# canonical/most-notable sense of the label -- "dog" -> Q144 the species, not a constellation or film).
# ---------------------------------------------------------------------------------------------------------------
TAXONOMY_ROOTS = [
    # animals
    "mammal", "bird", "fish", "reptile", "amphibian", "insect", "arachnid", "crustacean", "mollusk",
    "carnivore", "primate", "rodent", "ungulate", "marsupial", "cetacean",
    # plants / fungi
    "tree", "flower", "fruit", "vegetable", "grass", "fungus", "shrub", "conifer", "fern",
    # objects / artifacts
    "vehicle", "tool", "furniture", "weapon", "musical instrument", "clothing", "container", "machine",
    "building", "ship", "aircraft", "computer", "toy", "kitchenware",
    # food / substances
    "food", "beverage", "dairy product", "meat", "grain", "spice", "metal", "mineral", "gemstone",
    # abstract / natural kinds
    "color", "shape", "emotion", "disease", "profession", "sport", "language", "religion",
    "science", "art form", "musical genre", "dance", "season", "weather", "natural disaster",
    "celestial body", "planet", "star", "landform", "body of water", "ecosystem", "body part",
]
INSTANCE_ROOTS = {
    "country": "isa",     # sovereign states -> "france isa country"
    "planet": "isa",       # solar-system planets -> "mars isa planet"
    "chemical element": "isa",  # "gold isa element" (element itself simplifies via head-token)
}
HASPART_ROOTS = [
    "car", "bicycle", "airplane", "ship", "computer", "house", "tree", "flower", "human body",
    "guitar", "piano", "book", "cell", "atom", "eye", "heart", "brain", "leaf", "root system",
]


# QID_OVERRIDES: hand-checked entries carried from this project's ALREADY-VALIDATED pipelines
# (_fluidconv_phase15_wikidata_breadth_derisk.SEEDS/PARENTS, _realcorpus_wikidata_taxonomy_derisk.SUPERS) plus a
# handful this script's own dry run caught. `wbsearchentities` (label search) is convenient but NOT reliable for
# taxonomic root words -- e.g. "bird"/"fish" as plain English words resolve (by search relevance/popularity) to
# Wikidata items that are NOT the biological class and carry zero P279 children (measured: Q14915264 "bird" and
# Q16869951 "fish" both return 0 subclasses, vs the correct Q5113/Q152 which return real taxonomy). Overrides win;
# `_resolve_qid` is the fallback for every other label, with a post-fetch sanity check (see `_looks_empty`) that
# flags (does not silently drop) a resolved QID that yielded nothing everywhere it was tried.
QID_OVERRIDES = {
    "mammal": "Q7377", "bird": "Q5113", "fish": "Q152", "insect": "Q1390", "tree": "Q10884",
    "vehicle": "Q42889", "tool": "Q39546", "fruit": "Q3314483", "plant": "Q756", "dog": "Q144",
    "cat": "Q146", "river": "Q4022", "reptile": "Q10811", "amphibian": "Q10908",
}


def _resolve_qid(label, timeout=15, retries=3):
    """label -> best-match QID. Checks QID_OVERRIDES (this project's hand-verified taxonomic roots) FIRST; else
    falls back to the Wikidata search API (`wbsearchentities` -- purpose-built for label resolution, ranks by
    Wikidata's own notability). The fallback is not always right for a plain-English category word (see the
    QID_OVERRIDES docstring above); `_looks_empty` flags a resolution that returned nothing downstream."""
    if label in QID_OVERRIDES:
        return QID_OVERRIDES[label]
    url = _SEARCH_EP + "?" + urllib.parse.urlencode(
        {"action": "wbsearchentities", "search": label, "language": "en", "format": "json", "limit": 1})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            hits = data.get("search", [])
            return hits[0]["id"] if hits else None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as ex:
            last = ex
            time.sleep(1.5 * (attempt + 1))
    raise last if last is not None else RuntimeError("wbsearchentities failed")


def _subclasses(qid, limit=60):
    q = (f'SELECT ?cLabel WHERE {{ ?c wdt:P279 wd:{qid}. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("cLabel", {}).get("value", ""))
        if tok and len(tok) > 1:
            out.append(tok)
    return out


def _instances(qid, limit=250):
    q = (f'SELECT ?cLabel WHERE {{ ?c wdt:P31 wd:{qid}. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("cLabel", {}).get("value", ""))
        if tok and len(tok) > 1:
            out.append(tok)
    return out


def _haspart(qid, limit=10):
    q = (f'SELECT ?pLabel WHERE {{ wd:{qid} wdt:P527 ?p. '
         f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT {limit}')
    out = []
    for row in _sparql(q):
        tok = _head_token(row.get("pLabel", {}).get("value", ""))
        if tok and len(tok) > 1:
            out.append(tok)
    return out


def _capital(qid):
    q = f'SELECT ?capLabel WHERE {{ wd:{qid} wdt:P36 ?cap. SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT 1'
    for row in _sparql(q):
        tok = _head_token(row.get("capLabel", {}).get("value", ""))
        if tok and len(tok) > 1:
            return tok
    return None


def fetch_all(refetch=False, out_path=OUT_DEFAULT, max_seconds=1800):
    out_path = Path(out_path)
    if out_path.exists() and not refetch:
        return json.loads(out_path.read_text())

    t0 = time.time()
    facts = []  # list of [agent, relation, patient]
    qids = {}
    errors = []
    empty_roots = []  # roots whose resolved QID returned ZERO results (a likely wrong-QID resolution -- flagged, not hidden)

    def _add(a, rel, p):
        if a and p and a != p:
            facts.append([a, rel, p])

    print(f"[kb-fetch] resolving {len(TAXONOMY_ROOTS)} taxonomy roots + {len(INSTANCE_ROOTS)} instance roots + "
          f"{len(HASPART_ROOTS)} has-part roots ...", flush=True)

    # --- P279 taxonomy: root -> children (level 1), then a sample of children -> grandchildren (level 2) ---
    level1 = {}
    for name in TAXONOMY_ROOTS:
        if time.time() - t0 > max_seconds:
            errors.append(f"TIMEOUT before finishing taxonomy roots at {name!r}"); break
        try:
            qid = qids.get(name) or _resolve_qid(name)
            qids[name] = qid
            if qid is None:
                errors.append(f"no QID for root {name!r}"); continue
            kids = _subclasses(qid, limit=60)
            level1[name] = kids
            for k in kids:
                _add(k, "isa", name)
            if not kids:
                empty_roots.append({"root": name, "qid": qid, "stage": "P279"})
            print(f"  [P279] {name} ({qid}): {len(kids)} subclasses", flush=True)
            time.sleep(0.2)
        except Exception as ex:
            errors.append(f"P279 root {name!r} failed: {type(ex).__name__}: {ex}")

    # level 2: for each root, fetch subclasses of up to 6 of its own children (depth, bounded to stay fast)
    for name, kids in level1.items():
        for child in kids[:6]:
            if time.time() - t0 > max_seconds:
                break
            try:
                cqid = qids.get(child) or _resolve_qid(child)
                qids[child] = cqid
                if cqid is None:
                    continue
                gkids = _subclasses(cqid, limit=15)
                for g in gkids:
                    _add(g, "isa", child)
                time.sleep(0.15)
            except Exception as ex:
                errors.append(f"P279 depth {child!r} (under {name!r}) failed: {type(ex).__name__}: {ex}")

    # --- P31 instance-of: named entities (countries, planets, elements) ---
    for name, rel in INSTANCE_ROOTS.items():
        if time.time() - t0 > max_seconds:
            errors.append(f"TIMEOUT before instance root {name!r}"); break
        try:
            qid = qids.get(name) or _resolve_qid(name)
            qids[name] = qid
            if qid is None:
                errors.append(f"no QID for instance root {name!r}"); continue
            insts = _instances(qid, limit=250)
            for i in insts:
                _add(i, rel, name)
            if not insts:
                empty_roots.append({"root": name, "qid": qid, "stage": "P31"})
            print(f"  [P31] {name} ({qid}): {len(insts)} instances", flush=True)
            time.sleep(0.2)
        except Exception as ex:
            errors.append(f"P31 root {name!r} failed: {type(ex).__name__}: {ex}")

    # --- P527 has-part: curated classes with clean, well-known parts ---
    for name in HASPART_ROOTS:
        if time.time() - t0 > max_seconds:
            errors.append(f"TIMEOUT before has-part root {name!r}"); break
        try:
            qid = qids.get(name) or _resolve_qid(name)
            qids[name] = qid
            if qid is None:
                errors.append(f"no QID for has-part root {name!r}"); continue
            parts = _haspart(qid, limit=10)
            for p in parts:
                _add(name, "has", p)
            if not parts:
                empty_roots.append({"root": name, "qid": qid, "stage": "P527"})
            print(f"  [P527] {name} ({qid}): {len(parts)} parts", flush=True)
            time.sleep(0.2)
        except Exception as ex:
            errors.append(f"P527 root {name!r} failed: {type(ex).__name__}: {ex}")

    # --- P36 capital: reuse the country QIDs already fetched under P31 if we resolved "country" ---
    country_qid = qids.get("country")
    if country_qid is not None:
        try:
            countries_q = (f'SELECT ?cLabel ?capLabel WHERE {{ ?c wdt:P31 wd:{country_qid}. '
                           f'OPTIONAL {{ ?c wdt:P36 ?cap. }} '
                           f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT 250')
            rows = _sparql(countries_q)
            n_cap = 0
            for row in rows:
                cname = _head_token(row.get("cLabel", {}).get("value", ""))
                capv = row.get("capLabel", {}).get("value", "")
                cap = _head_token(capv) if capv else None
                if cname and cap:
                    _add(cname, "has", cap)
                    n_cap += 1
            print(f"  [P36] {n_cap} country->capital facts", flush=True)
        except Exception as ex:
            errors.append(f"P36 capitals failed: {type(ex).__name__}: {ex}")

    # dedup, preserve order
    seen, uniq = set(), []
    for f in facts:
        k = tuple(f)
        if k not in seen:
            seen.add(k); uniq.append(f)

    payload = {
        "source": "wikidata_live",
        "props": {"P279": "isa", "P31": "isa", "P527": "has", "P36": "has (capital)"},
        "n_facts_raw": len(facts), "n_facts_dedup": len(uniq),
        "elapsed_seconds": round(time.time() - t0, 1),
        "errors": errors,
        "empty_roots": empty_roots,
        "facts": uniq,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[kb-fetch] {len(uniq)} unique facts (from {len(facts)} raw) in {payload['elapsed_seconds']}s, "
          f"{len(errors)} errors. Wrote {out_path}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max-seconds", type=int, default=1800)
    a = ap.parse_args()
    try:
        payload = fetch_all(refetch=a.refetch, out_path=a.out, max_seconds=a.max_seconds)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as ex:
        print(f"NOT-RUNNABLE: Wikidata fetch failed ({type(ex).__name__}: {ex}) and no cache present")
        return 2
    print(f"n_facts_dedup={payload['n_facts_dedup']} errors={len(payload.get('errors', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
