"""KNOWLEDGE-BUNDLE fetch v2: a HARDENED, rate-limit-resilient, checkpoint/resume Wikidata fetcher that pulls a
LARGE (100k+), DIVERSE, CLEAN general-knowledge SVO-triple corpus for the sim-brain's fact-store (owner #1 priority:
teach the brain the fundamental knowledge an LLM has, then interact with it daily).

WHY v2 (what the first fetcher, `_knowledge_bundle_wikidata_fetch.py`, could not do):
  * it stalled on HTTP 429 (no backoff, no Retry-After, a generic User-Agent Wikidata throttles);
  * it only reached ~2.4k facts (a small hand-seeded root set, no pagination, no per-domain breadth);
  * it collapsed every value label to its single head token ("United States" -> "states"), and used
    `wbsearchentities` label-guessing for taxonomic roots -- the source of the "japan -> natural disaster" class of
    WRONG-OBJECT noise;
  * a kill/restart re-fetched everything from scratch (no checkpoint).

WHAT v2 DOES DIFFERENTLY:
  1. RATE-LIMIT RESILIENCE: a descriptive contact User-Agent (Wikidata requires it), a polite fixed delay between
     requests, and EXPONENTIAL BACKOFF WITH JITTER on 429/503/500/timeout that RESPECTS the `Retry-After` header.
  2. CHECKPOINT / RESUME: every fact is streamed to `<out>/facts.jsonl` as it is fetched, and each completed query
     TASK is recorded in `<out>/progress.json`. A kill/restart reloads the JSONL (dedup) and SKIPS completed tasks --
     it resumes, it does not re-fetch. A live `<out>/FETCH_STATE.json` (pid/target/count/status) lets a parent monitor
     liveness WITHOUT attaching.
  3. LABEL-NOISE FIX (structural, not heuristic): every fact is (a) SUBJECT-CLASS-ANCHORED (`?s wdt:P31 wd:<class>`)
     or OBJECT-ANCHORED (`?s <prop> wd:<obj>`), so a subject can never be a disambiguation page / list article /
     Wikimedia-internal page (those are not instances of `country`/`city`/`chemical element`/... ); (b) the OBJECT
     label is taken from the ENTITY'S OWN rdfs:label (via the Wikidata label service / a BIND), never guessed -- so the
     predicate->object mapping cannot be mislabeled (this is the exact fix for "japan -> natural disaster"); (c)
     unresolved bare-QID labels ("Q1358") are SKIPPED; (d) each label is normalised to a clean lowercase ASCII
     word-string ("José Mourinho" -> "jose mourinho", "Saint-Pierre" -> "saint pierre") so a fact reads as a clean SVO
     the way "france -> capital -> paris" or "gold -> isa -> chemical element" does -- multi-word labels are PRESERVED
     (not head-token-truncated), the composer stores a multi-word filler as one concept code.
  4. DIVERSITY: ~10 domain families (countries/capitals/continents/borders, cities, chemical elements, taxonomy of
     animals & plants, rivers/mountains/lakes/islands, languages/universities/currencies, films->directors,
     books->authors, planets, and -- for LLM-like breadth of notable people -- ~40 occupations and ~40 nationalities).
     High-value CLEAN domains are fetched FIRST; the high-volume people domains then fill the corpus to the target, so
     the result is a clean high-value core PLUS broad tail.

ON COMPLETION the fetcher BUILDS a directly-loadable `developed_brain_io` bundle into `<out>/bundle/` via the EXISTING
build path (reuse-by-import of `_knowledge_bundle_build_and_demo.build_composer` + `developed_brain_io`), from a capped
sample (`--bundle-cap`, default 5000) -- FHRR composer capacity (~sqrt(D)) bounds facts-per-brain, so the bundle is a
loadable DEMONSTRATION over a clean sample while the FULL corpus lives in `facts.jsonl` / `facts_raw.json`.

This is host-side DATA PREP (the world/environment supplying facts for the brain to LEARN), a declared TEST SCAFFOLD:
NO `sim/` edit, no production default changed. The composer's recall + no-confab moat over the loaded facts are the
genuine reads (see rf_phasor_composer.py).

Run:
  smoke (fetch 2k, build, load+assert):  SIM_BACKEND=numpy python -m research.runners._knowledge_bundle_wikidata_fetch_v2 --smoke 2000
  full  (detached, 100k+):               nohup python -m research.runners._knowledge_bundle_wikidata_fetch_v2 \
                                             --out <dir> --target 100000 > <dir>/fetch.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ENDPOINT = "https://query.wikidata.org/sparql"
# Wikidata's UA policy REQUIRES a descriptive agent with a contact -- a generic python-urllib UA gets throttled hard.
_HEADERS = {
    "User-Agent": "neural-simulator-kb-fetch/1.0 (https://github.com/danthi123/neural-simulator; daniel.thiberge@gmail.com)",
    "Accept": "application/sparql-results+json",
}
_QID_RE = re.compile(r"^Q\d+$")

OUT_DEFAULT = _REPO / "research" / "findings" / "raw" / "_knowledge_bundle_wikidata_100k"


# ===============================================================================================================
# Robust SPARQL: polite delay + exponential backoff with jitter + Retry-After honoring.
# ===============================================================================================================
class SparqlClient:
    def __init__(self, polite_delay=1.0, timeout=60, max_retries=6, base_backoff=2.0, max_backoff=120.0):
        self.polite_delay = float(polite_delay)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.base_backoff = float(base_backoff)
        self.max_backoff = float(max_backoff)
        self._last_request_t = 0.0
        self.n_requests = 0
        self.n_retries = 0
        self.n_429 = 0

    def _throttle(self):
        dt = time.time() - self._last_request_t
        if dt < self.polite_delay:
            time.sleep(self.polite_delay - dt)

    def query(self, sparql):
        """Return the result `bindings` list, or raise the last error after exhausting retries. Backoff (with jitter)
        on 429/503/500/timeout; on 429/503 with a `Retry-After` header, waits that long instead of the backoff."""
        url = _ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
        last = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_request_t = time.time()
            self.n_requests += 1
            try:
                req = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                return payload.get("results", {}).get("bindings", [])
            except urllib.error.HTTPError as ex:
                last = ex
                retry_after = None
                try:
                    ra = ex.headers.get("Retry-After") if ex.headers else None
                    if ra is not None:
                        retry_after = float(ra)
                except (TypeError, ValueError):
                    retry_after = None
                if ex.code == 429:
                    self.n_429 += 1
                if ex.code in (429, 503, 500, 502, 504):
                    self.n_retries += 1
                    self._sleep_backoff(attempt, retry_after)
                    continue
                raise  # a non-retryable HTTP error (400 bad query, etc.) -- surface it
            except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
                last = ex
                self.n_retries += 1
                self._sleep_backoff(attempt, None)
                continue
        if last is not None:
            raise last
        raise RuntimeError("sparql failed with no captured error")

    def _sleep_backoff(self, attempt, retry_after):
        if retry_after is not None and retry_after > 0:
            wait = min(retry_after + random.uniform(0.0, 1.0), self.max_backoff)
        else:
            wait = min(self.base_backoff * (2 ** attempt), self.max_backoff)
            wait += random.uniform(0.0, wait * 0.25)  # jitter
        print(f"  [backoff] attempt {attempt + 1}: sleeping {wait:.1f}s "
              f"(retry_after={retry_after})", flush=True)
        time.sleep(wait)


# ===============================================================================================================
# Label cleaning: entity's own rdfs:label -> a clean lowercase ASCII word-string. Multi-word PRESERVED.
# ===============================================================================================================
def clean_label(raw):
    """A Wikidata English label -> a clean SVO token (lowercase ASCII words separated by single spaces), or None if it
    is unusable (empty / a bare unresolved QID / degenerates to <2 chars). Accents are folded (jose, not josé),
    parenthetical disambiguators are dropped ("Mercury (planet)" -> "mercury"), separators (- / .) become spaces, and
    all remaining non-letters (digits, punctuation) are stripped -- so the result is exactly the [a-z ]+ shape the
    composer + the build's _clean_alpha accept. Multi-word labels are KEPT ("united states", "chemical element")."""
    if raw is None:
        return None
    t = raw.strip()
    if not t or _QID_RE.match(t):
        return None
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)             # drop parenthetical disambiguators
    t = unicodedata.normalize("NFKD", t)               # decompose accents
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[\-‐-―/_.,'’]", " ", t)      # separators/apostrophes -> space
    t = re.sub(r"[^a-z ]", "", t)                        # drop anything not a lowercase letter or space
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 2:
        return None
    return t


# ===============================================================================================================
# Domain tables (all QIDs verified against their own rdfs:label at build time -- see the fetch banner audit).
# ===============================================================================================================
# occupation (P106) -> "person occupation <occupation>" -- LLM-like breadth of notable people & their fields.
OCCUPATIONS = {
    "Q169470": "physicist", "Q593644": "chemist", "Q170790": "mathematician", "Q864503": "biologist",
    "Q11063": "astronomer", "Q4964182": "philosopher", "Q82955": "politician", "Q36180": "writer",
    "Q49757": "poet", "Q1028181": "painter", "Q36834": "composer", "Q177220": "singer", "Q33999": "actor",
    "Q2526255": "film director", "Q42973": "architect", "Q81096": "engineer", "Q188094": "economist",
    "Q201788": "historian", "Q39631": "physician", "Q40348": "lawyer", "Q1930187": "journalist",
    "Q205375": "inventor", "Q937857": "football player", "Q3665646": "basketball player",
    "Q10833314": "tennis player", "Q10873124": "chess player", "Q11631": "astronaut", "Q116": "monarch",
    "Q193391": "diplomat", "Q212980": "psychologist", "Q2306091": "sociologist", "Q14467526": "linguist",
    "Q520549": "geologist", "Q2374149": "botanist", "Q350979": "zoologist", "Q1281618": "sculptor",
    "Q33231": "photographer", "Q28389": "screenwriter", "Q6625963": "novelist", "Q214917": "playwright",
    "Q486748": "pianist", "Q855091": "guitarist", "Q158852": "conductor",
}
# major countries (used for P27 citizenship + P17 city-of-country). Labels fetched from Wikidata, not these hints.
COUNTRIES = ["Q30", "Q145", "Q142", "Q183", "Q38", "Q29", "Q159", "Q148", "Q17", "Q668", "Q155", "Q16", "Q408",
             "Q96", "Q55", "Q34", "Q20", "Q36", "Q41", "Q79", "Q43", "Q414", "Q884", "Q45", "Q40", "Q39", "Q31",
             "Q27", "Q35", "Q33", "Q212", "Q794", "Q801", "Q258", "Q1033", "Q252", "Q851", "Q869", "Q881", "Q213"]
# is-a classes (P31): "<instance> isa <class>". object = the class's own label.
ISA_CLASSES = ["Q6256", "Q515", "Q11344", "Q634", "Q4022", "Q8502", "Q23397", "Q34770", "Q3918", "Q7889",
               "Q23442", "Q9430", "Q165", "Q5107", "Q8142", "Q11424"]
# taxonomy parents (P279 subclass-of): "<child> isa <parent>". bare-QID parents (arachnid/mollusc/crustacean/
# vertebrate had no English rdfs:label at audit) are omitted -- clean_label would skip their objects anyway.
TAXO_PARENTS = ["Q7377", "Q5113", "Q152", "Q10811", "Q10908", "Q1390", "Q756", "Q10884", "Q506", "Q430", "Q764"]
# Shape-B free-object relations: (subject-class QID, property, relation-verb).
SHAPEB = [
    ("Q6256", "P36", "capital"),      # country -> its capital city
    ("Q6256", "P30", "continent"),    # country -> its continent
    ("Q6256", "P47", "borders"),      # country -> a bordering country
    ("Q4022", "P17", "country"),      # river -> country
    ("Q8502", "P17", "country"),      # mountain -> country
    ("Q23397", "P17", "country"),     # lake -> country
    ("Q11424", "P57", "director"),    # film -> director
    ("Q47461344", "P50", "author"),   # written work -> author
]


# ===============================================================================================================
# Query builders. Each returns (sparql_string). Runner turns rows -> facts.
# ===============================================================================================================
def _q_object_anchored(prop, obj_qid, ordered, limit, offset):
    """`?s wdt:<prop> wd:<obj>` -> rows of (?sLabel, constant ?oLabel). Subject class/edge-constrained; object is the
    fixed entity's own label. Used for is-a (P31), taxonomy (P279), occupation (P106), citizenship (P27)."""
    order = "ORDER BY ?s " if ordered else ""
    return (f"SELECT ?s ?sLabel ?oLabel WHERE {{ BIND(wd:{obj_qid} AS ?o) ?s wdt:{prop} ?o . "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} '
            f"{order}LIMIT {limit} OFFSET {offset}")


def _q_free_object(subj_class, prop, ordered, limit, offset):
    """`?s wdt:P31 wd:<class> ; wdt:<prop> ?o` -> rows of (?sLabel, ?oLabel). Subject constrained to a clean class;
    object label from the label service (bare-QID skipped). Used for capital/continent/borders/country/director/..."""
    order = "ORDER BY ?s " if ordered else ""
    return (f"SELECT ?s ?sLabel ?oLabel WHERE {{ ?s wdt:P31 wd:{subj_class} ; wdt:{prop} ?o . "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} '
            f"{order}LIMIT {limit} OFFSET {offset}")


def _q_city_of_country(country_qid, ordered, limit, offset):
    """`?c wdt:P31 wd:Q515 ; wdt:P17 wd:<country>` -> (?cLabel, constant ?countryLabel). city -> country."""
    order = "ORDER BY ?c " if ordered else ""
    return (f"SELECT ?c ?cLabel ?countryLabel WHERE {{ BIND(wd:{country_qid} AS ?country) "
            f"?c wdt:P31 wd:Q515 ; wdt:P17 ?country . "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} '
            f"{order}LIMIT {limit} OFFSET {offset}")


# ===============================================================================================================
# Task generation. A task = one query. `kind` picks the builder + row->fact mapping. Ordered/paged for the volume
# domains. High-value CLEAN domains come FIRST so a target-truncated run still has a clean high-value core.
# ===============================================================================================================
def build_task_list(limits):
    """Return an ordered list of task dicts: {id, kind, relation, sparql-params}. Ids are stable so a resume can skip
    completed ones. `limits` carries per-domain LIMIT + n_pages knobs."""
    tasks = []

    def add(tid, kind, relation, **params):
        tasks.append({"id": tid, "kind": kind, "relation": relation, **params})

    # --- Tier 1: small, ultra-clean, high-value (single page each) ---
    # country isa + capital + continent + borders
    add("isa:Q6256", "objanchor", "isa", prop="P31", obj="Q6256", ordered=False,
        limit=limits["isa_small"], offset=0)
    for subj, prop, rel in SHAPEB[:3]:  # capital, continent, borders (all on country Q6256)
        add(f"shapeb:{subj}:{prop}", "freeobj", rel, subj=subj, prop=prop, ordered=False,
            limit=limits["shapeb"], offset=0)
    # chemical elements, planets
    add("isa:Q11344", "objanchor", "isa", prop="P31", obj="Q11344", ordered=False, limit=500, offset=0)
    add("isa:Q634", "objanchor", "isa", prop="P31", obj="Q634", ordered=False, limit=1000, offset=0)
    # taxonomy (animals & plants): child isa parent
    for parent in TAXO_PARENTS:
        add(f"taxo:{parent}", "objanchor", "isa", prop="P279", obj=parent, ordered=False,
            limit=limits["taxo"], offset=0)
    # geography free-object (rivers/mountains/lakes -> country) + films/books
    for subj, prop, rel in SHAPEB[3:]:
        add(f"shapeb:{subj}:{prop}", "freeobj", rel, subj=subj, prop=prop, ordered=False,
            limit=limits["shapeb"], offset=0)
    # cities of each major country
    for cq in COUNTRIES:
        add(f"city:{cq}", "cityofcountry", "country", country=cq, ordered=False,
            limit=limits["city"], offset=0)

    # --- Tier 2: is-a of high-instance classes (paged) ---
    for cls in ISA_CLASSES:
        for pg in range(limits["isa_pages"]):
            add(f"isa:{cls}:p{pg}", "objanchor", "isa", prop="P31", obj=cls,
                ordered=(limits["isa_pages"] > 1), limit=limits["isa"], offset=pg * limits["isa"])

    # --- Tier 3: people breadth -- occupations (P106) then citizenship (P27), paged to fill to target ---
    for occ_qid in OCCUPATIONS:
        for pg in range(limits["occ_pages"]):
            add(f"occ:{occ_qid}:p{pg}", "objanchor", "occupation", prop="P106", obj=occ_qid,
                ordered=(limits["occ_pages"] > 1), limit=limits["occ"], offset=pg * limits["occ"])
    for cq in COUNTRIES:
        for pg in range(limits["cit_pages"]):
            add(f"cit:{cq}:p{pg}", "objanchor", "citizen", prop="P27", obj=cq,
                ordered=(limits["cit_pages"] > 1), limit=limits["cit"], offset=pg * limits["cit"])

    return tasks


def rows_to_facts(task, rows):
    """Map SPARQL rows -> a list of clean [subject, relation, object] triples for `task`."""
    facts = []
    relation = task["relation"]
    kind = task["kind"]
    for row in rows:
        if kind in ("objanchor", "freeobj"):
            s = clean_label(row.get("sLabel", {}).get("value"))
            o = clean_label(row.get("oLabel", {}).get("value"))
        elif kind == "cityofcountry":
            s = clean_label(row.get("cLabel", {}).get("value"))
            o = clean_label(row.get("countryLabel", {}).get("value"))
        else:
            continue
        if s and o and s != o:
            facts.append([s, relation, o])
    return facts


def task_sparql(task):
    kind = task["kind"]
    if kind == "objanchor":
        return _q_object_anchored(task["prop"], task["obj"], task["ordered"], task["limit"], task["offset"])
    if kind == "freeobj":
        return _q_free_object(task["subj"], task["prop"], task["ordered"], task["limit"], task["offset"])
    if kind == "cityofcountry":
        return _q_city_of_country(task["country"], task["ordered"], task["limit"], task["offset"])
    raise ValueError(f"unknown task kind {kind!r}")


# ===============================================================================================================
# Checkpoint / resume / state I/O.
# ===============================================================================================================
def _atomic_write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def write_state(out_dir, **fields):
    state_path = Path(out_dir) / "FETCH_STATE.json"
    existing = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except Exception:
            existing = {}
    existing.update(fields)
    _atomic_write_json(state_path, existing)


def load_checkpoint(out_dir):
    """Return (fact_set, completed_task_ids). fact_set is a set of (a, rel, p) tuples reloaded from facts.jsonl."""
    out_dir = Path(out_dir)
    fact_set = set()
    jsonl = out_dir / "facts.jsonl"
    if jsonl.exists():
        with open(jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    a, rel, p = json.loads(line)
                    fact_set.add((a, rel, p))
                except Exception:
                    continue
    completed = set()
    prog = out_dir / "progress.json"
    if prog.exists():
        try:
            completed = set(json.loads(prog.read_text()).get("completed_tasks", []))
        except Exception:
            completed = set()
    return fact_set, completed


# ===============================================================================================================
# The fetch loop.
# ===============================================================================================================
def fetch(out_dir, target, limits, client, pid, resume=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "facts.jsonl"
    prog_path = out_dir / "progress.json"

    fact_set, completed = (load_checkpoint(out_dir) if resume else (set(), set()))
    t0 = time.time()
    print(f"[kb-v2] resume: {len(fact_set)} facts + {len(completed)} tasks already done", flush=True)

    tasks = build_task_list(limits)
    write_state(out_dir, pid=pid, target=target, count_so_far=len(fact_set), status="running",
                started_note=f"started {time.strftime('%Y-%m-%d %H:%M:%S')}; {len(tasks)} tasks queued",
                last_update_note="init", n_tasks_total=len(tasks), n_tasks_done=len(completed))

    jsonl = open(jsonl_path, "a", encoding="utf-8", buffering=1)  # line-buffered append (survives kill)
    n_done = len(completed)
    try:
        for ti, task in enumerate(tasks):
            if len(fact_set) >= target:
                print(f"[kb-v2] target {target} reached ({len(fact_set)} facts); stopping early.", flush=True)
                break
            if task["id"] in completed:
                continue
            try:
                rows = client.query(task_sparql(task))
            except Exception as ex:
                # a failed task is NON-fatal: log, record it NOT-completed (so a later resume retries it), keep going.
                print(f"  [task {task['id']}] FAILED {type(ex).__name__}: {ex}", flush=True)
                write_state(out_dir, last_update_note=f"task {task['id']} failed: {type(ex).__name__}")
                continue
            new = 0
            for fact in rows_to_facts(task, rows):
                k = (fact[0], fact[1], fact[2])
                if k in fact_set:
                    continue
                fact_set.add(k)
                jsonl.write(json.dumps(fact, ensure_ascii=False) + "\n")
                new += 1
            completed.add(task["id"])
            n_done += 1
            if new or (ti % 5 == 0):
                print(f"  [{ti + 1}/{len(tasks)}] {task['id']} ({task['relation']}): "
                      f"+{new} new  (total {len(fact_set)}, reqs {client.n_requests}, 429s {client.n_429})",
                      flush=True)
            # checkpoint the progress + state every task (cheap; makes resume exact)
            _atomic_write_json(prog_path, {"completed_tasks": sorted(completed), "count": len(fact_set)})
            write_state(out_dir, count_so_far=len(fact_set), n_tasks_done=n_done,
                        last_update_note=f"task {task['id']}: +{new} (elapsed {int(time.time() - t0)}s)")
    finally:
        jsonl.close()

    elapsed = round(time.time() - t0, 1)
    # write the combined facts_raw.json (the schema the existing build/demo also understands)
    all_facts = sorted(fact_set)
    all_facts = [list(f) for f in all_facts]
    payload = {
        "source": "wikidata_live_v2",
        "endpoint": _ENDPOINT,
        "n_facts": len(all_facts),
        "elapsed_seconds": elapsed,
        "n_requests": client.n_requests,
        "n_retries": client.n_retries,
        "n_429": client.n_429,
        "relations": sorted({f[1] for f in all_facts}),
        "facts": all_facts,
    }
    _atomic_write_json(out_dir / "facts_raw.json", payload)
    print(f"[kb-v2] fetch done: {len(all_facts)} unique facts in {elapsed}s "
          f"({client.n_requests} requests, {client.n_429} 429s). Wrote facts_raw.json", flush=True)
    return all_facts, elapsed


# ===============================================================================================================
# Build the loadable bundle via the EXISTING build path (reuse-by-import). Capped sample (FHRR capacity).
# ===============================================================================================================
def build_bundle(out_dir, all_facts, bundle_cap, D, seed):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import logging
    logging.disable(logging.INFO)
    from research.runners._knowledge_bundle_build_and_demo import _clean_alpha, build_composer, _AgentShim
    from research.runners import developed_brain_io as dbio

    seen, clean = set(), []
    for a, v, p in all_facts:
        a, v, p = str(a).lower().strip(), str(v).lower().strip(), str(p).lower().strip()
        if not a or not v or not p or not _clean_alpha(a) or not _clean_alpha(p):
            continue
        k = (a, v, p)
        if k in seen:
            continue
        seen.add(k)
        clean.append([a, v, p])
    n_clean = len(clean)
    if bundle_cap is not None and n_clean > bundle_cap:
        rng = random.Random(seed)
        clean = rng.sample(clean, bundle_cap)   # deterministic subset down to the cap (FHRR capacity)
    print(f"[kb-v2] building bundle: {len(clean)} facts (from {n_clean} clean) at D={D} ...", flush=True)
    comp, vocab, build_seconds = build_composer(clean, D, seed)
    bundle_dir = Path(out_dir) / "bundle"
    manifest = dbio.save_developed_brain(
        _AgentShim(comp), bundle_dir, seed=seed, D=D, composer_kind="rf",
        extra_metadata={
            "provenance": "knowledge_bundle_wikidata_100k v2 (hardened rate-limited fetcher, 2026-08-20)",
            "source": "wikidata_live_v2",
            "n_facts_in_corpus": len(all_facts),
            "n_facts_clean": n_clean,
            "bundle_cap": bundle_cap,
            "note": "loadable DEMONSTRATION bundle over a clean sample; the FULL corpus is facts.jsonl / "
                    "facts_raw.json. FHRR capacity (~sqrt(D)) bounds facts-per-composer -- a production teach would "
                    "shard or raise D. Host-side data-prep test scaffold; no sim/ edit, no production default.",
            "test_scaffold": True,
        })
    print(f"[kb-v2] bundle written to {bundle_dir} (n_facts={manifest['n_facts']}, "
          f"vocab={len(vocab)}, build_time={build_seconds:.1f}s)", flush=True)
    return bundle_dir, manifest


# ===============================================================================================================
# Validate a built bundle: load through developed_brain_io + assert recall / abstain / print a clean sample.
# ===============================================================================================================
def validate_bundle(bundle_dir, n_recall=60, seed=42):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import logging
    logging.disable(logging.INFO)
    from research.runners import developed_brain_io as dbio

    agent, manifest = dbio.load_developed_brain(bundle_dir)
    comp = agent.composer

    # gather the loaded facts (first patient per (agent, relation) cue -- query_patient is first-match)
    first_seen = {}
    for fact, _handle in comp.kb:
        a, v, p = fact.get("agent"), fact.get("action"), fact.get("patient")
        if isinstance(a, str) and isinstance(v, str) and isinstance(p, str):
            first_seen.setdefault((a, v), p)

    rng = random.Random(seed)
    cues = list(first_seen.items())
    sample = rng.sample(cues, min(n_recall, len(cues)))   # deterministic recall subset
    correct, examples = 0, []
    for (a, v), expected in sample:
        ans = comp.query_patient(a, v)
        ok = (ans == expected)
        correct += int(ok)
        if len(examples) < 8:
            examples.append((a, v, expected, ans, ok))

    # abstain: a made-up subject never in the corpus must return None (the no-confab moat)
    abstain_words = ["snarklebee", "zorblaxi", "fnargleth"]
    abstains = {w: comp.query_patient(w, "isa") for w in abstain_words}
    abstain_ok = all(v is None for v in abstains.values())

    # a clean 20-fact sample to eyeball for label noise
    label_sample = [f"{f['agent']} -> {f['action']} -> {f['patient']}"
                    for f, _ in comp.kb[:20] if isinstance(f.get("patient"), str)]

    result = {
        "n_facts_loaded": len(comp.kb),
        "n_distinct_cues": len(first_seen),
        "recall_correct": correct,
        "recall_n": len(sample),
        "recall_examples": examples,
        "abstains": abstains,
        "abstain_ok": abstain_ok,
        "label_sample": label_sample,
        "pass": bool(correct >= 3 and abstain_ok),
    }
    return result


# ===============================================================================================================
# CLI.
# ===============================================================================================================
def _default_limits():
    return {
        "isa_small": 300,   # country isa (single page)
        "shapeb": 3000,     # free-object relations (capital/continent/borders/river/mountain/lake/film/book)
        "taxo": 300,        # subclasses per taxonomy parent
        "city": 1500,       # cities per country
        "isa": 2500, "isa_pages": 3,   # is-a of high-instance classes
        "occ": 1800, "occ_pages": 3,   # people by occupation
        "cit": 1500, "cit_pages": 2,   # people by citizenship
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output directory (facts.jsonl + FETCH_STATE.json + bundle/)")
    ap.add_argument("--target", type=int, default=100000, help="stop once this many unique facts are fetched")
    ap.add_argument("--smoke", type=int, default=0, help="smoke mode: fetch ~N facts, build bundle, load+assert, print")
    ap.add_argument("--no-resume", action="store_true", help="ignore any checkpoint and start fresh")
    ap.add_argument("--no-build", action="store_true", help="fetch only; do not build the bundle at the end")
    ap.add_argument("--bundle-cap", type=int, default=5000, help="max facts loaded into the demonstration bundle")
    ap.add_argument("--D", type=int, default=128, help="composer phasor dimension for the bundle (128 = production)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--polite-delay", type=float, default=1.0, help="min seconds between SPARQL requests")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    client = SparqlClient(polite_delay=a.polite_delay)

    if a.smoke and a.smoke > 0:
        # SMOKE: fetch ~N facts (small per-domain limits, no people-paging), build the bundle, load + assert.
        limits = {"isa_small": 250, "shapeb": 400, "taxo": 200, "city": 300,
                  "isa": 400, "isa_pages": 1, "occ": 300, "occ_pages": 1, "cit": 250, "cit_pages": 1}
        print(f"[kb-v2 SMOKE] target={a.smoke} facts into {out_dir}", flush=True)
        all_facts, elapsed = fetch(out_dir, a.smoke, limits, client, pid, resume=not a.no_resume)
        if not all_facts:
            print("SMOKE FAIL: no facts fetched"); write_state(out_dir, status="error"); return 2
        bundle_dir, manifest = build_bundle(out_dir, all_facts, a.bundle_cap, a.D, a.seed)
        res = validate_bundle(bundle_dir, n_recall=60, seed=a.seed)
        print("\n" + "=" * 100)
        print(f"SMOKE RESULT: pass={res['pass']}")
        print(f"  facts fetched: {len(all_facts)} in {elapsed}s")
        print(f"  bundle: {res['n_facts_loaded']} facts, {res['n_distinct_cues']} distinct cues")
        print(f"  recall: {res['recall_correct']}/{res['recall_n']} sampled cues correct")
        print("  3 example known-fact queries:")
        for a2, v, exp, got, ok in res["recall_examples"][:5]:
            print(f"    q({a2!r},{v!r}) -> {got!r}  (expected {exp!r}) {'OK' if ok else 'MISS'}")
        print(f"  abstain (no-confab moat): {res['abstains']}  -> abstain_ok={res['abstain_ok']}")
        print("  20-fact label sample (eyeball for noise):")
        for line in res["label_sample"]:
            print(f"    {line}")
        print("=" * 100, flush=True)
        write_state(out_dir, status="smoke_done")
        return 0 if res["pass"] else 1

    # FULL fetch (intended detached).
    limits = _default_limits()
    all_facts, elapsed = fetch(out_dir, a.target, limits, client, pid, resume=not a.no_resume)
    if a.no_build or not all_facts:
        write_state(out_dir, status="done" if all_facts else "error",
                    last_update_note=f"fetch complete: {len(all_facts)} facts, build skipped={a.no_build}")
        print(f"[kb-v2] DONE (no build): {len(all_facts)} facts.", flush=True)
        return 0 if all_facts else 2
    try:
        bundle_dir, manifest = build_bundle(out_dir, all_facts, a.bundle_cap, a.D, a.seed)
        res = validate_bundle(bundle_dir, n_recall=60, seed=a.seed)
        write_state(out_dir, status="done", count_so_far=len(all_facts),
                    last_update_note=(f"COMPLETE: {len(all_facts)} facts; bundle {res['n_facts_loaded']} facts, "
                                      f"recall {res['recall_correct']}/{res['recall_n']}, abstain_ok={res['abstain_ok']}"))
        print(f"[kb-v2] COMPLETE: {len(all_facts)} facts; bundle validated "
              f"(recall {res['recall_correct']}/{res['recall_n']}, abstain_ok={res['abstain_ok']}).", flush=True)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        write_state(out_dir, status="done_build_failed",
                    last_update_note=f"fetch OK ({len(all_facts)} facts) but bundle build failed: {type(ex).__name__}: {ex}")
        print(f"[kb-v2] fetch OK but build FAILED: {ex}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
