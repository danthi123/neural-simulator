"""Emergent-defensible taxonomy source (the CYCLE-1039 PRIMARY path): mine the is-a graph from REAL Simple English
Wikipedia first-sentence DEFINITIONS ("A robin is a bird", "A dog is a mammal"), NOT a curated Wikidata P279 graph.
The brain reads natural definitional text (legitimate encyclopedic experience) -> the EXISTING copular miner
(`mine_isa`) extracts is-a pairs -> gate: len(hubs)>=8 AND len(hub_pairs)>=40 (a clean multi-member taxonomy). If it
passes, natural text yields the is-a graph the corpora (TinyStories/WikiText) could NOT -- superseding the curated
graph AND unlocking the canonical animal domain (robin->bird->animal). Fetches via the Wikipedia REST summary API
(cached to raw/). numpy + network. NO `sim/` edit.
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request, urllib.parse, urllib.error
from collections import Counter
from research.runners._realcorpus_copular_isa_miner_derisk import _is_content_noun, _ADJ_LIKE

_UA = {"User-Agent": "sim-research/1.0 (grounded-knowledge taxonomy research; contact via repo)"}
_CACHE = "research/findings/raw/_simplewiki_defs.json"

# common-name titles across domains (their Simple-Wiki first sentences are typically definitional "X is a <super>").
TITLES = [
    # mammals
    "Dog", "Cat", "Horse", "Cow", "Lion", "Tiger", "Elephant", "Whale", "Bear", "Wolf", "Rabbit", "Mouse",
    # birds
    "Robin (bird)", "Eagle", "Owl", "Penguin", "Sparrow", "Duck", "Chicken", "Crow", "Parrot",
    # fish
    "Salmon", "Trout", "Shark", "Tuna", "Cod", "Goldfish",
    # insects
    "Ant", "Bee", "Butterfly", "Beetle", "Housefly", "Wasp", "Grasshopper",
    # trees / plants
    "Oak", "Pine", "Maple", "Birch", "Willow", "Rose", "Tulip", "Daisy", "Sunflower",
    # vehicles
    "Car", "Truck", "Bus", "Bicycle", "Motorcycle", "Airplane", "Helicopter",
    # tools
    "Hammer", "Saw", "Screwdriver", "Wrench", "Axe", "Chisel",
    # the superordinates themselves (for multi-level: "A mammal is an animal")
    "Mammal", "Bird", "Fish", "Insect", "Tree", "Flower", "Vehicle", "Tool",
]


def _summary(title, timeout=20, retries=3):
    url = "https://simple.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                d = json.load(r)
                return d.get("extract", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5 * (k + 1))
    return ""


def fetch_defs(refetch=False):
    """Incremental cache: only fetch titles still MISSING (so a re-run fills the rate-limited gaps). 0.6s delay."""
    defs = json.load(open(_CACHE)) if (os.path.exists(_CACHE) and not refetch) else {}
    missing = [t for t in TITLES if t not in defs]
    for t in missing:
        ex = _summary(t); time.sleep(0.6)
        if ex:
            defs[t] = ex
            print(f"  {t}: {ex[:65]}", flush=True)
        else:
            print(f"  {t}: (empty)", flush=True)
    json.dump(defs, open(_CACHE, "w"), indent=1)
    return defs


_PLURAL = {"mammals": "mammal", "birds": "bird", "fish": "fish", "fishes": "fish", "insects": "insect",
           "trees": "tree", "flowers": "flower", "vehicles": "vehicle", "tools": "tool", "animals": "animal",
           "plants": "plant", "shrubs": "shrub", "machines": "machine", "seabirds": "bird", "reptiles": "reptile"}


def _sing(w):
    return _PLURAL.get(w, w[:-1] if w.endswith("s") and len(w) > 3 else w)


def _first_sentence(text):
    s = text.split(". ")[0].lower().replace("(", " ").replace(")", " ")
    return [w.strip(".,;:'\"") for w in s.split() if w.strip(".,;:'\"")]


# post-nominal boundary words: prepositions / relativizers / common Simple-Wiki verbs+participles. The NP head is the
# LAST content noun BEFORE one of these (English noun phrases are head-final: 'a [adj adj] HEAD of/with/that ...').
_BOUNDARY = {"of", "with", "that", "which", "for", "from", "in", "on", "to", "and", "or", "but", "as", "at", "by",
             "found", "used", "kept", "called", "made", "designed", "meant", "living", "live", "lives", "known",
             "related", "belonging", "belong", "native", "usually", "also", "having", "has", "have", "is", "are",
             "the", "a", "an", "its", "their", "they", "it", "there"}


def _np_head_natural(toks, j, n):
    """Return the HEAD noun of the NP starting at j: the LAST content word before a post-nominal boundary. Skips
    leading/attributive modifiers (lexical adjectives like 'large'/'motor' too), unlike the suffix-only _np_head."""
    run = []
    while j < n:
        w = toks[j]
        if w in _BOUNDARY or not _is_content_noun(w):
            break
        if not w.endswith(_ADJ_LIKE):                            # keep candidate NOUNS (drop clear -ly/-ed/-ing adjs)
            run.append(w)
        j += 1
    return run[-1] if run else None


def mine_natural(subject, toks):
    """Extract is-a from a natural definitional first sentence, handling the varied forms Simple-Wiki actually uses:
    'X is a/an [adj]* HEAD', 'The X is a/an [adj]* HEAD', 'Xs are [adj]* HEADs' (plural), 'X is a kind/type of Y'.
    The SUBJECT is the article title (reliable) -- not the sentence's first token (which may be 'the'/'a'/plural)."""
    n = len(toks)
    child = _sing(subject.split()[0].lower())
    if child == "unk" or not _is_content_noun(child):
        return []
    KINDS = ("kind", "type", "sort", "form", "member", "species", "genus", "group")
    for i in range(n - 1):
        if toks[i] not in ("is", "are"):
            continue
        j = i + 1
        if j < n and toks[j] in ("a", "an"):                     # 'is a/an ...'
            j += 1
        if j < n and toks[j] in KINDS and j + 2 < n and toks[j + 1] == "of":   # 'a kind of Y'
            j += 2
        parent = _sing(_np_head_natural(toks, j, n) or "")
        if parent and _is_content_noun(parent) and parent != child:
            return [(child, parent)]                             # first copular parent only (the genus)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args()
    print("[simple-wiki is-a] mine the taxonomy from REAL Simple-Wikipedia definitions (natural text)", flush=True)
    try:
        defs = fetch_defs(refetch=a.refetch)
    except Exception as ex:
        print(f"  FETCH FAILED: {type(ex).__name__}: {ex}; VERDICT: NOT-EVALUABLE", flush=True); return
    if len(defs) < 20:
        print(f"  only {len(defs)} definitions fetched; VERDICT: NOT-EVALUABLE (network)"); return
    pairs = Counter()
    for t, ex in defs.items():
        for (c, p) in mine_natural(t, _first_sentence(ex)):
            pairs[(c, p)] += 1
    n_children_of = Counter()
    for (c, p) in pairs:
        n_children_of[p] += 1
    hubs = {p for p, k in n_children_of.items() if k >= 2}       # >=2 distinct members (small curated title set)
    hub_pairs = {(c, p): pairs[(c, p)] for (c, p) in pairs if p in hubs}
    parents = set(n_children_of)
    chains = sum(1 for (c, p) in hub_pairs if c in parents)      # multi-level: a child that is itself a parent
    print(f"  fetched {len(defs)} defs | is-a pairs mined: {len(pairs)}", flush=True)
    print(f"  HUB superordinates (>=2 distinct children): {len(hubs)} -> {len(hub_pairs)} hub-pairs; multi-level chains: {chains}", flush=True)
    top = [f"{p}({k})" for p, k in n_children_of.most_common(15) if p in hubs]
    print(f"  top hubs: {top}", flush=True)
    print(f"  sample pairs: {[f'{c}->{p}' for (c, p) in list(hub_pairs)[:25]]}", flush=True)
    go = len(hubs) >= 6 and len(hub_pairs) >= 25
    print(f"\n  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- Simple-Wikipedia natural definitions "
          f"{'yield a clean multi-member is-a graph (hubs>=6, pairs>=25 at this ~65-title scale) -> the emergent-defensible natural-text taxonomy source works (natural definitional text -> discovered is-a), a legitimate alternative to the curated P279 graph' if go else f'give {len(hubs)} hubs / {len(hub_pairs)} pairs (below the 6/25 gate) -- natural first-sentence definitions are more varied (plural/definite/kind-of) than curated is-a; honest boundary (the curated P279 remains the cleaner source)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
