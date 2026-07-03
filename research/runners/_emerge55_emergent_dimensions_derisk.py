"""EMERGE-55 / EMERGENT DIMENSIONS — the per-dimension (Collins-Quillian) cancellation of EMERGE-54 no longer relies on a
HOST-SIDE property->dimension lexicon (`PROP_DIM`, the thing hand-listing that fly/walk/swim share a dimension while breathe
is separate). Instead the DIMENSION STRUCTURE is LEARNED from the STATISTICS OF EXPERIENCE, then fed into the SAME
per-dimension cancellation read. This burns down the last host-scaffolding shortcut disclosed honestly in EMERGE-54: which
properties are alternatives-on-one-dimension is now DISCOVERED, not hand-listed.

THE MECHANISM (learn dimensions from statistics; no dimension lexicon):
  Properties in the SAME dimension are ALTERNATIVES -- they are MUTUALLY EXCLUSIVE across members (a member has exactly ONE
  locomotion: a robin flies XOR a penguin walks XOR a trout swims XOR a pike lurks), and an EXCEPTION REPLACES the class
  default on that dimension. Properties in DIFFERENT dimensions CO-OCCUR freely (a member both flies AND breathes). So the
  grouping is discoverable from a property x property co-occurrence matrix built over the members: two properties that
  CO-OCCUR for some member are forced into DIFFERENT dimensions; two properties that are mutually-exclusive ALTERNATIVES (the
  same member never has both, but members that lose one to an exception gain another) fall into the SAME dimension. This is
  the same PPMI/co-occurrence family the project already uses (stream-cortex co-occurrence learning); biology: competitive /
  lateral-inhibition grouping of mutually-exclusive alternatives -- a "which one" WTA per dimension (Numenta semantic folding
  of alternatives; Bates-MacWhinney competition). We learn the grouping by: (1) build the member x property matrix from the
  taught statistics (class defaults inherited by members, replaced per-member by exceptions); (2) two properties CONFLICT
  (must be different dimensions) iff they co-occur for a member; (3) properties are ALTERNATIVES (same dimension) iff they
  never co-occur AND they substitute in the same slot (a member has one where a sibling-population member has the other);
  (4) connected components of the "alternates-with" graph = the discovered dimensions. Each discovered dimension is exactly a
  set of mutually-exclusive alternatives -- the "which one" WTA group.

Then the discovered grouping REPLACES `PROP_DIM` in the EMERGE-54 per-dimension read: an exception cancels the class default
ONLY within its LEARNED dimension; other LEARNED dimensions inherit untouched.

  you> ... (observe members + speak the taxonomy + teach class props by dimension + LOCOMOTION exceptions, as EMERGE-54) ...
  brain LEARNS: dimension {fly, walk, swim, lurk} (locomotion alternatives) and dimension {breathe} (respiration), from stats
  you> can a penguin fly?     brain> No, a penguin walks.        (LEARNED-LOCOMOTION overridden -- exception wins ITS dim)
  you> can a penguin breathe? brain> Yes, a penguin can breathe. (LEARNED-RESPIRATION INHERITED -- the exception does NOT leak)
  you> can a robin fly?       brain> Yes, a robin can fly.       (inherited, no exception)
  you> can a zzz breathe?     brain> I don't know what a zzz is. (no-confab MOAT)

DE-RISK GATES (3-seed 42/43/44):
  (1) DIMENSION-DISCOVERY accuracy: the LEARNED grouping matches the TRUE dimensions (fly/walk/swim/lurk together; breathe
      separate) -- a "same-dim pairs grouped, cross-dim pairs separated" score (== adjusted-rand for a 2-way partition here)
      >= 0.9;
  (2) per-dimension cancellation with the LEARNED dimensions still holds ('can a penguin fly?' No AND 'can a penguin
      breathe?' Yes) -- BOTH must hold, exactly EMERGE-54's fix, now over LEARNED dimensions;
  (3) the no-confab MOAT abstains on unknown tokens (0 false-accepts);
  (4) LOAD-BEARING DESTROYED-EXCLUSIVITY control: destroy the mutual-exclusivity/co-occurrence statistics the grouping is
      learned from (truth-blind: give every member the full property vocabulary, so every property pair co-occurs) -> the
      learner splits every property into its OWN singleton dimension -> the exception's learned dimension no longer matches
      the asked property's (fly and walk are no longer grouped) -> per-dimension cancellation cannot gate and BREAKS every
      seed (the exception fails to cancel; 'can a penguin fly?' wrongly answers Yes). This proves the LEARNED grouping is
      doing the work, NOT a hard-wired fallback -- there is deliberately NO host `PROP_DIM` fallback. (A pure LABEL BIJECTION
      is NOT used because it merely renames properties and preserves the partition, so it never breaks the read -- the signal
      lives in the ASSIGNMENT, which member holds which property, not the names.)

`--demo` / `--script "..."` / interactive; `--derisk --seeds 42 43 44`. CPU numpy; reuse-by-import (EMERGE-54 console +
its EMERGE-52 machinery); NO `sim/` edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, re, time, traceback
from pathlib import Path
from itertools import combinations
import numpy as np

from research.runners._emerge54_per_dimension_cancellation_derisk import (
    PerDimensionConsole, FLOOR, _art, _lemma,
    _BIRDS, _FISH, _BIRD_SUBCAT, _FISH_SUBCAT, _BIRD_POOL, _FISH_POOL,
    _BIRD_HELDOUT, _FISH_HELDOUT, _member_feats, _BIRD_EXC, _FISH_EXC)

OUT = Path("research/findings/raw/_emerge55_emergent_dimensions.json")


# =====================================================================================================================
# The LEARNED dimension grouping (replaces the host PROP_DIM lexicon).
# =====================================================================================================================
def learn_dimensions(member_props):
    """LEARN the property->dimension grouping from the statistics of experience.

    `member_props`: {member_name -> set(property_lemma) that are TRUE for that member} (class defaults inherited by members,
    with each member's exception REPLACING the class default on the same slot).

    Returns (dim_of, dims) where dim_of maps a property lemma -> a discovered dimension id, and dims is {dim_id -> set(props)}.

    ALGORITHM (co-occurrence + competitive alternation; the same co-occurrence family the project uses):
      1. co-occurrence count C[p][q] = number of members for which BOTH p and q are true. C[p][q] > 0 => p,q are in
         DIFFERENT dimensions (a member cannot have two values of one dimension). This is the CONFLICT signal.
      2. Two properties are ALTERNATIVES (candidate SAME dimension) iff they NEVER co-occur (C == 0) AND both appear in the
         population (each is true for at least one member). The alternation is what makes them a WTA "which one" group: a
         member that has one does NOT have the other, and across the population they substitute for each other.
      3. Build the ALTERNATES graph: an edge p--q iff C[p][q] == 0 (they can share a dimension). The CONFLICT edges
         (C>0) forbid an edge. Take connected components under the alternates relation, BUT a component must be an
         'anti-clique' under conflict (internally conflict-free) -- which it is by construction, since an edge requires C==0.
         The connected components = the discovered dimensions.

    Why this separates {fly,walk,swim,lurk} from {breathe}: breathe co-occurs with EVERY locomotion (every member breathes
    AND has a locomotion) => breathe conflicts with all of them => breathe is isolated in its own dimension. fly/walk/swim/
    lurk never co-occur with each other (each member has exactly one locomotion) => they are mutually alternates => one
    connected component = the locomotion dimension.
    """
    props = sorted({p for s in member_props.values() for p in s})
    if not props:
        return {}, {}
    idx = {p: i for i, p in enumerate(props)}
    n = len(props)
    C = np.zeros((n, n), int)                    # co-occurrence counts
    present = np.zeros(n, int)                    # how many members have each property
    for s in member_props.values():
        ps = [idx[p] for p in s]
        for i in ps:
            present[i] += 1
        for i, j in combinations(ps, 2):
            C[i, j] += 1; C[j, i] += 1
    # ALTERNATES graph: edge iff the two properties NEVER co-occur (C==0) and both are present in the population.
    # (Two properties that co-occur are FORCED apart; two that never co-occur MAY share a dimension.)
    adj = [set() for _ in range(n)]
    for i, j in combinations(range(n), 2):
        if C[i, j] == 0 and present[i] > 0 and present[j] > 0:
            adj[i].add(j); adj[j].add(i)
    # connected components of the alternates graph = discovered dimensions.
    comp = [-1] * n; d = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        stack = [i]; comp[i] = d
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = d; stack.append(v)
        d += 1
    dim_of = {props[i]: comp[i] for i in range(n)}
    dims = {}
    for p, c in dim_of.items():
        dims.setdefault(c, set()).add(p)
    return dim_of, dims


class EmergentDimensionConsole(PerDimensionConsole):
    """EMERGE-54's per-dimension cancellation, but the property->dimension grouping is LEARNED from the taught statistics
    (not the host `PROP_DIM` lexicon). After the transcript is fed, call `fit_dimensions()`; the per-dimension read then
    resolves an exception's dimension + the asked property's dimension through the LEARNED grouping. Reuse-by-import: only
    `_dim_of` and the exception-dimension tag are re-sourced from the learned grouping; the pooler, teaching, and bridge are
    unchanged."""

    def __init__(self, *a, shuffle_labels=False, **kw):
        super().__init__(*a, **kw)
        self.shuffle_labels = bool(shuffle_labels)   # LOAD-BEARING control: destroy the mutual-exclusivity statistics
        self.learned_dim_of = None                   # property lemma -> learned dimension id (None until fit_dimensions())
        self.learned_dims = None

    # ---- build the member x property statistics, then LEARN the dimensions ------------------------------------------
    def _member_property_stats(self):
        """Reconstruct {member -> set(property_lemma TRUE for it)} PURELY from the taught statistics: each observed member
        inherits every class default of a class it belongs to, EXCEPT it substitutes its own exception on that slot. This is
        the experience the brain has (it heard these facts); the dimension learner sees only this co-occurrence matrix."""
        stats = {}
        for m in self.member_feats:
            props = set()
            # class defaults the member inherits (any class in its ancestor chain that taught a property)
            for cname in self._ancestors(m):
                if cname in self.class_prop:
                    props.add(_lemma(self.class_prop[cname]))
            # the member's own exception REPLACES the class default on its slot (a member that walks does not fly)
            if m in self.ovr_prop:
                exc = _lemma(self.ovr_prop[m])
                # remove the class default this exception replaces = the class LOCOMOTION default (any class prop that is,
                # by co-occurrence, in the same slot). We identify it structurally: the exception is taught on the member's
                # identity; the default it replaces is the one class property the member would otherwise have that the
                # exception is a substitute for. We approximate the substitution by the taxonomy: the member's most-specific
                # class default (the class default at the deepest ancestor level) is the one the exception overrides.
                default_to_replace = self._deepest_class_default(m)
                if default_to_replace is not None:
                    props.discard(default_to_replace)
                props.add(exc)
            stats[m] = props
        if self.shuffle_labels:
            stats = self._shuffle_property_labels(stats)
        return stats

    def _deepest_class_default(self, member):
        """The class default at the DEEPEST (most-specific) ancestor level that taught a property = the one an exception
        substitutes for. For the demo taxonomy: 'bird can fly' (genus, depth 1) is deeper than 'animal breathes' (order,
        depth 2), so a penguin's 'walks' replaces 'fly' (locomotion), NOT 'breathe' (respiration)."""
        best = None; best_depth = 10 ** 9
        anc = self._ancestors(member)
        for depth, cname in enumerate(anc):
            if cname in self.class_prop and depth < best_depth:
                best = _lemma(self.class_prop[cname]); best_depth = depth
        return best

    def _shuffle_property_labels(self, stats):
        """LOAD-BEARING control: DESTROY the mutual-exclusivity / co-occurrence statistics that carry the dimension signal,
        so the learned grouping is wrong -> per-dimension cancellation breaks. This proves the LEARNED grouping is doing the
        work (there is no host `PROP_DIM` fallback to mask a failure).

        NOTE why a global LABEL BIJECTION is the WRONG control (confirmed empirically): a bijection merely RENAMES properties
        and PRESERVES the partition, so it never breaks the read. The signal lives in the ASSIGNMENT (which member holds which
        property), not the names. A per-member random RE-DRAW breaks the assignment but is seed-fragile (a lucky draw can
        re-create a valid RESP/LOCO structure). So the control here destroys the exclusivity DETERMINISTICALLY and
        truth-blindly: give every member the FULL property vocabulary, so EVERY pair of properties now co-occurs. By the
        learner's rule, co-occurring properties are forced into DIFFERENT dimensions -> every property becomes its OWN
        singleton dimension -> no dimension groups the locomotion alternates, and the exception's learned dimension no longer
        matches (nor isolates) the asked property's -> the per-dimension cancellation cannot gate and BREAKS every seed. This
        uses NO dimension labels (truth-blind); it is the maximal destruction of the mutual-exclusivity statistics the
        grouping is learned from."""
        vocab = sorted({p for s in stats.values() for p in s})
        return {m: set(vocab) for m in stats}

    def fit_dimensions(self):
        """LEARN the property->dimension grouping from the current statistics and install it as the per-dimension read's
        grouping (replacing the host `PROP_DIM`)."""
        stats = self._member_property_stats()
        self.learned_dim_of, self.learned_dims = learn_dimensions(stats)
        # re-tag every recorded exception's dimension from the LEARNED grouping (overriding the EMERGE-54 host-lexicon tag)
        if not hasattr(self, "ovr_dim"):
            self.ovr_dim = {}
        for member, prop in self.ovr_prop.items():
            self.ovr_dim[member] = self._learned_dim_of(prop)
        return self.learned_dim_of, self.learned_dims

    def _learned_dim_of(self, prop):
        """The LEARNED dimension id of a property lemma (None if never learned -> the per-dimension read abstains, so the
        shuffled-labels control genuinely breaks rather than silently falling back to a host lexicon)."""
        if self.learned_dim_of is None:
            return None
        return self.learned_dim_of.get(_lemma(prop))

    # ---- override EMERGE-54's dimension resolution to use the LEARNED grouping ---------------------------------------
    def _dim_of(self, prop):   # instance method (EMERGE-54 used a module `_dim_of`); this console reads the LEARNED grouping
        return self._learned_dim_of(prop)

    def _exception_in_dim(self, member, dim):
        """True iff `member`'s exception is in LEARNED dimension `dim` AND fires above the floor (the only cancellation
        route; gated to the SAME learned dimension as the asked property, so a locomotion exception can't touch a
        respiration query). `dim` is a LEARNED dimension id here."""
        if member not in self.ovr_slot or self._ovr_dim(member) != dim:
            return False
        dr = self._drive(member)
        if not dr:
            return False
        v = dr.get(("OVR", member))
        return v is not None and v > FLOOR

    def ask_can(self, member, prop):
        """Answer 'can a <member> <prop>?' with PER-DIMENSION cancellation over the LEARNED dimensions. (1) if the member has
        an exception in the asked property's LEARNED dimension -> answer the exception; (2) else if a class the member
        belongs to teaches this property -> inherit (Yes); (3) else abstain (moat). Identical control flow to EMERGE-54, but
        `dim` comes from the LEARNED grouping, not the host lexicon."""
        if member not in self.member_idx:
            return f"I don't know what {_art(member)} is."
        dim = self._dim_of(prop)     # LEARNED dimension of the asked property
        if dim is not None and self._exception_in_dim(member, dim):
            ep = self.ovr_prop.get(member, prop)
            return f"No, {_art(member)} {ep}."
        cls = self._best_class_for_prop(member, prop)
        if cls is not None:
            return f"Yes, {_art(member)} can {_lemma(prop)}."
        return f"I don't know whether {_art(member)} can {_lemma(prop)}."


# ---- NL front end (reuse EMERGE-54's parsing; route through THIS console's learned-dimension ask) --------------------
_OBS = re.compile(r"(?:a|an)\s+(\w+)\s+has\s+(.+)", re.I)
_ISA = re.compile(r"(?:a|an)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)", re.I)
_ASK = re.compile(r"can\s+(?:a|an)\s+(\w+)\s+(\w+)\??", re.I)
_CAN = re.compile(r"(?:a|an)\s+(\w+)\s+can\s+(\w+)", re.I)
_EXC = re.compile(r"(?:a|an)\s+(\w+)\s+(\w+)\s*$", re.I)


def handle(console, line):
    line = line.strip()
    if not line:
        return None
    m = _ASK.search(line)
    if m:
        return console.ask_can(m.group(1).lower(), m.group(2).lower())
    m = _OBS.search(line)
    if m:
        feats = [w for w in re.split(r"[\s,]+", m.group(2).strip()) if w]
        return console.observe(m.group(1).lower(), [f.lower() for f in feats])
    m = _ISA.search(line)
    if m:
        return console.learn_isa(m.group(1).lower(), m.group(2).lower())
    m = _CAN.search(line)
    if m:
        return console.learn_class(m.group(1).lower(), m.group(2).lower())
    m = _EXC.search(line)
    if m:
        x, y = m.group(1).lower(), m.group(2).lower()
        is_class = any(x in console._ancestors(mm) for mm in console.member_feats)
        if is_class:
            return console.learn_class(x, y)
        return console.learn_exception(x, y)
    return "(say 'a X has f1 f2', 'a X is a Y', 'a CLASS can P', 'a ORDERCLASS P', 'a MEMBER EXCEPTION', or 'can a X P?')"


# ---- the scripted world: EMERGE-54's taxonomy + LOCOMOTION exceptions; the dimensions are LEARNED, not host-listed ---
def _script_lines(seed):
    obs, isa, teach, ask = [], [], [], []
    for b in _BIRDS:
        obs.append(("a %s has %s" % (b, " ".join(_member_feats(seed, b, _BIRD_POOL))), None))
        isa.append(("a %s is a %s" % (b, _BIRD_SUBCAT[b]), None))
    for f in _FISH:
        obs.append(("a %s has %s" % (f, " ".join(_member_feats(seed, f, _FISH_POOL))), None))
        isa.append(("a %s is a %s" % (f, _FISH_SUBCAT[f]), None))
    for sc in sorted(set(_BIRD_SUBCAT.values())):
        isa.append(("a %s is a bird" % sc, None))
    for sc in sorted(set(_FISH_SUBCAT.values())):
        isa.append(("a %s is a fish" % sc, None))
    isa.append(("a bird is an animal", None))
    isa.append(("a fish is an animal", None))
    teach.append(("a bird can fly", "GENUS property -- locomotion (LEARNED)"))
    teach.append(("a fish can swim", "GENUS property -- locomotion (LEARNED)"))
    teach.append(("an animal breathes", "ORDER property -- respiration (LEARNED separate)"))
    teach.append(("a %s %s" % _BIRD_EXC, "member EXCEPTION -- locomotion alternate (penguin walks)"))
    teach.append(("a %s %s" % _FISH_EXC, "member EXCEPTION -- locomotion alternate (pike lurks)"))
    ask.append(("can a %s fly?" % _BIRD_EXC[0], "LEARNED-locomotion overridden -> No, walks"))
    ask.append(("can a %s breathe?" % _BIRD_EXC[0], "LEARNED-respiration INHERITED -> Yes (the exception does NOT leak)"))
    ask.append(("can a %s swim?" % _FISH_EXC[0], "LEARNED-locomotion overridden -> No, lurks"))
    ask.append(("can a %s breathe?" % _FISH_EXC[0], "LEARNED-respiration INHERITED -> Yes"))
    ask.append(("can a %s fly?" % _BIRD_HELDOUT, "non-overridden inherits locomotion -> Yes"))
    ask.append(("can a %s breathe?" % _BIRD_HELDOUT, "non-overridden inherits respiration -> Yes"))
    ask.append(("can a %s swim?" % _FISH_HELDOUT, "non-overridden inherits locomotion -> Yes"))
    ask.append(("can a %s breathe?" % _FISH_HELDOUT, "non-overridden inherits respiration -> Yes"))
    ask.append(("can a %s swim?" % _BIRD_HELDOUT, "SIBLING-DISCRIM -- owl is a bird, not a fish -> abstain"))
    ask.append(("can a zzz breathe?", "MOAT -- never observed"))
    return obs, isa, teach, ask


def _feed(c, obs, isa, teach):
    for line, _ in obs:
        handle(c, line)
    for line, _ in isa:
        handle(c, line)
    for line, _ in teach:
        handle(c, line)


# ---- the TRUE dimensions (for scoring dimension-discovery accuracy only; NOT used by the read) -----------------------
TRUE_DIM = {"fly": "LOCO", "walk": "LOCO", "swim": "LOCO", "lurk": "LOCO", "breathe": "RESP"}


def _dimension_discovery_score(learned_dim_of):
    """Fraction of property-PAIRS whose SAME/DIFFERENT-dimension status the LEARNED grouping matches the TRUE dimensions.
    For a 2-way partition this equals the pair-counting rand-index component; == 1.0 iff the learned partition equals the
    true partition (up to relabeling). Only scores properties that are BOTH in the true lexicon AND were learned."""
    props = [p for p in TRUE_DIM if p in (learned_dim_of or {})]
    if len(props) < 2:
        return 0.0
    ok = tot = 0
    for a, b in combinations(props, 2):
        true_same = (TRUE_DIM[a] == TRUE_DIM[b])
        learned_same = (learned_dim_of[a] == learned_dim_of[b])
        ok += int(true_same == learned_same); tot += 1
    return ok / tot if tot else 0.0


def _check(seed=42, shuffle_labels=False, lesion=False):
    """Run the scripted transcript, LEARN the dimensions, then evaluate the per-dimension cancellation + discovery."""
    c = EmergentDimensionConsole(seed=seed, lesion=lesion, shuffle_labels=shuffle_labels)
    obs, isa, teach, _ = _script_lines(seed)
    _feed(c, obs, isa, teach)
    c.fit_dimensions()                                   # LEARN dimensions from the taught statistics
    be, fe = _BIRD_EXC[0], _FISH_EXC[0]
    hb, hf = _BIRD_HELDOUT, _FISH_HELDOUT
    disc = _dimension_discovery_score(c.learned_dim_of)
    # per-dimension cancellation (over LEARNED dimensions): overridden member answers its exception on its dim AND inherits
    # on a DIFFERENT dim.
    per_dim = float(np.mean([
        c.answers_exception_on_dim(be, "fly") and c.inherits_on_dim(be, "breathe"),
        c.answers_exception_on_dim(fe, "swim") and c.inherits_on_dim(fe, "breathe")]))
    override_loco = float(np.mean([c.answers_exception_on_dim(be, "fly"), c.answers_exception_on_dim(fe, "swim")]))
    inherit_other = float(np.mean([c.inherits_on_dim(be, "breathe"), c.inherits_on_dim(fe, "breathe")]))
    nonoverride = float(np.mean([
        c.inherits_on_dim(hb, "fly"), c.inherits_on_dim(hb, "breathe"),
        c.inherits_on_dim(hf, "swim"), c.inherits_on_dim(hf, "breathe")]))
    sib = float(np.mean([c.sibling_confusion(hb, "swim"), c.sibling_confusion(hf, "fly")]))
    moat_unknown = c.moat_abstains("zzz", "breathe")
    replies = {
        "penguin_fly": handle(c, "can a %s fly?" % be),
        "penguin_breathe": handle(c, "can a %s breathe?" % be),
        "owl_fly": handle(c, "can a %s fly?" % hb),
        "owl_breathe": handle(c, "can a %s breathe?" % hb),
        "moat_unknown": handle(c, "can a zzz breathe?"),
    }
    learned_str = {str(k): sorted(v) for k, v in (c.learned_dims or {}).items()}
    return c, {"discovery": disc, "per_dim_cancellation": per_dim, "override_locomotion": override_loco,
               "inherit_other_dim": inherit_other, "nonoverride_inherit": nonoverride, "sibling_confusion": sib,
               "moat_unknown": bool(moat_unknown), "learned_dims": learned_str, "replies": replies}


def _demo(seed=42):
    c = EmergentDimensionConsole(seed=seed)
    obs, isa, teach, ask = _script_lines(seed)
    print("\n=== EMERGE-55 EMERGENT DIMENSIONS -- the per-dimension cancellation grouping is LEARNED from the statistics of "
          "experience (mutually-exclusive alternates = one dimension), NOT a host lexicon; no transformer ===\n")
    print("  --- OBSERVE + speak the taxonomy + TEACH class props + LOCOMOTION exceptions (as EMERGE-54) ---")
    for line, _ in obs + isa:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    for line, why in teach:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    c.fit_dimensions()
    print("\n  --- LEARNED DIMENSIONS (from the property x member co-occurrence statistics; competitive alternates) ---")
    for did, props in sorted((c.learned_dims or {}).items()):
        print(f"    dimension {did}: {{{', '.join(sorted(props))}}}   (mutually-exclusive alternates)")
    print(f"    discovery accuracy vs true dimensions: {_dimension_discovery_score(c.learned_dim_of):.2f}\n")
    print("  --- ASK: an exception cancels ONLY its LEARNED dimension; other LEARNED dimensions INHERIT ---")
    for line, why in ask:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    print()
    return c


def _derisk_one(seed):
    c, ch = _check(seed)
    fa = sum(0 if c.moat_abstains(t, "breathe") else 1 for t in ("zzz", "qqq", "wobble"))
    # LOAD-BEARING control: DESTROY the mutual-exclusivity statistics -> the learned dimensions are wrong (each property its
    # own singleton dimension) -> per-dimension cancellation breaks (no host PROP_DIM fallback to mask it).
    _, chs = _check(seed, shuffle_labels=True)
    # secondary (reported): dAP-lesion collapses inheritance (the substrate the read relies on is load-bearing).
    _, chl = _check(seed, lesion=True)
    return {"seed": seed, "discovery": ch["discovery"], "per_dim_cancellation": ch["per_dim_cancellation"],
            "override_locomotion": ch["override_locomotion"], "inherit_other_dim": ch["inherit_other_dim"],
            "nonoverride_inherit": ch["nonoverride_inherit"], "sibling_confusion": ch["sibling_confusion"],
            "moat_unknown": bool(ch["moat_unknown"]), "moat_false_accepts": int(fa),
            "learned_dims": ch["learned_dims"],
            "shuffled_per_dim_cancellation": chs["per_dim_cancellation"], "shuffled_discovery": chs["discovery"],
            "lesion_nonoverride_inherit": chl["nonoverride_inherit"], "replies": ch["replies"]}


def _derisk(seeds):
    print("EMERGE-55 emergent dimensions de-risk: LEARN the property->dimension grouping from experience statistics "
          "(mutually-exclusive alternates), then drive EMERGE-54's per-dimension cancellation with the LEARNED grouping",
          flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            print(f"  [seed {s}] DISCOVERY {d['discovery']:.2f} {d['learned_dims']} | PER-DIM cancel "
                  f"{d['per_dim_cancellation']:.2f} (override-loco {d['override_locomotion']:.2f} + inherit-other "
                  f"{d['inherit_other_dim']:.2f}) | non-override inherit {d['nonoverride_inherit']:.2f} | sibling-conf "
                  f"{d['sibling_confusion']:.2f} | moat {int(d['moat_unknown'])}/FA {d['moat_false_accepts']} || "
                  f"(control) DESTROYED-EXCLUSIVITY per-dim {d['shuffled_per_dim_cancellation']:.2f} (discovery "
                  f"{d['shuffled_discovery']:.2f}) | dAP-lesion inherit {d['lesion_nonoverride_inherit']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        disc = float(np.mean([d["discovery"] for d in per]))
        per_dim = float(np.mean([d["per_dim_cancellation"] for d in per]))
        override_loco = float(np.mean([d["override_locomotion"] for d in per]))
        inherit_other = float(np.mean([d["inherit_other_dim"] for d in per]))
        nonoverride = float(np.mean([d["nonoverride_inherit"] for d in per]))
        sib = float(np.mean([d["sibling_confusion"] for d in per]))
        moat_unknown_all = all(d["moat_unknown"] for d in per)
        moat_fa = int(sum(d["moat_false_accepts"] for d in per))
        shuffled_per_dim = float(np.mean([d["shuffled_per_dim_cancellation"] for d in per]))
        lesion_nonoverride = float(np.mean([d["lesion_nonoverride_inherit"] for d in per]))
        # GO gate: (1) dimensions LEARNED accurately (>=0.9); (2) per-dimension cancellation holds over the LEARNED
        # dimensions (both override on its dim AND inherit on the other); (3) non-overridden inherit + moat + real sibling;
        # (4) LOAD-BEARING SHUFFLED-LABELS breaks per-dimension cancellation (real per_dim >= shuffled + 0.30). The dAP-lesion
        # collapse is a secondary substrate control.
        go = bool(disc >= 0.90 and per_dim >= 0.99 and nonoverride >= 0.99 and sib <= 0.05 and moat_unknown_all
                  and moat_fa == 0 and per_dim >= shuffled_per_dim + 0.30 and nonoverride >= lesion_nonoverride + 0.30)
        if go:
            verdict = (f"GO -- the per-dimension (Collins-Quillian) cancellation grouping is now EMERGENT: the brain LEARNS "
                       f"which properties are alternatives-on-one-dimension from the STATISTICS OF EXPERIENCE (mutually-"
                       f"exclusive alternates never co-occur -> one dimension; co-occurring properties -> different "
                       f"dimensions), and that LEARNED grouping REPLACES the host `PROP_DIM` lexicon in the read. "
                       f"DIMENSION-DISCOVERY {disc:.2f} (the learned grouping matches the true dimensions: fly/walk/swim/lurk "
                       f"together as locomotion alternates, breathe separate as respiration). PER-DIMENSION CANCELLATION over "
                       f"the LEARNED dimensions {per_dim:.2f}: the overridden member answers its exception on its LEARNED "
                       f"dimension (locomotion 'walks', not 'flies') AND INHERITS the class default on a DIFFERENT LEARNED "
                       f"dimension (respiration 'breathes') -- both hold (override-locomotion {override_loco:.2f} + "
                       f"inherit-other {inherit_other:.2f}). Non-overridden members inherit on all learned dimensions "
                       f"({nonoverride:.2f}); the no-confab MOAT abstains ({moat_fa} false-accepts); sibling-discrimination "
                       f"stays real ({sib:.2f}). The LOAD-BEARING DESTROYED-EXCLUSIVITY control (give every member the full "
                       f"property vocabulary so the mutual-exclusivity statistics vanish) BREAKS the read (per-dim "
                       f"cancellation collapses to {shuffled_per_dim:.2f}): the LEARNED grouping is doing the work, NOT a "
                       f"hard-wired fallback (there is no host `PROP_DIM` fallback). dAP-lesion collapses inheritance to "
                       f"{lesion_nonoverride:.2f} (secondary substrate control). 3-seed. => the per-dimension structure is now "
                       f"EMERGENT (learned from experience), NOT host-listed; one spiking brain, NO sim/ edit.")
        else:
            miss = []
            if disc < 0.90: miss.append(f"dimension-discovery {disc:.2f} < 0.90 (learned grouping != true dimensions)")
            if per_dim < 0.99:
                miss.append(f"per-dimension cancellation {per_dim:.2f} < 0.99 (override-loco {override_loco:.2f}, "
                            f"inherit-other {inherit_other:.2f})")
            if nonoverride < 0.99: miss.append(f"non-override inheritance {nonoverride:.2f} < 0.99")
            if sib > 0.05: miss.append(f"sibling-confusion {sib:.2f} > 0.05")
            if not moat_unknown_all: miss.append("moat did not abstain on an unknown token")
            if moat_fa != 0: miss.append(f"moat false-accepts {moat_fa} != 0")
            if per_dim < shuffled_per_dim + 0.30:
                miss.append(f"destroyed-exclusivity control didn't break per-dim cancellation ({per_dim:.2f} vs control "
                            f"{shuffled_per_dim:.2f}) -- the learned grouping may not be load-bearing")
            if nonoverride < lesion_nonoverride + 0.30:
                miss.append(f"dAP-lesion didn't collapse inheritance ({nonoverride:.2f} vs {lesion_nonoverride:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". The dimension learner clusters properties by "
                       "co-occurrence (co-occur -> different dimension; never-co-occur alternates -> same dimension); if the "
                       "statistics don't separate the dimensions cleanly at this scale, the specific gap + next step: inspect "
                       "the learned_dims per seed vs the true grouping; the exception's substitution-of-the-class-default (the "
                       "deepest-class-default heuristic) is the load-bearing statistics step. Not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge55_emergent_dimensions", "verdict": verdict,
               "mechanism": "LEARN the property->dimension grouping from the statistics of experience: build the member x "
                            "property matrix from the taught facts (class defaults inherited by members, each member's "
                            "exception REPLACING the class default on its slot); two properties that CO-OCCUR for a member are "
                            "forced into DIFFERENT dimensions, two that are mutually-exclusive ALTERNATES (never co-occur, both "
                            "present) share a dimension; connected components of the alternates graph = the discovered "
                            "dimensions (each a 'which one' WTA group of mutually-exclusive alternatives). That LEARNED grouping "
                            "REPLACES the host PROP_DIM lexicon in EMERGE-54's per-dimension cancellation read (an exception "
                            "cancels only its LEARNED dimension; other LEARNED dimensions inherit). Same co-occurrence family "
                            "the project uses (stream-cortex co-occurrence); biology: competitive/lateral-inhibition grouping "
                            "of mutually-exclusive alternatives. Reuse-by-import (EMERGE-54 console); NO sim/ edit.",
               "task": "observe members + speak the taxonomy + teach class props by dimension + LOCOMOTION exceptions -> LEARN "
                       "the dimensions from statistics -> drive EMERGE-54's per-dimension cancellation with the LEARNED "
                       "grouping; gates: dimension-discovery accuracy + per-dimension cancellation over learned dims + moat + "
                       "LOAD-BEARING destroyed-exclusivity control (breaks the read) + secondary dAP-lesion collapse; 3-seed",
               "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "burns down the EMERGE-54 host `PROP_DIM` lexicon: the property->dimension grouping is now LEARNED "
                              "from experience statistics (mutually-exclusive alternates), NOT hand-listed. The learner sees "
                              "ONLY the member x property co-occurrence matrix reconstructed from the taught facts (class "
                              "defaults + per-member exception substitution). The read, teaching, discovery pooler, and bridge "
                              "are unchanged from EMERGE-54 (reuse-by-import). There is deliberately NO host PROP_DIM fallback: "
                              "an unlearned/wrong dimension makes the exception's dimension not match the asked property's, so "
                              "the DESTROYED-EXCLUSIVITY control (give every member the full property vocab -> every property a "
                              "singleton dimension) genuinely breaks per-dimension cancellation, proving the learned grouping is "
                              "load-bearing. A pure label BIJECTION is NOT used as the control: it merely renames properties and "
                              "preserves the partition (the signal lives in the assignment, not the names), so it never breaks. "
                              "The "
                              "one host-side statistics heuristic is the exception-substitutes-the-deepest-class-default step "
                              "(which class default an exception replaces on its slot) -- a taxonomy-depth rule, not a dimension "
                              "lexicon; corpus-scale dimension discovery (many dimensions, learned substitution) is a follow-on. "
                              "TRUE_DIM is used ONLY to SCORE discovery, never in the read."}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge55] VERDICT: {verdict}", flush=True)
    print(f"[emerge55] wrote {OUT}\n" + "=" * 108, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--script", default=None)
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    if a.demo:
        _demo(a.seed); return 0
    c = EmergentDimensionConsole(seed=a.seed)
    print("emergent-dimensions console -- observe: 'a X has f1 f2'; taxonomy: 'a X is a Y'; class: 'a CLASS can P' + "
          "'a ORDERCLASS P'; exception: 'a MEMBER WORD'; ask: 'can a X P?'  (dimensions are LEARNED; Ctrl-D to exit)")
    if a.script:
        for line in a.script.split(";"):
            r = handle(c, line)
            if r is not None:
                print(f"  you> {line.strip()}\n  brain> {r}")
        c.fit_dimensions()
        print(f"  [learned dimensions: { { str(k): sorted(v) for k, v in (c.learned_dims or {}).items() } }]")
        return 0
    try:
        while True:
            r = handle(c, input("you> "))
            if r is not None:
                print(f"brain> {r}")
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
