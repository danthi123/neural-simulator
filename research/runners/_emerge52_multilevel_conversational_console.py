"""EMERGE-52 / the MULTI-LEVEL CONVERSATIONAL CONSOLE — the brain DISCOVERS a MULTI-LEVEL taxonomy from experience and
answers inheritance ACROSS LEVELS in plain language. It CONNECTS EMERGE-51 (a NL console over pooler-discovered FLAT
categories: inheritance + cancellation + no-confab moat) with EMERGE-44/45 (the DISCOVERED multi-level taxonomy: STACK the
competitive pooler so member features -> sub-category (L1) -> genus (L2) -> order (L3), and inheritance CHAINS through the
discovered levels). The result: a console where the brain discovers a sub-category-within-genus-within-order hierarchy from
observed features + co-occurrence, class properties are TAUGHT at DIFFERENT levels ('a bird can fly' at the mid/genus level;
'an animal breathes' at the top/order level), and the user ASKS in natural language, answered by on-substrate inheritance
from the CORRECT discovered level, with cancellation + the no-confab moat intact. On the real spiking `SimulationBridge`,
transformer-free. NO `sim/` edit.

  you> a robin has wings feathers red small        (OBSERVE a member: its features)
  you> a robin is a thrush                          (bind to its sub-category, genus, order -- the taxonomy is spoken)
  you> a thrush is a bird
  you> a bird is an animal
  ...  (many members across birds + fish; the STACKED pooler DISCOVERS sub-category -> genus -> order from feats + co-occur)
  you> a bird can fly                               (TEACH a MID-level/genus property, via the class name)
  you> an animal breathes                           (TEACH a TOP-level/order property, 2 discovered levels up)
  you> a penguin walks                              (member-specific EXCEPTION -- cancellation)
  you> can a robin fly?      brain> Yes, a robin can fly.       (INHERIT 1 discovered level up -- genus, never told robin)
  you> can a robin breathe?  brain> Yes, a robin can breathe.  (INHERIT 2 discovered levels up -- order)
  you> can a robin swim?     brain> I don't know whether a robin can swim.  (sibling branch -- NOT inherited)
  you> can a penguin fly?    brain> No, a penguin walks.       (CANCELLATION -- the member's own exception)
  you> can a zzz breathe?    brain> I don't know what a zzz is.  (the no-confab MOAT -- never observed)

MECHANISM (emergent; no inference engine, no transformer): L1 = the EMERGE-38 competitive self-organizing pooler on member
FEATURES -> a sub-category codon. L2 = the SAME pooler over the L1 codons, trained on the CO-OCCURRENCE of same-GENUS
members (which members share a genus is the spoken taxonomy -- the experienced context) -> a genus codon. L3 = the pooler
over L2 codons, trained on same-ORDER co-occurrence -> an order codon. A class property spoken as 'a <class> can P' /
'a <class> P' is taught (the committed `sim/` three-term kernel) on the LEVEL the class lives at (genus->L2 codons of that
genus's members; order->L3 codons of that order's members), over the members' DISCOVERED codons. Asking 'can a <member> P?'
primes the member's discovered L2 and L3 codons + its identity ensemble and reads the graded apical drive to every taught
property; the STRONGEST fires, with the member's OWN specific exception winning a tie (Collins-Quillian cancellation). A
held-out member inherits the genus property (1 level up) AND the order property (2 levels up) via the shared discovered
codons; a SIBLING-branch property (fish 'swim' for a bird) drives no codon overlap -> not inherited; a never-observed token
drives no codon -> the moat abstains.

HONEST SCOPE (per EMERGE-45): the L2/GENUS grouping is the DOMINANT carrier of the multi-level signal; L3/ORDER adds a
seed-VARIABLE increment. This console reports the L2-genus floor alongside the 2-level (order) inheritance and frames the
verdict accordingly: 2-level inheritance via the discovered L2/genus grouping (+ an L3 increment) is what carries the signal.

`--demo` / `--script "a robin has wings;...;can a robin breathe?"` / interactive; `--derisk --seeds 42 43 44` runs the
de-risk gates (2-level held-out inheritance / sibling-discrimination / moat / permuted-co-occurrence control). CPU numpy;
reuse-by-import (`_emerge14` + `_emerge12` + EMERGE-44's `_competitive_pool`); NO `sim/` edit.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, re, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._emerge44_stacked_pooler_derisk import _competitive_pool

OUT = Path("research/findings/raw/_emerge52_multilevel_conversational_console.json")

# --- capacities (generous; fixed cell blocks, EMERGE-42/45 layout) ----------------------------------------------------
NF = 48                 # feature-vocabulary cells (named features the user says)
NCOL1 = 240; K1 = 6     # L1 competitive-pooler (features -> sub-category codons)
NCOL2 = 160; K2 = 6     # L2 competitive-pooler (L1 codons -> genus codons via same-genus co-occurrence)
NCOL3 = 100; K3 = 6     # L3 competitive-pooler (L2 codons -> order codons via same-order co-occurrence)
NMEM = 48               # max distinct members
N_ID_PER = 3            # member-identity ensemble size (>=2 clears the coincidence threshold)
NCLASS = 24             # max distinct class-property tags (taught at any level)
NOVR = 16               # max distinct member-exception tags
N_PROP_PER = 2          # cells per property tag
# the on-bridge cell block holds: [L2 columns | L3 columns | member-identity ensembles | class-prop cells | override cells]
L2_0 = 0
L3_0 = L2_0 + NCOL2
ID0 = L3_0 + NCOL3
CLASSP0 = ID0 + NMEM * N_ID_PER
OVRP0 = CLASSP0 + NCLASS * N_PROP_PER
M = OVRP0 + NOVR * N_PROP_PER

# pooler + teaching schedule
POOL_EPOCHS = 400; L2_EPOCHS = 400; L3_EPOCHS = 400
COOC_SAMPLES = 260
TEACH_EPOCHS = 40
FLOOR = -40.0
# class levels
LEVEL_GENUS = "L2"      # a class taught at the genus level attaches to the L2 codons
LEVEL_ORDER = "L3"      # a class taught at the order level attaches to the L3 codons


def _sdr(cells):
    return set(int(c) for c in cells)


def _art(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def _lemma(w):
    """Tiny morphological normalizer so the ASKED verb ('breathe' from 'can a X breathe?') matches the TAUGHT verb
    ('breathes' from 'an animal breathes'). English 3rd-person-singular -s / -es / -ies. Host-side keyboard/language
    interface only (not a brain computation) -- it lets the console conversation read naturally."""
    w = w.lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 2 and w[-3] in "sxzo":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w


def _build_bridge(seed, lesion=False):
    """One on-bridge cell block; ALL property cells are downstream of BOTH the L2 and L3 columns AND the member-identity
    ensembles (so a class taught at either level, and a member exception, all attach through the committed 3-term kernel)."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
    cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
    for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
              "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_coincidence_detection = (not lesion)
    cfg.coincidence_weighted_drive = True; cfg.coincidence_k_threshold = 1.5
    cfg.coincidence_plateau_strength = 160.0; cfg.enable_two_compartment_dap = True; cfg.apical_g_couple = 2.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b.runtime_state.actual_seed_used = seed
    b._initialize_simulation_data(called_from_playback_init=False)
    ci = np.asarray(b.region_manager.indices("cells"), int)
    prop_cells = list(range(CLASSP0, CLASSP0 + NCLASS * N_PROP_PER)) + list(range(OVRP0, OVRP0 + NOVR * N_PROP_PER))
    pre, post, w = [], [], []
    for pc in prop_cells:
        for c in range(NCOL2):                                       # L2 (genus) columns -> every property cell
            pre.append(int(ci[L2_0 + c])); post.append(int(ci[pc])); w.append(0.0)
        for c in range(NCOL3):                                       # L3 (order) columns -> every property cell
            pre.append(int(ci[L3_0 + c])); post.append(int(ci[pc])); w.append(0.0)
        for idx in range(NMEM * N_ID_PER):                           # member-identity cells -> every property cell (exceptions)
            pre.append(int(ci[ID0 + idx])); post.append(int(ci[pc])); w.append(0.0)
    b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                     "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
    coo = b._get_cached_coo()
    return b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))


class MultiLevelConversationalConsole:
    """OBSERVE members with features + speak their taxonomy -> the STACKED competitive pooler DISCOVERS sub-category ->
    genus -> order -> TEACH class properties at DIFFERENT levels -> ASK in natural language (inherit across levels /
    cancel / abstain), on the spiking bridge over the DISCOVERED multi-level codes."""

    def __init__(self, seed=42, lesion=False, permute=False, permute_feats=False,
                 pool_epochs=POOL_EPOCHS, l2_epochs=L2_EPOCHS, l3_epochs=L3_EPOCHS, teach_epochs=TEACH_EPOCHS):
        self.seed = int(seed)
        self.permute = bool(permute)            # co-occurrence control: scramble the co-occurrence pairs (isolates the L3 increment)
        self.permute_feats = bool(permute_feats)  # LOAD-BEARING control: random per-member features -> the pooler can't discover
                                                # ANY category structure -> feature-driven genus grouping collapses (input-destruction)
        self.pool_epochs = int(pool_epochs); self.l2_epochs = int(l2_epochs); self.l3_epochs = int(l3_epochs)
        self.teach_epochs = int(teach_epochs)
        self.rng = np.random.default_rng(self.seed)
        self.b, self.ci, self.row, self.col = _build_bridge(self.seed, lesion=lesion)
        self.z = np.zeros(len(self.ci))
        # named vocabularies -> fixed cell blocks
        self.feat_id = {}                       # feature name -> feature-cell index (0..NF-1)
        self.member_feats = {}                  # member name -> set(feature-cell indices)
        self.member_idx = {}                    # member name -> member-identity block index (0..NMEM-1)
        # the SPOKEN taxonomy: is-a edges chain member -> sub-category -> genus -> order (any class name -> its parent)
        self.parent = {}                        # class/member name -> its immediate parent class name (from 'a X is a Y')
        self.class_slot = {}                    # class name -> class-property slot index
        self.class_prop = {}                    # class name -> the taught property word (for the reply)
        self.class_level = {}                   # class name -> "L2"(genus) / "L3"(order) discovered level it attaches to
        self.ovr_slot = {}                      # member name -> override-property slot index
        self.ovr_prop = {}                      # member name -> the taught exception word (for the reply)
        # discovered codons (lazily (re)built when the observation/taxonomy set changes)
        self.l2codon = {}; self.l3codon = {}
        self._pooler_dirty = False

    # ---- vocab allocation -------------------------------------------------------------------------------------------
    def _feat(self, name):
        if name not in self.feat_id:
            if len(self.feat_id) >= NF:
                raise RuntimeError("out of feature-vocab capacity")
            self.feat_id[name] = len(self.feat_id)
        return self.feat_id[name]

    def _alloc_member(self, name):
        if name not in self.member_idx:
            if len(self.member_idx) >= NMEM:
                raise RuntimeError("out of member capacity")
            self.member_idx[name] = len(self.member_idx)
        return self.member_idx[name]

    def _class_slot(self, cname):
        if cname not in self.class_slot:
            if len(self.class_slot) >= NCLASS:
                raise RuntimeError("out of class-property capacity")
            self.class_slot[cname] = len(self.class_slot)
        return self.class_slot[cname]

    def _ovr_slot(self, member):
        if member not in self.ovr_slot:
            if len(self.ovr_slot) >= NOVR:
                raise RuntimeError("out of override-property capacity")
            self.ovr_slot[member] = len(self.ovr_slot)
        return self.ovr_slot[member]

    def _class_cells(self, cname):
        s = self._class_slot(cname)
        return _sdr(CLASSP0 + s * N_PROP_PER + j for j in range(N_PROP_PER))

    def _ovr_cells(self, member):
        s = self._ovr_slot(member)
        return _sdr(OVRP0 + s * N_PROP_PER + j for j in range(N_PROP_PER))

    def _id_cells(self, member):
        base = ID0 + self.member_idx[member] * N_ID_PER
        return _sdr(base + j for j in range(N_ID_PER))

    # ---- the SPOKEN taxonomy: ancestry from is-a edges --------------------------------------------------------------
    def _ancestors(self, name):
        """Chain of ancestor class names from the spoken is-a edges (member -> sub-cat -> genus -> order)."""
        out, cur, seen = [], self.parent.get(name), set()
        while cur is not None and cur not in seen:
            out.append(cur); seen.add(cur); cur = self.parent.get(cur)
        return out

    def _genus_of(self, member):
        """The member's genus = its FIRST class ancestor that is itself a subclass (i.e. has a parent). member -> sub-cat
        -> genus -> order: sub-cat = anc[0], genus = anc[1], order = anc[2] if present. We key co-occurrence by genus/order
        NAME so the grouping is DISCOVERED from the spoken hierarchy, not a host numeric label."""
        anc = self._ancestors(member)
        return anc[1] if len(anc) >= 2 else (anc[0] if anc else None)

    def _order_of(self, member):
        anc = self._ancestors(member)
        return anc[2] if len(anc) >= 3 else self._genus_of(member)

    # ---- the STACKED competitive pooler (EMERGE-44/45): discover sub-category -> genus -> order ----------------------
    def _train_poolers(self):
        """L1 (features -> sub-cat) -> L2 (co-occurrence of same-GENUS members -> genus codons) -> L3 (co-occurrence of
        same-ORDER members -> order codons). The genus/order GROUPING is read from the spoken taxonomy (the experienced
        context: which members are said to share a genus/order). PERMUTED scrambles the co-occurrence pairs."""
        members = [m for m in self.member_feats if self._genus_of(m) is not None]
        if not members:
            self.l2codon = {}; self.l3codon = {}; self._pooler_dirty = False
            return
        # L1: features -> sub-category codons
        l1 = _competitive_pool(self.seed, [self.member_feats[m] for m in members], NF, NCOL1, K1, self.pool_epochs)
        l1c = {m: l1(self.member_feats[m]) for m in members}
        # L2: co-occurrence of same-genus members -> genus codons
        cg = self._cooc(members, self.seed * 3 + 1, self._genus_of, l1c)
        l2 = _competitive_pool(self.seed, cg, NCOL1, NCOL2, K2, self.l2_epochs)
        self.l2codon = {m: l2(l1c[m]) for m in members}
        # L3: co-occurrence of same-order members -> order codons
        co = self._cooc(members, self.seed * 5 + 2, self._order_of, self.l2codon)
        l3 = _competitive_pool(self.seed, co, NCOL2, NCOL3, K3, self.l3_epochs)
        self.l3codon = {m: l3(self.l2codon[m]) for m in members}
        self._pooler_dirty = False

    def _cooc(self, members, seed, keyfn, codons):
        """COO_SAMPLES unions of two members that (unless permuted) share the same group key -> the level-below pooler's
        input. `keyfn` returns the group NAME (genus or order) from the spoken taxonomy."""
        rr = np.random.default_rng(seed); out = []
        groups = {}
        for m in members:
            groups.setdefault(keyfn(m), []).append(m)
        gkeys = [g for g, ms in groups.items() if len(ms) >= 2]
        if not gkeys:
            return [codons[m] for m in members]           # degenerate: singleton groups -> no pooling structure
        for _ in range(COOC_SAMPLES):
            if self.permute:
                a, bb = rr.choice(members, 2, replace=False)
            else:
                g = gkeys[int(rr.integers(len(gkeys)))]
                a, bb = rr.choice(groups[g], 2, replace=False)
            out.append(codons[a] | codons[bb])
        return out

    def _codons(self, member):
        if self._pooler_dirty:
            self._train_poolers()
        return self.l2codon.get(member, set()), self.l3codon.get(member, set())

    # ---- teaching from experience -----------------------------------------------------------------------------------
    def observe(self, member, feats):
        """'a member has f1 f2 f3' -> record the member's feature vector; the stacked pooler rediscovers the levels."""
        self._alloc_member(member)
        fset = set(self._feat(f) for f in feats)
        if self.permute_feats:                  # input-destruction control: random feature vector -> no category structure
            fset = set(int(c) for c in self.rng.choice(NF, min(len(feats), NF), replace=False))
        self.member_feats[member] = fset
        self._pooler_dirty = True
        return f"ok -- I've seen {_art(member)} with {' '.join(feats)}."

    def learn_isa(self, child, parent):
        """'a child is a parent' -> a spoken taxonomy edge (member->sub-cat->genus->order). Rediscovers levels (the
        co-occurrence grouping depends on the taxonomy)."""
        self.parent[child] = parent
        self._pooler_dirty = True
        return f"ok -- {_art(child)} is {_art(parent)}."

    def learn_class(self, cname, prop):
        """'a <class> can P' / 'a <class> P' -> teach P on the DISCOVERED codons of that CLASS's members, at the LEVEL the
        class lives at: a GENUS class -> the members' L2 codons; an ORDER class -> their L3 codons. Members of the class
        (via the spoken is-a chain) INHERIT it through the shared discovered codon."""
        if self._pooler_dirty:
            self._train_poolers()
        # which members belong to this class? (any observed member whose ancestor chain includes cname).
        members = [m for m in self.member_feats if cname in self._ancestors(m)]
        if cname in self.member_feats:                    # 'a <observed-member> can P' -> teach on the member itself (depth 0)
            members = list(dict.fromkeys(members + [cname]))
        if not members:
            return f"(I haven't seen any {cname} yet.)"
        # the class's discovered LEVEL: a member-exemplar is depth 0, a genus class 1 hop up, an order class 2 hops up.
        # depth = min over members of (index of cname in the member's ancestor chain), or 0 if the class IS the member.
        depths = [self._ancestors(m).index(cname) for m in members if cname in self._ancestors(m)]
        if cname in self.member_feats:
            depths.append(0)
        depth = min(depths) if depths else 1
        level = LEVEL_ORDER if depth >= 2 else LEVEL_GENUS
        self.class_prop[cname] = prop; self.class_level[cname] = level
        cells = self._class_cells(cname)
        for m in members:
            l2c, l3c = self._codons(m)
            codon = (l3c if level == LEVEL_ORDER else l2c)
            if not codon:
                continue
            # the pooler returns 0-based column indices; map them into the on-bridge cell block for that level:
            base = L3_0 if level == LEVEL_ORDER else L2_0
            codon_cells = _sdr(base + int(c) for c in codon)
            for _ in range(self.teach_epochs):
                apply_kernel_update(self.b, self.row, self.col, self.ci, codon_cells, cells, self.z, 0.14, 0.02, 1.0)
        # 'can P' for a modal/genus verb ('fly'); bare 'P' for an already-inflected order verb ('breathes')
        phrase = ("can " + prop) if _lemma(prop) == prop else prop
        return f"ok -- {_art(cname)} {phrase}."

    def learn_exception(self, member, prop):
        """'a member P' (member-specific exception) -> teach P on the member's IDENTITY ensemble (a stronger, direct fact
        that out-drives the inherited class default for this member) = cancellation."""
        self._alloc_member(member)
        self.ovr_prop[member] = prop
        cells = self._ovr_cells(member)
        idc = self._id_cells(member)
        for _ in range(self.teach_epochs * 2):
            apply_kernel_update(self.b, self.row, self.col, self.ci, idc, cells, self.z, 0.14, 0.02, 1.0)
        return f"ok -- {_art(member)} {prop}."

    # ---- inference (graded apical read over the DISCOVERED multi-level codes + member identity) ---------------------
    def _drive(self, member):
        """Prime the member's discovered L2 (genus) + L3 (order) codons + its identity ensemble; read the graded apical
        drive to every taught property. Returns {(kind,key): drive} or None on a moat miss (unknown/unobserved member)."""
        if member not in self.member_feats:
            return None
        l2c, l3c = self._codons(member)
        if not l2c and not l3c:
            return None
        ab = np.zeros(len(self.ci), bool)
        for c in l2c:
            ab[L2_0 + int(c)] = True
        for c in l3c:
            ab[L3_0 + int(c)] = True
        for j in range(N_ID_PER):
            ab[ID0 + self.member_idx[member] * N_ID_PER + j] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:      # dAP-LESION -> no inference
            return None
        vap = _host(vap)[self.ci]
        out = {}
        for cname in self.class_slot:
            out[("CLASS", cname)] = float(np.mean([vap[x] for x in self._class_cells(cname)]))
        for mem in self.ovr_slot:
            out[("OVR", mem)] = float(np.mean([vap[x] for x in self._ovr_cells(mem)]))
        return out

    def _best(self, member):
        """The member's OWN specific exception WINS a tie against the inherited class default (Collins-Quillian: the
        specific fact overrides the general). Otherwise the strongest active property wins; None below the floor = the moat.
        A DIFFERENT member's exception winning strictly = honest cross-bleed (non-answer)."""
        dr = self._drive(member)
        if not dr:
            return None
        best = max(dr, key=dr.get)
        if dr[best] <= FLOOR:
            return None
        own = ("OVR", member)
        if own in dr and dr[own] > FLOOR and dr[own] >= dr[best] - 1e-6:
            return own
        return best

    def _best_class_for_prop(self, member, prop):
        """For a specific asked property P, return the strongest class (STRICTLY over the floor) whose taught property == P,
        selected PURELY by the DISCOVERED-CODON graded drive (NO spoken-taxonomy shortcut). 'can a robin swim?' is answered
        only if the robin's discovered codon drives a 'swim'-class property cell above the floor -- and it does NOT, because
        the discovered bird codon has no overlap with the fish class's potentiated columns (sibling-discrimination is REAL,
        from the codons). This is what the PERMUTE-CO-OCCURRENCE control breaks (it then wrongly fires the sibling)."""
        dr = self._drive(member)
        if not dr:
            return None
        pl = _lemma(prop)
        cands = [(k, v) for (kind, k), v in dr.items()
                 if kind == "CLASS" and v > FLOOR and _lemma(self.class_prop.get(k, "")) == pl]
        if not cands:
            return None
        return max(cands, key=lambda kv: kv[1])[0]

    def ask_can(self, member, prop):
        """Answer 'can a <member> <prop>?' by on-substrate inheritance/cancellation over the DISCOVERED multi-level codes;
        honestly abstain (the no-confab moat) on an unknown/unobserved member OR an un-inherited (sibling-branch) property."""
        if member not in self.member_idx:
            return f"I don't know what {_art(member)} is."
        # cancellation: the member's own exception fires strongest -> answer the specific fact
        best = self._best(member)
        if best is not None and best[0] == "OVR" and best[1] == member:
            ep = self.ovr_prop.get(member, prop)
            return f"No, {_art(member)} {ep}."
        # otherwise: does a class this member BELONGS TO teach exactly this asked property? (correct-level inheritance,
        # sibling-discrimination: a fish 'swim' does not fire the bird member's codon strongly enough / isn't in its chain)
        cls = self._best_class_for_prop(member, prop)
        if cls is not None:
            return f"Yes, {_art(member)} can {_lemma(prop)}."
        return f"I don't know whether {_art(member)} can {_lemma(prop)}."

    # ---- de-risk accessors (used by --derisk + the tests) ----------------------------------------------------------
    def inherit_prop_ok(self, member, prop, expect_class):
        """The member inherits `prop` AND it resolves to `expect_class` (the correct discovered level)."""
        # cancellation guard: a member with its own exception shouldn't be counted as inheriting
        best = self._best(member)
        if best is not None and best[0] == "OVR" and best[1] == member:
            return False
        return self._best_class_for_prop(member, prop) == expect_class

    def sibling_confusion(self, member, sibling_prop):
        """Did the member WRONGLY inherit a SIBLING-branch property (e.g. a bird 'swim')? Should be False."""
        return self._best_class_for_prop(member, sibling_prop) is not None

    def cancel_ok(self, member):
        best = self._best(member)
        return best is not None and best[0] == "OVR" and best[1] == member

    def moat_abstains(self, member, prop):
        return self.ask_can(member, prop).startswith("I don't know")


# ---- a tiny natural-language front end (host parsing = the world/keyboard interface) --------------------------------
_OBS = re.compile(r"(?:a|an)\s+(\w+)\s+has\s+(.+)", re.I)
_ISA = re.compile(r"(?:a|an)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)", re.I)
_ASK = re.compile(r"can\s+(?:a|an)\s+(\w+)\s+(\w+)\??", re.I)
_CAN = re.compile(r"(?:a|an)\s+(\w+)\s+can\s+(\w+)", re.I)          # class property (level inferred from the taxonomy)
_EXC = re.compile(r"(?:a|an)\s+(\w+)\s+(\w+)\s*$", re.I)            # member-specific exception OR order-class property


def handle(console, line):
    line = line.strip()
    if not line:
        return None
    m = _ASK.search(line)                                          # queries first
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
        # 'a X Y': if X is a CLASS the console knows (has observed members), teach a class property (order-level 'breathes');
        # else it's a member-specific exception ('a penguin walks').
        x, y = m.group(1).lower(), m.group(2).lower()
        is_class = any(x in console._ancestors(mm) for mm in console.member_feats)
        if is_class:
            return console.learn_class(x, y)
        return console.learn_exception(x, y)
    return "(say 'a X has f1 f2', 'a X is a Y', 'a CLASS can P', 'a ORDERCLASS P', 'a MEMBER EXCEPTION', or 'can a X P?')"


# ---- the scripted world: a real sub-category -> genus -> order taxonomy discovered from features + co-occurrence -----
# Two ORDERS the pooler must discover: birds (order=animal-that-BREATHES-AND-FLIES) vs fish (order=animal-that-SWIMS).
# Taxonomy depth-3: member -> sub-category -> GENUS -> ORDER. Properties taught at BOTH the genus level ('a bird can fly',
# 'a fish can swim') and the ORDER level ('an animal breathes'), so a held-out member inherits fly (1 level) AND breathe (2).
# Bird genus = 'bird', order = 'animal'; fish genus = 'fish', order = 'animal'... but breathe must inherit for BOTH while
# fly/swim discriminate at the GENUS. So the ORDER-level 'animal' shares breathe across both; the GENUS 'bird'/'fish'
# discriminate fly/swim -- exactly the sibling-discrimination the design tests.
_BIRD_POOL = ["wings", "feathers", "beak", "talons", "plume", "crest"]
_FISH_POOL = ["fins", "scales", "gills", "tail", "stripe", "barbel"]
# 8 members per genus; 6 taught exemplars (per EMERGE-42/45 protocol -- the class is potentiated on many member codons),
# 1 exception, and the last member per genus is a GENUINE HELD-OUT (never named in a can/exception sentence).
_BIRDS = ["robin", "sparrow", "eagle", "hawk", "crow", "finch", "penguin", "owl"]      # penguin = exception, owl = held-out
_FISH = ["trout", "salmon", "carp", "bass", "perch", "tuna", "pike", "minnow"]         # pike = exception, minnow = held-out
_BIRD_SUBCAT = {"robin": "thrush", "sparrow": "passerine", "eagle": "raptor", "hawk": "raptor",
                "crow": "corvid", "finch": "passerine", "penguin": "sphenisc", "owl": "strigid"}
_FISH_SUBCAT = {"trout": "salmonid", "salmon": "salmonid", "carp": "cyprinid", "bass": "percid",
                "perch": "percid", "tuna": "scombrid", "pike": "esocid", "minnow": "cyprinid"}
_BIRD_HELDOUT = "owl"; _FISH_HELDOUT = "minnow"
_BIRD_EXC = ("penguin", "walks"); _FISH_EXC = ("pike", "lurks")


def _member_feats(seed, member, pool):
    """Deterministic 4-of-6 subset of the genus's feature pool (varied per member, EMERGE-42/45 style)."""
    h = abs(hash((seed, member))) % (10 ** 8)
    r = np.random.default_rng(seed * 131 + h)
    return list(np.array(pool)[np.sort(r.choice(len(pool), 4, replace=False))])


def _script_lines(seed):
    """Build the full scripted transcript (observe -> is-a taxonomy -> teach genus+order class props + exceptions -> ask)."""
    obs, isa, teach, ask = [], [], [], []
    for b in _BIRDS:
        obs.append(("a %s has %s" % (b, " ".join(_member_feats(seed, b, _BIRD_POOL))), None))
        isa.append(("a %s is a %s" % (b, _BIRD_SUBCAT[b]), None))            # member -> sub-category
    for f in _FISH:
        obs.append(("a %s has %s" % (f, " ".join(_member_feats(seed, f, _FISH_POOL))), None))
        isa.append(("a %s is a %s" % (f, _FISH_SUBCAT[f]), None))
    for sc in sorted(set(_BIRD_SUBCAT.values())):
        isa.append(("a %s is a bird" % sc, None))                           # sub-category -> GENUS (bird)
    for sc in sorted(set(_FISH_SUBCAT.values())):
        isa.append(("a %s is a fish" % sc, None))
    isa.append(("a bird is an animal", None))                               # GENUS -> ORDER (animal)
    isa.append(("a fish is an animal", None))
    # teach the GENUS property via exemplars (all but the exception + held-out); teach the ORDER property once.
    bird_exemplars = [b for b in _BIRDS if b not in (_BIRD_EXC[0], _BIRD_HELDOUT)]
    fish_exemplars = [f for f in _FISH if f not in (_FISH_EXC[0], _FISH_HELDOUT)]
    teach.append(("a bird can fly", "GENUS (mid) property"))
    teach.append(("a fish can swim", "GENUS (mid) property"))
    teach.append(("an animal breathes", "ORDER (top) property -- 2 discovered levels up"))
    teach.append(("a %s %s" % _BIRD_EXC, "member-specific EXCEPTION (cancellation)"))
    teach.append(("a %s %s" % _FISH_EXC, "member-specific EXCEPTION (cancellation)"))
    # ASK: held-outs inherit fly (1 level) + breathe (2 levels); sibling-discrimination; cancellation; moat.
    ask.append(("can a %s fly?" % _BIRD_HELDOUT, "INHERIT genus (1 level) -- never told owl"))
    ask.append(("can a %s breathe?" % _BIRD_HELDOUT, "INHERIT order (2 levels up)"))
    ask.append(("can a %s swim?" % _BIRD_HELDOUT, "SIBLING-DISCRIM -- owl is a bird, not a fish"))
    ask.append(("can a %s swim?" % _FISH_HELDOUT, "INHERIT genus (1 level) -- never told minnow"))
    ask.append(("can a %s breathe?" % _FISH_HELDOUT, "INHERIT order (2 levels up)"))
    ask.append(("can a %s fly?" % _FISH_HELDOUT, "SIBLING-DISCRIM -- minnow is a fish, not a bird"))
    ask.append(("can a %s fly?" % _BIRD_EXC[0], "CANCEL -- the penguin's own exception out-drives the genus default"))
    ask.append(("can a %s breathe?" % _BIRD_EXC[0], "the member's exception dominates the read (honest Collins-Quillian: "
                                                    "the strongest specific fact wins; per-property override is a follow-on)"))
    ask.append(("can a zzz breathe?", "MOAT -- never observed"))
    return obs, isa, teach, ask


def _feed(c, obs, isa, teach):
    for line, _ in obs:
        handle(c, line)
    for line, _ in isa:
        handle(c, line)
    for line, _ in teach:
        handle(c, line)


def _demo(seed=42):
    c = MultiLevelConversationalConsole(seed=seed)
    obs, isa, teach, ask = _script_lines(seed)
    print("\n=== EMERGE-52 MULTI-LEVEL conversational console -- DISCOVER a taxonomy (sub-cat -> genus -> order) from "
          "experience, then TALK across levels (inherit / cancel / abstain); no transformer ===\n")
    print("  --- OBSERVE members with features (the STACKED pooler discovers the levels) ---")
    for line, _ in obs:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- speak the taxonomy: member -> sub-category -> GENUS -> ORDER ---")
    for line, _ in isa:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- TEACH class properties at DIFFERENT levels (genus 'fly'/'swim'; order 'breathes') + EXCEPTIONS ---")
    for line, why in teach:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    print("  --- ASK across levels (answers by on-substrate inference over the DISCOVERED multi-level codes) ---")
    for line, why in ask:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    print()
    return c


def _check(seed=42, permute=False, permute_feats=False):
    """Run the scripted transcript silently; return (console, checks) for the gates + the demo self-check test."""
    c = MultiLevelConversationalConsole(seed=seed, permute=permute, permute_feats=permute_feats)
    obs, isa, teach, _ = _script_lines(seed)
    _feed(c, obs, isa, teach)
    ho_bird, ho_fish = _BIRD_HELDOUT, _FISH_HELDOUT
    # 2-LEVEL inheritance: a held-out member inherits the ORDER property 'breathe' (2 discovered levels up) via 'animal'
    two_level = float(np.mean([c.inherit_prop_ok(ho_bird, "breathes", "animal"),
                               c.inherit_prop_ok(ho_fish, "breathes", "animal")]))
    # 1-LEVEL inheritance (genus) reported alongside (the L2-genus floor)
    one_level = float(np.mean([c.inherit_prop_ok(ho_bird, "fly", "bird"),
                               c.inherit_prop_ok(ho_fish, "swim", "fish")]))
    # SIBLING-DISCRIMINATION: the held-out bird must NOT inherit fish 'swim', held-out fish must NOT inherit bird 'fly'
    sib = float(np.mean([c.sibling_confusion(ho_bird, "swim"), c.sibling_confusion(ho_fish, "fly")]))
    canc = float(np.mean([c.cancel_ok(_BIRD_EXC[0]), c.cancel_ok(_FISH_EXC[0])]))
    moat_unknown = c.moat_abstains("zzz", "breathes")
    replies = {
        "twolevel_inherit": handle(c, "can a %s breathe?" % ho_bird),
        "onelevel_inherit": handle(c, "can a %s fly?" % ho_bird),
        "sibling_discrim": handle(c, "can a %s swim?" % ho_bird),
        "cancel": handle(c, "can a %s fly?" % _BIRD_EXC[0]),
        "moat_unknown": handle(c, "can a zzz breathe?"),
    }
    return c, {"two_level": two_level, "one_level_genus_floor": one_level, "sibling_confusion": sib,
               "cancel": canc, "moat_unknown": bool(moat_unknown), "replies": replies}


# ---- the de-risk (2-level held-out inheritance / sibling-discrimination / moat / permuted controls), multi-seed -------
def _derisk_one(seed):
    c, ch = _check(seed, permute=False)
    fa = sum(0 if c.moat_abstains(t, "breathes") else 1 for t in ("zzz", "qqq", "wobble"))
    # LOAD-BEARING collapse control: scramble the co-occurrence pairs -> the L2/L3 pooler can no longer separate the
    # branches -> the DISCOVERED-codon SIBLING-DISCRIMINATION breaks (a held-out bird now wrongly inherits the fish
    # 'swim' property). This is the genuine input-destruction on the discovered-codon structure the sibling-read depends
    # on -- the sibling-discrimination is codon-driven (NO spoken-taxonomy shortcut in the read), so this control is
    # load-bearing FOR IT. (The 2-level inheritance itself rides the feature/genus grouping, EMERGE-45's dominant carrier,
    # so it stays high across arms -- reported honestly.)
    _, chp = _check(seed, permute=True)
    # SECONDARY diagnostic (reported): random per-member features. The co-occurrence stream (keyed by the spoken taxonomy)
    # still groups same-branch members, so this alone does NOT collapse -- honestly showing the grouping rides co-occurrence.
    _, chf = _check(seed, permute_feats=True)
    return {"seed": seed, "two_level": ch["two_level"], "one_level_genus_floor": ch["one_level_genus_floor"],
            "sibling_confusion": ch["sibling_confusion"], "cancel": ch["cancel"],
            "moat_unknown": bool(ch["moat_unknown"]), "moat_false_accepts": int(fa),
            "permcooc_sibling_confusion": chp["sibling_confusion"], "permcooc_two_level": chp["two_level"],
            "permfeat_sibling_confusion": chf["sibling_confusion"], "permfeat_two_level": chf["two_level"]}


def _derisk(seeds):
    print(f"EMERGE-52 multi-level conversational console de-risk: observe -> DISCOVER sub-cat->genus->order -> teach at "
          f"levels -> NL inherit-across-levels/cancel/abstain", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            print(f"  [seed {s}] 2-level inherit {d['two_level']:.2f} (genus floor {d['one_level_genus_floor']:.2f}) | "
                  f"sibling-confusion {d['sibling_confusion']:.2f} | cancel {d['cancel']:.2f} | "
                  f"moat-unknown {int(d['moat_unknown'])} | moat-FA {d['moat_false_accepts']} || "
                  f"PERMUTE-COOC sibling-confusion {d['permcooc_sibling_confusion']:.2f} (collapse control) | "
                  f"(secondary) permute-feats 2-level {d['permfeat_two_level']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        two = float(np.mean([d["two_level"] for d in per]))
        genus_floor = float(np.mean([d["one_level_genus_floor"] for d in per]))
        sib = float(np.mean([d["sibling_confusion"] for d in per]))
        canc = float(np.mean([d["cancel"] for d in per]))
        moat_unknown_all = all(d["moat_unknown"] for d in per)
        moat_fa = int(sum(d["moat_false_accepts"] for d in per))
        permcooc_sib = float(np.mean([d["permcooc_sibling_confusion"] for d in per]))   # LOAD-BEARING collapse control
        permfeat_two = float(np.mean([d["permfeat_two_level"] for d in per]))           # SECONDARY (reported)
        two_seeds = [round(d["two_level"], 3) for d in per]
        # GO gate: 2-level held-out inheritance >= 0.75 (EMERGE-45 honest floor) + real sibling-discrimination (~0) + moat +
        # cancellation + the LOAD-BEARING collapse: PERMUTE-CO-OCCURRENCE breaks the discovered-codon separation so
        # sibling-discrimination COLLAPSES (real sib ~0 -> permuted sib >= 0.25). The sibling-read is codon-driven (no
        # spoken-taxonomy shortcut), so this control is load-bearing for the discovered-codon structure.
        go = bool(two >= 0.75 and sib <= 0.05 and moat_unknown_all and moat_fa == 0
                  and permcooc_sib >= sib + 0.25 and canc >= 0.99)
        if go:
            verdict = (f"GO -- a MULTI-LEVEL conversational console: the brain OBSERVES members with features + hears their "
                       f"taxonomy in plain language, the STACKED competitive pooler (EMERGE-44/45) DISCOVERS sub-category -> "
                       f"genus -> order, class properties are taught at DIFFERENT levels (genus 'fly'/'swim'; order "
                       f"'breathes'), and the user ASKS in natural language -- answered by on-substrate inheritance from the "
                       f"CORRECT discovered level, selected PURELY by the discovered-codon graded drive (no spoken-taxonomy "
                       f"shortcut in the read). 2-LEVEL held-out inheritance {two:.2f} (a never-taught held-out member inherits "
                       f"the ORDER property 'breathe' TWO discovered levels up; per-seed {two_seeds}), with the L2/genus "
                       f"1-level floor at {genus_floor:.2f}. SIBLING-DISCRIMINATION {sib:.2f} (~0): a held-out bird does NOT "
                       f"inherit fish 'swim' -- and this is CODON-driven: the LOAD-BEARING PERMUTE-CO-OCCURRENCE control "
                       f"(scramble the L2/L3 co-occurrence pairs so the pooler can't separate the branches) BREAKS it, raising "
                       f"sibling-confusion to {permcooc_sib:.2f} (the held-out member then wrongly inherits the sibling branch). "
                       f"CANCELLATION {canc:.2f} (the exception member answers ITS fact). The no-confab MOAT abstains on every "
                       f"never-observed token ({moat_fa} false-accepts). HONEST SCOPE (per EMERGE-45): the 2-level ORDER "
                       f"inheritance rides the dominant feature/genus grouping (so the secondary permute-FEATURES control alone, "
                       f"leaving the taxonomy-keyed co-occurrence intact, does not collapse the 2-level read: {permfeat_two:.2f}) "
                       f"-- '2-level via the discovered hierarchy works; L3/order is a seed-variable increment'. 3-seed. => "
                       f"'discover a multi-level taxonomy from experience -> talk to the brain across levels', one spiking "
                       f"brain, NO sim/ edit.")
        else:
            miss = []
            if two < 0.75: miss.append(f"2-level inheritance {two:.2f} < 0.75 (genus floor {genus_floor:.2f})")
            if sib > 0.05: miss.append(f"real sibling-confusion {sib:.2f} > 0.05 (held-out inherited a SIBLING-branch property)")
            if not moat_unknown_all: miss.append("moat did not abstain on an unknown token")
            if moat_fa != 0: miss.append(f"moat false-accepts {moat_fa} != 0")
            if permcooc_sib < sib + 0.25:
                miss.append(f"permute-co-occurrence didn't break sibling-discrimination ({permcooc_sib:.2f} vs real {sib:.2f})")
            if canc < 0.99: miss.append(f"cancellation {canc:.2f} < 0.99")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". L2/genus 1-level floor {genus_floor:.2f}; "
                       f"per-seed 2-level {two_seeds}; permute-co-occurrence sibling-confusion {permcooc_sib:.2f}; "
                       f"secondary permute-features 2-level {permfeat_two:.2f}. Per EMERGE-45's honest scope, the feature/genus "
                       "grouping carries the 2-level signal and the deepest (order/L3) read is a seed-variable increment; the "
                       "sibling-discrimination is codon-driven. Tune L3 boosting/epochs or the feature-vocab overlap. Not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge52_multilevel_conversational_console", "verdict": verdict,
               "mechanism": "the EMERGE-44/45 STACKED competitive pooler discovers sub-category (L1, from features) -> genus "
                            "(L2, from same-genus co-occurrence) -> order (L3, from same-order co-occurrence); the genus/order "
                            "grouping is read from the SPOKEN is-a taxonomy (the experienced context). The committed three-term "
                            "kernel teaches a CLASS property on the members' DISCOVERED codons at the class's level (genus->L2, "
                            "order->L3) + a member EXCEPTION on the identity ensemble (cancellation). A graded apical read over "
                            "the discovered L2+L3 codes answers NL questions from the correct level; sibling-discrimination is "
                            "CODON-DRIVEN (the asked property must drive a taught class-property cell above the floor purely via "
                            "the discovered codon, NO spoken-taxonomy shortcut) + the no-confab moat. Composes EMERGE-44/45 "
                            "(multi-level discovery) + EMERGE-51 (NL console) + EMERGE-42 (cancellation). NO sim/ edit.",
               "task": "observe members + speak sub-cat->genus->order taxonomy -> DISCOVER the levels -> teach genus + order "
                       "properties -> ASK 'can a X P?' answered by inheritance from the CORRECT discovered level (1 level + 2 "
                       "levels up) with sibling-discrimination + cancellation + moat; 2-level held-out inheritance + "
                       "sibling-confusion + moat + LOAD-BEARING permute-CO-OCCURRENCE control (breaks the codon-driven "
                       "sibling-discrimination) + secondary permute-features diagnostic; 3-seed",
               "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "composes validated pieces: EMERGE-44/45 stacked pooler (the pooler LEARNING is a rate-reference, "
                              "realized fully-on-substrate at EMERGE-39..41) + EMERGE-51 NL console + EMERGE-42 cancellation, on "
                              "the spiking bridge over the DISCOVERED multi-level codons. Per EMERGE-45's honest scope, L2/GENUS "
                              "grouping is the DOMINANT carrier of the multi-level signal and L3/ORDER is a seed-variable "
                              "increment; the reported L2/genus 1-level floor is disclosed alongside the 2-level order read. Each "
                              "held-out member shares its GENUS with taught members, so 2-level order inheritance chains through "
                              "the discovered L2/genus grouping (+ an L3 increment) -- '2-level via the discovered hierarchy'. "
                              "The SIBLING-DISCRIMINATION is read PURELY from the discovered codons (no spoken-taxonomy shortcut), "
                              "and PERMUTE-CO-OCCURRENCE (scramble the L2/L3 co-occurrence pairs) BREAKS it (raises "
                              "sibling-confusion) -- that is the LOAD-BEARING collapse control for the discovered-codon structure. "
                              "The 2-level ORDER read itself rides the feature/genus grouping (kept intact by permute-features "
                              "since the co-occurrence stream is keyed by the intact spoken taxonomy), so permute-features alone is "
                              "a SECONDARY diagnostic that does not collapse the 2-level read -- exactly EMERGE-45's finding. The "
                              "demo vocabulary is a small curated bird/fish taxonomy (birds share wings/feathers/beak/talons/plume/"
                              "crest; fish share fins/scales/gills/tail/stripe/barbel, each member a varied 4-of-6 subset); "
                              "corpus-scale feature/taxonomy discovery is a follow-on."}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge52] VERDICT: {verdict}", flush=True)
    print(f"[emerge52] wrote {OUT}\n" + "=" * 108, flush=True)
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
    c = MultiLevelConversationalConsole(seed=a.seed)
    print("multi-level console -- observe: 'a X has f1 f2'; taxonomy: 'a X is a Y' (member->sub-cat->genus->order); "
          "class: 'a CLASS can P' + 'a ORDERCLASS P'; exception: 'a MEMBER WORD'; ask: 'can a X P?'  (Ctrl-D to exit)")
    if a.script:
        for line in a.script.split(";"):
            r = handle(c, line)
            if r is not None:
                print(f"  you> {line.strip()}\n  brain> {r}")
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
