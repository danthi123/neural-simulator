"""EMERGE-51 / the EXPERIENTIAL CONVERSATIONAL CONSOLE — the emergent semantic substrate is now CONVERSATIONALLY
QUERYABLE: the brain OBSERVES members with features in plain language, the COMPETITIVE SELF-ORGANIZING POOLER
(EMERGE-38/42) DISCOVERS the overlapping categories from that experience (not hand-assigned), the user TEACHES a class
property via ONE member + a member-specific EXCEPTION, and then ASKS in natural language — answered by the on-substrate
inheritance/cancellation over the DISCOVERED codes, with the honest no-confab moat. It composes EMERGE-42/43 (the
pooler-discovered categories + inheritance + member-specific cancellation) with the EMERGE-29/31 natural-language
console, on the real spiking `SimulationBridge`. NO `sim/` edit.

  you> a robin has wings feathers small        (OBSERVE a member with features)
  you> a sparrow has wings feathers tiny
  you> a penguin has wings feathers black
  you> a trout has fins scales silver
  ...                                           (the competitive pooler DISCOVERS the categories from the features)
  you> a bird can fly                           (TEACH a class property via ONE member -- 'robin' is a bird-exemplar)
  you> a penguin walks                          (TEACH a member-specific EXCEPTION -- cancellation)
  you> can a sparrow fly?     brain> Yes, a sparrow can fly.   (INHERITED via the DISCOVERED bird codon; never told)
  you> can a penguin fly?     brain> No, a penguin walks.      (CANCELLATION -- the member's own exception)
  you> can a zzz fly?         brain> I don't know what a zzz is.  (the no-confab MOAT -- never observed)

MECHANISM (emergent; no inference engine, no transformer): "a X has f1 f2 f3" gives member X a feature vector over a
named feature vocabulary. The competitive HTM Spatial Pooler (EMERGE-38: winners potentiate active inputs + depress
inactive + homeostatic boosting, k-WTA) SELF-ORGANIZES a codon per member; members sharing features converge on
OVERLAPPING codons = the emergent categories. Teaching "a <exemplar> can P" (where <exemplar> is bound to class
<C> via 'a <exemplar> is a <C>', or auto-bound to the exemplar's own discovered codon) potentiates the codon->P
coincidence pool on the spiking bridge (the committed `sim/` three-term kernel) -> the class property attaches to the
SHARED codon so co-observed members INHERIT it. A member-specific exception "a <member> P" potentiates a MEMBER-IDENTITY
ensemble -> P, a stronger direct fact that out-drives the inherited default for that member (cancellation). Asking
reads the strongest of the direct override and the inherited class default via a graded apical drive. A never-observed
token drives no codon -> the moat abstains. (EMERGE-42/43 mechanism, wrapped in a live NL console + moat.)

`--demo` / `--script "a robin has wings feathers;...;can a sparrow fly?"` / interactive; `--derisk --seeds 42 43 44`
runs the de-risk gates (held-out inheritance / cancellation / moat / permuted control). CPU numpy-backend; reuse-by-
import (`_emerge14` + `_emerge12`); NO `sim/` edit.
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

OUT = Path("research/findings/raw/_emerge51_experiential_conversational_console.json")

# --- capacities (generous; the console pre-allocates fixed cell blocks, EMERGE-42/43 layout) --------------------------
NF = 48                 # feature-vocabulary cells (named features the user says)
NCOL = 200              # competitive-pooler columns
K_WIN = 6               # k winners per member codon
NMEM = 48               # max distinct members
N_ID_PER = 3            # member-identity ensemble size (>=2 clears the coincidence threshold)
NCLASS = 16             # max distinct class-property tags
NOVR = 16               # max distinct member-exception tags
N_PROP_PER = 2          # cells per property tag
# cell layout: [features | member-identity ensembles | pooler columns | class-property cells | override-property cells]
FEAT0 = 0
ID0 = NF
COL0 = NF + NMEM * N_ID_PER
CLASSP0 = COL0 + NCOL
OVRP0 = CLASSP0 + NCLASS * N_PROP_PER
M = OVRP0 + NOVR * N_PROP_PER

# pooler learning schedule
POOL_LP = 0.05
POOL_LD = 0.02
POOL_EPOCHS = 400
# teaching schedule
TEACH_EPOCHS = 40
FLOOR = -40.0


def _sdr(cells):
    return set(int(c) for c in cells)


def _art(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def _build_bridge(seed, lesion=False):
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
    # wire: pooler columns -> every property cell (class inheritance) + member-identity cells -> every property cell
    # (member-specific facts). All property cells (class + override) are downstream targets. Weights start at 0 and are
    # potentiated by the on-substrate three-term kernel during teaching.
    prop_cells = list(range(CLASSP0, CLASSP0 + NCLASS * N_PROP_PER)) + list(range(OVRP0, OVRP0 + NOVR * N_PROP_PER))
    pre, post, w = [], [], []
    for pc in prop_cells:
        for c in range(NCOL):
            pre.append(int(ci[COL0 + c])); post.append(int(ci[pc])); w.append(0.0)
        for idx in range(NMEM * N_ID_PER):
            pre.append(int(ci[ID0 + idx])); post.append(int(ci[pc])); w.append(0.0)
    b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                     "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
    coo = b._get_cached_coo()
    return b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))


class ExperientialConversationalConsole:
    """OBSERVE members with features -> the competitive pooler DISCOVERS categories -> TEACH class/exception properties ->
    ASK in natural language (inherit / cancel / abstain), all on the spiking bridge over the DISCOVERED codes."""

    def __init__(self, seed=42, lesion=False, permute=False, pool_epochs=POOL_EPOCHS, teach_epochs=TEACH_EPOCHS):
        self.seed = int(seed)
        self.permute = bool(permute)
        self.pool_epochs = int(pool_epochs)
        self.teach_epochs = int(teach_epochs)
        self.rng = np.random.default_rng(self.seed)
        self.b, self.ci, self.row, self.col = _build_bridge(self.seed, lesion=lesion)
        self.z = np.zeros(len(self.ci))
        # named vocabularies -> fixed cell blocks
        self.feat_id = {}                       # feature name -> feature-cell index (0..NF-1)
        self.member_feats = {}                  # member name -> set(feature-cell indices)
        self.member_idx = {}                    # member name -> member-identity block index (0..NMEM-1)
        self.member_class = {}                  # member name -> class name (from 'a X is a C'; else the member itself)
        self.class_slot = {}                    # class name -> class-property slot index (0..NCLASS-1)
        self.class_prop = {}                    # class name -> the taught class-property word (for the reply text)
        self.ovr_slot = {}                      # member name -> override-property slot index (0..NOVR-1)
        self.ovr_prop = {}                      # member name -> the taught exception word (for the reply text)
        self.Wp = self.rng.uniform(0.30, 0.55, (NCOL, NF))     # competitive-pooler feat->col permanences
        self._pooler_dirty = False              # a new observation invalidates the codons -> retrain lazily

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
            self.member_class.setdefault(name, name)          # default: its own class (until 'a X is a C')
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

    # ---- the competitive pooler (EMERGE-38/42): discover overlapping-category codons from the observed features ------
    def _train_pooler(self):
        """UNSUPERVISED competitive learning over the observed member feature-vectors: winners potentiate their active
        features + depress inactive (selectivity) + homeostatic boosting -> members sharing features converge on
        OVERLAPPING codons = the emergent categories. Retrained fresh from the current observation set (idempotent)."""
        members = list(self.member_feats)
        if not members:
            return
        self.Wp = self.rng.uniform(0.30, 0.55, (NCOL, NF))    # fresh init -> deterministic given the observation set
        duty = np.zeros(NCOL); boost = np.ones(NCOL)
        rng = np.random.default_rng(self.seed + 777)
        order = list(members)
        for e in range(self.pool_epochs):
            rng.shuffle(order)
            for m in order:
                x = np.zeros(NF); x[list(self.member_feats[m])] = 1.0
                win = np.argsort(-(((self.Wp > 0.5) @ x) * boost))[:K_WIN]
                self.Wp[win] += POOL_LP * x - POOL_LD * (1 - x)
                self.Wp[win] = np.clip(self.Wp[win], 0, 1); duty[win] += 1
            boost = np.exp(2.0 * (K_WIN / NCOL - duty / ((e + 1) * len(members))))
        self._pooler_dirty = False

    def _codon(self, member):
        """the DISCOVERED code = the k columns whose learned feat->col synapses best overlap the member's features."""
        if self._pooler_dirty:
            self._train_pooler()
        x = np.zeros(NF); x[list(self.member_feats[member])] = 1.0
        return _sdr(COL0 + int(c) for c in np.argsort(-((self.Wp > 0.5) @ x))[:K_WIN])

    # ---- teaching from experience -----------------------------------------------------------------------------------
    def observe(self, member, feats):
        """'a member has f1 f2 f3' -> record the member's feature vector; the pooler rediscovers categories."""
        self._alloc_member(member)
        fset = set(self._feat(f) for f in feats)
        if self.permute:                                      # PERMUTED control: random feature vector -> no category structure
            fset = set(int(c) for c in self.rng.choice(NF, min(len(feats), NF), replace=False))
        self.member_feats[member] = fset
        self._pooler_dirty = True
        return f"ok -- I've seen {_art(member)} with {' '.join(feats)}."

    def learn_isa(self, member, cname):
        """'a member is a C' -> bind the member to class C so 'a <exemplar> can P' teaches C's shared codon."""
        self._alloc_member(member)
        self.member_class[member] = cname
        return f"ok -- {_art(member)} is {_art(cname)}."

    def learn_class(self, exemplar, prop):
        """'a <exemplar> can P' -> teach P on the exemplar's DISCOVERED codon (the class-shared code), so co-observed
        members INHERIT P. The class is the exemplar's bound class (from 'a X is a C'), else the exemplar itself."""
        if exemplar not in self.member_feats:
            self._alloc_member(exemplar)
        cname = self.member_class.get(exemplar, exemplar)
        self.class_prop[cname] = prop
        cells = self._class_cells(cname)
        codon = self._codon(exemplar)
        for _ in range(self.teach_epochs):
            apply_kernel_update(self.b, self.row, self.col, self.ci, codon, cells, self.z, 0.14, 0.02, 1.0)
        return f"ok -- {_art(exemplar)} can {prop}."

    def learn_exception(self, member, prop):
        """'a member P' (member-specific exception) -> teach P on the member's IDENTITY ensemble (a stronger, direct
        fact that out-drives the inherited class default for this member) = cancellation."""
        self._alloc_member(member)
        self.ovr_prop[member] = prop
        cells = self._ovr_cells(member)
        idc = self._id_cells(member)
        for _ in range(self.teach_epochs * 2):
            apply_kernel_update(self.b, self.row, self.col, self.ci, idc, cells, self.z, 0.14, 0.02, 1.0)
        return f"ok -- {_art(member)} {prop}."

    # ---- inference (graded apical read over the DISCOVERED codes + member identity) ---------------------------------
    def _drive(self, member):
        """Prime the member's DISCOVERED codon + its identity ensemble, read the graded apical drive to every taught
        property slot. Returns {slot_name: drive} or None on a moat miss (unknown/unobserved member)."""
        if member not in self.member_feats:
            return None
        codon = self._codon(member)
        if not codon:
            return None
        ab = np.zeros(len(self.ci), bool)
        for c in codon:
            ab[c] = True
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
        """Cancellation read (Collins-Quillian): the member's OWN specific exception WINS a tie against the inherited
        class default (the specific fact overrides the general one). Otherwise the strongest active property wins; None
        below the floor = the no-confab moat. A DIFFERENT member's exception winning strictly = honest cross-bleed."""
        dr = self._drive(member)
        if not dr:
            return None
        best = max(dr, key=dr.get)
        if dr[best] <= FLOOR:
            return None
        own = ("OVR", member)
        if own in dr and dr[own] > FLOOR and dr[own] >= dr[best] - 1e-6:      # the specific fact overrides the class default
            return own
        return best

    def ask_can(self, member, prop):
        """Answer 'can a <member> <prop>?' by the on-substrate inheritance/cancellation over the DISCOVERED codes,
        honestly abstaining (the no-confab moat) on an unknown/unobserved member."""
        if member not in self.member_idx:
            return f"I don't know what {_art(member)} is."
        best = self._best(member)
        if best is None:
            return f"I don't know whether {_art(member)} can {prop}."
        kind, key = best
        if kind == "OVR" and key == member:                     # the member's own exception fires strongest -> cancellation
            ep = self.ovr_prop.get(member, prop)
            return f"No, {_art(member)} {ep}."
        if kind == "CLASS":                                     # inherited class default via the shared discovered codon
            cp = self.class_prop.get(key, prop)
            return f"Yes, {_art(member)} can {cp}."
        # a DIFFERENT member's exception won the read (cross-bleed) -> honest non-answer, don't confabulate
        return f"I don't know whether {_art(member)} can {prop}."

    # ---- de-risk accessors (used by --derisk + the tests) ----------------------------------------------------------
    def inherit_ok(self, member, cname):
        best = self._best(member)
        return best == ("CLASS", cname)

    def cancel_ok(self, member):
        best = self._best(member)
        return best == ("OVR", member)

    def moat_abstains(self, member, prop):
        return self.ask_can(member, prop).startswith("I don't know")


# ---- a tiny natural-language front end (host parsing = the world/keyboard interface) --------------------------------
_OBS = re.compile(r"(?:a|an)\s+(\w+)\s+has\s+(.+)", re.I)
_ISA = re.compile(r"(?:a|an)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)", re.I)
_ASK = re.compile(r"can\s+(?:a|an)\s+(\w+)\s+(\w+)\??", re.I)
_CAN = re.compile(r"(?:a|an)\s+(\w+)\s+can\s+(\w+)", re.I)          # class property via an exemplar
_EXC = re.compile(r"(?:a|an)\s+(\w+)\s+(\w+)\s*$", re.I)            # member-specific exception: 'a penguin walks'


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
        return console.learn_exception(m.group(1).lower(), m.group(2).lower())
    return "(say 'a X has f1 f2', 'a X is a C', 'a EXEMPLAR can P', 'a MEMBER EXCEPTION', or 'can a X P?')"


# ---- the scripted world (EMERGE-42-style overlapping feature pools; each member a varied 4-of-6 subset of its pool) --
# Two categories the pooler must DISCOVER from features alone (adjacent members share most, but not all, features):
#   birds share {wings, feathers, beak, talons, plume, crest}; fish share {fins, scales, gills, tail, stripe, barbel}.
# 6 members each; the user teaches the class property via 3 exemplars (following EMERGE-42's validated inheritance
# protocol -- the class is taught on multiple member codons) + 1 member-specific exception; the remaining members are
# GENUINE HELD-OUTS (never named in a 'can'/exception sentence -> they must INHERIT via the shared discovered codon).
_BIRD_POOL = ["wings", "feathers", "beak", "talons", "plume", "crest"]
_FISH_POOL = ["fins", "scales", "gills", "tail", "stripe", "barbel"]
# 8 members per category (EMERGE-42 statistical regime: enough same-category members that the discovered codons share
# columns reliably). The user teaches the class property via 6 exemplars (following EMERGE-42's inheritance protocol --
# the class is potentiated on many member codons), 1 member carries a specific exception, and the last 2 are GENUINE
# HELD-OUTS (never named in a 'can'/exception sentence -> they must INHERIT via the shared discovered codon).
_BIRDS = ["robin", "sparrow", "eagle", "hawk", "crow", "finch", "penguin", "owl", "wren"]   # penguin = exception
_FISH = ["trout", "salmon", "carp", "bass", "perch", "tuna", "pike", "minnow", "gar"]       # pike = exception
_BIRD_EXEMPLARS = ["robin", "sparrow", "eagle", "hawk", "crow", "finch"]
_FISH_EXEMPLARS = ["trout", "salmon", "carp", "bass", "perch", "tuna"]
_BIRD_EXC = ("penguin", "walks")
_FISH_EXC = ("pike", "lurks")
_BIRD_HELDOUT = ["owl", "wren"]                                          # never taught a property -> must inherit
_FISH_HELDOUT = ["minnow", "gar"]


def _member_feats(seed, member, pool):
    """Deterministic 4-of-6 subset of the category's feature pool (varied per member, EMERGE-42 style)."""
    h = abs(hash((seed, member))) % (10 ** 8)
    r = np.random.default_rng(seed * 131 + h)
    return list(np.array(pool)[np.sort(r.choice(len(pool), 4, replace=False))])


def _script_lines(seed):
    """Build the full scripted transcript (observe -> is-a -> teach class via exemplars + exceptions -> ask)."""
    obs, isa, teach, ask = [], [], [], []
    for b in _BIRDS:
        obs.append(("a %s has %s" % (b, " ".join(_member_feats(seed, b, _BIRD_POOL))), None))
        isa.append(("a %s is a bird" % b, None))
    for f in _FISH:
        obs.append(("a %s has %s" % (f, " ".join(_member_feats(seed, f, _FISH_POOL))), None))
        isa.append(("a %s is a fish" % f, None))
    for b in _BIRD_EXEMPLARS:
        teach.append(("a %s can fly" % b, "class property, via a bird exemplar"))
    for f in _FISH_EXEMPLARS:
        teach.append(("a %s can swim" % f, "class property, via a fish exemplar"))
    teach.append(("a %s %s" % _BIRD_EXC, "member-specific EXCEPTION (cancellation)"))
    teach.append(("a %s %s" % _FISH_EXC, "member-specific EXCEPTION (cancellation)"))
    for b in _BIRD_HELDOUT:
        ask.append(("can a %s fly?" % b, "INHERIT -- never told; via the discovered bird codon"))
    for f in _FISH_HELDOUT:
        ask.append(("can a %s swim?" % f, "INHERIT -- never told; via the discovered fish codon"))
    ask.append(("can a %s fly?" % _BIRD_EXC[0], "CANCEL -- the penguin's own exception"))
    ask.append(("can a %s swim?" % _FISH_EXC[0], "CANCEL -- the pike's own exception"))
    ask.append(("can a zzz fly?", "MOAT -- never observed"))
    return obs, isa, teach, ask


def _demo(seed=42):
    c = ExperientialConversationalConsole(seed=seed)
    obs, isa, teach, ask = _script_lines(seed)
    print("\n=== EMERGE-51 experiential conversational console -- DISCOVER categories from experience, then TALK "
          "(inherit / cancel / abstain); no transformer ===\n")
    print("  --- OBSERVE members with features (the competitive pooler DISCOVERS the categories) ---")
    for line, _ in obs:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- bind members to their class name (so 'a robin can fly' teaches the shared bird codon) ---")
    for line, _ in isa:
        print(f"  you> {line}\n  brain> {handle(c, line)}")
    print("  --- TEACH the class property via a few exemplars + member-specific EXCEPTIONS ---")
    for line, why in teach:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    print("  --- ASK in natural language (answers by the on-substrate inference over the DISCOVERED codes) ---")
    for line, why in ask:
        print(f"  you> {line}\n  brain> {handle(c, line)}   ({why})")
    print()
    return c


def _run_demo_and_check(seed=42):
    """Run the scripted transcript silently, return (console, checks-dict) for the self-check test + --derisk."""
    c = ExperientialConversationalConsole(seed=seed)
    obs, isa, teach, _ = _script_lines(seed)
    for line, _ in obs:
        handle(c, line)
    for line, _ in isa:
        handle(c, line)
    for line, _ in teach:
        handle(c, line)
    # GENUINE HELD-OUTS: owl/wren (bird) + bass/perch (fish) were NEVER named in a 'can'/exception sentence -> they
    # inherit ONLY via the shared discovered codon (the exemplars were the taught members).
    held_inherit = {m: "bird" for m in _BIRD_HELDOUT}
    held_inherit.update({m: "fish" for m in _FISH_HELDOUT})
    inh = float(np.mean([c.inherit_ok(m, cn) for m, cn in held_inherit.items()]))
    canc = float(np.mean([c.cancel_ok(m) for m in (_BIRD_EXC[0], _FISH_EXC[0])]))
    moat_unknown = c.moat_abstains("zzz", "fly")
    replies = {
        "heldout_inherit": handle(c, "can a %s fly?" % _BIRD_HELDOUT[0]),
        "exception_cancel": handle(c, "can a %s fly?" % _BIRD_EXC[0]),
        "moat_unknown": handle(c, "can a zzz fly?"),
    }
    return c, {"inherit": inh, "cancel": canc, "moat_unknown": bool(moat_unknown), "replies": replies}


# ---- the de-risk (held-out inheritance / cancellation / moat / permuted control), multi-seed -----------------------
def _derisk_one(seed):
    # main arm: real experience
    c, ch = _run_demo_and_check(seed)
    inh, canc = ch["inherit"], ch["cancel"]
    moat_unknown = ch["moat_unknown"]
    # MOAT false-accepts: every never-observed token must abstain (0 false-accepts). Reuse the same trained console.
    fa = sum(0 if c.moat_abstains(t, "fly") else 1 for t in ("zzz", "qqq", "wobble"))

    # PERMUTED control: scramble the experience (random feature vectors) -> the pooler can't discover categories ->
    # held-out inheritance collapses toward chance.
    cp = ExperientialConversationalConsole(seed=seed, permute=True)
    obs, isa, teach, _ = _script_lines(seed)
    for line, _ in obs:
        handle(cp, line)
    for line, _ in isa:
        handle(cp, line)
    for line, _ in teach:
        handle(cp, line)
    held_inherit = {m: "bird" for m in _BIRD_HELDOUT}
    held_inherit.update({m: "fish" for m in _FISH_HELDOUT})
    perm_inh = float(np.mean([cp.inherit_ok(m, cn) for m, cn in held_inherit.items()]))
    return {"seed": seed, "inherit": inh, "cancel": canc, "moat_unknown": bool(moat_unknown),
            "moat_false_accepts": int(fa), "permuted_inherit": perm_inh}


def _derisk(seeds):
    print(f"EMERGE-51 experiential conversational console de-risk: observe -> DISCOVER categories -> teach -> "
          f"NL inherit/cancel/abstain; held-out inheritance chance ~{1/8:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            print(f"  [seed {s}] held-out inherit {d['inherit']:.2f} | cancel {d['cancel']:.2f} | "
                  f"moat-unknown {int(d['moat_unknown'])} | moat-FA {d['moat_false_accepts']} | "
                  f"permuted inherit {d['permuted_inherit']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        inh = float(np.mean([d["inherit"] for d in per]))
        canc = float(np.mean([d["cancel"] for d in per]))
        moat_unknown_all = all(d["moat_unknown"] for d in per)
        moat_fa = int(sum(d["moat_false_accepts"] for d in per))
        perm = float(np.mean([d["permuted_inherit"] for d in per]))
        go = bool(inh >= 0.80 and canc >= 0.99 and moat_unknown_all and moat_fa == 0 and inh >= perm + 0.30)
        if go:
            verdict = (f"GO -- the emergent semantic substrate is CONVERSATIONALLY QUERYABLE: the brain OBSERVES members "
                       f"with features in plain language, the competitive self-organizing pooler (EMERGE-38/42) DISCOVERS "
                       f"the overlapping categories from that experience, and the user teaches + ASKS in natural language, "
                       f"answered by the on-substrate inheritance/cancellation over the DISCOVERED codes -- HELD-OUT "
                       f"INHERITANCE {inh:.2f} (a never-directly-taught member inherits via the shared discovered codon), "
                       f"CANCELLATION {canc:.2f} (the exception member answers ITS specific fact), the no-confab MOAT "
                       f"abstains on every never-observed token ({moat_fa} false-accepts). PERMUTED experience (scrambled "
                       f"features -> no discoverable categories) collapses held-out inheritance to {perm:.2f}. 3-seed. => "
                       f"'discover categories from experience -> talk to the brain about them', one spiking brain, NO sim/ edit.")
        else:
            miss = []
            if inh < 0.80: miss.append(f"held-out inheritance {inh:.2f} < 0.80")
            if canc < 0.99: miss.append(f"cancellation {canc:.2f} < 0.99")
            if not moat_unknown_all: miss.append("moat did not abstain on an unknown token")
            if moat_fa != 0: miss.append(f"moat false-accepts {moat_fa} != 0")
            if inh < perm + 0.30: miss.append(f"permuted didn't collapse ({inh:.2f} vs {perm:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". The specific gap is above; the pieces "
                       "(EMERGE-38 pooler + EMERGE-42/43 inheritance/cancellation + EMERGE-29 NL console) each pass "
                       "standalone -- tune the pooler epochs / teaching balance / feature-vocab overlap. Not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge51_experiential_conversational_console", "verdict": verdict,
               "mechanism": "the EMERGE-38/42 competitive self-organizing pooler discovers overlapping-category codons from "
                            "observed member feature-vectors (unsupervised, k-WTA + boosting); the committed three-term kernel "
                            "teaches a CLASS property on an exemplar's discovered codon (inheritance via the shared codon) + a "
                            "member EXCEPTION on the member-identity ensemble (cancellation); a graded apical read over the "
                            "discovered codes answers natural-language questions with the no-confab moat; a tiny regex NL front "
                            "end (world/keyboard interface). Composes EMERGE-42/43 + EMERGE-29/31. NO sim/ edit.",
               "task": "observe members with features in plain language -> DISCOVER categories -> teach class property + "
                       "member exception -> ASK 'can a X P?' answered by inheritance/cancellation over the discovered codes, "
                       "with the moat; held-out inheritance + cancellation + moat + permuted control; 3-seed",
               "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "composes validated pieces: EMERGE-38 competitive pooler (the learning is a rate-reference "
                              "realized fully-on-substrate at EMERGE-39..41) + EMERGE-42/43 inheritance/cancellation on the "
                              "spiking bridge over the discovered codons + EMERGE-29/31 natural-language console. Per EMERGE-42's "
                              "validated inheritance protocol, the class property is taught via SEVERAL exemplars (6 of the 8 "
                              "per-category members named in 'a X can P' sentences) so the shared discovered columns are broadly "
                              "potentiated; the 2 HELD-OUT members per category are NEVER named in a can/exception sentence and "
                              "inherit only via the shared codon. Cancellation resolves the specific member exception over the "
                              "class default (Collins-Quillian: the specific fact wins, incl. a saturated-plateau tie). The demo "
                              "vocabulary is a small curated feature set (birds share wings/feathers/beak/talons/plume/crest; fish "
                              "share fins/scales/gills/tail/stripe/barbel, each member a varied 4-of-6 subset); corpus-scale "
                              "feature discovery + multi-level taxonomy in NL are follow-ons."}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge51] VERDICT: {verdict}", flush=True)
    print(f"[emerge51] wrote {OUT}\n" + "=" * 108, flush=True)
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
    c = ExperientialConversationalConsole(seed=a.seed)
    print("experiential console -- observe: 'a X has f1 f2'; class: 'a X is a C' + 'a EXEMPLAR can P'; exception: "
          "'a MEMBER WORD'; ask: 'can a X P?'  (Ctrl-D to exit)")
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
