"""Word-reading role map: the question parser's (agent, action) is read from the WORDS, on spiking neurons.

Packet `word-role-parse` (session-2 packet factory, 2026-10-01). Target: `ChatBrain._neural_question_parse` in
research/runners/brain_chat_tui.py. Today it presents (content[0], content[1], "__q__") to `BridgeParser.role_of(pos)`,
whose (position, voice) -> role conjunction never sees a word: the parse is a fixed positional map (content[0] is the
agent, content[1] the action). The integration-wave-2 review ruled the self-ID faculty (`BRAIN_NEURAL_SELFID`) "the same
positional role map reaching self questions": the faculty routes a self question to that map but the map reads no word.

WHAT THIS ORGAN DOES INSTEAD (one `SimulationBridge`, Izhikevich neurons, no region framework):

  * `word` inputs: one small assembly per vocabulary word (`NW` excitatory neurons), plus `N_SPARE` untrained assemblies
    that carry word forms met at test time which no heard sentence contained ("usually", "zorp"). A spare is the same
    sensory form-input as any word, with no learned projection.
  * `pos` inputs: three assemblies, position 0, 1, 2.
  * `role` ensembles: agent / action / patient (`R` neurons each) with LATERAL INHIBITION between the ensembles: each
    role ensemble drives its own inhibitory pool (`NI` neurons) and that pool inhibits the OTHER two ensembles, so one
    role wins per presentation.
  * word -> role and pos -> role are plastic projections trained by the engine's Hebbian co-firing rule
    (`enable_hebbian_learning`, the rule `BridgeParser._train` uses) from the brain's OWN heard sentences: for each
    training fact (agent, action, patient), (word(agent), pos0) is presented while the agent ensemble is teacher-driven,
    (word(action), pos1) with the action ensemble, (word(patient), pos2) with the patient ensemble. That is experience
    replay of heard sentences (the teacher channel), not a host rule. The pos -> role projection learns through its own
    plasticity gate at a lower gain, so after training it is SUBTHRESHOLD alone: an unheard word presented at a position
    does not yield a confident role. Word + position together are suprathreshold.
  * Frozen confidence rule (host read-out of spike counts, a constant): a presented word gets role r iff r's ensemble
    spike count >= ROLE_MIN and >= ROLE_MARGIN x the runner-up; else the word has role None.

PARSE. `parse(content_words)` presents each content word once, at position = its order among the content words capped
at 2 (a fourth content word reuses position 2), one at a time with a quiescent-state reset between. agent = the word
whose role is agent with the highest agent-ensemble count; action likewise. Words whose role is None are skipped, so a
modifier the brain never heard ("usually") drops out; it still occupies an order slot, so [brain, usually, use] presents
"use" at position 2, where its word drive alone selects action (a positional map would have read "usually" as the
action). Agent or action missing -> that field is None and the caller declines.

LESION. `lesion=True` closes the transmission gate `role_word` on the word -> role projection. Only position then drives
the roles, which is subthreshold by construction, so every word reads role None and `parse` declines (None, None).

NAMED RESIDUALS (host code that remains, declared).
  * Training stimuli are transcribed from the composer's fact index (`chat.inner.composer.kb`): the host copies each
    stored (agent, action, patient) string into a presentation schedule. That is a host transcription of what was heard.
  * The content-word strip (stop-words) in `_extract_route` and the self-alias -> 'brain' resolution
    (`QuestionRouter._resolve_self`) stay host.
  * The frozen confidence rule (ROLE_MIN / ROLE_MARGIN over spike counts) is a host read-out of the role ensembles.
  * Assigning a word form to a word assembly (a dictionary word -> index) is the sensory front end, host.

Backend: process backend (numpy on the pool). No `sim/` edit. Same global-RNG isolation as the W3 organ.

DEV MEASUREMENTS (dev seeds 1-8 only; gate seeds 42/43/44/100/101/102 untouched; a dev read, not a generalization
claim, docs/TERMS.md). Organ battery, numpy backend, ~1 s per seed and op (the whole 8 x 8 sweep takes about 90 s):

  op  all six cases pass   dual-role words right   max pos-alone spikes   min word-alone spikes   note
  0   8/8                  17/32                   7                      10                      the default
  1   8/8                  17/32                   7                      10
  2   8/8                  13/32                   5                      9
  3   8/8                  14/32                   7                      10
  4   8/8                   2/32                   0                      15                      word-dominant
  5   3/8                  21/32                   16                     17                      pos too strong
  6   6/8                  25/32                   1                      5                       adverb fails
  7   6/8                  14/32                   8                      9                       swapped fails

Mean learned weights at op 0 (seed 1): word -> taught role 11.0, word -> untaught role 0.05 (the initial weight),
pos -> matched role 4.7, pos -> other role 0.05. Position alone reads 0-7 spikes against ROLE_MIN 9 (subthreshold, no
role); a word alone reads 10 or more (suprathreshold). That separation is thin at op 0 (7 < 9 < 10), which is why
ops 1-3 are kept as neighbours that also pass. The lateral inhibition carries the dual-role read: on seeds 1-4 with
the pool nearly silent (W_RI 6, W_IR 18, otherwise op 0) dual-role words were right 0 of 16 times, against 8-9 of 16
with W_RI 100-200 and W_IR 40-80.

KNOWN LIMIT (measured, not hidden). A word taught in two roles ("cat": patient of dog-chase-cat, agent of cat-eat-fish)
is read by its word synapses to BOTH roles; the position pathway tips the choice only when it adds enough above a role
ensemble's threshold. At op 0 the dual-role words cat/fish are read at the right position in 17 of 32 dev readings; the
rest decline (role None -> parse None -> the chat abstains) and none is read as the wrong role. A positional map answers
those words always; this organ trades some recall for reading the words. The self-ID row uses brain/use, which are
single-role words, and passes 8/8.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state, _reset_quiescent  # noqa: E402
from tools.lab import lever  # noqa: E402

ROLES = ("agent", "action", "patient")
GATE_WORD_TRANSMISSION = "role_word"        # transmission gate on word -> role (the lesion closes it)
GATE_WORD_PLASTIC = "wrp_word_plastic"      # plasticity gain of word -> role
GATE_POS_PLASTIC = "wrp_pos_plastic"        # plasticity gain of pos -> role (lower: stays subthreshold alone)
GATE_FIXED = "wrp_fixed"                    # frozen lateral-inhibition projections

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)

# Frozen operating-point GRID, most conservative first. Fields: NW/NP assembly sizes (word / position), R role-ensemble
# size, NI inhibitory pool size per role, DRIVE_PA input current while a word/pos is presented, TEACH_PA teacher current
# into the target role ensemble during training, LR Hebbian learning rate, W_MAX hebbian max weight, POS_GAIN plasticity
# gain of pos -> role, EPOCHS replays of the fact set, TRAIN_STEPS per presentation, TEST_STEPS per presentation,
# W_INIT initial plastic weight, W_RI role -> own inhibitory pool, W_IR pool -> other roles, ROLE_MIN / ROLE_MARGIN the
# frozen confidence rule, N_SPARE untrained word-form assemblies.
_G = dict(NW=8, NP=8, R=12, NI=8, DRIVE_PA=1500.0, TEACH_PA=1500.0, LR=0.1, W_MAX=20.0, POS_GAIN=0.04, EPOCHS=8,
          TRAIN_STEPS=40, TEST_STEPS=60, W_INIT=0.05, W_RI=200.0, W_IR=80.0, ROLE_MIN=9, ROLE_MARGIN=2.0, N_SPARE=4)
GRID: List[dict] = [
    dict(_G),                                                              # 0: working lateral inhibition, pos 0.04
    dict(_G, LR=0.08, EPOCHS=10, W_IR=60.0),                               # 1: slower word learning, longer replay
    dict(_G, POS_GAIN=0.03),                                               # 2: weaker position pathway
    dict(_G, ROLE_MIN=8, W_RI=150.0),                                      # 3: lower confidence floor, softer pool
    dict(_G, LR=0.3, POS_GAIN=0.01, EPOCHS=4, ROLE_MIN=8, W_IR=60.0),      # 4: word-dominant (dual-role words decline)
    dict(_G, NW=10, NP=10, R=16, NI=10, ROLE_MIN=12),                      # 5: larger assemblies / ensembles
    dict(_G, DRIVE_PA=1200.0, TEACH_PA=1200.0),                            # 6: weaker input drive
    dict(_G, NI=12, W_RI=120.0, W_IR=60.0),                                # 7: larger inhibitory pools
]
PARAMS: dict = dict(GRID[0])


def set_operating_point(i: int) -> dict:
    """Select GRID[i] as the module's PARAMS (mutated in place so importers holding PARAMS see it). Returns it."""
    i = int(i)
    if not (0 <= i < len(GRID)):
        raise ValueError("operating point %d outside 0..%d" % (i, len(GRID) - 1))
    PARAMS.clear()
    PARAMS.update(GRID[i])
    return PARAMS


def _isolated(fn):
    """Run `fn` and restore the host numpy (and backend) global RNG state afterwards, so building/training a bridge
    never perturbs the caller's random stream."""
    try:
        xp, _ = get_backend()
    except Exception:
        xp = None
    host_np = np.random.get_state()
    host_xp = None
    if xp is not None and xp is not np:
        try:
            host_xp = xp.random.get_random_state().get_state()
        except Exception:
            host_xp = None
    try:
        return fn()
    finally:
        np.random.set_state(host_np)
        if host_xp is not None:
            try:
                xp.random.get_random_state().set_state(host_xp)
            except Exception:
                pass


def _dense(pre_idx, post_idx, weight: float, *, plastic: bool, conn_type: str, plasticity_gate: str,
           transmission_gate: Optional[str] = None, receptor: Optional[str] = None) -> dict:
    pre = np.repeat(np.asarray(pre_idx, dtype=np.int64), len(post_idx))
    post = np.tile(np.asarray(post_idx, dtype=np.int64), len(pre_idx))
    d = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
         "initial_weights": np.full(pre.shape[0], float(weight), dtype=np.float32), "plastic": bool(plastic),
         "plasticity_gate": plasticity_gate, "conn_type": conn_type, "count": int(pre.size)}
    if transmission_gate:
        d["transmission_gate"] = transmission_gate
    if receptor:
        d["receptor"] = receptor
    return d


class WordRoleParserOrgan:
    """A word-reading (word, position) -> role parser on one bridge. Build with the heard `facts`
    ((agent, action, patient) strings); `train()` runs the Hebbian replay; `parse()` reads roles from the words."""

    def __init__(self, facts: Iterable[Tuple[str, str, str]], seed: int = 1, params: Optional[dict] = None):
        self.p = dict(params if params is not None else PARAMS)
        self.seed = int(seed)
        seen, uniq = set(), []
        for f in facts:
            f = tuple(str(x) for x in f)
            if len(f) == 3 and f not in seen:
                seen.add(f)
                uniq.append(f)
        self.facts: List[Tuple[str, str, str]] = uniq
        vocab: List[str] = []
        for f in uniq:
            for w in f:
                if w not in vocab:
                    vocab.append(w)
        self.vocab = vocab
        self._bridge = self._xp = self._snap = None
        self._lock = threading.Lock()
        self._spare_map: Dict[str, int] = {}
        self.trained = False
        self.weight_summary: dict = {}

    # ── build ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _ensure(self):
        if self._bridge is None:
            _isolated(self._build)

    def _build(self):
        p = self.p
        xp, _ = get_backend()
        NW, NP, R, NI = int(p["NW"]), int(p["NP"]), int(p["R"]), int(p["NI"])
        V, S = len(self.vocab), int(p["N_SPARE"])
        n_word = (V + S) * NW
        o_pos = n_word
        o_role = o_pos + 3 * NP
        o_inh = o_role + 3 * R
        n = o_inh + 3 * NI
        self.word_idx = {w: np.arange(i * NW, (i + 1) * NW, dtype=np.int64) for i, w in enumerate(self.vocab)}
        self.spare_idx = [np.arange((V + j) * NW, (V + j + 1) * NW, dtype=np.int64) for j in range(S)]
        self.pos_idx = [np.arange(o_pos + k * NP, o_pos + (k + 1) * NP, dtype=np.int64) for k in range(3)]
        self.role_idx = {r: np.arange(o_role + k * R, o_role + (k + 1) * R, dtype=np.int64)
                         for k, r in enumerate(ROLES)}
        self.inh_idx = {r: np.arange(o_inh + k * NI, o_inh + (k + 1) * NI, dtype=np.int64) for k, r in enumerate(ROLES)}
        all_word = np.arange(0, n_word, dtype=np.int64)
        all_role = np.arange(o_role, o_role + 3 * R, dtype=np.int64)
        all_pos = np.arange(o_pos, o_pos + 3 * NP, dtype=np.int64)

        cfg = CoreSimConfig()
        cfg.num_neurons = int(n)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_watts_strogatz", "enable_ou_process"):
            setattr(cfg, f, False)
        cfg.enable_gabab = bool(p.get("RECEPTOR") == "gaba_b")
        cfg.enable_hebbian_learning = True       # the co-firing rule BridgeParser._train uses (pre t-1 & post t)
        cfg.hebbian_learning_rate = float(p["LR"])
        cfg.hebbian_max_weight = float(p["W_MAX"])
        cfg.enable_parameter_heterogeneity = True
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                                  gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)

        plan = {}
        plan["word_role"] = _dense(all_word, all_role, p["W_INIT"], plastic=True, conn_type="E_TO_E",
                                   plasticity_gate=GATE_WORD_PLASTIC, transmission_gate=GATE_WORD_TRANSMISSION)
        plan["pos_role"] = _dense(all_pos, all_role, p["W_INIT"], plastic=True, conn_type="E_TO_E",
                                  plasticity_gate=GATE_POS_PLASTIC)
        for r in ROLES:
            others = np.concatenate([self.role_idx[o] for o in ROLES if o != r])
            plan["ri_" + r] = _dense(self.role_idx[r], self.inh_idx[r], p["W_RI"], plastic=False, conn_type="E_TO_I",
                                     plasticity_gate=GATE_FIXED)
            plan["ir_" + r] = _dense(self.inh_idx[r], others, p["W_IR"], plastic=False, conn_type="I_TO_E",
                                     plasticity_gate=GATE_FIXED, receptor=p.get("RECEPTOR"))
        inh = np.concatenate([self.inh_idx[r] for r in ROLES])
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=inh)
        bridge.set_plasticity_gate(GATE_FIXED, 0.0)
        bridge.set_plasticity_gate(GATE_WORD_PLASTIC, 1.0)
        bridge.set_plasticity_gate(GATE_POS_PLASTIC, float(p["POS_GAIN"]))
        bridge.set_transmission_gate(GATE_WORD_TRANSMISSION, 1.0)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(60):
            bridge._run_one_simulation_step()
        self._bridge, self._xp = bridge, xp
        self._snap = _snapshot_state(bridge, xp)
        self._n = int(n)

    # ── presentation ───────────────────────────────────────────────────────────────────────────────────────────
    def _word_dev(self, word: str):
        xp = self._xp
        if word in self.word_idx:
            return xp.asarray(self.word_idx[word])
        if word not in self._spare_map and len(self._spare_map) < len(self.spare_idx):
            self._spare_map[word] = len(self._spare_map)
        j = self._spare_map.get(word)
        return xp.asarray(self.spare_idx[j]) if j is not None else None

    def _present(self, word: Optional[str], pos: Optional[int], steps: int, teach: Optional[str] = None,
                 count: bool = True) -> Dict[str, int]:
        """Reset to the quiescent state, drive (word, pos) [and the teacher role], run `steps`; return the spike count
        of each role ensemble. `word=None` presents the position alone,
        `pos=None` the word alone (diagnostics)."""
        b, xp, p = self._bridge, self._xp, self.p
        _reset_quiescent(b, xp, self._snap)
        cur = xp.zeros(self._n, dtype=xp.float32)
        if word is not None:
            dev = self._word_dev(word)
            if dev is not None:
                cur[dev] = xp.float32(p["DRIVE_PA"])
        if pos is not None:
            cur[xp.asarray(self.pos_idx[min(max(int(pos), 0), 2)])] = xp.float32(p["DRIVE_PA"])
        if teach is not None:
            cur[xp.asarray(self.role_idx[teach])] = xp.float32(p["TEACH_PA"])
        b.cp_external_input_current[:] = cur
        acc = {r: 0 for r in ROLES}
        rdev = {r: xp.asarray(self.role_idx[r]) for r in ROLES}
        for _ in range(steps):
            b._run_one_simulation_step()
            if count:
                for r in ROLES:
                    acc[r] += int(to_host(b.cp_firing_states[rdev[r]].astype(xp.float64).sum()))
        b.cp_external_input_current[:] = 0.0
        return acc

    def train(self):
        """Hebbian replay of the heard facts (the teacher channel). Idempotent."""
        with self._lock:
            if self.trained:
                return self
            _isolated(self._train)
            self.trained = True
        return self

    def _train(self):
        self._ensure()
        p, b = self.p, self._bridge
        b.set_plasticity_gate(GATE_WORD_PLASTIC, 1.0)
        b.set_plasticity_gate(GATE_POS_PLASTIC, float(p["POS_GAIN"]))
        for _ in range(int(p["EPOCHS"])):
            for fact in self.facts:
                for k, role in enumerate(ROLES):
                    self._present(fact[k], k, int(p["TRAIN_STEPS"]), teach=role, count=False)
        b.set_plasticity_gate(GATE_WORD_PLASTIC, 0.0)     # frozen after training: a parse never rewrites weights
        b.set_plasticity_gate(GATE_POS_PLASTIC, 0.0)
        self.weight_summary = self._weights()

    def _weights(self) -> dict:
        """Mean learned weights for the diagnostic record (a host read of the synapse matrix): word -> its taught roles,
        word -> roles it was never taught, position -> its matched role (pos k -> ROLES[k]), position -> other roles."""
        out = {}
        try:
            coo = self.bridge_coo()
            row, col, dat = coo
            taught = {}
            for f in self.facts:
                for k, r in enumerate(ROLES):
                    taught.setdefault(f[k], set()).add(r)
            tw, uw = [], []
            for w, idx in self.word_idx.items():
                for r in ROLES:
                    m = np.isin(row, idx) & np.isin(col, self.role_idx[r])
                    if m.any():
                        (tw if r in taught.get(w, ()) else uw).append(float(dat[m].mean()))
            pm, pu = [], []
            for k in range(3):
                for kr, r in enumerate(ROLES):
                    m = np.isin(row, self.pos_idx[k]) & np.isin(col, self.role_idx[r])
                    if m.any():
                        (pm if k == kr else pu).append(float(dat[m].mean()))
            for name, vals in (("word_taught", tw), ("word_untaught", uw), ("pos_matched", pm), ("pos_other", pu)):
                out[name] = round(float(np.mean(vals)), 3) if vals else 0.0
        except Exception:                                  # noqa: BLE001 -- a diagnostic, never fatal
            pass
        return out

    def bridge_coo(self):
        coo = self._bridge.cp_connections.tocoo()
        return to_host(coo.row), to_host(coo.col), to_host(coo.data)

    # ── read ───────────────────────────────────────────────────────────────────────────────────────────────────
    def read_word(self, word: str, pos: int, lesion: bool = False) -> Tuple[Optional[str], Dict[str, int]]:
        """Present one word at one position; return (role or None, role spike counts) under the frozen rule."""
        self.train()
        counts = _isolated(lambda: self._read(word, pos, lesion))
        return self._rule(counts), counts

    def _read(self, word, pos, lesion):
        self._bridge.set_transmission_gate(GATE_WORD_TRANSMISSION, 0.0 if lesion else 1.0)
        try:
            return self._present(word, pos, int(self.p["TEST_STEPS"]))
        finally:
            self._bridge.set_transmission_gate(GATE_WORD_TRANSMISSION, 1.0)

    def _rule(self, counts: Dict[str, int]) -> Optional[str]:
        order = sorted(ROLES, key=lambda r: counts[r], reverse=True)
        top, second = counts[order[0]], counts[order[1]]
        if top >= int(self.p["ROLE_MIN"]) and top >= float(self.p["ROLE_MARGIN"]) * second and top > 0:
            return order[0]
        return None

    def parse(self, content_words: Sequence[str], lesion: bool = False) -> dict:
        """Read the roles of the content words from the words. Position = order among the content words, capped at 2.
        Returns dict(roles={word: role|None}, counts={word: {role: n}}, agent=, action=); a missing one is None."""
        self.train()
        self._spare_map = {}                               # a fresh word-form -> spare-assembly dictionary per parse
        roles, counts = {}, {}
        for i, w in enumerate(content_words):
            r, c = self.read_word(w, min(i, 2), lesion=lesion)
            roles[w] = r
            counts[w] = c
        best = {}
        for w, r in roles.items():
            if r is not None and (r not in best or counts[w][r] > counts[best[r]][r]):
                best[r] = w
        return {"roles": roles, "counts": counts, "agent": best.get("agent"), "action": best.get("action")}


# ── the organ battery ───────────────────────────────────────────────────────────────────────────────────────────
BATTERY_FACTS = [("dog", "chase", "cat"), ("cat", "eat", "fish"), ("bird", "eat", "worm"), ("brain", "use", "spikes"),
                 ("brain", "have", "neurons"), ("fish", "swim", "lake")]


def organ_battery(seed: int, op_index: Optional[int] = None) -> dict:
    """The six organ cases on a fresh organ trained on BATTERY_FACTS. Returns dict(seed, op_index, grid_index, params,
    cases, all_pass, backend, weights, diagnostics); each case has a boolean `pass`. `diagnostics` (pos-alone and
    word-alone spike counts, and the dual-role words cat/fish read at their two positions) never enters `all_pass`."""
    if op_index is not None:
        set_operating_point(op_index)
    organ = WordRoleParserOrgan(BATTERY_FACTS, seed=seed, params=dict(PARAMS)).train()

    def case(name, content, want, lesion=False):
        r = organ.parse(content, lesion=lesion)
        got = (r["agent"], r["action"])
        return {"content": list(content), "want": list(want) if want is not None else None,
                "got": list(got) if got != (None, None) else None, "roles": r["roles"], "counts": r["counts"],
                "pass": bool((got == tuple(want)) if want is not None else got == (None, None))}

    cases = {
        "svo": case("svo", ["dog", "chase"], ("dog", "chase")),
        "self": case("self", ["brain", "use"], ("brain", "use")),
        "adverb": case("adverb", ["brain", "usually", "use"], ("brain", "use")),
        "swapped": case("swapped", ["chase", "dog"], ("dog", "chase")),
        "novel": case("novel", ["zorp", "glim"], None),
        "lesion_self": case("lesion_self", ["brain", "use"], None, lesion=True),
    }
    # attribution (tools.lab, gate class AT): did closing the word -> role gate move the (brain, use) parse at all?
    lesion_moved = bool(lever("word->role gate (brain, use)", cases["self"]["got"], cases["lesion_self"]["got"],
                              required=False))
    diag = {"pos_alone": {}, "word_alone": {}, "dual_role": {}}
    try:                                                   # diagnostics only: never part of all_pass, never raise
        for k in range(3):
            diag["pos_alone"]["pos%d" % k] = _isolated(lambda k=k: organ._present(None, k, int(PARAMS["TEST_STEPS"])))
        for w in ("dog", "chase", "brain", "use"):
            diag["word_alone"][w] = _isolated(lambda w=w: organ._present(w, None, int(PARAMS["TEST_STEPS"])))
        for w, k, want in (("cat", 0, "agent"), ("cat", 2, "patient"), ("fish", 0, "agent"), ("fish", 2, "patient")):
            role, c = organ.read_word(w, k)
            diag["dual_role"]["%s@%d" % (w, k)] = {"want": want, "got": role, "counts": c, "ok": role == want}
        diag["dual_role_ok"] = int(sum(v["ok"] for v in diag["dual_role"].values()))
    except Exception as e:                                 # noqa: BLE001
        diag["error"] = repr(e)
    try:
        backend = get_backend()[0].__name__
    except Exception:
        backend = "unknown"
    return {"seed": int(seed), "op_index": None if op_index is None else int(op_index),
            "grid_index": next((i for i, g in enumerate(GRID) if g == PARAMS), None), "params": dict(PARAMS),
            "cases": cases, "all_pass": bool(all(c["pass"] for c in cases.values())), "backend": backend,
            "weights": organ.weight_summary, "diagnostics": diag, "lesion_moved_parse": lesion_moved}


# ── the production seam ─────────────────────────────────────────────────────────────────────────────────────────
_ORGAN_CACHE: Dict[tuple, WordRoleParserOrgan] = {}


def facts_from_chat(chat) -> List[Tuple[str, str, str]]:
    """The composer's own stored facts as (agent, action, patient) string triples (non-str entries skipped)."""
    kb = getattr(getattr(getattr(chat, "inner", None), "composer", None), "kb", None) or []
    out = []
    for entry in kb:
        try:
            fact = entry[0]
            a, v, p = fact.get("agent"), fact.get("action"), fact.get("patient")
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(a, str) and isinstance(v, str) and isinstance(p, str):
            out.append((a, v, p))
    return out


def parse_from_brain(chat, content: Sequence[str], *, seed: int, lesion: bool = False) -> dict:
    """Parse the question's content words with an organ trained on the composer's own stored facts. The training
    stimuli are TRANSCRIBED from the composer's fact index (`chat.inner.composer.kb`): a host transcription residual.
    The self-alias -> 'brain' resolution (QuestionRouter._resolve_self) stays host (named residual). The trained organ
    is cached per (seed, operating point, frozenset of facts). Returns parse()'s dict."""
    facts = facts_from_chat(chat)
    key = (int(seed), tuple(sorted(PARAMS.items())), frozenset(facts))
    organ = _ORGAN_CACHE.get(key)
    if organ is None:
        organ = WordRoleParserOrgan(sorted(facts), seed=seed, params=dict(PARAMS)).train()
        _ORGAN_CACHE[key] = organ
    return organ.parse(list(content), lesion=lesion)


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _check_seed(seed: int, allow_gate: bool, prereg: Optional[str]) -> None:
    if int(seed) in GATE_SEEDS and not (allow_gate and prereg):
        raise SystemExit("seed %d is a gate seed: refused (pass --allow-gate-seed --prereg <path> to run it)" % seed)


def _write(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    ap.add_argument("--op", type=int, default=0, help="GRID index")
    ap.add_argument("--ops", type=int, nargs="*", default=None, help="GRID indices for --sweep (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args(argv)
    seeds = a.seeds if a.sweep else [a.seed]
    for s in seeds:
        _check_seed(s, a.allow_gate_seed, a.prereg)
    if a.sweep:
        out_dir = a.out_dir or os.path.join(_REPO, "research", "findings", "raw", "word_role_parse")
        for op in (a.ops if a.ops is not None else range(len(GRID))):
            for s in seeds:
                res = organ_battery(s, op)
                _write(os.path.join(out_dir, "op%d_s%d.json" % (op, s)), res)
                print("op%d seed%d all_pass=%s %s" % (op, s, res["all_pass"],
                                                       {k: v["pass"] for k, v in res["cases"].items()}))
        return 0
    res = organ_battery(a.seed, a.op)
    if a.prereg:
        res["prereg"] = a.prereg      # tools/s2p_score.py step O requires the registered prereg path here
    if a.out:
        _write(a.out, res)
    print(json.dumps({k: res[k] for k in ("seed", "op_index", "all_pass")}),
          {k: v["pass"] for k, v in res["cases"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
