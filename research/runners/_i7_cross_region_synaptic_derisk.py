"""INTEGRATION #7 -- CROSS-REGION SYNAPTIC INTERACTION (the one-brain merge crosses co-location -> interaction).

Burn-down #1 (`_i7_burndown1_one_brain_merge_derisk`) put the e-prop ACQUISITION slices onto the ONE conversational
`SimulationBridge` / ONE `cp_connections` as DISJOINT slices -- CO-RESIDENCY with ZERO conv<->eprop synapse
(co-location, NOT yet cross-region synaptic INTERACTION, per `project_one_brain_substrate_vs_functional`). Its own
HONEST SCOPE note flagged the next step: "a synaptic pathway conv-cue -> eprop_in AND eprop_out-spikes -> composer
render". THIS arc builds the INPUT half of that: a REAL cross-region synaptic pathway.

WHAT CHANGES (option (a), the acquisition-INPUT host hand-off):
  * Burn-down #1 forward: the host RENDERS the env percept (`_feat`, a legitimate sensory boundary -- "Host code is
    legitimate here EXACTLY as a retinal-image render is") and WRITES it as EXTERNAL CURRENT directly onto `eprop_in`
    (`drive[self.slices[0]] = in_cur`). The percept reaching the e-prop input layer is a HOST INJECTION -- there is no
    synapse between the rest of the brain and eprop_in.
  * THIS arc appends a co-resident SENSORY-RELAY region `eprop_cue` (N_IN neurons) LAST on the SAME merged bridge, and
    injects a FIXED one-to-one synaptic pathway `eprop_cue -> eprop_in` into the SAME union / SAME cp_connections
    (`SA.build_one_brain(..., co_resident_eprop_cue=True)`). The percept is now presented as sensory drive onto
    `eprop_cue`; the brain's OWN synapses carry it to `eprop_in`, which fires from SYNAPTIC transmission, not a host
    current write. eprop_in's external current stays 0.
  * Calibration (build-once, no-training probe; seed 42): at cue_w=2000 the one-to-one relay reproduces the direct
    drive's GRADED eprop_in firing almost exactly (direct 0.041/0.095/0.136 by feature level 0/0.5/1.0 -> cue-relay
    0.048/0.099/0.127), so the feature signal survives one extra spiking stage -> acquisition is preserved.

THE NEW CROSS-REGION TOOTH (load-bearing, real interaction not co-location): LESION the `eprop_cue -> eprop_in`
synapse (zero its cp_connections slots) -> eprop_in receives no drive -> the WHOLE acquisition read path (eprop_in ->
eprop_h1 -> leaky readout) collapses: heldout discrimination falls to a non-discriminating constant. The pathway is
REQUIRED, so the two slice families now genuinely INTERACT through a synapse.

GO (per seed, hook on): (1) conversational byte-identity WITH vs WITHOUT the eprop+cue slices HOLDS (append-LAST,
internal_density=0, no out-edge to any pre-existing region); (2) the merged-#7 chat still GOes with the cue arriving
via SYNAPSES -- taught-recall 3/3, moat FA=0, frozen-readout 0, lesion-gate load-bearing, OOD abstains, post-hoc moat
drops 100%; (3) the cross-region teeth: the cue synapse is real + in the shared cp_connections, AND lesioning it
collapses acquisition.

HONEST SCOPE (what this reaches, what remains): this is a GENUINE cross-region synaptic pathway on ONE substrate that
is LOAD-BEARING -- burn-down #1's ZERO cross-synapse co-location is crossed. It biologizes ONE host seam: the
sensory-relay -> e-prop-input transmission is now synaptic, not a host injection. It is a PARTIAL step toward full
conv<->eprop functional integration. What REMAINS: (i) the presynaptic drive onto eprop_cue is still the host-rendered
env percept (a legitimate sensory boundary, but the CHAT word-cue -> env-percept lookup remains a host seam); (ii) the
eprop_cue region is a purpose-built relay, not one of the pre-existing CONVERSATIONAL faculty regions -- so this is
"a new region interacts with the acquisition slices", one rung below "the composer's own cue representation drives
acquisition"; (iii) the OUTPUT half -- eprop_out spikes -> the patient render -- is untouched (still a host argmax over
the leaky-readout logits). Named scaffolds unchanged: the numpy familiarity gate, the argmax patient read-out, the
host leaky-readout integration, the AI-teacher presentation.

The ONLY sim/-adjacent change is the additive/default-off flag `co_resident_eprop_cue` on `SA.build_one_brain` (a
research runner -- NO sim/ edit). Flag off -> no eprop_cue region -> byte-identical to burn-down #1 / #6.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import logging as _logging  # noqa: E402

for _n in ("SIM_BRIDGE", "sim.bridge", "root"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402

from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners import _corpus_facts_into_live_chat_derisk as CF  # noqa: E402
from research.runners import _teacher_loop_facts_into_live_chat_derisk as I7  # noqa: E402
from research.runners import _i7_burndown1_one_brain_merge_derisk as I7B  # noqa: E402
from tools.lab import attributable_to  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The CROSS-REGION e-prop net: burn-down #1's CoResidentEpropNet, but the FORWARD drives the co-resident SENSORY
# RELAY `eprop_cue` (external current) and lets the fixed one-to-one `eprop_cue -> eprop_in` synapse carry the
# percept to the e-prop input layer (eprop_in external current = 0). Adds a lesion of that synapse.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
class CrossRegionEpropNet(I7B.CoResidentEpropNet):
    def __init__(self, merged_bridge, baseline_snap, eprop_idx, n_in, hidden, k, seed, cue_idx,
                 settle=25, eprop_lr=0.5, w_clip=4000.0, hp=None):
        super().__init__(merged_bridge, baseline_snap, eprop_idx, n_in, hidden, k, seed,
                         settle=settle, eprop_lr=eprop_lr, w_clip=w_clip, hp=hp)
        xp = self._xp
        g_in = np.asarray(eprop_idx["in"], dtype=np.int64)
        g_cue = np.asarray(cue_idx, dtype=np.int64)
        if len(g_cue) != len(g_in):
            raise RuntimeError("eprop_cue slice size must equal eprop_in slice size (one-to-one relay)")
        if not np.array_equal(g_cue, np.arange(int(g_cue[0]), int(g_cue[-1]) + 1)):
            raise RuntimeError("eprop_cue slice is not contiguous -- append-LAST region layout broken")
        self.cue_slice = slice(int(g_cue[0]), int(g_cue[-1]) + 1)
        # sparse position map for the DIAGONAL cue->in edge (34 entries) into cp_connections.data.
        coo = self.br._get_cached_coo()
        row = np.asarray(to_host(coo.row)).astype(np.int64)
        col = np.asarray(to_host(coo.col)).astype(np.int64)
        pos = {(int(row[i]), int(col[i])): i for i in range(row.shape[0])}
        slots = []
        for kk in range(len(g_cue)):
            key = (int(g_cue[kk]), int(g_in[kk]))
            if key not in pos:
                raise RuntimeError("cue->eprop_in edge missing from cp_connections (position map failed)")
            slots.append(pos[key])
        self._cue_data_idx = xp.asarray(np.asarray(slots, dtype=np.int64))
        self._cue_w_backup = np.asarray(to_host(self.br.cp_connections.data[self._cue_data_idx])).copy()
        self.lesion_cue = False

    def lesion_cue_synapse(self):
        """Zero the cue->eprop_in weights on the shared cp_connections (the cross-region lesion)."""
        self.br.cp_connections.data[self._cue_data_idx] = self._xp.asarray(
            np.zeros(int(self._cue_data_idx.shape[0]), dtype=np.float32))
        self.lesion_cue = True

    def restore_cue_synapse(self):
        self.br.cp_connections.data[self._cue_data_idx] = self._xp.asarray(
            self._cue_w_backup.astype(np.float32))
        self.lesion_cue = False

    def _forward_record(self, feat_row, reset_rates=True):
        xp = self._xp
        n = self.n_total
        if self.reset_state:
            SA._restore_state(self.br, self._baseline_snap)
        if reset_rates and self.br.cp_bdsp_E is not None:   # None on the merged bridge (enable_bdsp OFF) -> skipped
            self.br.cp_bdsp_E[...] = 0.0
            self.br.cp_bdsp_B[...] = 0.0
            self.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
        drive = self._base_drive()
        f = np.asarray(feat_row, dtype=np.float32)
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * f, 0.0, 1600.0)
        # CROSS-REGION: drive the SENSORY RELAY `eprop_cue`; eprop_in (slices[0]) stays at base (0) and fires only
        # from the cue->in synapse. Lesioning that synapse => eprop_in silent => acquisition collapses.
        drive[self.cue_slice] = self._broadcast(in_cur, 0).astype(np.float32)
        drive_xp = xp.asarray(drive)
        if getattr(self.br, "cp_bdsp_apical_drive", None) is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        T = self.settle_steps
        sp = np.zeros((T, n), dtype=np.float32)
        vv = np.zeros((T, n), dtype=np.float32)
        for t in range(T):
            self.br.cp_external_input_current = drive_xp
            self.br._run_one_simulation_step()
            sp[t] = np.asarray(to_host(self.br.cp_firing_states), dtype=np.float32)
            vv[t] = np.asarray(to_host(self.br.cp_membrane_potential_v), dtype=np.float32)
        acts = [np.zeros(self.sizes[li], dtype=np.float64) for li in range(len(self.sizes))]
        return sp, vv, acts


def _mk_net(merged, snap, eprop_idx, cue_idx, seed, freeze=False):
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    net = CrossRegionEpropNet(merged, snap, eprop_idx, n_in=I7.N_IN, hidden=I7.HIDDEN, k=I7.K, seed=seed,
                              cue_idx=cue_idx, settle=I7.SETTLE, eprop_lr=I7.EPROP_LR, w_clip=I7.W_CLIP, hp=hp)
    if freeze:
        net.eprop_lr = 0.0
    return net


def _teach(seed, env, merged, eprop_idx, cue_idx, snap, mispaired=False, single_class=False, freeze=False):
    """One teacher presentation on the cross-region net (== I7B._teach_merged, net swapped for CrossRegionEpropNet)."""
    net = _mk_net(merged, snap, eprop_idx, cue_idx, seed, freeze=freeze)
    fam = I7._make_fam(seed)
    for r in I7.TAUGHT:
        fam.imprint(env, r, "eats")
    ro0 = I7._readout_norm(net)
    if single_class:
        Xtr, ytr = I7._single_class_batch(env, seed, I7.N_DRAWS)
    else:
        Xtr, ytr = I7._contrastive_batch(env, seed, I7.N_DRAWS, mispaired=mispaired)
    I7._train_eprop(net, Xtr, ytr, I7.EPOCHS, I7.BATCH, seed)
    return net, fam, float(abs(I7._readout_norm(net) - ro0))


def _heldout_discrim(net, env):
    """Per-taught-referent heldout argmax accuracy + the majority-class discrimination summary (the CT1 quantities)."""
    held = {r: I7._heldout_acc(net, env, r, I7.PATIENT_WORDS.index(p)) for r, p in I7.TAUGHT.items()}
    maj = {r: I7.PATIENT_WORDS[I7._majority(net, env, r, "eats")[0]] for r in I7.TAUGHT}
    distinct = len(set(maj.values()))
    n_correct = sum(1 for r in I7.TAUGHT if maj[r] == I7.TAUGHT[r])
    return {"heldout": held, "heldout_mean": float(np.mean(list(held.values()))),
            "majority": maj, "distinct_classes": int(distinct), "n_facts_correct": int(n_correct)}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BYTE-IDENTITY: build the #7 brain WITHOUT vs WITH the appended eprop + eprop_cue slices; the CONVERSATIONAL
# neurons must stay byte-identical (reuses I7B's per-parameter-reseed heterogeneity hook + izh hashing).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def byte_identity_cue(seed, hook=True):
    """Conversational byte-identity WITH vs WITHOUT the appended eprop+cue slices. Two claims:
      (1) SUBSTRATE byte-identity: every per-neuron Izhikevich parameter over the first n_pre (conversational) neurons
          + the composer concept codes are hash-identical (append-LAST + internal_density=0 + the per-parameter reseed
          hook). This is the load-bearing substrate claim.
      (2) DECISION-transcript identity under a NOISE-CONTROLLED comparison: the appended slices grow num_neurons, and
          the arbiter/affect reads ride an UNSEEDED OU read-noise whose per-step `cp.random.randn(n)` stream position
          depends on n -- so more neurons shift the conversational neurons' OU samples from step 2 onward (the codebase
          excludes these raw floats as EVAL noise, _corpus_facts...:418; I7B's pin_ou_noise only fixes step 1). We
          ISOLATE the substrate's decision FUNCTION by zeroing the OU noise amplitude (`ou_noise_std=0` -> OU decays to
          its constant mean, position-independent) on BOTH bridges -> the decision comparison is deterministic and, if
          the substrate is truly identical, MUST match. This is the complete version of the pin (not a weakening: the
          real smoke runs the full OU; here we only remove the eval-noise confound from the identity assertion)."""
    if hook:
        I7B.apply_heterogeneity_append_invariant_hook()
    xp, _ = get_backend()
    turns = list(TT.HUMAN_TURNS)
    V = I7.V

    b0, c0, i0, s0 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True, vocab=V)
    n_pre = int(b0.core_config.num_neurons)
    cc0 = CF._concept_hash(c0)
    izh0 = {p: I7B._hash_first(getattr(b0, p), n_pre) for p in I7B._IZH_PARAMS if getattr(b0, p, None) is not None}
    _v0, f0 = SA._store_facts(c0)
    b0.ou_noise_std = 0.0                    # ISOLATE the substrate decision function (see docstring (2))
    t0 = CF.run_chat(b0, xp, i0, s0, c0, f0, turns)

    b1, c1, i1, s1 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True, vocab=V,
                                        co_resident_eprop=True, eprop_dims=(I7.N_IN, I7.HIDDEN, I7.K),
                                        co_resident_eprop_cue=True)
    cc1 = CF._concept_hash(c1)
    izh1 = {p: I7B._hash_first(getattr(b1, p), n_pre) for p in I7B._IZH_PARAMS if getattr(b1, p, None) is not None}
    _v1, f1 = SA._store_facts(c1)
    b1.ou_noise_std = 0.0
    t1 = CF.run_chat(b1, xp, i1, s1, c1, f1, turns)

    izh_diffs = sorted(p for p in izh0 if izh0[p] != izh1.get(p))
    dec_ident = bool(json.dumps(CF._decision_view(t0), sort_keys=True, default=str)
                     == json.dumps(CF._decision_view(t1), sort_keys=True, default=str))
    substrate_ident = bool(len(izh_diffs) == 0 and cc0 == cc1)
    return {
        "hook_applied": bool(hook),
        "izh_params_identical": bool(len(izh_diffs) == 0), "izh_params_that_differ": izh_diffs,
        "concept_codes_identical": bool(cc0 == cc1),
        "substrate_byte_identical": substrate_ident,
        "num_neurons_without_eprop": n_pre, "num_neurons_with_eprop_cue": int(b1.core_config.num_neurons),
        "n_appended_neurons": int(b1.core_config.num_neurons) - n_pre,
        "decision_transcript_identical": dec_ident,
        "held": bool(substrate_ident and dec_ident),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE 1-SEED CROSS-REGION GO SMOKE + the new cross-region tooth.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def smoke_seed(seed, hook=True):
    if hook:
        I7B.apply_heterogeneity_append_invariant_hook()
    xp, _ = get_backend()
    t_start = time.time()
    V = I7.V

    bridge, comp, idx, snap = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                                 vocab=V, co_resident_eprop=True,
                                                 eprop_dims=(I7.N_IN, I7.HIDDEN, I7.K), co_resident_eprop_cue=True)
    eprop_idx = idx["eprop"]
    cue_idx = eprop_idx["cue"]
    _vc, curated_facts = SA._store_facts(comp)
    kb_after_store = len(comp.kb)
    env = I7._make_env(int(seed))
    facts_all = list(curated_facts) + list(I7.TAUGHT_FACTS)
    turns = I7._turns()

    # ---- TREATMENT: the cross-region net trains on the merged bridge with the cue arriving via SYNAPSES ----
    net_t, fam_t, readout_moved = _teach(int(seed), env, bridge, eprop_idx, cue_idx, snap)
    kb_after_teach = len(comp.kb)

    # structural one-brain teeth (co-residency, from burn-down #1) ...
    single_bridge = bool(net_t.br is bridge and comp._merged is bridge)
    n_ff = int(sum(int(np.asarray(to_host(a)).shape[0]) for a in net_t._data_idx_flat))
    n_syn_total = int(bridge.cp_connections.nnz)

    # ... PLUS the NEW cross-region synapse teeth: the cue->eprop_in edge is real, in the shared cp_connections.
    n_cue_syn = int(np.asarray(to_host(net_t._cue_data_idx)).shape[0])
    cue_slots = set(int(x) for x in np.asarray(to_host(net_t._cue_data_idx)).tolist())
    cue_syn_in_shared = bool(n_cue_syn == I7.N_IN and max(cue_slots) < n_syn_total
                             and net_t._cue_data_idx is not None
                             and net_t.br.cp_connections is bridge.cp_connections)
    # the cue edge is DISJOINT from the plastic FF slots (a distinct, fixed pathway).
    ff_slots = set(int(x) for a in net_t._data_idx_flat for x in np.asarray(to_host(a)).tolist())
    cue_disjoint_from_ff = bool(len(cue_slots & ff_slots) == 0)

    shim_treat = I7.ChatShim(comp, env, net=net_t, fam=fam_t, enabled=True, use_gate=True)
    tr_treat = CF.run_chat(bridge, xp, idx, snap, shim_treat, facts_all, turns)
    sum_treat = CF._chat_summary(tr_treat)
    recall_treat, recalled = I7._taught_recall(tr_treat)

    moat_fa, moat_ex = I7._moat_battery(shim_treat)
    teeth = CF.posthoc_teeth(shim_treat, facts_all, seed=int(seed))

    off_shim = I7.ChatShim(comp, env, net=net_t, fam=fam_t, enabled=True, use_gate=False, use_conf=True)
    gate_off_fa, _ = I7._moat_battery(off_shim)
    intact_margin, lesion_margin = I7._lesion_margin(fam_t, env)

    # ---- CT1 discrimination WITH the cue synapse INTACT (the acquisition genuinely learned through the pathway) ----
    disc_intact = _heldout_discrim(net_t, env)
    heldout_head = disc_intact["heldout"][I7.HEADLINE_REFERENT]

    # ════════════════════════════════════════════════════════════════════════════════════════════════════════
    # THE CROSS-REGION LESION TOOTH: cut the cue->eprop_in synapse -> the acquisition INPUT collapses. Read the
    # discrimination with the synapse lesioned, then RESTORE (so nothing downstream is left clobbered).
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════
    net_t.lesion_cue_synapse()
    disc_lesion = _heldout_discrim(net_t, env)
    # the chat itself collapses too: the taught cues no longer recall (the read path is severed).
    shim_lesion = I7.ChatShim(comp, env, net=net_t, fam=fam_t, enabled=True, use_gate=True)
    tr_lesion = CF.run_chat(bridge, xp, idx, snap, shim_lesion, facts_all, turns)
    recall_lesion, _ = I7._taught_recall(tr_lesion)
    net_t.restore_cue_synapse()
    # confirm restoration returns discrimination (the lesion is the CAUSE, not an irreversible corruption).
    disc_restore = _heldout_discrim(net_t, env)

    # LOAD-BEARING = cutting the synapse DESTROYS the acquisition. The robust signal is the DISCRIMINATION collapse:
    # eprop_in goes fully silent (no cue, no tonic on the input slice) -> eprop_h1 sees only its constant tonic ->
    # the readout is a CONSTANT class -> distinct_classes 3->1 (the net can no longer tell the referents apart) and
    # heldout_mean drops sharply. Chat recall drops too, but not necessarily to 0 (if the constant class happens to
    # equal a taught patient, that ONE referent accidentally "matches") -> gate on recall DROPPING, not ==0. Restore
    # recovers discrimination (the lesion is the CAUSE, reversible).
    xreg_load_bearing = bool(
        disc_intact["heldout_mean"] > 0.6 and disc_intact["distinct_classes"] >= 2
        and disc_intact["n_facts_correct"] >= 2
        and disc_lesion["distinct_classes"] <= 1
        and disc_lesion["heldout_mean"] < disc_intact["heldout_mean"] - 0.30
        and recall_lesion < recall_treat
        and disc_restore["heldout_mean"] > 0.6 and disc_restore["distinct_classes"] >= 2)
    recall_collapse_attrib = attributable_to(
        "taught-fact chat recall through the cue->eprop_in synapse (intact vs lesioned)",
        float(recall_treat), float(recall_lesion))

    # one-brain tooth: an e-prop teaching pass moves ONLY the eprop FF slots (conv + cue synapses byte-unchanged).
    net_probe = _mk_net(bridge, snap, eprop_idx, cue_idx, int(seed))
    data_pre = np.asarray(to_host(bridge.cp_connections.data)).copy()
    Xb, yb = I7._contrastive_batch(env, int(seed), I7.N_DRAWS)
    net_probe.fit_readout_norm(Xb)
    net_probe.train_batch(Xb[:I7.BATCH], yb[:I7.BATCH])
    data_post = np.asarray(to_host(bridge.cp_connections.data))
    changed = set(int(x) for x in np.where(data_pre != data_post)[0].tolist())
    probe_ff = set(int(x) for a in net_probe._data_idx_flat for x in np.asarray(to_host(a)).tolist())
    probe_cue = set(int(x) for x in np.asarray(to_host(net_probe._cue_data_idx)).tolist())
    moves_confined = bool(changed.issubset(probe_ff) and len(changed) > 0)
    cue_synapse_fixed = bool(len(changed & probe_cue) == 0)   # e-prop never touches the cue pathway

    # ---- FROZEN-READOUT control (identical teaching, eprop_lr=0 -> zero readout -> taught patient not recalled) ----
    net_fz, fam_fz, readout_moved_frozen = _teach(int(seed), I7._make_env(int(seed)), bridge, eprop_idx, cue_idx,
                                                  snap, freeze=True)
    shim_frozen = I7.ChatShim(comp, env, net=net_fz, fam=fam_fz, enabled=True, use_gate=True)
    tr_frozen = CF.run_chat(bridge, xp, idx, snap, shim_frozen, facts_all, turns)
    recall_frozen, _ = I7._taught_recall(tr_frozen)

    recall_attrib = attributable_to(
        "taught-fact chat recall from the cross-region e-prop weight change (trained vs frozen-readout)",
        float(recall_treat), float(recall_frozen))

    # ---- GO flags ----
    recall_ok = bool(recall_treat == len(I7.TAUGHT))
    moat_ok = bool(moat_fa == 0)
    frozen_ok = bool(readout_moved_frozen <= 1e-3 and recall_frozen == 0)
    lesion_ok = bool(gate_off_fa > 0 and lesion_margin < intact_margin - 0.30)
    ood_ok = bool(sum_treat["ood_abstained"] == sum_treat["ood_turns"]
                  and sum_treat["ungrounded_word_total"] == 0 and sum_treat["confabulated"] == 0)
    posthoc_ok = bool(abs(teeth["unsupported_drop_rate"] - 1.0) < 1e-9 and teeth["unsupported_props"] > 0
                      and abs(teeth["supported_keep_rate"] - 1.0) < 1e-9)
    ct1_ok = bool(heldout_head > 0.6 and disc_intact["distinct_classes"] >= 2
                  and disc_intact["n_facts_correct"] >= 2)
    kb_unchanged = bool(kb_after_teach == kb_after_store)

    cross_region_teeth = {
        "single_SimulationBridge (net.br IS comp._merged IS bridge)": single_bridge,
        "cue->eprop_in synapse in SAME cp_connections": cue_syn_in_shared,
        "n_cue_synapses": n_cue_syn, "n_eprop_ff_synapses": n_ff, "n_total_synapses": n_syn_total,
        "cue_edge_disjoint_from_plastic_FF": cue_disjoint_from_ff,
        "e-prop_teach_moves_ONLY_eprop_ff (cue+conv fixed)": moves_confined and cue_synapse_fixed,
        "n_data_slots_changed_by_a_teach_pass": len(changed),
        "LESION cue->eprop_in: heldout_mean intact->lesion->restore":
            [round(disc_intact["heldout_mean"], 3), round(disc_lesion["heldout_mean"], 3),
             round(disc_restore["heldout_mean"], 3)],
        "LESION cue->eprop_in: distinct_classes intact->lesion":
            [disc_intact["distinct_classes"], disc_lesion["distinct_classes"]],
        "LESION cue->eprop_in: chat recall intact->lesion": [recall_treat, recall_lesion],
        "cross_region_pathway_load_bearing": xreg_load_bearing,
    }
    teeth_ok = bool(single_bridge and cue_syn_in_shared and cue_disjoint_from_ff
                    and moves_confined and cue_synapse_fixed and xreg_load_bearing)

    smoke_go = bool(recall_ok and moat_ok and frozen_ok and lesion_ok and ood_ok and posthoc_ok
                    and ct1_ok and kb_unchanged and teeth_ok)

    return {
        "seed": int(seed), "hook_applied": bool(hook), "elapsed_s": round(time.time() - t_start, 1),
        "num_neurons": int(bridge.core_config.num_neurons),
        "taught_recall": recall_treat, "recalled": recalled, "recall_ok_3of3": recall_ok,
        "moat_false_accepts": moat_fa, "moat_ok": moat_ok, "moat_examples": moat_ex[:3],
        "frozen_recall": recall_frozen, "frozen_readout_moved": readout_moved_frozen, "frozen_ok": frozen_ok,
        "gate_off_false_accepts": gate_off_fa, "intact_margin": intact_margin, "lesion_margin": lesion_margin,
        "lesion_gate_load_bearing": lesion_ok,
        "ood_abstained": sum_treat["ood_abstained"], "ood_turns": sum_treat["ood_turns"],
        "confabulated": sum_treat["confabulated"], "ungrounded_word_total": sum_treat["ungrounded_word_total"],
        "ood_ok": ood_ok,
        "posthoc_unsupported_drop_rate": teeth["unsupported_drop_rate"],
        "posthoc_supported_keep_rate": teeth["supported_keep_rate"],
        "posthoc_unsupported_props": teeth["unsupported_props"], "posthoc_ok": posthoc_ok,
        "heldout_headline_acc": heldout_head, "heldout_mean_intact": disc_intact["heldout_mean"],
        "heldout_mean_lesion": disc_lesion["heldout_mean"], "ct1_ok": ct1_ok,
        "readout_moved_treatment": readout_moved, "kb_unchanged": kb_unchanged,
        "recall_attributable_to_weight_change": recall_attrib,
        "recall_collapse_attributable_to_cue_lesion": recall_collapse_attrib,
        "CROSS_REGION_TEETH": cross_region_teeth, "teeth_ok": teeth_ok,
        "SMOKE_GO": smoke_go,
    }


def sweep(seeds, hook=True):
    per = []
    for sd in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            bi = byte_identity_cue(int(sd), hook=hook)
            sm = smoke_seed(int(sd), hook=hook)
        per.append({"seed": int(sd), "byte_identity_held": bool(bi["held"]),
                    "izh_params_that_differ": bi["izh_params_that_differ"], "SMOKE_GO": bool(sm["SMOKE_GO"]),
                    "taught_recall": sm["taught_recall"], "moat_false_accepts": sm["moat_false_accepts"],
                    "frozen_recall": sm["frozen_recall"], "lesion_gate_load_bearing": sm["lesion_gate_load_bearing"],
                    "cross_region_load_bearing": sm["CROSS_REGION_TEETH"]["cross_region_pathway_load_bearing"],
                    "heldout_mean_intact": sm["heldout_mean_intact"], "heldout_mean_lesion": sm["heldout_mean_lesion"],
                    "teeth_ok": sm["teeth_ok"], "cross_region_teeth": sm["CROSS_REGION_TEETH"]})
        print(f"seed {sd}: byte_id={per[-1]['byte_identity_held']} SMOKE_GO={per[-1]['SMOKE_GO']} "
              f"recall={per[-1]['taught_recall']} moat_fa={per[-1]['moat_false_accepts']} "
              f"xreg_load_bearing={per[-1]['cross_region_load_bearing']} "
              f"held(intact->lesion)={per[-1]['heldout_mean_intact']:.2f}->{per[-1]['heldout_mean_lesion']:.2f}")
    n = len(per)
    agg = {"hook_applied": bool(hook), "n_seeds": n,
           "n_byte_identity_held": sum(1 for r in per if r["byte_identity_held"]),
           "n_smoke_go": sum(1 for r in per if r["SMOKE_GO"]),
           "n_cross_region_load_bearing": sum(1 for r in per if r["cross_region_load_bearing"]),
           "GO_all": bool(n == len(seeds) and all(r["byte_identity_held"] and r["SMOKE_GO"]
                                                  and r["cross_region_load_bearing"] for r in per)),
           "per_seed": per}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated -> SELF-SWEEP (byte-id + smoke per seed)")
    ap.add_argument("--byte-identity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-hook", action="store_true", help="disable the append-invariant heterogeneity hook")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    hook = not args.no_hook

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        agg = sweep(seeds, hook=hook)
        print("=== CROSS-REGION SELF-SWEEP AGGREGATE ===")
        print(json.dumps(agg, indent=2, default=str))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(agg, fh, indent=2, default=str)
        return

    result = {}
    if args.byte_identity or not args.smoke:
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            bi = byte_identity_cue(int(args.seed), hook=hook)
        bi["elapsed_s"] = round(time.time() - t0, 1)
        result["byte_identity"] = bi
        print("=== BYTE-IDENTITY (conversational, WITH vs WITHOUT appended eprop+cue slices) ===")
        print(json.dumps(bi, indent=2, default=str))
        print("VERDICT:", "HELD" if bi["held"] else "FAILED -> STOP (see izh_params_that_differ)")

    if args.smoke:
        sm = smoke_seed(int(args.seed), hook=hook)
        result["smoke"] = sm
        print("=== 1-SEED CROSS-REGION GO SMOKE + TEETH ===")
        print(json.dumps(sm, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
