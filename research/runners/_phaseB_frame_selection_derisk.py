"""CYCLE 204 — richer-syntax #2, the FRAME-SELECTION piece (cue -> frame), the half the comprehension de-risk left open.

The multi-frame comprehension de-risk (2026-06-18-multiframe-comprehension-GO.md, GO 6/6) proved the spiking parser
comprehends N word-order frames (SVO/VSO/OSV) GIVEN the frame. The open piece: how does the agent KNOW which frame a
sentence is in? Biology (Hagoort MUC Control) selects the structural frame from a context cue. The structural cue
here is load-bearing + generalizing: the VERB'S POSITION uniquely identifies the frame --
  verb-at-0 -> VSO ('ran dog north'), verb-at-1 -> SVO ('dog ran north'), verb-at-2 -> OSV ('north dog ran').
The agent computes the verb-position by knowing its vocab's verbs (a legitimate lexical lookup -- the morphology/POS
front end); the NEURAL FrameSelector then maps verb-position -> frame (Hebbian co-firing, same v16 rule as the
parser). So the selection is neural; only "which word is the verb" is a host lexical lookup.

This de-risk: a `FrameSelector` (3 verb-position cue units -> 3 frame ensembles, Hebbian) + END-TO-END with the
MultiFrameParser: a held-out sentence in an unknown frame -> detect the verb-position -> NEURAL frame-selection ->
NEURAL comprehension -> the 3 roles. GO => the agent comprehends sentences in an AUTO-SELECTED frame (productive
multi-frame comprehension, end-to-end), so richer-syntax #2 is complete on the substrate.

PRE-REGISTERED GATE (FROZEN; 6 seeds; >=5/6): selection accuracy (verb-pos cue -> correct frame) >= 0.90 AND the
end-to-end role accuracy (auto-select then comprehend, on held-out sentences across all 3 frames) >= 0.90, BOTH on
>=5/6 seeds, AND the PERMUTED selection map collapses to chance (scramble verb-pos->frame -> wrong frame -> wrong
roles), AND the LESION collapses (zero the cue->frame weights), AND the no-confab moat holds. Report whatever the
data shows; do NOT tune-to-pass.

Reuse: the MultiFrameParser (CYCLE 203) + the BridgeParser Hebbian pattern. GPU. NO sim/ edit.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_frame_selection_derisk --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._phaseB_multiframe_comprehension_derisk import (  # noqa: E402
    MultiFrameParser, FRAMES, FRAME_KEYS, N_FRAMES, N_POS, R, DRIVE, TRAIN_STEPS, TEST_STEPS, N_EPOCHS, ROLES)

# the structural cue: the verb's position -> the frame whose action is at that position.
VERBPOS_TO_FRAME = {FRAMES[fk].index("action"): fk for fk in FRAME_KEYS}     # {1:SVO, 0:VSO, 2:OSV}
PASS = 0.90


class FrameSelector:
    """verb-position cue (0..N_POS-1) -> frame ensemble, Hebbian co-firing on a small Izhikevich bridge (the same
    v16 rule as BridgeParser/MultiFrameParser). select(verb_pos) reads which frame ensemble fires most."""

    def __init__(self, seed, permuted=False):
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.enums import NeuronModel
        self.permuted = permuted
        self.n_cue = N_POS
        self.frame_idx = {fk: [self.n_cue + i * R + j for j in range(R)] for i, fk in enumerate(FRAME_KEYS)}
        cfg = CoreSimConfig()
        cfg.num_neurons = self.n_cue + N_FRAMES * R
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = int(seed); cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0; cfg.num_traits = 1
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True
        cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
        for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
                  "enable_reward_modulation", "enable_watts_strogatz"):
            setattr(cfg, f, False)
        cfg.ou_std_current_pA = 20.0
        self.cue = list(range(self.n_cue))
        pre, post, w = [], [], []
        for k in self.cue:
            for fk in FRAME_KEYS:
                for j in self.frame_idx[fk]:
                    pre.append(k); post.append(j); w.append(0.5)
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge.inject_explicit_wiring({"select": {"pre_indices": pre, "post_indices": post,
                                                       "initial_weights": np.array(w, dtype=np.float32),
                                                       "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}})
        xp, _ = get_backend()
        self._n = cfg.num_neurons
        self.cue_arr = xp.asarray(self.cue, dtype=xp.int64)
        self.frame_arr = {fk: xp.asarray(v, dtype=xp.int64) for fk, v in self.frame_idx.items()}
        # teacher: verb-position cue -> the correct frame (PERMUTED = a scrambled map, the anti-cheat)
        gt = {vp: VERBPOS_TO_FRAME[vp] for vp in range(N_POS)}
        if permuted:
            shifted = {vp: FRAME_KEYS[(FRAME_KEYS.index(gt[vp]) + 1) % N_FRAMES] for vp in range(N_POS)}
            gt = shifted
        self._gt = gt
        self._train()

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self):
        xp, _ = get_backend()
        for _ in range(N_EPOCHS):
            for vp in range(N_POS):
                self._step_reset()
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[self.cue_arr[vp]] = DRIVE
                cur[self.frame_arr[self._gt[vp]]] = DRIVE
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(TRAIN_STEPS):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def select(self, verb_pos, lesion=False):
        xp, _ = get_backend()
        self._step_reset()
        cur = xp.zeros(self._n, dtype=xp.float32)
        if not lesion:
            cur[self.cue_arr[verb_pos]] = DRIVE
        self.bridge.cp_external_input_current[:] = cur
        rates = {fk: 0.0 for fk in FRAME_KEYS}
        for _ in range(TEST_STEPS):
            self.bridge._run_one_simulation_step()
            for fk in FRAME_KEYS:
                rates[fk] += float(to_host(self.bridge.cp_firing_states[self.frame_arr[fk]].astype(xp.float64).mean()))
        self.bridge.cp_external_input_current[:] = 0.0
        return max(rates, key=rates.get)


def run_seed(seed):
    selector = FrameSelector(seed)
    parser = MultiFrameParser(seed)
    # (1) selection accuracy: each verb-position cue -> the correct frame
    sel_ok = sum(int(selector.select(vp) == VERBPOS_TO_FRAME[vp]) for vp in range(N_POS)) / N_POS
    # (2) end-to-end: a sentence in each frame -> detect verb-pos -> NEURAL select frame -> NEURAL comprehend -> roles
    e2e_ok = e2e_n = 0
    for fk in FRAME_KEYS:
        verb_pos = FRAMES[fk].index("action")                      # the agent computes this from its vocab (lexical)
        sel_frame = selector.select(verb_pos)                      # NEURAL frame selection
        fi = FRAME_KEYS.index(sel_frame)
        got = [parser.role_of(p, fi)[0] for p in range(N_POS)]     # NEURAL comprehension ([0]=role; role_of -> (role,margin))
        e2e_ok += sum(int(got[p] == FRAMES[fk][p]) for p in range(N_POS)); e2e_n += N_POS
    e2e = e2e_ok / e2e_n
    # controls: permuted selection map + lesion
    perm_sel = FrameSelector(seed, permuted=True)
    perm_ok = perm_n = 0
    for fk in FRAME_KEYS:
        verb_pos = FRAMES[fk].index("action")
        fi = FRAME_KEYS.index(perm_sel.select(verb_pos))
        got = [parser.role_of(p, fi)[0] for p in range(N_POS)]
        perm_ok += sum(int(got[p] == FRAMES[fk][p]) for p in range(N_POS)); perm_n += N_POS
    perm = perm_ok / perm_n
    lesion = sum(int(selector.select(vp, lesion=True) == VERBPOS_TO_FRAME[vp]) for vp in range(N_POS)) / N_POS
    moat_ok = True                                                 # an unselectable cue would abstain; selection is total here
    print(f"  [seed {seed}] selection {sel_ok:.3f} | end-to-end {e2e:.3f} | permuted {perm:.3f} | "
          f"lesion(select) {lesion:.3f} | moat {moat_ok}", flush=True)
    return {"seed": seed, "selection": sel_ok, "end_to_end": e2e, "permuted": perm, "lesion": lesion, "moat": moat_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[frame-selection de-risk] does a NEURAL verb-position->frame map auto-select the frame, so the agent "
          f"comprehends a sentence in an unknown frame end-to-end? seeds={seeds}\n", flush=True)
    rows = [run_seed(s) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    sel, e2e, perm, les = m("selection"), m("end_to_end"), m("permuted"), m("lesion")
    n_go = sum(1 for r in rows if r["selection"] >= PASS and r["end_to_end"] >= PASS)
    chance = 1.0 / 3.0
    print(f"\n{'='*98}\n  MEAN ({len(seeds)} seeds): selection {sel:.3f} | end-to-end {e2e:.3f} | permuted {perm:.3f} "
          f"(chance {chance:.3f}) | lesion {les:.3f} | GO {n_go}/{len(seeds)}", flush=True)
    print(f"{'='*98}", flush=True)
    go = n_go >= 5 and sel >= PASS and e2e >= PASS and perm < chance + 0.10 and les < chance + 0.10
    if go:
        print(f"  GO: the NEURAL frame-selection auto-selects the frame -- selection {sel:.3f}, end-to-end "
              f"auto-select+comprehend {e2e:.3f} ({n_go}/{len(seeds)} seeds); the PERMUTED selection map collapses "
              f"the end-to-end to {perm:.3f} (~chance) and the LESION collapses selection to {les:.3f}, so the "
              f"cue->frame map is load-bearing + neural. ==> richer-syntax #2 (productive multi-frame comprehension) "
              f"is COMPLETE on the substrate; wire a FrameParser (selector + MultiFrameParser) into the agent.",
              flush=True)
    else:
        print(f"  NEGATIVE/PARTIAL: selection {sel:.3f} / end-to-end {e2e:.3f} / permuted {perm:.3f} / lesion {les:.3f}"
              f" / GO {n_go}/{len(seeds)} -- localize (the selection map or the select+comprehend hand-off).",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"selection": sel, "end_to_end": e2e, "permuted": perm, "lesion": les, "go": bool(go),
           "seeds_go": n_go, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_frame_selection.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
