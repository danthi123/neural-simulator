"""CONSOLIDATION Probe 1 (option A) -- co-resident `OneBrainComposer` byte-identity on an OFFSET rf slice.

The owner-approved consolidation of the merged nav+conv bridge onto the persistent-loop `OneBrainComposer`. This is
the SMALLEST falsification of the ONLY unproven step in that arc: does `OneBrainComposer` reproduce its standalone
behavior when its bridge handle is an OFFSET `rf` slice of a LARGER (merged-stub) bridge, with EVERY absolute index
rebased by the slice base?

It is the exact `MergedRFComposer` co-residence anti-cheat (`tests/test_merged_rf_composer_coresident.py` 5/5 ==
standalone to atol 1e-9) applied ONE LEVEL UP -- to `OneBrainComposer` (the persistent-loop composer that holds the
whole who/what pipeline: parser front-end + synaptic multi-fact store + spiking cleanup + the no-confab moat).

DE-RISK SHAPE (per `research/findings/raw/_consolidation_onebrain_limbic_scoping.md` Probe 1):
  * Build a small framework-free Izhikevich bridge with a nav-STUB block forcing a non-zero offset + an `rf` region
    sized for `OneBrainComposer.n_total` (mirror the test's NAV_SLICE + RF_SIZE discipline).
  * Construct a co-resident `OneBrainComposer` whose bridge handle IS that merged-stub bridge and whose every absolute
    index (store_base/q_base/c_base/bat_q_base/bat_c_base/n_total + the parser slice + the rf_mask) is rebased by the
    rf-slice base -- the `CoResidentOneBrainComposer` subclass below (a NEW opt-in; `one_brain_composer.py` UNTOUCHED).
  * Run the full who/what/yes-no matrix + every `is None`/"unknown" abstention against a STANDALONE `OneBrainComposer`
    (same seed/V/D/persistent_loop/spiking-cleanup). Assert BYTE-IDENTICAL (atol 1e-9) on every answer.

WHY THE RF PATH IS BYTE-IDENTICAL ACROSS N + OFFSET: the resonate loop (`_rf_advance_one`, sim/bridge.py:5719) is PURE
complex dynamics -- NO OU noise -- and the masked write-back + the (N,N) complex CSR (nonzeros only inside the slice
block) make `(W@z)[slice]` depend ONLY on `z[slice]` + the slice's weights. So the store/query/cleanup ops reproduce
exactly when the kick + weights are the SAME relative pattern shifted by `rf_base`.

PARSER ISOLATION (scoping LAYOUT decision 2b -- the `MergedNavConvAgent.hear` pattern): comprehension (the parser) is a
SEPARATE, already-validated concern whose Izhikevich+OU dynamics depend on the bridge's N (the OU-noise array is size-N
-> a larger bridge shifts the RNG stream). To isolate the genuine de-risk (the RF index rebasing) from parser RNG-stream
noise, BOTH composers are driven via `store()` (resolved roles) exactly as the merged agent does (the parser supplies the
roles, the composer stores). The parser-front-end byte-identity is out of THIS probe's scope (it is the Route-D question).

ANTI-CHEATS (asserted below):
  1. BYTE-IDENTITY: co-resident == standalone to atol 1e-9, on EVERY stored-fact answer.
  2. MOAT-PRESERVED (HARD): every `is None`/"unknown" abstention is IDENTICAL on both composers.
  3. SLICE-ISOLATION: a co-resident Izhikevich (nav-stub) slice's v/u is BYTE-identical across a composer op.

CPU / numpy / small (SIM_BACKEND=numpy, V<=16, D<=64). The byte-identity IS the de-risk, NOT scale.
NO `sim/` edit (reuse-by-import: OneBrainComposer + BridgeParser(index_offset=) + the masked rf_kick).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Force numpy backend BEFORE importing sim (the de-risk is logic/CPU; do not contend the GPU).
os.environ.setdefault("SIM_BACKEND", "numpy")

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend
from research.runners.brain_conversational_agent import BridgeParser
from research.runners.one_brain_composer import OneBrainComposer


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The merged-stub bridge: a framework-free Izhikevich bridge with a nav-STUB block (forces a non-zero rf offset) +
# an `rf` region sized to hold the full OneBrainComposer layout. Mirrors tests/test_merged_rf_composer_coresident.py
# `_izh_bridge`, the same minimal co-resident host (no plasticity for the nav stub itself; Hebbian ON globally so the
# co-resident parser slice can train exactly as on the standalone composer's own bridge).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_merged_stub_bridge(n_total, seed=42):
    """An Izhikevich bridge of `n_total` neurons (= nav_stub + OneBrainComposer.n_total), config matched to
    `one_brain_composer.build_coresident_bridge` (Hebbian ON for the parser; the RF region has no cp_connections, so
    global Hebbian has nothing to touch there; OU on so the co-resident parser trains identically)."""
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n_total)
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
    cfg.enable_rf_cudagraph = False
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


class CoResidentOneBrainComposer(OneBrainComposer):
    """`OneBrainComposer` whose RF/store/cleanup ops run on a SLICE of a shared (merged) bridge instead of on its own
    private bridge -- the consolidation port. The SAME pattern `MergedRFComposer` performs for `RFPhasorComposer`,
    applied one level up.

    `OneBrainComposer.__init__` builds the layout from `self.P` and a PRIVATE bridge at `[0:n_total]`. This subclass
    redirects the bridge handle to `merged_bridge` and REBASES every absolute index by `rf_base`:
      * the parser slice moves to `[rf_base : rf_base + P_local]` (BridgeParser(index_offset=rf_base));
      * every base (P, store_base, q_base, c_base, bat_q_base, bat_c_base) is shifted += rf_base, so every downstream
        `self.<base> + i*...` lands inside the slice;
      * `self.n_total` is REDEFINED to the merged bridge's N (so every `np.zeros(self.n_total)` kick + every
        `_build_complex_csr(self.n_total, ...)` is full-N, the merged-bridge size);
      * `self.rf_mask` covers exactly the composer's layout span on the merged bridge ([rf_base : rf_base + span]).

    Because the resonate loop is pure complex dynamics (no OU) + masked + the CSR is block-local, the RF ops reproduce
    the standalone composer's results bit-for-bit. The parser is given the SAME relative wiring at index_offset=rf_base
    (comprehension is driven via `store()` in the de-risk, so parser RNG-stream differences do not enter the compared
    answers; see the module docstring)."""

    def __init__(self, merged_bridge, rf_base, **kwargs):
        # Reproduce OneBrainComposer.__init__'s feature/layout computation WITHOUT building the private bridge/parser
        # (which __init__ does at lines 282-283). We mirror the relevant body, then rebase. Keeping this in the subclass
        # leaves one_brain_composer.py byte-untouched (a NEW opt-in alongside MergedRFComposer).
        from research.runners.rf_phasor_composer import RFPhasorComposer
        seed = int(kwargs.get("seed", 42)); D = int(kwargs.get("D", 128))
        vocab = kwargs.get("vocab", None); period = int(kwargs.get("period", 200))
        k_max = int(kwargs.get("k_max", 32))
        grounded_codes = kwargs.get("grounded_codes", None)
        # --- the flag fields (defaults match OneBrainComposer.__init__ signature) ---
        self.seed = seed; self.D = D; self.period = period
        self.trace = bool(kwargs.get("trace", False)); self.last_trace = None
        self.integrated_loop = bool(kwargs.get("integrated_loop", False))
        self.persistent_loop = bool(kwargs.get("persistent_loop", False))
        self.sequencer_match_thresh = float(kwargs.get("sequencer_match_thresh", 0.06))
        self.sequencer_gain = float(kwargs.get("sequencer_gain", 0.11))
        self.sequencer_sigma = float(kwargs.get("sequencer_sigma", 1.0))
        self.sequencer_input_gain = float(kwargs.get("sequencer_input_gain", 1.0))
        self._seq = None; self._seq_score = None; self._seq_K = None; self._seq_drives = None; self._seq_dirty = True
        self.enable_seq_vocab_shrink = bool(kwargs.get("enable_seq_vocab_shrink", True))
        self._seq_mapA = None; self._seq_mapX = None; self._seq_cuevocab_sig = None; self._seq_cleanup_conns_cache = None
        self.local_reciprocal_unbind = bool(kwargs.get("local_reciprocal_unbind", True))
        self.encoding_gain_fn = kwargs.get("encoding_gain_fn", None)
        self.enable_spiking_cleanup = bool(kwargs.get("enable_spiking_cleanup", True))
        self.enable_multiframe = bool(kwargs.get("enable_multiframe", False)); self._frame_parser = None
        self.enable_batched = bool(kwargs.get("enable_batched", True))
        self.enable_rf_cudagraph = bool(kwargs.get("enable_rf_cudagraph", False))   # numpy path => no megakernel
        self.enable_csr_cache = bool(kwargs.get("enable_csr_cache", True))
        self._csr_cache = {}; self._store_csr = None; self._store_dirty = True
        self.confidence_gate = float(kwargs.get("confidence_gate", 0.0))
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=period, grounded_codes=grounded_codes,
                                     local_reciprocal_unbind=self.local_reciprocal_unbind)
        self.words = list(self.comp.words); self.V = len(self.words)
        self.R = 40; self.P = 6 + 3 * self.R; self.k_max = int(k_max)
        self.pol_words = list(self.comp.pol_words); self.NP = len(self.pol_words)
        self.enable_attributed = bool(kwargs.get("enable_attributed", False))
        self.bind_roles = (["agent", "action", "patient", "attribute", "polarity"] if self.enable_attributed
                           else ["agent", "action", "patient", "polarity"])
        self.n_roles = len(self.bind_roles)
        self.main_roles = [r for r in self.bind_roles if r != "polarity"]; self.n_main = len(self.main_roles)
        # the standalone (pre-offset) layout:
        self.store_base = self.P + (2 * self.n_roles + 1) * D
        self.block = 1 + D
        self.q_base = self.store_base + self.k_max * self.block
        self.c_base = self.q_base + self.n_roles * D
        self.cb = self.n_main * self.V + self.NP
        self.bat_q_base = self.c_base + self.cb
        self.bat_c_base = self.bat_q_base + self.k_max * self.n_roles * D
        layout_span = self.bat_c_base + self.k_max * self.cb     # == standalone n_total = the slice span on the merged bridge

        # --- THE REBASE: shift every base by rf_base; n_total becomes the merged bridge N; rf_mask = the slice. ---
        self._rf_base = int(rf_base)
        N = int(merged_bridge.core_config.num_neurons)
        if self._rf_base + layout_span > N:
            raise ValueError(f"co-resident OneBrainComposer needs {layout_span} rf neurons at base {self._rf_base} "
                             f"but the merged bridge has only {N} (raise the rf region size)")
        self.P += self._rf_base
        self.store_base += self._rf_base
        self.q_base += self._rf_base
        self.c_base += self._rf_base
        self.bat_q_base += self._rf_base
        self.bat_c_base += self._rf_base
        self.n_total = N                                         # array-sizing is full merged-bridge N
        self.b = merged_bridge
        # the parser slice lives at [rf_base : rf_base + P_local] on the merged bridge (same relative wiring).
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=self._rf_base)
        self.rf_mask = np.zeros(self.n_total, dtype=bool)
        self.rf_mask[self._rf_base:self._rf_base + layout_span] = True
        # the per-op `v/u <- 0` reset is restricted to the rf slice (so a co-resident Izhikevich/nav slice's v/u is
        # byte-untouched across a composer op) -- the masked-rf-kick co-residence guarantee, the MergedRFComposer 5b
        # precedent applied to OneBrainComposer's extra reset sites.
        self._rf_reset_mask = self.rf_mask
        self.kb = []
        self.store_conns = []
        self._word_index = {w: i for i, w in enumerate(self.words)}
        self._layout_span = int(layout_span)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The fact set + the full who/what/yes-no/abstention matrix (covers stored answers AND the no-confab moat).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _build_vocab():
    # small (V<=16) but covers agents/actions/patients + a polarity case; orthogonal-ish via the composer's own codes.
    return ["dog", "cat", "fox", "rabbit", "bird", "chase", "see", "eat", "north", "south",
            "river", "tree", "fish", "seed", "worm", "hill"]


FACTS = [
    # (agent, action, patient, polarity)
    ("dog", "chase", "cat", None),          # affirmative (default)
    ("cat", "see", "bird", None),
    ("fox", "eat", "rabbit", "NEGATE"),     # a negated fact -> ask_yes_no = "no"
    ("rabbit", "eat", "seed", None),
    ("bird", "see", "worm", None),
]


def _store_all(comp):
    for (a, x, p, pol) in FACTS:
        comp.store(a, x, p, polarity=pol)


def _run_matrix(comp):
    """Return an ordered list of (label, answer) over the full matrix: who/what/yes-no for stored facts + abstentions
    for unstored cues (the no-confab moat). Answers are strings / None / 'unknown' -- compared verbatim."""
    out = []
    # query_patient (what does AGENT ACTION?) for every stored fact + two unstored (agent, action) cues (moat -> None)
    for (a, x, p, pol) in FACTS:
        out.append((f"patient[{a},{x}]", comp.query_patient(a, x)))
    out.append(("patient[dog,see]", comp.query_patient("dog", "see")))        # (agent stored, action mismatch) -> None
    out.append(("patient[hill,chase]", comp.query_patient("hill", "chase")))  # absent agent -> None
    # query_agent (who ACTION PATIENT?) for stored + an unstored (action, patient) cue (moat -> None)
    for (a, x, p, pol) in FACTS:
        out.append((f"agent[{x},{p}]", comp.query_agent(x, p)))
    out.append(("agent[chase,bird]", comp.query_agent("chase", "bird")))      # unstored combo -> None
    out.append(("agent[fly,fish]", comp.query_agent("fly", "fish")))          # absent action word -> None
    # ask_yes_no for stored (yes/no) + unstored (unknown)
    for (a, x, p, pol) in FACTS:
        out.append((f"yesno[{a},{x},{p}]", comp.ask_yes_no(a, x, p)))
    out.append(("yesno[dog,chase,bird]", comp.ask_yes_no("dog", "chase", "bird")))   # wrong patient -> unknown
    out.append(("yesno[hill,eat,seed]", comp.ask_yes_no("hill", "eat", "seed")))     # absent agent -> unknown
    # render_fact (describe AGENT) for stored + an unknown subject (moat -> None)
    for (a, x, p, pol) in FACTS:
        out.append((f"render[{a}]", comp.render_fact(a)))
    out.append(("render[hill]", comp.render_fact("hill")))                    # unknown subject -> None
    # query_chain (multi-hop): rabbit -eat-> seed (1-hop); a 2-hop with a dead end -> None
    out.append(("chain[rabbit;eat]", comp.query_chain("rabbit", ["eat"])))
    out.append(("chain[dog;chase,see]", comp.query_chain("dog", ["chase", "see"])))   # cat -see-> bird => 'bird'
    out.append(("chain[dog;chase,eat]", comp.query_chain("dog", ["chase", "eat"])))   # cat -eat-> ? dead end -> None
    return out


def _compare(standalone_rows, coresident_rows):
    """Compare two answer matrices verbatim. Returns (n_total, n_match, n_abstain_total, n_abstain_match, mismatches)."""
    assert [r[0] for r in standalone_rows] == [r[0] for r in coresident_rows], "matrix labels diverged"
    n = len(standalone_rows); n_match = 0; mism = []
    n_abstain_total = 0; n_abstain_match = 0
    for (lbl, sa), (_, co) in zip(standalone_rows, coresident_rows):
        sa_abstain = (sa is None) or (sa == "unknown")
        if sa_abstain:
            n_abstain_total += 1
        if sa == co:
            n_match += 1
            if sa_abstain:
                n_abstain_match += 1
        else:
            mism.append({"label": lbl, "standalone": sa, "coresident": co})
    return n, n_match, n_abstain_total, n_abstain_match, mism


def _seed_izh_stub(bridge, n):
    """Put non-trivial Izhikevich state on the first `n` (nav-stub) neurons; return a (v,u) snapshot for the isolation check."""
    xp, _ = get_backend()
    v = bridge.cp_membrane_potential_v; u = bridge.cp_recovery_variable_u
    v[:n] = xp.asarray(np.linspace(-65.0, -50.0, n), dtype=v.dtype)
    u[:n] = xp.asarray(np.linspace(-13.0, -8.0, n), dtype=u.dtype)
    return np.asarray(v[:n]).copy(), np.asarray(u[:n]).copy()


def run_probe(seed=42, D=48, nav_stub=37, persistent_loop=True, enable_spiking_cleanup=False, atol=1e-9):
    """Build a standalone + a co-resident OneBrainComposer (same seed/D/vocab/flags), store the facts, run the full
    matrix on each, compare verbatim. Also assert the nav-stub Izhikevich slice is byte-identical across a composer op."""
    vocab = _build_vocab()
    common = dict(seed=seed, D=D, vocab=vocab, period=200, persistent_loop=persistent_loop,
                  enable_spiking_cleanup=enable_spiking_cleanup, integrated_loop=False, enable_batched=True,
                  enable_rf_cudagraph=False)

    # standalone (the oracle) -- its own private bridge at [0:n_total].
    standalone = OneBrainComposer(**common)

    # co-resident -- a merged-stub bridge with a nav stub forcing rf_base = nav_stub (>0).
    layout_span = int(standalone.n_total)            # the slice span the composer needs on the merged bridge
    merged = build_merged_stub_bridge(nav_stub + layout_span, seed=seed)
    coresident = CoResidentOneBrainComposer(merged, rf_base=nav_stub, **common)
    # The isolation control (SPEC: "byte-isolated ACROSS AN OP") is seeded AFTER construction -- the parser's
    # Hebbian training runs full Izhikevich steps that legitimately evolve the WHOLE bridge (incl. the nav stub),
    # so the isolation snapshot is taken POST-construction (just before the measured op), and the assertion is that a
    # single RF composer OP does not perturb the nav slice (the masked-rf-kick guarantee), not that construction never
    # touches it (the parser training, a full Izhikevich step, legitimately evolves the whole bridge).

    # sanity: the rebasing landed inside the slice + the parser slice is at the offset.
    assert coresident._rf_base == nav_stub
    assert coresident.store_base == standalone.store_base + nav_stub
    assert coresident.bat_c_base == standalone.bat_c_base + nav_stub
    assert coresident.n_total == int(merged.core_config.num_neurons)
    assert coresident.parser.index_offset == nav_stub

    _store_all(standalone)
    _store_all(coresident)

    sa_rows = _run_matrix(standalone)
    co_rows = _run_matrix(coresident)
    n, n_match, n_abst_tot, n_abst_match, mism = _compare(sa_rows, co_rows)

    # numeric byte-identity at the membrane level: compare the raw batched cleanup membrane read-out for a known query
    # (one full reconstruct->unbind->cleanup pass), not just the decoded word -- a tighter atol-1e-9 assertion.
    sa_mem = _raw_cleanup_membrane(standalone)
    co_mem = _raw_cleanup_membrane(coresident)
    max_abs_dmem = float(np.max(np.abs(sa_mem - co_mem))) if sa_mem.shape == co_mem.shape else float("inf")

    # SLICE ISOLATION (SPEC anti-cheat 3): a single composer RF op on the co-resident bridge must not touch the
    # nav-stub v/u. Seed the nav stub HERE (immediately before the op, after all prior construction/matrix work) so
    # the assertion measures exactly the masked-rf-kick guarantee across ONE op (the MergedRFComposer 5b precedent).
    v0, u0 = _seed_izh_stub(merged, nav_stub)
    coresident.query_patient("dog", "chase")          # a full RF op (kick + resonate + unbind + cleanup)
    v_after = np.asarray(merged.cp_membrane_potential_v[:nav_stub])
    u_after = np.asarray(merged.cp_recovery_variable_u[:nav_stub])
    nav_v_identical = bool(np.array_equal(v_after, v0))
    nav_u_identical = bool(np.array_equal(u_after, u0))

    return dict(
        n_matrix=n, n_match=n_match, n_mismatch=len(mism),
        n_abstain_total=n_abst_tot, n_abstain_match=n_abst_match,
        max_abs_membrane_delta=max_abs_dmem, atol=atol,
        byte_identical=(len(mism) == 0 and max_abs_dmem <= atol),
        moat_preserved=(n_abst_match == n_abst_tot and n_abst_tot > 0),
        nav_slice_byte_identical=(nav_v_identical and nav_u_identical),
        mismatches=mism[:20],
        sample_rows=[{"label": l, "answer": (a if not isinstance(a, float) else a)} for (l, a) in sa_rows],
    )


def _raw_cleanup_membrane(comp):
    """The raw batched cleanup membrane read-out over ALL stored blocks (one reconstruct->unbind->cleanup pass) --
    the tightest numeric byte-identity surface (the values _decode_batched_mem then argmaxes). Read off the SLICE so
    the standalone (base 0) + co-resident (offset) return the SAME-length vector aligned to the layout."""
    # Run the batched read (populates the cleanup membrane), then slice out the batched cleanup region for n facts.
    comp._read_all_blocks()
    mem = np.asarray(comp.b.cp_membrane_potential_v).astype(float)
    n = len(comp.kb)
    lo = comp.bat_c_base
    hi = comp.bat_c_base + n * comp.cb
    return mem[lo:hi].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--nav-stub", type=int, default=37)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--out", default="research/findings/raw/_consolidation_probe1_byteident.json")
    args = ap.parse_args()

    results = {}
    # two configs: the numpy-CPU/oracle cleanup (host argmax) AND the spiking-cleanup path, both persistent_loop=True.
    for cfg_name, spiking in [("oracle_cleanup", False), ("spiking_cleanup", True)]:
        r = run_probe(seed=args.seed, D=args.D, nav_stub=args.nav_stub, persistent_loop=True,
                      enable_spiking_cleanup=spiking, atol=args.atol)
        results[cfg_name] = r
        print(f"[{cfg_name}] persistent_loop=True spiking_cleanup={spiking}: "
              f"matrix {r['n_match']}/{r['n_matrix']} match, abstain {r['n_abstain_match']}/{r['n_abstain_total']}, "
              f"max|dmem|={r['max_abs_membrane_delta']:.2e}, byte_identical={r['byte_identical']}, "
              f"moat_preserved={r['moat_preserved']}, nav_slice_byte_identical={r['nav_slice_byte_identical']}")
        if r["mismatches"]:
            print(f"  MISMATCHES: {r['mismatches']}")

    overall_go = all(r["byte_identical"] and r["moat_preserved"] and r["nav_slice_byte_identical"]
                     for r in results.values())
    out = {
        "probe": "consolidation_probe1_byteident",
        "what": "co-resident OneBrainComposer on an OFFSET rf slice == standalone OneBrainComposer (atol 1e-9), "
                "the full who/what/yes-no matrix + every is-None/unknown abstention, persistent_loop=True",
        "scoping": "research/findings/raw/_consolidation_onebrain_limbic_scoping.md (Probe 1)",
        "backend": get_backend()[1],
        "seed": args.seed, "D": args.D, "nav_stub_offset": args.nav_stub, "vocab_size": len(_build_vocab()),
        "n_facts": len(FACTS), "atol": args.atol,
        "configs": results,
        "GO": bool(overall_go),
        "verdict": ("GO -- co-resident OneBrainComposer is byte-identical (atol 1e-9) to standalone on the full matrix; "
                    "moat preserved; nav slice byte-isolated; the index rebasing is correct"
                    if overall_go else
                    "NO-GO -- see per-config mismatches (which index / which op the rebasing breaks)"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGO={overall_go}  ->  wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
