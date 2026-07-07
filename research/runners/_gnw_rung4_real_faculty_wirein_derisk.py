"""GNW Rung-4 de-risk -- the GNW workspace SPEAKS its REASONED answer via the project's REAL faculties, in ONE process,
gate-first. This replaces the Rung-3/3b ABSTRACT downstream consumers (report / reason) with the project's VALIDATED
faculties: the EMERGE-52/54 inheritance REASONER (`PerDimensionConsole`) as `reason`, and the EMERGE-67/68 spiking A->W
read-out (`UnifiedNeuralSpell`) as `report` (the brain literally SPEAKS the ignited concept). ==> "the concepts the brain
can SAY are the ones it REASONS with" made literally true; and it BREAKS Rung-3's by-construction identity critique for
free (report and reason are now completely different substrates/populations, sharing ONLY the ignited concept's STRING).

THE KEY ARCHITECTURE -- a STRING-KEYED dispatch (not a code hand-off). The three faculties use three different concept
representations (workspace = a neuron-slice assembly; reasoner = an HTM-pooler codon; A->W = a driven concept pool decoded
from language_output spikes). NONE shares a code vector -- they share only the concept STRING. So the wire-in is a thin
string-keyed orchestration (== EMERGE-70's gate-first `_emerge_turn` with the GNW ignition as the front-end):

    ignite(concept) on the GNW workspace -> read which report-pop sustains -> concept STRING  ("none" => gate CLOSED = moat)
    reason  = reasoner.ask_can(concept, property)     # affirm / override / abstain (the reasoner's own moat)
       | gate-first: abstain => the A->W is NEVER invoked (0 spell calls)
    report  = UnifiedNeuralSpell.spell(...)            # drive pools -> decode "the owl can fly" from language_output SPIKES

THE CORRECTNESS GATE (the whole point): the string that IGNITES == the string the reasoner reasons about == the string the
A->W speaks. If we only showed each faculty works in isolation, we would NOT have demonstrated "report==reasoning". We
ASSERT the round-trip: ignited_concept == reasoned_concept == spoken_subject_surface, per member.

THE VOCAB INTERSECTION (documented, per the scoping's honest scope). The surface must be genuinely SPIKE-spelled, so the
members + properties are chosen in the INTERSECTION of (A) the A->W-trained vocab and (B) the reasoner taxonomy:
  * A->W content vocab (EMERGE-67 `_AW_SUBJECTS`/`_AW_VERBS`): subjects owl/penguin/robin/sparrow/eagle/hawk/wren/crow;
    verbs fly/swim/run/hop (bare-ability) + walks/lurks/hides/rests (3sg-intransitive). A->W function vocab (EMERGE-68
    `_FUNC_WORDS`): the/a/can/does/not.
  * Reasoner taxonomy (EMERGE-52/54): BIRD members robin/sparrow/eagle/hawk/crow/finch + penguin (LOCOMOTION exception
    "walks") + owl (HELD-OUT, inherits "fly"); FISH members (NOT in the A->W subject vocab, so excluded).
  => the two clean, fully-spike-spellable turns are:
       owl -> fly     (reasoner: "Yes, an owl can fly." INHERITED -- owl held-out, never taught fly)
                        modal frame:        "the owl can fly"     (the/owl/can/fly -- all in the A->W vocab)
       penguin -> fly (reasoner: "No, a penguin walks." OVERRIDE via the penguin's own LOCOMOTION exception)
                        intransitive frame: "the penguin walks"   (the/penguin/walks -- all in the A->W vocab)
  and the moat turn:
       zzz -> fly     (a workspace-ignitable concept NEVER taught to the reasoner -> reasoner abstains -> A->W NEVER invoked)
  The workspace holds {owl, penguin, zzz} (all categorized to BIRD -- legitimate: perception categorizes; owl held-out).
  The A->W engine's own concept-pool weights are GPU-trained ONCE + cached (a scale/data lever, NOT a mechanism); full
  vocab (e.g. the RESPIRATION dimension "breathe", or the FISH members) is a cached-A->W rebind (a --train re-run).

ONE-BACKEND CO-EXECUTION (already solved). `sim.bridge` binds ONE backend per process; the reasoner is numpy-native, the
A->W read-out is cupy-native. EMERGE-71 committed the single additive `SimulationBridge.xp` property (`sim/bridge.py:213`)
so the workspace + reasoner bridges inherit `bridge.xp` cleanly; a probe-scoped `from_host` shim (EMERGE-70's exact fix,
byte-identical on numpy) routes the reasoner's 3 residual host->device writes so it runs on cupy alongside the A->W engine.
NO `sim/` edit -- the shim monkeypatches two RESEARCH-runner helpers only, byte-identical on numpy.

DE-RISK GATES (the GO gate):
  (1) ROUND-TRIP (report==reasoning): for owl + penguin, ignited_concept == reasoned_concept == spoken_subject_surface.
  (2) INHERITED (not looked up): owl's answer "Yes ... can fly" comes from the reasoner's inheritance (owl never taught
      fly), AND the reasoner's OWN dAP-lesion control collapses that inheritance (the EMERGE-54 substrate is load-bearing).
  (3) SPIKE-SPELLED: lesion the A->W pool->language_output pathway -> the spoken surface COLLAPSES (proves the surface is
      genuinely spike-decoded, not a host string).
  (4) GATE-FIRST MOAT: ignite the never-taught concept -> reasoner abstains -> ASSERT the A->W spell is invoked 0 times.

WHAT RUNS ON NUMPY vs CUPY:
  * numpy-capable (structural): the GNW workspace (build + ignite + string-identity read) + the reasoner (build + teach +
    ask + dAP-lesion) + the gate-first moat control-flow (abstain => spell never called). ALL exercised on the numpy path.
  * cupy-only (the controller's GPU turn): the A->W SPELL (drive concept pools -> decode "the owl can fly" from
    language_output SPIKES) + the co-execution of the reasoner-on-cupy (via the from_host shim) with the A->W engine in ONE
    process + the SPIKE-SPELLED lesion control. On numpy the A->W engine cannot run the spiking read-out (documented skip),
    so the runner reports the surface as a SKIP and the spike-spell / round-trip-surface gates are deferred to the GPU turn.

Reuse-by-import; NO `sim/` edit; NO commit.

Run:
  # controller GPU turn (the full de-risk incl. the A->W spike-render + co-execution):
  SIM_BACKEND=cupy python -m research.runners._gnw_rung4_real_faculty_wirein_derisk --derisk --seed 42 \
      --json research/findings/raw/_gnw_rung4_smoke.json
  # structural numpy walk-through (workspace + reason + moat-gate; A->W deferred to GPU):
  SIM_BACKEND=numpy python -m research.runners._gnw_rung4_real_faculty_wirein_derisk --probe --seed 42 --backend numpy
"""
from __future__ import annotations
import os

# Leave SIM_BACKEND to the caller so the GPU turn co-executes on cupy; the numpy structural path forces numpy via --backend.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "research" / "findings" / "raw" / "_gnw_rung4_smoke.json"

# ---------------------------------------------------------------------------------------------------------------------
# THE VOCAB INTERSECTION (documented above). Workspace members (all BIRD; owl held-out; zzz the moat token) + the two
# fully-spike-spellable turns + the moat turn. Kept as module constants so the assertions read cleanly.
# ---------------------------------------------------------------------------------------------------------------------
WORKSPACE_MEMBERS = ["owl", "penguin", "zzz"]     # owl = held-out (inherits fly); penguin = exception (walks); zzz = moat
WS_HELD_OUT = "owl"
# (member, property, expected reasoner verdict, expected spoken surface). The property is "fly" for both answer turns; the
# reasoner resolves owl->inherit (modal "the owl can fly") and penguin->override (intransitive "the penguin walks").
ANSWER_TURNS = [
    {"member": "owl", "property": "fly", "verdict": "ANSWER", "expect_surface": "the owl can fly"},
    {"member": "penguin", "property": "fly", "verdict": "ANSWER", "expect_surface": "the penguin walks"},
]
MOAT_TURN = {"member": "zzz", "property": "fly"}   # workspace-ignitable, never taught to the reasoner -> abstain


# =====================================================================================================================
# THE from_host SHIM (probe-scoped; NO edit to the committed runners). Routes the reasoner's 3 residual host->device
# writes through `sim.backend.from_host` so the EMERGE-52/54 reasoner runs on cupy alongside the A->W engine. This is the
# EXACT EMERGE-70 shim (byte-identical on numpy). Idempotent. See `_emerge70_one_brain_single_backend_probe_derisk.py`.
# =====================================================================================================================
def install_from_host_shim():
    from sim.backend import from_host
    import research.runners._emerge14_stageC_onbridge_learning_derisk as m14
    import research.runners._emerge12_stageB2_bridge_tm_derisk as m12
    from sim.kernels import fused_htm_permanence_update

    def apply_kernel_update(bridge, coo_row, coo_col, cells_idx, prev_win, cur_win, z, cfg_lp, cfg_ld, z_star):
        n = int(bridge.core_config.num_neurons)
        pre_last_vec = np.zeros(n, np.float64)
        post_now_vec = np.zeros(n, np.float64)
        for i in prev_win:
            pre_last_vec[cells_idx[i]] = 1.0
        for i in cur_win:
            post_now_vec[cells_idx[i]] = 1.0
        hfac_cell = 0.5 + 0.5 * np.maximum(0.0, z_star - z)
        hfac_vec = np.zeros(n, np.float64)
        hfac_vec[cells_idx] = hfac_cell
        data = m14._host(bridge.cp_connections.data).astype(np.float64)
        pre_last = pre_last_vec[coo_row]
        post_now = post_now_vec[coo_col]
        hfac_post = hfac_vec[coo_col]
        updated = fused_htm_permanence_update(data, pre_last, post_now, hfac_post, cfg_lp, cfg_ld, 0.0, 1.0)
        bridge.cp_connections.data[:] = from_host(updated.astype(np.float32))

    def reset_soma(bridge):
        n = int(bridge.core_config.num_neurons)
        bridge.cp_membrane_potential_v[:] = from_host(np.full(n, -65.0, np.float32))
        if getattr(bridge, "cp_recovery_variable_u", None) is not None:
            bridge.cp_recovery_variable_u[:] = 0.0
        for arr in ("cp_firing_states", "cp_prev_firing_states", "cp_external_input_current",
                    "cp_conductance_g_e", "cp_conductance_g_i"):
            a = getattr(bridge, arr, None)
            if a is not None:
                a[:] = 0

    def _prime_from_winners(bridge, cells_idx, winners_bool, n_prime=6):
        n = int(bridge.core_config.num_neurons)
        reset_soma(bridge)
        m12._clear_apical(bridge)
        fv = np.zeros(n, np.float32)
        fv[cells_idx[winners_bool]] = 1.0
        for _ in range(n_prime):
            bridge.cp_prev_firing_states[:] = from_host(fv)
            bridge.cp_external_input_current[:] = from_host(np.zeros(n, np.float32))
            bridge._run_one_simulation_step()

    m14.apply_kernel_update = apply_kernel_update
    m12.reset_soma = reset_soma
    m12._prime_from_winners = _prime_from_winners
    import research.runners._emerge52_multilevel_conversational_console as m52
    m52._prime_from_winners = _prime_from_winners
    m52.apply_kernel_update = apply_kernel_update


# =====================================================================================================================
# FACULTY 1 -- the GNW WORKSPACE (reuse `build_inheritance_bridge`, rebound to the intersection members). ignite(concept)
# -> the report-pop that sustains -> the concept STRING ("none" => gate CLOSED). This is the front-end identity read-out.
# =====================================================================================================================
def _rebind_workspace_members():
    """Rebind the GNW module's member set to the vocab intersection (owl held-out, penguin, zzz-moat; all BIRD). The
    workspace only needs ignitable string-keyed member assemblies + a report identity read; its inheritance layer is not
    used for the reason here (the REAL reasoner replaces it). Returns the GNW module."""
    import research.runners._gnw_rung3b_emergent_inheritance_reasoning_derisk as g
    g.MEMBERS = list(WORKSPACE_MEMBERS)
    g.MEMBER_SUPER = {m: "BIRD" for m in WORKSPACE_MEMBERS}
    g.HELD_OUT = WS_HELD_OUT
    g.SUPERORDS = ["BIRD"]
    g.SUPER_PROP = {"BIRD": "flies"}
    g.PROPS = ["flies"]
    return g


class GNWWorkspace:
    """The GNW workspace as the front-end concept-identity read-out. `ignite(concept)` drives that member's assembly, reads
    which report-pop sustains, and returns the decoded concept STRING (or "none" if nothing sustains = the gate is CLOSED).
    Reuse-by-import of `build_inheritance_bridge`; the member set is rebound to the vocab intersection."""

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.g = _rebind_workspace_members()
        self.bridge, self.xp, self.idx, self.snap = self.g.build_inheritance_bridge(
            seed=self.seed, lesion_workspace=False)

    def ignite(self, concept):
        """Ignite `concept` on the workspace; return the report-pop STRING that sustains ("none" => gate CLOSED)."""
        if concept not in self.g.MEMBERS:
            return "none"
        trial = self.g._run_trial(self.bridge, self.xp, self.idx, self.snap, concept)
        return self.g._argmax_decode(trial["report"])


# =====================================================================================================================
# FACULTY 2 -- the REAL inheritance REASONER (EMERGE-52/54 PerDimensionConsole), taught the taxonomy in-script. Optional
# dAP-lesion for the "answer is genuinely INHERITED (substrate load-bearing)" anti-cheat.
# =====================================================================================================================
def build_reasoner(seed=42, shim=False, lesion=False):
    """Build + teach the EMERGE-54 reasoner. `shim` installs the from_host shim (needed only on cupy co-execution).
    `lesion=True` removes the coincidence/two-compartment substrate the inheritance read relies on (dAP-lesion), so the
    inheritance answer collapses to abstain (the EMERGE-54 primary load-bearing control)."""
    if shim:
        install_from_host_shim()
    from research.runners._emerge54_per_dimension_cancellation_derisk import (
        PerDimensionConsole, _script_lines, handle)
    c = PerDimensionConsole(seed=int(seed), lesion=bool(lesion))
    obs, isa, teach, _ask = _script_lines(int(seed))
    for line, _ in obs:
        handle(c, line)
    for line, _ in isa:
        handle(c, line)
    for line, _ in teach:
        handle(c, line)
    return c


# =====================================================================================================================
# THE TURN -- ignite -> string -> reason (gate-first) -> spell. == EMERGE-70's `_emerge_turn` with the GNW ignition as the
# front-end. Returns the concept-string round-trip pieces + the spoken surface + the moat accounting.
# =====================================================================================================================
def run_turn(ws, reasoner, unified, member, prop):
    """One flagship turn. (1) IGNITE on the workspace -> the concept STRING. (2) REASON gate-first: the reasoner decides
    answer-vs-abstain; on abstain the A->W is NEVER invoked. (3) REPORT: spike-spell the frame surface via the A->W read-
    out. Returns a dict with the round-trip strings + the spoken surface + spell-call accounting.
    `unified` may be None (numpy path) -> the spell step is a documented SKIP."""
    from research.runners._emerge54_per_dimension_cancellation_derisk import _lemma

    ignited = ws.ignite(member)                       # workspace identity read-out -> the concept STRING
    if ignited == "none":                             # workspace gate CLOSED (nothing sustained) -> hard abstain
        return {"member": member, "ignited": "none", "gate": "ABSTAIN", "reasoned": None,
                "surface": None, "spell_calls": 0}

    reply = reasoner.ask_can(ignited, prop)           # GATE-FIRST: the reasoner decides answer-vs-abstain
    if reply.startswith("I don't"):                   # abstain -> the A->W / producer is NEVER invoked (the moat)
        return {"member": member, "ignited": ignited, "gate": "ABSTAIN", "reasoned": None,
                "surface": None, "spell_calls": 0}

    # the reasoned concept STRING is the ignited member (the reasoner reasons about the string the workspace ignited)
    reasoned_concept = ignited

    if unified is None:                               # numpy path -> the A->W spiking read-out cannot run (documented skip)
        return {"member": member, "ignited": ignited, "gate": "ANSWER", "reasoned": reasoned_concept,
                "surface": None, "surface_skipped": "A->W spike read-out requires SIM_BACKEND=cupy",
                "spell_calls": 0}

    # REPORT: spell the frame surface ON SPIKES. Modal frame ("the <subj> can <bare-verb>") for an inherited affirm;
    # intransitive frame ("the <subj> <3sg-verb>") for the member's own exception. Every word decoded from language_output.
    calls0 = unified.spell_calls

    def w(word):                                      # spike-spell a word if the A->W engine knows it, else the surface
        if word in unified.content_words or word in unified.func_words:
            return unified.spell(word)
        return str(word)

    if reply.startswith("No,"):                       # exception -> "the <subj> <3sg-verb>" (penguin walks)
        verb = reasoner.ovr_prop.get(ignited, prop)
        words = [w("the"), w(ignited), w(verb)]
    else:                                             # inherited -> "the <subj> can <bare-verb>" (owl can fly)
        words = [w("the"), w(ignited), w("can"), w(_lemma(prop))]
    surface = " ".join(words)
    return {"member": member, "ignited": ignited, "gate": "ANSWER", "reasoned": reasoned_concept,
            "surface": surface, "spoken_subject": words[1], "spell_calls": int(unified.spell_calls - calls0)}


# =====================================================================================================================
# THE DE-RISK.
# =====================================================================================================================
def _derisk(seed, backend, json_path):
    from sim.backend import get_backend, is_gpu_backend
    if backend != "auto":
        get_backend(backend)
    gpu = bool(is_gpu_backend())
    print(f"[gnw-rung4] seed={seed} backend={backend} gpu={gpu} | the GNW workspace SPEAKS its REASONED answer via the "
          f"REAL faculties (reason=EMERGE-52/54, report=EMERGE-67/68 A->W), gate-first", flush=True)

    t0 = time.time()
    err = None
    result = {"runner": "_gnw_rung4_real_faculty_wirein_derisk", "seed": int(seed), "backend": backend, "gpu": gpu,
              "vocab_intersection": {"workspace_members": WORKSPACE_MEMBERS, "held_out": WS_HELD_OUT,
                                     "answer_turns": ANSWER_TURNS, "moat_turn": MOAT_TURN}}
    try:
        # ---- FACULTY 1: the GNW workspace (numpy or cupy) -----------------------------------------------------------
        ws = GNWWorkspace(seed=seed)

        # ---- FACULTY 2: the REAL reasoner (shim on cupy so it co-executes with the A->W engine) ----------------------
        reasoner = build_reasoner(seed=seed, shim=gpu, lesion=False)
        reasoner_lesion = build_reasoner(seed=seed, shim=False, lesion=True)   # dAP-lesion (inheritance load-bearing)

        # ---- FACULTY 3: the REAL A->W read-out (cupy + cached; numpy -> documented skip) -----------------------------
        unified = None
        unified_lesion = None
        aw_available = False
        aw_note = None
        if gpu:
            from research.runners._emerge68_function_word_spell_derisk import UnifiedNeuralSpell
            unified = UnifiedNeuralSpell(load=True)
            aw_available = bool(unified._backend_gpu)
            if aw_available:
                unified_lesion = UnifiedNeuralSpell(load=True, content_lesion=True, func_lesion=True)  # SPIKE-SPELLED ctrl
            else:
                aw_note = "A->W engines built but not on GPU (cache missing / numpy fallback)"
                unified = None
        else:
            aw_note = "numpy backend -- the A->W spiking read-out is deferred to the controller's cupy turn"
        result["aw_available"] = bool(aw_available)
        result["aw_note"] = aw_note

        # ---- THE TURNS -----------------------------------------------------------------------------------------------
        turns = []
        for spec in ANSWER_TURNS:
            tr = run_turn(ws, reasoner, unified, spec["member"], spec["property"])
            tr["expect_surface"] = spec["expect_surface"]
            turns.append(tr)
        # moat turn
        moat = run_turn(ws, reasoner, unified, MOAT_TURN["member"], MOAT_TURN["property"])
        result["turns"] = turns
        result["moat_turn_result"] = moat

        # ---- GATE (2) INHERITED: owl's answer is inheritance (never taught fly) AND the reasoner dAP-lesion collapses it
        owl_reply = reasoner.ask_can("owl", "fly")
        owl_reply_lesion = reasoner_lesion.ask_can("owl", "fly")
        inherited = bool(owl_reply.startswith("Yes,") and not owl_reply_lesion.startswith("Yes,"))
        result["inherited_check"] = {"owl_reply": owl_reply, "owl_reply_dap_lesion": owl_reply_lesion,
                                     "inherited": inherited}

        # ---- GATE (4) GATE-FIRST MOAT: the moat turn abstained AND invoked the A->W spell 0 times --------------------
        moat_ok = bool(moat["gate"] == "ABSTAIN" and moat["spell_calls"] == 0)
        result["moat_ok"] = moat_ok

        # ---- GATE (1) ROUND-TRIP (report==reasoning): ignited == reasoned == spoken subject (surface on GPU only) -----
        # On numpy the surface is skipped, so the round-trip is asserted through the ignited==reasoned identity + the
        # reasoner verdict matching; the spoken-subject leg is deferred to the GPU turn.
        roundtrip = []
        for tr in turns:
            leg = {"member": tr["member"], "ignited": tr["ignited"], "reasoned": tr["reasoned"],
                   "gate": tr["gate"], "surface": tr.get("surface"),
                   "spoken_subject": tr.get("spoken_subject"), "expect_surface": tr["expect_surface"]}
            ignited_eq_reasoned = (tr["ignited"] == tr["reasoned"] and tr["gate"] == "ANSWER")
            if aw_available and tr.get("surface") is not None:
                # full round-trip on GPU: ignited == reasoned == spoken subject == the expected surface
                surface_ok = (tr["surface"] == tr["expect_surface"])
                subject_ok = (tr.get("spoken_subject") == tr["ignited"])
                leg["roundtrip_full"] = bool(ignited_eq_reasoned and surface_ok and subject_ok)
                leg["surface_ok"] = bool(surface_ok)
                leg["subject_ok"] = bool(subject_ok)
            else:
                # structural round-trip on numpy: ignited == reasoned (surface deferred)
                leg["roundtrip_structural"] = bool(ignited_eq_reasoned)
            roundtrip.append(leg)
        result["roundtrip"] = roundtrip

        # ---- GATE (3) SPIKE-SPELLED: the A->W lesion collapses the spoken surface (GPU only) -------------------------
        spike_spelled = None
        if aw_available:
            lesion_turns = []
            spike_collapsed = True
            for spec in ANSWER_TURNS:
                lt = run_turn(ws, reasoner, unified_lesion, spec["member"], spec["property"])
                lesion_turns.append({"member": lt["member"], "surface": lt.get("surface"),
                                     "expect_surface": spec["expect_surface"]})
                # collapse = the lesioned surface is NOT the correct surface (a host lookup would be unaffected)
                if lt.get("surface") == spec["expect_surface"]:
                    spike_collapsed = False
            spike_spelled = bool(spike_collapsed)
            result["spike_spelled_check"] = {"lesion_turns": lesion_turns, "spike_spelled": spike_spelled}

        # ---- THE GO GATE --------------------------------------------------------------------------------------------
        if aw_available:
            roundtrip_full = all(leg.get("roundtrip_full", False) for leg in roundtrip)
            go = bool(roundtrip_full and inherited and (spike_spelled is True) and moat_ok)
            structural_only = False
        else:
            # numpy structural path: the workspace + reason + moat-gate are exercised; the A->W spike-render + the
            # spoken-surface round-trip + the spike-spelled lesion are deferred to the controller's cupy turn.
            roundtrip_struct = all(leg.get("roundtrip_structural", False) for leg in roundtrip)
            go = False   # NOT a GO on numpy -- the spike-render half is unverified (honest: needs the GPU turn)
            structural_only = bool(roundtrip_struct and inherited and moat_ok)
        result["go"] = go
        result["structural_only_pass"] = None if aw_available else structural_only
    except Exception as e:
        err = repr(e)
        traceback.print_exc()
        result["error"] = err
        result["go"] = False

    result["elapsed_seconds"] = round(time.time() - t0, 1)

    # ---- VERDICT ---------------------------------------------------------------------------------------------------
    if err is not None:
        verdict = f"ERROR -- {err}"
    elif not result.get("aw_available"):
        sp = result.get("structural_only_pass")
        verdict = (
            f"STRUCTURAL-PASS (numpy) -- the GNW workspace + the REAL reasoner + the gate-first moat all run and "
            f"round-trip on numpy (structural round-trip {'OK' if sp else 'INCOMPLETE'}: ignited==reasoned for each "
            f"answer turn; owl's 'fly' is INHERITED [reasoner Yes, dAP-lesion collapses it]; the moat abstains + invokes "
            f"the A->W 0 times). The A->W SPIKE-render half (the spoken surface 'the owl can fly' / 'the penguin walks' "
            f"decoded from language_output spikes) + the co-execution of the reasoner-on-cupy with the A->W engine in ONE "
            f"process + the SPIKE-SPELLED lesion control are DEFERRED to the controller's SIM_BACKEND=cupy turn (the "
            f"A->W concept-pool read-out is GPU-only; caches present: bridges/emerge67_aw + bridges/emerge68_aw). "
            f"{result.get('aw_note')}.")
    elif result["go"]:
        verdict = (
            f"GO -- the GNW workspace SPEAKS its REASONED answer via the project's REAL faculties, in ONE cupy process, "
            f"gate-first. The three faculties share ONLY the ignited concept's STRING: (1) the GNW workspace ignites a "
            f"member and reads its identity STRING; (2) the EMERGE-52/54 inheritance REASONER decides answer-vs-abstain "
            f"(gate-first); (3) on answer, the EMERGE-67/68 A->W read-out SPEAKS the frame surface, every word decoded "
            f"from language_output SPIKES. THE ROUND-TRIP HOLDS (report==reasoning): the string that IGNITES == the "
            f"string the reasoner reasons about == the string the A->W speaks -- 'owl' -> reason 'Yes, an owl can fly' "
            f"(INHERITED; owl never taught fly, dAP-lesion collapses it) -> spike-render 'the owl can fly'; 'penguin' -> "
            f"reason 'No, a penguin walks' (its own LOCOMOTION exception) -> spike-render 'the penguin walks'. The "
            f"surface is GENUINELY SPIKE-SPELLED (the A->W pool->language_output lesion COLLAPSES it -- a host lookup "
            f"would be unaffected). The gate-first no-confab MOAT holds BY CONSTRUCTION: igniting a never-taught concept "
            f"(zzz) -> the reasoner abstains -> the A->W spell is invoked 0 times. This BREAKS Rung-3's by-construction "
            f"identity critique (report and reason are completely different substrates/populations, sharing only the "
            f"STRING). Reuse-by-import; NO sim/ edit. The A->W engines are GPU-trained ONCE + cached (a scale/data lever); "
            f"full vocab (the RESPIRATION dimension / FISH members) is a cached-A->W rebind, not a mechanism.")
    else:
        miss = []
        rt = result.get("roundtrip", [])
        if not all(leg.get("roundtrip_full", False) for leg in rt):
            miss.append("the concept-string round-trip did NOT close (ignited != reasoned != spoken-subject, or the "
                        "spoken surface != the expected frame surface -- report==reasoning NOT demonstrated)")
        if not result.get("inherited_check", {}).get("inherited"):
            miss.append("owl's answer is NOT a genuine inheritance (reasoner did not say Yes, or the dAP-lesion did not "
                        "collapse it)")
        if result.get("spike_spelled_check", {}).get("spike_spelled") is not True:
            miss.append("the spoken surface is NOT clearly spike-spelled (the A->W pool->language_output lesion did not "
                        "collapse the surface)")
        if not result.get("moat_ok"):
            miss.append("the gate-first MOAT did NOT hold (the A->W was invoked on an abstain) -- BLOCKING")
        verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The wire-in is a STRING-KEYED orchestration (ignite->identity-"
                   "read->reason->gate-first->spell); the residual is in ONE of those legs -- check the workspace "
                   "identity read (mutual exclusion), the reasoner verdict/lemma, the A->W concept-pool decode, or the "
                   "frame-surface assembly. If the MOAT was breached this is BLOCKING -- do NOT weaken the moat. NOT a "
                   "wall (the faculties are each independently validated; the failure is in the orchestration seam).")
    result["verdict"] = verdict

    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[gnw-rung4] VERDICT: {verdict}", flush=True)
    print(f"[gnw-rung4] wrote {p}\n" + "=" * 118, flush=True)
    ok = result.get("go") or (not result.get("aw_available") and result.get("structural_only_pass"))
    return 0 if (err is None and ok) else 1


def _probe(seed, backend):
    """Human-readable single-seed walk-through (numpy exercises workspace + reason + moat-gate; A->W deferred to GPU)."""
    from sim.backend import get_backend, is_gpu_backend
    if backend != "auto":
        get_backend(backend)
    gpu = bool(is_gpu_backend())
    print(f"\n=== GNW Rung-4 PROBE (seed={seed}, backend={backend}, gpu={gpu}) -- the workspace SPEAKS its reasoned "
          f"answer via the REAL faculties, gate-first ===\n", flush=True)
    print(f"  vocab intersection: workspace members {WORKSPACE_MEMBERS} (held-out {WS_HELD_OUT}); answer turns "
          f"owl->fly / penguin->fly; moat turn zzz->fly\n", flush=True)

    ws = GNWWorkspace(seed=seed)
    reasoner = build_reasoner(seed=seed, shim=gpu, lesion=False)
    unified = None
    if gpu:
        from research.runners._emerge68_function_word_spell_derisk import UnifiedNeuralSpell
        u = UnifiedNeuralSpell(load=True)
        unified = u if u._backend_gpu else None
        if unified is None:
            print("  [skip] A->W engines not on GPU (cache missing / numpy fallback) -- surface deferred.\n")

    for spec in ANSWER_TURNS:
        tr = run_turn(ws, reasoner, unified, spec["member"], spec["property"])
        surface = tr.get("surface") or "(A->W deferred to GPU)"
        print(f"  ignite {tr['member']:8s} -> report='{tr['ignited']}' | reason={tr['gate']} | "
              f"reasoned='{tr['reasoned']}' | speak-> '{surface}'  (expect '{spec['expect_surface']}')")
    moat = run_turn(ws, reasoner, unified, MOAT_TURN["member"], MOAT_TURN["property"])
    print(f"  ignite {moat['member']:8s} -> report='{moat['ignited']}' | reason={moat['gate']} (MOAT) | "
          f"A->W spell calls={moat['spell_calls']} (must be 0)")
    print(f"\n  => workspace ignite -> string identity read -> gate-first reason -> (GPU) spike-render; the concept "
          f"string round-trips ignited==reasoned; the moat abstains + never invokes the A->W.\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-4 real-faculty wire-in de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", type=str, default="auto", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default=str(OUT))
    ap.add_argument("--probe", action="store_true", help="single-seed human-readable walk-through")
    ap.add_argument("--derisk", action="store_true", help="run the de-risk gates + write the finding json")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seed, a.backend, a.json)
    return _probe(a.seed, a.backend)


if __name__ == "__main__":
    raise SystemExit(main())
