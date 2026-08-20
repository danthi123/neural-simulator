"""IDLE-REPLAY-ON-D5-EPISODIC de-risk -- does the 6-seed GO emergent pattern-completion replay mechanism
(`_emergent_replay_specificity_derisk.py`, 2026-08-20: a plastic recurrent assembly reactivated by UNTARGETED noise,
pattern-completes ONLY a real encoded assembly) TRANSFER to the REAL production episodic-memory bridge
(`research.runners.d5_episodic_production_organ.EpisodicRecallOrgan` / `_episodic_dap_dialogue_memory.EpisodicDapMemory`,
the standing 6/6-GO gap#5 dendritic-dAP readout, n_ca3=2000)? This is the next step toward wiring learn-through-use
LIVE: the prior de-risk proved the mechanism on a SYNTHETIC toy network (n_pre=64, ALL of "pre" IS the one assembly,
`internal_density=1.0` full recurrent fan-in); this runner asks whether the SAME untargeted-noise-driven pattern-
completion + strengthening shows up on the REAL sparse hippocampal code (an emergently-selected ~13-27-cell CA3
assembly out of n_ca3=2000, i.e. ~1% of the population), reusing the REAL organ UNCHANGED (import only, no sim/ edit,
no existing runner edit).

THE KEY QUESTION, answered by READING the D5 bridge BEFORE running anything (corpus-first; the answer changes what
this runner even measures): does the D5 CA3 bridge have RECURRENT CA3->CA3 connectivity able to PATTERN-COMPLETE a
stored assembly from a noise-driven PARTIAL cue -- the mechanism emergent-replay's noise-driven recruitment needs?

  YES, recurrence EXISTS and is REAL: `research.runners._riii_ca3_coincidence_completion_derisk._build` constructs
  a genuine ca3->ca3 RegionPathway (`ca3_recurrent_density`, default 0.5 in D5's GO_DEFAULTS -- i.e. a DENSE recurrent
  net among the 2000 CA3 cells) and routes it through `coincidence_detector=True` -> every CA3->CA3 synapse's spikes
  feed a per-POST-cell coincidence COUNT (`c_count`, the number of SYNCHRONOUS clustered inputs this step) into
  `sim/kernels.py:fused_coincidence_plateau`: >=k_thresh coincident inputs this step triggers a regenerative apical
  plateau (self-sustaining via `self_regen`+KIR) on `cp_v_apical`. BTSP-forming a topic (`store()`) potentiates
  EXACTLY that topic's within-assembly recurrent synapses (`w_within` grows from baseline ~1.5 to near `btsp_w_max`
  =100), so a REAL recurrent substrate for pattern completion is present and load-bearing (dapB's own LINEAR-control
  anti-cheat: turning coincidence OFF, same formed weights, completion FAILS -- the recurrent+dendritic mechanism is
  what completes, not raw weight).

  BUT (the reason this is a genuine transfer risk, not a rubber-stamp): D5's OWN precedent
  (`_gap5_dendritic_dap_readout_completion_derisk.py` docstring, "THE SEAM") explicitly characterizes that at THIS
  assembly scale (~13-27 emergent cells) reading the RECURRENT ATTRACTOR AT THE POPULATION LEVEL (a SOMA-firing
  read driven by a host-known partial CUE) is NON-SPECIFIC: "cue-completion + self-ignition share the WITHIN-assembly
  recurrent gain -- a ~23-cell set is too small for a RECURRENT bistable attractor at any inhibition." D5's fix was
  to REPLACE the population-level recurrent-attractor READ with an INTRINSIC PER-CELL dendritic dAP latch, driven by
  a HOST-KNOWN cue (specific, pre-identified assembly member indices from `_held_cue_perm`) with DELIBERATELY WEAK
  apical<->soma back-coupling (apical_gc=apical_gc_read=0.3) specifically so the read is DECOUPLED from population
  reverberation. Every drive in D5's pipeline -- encode (`_form_one_assembly`: drives ONLY the assembly's own known
  members), recall (`_apical_up_read`: drives ONLY the cue-fraction of a NAMED topic's known members) -- is
  HOST-ADDRESSED. Nothing in the existing pipeline ever drives an UNTARGETED random subset of the WHOLE 2000-cell
  CA3 population and asks the recurrent net to figure out, on its own, which ~1%-sparse assembly (if any) that noise
  overlaps. That untargeted-recruitment operating regime is exactly what emergent-replay's mechanism needs and is
  the genuinely UNTESTED composition this runner closes or bounds.

MECHANISM (reuse, not re-derive; NO sim/ edit; NO existing runner edit -- pure runner-level driving of the ALREADY-
BUILT `EpisodicDapMemory.bridge`/`.R`):
  A key implementation finding (read `sim/bridge.py:10058-10264`, the BTSP block) that SIMPLIFIES this de-risk vs the
  synthetic emergent-replay runner: BTSP's postsynaptic instructive signal `IS_post = max(cp_v_apical - v_hold, 0)`
  (bridge.py:10088) reads WHATEVER `cp_v_apical` currently is -- it does NOT require the host-set `cp_bdsp_apical_drive`
  BDSP path `_form_one_assembly` uses for ENCODE. Because the READOUT bridge's ca3->ca3 pathway is PERMANENTLY routed
  through `coincidence_detector=True` (a per-pathway BUILD-time property, always active regardless of `enable_btsp`),
  `cp_v_apical` already reflects genuine recurrent-coincidence dendritic state on EVERY step. So simply flipping
  `bridge.core_config.enable_btsp = True` while driving UNTARGETED noise into a random CA3 subset lets the SAME
  dAP-completion machinery D5 already uses for READS also WRITE -- an emergent, self-organized potentiation gated by
  the bridge's OWN existing kernels (`fused_coincidence_plateau` + `fused_btsp_update`), not a foreign runner-level
  Hebbian model (unlike the synthetic emergent-replay runner's `_hebbian_assembly_step`, which was necessary there
  because that toy network's "pre" region carries no dendritic coincidence machinery at all).

PROTOCOL per instance (`EpisodicDapMemory(seed, topics=["dog","cat"])`, GO_DEFAULTS unchanged, kt=8):
  1. `note_topic("dog")` -- BTSP-forms dog's CA3 assembly (real spiking encode). "cat" stays UNSTORED (the never-
     formed, baseline-recurrent-weight SPECIFICITY control -- matches the emergent-replay precedent's moat_replay).
  2. BASELINE recall("dog")/recall("cat") -- the standard host-cued read (sanity + apical_nocue chance baseline).
  3. QUIET window (G2 "skip-the-replay-pass" lesion): N steps with NO drive, BTSP OFF -- recall again ("quiet").
     Because nothing plastic ever runs and every recall() independently hard-silences+resets state before its own
     drive, quiet-vs-baseline recall should be identical if-and-only-if nothing leaked state across phases (an
     INSTRUMENT check, not just a control).
  4. REPLAY window (the mechanism, both halves at once): reset to a clean rest state (`R.hard_silence()` +
     `_reset_apical_latch`), enable_btsp=True (mem.p's own btsp_lr/wmax/elig_tau -- the SAME formation constants),
     drive a RANDOM subset (`--noise-frac` of n_ca3, drawn from the WHOLE CA3 index range, blind to topic identity)
     with `--noise-pA` for warm+read steps (mem.p's own warm_steps/read_steps, matching the standard recall window
     scale for comparability). DURING this window: sample the per-step fraction of dog's held-out assembly cells and
     cat's held-out assembly cells whose `cp_v_apical` is elevated (apical UP fraction, `up_thresh`) and whose soma
     fires -- the PC (pattern-completion) diagnostic, the direct empirical read of THE KEY QUESTION. Freeze BTSP
     again after. Record `w_within(dog)`/`w_within(cat)` before/after (does untargeted noise potentiate the FORMED
     assembly specifically, or not at all, or non-specifically). recall("dog")/recall("cat") again ("after_replay").
  5. Verdict: PC (KEY QUESTION: does noise recruit dog's assembly more than cat's, same dose) -- the gate everything
     else is contingent on; G1 (recall improves after replay vs quiet); G2 (quiet==baseline, the lesion has no
     spurious gain); G3 (cat's recall stays at chance after replay, specificity). A failed PC is reported as the
     HONEST TRANSFER-NEGATIVE this runner exists to characterize, not tuned away (THE LAW: name the residual, not a
     stop) -- see the runner's own NEXT_RUNG for the smallest bridging step.

SCOPE (declared): this uses ONE seed (n_ca3=2000 numpy is ~15-25 min PER INSTANCE build+store; the task's own
"reduce n_ca3 via the organ's overrides" escape does NOT exist on the real production organ -- `EpisodicDapMemory`
accepts `**overrides` only into `GO_DEFAULTS` params (density/wmax/kthresh/...), NOT n_ca3, which is hardcoded to
2000 inside `_gap5_emergent_end_to_end_episodic_loop_derisk.emergent_assemblies`'s `R1` dict, imported unchanged --
"the n_ca3=400 GO does NOT reproduce" per that module's own docstring, so there is no smaller-scale variant of the
REAL organ to fall back to without editing an existing runner, which is out of scope here). Single-seed is a scoped
CPU de-risk (transfer risk / architecture question, not a generalization claim) -- 6-seed is the natural next step
IF this seed's answer is GO.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._idle_replay_on_d5_episodic_derisk --seed 42 \
        --out research/findings/raw/_idle_replay_on_d5_episodic/seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from tools.lab import attributable_to, void_if, assert_backend  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
# reuse-by-import, UNCHANGED: the real D5 production mechanism.
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_idle_replay_on_d5_episodic" / "seed42.json"

# Completion criterion, reused verbatim from D5's own GO bar (_episodic_dap_dialogue_memory.py).
COMPLETE_MIN, CUE_OVER_CTRL, CTRL_MAX = 0.20, 3.0, 0.10
# PC (pattern-completion) separation bar: a real margin, not noise. Loose on purpose (n=1 seed, exploratory) --
# the point is SIGN + MAGNITUDE, a strict bar belongs to a later 6-seed confirmatory run if this is GO.
PC_MIN_SEPARATION = 0.05


def _topic_held_global(mem, topic):
    """GLOBAL neuron indices of `topic`'s HELD-OUT assembly cells (cp_v_apical / cp_firing_states are global-indexed;
    held_pos_by_asm stores LOCAL ca3 positions -- convert via R.ca3_idx, exactly like _apical_up_read does)."""
    slot = mem.topic_slot[topic]
    held_pos = mem.held_pos_by_asm[slot]
    return np.asarray([int(mem.R.ca3_idx[p]) for p in held_pos], dtype=np.int64)


def _topic_assembly_global(mem, topic):
    """GLOBAL indices of `topic`'s FULL assembly (cue+held), for the w_within read."""
    slot = mem.topic_slot[topic]
    return np.asarray(mem.assemblies[slot], dtype=np.int64)


def _w_within(mem, topic):
    slot = mem.topic_slot[topic]
    m = mem.R.withinA_masks[slot]
    cp = mem.R.cp
    n = int(to_host(cp.sum(m)))
    return float(to_host(cp.mean(mem.R.C.data[m]))) if n else 0.0


def _w_rec_all(mem):
    cp = mem.R.cp
    m = mem.R.rec_mask
    n = int(to_host(cp.sum(m)))
    return float(to_host(cp.mean(mem.R.C.data[m]))) if n else 0.0


def _quiet_window(mem, n_steps):
    """N steps, zero drive, BTSP/BDSP left at their (already-permanent) OFF state -- the 'skip the replay pass'
    lesion / G2 control. Snapshots w_rec before/after to prove this phase is genuinely inert (an instrument check,
    not merely asserted)."""
    bridge = mem.bridge
    n = bridge.cp_membrane_potential_v.size
    w_before = _w_rec_all(mem)
    mem.R.hard_silence()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    w_after = _w_rec_all(mem)
    return {"w_rec_before": w_before, "w_rec_after": w_after, "inert": bool(abs(w_after - w_before) < 1e-9)}


def _replay_window(mem, args, rng):
    """THE MECHANISM: reset clean, enable_btsp, drive an UNTARGETED noise subset of the WHOLE CA3 population (blind
    to topic identity), sample the PC diagnostic (dog vs cat held-cell activation under the identical noise dose)
    every `--pc-sample-every` steps, freeze BTSP again. Returns per-topic PC time series + w_within before/after."""
    bridge = mem.bridge
    cp = mem.R.cp
    n = bridge.cp_membrane_potential_v.size

    dog_held = _topic_held_global(mem, "dog")
    cat_held = _topic_held_global(mem, "cat")

    w_within_dog_before = _w_within(mem, "dog")
    w_within_cat_before = _w_within(mem, "cat")
    w_rec_before = _w_rec_all(mem)

    # untargeted noise subset: a random draw from the WHOLE ca3 index range, independent of any topic's membership.
    n_noise = max(1, int(round(len(mem.R.ca3_idx) * args.noise_frac)))
    noise_idx = rng.choice(mem.R.ca3_idx, size=n_noise, replace=False)

    mem.R.hard_silence()
    _reset_apical_latch(bridge)
    cfg = bridge.core_config
    saved_btsp = bool(getattr(cfg, "enable_btsp", False))
    saved_lr = float(getattr(cfg, "btsp_learning_rate", 0.0))
    saved_wmax = float(getattr(cfg, "btsp_w_max", 5.0))
    saved_tau = float(getattr(cfg, "btsp_elig_tau_ms", 1000.0))
    cfg.enable_btsp = True
    cfg.btsp_learning_rate = float(mem.p["btsp_lr"])
    cfg.btsp_w_max = float(mem.p["wmax"])
    cfg.btsp_elig_tau_ms = 1000.0
    if getattr(bridge, "cp_btsp_pre_elig", None) is not None:
        bridge.cp_btsp_pre_elig[:] = 0.0

    drive = np.zeros(n, dtype=np.float32)
    drive[noise_idx] = args.noise_pA
    bridge.cp_external_input_current[:] = cp.asarray(drive)

    n_total = int(mem.p["warm_steps"]) + int(mem.p["read_steps"])
    dog_apical_up, cat_apical_up = [], []
    dog_fire, cat_fire = [], []
    for step in range(n_total):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        if step % max(1, args.pc_sample_every) == 0 or step == n_total - 1:
            va = to_host(bridge.cp_v_apical) if getattr(bridge, "cp_v_apical", None) is not None else None
            if va is not None:
                dog_apical_up.append(float(np.mean(va[dog_held] > mem.p["up_thresh"])) if len(dog_held) else 0.0)
                cat_apical_up.append(float(np.mean(va[cat_held] > mem.p["up_thresh"])) if len(cat_held) else 0.0)
            fired = to_host(bridge.cp_firing_states).astype(bool)
            dog_fire.append(float(np.mean(fired[dog_held])) if len(dog_held) else 0.0)
            cat_fire.append(float(np.mean(fired[cat_held])) if len(cat_held) else 0.0)

    bridge.cp_external_input_current[:] = 0.0
    cfg.enable_btsp = saved_btsp
    cfg.btsp_learning_rate = saved_lr
    cfg.btsp_w_max = saved_wmax
    cfg.btsp_elig_tau_ms = saved_tau

    w_within_dog_after = _w_within(mem, "dog")
    w_within_cat_after = _w_within(mem, "cat")
    w_rec_after = _w_rec_all(mem)

    return {
        "n_noise_driven": int(n_noise), "noise_frac": args.noise_frac, "noise_pA": args.noise_pA,
        "n_steps": n_total,
        "dog_apical_up_series": dog_apical_up, "cat_apical_up_series": cat_apical_up,
        "dog_fire_series": dog_fire, "cat_fire_series": cat_fire,
        "dog_apical_up_mean": float(np.mean(dog_apical_up)) if dog_apical_up else 0.0,
        "cat_apical_up_mean": float(np.mean(cat_apical_up)) if cat_apical_up else 0.0,
        "dog_apical_up_max": float(np.max(dog_apical_up)) if dog_apical_up else 0.0,
        "cat_apical_up_max": float(np.max(cat_apical_up)) if cat_apical_up else 0.0,
        "dog_fire_mean": float(np.mean(dog_fire)) if dog_fire else 0.0,
        "cat_fire_mean": float(np.mean(cat_fire)) if cat_fire else 0.0,
        "w_within_dog_before": w_within_dog_before, "w_within_dog_after": w_within_dog_after,
        "w_within_cat_before": w_within_cat_before, "w_within_cat_after": w_within_cat_after,
        "w_rec_before": w_rec_before, "w_rec_after": w_rec_after,
    }


def run(seed, args):
    t0 = time.time()
    result = {"seed": seed, "backend": os.environ.get("SIM_BACKEND", "(unset)"), "timing": {}}

    t_build = time.time()
    mem = EpisodicDapMemory(seed, ["dog", "cat"], verbose=True)
    result["timing"]["init_s"] = round(time.time() - t_build, 1)
    result["n_ca3"] = mem.n_ca3
    result["assembly_sizes"] = mem.assembly_sizes
    result["topic_slot"] = dict(mem.topic_slot)
    result["held_sizes"] = {t: len(mem.held_pos_by_asm[mem.topic_slot[t]]) for t in ("dog", "cat")}

    t_store = time.time()
    wrote = mem.store("dog")
    result["timing"]["store_s"] = round(time.time() - t_store, 1)
    result["store_wrote"] = bool(wrote)
    result["w_within_dog_poststore"] = _w_within(mem, "dog")
    result["w_within_cat_poststore"] = _w_within(mem, "cat")   # baseline (never formed)

    t_r = time.time()
    baseline = {"dog": mem.recall("dog"), "cat": mem.recall("cat")}
    result["timing"]["baseline_recall_s"] = round(time.time() - t_r, 1)
    result["baseline"] = baseline
    print(f"[idle-replay-d5] baseline dog={baseline['dog']} cat={baseline['cat']} "
          f"(+{time.time()-t0:.0f}s)", flush=True)

    t_q = time.time()
    quiet_instr = _quiet_window(mem, args.quiet_steps)
    quiet = {"dog": mem.recall("dog"), "cat": mem.recall("cat")}
    result["timing"]["quiet_phase_s"] = round(time.time() - t_q, 1)
    result["quiet_instrument"] = quiet_instr
    result["quiet"] = quiet
    print(f"[idle-replay-d5] quiet(no-replay lesion) dog={quiet['dog']} cat={quiet['cat']} "
          f"instrument={quiet_instr} (+{time.time()-t0:.0f}s)", flush=True)

    t_rep = time.time()
    rng = np.random.default_rng(seed * 7919 + 1)
    replay_diag = _replay_window(mem, args, rng)
    result["timing"]["replay_phase_s"] = round(time.time() - t_rep, 1)
    result["replay_diagnostic"] = replay_diag
    print(f"[idle-replay-d5] REPLAY pc: dog_apical_up={replay_diag['dog_apical_up_mean']:.3f} "
          f"(max {replay_diag['dog_apical_up_max']:.3f}) cat_apical_up={replay_diag['cat_apical_up_mean']:.3f} "
          f"(max {replay_diag['cat_apical_up_max']:.3f}) | dog_fire={replay_diag['dog_fire_mean']:.3f} "
          f"cat_fire={replay_diag['cat_fire_mean']:.3f} | w_within dog {replay_diag['w_within_dog_before']:.2f}->"
          f"{replay_diag['w_within_dog_after']:.2f} cat {replay_diag['w_within_cat_before']:.2f}->"
          f"{replay_diag['w_within_cat_after']:.2f} (+{time.time()-t0:.0f}s)", flush=True)

    t_a = time.time()
    after = {"dog": mem.recall("dog"), "cat": mem.recall("cat")}
    result["timing"]["after_recall_s"] = round(time.time() - t_a, 1)
    result["after_replay"] = after
    print(f"[idle-replay-d5] after-replay dog={after['dog']} cat={after['cat']} "
          f"(+{time.time()-t0:.0f}s)", flush=True)

    # ---- lesion recall (D5's own load-bearing teeth: baseline weights -> completion must collapse) --------------
    t_l = time.time()
    lesion_dog = mem.recall("dog", lesion=True)
    result["timing"]["lesion_recall_s"] = round(time.time() - t_l, 1)
    result["lesion_dog"] = lesion_dog

    result["timing"]["total_s"] = round(time.time() - t0, 1)
    return result


def build_verdict(r, args):
    base_dog, base_cat = r["baseline"]["dog"], r["baseline"]["cat"]
    quiet_dog, quiet_cat = r["quiet"]["dog"], r["quiet"]["cat"]
    after_dog, after_cat = r["after_replay"]["dog"], r["after_replay"]["cat"]
    rep = r["replay_diagnostic"]

    # ---- instrument checks ----
    intact_fires = bool(base_dog["in_memory"] and base_dog["apical_cue"] >= COMPLETE_MIN
                        and base_dog["apical_perm"] <= CTRL_MAX and base_dog["apical_nocue"] <= CTRL_MAX)
    unstored_abstains = bool((not base_cat["in_memory"]) and base_cat["apical_cue"] <= CTRL_MAX)
    lesion_collapses = bool((not r["lesion_dog"]["in_memory"]) and r["lesion_dog"]["apical_cue"] <= CTRL_MAX)
    quiet_inert = bool(r["quiet_instrument"]["inert"])
    recall_reproducible = bool(abs(quiet_dog["apical_cue"] - base_dog["apical_cue"]) < 1e-6
                               and abs(quiet_cat["apical_cue"] - base_cat["apical_cue"]) < 1e-6)
    instr_ok = bool(intact_fires and unstored_abstains and lesion_collapses and quiet_inert and recall_reproducible)

    # ---- THE KEY QUESTION: PC (pattern completion) -- does untargeted noise recruit dog's (formed) assembly more
    # than cat's (never-formed), under the IDENTICAL noise dose? Read via BOTH the apical dAP state (the D5 read
    # variable) and raw soma firing (a coarser, assumption-light cross-check).
    pc_apical_gap = rep["dog_apical_up_mean"] - rep["cat_apical_up_mean"]
    pc_apical_max_gap = rep["dog_apical_up_max"] - rep["cat_apical_up_max"]
    pc_fire_gap = rep["dog_fire_mean"] - rep["cat_fire_mean"]
    PC = bool(pc_apical_gap >= PC_MIN_SEPARATION or pc_apical_max_gap >= PC_MIN_SEPARATION)

    # ---- write-side: did untargeted replay specifically potentiate dog's within-assembly weight (vs cat's, vs the
    # recurrent population mean)? ----
    dw_dog = rep["w_within_dog_after"] - rep["w_within_dog_before"]
    dw_cat = rep["w_within_cat_after"] - rep["w_within_cat_before"]
    WRITE_SPECIFIC = bool(dw_dog > dw_cat + 1e-6 and dw_dog > 1e-6)

    # ---- behavioral G1/G2/G3 (contingent on completion existing at all -- if PC is flat, these are expected flat
    # too and that IS the honest read, not evidence against the instrument) ----
    g1_gain_dog = after_dog["apical_cue"] - quiet_dog["apical_cue"]
    G1 = bool(g1_gain_dog > 1e-3)
    G2_no_spurious_gain = bool(abs(quiet_dog["apical_cue"] - base_dog["apical_cue"]) < 1e-3)
    g3_gain_cat = after_cat["apical_cue"] - quiet_cat["apical_cue"]
    G3_specific = bool(g3_gain_cat < 1e-3)

    transfers = bool(instr_ok and PC and WRITE_SPECIFIC)

    v = Verdict("idle-replay-on-D5-episodic transfer de-risk (untargeted-noise pattern completion + BTSP write, "
               "REAL production organ, n_ca3=2000)")
    v.require("instrument: D5's own intact/unstored/lesion/quiet-inert/recall-reproducible all hold",
              instr_ok, expect=True)
    v.require("KEY QUESTION -- PC: untargeted noise recruits dog's (formed) assembly's apical dAP state above "
              "cat's (never-formed), same noise dose", PC, expect=True)
    v.require("write-side: replay's BTSP write specifically potentiates dog's within-assembly weight, not cat's",
              WRITE_SPECIFIC, expect=True)
    v.control("PC: dog vs cat apical-UP fraction under identical untargeted-noise dose",
              treatment=rep["dog_apical_up_mean"], control=rep["cat_apical_up_mean"], min_separation=0.0)
    v.control("write: dog vs cat within-assembly weight delta under identical replay dose",
              treatment=dw_dog, control=dw_cat, min_separation=0.0)
    v.reaches("store() potentiates dog's within-assembly weight above the never-formed baseline (cat)",
              before=r["w_within_cat_poststore"], after=r["w_within_dog_poststore"])
    for proc in ("STDP", "Hebbian (kernel-driven)", "homeostasis", "short-term plasticity", "reward modulation",
                 "structural plasticity", "BDSP learning (the tested-NEGATIVE hidden-credit rule)"):
        v.disabled(proc, why="isolation: only the pathway's permanent coincidence-plateau routing + BTSP (toggled "
                             "only during the replay window) are live; this is the SAME instrument D5's own dapB/"
                             "btsp-forms precedents use")
    decided = v.decide(go=transfers)

    void_if(not instr_ok, "an instrument check failed (intact-fires / unstored-abstains / lesion-collapses / "
                          "quiet-inert / recall-reproducible)")
    attributable_to("replay recall gain (dog): after-replay vs quiet(no-replay)", after_dog["apical_cue"],
                    quiet_dog["apical_cue"])
    attributable_to("PC pattern-completion (dog vs cat apical-UP, identical noise dose)",
                    rep["dog_apical_up_mean"], rep["cat_apical_up_mean"])

    if not instr_ok:
        verdict = ("UNDEFINED -- an instrument precondition failed (see 'instrument' checks); the transfer claim is "
                   "NOT cleanly attributable at this run. Re-check before re-running.")
    elif transfers:
        verdict = (f"GO -- untargeted-noise-driven idle reactivation on the REAL D5 CA3 bridge (n_ca3=2000, "
                  f"emergent {r['assembly_sizes']}-cell assembly) DOES pattern-complete the formed 'dog' assembly "
                  f"specifically (apical-UP dog={rep['dog_apical_up_mean']:.3f} vs cat={rep['cat_apical_up_mean']:.3f}) "
                  f"and the SAME bridge's own BTSP kernel writes specifically to dog's within-assembly weight "
                  f"(dw dog={dw_dog:.3f} vs cat={dw_cat:.3f}) from that noise-gated coincidence-plateau alone -- "
                  f"the emergent-replay mechanism TRANSFERS to the real sparse hippocampal code with NO new "
                  f"machinery beyond toggling enable_btsp during the reactivation window.")
    else:
        miss = []
        if not PC:
            miss.append(f"PC FAILED -- untargeted noise did NOT preferentially recruit dog's assembly over cat's "
                       f"(apical-UP dog={rep['dog_apical_up_mean']:.3f} max={rep['dog_apical_up_max']:.3f} vs "
                       f"cat={rep['cat_apical_up_mean']:.3f} max={rep['cat_apical_up_max']:.3f}; fire dog="
                       f"{rep['dog_fire_mean']:.3f} cat={rep['cat_fire_mean']:.3f})")
        if not WRITE_SPECIFIC:
            miss.append(f"WRITE not specific (dw dog={dw_dog:.4f} cat={dw_cat:.4f})")
        verdict = ("UNDEFINED/HONEST-TRANSFER-NEGATIVE -- " + "; ".join(miss) + ". Per THE LAW this is a verdict on "
                  "a METHOD (untargeted whole-population noise at this dose/duration/assembly-scale), not a "
                  "closure of the capability. See NEXT_RUNG for the smallest bridging step.")

    return {"GO": transfers, "verdict": verdict, "decided": decided,
            "instr_ok": instr_ok, "PC": PC, "WRITE_SPECIFIC": WRITE_SPECIFIC,
            "G1_gain_after_replay": G1, "G2_no_spurious_quiet_gain": G2_no_spurious_gain,
            "G3_specificity_cat_no_gain": G3_specific,
            "pc_apical_gap": pc_apical_gap, "pc_apical_max_gap": pc_apical_max_gap, "pc_fire_gap": pc_fire_gap,
            "dw_dog": dw_dog, "dw_cat": dw_cat, "g1_gain_dog": g1_gain_dog, "g3_gain_cat": g3_gain_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet-steps", type=int, default=100, dest="quiet_steps")
    # noise_frac: fraction of the WHOLE n_ca3 population driven directly (untargeted, blind to any assembly).
    # 0.18 matches the emergent-replay precedent's calibrated fraction on its toy "pre" region; here it is applied
    # to a population ~30x larger relative to the assembly size, so it is expected (not merely possible) to be a
    # much weaker dose relative to assembly size -- reported honestly, not pre-tuned to succeed.
    ap.add_argument("--noise-frac", type=float, default=0.18, dest="noise_frac")
    ap.add_argument("--noise-pA", type=float, default=900.0, dest="noise_pA")
    ap.add_argument("--pc-sample-every", type=int, default=10, dest="pc_sample_every")
    a = ap.parse_args()

    try:
        assert_backend("numpy", note="(CPU lane; n_ca3=2000 dense CA3 recurrence is faithful-but-slow -- see the "
                                     "runner docstring's SCOPE note on why n_ca3 cannot be reduced without editing "
                                     "an existing runner)")
    except AssertionError as e:
        print("BACKEND WARNING: %s" % e)

    err = None
    result = None
    verdict_block = None
    try:
        result = run(a.seed, a)
        verdict_block = build_verdict(result, a)
    except (RuntimeError, ValueError, AttributeError, KeyError, IndexError, TypeError) as e:
        err = "%s: %s" % (type(e).__name__, e)
        traceback.print_exc()

    summary = {"probe": "idle_replay_on_d5_episodic_transfer", "seed": a.seed,
              "params": {"quiet_steps": a.quiet_steps, "noise_frac": a.noise_frac, "noise_pA": a.noise_pA,
                        "pc_sample_every": a.pc_sample_every},
              "backend": os.environ.get("SIM_BACKEND", "(unset)")}
    if err is not None:
        summary["error"] = err
        summary["GO"] = False
        summary["verdict"] = f"ERROR -- {err}"
    else:
        summary["run"] = result
        summary.update(verdict_block)

    summary["HONEST_NOTE"] = (
        "This runner imports the REAL production D5 organ (research.runners._episodic_dap_dialogue_memory."
        "EpisodicDapMemory / research.runners.d5_episodic_production_organ.EpisodicRecallOrgan) UNCHANGED -- no "
        "sim/ edit, no existing runner edit, single seed (a scoped CPU transfer de-risk, not a generalization "
        "claim; 6-seed is the natural next rung IF GO). The 'replay' pattern-completion + write mechanism reuses "
        "the bridge's OWN existing kernels (fused_coincidence_plateau for the per-cell dAP state, fused_btsp_update "
        "for the write) -- toggling only enable_btsp/btsp_learning_rate/btsp_w_max during the reactivation window "
        "and driving cp_external_input_current into an untargeted random CA3 subset; NO runner-level Hebbian model "
        "was needed here (unlike the synthetic emergent-replay precedent), because this bridge's ca3->ca3 pathway "
        "already permanently routes through the coincidence-plateau kernel. n_ca3=2000 is NOT reducible without "
        "editing research/runners/_gap5_emergent_end_to_end_episodic_loop_derisk.py's hardcoded R1 dict (out of "
        "scope: 'import UNCHANGED'). NOT 'consolidation' or 'stabilization' in the docs/TERMS.md sense -- no source-"
        "structure lesion / systems-level independence was tested, and no forgetting/decay curve was modeled; this "
        "is a single-tick transfer-mechanism probe.")
    summary["NEXT_RUNG"] = (
        "IF GO at this seed: re-run at 6 seeds (42 43 44 100 101 102) to confirm before calling it closed; add the "
        "starting-weight-gated metaplastic write suppression from the synthetic precedent if raw BTSP alone shows a "
        "moat leak at scale; wire under continuous_engine.py's idle tick, default-off first. "
        "IF HONEST-NEGATIVE (the expected outcome per the dapB precedent's own characterized SEAM -- a ~1%-of-"
        "population sparse assembly is very unlikely to be preferentially hit by population-blind noise at a dose "
        "sized for a 64-cell all-assembly toy net): the smallest bridging step is NOT 'a bigger noise dose' (that "
        "risks the SAME non-specificity dapB already found at the population-recurrent level -- more noise recruits "
        "cat's baseline connectivity too, just via raw excitability rather than genuine completion). The bridging "
        "step this runner's own architecture points to is TARGETED-BUT-CONTENT-BLIND replay: reactivate from a "
        "SPARSE-CODE-AWARE noise source (e.g. drive the DG/EC afferent layer, whose mossy-fiber detonator ALREADY "
        "concentrates activity onto whichever CA3 cells were originally selected for a stored pattern, per "
        "emergent_assemblies' own DG->CA3 detonation mechanism) rather than uniform CA3-wide noise -- this is "
        "closer to the biological SWR replay trigger (CA3-internal sharp-wave initiation via recurrent excitation "
        "concentrated near recently-potentiated synapses) than i.i.d. noise over 2000 cells, and does not require "
        "knowing which topic to replay (still untargeted at the TOPIC level, just not blind to the DG->CA3 pathway "
        "structure). A second, cheaper lever: raise noise_frac substantially while WATCHING cat's apical-UP for the "
        "onset of non-specific self-ignition (dapB's own failure mode) as the ceiling on how far that lever can go.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[idle-replay-d5] VERDICT: {summary.get('verdict')}", flush=True)
    print(f"[idle-replay-d5] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
