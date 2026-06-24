"""Assemble research/findings/raw/_self_knowledge_chat_fix.json from the probe JSONs (the deliverable)."""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
RAW = os.path.join(_REPO, "research", "findings", "raw")


def _load(name):
    with open(os.path.join(RAW, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


recall = _load("_self_knowledge_recall_probe.json")
e2e = _load("_self_knowledge_chat_e2e_probe.json")
qwen = _load("_self_knowledge_qwen_realrepro.json")
verify = _load("_self_knowledge_chat_verify.json")

rows = recall["recall_vs_D_and_codes"]


def pick(codes, D):
    return next(r for r in rows if r["codes"] == codes and r["D"] == D)


def e2erun(codes):
    return next(r for r in e2e["runs"] if r["codes"] == codes)


def qarm(arm):
    return next(r for r in qwen["arms"] if r["arm"] == arm)


dc = e2erun("grounded_decorr")
out = {
    "probe": "self_knowledge_chat_fix__recall_root_cause_plus_qwen_crash_fix",
    "resolves": ("Make the self-knowledge chat brain chat-ready: recall >=0.8 at 52 facts (was 0.21) + the "
                 "off-bridge Qwen fluency working (was a hard-crash) + the firewall (0-FA) intact."),
    "backend": "cupy (GPU) for the faculty/bridge arms; numpy (CPU) for the recall isolation (seconds)",
    "seed": 42,

    "PART1_RECALL": {
        "root_cause": (
            "The stream-learned GROUNDED codes COLLAPSE at scale: ~22% of the 88 concept-code PAIRS have "
            "phase-cosine > 0.9 (max 0.995) because many concepts are heard in near-identical hub contexts -> "
            "near-identical double-centred code rows -> near-identical phasors. Composer recall = unbind + "
            "cleanup(argmax phase-cos over the vocab); on collapsed codes the cleanup returns a near-neighbour "
            "(e.g. 'faculty' when the answer is 'development', cos 0.995) -> recall ~0.21 at 52 facts. NOT D, "
            "NOT scale-of-store, NOT the cue-matcher: clean rng.uniform codes give 0.94 at EVERY D (128..1024); "
            "the cue-matcher/integrated_loop routes identically. This is the documented graded-magnitude / "
            "code-correlation family (the point-neuron decorrelation wall)."),
        "evidence_recall_vs_D_and_codes": {
            "grounded": {str(D): pick("grounded", D)["recall_acc"] for D in (128, 256, 512, 1024)},
            "grounded_decorr_THE_FIX": {str(D): pick("grounded_decorr", D)["recall_acc"] for D in (128, 256, 512, 1024)},
            "random_clean": {str(D): pick("random", D)["recall_acc"] for D in (128, 256, 512, 1024)},
        },
        "evidence_code_xcorr": {
            "grounded_D128": {k: pick("grounded", 128)["code_xcorr"][k] for k in ("mean", "max", "p95", "frac_above_0.9")},
            "grounded_decorr_D128": {k: pick("grounded_decorr", 128)["code_xcorr"][k] for k in ("mean", "max", "p95", "frac_above_0.9")},
            "random_D128": {k: pick("random", 128)["code_xcorr"][k] for k in ("mean", "max", "p95", "frac_above_0.9")},
        },
        "degradation_curve_D128": recall["recall_degradation_curve"],
        "THE_FIX": (
            "ZCA-decorrelate the grounded codes (a HOST post-processing of the codes -- the codes are a legitimate "
            "host-shaped INPUT to the composer, like the project flat-distinct path that uses per-bridge distinct "
            "seeds; the composer spiking bind/unbind/cleanup algebra is UNTOUCHED). decorrelate_grounded_codes() "
            "applied in build_qa_agent (default ON). Recall 0.21 -> 0.94 at the build D=128 (NO D bump, NO extra "
            "VRAM)."),
        "fix_e2e_through_full_BrainConversationalAgent": {
            "grounded_recall": e2erun("grounded")["agent_recall_acc"],
            "grounded_decorr_recall": dc["agent_recall_acc"],
            "grounded_decorr_firewall_project": f"{dc['firewall_positive_answered']}/{dc['firewall_positive_total']}",
            "grounded_decorr_general_leaks": dc["firewall_general_leaks"],
            "grounded_decorr_untaught_leaks": dc["firewall_untaught_leaks"],
        },
        "residual": (
            "recall 0.94 not 1.00 = 3 first-match collisions where the agent+action cue has TWO facts (brain "
            "learns words AND brain learns listening; brain remembers days AND brain remembers facts; brain is "
            "neural AND brain is spiking) -> query returns the OTHER valid patient. Inherent to the curriculum, "
            "identical on clean random codes; not a code/cleanup defect."),
    },

    "PART2_QWEN": {
        "hypothesis_tested": (
            "cupy-mempool-vs-PyTorch VRAM conflict: the bridge log shows mempool limit 80%; the develop loop + "
            "firewall composer bridges leave ~12.4 GB of CACHED cupy-pool blocks held (run.log constant 12.4GB)."),
        "finding": (
            "HONEST: the bare conflict did NOT reproduce as a deterministic OOM -- even at the FAITHFUL 12.2 GB "
            "pre-Qwen state (real MultiTurnAgent WM bridge + composer recall, matching run.log), torch+Qwen LOAD "
            "and RENDER fine on the 24 GB card (as_is arm OK, free 10.98 GB after Qwen). cupy set_limit is a CAP, "
            "not a reservation. The original hard-kill (no traceback, ~31 min, GPU NOT OOM at 12.4/24) was most "
            "likely an INTERMITTENT torch-CUDA-context-init x cupy-12.4GB-pool contention ('died under GPU "
            "contention' per the run history) or host-RAM pressure from the long loop -- not a clean reproducible "
            "OOM."),
        "THE_FIX": (
            "Release the cupy pool BEFORE loading the faculty: cp.get_default_memory_pool().free_all_blocks() (+ "
            "pinned pool + synchronize), wired as _free_cupy_pool() called right before the SpikingQwenFaculty "
            "load in run_demo + run_repl. In the faithful repro this drops held VRAM 12.2 GB -> 1.6 GB, so torch "
            "loads into free VRAM by construction -> contention eliminated. Belt-and-suspenders: "
            "GPUConfig.memory_pool_limit_fraction can be lowered (0.5)."),
        "repro_arms": {
            "as_is": {"ok": qwen["summary"]["as_is_ok"],
                      "vram_after_agent_used": qarm("as_is").get("vram_after_agent"),
                      "vram_after_qwen": qarm("as_is").get("vram_after_qwen")},
            "fixed": {"ok": qwen["summary"]["fixed_ok"],
                      "vram_after_freeall": qarm("fixed").get("vram_after_freeall"),
                      "vram_after_qwen": qarm("fixed").get("vram_after_qwen")},
        },
    },

    "END_TO_END_VERIFY_on_saved_codes": {
        "CHAT_READY": verify["CHAT_READY"],
        "agent_recall": verify["agent_recall_acc"],
        "firewall_project": f"{verify['firewall_project_answered']}/{verify['firewall_project_total']}",
        "general_leaks": verify["firewall_general_leaks"],
        "untaught_leaks": verify["firewall_untaught_leaks"],
        "faculty_loaded_no_crash": verify["faculty_loaded"],
        "selfreflect_verified": f"{verify['selfreflect_verified']}/{verify['selfreflect_total']}",
        "sample_chat": verify["chat"],
    },

    "recommended_chat_config": {
        "note": (
            "The fixes are in build_qa_agent (decorrelate_codes=True default) + _free_cupy_pool() before the "
            "faculty. The REPL loads the SAVED grounded codes and decorrelates-on-build, so it is chat-ready "
            "immediately on the existing codes -- NO need to re-run the 50-min develop loop. A FUTURE develop run "
            "also benefits (the firewall/Q&A stages decorrelate the codes they were handed)."),
        "repl_command_chat_ready": ("SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_demo --repl "
                                    "--load research/findings/raw/_self_knowledge_grounded_codes.json --seed 42"),
        "repl_command_brain_triples_only_no_GPU": ("SIM_BACKEND=numpy python -u -m research.runners._self_knowledge_demo "
                                                   "--repl --load research/findings/raw/_self_knowledge_grounded_codes.json "
                                                   "--seed 42 --no-faculty"),
        "full_demo_with_fixes": "SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_demo --n-days 4 --seed 42",
    },
    "sim_edits": "NONE (runner-level only: research/runners/_self_knowledge_demo.py).",
    "probe_artifacts": [
        "research/runners/_self_knowledge_recall_probe.py", "research/findings/raw/_self_knowledge_recall_probe.json",
        "research/runners/_self_knowledge_chat_e2e_probe.py", "research/findings/raw/_self_knowledge_chat_e2e_probe.json",
        "research/runners/_self_knowledge_qwen_realrepro.py", "research/findings/raw/_self_knowledge_qwen_realrepro.json",
        "research/runners/_self_knowledge_qwen_fix_probe.py", "research/findings/raw/_self_knowledge_qwen_fix_probe.json",
        "research/runners/_self_knowledge_chat_verify.py", "research/findings/raw/_self_knowledge_chat_verify.json",
    ],
}

dest = os.path.join(RAW, "_self_knowledge_chat_fix.json")
with open(dest, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=str)
print("WROTE", dest)
print("CHAT_READY:", out["END_TO_END_VERIFY_on_saved_codes"]["CHAT_READY"])
print("recall grounded->decorr @D128:",
      out["PART1_RECALL"]["evidence_recall_vs_D_and_codes"]["grounded"]["128"], "->",
      out["PART1_RECALL"]["evidence_recall_vs_D_and_codes"]["grounded_decorr_THE_FIX"]["128"])
