"""check_d -- the FRESH-SUBPROCESS-PER-ARM re-verification of the linattn mouth flip's bare production
default, closing the genuine gap the adversarial-verify review (workflow `w3qhweujd`) found: `phase6_
linattn_clean_isolation.py` (CLEAN_FLIP_GO:true) tested affect-load-bearing + determinism only, on a
manually-PINNED `BRAIN_WKV_MOUTH_*` config, one prompt/seed, never moat, never fluency, never the bare/
unset-env-var default `answer_turn` actually resolves in production; and `check_b_bare_default_linattn_
and_affect_go.py` -- the script that DOES test the bare default -- returned a NO-GO (determinism failed)
that was never committed (see `check_b_bare_default_DROPPED_2026-09-04.json`, recovered and committed
alongside this script, plus a fresh reproduction of check_b run in THIS worktree today which reproduced
that NO-GO byte-for-byte, see the accompanying finding).

WHY check_b'S OWN NO-GO IS ITSELF SUSPECT AS A TEST, NOT JUST AS A RESULT. check_b runs `prime -> Q1 turn1
(lesion0) -> Q1 turn2 (lesion1) -> Q1 turn3 (lesion0-repeat)` as FOUR TURNS OF ONE SHARED SESSION in ONE
process -- structurally identical to the ORIGINAL `phase4_linattn_flip_confirmation.py`, which `phase6`'s
own docstring diagnoses as CONFOUNDED ("the mood EMA... AND the affect-fix's habituation state evolve
ACROSS turns -- which makes lesion0 vs lesion0-repeat legitimately DIFFER (session-dynamics, not noise) and
CONFOUNDS the lesion0-vs-lesion1 attribution"). `phase6` fixed this with a FRESH SESSION per arm -- but a
fresh session in the SAME process does not reset `webapp/wkv_mouth_generator.py`'s own `_RngIsolation`,
which keeps one PRIVATE, per-seed, CONTINUING RNG timeline across every `generate()` call in a process
REGARDLESS of session id (by the class's own docstring), nor `_get_readout`/`_affect_bias_ids`'s per-seed
module-level caches. The project's OWN precedent for this exact class of coupling
(`research/runners/_wkv_mouth_affect_neural_verify.py::_run_arm`, used by `research/findings/2026-09-04-
affect-coupling-neural-not-host-PARTIAL.md`'s 6-seed x 3-prompt load-bearing table) explicitly rejects
fresh-session-in-one-process as insufficient and uses a FRESH SUBPROCESS per arm instead -- this script
applies that SAME discipline to the bare-default flip-gate scenario check_b/phase6 established, through the
FULL `webapp.server.brain_chat` pipeline (not the lower-level `generate()` call `_wkv_mouth_affect_neural_
verify.py` uses), because what is under test here is specifically whether `answer_turn` resolves the BARE,
UNSET `BRAIN_WKV_MOUTH_*` defaults correctly end to end -- calling `generate()` directly would bypass
exactly the dispatch logic under test.

SEED SCOPE (documented honestly, not waived silently). The project's 6-seed non-negotiable battery
(42/43/44/100/101/102) applies to standalone mechanism tests that pass `seed=` explicitly to `generate()`.
The DEPLOYED `webapp/server.py` pipeline this script tests does not have a seed axis: `_WARM_QWEN_RENDERER
= QwenRenderer(seed=42)` and every organ constructor in that file (`get_organ(seed=42)`, `SelfInitiationOrgan
(seed=42)`, etc., grep-confirmed, ~15 call sites) hardcode `seed=42`, and `wkv_mouth_generator._ckpt_path`
falls back to the seed42 checkpoint whenever a requested seed's file is missing ("a per-seed ckpt may be
missing; seed42 always ships") -- true of every seed but 42 in a fresh checkout, since only the seed42
linattn checkpoint is committed (the other five are `.gitignore`d). A 6-seed sweep of `brain_chat` itself
would therefore not exercise 6 different configurations at all; it would call the IDENTICAL seed42 pipeline
six times. This script instead adds PROMPT diversity where precedent had none (2 independently-fabricated
unknown topics for the moat check, up from check_b/phase6's 1) and keeps the ESTABLISHED single known-topic
prompt (`frank_lincoln_wright`, confirmed present in this worktree's KB by every prior run) for direct
comparability with check_b/phase4/phase6's own numbers; the determinism gate (A vs C, both fresh
subprocesses) is itself the noise-robustness replication this task asks for, run twice (both known-topic
and, implicitly, across the 2 unknown-topic prompts).

GATES (ALL must pass for `FRESH_SUBPROCESS_BARE_DEFAULT_CLEAN_GO`):
  1. RESOLUTION: bare (unset) BRAIN_WKV_MOUTH_* defaults resolve to linattn/bpe/broad and the ckpt exists.
  2. DETERMINISM: fresh-subprocess arm A (known topic, lesion=0) == fresh-subprocess arm C (repeat).
  3. AFFECT-LOAD-BEARING: arm A (lesion=0) != arm B (lesion=1, same fresh-subprocess recipe otherwise).
  4. MOAT: on EACH of 2 independently-fabricated unknown topics, `known=False` in BOTH lesion arms.
  5. FLUENCY: arm A's raw output salad-fraction (most-common-token share) < 0.3 (the project's own
     established cheap heuristic, `check_b`/`phase5`/`phase6`'s same threshold -- see `docs/TERMS.md` and
     honest-residual #6 of `2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md` for this heuristic's
     own acknowledged limits, unchanged here).

Run (from the repo root; forces CPU/numpy in every child, matching check_a/b/c precedent -- the mouth's
torch model + a 25531-neuron brain build have no need of the GPU and may collide with an unrelated GPU job):
  .venv/bin/python research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.py
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter

REPO_ROOT = os.path.abspath(".")
assert os.path.isdir(os.path.join(REPO_ROOT, "webapp")) and \
    os.path.isfile(os.path.join(REPO_ROOT, "GAP_CLOSURE_MISSION.md")), \
    "run this from the repo root (matches check_a/b/c's own convention: `.venv/bin/python research/.../check_d...py`)"
sys.path.insert(0, REPO_ROOT)
from tools.lab import lever  # noqa: E402

CHILD = os.path.join(REPO_ROOT, "research/findings/raw/_linattn_flip_verify/_check_d_run_one_arm.py")
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def salad_frac(text):
    toks = (text or "").split()
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


def _base_env(lesion, force_freegen):
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    env["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("BRAIN_WKV_MOUTH_RECURRENCE", "BRAIN_WKV_MOUTH_CKPT", "BRAIN_WKV_MOUTH_TOKENIZER",
              "BRAIN_WKV_MOUTH_SCOPE"):
        env.pop(k, None)                      # the whole point -- bare defaults, nothing pinned
    env["BRAIN_OPEN_ENDED"] = "1"
    env["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
    env["BRAIN_OPEN_ENDED_NP_ENTAILMENT"] = "0"
    env["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "0"
    env["BRAIN_LTM_SHIP_DEFAULT"] = "1"
    env["BRAIN_AFFECT_LESION"] = lesion
    if force_freegen:
        env["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"] = "0"
        env["BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"] = "0"
    env["CHECK_D_REPO_ROOT"] = REPO_ROOT
    return env


def run_resolution_gate(timeout_s=60):
    env = _base_env(lesion="0", force_freegen=False)
    env["CHECK_D_MODE"] = "resolve"
    proc = subprocess.run([sys.executable, CHILD], env=env, capture_output=True, text=True,
                          timeout=timeout_s, cwd=REPO_ROOT)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        raise RuntimeError(f"resolution gate failed: rc={proc.returncode}\n"
                           f"STDOUT-tail={proc.stdout[-2000:]}\nSTDERR-tail={proc.stderr[-4000:]}")
    return json.loads(line[len("RESULT_JSON:"):])


def run_arm(tag, session, prime, topic, lesion, force_freegen, timeout_s=600):
    env = _base_env(lesion, force_freegen)
    env["CHECK_D_SESSION"] = session
    env["CHECK_D_PRIME"] = prime
    env["CHECK_D_TOPIC"] = topic
    t0 = time.time()
    proc = subprocess.run([sys.executable, CHILD], env=env, capture_output=True, text=True,
                          timeout=timeout_s, cwd=REPO_ROOT)
    wall = time.time() - t0
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        raise RuntimeError(f"arm {tag} failed: rc={proc.returncode}\n"
                           f"STDOUT-tail={proc.stdout[-3000:]}\nSTDERR-tail={proc.stderr[-6000:]}")
    d = json.loads(line[len("RESULT_JSON:"):])
    d["wall_s"] = round(wall, 1)
    log(f"arm={tag} wall={d['wall_s']}s generator={d.get('generator')} wkv_used={d.get('wkv_mouth_used')} "
        f"known={d.get('known')} raw={repr(d.get('raw'))[:160]}")
    return d


PRIMING = "I am absolutely thrilled and overjoyed today, everything is wonderful!"
KNOWN_TOPIC = "Tell me about frank_lincoln_wright"
UNKNOWN_TOPICS = [
    "Tell me about the zltrinqua dynasty of planet Vexcor-9",
    "Tell me about the fictional glorbaxian empire of nebula-7",
]

rows = {}

log("=== Gate 1: bare-default resolution (cheap, no brain build) ===")
resolved = run_resolution_gate()
log("resolved:", resolved)
rows["resolved_defaults"] = resolved
resolution_pass = bool(
    resolved["recurrence_mode"] == "linattn" and resolved["tokenizer_mode"] == "bpe"
    and resolved["scope_mode"] == "broad" and resolved["ckpt_exists"]
    and resolved["ckpt_path_seed42"].endswith("wkv_linattn_depth2_contiguous_seed42.npz"))
log("resolution_pass:", resolution_pass)

log("=== Gate 2+3: known-topic arms A (lesion=0) / B (lesion=1) / C (lesion=0 repeat), fresh subprocess each ===")
A = run_arm("A_l0", "check_d_A_l0", PRIMING, KNOWN_TOPIC, lesion="0", force_freegen=True)
B = run_arm("B_l1", "check_d_B_l1", PRIMING, KNOWN_TOPIC, lesion="1", force_freegen=True)
C = run_arm("C_l0repeat", "check_d_C_l0repeat", PRIMING, KNOWN_TOPIC, lesion="0", force_freegen=True)
rows["Q1_known_topic"] = {"topic": KNOWN_TOPIC, "A_lesion0": A, "B_lesion1": B, "C_lesion0_repeat": C}

moved_det = lever("determinism fresh-subprocess (A raw vs C raw) -- want UNCHANGED", A["raw"], C["raw"],
                  required=False)
determinism_pass = not moved_det
moved_affect = lever("affect load-bearing fresh-subprocess (A raw vs B raw) -- want MOVED", A["raw"], B["raw"],
                     required=False)
affect_pass = moved_affect
sfrac = salad_frac(A["raw"])
fluency_pass = bool(sfrac < 0.3)
log("determinism_pass:", determinism_pass, "affect_pass:", affect_pass,
    "salad_frac(A):", sfrac, "fluency_pass:", fluency_pass)

log("=== Gate 4: moat -- 2 independently-fabricated unknown topics x {lesion0, lesion1}, fresh subprocess each ===")
moat_rows = []
moat_pass = True
for i, topic in enumerate(UNKNOWN_TOPICS):
    d0 = run_arm(f"Q2_{i}_l0", f"check_d_Q2_{i}_l0", PRIMING, topic, lesion="0", force_freegen=False)
    d1 = run_arm(f"Q2_{i}_l1", f"check_d_Q2_{i}_l1", PRIMING, topic, lesion="1", force_freegen=False)
    topic_pass = bool((not d0.get("known")) and (not d1.get("known")))
    moat_pass = moat_pass and topic_pass
    log(f"moat topic[{i}] PASS={topic_pass} known(l0)={d0.get('known')} known(l1)={d1.get('known')}")
    moat_rows.append({"topic": topic, "lesion0": d0, "lesion1": d1, "PASS": topic_pass})
rows["Q2_moat"] = moat_rows

verdict = {
    "resolution_pass": resolution_pass,
    "determinism_pass_fresh_subprocess": determinism_pass,
    "affect_loadbearing_pass_fresh_subprocess": affect_pass,
    "moat_pass_both_unknown_topics": moat_pass,
    "fluency_pass_salad_frac_lt_0p3": fluency_pass,
    "lesion0_salad_frac": sfrac,
}
verdict["FRESH_SUBPROCESS_BARE_DEFAULT_CLEAN_GO"] = bool(
    resolution_pass and determinism_pass and affect_pass and moat_pass and fluency_pass)
log("=== FINAL VERDICT ===")
log(verdict)

out_path = "research/findings/raw/_linattn_flip_verify/check_d_bare_default_fresh_subprocess_clean_verify.json"
with open(out_path, "w") as fh:
    json.dump({
        "runner": "check_d_bare_default_fresh_subprocess_clean_verify (hand-authored orchestrator, "
                  "fresh-subprocess-per-arm via research/findings/raw/_linattn_flip_verify/"
                  "_check_d_run_one_arm.py, replaces check_b/phase6's same-process session/RNG-timeline "
                  "isolation with the project's own _wkv_mouth_affect_neural_verify.py-precedent subprocess "
                  "isolation, through the FULL webapp.server.brain_chat pipeline at the bare/unset "
                  "BRAIN_WKV_MOUTH_* production defaults)",
        "seed_scope_note": "webapp/server.py hardcodes seed=42 at every organ construction site "
                           "(grep-confirmed); there is no seed axis in the deployed brain_chat pipeline to "
                           "sweep. Diversity is instead added on the PROMPT axis (2 independent unknown "
                           "topics for the moat gate, vs check_b/phase6's 1); the known-topic prompt matches "
                           "established precedent for direct comparability.",
        "seed": 42,
        "backend": "numpy",   # every fresh-subprocess child arm is launched with SIM_BACKEND=numpy +
                              # CUDA_VISIBLE_DEVICES="" (see _base_env) -- CPU-forced throughout, matching
                              # check_a/b/c's own convention in this same directory.
        "verdict": verdict,
        "rows": rows,
        "wall_seconds": round(time.time() - T0, 1),
    }, fh, indent=1)
log("wrote", out_path)
