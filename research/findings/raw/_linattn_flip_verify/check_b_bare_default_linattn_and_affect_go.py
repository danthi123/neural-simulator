"""Safety check (b) + a phase6-equivalent no-regression check (e) for the 2026-09-04 linattn production-default
flip: with `BRAIN_OPEN_ENDED=1` and NO OTHER BRAIN_WKV_MOUTH_* env var set at all (the bare new defaults), does
`answer_turn` genuinely route to the linattn mouth (loads the linattn ckpt, bpe tokenizer, broad scope) AND
reproduce the SAME affect-load-bearing GO verdict `research/findings/2026-09-04-linattn-affect-coupling-
sharpness-aware-GO.md` measured with recurrence/ckpt/tokenizer/scope all pinned EXPLICITLY
(`_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation_rerun.py`)?

This is the DIRECT substitute for the task brief's named `phase6_linattn_clean_isolation.py` re-run: that exact
script/its JSON artifact do not exist in this worktree (verified: absent from `git log --all`, absent from
`main`, absent on disk under `research/findings/raw/_affect_wkv_mouth_verify/` -- see the shipped finding's own
honest note). This script is scenario-identical to the EXISTING, committed `phase4_..._rerun.py` (same priming
message, same known/unknown topics, same Q1/Q2 structure, same CPU-forced backend) with exactly ONE change:
every `BRAIN_WKV_MOUTH_*` env var is left UNSET instead of pinned, so what is actually under test is "does the
bare post-flip DEFAULT reproduce the pinned-explicit GO" -- the flip's own central claim.

CORRECTION (2026-09-05, added after this file's own NO-GO result was recovered from an abandoned worktree and
found never to have been committed -- see `research/findings/2026-09-05-linattn-check-b-dropped-nogo-recovered-
and-reproduced.md`): running THIS file (as committed, smoke-turn-free) still returns `BARE_DEFAULT_FLIP_CONFIRM_
GO: false` -- byte-identical raw text and the identical determinism-control failure as the smoke-turn-containing
revision the "self-correcting" note below blames. The smoke turn was NEVER the cause; removing it changed
nothing. The real confound is the one `phase6_linattn_clean_isolation.py`'s own docstring names for the ORIGINAL
`phase4` script: this file still runs `prime -> turn1(lesion0) -> turn2(lesion1) -> turn3(lesion0-repeat)` as ONE
shared session in ONE process, so mood-EMA/habituation state evolves across turns exactly like phase4 did --
`phase6`'s fresh-session-per-arm fix (and this project's own stronger fresh-SUBPROCESS-per-arm precedent,
`research/runners/_wkv_mouth_affect_neural_verify.py::_run_arm`) was never back-ported here. See
`check_d_bare_default_fresh_subprocess_clean_verify.py` (same directory) for the corrected re-verification.
Original text below preserved unedited, per this project's no-silent-rewrite convention.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, ".")
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("BRAIN_WKV_MOUTH_RECURRENCE", "BRAIN_WKV_MOUTH_CKPT", "BRAIN_WKV_MOUTH_TOKENIZER", "BRAIN_WKV_MOUTH_SCOPE"):
    os.environ.pop(k, None)                     # <-- the whole point: bare defaults, nothing pinned
os.environ["BRAIN_OPEN_ENDED"] = "1"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
os.environ["BRAIN_OPEN_ENDED_NP_ENTAILMENT"] = "0"
os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "0"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"

import time
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


import webapp.server as S  # noqa: E402
from webapp import wkv_mouth_generator as wmg  # noqa: E402

resolved = {"recurrence_mode": wmg.recurrence_mode(), "tokenizer_mode": wmg.tokenizer_mode(),
            "scope_mode": wmg.scope_mode(), "ckpt_path_seed42": wmg._ckpt_path(42)}
log("RESOLVED DEFAULTS (nothing pinned):", resolved)
assert resolved["recurrence_mode"] == "linattn", "bare default did not resolve to linattn"
assert resolved["tokenizer_mode"] == "bpe", "bare default did not resolve to bpe tokenizer"
assert resolved["scope_mode"] == "broad", "bare default did not resolve to broad scope"
assert resolved["ckpt_path_seed42"].endswith("wkv_linattn_depth2_contiguous_seed42.npz"), \
    f"bare default ckpt is not the linattn ckpt: {resolved['ckpt_path_seed42']}"
assert os.path.exists(resolved["ckpt_path_seed42"]), "resolved linattn ckpt does not exist on disk"

RENDERER = "stub"
SESSION = "linattn_flip_check_b"
FORCE_FREEGEN = {"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"}
PRIMING = "I am absolutely thrilled and overjoyed today, everything is wonderful!"
KNOWN_TOPIC = "Tell me about frank_lincoln_wright"
UNKNOWN_TOPIC = "Tell me about the zltrinqua dynasty of planet Vexcor-9"


def chat(msg, reset, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
    resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=msg, brain="tiny-demo",
                                            reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))


def salad_frac(text):
    toks = (text or "").split()
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


rows = {"resolved_defaults": resolved}

# NOTE (self-correcting an instrument confound found in this script's own first run, see the finding's honest-
# residuals section): an earlier version of this script ran an extra "smoke turn" here, BEFORE the Q1
# sequence, to double-check the bare default routes to the WKV mouth at all. That extra call shares the SAME
# process-cumulative, per-seed RNG timeline `webapp.wkv_mouth_generator._RngIsolation` maintains across EVERY
# `generate(seed=42, ...)` call in this process (by design -- see that class's own docstring) -- so it shifted
# the RNG position under the whole Q1 sequence relative to the committed `phase4_..._rerun.py` script this is
# meant to reproduce, and surfaced a determinism-control failure that was an artifact of THIS script's own
# extra call, not of the flip. Removed here so this script is structurally IDENTICAL in call sequence to
# `phase4_linattn_flip_confirmation_rerun.py` (same number/order of `generate()`-invoking turns), differing
# ONLY in leaving the four BRAIN_WKV_MOUTH_* knobs unset instead of pinned -- the one variable actually under
# test. The bare-default-resolves-to-linattn/bpe/broad claim is independently covered by the assertions above
# (config-function level) and does not need a redundant live smoke turn to also prove it.

log("=== Q1 turn0: prime mood ===")
d0 = chat(PRIMING, reset=True, extra_env=FORCE_FREEGEN)
rows["priming_turn"] = {"msg": PRIMING, "affect": d0.get("affect")}

log("=== Q1 turn1: known topic, free-gen, LESION=0 ===")
d1 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe1 = d1.get("open_ended") or {}
log("gen:", oe1.get("generator"), "wkv_used:", oe1.get("wkv_mouth_used"), "raw1:", oe1.get("raw"))

log("=== Q1 turn2: same topic, free-gen, LESION=1 ===")
d2 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "1"})
oe2 = d2.get("open_ended") or {}
log("gen:", oe2.get("generator"), "raw2:", oe2.get("raw"))

log("=== Q1 turn3: same topic, free-gen, LESION=0 again (determinism control) ===")
d3 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe3 = d3.get("open_ended") or {}
log("raw3:", oe3.get("raw"))

raw_diff_lesion = oe1.get("raw") != oe2.get("raw")
raw_repro_lesion0 = oe1.get("raw") == oe3.get("raw")
sfrac_l0 = salad_frac(oe1.get("raw"))
log("Q1 affect load-bearing (raw differs lesion0 vs lesion1):", raw_diff_lesion)
log("Q1 determinism (raw same lesion0 vs lesion0-repeat):", raw_repro_lesion0)
log("Q1 lesion0 salad_frac:", sfrac_l0)

rows["Q1_affect_loadbearing"] = {
    "topic": KNOWN_TOPIC,
    "wkv_used": {"l0": oe1.get("wkv_mouth_used"), "l1": oe2.get("wkv_mouth_used")},
    "generator": {"l0": oe1.get("generator"), "l1": oe2.get("generator")},
    "raw": {"l0": oe1.get("raw"), "l1": oe2.get("raw"), "l0_repeat": oe3.get("raw")},
    "raw_differs_lesion0_vs_lesion1": raw_diff_lesion,
    "raw_reproduces_lesion0_vs_lesion0_repeat": raw_repro_lesion0,
    "lesion0_salad_frac": sfrac_l0,
    "lesion0_fluent_not_salad": bool(sfrac_l0 < 0.3),
    "PASS": bool(raw_diff_lesion and raw_repro_lesion0),
}

log("=== Q2 turn0: re-prime mood ===")
chat(PRIMING, reset=True)

log("=== Q2 turn1: UNKNOWN topic, moat ON, LESION=0 ===")
u1 = (chat(UNKNOWN_TOPIC, reset=False, extra_env={"BRAIN_AFFECT_LESION": "0"}).get("open_ended") or {})
log("known:", u1.get("known"), "abstained:", u1.get("abstained"), "raw:", u1.get("raw"))

log("=== Q2 turn2: UNKNOWN topic, moat ON, LESION=1 ===")
u2 = (chat(UNKNOWN_TOPIC, reset=False, extra_env={"BRAIN_AFFECT_LESION": "1"}).get("open_ended") or {})
log("known:", u2.get("known"), "abstained:", u2.get("abstained"), "raw:", u2.get("raw"))

q2_pass = (not u1.get("known")) and (not u2.get("known"))
log("Q2 moat holds (unknown topic not claimed known, both arms):", q2_pass)

rows["Q2_moat_with_affect"] = {
    "topic": UNKNOWN_TOPIC,
    "lesion0_affect_active": {"known": u1.get("known"), "abstained": u1.get("abstained"), "raw": u1.get("raw")},
    "lesion1_affect_off": {"known": u2.get("known"), "abstained": u2.get("abstained"), "raw": u2.get("raw")},
    "PASS": bool(q2_pass),
}

os.environ["BRAIN_AFFECT_LESION"] = "0"
verdict = {
    "bare_default_resolved_linattn_bpe_broad": all([
        resolved["recurrence_mode"] == "linattn", resolved["tokenizer_mode"] == "bpe",
        resolved["scope_mode"] == "broad"]),
    "q1_turn1_used_wkv_mouth": bool(oe1.get("wkv_mouth_used")),
    "Q1_affect_loadbearing_PASS": rows["Q1_affect_loadbearing"]["PASS"],
    "Q1_lesion0_fluent_not_salad": rows["Q1_affect_loadbearing"]["lesion0_fluent_not_salad"],
    "Q2_moat_with_affect_PASS": rows["Q2_moat_with_affect"]["PASS"],
}
verdict["BARE_DEFAULT_FLIP_CONFIRM_GO"] = bool(
    verdict["bare_default_resolved_linattn_bpe_broad"]
    and verdict["Q1_affect_loadbearing_PASS"]
    and verdict["Q1_lesion0_fluent_not_salad"]
    and verdict["Q2_moat_with_affect_PASS"])
log("VERDICT:", verdict)

out_path = "research/findings/raw/_linattn_flip_verify/check_b_bare_default.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "linattn_flip_verify check_b (bare-default coherence + phase6-equivalent affect GO, "
                         "hand-authored, scenario-identical to the committed phase4_..._rerun.py with all "
                         "BRAIN_WKV_MOUTH_* env vars left UNSET)",
               "seed": 42, "backend": os.environ.get("SIM_BACKEND"), "verdict": verdict, "rows": rows,
               "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
