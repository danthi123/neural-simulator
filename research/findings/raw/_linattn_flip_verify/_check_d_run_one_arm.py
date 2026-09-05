"""Child process for `check_d_bare_default_fresh_subprocess_clean_verify.py` -- ONE arm = [prime
(reset=True) -> topic(reset=False)] through the REAL `webapp.server.brain_chat`, in a BRAND NEW
interpreter. See check_d's own module docstring for why fresh-subprocess-per-arm (not just fresh-session-
per-arm, which is all `phase6_linattn_clean_isolation.py` did) is required: `webapp/wkv_mouth_generator.py`'s
`_RngIsolation` keeps a PRIVATE, per-seed, CONTINUING RNG timeline across every `generate()` call in a
process regardless of session id, and `_get_readout`/`_affect_bias_ids` cache per-seed state at module level
-- both persist across a fresh session in the SAME process, only a brand new process actually zeroes them.

Every BRAIN_*/SIM_BACKEND/CUDA_VISIBLE_DEVICES env var this arm needs is set by the PARENT (check_d) before
`subprocess.run` launches this file -- this file only reads three carrier vars (CHECK_D_REPO_ROOT/_SESSION/
_PRIME/_TOPIC) that are not real product config, just this harness's own argument-passing mechanism (env,
not argv, to avoid shell-quoting the free-text prompt strings).

Prints exactly one line "RESULT_JSON:{...}" to stdout; writes nothing to disk itself (the parent aggregates
and writes the combined artifact).
"""
import json
import os
import sys

_ROOT = os.environ["CHECK_D_REPO_ROOT"]
sys.path.insert(0, _ROOT)

if os.environ.get("CHECK_D_MODE") == "resolve":
    # Cheap config-resolution-only gate (no brain build): what do the bare, unset BRAIN_WKV_MOUTH_* knobs
    # resolve to, with only BRAIN_OPEN_ENDED=1 forced on top? Mirrors check_b's own top-of-script asserts.
    from webapp import wkv_mouth_generator as wmg  # noqa: E402
    ckpt = wmg._ckpt_path(42)
    print("RESULT_JSON:" + json.dumps({
        "recurrence_mode": wmg.recurrence_mode(),
        "tokenizer_mode": wmg.tokenizer_mode(),
        "scope_mode": wmg.scope_mode(),
        "ckpt_path_seed42": ckpt,
        "ckpt_exists": os.path.exists(ckpt),
    }))
    sys.exit(0)

import webapp.server as S  # noqa: E402

SESSION = os.environ["CHECK_D_SESSION"]
PRIME = os.environ["CHECK_D_PRIME"]
TOPIC = os.environ["CHECK_D_TOPIC"]
RENDERER = "stub"

S.brain_chat(S.BrainChatRequest(session=SESSION, message=PRIME, brain="tiny-demo",
                                 reset=True, rich=True, renderer=RENDERER))
resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=TOPIC, brain="tiny-demo",
                                        reset=False, rich=True, renderer=RENDERER))
body = json.loads(bytes(resp.body))
oe = body.get("open_ended") or {}
print("RESULT_JSON:" + json.dumps({
    "raw": oe.get("raw"),
    "generator": oe.get("generator"),
    "wkv_mouth_used": oe.get("wkv_mouth_used"),
    "known": oe.get("known"),
    "abstained": oe.get("abstained"),
}))
