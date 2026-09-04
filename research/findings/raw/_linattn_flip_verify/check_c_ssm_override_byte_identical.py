"""Safety check (c) for the 2026-09-04 linattn production-default flip: an EXPLICIT
`BRAIN_WKV_MOUTH_RECURRENCE=ssm` override (with no other BRAIN_WKV_MOUTH_* knob set) must still resolve the
OLD ssm/word/vocab config and produce a BYTE-IDENTICAL live brain_chat reply to before the flip existed. Run
once BEFORE the flip (git stash webapp/wkv_mouth_generator.py) and once AFTER (git stash pop), diff the two
JSON outputs byte-for-byte.
"""
import json
import os
import sys

sys.path.insert(0, ".")
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("BRAIN_WKV_MOUTH_CKPT", "BRAIN_WKV_MOUTH_TOKENIZER", "BRAIN_WKV_MOUTH_SCOPE", "BRAIN_AFFECT_LESION"):
    os.environ.pop(k, None)
os.environ["BRAIN_OPEN_ENDED"] = "1"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"          # the EXPLICIT override under test
os.environ["BRAIN_OPEN_ENDED_NP_ENTAILMENT"] = "0"
os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "0"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"] = "0"
os.environ["BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"] = "0"

import webapp.server as S  # noqa: E402
from webapp import wkv_mouth_generator as wmg  # noqa: E402

resp = S.brain_chat(S.BrainChatRequest(session="linattn_flip_check_c", message="Tell me a short story about a dog",
                                        brain="tiny-demo", reset=True, rich=True, renderer="stub"))
body = json.loads(bytes(resp.body))

out = {
    "recurrence_mode": wmg.recurrence_mode(),
    "tokenizer_mode": wmg.tokenizer_mode(),
    "scope_mode": wmg.scope_mode(),
    "ckpt_path_seed42": wmg._ckpt_path(42),
    "response_body": body,
}
tag = sys.argv[1] if len(sys.argv) > 1 else "unknown"
out_path = f"research/findings/raw/_linattn_flip_verify/check_c_{tag}.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print("recurrence_mode:", wmg.recurrence_mode(), "ckpt:", wmg._ckpt_path(42))
print("open_ended.raw:", (body.get("open_ended") or {}).get("raw"))
print("wrote", out_path)
