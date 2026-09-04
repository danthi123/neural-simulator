"""Safety check (a) for the 2026-09-04 linattn production-default flip: with `BRAIN_OPEN_ENDED` unset (the
existing, unchanged default), a brain_chat turn must be BYTE-IDENTICAL to before the flip -- the mouth module
whose defaults just changed must never even be imported on this path. Run once BEFORE the flip (git stash
webapp/wkv_mouth_generator.py) and once AFTER (git stash pop), diff the two JSON outputs byte-for-byte.

Also asserts `webapp.wkv_mouth_generator` never enters `sys.modules` -- the direct mechanism proof (not just an
outcome byte-diff): the module whose defaults changed was structurally never touched, not merely "touched but
had no visible effect."
"""
import json
import os
import sys

sys.path.insert(0, ".")
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("BRAIN_OPEN_ENDED", "BRAIN_OPEN_ENDED_WKV_MOUTH", "BRAIN_WKV_MOUTH_RECURRENCE", "BRAIN_WKV_MOUTH_CKPT",
          "BRAIN_WKV_MOUTH_TOKENIZER", "BRAIN_WKV_MOUTH_SCOPE", "BRAIN_AFFECT_LESION"):
    os.environ.pop(k, None)   # BRAIN_OPEN_ENDED left UNSET -- this is the "default off" path under test

import webapp.server as S  # noqa: E402

resp = S.brain_chat(S.BrainChatRequest(session="linattn_flip_check_a", message="Hello, how are you today?",
                                        brain="tiny-demo", reset=True, rich=True, renderer="stub"))
body = json.loads(bytes(resp.body))

mouth_imported = "webapp.wkv_mouth_generator" in sys.modules

out = {
    "mouth_module_imported": mouth_imported,
    "response_body": body,
}
tag = sys.argv[1] if len(sys.argv) > 1 else "unknown"
out_path = f"research/findings/raw/_linattn_flip_verify/check_a_{tag}.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print("mouth_module_imported:", mouth_imported)
print("wrote", out_path)
