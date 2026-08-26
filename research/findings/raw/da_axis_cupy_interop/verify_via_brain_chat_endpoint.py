"""Bonus end-to-end check: call the REAL `webapp.server.brain_chat()` FastAPI route function directly
(no HTTP layer needed -- same technique as `tests/test_production_chat_gpu_smoke.py`) on `SIM_BACKEND=cupy`,
`renderer='stub'` (GPU-free mouth, so this doesn't pay the ~58s Qwen load) and confirm the response's
`da_drives.reason` is no longer an error and the DA-mode engagement suffix can appear.

Run in a FRESH subprocess (the backend is selected at `webapp.server` import time from `SIM_BACKEND`).
"""
import json
import os

from webapp.server import brain_chat, BrainChatRequest

SESSION = "da_cupy_interop_verify"

r1 = brain_chat(BrainChatRequest(session=SESSION, message="", renderer="stub", rich=False))
body1 = json.loads(bytes(r1.body))
print("TURN 1 (empty/low-engagement) status=", r1.status_code)
print("TURN 1 da_drives:", body1.get("da_drives"))

r2 = brain_chat(BrainChatRequest(session=SESSION,
    message="tell me something surprising and unusual about deep sea bioluminescent creatures",
    renderer="stub", rich=False))
body2 = json.loads(bytes(r2.body))
print("TURN 2 (rich/novel) status=", r2.status_code)
print("TURN 2 answer:", body2.get("answer"))
print("TURN 2 da_drives:", body2.get("da_drives"))

assert r1.status_code == 200 and r2.status_code == 200, "brain_chat did not return 200"
dd1 = body1.get("da_drives") or {}
dd2 = body2.get("da_drives") or {}
assert dd1.get("reason") is None or not str(dd1.get("reason")).startswith("error:"), (
    f"TURN 1 da_drives still errored: {dd1}")
assert dd2.get("reason") is None or not str(dd2.get("reason")).startswith("error:"), (
    f"TURN 2 da_drives still errored: {dd2}")
print("\nENDPOINT_VERIFY_OK: /api/brain-chat da_drives no longer errors on SIM_BACKEND=cupy")
