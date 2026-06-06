"""Watch the on-bridge spiking-realization subagent. Fire (exit) the moment EITHER:
  (a) its GO/BOUNDARY finding doc appears (it concluded), OR
  (b) any of its result JSONs shows SPIKING composition ABOVE raw (26/39) — i.e. the whitening starts working.
Polls every 3 min; 3 h deadline. One completion notification to the main session with the trigger details."""
import glob
import json
import os
import time

RAW = 26
DEADLINE = time.time() + 3 * 3600

while time.time() < DEADLINE:
    found = glob.glob("research/findings/2026-06-06-option1-onbridge-spiking-realization-*.md")
    if found:
        print(f"TRIGGER finding-written: {[os.path.basename(f) for f in found]}", flush=True)
        break
    best, bestf, bestseed = 0, None, None
    for f in glob.glob("research/findings/raw/_onbridge*.json"):
        try:
            d = json.load(open(f))
            sp = d.get("SPIKING", {})
            if isinstance(sp, dict):
                for s, v in sp.items():
                    c = v.get("compose", [0])[0] if isinstance(v, dict) else 0
                    if c > best:
                        best, bestf, bestseed = c, os.path.basename(f), s
        except Exception:
            pass
    if best > RAW:
        print(f"TRIGGER above-raw: SPIKING {best}/39 (> raw {RAW}) in {bestf} seed {bestseed}", flush=True)
        break
    time.sleep(180)
else:
    print("watcher TIMEOUT (3h): no above-raw result and no finding yet — subagent still tuning", flush=True)
