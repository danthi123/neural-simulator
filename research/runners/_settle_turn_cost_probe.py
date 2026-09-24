"""Wall-clock cost per chat turn of BRAIN_AFFECT_MARKER_SETTLE (2026-09-24).

SETTLE (research/runners/_affect_marker_wta_derisk.py) lengthens the affect-marker circuit's deliberation window
(60 -> 500 ms simulated) and inter-turn rest (40 -> 1000 ms simulated). Before SETTLE joins a default-on flip batch its
per-turn cost has to be measured against the consumer-hardware reference. This probe times exactly what the chat path
calls per turn (webapp/affect_drives_chat.py: `get_reader(seed).select_valence(mood)` then `select_arousal(arousal)`)
with the flag off and on, on the requested backend. It measures time only; it changes nothing.

    SIM_BACKEND=cupy python -m research.runners._settle_turn_cost_probe --mode on --out research/findings/raw/_settle_cost/cupy_on.json
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["on", "off"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["BRAIN_AFFECT_MARKER_SETTLE"] = "1" if a.mode == "on" else "0"
    backend = os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners import _affect_marker_wta_derisk as m
    t0 = time.perf_counter()
    reader = m.get_reader(seed=a.seed)
    build_s = time.perf_counter() - t0
    moods = [0.068, 0.02, 0.12, -0.05, 0.068, 0.2, -0.12, 0.068]   # boundary, centres, negatives, repeats
    turn_s = []
    for mood in moods:
        t = time.perf_counter()
        reader.select_valence(mood)
        reader.select_arousal(0.075)
        turn_s.append(time.perf_counter() - t)
    steady = sorted(turn_s[1:])                                     # the first read also pays lazy construction
    res = {"mode": a.mode, "backend": backend, "seed": a.seed, "settle_enabled": bool(m.settle_enabled()),
           "deliberation_ms": m.DELIBERATION_MS, "interturn_rest_ms": m.INTERTURN_REST_MS,
           "build_s": build_s, "first_turn_s": turn_s[0], "turn_s": turn_s,
           "steady_median_turn_s": steady[len(steady) // 2], "steady_max_turn_s": steady[-1]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("mode", "backend", "settle_enabled", "first_turn_s",
                                          "steady_median_turn_s", "steady_max_turn_s")}))


if __name__ == "__main__":
    sys.exit(main())
