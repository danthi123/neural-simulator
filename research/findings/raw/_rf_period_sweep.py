"""(c-opt) RF composer resonate-period sweep: find the minimum period (steps per op) that preserves the full
capability matrix -> a direct response-latency win. Shorter period = fewer steps/op but coarser phase resolution.
"""
import time
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer


def c1_check(comp):
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")
    comp.store("river", "look", ("big", "apple"))
    comp.store("dog", "stop", "east", polarity="AFFIRM")
    comp.store("cat", "look", "west", polarity="NEGATE")
    checks = [
        comp.query_agent("go", "north") == "dog",
        comp.query_patient("cat", "run") == "south",
        comp.query_patient("river", "look") == "big apple",
        comp.render_fact("river") == "river look big apple",
        comp.ask_yes_no("dog", "stop", "east") == "yes",
        comp.ask_yes_no("cat", "look", "west") == "no",
        comp.query_agent("go", "south") is None,
        comp.ask_yes_no("dog", "go", "west") == "unknown",
        comp.render_fact("apple") is None,
    ]
    return sum(checks), len(checks)


if __name__ == "__main__":
    for period in [80, 100, 150, 200, 300, 400]:
        ok_all = True
        t0 = time.time()
        for seed in (42, 43, 44):
            comp = RFPhasorComposer(seed=seed, D=96, period=period)
            n, total = c1_check(comp)
            if n != total:
                ok_all = False
                print(f"  period={period} seed={seed}: {n}/{total} FAIL", flush=True)
        dt = time.time() - t0
        print(f"period={period}: {'PASS 3/3' if ok_all else 'FAIL'} ({dt:.1f}s for 3 seeds)", flush=True)
