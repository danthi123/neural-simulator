"""Parse the 787-concept scale develop-loop log into the per-day battery curve + evaluate the
FROZEN pre-registration gate (2026-07-17-stream-cortex-787-concept-scale-test-PREREGISTRATION.md).

Reproducible instrument so the RESULT doc's numbers + verdict are extracted, not eyeballed.

    python -m research.runners._scale787_analyze <log_path>

The gate (frozen before results):
  PRIMARY (emergence, D-independent): corr(M,C) >= 0.70 at the FINAL vocab, no collapse across curve
  SECONDARY: retain >= 0.80 across the curve; moat_fa stays O(single digits), not monotone-blowup
  CHARACTERIZATION (not pass/fail): recall(vocab) = the develop_D=128 FHRR capacity curve
"""

import re
import sys

DAY_RE = re.compile(
    r"\[day (\d+)\]\s+vocab=(\d+)\s+heard=(\d+)\s+facts=\s*(\d+)\s+"
    r"recall=([\d.]+|-)\s+heldout=([\d.]+|-)\s+retain=([\d.]+|-)\s+chain=([\d.]+|-)\s+"
    r"moat_fa=(\d+)\s+corr\(M,C\)=([+\-][\d.]+)"
)


def _f(x):
    return None if x == "-" else float(x)


def parse(log_text):
    rows = []
    for m in DAY_RE.finditer(log_text):
        rows.append({
            "day": int(m.group(1)), "vocab": int(m.group(2)), "heard": int(m.group(3)),
            "facts": int(m.group(4)), "recall": _f(m.group(5)), "heldout": _f(m.group(6)),
            "retain": _f(m.group(7)), "chain": _f(m.group(8)), "moat_fa": int(m.group(9)),
            "corr": float(m.group(10)),
        })
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: _scale787_analyze <log_path>")
        return 2
    rows = parse(open(path, encoding="utf-8", errors="ignore").read())
    if not rows:
        print("no [day ...] lines parsed (run not started, or format changed)")
        return 1

    print(f"{'day':>3} {'vocab':>5} {'facts':>5} {'recall':>6} {'retain':>6} {'moat':>4} {'corr':>6}")
    for r in rows:
        rc = "  -  " if r["recall"] is None else f"{r['recall']:.2f}"
        rt = "  -  " if r["retain"] is None else f"{r['retain']:.2f}"
        print(f"{r['day']:>3} {r['vocab']:>5} {r['facts']:>5} {rc:>6} {rt:>6} "
              f"{r['moat_fa']:>4} {r['corr']:+.2f}")

    last = rows[-1]
    corrs = [r["corr"] for r in rows]
    retains = [r["retain"] for r in rows if r["retain"] is not None]
    moats = [r["moat_fa"] for r in rows]
    corr_min = min(corrs)
    retain_min = min(retains) if retains else None

    print("\n=== FROZEN GATE (do not move the bar) ===")
    print(f"reached: day {last['day']}, vocab {last['vocab']}, {last['facts']} facts")
    # PRIMARY
    prim = last["corr"] >= 0.70 and corr_min >= 0.70
    print(f"PRIMARY  corr(M,C) final={last['corr']:+.2f}  min-across={corr_min:+.2f}  "
          f">=0.70 & no-collapse -> {'PASS' if prim else 'FAIL'}")
    # SECONDARY retain
    if retain_min is not None:
        sec_ret = retain_min >= 0.80
        print(f"SECONDARY retain min-across={retain_min:.2f}  >=0.80 -> {'PASS' if sec_ret else 'FAIL'}")
    # SECONDARY moat
    moat_max = max(moats)
    sec_moat = moat_max <= 9  # O(single digits); the frozen interpretation is code-fidelity cost, not mechanism fail
    print(f"SECONDARY moat_fa max-across={moat_max}  O(single digits) -> {'PASS' if sec_moat else 'REVIEW'}")
    # CHARACTERIZATION
    recalls = [(r["vocab"], r["recall"]) for r in rows if r["recall"] is not None]
    if recalls:
        print(f"CHAR      recall(vocab): {recalls[0][1]:.2f}@{recalls[0][0]} -> "
              f"{recalls[-1][1]:.2f}@{recalls[-1][0]}  (develop_D=128 capacity curve; lever=bigger develop_D)")

    verdict = ("GO — emergent structure HOLDS at scale (stream cortex keeps learning; recall is a "
               "separate D-tunable knob)") if prim else \
              ("BOUNDARY — the stream-cortex LEARNING has a scale limit (corr degraded); research-gate WHY")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
