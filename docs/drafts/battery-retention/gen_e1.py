"""Print the E1 command lines (markdown) to stdout, generated from the registered JOBS line 46. Run from the repo
root: python3 docs/drafts/battery-retention/gen_e1.py [A_REV B_REV] > e1_lines.md
RULING battery-retention-e1 (Claude 2026-10-13): run with A_REV = B_REV = the main merge that contains the release code,
so A and B differ ONLY in LB_BATTERY_RELEASE_SESSIONS (no cross-revision code drift)."""
import sys
src = open("research/coordination/2026-10-12-validation-battery-JOBS.md").read().splitlines()
tmpl = src[45]   # line 46: seed 42 OFF, the registered line verbatim
assert tmpl.startswith("python3 -c") and "BRAIN_CHAT_SEED=42 BRAIN_IDLE_REPLAY=0 " in tmpl
PIN = "f24963bfb05ca535f10f70bf4380fe4b2f68a9ea"  # the registered JOBS line 46 revision (template anchor)
A = sys.argv[1] if len(sys.argv) > 2 else PIN
B = sys.argv[2] if len(sys.argv) > 2 else "97a596734"
OUT = "research/findings/raw/_regression_battery/battery_retention_equiv"
arms = {"off": "BRAIN_IDLE_REPLAY=0 ", "on": "", "lesion": "BRAIN_IDLE_REPLAY_COMMIT_LESION=1 "}


def line(rev, seed, arm, release, tag):
    s = tmpl.replace(PIN, rev).replace("research/findings/raw/_regression_battery/c5val_1012", OUT)
    s = s.replace("BRAIN_CHAT_SEED=42 BRAIN_IDLE_REPLAY=0 ",
                  "BRAIN_CHAT_SEED=%d %s%s" % (seed, arms[arm], "LB_BATTERY_RELEASE_SESSIONS=1 " if release else ""))
    s = s.replace("c5_s42_off", tag)
    assert "c5_s42" not in s and "c5val_1012" not in s and s.count(tag) == 3
    return s


out = []
n = 0
for seed in (7, 8):
    for arm in ("off", "on", "lesion"):
        for rev, name, rel in ((A, "A", False), (B, "B", True)):
            n += 1
            tag = "%s_s%d_%s" % (name, seed, arm)
            out.append("E1.%d `%s` (instrument %s, seed %d, arm %s):\n\n```\n%s\n```\n"
                       % (n, tag, name, seed, arm, line(rev, seed, arm, rel, tag)))
n += 1
out.append("E1.%d `Bnorel_s7_on` (instrument B with release UNSET, seed 7, arm on):\n\n```\n%s\n```\n"
           % (n, line(B, 7, "on", False, "Bnorel_s7_on")))
print("\n".join(out))
