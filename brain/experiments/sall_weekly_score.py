# Capability: Self and identity
# Record: brain/research/2026-11-15-sall-slept-bar-1.5b.md (stage 2, W1-W4)
"""W1-W4 for the S_all slept-bar weekly. usage: sall_weekly_score.py <dir> <seeds...>
expects <dir>/wks0_s<seed>.md (default S0) and <dir>/wksa_s<seed>.md (--key S_all --key-bar slept)."""
import re, sys
from statistics import mean
from brain.experiments.self_completion_score import parse, strip, ok

ABST = "I don't know about that."


def own_recalled(a):
    m = re.search(r"_\(recalled: (.*?)\)_", a, re.S)
    return bool(m) and m.group(1).strip().startswith("User: What's your name?")


def row(p):
    head, secs = parse(p)
    s1 = secs[[k for k in secs if k.startswith("session 1")][0]]
    intact = secs["intact"]
    get = lambda S, q: next((a for qq, a in S if q in qq), "")
    return dict(key_hdr="key S_all (slept bar)" in head, d1_own=ok(s1[0][1]),
                w1=own_recalled(get(intact, "What's your name")), s2_own=ok(get(intact, "What's your name")),
                s2_own_txt=strip(get(intact, "What's your name")), keyed=sum(ok(a) for _, a in intact),
                friend=ok(get(intact, "remember my name")), abst=sum(ABST in a for _, a in s1 + intact))


if __name__ == "__main__":
    d, seeds = sys.argv[1], sys.argv[2:]
    R = {(arm, s): row(f"{d}/wk{arm}_s{s}.md") for arm in ("s0", "sa") for s in seeds}
    for s in seeds:
        for arm in ("s0", "sa"):
            r = R[arm, s]
            print(f"s{s} {arm} hdr {int(r['key_hdr'])} d1own {int(r['d1_own'])} s2 own-recalled {int(r['w1'])} s2own {int(r['s2_own'])} "
                  f"'{r['s2_own_txt'][:34]}' keyed {r['keyed']} friend {int(r['friend'])} abstain {r['abst']}")
    n = lambda arm, k: sum(R[arm, s][k] for s in seeds)
    mk = lambda arm, k: mean(R[arm, s][k] for s in seeds)
    eng = all(R["sa", s]["key_hdr"] and not R["s0", s]["key_hdr"] for s in seeds)
    w1 = n("sa", "w1") >= 5
    w2 = n("sa", "s2_own") >= 5 and n("sa", "s2_own") >= n("s0", "s2_own")
    w3 = (mk("sa", "keyed") >= mk("s0", "keyed") - 0.5 and n("sa", "d1_own") >= n("s0", "d1_own") - 1
          and n("sa", "friend") >= n("s0", "friend") - 1)
    w4 = all(R["sa", s]["abst"] <= R["s0", s]["abst"] for s in seeds)
    print(f"own recalled s0 {n('s0','w1')} sa {n('sa','w1')}; s2 own s0 {n('s0','s2_own')} sa {n('sa','s2_own')}; keyed s0 "
          f"{mk('s0','keyed'):.2f} sa {mk('sa','keyed'):.2f}; d1 own s0 {n('s0','d1_own')} sa {n('sa','d1_own')}; friend s0 "
          f"{n('s0','friend')} sa {n('sa','friend')}; abstentions s0 {n('s0','abst')} sa {n('sa','abst')}")
    print(f"engaged {eng} W1 {w1} W2 {w2} W3 {w3} W4 {w4} -> "
          f"{'VOID' if not eng else 'GO' if w1 and w2 and w3 and w4 else 'NOT A LEAD' if not (w3 and w4) else 'NARROWED'}")
