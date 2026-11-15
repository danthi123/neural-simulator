# Capability: Self and identity
# Record: brain/research/2026-11-15-self-completion-weekly-1.5b.md (C0-C3)
"""C0-C3 for brain/research/2026-11-15-self-completion-weekly-1.5b.md. usage: score_sc.py <dir> <seeds...>
expects <dir>/scoff_s<seed>.md and scon_s<seed>.md (weekly transcripts)."""
import re, sys
from statistics import mean
def parse(p):
    t = open(p).read()
    head = t.split("\n## ", 1)[0]
    secs = {}
    for sec in re.split(r"\n## ", t)[1:]:
        name = sec.split("\n", 1)[0]
        secs[name] = [(q.strip(), a.strip()) for q, a in
                      re.findall(r"- \*\*you:\*\* (.*?)\n  \*\*brain:\*\* (.*?)(?=\n- \*\*you|\n\n#|\Z)", sec, re.S)]
    return head, secs
def strip(a): return re.sub(r" ?_\(recalled:.*", "", a, flags=re.S).strip()
def ok(a): return "✅" in a
d, seeds = sys.argv[1], sys.argv[2:]
R = {}
for arm in ("off", "on"):
    for s in seeds:
        head, secs = parse(f"{d}/sc{arm}_s{s}.md")
        s1 = secs[[k for k in secs if k.startswith("session 1")][0]]
        intact = secs["intact"]
        hoff = secs.get([k for k in secs if "hippocampus off" in k][0], []) if any("hippocampus off" in k for k in secs) else []
        get = lambda S, q: next((a for qq, a in S if q in qq), "")
        R[arm, s] = dict(
            hdr="self episodes store their completion" in head,
            d1_own=ok(s1[0][1]), d1_own_txt=strip(s1[0][1]),
            d1_cat=ok(get(s1, "What is my cat called")),
            keyed=sum(ok(a) for _, a in intact),
            told=sum(ok(get(intact, q)) for q in ("cat called", "sister", "car do I", "for work")),
            friend=ok(get(intact, "remember my name")), friend_txt=strip(get(intact, "remember my name")),
            s2_own=ok(get(intact, "What's your name")), s2_own_txt=strip(get(intact, "What's your name")),
            hoff_keyed=sum(ok(a) for _, a in hoff),
            echo=sum("What's your name?" in strip(a) for _, a in s1 + intact if a))
for s in seeds:
    for arm in ("off", "on"):
        r = R[arm, s]
        print(f"s{s} {arm:3} hdr {int(r['hdr'])} d1own {int(r['d1_own'])} '{r['d1_own_txt'][:40]}' d1cat {int(r['d1_cat'])} "
              f"keyed {r['keyed']} told {r['told']} friend {int(r['friend'])} '{r['friend_txt'][:45]}' s2own {int(r['s2_own'])} "
              f"'{r['s2_own_txt'][:30]}' hoff {r['hoff_keyed']} echo {r['echo']}")
n = lambda arm, k: sum(R[arm, s][k] for s in seeds)
mk = lambda arm, k: mean(R[arm, s][k] for s in seeds)
c0 = all(R["on", s]["hdr"] and not R["off", s]["hdr"] for s in seeds)
c1 = n("on", "d1_own") >= 5 and n("on", "d1_own") >= n("off", "d1_own") + 2
c2 = mk("on", "keyed") >= mk("off", "keyed") - 0.5 and n("on", "d1_cat") >= n("off", "d1_cat")
c3 = n("on", "friend") >= n("off", "friend")
print(f"day-1 own off {n('off','d1_own')}/{len(seeds)} on {n('on','d1_own')}/{len(seeds)}; keyed mean off {mk('off','keyed'):.2f} on "
      f"{mk('on','keyed'):.2f}; told mean off {mk('off','told'):.2f} on {mk('on','told'):.2f}; d1 cat off {n('off','d1_cat')} on "
      f"{n('on','d1_cat')}; friend off {n('off','friend')} on {n('on','friend')}; s2 own off {n('off','s2_own')} on {n('on','s2_own')}; "
      f"hoff keyed off {mk('off','hoff_keyed'):.2f} on {mk('on','hoff_keyed'):.2f}")
print(f"C0 {c0} C1 {c1} C2 {c2} C3 {c3} -> {'GO' if c0 and c1 and c2 and c3 else 'NOT A LEAD' if not c1 else 'NARROWED' if c0 else 'VOID'}")
