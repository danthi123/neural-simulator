"""Read-only CLASS HD check of the round-2 DESIGN and PACKET at their planned repo paths (scratch; run with python -B)."""
import sys, os, importlib.util
ROOT = "/home/dant123/Projects/sim"
R2 = "/home/dant123/Projects/sim-rev/build8/r271/round2"
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("hdd", os.path.join(ROOT, "tools/gates/host_decision_declared.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pra, sf = m._pra(), m._sf()
env = pra._git_env(ROOT)
files = {"research/coordination/deixis-271-s1-PACKET.md": os.path.join(R2, "PACKET.md"),
         "docs/drafts/deixis/DESIGN-s1.md": os.path.join(R2, "DESIGN.md")}
texts = {rel: open(p).read() for rel, p in files.items()}
cache = {}
def runbook_fn():
    if "rb" not in cache: cache["rb"] = sf.read_runbook(ROOT)
    return cache["rb"]
def peers_fn():
    if "peers" not in cache:
        r = pra._git(["grep", "--cached", "-z", "-l", "-I", "-i", "-F", "-e", "host-decision", "--"] + list(m._specs()),
                     ROOT, env, ok_codes=(0, 1))
        out = {}
        for rel in r.stdout.decode("utf-8", "surrogateescape").split("\0"):
            k = m.kind_of(rel) if rel else None
            if k: out[rel] = (k, pra._git(["cat-file", "blob", ":%s" % rel], ROOT, env).stdout.decode("utf-8", "replace"))
        for rel, t in texts.items(): out[rel] = (m.kind_of(rel), t)
        cache["peers"] = out
    return cache["peers"]
for rel, t in texts.items():
    k = m.kind_of(rel); h = m._parse(t)
    n_hd = sum(1 for _, key, _v in h["decl"] if key == "host-decision")
    print("==", rel, "kind", k, "state", h["state"], "header line", (h["line"] or 0) + 1, "first ## line", h["first"] + 1,
          "host-decision lines", n_hd, "cap", h["cap"])
    print("   fields:", {f: len(v) for f, v in h["fields"].items()})
    probs = m.judge(rel, t, k, runbook_fn, peers_fn=peers_fn)
    print("   problems:", probs if probs else "NONE")
same = all(m._parse(texts[a])["vals"].get(f) == m._parse(texts[b])["vals"].get(f)
           for a in texts for b in texts for f in ("decides", "allowed-by", "replaced-by", "cap"))
print("identical decides/allowed-by/replaced-by/cap across the two files:", same)
