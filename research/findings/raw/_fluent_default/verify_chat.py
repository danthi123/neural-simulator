"""Live verification of the fluent-default /api/brain-chat change.

Assumes a uvicorn server is already running at BASE (launched with
SIM_BACKEND=cupy). Exercises: default (no rich) turn -> fluent multi-sentence
+ renderer=qwen + composer=onebrain; moat abstain; teach-then-recall; and the
rich=False escape (single-SVO) vs default byte-comparison.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791"
# Use the 'default' session so the warm-cached onebrain+qwen ChatBrain is
# reused (avoids a second ~180s build while the GPU is shared with gaming).
SESS_DEFAULT = "default"
SESS_ESC = "default"


def post(path, body, timeout=400):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    out["_elapsed_s"] = round(time.time() - t0, 1)
    return out


def brains():
    with urllib.request.urlopen(BASE + "/api/brains", timeout=30) as r:
        return json.loads(r.read().decode())


def show(tag, d):
    keys = ("answer", "abstained", "recalled_svo", "verified", "renderer",
            "rich", "n_sentences", "supporting_facts", "_elapsed_s")
    a = (d.get("activity") or {})
    comp = a.get("composer") if isinstance(a, dict) else None
    print(f"\n=== {tag} ===")
    for k in keys:
        if k in d:
            print(f"  {k}: {d[k]!r}")
    print(f"  activity.composer: {comp!r}")


def main():
    print("brains:", json.dumps(brains())[:200])

    # (a) DEFAULT turn (no rich field) -> fluent multi-sentence, qwen, onebrain
    d1 = post("/api/brain-chat", {"session": SESS_DEFAULT, "brain": "tiny-demo",
                                  "message": "what are you"})
    show("DEFAULT no-rich  'what are you'", d1)

    d2 = post("/api/brain-chat", {"session": SESS_DEFAULT, "brain": "tiny-demo",
                                  "message": "how do you learn"})
    show("DEFAULT no-rich  'how do you learn'", d2)

    # (b) MOAT: unknown question abstains
    d3 = post("/api/brain-chat", {"session": SESS_DEFAULT, "brain": "tiny-demo",
                                  "message": "what is the capital of france"})
    show("DEFAULT no-rich  moat 'capital of france'", d3)

    # (b) teach a fact, then recall it in prose
    d4 = post("/api/brain-chat", {"session": SESS_DEFAULT, "brain": "tiny-demo",
                                  "message": "wolf hunt deer"})
    show("TEACH 'wolf hunt deer'", d4)
    d5 = post("/api/brain-chat", {"session": SESS_DEFAULT, "brain": "tiny-demo",
                                  "message": "what does the wolf hunt"})
    show("RECALL 'what does the wolf hunt'", d5)

    # (c) ESCAPE: rich=False -> single-SVO path (old behavior)
    e1 = post("/api/brain-chat", {"session": SESS_ESC, "brain": "tiny-demo",
                                  "message": "what does the dog chase", "rich": False})
    show("ESCAPE rich=False 'what does the dog chase'", e1)
    e2 = post("/api/brain-chat", {"session": SESS_ESC, "brain": "tiny-demo",
                                  "message": "what does the dragon breathe", "rich": False})
    show("ESCAPE rich=False moat 'dragon breathe'", e2)

    # summary assertions
    print("\n----- CHECKS -----")
    checks = []
    checks.append(("default d1 rich==True", d1.get("rich") is True))
    checks.append(("default d1 n_sentences>1", (d1.get("n_sentences") or 0) > 1))
    checks.append(("default d1 renderer qwen", "qwen" in str(d1.get("renderer", "")).lower()))
    checks.append(("default d1 composer onebrain",
                   (d1.get("activity") or {}).get("composer") == "onebrain"))
    checks.append(("default d1 not abstained", d1.get("abstained") is False))
    checks.append(("moat d3 abstained", d3.get("abstained") is True))
    checks.append(("teach d4 recalled wolf/hunt/deer",
                   d4.get("recalled_svo") == ["wolf", "hunt", "deer"]))
    checks.append(("recall d5 mentions deer", "deer" in str(d5.get("answer", "")).lower()))
    checks.append(("escape e1 rich==False", e1.get("rich") is False))
    checks.append(("escape e1 single recalled_svo dog/chase/cat",
                   e1.get("recalled_svo") == ["dog", "chase", "cat"]))
    checks.append(("escape e2 abstained", e2.get("abstained") is True))
    ok = True
    for name, val in checks:
        print(f"  [{'PASS' if val else 'FAIL'}] {name}")
        ok = ok and bool(val)
    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
