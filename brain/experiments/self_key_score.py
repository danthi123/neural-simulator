# Capability: Self and identity
# Record: brain/research/2026-11-15-self-key-question-1.5b.md (K1-K3)
"""K1-K3 for the self-key probe. usage: self_key_score.py <dir with skp_*.json>"""
import glob, json, sys
TOLD = {"What is my cat called?": "cat is called", "What is my sister's name?": "sister", "What car do I drive?": "Volvo",
        "What do I do for work?": "work as", "What is my cat called? (session 1)": "cat is called"}
SELF = lambda w: "\n\nAssistant:" in w
CHAT = ["Tell me a short story about a fox who gets lost.", "How are you feeling today?"]
homes = [h for f in sorted(glob.glob(sys.argv[1] + "/skp_*.json")) for h in json.load(open(f))["homes"]]
k1 = k2 = 0; k3 = {"exchange": 0, "question": 0}; rep = {}
for h in homes:
    s = h["home"].split("_s")[-1].split("/")[0]
    E, Q = h["exchange"], h["question"]
    own = Q["What's your name?"]; ok1 = own["passes"] and own["winner"].startswith("User: What's your name?")
    k1 += ok1
    ok2 = True
    for cue, tag in TOLD.items():
        was = E[cue]["passes"] and tag in E[cue]["winner"]; now = Q[cue]["passes"] and tag in Q[cue]["winner"]
        stolen = Q[cue]["passes"] and SELF(Q[cue]["winner"]) and not (E[cue]["passes"] and SELF(E[cue]["winner"]))
        if (was and not now) or stolen: ok2 = False
    k2 += ok2
    for arm, R in (("exchange", E), ("question", Q)):
        k3[arm] += any(R[c]["passes"] and SELF(R[c]["winner"]) for c in CHAT)
    told_now = sum(Q[c]["passes"] and t in Q[c]["winner"] for c, t in TOLD.items())
    told_was = sum(E[c]["passes"] and t in E[c]["winner"] for c, t in TOLD.items())
    fr = Q["Hi again! Do you remember my name?"]; wh = Q["Who are you?"] if "Who are you?" in Q else None
    print(f"s{s} K1 {ok1} ({own['best3'][0][1]} margin {own['margin']}) K2 {ok2} told {told_was}->{told_now}/5 "
          f"friend-cue -> {fr['winner'][:34]!r} {fr['passes']} | chat self-pass ex/q "
          f"{any(E[c]['passes'] and SELF(E[c]['winner']) for c in CHAT)}/{any(Q[c]['passes'] and SELF(Q[c]['winner']) for c in CHAT)}")
n = len(homes)
print(f"K1 {k1}/{n} (>= 10) {k1 >= 10}; K2 {k2}/{n} (>= 11) {k2 >= 11}; K3 chat self-pass exchange {k3['exchange']} question {k3['question']} "
      f"(<= +1) {k3['question'] <= k3['exchange'] + 1}")
