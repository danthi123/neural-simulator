"""Offline separating read of the language drift (scratch; read-only on the source homes).
usage: drift_read.py HOME LABEL OUT.json [EXTRA_ADAPTERS.pt ...]
Opens a disk COPY of HOME (sleep_on_restart=False). For the home's adapters (and each extra adapters file, same cortex)
reads KL(off||on) [the life_langguard.measure form] + NLL on/off per text on: ORDINARY keep (3), HELDOUT (24), replayed
episodes (created <= last_sleep), pending (not yet replayed) episodes, and 8 plain-prose controls. Also per-position KL,
role split (positions predicting User vs Assistant tokens), and the vocab tokens whose probability rose most on HELDOUT.
With extras, also KL between consecutive adapter states (what the later night changed) per set.
K-P-U gate copy (_scratch/kpid, prereg 2026-11-23-kp-identity section 2): a verbatim copy of _scratch/kpgate_w/drift_read.py
with three changes only: DRIFT_MODEL / DRIFT_DEVICE / DRIFT_PRECISION env (defaults 1.5b / cpu / fp16 = the original),
lp_of returns its log-probs on the CPU (a no-op on CPU; on cuda the reads below are CPU tensors), and the result records
the cortex strategy. Read numbers are unchanged on CPU."""
import json, os, shutil, sys, time
from collections import defaultdict
import torch
import torch.nn.functional as F
from brain.brain import Brain, ORDINARY
from brain.eval.life_heldout import HELDOUT
from brain.eval.life_langguard import measure

PROSE = [  # non-chat ordinary prose controls (no names, no first person)
    "The village sits at the foot of a low hill. In spring the fields turn green and the river runs high with melted snow.",
    "To repot a plant, water it the day before, loosen the roots gently, and set it in fresh soil one size larger.",
    "Bread rises because yeast feeds on sugar and releases gas, which is trapped by the stretchy gluten in the dough.",
    "The train leaves the station at seven each morning and reaches the coast a little after noon.",
    "Owls hunt mostly at night. Their soft feathers let them fly almost silently over open ground.",
    "A good map shows roads, rivers and towns, and a small scale in the corner tells you how far apart they are.",
    "The library opens at nine on weekdays. Books can be borrowed for three weeks and renewed once.",
    "Copper is a soft metal that carries heat and electricity well, which is why it is used in wires and pans.",
]
PARA = [  # near-paraphrases of the 3 ORDINARY keep texts (same format, nearly same words)
    "User: Hi, how are you doing?\n\nAssistant: I'm fine, thanks! And you?\n\nUser: Pretty good. It snowed all day.\n\nAssistant: Snow can be cozy.\n\n",
    "User: Hello, how are you?\n\nAssistant: I'm well, thank you! How about yourself?\n\nUser: Not bad. It was windy all day.\n\nAssistant: Wind can be wild.\n\n",
    "User: What's a good way to spend a snowy morning?\n\nAssistant: Reading a book with coffee is nice.\n\n",
    "User: What's a nice way to spend a quiet evening?\n\nAssistant: Watching a film with some tea is lovely.\n\n",
    "User: Can you count to six?\n\nAssistant: One, two, three, four, five, six.\n\n",
    "User: Can you count backwards from five?\n\nAssistant: Five, four, three, two, one.\n\n"]
SPECIAL = [" Dan", "Dan", " I", "I", " my", " me", " Nova", " friend", " remember", " You", " you"]

home, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
extras = sys.argv[4:]
work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homes", label)
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(home, work, ignore=shutil.ignore_patterns(".lock", "*.tmp"))
meta = json.load(open(os.path.join(work, "meta.json")))
kw = {"replay_priority": meta["replay_priority"]} if meta.get("replay_priority") else {}
from brain.cortex import Cortex, resolve_model
cortex = Cortex(model_path=resolve_model(os.environ.get("DRIFT_MODEL", "1.5b")), device=os.environ.get("DRIFT_DEVICE", "cpu"),
                cuda_precision=os.environ.get("DRIFT_PRECISION", "fp16"))
b = Brain(work, cortex=cortex, lock=True, sleep_on_restart=False, **kw)
c = b.cortex
p = c.plastic
base = [x.float() for x in b.base]
last = meta.get("last_sleep", 0.0)
eps = b.hippocampus.episodes
replayed = [f"{e.text}\n\n" for e in eps if e.created <= last]
pending = [f"{e.text}\n\n" for e in eps if e.created > last]
ep_tok = set()
for t in replayed:
    ep_tok.update(c.pipe.encode(t))
dec = lambda i: c.pipe.decode([i])
spec_ids = {}
for s in SPECIAL:
    ids = c.pipe.encode(s)
    if len(ids) == 1:
        spec_ids[s] = ids[0]


def roles(ids):
    """role of each token: 'A' if inside an Assistant turn else 'U' (by decoded char offsets)."""
    txt, r, cur = "", [], "U"
    for i in ids:
        txt += dec(i)
        if txt.endswith("Assistant:"):
            cur = "A"
        elif txt.endswith("User:"):
            cur = "U"
        r.append(cur)
    return r


def lp_of(ids, state_dict=None, enabled=True):
    if state_dict is not None:
        p.load_state_dict(state_dict)
    p.enabled = enabled
    return F.log_softmax(p(ids, base)[0].float(), -1).cpu()


def adapter_norm():
    return float(torch.sqrt(sum((prm.detach().float() ** 2).sum() for prm in p.parameters() if prm.requires_grad)))


home_sd = {k: v.clone() for k, v in p.state_dict().items()}
states = [("home", home_sd)] + [(os.path.basename(x), torch.load(x, weights_only=False)["adapters"]) for x in extras]

sets_all = {"para_keep": PARA, "keep_ORDINARY": ORDINARY, "HELDOUT": HELDOUT, "episodes_replayed": replayed, "episodes_pending": pending,
        "prose_ctrl": PROSE}
sets = {k: v for k, v in sets_all.items() if not os.environ.get("SETS") or k in os.environ["SETS"].split(",")}
res = {"strategy": getattr(cortex, "strategy", None), "home": home, "label": label, "meta": meta, "n": {k: len(v) for k, v in sets.items()}, "states": {}}
t0 = time.time()
norms = {}
for sname, sd in states:
    p.load_state_dict(sd)
    norms[sname] = adapter_norm()
acc = {sn: {k: {"kl": [], "non": [], "noff": [], "klprev": [], "L": [], "pos": defaultdict(list), "role": defaultdict(list),
               "rise": None, "rev": None, "spec": defaultdict(lambda: [0.0, 0.0])} for k in sets} for sn, _ in states}
with torch.no_grad():
    for k, texts in sets.items():
        for j, t in enumerate(texts):
            ids = c.pipe.encode(t)
            tgt = torch.tensor(ids[1:])
            off = lp_of(ids, enabled=False)
            rl = roles(ids)
            prev = None
            for sname, sd in states:
                a = acc[sname][k]
                on = lp_of(ids, sd)
                kpos = (off.exp() * (off - on)).sum(-1)
                a["kl"].append(float(kpos.mean()))
                a["non"].append(float(F.nll_loss(on[:-1], tgt)))
                a["noff"].append(float(F.nll_loss(off[:-1], tgt)))
                a["L"].append(len(ids))
                if prev is not None:
                    a["klprev"].append(float((prev.exp() * (prev - on)).sum(-1).mean()))
                for pos, v in enumerate(kpos.tolist()):
                    a["pos"]["0-9" if pos < 10 else "10-29" if pos < 30 else "30-59" if pos < 60 else "60+"].append(v)
                for pos in range(len(ids) - 1):
                    a["role"][rl[pos + 1]].append(float(kpos[pos]))
                if k in ("HELDOUT", "prose_ctrl"):
                    d = (on.exp() - off.exp())[:-1].sum(0)
                    a["rise"] = d if a["rise"] is None else a["rise"] + d
                    rc = (on.exp() * (on - off))[:-1].sum(0)
                    a["rev"] = rc if a["rev"] is None else a["rev"] + rc
                    for s_, i in spec_ids.items():
                        a["spec"][s_][0] += float(on.exp()[:-1, i].sum())
                        a["spec"][s_][1] += float(off.exp()[:-1, i].sum())
                prev = on
            print(k, j, len(ids), [round(acc[sn][k]["kl"][-1], 3) for sn, _ in states], flush=True)
mean = lambda v: sum(v) / max(len(v), 1)
for sname, sd in states:
    r = {"adapter_norm": norms[sname], "sets": {}}
    for k in sets:
        a = acc[sname][k]
        if not a["kl"]:
            continue
        row = {"kl": mean(a["kl"]), "nll_on": mean(a["non"]), "nll_off": mean(a["noff"]),
               "per_text_kl": [round(x, 4) for x in a["kl"]], "per_text_len": a["L"],
               "pos_kl": {b_: (mean(v), len(v)) for b_, v in a["pos"].items()},
               "role_kl": {r_: (mean(v), len(v)) for r_, v in a["role"].items()}}
        if a["klprev"]:
            row["kl_vs_prev_state"] = mean(a["klprev"])
        if a["rise"] is not None:
            rise, npos = a["rise"], sum(x - 1 for x in a["L"])
            top = torch.topk(rise, 30)
            row["top_rise"] = [(dec(int(i)), round(float(v) / npos, 5), int(i) in ep_tok) for v, i in zip(top.values, top.indices)]
            topk = torch.topk(a["rev"], 30)
            row["top_revKL_contrib"] = [(dec(int(i)), round(float(v) / npos, 5), int(i) in ep_tok) for v, i in zip(topk.values, topk.indices)]
            pr = rise.clamp(min=0)
            m = torch.zeros(rise.numel(), dtype=torch.bool)
            m[list(ep_tok)] = True
            row["rise_mass_on_episode_vocab_frac"] = float(pr[m].sum() / pr.sum())
            row["episode_vocab_frac_of_vocab"] = len(ep_tok) / rise.numel()
            row["special_mean_p_on_off"] = {s_: (v[0] / npos, v[1] / npos) for s_, v in a["spec"].items()}
        r["sets"][k] = row
    p.load_state_dict(sd)
    r["measure_check_heldout"] = measure(c, base) if sname == "home" else None
    res["states"][sname] = r
res["seconds"] = time.time() - t0
b.release()
shutil.rmtree(work, ignore_errors=True)
json.dump(res, open(out, "w"), indent=1, default=str)
print("done", res["seconds"])
