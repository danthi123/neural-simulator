"""Stand-in organ preview on DEV seeds 1-8 only (no brain, no battery shard): the Amendment-B session shapes on the
idle-replay organ alone, to set the tag-lesion and re-tell bars before any gate data. Usage:
    python docs/drafts/idle-replay/c5_amendB_standin_dev.py docs/drafts/idle-replay/c5_amendB_standin_dev.json
(run 2026-10-13 on the workstation, numpy, under memcap 3; dev seeds 1-8 only)
"""
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from research.runners import idle_replay_organ as IR  # noqa: E402
from research.runners import spiking_commit as SC  # noqa: E402

BUILD = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
         ("dog", "chase", "cat"), ("cat", "eat", "fish")]
F = [("lynx", "chase", "vole"), ("heron", "hunt", "frog"), ("yak", "eat", "grass"), ("crow", "chase", "wren")]
SHAPES = {
    # (fact, idx, t_enc) after the build facts; idle turn number
    "stagger": ([(F[0], 5, 1), (F[1], 6, 2), (F[2], 7, 3), (F[3], 8, 4)], 5),
    # re-tell, case A: F2 re-tell appends a slot (t5), F1 rewritten in place with a new patient (t6)
    "retell_append_inplace": ([(("lynx", "chase", "frog"), 5, 6), (F[2], 7, 3), (F[3], 8, 4), (F[1], 9, 5)], 7),
    # re-tell, case B: F2 re-tell is a no-op (no slot, no re-tag), F1 rewritten in place (t6)
    "retell_noop_inplace": ([(("lynx", "chase", "frog"), 5, 6), (F[1], 6, 2), (F[2], 7, 3), (F[3], 8, 4)], 7),
}
N_EXTRA = 10


def organ(seed, shape):
    facts, turn = SHAPES[shape]
    o = IR.IdleReplayOrgan(seed=seed, params=dict(IR.REPLAY_GRID[0], OP=0))
    for i, f in enumerate(BUILD):
        o.note_encoding(f, i, 0)
    for f, idx, t in facts:
        o.note_encoding(f, idx, t)
    o.turn = turn
    return o


def winner(o, cands, ns, **kw):
    o.noise_seed = (lambda K, n=ns: n)
    r = o.select(cands, **kw)
    w = r.get("winner")
    return None if w is None else list(cands[w]["fact"]), int(sum(r.get("gate_counts") or [0]))


def main(out):
    res = {}
    for shape in SHAPES:
        res[shape] = {}
        for seed in IR.DEV_SEEDS:
            o = organ(seed, shape)
            cands = o.candidates()
            K = len(cands)
            battery_ns = int(seed) * 1000003 + int(o.turn) * 7919 + K * 31 + 17   # IdleReplayOrgan.noise_seed
            newest = max(cands, key=lambda c: c["x"])["fact"]
            last = max(cands, key=lambda c: c["kb_idx"])["fact"]
            rec = {"K": K, "newest": newest, "last_slot": last, "battery_draw": {}, "extra": {"intact": [], "tag": []}}
            rec["battery_draw"]["intact"] = winner(o, cands, battery_ns)[0]
            rec["battery_draw"]["tag"] = winner(o, cands, battery_ns, tag_lesion=True)[0]
            for n in range(N_EXTRA):
                ns = 100003 * int(seed) + 7 * n + 1
                rec["extra"]["intact"].append(winner(o, cands, ns)[0])
                rec["extra"]["tag"].append(winner(o, cands, ns, tag_lesion=True)[0])
            res[shape][seed] = rec
            bi, bt = rec["battery_draw"]["intact"], rec["battery_draw"]["tag"]
            print(shape, seed, "K", K, "battery intact", bi, "tag", bt,
                  "| extra newest-wins intact %d/%d tag %d/%d, tag==intact %d/%d" % (
                      sum(1 for w in rec["extra"]["intact"] if w == list(newest)), N_EXTRA,
                      sum(1 for w in rec["extra"]["tag"] if w == list(newest)), N_EXTRA,
                      sum(1 for a, b in zip(rec["extra"]["intact"], rec["extra"]["tag"]) if a == b), N_EXTRA),
                  flush=True)
    try:
        from sim.backend import get_backend
        backend = get_backend()[0].__name__
    except Exception:  # noqa: BLE001
        backend = "unknown"
    with open(out, "w") as fh:
        json.dump({"backend": backend, "device": "cpu", "seeds": "dev 1-8 only", "op": 0,
                   "runner": "docs/drafts/idle-replay/c5_amendB_standin_dev.py", "shapes": res}, fh, indent=1,
                  sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
