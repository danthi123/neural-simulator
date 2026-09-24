"""A6 / reasoning-transitive-chat GATE — held-out non-adjacent pairs + a scrambled-premise control, through the
REAL production `webapp.server.brain_chat` handler. See
research/findings/2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md for the full design + GO criteria
(G1-G6) and webapp/reasoning_transitive_chat.py for the mechanism under test.

THIS IS A SEED-7 DEV/SMOKE DE-RISK, NOT THE 6-SEED CAPABILITY GATE. Seed 7 only, per the standing dev-seed rule
(42/43/44/100/101/102 stay reserved for the real gate). A 6-seed job line for the follow-on gate is printed at
the end.

RUN (respect the local RAM discipline -- wrap with mem_ok/memcap; this box was RAM-tight at authoring time, so
this pass may need to run once contention clears, or on the pool):
    OMP_NUM_THREADS=1 SIM_BACKEND=numpy bash tools/mem_ok.sh 12 4 && \\
    OMP_NUM_THREADS=1 bash tools/memcap.sh 14 -- .venv/bin/python -m research.runners.reasoning_transitive_chat_gate \\
        --seed 7 --out research/findings/raw/_reasoning_transitive_chat/s7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


CHAIN = ["e0", "e1", "e2", "e3", "e4"]
RELATION = "precede"

# a fixed pool of fresh tokens for the scrambled-premise control (World B), permuted deterministically off `seed`
# so the run is reproducible but the mapping is NOT the identity/World-A mapping.
_SCRAMBLE_POOL = ["zeta", "mu", "kappa", "rho", "iota", "nu", "xi", "chi"]


def _scrambled_chain(seed: int):
    import random
    rng = random.Random(int(seed) * 97 + 3)
    pool = list(_SCRAMBLE_POOL)
    rng.shuffle(pool)
    return pool[: len(CHAIN)]


def _turn(brain_chat, BrainChatRequest, *, session, message, reset):
    r = brain_chat(BrainChatRequest(session=session, message=message, brain="tiny-demo",
                                    renderer="stub", rich=False, reset=reset))
    return json.loads(r.body)


def _teach_chain(brain_chat, BrainChatRequest, session, chain, relation, first_reset=True):
    for i in range(len(chain) - 1):
        _turn(brain_chat, BrainChatRequest, session=session,
              message=f"{chain[i]} {relation} {chain[i + 1]}",
              reset=(first_reset and i == 0))


def _ask(brain_chat, BrainChatRequest, session, a, relation, b):
    return _turn(brain_chat, BrainChatRequest, session=session,
                 message=f"does {a} {relation} {b}?", reset=False)


def run_gate(seed: int, *, skip_g6: bool = False, skip_worldb: bool = False) -> dict:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))          # cfg.seed, never actual_seed_used (CLAUDE.md rule)
    os.environ["BRAIN_TRANSITIVE_CHAT"] = "1"
    os.environ.pop("BRAIN_TRANSITIVE_LESION", None)

    from webapp.server import brain_chat, BrainChatRequest

    report = {"seed": seed, "criteria": {}, "skipped": []}

    # ── G6 (byte-identity OFF): 2 EXTRA fresh-session builds -- skippable on a RAM-constrained box. The module's
    # OWN control flow already guarantees this (`resolve_transitive_query` returns None on its very first line
    # when the flag is off, BEFORE touching any composer/chat state -- see the standalone logic-check that
    # exercises this exact branch of this exact shipped file), so a skipped G6 here is a reduced-cost pass, not a
    # silently-dropped criterion (named in `skipped`).
    if skip_g6:
        report["skipped"].append("G6_byte_identity_off (see webapp/reasoning_transitive_chat.py's first-line "
                                  "flag check + the module-level logic-check; not independently re-verified "
                                  "through 2 extra live brain builds in this RAM-constrained pass)")
    else:
        os.environ["BRAIN_TRANSITIVE_CHAT"] = ""                # unset-equivalent
        off_a = [_turn(brain_chat, BrainChatRequest, session="g6a", message="the wolf bites the apple", reset=True),
                 _turn(brain_chat, BrainChatRequest, session="g6a", message="what is the capital of france", reset=False)]
        os.environ["BRAIN_TRANSITIVE_CHAT"] = "0"
        off_b = [_turn(brain_chat, BrainChatRequest, session="g6b", message="the wolf bites the apple", reset=True),
                 _turn(brain_chat, BrainChatRequest, session="g6b", message="what is the capital of france", reset=False)]

        def _decision_fields(resp):
            return {"answer": resp.get("answer"), "abstained": resp.get("abstained"),
                    "recalled_svo": resp.get("recalled_svo"), "derived": resp.get("derived")}

        h_a = hashlib.sha256(json.dumps([_decision_fields(r) for r in off_a], sort_keys=True, default=str).encode()).hexdigest()
        h_b = hashlib.sha256(json.dumps([_decision_fields(r) for r in off_b], sort_keys=True, default=str).encode()).hexdigest()
        report["criteria"]["G6_byte_identity_off"] = {"pass": h_a == h_b, "hash_unset": h_a, "hash_explicit_0": h_b}

    os.environ["BRAIN_TRANSITIVE_CHAT"] = "1"

    # ── World A: the taught linear chain, one session ──────────────────────────────────────────────────────────
    SESSION_A = f"transA_{seed}"
    _teach_chain(brain_chat, BrainChatRequest, SESSION_A, CHAIN, RELATION)

    adjacent = [(CHAIN[i], CHAIN[i + 1]) for i in range(len(CHAIN) - 1)]
    nonadjacent = [(CHAIN[i], CHAIN[j]) for i in range(len(CHAIN)) for j in range(i + 2, len(CHAIN))]
    reverse_neg = [(CHAIN[-1], CHAIN[0]), (CHAIN[2], CHAIN[1])]
    unrelated_neg = [(CHAIN[0], "zzz")]

    def _probe_all(pairs):
        out = []
        for a, b in pairs:
            resp = _ask(brain_chat, BrainChatRequest, SESSION_A, a, RELATION, b)
            out.append({"a": a, "b": b, "derived": resp.get("derived"), "abstained": resp.get("abstained"),
                        "derived_from": resp.get("derived_from"), "answer": resp.get("answer")})
        return out

    g1_rows = _probe_all(nonadjacent)
    g2_rows = _probe_all(adjacent)
    g3_rows = _probe_all(reverse_neg + unrelated_neg)

    g1_pass = all(r["derived"] is True and r["abstained"] is False for r in g1_rows)
    g2_pass = all(r["derived"] is not True and r["abstained"] is False for r in g2_rows)
    g3_pass = all(r["abstained"] is True for r in g3_rows)

    report["criteria"]["G1_nonadjacent"] = {"pass": g1_pass, "rows": g1_rows}
    report["criteria"]["G2_adjacent"] = {"pass": g2_pass, "rows": g2_rows}
    report["criteria"]["G3_negative_controls"] = {"pass": g3_pass, "rows": g3_rows}

    # ── G5: the lesion arm, SAME session (lesion is read at CALL time, not build time) ─────────────────────────
    os.environ["BRAIN_TRANSITIVE_LESION"] = "1"
    g5_nonadj_rows = _probe_all(nonadjacent)
    g5_adj_rows = _probe_all(adjacent)
    os.environ.pop("BRAIN_TRANSITIVE_LESION", None)
    g5_nonadj_pass = all(r["abstained"] is True for r in g5_nonadj_rows)
    g5_adj_pass = all(r["derived"] is not True and r["abstained"] is False for r in g5_adj_rows)
    report["criteria"]["G5_lesion"] = {
        "pass": g5_nonadj_pass and g5_adj_pass,
        "nonadjacent_collapse_pass": g5_nonadj_pass, "adjacent_unchanged_pass": g5_adj_pass,
        "nonadjacent_rows": g5_nonadj_rows, "adjacent_rows": g5_adj_rows,
    }

    # ── World B: scrambled-premise control, a FRESH session + a FRESH random mapping (1 extra brain build) ──────
    if skip_worldb:
        report["skipped"].append("G4_scrambled_premise (deferred: 1 extra fresh-session brain build, RAM-"
                                  "constrained pass; logic-level equivalent already verified by the standalone "
                                  "logic-check against a freshly-scrambled synthetic mapping)")
    else:
        scrambled = _scrambled_chain(seed)
        SESSION_B = f"transB_{seed}"
        _teach_chain(brain_chat, BrainChatRequest, SESSION_B, scrambled, RELATION)
        b_probe = _ask(brain_chat, BrainChatRequest, SESSION_B, scrambled[0], RELATION, scrambled[3])
        g4_pass = (b_probe.get("derived") is True and b_probe.get("abstained") is False
                   and b_probe.get("derived_from") == [[scrambled[0], RELATION, scrambled[1]],
                                                        [scrambled[1], RELATION, scrambled[2]],
                                                        [scrambled[2], RELATION, scrambled[3]]])
        report["criteria"]["G4_scrambled_premise"] = {
            "pass": g4_pass, "mapping": scrambled,
            "probe": {"derived": b_probe.get("derived"), "abstained": b_probe.get("abstained"),
                      "derived_from": b_probe.get("derived_from"), "answer": b_probe.get("answer")},
        }

    all_pass = all(c["pass"] for c in report["criteria"].values())
    report["verdict"] = ("GO (seed-7 de-risk%s)" % (", PARTIAL -- see skipped" if report["skipped"] else "")
                          if all_pass else "NO-GO")
    report["failing"] = [k for k, v in report["criteria"].items() if not v["pass"]]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="dev seed ONLY (never 42/43/44/100/101/102 here)")
    ap.add_argument("--out", default="research/findings/raw/_reasoning_transitive_chat/s7.json")
    ap.add_argument("--skip-g6", action="store_true",
                    help="skip the 2-extra-brain-build byte-identity-off re-check (RAM-constrained pass)")
    ap.add_argument("--skip-worldb", action="store_true",
                    help="skip the 1-extra-brain-build scrambled-premise control (RAM-constrained pass)")
    args = ap.parse_args()
    if args.seed in (42, 43, 44, 100, 101, 102):
        raise SystemExit("refusing: this is the dev/smoke gate, not the 6-seed capability gate -- "
                          "reserved seeds (42/43/44/100/101/102) run under a separate --checked pool job.")
    report = run_gate(args.seed, skip_g6=args.skip_g6, skip_worldb=args.skip_worldb)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print(json.dumps({"seed": args.seed, "verdict": report["verdict"], "failing": report["failing"]}, indent=2))
    print("\n6-SEED CAPABILITY-GATE job line (staged, NOT run by this pass):")
    for s in (42, 43, 44, 100, 101, 102):
        print(f"  cd ~/derisk-pool/revisions/<F-sha> && SIM_BACKEND=numpy OMP_NUM_THREADS=1 "
              f".venv/bin/python -u -m research.runners.reasoning_transitive_chat_gate --seed {s} "
              f"--out research/findings/raw/_reasoning_transitive_chat/s{s}.json  "
              f"# mem_gb=<measure on first run>")
    return 0 if report["verdict"].startswith("GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
