"""Shared builder for the Stage-3 exit-gate bar-sanity specs (tools/bar_sanity.py).

The decision is the REAL aggregate: `research.runners._stage3_exit_draft.aggregate` run over synthetic six-seed shards
in the worker's format, reading one group's CRITERION verdict (capability AND ownership). Nothing about a bar is
reimplemented: the scorers, the lesion-effect / reproduction / engagement / dissociation clauses, the permuted-tracks
clause, the steer clause, the ownership instruments (F, ORDER, RECORD, RI, G) and the UNDEFINED rules are the runner's
own; the only glue is that the shards are handed to `aggregate` from memory (`_load_shards` swapped for the call), and
`selfcheck()` proves the two paths return the identical aggregate.

THE SYNTHETIC BRAINS are the runner's (`_stage3_exit_draft.synthetic_shard`, one `kind` per brain: `perfect`, the
production-at-main shape, every host shortcut PLAN.md section 7 names, and one inert-instrument brain per registered
clause). `realistic=True` wraps replies in the discourse leads / trailers the live handler prints (dev-7 Stage-2
surfaces) and scales every recorded rate by one per-read factor (the ratios the RI grid reads are preserved, as a real
read's would be under a gain change). Brains are DETERMINISTIC per (draw, seed, kind): intact_b is byte-identical to
intact_a and a lesion arm to its repeat, which is what the Stage-2 dev-7 nulls showed.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import _stage3_exit_draft as G  # noqa: E402

SEEDS = list(G.GATE_SEEDS)
LESION = {"chain": "ws_lesion", "meta": "inherit_lesion", "tom": "store_lesion", "credit": "no_teacher"}


def _sub(master, tag, salt=0):
    return np.random.default_rng([int(master), zlib.crc32(str(tag).encode()), int(salt)])


def _decorate(resp, rng, label):
    """The live handler's discourse lead / trailer around a reply (dev-7 Stage-2 surfaces); never inside a frame."""
    a = str(resp.get("answer") or "")
    r = rng.random()
    subj = label.split("_")[-1] if "_" in label else "it"
    lead = ("" if r < 0.4 else "As for it — " if r < 0.6 else "Setting the held thread aside — On %s, then — " % subj)
    tail = " — worth going further here." if rng.random() < 0.2 else ""
    if resp.get("abstained") and rng.random() < 0.3:
        tail += " My curiosity is piqued — I haven't learned about %s yet: what can you tell me about %s?" % (subj, subj)
    resp["answer"] = lead + a + tail
    return resp


def _scale_rates(sh, rng):
    """One gain factor per recorded read (ratios preserved)."""
    for r in (sh.get("responses") or {}).values():
        ci = (r.get("category_inherit") or {}).get("read") if isinstance(r, dict) else None
        if isinstance(ci, dict) and isinstance(ci.get("rates"), dict):
            f = float(rng.uniform(0.85, 1.15))
            ci["rates"] = {k: round(v * f, 6) for k, v in ci["rates"].items()}
        fb = r.get("false_belief_tom") if isinstance(r, dict) else None
        if isinstance(fb, dict) and isinstance(fb.get("belief_rates"), dict):
            f = float(rng.uniform(0.85, 1.15))
            fb["belief_rates"] = {k: round(v * f, 6) for k, v in fb["belief_rates"].items()}
    obs = sh.get("observe_patch")
    if isinstance(obs, dict):
        for e in obs.get("events") or []:
            if e.get("rates"):
                f = float(rng.uniform(0.9, 1.1))
                e["rates"] = [round(x * f, 6) for x in e["rates"]]


def make_group(rng, group, kind="perfect", over=None, realistic=True, unclean=False, unreproduced=(), permuted_inert=False):
    """One six-seed run of ONE group: {seed: {arm: shard}}.
    over            : {arm_base: kind} replaces that arm's brain kind (a repeat follows its base)
    unclean         : intact_b differs from intact_a on its first turn (a non-deterministic build: the null fails)
    unreproduced    : lesion names whose `.rep0` differs from the first run on the scored turns
    permuted_inert  : the permuted arm ignores the exposure (replies as intact)"""
    data = {}
    for seed in SEEDS:
        master = int(rng.integers(0, 2 ** 31 - 1))
        data[seed] = {}
        for arm in G.counted_arms(group):
            base = arm.split(".rep")[0]
            k = (over or {}).get(base, kind)
            sub = _sub(master, k)                                   # same draws for intact_a/intact_b and a lesion/its repeat
            if base == "permuted" and permuted_inert:
                sh = G.synthetic_shard(group, seed, G._intact_names(group)[0], k, rng=sub)
                sh["arm"] = arm
                sh["turns"] = [{"label": l, "message": m, "role": r} for l, m, r in G.turns_for(group, seed, arm)]
            else:
                sh = G.synthetic_shard(group, seed, arm, k, rng=sub)
            if realistic:
                dec = _sub(master, k, 1)
                for label, resp in sh["responses"].items():
                    if isinstance(resp, dict) and "answer" in resp:
                        _decorate(resp, dec, label)
                _scale_rates(sh, _sub(master, k, 2))
            if arm.endswith(".rep0") and base in unreproduced:
                for l in G.SCORED_LABELS[group][base]:
                    sh["responses"][l] = dict(sh["responses"][l], answer=sh["responses"][l]["answer"] + " (repeat)")
            if arm == G._intact_names(group)[1] and unclean:
                l0 = next(iter(sh["responses"]))
                sh["responses"][l0] = dict(sh["responses"][l0], answer=sh["responses"][l0]["answer"] + " (b)")
            data[seed][arm] = sh
    return data


def _aggregate(shards, seeds, root="<memory>"):
    orig = G._load_shards
    G._load_shards = lambda root, sd: shards
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return G.aggregate(root, seeds=seeds, informational=False)
    finally:
        G._load_shards = orig


def decide_group(data, group):
    """The real aggregate's CRITERION verdict (capability AND ownership) for this group over six seeds is PASS."""
    shards = {(seed, group, arm): sh for seed, arms in data.items() for arm, sh in arms.items()}
    return _aggregate(shards, sorted(data))["groups"][group]["verdict"] == "PASS"


def write_tree(root, data_by_group):
    for g, d in data_by_group.items():
        for seed, arms in d.items():
            for arm, sh in arms.items():
                p = G.shard_path(root, seed, g, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    json.dump(sh, fh)


def selfcheck(n=6, seed=3):
    """The in-memory glue returns the identical aggregate as the on-disk path, over a spread of brains, every group."""
    rng = np.random.default_rng(seed)
    bad = 0
    for g in G.PRIMARY_GROUPS:
        for i in range(n):
            kw = [dict(), dict(permuted_inert=True), dict(realistic=False), dict(over={LESION[g]: "lesion_inert"}),
                  dict(unclean=True), dict(kind="ri_unstable")][i % 6]
            data = make_group(rng, g, **kw)
            with tempfile.TemporaryDirectory() as td:
                write_tree(td, {g: data})
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    disk = G.aggregate(td, seeds=sorted(data), informational=False)
                mem = _aggregate({(s, g, a): sh for s, arms in data.items() for a, sh in arms.items()}, sorted(data), root=td)
                if json.dumps(disk, sort_keys=True, default=str) != json.dumps(mem, sort_keys=True, default=str):
                    bad += 1
    return bad


# ══════════════════════════════════════════════ the generators a spec exposes ═══════════════════════════════════════
_NULL_KINDS = {
    "chain": {"null_production_at_main": "main", "null_kb_closure_first_match": "closure",
              "null_kb_closure_all_leaves": "closure_all_leaves", "null_closure_relay_ignition_check": "relay",
              "null_unmarked_terminal": "unmarked", "null_single_branch_fork": "single_branch_fork",
              "null_lesion_kills_recall": "lesion_kills_recall", "null_order_dependent": "order_dependent",
              "null_kselect_host": "f_host", "null_ri_unstable": "ri_unstable"},
    "meta": {"null_never_abstains": "never_abstains", "null_hedge_by_taught": "hedge_by_taught", "null_all_abstain": "all_abstain",
             "null_item_keyed": "item_keyed", "null_random_optout": "random_optout", "null_conflict_counter": "conflict_counter",
             "null_ri_unstable": "ri_unstable"},
    "tom": {"null_first_location": "first_location", "null_present_means_reality": "present_means_reality",
            "null_host_rule_unread_store": "host_rule_unread_store", "null_host_rule_store_veto": "host_rule_store_veto",
            "null_host_written_belief": "host_written_belief", "null_ri_unstable": "ri_unstable"},
    "credit": {"null_production_at_main": "main", "null_recent_correction": "recent_correction", "null_exemplar_nn": "exemplar_nn",
               "null_host_inferred_block": "host_inferred_block", "null_activity_only": "activity_only",
               "null_ri_unstable": "ri_unstable"},
}
_PERFECT_EXTRA = {"chain": {"perfect_swap_wording": "perfect_swap"}, "meta": {"perfect_verbal_hedge": "perfect_hedge"},
                  "tom": {}, "credit": {}}


def generators(group):
    """{name: callable(rng) -> data}: the perfect brains, every host shortcut and inert instrument as a null."""
    les = LESION[group]

    def mk(**kw):
        return lambda rng: make_group(rng, group, **kw)
    out = {"perfect": mk(), "perfect_exact": mk(realistic=False),
           "null_permuted_inert": mk(permuted_inert=True), "null_unclean": mk(unclean=True),
           "null_lesion_inert": mk(kind="lesion_inert"), "null_lesion_unreproduced": mk(unreproduced=(les,)),
           "null_steer_inert": mk(kind="steer_inert")}
    for name, k in _NULL_KINDS[group].items():
        out[name] = mk(kind=k)
    for name, k in _PERFECT_EXTRA[group].items():
        out[name] = mk(kind=k)
    return out


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        b = selfcheck()
        print("selfcheck: %d mismatches between the in-memory and the on-disk aggregate" % b)
        sys.exit(1 if b else 0)
