"""0a module 1: the Appendix-A toy world with its morphology markers, as canonical JSONL streams.

Port sources (arithmetic kept verbatim; see the packet module-1 spec):
  * ``docs/drafts/deixis/s1-scratch/fix2/replica4_markers.py::world_m`` -- the world with
    the five markers (ART/ASP/IND/AGR/DEF). Marker rendering draws nothing from the world
    RNG; ASP draws from its own ``rng_asp = default_rng(10**6 + seed)``.
  * ``docs/drafts/deixis/s1-scratch/fix2/stream_audit.py::write_stream`` -- the canonical
    JSONL format: one header line, one line per utterance, ``sort_keys=True`` everywhere,
    y sparse as ``[index, value]`` pairs with values rounded to 6 decimals.

Differences from the port sources (each declared in the packet / ruling
deixis-271-s1-0a-q-1022 of 2026-10-22):
  * Rows carry the world's row metadata m (replica3.world's shape: logical S/A, the heard
    Sh/Ah, parts, given, the LEFT/NEWEST coin dicts) so the ``H`` field can be RECORDED
    from the world's own state. Recording makes NO extra RNG draw (m is bookkeeping).
  * ``S`` and ``A`` are written as entity indices (int over ENT = PERS+OBJ), holding the
    LOGICAL speaker/addressee (the committed consumer, ``_deixis_calibration.row_parts``,
    casts int). In the voice arm the ``H`` SPK/ADR pairs follow the HEARD voices Sh/Ah --
    the affected afferents -- while S/A stay logical.
  * Per-utterance field ``H`` (capital, per the ruling): a flat list of ``[k, e]`` int
    pairs -- k over POOLS = (SPK, ADR, GIV, LEFT, NEWEST) in header-POST order, e over
    ENT = PERS+OBJ in header-POST order -- the entities for which relation k holds. Each
    value is the world's own raw state at that utterance, NOT participation-gated:
      SPK  [Sh], ADR [Ah]              the heard voices (== logical S/A except voice_perm)
      GIV  the world's givenness set   replica3 m["given"]: previous-utterance named
                                        entities minus (S, A), plus the pointed entity.
                                        The port's world() KEEPS this state per row, so
                                        the ruling's previous-row fallback is not needed.
      LEFT / NEWEST                    exactly the per-person coin dicts world() already
                                        draws each utterance, for the present entities
                                        that show true.
    ``H`` is recorded once from the row's own world state and never re-derived: the null
    arms perturb y only (label permutes the 5 pool columns per utterance; scene takes y
    from another utterance) -- x, S, A and H stay the raw utterance i, matching
    replica3.nulls, which keeps the row's metadata with x.
  * Header carries the ruling's ``vocab`` ({form: token id} over the replica's TOKM
    order), ``nonce_map: null`` and ``nonce_map_note: "step-3 artifact; 0a streams are
    symbolic"``; the schema stays ``deixis-toy-stream-v1`` and the manifest sha covers
    the header as written. Both ``T`` and ``length`` are written (module 4's contract
    says "T/length").
  * The manifest lists streams RELATIVE to the manifest's directory
    (``<arm>/s<seed>.jsonl``), so two writes to different directories are byte-identical.

CLI: --write-streams --arm {base,label,scene,voice,indep,calib} --seeds A-B --length N
--out <manifest.json>.  Arms: base/calib/indep are plain worlds (calib and indep differ
only in their seed ranges: 61-70 and 1001-1015 by packet convention); voice runs
world_m(voice_perm=True); label and scene apply the replica3 null permutation with
default_rng(777 + seed) / default_rng(555 + seed) -- side RNGs created AFTER the world,
so the world's draw sequence is untouched.

Streams are not committed; manifests are (packet). --selftest asserts: two writes are
byte-identical with the H field present; the original-vocabulary token bits equal
replica3.world's on the same seed (same world draws); header/H domain contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners._deixis_rule_replica import TOKM, POST   # noqa: E402

# --- replica3 constants, copied (the port keeps the source's arithmetic) ---
VOICES = ["V1", "V2", "V3", "V4", "V5"]
PERS = VOICES + ["B1", "SELF"]
OBJ = ["O1", "O2", "O3"]
ENT = PERS + OBJ
POOLS = ["SPK", "ADR", "GIV", "LEFT", "NEWEST"]
NPOOL = len(POOLS)
PRES = 0.2
pi = {p: i for i, p in enumerate(POOLS + ENT)}
ei = {e: i for i, e in enumerate(ENT)}
ki = {k: i for i, k in enumerate(POOLS)}
SCHEMA = "deixis-toy-stream-v1"
MANIFEST_SCHEMA = "deixis-streams-manifest-v1"
ARMS = ("base", "label", "scene", "voice", "indep", "calib")
LABEL_ARM_SEED = 777           # replica3.shuffle_labels(777 + seed)
SCENE_ARM_SEED = 555           # replica3.shuffle_scene(555 + seed)


def world_m(seed, T, f_self=0.4, n_voices=5, turn_taking=0.0, v5_speaks=False,
            prev_use=0.5, joint_p=0.10, p_point=0.30, ana_use=0.7, ord_use=0.6,
            dem_use=0.8, possg_use=0.8, voice_perm=False, p_prog=0.2):
    """replica4 world_m, copied, returning (x, y, m) -- m is the replica3-style row
    metadata (bookkeeping only, no draw) the H recording needs."""
    rng = np.random.default_rng(seed)
    rng_asp = np.random.default_rng(10 ** 6 + seed)
    rows = []
    prev = None
    disc = 0
    present = None
    prev_named = []
    voices = VOICES[:n_voices]
    for _ in range(T):
        if disc == 0:
            k = min(3, len(voices))
            present = list(rng.choice(voices, k, replace=False)) + ["B1", "SELF"]
            disc = 3
            prev = None
            prev_named = []
        disc -= 1
        spk_pool = [p for p in present if p in voices
                    and (v5_speaks or p != "V5")] or \
                   [p for p in present if p in voices]
        S = spk_pool[rng.integers(len(spk_pool))]
        others = [p for p in present if p != S and p != "SELF" and p in voices]
        if rng.random() < f_self or not others:
            A = "SELF"
        elif prev is not None and prev != S and prev in others and \
                rng.random() < turn_taking:
            A = prev
        else:
            A = others[rng.integers(len(others))]
        pointed = None
        if rng.random() < p_point:
            cands = [e for e in present + OBJ if e not in (S, A)]
            pointed = cands[rng.integers(len(cands))]
        given = set(e for e in prev_named if e not in (S, A)) | \
                ({pointed} if pointed else set())
        agent = present[rng.integers(len(present))]
        joint = rng.random() < joint_p and A != "SELF"
        pat_person = None
        obj = None
        owner = None
        if rng.random() < 0.2:
            cands = [p for p in present if p != agent]
            pat_person = cands[rng.integers(len(cands))]
        else:
            obj = OBJ[rng.integers(3)]
            if rng.random() < 0.35:
                owner = present[rng.integers(len(present))]
        left = {p: rng.random() < 0.5 for p in present}
        newest = {p: rng.random() < 0.5 for p in present}
        named_now = []

        def name(p):
            named_now.append(p)
            if p == "V3":
                return "CN" if rng.random() < 0.5 else "N_V3"
            if p == "V2" and p == S and rng.random() < 0.5:
                return "IDIO"
            return "N_" + p

        def refer(p):
            if prev is not None and p == prev and p != S and rng.random() < prev_use:
                return "PREV"
            if p == S:
                return "SHS" if rng.random() < 0.85 else name(p)
            if p == A:
                return "SHA" if rng.random() < 0.85 else name(p)
            if p in given:
                if p == pointed and rng.random() < dem_use:
                    return "DEM"
                if p in prev_named:
                    if len(prev_named) >= 2 and p in prev_named[:2] and \
                            rng.random() < ord_use:
                        return "ORD1" if p == prev_named[0] else "ORD2"
                    if rng.random() < ana_use:
                        return "ANA"
            if rng.random() < 0.10:
                return "OOI1"
            return name(p)

        toks = []
        if joint:
            toks.append("PLUR")
            parts = {S, A}
        else:
            toks.append(refer(agent))
            parts = {agent}
        toks.append("P%d" % (1 + rng.integers(5)))
        if joint or agent in (S, A):
            toks.append("AGR")                 # stress: person agreement
        if pat_person:
            toks.append(refer(pat_person))
            parts.add(pat_person)
        else:
            det = False
            if owner is not None:
                if owner == S:
                    toks.append("POSS_S")
                elif owner == A:
                    toks.append("POSS_A")
                elif owner in given and rng.random() < possg_use:
                    toks.append("POSS_G")
                else:
                    toks.append(name(owner))
                parts.add(owner)
                det = True
            if obj in given and obj == pointed and rng.random() < dem_use:
                toks += ["DEM", "T_" + obj]
                named_now.append(obj)
            elif obj in given and obj in prev_named and len(prev_named) >= 2 and \
                    obj in prev_named[:2] and rng.random() < ord_use:
                toks.append("ORD1" if obj == prev_named[0] else "ORD2")
            elif obj in given and obj in prev_named and rng.random() < ana_use:
                toks.append("ANA")
            else:
                if not det:
                    toks.append("ART")             # EN-matching uniform article
                toks.append("DEF" if obj in given else "IND")
                toks.append("T_" + obj)
                named_now.append(obj)
            parts.add(obj)
        if "ORD1" in toks or "ORD2" in toks:
            toks.append("ART")                     # 'the former' / 'the latter'
        if rng_asp.random() < p_prog:
            toks.append("ASP")                     # EN-matching aspect frame
        if rng.random() < 0.5:
            toks.append("FILL")
        x = np.zeros(len(TOKM))
        for t in toks:
            x[TOKM.index(t)] = 1
        Sh, Ah = S, A
        if voice_perm:
            vv = [p for p in present if p in voices]
            Sh = vv[rng.integers(len(vv))]
            aa = [p for p in present if p != Sh]
            Ah = aa[rng.integers(len(aa))]
        y = np.zeros(len(POST))
        y[pi["SPK"]] = Sh in parts
        y[pi["ADR"]] = Ah in parts
        y[pi["GIV"]] = any(e in given for e in parts)
        y[pi["LEFT"]] = any(left[p] for p in parts if p in PERS)
        y[pi["NEWEST"]] = any(newest[p] for p in parts if p in PERS)
        for p in present:
            y[pi[p]] = PRES
        for p in parts:
            y[pi[p]] = 1.0
        m = dict(S=S, A=A, Sh=Sh, Ah=Ah, parts=set(parts), given=set(given),
                 left=dict(left), newest=dict(newest), prev=prev)
        rows.append((x, y, m))
        prev = S
        prev_named = [e for e in dict.fromkeys(named_now)]
    return rows


def h_pairs(m):
    """The row's H field: raw world state, [(relation k, entity e), ...] sorted.

    See the module docstring for each relation's source. No RNG is touched."""
    pairs = [(ki["SPK"], ei[m["Sh"]]), (ki["ADR"], ei[m["Ah"]])]
    for g in m["given"]:
        pairs.append((ki["GIV"], ei[g]))
    for p, on in m["left"].items():
        if on:
            pairs.append((ki["LEFT"], ei[p]))
    for p, on in m["newest"].items():
        if on:
            pairs.append((ki["NEWEST"], ei[p]))
    return sorted(set(pairs))


def arm_rows(rows, arm, seed):
    """The null arms perturb y only, keeping x/S/A/H on the row's own utterance
    (replica3.nulls parity). label: pool columns per-utterance permuted with
    default_rng(777 + seed); scene: y taken from row perm[i] with
    default_rng(555 + seed); voice is a world kwarg (voice_perm=True), not a
    post-processing step; base/calib/indep are identity. Side RNGs are created
    HERE, after the world, so the world's draw sequence is untouched."""
    if arm == "label":
        rng = np.random.default_rng(LABEL_ARM_SEED + seed)
        out = []
        for x, y, m in rows:
            perm = rng.permutation(NPOOL)
            y2 = y.copy()
            y2[:NPOOL] = y[perm]
            out.append((x, y2, m))
        return out
    if arm == "scene":
        rng = np.random.default_rng(SCENE_ARM_SEED + seed)
        perm = rng.permutation(len(rows))
        return [(rows[i][0], rows[int(perm[i])][1], rows[i][2])
                for i in range(len(rows))]
    return rows


def header(seed, length, arm, world_kw):
    """The canonical header line (ruling dejits-271-s1-0a-q-1022: vocab,
    nonce_map null + note; both T and length)."""
    return {"schema": SCHEMA, "seed": int(seed), "T": int(length),
            "length": int(length), "arm": arm, "world_kw": dict(world_kw),
            "TOK": list(TOKM), "POST": list(POST),
            "vocab": {t: i for i, t in enumerate(TOKM)},
            "nonce_map": None,
            "nonce_map_note": "step-3 artifact; 0a streams are symbolic"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_streams(arm, seeds, length, out_manifest):
    """Write one canonical JSONL per seed under <dir of out>/<arm>/s<seed>.jsonl
    plus manifest_<arm>.json at <dir of out>; return the manifest dict."""
    if arm not in ARMS:
        raise ValueError("unknown arm: %s" % arm)
    mdir = os.path.dirname(os.path.abspath(out_manifest))
    os.makedirs(mdir, exist_ok=True)
    entries = []
    for seed in seeds:
        world_kw = {"voice_perm": True} if arm == "voice" else {}
        rows = world_m(int(seed), int(length), **world_kw)
        rows = arm_rows(rows, arm, int(seed))
        rel = os.path.join(arm, "s%d.jsonl" % int(seed))
        path = os.path.join(mdir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(header(seed, length, arm, world_kw),
                               sort_keys=True) + "\n")
            for t, (x, y, m) in enumerate(rows):
                yv = [[i, round(float(v), 6)]
                      for i, v in enumerate(y) if v > 0.0]
                row = {"t": t,
                       "x": [int(i) for i in np.flatnonzero(x)],
                       "y": yv,
                       "S": ei[m["S"]], "A": ei[m["A"]],
                       "H": [[int(k), int(e)] for k, e in h_pairs(m)]}
                f.write(json.dumps(row, sort_keys=True) + "\n")
        entries.append({"arm": arm, "seed": int(seed), "file": rel,
                        "sha256": sha256(path)})
    manifest = {"schema": MANIFEST_SCHEMA, "arm": arm,
                "seeds": [int(s) for s in seeds], "length": int(length),
                "streams": entries}
    with open(out_manifest, "w") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def _load_replica3_docs():
    """The port source replica3 (docs/drafts/...), for the selftest comparison."""
    path = os.path.join(REPO, "docs", "drafts", "deixis", "s1-scratch",
                        "fix", "replica3.py")
    spec = importlib.util.spec_from_file_location("_deixis_replica3_docs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest():
    """(1) two writes byte-identical -- files AND manifest -- with H present in
    every row; (2) the original-vocabulary token bits equal replica3.world's on
    the same seed (identical world draws); (3) header + H domain contract."""
    res = {}
    with tempfile.TemporaryDirectory() as td1, \
            tempfile.TemporaryDirectory() as td2:
        m1 = write_streams("base", [1, 2], 40,
                           os.path.join(td1, "streams", "manifest_base.json"))
        m2 = write_streams("base", [1, 2], 40,
                           os.path.join(td2, "streams", "manifest_base.json"))
        assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
        for e in m1["streams"]:
            a = os.path.join(td1, "streams", e["file"])
            b = os.path.join(td2, "streams", e["file"])
            ba, bb = open(a, "rb").read(), open(b, "rb").read()
            assert ba == bb, "byte-identical write failed: %s" % e["file"]
            lines = ba.decode().splitlines()
            head = json.loads(lines[0])
            assert head["TOK"] == list(TOKM) and head["POST"] == list(POST)
            assert head["nonce_map"] is None
            assert head["nonce_map_note"] == \
                "step-3 artifact; 0a streams are symbolic"
            assert head["vocab"] == {t: i for i, t in enumerate(TOKM)}
            assert len(lines) - 1 == 40
            n_h = 0
            for ln in lines[1:]:
                d = json.loads(ln)
                assert "H" in d, "H field missing from a written row"
                for k, e in d["H"]:
                    assert 0 <= k < NPOOL and 0 <= e < len(ENT)
                assert 0 <= d["S"] < len(ENT) and 0 <= d["A"] < len(ENT)
                n_h += 1
            assert n_h == 40
        res["byte_identical_two_writes"] = True
        res["h_present_all_rows"] = True
        res["header_contract"] = True

    # (2) original-vocabulary tokens (and y) equal replica3.world's on seed 7
    R3 = _load_replica3_docs()
    mine = world_m(7, 60)
    theirs = R3.world(7, 60)
    assert len(mine) == len(theirs) == 60
    for (x, y, m), (x3, y3, _m3) in zip(mine, theirs):
        assert all(x[i] == x3[i] for i in range(len(R3.TOK))), \
            "original-vocabulary token bits differ from replica3.world"
        assert np.array_equal(y, y3), "y differs from replica3.world"
    res["original_vocab_equal_replica3"] = True

    # (3) H pairs: SPK/ADR singletons from the heard voices; domain of GIV/coins
    for x, y, m in mine[:10]:
        hp = dict()
        for k, e in h_pairs(m):
            hp.setdefault(k, []).append(e)
        assert hp.get(ki["SPK"], []) == [ei[m["Sh"]]]
        assert hp.get(ki["ADR"], []) == [ei[m["Ah"]]]
        assert all(g in (m["given"]) for g in
                   [ENT[e] for e in hp.get(ki["GIV"], [])])
    res["h_shape"] = True

    # voice arm differs from base only through the world's own voice_perm draws
    with tempfile.TemporaryDirectory() as td:
        mv = write_streams("voice", [3], 30,
                           os.path.join(td, "streams", "manifest_voice.json"))
        mb = write_streams("base", [3], 30,
                           os.path.join(td, "streams", "manifest_base.json"))
        v = open(os.path.join(td, "streams", mv["streams"][0]["file"])).read()
        b = open(os.path.join(td, "streams", mb["streams"][0]["file"])).read()
        assert v != b and mv["streams"][0]["sha256"] != mb["streams"][0]["sha256"]
    res["voice_arm_world_kw"] = True
    return res


def _rng(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="0a: the toy world as canonical JSONL streams")
    ap.add_argument("--write-streams", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seeds", help="e.g. 1-60 or 61,62")
    ap.add_argument("--length", type=int)
    ap.add_argument("--out", help="manifest_<arm>.json path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    missing = [n for n, v in (("--write-streams", args.write_streams),
                              ("--arm", args.arm),
                              ("--seeds", args.seeds),
                              ("--length", args.length),
                              ("--out", args.out)) if not v]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))
    man = write_streams(args.arm, _rng(args.seeds), args.length, args.out)
    print("wrote %d streams for arm %s -> %s"
          % (len(man["streams"]), args.arm, args.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())