"""DG-CA3 SPARSE-INDEX over the ONEBRAIN COMPOSER'S FACT BLOCKS -- a SUBLINEAR, brain-based recall de-risk.

THE WALL (rank-1, 2026-09-05). The spiking `OneBrainComposer` is CORRECT (6/6, recall==rf, moat clean) and FITS
memory (563 MiB at 404 facts). Its ONLY residual is per-query LATENCY: recall is an O(k_max) LINEAR SCAN over the
co-resident fact blocks -- every recall reconstructs + unbinds + cleans up EVERY stored block (`_read_blocks` ->
`_read_all_blocks` batched, or `[_read_block(i) for i in range(len(kb))]`), so ~114 s / recall at 404 facts, which
is why `k_max` is pinned at 32. This is the single biggest host-shortcut cost on the conversational spine (the
composer runs on every live recall).

IMPORTANT -- WHY THE EXISTING `enable_sparse_index` DOES NOT CLOSE THIS. `OneBrainComposer.enable_sparse_index`
(the DG index, `research/biology/dg-ca3-sparse-index.md`) shards the VOCABULARY axis: it routes each role's
recovered phasor to a small shard of the V-wide concept CODEBOOK so the per-block CLEANUP is O(shard_V) not O(V).
But `_read_blocks_indexed` STILL loops `for i in range(len(self.kb))` -- it decodes EVERY fact block. So the
FACT-COUNT axis (k_max) stays a full linear scan. That is the wall this de-risk attacks: a DG-CA3 sparse index over
the FACT BLOCKS, so a query cue routes to a SMALL SHARD of candidate blocks and recall decodes O(shard) blocks
instead of O(k_max).

WHY THIS IS DISTINCT FROM `ShardedPhasorStore` (the tiered LTM, already sublinear-at-scale). That store IS
sublinear (2026-08-27 finding), but by a HOST-HASH agent-router (a python dict keyed on the cued agent -> one of
~395 shards) -- a DECLARED host shortcut (`scaffold_retired`=0), NOT on the spiking one-brain composer, and its
reverse lookups (`query_agent`) fan out to ALL shards (that finding's residual #2). This de-risk is the BRAIN-BASED
version ON the spiking `OneBrainComposer`: a DG-CA3 sparse index (pattern separation + conjunctive routing, the
`DGSparseIndex` class the composer already imports), routing on the CLEAN cue-word codes, with a PER-ROLE inverted
index so reverse lookups (`query_agent`, cue = action+patient) shard TOO.

WHY IT WILL WORK WHERE THE `RFPhasorComposer` DG PORT FAILED (99.5% escalation, 2026-08-28). That port routed on
the NOISY RECOVERED phasor read off the substrate (sigma=1.27 rad -> misroute). HERE the routing key is the CLEAN
cue-word concept code -- the caller ASSERTS `agent="dog"`, so we look up `comp.comp.concepts["dog"]` (sigma=0, the
exact stored code). The block that stored `agent="dog"` was indexed under that SAME code -> deterministic,
by-construction hit. No recovery noise -> no misroute. This is the key structural difference.

MECHANISM (per-role DG-CA3 inverted index over the fact blocks; reuse-by-import of `DGSparseIndex`).
  * At INDEX-BUILD time (encoding-time role knowledge, legitimate -- the fact's roles are known when it is stored):
    for each MAIN role r in (agent, action, patient), build a `DGSparseIndex` over the (K, D) matrix whose row i is
    the CONCEPT CODE of block i's filler in role r. The bucket-member id IS the block index. DG expansion + hard
    per-band k-WTA + CA3 conjunction routing (`m ~ K^(1/g)` -> O(1) bucket occupancy) -- the SAME class + math the
    vocabulary index uses, applied to the fact-block axis.
  * At QUERY time: for each asserted cue role r with clean word w, route `comp.comp.concepts[w]` -> the DG shard of
    candidate BLOCK indices for that role; INTERSECT the per-role shards (conjunctive cue: the block must carry
    ALL cue roles). The candidate SHARD is a SUPERSET of the true matches BY CONSTRUCTION (block i with the cued
    filler in role r routes to the SAME bucket its filler was stored in -> i is in role r's shard; intersection
    keeps every block carrying all cue roles). Extra collisions are harmless -- they are DECODED and rejected.
  * Then decode ONLY the shard blocks via the composer's EXISTING spiking `_read_block` (reconstruct + unbind +
    cleanup on FIRING NEURONS -- the CA3 pattern-completion, restricted to the routed ensemble), first-match in
    ascending block order (== the full scan's first-match), and read the answer role OFF THE SPIKING DECODE (never
    off the host `kb`).

BRAIN-BASED (honest). The in-shard reconstruct/unbind/cleanup IS the composer's on-substrate op (unchanged, over
fewer blocks) -- the CA3 completion. The DG sparse PROJECTION is the SAME declared host-rate stand-in the vocabulary
index already uses (`research/biology/dg-ca3-sparse-index.md`: fixed random sparse granule bands + hard k-WTA; its
named spiking burn-down is the granule-cell WTA in `_riii_ca3_completion_specificity_derisk.py` /
`cortex_dg_ca3_cleanup_probe.py` / `_gap5_emergent_dg_selection_derisk.py`). NO `sim/` edit; NO composer edit (this
de-risk drives the PUBLIC + existing spiking read methods only). The answer always comes from the spiking decode.

GO CRITERIA (6-seed 42 43 44 100 101 102; wired into the printed VERDICT):
  1. CORRECTNESS PARITY: the sharded recall returns the SAME answer as the full O(k_max) scan for query_patient /
     query_agent / ask_yes_no on every stored-fact probe (a fast-but-wrong index is a NO-GO). Anchored: a sample of
     REAL `comp.query_patient/…` calls must equal the host-matched full reference (proves the reference == the real
     full path).
  2. MOAT PRESERVED: a genuinely out-of-store cue (absent word, or valid words in an unstored combination) ABSTAINS
     under BOTH paths; the sharding introduces 0 new confabulation.
  3. SUBLINEAR COST: blocks DECODED per recall grows sublinearly -- shard (mean/max) << k_max=404 -- and wall-clock
     sharded << full. Report the speedup + the right-sized-bridge projection.

ANTI-CHEATS (wired in):
  (a) CONTENT-ADDRESSABLE: the routing key is the CUE WORD's concept code, never the answer id; the answer is read
      off the spiking `_read_block` decode, never off `kb`. `kb` is touched ONLY at index-build (encoding).
  (b) PARITY vs the full scan is a hard gate (criterion 1).
  (c) SCRAMBLE control: permuting the query's band-winner tuple (routing decorrelated from content) collapses recall
      -- proves the routing is load-bearing, not luck.

Determinism: every RNG seeds from --seeds (cfg.seed discipline). SIM_BACKEND defaults to numpy (CPU; cost-routing --
no GPU brain-load; a GPU re-verify would refine the absolute latency, not the sublinear verdict).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._sparse_indexed_retrieval_derisk import DGSparseIndex  # reuse-by-import (mandated)
from tools.lab import attributable_to  # scramble-control attribution (anti-cheat c)
from tools.verdict import Verdict       # the verdict must travel with its preconditions (gates/verdict_preconditions)

MAIN_ROLES = ("agent", "action", "patient")


# --------------------------------------------------------------------------------------------------
class FactShardIndex:
    """A per-role DG-CA3 sparse index over an OneBrainComposer's stored FACT BLOCKS. Routes a CLEAN cue (asserted
    role words) to a small SHARD of candidate block indices. Reuse-by-import of `DGSparseIndex` (NOT reimplemented)."""

    def __init__(self, composer, roles=MAIN_ROLES, g=2, G=4, c=8, seed=42):
        self.comp = composer
        self.roles = tuple(roles)
        self.g, self.G, self.c, self.seed = int(g), int(G), int(c), int(seed)
        self._idx = {}          # role -> DGSparseIndex over the K block-filler codes (id = block index)
        self._built_K = -1
        self.build_seconds = 0.0

    def build(self):
        """One DGSparseIndex per role over the (K, D) matrix of block filler codes (encoding-time role knowledge:
        the fact's roles are known when it is stored). id i = block index. m ~ K^(1/g) -> O(1) bucket occupancy."""
        t0 = time.time()
        comp = self.comp
        K = len(comp.kb)
        m = max(2, int(np.ceil(max(1, K) ** (1.0 / self.g))))
        self._idx = {}
        for ri, r in enumerate(self.roles):
            codes = np.stack([np.asarray(comp.comp.concepts[comp.kb[i][0][r]], dtype=float)
                              for i in range(K)])                       # (K, D) fractional-cycle phases
            idx = DGSparseIndex(D=comp.D, m=m, g=self.g, G=self.G, c=self.c, seed=self.seed + 101 * (ri + 1))
            idx.build(codes * (2.0 * np.pi))                           # radians convention (== _ensure_dg_index)
            self._idx[r] = idx
        self._built_K = K
        self.build_seconds = time.time() - t0
        return self

    def candidates(self, cue_roles, scramble_rng=None):
        """Route a CLEAN cue {role: word} to the intersected shard of candidate block indices (ascending). Returns
        [] for an absent cue word (moat: no block -> abstain) and None to signal 'cannot route -> escalate'."""
        comp = self.comp
        sets = []
        for r, w in cue_roles.items():
            if r not in self._idx:
                return None                                            # unindexed role -> escalate to full scan
            if w not in comp.comp.concepts:
                return []                                              # absent cue word -> empty shard -> abstain
            code = np.asarray(comp.comp.concepts[w], dtype=float)
            cand = self._idx[r].query(code * (2.0 * np.pi), scramble_rng=scramble_rng)
            sets.append(set(int(x) for x in cand.tolist()))
        if not sets:
            return None
        shard = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return sorted(shard)


# --------------------------------------------------------------------------------------------------
# Sharded recall: route -> decode ONLY the shard blocks via the composer's spiking `_read_block` -> first-match.
def sharded_query_patient(comp, index, agent, action, scramble_rng=None):
    shard = index.candidates({"agent": agent, "action": action}, scramble_rng=scramble_rng)
    if shard is None:
        return comp.query_patient(agent, action), len(comp.kb)         # escalation (not expected on the flat path)
    for i in shard:                                                    # ascending -> first-match == full scan
        got = comp._read_block(i)                                      # SPIKING decode (CA3 completion)
        if got.get("agent") == agent and got.get("action") == action:
            return got.get("patient"), len(shard)
    return None, len(shard)                                            # moat: no shard block matches -> abstain


def sharded_query_agent(comp, index, action, patient, scramble_rng=None):
    shard = index.candidates({"action": action, "patient": patient}, scramble_rng=scramble_rng)
    if shard is None:
        return comp.query_agent(action, patient), len(comp.kb)
    for i in shard:
        got = comp._read_block(i)
        if got.get("action") == action and got.get("patient") == patient:
            return got.get("agent"), len(shard)
    return None, len(shard)


def sharded_ask_yes_no(comp, index, agent, action, patient, scramble_rng=None):
    shard = index.candidates({"agent": agent, "action": action}, scramble_rng=scramble_rng)
    if shard is None:
        return comp.ask_yes_no(agent, action, patient), len(comp.kb)
    for i in shard:
        got = comp._read_block(i)
        if got.get("agent") == agent and got.get("action") == action:
            if got.get("patient") != patient:
                return "unknown", len(shard)                          # (agent,action) block's patient != asserted
            return ("yes" if got.get("polarity") == "AFFIRM" else "no"), len(shard)
    return "unknown", len(shard)                                       # moat: no (agent,action) block -> unknown


# --------------------------------------------------------------------------------------------------
# Full reference (the O(k_max) scan): one batched `_read_blocks()` decode -> host first-match (== the real methods).
def full_reference(rows, kind, a=None, x=None, p=None):
    """rows = comp._read_blocks() (the spiking decode of EVERY block). Replays the composer's own host first-match."""
    if kind == "patient":
        for got in rows:
            if got.get("agent") == a and got.get("action") == x:
                return got.get("patient")
        return None
    if kind == "agent":
        for got in rows:
            if got.get("action") == x and got.get("patient") == p:
                return got.get("agent")
        return None
    if kind == "yesno":
        for got in rows:
            if got.get("agent") == a and got.get("action") == x:
                if got.get("patient") != p:
                    return "unknown"
                return "yes" if got.get("polarity") == "AFFIRM" else "no"
        return "unknown"
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------------
def make_facts(n_facts, seed):
    """n_facts UNIQUE-(agent, action) flat SVO facts drawn from moderate reused pools (realistic co-residence:
    agents/patients recur, so per-role shards are >1 and the conjunctive INTERSECTION is genuinely exercised)."""
    rng = np.random.default_rng(seed + 7)
    n_ag = max(8, n_facts // 3)
    n_ac = max(6, n_facts // 10)
    n_pt = max(8, n_facts // 3)
    agents = [f"agent{i}" for i in range(n_ag)]
    actions = [f"act{i}" for i in range(n_ac)]
    patients = [f"pat{i}" for i in range(n_pt)]
    facts, seen = [], set()
    guard = 0
    while len(facts) < n_facts and guard < n_facts * 100:
        guard += 1
        a = agents[int(rng.integers(n_ag))]
        x = actions[int(rng.integers(n_ac))]
        if (a, x) in seen:
            continue
        seen.add((a, x))
        p = patients[int(rng.integers(n_pt))]
        facts.append((a, x, p))
    vocab = sorted(set(agents + actions + patients))
    return facts, vocab


def asis_n_total(comp):
    """What n_total WOULD be WITH the batched region (the production as-is composer), for the inflation comparison."""
    return int(comp.bat_q_base + comp.k_max * comp.n_roles * comp.D + comp.k_max * comp.cb)


def run_seed(seed, n_facts, D, g, G, c, n_parity, n_moat, n_real_anchor, verbose=True):
    from research.runners.one_brain_composer import OneBrainComposer
    out = {"seed": seed, "n_facts": n_facts, "D": D, "dg": {"g": g, "G": G, "c": c}}
    facts, vocab = make_facts(n_facts, seed)
    out["vocab_V"] = len(vocab)
    out["n_facts_actual"] = len(facts)
    truth = {(a, x): p for (a, x, p) in facts}

    # RIGHT-SIZED composer: no_batched_region drops the k_max*(n_roles*D+cb) batched region -- dead weight for a
    # per-block sharded read. This is BOTH the de-risk enabler (an as-is k_max=420 bridge is ~1.3M neurons -> store +
    # read intractable on CPU) AND a core wire-in lever (a fact-shard composer never batches). Per-block reads are
    # BYTE-IDENTICAL to the default composer (verified: [q_base:c_base+cb] is unchanged).
    t0 = time.time()
    comp = OneBrainComposer(seed=seed, D=D, vocab=vocab, k_max=len(facts) + 16, no_batched_region=True)
    out["n_total"] = int(comp.n_total)
    out["asis_n_total_with_batched_region"] = asis_n_total(comp)
    out["bridge_shrink_ratio"] = out["asis_n_total_with_batched_region"] / out["n_total"]
    t_build = time.time() - t0
    t0 = time.time()
    for (a, x, p) in facts:
        comp.store(a, x, p)
    out["store_seconds"] = time.time() - t0
    out["build_seconds"] = t_build

    index = FactShardIndex(comp, roles=MAIN_ROLES, g=g, G=G, c=c, seed=seed).build()
    out["index_build_seconds"] = index.build_seconds

    # ---- FULL reference = the O(k_max) LINEAR SCAN: decode EVERY block per-block (== the composer's non-batched
    #      recall path when no_batched_region), cache the rows, host first-match. Timed = the full recall latency. ----
    t0 = time.time()
    rows = [comp._read_block(i) for i in range(len(comp.kb))]
    out["full_perblock_scan_seconds"] = time.time() - t0
    out["full_blocks_decoded"] = len(rows)
    out["asis_perblock_seconds"] = out["full_perblock_scan_seconds"] / max(1, len(rows))

    # recall correctness of the FULL path vs GROUND TRUTH (independent of sharding; confirms the composer recalls at
    # this scale). A sharded==full parity below then transfers this correctness to the sharded path.
    rec_ok = sum(1 for i, (a, x, p) in enumerate(facts) if rows[i].get("agent") == a
                 and rows[i].get("action") == x and rows[i].get("patient") == p)
    out["full_recall_vs_truth"] = [rec_ok, len(facts)]

    rng = np.random.default_rng(seed + 999)
    idxs = list(range(len(facts)))
    rng.shuffle(idxs)

    # ---- PARITY (criterion 1): sharded == full reference, all 3 query kinds ----
    parity = {"patient": [0, 0], "agent": [0, 0], "yesno": [0, 0]}   # [ok, total]
    shard_sizes = []
    lat_shard = {"patient": [], "agent": [], "yesno": []}
    mism = []
    for k in idxs[:n_parity]:
        a, x, p = facts[k]
        # patient
        ref = full_reference(rows, "patient", a=a, x=x)
        t0 = time.time(); sh, ssz = sharded_query_patient(comp, index, a, x); lat_shard["patient"].append(time.time() - t0)
        shard_sizes.append(ssz)
        parity["patient"][1] += 1; parity["patient"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("patient", a, x, p, ref, sh))
        # agent (reverse lookup: cue = action+patient)
        ref = full_reference(rows, "agent", x=x, p=p)
        t0 = time.time(); sh, ssz = sharded_query_agent(comp, index, x, p); lat_shard["agent"].append(time.time() - t0)
        shard_sizes.append(ssz)
        parity["agent"][1] += 1; parity["agent"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("agent", a, x, p, ref, sh))
        # yes/no
        ref = full_reference(rows, "yesno", a=a, x=x, p=p)
        t0 = time.time(); sh, ssz = sharded_ask_yes_no(comp, index, a, x, p); lat_shard["yesno"].append(time.time() - t0)
        shard_sizes.append(ssz)
        parity["yesno"][1] += 1; parity["yesno"][0] += int(sh == ref)
        if sh != ref:
            mism.append(("yesno", a, x, p, ref, sh))
    out["parity"] = parity
    out["parity_mismatches"] = mism[:20]
    out["shard_size_mean"] = float(np.mean(shard_sizes)) if shard_sizes else None
    out["shard_size_max"] = int(np.max(shard_sizes)) if shard_sizes else None
    out["latency_shard_median_seconds"] = {k: (float(np.median(v)) if v else None) for k, v in lat_shard.items()}

    # ---- ANCHOR: REAL public-API comp.query_* == the host-matched reference (proves the reference IS the real full
    #      path). Expensive (query_patient runs the per-block scan twice), so gated to n_real_anchor calls -- run only
    #      on the first seed via main(); the other seeds rest on the structural identity (same `_read_block` + first-
    #      match) + the recall-vs-truth check. Also cross-checks query_agent + ask_yes_no once. ----
    anchor = {"checked": 0, "ok": 0}
    lat_full_real = []
    for k in idxs[:n_real_anchor]:
        a, x, p = facts[k]
        t0 = time.time(); real = comp.query_patient(a, x); lat_full_real.append(time.time() - t0)
        anchor["checked"] += 1; anchor["ok"] += int(real == full_reference(rows, "patient", a=a, x=x))
        real_ag = comp.query_agent(x, p)
        anchor["checked"] += 1; anchor["ok"] += int(real_ag == full_reference(rows, "agent", x=x, p=p))
        real_yn = comp.ask_yes_no(a, x, p)
        anchor["checked"] += 1; anchor["ok"] += int(real_yn == full_reference(rows, "yesno", a=a, x=x, p=p))
    out["real_full_anchor"] = anchor
    out["latency_full_real_query_patient_median_seconds"] = float(np.median(lat_full_real)) if lat_full_real else None

    # ---- MOAT (criterion 2): out-of-store cues abstain under both; 0 new confab ----
    moat = {"checked": 0, "both_abstain": 0, "new_confab": 0}
    stored_pairs = set((a, x) for (a, x, p) in facts)
    words = vocab
    mrng = np.random.default_rng(seed + 4242)
    tries = 0
    while moat["checked"] < n_moat and tries < n_moat * 200:
        tries += 1
        if tries % 3 == 0:
            a = f"__absent_agent_{tries}__"; x = words[int(mrng.integers(len(words)))]   # absent word cue
        else:
            a = words[int(mrng.integers(len(words)))]; x = words[int(mrng.integers(len(words)))]  # valid words, unstored combo
            if (a, x) in stored_pairs:
                continue
        ref = full_reference(rows, "patient", a=a, x=x)                # full-scan answer (should be None)
        sh, _ = sharded_query_patient(comp, index, a, x)
        moat["checked"] += 1
        moat["both_abstain"] += int(ref is None and sh is None)
        moat["new_confab"] += int(ref is None and sh is not None)     # sharding invented an answer the full scan did not
    out["moat"] = moat

    # ---- SCRAMBLE control (anti-cheat c): permuted band-winners (routing decorrelated from content) -> recall
    #      collapses. Measure REAL vs SCRAMBLED recall on the IDENTICAL query set, then ATTRIBUTE the recall to the
    #      content-derived routing via tools.lab.attributable_to (treatment=real, control=scrambled). ~100% => the
    #      routing is load-bearing, not luck; a low fraction would mean the shard recalls even with random routing. ----
    scr = {"checked": 0, "recovered": 0, "real_recovered": 0}
    srng = np.random.default_rng(seed + 31337)
    for k in idxs[:min(n_parity, 30)]:
        a, x, p = facts[k]
        ref = full_reference(rows, "patient", a=a, x=x)
        sh_s, _ = sharded_query_patient(comp, index, a, x, scramble_rng=srng)
        sh_r, _ = sharded_query_patient(comp, index, a, x)                # same query, REAL (content) routing
        scr["checked"] += 1
        scr["recovered"] += int(sh_s == ref and ref is not None)         # scrambled arm (control)
        scr["real_recovered"] += int(sh_r == ref and ref is not None)    # real arm (treatment)
    real_rate = scr["real_recovered"] / max(1, scr["checked"])
    scr_rate = scr["recovered"] / max(1, scr["checked"])
    scr["real_rate"] = real_rate
    scr["scrambled_rate"] = scr_rate
    scr["attributable_to_routing"] = attributable_to(
        "fact-shard recall: content routing vs scrambled", real_rate, scr_rate)
    out["scramble_control"] = scr

    # (the composer is now right-sized via no_batched_region, so the measured sharded latency IS the achievable one --
    #  no separate projection needed.)

    # ---- per-seed GO ----
    par_ok = all(parity[k][0] == parity[k][1] and parity[k][1] > 0 for k in parity)
    moat_ok = (moat["new_confab"] == 0 and moat["checked"] > 0)
    anchor_ok = (anchor["checked"] == 0) or (anchor["ok"] == anchor["checked"])  # pass if not run this seed
    scramble_ok = (scr["recovered"] < 0.5 * scr["checked"]) if scr["checked"] else False
    sublinear_ok = (out["shard_size_mean"] is not None and out["shard_size_mean"] < 0.25 * len(rows))
    recall_ok = (out["full_recall_vs_truth"][0] == out["full_recall_vs_truth"][1])
    out["speedup_full_over_shard"] = (out["full_perblock_scan_seconds"]
                                      / max(1e-9, out["latency_shard_median_seconds"]["patient"])
                                      if out["latency_shard_median_seconds"]["patient"] else None)
    out["go_flags"] = {"parity": par_ok, "moat": moat_ok, "real_anchor": anchor_ok,
                       "scramble": scramble_ok, "sublinear": sublinear_ok, "full_recall": recall_ok}
    out["go"] = bool(par_ok and moat_ok and anchor_ok and scramble_ok and sublinear_ok and recall_ok)
    if verbose:
        print(f"[seed {seed}] V={len(vocab)} facts={len(facts)} n_total={comp.n_total} "
              f"(as-is w/batched={out['asis_n_total_with_batched_region']}, shrink={out['bridge_shrink_ratio']:.1f}x)  "
              f"store={out['store_seconds']:.0f}s idx={index.build_seconds:.2f}s")
        print(f"  parity P/A/YN={parity['patient']} {parity['agent']} {parity['yesno']}  "
              f"full_recall_vs_truth={out['full_recall_vs_truth']}  "
              f"moat new_confab={moat['new_confab']}/{moat['checked']}  anchor={anchor['ok']}/{anchor['checked']}  "
              f"scramble_recovered={scr['recovered']}/{scr['checked']}")
        print(f"  shard mean/max={out['shard_size_mean']:.2f}/{out['shard_size_max']} vs full={len(rows)} blocks | "
              f"full_scan={out['full_perblock_scan_seconds']:.1f}s ({out['asis_perblock_seconds']:.2f}s/block)  "
              f"shard_median={out['latency_shard_median_seconds']['patient']:.2f}s  "
              f"speedup={out['speedup_full_over_shard']:.0f}x")
        print(f"  GO={out['go']}  flags={out['go_flags']}")
    return out


def build_verdict(summary):
    """Earn the verdict via tools.verdict.Verdict so it travels with its preconditions into the artifact
    (gates/verdict_preconditions). Pure function of the already-measured per-seed results -> a re-decide on the
    same JSON is identical (the --finalize path relies on this: it re-earns the verdict from a run's own
    measurements without re-running the sims). Every precondition here is a HARD GO requirement."""
    agg = summary["aggregate"]
    per = summary["per_seed"]
    v = Verdict("onebrain fact-block DG-CA3 sublinear spiking retrieval @ %d facts" % summary["n_facts"])
    v.require("parity: sharded == full O(k_max) scan (all seeds, patient/agent/yes-no)",
              bool(agg["parity_all_ok"]), expect=True)
    v.require("moat: 0 new confabulation on out-of-store cues (all seeds)",
              bool(agg["moat_all_ok"]), expect=True)
    v.require("full recall == ground truth at scale (all seeds)",
              bool(agg["full_recall_all_ok"]), expect=True)
    v.require("real public-API anchor == full reference (query_patient/query_agent/ask_yes_no)",
              bool(agg["anchor_ok_where_run"]), expect=True)
    v.require("sublinear: shard mean << k_max (all seeds)",
              bool(agg["sublinear_all_ok"]), expect=True)
    real_rates = [p["scramble_control"]["real_rate"] for p in per if "scramble_control" in p]
    scr_rates = [p["scramble_control"]["scrambled_rate"] for p in per if "scramble_control" in p]
    if real_rates and scr_rates:
        v.control("scramble control: content-derived routing vs scrambled routing recall",
                  treatment=float(np.mean(real_rates)), control=float(np.mean(scr_rates)))
    v.require("SIM_BACKEND == numpy (declared CPU de-risk backend)",
              os.environ.get("SIM_BACKEND", "").lower() == "numpy", expect=True)
    go = all(c.ok for c in v.checks)
    return v.decide(go=go, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-facts", type=int, default=404)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--g", type=int, default=2)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--n-parity", type=int, default=40)
    ap.add_argument("--n-moat", type=int, default=20)
    ap.add_argument("--n-real-anchor", type=int, default=1, help="REAL public-API calls (first seed only); expensive")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny scale (fast mechanism check)")
    ap.add_argument("--finalize", type=str, default=None,
                    help="re-earn the verdict+preconditions from an existing results JSON (its own measurements) "
                         "and rewrite it, WITHOUT re-running the sims (measurements are deterministic + preserved)")
    args = ap.parse_args()
    if args.finalize:
        with open(args.finalize) as f:
            summary = json.load(f)
        decided = build_verdict(summary)
        summary["verdict"] = decided["status"]
        summary["preconditions"] = decided["preconditions"]
        summary["verdict_block"] = decided
        outp = args.out or args.finalize
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        print("finalized verdict=%s preconditions=%d -> %s" % (decided["status"], len(decided["preconditions"]), outp))
        return 0 if decided["go"] else 1
    if args.smoke:
        args.n_facts = 40; args.seeds = [42, 43]; args.n_parity = 20; args.n_moat = 12; args.n_real_anchor = 1

    results = []
    for si, s in enumerate(args.seeds):
        anchor_n = args.n_real_anchor if si == 0 else 0    # anchor only on the first seed (query_patient is ~2 full scans)
        results.append(run_seed(s, args.n_facts, args.D, args.g, args.G, args.c,
                                args.n_parity, args.n_moat, anchor_n))
    go_all = all(r["go"] for r in results)
    anchored = [r for r in results if r["real_full_anchor"]["checked"] > 0]
    summary = {
        "verdict": "GO" if go_all else "NO-GO",
        "n_seeds": len(results),
        "seeds": args.seeds,
        "n_facts": args.n_facts,
        "D": args.D,
        "dg": {"g": args.g, "G": args.G, "c": args.c},
        "per_seed": results,
        "aggregate": {
            "parity_all_ok": all(r["go_flags"]["parity"] for r in results),
            "moat_all_ok": all(r["go_flags"]["moat"] for r in results),
            "full_recall_all_ok": all(r["go_flags"]["full_recall"] for r in results),
            "anchor_ok_where_run": all(r["go_flags"]["real_anchor"] for r in results) and len(anchored) > 0,
            "scramble_all_ok": all(r["go_flags"]["scramble"] for r in results),
            "sublinear_all_ok": all(r["go_flags"]["sublinear"] for r in results),
            "shard_size_mean": float(np.mean([r["shard_size_mean"] for r in results if r["shard_size_mean"] is not None])),
            "shard_size_max": int(np.max([r["shard_size_max"] for r in results if r["shard_size_max"] is not None])),
            "full_perblock_scan_seconds_median": float(np.median([r["full_perblock_scan_seconds"] for r in results])),
            "shard_patient_latency_median": float(np.median(
                [r["latency_shard_median_seconds"]["patient"] for r in results
                 if r["latency_shard_median_seconds"]["patient"] is not None])),
            "speedup_full_over_shard_median": float(np.median(
                [r["speedup_full_over_shard"] for r in results if r["speedup_full_over_shard"] is not None])),
            "bridge_shrink_ratio": float(np.median([r["bridge_shrink_ratio"] for r in results])),
        },
    }
    decided = build_verdict(summary)                        # earn the verdict WITH its preconditions
    summary["verdict"] = decided["status"]
    summary["preconditions"] = decided["preconditions"]
    summary["verdict_block"] = decided
    print("\n=== VERDICT ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"VERDICT: {summary['verdict']}  ({sum(r['go'] for r in results)}/{len(results)} seeds GO)")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if go_all else 1


if __name__ == "__main__":
    sys.exit(main())
