# Capability: Lifelong learning
# Record: brain/research/2026-11-27-sleep-drift-recipe-check.md
# Expected-fail: partly. A screen of four sleep-recipe constants on the K-P slept store plus two placebo nights, 3 DEV seeds,
#   3 chained nights, 1.5B CPU. A negative is a verdict on these recipe changes at this scale, not on consolidation without
#   drift. The placebo arms bracket the content-independent part of the drift (A5s is biased DOWN, see the note); they do
#   not measure it exactly. Nothing here flips a default.
"""Offline mix-night screen. Switch-free: this file only monkeypatches brain.brain.replay_into_cortex and
Brain._night_episodes inside a context manager (restored on exit); brain/sleep.py, brain/brain.py and every default are
untouched. Nothing runs without POOL_NODE=1 (or BRAIN_ALLOW_W3090=1). The rule is brain/research/2026-11-27-mix-night-prereg.md
(REGISTERED 2026-11-27 after pre-data review 2); the design and record brief are brain/research/2026-11-27-sleep-drift-recipe-check.md.

Population: the KR night snapshots (life_kw --night-snapshots; s319/s320/s321). Start = snapshot n12 (adapters, keep bank).
Night k (13..15) is slept on the store of snapshot n<k> (episodes.json: text, strength, uses, last_used, created), at the
clock of that snapshot's meta last_sleep, with the sleep counter set to k-1, the seed rule of Brain.sleep
(sleep_seed + sleeps) and the arm's OWN adapters carried forward (the arm's chain, not the life's). What the life's
trajectory contributes to a night is therefore the store text and its strengths/uses/last_used only (the sleep reads
nothing else of an episode: no keys, no deltas); keys and strengths come from the life's adapter state, identical across
arms, and are declared as such in the note.

RNG decoupling (the check of 11-27): in brain/sleep.py ONE random.Random(seed) draws the episode (rng.choices) and the keep
text (rng.randrange(len(generic))) in turn, so any change to the keep set size or the keeps per step re-times the episode
stream too. `mix_sleep(loop="split")` gives the episode stream and the keep-index stream their own generators; the shipped
subset is kept, and a larger subset is that same 64 plus extras drawn from a third generator (nested). loop="legacy" is an
exact replica of brain.sleep.sleep (tested equal) and is used only by arm A0L, the instrument check against the life.

    POOL_NODE=1 python -m brain.experiments.mix_night_screen run --seed 319 --arm A0 --life L.json --snaps DIR \\
        --offframe F.json --offframe-sha256 HEX --out-dir D [--model 1.5b] [--device cpu] [--foreign-snaps DIR2]
    python -m brain.experiments.mix_night_screen score --dir D --lives L1.json L2.json L3.json [--out verdict.json]
    python -m brain.experiments.mix_night_screen strata --snaps DIR [--min-n 5]     (store-only; no model, no guard)
    python -m brain.experiments.mix_night_screen estimate [--night-min 25 --read-min 10]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import time

import torch
import torch.nn.functional as F

SEEDS = (319, 320, 321)
START_NIGHT = 12
NIGHTS = (13, 14, 15)
END = 15
KEEP_RNG_OFFSET = 31337        # the keep-index stream: seed + this (+ the arm's keep_add)
EXTRA_SUBSET_OFFSET = 1        # extras of a larger subset: Random(sleep_seed + sleeps + 104729 + this)
PLACEBO_SEED = 20261127
RET_SEED = 20261128
RET_PER_STRATUM = 20
REPLY_TOKENS = 40
STRATA = (("new", 0.0, 3.0), ("mid", 3.0, 5.0), ("old", 5.0, float("inf")))   # age in clock days (Amendment S1: own+recent merged)
P2_STRATA = ("new", "old")
P2_MIN_N = 5                   # a P2 stratum read on fewer texts (or none) is UNREAD: the candidate is INCONCLUSIVE, never a P2 pass
REF_CACHE_MAX = 64             # keep sets up to this size hold their frozen reference log-probs; larger (A3) recompute per step
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFFRAME_FILE = os.path.join(REPO, "brain", "eval", "mix_night_offframe.json")
OFFFRAME_SHA256 = "b1a95684b8caeec1f6564493afa35a6cddc280d791f8466a2c73ccbc49e6f268"   # registered in the prereg

# arm table. Defaults: split RNG, the shipped K-P night, 3 chained nights.
DEFAULT = {"loop": "split", "ep_add": 0, "keep_add": 0, "ce_keep_w": 0.0, "keep_per_step": 1, "subset_n": None,
           "kl_weight": None, "placebo": None, "nights": 3}
ARMS = {
    "A0L": {"loop": "legacy", "nights": 1},       # exact shipped loop: instrument check vs the life (night 13 only)
    "A0": {},                                     # shipped night, decoupled streams: the baseline of every contrast
    "A0r": {"nights": 1},                         # identical rerun of A0 night 13: determinism check
    "A0e": {"ep_add": 1000},                      # noise: episode stream only
    "A0k": {"keep_add": 1000},                    # noise: keep-index stream only
    "A1": {"ce_keep_w": 1.0},                     # literal REMIX: + 1.0 x CE on the same keep text
    "A2": {"keep_per_step": 2},                   # 2 keep texts per step, MEAN keep (dose fixed: variance lever only)
    "A3": {"subset_n": 256},                      # nightly subset 64 -> 256 (the shipped 64 plus 192, nested)
    "A4": {"kl_weight": 6.0},                     # dose
    "A5s": {"placebo": "self"},                   # episodes -> unused bank texts (the cortex's own samples)
    "A5f": {"placebo": "foreign"},                # episodes -> another life's episodes (non-self, lived register)
}
CANDIDATES = ("A1", "A2", "A3", "A4")
NOISE_ARMS = ("A0e", "A0k")

# Registered pass rule (brain/research/2026-11-27-mix-night-prereg.md section 5; REGISTERED 2026-11-27).
P1_RATIO, P1_NOISE_MULT = 0.75, 2.0
P1B_RATIO, P1B_SEEDS = 0.85, 2
P2_MARGIN = 0.10
P3_MARGIN = 0.05
KILL_SEEDS = 2
K_SANE_TOL = 0.25
DETERMINISM_TOL = 1e-3         # relative KL difference A0r vs A0 at n13
MOTION_RATIO, CONTENT_RATIO, PLACEBO_SEEDS = 0.6, 0.25, 2
# Noise-floor minimum (pre-data review 2, B2): noise_s = max(raw two-draw floor, NOISE_FLOOR_REL x KL(A0, n15)_s). Two draws
# can land near A0 by chance and make "> 2 x noise" trivial. 0.05 is set before data from the record: the smallest one-night
# life change at this point of the KR lives, s319 n12 -> n13 +0.015 on 0.285 (5.3%).
NOISE_FLOOR_REL = 0.05
# Engagement (pre-data review 2, B1): what each arm's sleep must have REALIZED on every night unit (sleep_info), or the
# arm is void (INCONCLUSIVE), never scored: a KILL on a null arm would be a verdict on nothing.
ENGAGE = {"A1": "mean_ce_keep > 0 and ce_keep_w == 1.0", "A2": "keep_per_step == [2]", "A3": "keep_n == 256",
          "A4": "kl_weight == 6.0", "A5s": "placebo_pool > 0, placebo_excluded recorded, placebo_overlap == 0",
          "A5f": "placebo_pool > 0, placebo_excluded recorded, placebo_overlap == 0"}

PROMPTS = [   # 24 ordinary prompts, none from the life banks (test_prompts_not_in_banks checks)
    "What time does the sun usually set in winter?", "Can you suggest a quiet hobby for the evening?",
    "How do I keep bread fresh for longer?", "Tell me something nice about autumn.",
    "What is a good way to learn a new language?", "Why is the sky blue?",
    "Can you recommend a short walk after dinner?", "How many days are there in a leap year?",
    "What should I pack for a day at the beach?", "Give me a tip for sleeping better.",
    "What is the difference between a lake and a pond?", "How do I boil an egg?",
    "Can you describe a calm morning?", "Is it better to read in the morning or at night?",
    "What do bees do in the winter?", "How can I stay focused while studying?",
    "What is a simple dinner I can make in twenty minutes?", "Why do cats purr?",
    "What does a lighthouse do?", "Can you explain what a budget is?",
    "How far is a mile compared with a kilometre?", "What is a good name for a small boat?",
    "How do plants know which way is up?", "Say something encouraging before an exam."]
REFUSAL = re.compile(r"\bI(?:'|’)?m sorry\b|\bI am sorry\b|\bI don(?:'|’)?t know what you mean\b"
                     r"|\bI do not know what you mean\b", re.I)
NAMEPAT = re.compile(r"\bmy name is\b", re.I)


# ------------------------------------------------------------------ guard and provenance
def guard(env=None):
    """Heavy run only on a pool node (POOL_NODE=1) or with the W 3090 explicitly allowed (BRAIN_ALLOW_W3090=1)."""
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("mix_night_screen run: refuses to run unless POOL_NODE=1 (pool node) or BRAIN_ALLOW_W3090=1")


def precision_of(device: str, cuda_precision: str) -> str:
    return cuda_precision if str(device).startswith("cuda") else "fp32"


def provenance(device="?", precision="fp32", model=None, env=None, repo=REPO) -> dict:
    """host, commit, device, precision, model. On a pool node (POOL_NODE=1) the commit is READ from <repo>/COMMIT (the
    route_repo copy, ~/brain-lane/route_repo/COMMIT, is a git-archive copy, so `git rev-parse` there would report nothing or
    an enclosing repo); a missing or empty COMMIT file refuses the run. Elsewhere `git rev-parse HEAD` in the repo."""
    env = os.environ if env is None else env
    if env.get("POOL_NODE") == "1":
        cf = os.path.join(repo, "COMMIT")
        commit = open(cf).read().strip() if os.path.exists(cf) else ""
        if not commit:
            raise SystemExit(f"pool node: {cf} missing or empty; ship the code with its COMMIT file")
        src = "COMMIT"
    else:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo).stdout.strip()
        except OSError:
            commit = ""
        src = "git"
        if not commit:
            raise SystemExit(f"no git commit readable in {repo}")
    return {"host": socket.gethostname(), "commit": commit, "commit_source": src, "device": str(device),
            "precision": precision, "model": model, "threads": torch.get_num_threads()}


SNAP_FILES = ("episodes.json", "keys.pt", "meta.json", "keep_bank.json")   # per reading night; keep_bank.json: review defect 1
PROV_KEYS = ("host", "commit", "device", "precision", "model", "threads", "offframe_sha256")   # threads: determinism fallback


def same_provenance(u: dict, prov: dict, off_sha: str) -> bool:
    want = {**prov, "offframe_sha256": off_sha}
    return all(u.get(k) == want.get(k) for k in PROV_KEYS)


def _resume(path: str, prov: dict, off_sha: str) -> dict:
    u = json.load(open(path))
    if not same_provenance(u, prov, off_sha):
        got = {k: u.get(k) for k in PROV_KEYS}
        raise SystemExit(f"{path}: written under {got}, now {({**prov, 'offframe_sha256': off_sha})}; use a fresh out dir "
                         f"(no mixing)")
    return u


def atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def arm_cfg(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return {**DEFAULT, **ARMS[arm]}


# ------------------------------------------------------------------ the keep subset (shipped rule, nested extension)
def shipped_subset(bank: list, sleep_seed: int, sleeps: int) -> list:
    """What Brain.sleep draws: random.Random(sleep_seed + sleeps + KEEP_BANK_SEED_OFFSET).sample(bank, min(KEEP_BANK_NIGHT,
    len(bank))), the constants read from brain.brain at call time."""
    import brain.brain as bb
    return random.Random(sleep_seed + sleeps + bb.KEEP_BANK_SEED_OFFSET).sample(bank, min(bb.KEEP_BANK_NIGHT, len(bank)))


def keep_set(bank: list, sleep_seed: int, sleeps: int, n: int | None) -> list:
    """The night's keep set: n None (or <= the shipped size) = the shipped draw as is; a larger n = the shipped draw first
    (identical to the life's), then n - shipped extras drawn without replacement from the rest of the bank by their own
    generator (nested, so the shipped keep texts stay a subset of the arm's)."""
    import brain.brain as bb
    base = shipped_subset(bank, sleep_seed, sleeps)
    if n is None or n <= len(base):
        return base
    rest = [t for t in bank if t not in set(base)]
    extra = random.Random(sleep_seed + sleeps + bb.KEEP_BANK_SEED_OFFSET + EXTRA_SUBSET_OFFSET).sample(
        rest, min(n - len(base), len(rest)))
    return base + extra


# ------------------------------------------------------------------ the sleep loop
def _ids(cortex, text):
    return cortex.pipe.encode(text)


def mix_sleep(plastic, base_state, episodes, generic, *, steps=200, lr=5e-4, kl_weight=1.0, seed=0, log=None, guard=(),
              guard_weight=None, loop="split", ep_add=0, keep_add=0, ce_keep_w=0.0, keep_per_step=1, trace=None,
              ref_cache=None, stats=None):
    """brain.sleep.sleep with extra arm options and optional RNG decoupling. generic: list of str (K-P only, no guard).
    loop="legacy": ONE random.Random(seed) draws episode then keep index exactly as brain/sleep.py (replica, tested equal to
    it; ep_add / keep_add / keep_per_step must be 0 / 0 / 1).
    loop="split": the episode stream is Random(seed + ep_add) and the keep-index stream Random(seed + KEEP_RNG_OFFSET +
    keep_add), so no keep option moves the episode sequence and no episode option moves the keep sequence.
    ce_keep_w > 0 adds ce_keep_w x CE(keep text) (mean over the step's keep texts). keep_per_step > 1: that many keep texts
    per step, KL and CE MEAN-reduced (the dose stays kl_weight; the sum would equal kl_weight x k in expectation = arm A4).
    trace (a list) receives (episode index, tuple of keep indices) per step.
    ref_cache: None = hold the frozen reference log-probs of every keep text when the set has <= REF_CACHE_MAX texts (the
    shipped path), else recompute each step's reference from the frozen cortex (adapters off, no grad, no RNG draw): the same
    numbers without holding 256 x tokens x 65536 floats (about 4.5 GB at 1.5B) for A3. True / False force either path.
    stats (a dict) receives what the loop REALIZED (pre-data review 2, B1: the engagement gate reads it): keep_n (texts in
    the keep set), kl_weight, ce_keep_w, keep_per_step (the set of keep texts drawn per step, as observed), steps and
    mean_ce_keep (mean of the CE keep term as added to the loss; None when ce_keep_w is 0). The returned losses are
    unchanged (the legacy replica stays equal to brain/sleep.py)."""
    if guard:
        raise ValueError("mix_sleep: the near-miss guard is not part of the K-P screen")
    if any(not isinstance(g, str) for g in generic):
        raise ValueError("mix_sleep: K-P-U (kept-position) keep texts are not part of the screen")
    if loop not in ("legacy", "split"):
        raise ValueError(loop)
    if loop == "legacy" and (ep_add or keep_add or keep_per_step != 1):
        raise ValueError("legacy loop takes no ep_add / keep_add / keep_per_step")
    rng = random.Random(seed + (ep_add if loop == "split" else 0))
    krng = rng if loop == "legacy" else random.Random(seed + KEEP_RNG_OFFSET + keep_add)
    torch.manual_seed(seed)
    c = plastic.cortex
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    weights = [max(s, 1e-3) for _, s in episodes]
    gen_ids = [_ids(c, g) for g in generic]
    cache = (len(gen_ids) <= REF_CACHE_MAX) if ref_cache is None else bool(ref_cache)

    def ref_of(ids):
        plastic.enabled = False
        try:
            with torch.no_grad():
                return F.log_softmax(plastic(ids, base_state)[0], -1)
        finally:
            plastic.enabled = True

    ref = [ref_of(ids) for ids in gen_ids] if cache else None
    losses, ce_vals, kps = [], [], set()
    for step in range(steps):
        ei = rng.choices(range(len(episodes)), weights=weights)[0]
        text = episodes[ei][0]
        ids = _ids(c, text)
        logits, _ = plastic(ids, base_state)
        replay = F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=logits.device))
        js = tuple(krng.randrange(len(gen_ids)) for _ in range(keep_per_step))
        kls, ces = [], []
        for j in js:
            g_logits, _ = plastic(gen_ids[j], base_state)
            r = ref[j] if cache else ref_of(gen_ids[j])
            kls.append(F.kl_div(F.log_softmax(g_logits, -1), r, log_target=True, reduction="batchmean"))
            if ce_keep_w:
                ces.append(F.cross_entropy(g_logits[:-1], torch.tensor(gen_ids[j][1:], device=g_logits.device)))
        keep = kls[0] if len(kls) == 1 else sum(kls) / len(kls)
        loss = replay + kl_weight * keep
        if ce_keep_w:
            ce_term = ces[0] if len(ces) == 1 else sum(ces) / len(ces)
            loss = loss + ce_keep_w * ce_term
            ce_vals.append(float(ce_term))
        kps.add(len(js))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append((float(replay), float(keep)))
        if trace is not None:
            trace.append((ei, js))
        if log and step % 20 == 0:
            log(f"sleep step {step}: replay {float(replay):.3f} keep {float(keep):.4f}")
    if stats is not None:
        stats.update(keep_n_loop=len(gen_ids), kl_weight=float(kl_weight), ce_keep_w=float(ce_keep_w),
                     keep_per_step=sorted(kps), steps_run=len(losses),
                     mean_ce_keep=(sum(ce_vals) / len(ce_vals)) if ce_vals else None)
    return losses


# ------------------------------------------------------------------ placebo texts
def placebo_pool(candidates: list, exclude: list, seed: int) -> list:
    """candidates minus any text in `exclude` (compared stripped), in a seeded shuffle. Returns (pool, n_excluded)."""
    ex = {t.strip() for t in exclude}
    kept = [t for t in candidates if t.strip() not in ex]
    random.Random(seed).shuffle(kept)
    return kept, len(candidates) - len(kept)


def assign_placebo(n: int, pool: list) -> list:
    """The i-th episode gets the i-th pool text (cycling when the pool is shorter); trailing blank lines are stripped
    because Brain.sleep appends them."""
    if not pool:
        raise ValueError("empty placebo pool")
    return [pool[i % len(pool)].rstrip("\n") for i in range(n)]


# ------------------------------------------------------------------ patching Brain.sleep for one arm-night
@contextlib.contextmanager
def patched(b, cfg: dict, foreign_texts: list | None = None, info: dict | None = None):
    """Route this Brain's next sleep() through mix_sleep with the arm's options. Restored on exit."""
    import brain.brain as bb
    real_rep, real_ne = bb.replay_into_cortex, bb.Brain._night_episodes
    info = {} if info is None else info

    def rep(plastic, base, eps, generic, *, steps, lr, kl_weight, seed, log=None, guard=(), guard_weight=None):
        bank = b._keep_bank()
        sleeps = b.meta.get("sleeps", 0)
        gen = keep_set(bank, b.sleep_seed, sleeps, cfg["subset_n"])
        assert gen[:len(generic)] == list(generic), "the shipped keep subset must be the prefix of the arm's keep set"
        info.update(keep_n=len(gen), keep_hash=hashlib.sha1("\x00".join(gen).encode()).hexdigest()[:12],
                    seed=seed, sleeps=sleeps, steps=steps)
        stats = {}
        out = mix_sleep(plastic, base, eps, gen, steps=steps, lr=lr,
                        kl_weight=cfg["kl_weight"] if cfg["kl_weight"] is not None else kl_weight, seed=seed, log=log,
                        guard=guard, guard_weight=guard_weight, loop=cfg["loop"], ep_add=cfg["ep_add"],
                        keep_add=cfg["keep_add"], ce_keep_w=cfg["ce_keep_w"], keep_per_step=cfg["keep_per_step"],
                        stats=stats)
        info.update(stats)          # realized values: the engagement gate (B1) reads them from sleep_info
        return out

    def night_eps(self):
        eps = real_ne(self)
        if cfg["placebo"] is None:
            return eps
        bank = self._keep_bank()
        sleeps = self.meta.get("sleeps", 0)
        own = [e.text for e in eps]
        if cfg["placebo"] == "self":
            pool, nex = placebo_pool(bank, shipped_subset(bank, self.sleep_seed, sleeps), PLACEBO_SEED + sleeps)
        else:
            pool, nex = placebo_pool(foreign_texts or [], own, PLACEBO_SEED + sleeps)
        assigned = assign_placebo(len(eps), pool)
        own_set = {t.strip() for t in own}
        # placebo_overlap: replayed texts that are in the night's real store (must be 0 for the sham to be a sham; B1)
        info.update(placebo=cfg["placebo"], placebo_pool=len(pool), placebo_excluded=nex,
                    placebo_overlap=sum(t.strip() in own_set for t in assigned), placebo_n=len(assigned))
        out = []
        for e, t in zip(eps, assigned):
            e2 = copy.copy(e)
            e2.text = t
            out.append(e2)
        return out

    bb.replay_into_cortex = rep
    bb.Brain._night_episodes = night_eps
    try:
        yield info
    finally:
        bb.replay_into_cortex, bb.Brain._night_episodes = real_rep, real_ne


# ------------------------------------------------------------------ readings
def stratum(age_days: float) -> str:
    for name, lo, hi in STRATA:
        if lo <= age_days < hi:
            return name
    return "new"      # negative age (clock skew): the night's own


def sample_strata(created: list, now: float, k: int, per: int = RET_PER_STRATUM) -> dict:
    """{stratum: sorted episode indices}: up to `per` per stratum, drawn by Random(RET_SEED + k) from the night's store, so
    every arm of a seed reads the same texts."""
    by = {n: [] for n, _, _ in STRATA}
    for i, c in enumerate(created):
        by[stratum((now - c) / 86400.0)].append(i)
    rng = random.Random(RET_SEED + k)
    return {n: sorted(rng.sample(v, min(per, len(v)))) for n, v in by.items()}


def reply_flags(replies: list) -> dict:
    n = max(len(replies), 1)
    return {"refusal": sum(bool(REFUSAL.search(r)) for r in replies) / n,
            "name": sum(bool(NAMEPAT.search(r)) for r in replies) / n}


def read_retention(b, now: float, k: int) -> dict:
    """Mean per-token NLL of sampled store texts by age stratum, adapters on, no hippocampus, no RNG draw."""
    eps = b.hippocampus.episodes
    pick = sample_strata([e.created for e in eps], now, k)
    p, base = b.cortex.plastic, [s.float() for s in b.base]
    out = {}
    was = p.enabled
    try:
        p.enabled = True
        with torch.no_grad():
            for name, idx in pick.items():
                vals = []
                for i in idx:
                    ids = b.cortex.pipe.encode(f"{eps[i].text}\n\n")
                    lg = p(ids, base)[0].float()
                    vals.append(float(F.cross_entropy(lg[:-1], torch.tensor(ids[1:], device=lg.device))))
                out[name] = {"n": len(vals), "nll": (sum(vals) / len(vals)) if vals else None,
                             "under1": (sum(v < 1.0 for v in vals) / len(vals)) if vals else None}
    finally:
        p.enabled = was
    return out


def read_replies(b) -> dict:
    """Greedy replies (REPLY_TOKENS) to the fixed prompts from the brain's working state; adapters on, no hippocampus."""
    reps = []
    with torch.no_grad():
        for q in PROMPTS:
            lg, st = b.cortex.read(f"User: {q}\n\nAssistant:", b.state)
            toks = []
            for _ in range(REPLY_TOKENS):
                tok = int(torch.argmax(lg))
                if tok == 0:
                    break
                toks.append(tok)
                if "\n\n" in b.cortex.pipe.decode(toks):
                    break
                lg, st = b.cortex.forward([tok], st)
            reps.append(b.cortex.pipe.decode(toks).strip())
    return {**reply_flags(reps), "replies": reps}


def read_all(b, offframe: list, now: float, k: int) -> dict:
    from brain.eval.life_langguard import measure
    base = [s.float() for s in b.base]
    t0 = time.time()
    held = measure(b.cortex, base)
    off = measure(b.cortex, base, texts=offframe)
    rec = {"kl": held["kl"], "nll_on": held["nll_on"], "nll_off": held["nll_off"], "off_kl": off["kl"],
           "off_nll_on": off["nll_on"], "ret": read_retention(b, now, k), **read_replies(b),
           "adapter_norm": float(b.cortex.plastic.adapter_norm())}
    rec["read_seconds"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------ the chain
def _load_adapters_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_dir(snaps: str, k: int, adapters: str, work: str, tag: str) -> str:
    """A snapshot-shaped dir for routing_replay.open_snapshot: the store of snapshot n<k> (episodes.json, keys.pt,
    keep_bank.json linked), the arm's own adapters, meta.json with the sleep counter set to k-1."""
    d = os.path.join(work, f"chain_{tag}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    snap = os.path.join(snaps, f"n{k}")
    for f in ("episodes.json", "keys.pt", "keep_bank.json"):
        if not os.path.exists(os.path.join(snap, f)):
            # keep_bank.json: without it the Brain would silently sample a NEW 512-text bank from the frozen cortex, so
            # every arm would sleep on another bank than the life (review defect 1). Refuse instead.
            raise SystemExit(f"{snap} lacks {f}: the chain needs the life's own store and keep bank")
        os.symlink(os.path.abspath(os.path.join(snap, f)), os.path.join(d, f))
    os.symlink(os.path.abspath(adapters), os.path.join(d, "cortex_adapters.pt"))
    with open(os.path.join(snap, "meta.json")) as fh:
        meta = json.load(fh)
    meta["sleeps"] = meta["sleeps"] - 1
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return d


def save_state(b, path: str) -> None:
    p = b.cortex.plastic
    tmp = path + ".tmp"
    torch.save({"rank": b.adapter_rank, "layers": p.layers,
                "adapters": {k: v.detach().cpu() for k, v in p.state_dict().items()}}, tmp)
    os.replace(tmp, path)


def rel_distance(a: dict, b: dict) -> float:
    """Relative L2 distance between two adapter state dicts (the "adapters" entries)."""
    num = sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in a)
    den = sum(float((b[k].float() ** 2).sum()) for k in b)
    return (num / den) ** 0.5 if den else float("nan")


def foreign_texts_of(foreign_snaps: str | None, k: int) -> list:
    if not foreign_snaps:
        return []
    from brain.experiments import routing_replay as RR
    return [r["text"] for r in RR._episode_rows(os.path.join(foreign_snaps, f"n{k}"))]


def foreign_check(foreign_snaps: str | None, nights=NIGHTS) -> dict:
    """A5f needs the foreign seed's episodes.json for every chained night (on a node only the own seed's snapshots are
    held, so run_node.sh ships the three small files). Refuses before any model load; returns {night: sha256}."""
    if not foreign_snaps:
        raise SystemExit("arm A5f needs --foreign-snaps (the foreign seed's n13..n15 episodes.json)")
    out = {}
    for k in nights:
        f = os.path.join(foreign_snaps, f"n{k}", "episodes.json")
        if not os.path.exists(f):
            raise SystemExit(f"arm A5f: {f} missing")
        out[k] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    return out


def unit_path(out_dir: str, arm: str, seed: int, night: int) -> str:
    return os.path.join(out_dir, f"u_{arm}_s{seed}_n{night}.json")


def state_path(out_dir: str, arm: str, seed: int, night: int) -> str:
    return os.path.join(out_dir, "state", f"{arm}_s{seed}_n{night}.pt")


def run_start(res, snaps, cortex, work, out_dir, offframe, prov, off_sha) -> dict:
    """The shared n12 reading (every arm of a seed starts here)."""
    from brain.experiments import routing_replay as RR
    from brain import clock
    seed = res["seed"]
    path = unit_path(out_dir, "start", seed, START_NIGHT)
    if os.path.exists(path):
        return _resume(path, prov, off_sha)
    snap = os.path.join(snaps, f"n{START_NIGHT}")
    home = os.path.join(work, f"home_start_s{seed}")
    b = RR.open_snapshot(snap, home, cortex, res["brain_kw"])
    try:
        meta = json.load(open(os.path.join(snap, "meta.json")))
        now = float(meta["last_sleep"])
        clock.set(lambda: now)
        rec = {"arm": "start", "seed": seed, "night": START_NIGHT, **read_all(b, offframe, now, START_NIGHT),
               "offframe_sha256": off_sha, **prov}
    finally:
        clock.set(None)
        b.release()
        shutil.rmtree(home, ignore_errors=True)
    atomic_write(path, rec)
    return rec


def run_chain(res, snaps, arm, cortex, work, out_dir, offframe, prov, off_sha, foreign_snaps=None, log=print,
              max_night=None) -> list:
    from brain.experiments import routing_replay as RR
    from brain import clock
    cfg = arm_cfg(arm)
    seed = res["seed"]
    os.makedirs(os.path.join(out_dir, "state"), exist_ok=True)
    os.makedirs(work, exist_ok=True)
    run_start(res, snaps, cortex, work, out_dir, offframe, prov, off_sha)
    done = []
    for i, k in enumerate(NIGHTS[:cfg["nights"]]):
        if max_night is not None and k > max_night:       # step 0 runs A0 and A2 night 13 only; the run resumes them
            break
        up = unit_path(out_dir, arm, seed, k)
        if os.path.exists(up):
            done.append(_resume(up, prov, off_sha))
            continue
        if k == NIGHTS[0]:
            adapters = os.path.join(snaps, f"n{START_NIGHT}", "cortex_adapters.pt")
        else:
            adapters = state_path(out_dir, arm, seed, k - 1)
            if not os.path.exists(adapters):
                raise RuntimeError(f"{arm} s{seed}: no carried adapters for night {k} ({adapters}); rerun from night 13")
        cd = chain_dir(snaps, k, adapters, work, f"{arm}_s{seed}_n{k}")
        home = os.path.join(work, f"home_{arm}_s{seed}_n{k}")
        b = RR.open_snapshot(cd, home, cortex, res["brain_kw"])
        t0 = time.time()
        info = {}
        try:
            meta = json.load(open(os.path.join(snaps, f"n{k}", "meta.json")))
            now = float(meta["last_sleep"])
            clock.set(lambda: now)
            assert b.meta["sleeps"] == k - 1, (b.meta["sleeps"], k)
            foreign = foreign_texts_of(foreign_snaps, k) if cfg["placebo"] == "foreign" else None
            if foreign is not None:
                info["foreign_sha256"] = hashlib.sha256(
                    open(os.path.join(foreign_snaps, f"n{k}", "episodes.json"), "rb").read()).hexdigest()
            with patched(b, cfg, foreign, info):
                losses = b.sleep()
            assert b.meta["sleeps"] == k
            save_state(b, state_path(out_dir, arm, seed, k))
            t_sleep = time.time() - t0
            rec = {"arm": arm, "seed": seed, "night": k, "cfg": cfg, "sleep_seconds": round(t_sleep, 1),
                   "sleep_info": info, "mean_replay": sum(a for a, _ in losses) / max(len(losses), 1),
                   "mean_keep": sum(c for _, c in losses) / max(len(losses), 1),
                   "mean_ce_keep": info.get("mean_ce_keep"), "kl_weight_used": info.get("kl_weight"),
                   **read_all(b, offframe, now, k), "offframe_sha256": off_sha, **prov}
            if arm == "A0L" or k == NIGHTS[0]:
                life_ad = os.path.join(snaps, f"n{k}", "cortex_adapters.pt")
                if os.path.exists(life_ad):
                    rec["rel_dist_to_life_adapters"] = rel_distance(
                        _load_adapters_file(state_path(out_dir, arm, seed, k))["adapters"],
                        _load_adapters_file(life_ad)["adapters"])
        finally:
            clock.set(None)
            b.release()
            shutil.rmtree(home, ignore_errors=True)
            shutil.rmtree(cd, ignore_errors=True)
        atomic_write(up, rec)
        prev = state_path(out_dir, arm, seed, k - 1)
        if k > NIGHTS[0] and os.path.exists(prev):
            os.remove(prev)
        log(f"{arm} s{seed} n{k}: kl {rec['kl']:.4f} off_kl {rec['off_kl']:.4f} sleep {rec['sleep_seconds']}s")
        done.append(rec)
    return done


# ------------------------------------------------------------------ scoring (the registered rule, exact)
def _g(tab, arm, seed, night):
    return (tab.get(arm, {}).get(seed, {}) or {}).get(night)


def noise_raw(tab, seed, night=END):
    """max |KL(A0) - KL(noise arm)| over the two decoupled noise arms, held-out KL at `night`; None if absent."""
    a0 = _g(tab, "A0", seed, night)
    vals = []
    for n in NOISE_ARMS:
        r = _g(tab, n, seed, night)
        if a0 is None or r is None:
            return None
        vals.append(abs(a0["kl"] - r["kl"]))
    return max(vals)


def noise_floor(tab, seed, night=END):
    """The registered floor: max(noise_raw, NOISE_FLOOR_REL x KL(A0, night)); None if absent (B2: two draws can land near
    A0 by chance, so the floor never goes below 5% of the A0 level)."""
    raw = noise_raw(tab, seed, night)
    if raw is None:
        return None
    return max(raw, NOISE_FLOOR_REL * abs(_g(tab, "A0", seed, night)["kl"]))


def _engaged(arm: str, si: dict) -> bool:
    si = si or {}
    if arm == "A1":
        return si.get("ce_keep_w") == 1.0 and si.get("mean_ce_keep") is not None and si["mean_ce_keep"] > 0
    if arm == "A2":
        return si.get("keep_per_step") == [2]
    if arm == "A3":
        return si.get("keep_n") == 256 and si.get("keep_n_loop", 256) == 256
    if arm == "A4":
        return si.get("kl_weight") == 6.0
    if arm in ("A5s", "A5f"):
        return ((si.get("placebo_pool") or 0) > 0 and si.get("placebo_excluded") is not None
                and si.get("placebo_overlap") == 0)
    return True


def engagement(tab, arm, seeds=SEEDS) -> list:
    """Units of `arm` (every night present, n13..n15) whose sleep did not realize the arm's change (ENGAGE). [] = engaged.
    An arm with no unit at all is reported by the missing-unit checks, not here."""
    bad = []
    for s in seeds:
        for k, u in sorted(((tab.get(arm) or {}).get(s) or {}).items()):
            if not _engaged(arm, u.get("sleep_info")):
                bad.append(f"{arm}:s{s}:n{k}")
    return bad


def candidate_verdict(tab, X, seeds=SEEDS) -> dict:
    need = [(a, s) for a in ("A0", X) for s in seeds] + [(n, s) for n in NOISE_ARMS for s in seeds]
    missing = [f"{a}:s{s}" for a, s in need if _g(tab, a, s, END) is None]
    if missing:
        return {"status": f"INCONCLUSIVE: missing n{END} units {missing}"}
    not_engaged = engagement(tab, X, seeds)
    if not_engaged:
        return {"status": f"INCONCLUSIVE: arm not engaged ({ENGAGE[X]}) on {not_engaged}", "not_engaged": not_engaged}
    noise = {s: noise_floor(tab, s) for s in seeds}
    rows, p2_unread = {}, []
    for s in seeds:
        x, a = _g(tab, X, s, END), _g(tab, "A0", s, END)
        unread = [st for st in P2_STRATA if not (_stratum_read(x, st) and _stratum_read(a, st))]
        p2_unread += [f"s{s}:{st}" for st in unread]
        rows[s] = {"kl_x": x["kl"], "kl_a0": a["kl"], "noise": noise[s], "noise_raw": noise_raw(tab, s),
                   "p1_seed": x["kl"] < a["kl"] and (a["kl"] - x["kl"]) > P1_NOISE_MULT * noise[s],
                   "p1b_seed": x["off_kl"] <= P1B_RATIO * a["off_kl"],
                   "p2_seed": not unread and all(x["ret"][st]["nll"] <= a["ret"][st]["nll"] + P2_MARGIN for st in P2_STRATA),
                   "p2_unread": unread,
                   "p3_seed": x["refusal"] <= a["refusal"] + P3_MARGIN and x["name"] <= a["name"] + P3_MARGIN}
    mean_x = sum(r["kl_x"] for r in rows.values()) / len(seeds)
    mean_a = sum(r["kl_a0"] for r in rows.values()) / len(seeds)
    p1 = all(r["p1_seed"] for r in rows.values()) and mean_x <= P1_RATIO * mean_a
    p1b = sum(r["p1b_seed"] for r in rows.values()) >= P1B_SEEDS
    p2 = all(r["p2_seed"] for r in rows.values())
    p3 = all(r["p3_seed"] for r in rows.values())
    kill = sum(not r["p1_seed"] for r in rows.values()) >= KILL_SEEDS
    if kill:
        status = "KILL"
    elif p2_unread:
        status = f"INCONCLUSIVE: P2 strata read on < {P2_MIN_N} texts {p2_unread}"
    elif p1 and p1b and p2 and p3:
        status = "PASS"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "P1": p1, "P1b": p1b, "P2": p2, "P3": p3, "mean_kl_x": mean_x, "mean_kl_a0": mean_a,
            "ratio": mean_x / mean_a if mean_a else None, "per_seed": rows}


def _stratum_read(rec: dict, st: str) -> bool:
    r = (rec.get("ret") or {}).get(st) or {}
    return r.get("nll") is not None and (r.get("n") or 0) >= P2_MIN_N


def placebo_read(tab, start, seeds=SEEDS) -> dict:
    """Per seed D = KL(n15) - KL(n12) for A0, A5s, A5f. MOTION-DOMINANT if A5s >= 0.6 x A0 on >= 2 seeds (the self-text
    placebo is biased DOWN, so this direction survives the bias); CONTENT-DRIVEN only if BOTH placebos <= 0.25 x A0 on >= 2
    seeds; else BRACKETED. A seed whose real drift is not above twice the noise floor is UNREAD."""
    out, mot, con, n_read = {}, 0, 0, 0
    for s in seeds:
        a0, ps, pf, st = (_g(tab, "A0", s, END), _g(tab, "A5s", s, END), _g(tab, "A5f", s, END), start.get(s))
        nf = noise_floor(tab, s)
        if None in (a0, ps, pf, st) or nf is None:
            return {"status": f"INCONCLUSIVE: missing A0/A5s/A5f n{END}, start or noise arms for s{s}"}
    not_engaged = engagement(tab, "A5s", seeds) + engagement(tab, "A5f", seeds)
    if not_engaged:
        return {"status": f"INCONCLUSIVE: placebo not engaged ({ENGAGE['A5s']}) on {not_engaged}",
                "not_engaged": not_engaged}
    for s in seeds:
        a0, ps, pf, st = (_g(tab, "A0", s, END), _g(tab, "A5s", s, END), _g(tab, "A5f", s, END), start.get(s))
        nf = noise_floor(tab, s)
        d_real, d_s, d_f = a0["kl"] - st["kl"], ps["kl"] - st["kl"], pf["kl"] - st["kl"]
        readable = d_real > P1_NOISE_MULT * nf
        out[s] = {"d_real": d_real, "d_self": d_s, "d_foreign": d_f, "noise": nf, "readable": readable,
                  "self_over_real": d_s / d_real if d_real else None, "foreign_over_real": d_f / d_real if d_real else None}
        if readable:
            n_read += 1
            mot += d_s >= MOTION_RATIO * d_real
            con += (d_s <= CONTENT_RATIO * d_real) and (d_f <= CONTENT_RATIO * d_real)
    if n_read < PLACEBO_SEEDS:
        status = "UNREAD: real drift within twice the noise floor on too many seeds"
    elif mot >= PLACEBO_SEEDS:
        status = "MOTION-DOMINANT"
    elif con >= PLACEBO_SEEDS:
        status = "CONTENT-DRIVEN"
    else:
        status = "BRACKETED"
    return {"status": status, "per_seed": out}


def determinism_pair(tab, s: int) -> dict:
    """The instrument's repeatability on one seed: A0r vs A0 held-out KL at n13 within DETERMINISM_TOL relative. ok is
    None when either unit is missing. Used by instrument_checks and, at step 0, by `detcheck` (prereg section 5: a miss at
    step 0 triggers the registered single-thread fallback before any other unit runs)."""
    a, r = _g(tab, "A0", s, 13), _g(tab, "A0r", s, 13)
    if a is None or r is None:
        return {"seed": s, "ok": None, "rel_diff": None}
    rel = abs(a["kl"] - r["kl"]) / max(abs(a["kl"]), 1e-12)
    return {"seed": s, "ok": abs(a["kl"] - r["kl"]) <= DETERMINISM_TOL * max(abs(a["kl"]), 1e-12), "rel_diff": rel,
            "threads": [a.get("threads"), r.get("threads")]}


def p1_change_diag(tab, start, X, seeds=SEEDS) -> dict:
    """DIAGNOSTIC, never a pass condition (pre-data review, P1 power). The share of A0's own n12 -> n15 drift that X
    removes: per seed d_a0 = KL(A0,n15) - KL(n12), d_x = KL(X,n15) - KL(n12); a seed is readable only if d_a0 > 2 x noise_s.
    cut = 1 - mean(d_x) / mean(d_a0) over readable seeds; UNREAD with fewer than 2 readable seeds. (The per-seed P1 clause is
    already a change contrast, since X and A0 share the n12 start; only P1's 0.75 ratio is on the level.)"""
    rows, read = {}, []
    for s in seeds:
        x, a, st, nf = _g(tab, X, s, END), _g(tab, "A0", s, END), start.get(s), noise_floor(tab, s)
        if None in (x, a, st, nf):
            return {"status": f"UNREAD: missing {X}/A0 n{END}, start or noise arms for s{s}"}
        d_a, d_x = a["kl"] - st["kl"], x["kl"] - st["kl"]
        rows[s] = {"d_a0": d_a, "d_x": d_x, "noise": nf, "readable": d_a > P1_NOISE_MULT * nf}
        if rows[s]["readable"]:
            read.append(s)
    if len(read) < 2:
        return {"status": "UNREAD: A0's own drift within twice the noise floor on too many seeds", "per_seed": rows}
    ma = sum(rows[s]["d_a0"] for s in read) / len(read)
    mx = sum(rows[s]["d_x"] for s in read) / len(read)
    return {"status": "READ", "cut_of_change": 1 - mx / ma, "readable": read, "per_seed": rows}


def instrument_checks(tab, banked: dict, seeds=SEEDS) -> dict:
    """determinism: A0r vs A0 held-out KL at n13 within DETERMINISM_TOL relative; k_sane: A0L and A0 n13 KL within 25% of
    the life's banked lang_guard.kl at n13. A check that cannot be read (its unit, or the banked KL, missing) is None and
    voids the screen (INCONCLUSIVE); a K-sane MISS (False) is a device caveat, not a void; a determinism miss voids."""
    det, ks = {}, {}
    for s in seeds:
        det[s] = determinism_pair(tab, s)["ok"]
        for arm in ("A0L", "A0"):
            x = _g(tab, arm, s, 13)
            bk = banked.get(s)
            ks[f"{arm}:s{s}"] = None if x is None or not bk else abs(x["kl"] - bk) <= K_SANE_TOL * bk
    void = []
    if [s for s, v in det.items() if v is None]:
        void.append(f"determinism unread (A0r or A0 n13 missing) on seeds {[s for s, v in det.items() if v is None]}")
    if [s for s, v in det.items() if v is False]:
        void.append(f"A0r differs from A0 at n13 on seeds {[s for s, v in det.items() if v is False]} (the instrument is not "
                    f"deterministic)")
    if [k for k, v in ks.items() if v is None]:
        void.append(f"K-sane unread (A0L / A0 n13 unit or the life's banked n13 KL missing) for {[k for k, v in ks.items() if v is None]}")
    return {"determinism": det, "k_sane": ks, "k_sane_caveat": [k for k, v in ks.items() if v is False], "void": void}


def provenance_check(start: dict, tab: dict, seeds=SEEDS) -> list:
    """Reasons the unit set is not one instrument: within a seed every unit (start and arms) on one host, commit, device,
    precision, model and off-frame set; across seeds one commit, device, precision, model and off-frame set (hosts differ
    by seed by design: one seed per node). A missing commit is a reason too."""
    units = [u for u in start.values()] + [u for by in tab.values() for ns in by.values() for u in ns.values()]
    bad = []
    if any(not u.get("commit") for u in units):
        bad.append("units without a commit")
    for k in ("commit", "device", "precision", "model", "threads", "offframe_sha256"):
        vals = {str(u.get(k)) for u in units}
        if len(vals) > 1:
            bad.append(f"{k} differs across units {sorted(vals)}")
    for s in seeds:
        hosts = {str(u.get("host")) for u in units if u.get("seed") == s}
        if len(hosts) > 1:
            bad.append(f"s{s} units on several hosts {sorted(hosts)}")
    return bad


def verdict(tab, start, banked=None, seeds=SEEDS) -> dict:
    chk = instrument_checks(tab, banked or {}, seeds)
    cands = {X: candidate_verdict(tab, X, seeds) for X in CANDIDATES}
    plc = placebo_read(tab, start, seeds)
    for X, v in cands.items():
        v["p1_change_diag"] = p1_change_diag(tab, start, X, seeds)     # report only; never changes PASS / INCONCLUSIVE
        if v["status"] == "KILL" and v["p1_change_diag"]["status"].startswith("UNREAD"):
            # pre-data review 2, N1: a KILL whose window shows no readable A0 drift says "this window cannot show the
            # effect", not "the recipe fails"; it is named apart so no digest rolls it into a KILL
            v["status"] = "KILL-UNREAD"
    if chk["void"]:
        why = "INCONCLUSIVE: " + "; ".join(chk["void"])
        for v in cands.values():
            v["status"] = why
        plc["status"] = why
    return {"candidates": cands, "placebo": plc, "instrument": chk}


def load_tab(d: str) -> tuple:
    tab, start = {}, {}
    for f in sorted(os.listdir(d)):
        if f.startswith("u_") and f.endswith(".json"):
            u = json.load(open(os.path.join(d, f)))
            if u["arm"] == "start":
                start[u["seed"]] = u
            else:
                tab.setdefault(u["arm"], {}).setdefault(u["seed"], {})[u["night"]] = u
    return tab, start


def banked_kl(res: dict, night: int):
    for n in res.get("nights", []):
        if n.get("line") == "trunk" and n.get("night") == night and n.get("lang_guard"):
            return n["lang_guard"]["kl"]
    return None


def score(d: str, lives: list | None = None) -> dict:
    tab, start = load_tab(d)
    banked = {}
    for f in lives or []:
        r = json.load(open(f))
        banked[r["seed"]] = banked_kl(r, 13)
    v = verdict(tab, start, banked)
    bad = provenance_check(start, tab)
    v["provenance"] = sorted({json.dumps({k: u.get(k) for k in PROV_KEYS}, sort_keys=True)
                              for u in list(start.values()) + [u for by in tab.values() for ns in by.values()
                                                               for u in ns.values()]})
    if bad:
        v["mixed_provenance"] = bad
        why = "INCONCLUSIVE: mixed provenance (" + "; ".join(bad) + ")"
        for c in v["candidates"].values():
            c["status"] = why
        v["placebo"]["status"] = why
    return v


def strata_counts(snaps: str, nights=(START_NIGHT,) + NIGHTS) -> dict:
    """Store-only (no model): the retention sample size per age stratum at each reading night, from the snapshot's
    episodes.json created stamps and meta.last_sleep, exactly as read_retention samples them."""
    out = {}
    for k in nights:
        d = os.path.join(snaps, f"n{k}")
        rows = json.load(open(os.path.join(d, "episodes.json")))
        rows = rows if isinstance(rows, list) else rows.get("episodes", [])
        now = float(json.load(open(os.path.join(d, "meta.json")))["last_sleep"])
        out[k] = {n: len(v) for n, v in sample_strata([r["created"] for r in rows], now, k).items()}
    return out


# ------------------------------------------------------------------ runtime estimate (pool CPU; inputs are UNMEASURED)
def estimate(night_min: float = 25.0, read_min: float = 10.0, nodes: int = 3, arms=None, seeds: int = 3) -> dict:
    """Node-hours and wall hours. night_min / read_min are placeholders until step 0 times one A0 night (the note's step 0).
    Per (arm, seed): one sleep + one reading per chained night, plus one shared start reading per seed. A3 keeps ~4x the
    reference forwards (+ ~1.3x per step for A2's second keep text), and A3 recomputes the frozen reference each step (one
    no-grad forward more per step, to stay inside the node RAM); not modelled, so the figure is a floor."""
    arms = list(ARMS) if arms is None else list(arms)
    nights = sum(arm_cfg(a)["nights"] for a in arms)
    units = nights * seeds
    hours = (units * (night_min + read_min) + seeds * read_min) / 60.0
    return {"arms": arms, "night_sleeps": units, "node_hours": round(hours, 1), "wall_hours_on_nodes": round(hours / max(nodes, 1), 1),
            "inputs": {"night_min": night_min, "read_min": read_min, "nodes": nodes, "measured": False}}


# ------------------------------------------------------------------ CLI
def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--offframe", required=True)
    r.add_argument("--offframe-sha256", required=True)
    r.add_argument("--foreign-snaps", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--cuda-precision", default="fp32")
    r.add_argument("--work", default=None)
    r.add_argument("--max-night", type=int, default=None, help="stop after this night (step 0: A0 / A2 night 13 only)")
    dc = sub.add_parser("detcheck", help="step 0: A0r vs A0 night-13 determinism for one seed (exit 3 on a miss)")
    dc.add_argument("--dir", required=True)
    dc.add_argument("--seed", type=int, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--lives", nargs="+", required=True, help="the three KR life JSONs (K-sane reads their banked n13 KL)")
    sc.add_argument("--out", default=None)
    st = sub.add_parser("strata")
    st.add_argument("--snaps", required=True)
    st.add_argument("--min-n", type=int, default=P2_MIN_N)
    es = sub.add_parser("estimate")
    es.add_argument("--night-min", type=float, default=25.0)
    es.add_argument("--read-min", type=float, default=10.0)
    es.add_argument("--nodes", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        out = estimate(a.night_min, a.read_min, a.nodes)
        print(json.dumps(out, indent=1))
        return out
    if a.cmd == "strata":
        c = strata_counts(a.snaps)
        short = {k: [n for n in P2_STRATA if v[n] < a.min_n] for k, v in c.items() if k == END}
        nobank = [k for k in (START_NIGHT,) + NIGHTS if not os.path.exists(os.path.join(a.snaps, f"n{k}", "keep_bank.json"))]
        out = {"counts": c, "p2_short_at_n15": short[END], "no_keep_bank": nobank, "ok": not short[END] and not nobank}
        print(json.dumps(out, indent=1))
        return out
    if a.cmd == "detcheck":
        tab, _ = load_tab(a.dir)
        out = determinism_pair(tab, a.seed)
        print(json.dumps(out, indent=1))
        if out["ok"] is not True:
            raise SystemExit(3)
        return out
    if a.cmd == "score":
        v = score(a.dir, a.lives)
        if a.out:
            atomic_write(a.out, v)
        print(json.dumps({"candidates": {k: x["status"] for k, x in v["candidates"].items()},
                          "placebo": v["placebo"]["status"], "instrument": v["instrument"]}, indent=1))
        return v
    guard()
    cfg = arm_cfg(a.arm)
    if cfg["placebo"] == "foreign":
        foreign_check(a.foreign_snaps, NIGHTS[:cfg["nights"]])
    raw = open(a.offframe, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != a.offframe_sha256 or sha != OFFFRAME_SHA256:
        raise SystemExit(f"off-frame set sha256 {sha} != given {a.offframe_sha256} / registered {OFFFRAME_SHA256}")
    offframe = json.loads(raw)
    if not (isinstance(offframe, list) and len(offframe) == 12 and all(isinstance(t, str) for t in offframe)):
        raise SystemExit("off-frame set must be a JSON list of 12 strings")
    need = [os.path.join(a.snaps, f"n{START_NIGHT}", "cortex_adapters.pt")] + [
        os.path.join(a.snaps, f"n{k}", f) for k in (START_NIGHT,) + NIGHTS[:cfg["nights"]]
        for f in SNAP_FILES]
    lost = [f for f in need if not os.path.exists(f)]
    if lost:
        raise SystemExit(f"--snaps lacks {lost[:3]} ({len(lost)} files): this node does not hold seed {a.seed}'s snapshots")
    res = json.load(open(a.life))
    if res["seed"] != a.seed:
        raise SystemExit(f"--life is seed {res['seed']}, not {a.seed}")
    if res["brain_kw"].get("keep_bank") != "prior":
        raise SystemExit("the screen is for K-P (keep_bank='prior') lives")
    if cortex is None:
        from brain.cortex import Cortex, resolve_model
        cortex = Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)
    prov = provenance(a.device, precision_of(a.device, a.cuda_precision), getattr(cortex, "model_name", None))
    work = a.work or os.path.join(a.out_dir, f"work_{a.arm}_s{a.seed}")
    os.makedirs(a.out_dir, exist_ok=True)
    return run_chain(res, a.snaps, a.arm, cortex, work, a.out_dir, offframe, prov, sha, a.foreign_snaps,
                     max_night=a.max_night)


if __name__ == "__main__":
    main()
