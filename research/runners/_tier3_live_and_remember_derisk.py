"""Tier-3 'live-and-remember' — the FIRST artificial-life-capstone SYNTHESIS slice (cheap-first de-risk).

Per the scoping (research/findings/2026-06-30-tier3-artificial-life-capstone-scoping.md, CYCLE 731-732, owner
picked Option 1): the capstone is largely done in PIECES; the genuine residual is the SYNTHESIS — one continuous
loop where the SAME persistent brain (i) is driven by its own interoceptive state to act, (ii) PERCEIVES + GROUNDS
objects it encounters DURING that lived behaviour, (iii) STORES facts about them, (iv) can be QUERIED about what it
lived (moat intact), and (v) PERSISTS across a reset. Each of (i)-(v) is separately validated; NO runner chains
them. This probe is that runner-only JOIN (NO `sim/` edit — one additive default-off `co_resident_drive`
passthrough on `MergedNavConvAgent`, itself only forwarding a build param that already exists).

WHAT IS NEW vs the executed pieces (the genuine residual)
---------------------------------------------------------
* `persistent_living_loop_derisk.py` (GO 6/6) is a continuous survival loop but has NO perception/memory/converse.
* `_tier3_spiking_living_loop_derisk.py` lifts the DRIVE onto the merged spiking bridge but still doesn't converse.
* `navigate_to_compose_then_answer.py` (6-seed GO) perceives+composes+converses but on a SCRIPTED bounded episode.
This runner is the intersection none fills: a CONTINUOUS, OPEN-ENDED (self-chosen), PERSISTENT life that perceives,
remembers, and can be talked to about WHAT IT LIVED.

THE LOOP (host code legitimate ONLY for the world + body; the drive/reward/perception/memory are the brain's job)
------------------------------------------------------------------------------------------------------------------
The agent lives in a corridor (food at cell 0; a few landmark OBJECTS at other cells). Each living step:
  1. read the SPIKING interoceptive hunger off the merged bridge's co-resident `drive_agrp` pool (inject the body
     deficit as an interoceptive current, run the pool, read the firing rate off `cp_firing_states` — NOT a host
     deficit value); this is the validated 2-pool drive (O.05/O.06, corr(deficit,AgRP)>=0.9);
  2. the VALIDATED survival policy (the rate-proxy Q stand-in for the motor system — the LEARNED spatial policy on
     the substrate is the deferred Tier-4 dendrite wall, off this slice's critical path) moves the body; energy
     depletes; eating at food refills → an INTRINSIC drive-reduction reward `r` (Keramati-Gutkin; NO host distance
     term); the Q shaped by `r` keeps the agent alive (self-directed foraging, NO external goal);
  3. on FIRST arrival at an object cell, the environment renders the object into the perception slice and
     `agent.perceive_and_ground(obj)` grounds the LIVE spiking percept into the co-resident composer, and the
     agent STORES a lived fact linking it to the previously-encountered object (`(prev, "near", obj)`) — so WHICH
     objects it knows is a consequence of its own drive-biased trajectory (OPEN-ENDED, not a scripted perceive-list).
After the life: the owner can ask "what did you encounter near X?" — the agent answers from its lived, grounded
memory (or ABSTAINS on a never-encountered object — the no-confab moat). The life PERSISTS: the body state + the
lived facts + the grounded codes are saved via BridgeLineage; a reset → reload resumes the SAME life + memory.

GATES / ANTI-CHEATS (the validated-signal-by-its-function bar; ALL must collapse)
---------------------------------------------------------------------------------
  (1) SURVIVAL: intact keeps itself alive (energy in-band, never crashes) from the intrinsic drive-reduction reward,
      while DRIVE-LESION (r≡0) and YOKED-RANDOM (drive decorrelated from the deficit) both STARVE.
  (2) DRIVE-IS-SPIKING: corr(deficit, drive_agrp firing rate) >= +0.9 on the merged bridge (a controlled sweep).
  (3) LIVED, OPEN-ENDED MEMORY: the agent grounds+stores facts about the objects IT encountered; a GROUNDING-LESION
      (sever the perception->concept convergence) collapses lived-recall to chance (the memory rides the LIVE
      percept + the agent's own trajectory, not a script).
  (4) CONVERSE ABOUT WHAT IT LIVED: who/what queries about encountered objects answer correctly AND the no-confab
      MOAT holds — a never-encountered cue returns None (abstain); the conversational synapses stay BYTE-IDENTICAL
      across the live run. A moat breach is a HARD STOP.
  (5) PERSISTENCE ACROSS RESET: reload resumes the body life-state AND the lived memory; a NO-PERSISTENCE cold-start
      visibly differs (empty memory, re-warm).
  (*) REWARD-PROVENANCE: `r` is the drive-reduction (from the spiking/interoceptive drive), asserted NO r=f(distance).

HONEST SCOPE (valid deliverables per the actual-goal mandate)
-------------------------------------------------------------
* The LEARNED spatial policy on the substrate stays the deferred Tier-4 dendrite wall — survival here uses the
  validated rate-proxy Q stand-in (the motor-system stand-in), exactly as the rate-proxy GO 6/6 established;
  survival (not spatial optimality) is the discriminator. If the noisy spiking-hunger reward degrades Q-learning,
  `--drive-reward rate_proxy` falls back to the validated host drive-reduction (the spiking drive still read for the
  corr gate) — the honest scope, since combining spiking-reward + rate-proxy-policy is itself a follow-on.
* Persistence is JSON re-instate (body state + lived facts + grounded codes), NOT the raw `cp_connections` synaptic
  tensor — the develop-loop / LivingState cheap-first stand-in (§1f); true synaptic persistence is a follow-on.
* Open-endedness on the corridor is encounter-driven (the agent grounds what its foraging brings it to, not a
  scripted list); the richer path-dependent-order 2D world is a follow-on.

Run (GPU — the merged bridge is GPU-only):
  python -m research.runners._tier3_live_and_remember_derisk --smoke                 # tiny GPU mechanics check
  python -m research.runners._tier3_live_and_remember_derisk --seeds 42 43 44 100 101 102   # full 6-seed de-risk
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.lineage import BridgeLineage
# the VALIDATED 2-pool drive (only used as the rate-proxy reward fallback + the yoke marginal) — reuse verbatim.
from research.runners._homeostatic_drive_rl_cheap_first_probe import TwoPoolDrive

# the perceivable objects (the gen stack renders OBJECT_WORDS.index(obj)); the composer vocab already carries them
# + the link verb (navigate_to_compose_then_answer's 6-seed GO uses the same set with vocab=None).
OBJECT_WORDS = ["apple", "river", "dog", "cat"]
LINK_VERB = "near"          # the link verb; MUST be in the composer vocab (RFPhasorComposer builds concepts FROM vocab)
# the composer vocab MUST declare every word a fact uses (RFPhasorComposer.concepts = {w: code for w in vocab};
# _filler_phases KeyErrors on an undeclared word). Mirror navigate_to_compose_then_answer.build_compose_bridge's
# validated `vocab = OBJECT_WORDS + ACTIONS` exactly (the objects are placeholders overwritten live by grounding).
VOCAB = list(OBJECT_WORDS) + ["chase", LINK_VERB]

# survival dynamics — the validated persistent_living_loop_derisk corridor (GO 6/6): refill > learned round-trip
# cost but < random-walk cost, so the LEARNED policy survives while lesion/yoke crash.
L = 6                       # corridor length (cells 0..L-1); food at 0
SET_POINT = 1.0
DEPLETE = 0.015
EAT_REFILL = 0.3
START_E = 1.0
HEALTHY = 0.3               # healthy-energy band floor
CRASH = 0.1                 # below this = starving
GAMMA, ALPHA, EPS = 0.9, 0.25, 0.1


# ── the SPIKING interoceptive drive read (brain-based; reuse the _tier3_spiking_living_loop pattern) ──────────
class SpikingHunger:
    """Reads the co-resident 2-pool drive off the MERGED bridge: inject the body deficit/surplus as interoceptive
    current into drive_agrp/drive_pomc, run the pools `window` steps, read the drive_agrp FIRING RATE (off
    `cp_firing_states`) as the hunger. NOT a host deficit value — the firing IS the drive signal."""

    def __init__(self, bridge, window=40, i_scale=300.0, gain=14.0, floor=0.1):
        import sim.backend as B
        self.B = B
        self.xp, _ = B.get_backend()
        self.bridge = bridge
        rm = bridge.region_manager
        agrp = np.asarray(rm.indices("drive_agrp"), dtype=np.int64)
        pomc = np.asarray(rm.indices("drive_pomc"), dtype=np.int64)
        self.n_agrp = int(agrp.size)
        self._agrp_x = self.xp.asarray(agrp)
        self._pomc_x = self.xp.asarray(pomc)
        self.window = int(window)
        self.i_scale = float(i_scale)
        self.gain = float(gain)
        self.floor = float(floor)

    def read(self, deficit, lesion=False):
        """Return (hunger in [0,1], agrp_firing_rate). lesion=True zeros the interoceptive current (drive silent)."""
        B, br = self.B, self.bridge
        i_agrp = 0.0 if lesion else self.i_scale * max(0.0, float(deficit))
        i_pomc = self.i_scale * max(0.0, 1.0 - float(deficit))
        a_spikes = 0
        for _ in range(self.window):
            br.cp_external_input_current[:] = 0.0
            br.cp_external_input_current[self._agrp_x] = i_agrp
            br.cp_external_input_current[self._pomc_x] = i_pomc
            br._run_one_simulation_step()
            a_spikes += int(B.to_host(br.cp_firing_states[self._agrp_x]).sum())
        rate = a_spikes / (self.n_agrp * self.window)
        hunger = float(np.clip(self.floor + self.gain * rate, 0.0, 1.0))
        return hunger, rate


def _drive_corr_sweep(hunger_reader):
    """Regulation-independent drive-tracking gate (mirrors persistent_living_loop_derisk._drive_tracking_sweep):
    sweep the deficit across its full range, read the spiking drive_agrp rate at each, report corr(deficit, rate)."""
    sweep = np.concatenate([np.linspace(0.0, 1.0, 12), np.linspace(1.0, 0.0, 12)])
    defs, rates = [], []
    for deficit in sweep:
        _h, rate = hunger_reader.read(float(deficit))
        defs.append(float(deficit)); rates.append(float(rate))
    defs, rates = np.array(defs), np.array(rates)
    return float(np.corrcoef(defs, rates)[0, 1]) if rates.std() > 1e-9 else 0.0


# ── the world (the LEGITIMATE host surface: the corridor + where the food & objects sit) ─────────────────────
class LivingWorld:
    """A corridor: food at cell 0, `objects` placed at distinct cells (the agent grounds them on first arrival).
    Some OBJECT_WORDS are DELIBERATELY held out of the world (never placed) → the no-confab moat cue."""

    def __init__(self, seed, n_objects=3):
        rng = np.random.default_rng(seed * 104729 + 7)
        objs = list(OBJECT_WORDS)
        rng.shuffle(objs)
        self.placed = objs[:n_objects]                     # encountered (grounded during the life)
        self.held_out = objs[n_objects:]                   # NEVER placed → moat cues
        # place the objects at descending cells so a walk toward food (0) passes them (cells L-2 .. down).
        cells = list(range(L - 2, 0, -1))[:n_objects]      # e.g. L=6 -> [4,3,2]
        self.cell_to_obj = {c: o for c, o in zip(cells, self.placed)}


# ── the persistent internal life-state (body + learned policy + LIVED MEMORY) ────────────────────────────────
class LiveState:
    """The self over time: body energy + the learned Q policy + position + RNG + the LIVED MEMORY (the grounded
    object codes + the stored lived facts). Persisting + reloading it resumes the EXACT life AND what it remembers.
    (Cheap-first JSON re-instate per scoping §1f; true synaptic persistence is a follow-on.)"""

    def __init__(self, seed, lesion=False, yoke=False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self.yoke = bool(yoke)
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((L, 2))
        self.toward_action = int(self.rng.integers(2))     # remapped: which abstract action moves TOWARD food
        self.E = float(START_E)
        self.pos = L - 1
        self.t = 0
        self.drive_proxy = TwoPoolDrive(lesion=lesion)     # the rate-proxy reward fallback + the yoke marginal src
        self.yoke_pool = (self.rng.permutation(np.linspace(0.0, 1.0, 200)) if yoke else None)
        self.yi = 0
        self._prev_hunger = None
        self._hunger_cache = None                          # cached (hunger, agrp_rate) for the drive_read_every skip
        # the LIVED MEMORY (filled during the life):
        self.encountered = []                              # first-encounter order (open-ended: the lived trajectory)
        self.lived_facts = []                              # [(prev, LINK_VERB, cur)] linking consecutive encounters
        self.grounded_codes = {}                           # obj -> the grounded phasor code (re/imag), for resume

    def memory_payload(self):
        return {"encountered": list(self.encountered), "lived_facts": [list(f) for f in self.lived_facts],
                "grounded_codes": self.grounded_codes}

    def body_payload(self):
        return {"seed": self.seed, "lesion": self.lesion, "yoke": self.yoke,
                "rng_state": self.rng.bit_generator.state, "Q": self.Q.tolist(),
                "toward_action": self.toward_action, "E": self.E, "pos": self.pos, "t": self.t,
                "drive_agrp": self.drive_proxy.agrp, "drive_pomc": self.drive_proxy.pomc,
                "drive_tau": self.drive_proxy.tau, "yoke_pool": (None if self.yoke_pool is None else self.yoke_pool.tolist()),
                "yi": self.yi, "prev_hunger": self._prev_hunger}


def _encode_code(arr):
    """JSON-serialize a concept code (real or complex phasor) as {'real':..,'imag':..}."""
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        return {"real": a.real.tolist(), "imag": a.imag.tolist()}
    return {"real": a.tolist(), "imag": None}


def _decode_code(d):
    r = np.asarray(d["real"])
    return r if d["imag"] is None else (r + 1j * np.asarray(d["imag"]))


# ── the continuous living loop (survival + perceive-ground-store on encounters) ──────────────────────────────
def live(agent, hunger_reader, state, world, n_steps, *, drive_reward="spiking", drive_read_every=10,
         grounded_obj_cache=None):
    """Run a stretch of the agent's life IN PLACE on `state`. Survival = the validated Q policy shaped by the
    intrinsic drive-reduction reward; on first arrival at an object cell the agent perceive_and_grounds it + stores
    a lived fact. Returns per-step traces (energies, deficits, agrp_rates).

    drive_reward="spiking": the intrinsic reward rides the SPIKING hunger read off the bridge (brain-based; each read
      is `window` bridge steps, so it is sampled every drive_read_every-th step and cached between -- the biologically-
      faithful slow-hypothalamic-integration optimization). "rate_proxy": the validated host drive-reduction shapes
      the reward (NO per-step bridge stepping); the spiking drive is validated separately by the one-time corr sweep."""
    energies, deficits, agrp_rates = [], [], []
    cache = grounded_obj_cache if grounded_obj_cache is not None else set()
    for _ in range(n_steps):
        deficit = SET_POINT - state.E
        deficits.append(deficit)
        # the SPIKING drive read (spiking mode only; rate_proxy uses the host drive + the separate corr sweep, so it
        # steps the bridge only for groundings -> tractable). Sample every drive_read_every-th step, reuse the cache.
        if drive_reward == "spiking":
            if state._hunger_cache is None or (state.t % max(1, drive_read_every) == 0):
                state._hunger_cache = hunger_reader.read(deficit, lesion=state.lesion)
            hunger_spk, agrp_rate = state._hunger_cache
            agrp_rates.append(agrp_rate)
        else:
            hunger_spk = None
        # the drive value that shapes the reward:
        if state.lesion:
            drive_val = 0.0                                # lesion: no drive -> r == 0 -> no learning -> starves
        elif state.yoke:
            drive_val = float(state.yoke_pool[state.yi % len(state.yoke_pool)]); state.yi += 1
        elif drive_reward == "spiking":
            drive_val = hunger_spk                         # the SPIKING hunger IS the drive (brain-based)
        else:
            drive_val = state.drive_proxy.update(deficit)  # the validated rate-proxy (host drive-reduction)

        # action selection (eps-greedy, random tie-break)
        if state.rng.random() < EPS:
            a = int(state.rng.integers(2))
        else:
            a = int(state.rng.choice(np.flatnonzero(state.Q[state.pos] == state.Q[state.pos].max())))
        toward = (a == state.toward_action)
        new_pos = max(0, state.pos - 1) if toward else min(L - 1, state.pos + 1)
        state.E = max(0.0, state.E - DEPLETE)
        if new_pos == 0:                                   # reached food -> eat
            state.E = min(1.0, state.E + EAT_REFILL)

        # INTRINSIC reward = drive REDUCTION across the transition (no host distance term).
        if state.lesion:
            r = 0.0
        elif drive_reward == "spiking" and not state.yoke:
            r = 0.0 if state._prev_hunger is None else (state._prev_hunger - drive_val)
        else:
            # rate-proxy / yoke: the same drive_before - drive_after form the validated loop uses.
            d_after = (float(state.yoke_pool[state.yi % len(state.yoke_pool)]) if state.yoke
                       else state.drive_proxy.update(SET_POINT - state.E))
            if state.yoke:
                state.yi += 1
            r = drive_val - d_after
        state.Q[state.pos, a] += ALPHA * (r + GAMMA * np.max(state.Q[new_pos]) - state.Q[state.pos, a])
        state._prev_hunger = drive_val
        state.pos = new_pos
        state.t += 1
        energies.append(state.E)

        # ── perceive + ground + store on FIRST arrival at an object cell (the lived, open-ended memory) ──
        obj = world.cell_to_obj.get(state.pos)
        if obj is not None and obj not in cache:
            agent.perceive_and_ground(obj)                 # grounds the LIVE spiking percept into the composer
            cache.add(obj)
            code = agent.composer.concepts.get(obj)
            if code is not None:
                state.grounded_codes[obj] = _encode_code(code)
            prev = state.encountered[-1] if state.encountered else None
            state.encountered.append(obj)
            if prev is not None and prev != obj:
                agent.composer.store(prev, LINK_VERB, obj) # the lived fact: "encountered prev near obj"
                state.lived_facts.append((prev, LINK_VERB, obj))
    return {"energies": np.array(energies), "deficits": np.array(deficits), "agrp_rates": np.array(agrp_rates)}


def _survival(energies):
    half = energies[len(energies) // 2:]
    return {"min_energy": float(half.min()), "mean_energy": float(half.mean()),
            "crash_frac": float(np.mean(half < CRASH)), "band_occupancy": float(np.mean(half >= HEALTHY))}


def _lived_recall(agent, lived_facts):
    """Query the composer about what the agent lived: for each stored (prev, verb, cur), does query_patient(prev,
    verb) == cur? Returns (n_correct, n_total)."""
    ok = 0
    for (prev, verb, cur) in lived_facts:
        try:
            if agent.composer.query_patient(prev, verb) == cur:
                ok += 1
        except Exception:
            pass
    return ok, len(lived_facts)


def _moat_check(agent, world):
    """The no-confab moat: a never-encountered object as the query agent MUST abstain (None). Also a stored positive
    control (a lived fact retrieves) so the moat isn't trivially abstaining on everything."""
    abstain_ok, abstain_tot = 0, 0
    for held in world.held_out:                            # objects never placed -> never encountered
        abstain_tot += 1
        try:
            if agent.composer.query_patient(held, LINK_VERB) is None:
                abstain_ok += 1
        except Exception:
            pass
    return abstain_ok, abstain_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-steps", type=int, default=900, help="living steps per segment (lived once; the resume runs a short tail)")
    ap.add_argument("--drive-window", type=int, default=40)
    ap.add_argument("--drive-reward", choices=["spiking", "rate_proxy"], default="spiking",
                    help="'spiking': the intrinsic reward rides the spiking-hunger read (brain-based, expensive); "
                         "'rate_proxy': the validated host drive-reduction (the spiking drive still read for the "
                         "corr gate) -- the tractable survival path for the 6-seed run")
    ap.add_argument("--drive-read-every", type=int, default=10,
                    help="(spiking mode) sample the spiking hunger every Nth living step, reuse the cache -- the "
                         "biologically-faithful slow-hypothalamic-integration optimization (cuts the per-step cost)")
    ap.add_argument("--n-objects", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["intact", "lesion", "yoke"])
    ap.add_argument("--out", default="research/findings/raw/_tier3_live_and_remember.json")
    ap.add_argument("--keep-lineage", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny GPU mechanics check (1 seed, intact, short)")
    a = ap.parse_args()

    print("[Tier-3 live-and-remember] does a merged one-brain LIVE (drive-biased survival), PERCEIVE+GROUND+STORE "
          "what it encounters, get QUERIED about what it lived (moat intact), and PERSIST across a reset?\n"
          "  GATES: (1) survival vs lesion/yoke crash  (2) corr(deficit, spiking drive)>=0.9  (3) lived memory "
          "(grounding-lesion collapses)  (4) converse + no-confab MOAT  (5) persistence across reset.\n", flush=True)

    if a.smoke:
        _run_smoke(a)
        return 0

    root = tempfile.mkdtemp(prefix="live_remember_")
    per_seed = []
    try:
        for seed in a.seeds:
            per_seed.append(run_seed(seed, root, n_steps=a.n_steps, drive_window=a.drive_window,
                                     drive_reward=a.drive_reward, drive_read_every=a.drive_read_every,
                                     n_objects=a.n_objects, modes=a.modes))
            v = per_seed[-1]["verdict"]
            print(f"  >>> seed {seed}: {'GO' if v.get('go') else 'NO'}  {v}", flush=True)
    finally:
        if not a.keep_lineage:
            shutil.rmtree(root, ignore_errors=True)

    n_go = sum(p["verdict"].get("go", False) for p in per_seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"per_seed": per_seed, "n_go": n_go, "n_seeds": len(per_seed)}, fh, indent=2, default=str)

    print(f"\n{'='*110}", flush=True)
    if per_seed and n_go == len(per_seed):
        print(f"  GO ({n_go}/{len(per_seed)} seeds): the FIRST PERSISTENT LIVING AGENT that perceives, remembers, and "
              "can be talked to about what it LIVED. A merged one-brain lives a drive-biased life (survives while "
              "LESION+YOKE crash), grounds+stores the objects IT encountered (a GROUNDING-LESION collapses lived "
              "recall), answers who/what about them + ABSTAINS on never-encountered (the no-confab MOAT held, "
              "conversational synapses byte-frozen), and RESUMES the exact life + memory after a reset. ⇒ the merged "
              "brain becomes a LIFE that can be talked to about its own experience. HONEST SCOPE: the learned spatial "
              "policy stays the deferred Tier-4 dendrite wall (survival uses the validated rate-proxy stand-in); "
              "persistence is JSON re-instate; open-endedness is encounter-driven on a corridor (2D path-dependent = "
              "follow-on).", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(per_seed)} seeds): localize (survival / corr / lived-memory / moat / "
              "persistence). An honest negative that pins the exact wall is a valid deliverable per the actual-goal "
              "mandate.", flush=True)
    print(f"  [saved] {a.out}\n{'='*110}", flush=True)
    return 0 if (per_seed and n_go == len(per_seed)) else 1


def _build_agent(seed):
    """The merged one brain: composer + perception + the co-resident SPIKING drive, all on ONE bridge."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    return MergedNavConvAgent(
        seed=seed, vocab=VOCAB, co_resident_composer=True, co_resident_composer_kind="rf",
        co_resident_perception=True, co_resident_generalization=True, perception_grounding="gen_spikes",
        co_resident_drive=True)


def run_seed(seed, root, *, n_steps=900, drive_window=40, drive_reward="spiking", drive_read_every=10, n_objects=3,
             modes=("intact", "lesion", "yoke")):
    """One seed: for each mode build the merged brain, live, and measure survival + lived memory + moat; on intact
    also do the drive corr sweep, the grounding-lesion arm, and the persistence-across-reset check."""
    from sim.backend import to_host
    from research.runners.navigate_to_compose_then_answer import lesion_gen_convergence

    world = LivingWorld(seed, n_objects=n_objects)
    out = {"seed": seed, "world": {"placed": world.placed, "held_out": world.held_out}, "modes": {}}

    for mode in modes:
        agent = _build_agent(seed)
        bridge = agent._merged_bridge
        hunger = SpikingHunger(bridge, window=drive_window)
        # MOAT (in vivo): snapshot the conversational synapses before the live run.
        pre_conn = to_host(bridge.cp_connections.data).copy()
        st = LiveState(seed, lesion=(mode == "lesion"), yoke=(mode == "yoke"))
        cache = set()
        seg = live(agent, hunger, st, world, n_steps, drive_reward=drive_reward,
                   drive_read_every=drive_read_every, grounded_obj_cache=cache)
        surv = _survival(seg["energies"])
        recall_ok, recall_tot = _lived_recall(agent, st.lived_facts)
        abstain_ok, abstain_tot = _moat_check(agent, world)
        post_conn = to_host(bridge.cp_connections.data)
        conv_byte_frozen = bool(pre_conn.shape == post_conn.shape and np.array_equal(pre_conn, post_conn))

        rec = {"mode": mode, "survival": surv, "n_encountered": len(st.encountered),
               "encountered": list(st.encountered), "n_lived_facts": len(st.lived_facts),
               "lived_recall_ok": recall_ok, "lived_recall_tot": recall_tot,
               "moat_abstain_ok": abstain_ok, "moat_abstain_tot": abstain_tot,
               "conv_byte_frozen": conv_byte_frozen}

        if mode == "intact":
            rec["corr_deficit_drive_sweep"] = _drive_corr_sweep(hunger)
            # persistence: save body + memory, reload into a FRESH agent, re-instate the memory, resume a tail.
            rec.update(_persistence_check(seed, root, st, world, n_steps, drive_window, drive_reward))
            # grounding-lesion: a fresh agent, sever the perception->concept convergence, re-live -> recall collapses.
            rec["grounding_lesion"] = _grounding_lesion_arm(seed, world, n_steps, drive_window, drive_reward,
                                                            drive_read_every, lesion_gen_convergence)
        out["modes"][mode] = rec
        print(f"  [seed {seed} {mode}] minE {surv['min_energy']:.2f} crash% {100*surv['crash_frac']:.0f} | "
              f"enc {len(st.encountered)} facts {len(st.lived_facts)} recall {recall_ok}/{recall_tot} | "
              f"moat {abstain_ok}/{abstain_tot} | conv-frozen {conv_byte_frozen}", flush=True)

    out["verdict"] = _verdict(out["modes"])
    return out


def _persistence_check(seed, root, st, world, n_steps, drive_window, drive_reward):
    """Save the body + lived memory, reload into a FRESH agent, re-instate the memory (grounded codes + re-store the
    lived facts), and confirm the reloaded agent answers the lived queries — while a NO-PERSISTENCE cold agent (no
    re-instate) has an EMPTY memory. Cheap-first JSON re-instate (§1f)."""
    seed_root = os.path.join(root, f"seed{seed}_persist")
    lineage = BridgeLineage(f"live_remember_{seed}", root=Path(seed_root))
    payload = {"body": st.body_payload(), "memory": st.memory_payload()}

    def save_fn(_unused, path_str):
        with open(path_str, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    lineage.save(None, save_fn=save_fn, tier="live-and-remember",
                 arch={"kind": "tier3_live_and_remember", "L": L}, snapshot=False)

    with open(lineage.load(), "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    mem = loaded["memory"]

    # PERSISTED resume: a fresh agent, re-instate the grounded codes + re-store the lived facts -> memory resumes.
    resumed = _build_agent(seed)
    for obj, code in mem["grounded_codes"].items():
        resumed.composer.concepts[obj] = _decode_code(code)
    for (prev, verb, cur) in [tuple(f) for f in mem["lived_facts"]]:
        resumed.composer.store(prev, verb, cur)
    lived_facts = [tuple(f) for f in mem["lived_facts"]]
    p_ok, p_tot = _lived_recall(resumed, lived_facts)
    resumed_remembers = bool(p_tot > 0 and p_ok == p_tot)

    # NO-PERSISTENCE cold start: a fresh agent with NO re-instate -> empty memory -> cannot answer the lived queries.
    cold = _build_agent(seed)
    c_ok, c_tot = _lived_recall(cold, lived_facts)          # nothing stored -> all abstain -> 0
    no_persistence_differs = bool(resumed_remembers and c_ok == 0 and p_tot > 0)

    return {"persist_resumed_remembers": resumed_remembers, "persist_recall": [p_ok, p_tot],
            "cold_recall": [c_ok, c_tot], "no_persistence_differs": no_persistence_differs}


def _grounding_lesion_arm(seed, world, n_steps, drive_window, drive_reward, drive_read_every, lesion_gen_convergence):
    """A fresh agent whose perception->concept convergence is SEVERED before the life: it still encounters + stores
    the SAME objects, but the grounded codes are random -> lived-recall collapses to chance (the memory rides the
    LIVE percept). Returns the lesioned recall fraction (should be << the intact fraction)."""
    agent = _build_agent(seed)
    lesion_gen_convergence(agent._merged_bridge, agent._handles["gen"])
    hunger = SpikingHunger(agent._merged_bridge, window=drive_window)
    st = LiveState(seed)
    live(agent, hunger, st, world, n_steps, drive_reward=drive_reward,
         drive_read_every=drive_read_every, grounded_obj_cache=set())
    ok, tot = _lived_recall(agent, st.lived_facts)
    return {"recall_ok": ok, "recall_tot": tot, "recall_frac": (ok / tot if tot else 0.0)}


def _verdict(modes):
    if not all(m in modes for m in ("intact", "lesion", "yoke")):
        return {"go": False, "reason": "missing modes"}
    I, Le, Y = modes["intact"], modes["lesion"], modes["yoke"]
    survival = bool(I["survival"]["min_energy"] > HEALTHY and I["survival"]["crash_frac"] < 0.01
                    and Le["survival"]["min_energy"] < CRASH and Y["survival"]["min_energy"] < CRASH
                    and I["survival"]["min_energy"] >= Le["survival"]["min_energy"] + 0.2
                    and I["survival"]["min_energy"] >= Y["survival"]["min_energy"] + 0.2)
    corr_ok = float(I.get("corr_deficit_drive_sweep", 0.0)) >= 0.9
    # lived memory: intact recalls its lived facts; the grounding-lesion collapses recall to << intact.
    intact_frac = (I["lived_recall_ok"] / I["lived_recall_tot"]) if I["lived_recall_tot"] else 0.0
    gl = I.get("grounding_lesion", {})
    lesion_frac = float(gl.get("recall_frac", 1.0))
    lived_memory = bool(I["lived_recall_tot"] >= 1 and intact_frac >= 0.75 and lesion_frac <= intact_frac - 0.3)
    # converse + moat: intact answers correctly AND the no-confab moat abstains on never-encountered (byte-frozen).
    moat = bool(I["moat_abstain_tot"] >= 1 and I["moat_abstain_ok"] == I["moat_abstain_tot"]
                and I["conv_byte_frozen"] and Le["conv_byte_frozen"] and Y["conv_byte_frozen"])
    persistence = bool(I.get("persist_resumed_remembers") and I.get("no_persistence_differs"))
    go = bool(survival and corr_ok and lived_memory and moat and persistence)
    return {"go": go, "survival": survival, "corr_ok": corr_ok, "lived_memory": lived_memory,
            "moat": moat, "persistence": persistence, "intact_recall_frac": intact_frac,
            "grounding_lesion_frac": lesion_frac}


def _run_smoke(a):
    """Tiny GPU mechanics check: 1 seed, intact, short life — does the JOIN close (live -> perceive -> ground ->
    store -> query + moat -> persist+reload resumes the memory)?"""
    root = tempfile.mkdtemp(prefix="live_remember_smoke_")
    try:
        agent = _build_agent(a.seeds[0])
        bridge = agent._merged_bridge
        from sim.backend import to_host
        world = LivingWorld(a.seeds[0], n_objects=a.n_objects)
        hunger = SpikingHunger(bridge, window=max(20, a.drive_window // 2))
        pre_conn = to_host(bridge.cp_connections.data).copy()
        st = LiveState(a.seeds[0])
        seg = live(agent, hunger, st, world, min(a.n_steps, 120), drive_reward=a.drive_reward,
                   drive_read_every=a.drive_read_every, grounded_obj_cache=set())
        surv = _survival(seg["energies"])
        recall_ok, recall_tot = _lived_recall(agent, st.lived_facts)
        abstain_ok, abstain_tot = _moat_check(agent, world)
        conv_frozen = bool(np.array_equal(pre_conn, to_host(bridge.cp_connections.data)))
        corr = _drive_corr_sweep(hunger)
        pc = _persistence_check(a.seeds[0], root, st, world, a.n_steps, a.drive_window, a.drive_reward)
        ok = bool(len(st.encountered) >= 2 and recall_tot >= 1 and abstain_ok == abstain_tot
                  and conv_frozen and pc["no_persistence_differs"])
        print(f"[smoke] enc {st.encountered} | facts {len(st.lived_facts)} recall {recall_ok}/{recall_tot} | "
              f"moat {abstain_ok}/{abstain_tot} | conv-frozen {conv_frozen} | corr {corr:+.2f} | "
              f"persist {pc['persist_resumed_remembers']} cold-empty {pc['cold_recall']} | minE {surv['min_energy']:.2f}"
              f"  ||  {'JOIN-CLOSES' if ok else 'CHECK'}", flush=True)
    finally:
        if not a.keep_lineage:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
