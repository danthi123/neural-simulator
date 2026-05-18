"""Q2R: a FRESH larger-KB experiment of the VALIDATED Q2 constrained-
decoding mechanism. This module introduces NO new mechanism -- it
IMPORTS the validated Q2 per-token grounded constrained-decoding LM
(_GroundedConstrainedLM) and the validated Q2 soundness instrument
(cdc_verdict) BYTE-UNMODIFIED, and runs them over a NET-NEW, larger,
genuinely-varied knowledge base across the ladder K in {12,24,48,96}.
Every rung runs FRESH (each rung is a real fresh decode pass over the
new KB; this is NOT a re-score of any Q2 result). The cross-rung
trend is aggregated by the already-built, a-priori-frozen
q2r_core.q2r_scale_confidence (its bars are NEVER tuned here).

HONEST CEILING (never spun): this is a scale-confidence proof-of-
concept. It only asks whether the VALIDATED constrained-decoding
faithfulness holds (and does not degrade) as a genuine local KB scales
up, and whether it still clears the SAME 0.50 non-vacuity floor at the
largest local scale with the validated instrument unmodified. It is
NOT an LLM, NOT a fluent or GPT-class model, NOT open-ended fluent
composition, NOT conversation-solved. Constrained decoding TRADES
fluency for faithfulness BY DESIGN; that trade is the point of the
mechanism, not a defect to be explained away.

Generator-F inference is reused via the imported _GroundedConstrainedLM
(INFERENCE ONLY -- no new training, no autograd, no gradient-descent
state, no training objective; the imported class already places the
module in inference mode internally, so this module never needs to).
CUDA when available. ASCII.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

from sim.grounded_decode import grounded_decode
from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD
from research.runners.generator_g_core import ungrounded_entity_rate, \
    FUNCTION_WORDS
from research.runners.constrained_decode_gate import _GroundedConstrainedLM
from research.runners.constrained_decode_core import (
    cdc_verdict, nonvacuous_answered)
from research.runners.q2r_core import q2r_scale_confidence, _Q2R_LADDER

# SAME Generator-F prefix path string constrained_decode_gate uses.
_GEN_F = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"

# Net-new, larger, genuinely-varied KB. Authored by hand across many
# domains (animals, crafts, weather, food, tools, nature, music,
# travel, sea, garden, sky, etc). Each proposition is a simple
# TinyStories-style sentence with >= 4 CONTENT words (function words
# per generator_g_core.FUNCTION_WORDS removed), and the normalized
# content-word set of EVERY proposition is pairwise-distinct from
# every other's. NOT a template, NOT a permuted tiny word pool, NOT
# auto-generated -- subjects, verbs, objects and adjectives all vary.
_Q2R_GROUNDED = {
 "ada": "ada studies the old star maps",
 "bo": "bo carved a whistle from cedar",
 "cy": "cy spilled warm cocoa on snow",
 "di": "di trains a clever grey parrot",
 "ed": "ed welds a broken iron gate",
 "fern": "fern brews tea from wild mint",
 "gio": "gio juggles five painted wooden clubs",
 "hana": "hana folds a paper crane swiftly",
 "ira": "ira maps a hidden cave river",
 "jun": "jun tunes a cracked cello string",
 "kit": "kit races a small wooden sail cart",
 "lev": "lev sketches a tall harbor crane",
 "mira": "mira bakes a spiced apple pie",
 "nico": "nico repairs a rusty bicycle chain",
 "ola": "ola knits a striped winter scarf",
 "pax": "pax plants a row of beans",
 "quin": "quin polishes a dull brass lamp",
 "ria": "ria paddles a leaky canoe upstream",
 "soren": "soren grinds fresh pepper for stew",
 "tilda": "tilda braids a long horse mane",
 "umi": "umi nets a silver leaping trout",
 "vesna": "vesna stitches a quilt from rags",
 "wren": "wren whittles a duck from driftwood",
 "xan": "xan flies a striped box kite",
 "yara": "yara dyes wool with crushed berries",
 "zeb": "zeb shoes a restless brown mare",
 "arlo": "arlo collects shells along rocky tide pools",
 "bex": "bex roasts chestnuts over glowing coals",
 "ciro": "ciro carves runes into smooth stones",
 "dot": "dot waters ferns in clay troughs",
 "esme": "esme tracks deer through frosty woods",
 "finn": "finn casts bronze bells in sand",
 "gala": "gala spins flax into fine thread",
 "hugo": "hugo prunes thorny rose hedges carefully",
 "ines": "ines charts distant comets each autumn",
 "jad": "jad forges a curved garden blade",
 "kemi": "kemi drums a steady marching beat",
 "liu": "liu paints misty mountain waterfalls",
 "moss": "moss gathers dry kindling at dusk",
 "nour": "nour kneads dough for crusty loaves",
 "oki": "oki carves jade into tiny turtles",
 "pia": "pia pumps air into flat tires",
 "quill": "quill sorts buttons by faded color",
 "remy": "remy smokes herring over oak chips",
 "sage": "sage grafts pear shoots onto plum",
 "tovi": "tovi rakes amber leaves into mounds",
 "ula": "ula sews tiny brass bells on cloth",
 "vito": "vito hammers copper into shallow bowls",
 "wim": "wim splices frayed sailing rope tightly",
 "xenia": "xenia presses violets between heavy books",
 "yusuf": "yusuf herds goats up steep slopes",
 "zola": "zola weaves baskets from split willow",
 "abe": "abe sands rough planks until smooth",
 "brisa": "brisa bottles honey from wild hives",
 "cleo": "cleo trims hooves on tired ponies",
 "drev": "drev solders thin wires onto chips",
 "elsa": "elsa freezes plums for winter jam",
 "fitz": "fitz climbs a creaking pine ladder",
 "gwen": "gwen hauls nets full of crabs",
 "haru": "haru sharpens dull garden shears keenly",
 "isla": "isla traces fossil prints in clay",
 "jem": "jem brews bitter coffee at dawn",
 "kade": "kade chops oak logs for fences",
 "lumi": "lumi sketches glowing northern auroras",
 "marek": "marek tames a wild marsh pony",
 "nia": "nia plucks ripe figs from branches",
 "orin": "orin maps quiet underground tunnels",
 "petra": "petra mends torn fishing nets nightly",
 "qadir": "qadir grills spicy lamb skewers",
 "rosa": "rosa transplants seedlings into deep beds",
 "saul": "saul tunes a wheezing old organ",
 "tess": "tess paddles past floating lily pads",
 "uri": "uri stacks mossy river stones high",
 "vera": "vera embroiders gold thread on silk",
 "wade": "wade dredges mud from clogged ditches",
 "xio": "xio folds origami frogs from leaves",
 "yael": "yael grinds barley into coarse meal",
 "zane": "zane patches a leaking canvas tent",
 "amaru": "amaru herds llamas across windy plateaus",
 "bru": "bru smelts tin in a furnace",
 "cael": "cael whistles to circling sheepdogs",
 "deja": "deja salts cod on wooden racks",
 "enzo": "enzo bends willow into round hoops",
 "freya": "freya gathers seaweed at low tide",
 "gunn": "gunn splits flint for sharp arrowheads",
 "hira": "hira spins clay pots on wheels",
 "ivo": "ivo nails shingles onto a roof",
 "juno": "juno charts tidal currents near reefs",
 "kira": "kira distills lavender into fragrant oil",
 "loki": "loki chases moths around lantern light",
 "mona": "mona threads beads onto thin cord",
 "neel": "neel rows passengers across calm lakes",
 "ona": "ona carves spoons from apple wood",
 "piet": "piet thatches a barn with reeds",
 "quru": "quru tracks foxes through dunes",
 "rune": "rune engraves names onto silver rings",
 "suvi": "suvi ferments cabbage in stone crocks",
 "tariq": "tariq trains hawks to return swiftly",
 "uma": "uma builds birdhouses from scrap timber",
 "vidal": "vidal trims grapevines before first frost",
 "wila": "wila churns cream into pale butter",
}
# Net-new nonsense queries; none equals any subject key above.
_Q2R_UNGROUNDED = ["xthar", "qoom", "vlex", "druskin", "plimp",
                   "wozzle"]


def _params(tiny):
    if tiny:
        return dict(ladder=(_Q2R_LADDER[0],), max_new=12,
                    n_ungrounded=3)
    return dict(ladder=_Q2R_LADDER, max_new=40, n_ungrounded=6)


def _q2r_run_rung(K, seeds, lm_c, lm_u, lm_s, max_new, n_ung):
    items = list(_Q2R_GROUNDED.items())[:K]
    props = [p for _, p in items]
    ung = list(_Q2R_UNGROUNDED)[:n_ung]
    per_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = list(range(len(items)))
        rng.shuffle(order)
        c_uer, u_uer, s_uer, c_nv, s_nv = [], [], [], [], []
        for idx in order:
            subj, prop = items[idx]
            ranked = [(subj, 900.0, "kb")]
            r = grounded_decode(ranked, lm_c, lm_c.tok,
                                retrieved_text=prop, query=subj,
                                threshold=DEFAULT_THRESHOLD,
                                max_new=max_new)
            ct = r["text"] or ""
            c_uer.append(ungrounded_entity_rate(ct, prop))
            c_nv.append(1.0 if nonvacuous_answered(ct, prop) else 0.0)
            ru = grounded_decode(ranked, lm_u, lm_u.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            u_uer.append(ungrounded_entity_rate(ru["text"] or "", prop))
            # WEAK#2: RNG-permuted shuffle source -- a per-seed
            # rng-chosen DIFFERENT proposition (asserted != idx), not
            # the fixed (idx+1)%len neighbour. Strengthens control
            # independence; deterministic per seed (same rng stream).
            if len(props) > 1:
                sidx = int(rng.integers(0, len(props) - 1))
                if sidx >= idx:
                    sidx += 1
            else:
                sidx = idx
            assert sidx != idx or len(props) <= 1
            lm_s._shuffle_text = props[sidx]
            rs = grounded_decode(ranked, lm_s, lm_s.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            st_ = rs["text"] or ""
            s_uer.append(ungrounded_entity_rate(st_, prop))
            s_nv.append(1.0 if nonvacuous_answered(st_, prop) else 0.0)
        n_abst = bare = 0
        for subj in ung:
            if gate([], DEFAULT_THRESHOLD) is None:
                bare += 1
            ra = grounded_decode([], lm_c, lm_c.tok, retrieved_text="",
                                 query=subj, threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            if ra["abstained"]:
                n_abst += 1
        nu = max(1, len(ung))
        # Fix-B per-seed instrument-validity metric: fraction of the K
        # KB props whose ALL content words are emittable under the
        # constructed faithful mask (each content word's enc(w) fully
        # traversable: prefixes + full present in the automaton).
        mt_emit = lm_c._props_fully_emittable_rate(props)
        per_seed[seed] = {
            "unconstrained_uer": float(np.mean(u_uer)),
            "constrained_uer": float(np.mean(c_uer)),
            "constrained_nonvac_rate": float(np.mean(c_nv)),
            "shuffled_uer": float(np.mean(s_uer)),
            "shuffled_nonvac_rate": float(np.mean(s_nv)),
            "bare_moat_abstain_rate": bare / nu,
            "abstain_on_ungrounded_rate": n_abst / nu,
            "constrained_multitoken_emittable_rate": float(mt_emit)}
    verdict = cdc_verdict(per_seed)
    nv_mean = float(np.mean(
        [per_seed[s]["constrained_nonvac_rate"] for s in per_seed]))
    return {"K": K, "verdict": verdict,
            "constrained_nonvac_rate_mean": nv_mean}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt", default="research/findings/raw/g11_bg/"
                    "q2r_gate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists(_GEN_F + ".pt")
            and os.path.exists(_GEN_F + ".bpe.json")):
        print("NOT-RUNNABLE: Generator-F artifact absent"); return 2
    P = _params(a.tiny)
    lm_c = _GroundedConstrainedLM(_GEN_F, mode="constrained")
    lm_u = _GroundedConstrainedLM(_GEN_F, mode="unconstrained")
    lm_s = _GroundedConstrainedLM(_GEN_F, mode="shuffled")
    print("DEVICE=%s (CUDA=%s) -- decisive run MUST be cuda"
          % (lm_c.device, lm_c._torch.cuda.is_available()))
    resume = str(a.ckpt) + ".resume.json"
    done = {}
    if Path(resume).exists():
        try:
            done = {int(k): v for k, v in json.loads(
                Path(resume).read_text()).get("done", {}).items()}
        except (ValueError, OSError):
            done = {}
    rungs = []
    t0 = time.time()
    try:
        for K in P["ladder"]:
            if K in done:
                rungs.append(done[K]); continue
            rg = _q2r_run_rung(K, a.seeds, lm_c, lm_u, lm_s,
                               P["max_new"], P["n_ungrounded"])
            rungs.append(rg); done[K] = rg
            tmp = resume + ".tmp"
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp).write_text(json.dumps(
                {"done": {str(k): v for k, v in done.items()}}))
            os.replace(tmp, resume)
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial resume flushed; resumable")
        return 130
    sc = (q2r_scale_confidence(rungs) if not a.tiny else
          {"scale_confident": False,
           "classification": "TINY (toy; NOT propagated)"})
    out = {"ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"],
           "scale_reason": sc.get("reason", ""),
           "device": lm_c.device, "tiny": bool(a.tiny),
           "note": ("TINY toy verdict -- NOT propagated" if a.tiny else
                    "multi-rung trend-primary scale-confidence verdict "
                    "-- recompute from this JSON; no re-run/no tuning"),
           "HONEST_CEILING": ("scale-confidence PoC: validated "
             "constrained-decoding faithfulness holds/improves up a "
             "genuine local KB ladder and clears the SAME 0.50 floor "
             "at the largest local scale with the validated instrument "
             "unmodified; NOT open-ended fluent composition, NOT an "
             "LLM, NOT GPT-class, NOT conversation-solved; constrained "
             "decoding TRADES fluency for faithfulness BY DESIGN")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("SCALE=%s class=%s device=%s"
          % (out["scale_confident"], out["scale_classification"],
             out["device"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
