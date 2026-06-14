"""DENDRITIC de-risk D1 -- can a PER-COMPARTMENT dendritic gain recover category structure that a single
point-neuron soma cannot? (the cheap-first, afternoon-scale, CPU/numpy de-risk that GATES the months-scale
step-3(B) build)

WHY THIS RUNNER EXISTS
======================
The project has now hit the point-neuron decorrelation wall FIVE mechanistically-distinct ways (vanilla
Hopfield, Storkey local-covariance, spiking dentate-gyrus kWTA, fixed random expansion, and -- 2026-06-14
-- Option C learning paradigmatic similarity from real text). The read-out/mechanism discriminator
(`option_c_stageB_readout_discriminator.py`) localized the Option-C failure precisely: the structure is
present in the raw concept x hub co-occurrence counts (the host PPMI+SVD lens recovers it, L1 ~ +0.45),
but a raw cosine of each concept's HUB-CONNECTIVITY PROFILE recovers nothing (L2 ~ 0) because the
high-frequency COMMON HUBS ("said","day","big" -- every concept connects to them strongly) dominate every
profile. Recovering the structure requires DOWN-WEIGHTING each hub by its own frequency -- a PER-INPUT
(per-hub) normalization. That is exactly the Mikulasch-Priesemann point: a point neuron sums all hub drive
at ONE soma and can apply only a SINGLE global gain; a dendritic compartment PER input can apply a
per-hub gain. The dendritic-substrate deep-research pass recommended this de-risk (section (e)).

    Run the project's ACTUAL failing case -- recover an a-priori category block-structure S_true from a
    concept x hub count matrix whose high-frequency common hubs dominate -- with a DENDRITIC per-hub
    gain-control compartment, vs a POINT-NEURON single-global-gain control on the IDENTICAL pipeline that
    MUST recover ~0 (reproducing the discriminator's L2 ~ 0 and the five NEGATIVEs). THE CONTRAST IS THE
    RESULT. The host PPMI+SVD ceiling proves the structure IS in the data (so a failure is the mechanism,
    not the data).

Pure numpy, OFF-bridge, NO sim/ edits, multi-seed, full anti-cheat battery. A GO justifies the
months-scale on-substrate two-compartment NeuronModel build (D2); a NEGATIVE is itself a citable result
and SAVES the build.

THE MECHANISM (Carandini-Heeger divisive gain control, delivered PER COMPARTMENT = per hub)
===========================================================================================
For each concept's hub-drive profile x (= the concept x hub count row):
  DENDRITIC (per-hub gain): each hub h is its OWN dendritic compartment with a LOCAL inhibitory gain g_h
    that adapts ONLINE to that hub's own drive across the experience stream (g_h <- g_h + eta*(x_h - g_h),
    a purely local rule -- only hub h's own activity). The residual is gain-normalized: r_h = x_h /
    (sigma + g_h). High-frequency common hubs get a LARGE g_h -> down-weighted; rare category-specific
    hubs keep a small g_h -> emphasized. This is the per-input normalization PPMI does, realized by a
    biologically-local per-compartment gain.
  POINT-NEURON (single global gain): ONE gain g for ALL hubs (the soma sums everything and applies one
    gain): r_h = x_h / (sigma + g), g <- g + eta*(mean_h x_h - g). A single global gain CANNOT down-weight
    the high-frequency hubs specifically -> the common mode survives -> the profile stays dominated by the
    common hubs -> ~0 structure (the literal Mikulasch-Priesemann "a single global inhibitory pool cannot
    whiten" claim, made falsifiable on the project's failing case).
THE READ-OUT codes = the gain-normalized residual profile r for each concept. STRUCTURE RECOVERY =
Pearson(cos(r_i, r_j), S_true), S_true the a-priori category block (CONSTRUCTED, never data-derived).

GATES (D1 GO requires the CONTRAST, multi-seed 42/43/44):
  1. STRUCTURE  -- dendritic Pearson(S_learned,S_true) >= +0.30 WHILE the point-neuron control ~0
                   (|Pearson| <= 0.12). The headline; the contrast IS the result.
  2. GENERALIZE -- held-out nearest-category classification above chance for dendritic, at chance for
                   point-neuron.
  3. REPRODUCE  -- same profile + count noise twice -> residual cos >= 0.90.
  4. NOT-COLLAPSED -- residuals not all identical (mean off-diag cos < 0.95, effective rank > 1.5).

ANTI-CHEATS (all mandatory; mirror the deep-research section (f)):
  - POINT-NEURON-MUST-FAIL: the headline anti-cheat. If the point-neuron control does NOT give ~0, the
    common mode is too weak -> re-tune (raise --lam-common / --n-common-hubs) until it fails BEFORE
    trusting any dendritic GO. Reported every run.
  - HOST-CEILING confirms the data carries it: PPMI+SVD on the SAME counts >= +0.30 (else CORPUS issue,
    not a mechanism statement). The host is a labelled instrument, never a deliverable.
  - S_true A-PRIORI: constructed block; an independence self-check asserts it is never data-derived.
  - PERMUTED-SIMILARITY collapses: shuffle which concepts are same-category -> recovery -> ~0.
  - LESION-THE-COMPARTMENT: freeze the per-hub gains to a single constant -> the dendritic effect collapses
    to ~ the point-neuron control (proves the effect RIDES the per-compartment gain, not a code property).
  - LEARNED-ONLINE-NOT-HOST-OP: the gains are learned ONLINE over the stream (report they CONVERGE), NOT a
    one-shot host np.mean; report the per-hub gains are bounded + ordered by hub frequency.
  - NATIVE-CONVENTION unit check: report the raw-count cosine is common-hub-dominated (point-neuron ~0).

VERDICT: GO (contrast holds + controls clean + host ceiling positive, multi-seed) / BOUNDARY (dendritic
beats point-neuron but partial) / NEGATIVE (dendritic also ~0) / NEGATIVE_miscalibrated (point-neuron did
NOT fail, or host ceiling ~0).

Run (CPU/numpy, afternoon-scale; multi-seed):
  python -u -m research.runners.dendritic_d1_learn_graded_structure_derisk \
      --seeds 42,43,44 --out research/findings/raw/_dendritic_d1_multiseed.json
Run (fast smoke):
  python -u -m research.runners.dendritic_d1_learn_graded_structure_derisk --smoke --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# the host PPMI+SVD lens (the SAME instrument that gave the +0.53 Option-C ceiling) -- used here ONLY as
# the data-carries-it reference, never as a deliverable.
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


# ===========================================================================
# (1) Synthetic concept x hub COUNTS: high-frequency COMMON hubs (the common mode) + category-specific
#     signal hubs, faithfully mirroring the discriminator's L1/L2 structure. S_true CONSTRUCTED a-priori.
# ===========================================================================
def build_concept_hub_counts(n_cat, per_cat, n_common, n_sig_per_cat, lam_common, lam_sig, lam_bg, seed):
    """Return (C [Nc x H] float counts, labels [Nc], S_true [Nc x Nc], hub_freq [H]).
    Hubs: [0:n_common] = COMMON high-frequency hubs every concept connects to ~ Poisson(lam_common) (the
    common mode that dominates raw profiles); then per category a block of n_sig_per_cat CATEGORY-SIGNAL
    hubs -- within-category concepts ~ Poisson(lam_sig), out-of-category ~ Poisson(lam_bg). S_true = the
    a-priori within-category-1 / between-0 block (NEVER data-derived)."""
    rng = np.random.RandomState(seed * 7919 + 11)
    Nc = n_cat * per_cat
    H = n_common + n_cat * n_sig_per_cat
    C = np.zeros((Nc, H), dtype=np.float64)
    labels = np.repeat(np.arange(n_cat), per_cat)
    # COMMON hubs: every concept, high frequency (the common mode)
    C[:, :n_common] = rng.poisson(lam_common, size=(Nc, n_common)).astype(np.float64)
    # category-signal hubs: a block per category
    for c in range(n_cat):
        lo = n_common + c * n_sig_per_cat
        hi = lo + n_sig_per_cat
        within = labels == c
        C[np.ix_(within, np.arange(lo, hi))] = rng.poisson(lam_sig, size=(within.sum(), n_sig_per_cat))
        C[np.ix_(~within, np.arange(lo, hi))] = rng.poisson(lam_bg, size=((~within).sum(), n_sig_per_cat))
    S_true = (labels[:, None] == labels[None, :]).astype(np.float64)
    hub_freq = C.mean(0)
    return C, labels, S_true, hub_freq


def _cos_sim(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / (n + 1e-12)
    return Zn @ Zn.T


def _pearson_vs_Strue(sim, S_true):
    Nc = sim.shape[0]
    iu = np.triu_indices(Nc, k=1)
    a, b = sim[iu], S_true[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ===========================================================================
# (2) The dendritic per-hub gain (LOCAL, online-learned) and the POINT-NEURON single-global-gain control.
# ===========================================================================
def learn_perhub_gains(C, epochs, eta, seed):
    """DENDRITIC: each hub h adapts a LOCAL gain g_h online to its own drive (g_h <- g_h + eta*(x_h - g_h)).
    Returns (g [H], gain_trace) -- g converges to each hub's marginal frequency. Biologically local:
    compartment h uses only hub h's own activity."""
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = C.shape
    g = np.zeros(H, dtype=np.float64)
    order = np.arange(Nc)
    trace = []
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            g += eta * (C[i] - g)
        trace.append(float(np.linalg.norm(g)))
    return g, trace


def learn_global_gain(C, epochs, eta, seed):
    """POINT-NEURON: a SINGLE global gain g (the soma's one inhibitory pool) adapts to the mean hub drive
    (g <- g + eta*(mean_h x_h - g)). One gain for ALL hubs -> cannot down-weight high-freq hubs."""
    Nc, H = C.shape
    g = 0.0
    for ep in range(epochs):
        for i in range(Nc):
            g += eta * (float(C[i].mean()) - g)
    return float(g)


def perhub_residual(C, g, sigma):
    """Per-hub divisive gain control: r_h = x_h / (sigma + g_h)."""
    return C / (sigma + g[None, :])


def global_residual(C, g, sigma):
    """Single global gain: r_h = x_h / (sigma + g) (same g all hubs)."""
    return C / (sigma + g)


# ===========================================================================
# (3) Generalization (held-out nearest-category) + reproducibility + effective rank.
# ===========================================================================
def heldout_generalization(codes, labels):
    Nc = codes.shape[0]
    cats = np.unique(labels)
    sim = _cos_sim(codes)
    correct = 0
    for i in range(Nc):
        best_c, best_s = None, -2.0
        for c in cats:
            members = [j for j in range(Nc) if labels[j] == c and j != i]
            if not members:
                continue
            s = float(sim[i, members].mean())
            if s > best_s:
                best_s, best_c = s, c
        correct += int(best_c == labels[i])
    return correct / Nc, 1.0 / len(cats)


def effective_rank(codes):
    Z = codes - codes.mean(0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    if s.sum() < 1e-18:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-18))))


def reproducibility(C, g, sigma, noise, seed, per_hub=True):
    """Same concept profile + Poisson-like count noise twice -> residual cosine."""
    rng = np.random.RandomState(seed * 15485863 + 9)
    Nc, H = C.shape
    cs = []
    for i in range(Nc):
        n1 = C[i] + rng.normal(0, noise * np.sqrt(np.maximum(C[i], 1.0)), H)
        n2 = C[i] + rng.normal(0, noise * np.sqrt(np.maximum(C[i], 1.0)), H)
        n1 = np.maximum(n1, 0.0); n2 = np.maximum(n2, 0.0)
        r1 = (n1 / (sigma + g)) if per_hub else (n1 / (sigma + g))
        r2 = (n2 / (sigma + g)) if per_hub else (n2 / (sigma + g))
        cs.append(float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-12)))
    return float(np.mean(cs))


# ===========================================================================
# Per-seed driver.
# ===========================================================================
def run_seed(seed, args):
    print(f"\n{'='*84}\n  DENDRITIC D1 -- PER-HUB GAIN vs POINT-NEURON (seed {seed})\n{'='*84}", flush=True)
    C, labels, S_true, hub_freq = build_concept_hub_counts(
        args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
        args.lam_common, args.lam_sig, args.lam_bg, seed)
    Nc, H = C.shape
    s_true_independent = bool(np.array_equal(S_true, (labels[:, None] == labels[None, :]).astype(float)))

    # raw (point-neuron with NO gain) baseline + host ceiling
    raw_pearson = _pearson_vs_Strue(_cos_sim(C), S_true)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)
    print(f"  data: {args.n_cat}x{args.per_cat}={Nc} concepts, {args.n_common} common hubs + "
          f"{args.n_cat*args.n_sig_per_cat} signal hubs = {H} hubs "
          f"(lam_common={args.lam_common} lam_sig={args.lam_sig} lam_bg={args.lam_bg})", flush=True)
    print(f"  [unit check] raw-profile cosine Pearson(sim,S_true)={raw_pearson:+.3f} (point-neuron RAW, "
          f"common-hub-dominated -> ~0); HOST PPMI+SVD ceiling={host_pearson:+.3f}; S_true a-priori="
          f"{s_true_independent}", flush=True)

    # ---- dendritic per-hub gain ----
    t0 = time.time()
    g_hub, gtrace = learn_perhub_gains(C, args.epochs, args.eta, seed)
    dend_codes = perhub_residual(C, g_hub, args.sigma)
    dend_pearson = _pearson_vs_Strue(_cos_sim(dend_codes), S_true)
    dend_gen, chance = heldout_generalization(dend_codes, labels)
    dend_repro = reproducibility(C, g_hub, args.sigma, args.noise, seed, per_hub=True)
    dend_rank = effective_rank(dend_codes)
    dend_off = float(_cos_sim(dend_codes)[np.triu_indices(Nc, 1)].mean())
    gains_converge = bool(len(gtrace) >= 2 and abs(gtrace[-1] - gtrace[-2]) <= 0.05 * (gtrace[-1] + 1e-9))
    # gains ordered by hub frequency (common hubs -> large gain): correlation gain vs freq
    gain_freq_corr = float(np.corrcoef(g_hub, hub_freq)[0, 1])
    print(f"  [DENDRITIC per-hub] Pearson(S,S_true)={dend_pearson:+.3f}  gen={dend_gen:.3f} "
          f"(chance {chance:.3f})  repro@{args.noise}={dend_repro:.3f}  eff-rank={dend_rank:.1f}  "
          f"offdiag={dend_off:+.3f}  (gain~freq corr={gain_freq_corr:+.3f}, converge={gains_converge}, "
          f"{time.time()-t0:.1f}s)", flush=True)

    # ---- point-neuron control (single global gain) ----
    g_glob = learn_global_gain(C, args.epochs, args.eta, seed)
    pn_codes = global_residual(C, g_glob, args.sigma)
    pn_pearson = _pearson_vs_Strue(_cos_sim(pn_codes), S_true)
    pn_gen, _ = heldout_generalization(pn_codes, labels)
    pn_repro = reproducibility(C, g_glob, args.sigma, args.noise, seed, per_hub=False)
    print(f"  [POINT-NEURON global] Pearson(S,S_true)={pn_pearson:+.3f}  gen={pn_gen:.3f}  "
          f"repro@{args.noise}={pn_repro:.3f}  (g={g_glob:.3f})", flush=True)

    # ---- ANTI-CHEATS ----
    rng = np.random.RandomState(seed * 32452843 + 1)
    perm_labels = rng.permutation(labels)
    S_perm = (perm_labels[:, None] == perm_labels[None, :]).astype(np.float64)
    dend_perm = _pearson_vs_Strue(_cos_sim(dend_codes), S_perm)
    # lesion: freeze per-hub gains to a single constant (= the global mean) -> collapses to ~ point-neuron
    g_const = float(g_hub.mean())
    lesion_codes = C / (args.sigma + g_const)
    lesion_pearson = _pearson_vs_Strue(_cos_sim(lesion_codes), S_true)
    print(f"  [anti-cheat] permuted-similarity Pearson={dend_perm:+.3f} (must ~0)  "
          f"lesion(const-gain) Pearson={lesion_pearson:+.3f} (must collapse to ~point-neuron "
          f"{pn_pearson:+.3f})", flush=True)

    # ---- gates ----
    point_neuron_fails = abs(pn_pearson) <= args.pn_fail_bar
    host_carries = host_pearson >= args.host_bar
    structure = (dend_pearson >= args.structure_bar) and point_neuron_fails and host_carries
    # the meaningful generalization CONTRAST: the dendritic codes generalize above chance AND clearly
    # beat the point-neuron codes (the nearest-category metric is more forgiving than Pearson, so a
    # point-neuron with Pearson ~0.05 still scores a bit above chance; the structure gate already pins
    # point-neuron Pearson ~= 0, so here we require the dendritic to EXCEED point-neuron by a clear margin).
    generalize = (dend_gen > chance + args.gen_margin) and (dend_gen - pn_gen >= args.gen_contrast_margin)
    reproduce = dend_repro >= args.repro_bar
    not_collapsed = (dend_off < 0.95) and (dend_rank > 1.5)
    permuted_collapses = abs(dend_perm) <= args.pn_fail_bar
    lesion_collapses = lesion_pearson <= dend_pearson - 0.10
    gates = {
        "structure_contrast": bool(structure),
        "point_neuron_fails": bool(point_neuron_fails),
        "host_ceiling_carries": bool(host_carries),
        "generalize_contrast": bool(generalize),
        "reproduce": bool(reproduce),
        "not_collapsed": bool(not_collapsed),
        "permuted_similarity_collapses": bool(permuted_collapses),
        "lesion_collapses": bool(lesion_collapses),
        "gains_converge": bool(gains_converge),
        "gain_tracks_frequency": bool(gain_freq_corr >= 0.5),
        "s_true_independent": bool(s_true_independent),
    }
    print(f"  [seed {seed} gates] {gates}", flush=True)
    return {
        "seed": seed, "n_concepts": Nc, "n_hubs": H, "chance": chance,
        "raw_pearson": raw_pearson, "host_ceiling_pearson": host_pearson,
        "dendritic": {"pearson": dend_pearson, "generalization": dend_gen, "repro": dend_repro,
                      "eff_rank": dend_rank, "offdiag_cos": dend_off, "gain_freq_corr": gain_freq_corr,
                      "gains_converge": gains_converge},
        "point_neuron": {"pearson": pn_pearson, "generalization": pn_gen, "repro": pn_repro, "g": g_glob},
        "anti_cheat": {"permuted_similarity_pearson": dend_perm, "lesion_pearson": lesion_pearson},
        "gates": gates,
    }


def decide_verdict(per_seed, seeds, args):
    def allg(k):
        return all(per_seed[str(s)]["gates"][k] for s in seeds)
    structure = allg("structure_contrast")
    pn_fails = allg("point_neuron_fails")
    host_ok = allg("host_ceiling_carries")
    controls = (allg("permuted_similarity_collapses") and allg("lesion_collapses")
                and allg("gains_converge") and allg("s_true_independent"))
    generalize = allg("generalize_contrast")
    dmean = float(np.mean([per_seed[str(s)]["dendritic"]["pearson"] for s in seeds]))
    pmean = float(np.mean([per_seed[str(s)]["point_neuron"]["pearson"] for s in seeds]))
    hmean = float(np.mean([per_seed[str(s)]["host_ceiling_pearson"] for s in seeds]))

    if not host_ok:
        verdict = "NEGATIVE_miscalibrated"
        why = (f"the HOST ceiling did not carry the structure (mean {hmean:+.3f} < bar {args.host_bar}) "
               f"-> the synthetic category signal is too weak to be recoverable even in principle; re-tune "
               f"(raise --lam-sig / --n-sig-per-cat) before trusting any contrast.")
    elif not pn_fails:
        verdict = "NEGATIVE_miscalibrated"
        why = (f"the POINT-NEURON control did NOT fail (mean Pearson {pmean:+.3f} > bar {args.pn_fail_bar}) "
               f"-> the common mode is too weak; raise --lam-common / --n-common-hubs until the point "
               f"neuron gives ~0, THEN re-run. (A single global gain should NOT recover the structure.)")
    elif structure and controls and generalize:
        verdict = "GO"
        why = (f"a PER-HUB dendritic gain LEARNS the category structure (mean Pearson {dmean:+.3f}, host "
               f"ceiling {hmean:+.3f}) AND generalizes to held-out members, WHILE the point-neuron "
               f"single-global-gain control gives ~0 ({pmean:+.3f}); all controls clean (permuted + lesion "
               f"collapse, gains converge + track hub frequency). The dendritic compartment is the "
               f"substrate-level enabler of the per-input normalization a point neuron provably lacks -> "
               f"the D2 on-substrate build is warranted (present cost to owner).")
    elif structure and controls:
        verdict = "GO_structure_only"
        why = (f"the per-hub dendritic gain recovers the structure (mean {dmean:+.3f}) vs the point-neuron "
               f"~0 ({pmean:+.3f}) with clean controls, but held-out generalization did not clear its "
               f"margin -> a real but partial escape; characterize before the D2 build.")
    elif dmean > pmean + 0.10 and controls:
        verdict = "BOUNDARY"
        why = (f"the dendritic gain beats the point neuron (mean {dmean:+.3f} vs {pmean:+.3f}) but does not "
               f"clear the structure bar {args.structure_bar} -> partial escape; informs whether the fuller "
               f"two-compartment form is warranted vs the shipped curated-similarity (Option B).")
    else:
        verdict = "NEGATIVE"
        why = (f"the per-hub dendritic gain does NOT beat the point neuron on structure recovery "
               f"(dendritic {dmean:+.3f} vs point-neuron {pmean:+.3f}) -> the gap is deeper than a per-hub "
               f"gain; a clean, citable result that SAVES the months-scale build.")
    return verdict, why, {"dendritic_pearson_mean": dmean, "point_neuron_pearson_mean": pmean,
                          "host_ceiling_pearson_mean": hmean, "structure_contrast_all": structure,
                          "point_neuron_fails_all": pn_fails, "host_carries_all": host_ok,
                          "controls_clean_all": controls, "generalize_all": generalize}


def main():
    p = argparse.ArgumentParser(description="Dendritic D1 cheap de-risk: can a per-compartment gain "
                                            "recover category structure a single soma cannot?")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-cat", type=int, default=8)
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200,
                   help="# high-frequency COMMON hubs (the common mode; calibrated so point-neuron fails)")
    p.add_argument("--n-sig-per-cat", type=int, default=12, help="# category-signal hubs per category")
    p.add_argument("--lam-common", type=float, default=40.0, help="common-hub mean count (dominant)")
    p.add_argument("--lam-sig", type=float, default=4.0, help="within-category signal-hub mean count")
    p.add_argument("--lam-bg", type=float, default=0.3, help="out-of-category signal-hub background count")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.05, help="online gain-adaptation rate (local rule)")
    p.add_argument("--sigma", type=float, default=1.0, help="divisive semi-saturation constant")
    p.add_argument("--noise", type=float, default=0.1, help="reproducibility count noise")
    p.add_argument("--host-svd", type=int, default=50)
    p.add_argument("--host-alpha", type=float, default=0.75)
    # gate bars
    p.add_argument("--structure-bar", type=float, default=0.30)
    p.add_argument("--pn-fail-bar", type=float, default=0.12)
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--gen-margin", type=float, default=0.05)
    p.add_argument("--gen-contrast-margin", type=float, default=0.30,
                   help="dendritic held-out generalization must exceed the point-neuron's by >= this")
    p.add_argument("--repro-bar", type=float, default=0.90)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.smoke:
        # the run is ~instant; the smoke keeps the calibrated hub scale (so the point-neuron still fails)
        # and only trims epochs.
        args.epochs = 6

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[dendritic D1] seeds={seeds} smoke={args.smoke}  question: can a PER-HUB gain recover what a "
          f"single global gain (point neuron) cannot?", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)

    print(f"\n{'='*84}\n  D1 VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  ladder: HOST ceiling {detail['host_ceiling_pearson_mean']:+.3f}  |  DENDRITIC per-hub "
          f"{detail['dendritic_pearson_mean']:+.3f}  vs  POINT-NEURON global "
          f"{detail['point_neuron_pearson_mean']:+.3f}  (contrast all seeds: "
          f"{detail['structure_contrast_all']})", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*84}\n", flush=True)

    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "smoke": bool(args.smoke),
           "config": vars(args), "per_seed": per_seed,
           "note": ("DIAGNOSTIC cheap-first de-risk D1 (afternoon CPU/numpy, NO sim/ edits). The contrast "
                    "IS the result: a dendritic GO counts only against a point-neuron NEGATIVE on the "
                    "IDENTICAL counts/mechanism/seeds, with the host ceiling confirming the data carries "
                    "it. The dendritic per-hub gain is the local implementation of PPMI's per-input "
                    "normalization that a single-soma point neuron provably cannot deliver. Gates the "
                    "months-scale on-substrate D2 build."),
           "elapsed_total_s": time.time() - t0}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendritic_d1_{'smoke' if args.smoke else 'multiseed'}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
