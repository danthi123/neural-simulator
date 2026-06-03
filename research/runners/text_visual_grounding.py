"""Production build, step 1: a GPU bridge that reads TEXT-AS-PIXELS through the real visual pathway
(retina -> V1_simple -> V1_complex -> V2 -> IT), at a configurable (un-capped) retina size, with Gabor V1
weights scaled to the retina. This replaces the tokenizer's orthogonal lang_input with EARNED visual
transduction (the owner's input-side-fidelity fix). This step: construct the bridge, install scaled Gabor V1,
render a word as pixels, drive the retina, step, and confirm the visual hierarchy RESPONDS (V1/V2/IT fire).
Word-recognition learning (STDP V2/IT readout) is step 2.

Reuses sim.visual_cortex (Gabor V1) + the g11 visual region/pathway pattern + the standard region-framework
bridge construction. GPU (CuPy) when available. No protected-module change.

  python -m research.runners.text_visual_grounding --retina 64
"""
from __future__ import annotations
import argparse, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sim.visual_cortex as VC


def render_word_image(word, rs, font):
    """Render a word as an (2, rs, rs) ON/OFF image for the retina."""
    img = Image.new("L", (rs, rs), 0)
    d = ImageDraw.Draw(img)
    band = rs // max(1, len(word))
    for i, ch in enumerate(word):
        d.text((i * band + max(1, rs // 32), rs // 3, ), ch, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([a, (1.0 - a) * (a.max() > 0)]).astype(np.float32)


def build_scaled_gabor_v1_weights(retina_size, n_orientations, n_frequencies, n_positions_per_dim):
    """Like VC.build_v1_simple_weights but Gabor freqs/sigmas/RF scaled by retina_size/32 (letter-scale)."""
    scale = retina_size / 32.0
    stride = retina_size // n_positions_per_dim
    base_freqs = [0.05, 0.10, 0.20, 0.40]; base_sig = [3.0, 2.5, 2.0, 1.5]
    freqs = [f / scale for f in base_freqs]; sigmas = [s * scale for s in base_sig]
    rf = max(4, int(4 * scale))
    thetas = [i * math.pi / n_orientations for i in range(n_orientations)]
    pre, post, wts = [], [], []
    for oi, theta in enumerate(thetas):
        for fi in range(n_frequencies):
            k = VC.gabor_kernel(sigmas[fi], sigmas[fi], theta, freqs[fi], phase=0.0)
            for py in range(n_positions_per_dim):
                for px in range(n_positions_per_dim):
                    cx, cy = px * stride + stride // 2, py * stride + stride // 2
                    v1 = (oi * (n_frequencies * n_positions_per_dim ** 2)
                          + fi * (n_positions_per_dim ** 2) + py * n_positions_per_dim + px)
                    for dy in range(-rf, rf + 1):
                        for dx in range(-rf, rf + 1):
                            x, y = cx + dx, cy + dy
                            if not (0 <= x < retina_size and 0 <= y < retina_size):
                                continue
                            wv = k(dx, dy)
                            if abs(wv) < 0.01:
                                continue
                            ch = 0 if wv > 0 else 1
                            pre.append(ch * retina_size * retina_size + y * retina_size + x)
                            post.append(v1); wts.append(abs(wv))
    return (np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(wts, np.float32))


def build_visual_text_bridge(retina_size=64, n_orientations=8, n_frequencies=4, n_v2=256, n_it=64, seed=42,
                             word_pools=None, n_per_pool=64, verbose=True):
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    npos = retina_size // 2
    n_retina = 2 * retina_size * retina_size
    n_v1s = n_orientations * n_frequencies * npos * npos
    n_v1c = n_orientations * npos * npos

    def vis_region(name, n, exc=1.0, dens=0.0, ew=0.0, iw=0.0, jit=0.0, plast=False):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=exc, internal_density=dens,
                           exc_weight_mean=ew, inh_weight_mean=iw, weight_jitter=jit, plastic_internal=plast)
    regions = [
        vis_region("retina", n_retina),
        vis_region("cortex_v1_simple", n_v1s),
        vis_region("cortex_v1_complex", n_v1c),
        vis_region("cortex_v2", n_v2, exc=0.8, dens=0.05, ew=2.0, iw=4.0, jit=0.2, plast=True),
        vis_region("cortex_it", n_it, exc=0.8, dens=0.10, ew=2.0, iw=4.0, jit=0.2, plast=True),
    ]
    pathways = [
        RegionPathway(from_region="retina", to_region="cortex_v1_simple", density=0.05,
                      weight_mean=0.5, weight_jitter=0.5, plastic=True, plasticity_gate="visual_cortex_v1"),
        # text is SPARSE (thin strokes) vs g11's dense gridworld blocks -> few V1s spikes; strengthen the
        # phase-pooling weight so sparse coincidences still fire V1_complex (diagnosed break point).
        RegionPathway(from_region="cortex_v1_simple", to_region="cortex_v1_complex",
                      density=4.0 * n_frequencies / float(n_v1s), weight_mean=20.0, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="cortex_v1_complex", to_region="cortex_v2", density=0.15,
                      weight_mean=6.0, weight_jitter=0.5, plastic=True, plasticity_gate="visual_cortex_v2"),
        RegionPathway(from_region="cortex_v2", to_region="cortex_it", density=0.25,
                      weight_mean=7.0, weight_jitter=0.5, plastic=True, plasticity_gate="visual_cortex_it"),
    ]
    # step-2a: word-recognition pools fed by a PLASTIC V1_simple->pool pathway (one gate per pool so each
    # word's training is isolated). Recognition is read DIRECTLY off the working V1_simple word-form via STDP,
    # bypassing the (sparse-text-starved) V1c->V2->IT cascade. Still cortically faithful: V1 simple cells ->
    # cortico-cortical STDP. Non-zero init (0.5+-0.3) per the readout-init lesson (STDP can't grow from exact 0).
    for nm in (word_pools or []):
        regions.append(vis_region(nm, n_per_pool, exc=0.9, dens=0.05, ew=2.0, iw=1.5, jit=0.2, plast=True))
        pathways.append(RegionPathway(from_region="cortex_v1_simple", to_region=nm, density=0.05,
                                      weight_mean=0.5, weight_jitter=0.3, plastic=True,
                                      plasticity_gate=f"v1s_to_{nm}"))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5; cfg.seed = seed
    cfg.enable_nmda = False; cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False; cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False; cfg.stdp_w_max = 10.0; cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # install SCALED Gabor retina->V1_simple weights
    pre, post, wts = build_scaled_gabor_v1_weights(retina_size, n_orientations, n_frequencies, npos)
    rm = bridge.region_manager
    r0 = rm.indices("retina")[0]; v0 = rm.indices("cortex_v1_simple")[0]
    bridge.set_pathway_weights(
        pathway_name="retina_to_v1_simple_gabor_scaled",
        pre_indices=(pre + int(r0)).astype(np.int64),
        post_indices=(post + int(v0)).astype(np.int64),
        weights=(wts * 5.0).astype(np.float32),
        add_missing=True)
    if verbose:
        print(f"[visual-text bridge] retina {retina_size}x{retina_size} ({n_retina}) -> V1s {n_v1s} -> "
              f"V1c {n_v1c} -> V2 {n_v2} -> IT {n_it}; scaled Gabor installed ({len(wts)} synapses)", flush=True)
    return bridge, retina_size, npos


def _retina_drive_gpu(word, rs, font, xp):
    img = render_word_image(word, rs, font)
    drive = VC.image_to_retina_drive(img, drive_max_pA=2500.0)
    return xp.asarray(drive, dtype=xp.float32)


def train_recognition(bridge, vocab, rs, font, n_events=80, stim_steps=60, reset_steps=30,
                      teacher_pA=2500.0, seed=42, verbose=True):
    """Teacher-supervised STDP: drive retina(word) -> V1_simple word-form fires; simultaneously drive the
    target word-pool with teacher current; STDP on the (open-gated) V1s->target-pool pathway binds the word-form
    to the pool. Interleaved (shuffled) events, one gate open at a time -> isolated per-word training."""
    import sim.backend as B
    xp, _ = B.get_backend()
    rm = bridge.region_manager
    r_idx = xp.asarray(rm.indices("retina"))
    pools = {w: xp.asarray(rm.indices(f"word_{w}")) for w in vocab}
    drives = {w: _retina_drive_gpu(w, rs, font, xp) for w in vocab}
    schedule = [w for w in vocab for _ in range(n_events)]
    np.random.default_rng(seed).shuffle(schedule)
    for i, w in enumerate(schedule):
        gate = f"v1s_to_word_{w}"
        bridge.set_plasticity_gate(gate, 1.0)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[r_idx] = drives[w]
        bridge.cp_external_input_current[pools[w]] += float(teacher_pA)
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
        bridge.set_plasticity_gate(gate, 0.0)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [train] {i+1}/{len(schedule)} events", flush=True)


def test_recognition(bridge, vocab, rs, font, stim_steps=60, reset_steps=30):
    """Drive retina(word) with NO teacher; the word-pool with the highest firing is the recognition. Earned
    visual word recognition off the V1_simple word-form -- no tokenizer, no orthogonal lang_input."""
    import sim.backend as B
    xp, _ = B.get_backend()
    rm = bridge.region_manager
    r_idx = xp.asarray(rm.indices("retina"))
    pools = {w: xp.asarray(rm.indices(f"word_{w}")) for w in vocab}
    drives = {w: _retina_drive_gpu(w, rs, font, xp) for w in vocab}
    ok = 0
    for w in vocab:
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[r_idx] = drives[w]
        counts = {ww: 0.0 for ww in vocab}
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
            fs = bridge.cp_firing_states
            for ww in vocab:
                counts[ww] += float(B.to_host(fs[pools[ww]]).sum())
        pred = max(vocab, key=lambda ww: counts[ww])
        ok += int(pred == w)
        rates = ", ".join(f"{ww}:{counts[ww]/stim_steps:.1f}" for ww in vocab)
        print(f"  {'OK ' if pred == w else 'XX '}'{w}' -> '{pred}'  [{rates}]", flush=True)
    acc = ok / len(vocab)
    print(f"  RECOGNITION {ok}/{len(vocab)} = {acc:.2f} (chance {1/len(vocab):.2f})  "
          f"-- earned visual word recognition off V1_simple word-form", flush=True)
    return acc


def read_letters_test(bridge, rs, npos, n_letters=8, n_pos=3, n_words=80,
                      stim_steps=100, reset_steps=30, seed=42, code="rate"):
    """PATH 2 cheap-first decisive test: per-position letter reading on REAL spiking V1_simple.
    Render multi-letter words, collect V1_simple features, read each letter from its V1s BAND
    (position-specific -- the structure that produced the cheap probe's 0.91, never tested in spiking).
    Per-position classifier trained on K words, tested on NOVEL words (the data-efficiency / open-vocab test).

    code="rate":    per-cell spike COUNT over the window (noisy at ~3 spikes/cell -- the failing readout).
    code="latency": per-cell FIRST-SPIKE recency (stim_steps - first_spike_step; 0 if never fired). The
                    biologically-proven spiking-recognition code (Thorpe/Masquelier/Kheradpisheh 2018):
                    strongest-responding cell fires FIRST -> robust to sparsity, preserves the Gabor-magnitude
                    structure the continuous features (0.91) used."""
    import sim.backend as B
    xp, _ = B.get_backend()
    from research.findings.raw._text_as_pixels_probe import softmax, train_logreg
    rm = bridge.region_manager
    r_idx = xp.asarray(rm.indices("retina"))
    v1 = rm.indices("cortex_v1_simple")
    v1_arr = xp.asarray(v1)
    n_v1s = len(v1)
    alphabet = list("aeotxscn")[:n_letters]
    rng = np.random.default_rng(seed)
    words = set()
    while len(words) < n_words:
        words.add("".join(rng.choice(alphabet, size=n_pos)))
    words = list(words)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(rs * 0.5))
    except Exception:
        font = ImageFont.load_default()
    print(f"[read-letters] retina {rs}, {n_words} {n_pos}-letter words over {n_letters}-letter alphabet; "
          f"collecting spiking V1_simple per-word...", flush=True)
    feats = {}
    for wi, w in enumerate(words):
        drive = xp.asarray(VC.image_to_retina_drive(render_word_image(w, rs, font), drive_max_pA=2500.0),
                           dtype=xp.float32)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[r_idx] = drive
        if code == "latency":
            first = xp.full(n_v1s, float(stim_steps), dtype=xp.float32)  # default = never fired (latest)
            for t in range(stim_steps):
                bridge._run_one_simulation_step()
                fired = bridge.cp_firing_states[v1_arr] > 0
                first = xp.where(fired & (first == float(stim_steps)), xp.float32(t), first)
            feats[w] = B.to_host(float(stim_steps) - first)  # earlier first-spike -> larger; never -> 0
        else:
            acc = xp.zeros(n_v1s, dtype=xp.float32)
            for _ in range(stim_steps):
                bridge._run_one_simulation_step()
                acc += bridge.cp_firing_states[v1_arr].astype(xp.float32)
            feats[w] = B.to_host(acc)
        if (wi + 1) % 25 == 0:
            print(f"  collected {wi+1}/{n_words}", flush=True)
    # V1_simple cell px (x-position) = (local_idx % npos^2) % npos; band p covers a third of px
    px = (np.arange(n_v1s) % (npos * npos)) % npos
    band_cells = {p: np.where((px >= p * npos // n_pos) & (px < (p + 1) * npos // n_pos))[0]
                  for p in range(n_pos)}
    def band_vec(w, p):
        v = feats[w][band_cells[p]]
        return v / (np.linalg.norm(v) + 1e-9)
    n_ho = n_words // 3
    ho, pool = words[-n_ho:], words[:-n_ho]
    L = len(alphabet); lidx = {c: i for i, c in enumerate(alphabet)}
    print(f"  per-position letter read on NOVEL words (chance per-letter {1/L:.2f}):", flush=True)
    for K in (15, 30, len(pool)):
        if K > len(pool):
            continue
        tr = pool[:K]
        Ws = [train_logreg(np.array([band_vec(w, p) for w in tr]),
                           np.array([lidx[w[p]] for w in tr]), L, seed=seed) for p in range(n_pos)]
        ok = tot = word_ok = 0
        for w in ho:
            preds = [int(softmax(band_vec(w, p)[None, :] @ Ws[p]).argmax(1)[0]) for p in range(n_pos)]
            ok += sum(int(preds[p] == lidx[w[p]]) for p in range(n_pos)); tot += n_pos
            word_ok += int(all(preds[p] == lidx[w[p]] for p in range(n_pos)))
        print(f"  K={K:>3} train words | per-letter {ok/tot:.3f} | per-word(novel) {word_ok/len(ho):.3f}",
              flush=True)
    print("  -> if per-letter >> chance on NOVEL words, spiking V1_simple per-position bands carry letter "
          "structure -> data-efficient open-vocab recognizer viable in spiking (path 2). If ~chance -> "
          "spiking sparsity kills per-position too -> path 1 (full V1->V2->IT hierarchy) required.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retina", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--recognize", action="store_true",
                    help="step-2a: STDP-train word-recognition pools off V1_simple, then test recognition")
    ap.add_argument("--read-letters", action="store_true",
                    help="path-2 cheap-first: per-position letter reading on spiking V1_simple (novel words)")
    ap.add_argument("--n-words", type=int, default=80)
    ap.add_argument("--integration-steps", type=int, default=100,
                    help="V1_simple spike-count integration window for --read-letters")
    ap.add_argument("--latency", action="store_true",
                    help="--read-letters: use first-spike latency code (Thorpe/Masquelier) instead of rate")
    ap.add_argument("--events", type=int, default=80, help="training events per word (--recognize)")
    ap.add_argument("--vocab", type=str, default="dog,cat,run,sun")
    ap.add_argument("--test-steps", type=int, default=60, help="test integration window (temporal denoise)")
    ap.add_argument("--train-steps", type=int, default=60, help="stim steps per training event")
    ap.add_argument("--teacher-pa", type=float, default=2500.0)
    a = ap.parse_args()

    if a.read_letters:
        bridge, rs, npos = build_visual_text_bridge(retina_size=a.retina, seed=a.seed)
        read_letters_test(bridge, rs, npos, n_words=a.n_words, stim_steps=a.integration_steps,
                          seed=a.seed, code=("latency" if a.latency else "rate"))
        return

    if a.recognize:
        vocab = [w.strip() for w in a.vocab.split(",") if w.strip()]
        pools = [f"word_{w}" for w in vocab]
        bridge, rs, npos = build_visual_text_bridge(retina_size=a.retina, seed=a.seed, word_pools=pools)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(rs * 0.5))
        except Exception:
            font = ImageFont.load_default()
        print(f"[recognize] vocab={vocab}; STDP-training {a.events} events/word off V1_simple word-form "
              f"(retina {rs}, train-steps {a.train_steps}, test-steps {a.test_steps})...", flush=True)
        train_recognition(bridge, vocab, rs, font, n_events=a.events, stim_steps=a.train_steps,
                          teacher_pA=a.teacher_pa, seed=a.seed)
        test_recognition(bridge, vocab, rs, font, stim_steps=a.test_steps)
        return
    bridge, rs, npos = build_visual_text_bridge(retina_size=a.retina, seed=a.seed)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(rs * 0.5))
    except Exception:
        font = ImageFont.load_default()
    import sim.backend as B
    xp, _ = B.get_backend()
    rm = bridge.region_manager
    r_idx = xp.asarray(rm.indices("retina"))
    for word in ("dog", "cat", "run"):
        img = render_word_image(word, rs, font)
        drive = VC.image_to_retina_drive(img, drive_max_pA=2500.0)
        ret = rm.indices("retina"); v1 = rm.indices("cortex_v1_simple")
        v1c = rm.indices("cortex_v1_complex"); v2 = rm.indices("cortex_v2"); it = rm.indices("cortex_it")
        acc = {"ret": 0.0, "v1": 0.0, "v1c": 0.0, "v2": 0.0, "it": 0.0}
        for t in range(a.steps):
            bridge.cp_external_input_current[r_idx] = xp.asarray(drive, dtype=xp.float32)
            bridge._run_one_simulation_step()
            fs = bridge.cp_firing_states
            acc["ret"] += float(B.to_host(fs[ret]).mean()); acc["v1"] += float(B.to_host(fs[v1]).mean())
            acc["v1c"] += float(B.to_host(fs[v1c]).mean()); acc["v2"] += float(B.to_host(fs[v2]).mean())
            acc["it"] += float(B.to_host(fs[it]).mean())
        n = a.steps
        print(f"  '{word}': mean firing -- retina {acc['ret']/n:.3f}  V1s {acc['v1']/n:.3f}  "
              f"V1c {acc['v1c']/n:.3f}  V2 {acc['v2']/n:.3f}  IT {acc['it']/n:.3f}", flush=True)
    print("  -> if V1/V2/IT fire in response to the rendered word, the GPU visual-text pipeline is live; "
          "step 2 = STDP-train an IT readout to RECOGNISE words (replacing the tokenizer).", flush=True)


if __name__ == "__main__":
    main()
