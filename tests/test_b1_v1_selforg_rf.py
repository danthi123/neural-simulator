"""CPU smoke tests for the B1 self-organizing V1 RF de-risk
(research/runners/_b1_v1_selforg_rf_derisk.py).

Pins the load-bearing claims at a tiny scale (fast, CPU/numpy):
  - a self-organized RF bank (learned mechanism A, dev-random mechanism B)
    PRESERVES the host Gabor bank's pixel-similarity geometry (high RSA-to-host).
  - the LEARNED + DEV-RANDOM banks are ORIENTED (OSI), while the NO-LEARNING +
    NOISE-INPUT controls are NOT (the discriminating control collapses).
A drift here would silently break the discharge claim.
"""
import numpy as np
import pytest

from research.runners import _b1_v1_selforg_rf_derisk as m


def test_learned_bank_is_oriented_controls_are_not():
    """Mechanism A (learned from oriented patches) + B (dev-random) produce
    ORIENTED filters; the no-learning random bank + a bank trained on white
    noise do NOT (OSI collapses). The L.05 content-matters discriminator."""
    oriented = m.make_patch_stream(1500, np.random.default_rng(43), kind="oriented")
    noise = m.make_patch_stream(1500, np.random.default_rng(44), kind="noise")

    W_A = m.learn_rf_bank_sailnet(oriented, seed=42, n_epochs=60)
    W_B = m.devrandom_rf_bank(seed=42)
    W_rand = m.random_rf_bank(seed=42)
    W_noise = m.learn_rf_bank_sailnet(noise, seed=42, n_epochs=60)

    _, A_frac = m.gabor_orientation_tuning(W_A)
    _, B_frac = m.gabor_orientation_tuning(W_B)
    _, rand_frac = m.gabor_orientation_tuning(W_rand)
    _, noise_frac = m.gabor_orientation_tuning(W_noise)

    # self-org banks oriented; controls collapse
    assert A_frac >= 0.5, f"learned bank not oriented (OSI frac {A_frac})"
    assert B_frac >= 0.5, f"dev-random bank not oriented (OSI frac {B_frac})"
    assert rand_frac <= 0.2, f"no-learning control oriented?! (OSI frac {rand_frac})"
    assert noise_frac <= 0.2, f"noise-input control oriented?! (OSI frac {noise_frac})"


def test_selforg_bank_preserves_host_geometry():
    """A self-org bank's codes carry the same pixel-similarity geometry as the
    host Gabor bank's codes (RSA-to-host high) on the Option-B shape set."""
    rng = np.random.default_rng(42)
    images, labels, _ = m.build_shape_set(4, 4, rng)
    Whost = m.build_host_v1_matrix()
    host_code = m.encode_host_v1(images, Whost)

    oriented = m.make_patch_stream(1500, np.random.default_rng(43), kind="oriented")
    W_A = m.learn_rf_bank_sailnet(oriented, seed=42, n_epochs=60)
    code_A = m.encode_with_bank(images, W_A)
    code_B = m.encode_with_bank(images, m.devrandom_rf_bank(seed=42))

    rsa_A = m.rsa_between_codes(code_A, host_code)
    rsa_B = m.rsa_between_codes(code_B, host_code)
    assert rsa_A >= 0.7, f"learned bank does not preserve host geometry (RSA {rsa_A})"
    assert rsa_B >= 0.7, f"dev-random bank does not preserve host geometry (RSA {rsa_B})"

    # and the within>between margin is reproduced (positive, like the host)
    _, _, A_margin = m.within_between_margin(code_A, labels)
    assert A_margin > 0.1, f"learned bank margin too low ({A_margin})"


def test_run_seed_verdict_go():
    """End-to-end per-seed run returns GO at the tiny smoke scale (both the
    discharge bar and the discriminating control pass)."""
    r = m.run_seed(42, n_categories=4, n_exemplars=4, n_patches=1500,
                   n_epochs=50, n_orient=8, n_orient_ex=6)
    assert r["A_geom"] and r["B_geom"], "geometry not preserved"
    assert r["controls_unoriented"], "controls did not collapse on OSI"
    assert r["verdict"] == "GO", f"verdict {r['verdict']} != GO"
