"""Pure CPU tests. LOAD-BEARING: (A) the FIXED-RANDOM apical feedback
matrix is provably never mutated by any call (no weight transport);
(B) apical depolarization genuinely LOWERS the somatic effective
threshold (Larkum BAC); (C) no autograd/torch imported."""
import numpy as np
import sim.dendritic_neuron as dn


def test_no_autograd_imported():
    import inspect
    src = inspect.getsource(dn)
    assert "torch" not in src and "autograd" not in src


def test_fixed_apical_feedback_never_mutated():
    layer = dn.DendriticLayer(n_pre=4, n_post=3, n_teacher=2, seed=7)
    B0 = layer.B_apical.copy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        layer.step(x_basal=rng.normal(size=4),
                   teacher=rng.normal(size=2))
    assert np.array_equal(layer.B_apical, B0)  # FIXED random, untouched


def test_apical_depolarization_lowers_threshold():
    layer = dn.DendriticLayer(n_pre=4, n_post=3, n_teacher=2, seed=1)
    x = np.ones(4) * 0.5
    s_noap = layer.effective_threshold(teacher=np.zeros(2))
    s_ap = layer.effective_threshold(teacher=np.ones(2) * 5.0)
    assert np.all(s_ap <= s_noap)            # BAC: apical eases firing
    assert np.any(s_ap < s_noap)


def test_step_is_deterministic_given_state():
    a = dn.DendriticLayer(n_pre=3, n_post=2, n_teacher=2, seed=42)
    b = dn.DendriticLayer(n_pre=3, n_post=2, n_teacher=2, seed=42)
    x = np.array([0.2, -0.4, 0.7])
    t = np.array([1.0, 0.0])
    o1 = a.step(x_basal=x, teacher=t)
    o2 = b.step(x_basal=x, teacher=t)
    assert np.array_equal(o1["soma_rate"], o2["soma_rate"])
    assert np.array_equal(o1["v_basal"], o2["v_basal"])
