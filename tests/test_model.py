"""Einheiten von ent_model.py, jede an der Mini-Instanz oder an einer kleinen Aufgabe von Hand gerechnet (DEMO-PLAYBOOK §1b). Orakel gegen andere Rechenwege: test_oracle_ent.py."""

import math

import numpy as np
import pytest

import ent_constants as C
import ent_model as m
from inverse_generator import SplitMix64, grid


# ----------------------------------------------------------------------------- Grundbausteine

def test_soft_hand_cases():
    assert m.soft(np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])
    assert m.soft(np.array([3.0, -2.0]), 0.0).tolist() == [3.0, -2.0]


def test_step_size_is_one_over_the_squared_spectral_norm(mini):
    a, b, x = mini
    assert m.step_size(a) == pytest.approx(1.0, abs=1e-12) and m.step_size(2.0 * a) == pytest.approx(0.25, abs=1e-12)


def test_op_is_cached_and_read_only():
    a = m.op()
    assert a is m.op(48, 0.03 + 1e-9) and not a.flags.writeable and a.shape == (48, 48)


def test_prior_cov_and_exp_kernel_hand_values():
    k = m.exp_kernel(np.array([0.0, 0.12, 0.24]), 0.12)
    assert k[0, 1] == pytest.approx(math.exp(-1.0)) and k[0, 2] == pytest.approx(math.exp(-2.0)) and np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T)
    p = m.prior_cov(np.array([0.0, 0.12]))
    assert p[0, 0] - 0.25 == pytest.approx(1e-10, rel=1e-3) and p[0, 1] == pytest.approx(0.25 * math.exp(-1.0)) and p[1, 0] == p[0, 1]


# ----------------------------------------------------------------------------- Netze: Start, Vorwärts, Rückwärts

def test_init_params_are_ista(mini):
    a, b, x = mini
    s = m.init_params("scalar", 3, 0.2, a)
    assert s["gam"].tolist() == [1.0] * 3 and s["th"] == pytest.approx([0.2] * 3)
    li = m.init_params("lista", 2, 0.2, a)
    assert li["We"] == pytest.approx(a.T) and li["Sm"] == pytest.approx(np.eye(3) - a.T @ a) and li["th"] == pytest.approx([0.2, 0.2])
    assert m.init_params("scalar", 1, 0.1)["gam"][0] == pytest.approx(m.step_size(m.op()))


def test_forward_hand_values(mini):
    a, b, x = mini
    p = {"gam": np.array([1.0]), "th": np.array([0.1])}
    assert m.apply(p, b, "scalar", a)[0] == pytest.approx([1.23, 1.24, 1.23])
    q = {"We": a.T.copy(), "Sm": np.zeros((3, 3)), "th": np.array([0.1])}
    assert m.apply(q, b, "lista", a)[0] == pytest.approx([1.23, 1.24, 1.23])
    big = {"gam": np.array([1.0]), "th": np.array([5.0])}
    assert m.apply(big, b, "scalar", a)[0].tolist() == [0.0, 0.0, 0.0]


def test_untrained_network_is_ista_after_two_layers(mini):
    a, b, x = mini
    for kind in ("scalar", "lista"):
        assert m.apply(m.init_params(kind, 2, 0.1, a), b, kind, a)[0] == pytest.approx([1.2267, 1.2466, 1.2267], abs=1e-4)
        assert m.apply(m.init_params(kind, 2, 0.1, a), b, kind, a)[0] == pytest.approx(m.ista(b, 0.1, 2, a=a)[0], abs=1e-12)


def test_loss_and_gradients_hand_values_scalar(mini):
    a, b, x = mini
    w = 1.0 / np.sum(x ** 2, axis=1)
    loss, g = m.loss_and_grad({"gam": np.array([1.0]), "th": np.array([0.1])}, b, x, "scalar", w, a)
    assert loss == pytest.approx(0.1139, abs=1e-9) and g["th"][0] == pytest.approx(0.1, abs=1e-9) and g["gam"][0] == pytest.approx(-0.135533, abs=1e-6)


def test_loss_and_gradients_hand_values_lista(mini):
    a, b, x = mini
    w = 1.0 / np.sum(x ** 2, axis=1)
    loss, g = m.loss_and_grad({"We": a.T.copy(), "Sm": np.eye(3), "th": np.array([0.1])}, b, x, "lista", w, a)
    assert loss == pytest.approx(0.1139, abs=1e-9) and g["th"][0] == pytest.approx(0.1, abs=1e-9) and g["We"][0, 0] == pytest.approx(0.099667, abs=1e-6)
    assert g["We"][1, 0] == pytest.approx(-0.253333 * 1.3, abs=1e-6) and np.abs(g["Sm"]).max() == 0.0                    # der Start ist 0: S wirkt in der ersten Schicht nicht


def test_gradients_vanish_when_every_input_is_below_the_threshold(mini):
    a, b, x = mini
    w = 1.0 / np.sum(x ** 2, axis=1)
    loss, g = m.loss_and_grad({"gam": np.array([1.0]), "th": np.array([5.0])}, b, x, "scalar", w, a)
    assert loss == pytest.approx(1.0) and g["th"][0] == 0.0 and g["gam"][0] == 0.0


def test_the_loss_is_symmetric_under_mirroring_the_signal(mini):
    a, b, x = mini
    w = np.array([1.0])
    p = {"gam": np.array([1.0]), "th": np.array([0.1])}
    l1, g1 = m.loss_and_grad(p, b, x, "scalar", w, a)
    l2, g2 = m.loss_and_grad(p, -b, -x, "scalar", w, a)
    assert l1 == pytest.approx(l2) and g1["th"][0] == pytest.approx(g2["th"][0]) == pytest.approx(0.6, abs=1e-9) and g1["gam"][0] == pytest.approx(g2["gam"][0])


# ----------------------------------------------------------------------------- Zufallsströme

def test_batch_order_is_the_rank_order_of_the_stream():
    o = m.batch_order(20, 5)
    assert sorted(o.tolist()) == list(range(20)) and o.tolist() == np.argsort(SplitMix64(5).uniform(20)).tolist()
    assert o.tolist() != m.batch_order(20, 6).tolist() and o.tolist() == m.batch_order(20, 5).tolist()


def test_spike_distribution_hand_values():
    r = np.zeros(C.DRAW_WIDTH)
    r[1], r[2], r[3], r[4] = 0.25, 0.0, 0.5, 1.0                                                      # zwei Spitzen: Zelle 12 mit Höhe 1, Zelle 24 mit Höhe 2,5
    x = m._spikes(r, 48)
    assert np.flatnonzero(x).tolist() == [12, 24] and x[12] == 1.0 and x[24] == 2.5
    r[0] = 0.99                                                                                       # vier Spitzen: die weiteren liegen bei Zelle 0 mit Höhe 1
    y = m._spikes(r, 48)
    assert y[0] == 2.0 and y[12] == 1.0 and y[24] == 2.5 and np.flatnonzero(y).tolist() == [0, 12, 24]


def test_block_distribution_hand_values():
    r = np.zeros(C.DRAW_WIDTH)
    r[2] = 0.5
    x = m._blocks(r, 48)
    assert x[22:25].tolist() == [0.3, 0.3, 0.3] and x[0:3].tolist() == [0.3, 0.3, 0.3] and x.sum() == pytest.approx(1.8)


def test_wave_distribution_hand_values():
    x = m._waves(np.zeros(C.DRAW_WIDTH), 48)
    assert x[11] == pytest.approx(0.4375) and x[4] == 0.0 and x[43] == 0.0 and x.max() <= 1.0
    assert m._waves(np.array([0.99] + [0.0] * 15), 48)[11] == pytest.approx(abs((5 * (11.5 / 48)) % 1.0 - 0.5) * 2.0)


def test_phantom_distribution_hand_value():
    x = m._phantom(np.zeros(C.DRAW_WIDTH), 48)
    s = 9.5 / 48                                                                                      # zwei gleiche Hügel bei 0,2 (Höhe 0,5, Breite 0,03) und ein Kasten ab 0,1 (Höhe 0,3, Breite 0,12)
    assert x[9] == pytest.approx(2 * 0.5 * math.exp(-((s - 0.2) ** 2) / (2 * 0.03 ** 2)) + 0.3) and x[40] == pytest.approx(0.0, abs=1e-12) and x[0] == pytest.approx(0.0, abs=1e-6)


def test_draw_signals_shapes_determinism_and_mixture():
    for world in ("spikes", "blocks", "waves", "phantom", "gauss", "all"):
        s = m.draw_signals(world, 12, 5)
        assert s.shape == (12, 48) and s.tolist() == m.draw_signals(world, 12, 5).tolist() and s.tolist() != m.draw_signals(world, 12, 6).tolist()
    mix = m.draw_signals("all", 10, 5)
    assert mix.shape == (10, 48) and mix[:3].tolist() == m.draw_signals("spikes", 3, 6).tolist() and mix[3:6].tolist() == m.draw_signals("blocks", 3, 7).tolist()
    assert mix[6:8].tolist() == m.draw_signals("waves", 2, 8).tolist() and mix[8:].tolist() == m.draw_signals("phantom", 2, 9).tolist()


def test_gauss_draws_are_cholesky_times_normals():
    s = m.draw_signals("gauss", 4, 3)
    expect = SplitMix64(3).normal(4 * 48).reshape(4, 48) @ np.linalg.cholesky(m.prior_cov(grid(48))).T
    assert s == pytest.approx(expect)


def test_make_data_noise_scale_and_stream():
    x = m.draw_signals("spikes", 5, 1)
    b, sig = m.make_data(x, 0.01, 2)
    clean = x @ m.op().T
    assert sig == pytest.approx(0.01 * np.sqrt(np.mean(clean ** 2, axis=1))) and (b - clean) / sig[:, None] == pytest.approx(SplitMix64(2).normal(5 * 48).reshape(5, 48))
    assert m.make_data(x, 0.1, 2)[1] == pytest.approx(10.0 * sig)


def test_rel_err_per_row():
    assert m.rel_err(np.array([[3.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 4.0], [0.0, 2.0]])) == pytest.approx([1.25, 1.0])


# ----------------------------------------------------------------------------- Vergleichsverfahren

def test_ista_hand_values_and_momentum(mini):
    a, b, x = mini
    assert m.ista(b, 0.1, 1, a=a)[0] == pytest.approx([1.23, 1.24, 1.23]) and m.ista(b, 0.1, 2, a=a)[0] == pytest.approx([1.2267, 1.2466, 1.2267], abs=1e-4)
    assert m.ista(b, 0.1, 0, a=a)[0].tolist() == [0.0, 0.0, 0.0]
    assert m.ista(b, 0.1, 2, True, a=a)[0] == pytest.approx(m.ista(b, 0.1, 2, a=a)[0]) and not np.allclose(m.ista(b, 0.1, 3, True, a=a), m.ista(b, 0.1, 3, a=a))


def test_tikhonov_svd_and_posterior_gain_hand_values(mini):
    a, b, x = mini
    assert m.tikhonov_svd(a, b, [0.1])[0, 0] == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    assert (m.posterior_gain(a, 0.1, np.eye(3)) @ b[0]) == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)


def test_lasso_hand_cases():
    assert m.lasso(np.eye(3), np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])
    assert m.lasso(2.0 * np.eye(2), np.array([3.0, -3.0]), 1.0) == pytest.approx([1.25, -1.25])
    assert m.lasso(np.eye(2), np.array([1.0, 0.0]), 1.0) == pytest.approx([0.0, 0.0])


def test_lstsq_linear_inverts_exact_data(mini):
    a, b, x = mini
    xs = np.random.default_rng(0).normal(size=(6, 3))
    bs = xs @ a.T
    we = m.lstsq_linear(bs, xs)
    assert bs @ we.T == pytest.approx(xs, abs=1e-6) and we == pytest.approx(np.linalg.inv(a), abs=1e-6)


def test_lstsq_linear_weights_by_the_inverse_squared_norm():
    b = np.array([[1.0], [1.0]])
    xt = np.array([[1.0], [4.0]])
    assert m.lstsq_linear(b, xt, ridge=0.0)[0, 0] == pytest.approx((1.0 / 1.0 + 4.0 / 16.0) / (1.0 / 1.0 + 1.0 / 16.0))      # min Σ (w - x_i)²/x_i²: w = Σ(1/x_i) / Σ(1/x_i²)


def test_wave_phase_shifts_the_triangle():
    r = np.zeros(C.DRAW_WIDTH)
    r[1] = 0.25                                                                                       # Phase 0,25: |(3 s + 0,25) mod 1 - 0,5| · 2 bei s = 11,5 / 48
    assert m._waves(r, 48)[11] == pytest.approx(abs((3 * (11.5 / 48) + 0.25) % 1.0 - 0.5) * 2.0) and m._waves(r, 48)[11] == pytest.approx(0.9375)


def test_threshold_mask_is_strict():
    a = np.eye(3)
    b = np.array([[1.0, 2.0, 3.0]])
    p = {"gam": np.array([1.0]), "th": np.array([2.0])}                                             # z = b; die zweite Komponente liegt genau auf der Schwelle
    assert m.apply(p, b, "scalar", a)[0].tolist() == [0.0, 0.0, 1.0]
    _, g = m.loss_and_grad(p, b, b, "scalar", np.array([1.0]), a)
    assert g["th"][0] == pytest.approx(4.0)                                                           # nur die dritte Komponente zählt: dz = 2 (1 - 3) = -4, dtheta = -sign(z) dz


def test_lstsq_linear_ridge_enters_the_denominator():
    b = np.array([[1.0], [1.0]])
    xt = np.array([[1.0], [4.0]])
    assert m.lstsq_linear(b, xt, ridge=1.0)[0, 0] == pytest.approx(1.25 / 2.0625)                    # Σ w b x / (Σ w b² + ridge) mit w = 1, 1/16


def test_minibatch_epochs_reshuffle_only_when_the_data_run_out(monkeypatch):
    seen = []
    real = m.loss_and_grad

    def spy(p, b, xt, kind, weights, a=None):
        seen.append(b[:, 0].astype(int).tolist())
        return real(p, b, xt, kind, weights, a)

    monkeypatch.setattr(m, "loss_and_grad", spy)
    b = np.tile(np.arange(512, dtype=float)[:, None], (1, 48))
    x = np.ones((512, 48))
    m.train("scalar", 1, b, x, 0.01, 3, 7)
    o0, o1 = m.batch_order(512, 7), m.batch_order(512, 8)
    assert seen[0] == o0[:256].tolist() and seen[1] == o0[256:].tolist() and seen[2] == o1[:256].tolist()
