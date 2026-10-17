"""Orakel: jede Einheit gegen einen anderen Rechenweg (Differenzenquotient, ISTA aus Stück 13, exakte Lösung der kleinsten Quadrate, Posterior-Mittel aus Stück 7) und gegen die Mini-Instanz.
Das Training läuft hier nur in kleiner Konfiguration (wenige Schichten, Schritte, Signale), die Aussagen sind Schranken. Linux-CI: in der letzten Stelle nichts festlegen."""

import numpy as np
import pytest

import ent_constants as C
import ent_evaluation as ev
import ent_model as m
from inverse_generator import grid


# ----------------------------------------------------------------------------- 1. Rückwärtsrechnung gegen den Differenzenquotienten

@pytest.mark.parametrize("kind", ["scalar", "lista"])
@pytest.mark.parametrize("k_layers", [1, 3])
def test_gradients_match_finite_differences(kind, k_layers):
    rng = np.random.default_rng(0)
    xt = np.abs(rng.normal(size=(6, C.N))) * (rng.random((6, C.N)) < 0.2) + 0.01
    b = xt @ m.op().T + 0.01 * rng.normal(size=(6, C.N))
    p = m.init_params(kind, k_layers, 1e-3)
    p["th"] = np.array([2e-3, 1e-3, 5e-4][:k_layers])
    if kind == "scalar":
        p["gam"] = np.array([1.1, 0.9, 1.3][:k_layers]) * m.step_size(m.op())
    else:
        p["We"] = p["We"] + 0.01 * rng.normal(size=p["We"].shape)
        p["Sm"] = p["Sm"] + 0.01 * rng.normal(size=p["Sm"].shape)
    w = 1.0 / np.sum(xt ** 2, axis=1)
    _, g = m.loss_and_grad(p, b, xt, kind, w)
    for key in p:
        flat = p[key].reshape(-1)
        for i in rng.choice(flat.size, size=min(4, flat.size), replace=False):
            h = 1e-6 * max(abs(flat[i]), 1e-3)
            q1 = {k: v.copy() for k, v in p.items()}
            q2 = {k: v.copy() for k, v in p.items()}
            q1[key].reshape(-1)[i] += h
            q2[key].reshape(-1)[i] -= h
            fd = (m.loss_and_grad(q1, b, xt, kind, w)[0] - m.loss_and_grad(q2, b, xt, kind, w)[0]) / (2 * h)
            an = g[key].reshape(-1)[i]
            assert abs(fd - an) <= 1e-4 * max(abs(fd), abs(an), 1e-9), (kind, k_layers, key, fd, an)


# ----------------------------------------------------------------------------- 2. Start = ISTA

@pytest.mark.parametrize("kind", ["scalar", "lista"])
def test_the_untrained_network_is_ista_in_every_layer(kind):
    x = m.draw_signals("spikes", 20, 3)
    b, _ = m.make_data(x, 1e-2, 4)
    for k_layers in (1, 5, 12):
        out = m.apply(m.init_params(kind, k_layers, 0.03), b, kind)
        assert out == pytest.approx(m.ista(b, 0.03, k_layers), abs=1e-10)


def test_ista_matches_the_exact_lasso_on_a_wellposed_matrix():
    n = C.N
    a = np.zeros((n, n))
    for d, v in zip((-1, 0, 1), (0.2, 0.6, 0.2)):
        idx = np.arange(max(d, 0), min(n, n + d))
        a[idx, idx - d] = v
    x = m.draw_signals("blocks", 1, 2)
    b = x @ a.T + 0.01 * np.random.default_rng(1).normal(size=(1, n))
    assert m.ista(b, 0.01, 800, a=a)[0] == pytest.approx(m.lasso(a, b[0], 0.01), abs=1e-8)


# ----------------------------------------------------------------------------- 3. Lineares Netz = Posterior-Mittel

def test_the_least_squares_matrix_approaches_the_posterior_mean_with_more_signals():
    a = m.op()
    p = m.prior_cov(grid(C.N))
    sig = 1e-2
    xv = m.draw_signals("gauss", 300, 9)
    bv, sv = m.make_data(xv, sig, 10)
    gain = m.posterior_gain(a, float(np.sqrt(np.mean(sv ** 2))), p)
    pm = float(np.median(m.rel_err(bv @ gain.T, xv)))
    dists, errs = [], []
    for size in (100, 1000, 8000):
        x = m.draw_signals("gauss", size, 11)
        b, _ = m.make_data(x, sig, 12)
        we = m.lstsq_linear(b, x)
        dists.append(float(np.linalg.norm(we - gain) / np.linalg.norm(gain)))
        errs.append(float(np.median(m.rel_err(bv @ we.T, xv))))
    assert dists[0] > dists[1] > dists[2] and dists[2] < 0.3 and errs[0] > errs[1] and errs[2] < pm * 1.02 and errs[2] > pm * 0.98
    assert all(e >= pm - 1e-3 for e in errs)                                                           # unter das Posterior-Mittel kommt keine lineare Abbildung


def test_the_linear_network_with_zero_threshold_reproduces_the_least_squares_map():
    x = m.draw_signals("gauss", 300, 21)
    b, _ = m.make_data(x, 1e-2, 22)
    we = m.lstsq_linear(b, x)
    p = {"We": we, "Sm": np.zeros((C.N, C.N)), "th": np.zeros(1)}
    assert m.apply(p, b, "lista") == pytest.approx(b @ we.T, abs=1e-12)
    w = 1.0 / np.sum(x ** 2, axis=1)
    loss0, g = m.loss_and_grad(p, b, x, "lista", w)
    assert np.abs(g["We"]).max() < 1e-6 * max(np.abs(b).max(), 1.0)                                    # die Lösung der kleinsten Quadrate ist stationär (Gradient null)


# ----------------------------------------------------------------------------- 4. Training

def _small(kind, k_layers=3, count=200, steps=150, world="spikes", seed=5):
    x = m.draw_signals(world, count, seed)
    b, _ = m.make_data(x, 1e-2, seed + 1)
    _, lam0 = ev.best_ista(b, x, k_layers)
    p, hist = m.train(kind, k_layers, b, x, lam0, steps, seed, val=(b, x))
    return p, hist, b, x, lam0


@pytest.mark.parametrize("kind", ["scalar", "lista"])
def test_training_lowers_the_error_below_ista_and_keeps_thresholds_non_negative(kind):
    p, hist, b, x, lam0 = _small(kind)
    ista_err = float(np.median(m.rel_err(m.ista(b, lam0, 3), x)))
    assert len(hist) == C.LOG_POINTS + 1 and hist[0]["step"] == 0 and hist[-1]["step"] == 150 and hist[0]["train"] == pytest.approx(ista_err)
    assert hist[-1]["train"] < 0.9 * hist[0]["train"] and hist[-1]["val"] == pytest.approx(hist[-1]["train"]) and (p["th"] >= 0.0).all()


def test_training_is_deterministic_and_the_seed_changes_it():
    a1, _, _, _, _ = _small("scalar", steps=40)
    a2, _, _, _, _ = _small("scalar", steps=40)
    assert a1["gam"].tolist() == a2["gam"].tolist() and a1["th"].tolist() == a2["th"].tolist()
    x = m.draw_signals("spikes", 400, 5)
    b, _ = m.make_data(x, 1e-2, 6)
    _, lam0 = ev.best_ista(b, x, 3)
    q1, _ = m.train("scalar", 3, b, x, lam0, 40, 0)
    q2, _ = m.train("scalar", 3, b, x, lam0, 40, 1)
    assert q1["gam"].tolist() != q2["gam"].tolist()                                                    # andere Minibatch-Reihenfolge (mehr Signale als ein Minibatch)


def test_zero_steps_return_the_ista_start():
    x = m.draw_signals("spikes", 30, 5)
    b, _ = m.make_data(x, 1e-2, 6)
    p, hist = m.train("lista", 2, b, x, 0.05, 0, 0)
    q = m.init_params("lista", 2, 0.05)
    assert all(np.array_equal(p[k], q[k]) for k in p) and len(hist) == 1


def test_lista_with_few_signals_memorizes_them_and_the_scalar_network_does_not():
    x = m.draw_signals("spikes", 8, 31)
    b, _ = m.make_data(x, 1e-2, 32)
    xv = m.draw_signals("spikes", 100, 33)
    bv, _ = m.make_data(xv, 1e-2, 34)
    _, lam0 = ev.best_ista(b, x, 2)
    li, _ = m.train("lista", 2, b, x, lam0, 800, 0)
    sc, _ = m.train("scalar", 2, b, x, lam0, 800, 0)
    tr_l, te_l = np.median(m.rel_err(m.apply(li, b, "lista"), x)), np.median(m.rel_err(m.apply(li, bv, "lista"), xv))
    tr_s, te_s = np.median(m.rel_err(m.apply(sc, b, "scalar"), x)), np.median(m.rel_err(m.apply(sc, bv, "scalar"), xv))
    assert tr_l < 0.2 * te_l and tr_s > 0.5 * te_s


def test_a_network_trained_on_spikes_is_worse_than_tikhonov_on_blocks():
    x = m.draw_signals("spikes", 400, 41)
    b, _ = m.make_data(x, 1e-2, 42)
    _, lam0 = ev.best_ista(b, x, 3)
    p, _ = m.train("lista", 3, b, x, lam0, 300, 0)
    xb = m.draw_signals("blocks", 100, 43)
    bb, _ = m.make_data(xb, 1e-2, 44)
    assert np.median(m.rel_err(m.apply(p, bb, "lista"), xb)) > ev.best_tikhonov(bb, xb)[0]
    xs = m.draw_signals("spikes", 100, 45)
    bs, _ = m.make_data(xs, 1e-2, 46)
    assert np.median(m.rel_err(m.apply(p, bs, "lista"), xs)) < ev.best_ista(bs, xs, 3)[0]


# ----------------------------------------------------------------------------- 5. Verteilungen

def test_the_signal_distributions_have_their_documented_ranges():
    sp = m.draw_signals("spikes", 300, 1)
    assert ((sp > 0).sum(axis=1) <= 4).all() and ((sp > 0).sum(axis=1) >= 1).all() and sp.max() <= 2.5 * 4 and sp[sp > 0].min() >= 1.0
    bl = m.draw_signals("blocks", 300, 2)
    assert bl.min() >= 0.0 and bl.max() <= 4 * 1.2 + 1e-12 and ((bl > 0).sum(axis=1) >= 3).all()
    wv = m.draw_signals("waves", 300, 3)
    assert wv.min() >= 0.0 and wv.max() <= 1.0 and (wv[:, grid(C.N) <= 0.1] == 0.0).all()
    ph = m.draw_signals("phantom", 300, 4)
    assert ph.min() >= 0.0 and ph.max() < 2.4
