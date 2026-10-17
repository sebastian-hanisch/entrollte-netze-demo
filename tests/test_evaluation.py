"""Formatierung, Trainings- und Testdaten, Analyse einer Einstellung (kleine Konfiguration) und die Verzweigungen des Urteils (jede Schwelle auf der Kante)."""

import dataclasses
import math

import numpy as np
import pytest

import ent_constants as C
import ent_evaluation as ev
import ent_model as m


def S(**kw) -> ev.Settings:
    return ev.Settings(**kw)


SMALL = dict(kind="lista", k=2, n_train=100, steps=100)


@pytest.fixture(scope="module")
def small():
    return ev.analyse(S(**SMALL))


@pytest.fixture(scope="module")
def small_scalar():
    return ev.analyse(S(**dict(SMALL, kind="scalar")))


# ----------------------------------------------------------------------------- Formatierung

def test_de_uses_comma_and_typographic_minus():
    assert ev.de(-1.5, 1) == "−1,5" and ev.de(0.04, 3) == "0,040"


def test_sci_formats_powers_of_ten_and_rounds_up_the_mantissa():
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(0.00039) == "3,9·10⁻⁴" and ev.sci(9.96e-3) == "1,0·10⁻²" and ev.sci(0.0) == "0,0" and ev.sci(math.inf) == "unendlich"


def test_err_pct_has_four_ranges_a_factor_form_and_zero():
    assert ev.err_pct(0.5) == "50,0 %" and ev.err_pct(1.19) == "119 %" and ev.err_pct(0.0123) == "1,2 %" and ev.err_pct(0.00123) == "0,12 %" and ev.err_pct(1e-5) == "1,0·10⁻³ %"
    assert ev.err_pct(12.0) == "1,2·10¹-fach" and ev.err_pct(math.inf) == "unendlich" and ev.err_pct(0.9951) == "100 %" and ev.err_pct(8.5e-17) == "0 %" and ev.err_pct(1e-10) == "1,0·10⁻⁸ %" and ev.err_pct(0.0) == "0 %"


def test_noise_label():
    assert ev.noise_label(0.01) == "1 %" and ev.noise_label(0.001) == "0,1 %" and ev.noise_label(0.1) == "10 %" and ev.noise_label(0.0) == "0 %"


def test_by_finds_the_row_with_all_keys():
    rows = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert ev._by(rows, a=1, b=3) is rows[1] and ev._by(rows, a=2) is rows[2]


def test_default_settings_equal_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


# ----------------------------------------------------------------------------- Daten und Vergleichsverfahren

def test_train_and_test_data_use_different_streams():
    xt, bt = ev.train_data("spikes", 20, 1e-2, 7)
    xv, bv = ev.test_data("spikes", 20, 1e-2, 7)
    assert xt.tolist() == m.draw_signals("spikes", 20, 8).tolist() and xv.tolist() == m.draw_signals("spikes", 20, 10).tolist() and xt.tolist() != xv.tolist()
    assert bt.tolist() == m.make_data(xt, 1e-2, 9)[0].tolist() and bv.tolist() == m.make_data(xv, 1e-2, 11)[0].tolist()


def test_best_lambda_picks_the_grid_value_with_the_smallest_median_error(monkeypatch):
    monkeypatch.setattr(C, "LAMBDA_GRID", (1e-4, 1e-1, 10.0))
    xv, bv = ev.test_data("spikes", 20, 1e-2, 3)
    errs = [float(np.median(m.rel_err(m.ista(bv, lam, 5), xv))) for lam in C.LAMBDA_GRID]
    e, lam = ev.best_ista(bv, xv, 5)
    assert e == pytest.approx(min(errs)) and lam == C.LAMBDA_GRID[int(np.argmin(errs))]
    tik = [float(np.median(m.rel_err(s, xv))) for s in m.tikhonov_svd(m.op(), bv, C.LAMBDA_GRID)]
    assert ev.best_tikhonov(bv, xv) == (pytest.approx(min(tik)), C.LAMBDA_GRID[int(np.argmin(tik))])
    e1, lam1, x1 = ev.best_l1(bv[:5], xv[:5])
    l1 = [float(np.median(m.rel_err(np.array([m.lasso(m.op(), bb, lam) for bb in bv[:5]]), xv[:5]))) for lam in C.LAMBDA_GRID]
    assert e1 == pytest.approx(min(l1)) and lam1 == C.LAMBDA_GRID[int(np.argmin(l1))] and x1 == pytest.approx(m.lasso(m.op(), bv[0], lam1))


# ----------------------------------------------------------------------------- Analyse

def test_analysis_fields_of_a_small_lista(small):
    a, st = small, S(**SMALL)
    xv, bv = ev.test_data("spikes", C.TEST_COUNT, 1e-2, st.seed)
    assert a.settings == st and a.s == pytest.approx((np.arange(48) + 0.5) / 48) and a.t == pytest.approx(m.step_size(m.op())) and len(a.hist) == C.LOG_POINTS + 1
    assert a.x.tolist() == xv[0].tolist() and a.b.tolist() == bv[0].tolist() and a.x_net == pytest.approx(m.apply(a.params, bv, "lista")[0]) and a.err_net == pytest.approx(float(np.median(m.rel_err(m.apply(a.params, bv, "lista"), xv))))
    xt, bt = ev.train_data("spikes", 100, 1e-2, st.seed)
    assert a.err_train == pytest.approx(float(np.median(m.rel_err(m.apply(a.params, bt, "lista"), xt)))) and a.lam0 == ev.best_ista(bt, xt, 2)[1]
    assert a.err_ista == pytest.approx(ev.best_ista(bv, xv, 2)[0]) and a.err_fista == pytest.approx(ev.best_ista(bv, xv, 2, True)[0]) and a.lam_ista in C.LAMBDA_GRID
    assert a.err_tik == pytest.approx(ev.best_tikhonov(bv, xv)[0]) and a.err_l1 == pytest.approx(ev.best_l1(bv[:C.BASE_COUNT], xv[:C.BASE_COUNT])[0])
    assert a.x_ista == pytest.approx(m.ista(bv[:1], a.lam_ista, 2)[0]) and a.x_fista == pytest.approx(m.ista(bv[:1], a.lam_fista, 2, True)[0]) and math.isnan(a.gam_max) and a.hist[-1]["val"] == pytest.approx(a.err_net)
    assert a.train_noise == 1e-2 and isinstance(a.th_zero, bool) and a.err_net < a.hist[0]["val"]


def test_scalar_analysis_reports_the_largest_step(small_scalar):
    a = small_scalar
    assert a.gam_max == pytest.approx(float(a.params["gam"].max() / a.t)) and set(a.params) == {"gam", "th"}


def test_test_world_decides_the_test_signals():
    a = ev.analyse(S(kind="scalar", k=2, n_train=100, steps=40, test_world="blocks"))
    xv, _ = ev.test_data("blocks", C.TEST_COUNT, 1e-2, C.DEFAULT_SEED)
    assert a.x.tolist() == xv[0].tolist()


# ----------------------------------------------------------------------------- Verdikt: jede Verzweigung

def _v(small, **kw):
    base = dataclasses.replace(small, err_net=0.5, err_train=0.5, err_ista=1.0, err_fista=1.0, err_l1=0.9, err_tik=0.8, gam_max=math.nan, th_zero=False,
                               settings=dataclasses.replace(small.settings, train_world="spikes", test_world="spikes"))
    return ev.verdict_instance(dataclasses.replace(base, **kw))


def test_verdict_worth_it_with_the_documented_text(small):
    kind, text = _v(small, err_net=0.2)
    assert kind == "success" and text.startswith("LISTA, 2 Schichten, Trainingswelt Spitzen (100 Signale, 100 Schritte), Testwelt Spitzen, Rauschen 1 %: Testfehler 20,0 % (Trainingsfehler 50,0 %; ISTA 100 %, FISTA 100 % mit 2 Schritten; ")
    assert "exaktes L1 90,0 %, Tikhonov 80,0 %)" in text and "Das Netz lohnt" in text and "Überanpassung" not in text and "Verteilungswechsel" not in text


def test_gain_threshold_is_inclusive(monkeypatch, small):
    monkeypatch.setattr(C, "GAIN", 0.5)
    assert _v(small, err_net=0.5)[0] == "success" and _v(small, err_net=0.5000001)[0] == "info"


def test_tie_and_loss_against_ista_and_fista(small):
    assert _v(small, err_net=1.0)[0] == "info" and _v(small, err_net=1.0000001)[0] == "warning"
    assert _v(small, err_net=0.95, err_ista=1.0, err_fista=0.9)[0] == "warning"                           # das bessere von beiden zählt
    assert "lohnt nicht" in _v(small, err_net=1.5)[1] and "kaum besser" in _v(small, err_net=0.95)[1]


def test_exact_l1_remark_is_strict(small):
    assert "Das exakte L1 ist besser (10,0 %): das Netz holt die Iteration ein, nicht die Lösung" in _v(small, err_net=0.5, err_l1=0.1)[1]
    assert "Das exakte L1 ist besser" not in _v(small, err_net=0.5, err_l1=0.5)[1]


def test_overfit_remark_is_inclusive(monkeypatch, small):
    monkeypatch.setattr(C, "OVERFIT", 0.5)
    assert "Überanpassung: der Trainingsfehler (25,0 %) ist höchstens halb so groß wie der Testfehler" in _v(small, err_net=0.5, err_train=0.25)[1]
    assert "Überanpassung" not in _v(small, err_net=0.5, err_train=0.2500001)[1]


def test_distribution_shift_remark(small):
    st = dataclasses.replace(small.settings, train_world="spikes", test_world="blocks")
    assert "Verteilungswechsel: Trainingswelt Spitzen, Testwelt Blöcke." in _v(small, settings=st)[1]
    assert "Verteilungswechsel" not in _v(small)[1]


def test_zero_threshold_remark(small):
    assert "Alle Schwellen sind abgeschaltet: das Netz ist linear und schätzt das Posterior-Mittel der Trainingsverteilung." in _v(small, th_zero=True)[1]
    assert "abgeschaltet" not in _v(small)[1]


def test_step_over_remark_is_strict(monkeypatch, small):
    monkeypatch.setattr(C, "STEP_OVER", 3.0)
    assert "keine konvergente Iteration" not in _v(small, gam_max=3.0)[1] and "das 3,1-Fache von 1/‖A‖² (über dem Doppelten)" in _v(small, gam_max=3.1)[1]
    assert "keine konvergente Iteration" not in _v(small, gam_max=math.nan)[1]
