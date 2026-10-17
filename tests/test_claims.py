"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE; ein Training je Zelle). Ändert sich eine Zahl, muss der Text mit.
Linux-CI: Zahlen ohne Training (ISTA, FISTA, L1, Tikhonov, Posterior-Mittel, kleinste Quadrate) eng pinnen, trainierte Zahlen mit weiter Toleranz und als Ungleichungen."""

import pytest

import ent_constants as C
import ent_evaluation as ev


@pytest.fixture(scope="module")
def post():
    return ev.posterior_experiment()


@pytest.fixture(scope="module")
def layers():
    return ev.layers_experiment()


@pytest.fixture(scope="module")
def shift():
    return ev.shift_experiment()


@pytest.fixture(scope="module")
def over():
    return ev.overfit_experiment()


def cell(rows, **kw):
    return ev._by(rows, **kw)


# ----------------------------------------------------------------------------- Experiment 1

def test_the_posterior_mean_is_the_floor_and_least_squares_approaches_it(post):
    assert post["pm"] == pytest.approx(0.20583, abs=3e-4) and post["tik"] == pytest.approx(0.2066, abs=0.003) and post["l1"] == pytest.approx(0.660, abs=0.01)
    assert post["ista"] == pytest.approx(0.2441, abs=0.003) and post["fista"] == pytest.approx(0.2109, abs=0.003) and post["k"] == 5
    ls = post["ls"]
    assert [r["size"] for r in ls] == [100, 1000, 10000] and [r["dist"] for r in ls] == pytest.approx([1.747, 0.424, 0.197], abs=0.01) and [r["err"] for r in ls] == pytest.approx([0.28656, 0.20952, 0.20759], abs=5e-4)
    assert ls[0]["dist"] > ls[1]["dist"] > ls[2]["dist"] and all(r["err"] >= post["pm"] - 1e-3 for r in ls)


def test_the_networks_cannot_beat_the_posterior_mean_and_lista_turns_the_thresholds_off(post):
    lista, scal = cell(post["nets"], kind="lista"), cell(post["nets"], kind="scalar")
    assert lista["th_zero"] and lista["th"] == [0.0] * 5 and not scal["th_zero"]
    assert lista["err"] == pytest.approx(0.240, abs=0.04) and scal["err"] == pytest.approx(0.271, abs=0.05) and lista["err"] >= post["pm"] and scal["err"] >= post["pm"] and lista["err"] < post["l1"] / 2
    text = ev.verdict_posterior(post)[1]
    assert "Bei Prior-Ziehungen ist das Posterior-Mittel mit 20,6 % nicht zu unterbieten (Tikhonov 20,7 %, exaktes L1 66,0 %)" in text and "Matrixabstand 1,75 / 0,42 / 0,20" in text
    assert "Testfehler 28,7 % / 21,0 % / 20,8 % bei 100 / 1000 / 10000 Signalen" in text and "und schaltet alle Schwellen ab (linear)" in text and f"das Skalar-Netz {ev.err_pct(scal['err'])}" in text and "FISTA mit 50 Schritten 21,1 %" in text


# ----------------------------------------------------------------------------- Experiment 2

def test_layers_beat_steps_but_not_the_exact_solution(layers):
    rows = {r["k"]: r for r in layers["rows"]}
    assert [rows[k]["ista"] for k in (2, 5, 10)] == pytest.approx([0.831, 0.783, 0.735], abs=0.003) and [rows[k]["fista"] for k in (2, 5, 10)] == pytest.approx([0.831, 0.763, 0.670], abs=0.003)
    assert layers["l1"] == pytest.approx(0.0098, abs=0.003) and layers["tik"] == pytest.approx(0.5057, abs=0.003)
    assert [rows[k]["lista"] for k in (2, 5, 10)] == pytest.approx([0.669, 0.276, 0.162], abs=0.06) and [rows[k]["scalar"] for k in (2, 5, 10)] == pytest.approx([0.642, 0.528, 0.457], abs=0.07)
    assert all(rows[k]["lista"] < rows[k]["scalar"] < rows[k]["ista"] for k in (5, 10)) and rows[10]["lista"] > 5 * layers["l1"] and rows[10]["lista"] < rows[2]["lista"] / 2
    assert rows[10]["lista_equiv_fista"] in (25, 50, 100) and rows[10]["lista_equiv_ista"] in (150, 300, 500) and rows[5]["lista_equiv_ista"] >= 150 and rows[2]["lista_equiv_fista"] >= 10


def test_the_scalar_network_learns_steps_far_above_two_over_l(layers):
    lr = layers["learned"]
    assert lr["k"] == 10 and len(lr["gam"]) == len(lr["lam"]) == 10 and lr["lam0"] == pytest.approx(0.1)
    assert max(lr["gam"]) > 8.0 and min(lr["gam"]) < 1.0 and sum(lr["gam"]) > 30.0 and max(lr["lam"]) > 1.5 and min(lr["lam"]) < 0.5
    text = ev.verdict_layers(layers)[1]
    rows = layers["rows"]
    assert f"LISTA mit 2 / 5 / 10 Schichten {' / '.join(ev.err_pct(r['lista']) for r in rows)}" in text and "Das exakte L1 liegt bei 1,0 %, Tikhonov bei 50,6 %: das Netz holt die Iteration ein, nicht die Lösung." in text
    assert "ISTA mit denselben Schichten 83,1 % / 78,3 % / 73,5 %, FISTA 83,1 % / 76,3 % / 67,0 %" in text and f"(Summe {ev.de(sum(lr['gam']), 1)} gegen 10 bei ISTA)" in text


# ----------------------------------------------------------------------------- Experiment 3

def test_the_baselines_per_world_are_exact(shift):
    ista, tik = shift["base"]["ista"], shift["base"]["tik"]
    assert [ista[w] for w in C.TEST_WORLDS] == pytest.approx([0.783, 0.2545, 0.1174, 0.1293], abs=0.003) and [tik[w] for w in C.TEST_WORLDS] == pytest.approx([0.5057, 0.1791, 0.0935, 0.0991], abs=0.003)
    assert shift["k"] == 5 and len(shift["rows"]) == 5 and len(shift["noise_rows"]) == 3


def test_a_network_is_the_prior_of_its_training_distribution(shift):
    sp = cell(shift["rows"], kind="lista", train="spikes")["errs"]
    bl = cell(shift["rows"], kind="lista", train="blocks")["errs"]
    wv = cell(shift["rows"], kind="lista", train="waves")["errs"]
    al = cell(shift["rows"], kind="lista", train="all")["errs"]
    sc = cell(shift["rows"], kind="scalar", train="spikes")["errs"]
    tik, ista = shift["base"]["tik"], shift["base"]["ista"]
    assert sp["spikes"] < 0.65 * sp["blocks"] and sp["blocks"] > tik["blocks"] * 2 and sp["waves"] > 3 * tik["waves"] and sp["phantom"] > 3 * tik["phantom"] and sp["spikes"] < 0.6 * ista["spikes"]
    assert bl["blocks"] == pytest.approx(tik["blocks"], abs=0.05) and bl["spikes"] > 0.6 and wv["waves"] < 0.7 * tik["waves"] and wv["waves"] == pytest.approx(0.051, abs=0.02) and wv["spikes"] > 0.6
    assert al["spikes"] > 1.5 * sp["spikes"] and al["blocks"] < 1.4 * tik["blocks"] and al["waves"] < tik["waves"] and sc["blocks"] > 1.0 and sc["blocks"] > 4 * bl["blocks"]
    text = ev.verdict_shift(shift)[1]
    assert "LISTA mit 5 Schichten, auf Spitzen trainiert: Spitzen" in text and f"Das Skalar-Netz von den Spitzen erzeugt auf Blöcken {ev.err_pct(sc['blocks'])}" in text and "Ohne Training: ISTA 25,5 % und Tikhonov 17,9 % auf Blöcken." in text


def test_the_training_noise_hardly_matters_at_few_layers(shift):
    nz = {r["train"]: r["errs"] for r in shift["noise_rows"]}
    assert set(nz) == {1e-3, 1e-2, 1e-1} and abs(nz[1e-3][1e-2] - nz[1e-2][1e-2]) < 0.03 and abs(nz[1e-1][1e-2] - nz[1e-2][1e-2]) < 0.06
    assert all(nz[t][1e-1] > nz[t][1e-3] for t in nz) and all(e < 0.5 for r in nz.values() for e in r.values())
    assert "Test bei 1 % nach Training bei 0,1 % / 1 % / 10 %:" in ev.verdict_shift(shift)[1]


# ----------------------------------------------------------------------------- Experiment 4

def test_lista_memorizes_few_signals_and_the_scalar_network_does_not(over):
    lista = {r["size"]: r for r in over["rows"] if r["kind"] == "lista"}
    scal = {r["size"]: r for r in over["rows"] if r["kind"] == "scalar"}
    assert sorted(lista) == [20, 100, 500, 2000] and sorted(scal) == [20, 2000] and all(r["params"] == 4613 for r in lista.values()) and all(r["params"] == 10 for r in scal.values())
    assert lista[20]["train"] < 1e-6 and lista[100]["train"] < 1e-2 and lista[20]["test"] > 0.8 and lista[100]["test"] > 0.8 and lista[500]["test"] > 1.3 * lista[500]["train"]
    assert lista[2000]["test"] < 0.5 * lista[100]["test"] and lista[2000]["test"] == pytest.approx(0.276, abs=0.06) and lista[2000]["train"] < lista[2000]["test"]
    assert scal[20]["train"] > 0.4 and scal[20]["test"] < 1.25 * scal[20]["train"] and scal[2000]["test"] == pytest.approx(0.528, abs=0.07) and over["ista"] == pytest.approx(0.783, abs=0.003) and over["k"] == 5
    text = ev.verdict_overfit(over)[1]
    assert "LISTA (4613 Parameter) mit 20 / 100 / 500 / 2000 Trainingssignalen: Trainingsfehler 0 % / " in text and "Das Skalar-Netz (10 Parameter) überanpasst nicht" in text and "ISTA mit denselben Schichten: 78,3 %" in text
