"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py, klickt jeden Preset-Button, trainiert kleine Netze über den Knopf, rechnet jedes Experiment einmal
und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=). Training nur in kleiner Konfiguration (zwei Schichten, 20 Signale)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 900
SMALL = {"kind_select": "lista", "k_select": 2, "n_train_select": 20, "steps_select": 300}


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _small_app(**extra) -> AppTest:
    at = _fresh_app()
    for key, value in {**SMALL, **extra}.items():
        (at.select_slider(key=key) if key in ("k_select", "n_train_select", "steps_select", "noise_select") else at.selectbox(key=key)).set_value(value)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _train(at: AppTest) -> AppTest:
    at.button(key="train_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception_and_asks_for_training():
    at = _fresh_app()
    assert any("noch kein Netz trainiert" in i.value for i in at.info)
    assert sum("🔬" in s.value for s in at.subheader) == 4


def test_presets_and_dice_buttons_do_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels
    for index, label in enumerate(labels):
        if "trainieren" in label or "Experiment" in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_does_not_raise():
    import ent_constants as C

    for key, options in (("noise_select", C.NOISE_CHOICES), ("k_select", C.K_CHOICES), ("n_train_select", C.N_TRAIN_CHOICES), ("steps_select", C.STEPS_CHOICES)):
        for value in options:
            at = _fresh_app()
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    import ent_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.selectbox(key="kind_select").value == preset["kind"] and at.select_slider(key="k_select").value == preset["k"]
        assert at.selectbox(key="train_world_select").value == preset["train_world"] and at.selectbox(key="test_world_select").value == preset["test_world"]
        assert at.select_slider(key="noise_select").value == preset["noise"] and at.select_slider(key="n_train_select").value == preset["n_train"]
        assert at.select_slider(key="steps_select").value == preset["steps"] and at.number_input(key="seed_input").value == preset["seed"]


def test_training_a_small_network_shows_metrics_tabs_and_a_verdict():
    at = _train(_small_app())
    assert len(at.metric) == 4 and any(v for v in [*at.success, *at.warning, *at.info] if "Testfehler" in v.value)
    assert not any("noch kein Netz trainiert" in i.value for i in at.info)


def test_every_network_kind_and_test_world_does_not_raise():
    import ent_constants as C

    for kind in C.KINDS:
        for world in C.TEST_WORLDS:
            at = _train(_small_app(kind_select=kind, test_world_select=world, steps_select=300))
            assert len(at.metric) == 4, (kind, world)


def test_every_training_world_does_not_raise():
    import ent_constants as C

    for world in C.TRAIN_WORLDS:
        _train(_small_app(train_world_select=world, kind_select="scalar"))


def test_the_standard_network_is_worth_it():
    at = _train(_fresh_app())
    assert any("Das Netz lohnt" in s.value for s in at.success)


def test_the_distribution_shift_preset_warns():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    at = _fresh_app()
    at.button[labels["Verteilungswechsel"]].click().run(timeout=TIMEOUT)
    _train(at)
    assert any("Das Netz lohnt nicht" in w.value and "Verteilungswechsel" in w.value for w in at.warning)


def test_changing_a_setting_after_training_asks_again():
    at = _train(_small_app())
    at.select_slider(key="steps_select").set_value(1000).run(timeout=TIMEOUT)
    assert any("noch kein Netz trainiert" in i.value for i in at.info) and len(at.metric) == 0


def test_pdf_is_created_only_after_the_click():
    at = _train(_small_app())
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_a_scalar_network_is_created():
    at = _train(_small_app(kind_select="scalar"))
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_each_experiment_runs_once_and_shows_its_verdict():
    at = _fresh_app()
    for key, marker in (("exp1_run", "Posterior-Mittel"), ("exp2_run", "In der Spitzen-Welt"), ("exp3_run", "LISTA mit 5 Schichten"), ("exp4_run", "Überanpassung")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
    texts = [e.value for e in [*at.info, *at.warning]]
    assert all(any(marker in t for t in texts) for marker in ("Posterior-Mittel", "In der Spitzen-Welt", "LISTA mit 5 Schichten", "LISTA (4613 Parameter)"))
    assert not any("Noch nicht gerechnet" in i.value for i in at.info)


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["net"] = "cnn"
    at.query_params["k"] = "7"
    at.query_params["train"] = "kreis"
    at.query_params["test"] = "all"
    at.query_params["r"] = "0.5"
    at.query_params["n"] = "7"
    at.query_params["steps"] = "5"
    at.query_params["seed"] = "99999999"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="kind_select").value == "lista" and at.select_slider(key="k_select").value == 10
    assert at.selectbox(key="train_world_select").value == "spikes" and at.selectbox(key="test_world_select").value == "spikes"
    assert at.select_slider(key="noise_select").value == 1e-2 and at.select_slider(key="n_train_select").value == 2000
    assert at.select_slider(key="steps_select").value == 1000 and at.number_input(key="seed_input").value == 999999


def test_valid_permalink_values_are_taken_over():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["net"] = "scalar"
    at.query_params["k"] = "5"
    at.query_params["train"] = "all"
    at.query_params["test"] = "waves"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="kind_select").value == "scalar" and at.select_slider(key="k_select").value == 5
    assert at.selectbox(key="train_world_select").value == "all" and at.selectbox(key="test_world_select").value == "waves"


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="kind_select").set_value("scalar").run(timeout=TIMEOUT)
    assert at.query_params["net"] == ["scalar"] or at.query_params["net"] == "scalar"
