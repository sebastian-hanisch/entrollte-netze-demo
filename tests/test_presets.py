"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen und die dokumentierten Ausgänge (trainierte Netze: Schranken und weite Toleranzen)."""

import dataclasses

import pytest

import ent_constants as C
import ent_evaluation as ev
import ent_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_on_the_choice_lists():
    for name, p in C.PRESETS.items():
        assert p["kind"] in C.KINDS and p["k"] in C.K_CHOICES and p["train_world"] in C.TRAIN_WORLDS and p["test_world"] in C.TEST_WORLDS and p["noise"] in C.NOISE_CHOICES, name
        assert p["n_train"] in C.N_TRAIN_CHOICES and p["steps"] in C.STEPS_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_settings_fields_follow_the_preset_keys():
    assert [f.name for f in dataclasses.fields(ev.Settings)] == list(P.PRESET_KEYS)


@pytest.fixture(scope="module")
def outcomes():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_the_presets_show_the_documented_outcomes(outcomes):
    std, scal, shift, few, waves, allw, two = (outcomes[k] for k in C.PRESETS)
    assert ev.verdict_instance(std)[0] == "success" and std.err_net == pytest.approx(0.200, abs=0.04) and std.err_ista == pytest.approx(0.7338, abs=0.003) and std.err_fista == pytest.approx(0.6685, abs=0.003)
    assert std.err_l1 == pytest.approx(0.0087, abs=0.003) and std.err_l1 < std.err_net / 5 and std.err_train < std.err_net
    assert ev.verdict_instance(scal)[0] == "success" and scal.err_net == pytest.approx(0.444, abs=0.06) and scal.gam_max > C.STEP_OVER and scal.err_net > std.err_net
    assert ev.verdict_instance(shift)[0] == "warning" and shift.err_net == pytest.approx(0.591, abs=0.08) and shift.err_net > shift.err_ista and shift.err_net > shift.err_tik and shift.err_net > 2.0 * std.err_net
    assert ev.verdict_instance(few)[0] == "warning" and few.err_train < 1e-6 and few.err_net > 0.7
    assert ev.verdict_instance(waves)[0] == "success" and waves.err_net == pytest.approx(0.0487, abs=0.015) and waves.err_net < waves.err_tik < waves.err_ista
    assert ev.verdict_instance(allw)[0] == "success" and allw.err_net > 1.8 * std.err_net and allw.err_net == pytest.approx(0.48, abs=0.08)
    assert ev.verdict_instance(two)[0] in ("info", "success") and two.err_net == pytest.approx(0.667, abs=0.06) and two.err_ista == pytest.approx(0.8307, abs=0.003) and two.err_fista == two.err_ista


def test_every_verdict_names_what_it_found(outcomes):
    assert "Verteilungswechsel: Trainingswelt Spitzen, Testwelt Blöcke." in ev.verdict_instance(outcomes["Verteilungswechsel"])[1]
    assert "Überanpassung" in ev.verdict_instance(outcomes["Wenige Signale"])[1] and "keine konvergente Iteration" in ev.verdict_instance(outcomes["Skalar-Netz"])[1]
    assert "Das exakte L1 ist besser" in ev.verdict_instance(outcomes["Standard"])[1]


def test_experiment_constants_are_consistent():
    assert C.N == 48 and C.EXP_K in C.K_CHOICES and C.POST_K in C.K_CHOICES and set(C.LAYER_KS) <= set(C.K_CHOICES) and C.EXP_STEPS in C.STEPS_CHOICES and C.DEFAULT_N_TRAIN in C.N_TRAIN_CHOICES
    assert set(C.SHIFT_WORLDS) <= set(C.TRAIN_WORLDS) and set(C.OVERFIT_SIZES) <= set(C.N_TRAIN_CHOICES) and set(C.SHIFT_NOISES) <= set(C.NOISE_CHOICES) and C.BASE_COUNT <= C.TEST_COUNT
    assert C.GAIN == 0.8 and C.OVERFIT == 0.5 and C.STEP_OVER == 2.0 and len(C.LAMBDA_GRID) == 13 and C.BATCH == 256 and set(C.LR) == set(C.KINDS) and set(C.SHOW_RANGE) == set(C.TEST_WORLDS)
    assert set(C.KIND_LABELS) == set(C.KIND_SHORT) == set(C.KINDS) and set(C.TRAIN_WORLDS) | {"gauss"} == set(C.WORLD_LABELS) == set(C.WORLD_SHORT)
