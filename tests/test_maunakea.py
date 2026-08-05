"""Mauna Kea extinction and the empirical-only sky model.

The analytic moon-phase fallback in maunakea_sky.pro is deliberately absent, so
these tests pin that down as much as they check the physics.
"""

import numpy as np
import pytest

from obscalc.atmosphere import (
    MAUNAKEA_RANGE,
    _MAUNAKEA_EXTINCT,
    _MAUNAKEA_WAVE,
    extinction_for,
    maunakea_trans,
    mtham_trans,
)
from obscalc.sky import (
    DEFAULT_MAUNAKEA_MODEL,
    FLG_SKY,
    KECK_INSTRUMENT_MODELS,
    MAUNAKEA_MODELS,
    PLAUSIBLE_SKY_MAG,
    UNUSABLE_MODELS,
    coverage,
    maunakea_sky,
    median_sky_magnitude,
    mtham_sky,
    sky_for,
)
from obscalc.telescopes import keck_telescope, telescope

# --- telescope ---------------------------------------------------------------


def test_keck_telescope_values():
    for name in ("KeckI", "KeckII"):
        tel = keck_telescope(name)
        assert tel.name == name
        assert tel.area == 723674.0
        assert tel.plate_scale == 1.379


def test_keck_requires_a_valid_name():
    # x_initkeck.pro left name blank and the instrument inits filled it in; the
    # extinction and sky dispatch depends on it, so it cannot be empty.
    with pytest.raises(ValueError, match="KeckI"):
        keck_telescope("Keck")


def test_both_kecks_are_in_the_telescope_registry():
    assert telescope("KeckI").name == "KeckI"
    assert telescope("KeckII").name == "KeckII"


# --- extinction --------------------------------------------------------------


def test_maunakea_extinction_reproduces_its_table():
    assert np.allclose(maunakea_trans(_MAUNAKEA_WAVE), _MAUNAKEA_EXTINCT)
    assert maunakea_trans([3000.0]) == pytest.approx(4.90)
    assert maunakea_trans([5000.0]) == pytest.approx(0.13)
    assert maunakea_trans([12000.0]) == pytest.approx(0.03)


def test_maunakea_extinction_interpolates_midpoints():
    # Halfway between 4000/0.25 and 4250/0.21.
    assert maunakea_trans([4125.0]) == pytest.approx(0.23)


def test_maunakea_extinction_falls_from_blue_to_red():
    extinct = maunakea_trans([3200.0, 4000.0, 5000.0, 7000.0, 9000.0])
    assert list(extinct) == sorted(extinct, reverse=True)


def test_maunakea_extinction_extrapolates_unlike_mtham():
    """maunakea_trans.pro passed straight to interpol; mtham_trans.pro clamped."""
    below = MAUNAKEA_RANGE[0] - 100.0
    assert maunakea_trans([below])[0] > _MAUNAKEA_EXTINCT[0]

    # Mt Hamilton holds its end value instead.
    table_min = 3200.0
    assert mtham_trans([table_min - 500.0]) == pytest.approx(mtham_trans([table_min]))


def test_maunakea_extinction_is_lower_than_mt_hamilton_over_most_of_the_optical():
    for wave in (4000.0, 5000.0, 6000.0, 9000.0):
        assert maunakea_trans([wave])[0] < mtham_trans([wave])[0], wave
    # 7000 A is the exception: the coarse Mauna Kea table sits at 0.10 there
    # while the Mt Hamilton file gives 0.095.  The tables come from different
    # sources and are not smooth against each other.
    assert maunakea_trans([7000.0])[0] > mtham_trans([7000.0])[0]


def test_extinction_registry_covers_both_kecks():
    assert extinction_for("KeckI") is maunakea_trans
    assert extinction_for("KeckII") is maunakea_trans
    assert extinction_for("APF") is mtham_trans
    with pytest.raises(NotImplementedError, match="Subaru"):
        extinction_for("Subaru")


# --- sky models --------------------------------------------------------------


def test_flg_sky_records_the_idl_numbering_including_the_unusable_file():
    # maunakea_sky.pro selected files by a numeric flg_sky.  2 is kept in the
    # mapping for traceability even though the file behind it is not offered.
    assert FLG_SKY == {0: "deimos600", 1: "deimos1200", 2: "lris"}
    assert set(FLG_SKY.values()) - set(MAUNAKEA_MODELS) == set(UNUSABLE_MODELS)


def test_model_coverage_is_what_the_files_measure():
    assert coverage("deimos600") == pytest.approx((5000.6, 9999.0), abs=0.1)
    assert coverage("deimos1200") == pytest.approx((6281.2, 9329.2), abs=0.1)
    # No argument: the Mt Hamilton measurement.
    assert coverage()[0] == pytest.approx(3163.4, abs=0.1)


def test_the_default_model_has_the_widest_usable_coverage():
    spans = {
        name: coverage(name)[1] - coverage(name)[0] for name in MAUNAKEA_MODELS
    }
    assert max(spans, key=spans.get) == DEFAULT_MAUNAKEA_MODEL


def test_nothing_is_measured_blueward_of_5000_angstroms():
    """The cost of dropping the analytic fallback.

    Both usable models start in the green, so any Keck instrument working in the
    blue has no measured Mauna Kea sky to interpolate.
    """
    assert min(coverage(name)[0] for name in MAUNAKEA_MODELS) > 5000.0


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown Mauna Kea sky model"):
        maunakea_sky([5000.0], model="nope")


def test_the_lris_model_is_refused_with_its_reason():
    """mkea_sky_LRIS_both.fits is not in f_lambda like the others.

    Its fluxes have a median of 0.23 against 5e-18 for the DEIMOS files, and the
    IDL's conversion turns that into roughly -19 AB mag/arcsec^2.
    """
    assert "lris" in UNUSABLE_MODELS
    with pytest.raises(ValueError, match="not in f_lambda"):
        maunakea_sky([6000.0], model="lris")
    with pytest.raises(ValueError, match="not in f_lambda"):
        coverage("lris")


def test_every_offered_model_has_physical_units():
    """A model in the wrong units shows up immediately as a silly median."""
    lo, hi = PLAUSIBLE_SKY_MAG
    for name in MAUNAKEA_MODELS:
        assert lo < median_sky_magnitude(name) < hi, name
    assert lo < median_sky_magnitude() < hi  # Mt Hamilton


def test_sky_brightness_is_plausible_where_it_is_measured():
    lo, hi = coverage(DEFAULT_MAUNAKEA_MODEL)
    wave = np.arange(lo + 100.0, hi - 100.0, 50.0)
    magsky = maunakea_sky(wave)
    assert np.all(np.isfinite(magsky))
    # Between the airglow lines a dark sky sits around 21-22; the lines
    # themselves brighten it to about 17.
    assert magsky.min() > 16.0
    assert magsky.max() < 24.0
    assert 20.0 < np.median(magsky) < 23.0


def test_moon_phase_is_ignored_entirely():
    """The analytic phase-dependent table is gone; all models are new moon."""
    wave = np.arange(4000.0, 9000.0, 100.0)
    dark = maunakea_sky(wave, phase=0)
    full = maunakea_sky(wave, phase=14)
    assert np.array_equal(dark, full)


def test_outside_the_measurement_the_nearest_value_is_held():
    wave_min, wave_max = coverage("deimos600")
    inside_blue = maunakea_sky([wave_min + 0.5], model="deimos600")
    below = maunakea_sky([wave_min - 500.0], model="deimos600")
    # Held in f_lambda, so the magnitude still moves with the lambda^2/c factor;
    # what matters is that it stays finite and close, not extrapolated away.
    assert np.isfinite(below).all()
    assert abs(below[0] - inside_blue[0]) < 1.0

    above = maunakea_sky([wave_max + 500.0], model="deimos600")
    assert np.isfinite(above).all()


def test_a_narrow_model_does_not_silently_cover_the_blue():
    """deimos1200 starts at 6281 A, so a blue request is held, not measured.

    This is the cost of dropping the analytic fallback: callers have to check
    coverage() rather than trusting the number.
    """
    lo, _ = coverage("deimos1200")
    assert lo > 6000.0
    held = maunakea_sky([4000.0], model="deimos1200")
    assert np.isfinite(held).all()


# --- dispatch ----------------------------------------------------------------


def test_sky_dispatch_is_per_instrument_for_keck():
    """spec_calcs2n.pro nested a case on str_instr.name inside the KeckII branch."""
    assert sky_for("KeckII", "ESI").model == "deimos600"
    assert sky_for("KeckII", "DEIMOS").model == "deimos600"


def test_unlisted_keck_instruments_fall_back_to_the_default():
    # Keck I and HIRES reached the analytic fallback, which no longer exists.
    # LRIS selected flg_sky = 2, whose file is unusable, so it falls back too.
    assert sky_for("KeckI", "HIRES").model == DEFAULT_MAUNAKEA_MODEL
    assert sky_for("KeckII", "LRIS").model == DEFAULT_MAUNAKEA_MODEL
    assert sky_for("KeckII", None).model == DEFAULT_MAUNAKEA_MODEL


def test_instrument_name_matching_is_case_insensitive():
    assert sky_for("KeckII", "esi").model == "deimos600"
    assert set(KECK_INSTRUMENT_MODELS) == {"DEIMOS", "ESI"}


def test_sky_dispatch_ignores_the_instrument_where_it_does_not_matter():
    assert sky_for("APF", "APFSPEC") is mtham_sky
    assert sky_for("Lick-3m", "Kast-blue") is mtham_sky


def test_sky_registry_still_rejects_unknown_telescopes():
    with pytest.raises(NotImplementedError, match="Subaru"):
        sky_for("Subaru")


def test_maunakea_sky_takes_no_noempir_option():
    """The /NOEMPIR switch is gone; empirical models are the only path."""
    import inspect

    parameters = inspect.signature(maunakea_sky).parameters
    assert list(parameters) == ["wave", "phase", "model"]
    assert not any("empir" in name.lower() for name in parameters)
