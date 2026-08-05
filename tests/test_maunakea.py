"""Mauna Kea extinction and the empirical-only sky models.

The analytic moon-phase fallback in maunakea_sky.pro is deliberately absent, and
the LRIS sky frames have their throughput divided back out, so these tests pin
down both decisions as much as they check the physics.
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
    RED_VALIDATION,
    _DEIMOS600_MIN,
    _LRIS_BLUE_MAX,
    _read_wcs_spectrum,
    coverage,
    maunakea_sky,
    median_sky_magnitude,
    mtham_sky,
    sky_for,
    validate_against_deimos,
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

    table_min = 3200.0
    assert mtham_trans([table_min - 500.0]) == pytest.approx(mtham_trans([table_min]))


def test_maunakea_extinction_is_lower_than_mt_hamilton_over_most_of_the_optical():
    for wave in (4000.0, 5000.0, 6000.0, 9000.0):
        assert maunakea_trans([wave])[0] < mtham_trans([wave])[0], wave
    # 7000 A is the exception: the coarse Mauna Kea table sits at 0.10 there
    # while the Mt Hamilton file gives 0.095.  Different sources, not smooth
    # against each other.
    assert maunakea_trans([7000.0])[0] > mtham_trans([7000.0])[0]


def test_extinction_registry_covers_both_kecks():
    assert extinction_for("KeckI") is maunakea_trans
    assert extinction_for("KeckII") is maunakea_trans
    assert extinction_for("APF") is mtham_trans
    with pytest.raises(NotImplementedError, match="Subaru"):
        extinction_for("Subaru")


# --- reading the LRIS frames -------------------------------------------------


def test_lris_frames_are_read_through_their_wcs():
    """One-dimensional images with CRVAL1/CDELT1, not wavelength arrays."""
    wave, values = _read_wcs_spectrum("bsky.eps_pang_parcsec.fits")
    assert wave.size == values.size == 1604
    assert wave[0] == pytest.approx(3001.8)
    assert wave[1] - wave[0] == pytest.approx(2.18)
    assert wave[-1] == pytest.approx(3001.8 + 1603 * 2.18)


def test_lris_frames_are_in_detected_electrons_not_flux():
    """Median near 0.1 e-/s/Ang/arcsec^2, which is why they need the undo.

    Reading them as f_lambda is what gave maunakea_sky.pro's flg_sky = 2 model
    sky brightnesses around -19 AB mag/arcsec^2.
    """
    _, values = _read_wcs_spectrum("bsky.eps_pang_parcsec.fits")
    assert 0.01 < np.median(values) < 1.0


# --- sky models --------------------------------------------------------------


def test_flg_sky_records_the_idl_numbering():
    assert FLG_SKY == {0: "deimos600", 1: "deimos1200", 2: "lris_blue"}
    assert set(FLG_SKY.values()) <= set(MAUNAKEA_MODELS)


def test_model_coverage():
    assert coverage("deimos600") == pytest.approx((5000.6, 9999.0), abs=0.1)
    assert coverage("deimos1200") == pytest.approx((6281.2, 9329.2), abs=0.1)
    assert coverage("lris_blue") == pytest.approx((3101.9, 4999.0), abs=1.0)
    assert coverage("combined") == pytest.approx((3101.9, 9999.0), abs=1.0)
    # No argument: the Mt Hamilton measurement.
    assert coverage()[0] == pytest.approx(3163.4, abs=0.1)


def test_the_default_model_has_the_widest_coverage():
    spans = {
        name: coverage(name)[1] - coverage(name)[0] for name in MAUNAKEA_MODELS
    }
    assert max(spans, key=spans.get) == DEFAULT_MAUNAKEA_MODEL


def test_the_blue_channel_is_the_only_thing_measuring_below_5000_angstroms():
    """This is what the LRIS frames buy: DEIMOS starts at 5001 A."""
    assert coverage("lris_blue")[0] < 3200.0
    assert min(coverage(n)[0] for n in ("deimos600", "deimos1200")) > 5000.0
    assert coverage(DEFAULT_MAUNAKEA_MODEL)[0] < 3200.0


def test_the_blue_channel_stops_at_the_dichroic_handover():
    """Past dichroic 500 the blue channel's recovered sky is not reliable."""
    assert coverage("lris_blue")[1] <= _LRIS_BLUE_MAX


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown Mauna Kea sky model"):
        maunakea_sky([5000.0], model="nope")


def test_every_model_has_physical_units():
    """A model in the wrong units shows up immediately as a silly median."""
    lo, hi = PLAUSIBLE_SKY_MAG
    for name in MAUNAKEA_MODELS:
        assert lo < median_sky_magnitude(name) < hi, name
    assert lo < median_sky_magnitude() < hi  # Mt Hamilton


def test_the_recovered_blue_sky_is_a_plausible_dark_blue_sky():
    wave = np.arange(3300.0, 4950.0, 25.0)
    magsky = maunakea_sky(wave, model="lris_blue")
    assert np.all(np.isfinite(magsky))
    # Dark sky at Mauna Kea is around 22.7 AB at B.
    assert 22.0 < np.median(magsky) < 23.5
    assert magsky.min() > 20.0
    assert magsky.max() < 24.5


def test_the_recovered_blue_sky_does_not_darken_toward_the_red():
    """The check that caught the wrong throughput curve.

    Real sky brightness rises slightly toward the red.  Undoing with the
    300/5000 grism, blazed 1600 A away from the 400/3400 actually used, gave a
    sky that darkened by 1.4 mag from 4000 to 5000 A.  The 600/4000 curve does
    not.
    """
    blue = maunakea_sky([4000.0], model="lris_blue")[0]
    red = maunakea_sky([4900.0], model="lris_blue")[0]
    assert red - blue < 0.5


def test_undoing_the_throughput_reproduces_the_deimos_measurement():
    """The evidence that the method works.

    The red channel adds no coverage the DEIMOS models lack, and is kept purely
    for this: recovering it with the same-ruling 600/7500 curve agrees with an
    independent sky measurement to well under a tenth of a magnitude.
    """
    offset, scatter = validate_against_deimos("lris_red")
    assert offset == pytest.approx(RED_VALIDATION[0], abs=0.01)
    assert scatter == pytest.approx(RED_VALIDATION[1], abs=0.01)
    assert abs(offset) < 0.15
    assert scatter < 0.4


def test_the_combined_model_joins_without_a_step():
    """Blue below 5000 A, DEIMOS above 5200, interpolated across the bridge.

    Neither measures 5000-5200: the blue channel has fallen off the dichroic and
    DEIMOS has not come up off its blue edge, where it reads 24.1 mag at 5001 A
    against 22.5 at 5200.
    """
    wave = np.arange(_LRIS_BLUE_MAX - 200.0, _DEIMOS600_MIN + 200.0, 20.0)
    magsky = maunakea_sky(wave, model="combined")
    assert np.all(np.isfinite(magsky))
    # No discontinuity: successive points stay within a few tenths.
    assert np.abs(np.diff(magsky)).max() < 0.6


def test_the_combined_model_matches_its_two_sources_away_from_the_bridge():
    for wave, source in ((4000.0, "lris_blue"), (7000.0, "deimos600")):
        assert maunakea_sky([wave], model="combined")[0] == pytest.approx(
            maunakea_sky([wave], model=source)[0], abs=0.01
        )


def test_sky_brightness_is_plausible_where_it_is_measured():
    lo, hi = coverage(DEFAULT_MAUNAKEA_MODEL)
    wave = np.arange(lo + 100.0, hi - 100.0, 50.0)
    magsky = maunakea_sky(wave)
    assert np.all(np.isfinite(magsky))
    assert magsky.min() > 16.0
    assert magsky.max() < 24.5
    assert 20.0 < np.median(magsky) < 23.0


def test_moon_phase_is_ignored_entirely():
    """The analytic phase-dependent table is gone; all models are new moon."""
    wave = np.arange(4000.0, 9000.0, 100.0)
    assert np.array_equal(maunakea_sky(wave, phase=0), maunakea_sky(wave, phase=14))


def test_outside_the_measurement_the_nearest_value_is_held():
    wave_min, wave_max = coverage("deimos1200")
    inside = maunakea_sky([wave_min + 0.5], model="deimos1200")
    below = maunakea_sky([wave_min - 500.0], model="deimos1200")
    assert np.isfinite(below).all()
    assert abs(below[0] - inside[0]) < 1.0
    assert np.isfinite(maunakea_sky([wave_max + 500.0], model="deimos1200")).all()


# --- dispatch ----------------------------------------------------------------


def test_sky_dispatch_is_per_instrument_for_keck():
    """spec_calcs2n.pro nested a case on str_instr.name inside the KeckII branch."""
    assert sky_for("KeckII", "ESI").model == "deimos600"
    assert sky_for("KeckII", "DEIMOS").model == "deimos600"


def test_unlisted_keck_instruments_fall_back_to_the_default():
    # Keck I and HIRES reached the analytic fallback, which no longer exists.
    # LRIS selected flg_sky = 2, superseded by the recovered blue model.
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
