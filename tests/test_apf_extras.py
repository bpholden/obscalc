import numpy as np
import pytest

from obscalc import Observation, apf_spectrograph, apf_telescope, apf_thruput
from obscalc.apf_extras import (
    I2_WINDOW,
    apf_extras,
    exposure_meter_value,
    i2counts,
    rv_precision,
    template_bmv,
)
from obscalc.s2n import spec_calcs2n


@pytest.fixture
def result():
    wave = np.arange(3742.0, 7700.0, 10.0)
    tel = apf_telescope()
    instr = apf_spectrograph(decker="N")
    obs = Observation(seeing=1.2, mstar=9.0, mtype=1, exptime=600.0)
    return spec_calcs2n(wave, apf_thruput(wave), tel, instr, obs)


def test_i2counts_is_the_median_over_the_iodine_window(result):
    window = (result.wave >= I2_WINDOW[0]) & (result.wave <= I2_WINDOW[1])
    assert i2counts(result.wave, result.star) == pytest.approx(
        np.median(result.star[window])
    )


def test_i2counts_rejects_a_grid_that_misses_the_window():
    with pytest.raises(ValueError, match="iodine"):
        i2counts(np.array([4000.0, 4100.0]), np.array([1.0, 2.0]))


def test_rv_precision_improves_with_more_counts():
    bright = rv_precision(1e5, bmv=0.65)
    faint = rv_precision(1e3, bmv=0.65)
    assert bright < faint


def test_rv_precision_uses_the_red_star_calibration_above_the_break():
    # Different coefficients either side of B-V = 1.2.
    assert rv_precision(1e4, bmv=1.0) != rv_precision(1e4, bmv=1.4)


def test_rv_precision_is_plausible_for_a_bright_star():
    # A V=9 G star in 600 s on APF should land within a few m/s, not hundreds.
    # This is what the precision_value bug in apf_calcs2n.pro got wrong.
    assert 0.5 < rv_precision(6.3e3, bmv=0.69) < 20.0


def test_zero_counts_give_zero_rather_than_a_log_of_zero():
    assert rv_precision(0.0, bmv=0.65) == 0.0
    assert exposure_meter_value(0.0, 0.65) == 0.0


def test_exposure_meter_grows_with_counts():
    assert exposure_meter_value(1e4, 0.65) > exposure_meter_value(1e3, 0.65)


def test_template_bmv_of_vega_is_zero():
    obs = Observation(template="alpha_lyr_stis_005.fits", filter="Buser_V.dat")
    assert template_bmv(obs) == pytest.approx(0.0, abs=1e-9)


def test_template_bmv_ordering_across_spectral_types():
    # O5V is bluer than G5V is bluer than M5V.
    colours = [
        template_bmv(Observation(template=name, filter="Buser_V.dat"))
        for name in (
            "O5V_pickles_1.fits",
            "G5V_pickles_27.fits",
            "M5V_pickles_44.fits",
        )
    ]
    assert colours == sorted(colours)
    assert colours[0] < 0.0 < colours[1] < colours[2]


def test_extras_without_a_template_have_no_colour(result):
    obs = Observation(seeing=1.2, mstar=9.0, mtype=1, exptime=600.0)
    extras = apf_extras(result, obs)
    assert extras.bmv is None
    assert extras.expmeter == 0.0
    assert extras.precision == 0.0
    assert extras.i2counts > 0.0


def test_extras_with_a_template_report_colour_and_precision():
    wave = np.arange(3742.0, 7700.0, 10.0)
    obs = Observation(
        seeing=1.2,
        mstar=9.0,
        mtype=1,
        exptime=600.0,
        template="G5V_pickles_27.fits",
        filter="Buser_V.dat",
    )
    res = spec_calcs2n(
        wave, apf_thruput(wave), apf_telescope(), apf_spectrograph(decker="N"), obs
    )
    extras = apf_extras(res, obs)
    assert extras.bmv == pytest.approx(0.688, abs=0.01)
    assert 0.5 < extras.precision < 20.0
    assert extras.expmeter > 0.0
