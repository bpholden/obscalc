import numpy as np
import pytest

from obscalc import Observation, apf_spectrograph, apf_telescope, apf_thruput
from obscalc.s2n import spec_calcs2n


@pytest.fixture
def setup():
    wave = np.arange(3742.0, 7700.0, 10.0)
    return wave, apf_telescope(), apf_thruput(wave)


def run(wave, tel, thru, decker="W", **obs_kwargs):
    kwargs = dict(seeing=1.2, mstar=13.0, mtype=2, exptime=1200.0)
    kwargs.update(obs_kwargs)
    bins = kwargs.pop("bins", 1)
    bind = kwargs.pop("bind", 1)
    instr = apf_spectrograph(decker=decker, bins=bins, bind=bind)
    return spec_calcs2n(wave, thru, tel, instr, Observation(**kwargs))


def test_geometry_matches_the_hand_calculation(setup):
    result = run(*setup)
    # scale_perp = 5.8241 * 5.5136 * 0.0135 = 0.4335 arcsec/pixel.
    # rows = long(3*1.2/0.4335 + 0.999) = 9; the 3 arcsec decker spans 7 rows.
    assert result.rows == 9
    assert result.nsky == pytest.approx((7 - 9) / 9)
    assert result.columns == pytest.approx(min(1.0, 3.6) / 0.392442709455)
    assert result.noise == pytest.approx(3.75 * np.sqrt(9.0))


def test_no_nans_where_the_idl_produced_them(setup):
    # nsky < 0 for the W decker at 1.2 arcsec seeing.  The IDL applied
    # (1 + 1/nsky) = -3.5 regardless and got the square root of a negative
    # number; this must not happen here.
    result = run(*setup)
    assert result.nsky < 0
    assert np.all(np.isfinite(result.sn))
    assert np.all(result.sn > 0)


def test_tall_decker_keeps_the_sky_subtraction_penalty(setup):
    # The 8 arcsec deckers do give sky rows, so the penalty applies and the
    # result is noisier than the same setup without it.
    result = run(*setup, decker="M")
    assert result.nsky > 0
    expected = np.sqrt(
        result.star
        + (1.0 + 1.0 / result.nsky) * (result.noise**2 + result.sky + result.ndark)
    )
    assert np.allclose(result.tnoise, expected)


def test_signal_to_noise_scales_as_root_time_when_source_dominated(setup):
    # A flat throughput, so this exercises the engine rather than the echelle
    # blaze: with the blaze applied, wavelengths near an order edge are
    # read-noise dominated and scale closer to linearly in time.
    wave, tel, _ = setup
    thru = np.full(wave.shape, 0.15)
    bright = dict(mstar=8.0)
    short = run(wave, tel, thru, exptime=100.0, **bright)
    long = run(wave, tel, thru, exptime=400.0, **bright)
    ratio = np.median(long.sn / short.sn)
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_counts_scale_correctly_with_magnitude(setup):
    wave, tel, thru = setup
    faint = run(wave, tel, thru, mstar=15.0)
    bright = run(wave, tel, thru, mstar=10.0)
    assert np.allclose(bright.star / faint.star, 10.0 ** (0.4 * 5.0))


def test_higher_airmass_reduces_the_signal(setup):
    wave, tel, thru = setup
    low = run(wave, tel, thru, airmass=1.0)
    high = run(wave, tel, thru, airmass=2.0)
    assert np.all(high.star < low.star)


def test_worse_seeing_reduces_the_slit_fraction(setup):
    wave, tel, thru = setup
    assert run(wave, tel, thru, seeing=2.0).slit0 < run(
        wave, tel, thru, seeing=0.7
    ).slit0


def test_dark_current_scales_with_dispersion_binning_and_rows(setup):
    result = run(*setup, bind=2)
    assert result.ndark == pytest.approx(2 * 7.3 * result.rows * 1200.0 / 3600.0)


def test_read_noise_falls_with_spatial_binning(setup):
    wave, tel, thru = setup
    assert run(wave, tel, thru, bins=4).noise < run(wave, tel, thru, bins=1).noise


def test_every_decker_produces_finite_results(setup):
    wave, tel, thru = setup
    for decker in ("N", "S", "M", "W", "O", "T", "B"):
        result = run(wave, tel, thru, decker=decker)
        assert np.all(np.isfinite(result.sn)), decker
        assert result.slit0 > 0.0


def test_mismatched_wave_and_thru_is_rejected(setup):
    wave, tel, thru = setup
    with pytest.raises(ValueError, match="same shape"):
        spec_calcs2n(wave, thru[:-1], tel, apf_spectrograph(), Observation())


def test_zero_seeing_is_rejected(setup):
    wave, tel, thru = setup
    with pytest.raises(ValueError, match="seeing"):
        spec_calcs2n(
            wave, thru, tel, apf_spectrograph(), Observation(seeing=0.0)
        )


def test_default_magnitude_and_mtype_follow_the_idl(setup):
    wave, tel, thru = setup
    # mstar < 1 falls back to 17, mtype == 0 falls back to AB.
    result = run(wave, tel, thru, mstar=0.0, mtype=0)
    assert result.mstar == 17.0
    assert result.mtype == 2


def test_resolution_element_signal_to_noise_is_higher_than_per_pixel(setup):
    result = run(*setup)
    assert np.all(result.sn_per_resolution_element > result.sn)
