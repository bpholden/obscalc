import numpy as np
import pytest

from obscalc.atmosphere import _extinction_table, extinction_for, mtham_trans
from obscalc.sky import _LICK_FILE, _read_sky, mtham_sky, sky_for


def test_extinction_reproduces_table_values_at_nodes():
    table_wave, table_mag = _extinction_table()
    assert np.allclose(mtham_trans(table_wave), table_mag)


def test_extinction_is_held_flat_outside_the_table():
    table_wave, table_mag = _extinction_table()
    below = mtham_trans([table_wave.min() - 500.0])
    above = mtham_trans([table_wave.max() + 500.0])
    assert below == pytest.approx(table_mag[table_wave.argmin()])
    assert above == pytest.approx(table_mag[table_wave.argmax()])


def test_extinction_decreases_from_blue_to_red():
    extinct = mtham_trans([3500.0, 4500.0, 5500.0, 7000.0])
    assert list(extinct) == sorted(extinct, reverse=True)


def test_extinction_registry_rejects_unsupported_telescopes():
    assert extinction_for("APF") is mtham_trans
    with pytest.raises(NotImplementedError, match="Subaru"):
        extinction_for("Subaru")


def test_sky_registry_rejects_unsupported_telescopes():
    assert sky_for("APF") is mtham_sky
    with pytest.raises(NotImplementedError, match="Subaru"):
        sky_for("Subaru")


def test_sky_brightness_is_plausible_across_the_apf_range():
    wave = np.arange(3800.0, 7600.0, 50.0)
    magsky = mtham_sky(wave)
    # Dark sky at a mid-altitude site: roughly 18-23 AB mag/arcsec^2.
    assert np.all(np.isfinite(magsky))
    assert magsky.min() > 17.0
    assert magsky.max() < 24.0


def test_sky_ignores_moon_phase():
    # mtham_sky interpolates a single dark-sky measurement; phase does nothing.
    wave = np.array([5000.0, 6000.0])
    assert np.allclose(mtham_sky(wave, 0), mtham_sky(wave, 14))


def test_sky_below_the_table_uses_the_bluest_measurement():
    sky_wave, sky_flam = _read_sky(_LICK_FILE)
    blue = sky_wave.min() - 100.0
    # The fix for mtham_sky.pro:56: below the table, hold the first flux value.
    expected_fnu = sky_flam[0] / 3e10 * blue * (blue * 1e-8)
    with np.errstate(divide="ignore"):
        expected = -np.log10(expected_fnu) / 0.4 - 48.6
    assert mtham_sky([blue]) == pytest.approx(expected)
