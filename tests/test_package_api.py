"""The top-level obscalc namespace: what a library caller can reach directly."""

import numpy as np
import pytest

import obscalc


SINGLE_CHANNEL = ["apf", "deimos", "hires"]
DOUBLE_CHANNEL = ["kast", "lris"]


@pytest.mark.parametrize("name", SINGLE_CHANNEL + DOUBLE_CHANNEL)
def test_every_instrument_is_reachable_from_the_top_level(name):
    """Not just APF, which was all the package exported to begin with."""
    for suffix in ("spectrograph", "thruput"):
        assert hasattr(obscalc, f"{name}_{suffix}"), f"{name}_{suffix}"
        assert f"{name}_{suffix}" in obscalc.__all__


@pytest.mark.parametrize("name", DOUBLE_CHANNEL)
def test_two_channel_instruments_export_a_sides_helper(name):
    """Without it the caller would need split_wavelengths from the submodule."""
    assert hasattr(obscalc, f"{name}_sides")
    assert f"{name}_sides" in obscalc.__all__


def test_the_engine_and_telescopes_are_exported():
    for name in (
        "spec_calcs2n",
        "run_sides",
        "Side",
        "S2NResult",
        "StackedResult",
        "Instrument",
        "Observation",
        "Telescope",
        "apf_telescope",
        "keck_telescope",
        "lick_telescope",
        "telescope",
    ):
        assert hasattr(obscalc, name), name
        assert name in obscalc.__all__


def test_everything_named_in_all_actually_exists():
    missing = [name for name in obscalc.__all__ if not hasattr(obscalc, name)]
    assert missing == []


def test_a_single_channel_run_needs_only_the_top_level():
    wave = np.arange(3000.0, 9500.0, 10.0)
    tel = obscalc.keck_telescope("KeckI")
    instr = obscalc.hires_spectrograph(decker="C5", epoch="new", str_tel=tel)
    obs = obscalc.Observation(seeing=0.7, mstar=15.0, mtype=2, exptime=1800.0)

    result = obscalc.spec_calcs2n(
        wave, obscalc.hires_thruput(wave, instr), tel, instr, obs
    )
    assert np.all(np.isfinite(result.sn)) and np.all(result.sn > 0)
    assert result.resolving_power == pytest.approx(instr.R / result.columns)


def test_a_two_channel_run_needs_only_the_top_level():
    wave = np.arange(3500.0, 10000.0, 10.0)
    tel = obscalc.keck_telescope("KeckI")
    blue, red = obscalc.lris_spectrograph(
        grism="B600", grating="600/7500", str_tel=tel
    )
    obs = obscalc.Observation(seeing=1.0, mstar=20.0, mtype=2, exptime=3600.0)

    result = obscalc.run_sides(wave, tel, obscalc.lris_sides(wave, blue, red), obs)
    assert [name for name, _, _ in result.sides] == ["blue", "red"]
    assert np.all(np.isfinite(result.sn))
    # Read noise steps across the dichroic, so both detectors really ran.
    assert result.side("blue").noise < result.side("red").noise


def test_the_sides_helper_matches_building_them_by_hand():
    """It is a convenience, not a different calculation."""
    from obscalc.instruments.kast import kast_thruput, split_wavelengths

    wave = np.arange(3150.0, 8000.0, 10.0)
    tel = obscalc.lick_telescope()
    blue, red = obscalc.kast_spectrograph(dichroic="d55", str_tel=tel)

    thru = kast_thruput(wave, blue, red)
    blue_index, red_index = split_wavelengths(wave, "d55")
    by_hand = [
        obscalc.Side("blue", blue, blue_index, thru[blue_index]),
        obscalc.Side("red", red, red_index, thru[red_index]),
    ]

    for helper, manual in zip(obscalc.kast_sides(wave, blue, red), by_hand):
        assert helper.name == manual.name
        assert np.array_equal(helper.index, manual.index)
        assert np.array_equal(helper.thru, manual.thru)
