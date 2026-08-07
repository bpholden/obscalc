"""DEIMOS: a single channel whose throughput depends on grating *and* tilt."""

import itertools

import numpy as np
import pytest

from obscalc.instruments.base import ParameterError
from obscalc.instruments.deimos import (
    CENTRAL_WAVES,
    DEFAULT_CWAVE,
    DEFAULT_GRATING,
    DEFAULT_RANGE,
    GRATINGS,
    MIN_THRUPUT,
    SENS_FILES,
    DeimosBackend,
    _sensitivity,
    deimos_spectrograph,
    deimos_thruput,
    sens_file,
    sensitivity_range,
)
from obscalc.s2n import Side, run_sides
from obscalc.structures import Observation
from obscalc.telescopes import keck_telescope

GRID = np.arange(*DEFAULT_RANGE, 10.0)


def result_for(**kwargs):
    tel = keck_telescope("KeckII")
    obs_kwargs = {"seeing": 0.7, "mstar": 22.0, "mtype": 2, "exptime": 1200.0}
    obs_kwargs.update(kwargs.pop("obs", {}))
    instr = deimos_spectrograph(**kwargs, str_tel=tel)
    thru = deimos_thruput(GRID, instr)
    return run_sides(
        GRID, tel, [Side("", instr, np.arange(GRID.size), thru)], Observation(**obs_kwargs)
    )


# --- instrument --------------------------------------------------------------


def test_deimos_sits_on_keck_two():
    instr = deimos_spectrograph()
    assert instr.name == "DEIMOS"
    # KeckII, and spec_calcs2n.pro's nested case picks the DEIMOS sky model.
    assert keck_telescope("KeckII").name == "KeckII"


def test_pixel_scale_is_set_from_the_observed_resolution():
    # MAG 8.03 was chosen so 0.75 arcsec maps onto 4.5 pixels.
    instr = deimos_spectrograph()
    assert instr.scale_perp == pytest.approx(1.379 * 8.03 * 0.015)
    assert instr.scale_perp == pytest.approx(0.1661, abs=1e-4)
    assert instr.scale_perp == instr.scale_para


def test_detector_and_slit_values():
    instr = deimos_spectrograph(slit=0.75)
    assert instr.readno == 2.6
    assert instr.dark == 4.0
    assert instr.swidth == 0.75
    assert instr.sheight == 10.0
    assert instr.bins == 1 and instr.bind == 1


def test_resolving_power_follows_the_grating():
    for grating, resolution in GRATINGS.items():
        assert deimos_spectrograph(grating=grating).R == resolution
    # The two 1200 line gratings differ in blaze, not dispersion.
    assert GRATINGS["1200G"] == GRATINGS["1200B"]


def test_the_default_grating_is_valid():
    """x_initdeimos.pro defaulted to '1200', which its own case list rejects.

    Both it and deimos_calcs2n_wrapper.pro set that default, so the default
    configuration ran into `else: stop`.
    """
    assert "1200" not in GRATINGS
    assert DEFAULT_GRATING in GRATINGS
    assert deimos_spectrograph().grating == DEFAULT_GRATING


def test_the_central_wavelength_is_recorded():
    assert deimos_spectrograph(cwave=5000).cwave == 5000.0
    assert deimos_spectrograph().cwave == float(DEFAULT_CWAVE)


def test_unknown_configurations_are_rejected():
    with pytest.raises(ValueError, match="grating"):
        deimos_spectrograph(grating="1200")
    with pytest.raises(ValueError, match="central wavelength"):
        deimos_spectrograph(cwave=6500)
    with pytest.raises(ValueError, match="must be a number"):
        deimos_spectrograph(cwave="red")


# --- which measurement applies -----------------------------------------------


def test_every_configuration_maps_to_a_readable_file():
    for grating, cwave in itertools.product(sorted(GRATINGS), CENTRAL_WAVES):
        wave, eff = _sensitivity(sens_file(grating, cwave))
        assert wave.size > 100, (grating, cwave)
        assert eff.max() < 0.4


def test_the_1200_gratings_use_one_file_for_every_tilt():
    for grating in ("1200G", "1200B"):
        files = {sens_file(grating, c) for c in CENTRAL_WAVES}
        assert len(files) == 1


def test_600z_reuses_the_750nm_measurement_at_the_reddest_tilt():
    """No 850 nm measurement was ever taken; deimos_thruput.pro reused 750 nm."""
    assert sens_file("600Z", 8000) == sens_file("600Z", 7000)
    assert SENS_FILES["600Z"][8000] == "sens_DEIMOS_600_750nm.fits"


def test_900z_has_a_distinct_measurement_per_tilt():
    files = {sens_file("900Z", c) for c in CENTRAL_WAVES}
    assert len(files) == 4


def test_the_tilt_moves_where_the_measurement_lies():
    ranges = [sensitivity_range("900Z", c) for c in CENTRAL_WAVES]
    assert [low for low, _ in ranges] == sorted(low for low, _ in ranges)
    assert sensitivity_range("900Z", 5000)[0] < sensitivity_range("900Z", 8000)[0]


# --- throughput --------------------------------------------------------------


def test_throughput_is_a_plausible_fraction_everywhere():
    for grating, cwave in itertools.product(sorted(GRATINGS), CENTRAL_WAVES):
        instr = deimos_spectrograph(grating=grating, cwave=cwave)
        thru = deimos_thruput(GRID, instr)
        label = f"{grating} {cwave}"
        assert np.all(thru >= 0.0), label
        # No configuration can beat the best efficiency ever measured for DEIMOS.
        assert thru.max() <= 0.36, label


def test_outside_the_measurement_the_throughput_is_dead():
    """Neither extrapolated nor held, as for LRIS and Kast.

    deimos_thruput.pro extrapolated and clipped only at zero: for the 600Z
    grating at its 5000 A tilt, measured only to 8035 A, that reached 0.848 by
    10000 A -- more than twice the best efficiency measured for any DEIMOS
    configuration.
    """
    from obscalc.cli_common import DEAD_THRUPUT

    instr = deimos_spectrograph(grating="600Z", cwave=5000)
    low, high = sensitivity_range("600Z", 5000)

    assert deimos_thruput([high + 1.0], instr)[0] == MIN_THRUPUT
    assert deimos_thruput([10000.0], instr)[0] == MIN_THRUPUT
    assert deimos_thruput([low - 1.0], instr)[0] == MIN_THRUPUT
    assert MIN_THRUPUT < DEAD_THRUPUT
    # Still a real measurement inside the range.  Taken at the middle, since the
    # curve is only 0.009 a hundred Angstroms in from its blue edge.
    assert deimos_thruput([(low + high) / 2.0], instr)[0] > 0.1


def test_negative_efficiencies_inside_a_measurement_are_clipped():
    """sens_DEIMOS_900_500nm holds 133 slightly negative efficiencies."""
    _, eff = _sensitivity(sens_file("900Z", 5000))
    assert (eff < 0).sum() > 0
    instr = deimos_spectrograph(grating="900Z", cwave=5000)
    assert np.all(deimos_thruput(GRID, instr) >= 0.0)


def test_columns_are_read_by_name_not_position():
    # sens_DEIMOS_900_500nm stores EFF before WAV.
    wave, eff = _sensitivity(sens_file("900Z", 5000))
    assert wave.min() > 1000.0  # wavelengths, not efficiencies
    assert eff.max() < 1.0


def test_both_ends_of_the_grid_go_dead():
    """Every DEIMOS measurement is narrower than the 4000-10000 A grid."""
    instr = deimos_spectrograph(grating="1200G", cwave=7000)
    low, high = sensitivity_range("1200G", 7000)
    thru = deimos_thruput(GRID, instr)

    assert np.all(thru[GRID < low] == MIN_THRUPUT)
    assert np.all(thru[GRID > high] == MIN_THRUPUT)
    assert (thru == MIN_THRUPUT).sum() == ((GRID < low) | (GRID > high)).sum()


def test_a_grid_inside_the_measurement_has_nothing_dead():
    low, high = sensitivity_range("1200G", 7000)
    inside = np.arange(low + 10.0, high - 10.0, 10.0)
    instr = deimos_spectrograph(grating="1200G", cwave=7000)
    assert np.all(deimos_thruput(inside, instr) > MIN_THRUPUT)


# --- through the engine ------------------------------------------------------


def test_signal_to_noise_is_finite_across_the_range():
    result = result_for()
    assert np.all(np.isfinite(result.sn))
    assert np.all(result.sn > 0)


def test_every_configuration_runs():
    for grating, cwave in itertools.product(sorted(GRATINGS), CENTRAL_WAVES):
        result = result_for(grating=grating, cwave=cwave)
        assert np.all(np.isfinite(result.sn)), f"{grating} {cwave}"


def test_the_tall_slit_always_leaves_sky_rows():
    # 10 arcsec, so the negative-nsky case does not arise at sane seeing.
    for seeing in (0.5, 0.7, 1.5):
        result = result_for(obs={"seeing": seeing, "mstar": 22.0})
        assert result.sides[0][2].nsky > 0, seeing


def test_a_faint_target_is_sky_limited_not_source_limited():
    # V=22 in 1200 s on a 10 m: the sky should dominate the noise budget.
    result = result_for()
    assert np.median(result.sky) > np.median(result.star)


def test_signal_to_noise_falls_with_magnitude():
    medians = [np.median(result_for(obs={"mstar": m}).sn) for m in (20.0, 22.0, 24.0)]
    assert medians == sorted(medians, reverse=True)


def test_a_template_run_succeeds():
    result = result_for(
        obs={
            "mstar": 22.0,
            "mtype": 2,
            "template": "sn1a10d_template.fits",
            "filter": "sdss_r.dat",
        }
    )
    assert np.all(np.isfinite(result.sn))


# --- backend -----------------------------------------------------------------


def test_backend_returns_one_side():
    backend = DeimosBackend()
    tel, sides = backend.sides(
        GRID, {"bins": 1, "bind": 1, "slitwidth": "1.0", "grating": "1200G"}
    )
    assert tel.name == "KeckII"
    assert len(sides) == 1


def test_backend_slitwidth_is_a_width():
    backend = DeimosBackend()
    _, sides = backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "0.75"})
    assert sides[0].instr.swidth == 0.75
    with pytest.raises(ParameterError, match="Slitwidth"):
        backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "wide"})


@pytest.mark.parametrize(
    "override,label",
    [
        ({"grating": "1200"}, "Grating"),
        ({"grating": "600"}, "Grating"),
        ({"cwave": "6500"}, "Central Wavelength"),
        ({"cwave": "red"}, "Central Wavelength"),
    ],
)
def test_backend_rejects_bad_configurations(override, label):
    backend = DeimosBackend()
    values = {"bins": 1, "bind": 1, "slitwidth": "1.0", **override}
    with pytest.raises(ParameterError, match=label):
        backend.sides(GRID, values)


def test_backend_defaults_need_no_grating_or_cwave():
    backend = DeimosBackend()
    _, sides = backend.sides(GRID, {"bins": 1, "bind": 1})
    assert sides[0].instr.grating == DEFAULT_GRATING
    assert sides[0].instr.cwave == float(DEFAULT_CWAVE)


def test_backend_has_no_instrument_specific_extras():
    assert DeimosBackend().extras(result_for(), Observation()) is None
