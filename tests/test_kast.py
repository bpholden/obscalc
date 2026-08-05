"""Kast: the first two-channel instrument, so the dichroic split matters."""

import itertools

import numpy as np
import pytest

from obscalc.instruments.kast import (
    BLUE_SENS_FILES,
    DICHROICS,
    GRATINGS,
    GRISMS,
    MIN_THRUPUT,
    RED_SENS_FILES,
    KastBackend,
    kast_spectrograph,
    kast_thruput,
    sensitivity_range,
    split_wavelengths,
)
from obscalc.instruments.base import ParameterError
from obscalc.s2n import Side, run_sides
from obscalc.structures import Observation
from obscalc.telescopes import lick_telescope

GRID = np.arange(3150.0, 8000.0, 10.0)


def sides_for(dichroic="d46", grism="G2", grating="600/7500", slit=1.5, bins=1, bind=1):
    tel = lick_telescope()
    blue, red = kast_spectrograph(
        grism=grism,
        grating=grating,
        dichroic=dichroic,
        slit=slit,
        bins=bins,
        bind=bind,
        str_tel=tel,
    )
    thru = kast_thruput(GRID, blue, red)
    blue_index, red_index = split_wavelengths(GRID, dichroic)
    return tel, [
        Side("blue", blue, blue_index, thru[blue_index]),
        Side("red", red, red_index, thru[red_index]),
    ]


# --- telescope and instrument ------------------------------------------------


def test_lick_telescope_values():
    tel = lick_telescope()
    assert tel.name == "Lick-3m"
    assert tel.area == 63617.0
    assert tel.plate_scale == 1.379


def test_pixel_scale_is_the_same_on_both_sides():
    """Both channels are 0.43"/pixel, per the current instrument documentation.

    MAG 20.9 with 15 micron pixels at 1.379"/mm gives 0.4323" on each side.  The
    "(0.78")" comment on the red side in x_initkast.pro is out of date and
    refers to the pre-upgrade red CCD.
    """
    blue, red = kast_spectrograph()
    expected = 1.379 * 20.9 * 0.015
    assert blue.scale_perp == pytest.approx(expected)
    assert red.scale_perp == pytest.approx(expected)
    assert blue.scale_perp == pytest.approx(0.4323, abs=1e-4)


def test_read_noise_differs_per_detector():
    """x_initkast.pro meant 3.7 blue and 3.8 red but clobbered both to 3.8.

    ``kastinstr.readno = 3.7`` then ``kastinstr.readno = 3.8`` on a two-element
    array overwrites the first assignment.
    """
    blue, red = kast_spectrograph()
    assert blue.readno == 3.7
    assert red.readno == 3.8


def test_resolving_power_follows_the_disperser():
    for grism, resolution in GRISMS.items():
        blue, _ = kast_spectrograph(grism=grism)
        assert blue.R == resolution
    for grating, resolution in GRATINGS.items():
        _, red = kast_spectrograph(grating=grating)
        assert red.R == resolution


def test_slit_height_is_the_long_slit():
    blue, red = kast_spectrograph(slit=2.0)
    assert blue.sheight == 120.0 and red.sheight == 120.0
    assert blue.swidth == 2.0 and red.swidth == 2.0


def test_unknown_configurations_are_rejected():
    with pytest.raises(ValueError, match="grism"):
        kast_spectrograph(grism="G9")
    with pytest.raises(ValueError, match="grating"):
        kast_spectrograph(grating="1200/5000")
    with pytest.raises(ValueError, match="dichroic"):
        kast_spectrograph(dichroic="d99")


# --- the dichroic ------------------------------------------------------------


@pytest.mark.parametrize("dichroic,crossover", sorted(DICHROICS.items()))
def test_dichroic_splits_at_its_wavelength(dichroic, crossover):
    blue_index, red_index = split_wavelengths(GRID, dichroic)
    assert GRID[blue_index].max() < crossover
    assert GRID[red_index].min() >= crossover
    # The two sides partition the grid exactly.
    assert blue_index.size + red_index.size == GRID.size
    assert not set(blue_index) & set(red_index)


def test_d55_gives_the_blue_side_more_of_the_grid_than_d46():
    blue_46, _ = split_wavelengths(GRID, "d46")
    blue_55, _ = split_wavelengths(GRID, "d55")
    assert blue_55.size > blue_46.size


def test_split_rejects_an_unknown_dichroic():
    with pytest.raises(ValueError, match="dichroic"):
        split_wavelengths(GRID, "d99")


# --- throughput --------------------------------------------------------------


def test_sensitivity_files_cover_plausible_ranges():
    assert sensitivity_range(BLUE_SENS_FILES["G2"]) == pytest.approx(
        (3221.3, 5241.0), abs=0.1
    )
    assert sensitivity_range(BLUE_SENS_FILES["G3"]) == pytest.approx(
        (3169.3, 4429.4), abs=0.1
    )
    assert sensitivity_range(RED_SENS_FILES["600/7500"]) == pytest.approx(
        (5135.8, 7632.2), abs=0.1
    )


def test_throughput_is_a_fraction_not_a_percentage():
    # Unlike the APF file, EFF in the Kast files is already 0-1, so there is no
    # division by 100.
    blue, red = kast_spectrograph()
    thru = kast_thruput(GRID, blue, red)
    assert thru.max() < 0.5
    assert thru.max() > 0.1


def test_throughput_is_floored_never_negative():
    # kast_thruput.pro extrapolates past the red end of the measurement, which
    # runs a falling curve negative; the floor is what keeps counts sensible.
    blue, red = kast_spectrograph(dichroic="d46")
    thru = kast_thruput(GRID, blue, red)
    assert thru.min() >= MIN_THRUPUT
    assert np.all(thru > 0)


def test_throughput_holds_flat_below_the_measured_range():
    blue, red = kast_spectrograph(dichroic="d46", grism="G3")
    # G3's measurement starts at 3169 A; below that the first value is held.
    wave = np.array([3000.0, 3100.0, 3169.3])
    thru = kast_thruput(wave, blue, red)
    assert thru[0] == pytest.approx(thru[1])
    assert thru[0] == pytest.approx(thru[2], rel=1e-3)


def test_grism_without_a_measurement_is_reported_clearly():
    # G1 is defined in x_initkast with R = 2344 but has no sensitivity file, so
    # kast_thruput.pro hit its `else: stop`.
    assert "G1" in GRISMS
    assert "G1" not in BLUE_SENS_FILES
    blue, red = kast_spectrograph(grism="G1")
    with pytest.raises(ParameterError, match="No throughput measurement"):
        kast_thruput(GRID, blue, red)


# --- the engine over two sides ----------------------------------------------


def test_both_sides_produce_finite_results():
    tel, sides = sides_for()
    result = run_sides(GRID, tel, sides, Observation(seeing=1.5, mstar=19.0, mtype=2))
    assert np.all(np.isfinite(result.sn))
    assert np.all(result.sn > 0)
    assert result.wave.size == GRID.size


def test_stacked_arrays_are_piecewise_constant_across_the_split():
    tel, sides = sides_for()
    result = run_sides(GRID, tel, sides, Observation(seeing=1.5, mstar=19.0, mtype=2))
    # Read noise differs per detector: 3.7 vs 3.8 electrons before extraction.
    assert len(set(np.round(result.noise, 6))) == 2
    blue = result.side("blue")
    red = result.side("red")
    assert red.noise / blue.noise == pytest.approx(3.8 / 3.7)


def test_resolution_differs_between_the_sides():
    tel, sides = sides_for()
    result = run_sides(GRID, tel, sides, Observation(seeing=1.5, mstar=19.0))
    assert result.side("blue").R == GRISMS["G2"]
    assert result.side("red").R == GRATINGS["600/7500"]
    # Angstroms per pixel scales as wave/R, so the lower-R red side is coarser
    # at its own wavelengths.
    assert result.side("red").pixel.mean() > result.side("blue").pixel.mean()


def test_side_lookup_rejects_an_unknown_name():
    tel, sides = sides_for()
    result = run_sides(GRID, tel, sides, Observation(seeing=1.5))
    with pytest.raises(KeyError, match="green"):
        result.side("green")


def test_the_long_slit_always_leaves_sky_rows():
    # The 120 arcsec slit means Kast never hits the negative-nsky case that
    # affects the APF short deckers.
    tel, sides = sides_for()
    result = run_sides(GRID, tel, sides, Observation(seeing=3.0, mstar=19.0))
    for _, _, side in result.sides:
        assert side.nsky > 0


def test_a_grid_entirely_on_one_side_still_works():
    # Only red wavelengths: the blue side contributes no points.
    wave = np.arange(6000.0, 7000.0, 10.0)
    tel = lick_telescope()
    blue, red = kast_spectrograph(dichroic="d46")
    thru = kast_thruput(wave, blue, red)
    blue_index, red_index = split_wavelengths(wave, "d46")
    assert blue_index.size == 0
    result = run_sides(
        wave,
        tel,
        [
            Side("blue", blue, blue_index, thru[blue_index]),
            Side("red", red, red_index, thru[red_index]),
        ],
        Observation(seeing=1.5, mstar=19.0),
    )
    assert [name for name, _, _ in result.sides] == ["red"]
    assert np.all(np.isfinite(result.sn))


def test_no_detector_at_all_is_an_error():
    tel = lick_telescope()
    blue, red = kast_spectrograph()
    empty = np.array([], dtype=int)
    with pytest.raises(ValueError, match="no detector"):
        run_sides(
            GRID,
            tel,
            [Side("blue", blue, empty, np.array([]))],
            Observation(seeing=1.5),
        )


# --- backend -----------------------------------------------------------------


def test_backend_sides_returns_two_detectors():
    backend = KastBackend()
    values = {"bins": 1, "bind": 1, "slitwidth": "1.5", "dichroic": "d55"}
    tel, sides = backend.sides(GRID, values)
    assert tel.name == "Lick-3m"
    assert [side.name for side in sides] == ["blue", "red"]


def test_backend_slitwidth_is_a_width_not_a_decker():
    backend = KastBackend()
    tel, sides = backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "2.0"})
    assert sides[0].instr.swidth == 2.0
    with pytest.raises(ParameterError, match="Slitwidth"):
        backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "W"})
    with pytest.raises(ParameterError, match="Slitwidth"):
        backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "-1"})


@pytest.mark.parametrize(
    "override,label",
    [
        ({"grism": "G9"}, "Grism"),
        ({"grating": "1200/5000"}, "Grating"),
        ({"dichroic": "d99"}, "Dichroic"),
    ],
)
def test_backend_rejects_unknown_dispersers(override, label):
    backend = KastBackend()
    values = {"bins": 1, "bind": 1, "slitwidth": "1.5", **override}
    with pytest.raises(ParameterError, match=label):
        backend.sides(GRID, values)


def test_backend_has_no_instrument_specific_extras():
    # Only APF has iodine-cell quantities.
    backend = KastBackend()
    tel, sides = backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "1.5"})
    result = run_sides(GRID, tel, sides, Observation(seeing=1.5, mstar=19.0))
    assert backend.extras(result, Observation()) is None


def test_every_configuration_is_finite():
    checked = 0
    for grism, dichroic, binning, seeing in itertools.product(
        sorted(BLUE_SENS_FILES), sorted(DICHROICS), ["1x1", "2x1", "2x2"], [0.7, 1.5]
    ):
        bins, bind = (int(x) for x in binning.split("x"))
        tel, sides = sides_for(
            dichroic=dichroic, grism=grism, bins=bins, bind=bind
        )
        result = run_sides(
            GRID, tel, sides, Observation(seeing=seeing, mstar=19.0, mtype=2)
        )
        label = f"{grism} {dichroic} {binning} {seeing}"
        assert np.all(np.isfinite(result.sn)), label
        assert np.all(result.star > 0), label
        checked += 1
    assert checked == 2 * 2 * 3 * 2
