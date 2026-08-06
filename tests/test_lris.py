"""LRIS: two channels like Kast, but the throughput file follows the dichroic."""

import itertools

import numpy as np
import pytest

from obscalc.instruments.base import ParameterError
from obscalc.instruments.lris import (
    BLUE_SENS_FILES,
    DEFAULT_RANGE,
    DICHROICS,
    GRATINGS,
    GRISMS,
    MIN_THRUPUT,
    RED_SENS_FILES,
    UNUSABLE_SENS_FILES,
    LrisBackend,
    _sensitivity,
    lris_spectrograph,
    lris_thruput,
    sens_files,
    sensitivity_range,
    split_wavelengths,
    unmeasured_ranges,
)
from obscalc.s2n import Side, run_sides
from obscalc.sky import sky_for
from obscalc.structures import Observation
from obscalc.telescopes import keck_telescope

GRID = np.arange(*DEFAULT_RANGE, 10.0)


def sides_for(grism="B600", grating="600/7500", dichroic="D560", slit=1.0, bins=1, bind=1):
    tel = keck_telescope("KeckI")
    blue, red = lris_spectrograph(
        grism=grism,
        grating=grating,
        dichroic=dichroic,
        slit=slit,
        bins=bins,
        bind=bind,
        str_tel=tel,
    )
    thru = lris_thruput(GRID, blue, red)
    blue_index, red_index = split_wavelengths(GRID, dichroic)
    return tel, [
        Side("blue", blue, blue_index, thru[blue_index]),
        Side("red", red, red_index, thru[red_index]),
    ]


# --- telescope and instrument ------------------------------------------------


def test_lris_is_on_keck_one():
    """x_initlris.pro sets str_tel.name = 'KeckI' after calling x_initkeck."""
    blue, red = lris_spectrograph()
    tel = keck_telescope("KeckI")
    assert tel.name == "KeckI"
    assert tel.area == 723674.0
    assert blue.name == "LRIS-blue" and red.name == "LRIS-red"


def test_pixel_scale_matches_the_comment_in_the_idl():
    """x_initlris.pro says MAG_PERP = 6.5 was "Modified to give 0.135" pixels"."""
    blue, red = lris_spectrograph()
    for instr in (blue, red):
        assert instr.mag_perp == 6.5
        assert instr.pixel_size == 15.0
        assert instr.scale_perp == pytest.approx(0.1344, abs=5e-4)
        assert instr.scale_para == pytest.approx(instr.scale_perp)


def test_read_noise_differs_per_detector():
    """x_initlris.pro meant 3.7 blue and 4.5 red but clobbered both.

    ``lrisinstr.readno = 3.7`` then ``lrisinstr.readno = 4.5`` on a two-element
    array leaves 4.5 on each side.  Same mistake as x_initkast.pro.
    """
    blue, red = lris_spectrograph()
    assert blue.readno == 3.7
    assert red.readno == 4.5
    assert blue.dark == red.dark == 0.001


def test_resolving_power_follows_the_disperser():
    blue, red = lris_spectrograph(grism="B300", grating="1200/9000")
    assert blue.R == 3304.0
    assert red.R == 23640.0


def test_the_two_600_gratings_share_a_resolving_power():
    """Same ruling, different blaze, so x_initlris gives both R = 11820."""
    assert GRATINGS["600/7500"] == GRATINGS["600/10000"] == 11820.0


def test_slit_height_is_the_long_slit():
    blue, red = lris_spectrograph(slit=1.5)
    for instr in (blue, red):
        assert instr.swidth == 1.5
        assert instr.sheight == 120.0


def test_unknown_configurations_are_rejected():
    with pytest.raises(ValueError, match="grism"):
        lris_spectrograph(grism="B1200")
    with pytest.raises(ValueError, match="grating"):
        lris_spectrograph(grating="150/7500")
    with pytest.raises(ValueError, match="dichroic"):
        lris_spectrograph(dichroic="d46")


# --- the dichroic split ------------------------------------------------------


def test_only_the_d560_dichroic_is_offered():
    """lris_thruput.pro stopped on anything else; its d46 entry is commented out."""
    assert set(DICHROICS) == {"D560"}
    assert DICHROICS["D560"] == 5600.0


def test_dichroic_splits_at_its_wavelength():
    blue_index, red_index = split_wavelengths(GRID, "D560")
    assert GRID[blue_index].max() < 5600.0
    assert GRID[red_index].min() >= 5600.0
    assert blue_index.size + red_index.size == GRID.size


def test_split_rejects_an_unknown_dichroic():
    with pytest.raises(ValueError, match="unknown LRIS dichroic"):
        split_wavelengths(GRID, "D680")


# --- throughput --------------------------------------------------------------


def test_sensitivity_files_are_hdu_two_vector_columns():
    """Same layout as the Kast files, unlike the DEIMOS ones in HDU 1."""
    for name in list(BLUE_SENS_FILES.values()) + list(RED_SENS_FILES.values()):
        wave, eff = _sensitivity(name)
        assert wave.size == eff.size == 999
        assert wave[0] < wave[-1]


def test_throughput_is_a_fraction_not_a_percentage():
    for name in list(BLUE_SENS_FILES.values()) + list(RED_SENS_FILES.values()):
        _, eff = _sensitivity(name)
        assert 0.0 <= eff.min()
        assert eff.max() < 0.5


def test_every_configuration_has_a_measurement_for_d560():
    for grism, grating in itertools.product(GRISMS, GRATINGS):
        blue, red = lris_spectrograph(grism=grism, grating=grating)
        blue_file, red_file = sens_files(blue, red)
        assert blue_file in BLUE_SENS_FILES.values()
        assert red_file in RED_SENS_FILES.values()


def test_throughput_is_floored_and_finite_everywhere():
    for grism, grating in itertools.product(GRISMS, GRATINGS):
        blue, red = lris_spectrograph(grism=grism, grating=grating)
        thru = lris_thruput(GRID, blue, red)
        assert np.all(np.isfinite(thru))
        assert np.all(thru >= MIN_THRUPUT)
        assert np.all(thru < 1.0)


def test_both_ends_are_held_not_extrapolated():
    """The divergence from lris_thruput.pro, which held only the blue end.

    600/7500 stops at 8191 A.  Continuing its last interval to 10000 A gives 0.415,
    above every LRIS measurement on either side; holding gives the 0.125 measured
    at the red end of the curve.
    """
    blue, red = lris_spectrograph(grating="600/7500")
    _, red_file = sens_files(blue, red)
    sens_wave, sens_eff = _sensitivity(red_file)

    thru = lris_thruput(np.array([sens_wave[-1] + 1000.0]), blue, red)
    assert thru[0] == pytest.approx(sens_eff[-1])
    assert thru[0] < 0.15

    below = lris_thruput(np.array([5600.0]), blue, red)
    assert below[0] == pytest.approx(sens_eff[0])


def test_the_idl_extrapolation_would_have_exceeded_every_measurement():
    """Why the red end is held: the extrapolation rises rather than falling.

    kast_thruput.pro's floor catches a curve going negative.  Here the worst case
    goes up instead, so the floor never sees it.
    """
    from obscalc.idl_compat import interpol

    best_red = max(_sensitivity(name)[1].max() for name in RED_SENS_FILES.values())
    sens_wave, sens_eff = _sensitivity(RED_SENS_FILES["600/7500"])
    extrapolated = interpol(sens_eff, sens_wave, np.array([10000.0]))[0]

    assert extrapolated == pytest.approx(0.415, abs=0.01)
    # Above every red-side measurement, and 3.3x its own curve's last value.
    assert extrapolated > best_red == pytest.approx(0.378, abs=0.01)
    assert extrapolated > 3.0 * sens_eff[-1]
    assert extrapolated > sens_eff.max()


def test_unmeasured_ranges_only_counts_wavelengths_a_side_records():
    """The blue curve stopping at 5596 A does not matter: the red side has it."""
    blue, red = lris_spectrograph(grism="B600", grating="600/7500")
    held = unmeasured_ranges(GRID, blue, red)
    assert all(name == "red" for name, _, _ in held)
    spans = [(lo, hi) for _, lo, hi in held]
    assert (8200.0, 9990.0) in spans


def test_a_fully_covering_configuration_holds_nothing():
    """B300 is measured 2148-7652 and 400/8500 to 10354, so nothing is held."""
    blue, red = lris_spectrograph(grism="B300", grating="400/8500")
    assert unmeasured_ranges(GRID, blue, red) == []


def test_sensitivity_ranges_are_plausible():
    low, high = sensitivity_range(BLUE_SENS_FILES["B600"])
    assert (low, high) == pytest.approx((3101.2, 5595.9), abs=0.1)
    low, high = sensitivity_range(RED_SENS_FILES["600/7500"])
    assert (low, high) == pytest.approx((5628.3, 8190.7), abs=0.1)


def test_the_d680_measurement_is_bundled_but_unusable():
    """A real B300 measurement behind D680 with no red-side counterpart.

    lris_thruput.pro names it inside a case nested in the dichroic case that
    stops unless the dichroic is D560, so it was unreachable there too.
    """
    assert UNUSABLE_SENS_FILES == {("B300", "D680"): "sens_LRISb_300_5000_D680.fits"}
    wave, eff = _sensitivity(UNUSABLE_SENS_FILES[("B300", "D680")])
    assert wave.size == 999 and eff.max() < 0.5
    assert not any("D680" in name for name in RED_SENS_FILES.values())


# --- the engine over both sides ----------------------------------------------


def test_both_sides_produce_finite_results():
    tel, sides = sides_for()
    obs = Observation(seeing=1.0, mstar=20.0, mtype=2, exptime=3600.0)
    result = run_sides(GRID, tel, sides, obs)
    assert np.all(np.isfinite(result.sn))
    assert np.all(result.sn > 0)


def test_stacked_arrays_are_piecewise_constant_across_the_split():
    tel, sides = sides_for()
    obs = Observation(seeing=1.0, mstar=20.0, mtype=2, exptime=3600.0)
    result = run_sides(GRID, tel, sides, obs)

    blue_index, red_index = split_wavelengths(GRID, "D560")
    assert np.allclose(result.noise[blue_index], result.noise[blue_index][0])
    assert np.allclose(result.noise[red_index], result.noise[red_index][0])
    # 3.7 against 4.5 electrons, scaled the same way by the extraction height.
    assert result.noise[blue_index][0] < result.noise[red_index][0]


def test_the_long_slit_always_leaves_sky_rows():
    """120 arcsec of slit against a seeing-sized extraction: nsky stays positive."""
    for seeing in (0.6, 1.0, 1.5, 2.0):
        tel, sides = sides_for()
        obs = Observation(seeing=seeing, mstar=20.0, mtype=2, exptime=3600.0)
        result = run_sides(GRID, tel, sides, obs)
        for _, _, side in result.sides:
            assert side.nsky > 0


def test_lris_takes_the_combined_mauna_kea_sky():
    """spec_calcs2n.pro could not reach its own LRIS sky branch.

    The 'LRIS' case selecting flg_sky = 2 sits inside the 'KeckII' branch, but
    x_initlris.pro sets the telescope to 'KeckI', so LRIS took the KeckI line --
    maunakea_sky(wave, phase, /NOEMPIR), which the NOEMPIRI/NOEMPIRIC typo turned
    back into the empirical default, the DEIMOS 600 model.  Here it gets the
    combined model, which covers the blue half LRIS actually uses.
    """
    assert sky_for("KeckI", "LRIS-blue").model == "combined"
    assert sky_for("KeckI", "LRIS-red").model == "combined"


# --- the web backend ---------------------------------------------------------


def test_backend_sides_returns_two_detectors():
    tel, sides = LrisBackend().sides(
        GRID, {"bins": 1, "bind": 1, "slitwidth": "1.0"}
    )
    assert tel.name == "KeckI"
    assert [side.name for side in sides] == ["blue", "red"]


def test_backend_slitwidth_is_a_width_not_a_decker():
    backend = LrisBackend()
    _, sides = backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "1.5"})
    assert sides[0].instr.swidth == 1.5
    with pytest.raises(ParameterError, match="Slitwidth"):
        backend.sides(GRID, {"bins": 1, "bind": 1, "slitwidth": "C5"})


@pytest.mark.parametrize(
    "override, label",
    [
        ({"grism": "B1200"}, "Grism"),
        ({"grating": "150/7500"}, "Grating"),
        ({"dichroic": "d46"}, "Dichroic"),
    ],
)
def test_backend_rejects_unknown_dispersers(override, label):
    values = {"bins": 1, "bind": 1, "slitwidth": "1.0", **override}
    with pytest.raises(ParameterError, match=label):
        LrisBackend().sides(GRID, values)


def test_backend_has_no_instrument_specific_extras():
    assert LrisBackend().extras(None, None) is None


# --- sweep -------------------------------------------------------------------


def test_every_configuration_is_finite():
    obs = Observation(seeing=1.0, mstar=20.0, mtype=2, exptime=3600.0)
    for grism, grating in itertools.product(GRISMS, GRATINGS):
        tel, sides = sides_for(grism=grism, grating=grating)
        result = run_sides(GRID, tel, sides, obs)
        assert np.all(np.isfinite(result.sn)), (grism, grating)
        assert np.all(result.sn > 0), (grism, grating)
