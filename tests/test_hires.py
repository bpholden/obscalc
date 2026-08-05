"""HIRES: a cross-dispersed echelle, so throughput is modelled, not measured."""

import itertools

import numpy as np
import pytest

from obscalc.instruments.base import ParameterError
from obscalc.instruments.hires import (
    CCD_BOOST_RANGE,
    CROSS_DISPERSER_BREAK,
    DECKERS,
    DEFAULT_RANGE,
    EPOCHS,
    MLAMBDA,
    HiresBackend,
    _THRU_BY_ORDER,
    blaze_efficiency,
    ccd_boost,
    cross_disperser_order,
    echelle_orders,
    hires_spectrograph,
    hires_thruput,
)
from obscalc.s2n import Side, run_sides
from obscalc.structures import Observation
from obscalc.telescopes import keck_telescope

GRID = np.arange(*DEFAULT_RANGE, 10.0)


def result_for(**kwargs):
    tel = keck_telescope("KeckI")
    blaze = kwargs.pop("blaze", False)
    obs_kwargs = {"seeing": 0.7, "mstar": 15.0, "mtype": 2, "exptime": 1800.0}
    obs_kwargs.update(kwargs.pop("obs", {}))
    instr = hires_spectrograph(**kwargs, str_tel=tel)
    thru = hires_thruput(GRID, instr, blaze=blaze)
    return run_sides(
        GRID,
        tel,
        [Side("", instr, np.arange(GRID.size), thru)],
        Observation(**obs_kwargs),
    )


# --- instrument --------------------------------------------------------------


def test_hires_sits_on_keck_one():
    instr = hires_spectrograph()
    assert instr.name == "HIRES"
    assert instr.mlambda == MLAMBDA
    # KeckI, so maunakea_trans and maunakea_sky apply.
    assert keck_telescope("KeckI").name == "KeckI"


def test_epochs_differ_in_detector_and_resolution():
    old = hires_spectrograph(epoch="old")
    new = hires_spectrograph(epoch="new")
    assert (old.pixel_size, old.readno, old.dark) == (24.0, 4.3, 2.0)
    assert (new.pixel_size, new.readno, new.dark) == (15.0, 2.2, 1.0)
    # R scales inversely with pixel size from 135000 at 24 microns.
    assert old.R == pytest.approx(135000.0)
    assert new.R == pytest.approx(135000.0 * 24.0 / 15.0)
    assert new.R > old.R


def test_spatial_binning_defaults_to_two():
    # x_inithires.pro sets bins = 2, unlike the other instruments here.
    assert hires_spectrograph().bins == 2
    assert hires_spectrograph().bind == 1


def test_all_eight_deckers_are_available():
    assert set(DECKERS) == {"B2", "B5", "C1", "C5", "D1", "D3", "E4", "E5"}
    for decker, (width, height) in DECKERS.items():
        instr = hires_spectrograph(decker=decker)
        assert (instr.swidth, instr.sheight) == (width, height), decker


def test_the_default_decker_uses_the_table_value():
    """x_inithires.pro's no-decker fallback called it C5 but set 1.1, not 1.148."""
    assert hires_spectrograph().swidth == pytest.approx(DECKERS["C5"][0])
    assert hires_spectrograph().swidth == pytest.approx(1.148)


def test_unknown_configurations_are_rejected():
    with pytest.raises(ValueError, match="decker"):
        hires_spectrograph(decker="Z9")
    with pytest.raises(ValueError, match="epoch"):
        hires_spectrograph(epoch="ancient")


# --- echelle format ----------------------------------------------------------


def test_echelle_order_geometry():
    """m = long(MLAMBDA/wave), centre = MLAMBDA/m, fsr = centre/m."""
    order, centre, fsr = echelle_orders([5000.0])
    assert order[0] == 71
    assert centre[0] == pytest.approx(MLAMBDA / 71)
    assert fsr[0] == pytest.approx(centre[0] / 71)


def test_orders_get_wider_and_lower_numbered_toward_the_red():
    order, _, fsr = echelle_orders([3000.0, 5000.0, 9500.0])
    assert list(order) == sorted(order, reverse=True)
    assert list(fsr) == sorted(fsr)
    assert order[0] == 118 and order[-1] == 37
    assert fsr[0] == pytest.approx(25.6, abs=0.1)
    assert fsr[-1] == pytest.approx(260.3, abs=0.1)


def test_the_order_centre_is_within_half_a_free_spectral_range():
    _, centre, fsr = echelle_orders(GRID)
    assert np.all(np.abs(centre - GRID) <= fsr)


def test_wavelengths_outside_the_echelle_format_are_rejected():
    with pytest.raises(ValueError, match="echelle format"):
        echelle_orders([MLAMBDA * 2])
    with pytest.raises(ValueError, match="positive"):
        echelle_orders([0.0])


def test_cross_disperser_switches_at_3800_angstroms():
    assert cross_disperser_order([CROSS_DISPERSER_BREAK - 1.0])[0] == 2
    assert cross_disperser_order([CROSS_DISPERSER_BREAK + 1.0])[0] == 1
    # The blue setting is much more efficient in the near UV, the red one is not.
    assert _THRU_BY_ORDER[2][0] > _THRU_BY_ORDER[1][0]
    assert _THRU_BY_ORDER[1][-1] > _THRU_BY_ORDER[2][-1]


# --- blaze -------------------------------------------------------------------


def test_blaze_peaks_at_the_order_centre():
    _, centre, fsr = echelle_orders([5000.0])
    assert blaze_efficiency(centre, centre, fsr)[0] == pytest.approx(1.0)


def test_blaze_falls_toward_the_order_edge():
    _, centre, fsr = echelle_orders([5000.0])
    half = blaze_efficiency(centre + fsr / 2, centre, fsr)[0]
    assert half == pytest.approx(0.405, abs=0.01)
    edge = blaze_efficiency(centre + fsr, centre, fsr)[0]
    assert edge < 1e-6  # sinc^2 zero one free spectral range away


def test_blaze_is_symmetric_about_the_centre():
    _, centre, fsr = echelle_orders([5000.0])
    offset = fsr / 3
    assert blaze_efficiency(centre - offset, centre, fsr)[0] == pytest.approx(
        blaze_efficiency(centre + offset, centre, fsr)[0]
    )


def test_blaze_is_off_by_default():
    """hires_calcs2n.pro passed BLAZE=blaze with blaze undefined, so it never fired.

    That makes the reported throughput the peak value within each order.
    """
    instr = hires_spectrograph()
    off = hires_thruput(GRID, instr)
    on = hires_thruput(GRID, instr, blaze=True)
    assert np.all(on <= off + 1e-12)
    # Averaged over an order, sinc^2 comes to about 0.4.
    assert np.median(on / off) == pytest.approx(0.40, abs=0.03)


# --- throughput --------------------------------------------------------------


def test_throughput_is_constant_across_an_order():
    """The table is read at the order centre, not at the wavelength."""
    instr = hires_spectrograph(epoch="old")  # no detector boost to confuse it
    order, _, _ = echelle_orders(GRID)
    thru = hires_thruput(GRID, instr)
    inside = order == order[len(order) // 2]
    assert inside.sum() > 1
    assert np.allclose(thru[inside], thru[inside][0])


def test_throughput_steps_between_orders():
    instr = hires_spectrograph(epoch="old")
    thru = hires_thruput(GRID, instr)
    assert len(np.unique(thru)) > 20  # one value per order, not a smooth curve


def test_throughput_is_a_plausible_fraction():
    for epoch in EPOCHS:
        instr = hires_spectrograph(epoch=epoch)
        thru = hires_thruput(GRID, instr)
        assert np.all(thru > 0), epoch
        assert np.all(thru < 0.5), epoch


def test_the_newer_detector_raises_the_throughput():
    old = hires_thruput(GRID, hires_spectrograph(epoch="old"))
    new = hires_thruput(GRID, hires_spectrograph(epoch="new"))
    assert np.all(new >= old)
    assert np.median(new / old) > 1.0


def test_detector_boost_is_held_flat_outside_its_table():
    """A quantum efficiency ratio cannot be linearly extrapolated.

    hires_thru_newccd passed straight to interpol; continuing its first interval
    from 3153.9 A down to 3000 A would treble the boost to 32.8, turning a 0.3
    per cent throughput into 9.8 per cent.
    """
    low, high = CCD_BOOST_RANGE
    assert ccd_boost([low - 200.0])[0] == pytest.approx(ccd_boost([low])[0])
    assert ccd_boost([high + 500.0])[0] == pytest.approx(ccd_boost([high])[0])
    assert ccd_boost([3000.0])[0] == pytest.approx(11.21)


def test_epoch_can_be_overridden_independently_of_the_instrument():
    instr = hires_spectrograph(epoch="old")
    assert not np.allclose(
        hires_thruput(GRID, instr), hires_thruput(GRID, instr, epoch="new")
    )
    with pytest.raises(ValueError, match="epoch"):
        hires_thruput(GRID, instr, epoch="ancient")


# --- through the engine ------------------------------------------------------


def test_signal_to_noise_is_finite_across_the_range():
    result = result_for()
    assert np.all(np.isfinite(result.sn))
    assert np.all(result.sn > 0)
    assert np.all(result.star > 0)


def test_the_short_decker_would_have_produced_nan_in_the_idl():
    """E5 is 1 arcsec tall, so the extraction window exceeds it.

    hires_calcs2n.pro had no guard on nsky, exactly like spec_calcs2n.pro.
    """
    result = result_for(decker="E5")
    assert result.sides[0][2].nsky < 0
    assert np.all(np.isfinite(result.sn))


def test_every_decker_and_epoch_runs():
    for decker, epoch in itertools.product(sorted(DECKERS), sorted(EPOCHS)):
        result = result_for(decker=decker, epoch=epoch)
        assert np.all(np.isfinite(result.sn)), f"{decker} {epoch}"
        assert result.sides[0][2].slit0 > 0


def test_the_newer_detector_trades_signal_per_pixel_for_resolution():
    """Smaller pixels mean higher R, so fewer photons land in each pixel.

    The newer detector has 1.29x the throughput but 15 micron pixels against 24,
    lifting R from 135000 to 216000 and so covering 0.625x the wavelength per
    pixel.  Per pixel that is a net loss of about 11 per cent; per resolution
    element it is a 13 per cent gain.  Comparing per-pixel S/N between the two
    epochs is therefore misleading on its own.
    """
    old = result_for(epoch="old")
    new = result_for(epoch="new")

    assert np.median(new.thru) > np.median(old.thru)
    assert np.median(new.pixel) < np.median(old.pixel)

    assert np.median(new.sn) < np.median(old.sn)
    assert np.median(new.sn_per_resolution_element) > np.median(
        old.sn_per_resolution_element
    )
    # Counts per unit wavelength, which is the fair comparison, favours the new one.
    assert np.median(new.star / new.pixel) > np.median(old.star / old.pixel)


def test_applying_the_blaze_lowers_signal_to_noise():
    assert np.median(result_for(blaze=True).sn) < np.median(result_for().sn)


def test_resolving_power_reaches_the_table_meta():
    result = result_for()
    assert result.sides[0][2].R == pytest.approx(216000.0)


def test_a_template_run_succeeds():
    result = result_for(
        obs={
            "seeing": 0.7,
            "mstar": 12.0,
            "mtype": 1,
            "exptime": 1800.0,
            "template": "G5V_pickles_27.fits",
            "filter": "Buser_V.dat",
        }
    )
    assert np.all(np.isfinite(result.sn))


# --- backend -----------------------------------------------------------------


def test_backend_returns_one_side():
    backend = HiresBackend()
    tel, sides = backend.sides(GRID, {"bins": 2, "bind": 1, "slitwidth": "C5"})
    assert tel.name == "KeckI"
    assert len(sides) == 1
    assert sides[0].instr.swidth == pytest.approx(1.148)


def test_backend_slitwidth_is_a_decker_name():
    backend = HiresBackend()
    with pytest.raises(ParameterError, match="Decker"):
        backend.sides(GRID, {"bins": 2, "bind": 1, "slitwidth": "1.0"})


def test_backend_rejects_an_unknown_epoch():
    backend = HiresBackend()
    with pytest.raises(ParameterError, match="Epoch"):
        backend.sides(
            GRID, {"bins": 2, "bind": 1, "slitwidth": "C5", "epoch": "ancient"}
        )


def test_backend_accepts_a_blaze_flag():
    backend = HiresBackend()
    values = {"bins": 2, "bind": 1, "slitwidth": "C5"}
    _, plain = backend.sides(GRID, values)
    _, blazed = backend.sides(GRID, {**values, "blaze": "true"})
    assert np.all(blazed[0].thru <= plain[0].thru + 1e-12)
    assert np.median(blazed[0].thru) < np.median(plain[0].thru)


def test_backend_has_no_instrument_specific_extras():
    backend = HiresBackend()
    assert backend.extras(result_for(), Observation()) is None
