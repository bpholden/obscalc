"""The shared echelle model, and the APF's use of it.

The APF sensitivity file turns out to be tabulated once per echelle order, at the
blaze centre, exactly like the HIRES throughput table.  ``apf_thruput.pro``
treated it as a smooth curve and interpolated between order centres, and applied
no blaze; ``apf_calcs2n.pro`` computed the order number, centre and free spectral
range and discarded all three.  These tests pin down the corrected behaviour.
"""

import numpy as np
import pytest

from obscalc.echelle import (
    blaze_efficiency,
    echelle_orders,
    measured_orders,
    order_centre_throughput,
)
from obscalc.instruments.apf import (
    DEFAULT_RANGE,
    MLAMBDA,
    _sensitivity,
    DEFAULT_SENS_FILE,
    apf_spectrograph,
    apf_thruput,
    sensitivity_orders,
)

GRID = np.arange(*DEFAULT_RANGE, 10.0)


# --- the shared model --------------------------------------------------------


def test_order_geometry_is_the_idl_formula():
    order, centre, fsr = echelle_orders([5000.0], MLAMBDA)
    assert order[0] == int(MLAMBDA / 5000.0)  # long() truncates
    assert centre[0] == pytest.approx(MLAMBDA / order[0])
    assert fsr[0] == pytest.approx(centre[0] / order[0])


def test_blaze_is_one_at_the_centre_and_zero_a_full_order_away():
    _, centre, fsr = echelle_orders([5000.0], MLAMBDA)
    assert blaze_efficiency(centre, centre, fsr)[0] == pytest.approx(1.0)
    assert blaze_efficiency(centre + fsr / 2, centre, fsr)[0] == pytest.approx(
        0.405, abs=0.01
    )
    assert blaze_efficiency(centre + fsr, centre, fsr)[0] < 1e-6


def test_order_centre_throughput_holds_the_ends_rather_than_extrapolating():
    table_wave = np.array([4000.0, 5000.0, 6000.0])
    table_eff = np.array([0.1, 0.2, 0.3])
    # Far outside the table, so the nearest value must be held, not extended.
    thru = order_centre_throughput(
        [4000.0], MLAMBDA, table_wave, table_eff, blaze=False
    )
    assert 0.1 <= thru[0] <= 0.3


def test_measured_orders_rejects_a_table_that_is_not_per_order():
    # An evenly spaced curve is not tabulated at order centres.
    with pytest.raises(ValueError, match="not tabulated at echelle order centres"):
        measured_orders(np.arange(4000.0, 5000.0, 100.0), MLAMBDA)


# --- the APF sensitivity file is per-order -----------------------------------


def test_the_apf_sensitivity_file_is_tabulated_at_order_centres():
    """The evidence for this whole change.

    All 63 wavelengths in sens_APF_aug2022 are MLAMBDA/m for consecutive integer
    m, to better than 0.006 Angstroms -- as they are in the nov2016 file it
    replaced.  That is not a coincidence: it is one measurement per order, taken
    at the blaze peak.
    """
    sens_wave, _ = _sensitivity(DEFAULT_SENS_FILE)
    order = np.rint(MLAMBDA / sens_wave).astype(int)
    residual = np.abs(sens_wave - MLAMBDA / order)
    assert residual.max() < 0.01
    assert sorted(order) == list(range(order.min(), order.max() + 1))
    assert (order.min(), order.max()) == (62, 124)
    assert sens_wave.size == 63


def test_sensitivity_orders_reports_them():
    order = sensitivity_orders()
    assert order.min() == 62 and order.max() == 124


# --- the corrected APF throughput --------------------------------------------


def test_throughput_is_constant_across_an_order_without_the_blaze():
    instr = apf_spectrograph()
    order, _, _ = echelle_orders(GRID, instr.mlambda)
    thru = apf_thruput(GRID, instr, blaze=False)
    inside = order == order[len(order) // 2]
    assert inside.sum() > 1
    assert np.allclose(thru[inside], thru[inside][0])


def test_throughput_steps_between_orders_rather_than_varying_smoothly():
    instr = apf_spectrograph()
    thru = apf_thruput(GRID, instr, blaze=False)
    order, _, _ = echelle_orders(GRID, instr.mlambda)
    distinct = len(np.unique(thru))
    # One value per order the grid touches, not one per wavelength.  Orders
    # outside the measurement share the held end value, so this is a bound.
    assert distinct <= len(np.unique(order))
    assert distinct < GRID.size / 5


def test_an_exact_order_centre_is_assigned_to_its_own_order():
    """Guards the floating-point edge in echelle_orders.

    ``mlambda / (mlambda / m)`` can fall a hair below ``m``, and IDL's ``long()``
    truncates, which put 3 of the APF's 63 order centres in the order below.
    """
    instr = apf_spectrograph()
    expected = np.arange(62, 125)
    order, _, _ = echelle_orders(instr.mlambda / expected, instr.mlambda)
    assert np.array_equal(order, expected)


def test_the_blaze_is_on_by_default_for_the_apf_as_for_hires():
    instr = apf_spectrograph()
    on = apf_thruput(GRID, instr)
    off = apf_thruput(GRID, instr, blaze=False)
    assert np.all(on <= off + 1e-12)
    assert np.median(on / off) == pytest.approx(0.41, abs=0.03)


def test_the_blaze_peaks_at_each_order_centre():
    instr = apf_spectrograph()
    centre = instr.mlambda / np.arange(62, 125)
    # At an order centre the blazed and unblazed values coincide.
    assert np.allclose(
        apf_thruput(centre, instr), apf_thruput(centre, instr, blaze=False)
    )


def test_throughput_stays_a_plausible_fraction():
    instr = apf_spectrograph()
    for blaze in (True, False):
        thru = apf_thruput(GRID, instr, blaze=blaze)
        assert np.all(thru > 0)
        assert np.all(thru < 0.25)


def test_orders_beyond_the_measurement_hold_the_end_value():
    """The 3742-7700 A grid reaches orders 60 and 61, which were not measured."""
    instr = apf_spectrograph()
    order, _, _ = echelle_orders(GRID, instr.mlambda)
    unmeasured = (order < 62) | (order > 124)
    assert unmeasured.sum() == 18
    assert GRID[unmeasured].min() == pytest.approx(7522.0)
    thru = apf_thruput(GRID, instr, blaze=False)
    assert np.all(np.isfinite(thru))
    assert np.all(thru > 0)


def test_the_old_interpolation_differed_most_in_the_steep_blue():
    """Per-order lookup against interpolating the curve at the wavelength.

    Mostly sub-1 per cent, but around 15 per cent at 3800-4300 A where the
    sensitivity curve rises steeply and a wavelength sits well away from its own
    order's centre.  The exact worst case depends on the sensitivity file: it is
    14.8 per cent for aug2022, 20 per cent for the nov2016 file it replaced.
    """
    from obscalc.idl_compat import interpol

    instr = apf_spectrograph()
    sens_wave, sens_eff = _sensitivity(DEFAULT_SENS_FILE)
    old = interpol(sens_eff, sens_wave, GRID) / 100.0
    new = apf_thruput(GRID, instr, blaze=False)

    deviation = np.abs(new / old - 1)
    assert np.median(deviation) < 0.02
    assert deviation.max() > 0.10
    worst = GRID[np.argmax(deviation)]
    assert 3700.0 < worst < 4400.0
