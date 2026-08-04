import numpy as np
import pytest

from obscalc.idl_compat import idl_long, interpol, linterp, tsum


def test_interpol_reproduces_nodes():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([10.0, 20.0, 40.0])
    assert np.allclose(interpol(y, x, x), y)


def test_interpol_extrapolates_linearly_off_both_ends():
    # np.interp would clamp to 10.0 and 40.0 here; IDL's interpol does not.
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([10.0, 20.0, 40.0])
    assert interpol(y, x, [0.0]) == pytest.approx(0.0)
    assert interpol(y, x, [5.0]) == pytest.approx(50.0)


def test_interpol_accepts_descending_abscissa():
    # The APF sensitivity file is stored in descending wavelength order.
    x = np.array([4.0, 2.0, 1.0])
    y = np.array([40.0, 20.0, 10.0])
    assert interpol(y, x, [3.0]) == pytest.approx(30.0)
    assert interpol(y, x, [5.0]) == pytest.approx(50.0)


def test_interpol_spline_passes_through_nodes():
    x = np.linspace(0.0, 10.0, 11)
    y = np.sin(x)
    assert np.allclose(interpol(y, x, x, spline=True), y)


def test_linterp_fills_instead_of_extrapolating():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    assert np.allclose(linterp(x, y, [0.0, 1.5, 4.0]), [0.0, 15.0, 0.0])
    assert linterp(x, y, [4.0], missing=-1.0) == pytest.approx(-1.0)


def test_tsum_trapezoid_and_index_subrange():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert tsum(x, y) == pytest.approx(4.5)
    # imin/imax are inclusive element indices, as in the IDL.
    assert tsum(x, y, 1, 2) == pytest.approx(1.5)


def test_idl_long_truncates_toward_zero():
    assert idl_long(2.7) == 2
    assert idl_long(-2.7) == -2  # np.floor would give -3
    assert np.array_equal(idl_long([0.999, 1.999]), [0, 1])


def test_idl_long_ceiling_idiom():
    # The xidl sources spell ceil(x) as long(x + 0.999); these are the row
    # counts that idiom produces for APF at 1.2" seeing with a 3" decker.
    assert idl_long(3 * 1.2 / 0.43350872976 + 0.999) == 9
    assert idl_long(3.0 / 0.43350872976 + 0.999) == 7
