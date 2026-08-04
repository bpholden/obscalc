import pytest

from obscalc.slit import _FLUX, _PSF, gauss_slit


def test_psf_table_length_covers_the_interpolation():
    # gauss_slit interpolates psf[irad] and psf[irad+1] for radius < 99, so
    # index 99 must exist.
    assert len(_PSF) == 100
    assert _PSF[0] == 1.0


def test_flux_grid_is_zero_outside_the_tabulated_profile():
    # Corner cells are at radius sqrt(2)*99 > 99 and must contribute nothing.
    assert _FLUX[0, 0] == 0.0
    assert _FLUX.shape == (199, 199)


def test_a_slit_much_wider_than_the_seeing_passes_everything():
    assert gauss_slit(20.0, 20.0) == pytest.approx(1.0)


def test_a_slit_much_narrower_than_the_seeing_passes_almost_nothing():
    assert gauss_slit(0.01, 0.01) < 0.001


def test_slit_fraction_increases_with_width():
    fractions = [gauss_slit(w, 3.0) for w in (0.25, 0.5, 1.0, 2.0)]
    assert fractions == sorted(fractions)
    assert all(0.0 < f <= 1.0 for f in fractions)


def test_offsetting_the_object_loses_light():
    centred = gauss_slit(1.0, 1.0, 0.0, 0.0)
    offset = gauss_slit(1.0, 1.0, 0.5, 0.0)
    assert offset < centred


def test_apf_w_decker_at_nominal_seeing():
    # 1.0 x 3.0 arcsec decker at 1.2 arcsec seeing.
    assert gauss_slit(1.0 / 1.2, 3.0 / 1.2) == pytest.approx(0.540476, abs=1e-6)
