import numpy as np
import pytest

from obscalc.photometry import (
    TemplateFilterMismatch,
    flux_ab,
    flux_johnson,
    generate_template,
    read_filter,
    read_template,
    spec_to_mag,
    template_magnitude,
    zero_point_flux,
)
from obscalc.structures import Observation


def test_johnson_flux_reproduces_its_table():
    assert flux_johnson([3062.0]) == pytest.approx(569.0)
    assert flux_johnson([5425.0]) == pytest.approx(1054.0)
    assert flux_johnson([11050.0]) == pytest.approx(263.0)


def test_johnson_flux_interpolates_midpoints():
    # Halfway between 5425/1054 and 5925/886.
    assert flux_johnson([5675.0]) == pytest.approx(970.0)


def test_ab_zero_point_matches_the_closed_form():
    wave = np.array([5000.0])
    assert flux_ab(wave) == pytest.approx(10.0 ** (-0.4 * 48.6) / 6.626e-27 / 5000.0)


def test_zero_point_flux_rejects_unknown_systems():
    with pytest.raises(ValueError, match="mtype"):
        zero_point_flux(3, np.array([5000.0]))


def test_filter_files_are_two_columns_of_wavelength_and_throughput():
    wave, thru = read_filter("Buser_V.dat")
    assert wave.shape == thru.shape
    assert wave.min() > 4000.0 and wave.max() < 8000.0
    assert thru.min() >= 0.0


def test_read_template_handles_one_row_per_wavelength():
    wave, flux = read_template("G5V_pickles_27.fits")
    assert wave.shape == (1895,) and flux.shape == (1895,)
    assert wave.min() == pytest.approx(1150.0)


def test_read_template_handles_vector_columns():
    # qso_template.fits is a single row of 7755-element vector columns, and
    # stores FLUX before WAVELENGTH.
    wave, flux = read_template("qso_template.fits")
    assert wave.shape == (7755,) and flux.shape == (7755,)
    assert wave.min() == pytest.approx(800.5)
    assert wave.max() == pytest.approx(8554.5)


def test_vega_is_zero_magnitude_in_the_vega_system():
    obs = Observation(
        template="alpha_lyr_stis_005.fits", filter="Buser_V.dat", mtype=1
    )
    assert template_magnitude(obs) == pytest.approx(0.0, abs=1e-9)


def test_vega_ab_magnitude_in_v_is_near_zero():
    # V_AB = V_Vega by construction, and Vega has V ~ 0.03.
    wave, flux = read_template("alpha_lyr_stis_005.fits")
    mag, _, frac = spec_to_mag(wave, flux * 1e17, "Buser_V.dat")
    assert frac == pytest.approx(1.0)
    assert abs(mag) < 0.1


def test_spec_to_mag_reports_partial_filter_coverage():
    wave, flux = read_template("G5V_pickles_27.fits")
    # Chop the spectrum so it covers only part of the V band.
    keep = wave < 5500.0
    mag, _, frac = spec_to_mag(wave[keep], flux[keep] * 1e17, "Buser_V.dat")
    assert mag == -99.0
    assert 0.0 < frac < 1.0


def test_spec_to_mag_rejects_a_bad_careful_value():
    wave, flux = read_template("G5V_pickles_27.fits")
    with pytest.raises(ValueError, match="careful"):
        spec_to_mag(wave, flux, "Buser_V.dat", careful=1.5)


def test_generate_template_is_zero_outside_the_template_range():
    obs = Observation(template="G5V_pickles_27.fits", filter="Buser_V.dat", mtype=2)
    wave = np.array([500.0, 5000.0, 50000.0])
    n0 = generate_template(obs, wave)
    assert n0[0] == 0.0
    assert n0[2] == 0.0
    assert n0[1] > 0.0


def test_generate_template_normalises_to_zero_magnitude():
    # A 0-mag source through the normalising filter should give a photon flux
    # comparable to the AB zero point at the filter's effective wavelength.
    obs = Observation(template="G5V_pickles_27.fits", filter="Buser_V.dat", mtype=2)
    wave = np.array([5500.0])
    assert generate_template(obs, wave)[0] == pytest.approx(flux_ab(wave)[0], rel=0.3)


def test_redshifting_a_template_off_the_filter_raises():
    # z = 5 lifts the 1150 A blue end of the template above the V band.
    obs = Observation(
        template="G5V_pickles_27.fits", filter="Buser_V.dat", mtype=2, redshift=5.0
    )
    with pytest.raises(TemplateFilterMismatch, match="covers only"):
        generate_template(obs, np.array([5000.0]))
