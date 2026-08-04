"""Magnitude systems, filter photometry and spectral templates.

Ports ``x_fluxjohnson.pro``, ``single_spec2mag.pro`` and the
``generate_template`` helper from ``spec_calcs2n.pro``.
"""

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.io import fits

from . import config
from .idl_compat import interpol, linterp, tsum

#: Speed of light in cm/s, as spelled in ``single_spec2mag.pro``.
_C_CGS = 2.9979000e10
#: Planck constant, erg s, and c in Angstrom/s, as spelled in the IDL.
_H_ERG_S = 6.626e-27
_C_ANGSTROM_S = 2.99792e18

#: AB zero point: an AB magnitude of 0 is f_nu = 10^(-0.4*48.6) erg/s/cm^2/Hz.
AB_ZERO_POINT_MAG = 48.6

# Johnson zero-point flux, photons/s/cm^2/Angstrom for a 0-mag star.
_JOHNSON_WAVE = np.array(
    [
        3062.0, 3312.0, 3562.0, 3812.0, 4062.0, 4212.0, 4462.0, 4712.0,
        4962.0, 5425.0, 5925.0, 6425.0, 6925.0, 7750.0, 8350.0, 11050.0,
    ]
)
_JOHNSON_FLUX = np.array(
    [
        569.0, 586.0, 590.0, 1326.0, 1770.0, 1707.0, 1530.0, 1356.0,
        1257.0, 1054.0, 886.0, 749.0, 641.0, 502.0, 435.0, 263.0,
    ]
)


class TemplateFilterMismatch(Exception):
    """The template does not cover the normalising filter.

    Raised where ``single_spec2mag`` returned its ``-99`` sentinel, which the
    IDL then propagated silently into the count rates.
    """


def flux_johnson(wave):
    """Photons/s/cm^2/Angstrom for a 0th magnitude Johnson star."""
    return interpol(_JOHNSON_FLUX, _JOHNSON_WAVE, np.asarray(wave, dtype=float))


def flux_ab(wave):
    """Photons/s/cm^2/Angstrom for a 0th magnitude AB source."""
    wave = np.asarray(wave, dtype=float)
    return 10.0 ** (-0.4 * AB_ZERO_POINT_MAG) / _H_ERG_S / wave


def zero_point_flux(mtype, wave):
    """Photon flux of a 0-magnitude source in system ``mtype``.

    ``mtype`` is 1 for Vega/Johnson and 2 for AB.
    """
    if mtype == 1:
        return flux_johnson(wave)
    if mtype == 2:
        return flux_ab(wave)
    raise ValueError(f"mtype must be 1 (Vega/Johnson) or 2 (AB), got {mtype!r}")


def read_filter(name):
    """Read a filter transmission curve as ``(wave, throughput)``.

    Takes the first two whitespace-separated columns of the non-comment lines,
    which is what ``readcol, file, fwave, fthru`` did in the IDL.  Some files
    have exactly two columns (the Buser and Gaia curves); the SDSS responses
    have five plus a ``#`` header, where the second column is the response
    through one airmass.
    """
    path = config.filter_file(name)
    table = ascii_io.read(path, format="no_header", comment="#")
    if len(table.colnames) < 2:
        raise ValueError(f"{path.name} has fewer than two columns")
    return (
        np.asarray(table[table.colnames[0]], dtype=float),
        np.asarray(table[table.colnames[1]], dtype=float),
    )


def read_template(name):
    """Read a spectral template as ``(wave, flux)`` 1-D arrays.

    Handles both layouts present in the template collection: one row per
    wavelength (the Pickles spectra, Vega), and a single row holding vector
    columns (``qso_template.fits`` is 1 row of ``7755E``).  Columns are matched
    by name because the order varies -- ``qso_template.fits`` stores ``FLUX``
    first.  Flux is erg/s/cm^2/Angstrom.
    """
    path = config.template_file(name)
    with fits.open(path) as hdus:
        data = hdus[1].data
        names = {n.upper(): n for n in data.columns.names}
        try:
            wave_col, flux_col = names["WAVELENGTH"], names["FLUX"]
        except KeyError:
            raise ValueError(
                f"{path.name} has columns {data.columns.names}, "
                "expected WAVELENGTH and FLUX"
            ) from None
        wave = np.asarray(data[wave_col], dtype=float).ravel()
        flux = np.asarray(data[flux_col], dtype=float).ravel()
    return wave, flux


def spec_to_mag(wave, flux, filter_name, careful=1.0):
    """AB magnitude of a spectrum through a filter.

    Port of ``single_spec2mag.pro``.  ``flux`` is in units of
    1e-17 erg/s/cm^2/Angstrom.

    ``careful`` is the fraction of the filter's integral that the spectrum must
    span; the default of 1.0 demands complete coverage.  Returns
    ``(mag, totflux, frac_filt)``, with ``mag`` set to -99 when coverage is
    insufficient, exactly as the IDL did.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if not 0.0 <= careful <= 1.0:
        raise ValueError(f"careful must be between 0 and 1, got {careful!r}")

    filter_wave, filter_thru = read_filter(filter_name)
    filter_thru = filter_thru / filter_thru.sum()

    covered = np.flatnonzero(
        (filter_wave >= wave.min()) & (filter_wave <= wave.max())
    )
    if covered.size == 0:
        return -99.0, 0.0, 0.0

    filter_total = tsum(filter_wave, filter_thru)
    frac_filt = (
        tsum(filter_wave, filter_thru, covered.min(), covered.max()) / filter_total
    )
    if frac_filt < careful:
        return -99.0, 0.0, frac_filt

    # Throughput-weighted mean wavelength of the filter.
    lambda_eff = tsum(filter_wave, filter_thru * filter_wave) / filter_total

    interp_filter = linterp(filter_wave, filter_thru, wave, missing=0.0)
    total_filter = tsum(wave, interp_filter)

    # f_lambda -> f_nu via lambda^2/c, with 1e-17 for the input units and 1e-8
    # to take lambda^2 from Angstroms^2 to cm*Angstrom.
    totflux = (
        1e-17
        * lambda_eff**2
        * 1e-8
        * tsum(wave, flux * interp_filter)
        / total_filter
        / _C_CGS
    )
    mag = -2.5 * np.log10(totflux) - AB_ZERO_POINT_MAG
    return float(mag), float(totflux), float(frac_filt)


def template_magnitude(obs, filter_name=None):
    """Magnitude of ``obs``'s redshifted template in ``obs.filter``.

    In the Vega system (``obs.mtype == 1``) the AB magnitude of Vega through the
    same filter is subtracted, so the result is a Vega magnitude.
    """
    filter_name = filter_name or obs.filter
    wave, flux = read_template(obs.template)
    wave = wave * (1.0 + obs.redshift)

    mag, _, frac = spec_to_mag(wave, flux * 1e17, filter_name)
    if mag <= -98:
        raise TemplateFilterMismatch(
            f"template {obs.template!r} covers only {frac:.3f} of filter "
            f"{filter_name!r}; redshift the template or choose another filter"
        )

    if obs.mtype == 1:
        vega_wave, vega_flux = read_template(obs.vega_template)
        vega_mag, _, vega_frac = spec_to_mag(vega_wave, vega_flux * 1e17, filter_name)
        if vega_mag <= -98:
            # The IDL did not check this, and would have carried the -99 through.
            raise TemplateFilterMismatch(
                f"Vega template {obs.vega_template!r} covers only "
                f"{vega_frac:.3f} of filter {filter_name!r}"
            )
        mag -= vega_mag

    return mag


def generate_template(obs, wave):
    """Photon flux of ``obs``'s template, normalised to 0 magnitude.

    Port of ``generate_template`` in ``spec_calcs2n.pro``.  Returns
    photons/s/cm^2/Angstrom at ``wave``, zero outside the template's range, for
    a source of magnitude 0 in ``obs.filter``.
    """
    wave = np.asarray(wave, dtype=float)

    template_wave, template_flux = read_template(obs.template)
    template_wave = template_wave * (1.0 + obs.redshift)

    mag = template_magnitude(obs)

    # Scale to 0 magnitude in the filter, then convert f_lambda to photons.
    flux = template_flux * 10.0 ** (0.4 * mag)
    flux = flux * template_wave / (_H_ERG_S * _C_ANGSTROM_S)

    n0 = interpol(flux, template_wave, wave, spline=True)
    outside = (wave > template_wave.max()) | (wave < template_wave.min())
    n0[outside] = 0.0
    return n0
