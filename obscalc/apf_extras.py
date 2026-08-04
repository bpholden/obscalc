"""APF-specific quantities derived from the count rates.

Ports the three helper functions at the top of ``apf_calcs2n.pro``: the counts
in the iodine cell region, the predicted exposure meter reading, and the
expected radial-velocity precision.  These are displayed by the web ETC, so they
are not optional.

One deliberate change.  ``i2counts_value`` in the IDL returned
``alog10(median(counts)) * binc * binc``, and that value was then used two
inconsistent ways: ``expmeter_value`` added it to a log-space offset (treating
it as a log, which is consistent), while ``precision_value`` took ``alog10`` of
it again (treating it as linear counts, which is not).  The published
coefficients ``A = 4.47``, ``B = -1.58`` only make sense against
``log10(counts)``, which for APF is 3-5; feeding them ``log10(log10(counts))``
instead inflates the reported RV precision by roughly two orders of magnitude.
Multiplying a logarithm by ``binc**2`` is also not a count scaling under any
reading.

So :func:`i2counts` here returns plain median counts, and each consumer takes
its own logarithm exactly once.  :data:`I2COUNTS_IS_LINEAR` records the change
for anyone diffing against the IDL.
"""

from dataclasses import dataclass

import numpy as np

from .photometry import TemplateFilterMismatch, read_template, spec_to_mag

#: Wavelength range of the iodine cell absorption used for RV work, Angstroms.
I2_WINDOW = (5000.0, 6200.0)

#: True: unlike the IDL, i2counts is linear counts rather than their logarithm.
I2COUNTS_IS_LINEAR = True

# Exposure meter calibration, from expmeter_value.
_EM_DELTA = 4.52
_EM_EPSILON = -0.196
_EM_ZETA = 0.262

# RV precision calibration, from precision_value.  The second pair applies to
# red stars.
_PREC_BLUE = (4.47, -1.58)
_PREC_RED = (4.14, -1.73)
_PREC_BMV_BREAK = 1.2

DEFAULT_B_FILTER = "Buser_B.dat"
DEFAULT_V_FILTER = "Buser_V.dat"


@dataclass
class APFExtras:
    """Derived APF quantities.  ``bmv`` is None when no template was used."""

    i2counts: float  # median object counts in the iodine region
    expmeter: float  # predicted exposure meter reading
    precision: float  # expected RV precision, m/s
    bmv: float | None  # Vega-system B-V of the template


def i2counts(wave, star):
    """Median object counts across the iodine cell region."""
    wave = np.asarray(wave, dtype=float)
    star = np.asarray(star, dtype=float)
    in_window = (wave >= I2_WINDOW[0]) & (wave <= I2_WINDOW[1])
    if not in_window.any():
        raise ValueError(
            f"no wavelengths in the iodine region {I2_WINDOW[0]:.0f}-"
            f"{I2_WINDOW[1]:.0f} A; widen the wavelength range"
        )
    return float(np.median(star[in_window]))


def exposure_meter_value(counts, bmv):
    """Predicted exposure meter reading for ``counts`` and colour ``bmv``."""
    if counts <= 0:
        return 0.0
    log_value = (
        np.log10(counts) + _EM_DELTA + _EM_EPSILON * bmv + _EM_ZETA * bmv * bmv
    )
    return float(10.0**log_value)


def rv_precision(counts, bmv, binc=1):
    """Expected radial-velocity precision in m/s."""
    if counts <= 0:
        return 0.0
    slope_intercept = _PREC_RED if bmv > _PREC_BMV_BREAK else _PREC_BLUE
    a, b = slope_intercept
    return float(10.0 ** ((np.log10(counts) + np.log10(binc) - a) / b))


def template_bmv(obs, b_filter=DEFAULT_B_FILTER, v_filter=DEFAULT_V_FILTER):
    """Vega-system B-V colour of ``obs``'s redshifted template.

    Both the template and Vega are measured through the same two filters and the
    AB colours differenced, which cancels the AB-to-Vega offset.
    """
    wave, flux = read_template(obs.template)
    wave = wave * (1.0 + obs.redshift)
    template_b, _, _ = spec_to_mag(wave, flux * 1e17, b_filter)
    template_v, _, _ = spec_to_mag(wave, flux * 1e17, v_filter)

    vega_wave, vega_flux = read_template(obs.vega_template)
    vega_b, _, _ = spec_to_mag(vega_wave, vega_flux * 1e17, b_filter)
    vega_v, _, _ = spec_to_mag(vega_wave, vega_flux * 1e17, v_filter)

    if min(template_b, template_v, vega_b, vega_v) <= -98:
        raise TemplateFilterMismatch(
            f"template {obs.template!r} at z={obs.redshift} does not cover both "
            f"{b_filter} and {v_filter}, so B-V is undefined"
        )

    return (template_b - template_v) - (vega_b - vega_v)


def apf_extras(result, obs):
    """Iodine counts, exposure meter reading and RV precision for a result.

    Without a template and filter there is no colour, so -- as in
    ``apf_calcs2n.pro`` -- the exposure meter and precision are reported as 0.
    """
    counts = i2counts(result.wave, result.star)

    if not (obs.template and obs.filter):
        return APFExtras(i2counts=counts, expmeter=0.0, precision=0.0, bmv=None)

    bmv = template_bmv(obs)
    return APFExtras(
        i2counts=counts,
        expmeter=exposure_meter_value(counts, bmv),
        precision=rv_precision(counts, bmv, binc=result.binc),
        bmv=bmv,
    )
