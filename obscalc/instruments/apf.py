"""The Levy spectrograph on the APF.

Ports ``x_initapfspec.pro`` (instrument parameters and deckers) and
``apf_thruput.pro`` (measured end-to-end throughput).
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from .. import config
from ..echelle import measured_orders, order_centre_throughput
from ..s2n import Side
from ..structures import Instrument
from ..telescopes import apf_telescope
from .base import Backend, ParameterError

#: Slit width and height in arcsec for each decker, from ``apfspec_setdecker``.
DECKERS = {
    "N": (0.5, 8.0),
    "S": (0.75, 8.0),
    "M": (1.0, 8.0),
    "W": (1.0, 3.0),
    "O": (8.0, 8.0),
    "T": (2.0, 3.0),
    "B": (2.0, 8.0),
}

#: Most recent throughput measurement.  ``apf_thruput.pro`` carried commented-out
#: paths for the earlier epochs; those files are not bundled here.
DEFAULT_SENS_FILE = "sens_APF_aug2022.fits"

#: Default wavelength grid, from ``x_initapfspec.pro``.
DEFAULT_RANGE = (3742.0, 7700.0)

#: ``2 sigma sin(delta) cos(theta)`` for the echelle, measured from the order
#: central wavelengths.  Order ``m`` is centred on ``MLAMBDA / m``.
MLAMBDA = 465980.24


def set_decker(instr, decker):
    """Set slit width and height on ``instr`` from a decker name."""
    try:
        instr.swidth, instr.sheight = DECKERS[str(decker).strip()]
    except KeyError:
        raise ValueError(
            f"unknown APF decker {decker!r}; have {sorted(DECKERS)}"
        ) from None
    return instr


def apf_spectrograph(decker="W", bins=1, bind=1, str_tel=None):
    """Instrument parameters for the Levy spectrograph.

    ``decker`` defaults to ``"W"``, matching ``apf_calcs2n_wrapper.pro`` (and
    the same 1.0 x 3.0 arcsec slit that ``x_initapfspec.pro`` fell back to).
    """
    tel = str_tel or apf_telescope()

    instr = Instrument(
        name="APFSPEC",
        mag_perp=5.5136,
        mag_para=4.9913,
        pixel_size=13.5,
        R=282160.8,
        mlambda=MLAMBDA,
        dely=0.154693,
        readno=3.75,
        dark=7.3,
        bins=bins,
        bind=bind,
        wvmnx=(3742.0, 7700.0),
    )
    instr.scale_perp = tel.plate_scale * instr.mag_perp * (instr.pixel_size / 1000.0)
    instr.scale_para = tel.plate_scale * instr.mag_para * (instr.pixel_size / 1000.0)

    return set_decker(instr, decker)


@lru_cache(maxsize=None)
def _sensitivity(sens_file):
    """Throughput measurement: columns ``wav`` (Angstroms) and ``eff`` (percent).

    The file is stored in *descending* wavelength order, which is why
    :func:`obscalc.idl_compat.interpol` has to accept a descending abscissa.
    """
    path = config.resolve(sens_file, config.THRUPUT_DIR)
    with fits.open(path) as hdus:
        data = hdus[1].data
        return (
            np.asarray(data["wav"], dtype=float),
            np.asarray(data["eff"], dtype=float),
        )


def apf_thruput(wave, instr=None, blaze=True, sens_file=DEFAULT_SENS_FILE):
    """End-to-end throughput (0-1) at ``wave``.

    The Levy is a cross-dispersed echelle and its sensitivity file is tabulated
    **once per order, at the blaze centre** -- see :func:`sensitivity_orders`.  So
    each wavelength takes the throughput of its own order, constant across that
    order, scaled by the blaze function.

    ``apf_thruput.pro`` did neither: it interpolated the curve at the wavelength,
    which mixes adjacent orders, and applied no blaze, which reports every order's
    peak.  ``apf_calcs2n.pro`` computed the order number, blaze centre and free
    spectral range and then discarded all three.  Pass ``blaze=False`` for the
    per-order value without the blaze; there is no way to ask for the old
    interpolation, which was simply wrong.

    Orders outside the measurement hold the nearest measured value rather than
    extrapolating.
    """
    mlambda = instr.mlambda if instr is not None else MLAMBDA
    sens_wave, sens_eff = _sensitivity(sens_file)
    return order_centre_throughput(
        np.asarray(wave, dtype=float), mlambda, sens_wave, sens_eff / 100.0, blaze=blaze
    )


def sensitivity_range(sens_file=DEFAULT_SENS_FILE):
    """Wavelength range actually covered by the throughput measurement."""
    sens_wave, _ = _sensitivity(sens_file)
    return float(sens_wave.min()), float(sens_wave.max())


def sensitivity_orders(sens_file=DEFAULT_SENS_FILE, mlambda=MLAMBDA):
    """Echelle orders the sensitivity file measures.

    The file's 63 wavelengths are the 63 consecutive order centres 62 to 124,
    matching ``MLAMBDA / m`` to better than 0.006 Angstroms.  That is the evidence
    the measurement is per-order, and :func:`measured_orders` raises if a
    replacement file does not share the property.
    """
    sens_wave, _ = _sensitivity(sens_file)
    return measured_orders(sens_wave, mlambda)


class APFBackend(Backend):
    """Web backend for the APF.

    ``slitwidth`` is a decker letter here, not a width in arcsec -- the APF form
    posts ``N``/``S``/``M``/``W``/``O``/``T``/``B``.  A single detector covers the
    whole range, so there is one side.
    """

    name = "apf"
    default_range = DEFAULT_RANGE

    def sides(self, wave, values):
        decker = str(values.get("slitwidth") or "W").strip()
        if decker not in DECKERS:
            raise ParameterError(
                "Inappropriate value for the input parameter Slitwidth: "
                f"expected an APF decker, one of {', '.join(sorted(DECKERS))}"
            )
        tel = apf_telescope()
        instr = apf_spectrograph(
            decker=decker,
            bins=values["bins"],
            bind=values["bind"],
            str_tel=tel,
        )
        blaze = str(values.get("blaze", "true")).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        index = np.arange(np.size(wave))
        return tel, [
            Side("", instr, index, apf_thruput(wave, instr, blaze=blaze))
        ]

    def extras(self, result, obs):
        # Imported here because apf_extras reads back from a finished result.
        from ..apf_extras import apf_extras

        _, _, single = result.sides[0]
        values = apf_extras(single, obs)
        return {
            "i2counts": values.i2counts,
            "exp": values.expmeter,
            "precision": values.precision,
        }
