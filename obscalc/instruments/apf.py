"""The Levy spectrograph on the APF.

Ports ``x_initapfspec.pro`` (instrument parameters and deckers) and
``apf_thruput.pro`` (measured end-to-end throughput).
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from .. import config
from ..idl_compat import interpol
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
DEFAULT_SENS_FILE = "sens_APF_nov2016.fits"

#: Default wavelength grid, from ``x_initapfspec.pro``.
DEFAULT_RANGE = (3742.0, 7700.0)


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
        mlambda=465980.24,  # measured from order central wavelengths
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


def apf_thruput(wave, sens_file=DEFAULT_SENS_FILE):
    """End-to-end throughput (0-1) at ``wave``.

    Outside the measured range this extrapolates, as the IDL did.  Unlike
    ``apflow_thruput.pro`` there is no floor on the result, so a wavelength far
    enough outside the table can give a non-physical value; the measured range
    is 3757.9-7515.8 Angstroms.
    """
    sens_wave, sens_eff = _sensitivity(sens_file)
    return interpol(sens_eff, sens_wave, np.asarray(wave, dtype=float)) / 100.0


def sensitivity_range(sens_file=DEFAULT_SENS_FILE):
    """Wavelength range actually covered by the throughput measurement."""
    sens_wave, _ = _sensitivity(sens_file)
    return float(sens_wave.min()), float(sens_wave.max())


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
        index = np.arange(np.size(wave))
        return tel, [Side("", instr, index, apf_thruput(wave))]

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
