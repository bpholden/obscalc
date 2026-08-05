"""The Kast double spectrograph on the Shane 3 m at Lick.

Ports ``x_initkast.pro`` and ``kast_thruput.pro``.

Kast is the first two-channel instrument here.  A dichroic splits the beam, and
the two sides have different resolutions, read noise and throughput curves, so
each needs its own pass through :func:`obscalc.s2n.spec_calcs2n` -- which is what
``kast_calcs2n.pro`` did before concatenating the two.
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from .. import config
from ..idl_compat import interpol
from ..s2n import Side
from ..structures import Instrument
from ..telescopes import lick_telescope
from .base import Backend, ParameterError

#: Wavelength at which each dichroic hands over from the blue to the red side.
DICHROICS = {"d46": 4600.0, "d55": 5500.0}

#: Blue-side grisms and their resolving power at one native pixel.
GRISMS = {"G1": 2344.0, "G2": 4254.0, "G3": 5492.0}

#: Red-side gratings and their resolving power at one native pixel.
GRATINGS = {"600/7500": 3164.0}

#: Throughput measurements, keyed on grism/grating.  ``kast_thruput.pro`` chose
#: the file by disperser and then had a dichroic sub-case whose branches were
#: identical, so the dichroic does not actually select a file.
BLUE_SENS_FILES = {
    "G2": "sens_Kastb600_4310_d55.fits",
    "G3": "sens_Kastb830_3460_d46.fits",
}
RED_SENS_FILES = {"600/7500": "sens_Kastr600_7500_d55.fits"}

#: Floor applied to the throughput, from ``kast_thruput.pro``.  Kast needs it:
#: the measurements do not span the full 3150-8000 A grid the wrapper used, and
#: extrapolating the red end of a falling curve goes negative.
MIN_THRUPUT = 1e-5

#: Default wavelength grid, from ``kast_calcs2n_wrapper.pro``.
DEFAULT_RANGE = (3150.0, 8000.0)

DEFAULT_GRISM = "G2"
DEFAULT_GRATING = "600/7500"
DEFAULT_DICHROIC = "d46"
DEFAULT_SLIT = 0.75  # arcsec, from x_initkast


def kast_spectrograph(
    grism=DEFAULT_GRISM,
    grating=DEFAULT_GRATING,
    dichroic=DEFAULT_DICHROIC,
    slit=DEFAULT_SLIT,
    bins=1,
    bind=1,
    str_tel=None,
):
    """Return the ``(blue, red)`` instrument pair.

    ``x_initkast.pro`` intended different read noise per detector -- its
    comments say 3.7 for the blue and 3.8 for the red -- but wrote
    ``kastinstr.readno = 3.7`` followed by ``kastinstr.readno = 3.8``, and since
    ``kastinstr`` is a two-element array the second assignment overwrote both.
    Every Kast calculation therefore used 3.8 on both sides.  The per-detector
    values are used here.
    """
    if grism not in GRISMS:
        raise ValueError(f"unknown Kast grism {grism!r}; have {sorted(GRISMS)}")
    if grating not in GRATINGS:
        raise ValueError(f"unknown Kast grating {grating!r}; have {sorted(GRATINGS)}")
    if dichroic not in DICHROICS:
        raise ValueError(
            f"unknown Kast dichroic {dichroic!r}; have {sorted(DICHROICS)}"
        )

    tel = str_tel or lick_telescope()

    sides = []
    for name, disperser, resolution, wvmnx, readno in (
        ("blue", grism, GRISMS[grism], (3000.0, 6000.0), 3.7),
        ("red", grating, GRATINGS[grating], (5000.0, 10000.0), 3.8),
    ):
        instr = Instrument(
            name=f"Kast-{name}",
            mag_perp=20.9,
            mag_para=20.9,
            pixel_size=15.0,  # microns
            R=resolution,
            grating=disperser,
            dichroic=dichroic,
            readno=readno,
            dark=0.001,
            swidth=slit,
            sheight=120.0,  # long slit
            wvmnx=wvmnx,
            bins=bins,
            bind=bind,
        )
        # Both channels use the same magnification and pixel size in the IDL, so
        # both come out at 0.43"/pixel.  The comment on the red side says 0.78",
        # which was true of the pre-upgrade red CCD; the code has always
        # computed 0.43" and that is kept here.
        instr.scale_perp = tel.plate_scale * instr.mag_perp * (
            instr.pixel_size / 1000.0
        )
        instr.scale_para = tel.plate_scale * instr.mag_para * (
            instr.pixel_size / 1000.0
        )
        sides.append(instr)

    return tuple(sides)


@lru_cache(maxsize=None)
def _sensitivity(sens_file):
    """Throughput measurement from HDU 2, as vector columns.

    Unlike the APF file these live in HDU 2, hold one row of 999-element vectors,
    and ``EFF`` is already a fraction rather than a percentage.
    """
    path = config.resolve(sens_file, config.THRUPUT_DIR)
    with fits.open(path) as hdus:
        data = hdus[2].data
        return (
            np.asarray(data["WAV"], dtype=float).ravel(),
            np.asarray(data["EFF"], dtype=float).ravel(),
        )


def _side_thruput(wave, sens_file):
    """Interpolate one side's throughput, holding the blue end flat."""
    sens_wave, sens_eff = _sensitivity(sens_file)
    thru = interpol(sens_eff, sens_wave, wave)
    # kast_thruput.pro clamps below the measured range but lets the red end
    # extrapolate, which is why MIN_THRUPUT is needed.
    return np.where(wave < sens_wave[0], sens_eff[0], thru)


def split_wavelengths(wave, dichroic):
    """Indices of the blue and red sides for a dichroic."""
    try:
        crossover = DICHROICS[dichroic]
    except KeyError:
        raise ValueError(
            f"unknown Kast dichroic {dichroic!r}; have {sorted(DICHROICS)}"
        ) from None
    wave = np.asarray(wave, dtype=float)
    return np.flatnonzero(wave < crossover), np.flatnonzero(wave >= crossover)


def kast_thruput(wave, blue, red):
    """End-to-end throughput across both sides.

    ``blue`` and ``red`` are the two instruments from :func:`kast_spectrograph`;
    the dichroic is read off the blue one, as ``kast_thruput.pro`` did.
    """
    wave = np.asarray(wave, dtype=float)
    thru = np.zeros(wave.shape)

    blue_index, red_index = split_wavelengths(wave, blue.dichroic)

    if blue_index.size:
        thru[blue_index] = _side_thruput(
            wave[blue_index], _blue_sens_file(blue.grating)
        )
    if red_index.size:
        thru[red_index] = _side_thruput(
            wave[red_index], _red_sens_file(red.grating)
        )

    return np.maximum(thru, MIN_THRUPUT)


def _blue_sens_file(grism):
    try:
        return BLUE_SENS_FILES[grism]
    except KeyError:
        raise ParameterError(
            f"No throughput measurement exists for Kast grism {grism!r}. "
            f"Measured grisms are {', '.join(sorted(BLUE_SENS_FILES))}."
        ) from None


def _red_sens_file(grating):
    try:
        return RED_SENS_FILES[grating]
    except KeyError:
        raise ParameterError(
            f"No throughput measurement exists for Kast grating {grating!r}. "
            f"Measured gratings are {', '.join(sorted(RED_SENS_FILES))}."
        ) from None


def sensitivity_range(sens_file):
    """Wavelength range actually covered by a throughput measurement."""
    sens_wave, _ = _sensitivity(sens_file)
    return float(sens_wave.min()), float(sens_wave.max())


class KastBackend(Backend):
    """Web backend for Kast.

    ``slitwidth`` is a width in arcsec here, not a decker letter as it is for
    APF.  ``grism`` selects the blue disperser, ``grating`` the red, and
    ``dichroic`` where the two meet.
    """

    name = "kast"
    default_range = DEFAULT_RANGE

    def sides(self, wave, values):
        grism = str(values.get("grism") or DEFAULT_GRISM).strip()
        grating = str(values.get("grating") or DEFAULT_GRATING).strip()
        dichroic = str(values.get("dichroic") or DEFAULT_DICHROIC).strip()

        if grism not in GRISMS:
            raise ParameterError(
                "Inappropriate value for the input parameter Grism: "
                f"expected one of {', '.join(sorted(BLUE_SENS_FILES))}"
            )
        if grating not in GRATINGS:
            raise ParameterError(
                "Inappropriate value for the input parameter Grating: "
                f"expected one of {', '.join(sorted(GRATINGS))}"
            )
        if dichroic not in DICHROICS:
            raise ParameterError(
                "Inappropriate value for the input parameter Dichroic: "
                f"expected one of {', '.join(sorted(DICHROICS))}"
            )

        try:
            slit = float(values.get("slitwidth", DEFAULT_SLIT))
        except (TypeError, ValueError):
            raise ParameterError(
                "Inappropriate value for the input parameter Slitwidth: "
                "expected a width in arcsec"
            ) from None
        if slit <= 0:
            raise ParameterError(
                "Inappropriate value for the input parameter Slitwidth: "
                "expected a width in arcsec"
            )

        tel = lick_telescope()
        blue, red = kast_spectrograph(
            grism=grism,
            grating=grating,
            dichroic=dichroic,
            slit=slit,
            bins=values["bins"],
            bind=values["bind"],
            str_tel=tel,
        )

        thru = kast_thruput(wave, blue, red)
        blue_index, red_index = split_wavelengths(wave, dichroic)

        return tel, [
            Side("blue", blue, blue_index, thru[blue_index]),
            Side("red", red, red_index, thru[red_index]),
        ]
