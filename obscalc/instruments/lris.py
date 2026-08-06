"""LRIS on Keck I.

Ports ``x_initlris.pro`` and ``lris_thruput.pro``.

Like Kast, LRIS is a double spectrograph: a dichroic splits the beam and the two
sides have their own disperser, resolving power, read noise and throughput curve,
so each needs its own pass through :func:`obscalc.s2n.spec_calcs2n` -- which is
what ``lris_calcs2n.pro`` did before concatenating the two.

Unlike Kast, the throughput file depends on the dichroic as well as the
disperser, because a sensitivity measurement carries the dichroic that was in the
beam when it was taken.  Only ``D560`` has measurements on both sides, which is
also all ``lris_thruput.pro`` would run: its outer ``case str_instr[0].dichroic``
stopped on anything else.  See :data:`UNUSABLE_SENS_FILES` for the one measured
configuration that leaves behind.
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from .. import config
from ..idl_compat import interpol
from ..s2n import Side
from ..structures import Instrument
from ..telescopes import keck_telescope
from .base import Backend, ParameterError

#: Wavelength at which the dichroic hands over from the blue to the red side.
#: ``lris_thruput.pro`` had a commented-out ``d46`` entry and stopped on anything
#: but ``D560``; the wrapper's ``dichroics`` list of four is the GUI's, not the
#: calculator's.
DICHROICS = {"D560": 5600.0}

#: Blue-side grisms and their resolving power at one native pixel.
GRISMS = {"B600": 7500.0, "B300": 3304.0}

#: Red-side gratings and their resolving power at one native pixel.  ``600/7500``
#: and ``600/10000`` share a resolving power: same ruling, different blaze.
GRATINGS = {
    "600/7500": 11820.0,
    "600/10000": 11820.0,
    "400/8500": 8151.0,
    "831/8200": 16303.0,
    "1200/9000": 23640.0,
}

#: Throughput measurements for the ``D560`` dichroic, keyed on disperser.
BLUE_SENS_FILES = {
    "B600": "sens_LRISb_600_4000_D560.fits",
    "B300": "sens_LRISb_300_5000_D560.fits",
}
RED_SENS_FILES = {
    "600/7500": "sens_LRISr_600_7500_D560.fits",
    "600/10000": "sens_LRISr_600_10000_D560.fits",
    "400/8500": "sens_LRISr_400_8500_D560.fits",
    "831/8200": "sens_LRISr_831_8200_D560.fits",
    "1200/9000": "sens_LRISr_1200_9000_D560.fits",
}

#: Measured but unusable.  ``sens_LRISb_300_5000_D680.fits`` is a real B300
#: measurement behind the D680 dichroic, and ``lris_thruput.pro`` names it -- but
#: in a ``case`` nested inside the dichroic ``case`` that stops unless the
#: dichroic is D560, so it could never be reached.  It stays unreachable here for
#: a different reason: every red-side measurement is D560, so a D680 run would
#: have a blue side and no red one.  The file is bundled so the gap is a matter
#: of record rather than a missing file.
UNUSABLE_SENS_FILES = {("B300", "D680"): "sens_LRISb_300_5000_D680.fits"}

#: Floor applied to the throughput, from ``lris_thruput.pro``.  Kept because
#: ``sens_LRISr_400_8500`` dips to 0.0004 at its blue end and the engine divides
#: by nothing but must not see a negative.
MIN_THRUPUT = 1e-5

#: Default wavelength grid, from ``lris_calcs2n_wrapper.pro``.
DEFAULT_RANGE = (3500.0, 10000.0)

DEFAULT_GRISM = "B600"
DEFAULT_GRATING = "600/7500"
DEFAULT_DICHROIC = "D560"
DEFAULT_SLIT = 1.0  # arcsec, from x_initlris
DEFAULT_SEEING = 1.0  # arcsec, from lris_calcs2n_wrapper.pro


def lris_spectrograph(
    grism=DEFAULT_GRISM,
    grating=DEFAULT_GRATING,
    dichroic=DEFAULT_DICHROIC,
    slit=DEFAULT_SLIT,
    bins=1,
    bind=1,
    str_tel=None,
):
    """Return the ``(blue, red)`` instrument pair.

    ``x_initlris.pro`` intended different read noise per detector -- 3.7 for the
    blue and 4.5 for the red, under comments naming each -- but wrote
    ``lrisinstr.readno = 3.7`` followed by ``lrisinstr.readno = 4.5`` on a
    two-element array, so the second assignment overwrote both and every LRIS
    calculation used 4.5 on each side.  The per-detector values are used here,
    as for the identical mistake in ``x_initkast.pro``.
    """
    if grism not in GRISMS:
        raise ValueError(f"unknown LRIS grism {grism!r}; have {sorted(GRISMS)}")
    if grating not in GRATINGS:
        raise ValueError(f"unknown LRIS grating {grating!r}; have {sorted(GRATINGS)}")
    if dichroic not in DICHROICS:
        raise ValueError(
            f"unknown LRIS dichroic {dichroic!r}; have {sorted(DICHROICS)}"
        )

    tel = str_tel or keck_telescope("KeckI")

    sides = []
    for name, disperser, resolution, wvmnx, readno in (
        ("blue", grism, GRISMS[grism], (3000.0, 6000.0), 3.7),
        ("red", grating, GRATINGS[grating], (5000.0, 10000.0), 4.5),
    ):
        instr = Instrument(
            name=f"LRIS-{name}",
            mag_perp=6.5,
            mag_para=6.5,
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
        # Both channels use the same magnification and pixel size, giving
        # 0.134"/pixel, which is what the "Modified to give 0.135" pixels"
        # comment in x_initlris.pro is aiming at.
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

    Same layout as the Kast files: HDU 2, one row of 999-element vectors, with
    ``EFF`` already a fraction rather than a percentage.
    """
    path = config.resolve(sens_file, config.THRUPUT_DIR)
    with fits.open(path) as hdus:
        data = hdus[2].data
        return (
            np.asarray(data["WAV"], dtype=float).ravel(),
            np.asarray(data["EFF"], dtype=float).ravel(),
        )


def sensitivity_range(sens_file):
    """Wavelength range a throughput measurement covers."""
    sens_wave, _ = _sensitivity(sens_file)
    return float(sens_wave.min()), float(sens_wave.max())


def _side_thruput(wave, sens_file):
    """Interpolate one side's throughput, holding both ends flat.

    ``lris_thruput.pro`` held the blue end (``wave < sens.wav[0]`` takes
    ``sens.eff[0]``) but let ``interpol`` extrapolate past the red end, clipping
    only with ``> 1e-5``.  That catches a curve falling negative and misses one
    rising, and here it is the rising case that bites: ``600/7500`` is measured
    only to 8191 A, and continuing its last interval to the 10000 A end of the
    default grid reaches an efficiency of 0.415 -- above every red-side
    measurement here (best 0.378), above anything on its own curve (peak 0.221),
    and 3.3 times the 0.125 measured at the red end of it.  ``1200/9000`` and
    ``831/8200`` instead run negative and would be floored, which hides 1600 A of
    grid behind a number that only looks small.

    So the red end is held too, as :func:`obscalc.instruments.deimos.deimos_thruput`
    does for the same reason.  :func:`unmeasured_ranges` reports where.
    """
    sens_wave, sens_eff = _sensitivity(sens_file)
    thru = interpol(sens_eff, sens_wave, wave)
    thru = np.where(wave < sens_wave[0], sens_eff[0], thru)
    thru = np.where(wave > sens_wave[-1], sens_eff[-1], thru)
    return thru


def split_wavelengths(wave, dichroic):
    """Indices of the blue and red sides for a dichroic."""
    try:
        crossover = DICHROICS[dichroic]
    except KeyError:
        raise ValueError(
            f"unknown LRIS dichroic {dichroic!r}; have {sorted(DICHROICS)}"
        ) from None
    wave = np.asarray(wave, dtype=float)
    return np.flatnonzero(wave < crossover), np.flatnonzero(wave >= crossover)


def _blue_sens_file(grism, dichroic):
    if (grism, dichroic) in UNUSABLE_SENS_FILES:
        raise ParameterError(
            f"LRIS grism {grism!r} was measured with dichroic {dichroic!r}, but no "
            f"red-side measurement exists for that dichroic. Use "
            f"{DEFAULT_DICHROIC}."
        )
    try:
        return BLUE_SENS_FILES[grism]
    except KeyError:
        raise ParameterError(
            f"No throughput measurement exists for LRIS grism {grism!r}. "
            f"Measured grisms are {', '.join(sorted(BLUE_SENS_FILES))}."
        ) from None


def _red_sens_file(grating):
    try:
        return RED_SENS_FILES[grating]
    except KeyError:
        raise ParameterError(
            f"No throughput measurement exists for LRIS grating {grating!r}. "
            f"Measured gratings are {', '.join(sorted(RED_SENS_FILES))}."
        ) from None


def sens_files(blue, red):
    """The ``(blue, red)`` measurement filenames a configuration uses.

    Only the blue lookup takes the dichroic: every red-side measurement here was
    taken behind D560, so there is nothing for it to choose between.
    """
    return _blue_sens_file(blue.grating, blue.dichroic), _red_sens_file(red.grating)


def lris_thruput(wave, blue, red):
    """End-to-end throughput across both sides.

    ``blue`` and ``red`` are the two instruments from :func:`lris_spectrograph`;
    the dichroic is read off the blue one, as ``lris_thruput.pro`` did.
    """
    wave = np.asarray(wave, dtype=float)
    thru = np.zeros(wave.shape)

    blue_index, red_index = split_wavelengths(wave, blue.dichroic)
    blue_file, red_file = sens_files(blue, red)

    if blue_index.size:
        thru[blue_index] = _side_thruput(wave[blue_index], blue_file)
    if red_index.size:
        thru[red_index] = _side_thruput(wave[red_index], red_file)

    return np.maximum(thru, MIN_THRUPUT)


def unmeasured_ranges(wave, blue, red):
    """Where each side reaches past its measurement, as ``(side, low, high)``.

    Throughput there is held at the nearest measured value, so the counts are an
    extrapolation however plausible they look.  Only the wavelengths a side
    actually records count: the blue measurement stopping at 5596 A does not
    matter for a grid that hands over to the red side at 5600 A.
    """
    wave = np.asarray(wave, dtype=float)
    blue_index, red_index = split_wavelengths(wave, blue.dichroic)
    blue_file, red_file = sens_files(blue, red)

    held = []
    for name, index, sens_file in (
        ("blue", blue_index, blue_file),
        ("red", red_index, red_file),
    ):
        if not index.size:
            continue
        low, high = sensitivity_range(sens_file)
        side_wave = wave[index]
        for outside in (side_wave < low, side_wave > high):
            if outside.any():
                held.append(
                    (
                        name,
                        float(side_wave[outside].min()),
                        float(side_wave[outside].max()),
                    )
                )
    return held


class LrisBackend(Backend):
    """Web backend for LRIS.

    ``slitwidth`` is a width in arcsec, as for Kast and DEIMOS.  ``grism``
    selects the blue disperser, ``grating`` the red, and ``dichroic`` where the
    two meet.
    """

    name = "lris"
    default_range = DEFAULT_RANGE

    def sides(self, wave, values):
        grism = str(values.get("grism") or DEFAULT_GRISM).strip()
        grating = str(values.get("grating") or DEFAULT_GRATING).strip()
        dichroic = str(values.get("dichroic") or DEFAULT_DICHROIC).strip()

        if grism not in GRISMS:
            raise ParameterError(
                "Inappropriate value for the input parameter Grism: "
                f"expected one of {', '.join(sorted(GRISMS))}"
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

        tel = keck_telescope("KeckI")
        blue, red = lris_spectrograph(
            grism=grism,
            grating=grating,
            dichroic=dichroic,
            slit=slit,
            bins=values["bins"],
            bind=values["bind"],
            str_tel=tel,
        )

        thru = lris_thruput(wave, blue, red)
        blue_index, red_index = split_wavelengths(wave, dichroic)

        return tel, [
            Side("blue", blue, blue_index, thru[blue_index]),
            Side("red", red, red_index, thru[red_index]),
        ]
