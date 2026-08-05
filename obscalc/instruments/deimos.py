"""DEIMOS on Keck II.

Ports ``x_initdeimos.pro`` and ``deimos_thruput.pro``.  DEIMOS reached
``spec_calcs2n`` through ``keck_calcs2n.pro`` with ``flg = 4``, as a single
channel, so nothing about the S/N arithmetic is special.

What is particular to DEIMOS is that the throughput depends on **two** settings:
the grating and the grating tilt, given as a central wavelength.  Between them
they select one of nine measured sensitivity files, each covering only the range
that configuration actually records.
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

#: Gratings and their resolving power at one native pixel, from
#: ``x_initdeimos.pro``.  The two 1200 line gratings differ in blaze, not
#: dispersion.
GRATINGS = {
    "600Z": 11538.5,
    "900Z": 17307.8,
    "1200G": 22727.3,
    "1200B": 22727.3,
}

#: Grating tilts the sensitivity measurements exist for, in Angstroms.
#: ``deimos_thruput.pro`` compared ``fix(str_instr.cwave)`` against these exact
#: integers, so nothing in between is available.
CENTRAL_WAVES = (5000, 6000, 7000, 8000)

#: Sensitivity file for each (grating, central wavelength).
#: ``600Z`` at 8000 A reuses the 750 nm measurement: no 850 nm one was ever taken,
#: and ``deimos_thruput.pro`` pointed both tilts at the same file.
#: The 1200 line gratings have a single measurement covering all four tilts.
SENS_FILES = {
    "600Z": {
        5000: "sens_DEIMOS_600_550nm.fits",
        6000: "sens_DEIMOS_600_650nm.fits",
        7000: "sens_DEIMOS_600_750nm.fits",
        8000: "sens_DEIMOS_600_750nm.fits",
    },
    "900Z": {
        5000: "sens_DEIMOS_900_500nm.fits",
        6000: "sens_DEIMOS_900_600nm.fits",
        7000: "sens_DEIMOS_900_700nm.fits",
        8000: "sens_DEIMOS_900_800nm.fits",
    },
    "1200G": dict.fromkeys(CENTRAL_WAVES, "sens_DEIMOS_1200G.fits"),
    "1200B": dict.fromkeys(CENTRAL_WAVES, "sens_DEIMOS_1200B.fits"),
}

#: ``x_initdeimos.pro`` and ``deimos_calcs2n_wrapper.pro`` both default the grating
#: to ``'1200'``, which is not one of the four names the routine then switches on,
#: so the default configuration ran into ``else: stop``.  The general-purpose 1200
#: line grating is used here instead.
DEFAULT_GRATING = "1200G"

#: Default tilt, from ``deimos_calcs2n_wrapper.pro``.
DEFAULT_CWAVE = 7000

#: Default wavelength grid, from ``deimos_calcs2n_wrapper.pro``.  Note this is
#: wider than any single configuration records; see :func:`sensitivity_range`.
DEFAULT_RANGE = (4000.0, 10000.0)

DEFAULT_SLIT = 1.0  # arcsec, from x_initdeimos.pro


def _validate(grating, cwave):
    """Check a configuration, returning ``(grating, int(cwave))``."""
    grating = str(grating).strip()
    if grating not in GRATINGS:
        raise ValueError(
            f"unknown DEIMOS grating {grating!r}; have {sorted(GRATINGS)}"
        )
    try:
        cwave = int(float(cwave))
    except (TypeError, ValueError):
        raise ValueError(f"central wavelength must be a number, got {cwave!r}") from None
    if cwave not in CENTRAL_WAVES:
        raise ValueError(
            f"no DEIMOS sensitivity measurement at a central wavelength of "
            f"{cwave} A; have {', '.join(str(c) for c in CENTRAL_WAVES)}"
        )
    return grating, cwave


def deimos_spectrograph(
    grating=DEFAULT_GRATING,
    cwave=DEFAULT_CWAVE,
    slit=DEFAULT_SLIT,
    bins=1,
    bind=1,
    str_tel=None,
):
    """Instrument parameters for DEIMOS.

    ``cwave`` is the grating tilt in Angstroms.  It selects the sensitivity
    measurement and nothing else -- it does not narrow the wavelength grid, so a
    calculation can be asked for wavelengths that configuration does not record.
    :func:`sensitivity_range` says where the measurement stops.
    """
    grating, cwave = _validate(grating, cwave)
    tel = str_tel or keck_telescope("KeckII")

    instr = Instrument(
        name="DEIMOS",
        # Chosen so 0.75 arcsec maps onto 4.5 pixels, matching the observed
        # resolution rather than the nominal optics.
        mag_perp=8.03,
        mag_para=8.03,
        pixel_size=15.0,
        R=GRATINGS[grating],
        grating=grating,
        cwave=float(cwave),
        readno=2.6,
        dark=4.0,  # electrons/pixel/hour
        swidth=slit,
        sheight=10.0,
        wvmnx=DEFAULT_RANGE,
        bins=bins,
        bind=bind,
    )
    instr.scale_perp = tel.plate_scale * instr.mag_perp * (instr.pixel_size / 1000.0)
    instr.scale_para = tel.plate_scale * instr.mag_para * (instr.pixel_size / 1000.0)
    return instr


def sens_file(grating=DEFAULT_GRATING, cwave=DEFAULT_CWAVE):
    """Sensitivity file a configuration uses."""
    grating, cwave = _validate(grating, cwave)
    return SENS_FILES[grating][cwave]


@lru_cache(maxsize=None)
def _sensitivity(filename):
    """Throughput measurement: HDU 1, columns ``WAV`` and ``EFF``, already 0-1.

    Columns are read by name because the order varies -- ``sens_DEIMOS_900_500nm``
    stores ``EFF`` first.
    """
    path = config.resolve(filename, config.THRUPUT_DIR)
    with fits.open(path) as hdus:
        data = hdus[1].data
        return (
            np.asarray(data["WAV"], dtype=float).ravel(),
            np.asarray(data["EFF"], dtype=float).ravel(),
        )


def sensitivity_range(grating=DEFAULT_GRATING, cwave=DEFAULT_CWAVE):
    """Wavelength range a configuration's measurement covers."""
    sens_wave, _ = _sensitivity(sens_file(grating, cwave))
    return float(sens_wave.min()), float(sens_wave.max())


def deimos_thruput(wave, instr=None, grating=None, cwave=None):
    """End-to-end throughput (0-1) at ``wave``.

    The configuration comes from ``instr`` unless ``grating``/``cwave`` override
    it.

    Outside the measured range the nearest measured value is held.
    ``deimos_thruput.pro`` extrapolated instead, clipping only at zero with
    ``> 0.``, which catches a curve falling negative but not one rising absurdly.
    The default 4000-10000 A grid is wider than every one of the nine
    measurements, and extrapolating the 600Z curve at its 5000 A tilt -- measured
    only to 8035 A -- reaches 0.848 by 10000 A, against a best measured efficiency
    of 0.358 across all DEIMOS configurations.  Use :func:`sensitivity_range` to
    see where a configuration stops being measured.

    The zero clip is kept: ``sens_DEIMOS_900_500nm`` holds 133 slightly negative
    efficiencies inside its own range.
    """
    if instr is not None:
        grating = grating or instr.grating
        cwave = cwave if cwave is not None else instr.cwave
    grating = grating or DEFAULT_GRATING
    cwave = DEFAULT_CWAVE if cwave is None else cwave

    wave = np.asarray(wave, dtype=float)
    sens_wave, sens_eff = _sensitivity(sens_file(grating, cwave))
    order = np.argsort(sens_wave)
    sens_wave, sens_eff = sens_wave[order], sens_eff[order]

    thru = interpol(sens_eff, sens_wave, wave)
    thru = np.where(wave < sens_wave[0], sens_eff[0], thru)
    thru = np.where(wave > sens_wave[-1], sens_eff[-1], thru)
    return np.maximum(thru, 0.0)


def unmeasured_ranges(wave, grating=DEFAULT_GRATING, cwave=DEFAULT_CWAVE):
    """Parts of ``wave`` outside the configuration's measurement, as (low, high).

    Throughput there is held at the nearest measured value rather than measured,
    so the counts are an extrapolation however plausible they look.
    """
    wave = np.asarray(wave, dtype=float)
    low, high = sensitivity_range(grating, cwave)
    ranges = []
    below = wave < low
    above = wave > high
    if below.any():
        ranges.append((float(wave[below].min()), float(wave[below].max())))
    if above.any():
        ranges.append((float(wave[above].min()), float(wave[above].max())))
    return ranges


class DeimosBackend(Backend):
    """Web backend for DEIMOS.

    ``slitwidth`` is a width in arcsec, as for Kast.  ``grating`` and ``cwave``
    together choose the sensitivity measurement.
    """

    name = "deimos"
    default_range = DEFAULT_RANGE

    def sides(self, wave, values):
        grating = str(values.get("grating") or DEFAULT_GRATING).strip()
        if grating not in GRATINGS:
            raise ParameterError(
                "Inappropriate value for the input parameter Grating: "
                f"expected one of {', '.join(sorted(GRATINGS))}"
            )
        try:
            _, cwave = _validate(grating, values.get("cwave", DEFAULT_CWAVE))
        except ValueError as exc:
            raise ParameterError(
                f"Inappropriate value for the input parameter Central "
                f"Wavelength: {exc}"
            ) from None

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

        tel = keck_telescope("KeckII")
        instr = deimos_spectrograph(
            grating=grating,
            cwave=cwave,
            slit=slit,
            bins=values["bins"],
            bind=values["bind"],
            str_tel=tel,
        )
        index = np.arange(np.size(wave))
        return tel, [Side("", instr, index, deimos_thruput(wave, instr))]
