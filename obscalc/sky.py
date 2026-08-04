"""Sky surface brightness, in AB magnitudes per square arcsecond.

Port of ``mtham_sky.pro``.  That routine ignores the moon phase entirely: it
interpolates a single empirical dark-sky measurement taken at Mt Hamilton on
2011 Aug 29.  The ``phase`` argument is kept so the signature matches the other
site functions, but it has no effect.
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from . import config
from .idl_compat import interpol

_SKY_FILE = "lick_sky_d55_2011aug29.fits"


@lru_cache(maxsize=None)
def _lick_sky():
    """Empirical Mt Hamilton dark sky: wavelength in HDU 0, f_lambda in HDU 1.

    Cached, standing in for the IDL COMMON block that held the same two arrays.
    """
    path = config.resolve(_SKY_FILE, config.SKY_DIR)
    with fits.open(path) as hdus:
        wave = np.asarray(hdus[0].data, dtype=float)
        flam = np.asarray(hdus[1].data, dtype=float)
    return wave, flam


def mtham_sky(wave, phase=0):
    """Sky brightness in AB mag/arcsec^2 at ``wave``. ``phase`` is ignored."""
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    sky_wave, sky_flam = _lick_sky()

    flam = interpol(sky_flam, sky_wave, wave)
    # mtham_sky.pro:56 wrote `where(wave LT mtham_swv)`, comparing two arrays of
    # different length, which IDL silently truncates to the shorter one.  The
    # intent was clearly the blue end of the table.
    flam = np.where(wave < sky_wave.min(), sky_flam[0], flam)

    with np.errstate(divide="ignore", invalid="ignore"):
        fnu = flam / 3e10 * wave * (wave * 1e-8)
        return -np.log10(fnu) / 0.4 - 48.6


SKY = {"APF": mtham_sky, "Lick-3m": mtham_sky}


def sky_for(telescope_name):
    """Sky brightness function for a telescope, by name."""
    try:
        return SKY[telescope_name.strip()]
    except KeyError:
        raise NotImplementedError(
            f"no sky model for telescope {telescope_name!r}; have {sorted(SKY)}"
        ) from None
