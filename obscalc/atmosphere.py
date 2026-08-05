"""Atmospheric extinction, in magnitudes per unit airmass.

Ports ``mtham_trans.pro`` (Mt Hamilton, read from a file) and
``maunakea_trans.pro`` (a table in the source, from CFHT Bulletin 19, 16, 1988).
:func:`extinction_for` raises for an unknown site rather than silently returning
the wrong curve.

The two differ in how they treat wavelengths outside their tables, and each
follows its own IDL: ``mtham_trans`` holds the end value, ``maunakea_trans``
extrapolates.
"""

from functools import lru_cache

import numpy as np
from astropy.io import ascii as ascii_io

from . import config
from .idl_compat import interpol

_EXTINCT_FILE = "mthamextinct.dat"


@lru_cache(maxsize=None)
def _extinction_table():
    """Mt Hamilton extinction curve, 3200-10870 Angstroms."""
    path = config.resolve(_EXTINCT_FILE, config.EXTINCTION_DIR)
    table = ascii_io.read(path, format="no_header", names=("wave", "mag"))
    return (
        np.asarray(table["wave"], dtype=float),
        np.asarray(table["mag"], dtype=float),
    )


def mtham_trans(wave):
    """Extinction in magnitudes at ``wave``, held flat outside the table."""
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    table_wave, table_mag = _extinction_table()

    extinct = np.empty(wave.shape, dtype=float)
    below = wave < table_wave.min()
    above = wave > table_wave.max()
    within = ~(below | above)

    extinct[within] = interpol(table_mag, table_wave, wave[within])
    extinct[below] = table_mag[table_wave.argmin()]
    extinct[above] = table_mag[table_wave.argmax()]
    return extinct


# Mauna Kea extinction, tabulated in maunakea_trans.pro from CFHT Bulletin 19,
# 16 (1988).  Magnitudes per airmass.
_MAUNAKEA_WAVE = np.array(
    [
        3000.0, 3100.0, 3200.0, 3300.0, 3400.0,
        3500.0, 3600.0, 3700.0, 3800.0, 3900.0,
        4000.0, 4250.0, 4500.0, 4750.0, 5000.0,
        5250.0, 5500.0, 5750.0, 6000.0, 6500.0,
        7000.0, 8000.0, 9000.0, 10000.0, 12000.0,
    ]
)
_MAUNAKEA_EXTINCT = np.array(
    [
        4.90, 1.37, 0.82, 0.57, 0.51,
        0.42, 0.37, 0.33, 0.30, 0.27,
        0.25, 0.21, 0.17, 0.14, 0.13,
        0.12, 0.12, 0.12, 0.11, 0.11,
        0.10, 0.07, 0.05, 0.04, 0.03,
    ]
)

#: Range the Mauna Kea table actually covers.
MAUNAKEA_RANGE = (float(_MAUNAKEA_WAVE[0]), float(_MAUNAKEA_WAVE[-1]))


def maunakea_trans(wave):
    """Extinction in magnitudes at ``wave`` for Mauna Kea.

    ``maunakea_trans.pro`` passed straight to ``interpol``, so this extrapolates
    outside 3000-12000 A rather than holding the end value the way
    :func:`mtham_trans` does.  That is kept, but note the extrapolation is only
    sane over a modest reach: continuing the last interval would reach zero
    extinction near 18000 A and go negative beyond, well outside the range of
    any optical spectrograph here.
    """
    return interpol(
        _MAUNAKEA_EXTINCT, _MAUNAKEA_WAVE, np.atleast_1d(np.asarray(wave, dtype=float))
    )


EXTINCTION = {
    "APF": mtham_trans,
    "Lick-3m": mtham_trans,
    "KeckI": maunakea_trans,
    "KeckII": maunakea_trans,
}


def extinction_for(telescope_name):
    """Extinction function for a telescope, by name."""
    try:
        return EXTINCTION[telescope_name.strip()]
    except KeyError:
        raise NotImplementedError(
            f"no extinction curve for telescope {telescope_name!r}; "
            f"have {sorted(EXTINCTION)}"
        ) from None
