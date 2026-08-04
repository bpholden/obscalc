"""Atmospheric extinction, in magnitudes per unit airmass.

Port of ``mtham_trans.pro``.  Only Mt Hamilton is implemented, which is what
the APF needs; :func:`extinction_for` raises for anything else rather than
silently returning the wrong site's curve.
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


EXTINCTION = {"APF": mtham_trans, "Lick-3m": mtham_trans}


def extinction_for(telescope_name):
    """Extinction function for a telescope, by name."""
    try:
        return EXTINCTION[telescope_name.strip()]
    except KeyError:
        raise NotImplementedError(
            f"no extinction curve for telescope {telescope_name!r}; "
            f"have {sorted(EXTINCTION)}"
        ) from None
