"""Sky surface brightness, in AB magnitudes per square arcsecond.

Ports ``mtham_sky.pro`` and ``maunakea_sky.pro``.  Both interpolate an empirical
measurement; neither depends on the phase of the moon.  The ``phase`` argument is
kept so the signatures match, but every bundled measurement was taken at new
moon and nothing here uses it.

``maunakea_sky.pro`` also carried an analytic fallback: a table of Vega-system
sky brightness against wavelength and moon phase, bilinearly interpolated, used
for wavelengths outside the empirical measurement and (nominally) whenever the
caller passed ``/NOEMPIR``.  That path is **removed** here, by decision, in favour
of empirical measurements only.  Two consequences worth knowing:

* The ``/NOEMPIR`` option is gone.  It never worked anyway -- the keyword was
  declared ``NOEMPIRI=noempiri`` but tested as ``keyword_set(NOEMPIRIC)``, so
  every caller asking to skip the empirical model silently got it.
* There is no moon-phase dependence at all for Mauna Kea.  That matches what
  ``spec_calcs2n.pro`` already forced for DEIMOS, ESI and LRIS, each of which
  hardcoded ``phase = 0L  ;; Only New Moon so far``.  It is a change for Keck I
  and for HIRES, which passed a real phase to a table that only mattered outside
  the empirical range.

Because the analytic fallback is gone, wavelengths outside a measurement hold its
nearest measured value.  Use :func:`coverage` to find where that happens and
report it -- the models do not all span the optical.
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from . import config
from .idl_compat import interpol

_LICK_FILE = "lick_sky_d55_2011aug29.fits"

#: Empirical Mauna Kea sky models, in f_lambda, erg/s/cm^2/Angstrom/arcsec^2.
#: All are new-moon measurements.
MAUNAKEA_MODELS = {
    "deimos600": "mkea_sky_newmoon_DEIMOS_600_2011oct.fits",
    "deimos1200": "mkea_sky_newmoon_DEIMOS_1200_2011oct.fits",
}

#: ``mkea_sky_LRIS_both.fits`` (``flg_sky = 2``) is **not** offered.  Its fluxes
#: have a median of 0.23 where the two DEIMOS files sit near 5e-18, and no
#: ``BUNIT`` says what they are.  Put through the same f_lambda conversion the
#: IDL applies to every model, it yields sky brightnesses around -19 AB
#: mag/arcsec^2, which is unphysical -- ``maunakea_sky.pro`` produces the same
#: nonsense for ``flg_sky = 2``.  The file and that routine were both last
#: touched 2025-03-12, so this looks unfinished rather than intended.
UNUSABLE_MODELS = {
    "lris": (
        "mkea_sky_LRIS_both.fits is not in f_lambda like the other models "
        "(median 0.23 against 5e-18) and gives negative sky magnitudes; "
        "rescale it or supply BUNIT before using it"
    )
}

#: ``flg_sky`` in ``maunakea_sky.pro`` mapped onto the names above.
FLG_SKY = {0: "deimos600", 1: "deimos1200", 2: "lris"}

#: Default Mauna Kea model: the wider of the two usable files, 5001-9999 A.
#: Note this leaves nothing measured blueward of 5000 A -- see :func:`coverage`.
DEFAULT_MAUNAKEA_MODEL = "deimos600"

#: Model each Keck instrument's branch of ``spec_calcs2n.pro`` implied.  LRIS
#: selected ``flg_sky = 2``, which is unusable, so it falls back with the rest.
KECK_INSTRUMENT_MODELS = {
    "DEIMOS": "deimos600",  # flg_sky = 1 only for the 1200 line grating
    "ESI": "deimos600",
}

#: A dark optical sky sits in this range; outside it the units are wrong.
PLAUSIBLE_SKY_MAG = (15.0, 26.0)


@lru_cache(maxsize=None)
def _read_sky(filename):
    """Wavelength in HDU 0, f_lambda in HDU 1.

    Cached, standing in for the IDL COMMON blocks that held the same arrays.
    """
    path = config.resolve(filename, config.SKY_DIR)
    with fits.open(path) as hdus:
        wave = np.asarray(hdus[0].data, dtype=float)
        flam = np.asarray(hdus[1].data, dtype=float)
    return wave, flam


def _flam_to_ab(flam, wave):
    """f_lambda to AB magnitudes per square arcsecond."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fnu = flam / 3e10 * wave * (wave * 1e-8)
        return -np.log10(fnu) / 0.4 - 48.6


def coverage(model=None):
    """Wavelength range a sky model actually measures.

    With no argument, the Mt Hamilton measurement; otherwise one of
    :data:`MAUNAKEA_MODELS`.
    """
    filename = _LICK_FILE if model is None else _maunakea_file(model)
    wave, _ = _read_sky(filename)
    return float(wave.min()), float(wave.max())


def _maunakea_file(model):
    if model in UNUSABLE_MODELS:
        raise ValueError(f"sky model {model!r} is unusable: {UNUSABLE_MODELS[model]}")
    try:
        return MAUNAKEA_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown Mauna Kea sky model {model!r}; have {sorted(MAUNAKEA_MODELS)}"
        ) from None


def median_sky_magnitude(model=None):
    """Median sky brightness a model implies, for checking its units.

    A model in the wrong units gives a wildly wrong answer here; that is how
    ``mkea_sky_LRIS_both.fits`` was caught.
    """
    filename = _LICK_FILE if model is None else _maunakea_file(model)
    wave, flam = _read_sky(filename)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmedian(_flam_to_ab(flam, wave)))


def mtham_sky(wave, phase=0):
    """Mt Hamilton sky brightness. ``phase`` is ignored.

    Interpolates a single dark-sky measurement from 2011 Aug 29.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    sky_wave, sky_flam = _read_sky(_LICK_FILE)

    flam = interpol(sky_flam, sky_wave, wave)
    # mtham_sky.pro:56 wrote `where(wave LT mtham_swv)`, comparing two arrays of
    # different length, which IDL silently truncates to the shorter one.  The
    # intent was clearly the blue end of the table.
    flam = np.where(wave < sky_wave.min(), sky_flam[0], flam)

    return _flam_to_ab(flam, wave)


def maunakea_sky(wave, phase=0, model=DEFAULT_MAUNAKEA_MODEL):
    """Mauna Kea sky brightness from an empirical model. ``phase`` is ignored.

    Outside the model's measured range the nearest measured value is held; see
    :func:`coverage` and the module docstring.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    sky_wave, sky_flam = _read_sky(_maunakea_file(model))

    flam = interpol(sky_flam, sky_wave, wave)
    # No analytic fallback any more, so hold the ends rather than let interpol
    # extrapolate a line-filled spectrum into negative flux.
    flam = np.where(wave < sky_wave.min(), sky_flam[0], flam)
    flam = np.where(wave > sky_wave.max(), sky_flam[-1], flam)

    return _flam_to_ab(flam, wave)


def _maunakea_for_instrument(instrument_name):
    """Bind :func:`maunakea_sky` to the model an instrument implies."""
    model = KECK_INSTRUMENT_MODELS.get(
        (instrument_name or "").strip().upper(), DEFAULT_MAUNAKEA_MODEL
    )

    def sky(wave, phase=0):
        return maunakea_sky(wave, phase, model=model)

    sky.model = model
    return sky


SKY = {"APF": mtham_sky, "Lick-3m": mtham_sky}

#: Telescopes whose sky model depends on which instrument is in use, as the
#: nested ``case str_instr.name`` in ``spec_calcs2n.pro`` did.
SKY_BY_INSTRUMENT = {"KeckI": _maunakea_for_instrument, "KeckII": _maunakea_for_instrument}


def sky_for(telescope_name, instrument_name=None):
    """Sky brightness function for a telescope, and instrument where it matters."""
    key = telescope_name.strip()
    if key in SKY:
        return SKY[key]
    if key in SKY_BY_INSTRUMENT:
        return SKY_BY_INSTRUMENT[key](instrument_name)
    raise NotImplementedError(
        f"no sky model for telescope {telescope_name!r}; "
        f"have {sorted(set(SKY) | set(SKY_BY_INSTRUMENT))}"
    )
