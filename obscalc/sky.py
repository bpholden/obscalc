"""Sky surface brightness, in AB magnitudes per square arcsecond.

Ports ``mtham_sky.pro`` and ``maunakea_sky.pro``.  Both interpolate empirical
measurements; neither depends on the phase of the moon.  The ``phase`` argument is
kept so the signatures match, but every bundled measurement was taken at new
moon and nothing here uses it.

``maunakea_sky.pro`` also carried an analytic fallback: a table of Vega-system
sky brightness against wavelength and moon phase, bilinearly interpolated, used
for wavelengths outside the empirical measurement and (nominally) whenever the
caller passed ``/NOEMPIR``.  That path is **removed** here, by decision, in favour
of empirical measurements only.  Two consequences:

* The ``/NOEMPIR`` option is gone.  It never worked anyway -- the keyword was
  declared ``NOEMPIRI=noempiri`` but tested as ``keyword_set(NOEMPIRIC)``, so
  every caller asking to skip the empirical model silently got it.  Dropping the
  fallback also disposes of the second bug in that routine, where the moon-phase
  index reused ``ngd`` from the preceding wavelength search.
* There is no moon-phase dependence for Mauna Kea.  That matches what
  ``spec_calcs2n.pro`` already forced for DEIMOS, ESI and LRIS, each of which
  hardcoded ``phase = 0L  ;; Only New Moon so far``.

Outside a model's measured range the nearest measured value is held.  Use
:func:`coverage` to find where that happens and report it.

The LRIS sky measurements
-------------------------
``bsky.eps_pang_parcsec.fits`` and ``rsky.eps_pang_parcsec_onemicron.fits`` are in
**electrons per second per Angstrom per square arcsecond**, so they already
include the throughput of the telescope and instrument.  Feeding them to an
f_lambda conversion, which is what ``maunakea_sky.pro`` does for its
``flg_sky = 2`` model, gives sky brightnesses near -19 AB mag/arcsec^2 -- that
model was ``mkea_sky_LRIS_both.fits``, bit-identical to the two files above
summed across their overlap, and it is not used here.

Instead the throughput is divided back out to recover a true surface brightness,
which is what the engine wants and which any instrument can then use.  The
recovery is only as good as the throughput curve, and the configuration these
data were taken in -- blue grism 400/3400, red grating 600/5000, dichroic 500 --
has no sensitivity measurement in the xidl tree.  The nearest same-ruling D560
curves are used instead, and each channel is restricted to where it dominates.

How well that works is measurable.  Recovering the red channel with
``sens_LRISr_600_7500_D560`` reproduces the independent DEIMOS 600 sky to
**+0.07 mag** median over 5700-8190 A, which is the main evidence that the method
is sound; :data:`RED_VALIDATION` records it.  The red channel is kept only for
that check, since it adds no coverage the DEIMOS models lack.  The blue channel
is the useful one: it is the only Mauna Kea measurement here blueward of 5000 A.
"""

from functools import lru_cache

import numpy as np
from astropy.io import fits

from . import config
from .idl_compat import interpol

# Physical constants as spelled in the IDL.
_H_ERG_S = 6.626e-27
_C_ANGSTROM_S = 2.99792e18
_C_CM_S = 3e10

#: Keck collecting area, cm^2, from x_initkeck.pro.  Needed to undo the
#: throughput baked into the LRIS measurements.
_KECK_AREA = 723674.0

_LICK_FILE = "lick_sky_d55_2011aug29.fits"

#: Blue LRIS sky frame and the throughput curve used to undo it.  As observed:
#: grism 400/3400, dichroic 500, Keck I, 2017-05-28, airmass 1.32.  The nearest
#: available curve is the 600 l/mm grism blazed at 4000 A, closest in blaze to
#: the 400/3400's 3400 A; the 300/5000 grism blazes at 5000 A and gives a
#: recovered sky that darkens toward the red where the real sky brightens.
_LRIS_BLUE_SKY = "bsky.eps_pang_parcsec.fits"
_LRIS_BLUE_SENS = "sens_LRISb_600_4000_D560.fits"

#: Red LRIS sky frame and curve, kept for validation only.  As observed: grating
#: 600/5000, dichroic 500, 2017-05-27.
_LRIS_RED_SKY = "rsky.eps_pang_parcsec_onemicron.fits"
_LRIS_RED_SENS = "sens_LRISr_600_7500_D560.fits"

#: Above this the blue channel is past the dichroic 500 handover and its
#: recovered sky is no longer reliable.
_LRIS_BLUE_MAX = 5000.0

#: Below this the DEIMOS 600 measurement is still ramping up off its blue edge:
#: it reads 24.1 mag at 5001 A against 22.5 at 5200 A.
_DEIMOS600_MIN = 5200.0

#: Median offset and scatter of the red-channel recovery against the DEIMOS 600
#: model over 5700-8190 A.  Recorded because it is the evidence that undoing the
#: throughput works; see :func:`validate_against_deimos`.
RED_VALIDATION = (0.07, 0.31)

#: A dark optical sky sits in this range; outside it the units are wrong.
PLAUSIBLE_SKY_MAG = (15.0, 26.0)


@lru_cache(maxsize=None)
def _read_two_hdu(filename):
    """Wavelength in HDU 0, f_lambda in HDU 1 -- the DEIMOS and Lick layout.

    Cached, standing in for the IDL COMMON blocks that held the same arrays.
    """
    path = config.resolve(filename, config.SKY_DIR)
    with fits.open(path) as hdus:
        return (
            np.asarray(hdus[0].data, dtype=float),
            np.asarray(hdus[1].data, dtype=float),
        )


@lru_cache(maxsize=None)
def _read_wcs_spectrum(filename):
    """One-dimensional image with a linear WCS -- the LRIS layout."""
    path = config.resolve(filename, config.SKY_DIR)
    with fits.open(path) as hdus:
        header = hdus[0].header
        values = np.asarray(hdus[0].data, dtype=float).ravel()
    step = header.get("CDELT1", header.get("CD1_1"))
    if step is None:
        raise ValueError(f"{path.name} has no CDELT1 or CD1_1")
    reference = header.get("CRPIX1", 1.0)
    wave = header["CRVAL1"] + (np.arange(values.size) - (reference - 1.0)) * step
    return wave, values


@lru_cache(maxsize=None)
def _read_sensitivity(filename):
    """Throughput measurement: HDU 2, vector columns ``WAV`` and ``EFF``."""
    path = config.resolve(filename, config.THRUPUT_DIR)
    with fits.open(path) as hdus:
        data = hdus[2].data
        return (
            np.asarray(data["WAV"], dtype=float).ravel(),
            np.asarray(data["EFF"], dtype=float).ravel(),
        )


def _flam_to_ab(flam, wave):
    """f_lambda to AB magnitudes per square arcsecond."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fnu = flam / _C_CM_S * wave * (wave * 1e-8)
        return -np.log10(fnu) / 0.4 - 48.6


def _undo_throughput(sky_file, sens_file, wave_min=None, wave_max=None):
    """Recover f_lambda per square arcsecond from a detected-electron spectrum.

    The measurement is electrons/s/Angstrom/arcsec^2, so dividing by the
    collecting area and the end-to-end throughput gives photons, and multiplying
    by the photon energy gives f_lambda.  Wavelengths where the throughput curve
    has no measurement are dropped, as are any outside the requested limits.
    """
    wave, electrons = _read_wcs_spectrum(sky_file)
    sens_wave, sens_eff = _read_sensitivity(sens_file)

    thru = np.interp(wave, sens_wave, sens_eff, left=np.nan, right=np.nan)
    keep = np.isfinite(thru) & (thru > 0)
    if wave_min is not None:
        keep &= wave >= wave_min
    if wave_max is not None:
        keep &= wave <= wave_max
    if not keep.any():
        raise ValueError(
            f"{sky_file} and {sens_file} share no wavelengths in "
            f"[{wave_min}, {wave_max}]"
        )

    wave = wave[keep]
    photons = electrons[keep] / (_KECK_AREA * thru[keep])
    flam = photons * _H_ERG_S * _C_ANGSTROM_S / wave
    return wave, flam


@lru_cache(maxsize=None)
def _lris_blue_flam():
    """LRIS blue sky as f_lambda per square arcsecond, 3101-5000 A."""
    return _undo_throughput(
        _LRIS_BLUE_SKY, _LRIS_BLUE_SENS, wave_max=_LRIS_BLUE_MAX
    )


@lru_cache(maxsize=None)
def _lris_red_flam():
    """LRIS red sky as f_lambda per square arcsecond.  Validation only."""
    return _undo_throughput(_LRIS_RED_SKY, _LRIS_RED_SENS)


@lru_cache(maxsize=None)
def _combined_flam():
    """LRIS blue below 5000 A joined to DEIMOS 600 above 5200 A.

    The 5000-5200 A bridge is spanned by interpolation: the blue channel has
    fallen off the dichroic 500 handover there and the DEIMOS measurement has
    not yet come up off its blue edge.  Nothing bundled measures that gap.
    """
    blue_wave, blue_flam = _lris_blue_flam()
    deimos_wave, deimos_flam = _read_two_hdu(MAUNAKEA_FILES["deimos600"])
    red = deimos_wave >= _DEIMOS600_MIN
    return (
        np.concatenate([blue_wave, deimos_wave[red]]),
        np.concatenate([blue_flam, deimos_flam[red]]),
    )


#: File-backed Mauna Kea models, in f_lambda per square arcsecond.
MAUNAKEA_FILES = {
    "deimos600": "mkea_sky_newmoon_DEIMOS_600_2011oct.fits",
    "deimos1200": "mkea_sky_newmoon_DEIMOS_1200_2011oct.fits",
}

#: Every Mauna Kea model, as a provider returning ``(wave, f_lambda)``.
MAUNAKEA_MODELS = {
    "deimos600": lambda: _read_two_hdu(MAUNAKEA_FILES["deimos600"]),
    "deimos1200": lambda: _read_two_hdu(MAUNAKEA_FILES["deimos1200"]),
    "lris_blue": _lris_blue_flam,
    "lris_red": _lris_red_flam,
    "combined": _combined_flam,
}

#: ``flg_sky`` in ``maunakea_sky.pro``.  2 selected ``mkea_sky_LRIS_both.fits``,
#: which is superseded: see the module docstring.
FLG_SKY = {0: "deimos600", 1: "deimos1200", 2: "lris_blue"}

#: Default: the widest usable coverage, 3101-9999 A.
DEFAULT_MAUNAKEA_MODEL = "combined"

#: Model each Keck instrument's branch of ``spec_calcs2n.pro`` implied.  Anything
#: not listed reached the analytic fallback, which no longer exists.
KECK_INSTRUMENT_MODELS = {
    "DEIMOS": "deimos600",  # flg_sky = 1 only for the 1200 line grating
    "ESI": "deimos600",
}


def _model(name):
    try:
        return MAUNAKEA_MODELS[name]()
    except KeyError:
        raise ValueError(
            f"unknown Mauna Kea sky model {name!r}; have {sorted(MAUNAKEA_MODELS)}"
        ) from None


def coverage(model=None):
    """Wavelength range a sky model measures.

    With no argument, the Mt Hamilton measurement.
    """
    wave, _ = _read_two_hdu(_LICK_FILE) if model is None else _model(model)
    return float(wave.min()), float(wave.max())


def median_sky_magnitude(model=None):
    """Median sky brightness a model implies, for checking its units.

    A model in the wrong units gives a wildly wrong answer here; that is how
    ``mkea_sky_LRIS_both.fits`` was caught.
    """
    wave, flam = _read_two_hdu(_LICK_FILE) if model is None else _model(model)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmedian(_flam_to_ab(flam, wave)))


def _interpolate_held(wave, model_wave, model_flam):
    """Interpolate, holding the end values outside the measured range."""
    flam = interpol(model_flam, model_wave, wave)
    flam = np.where(wave < model_wave.min(), model_flam[0], flam)
    return np.where(wave > model_wave.max(), model_flam[-1], flam)


def mtham_sky(wave, phase=0):
    """Mt Hamilton sky brightness. ``phase`` is ignored.

    Interpolates a single dark-sky measurement from 2011 Aug 29.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    sky_wave, sky_flam = _read_two_hdu(_LICK_FILE)

    flam = interpol(sky_flam, sky_wave, wave)
    # mtham_sky.pro:56 wrote `where(wave LT mtham_swv)`, comparing two arrays of
    # different length, which IDL silently truncates to the shorter one.  The
    # intent was clearly the blue end of the table.
    flam = np.where(wave < sky_wave.min(), sky_flam[0], flam)

    return _flam_to_ab(flam, wave)


def maunakea_sky(wave, phase=0, model=DEFAULT_MAUNAKEA_MODEL):
    """Mauna Kea sky brightness from empirical measurements. ``phase`` is ignored.

    Outside the model's measured range the nearest measured value is held; see
    :func:`coverage` and the module docstring.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    model_wave, model_flam = _model(model)
    return _flam_to_ab(_interpolate_held(wave, model_wave, model_flam), wave)


def validate_against_deimos(model="lris_red", wave_min=5700.0, wave_max=8190.0):
    """Compare a recovered model with the independent DEIMOS 600 measurement.

    Returns ``(median_offset, scatter)`` in magnitudes.  This is the check that
    says whether undoing the throughput recovered the right absolute level.
    """
    model_wave, model_flam = _model(model)
    inside = (model_wave > wave_min) & (model_wave < wave_max)
    if not inside.any():
        raise ValueError(f"{model!r} does not cover {wave_min}-{wave_max} A")

    wave = model_wave[inside]
    recovered = _flam_to_ab(model_flam[inside], wave)
    reference = maunakea_sky(wave, model="deimos600")
    difference = recovered - reference
    return float(np.median(difference)), float(np.std(difference))


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
SKY_BY_INSTRUMENT = {
    "KeckI": _maunakea_for_instrument,
    "KeckII": _maunakea_for_instrument,
}


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
