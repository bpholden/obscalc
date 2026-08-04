"""Count rates and signal-to-noise for a generic slit spectrograph.

Port of ``spec_calcs2n.pro``.
"""

from dataclasses import dataclass

import numpy as np

from .atmosphere import extinction_for
from .idl_compat import idl_long
from .photometry import generate_template, zero_point_flux
from .sky import sky_for
from .slit import gauss_slit


@dataclass
class S2NResult:
    """Per-wavelength count rates and noise.

    The first ten fields are those of the IDL ``fstrct``; the rest are
    intermediate quantities the IDL recomputed in its callers.  Counts are
    electrons per binned pixel over the whole exposure.
    """

    wave: np.ndarray  # Angstroms
    sn: np.ndarray  # signal-to-noise per binned pixel
    star: np.ndarray  # object counts
    tnoise: np.ndarray  # total noise
    extinct: np.ndarray  # atmospheric extinction, mag/airmass
    noise: float  # read noise summed over the extraction, e- rms
    ndark: float  # dark counts
    slit0: float  # fraction of the image passing the slit
    thru: np.ndarray  # end-to-end throughput
    sky: np.ndarray  # sky counts
    n0: np.ndarray  # photons/s/cm^2/Angstrom for a 0-mag source

    magsky: np.ndarray  # sky brightness, AB mag/arcsec^2
    pixel: np.ndarray  # Angstroms per binned pixel
    projslit: np.ndarray  # Angstroms per resolution element
    columns: float  # pixels the slit width projects to
    rows: int  # pixels extracted in the spatial direction
    nsky: float  # sky rows per object row
    mstar: float  # object magnitude actually used
    mtype: int  # magnitude system actually used
    binc: int  # dispersion binning used
    binr: int  # spatial binning used

    @property
    def sn_per_resolution_element(self):
        """S/N per resolution element rather than per pixel."""
        return self.sn * np.sqrt(self.columns)


def spec_calcs2n(wave, thru, str_tel, str_instr, str_obs):
    """Signal-to-noise for an observation.

    ``wave`` is in Angstroms and ``thru`` the matching end-to-end throughput
    (0-1).  ``str_tel``, ``str_instr`` and ``str_obs`` are
    :class:`~obscalc.structures.Telescope`, :class:`~obscalc.structures.Instrument`
    and :class:`~obscalc.structures.Observation`.

    This differs from ``spec_calcs2n.pro`` in one place.  The IDL always applied
    the sky-subtraction noise penalty ``(1 + 1/nsky)``, but ``nsky`` -- the
    number of sky rows per object row -- goes negative whenever the extraction
    window is taller than the slit, which happens for the APF ``W`` decker at
    seeing worse than about 1.0 arcsec.  That made ``tnoise`` the square root of
    a negative number and the S/N NaN, which is why the IDL wrappers filtered
    their output through ``where(finite(s2n))``.  Here, as in
    ``apf_calcs2n.pro``, the penalty is dropped when no sky rows are available.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    thru = np.atleast_1d(np.asarray(thru, dtype=float))
    if wave.shape != thru.shape:
        raise ValueError(
            f"wave and thru must have the same shape, got {wave.shape} and {thru.shape}"
        )
    if str_obs.seeing <= 0:
        raise ValueError(f"seeing must be positive, got {str_obs.seeing}")

    height = str_instr.sheight
    width = str_instr.swidth
    dark = str_instr.dark

    # spec_calcs2n.pro reads `bind` as the dispersion factor and `bins` as the
    # spatial one; the local names follow the IDL.
    binc = str_instr.bind
    binr = str_instr.bins

    seeing = str_obs.seeing
    phase = str_obs.mphase
    air = str_obs.airmass
    time = str_obs.exptime

    mstar = 17.0 if str_obs.mstar < 1 else str_obs.mstar
    mtype = 2 if str_obs.mtype == 0 else str_obs.mtype
    n0 = zero_point_flux(mtype, wave)

    # Spreading the light out: slit losses, and the extraction window in pixels.
    slit0 = gauss_slit(width / seeing, height / seeing, 0.0, 0.0)
    rows = int(idl_long(3 * seeing / str_instr.scale_perp + 0.999))
    columns = float(max(2.0, min(width, 3.0 * seeing) / str_instr.scale_para))
    slit_rows = int(idl_long(height / str_instr.scale_perp + 0.999))
    nsky = (slit_rows - rows) / rows

    read = str_instr.readno * np.sqrt(rows / float(binr))

    pixel = binc * (wave / str_instr.R)
    projslit = columns * (wave / str_instr.R)
    slarea = 3 * seeing * width

    extinct = extinction_for(str_tel.name)(wave)
    slit1 = slit0 * 10.0 ** (-0.4 * extinct * air)
    magsky = sky_for(str_tel.name)(wave, phase)

    # A template plus a normalising filter overrides the flat zero point.
    if str_obs.template and str_obs.filter:
        n0 = generate_template(str_obs, wave)

    star = n0 * 10.0 ** (-0.4 * mstar) * str_tel.area * thru * slit1 * pixel * time
    sky = n0 * 10.0 ** (-0.4 * magsky) * slarea * str_tel.area * thru * pixel * time
    ndark = binc * dark * rows * time / 3600.0
    noise = read

    background = noise * noise + sky + ndark
    if nsky > 0:
        tnoise = np.sqrt(star + (1.0 + 1.0 / nsky) * background)
    else:
        tnoise = np.sqrt(star + background)

    with np.errstate(divide="ignore", invalid="ignore"):
        sn = star / tnoise

    return S2NResult(
        wave=wave,
        sn=sn,
        star=star,
        tnoise=tnoise,
        extinct=extinct,
        noise=float(noise),
        ndark=float(ndark),
        slit0=slit0,
        thru=thru,
        sky=sky,
        n0=n0,
        magsky=magsky,
        pixel=pixel,
        projslit=projslit,
        columns=columns,
        rows=rows,
        nsky=nsky,
        mstar=mstar,
        mtype=mtype,
        binc=binc,
        binr=binr,
    )
