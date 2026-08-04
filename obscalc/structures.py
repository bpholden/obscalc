"""Telescope, instrument and observation parameters.

These mirror the IDL structures ``telestruct``, ``instrstruct`` and
``obsstruct``, keeping the original field names so the two can be compared
side by side.
"""

from dataclasses import dataclass, field

from astropy.io import ascii as ascii_io


@dataclass
class Telescope:
    """IDL ``telestruct``."""

    name: str = ""
    area: float = 0.0  # collecting area, cm^2
    plate_scale: float = 0.0  # arcsec/mm at the focal plane


@dataclass
class Instrument:
    """IDL ``instrstruct``.

    ``bins`` and ``bind`` are spatial and dispersion binning.  Note that
    ``spec_calcs2n`` reads ``bind`` as the *dispersion* factor (``binc``) and
    ``bins`` as the *spatial* factor (``binr``), while
    ``apf_calcs2n_wrapper.pro`` filled ``bins`` from the first character of a
    string like ``"1x1"`` and ``bind`` from the second.  The names are kept as
    they were; :func:`parse_binning` is the one place that maps a "AxB" string
    onto them.
    """

    name: str = ""
    mag_perp: float = 0.0  # magnification perpendicular to dispersion
    mag_para: float = 0.0  # magnification parallel to dispersion
    pixel_size: float = 0.0  # microns
    scale_perp: float = 0.0  # arcsec per unbinned pixel, spatial
    scale_para: float = 0.0  # arcsec per unbinned pixel, dispersion
    R: float = 0.0  # resolving power
    mlambda: float = 0.0  # order number times central wavelength
    dark: float = 0.0  # e-/pixel/hour
    readno: float = 0.0  # read noise, e- rms
    dichroic: str = ""
    grating: str = ""
    cwave: float = 0.0  # central wavelength
    swidth: float = 0.0  # slit width, arcsec
    sheight: float = 0.0  # slit height, arcsec
    wvmnx: tuple = (0.0, 0.0)  # useful wavelength range, Angstroms
    bins: int = 1  # spatial binning
    bind: int = 1  # dispersion binning
    dely: float = 0.0


@dataclass
class Observation:
    """IDL ``obsstruct``.

    ``mtype`` selects the magnitude system: 1 = Vega/Johnson, 2 = AB.  (The
    comment in ``obsstruct__define.pro`` says "1=AB, 2=Johnson", which is the
    reverse of what ``spec_calcs2n`` actually does.)

    Defaults are those of ``x_obsinit`` with its Mauna Kea branch.
    """

    seeing: float = 0.7  # arcsec FWHM
    airmass: float = 1.1
    mphase: int = 0  # moon phase, days from new
    mstar: float = 17.0  # object magnitude
    filter: str = ""  # filter used to normalise the template
    mtype: int = 1
    exptime: float = 3600.0  # seconds
    redshift: float = 0.0
    template: str = ""
    vega_template: str = "alpha_lyr_stis_005.fits"


# Cards understood in an INFIL parameter file, mapped to the field they set and
# the type they are read as.  ``x_obsinit`` and ``x_initapfspec`` between them
# recognised exactly these.
_OBS_CARDS = {
    "SEEING": ("seeing", float),
    "AIRMASS": ("airmass", float),
    "MPHASE": ("mphase", int),
    "EXPTIME": ("exptime", float),
    "MAGNITUDE": ("mstar", float),
    "MAGTYPE": ("mtype", int),
}

_INSTR_CARDS = {
    "SWIDTH": ("swidth", float),
    "BINC": ("bind", int),
    "BINR": ("bins", int),
}


def read_infil(path):
    """Read a two-column ``CARD value`` parameter file into a dict."""
    table = ascii_io.read(
        path, format="no_header", names=("card", "value"), converters={"value": str}
    )
    return {
        str(row["card"]).strip(): str(row["value"]).strip() for row in table
    }


def apply_infil_to_observation(obs, cards):
    """Apply INFIL cards to an :class:`Observation`, as ``x_obsinit`` did."""
    for card, (attr, cast) in _OBS_CARDS.items():
        if card in cards:
            setattr(obs, attr, cast(float(cards[card])))
    return obs


def apply_infil_to_instrument(instr, cards):
    """Apply INFIL cards to an :class:`Instrument`, as ``x_initapfspec`` did.

    The ``DECKER`` card is returned rather than applied, because setting the
    decker is instrument specific.
    """
    for card, (attr, cast) in _INSTR_CARDS.items():
        if card in cards:
            setattr(instr, attr, cast(float(cards[card])))
    return cards.get("DECKER")


def parse_binning(binning):
    """Split a ``"AxB"`` binning string into ``(bins, bind)``.

    Matches ``apf_calcs2n_wrapper.pro``: the first field is spatial (``bins``),
    the second is dispersion (``bind``).
    """
    spatial, _, dispersion = str(binning).lower().partition("x")
    if not dispersion:
        raise ValueError(f"binning must look like '1x1', got {binning!r}")
    return int(spatial), int(dispersion)
