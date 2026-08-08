"""HIRES on Keck I.

Ports ``x_inithires.pro``, ``hires_thruput.pro`` and the parts of
``hires_calcs2n.pro`` that are not already in :func:`obscalc.s2n.spec_calcs2n`.

HIRES differs from the other instruments here in where its throughput comes
from.  Kast and APF interpolate a measured sensitivity curve; HIRES is a
cross-dispersed echelle, and ``hires_thruput.pro`` models it:

1. The echelle order containing a wavelength is ``m = long(MLAMBDA / wave)``,
   whose blaze centre is ``MLAMBDA / m`` and whose free spectral range is
   ``centre / m`` -- about 70 A wide near 5000 A.
2. Throughput is read from a table **at the order centre**, not at the
   wavelength, so it is constant across each order.  Since the order centre is
   the peak of the blaze, this is a best-case-within-the-order convention.
3. Which of two tabulated cross-disperser curves applies depends on the order
   centre: the blue setting below 3800 A, the red one above.
4. The blaze function itself, ``(sin(gamma)/gamma)^2`` with
   ``gamma = pi (centre - wave) / fsr``, is applied by default, so the reported
   throughput is what a wavelength actually reaches rather than its order's peak.
   ``hires_calcs2n.pro`` never applied it: it passed ``BLAZE=blaze`` with
   ``blaze`` undefined, which in IDL leaves ``keyword_set(BLAZE)`` false, and for
   new HIRES it did not pass the keyword at all.  ``blaze=False`` restores that
   behaviour, and raises the median throughput by about 2.5x.

``x_inithires.pro`` also defines a third configuration, ``flg = 3``, for MTHR on
the TMT.  That is a different instrument on a different telescope with its own
throughput routine (``mthr_thruput.pro``) and is not ported.
"""

from functools import lru_cache

import numpy as np

from ..echelle import blaze_efficiency, echelle_orders
from ..idl_compat import interpol
from ..s2n import Side
from ..structures import Instrument
from ..telescopes import keck_telescope
from .base import Backend, ParameterError

__all__ = [
    "CCD_BOOST_RANGE",
    "CROSS_DISPERSER_BREAK",
    "DECKERS",
    "DEFAULT_DECKER",
    "DEFAULT_EPOCH",
    "DEFAULT_RANGE",
    "EPOCHS",
    "MLAMBDA",
    "HiresBackend",
    "blaze_efficiency",
    "ccd_boost",
    "cross_disperser_order",
    "echelle_orders",
    "hires_spectrograph",
    "hires_thruput",
    "set_decker",
]

#: Slit width and height in arcsec for each decker, from ``hires_setdecker``.
DECKERS = {
    "B2": (0.574, 7.0),
    "B5": (0.861, 7.0),
    "C1": (0.861, 7.0),
    "C5": (1.148, 7.0),
    "D1": (1.148, 14.0),
    "D3": (1.72, 7.0),
    "E4": (0.400, 7.0),
    "E5": (0.800, 1.0),
}

#: Detector epochs.  ``flg = 1`` and ``flg = 2`` in ``x_inithires.pro``; the
#: resolving power scales inversely with pixel size from the original 24 micron
#: detector, and the newer one also gets the sensitivity boost below.
EPOCHS = {
    "old": {"pixel_size": 24.0, "readno": 4.3, "dark": 2.0, "new_ccd": False},
    "new": {"pixel_size": 15.0, "readno": 2.2, "dark": 1.0, "new_ccd": True},
}

DEFAULT_EPOCH = "new"  # what hires_calcs2n_wrapper.pro hardcodes
DEFAULT_DECKER = "C5"

#: Order centres blueward of this use the blue cross-disperser curve.
CROSS_DISPERSER_BREAK = 3800.0

#: ``2 sigma sin(delta) cos(theta)`` for the echelle: 52.676 grooves/mm at a
#: 70.43 degree blaze.  Order m covers wavelengths near ``MLAMBDA / m``.
MLAMBDA = 356385.3016

#: Resolving power at the original 24 micron pixels.
_R_AT_24_MICRON = 135000.0

#: Default wavelength grid, from ``hires_calcs2n_wrapper.pro``.
DEFAULT_RANGE = (3000.0, 9500.0)

# Spectrograph throughput against wavelength, tabulated by S. Vogt (1993) for the
# two cross-disperser settings.  Index 0 is cross-disperser order 1 (red), index
# 1 is order 2 (blue).
_THRU_WAVE = np.array(
    [3000.0, 3200.0, 3500.0, 3800.0, 4000.0, 4500.0, 5000.0, 6000.0, 7000.0,
     8000.0, 9500.0]
)
_THRU_BY_ORDER = {
    1: np.array(
        [1e-4, 2e-4, 0.008, 0.020, 0.035, 0.061, 0.077, 0.080, 0.068, 0.051,
         0.017]
    ),
    2: np.array(
        [0.003, 0.008, 0.020, 0.019, 0.018, 0.009, 0.003, 3e-4, 3e-4, 3e-4,
         3e-4]
    ),
}

# Multiplicative sensitivity gain of the newer detector, from
# hires_thru_newccd.  Wavelength, boost.
_CCD_BOOST = np.array(
    [
        [3153.90, 11.21], [3183.67, 7.03], [3213.44, 9.32], [3243.22, 8.04],
        [3272.99, 7.56], [3302.76, 6.77], [3332.53, 6.09], [3362.30, 5.75],
        [3426.54, 3.76], [3462.23, 3.97], [3497.92, 3.61], [3533.61, 3.43],
        [3569.30, 3.14], [3604.99, 2.83], [3640.68, 2.61], [3676.37, 2.49],
        [3712.06, 2.19], [3831.73, 1.89], [3876.29, 1.82], [3920.85, 1.75],
        [3965.41, 1.69], [4054.16, 1.86], [4104.22, 1.84], [4154.28, 1.79],
        [4204.34, 1.76], [4254.40, 1.71], [4304.46, 1.70], [4354.52, 1.66],
        [4404.58, 1.65], [4454.64, 1.63], [4569.19, 1.52], [4635.40, 1.53],
        [4701.61, 1.53], [4767.82, 1.54], [4834.03, 1.53], [4900.24, 1.52],
        [4966.45, 1.49], [5032.66, 1.47], [5098.87, 1.43], [5165.08, 1.40],
        [5399.52, 1.34], [5491.01, 1.32], [5582.50, 1.31], [5673.99, 1.27],
        [5765.48, 1.24], [5856.97, 1.23], [5940.00, 1.37], [6046.00, 1.37],
        [6152.00, 1.34], [6258.00, 1.32], [6364.00, 1.27], [6479.53, 1.11],
        [6606.56, 1.09], [6733.59, 1.07], [6860.62, 1.09], [6987.65, 1.08],
        [7424.34, 1.08], [7589.31, 1.13], [7754.28, 1.16], [7919.70, 1.37],
        [8103.85, 1.45], [8288.00, 1.51], [8485.40, 1.58], [8697.50, 1.67],
        [8909.60, 1.78], [9378.60, 2.22], [9639.10, 2.16], [9899.60, 2.18],
        [10160.10, 2.04],
    ]
)

#: Range the detector-boost table covers.
CCD_BOOST_RANGE = (float(_CCD_BOOST[0, 0]), float(_CCD_BOOST[-1, 0]))


def set_decker(instr, decker):
    """Set slit width and height on ``instr`` from a decker name."""
    try:
        instr.swidth, instr.sheight = DECKERS[str(decker).strip()]
    except KeyError:
        raise ValueError(
            f"unknown HIRES decker {decker!r}; have {sorted(DECKERS)}"
        ) from None
    return instr


def hires_spectrograph(
    decker=DEFAULT_DECKER, epoch=DEFAULT_EPOCH, bins=2, bind=1, str_tel=None
):
    """Instrument parameters for HIRES.

    ``epoch`` selects the detector: ``"old"`` or ``"new"``.  It is recorded in
    the ``grating`` field, which HIRES has no other use for, so that
    :func:`hires_thruput` can read it back off the instrument.

    ``bins`` defaults to 2, matching ``x_inithires.pro``.  Note also that when no
    decker was given, ``x_inithires.pro`` called the default ``'C5'`` but set a
    0.5 per cent narrower slit than ``hires_setdecker`` gives for C5 (1.1 against
    1.148 arcsec); the table value is used here.
    """
    try:
        settings = EPOCHS[epoch]
    except KeyError:
        raise ValueError(
            f"unknown HIRES epoch {epoch!r}; have {sorted(EPOCHS)}"
        ) from None

    tel = str_tel or keck_telescope("KeckI")

    instr = Instrument(
        name="HIRES",
        mag_perp=5.776,
        mag_para=8.6713,
        pixel_size=settings["pixel_size"],
        R=_R_AT_24_MICRON * 24.0 / settings["pixel_size"],
        mlambda=MLAMBDA,
        dely=0.154693,  # = f2 * Ac
        grating=epoch,
        readno=settings["readno"],
        dark=settings["dark"],
        bins=bins,
        bind=bind,
        wvmnx=DEFAULT_RANGE,
    )
    instr.scale_perp = tel.plate_scale * instr.mag_perp * (instr.pixel_size / 1000.0)
    instr.scale_para = tel.plate_scale * instr.mag_para * (instr.pixel_size / 1000.0)

    return set_decker(instr, decker)


def cross_disperser_order(centre):
    """Which cross-disperser curve applies: 2 blueward of 3800 A, else 1.

    The APF has no equivalent -- ``apf_calcs2n.pro`` hardwires its ``iorder`` to
    1 -- so this stays here rather than moving to :mod:`obscalc.echelle`.
    """
    return np.where(np.asarray(centre, dtype=float) < CROSS_DISPERSER_BREAK, 2, 1)


@lru_cache(maxsize=None)
def _ccd_boost_table():
    return _CCD_BOOST[:, 0].copy(), _CCD_BOOST[:, 1].copy()


def ccd_boost(wave):
    """Sensitivity gain of the newer detector, held flat outside its table.

    ``hires_thru_newccd`` passed straight to ``interpol``, which extrapolates.
    That is untenable here: the table starts at 3153.9 A while HIRES is used from
    3000 A, and continuing the first interval would treble the boost to 32.8 by
    3000 A -- turning a 0.3 per cent throughput into 9.8 per cent.  A detector
    quantum efficiency ratio cannot be extrapolated that way, so the end values
    are held instead.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    table_wave, table_boost = _ccd_boost_table()
    boost = interpol(table_boost, table_wave, wave)
    boost = np.where(wave < table_wave[0], table_boost[0], boost)
    return np.where(wave > table_wave[-1], table_boost[-1], boost)


def hires_thruput(wave, instr=None, blaze=True, epoch=None):
    """End-to-end throughput (0-1) at ``wave``.

    ``epoch`` defaults to whatever ``instr`` records, else :data:`DEFAULT_EPOCH`.

    ``blaze`` applies the echelle blaze function, and is **on** by default so the
    result is the throughput actually reached at each wavelength.
    ``hires_calcs2n.pro`` never switched it on -- it passed ``BLAZE=blaze`` with
    ``blaze`` undefined -- so it reported the peak value within every order.  Pass
    ``blaze=False`` to reproduce that.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    if epoch is None:
        epoch = (instr.grating if instr is not None else None) or DEFAULT_EPOCH
    if epoch not in EPOCHS:
        raise ValueError(f"unknown HIRES epoch {epoch!r}; have {sorted(EPOCHS)}")

    _, centre, fsr = echelle_orders(wave, instr.mlambda if instr else MLAMBDA)
    order = cross_disperser_order(centre)

    # The table is read at the order centre, so throughput is flat across an
    # order.  Each cross-disperser setting has its own curve.
    thru = np.empty(wave.shape)
    for setting, table in _THRU_BY_ORDER.items():
        selected = order == setting
        if selected.any():
            thru[selected] = interpol(table, _THRU_WAVE, centre[selected])

    if blaze:
        thru = thru * blaze_efficiency(wave, centre, fsr)
    if EPOCHS[epoch]["new_ccd"]:
        thru = thru * ccd_boost(wave)

    return thru


class HiresBackend(Backend):
    """Web backend for HIRES.

    ``slitwidth`` is a decker name here, as it is for APF, not a width in arcsec
    -- the form posts things like ``C5``.  A single detector covers the whole
    range, so there is one side.
    """

    name = "hires"
    default_range = DEFAULT_RANGE

    def sides(self, wave, values):
        decker = str(values.get("slitwidth") or DEFAULT_DECKER).strip()
        if decker not in DECKERS:
            raise ParameterError(
                "Inappropriate value for the input parameter Decker: "
                f"expected one of {', '.join(sorted(DECKERS))}"
            )
        epoch = str(values.get("epoch") or DEFAULT_EPOCH).strip()
        if epoch not in EPOCHS:
            raise ParameterError(
                "Inappropriate value for the input parameter Epoch: "
                f"expected one of {', '.join(sorted(EPOCHS))}"
            )

        tel = keck_telescope("KeckI")
        instr = hires_spectrograph(
            decker=decker,
            epoch=epoch,
            bins=values["bins"],
            bind=values["bind"],
            str_tel=tel,
        )
        # On unless asked otherwise, as for the APF and as ``hires_thruput``
        # itself defaults.  Pass blaze=false for ``hires_calcs2n.pro``'s
        # per-order peak throughput.
        blaze = str(values.get("blaze", "true")).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        index = np.arange(np.size(wave))
        return tel, [Side("", instr, index, hires_thruput(wave, instr, blaze=blaze))]
