"""Echelle order geometry and the blaze function.

Shared by the two cross-dispersed instruments here, HIRES and the APF's Levy.
Both tabulate their throughput once per order, at the blaze centre, so both need
to know which order a wavelength falls in and how far down the blaze it sits.

``MLAMBDA`` in the instrument structure is ``2 sigma sin(delta) cos(theta)`` for
the echelle, so order ``m`` is centred on ``MLAMBDA / m`` and spans one free
spectral range, ``centre / m``.
"""

import numpy as np

from .idl_compat import idl_long


#: Relative nudge applied before truncating to an order number.  ``mlambda /
#: (mlambda / m)`` can land a hair below ``m`` in floating point, and IDL's
#: ``long()`` truncates, so a wavelength given as an exact order centre would be
#: assigned the order below it -- for the APF that happens for 3 of its 63 order
#: centres.  A relative 1e-12 cannot move the boundary for a wavelength genuinely
#: inside an order, so this only repairs the exact-centre case.
_ORDER_EPSILON = 1e-12


def echelle_orders(wave, mlambda):
    """Order number, blaze centre and free spectral range at ``wave``.

    The order is ``long(mlambda / wave)``, truncating toward zero as IDL does,
    with the epsilon described at :data:`_ORDER_EPSILON`.
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    if np.any(wave <= 0):
        raise ValueError("wavelengths must be positive")
    order = idl_long(mlambda / wave * (1.0 + _ORDER_EPSILON))
    if np.any(order < 1):
        raise ValueError(
            f"wavelengths above {mlambda:.0f} A fall outside the echelle format"
        )
    centre = mlambda / order
    return order, centre, centre / order


def blaze_efficiency(wave, centre, fsr):
    """Blaze function ``(sin(gamma)/gamma)^2``, peaking at the order centre.

    ``gamma = pi (centre - wave) / fsr``, so this is 1 at the centre, about 0.405
    half a free spectral range away, and zero a full one away.
    """
    wave = np.asarray(wave, dtype=float)
    gamma = np.pi * (np.asarray(centre, dtype=float) - wave) / np.asarray(
        fsr, dtype=float
    )
    blaze = np.ones(np.broadcast(wave, gamma).shape)
    nonzero = gamma != 0.0
    blaze[nonzero] = (np.sin(gamma[nonzero]) / gamma[nonzero]) ** 2
    return blaze


def order_centre_throughput(wave, mlambda, table_wave, table_eff, blaze=True):
    """Throughput for a per-order measurement, optionally times the blaze.

    ``table_wave``/``table_eff`` are a throughput measurement tabulated at order
    centres.  Each wavelength takes the value of *its own* order, so the result
    is constant across an order, and is then scaled by the blaze function unless
    ``blaze`` is false.  Orders outside the measurement hold the nearest
    measured value rather than extrapolating.
    """
    _, centre, fsr = echelle_orders(wave, mlambda)

    table_wave = np.asarray(table_wave, dtype=float)
    table_eff = np.asarray(table_eff, dtype=float)
    order = np.argsort(table_wave)
    table_wave, table_eff = table_wave[order], table_eff[order]

    thru = np.interp(centre, table_wave, table_eff)
    if blaze:
        thru = thru * blaze_efficiency(wave, centre, fsr)
    return thru


def measured_orders(table_wave, mlambda, tolerance=0.01):
    """Order numbers a per-order throughput table covers.

    Raises if the table's wavelengths are not order centres to within
    ``tolerance`` Angstroms, which is the assumption
    :func:`order_centre_throughput` rests on.
    """
    table_wave = np.asarray(table_wave, dtype=float)
    order = np.rint(mlambda / table_wave).astype(int)
    residual = np.abs(table_wave - mlambda / order)
    if residual.max() > tolerance:
        raise ValueError(
            "throughput table is not tabulated at echelle order centres: "
            f"largest residual {residual.max():.4g} A exceeds {tolerance} A"
        )
    return order
