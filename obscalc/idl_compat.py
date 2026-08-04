"""Numpy equivalents of the IDL routines the ported code relied on.

These exist because the obvious numpy call has different behaviour from the IDL
one in ways that change results:

* ``interpol`` extrapolates linearly off the end of its table; ``np.interp``
  clamps to the end value instead.
* ``interpol`` accepts a descending abscissa (the APF sensitivity file is stored
  in descending wavelength order); ``np.interp`` requires ascending.
* ``long()`` truncates toward zero; ``np.floor`` rounds toward -inf.
* ``linterp`` (idlutils) does *not* extrapolate -- it substitutes a fill value.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def _ascending(x, y):
    """Return ``x, y`` sorted so ``x`` ascends, as IDL's value_locate expects."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size > 1 and x[0] > x[-1]:
        return x[::-1], y[::-1]
    return x, y


def interpol(v, x, xout, spline=False):
    """IDL ``interpol(V, X, XOUT)``: linear interpolation with extrapolation.

    Outside ``[x.min(), x.max()]`` the slope of the nearest interval is
    continued, which is what IDL does and what ``np.interp`` does not.
    """
    x, v = _ascending(x, v)
    xout = np.asarray(xout, dtype=float)

    if spline:
        return CubicSpline(x, v, extrapolate=True)(xout)

    if x.size == 1:
        return np.full(xout.shape, v[0])

    # Clipping the interval index to the interior is what turns interpolation
    # into linear extrapolation at both ends.
    idx = np.clip(np.searchsorted(x, xout) - 1, 0, x.size - 2)
    slope = (v[idx + 1] - v[idx]) / (x[idx + 1] - x[idx])
    return v[idx] + (xout - x[idx]) * slope


def linterp(x, y, xout, missing=0.0):
    """idlutils ``linterp``: linear interpolation, ``missing`` outside the range."""
    x, y = _ascending(x, y)
    return np.interp(np.asarray(xout, dtype=float), x, y, left=missing, right=missing)


def tsum(x, y, imin=None, imax=None):
    """IDL Astronomy Library ``tsum``: trapezoidal integration of ``y`` d``x``.

    ``imin``/``imax`` are inclusive element indices, as in the IDL.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = 0 if imin is None else int(imin)
    hi = x.size - 1 if imax is None else int(imax)
    return float(np.trapezoid(y[lo : hi + 1], x[lo : hi + 1]))


def idl_long(x):
    """IDL ``long()``: truncate toward zero, unlike ``np.floor``."""
    return np.trunc(np.asarray(x, dtype=float)).astype(np.int64)
