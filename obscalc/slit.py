"""Fraction of a Gaussian image that falls through a rectangular slit.

Port of ``x_gaussslit.pro``.  The IDL summed a tabulated radial profile over a
199x199 grid of half-arcsecond-ish cells with two nested loops; this does the
same sum with numpy.  The tabulated profile, the ``radius >= 99`` cutoff, the
use of the un-offset radius, and the half-in/half-out treatment of cells lying
exactly on the slit edge are all preserved.
"""

import numpy as np

# Radial profile of the seeing disc, sampled every 1/20 FWHM.  Transcribed from
# x_gaussslit.pro; despite the routine's name this is an empirical profile, not
# an analytic Gaussian.
_PSF = np.array(
    [
        1.000, 0.995, 0.985, 0.971, 0.954,
        0.933, 0.911, 0.886, 0.860, 0.833,
        0.804, 0.774, 0.743, 0.713, 0.682,
        0.651, 0.620, 0.594, 0.559, 0.529,
        0.500, 0.471, 0.443, 0.417, 0.391,
        0.366, 0.342, 0.319, 0.297, 0.276,
        0.256, 0.237, 0.218, 0.202, 0.187,
        0.172, 0.158, 0.145, 0.132, 0.122,
        0.113, 0.104, 0.097, 0.089, 0.082,
        0.077, 0.072, 0.065, 0.059, 0.057,
        0.052, 0.049, 0.046, 0.042, 0.039,
        0.037, 0.034, 0.032, 0.029, 0.027,
        0.026, 0.024, 0.023, 0.021, 0.019,
        0.018, 0.017, 0.017, 0.016, 0.016,
        0.015, 0.014, 0.013, 0.012, 0.011,
        0.010, 0.010, 0.009, 0.009, 0.008,
        0.008, 0.007, 0.007, 0.006, 0.006,
        0.005, 0.005, 0.005, 0.004, 0.004,
        0.004, 0.004, 0.003, 0.003, 0.003,
        0.003, 0.003, 0.002, 0.002, 0.002,
    ]
)

# The IDL grid: i = 1..199 gives y = 99 down to -99, j = 1..199 gives x = -99
# up to 99.  Built once, since the geometry never changes.
_Y = (100.0 - np.arange(1, 200)).reshape(-1, 1)
_X = (np.arange(1, 200) - 100.0).reshape(1, -1)
_RADIUS = np.sqrt(_X**2 + _Y**2)

# Flux in each cell, from the profile.  Cells beyond the tabulated profile
# contribute nothing.
_FLUX = np.zeros_like(_RADIUS)
_INSIDE_PROFILE = _RADIUS < 99.0
_IRAD = np.trunc(_RADIUS[_INSIDE_PROFILE]).astype(int)
_DRAD = _RADIUS[_INSIDE_PROFILE] - _IRAD
_FLUX[_INSIDE_PROFILE] = (1.0 - _DRAD) * _PSF[_IRAD] + _DRAD * _PSF[_IRAD + 1]


def gauss_slit(w, h, xo=0.0, yo=0.0):
    """Fraction of the image passing a slit ``w`` by ``h``, in FWHM units.

    ``w`` and ``h`` are slit width and height divided by the seeing FWHM;
    ``xo``, ``yo`` are pointing offsets in the same units.
    """
    width = 20.0 * w
    height = 20.0 * h
    xoff = 40.0 * xo
    yoff = 40.0 * yo

    dx = np.abs(_X - xoff)
    dy = np.abs(_Y - yoff)

    inside = (dy < height) & (dx < width)
    outside = (dy > height) | (dx > width)
    # A cell sitting exactly on an edge is split between in and out.  With
    # floating point this only fires when the slit dimensions land on grid
    # lines, but it does fire for the round slit widths APF uses.
    on_edge = ((dy == height) & (dx <= width)) | ((dx == width) & (dy <= height))

    edge_flux = _FLUX[on_edge].sum()
    flux_in = _FLUX[inside].sum() + 0.5 * edge_flux
    flux_out = _FLUX[outside].sum() + 0.5 * edge_flux

    return float(flux_in / (flux_in + flux_out))
