"""Signal-to-noise calculator for slit spectrographs.

A Python port of the exposure-time calculator in xidl (``Obs/S2N``), whose entry
point was ``spec_calcs2n.pro``.  Five instruments are supported: the Levy on the
APF and Kast on the Shane 3 m at Lick, HIRES and LRIS on Keck I, and DEIMOS on
Keck II.

Each instrument contributes a pair of functions, ``<name>_spectrograph`` for its
parameters and ``<name>_thruput`` for its measured throughput.  For a
single-channel instrument that is all :func:`spec_calcs2n` needs::

    import numpy as np
    from obscalc import (Observation, hires_spectrograph, hires_thruput,
                         keck_telescope, spec_calcs2n)

    wave = np.arange(3000.0, 9500.0, 10.0)
    tel = keck_telescope("KeckI")
    instr = hires_spectrograph(decker="C5", epoch="new", str_tel=tel)
    obs = Observation(seeing=0.7, mstar=15.0, mtype=2, exptime=1800.0)
    result = spec_calcs2n(wave, hires_thruput(wave, instr), tel, instr, obs)

Kast and LRIS split the beam, so each has two detectors with their own
resolution, read noise and throughput.  They return an instrument *pair*, and
``<name>_sides`` packages it for :func:`run_sides`, which runs the engine once
per detector and stacks the results onto the shared grid::

    from obscalc import (Observation, lris_sides, lris_spectrograph,
                         keck_telescope, run_sides)

    wave = np.arange(3500.0, 10000.0, 10.0)
    tel = keck_telescope("KeckI")
    blue, red = lris_spectrograph(grism="B600", grating="600/7500", str_tel=tel)
    obs = Observation(seeing=1.0, mstar=20.0, mtype=2, exptime=3600.0)
    result = run_sides(wave, tel, lris_sides(wave, blue, red), obs)

Anything an instrument alone defines -- its deckers, dispersers, dichroics and
sensitivity files -- lives in its own module under :mod:`obscalc.instruments`.
For the dictionary the web forms expect, see :func:`obscalc.webapi.calculate`.
"""

from .instruments.apf import apf_spectrograph, apf_thruput
from .instruments.deimos import deimos_spectrograph, deimos_thruput
from .instruments.hires import hires_spectrograph, hires_thruput
from .instruments.kast import kast_sides, kast_spectrograph, kast_thruput
from .instruments.lris import lris_sides, lris_spectrograph, lris_thruput
from .s2n import S2NResult, Side, StackedResult, run_sides, spec_calcs2n
from .structures import Instrument, Observation, Telescope
from .telescopes import apf_telescope, keck_telescope, lick_telescope, telescope

__version__ = "0.1.0"

__all__ = [
    # engine and data structures
    "Instrument",
    "Observation",
    "S2NResult",
    "Side",
    "StackedResult",
    "Telescope",
    "run_sides",
    "spec_calcs2n",
    # telescopes
    "apf_telescope",
    "keck_telescope",
    "lick_telescope",
    "telescope",
    # instruments
    "apf_spectrograph",
    "apf_thruput",
    "deimos_spectrograph",
    "deimos_thruput",
    "hires_spectrograph",
    "hires_thruput",
    "kast_sides",
    "kast_spectrograph",
    "kast_thruput",
    "lris_sides",
    "lris_spectrograph",
    "lris_thruput",
]
