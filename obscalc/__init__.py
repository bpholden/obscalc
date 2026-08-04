"""Signal-to-noise calculator for slit spectrographs.

A Python port of the exposure-time calculator in xidl (``Obs/S2N``), whose
entry point was ``spec_calcs2n.pro``.  The first supported configuration is the
Levy spectrograph on the APF.

Typical use::

    from obscalc import Observation, apf_spectrograph, apf_thruput
    from obscalc import apf_telescope, spec_calcs2n
    import numpy as np

    wave = np.arange(3742.0, 7700.0, 10.0)
    tel = apf_telescope()
    instr = apf_spectrograph(decker="W")
    obs = Observation(seeing=1.2, mstar=13.0, mtype=2, exptime=1200.0)
    result = spec_calcs2n(wave, apf_thruput(wave), tel, instr, obs)
"""

from .instruments.apf import apf_spectrograph, apf_thruput
from .s2n import S2NResult, spec_calcs2n
from .structures import Instrument, Observation, Telescope
from .telescopes import apf_telescope

__version__ = "0.1.0"

__all__ = [
    "Instrument",
    "Observation",
    "S2NResult",
    "Telescope",
    "apf_spectrograph",
    "apf_telescope",
    "apf_thruput",
    "spec_calcs2n",
]
