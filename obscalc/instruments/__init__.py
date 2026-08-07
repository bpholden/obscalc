"""Instrument definitions, throughput curves and web backends."""

from .apf import DECKERS, APFBackend, apf_spectrograph, apf_thruput
from .base import Backend, ParameterError
from .deimos import (
    CENTRAL_WAVES,
    DeimosBackend,
    deimos_spectrograph,
    deimos_thruput,
)
from .deimos import GRATINGS as DEIMOS_GRATINGS
from .hires import EPOCHS, HiresBackend, hires_spectrograph, hires_thruput
from .hires import DECKERS as HIRES_DECKERS
from .kast import (
    DICHROICS,
    GRATINGS,
    GRISMS,
    KastBackend,
    kast_sides,
    kast_spectrograph,
    kast_thruput,
)
from .lris import LrisBackend, lris_sides, lris_spectrograph, lris_thruput
from .lris import DICHROICS as LRIS_DICHROICS
from .lris import GRATINGS as LRIS_GRATINGS
from .lris import GRISMS as LRIS_GRISMS

#: Backends the web layer can serve, keyed on the ``inst`` request parameter.
#: Registering a new instrument here is all :mod:`obscalc.webapi` needs.
BACKENDS = {
    "apf": APFBackend(),
    "deimos": DeimosBackend(),
    "hires": HiresBackend(),
    "kast": KastBackend(),
    "lris": LrisBackend(),
}


def get_backend(name):
    """Look up a backend by the name the web forms post as ``inst``."""
    try:
        return BACKENDS[str(name).strip().lower()]
    except KeyError:
        raise NotImplementedError(
            f"no backend for instrument {name!r}; have {sorted(BACKENDS)}"
        ) from None


def available_instruments():
    return sorted(BACKENDS)


__all__ = [
    "BACKENDS",
    "APFBackend",
    "Backend",
    "CENTRAL_WAVES",
    "DECKERS",
    "DEIMOS_GRATINGS",
    "DICHROICS",
    "DeimosBackend",
    "deimos_spectrograph",
    "deimos_thruput",
    "EPOCHS",
    "GRATINGS",
    "GRISMS",
    "HIRES_DECKERS",
    "HiresBackend",
    "KastBackend",
    "LRIS_DICHROICS",
    "LRIS_GRATINGS",
    "LRIS_GRISMS",
    "LrisBackend",
    "ParameterError",
    "apf_spectrograph",
    "apf_thruput",
    "available_instruments",
    "get_backend",
    "hires_spectrograph",
    "hires_thruput",
    "kast_sides",
    "kast_spectrograph",
    "kast_thruput",
    "lris_sides",
    "lris_spectrograph",
    "lris_thruput",
]
