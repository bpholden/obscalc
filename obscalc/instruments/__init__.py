"""Instrument definitions, throughput curves and web backends."""

from .apf import DECKERS, APFBackend, apf_spectrograph, apf_thruput
from .base import Backend, ParameterError

#: Backends the web layer can serve, keyed on the ``inst`` request parameter.
#: Registering a new instrument here is all :mod:`obscalc.webapi` needs.
BACKENDS = {
    "apf": APFBackend(),
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
    "DECKERS",
    "ParameterError",
    "apf_spectrograph",
    "apf_thruput",
    "available_instruments",
    "get_backend",
]
