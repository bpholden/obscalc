"""Adapter for the expcalc web ETC.

The web application in ``expcalc`` currently builds an IDL command string,
``Popen``s ``idl_wrapper``, and recovers the numbers by regexp-matching the
wrapper's printed output (``s2n_param.py``: ``gen_s2n`` and ``parse_return``).
:func:`calculate` returns the same dictionary that ``parse_return`` produced, so
that whole path can be replaced with a direct call::

    from obscalc.webapi import calculate
    output = calculate(request.params)

Keys, all as in the original: ``wave`` is a list of floats; ``s2n``, ``obj``,
``sky`` and ``noise`` are lists of ``[wavelength, value]`` pairs with ``j``-prefixed
plain-value twins; ``cts`` is a list of
``[wavelength, obj, sky, noise, s2n]`` rows; ``msg`` reports a bad request and
``errormsg`` a bad result.

Nothing here is instrument specific.  The ``inst`` parameter selects a backend
from :data:`obscalc.instruments.BACKENDS`, which owns the parameters only that
spectrograph understands.  Only ``apf`` is registered so far; any other value is
refused rather than quietly answered with APF numbers.
"""

import numpy as np

from .instruments import BACKENDS, ParameterError, get_backend
from .photometry import TemplateFilterMismatch
from .s2n import run_sides
from .structures import Observation, parse_binning

#: Wording the existing web UI shows for a template/filter that do not overlap.
NO_OVERLAP_MESSAGE = (
    "The selected filter and template do not overlap. Please select another "
    "filter, another template, or redshift the template. "
)

#: More than this many negative object counts means the result is unusable.
#: ``s2n_param.bad_obj`` used the same threshold.
MAX_NEGATIVE_COUNTS = 5

#: Instrument served when a request does not say.
DEFAULT_INSTRUMENT = "apf"

# Parameters every spectrograph shares.  Anything else in the request -- a
# dichroic, a grating, a decker letter -- belongs to a backend.
_DEFAULTS = {
    "inst": DEFAULT_INSTRUMENT,
    "mag": 17.0,
    "mtype": 1,
    "seeing": 1.2,
    "airmass": 1.1,
    "exptime": 3600.0,
    "redshift": 0.0,
    "binning": "1x1",
    "template": "",
    "ffilter": "",
}

_NUMERIC_LABELS = {
    "mag": "Mag",
    "seeing": "Seeing",
    "airmass": "Airmass",
    "exptime": "Exp. time",
    "redshift": "Redshift",
}


def empty_payload():
    """The output dictionary before any calculation, as ``build_exec_str`` built it."""
    return {
        "wave": [],
        "s2n": [],
        "s2nperres": [],
        "obj": [],
        "objperwavesec": [],
        "noise": [],
        "sky": [],
        "cts": [],
        "js2n": [],
        "jobj": [],
        "js2nperres": [],
        "jnoise": [],
        "jsky": [],
        "com": "",
        "dich": "",
        "msg": "",
        "errormsg": "",
        "i2counts": None,
        "exp": None,
        "precision": None,
    }


def _bad(label):
    return f"Inappropriate value for the input parameter {label}"


def _coerce(params):
    """Validate and type the instrument-independent parameters.

    Returns ``(values, msg)``; a non-empty ``msg`` means do not calculate, which
    is the same contract ``build_exec_str`` had.  Parameters this function does
    not recognise are passed through untouched for the backend to interpret.
    """
    values = {
        key: str(value).strip()
        for key, value in params.items()
        if str(value).strip() != ""
    }
    for key, default in _DEFAULTS.items():
        values.setdefault(key, default)

    for key, label in _NUMERIC_LABELS.items():
        try:
            values[key] = float(values[key])
        except (TypeError, ValueError):
            return None, _bad(label)

    try:
        values["mtype"] = int(values["mtype"])
    except (TypeError, ValueError):
        return None, _bad("Mag. Type")
    if values["mtype"] not in (1, 2):
        return None, _bad("Mag. Type")

    try:
        values["bins"], values["bind"] = parse_binning(values["binning"])
    except ValueError:
        return None, _bad("CCD Binning")

    if values["seeing"] <= 0:
        return None, _bad("Seeing")
    if values["exptime"] <= 0:
        return None, _bad("Exp. time")

    if bool(values["template"]) != bool(values["ffilter"]):
        return None, "A template requires a filter to normalise it, and vice versa"

    if str(values["inst"]).strip().lower() not in BACKENDS:
        return None, (
            f"Unknown instrument {values['inst']!r}. This calculator serves "
            f"{', '.join(sorted(BACKENDS))}."
        )

    return values, ""


def _pairs(wave, values):
    return [[float(w), float(v)] for w, v in zip(wave, values)]


def calculate(params, wvmn=None, wvmx=None, dwv=10.0):
    """Run a calculation for a web request.

    ``params`` is any mapping using the keys the existing forms post. Common to
    every instrument: ``inst``, ``mag``, ``mtype``, ``seeing``, ``airmass``,
    ``exptime``, ``redshift``, ``binning``, ``template``, ``ffilter``. Anything
    else is the selected backend's business -- for APF that is ``slitwidth``,
    which is a decker letter.
    """
    payload = empty_payload()

    values, msg = _coerce(params)
    if msg:
        payload["msg"] = msg
        return payload

    backend = get_backend(values["inst"])

    obs = Observation(
        seeing=values["seeing"],
        airmass=values["airmass"],
        mstar=values["mag"],
        mtype=values["mtype"],
        exptime=values["exptime"],
        redshift=values["redshift"],
        template=values["template"],
        filter=values["ffilter"],
    )

    if backend.default_range is None:
        raise NotImplementedError(
            f"backend {backend.name!r} does not declare a default_range"
        )
    default_min, default_max = backend.default_range
    wvmn = default_min if wvmn is None else float(wvmn)
    wvmx = default_max if wvmx is None else float(wvmx)
    count = int((wvmx - wvmn) / dwv) + 1
    wave = wvmn + np.arange(count) * dwv

    try:
        tel, sides = backend.sides(wave, values)
    except ParameterError as exc:
        payload["msg"] = str(exc)
        return payload

    try:
        result = run_sides(wave, tel, sides, obs)
        extras = backend.extras(result, obs)
    except TemplateFilterMismatch:
        payload["errormsg"] = NO_OVERLAP_MESSAGE
        return payload

    payload["wave"] = [float(w) for w in result.wave]
    for key, series in (
        ("s2n", result.sn),
        ("obj", result.star),
        ("sky", result.sky),
        # Read noise is per detector, so this is piecewise constant across the
        # dichroic split rather than a single value.
        ("noise", result.noise),
        ("s2nperres", result.sn_per_resolution_element),

    ):
        payload[key] = _pairs(result.wave, series)
        payload["j" + key] = [float(v) for v in series]

    payload["cts"] = [
        [float(w), float(o), float(s), float(n), float(sn)]
        for w, o, s, n, sn in zip(
            result.wave, result.star, result.sky, result.noise, result.sn
        )
    ]

    if extras:
        payload.update(extras)

    if int(np.sum(result.star < 0)) > MAX_NEGATIVE_COUNTS:
        payload["errormsg"] = NO_OVERLAP_MESSAGE

    return payload
