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
"""

import numpy as np

from .apf_extras import apf_extras
from .instruments.apf import DECKERS, apf_spectrograph, apf_thruput
from .photometry import TemplateFilterMismatch
from .s2n import spec_calcs2n
from .structures import Observation, parse_binning
from .telescopes import apf_telescope

#: Wording the existing web UI shows for a template/filter that do not overlap.
NO_OVERLAP_MESSAGE = (
    "The selected filter and template do not overlap. Please select another "
    "filter, another template, or redshift the template. "
)

#: More than this many negative object counts means the result is unusable.
#: ``s2n_param.bad_obj`` used the same threshold.
MAX_NEGATIVE_COUNTS = 5

_DEFAULTS = {
    "mag": 17.0,
    "mtype": 1,
    "seeing": 1.2,
    "airmass": 1.1,
    "exptime": 3600.0,
    "redshift": 0.0,
    "binning": "1x1",
    "slitwidth": "W",
    "template": "",
    "ffilter": "",
}


def empty_payload():
    """The output dictionary before any calculation, as ``build_exec_str`` built it."""
    return {
        "wave": [],
        "s2n": [],
        "obj": [],
        "objperwavesec": [],
        "noise": [],
        "sky": [],
        "cts": [],
        "js2n": [],
        "jobj": [],
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


def _coerce(params):
    """Validate and type the request parameters.

    Returns ``(values, msg)``; a non-empty ``msg`` means do not calculate, which
    is the same contract ``build_exec_str`` had.
    """
    values = dict(_DEFAULTS)
    for key in values:
        if key in params and str(params[key]).strip():
            values[key] = str(params[key]).strip()

    numeric = {
        "mag": "Mag",
        "seeing": "Seeing",
        "airmass": "Airmass",
        "exptime": "Exp. time",
        "redshift": "Redshift",
    }
    for key, label in numeric.items():
        try:
            values[key] = float(values[key])
        except (TypeError, ValueError):
            return None, f"Inappropriate value for the input parameter {label}"

    try:
        values["mtype"] = int(values["mtype"])
    except (TypeError, ValueError):
        return None, "Inappropriate value for the input parameter Mag. Type"
    if values["mtype"] not in (1, 2):
        return None, "Inappropriate value for the input parameter Mag. Type"

    try:
        parse_binning(values["binning"])
    except ValueError:
        return None, "Inappropriate value for the input parameter CCD Binning"

    if values["slitwidth"] not in DECKERS:
        return None, "Inappropriate value for the input parameter Slitwidth"

    if values["seeing"] <= 0:
        return None, "Inappropriate value for the input parameter Seeing"
    if values["exptime"] <= 0:
        return None, "Inappropriate value for the input parameter Exp. time"

    if bool(values["template"]) != bool(values["ffilter"]):
        return None, "A template requires a filter to normalise it, and vice versa"

    return values, ""


def _pairs(wave, values):
    return [[float(w), float(v)] for w, v in zip(wave, values)]


def calculate(params, wvmn=None, wvmx=None, dwv=10.0):
    """Run an APF calculation for a web request.

    ``params`` is any mapping using the keys the existing forms post: ``mag``,
    ``mtype``, ``seeing``, ``airmass``, ``exptime``, ``redshift``, ``binning``,
    ``slitwidth`` (an APF decker letter), ``template`` and ``ffilter``.
    """
    payload = empty_payload()

    values, msg = _coerce(params)
    if msg:
        payload["msg"] = msg
        return payload

    bins, bind = parse_binning(values["binning"])
    tel = apf_telescope()
    instr = apf_spectrograph(
        decker=values["slitwidth"], bins=bins, bind=bind, str_tel=tel
    )
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

    wvmn = instr.wvmnx[0] if wvmn is None else float(wvmn)
    wvmx = instr.wvmnx[1] if wvmx is None else float(wvmx)
    count = int((wvmx - wvmn) / dwv) + 1
    wave = wvmn + np.arange(count) * dwv

    try:
        result = spec_calcs2n(wave, apf_thruput(wave), tel, instr, obs)
        extras = apf_extras(result, obs)
    except TemplateFilterMismatch:
        payload["errormsg"] = NO_OVERLAP_MESSAGE
        return payload

    # The IDL wrapper broadcast the scalar read noise across the grid.
    noise = np.full(result.wave.shape, result.noise)

    payload["wave"] = [float(w) for w in result.wave]
    for key, values_array in (
        ("s2n", result.sn),
        ("obj", result.star),
        ("sky", result.sky),
        ("noise", noise),
    ):
        payload[key] = _pairs(result.wave, values_array)
        payload["j" + key] = [float(v) for v in values_array]

    payload["cts"] = [
        [float(w), float(o), float(s), float(n), float(sn)]
        for w, o, s, n, sn in zip(
            result.wave, result.star, result.sky, noise, result.sn
        )
    ]

    payload["i2counts"] = extras.i2counts
    payload["exp"] = extras.expmeter
    payload["precision"] = extras.precision

    if int(np.sum(result.star < 0)) > MAX_NEGATIVE_COUNTS:
        payload["errormsg"] = NO_OVERLAP_MESSAGE

    return payload
