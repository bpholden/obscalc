"""Telescope definitions."""

from .structures import Telescope


def apf_telescope():
    """The Automated Planet Finder, from ``x_initapftel.pro``.

    The area is that of the APF's 2.4 m primary; the comment in the IDL
    ("10m telescope with 7.9% central obscuration") was left over from the Keck
    routine this file was copied from.
    """
    return Telescope(name="APF", area=42053.0, plate_scale=5.8241)


def lick_telescope():
    """The Shane 3 m at Lick, from ``x_initlick.pro``.

    The plate scale is marked "should confirm; secondary dependent" in the IDL.
    """
    return Telescope(name="Lick-3m", area=63617.0, plate_scale=1.379)


def keck_telescope(name="KeckI"):
    """Keck I or II, from ``x_initkeck.pro``.

    The two are identical here.  ``x_initkeck.pro`` left ``name`` blank and each
    instrument's init routine filled it in afterwards, which is what
    ``spec_calcs2n``'s extinction and sky dispatch keys on, so it is required.
    """
    if name not in ("KeckI", "KeckII"):
        raise ValueError(f"name must be 'KeckI' or 'KeckII', got {name!r}")
    return Telescope(name=name, area=723674.0, plate_scale=1.379)


TELESCOPES = {
    "APF": apf_telescope,
    "Lick-3m": lick_telescope,
    "KeckI": lambda: keck_telescope("KeckI"),
    "KeckII": lambda: keck_telescope("KeckII"),
}


def telescope(name):
    try:
        return TELESCOPES[name.strip()]()
    except KeyError:
        raise NotImplementedError(
            f"no telescope definition for {name!r}; have {sorted(TELESCOPES)}"
        ) from None
