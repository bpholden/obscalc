"""Telescope definitions."""

from .structures import Telescope


def apf_telescope():
    """The Automated Planet Finder, from ``x_initapftel.pro``.

    The area is that of the APF's 2.4 m primary; the comment in the IDL
    ("10m telescope with 7.9% central obscuration") was left over from the Keck
    routine this file was copied from.
    """
    return Telescope(name="APF", area=42053.0, plate_scale=5.8241)


TELESCOPES = {"APF": apf_telescope}


def telescope(name):
    try:
        return TELESCOPES[name.strip()]()
    except KeyError:
        raise NotImplementedError(
            f"no telescope definition for {name!r}; have {sorted(TELESCOPES)}"
        ) from None
