"""What a web-facing instrument backend has to provide.

:mod:`obscalc.webapi` handles everything common to all spectrographs -- the
magnitude, exposure time, seeing, airmass, template and binning -- and delegates
the rest here.  A backend owns the parameters only its own instrument
understands: how ``slitwidth`` is spelled, whether there is a dichroic, a
grating, a grism or a central wavelength, and which throughput curve applies.
"""

import abc


class ParameterError(ValueError):
    """A request parameter this instrument cannot accept.

    The message is shown to the user through the web layer's ``msg`` field, so
    it should name the parameter the way the form labels it.
    """


class Backend(abc.ABC):
    """Adapter between a web request and one instrument."""

    #: Name the web forms post as ``inst``.
    name = ""

    @abc.abstractmethod
    def configure(self, values):
        """Return ``(telescope, instrument)`` for the request.

        ``values`` holds the parameters after the generic coercion in
        :mod:`obscalc.webapi`, including ``bins`` and ``bind`` already split out
        of the ``binning`` string.  Raise :class:`ParameterError` for anything
        this instrument rejects.
        """

    @abc.abstractmethod
    def thruput(self, wave, instr):
        """End-to-end throughput (0-1) on the ``wave`` grid."""

    def extras(self, result, obs):
        """Instrument-specific derived quantities, or None.

        Returns a mapping merged into the response.  Only APF has any -- the
        iodine counts, exposure meter reading and RV precision.
        """
        return None

    def wavelength_range(self, instr):
        """Default grid limits, in Angstroms."""
        return instr.wvmnx
