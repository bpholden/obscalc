"""What a web-facing instrument backend has to provide.

:mod:`obscalc.webapi` handles everything common to all spectrographs -- the
magnitude, exposure time, seeing, airmass, template and binning -- and delegates
the rest here.  A backend owns the parameters only its own instrument
understands: how ``slitwidth`` is spelled, whether there is a dichroic, a
grating, a grism or a central wavelength, and which throughput curve applies.

A backend describes its detectors rather than assuming there is one.  Kast
splits the beam with a dichroic and gives the two sides different resolutions,
read noise and throughput, so :meth:`Backend.sides` returns one
:class:`~obscalc.s2n.Side` per detector and the engine runs once for each.
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

    #: Default wavelength grid limits in Angstroms, used when the caller does
    #: not give any.  A class attribute because the grid has to exist before
    #: :meth:`sides` can be called.  Subclasses must set it.
    default_range = None

    @abc.abstractmethod
    def sides(self, wave, values):
        """Return ``(telescope, [Side, ...])`` for the request.

        ``values`` holds the parameters after the generic coercion in
        :mod:`obscalc.webapi`, including ``bins`` and ``bind`` already split out
        of the ``binning`` string.  Each :class:`~obscalc.s2n.Side` carries the
        instrument for one detector, the indices into ``wave`` it records, and
        its throughput there.  Raise :class:`ParameterError` for anything this
        instrument rejects.
        """

    def extras(self, result, obs):
        """Instrument-specific derived quantities, or None.

        ``result`` is the :class:`~obscalc.s2n.StackedResult`.  Returns a mapping
        merged into the response; only APF has any -- the iodine counts,
        exposure meter reading and RV precision.
        """
        return None
