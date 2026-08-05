"""Command line driver for the APF, replacing ``apf_calcs2n_wrapper.pro``.

Everything here is APF specific: the telescope, the Levy spectrograph, its
deckers, and the iodine-cell quantities in :mod:`obscalc.apf_extras`.  The parts
shared with other instruments live in :mod:`obscalc.cli_common`, and
:func:`obscalc.s2n.spec_calcs2n` is the generic engine.
"""

import argparse
import sys

import numpy as np

from . import config
from .apf_extras import apf_extras
from .echelle import echelle_orders
from .cli_common import (
    add_grid_arguments,
    add_observation_arguments,
    add_output_arguments,
    observation_from_args,
    results_table,
    summarise,
    wavelength_grid,
    write_outputs,
)
from .instruments.apf import (
    DECKERS,
    DEFAULT_RANGE,
    apf_spectrograph,
    apf_thruput,
    set_decker,
)
from .instruments.base import ParameterError
from .photometry import TemplateFilterMismatch
from .s2n import Side, run_sides
from .structures import (
    apply_infil_to_instrument,
    apply_infil_to_observation,
    parse_binning,
    read_infil,
)
from .telescopes import apf_telescope

__all__ = ["main", "results_table", "wavelength_grid"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="obscalc-apf",
        description="Signal-to-noise for the Levy spectrograph on the APF.",
    )
    add_grid_arguments(parser)

    inst = parser.add_argument_group("instrument")
    inst.add_argument(
        "--decker", default="W", choices=sorted(DECKERS), help="slit decker"
    )
    inst.add_argument(
        "--binning", default="1x1", help="spatial x dispersion, e.g. 2x1"
    )
    inst.add_argument(
        "--no-blaze",
        dest="blaze",
        action="store_false",
        help="report each echelle order's peak throughput instead of the value "
        "reached at each wavelength; reproduces apf_thruput.pro",
    )

    add_observation_arguments(parser, seeing=1.2, exptime=3600.0, mag=17.0)
    add_output_arguments(parser)

    data = parser.add_argument_group("data")
    data.add_argument(
        "--list-filters", action="store_true", help="list available filters and exit"
    )
    data.add_argument(
        "--list-templates",
        action="store_true",
        help="list available templates and exit",
    )
    return parser


def _configure(args):
    """Turn parsed arguments into telescope, instrument, observation and grid."""
    tel = apf_telescope()
    bins, bind = parse_binning(args.binning)
    instr = apf_spectrograph(decker=args.decker, bins=bins, bind=bind, str_tel=tel)
    obs = observation_from_args(args)

    if args.infil:
        cards = read_infil(args.infil)
        apply_infil_to_observation(obs, cards)
        decker = apply_infil_to_instrument(instr, cards)
        if decker:
            set_decker(instr, decker)

    wvmn = args.wvmn if args.wvmn is not None else DEFAULT_RANGE[0]
    wvmx = args.wvmx if args.wvmx is not None else DEFAULT_RANGE[1]
    return tel, instr, obs, wavelength_grid(wvmn, wvmx, args.dwv)


def _report_orders(result, instr, blaze=True, stream=None):
    """A line about the echelle format, as the HIRES driver prints."""
    stream = sys.stdout if stream is None else stream
    order, _, fsr = echelle_orders(result.wave, instr.mlambda)
    print(
        f"\nEchelle orders {order.min()}-{order.max()} across the range; "
        f"free spectral range {fsr.min():.1f}-{fsr.max():.1f} A",
        file=stream,
    )
    if not blaze:
        print(
            "Throughput is each order's peak value; drop --no-blaze for the "
            "value actually reached at each wavelength.",
            file=stream,
        )


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.list_filters:
        print("\n".join(config.available_filters()))
        return 0
    if args.list_templates:
        print("\n".join(config.available_templates()))
        return 0
    if bool(args.template) != bool(args.filter):
        print(
            "--template and --filter must be given together: the filter is what "
            "the template is normalised through.",
            file=sys.stderr,
        )
        return 2

    tel, instr, obs, wave = _configure(args)

    # One detector spans the whole range.
    thru = apf_thruput(wave, instr, blaze=args.blaze)
    sides = [Side("", instr, np.arange(wave.size), thru)]

    try:
        result = run_sides(wave, tel, sides, obs)
        extras = apf_extras(result.sides[0][2], obs)
    except (TemplateFilterMismatch, ParameterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        summarise(result, obs, instr=instr, extras=extras)
        _report_orders(result, instr, blaze=args.blaze)
    write_outputs(args, result, obs, instr=instr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
