"""Command line driver for DEIMOS, replacing ``deimos_calcs2n_wrapper.pro``."""

import argparse
import sys

import numpy as np

from . import config
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
from .instruments.deimos import (
    CENTRAL_WAVES,
    DEFAULT_CWAVE,
    DEFAULT_GRATING,
    DEFAULT_RANGE,
    DEFAULT_SLIT,
    GRATINGS,
    deimos_spectrograph,
    deimos_thruput,
    sens_file,
    sensitivity_range,
)
from .photometry import TemplateFilterMismatch
from .s2n import Side, run_sides
from .structures import (
    apply_infil_to_instrument,
    apply_infil_to_observation,
    parse_binning,
    read_infil,
)
from .telescopes import keck_telescope

__all__ = ["main", "results_table", "wavelength_grid"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="obscalc-deimos",
        description="Signal-to-noise for DEIMOS on Keck II.",
    )
    add_grid_arguments(parser)

    inst = parser.add_argument_group("instrument")
    inst.add_argument(
        "--grating",
        default=DEFAULT_GRATING,
        choices=sorted(GRATINGS),
        help="grating",
    )
    inst.add_argument(
        "--cwave",
        type=int,
        default=DEFAULT_CWAVE,
        choices=CENTRAL_WAVES,
        help="grating tilt, as a central wavelength in Angstroms; selects which "
        "sensitivity measurement applies",
    )
    inst.add_argument(
        "--slitwidth",
        type=float,
        default=DEFAULT_SLIT,
        help="slit width in arcsec",
    )
    inst.add_argument(
        "--binning", default="1x1", help="spatial x dispersion, e.g. 2x2"
    )

    add_observation_arguments(parser, seeing=0.7, exptime=1200.0, mag=22.0)
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
    tel = keck_telescope("KeckII")
    bins, bind = parse_binning(args.binning)
    instr = deimos_spectrograph(
        grating=args.grating,
        cwave=args.cwave,
        slit=args.slitwidth,
        bins=bins,
        bind=bind,
        str_tel=tel,
    )
    obs = observation_from_args(args)

    if args.infil:
        cards = read_infil(args.infil)
        apply_infil_to_observation(obs, cards)
        apply_infil_to_instrument(instr, cards)

    wvmn = args.wvmn if args.wvmn is not None else DEFAULT_RANGE[0]
    wvmx = args.wvmx if args.wvmx is not None else DEFAULT_RANGE[1]
    return tel, instr, obs, wavelength_grid(wvmn, wvmx, args.dwv)


def _report_configuration(instr, stream=None):
    """Say which measurement is in use and where it stops.

    Wavelengths outside it carry no throughput at all, which ``summarise``
    already reports as a dead range; this line says which measurement ran out.
    """
    stream = sys.stdout if stream is None else stream
    low, high = sensitivity_range(instr.grating, instr.cwave)
    print(
        f"\nGrating {instr.grating} tilted to {instr.cwave:.0f} A: throughput from "
        f"{sens_file(instr.grating, instr.cwave)}, measured over {low:.0f}-{high:.0f} A",
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
    if args.slitwidth <= 0:
        print("--slitwidth must be positive (arcsec).", file=sys.stderr)
        return 2

    tel, instr, obs, wave = _configure(args)

    try:
        thru = deimos_thruput(wave, instr)
        sides = [Side("", instr, np.arange(wave.size), thru)]
        result = run_sides(wave, tel, sides, obs)
    except (TemplateFilterMismatch, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        summarise(result, obs, instr=instr)
        _report_configuration(instr)
    write_outputs(args, result, obs, instr=instr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
