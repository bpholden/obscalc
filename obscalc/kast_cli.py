"""Command line driver for Kast, replacing ``kast_calcs2n_wrapper.pro``.

Kast is a double spectrograph: a dichroic splits the beam and the two sides have
different resolutions, read noise and throughput curves, so the engine runs once
per side and the results are stacked onto one wavelength grid.
"""

import argparse
import sys

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
from .instruments.base import ParameterError
from .instruments.kast import (
    BLUE_SENS_FILES,
    DEFAULT_DICHROIC,
    DEFAULT_GRATING,
    DEFAULT_GRISM,
    DEFAULT_RANGE,
    DEFAULT_SLIT,
    DICHROICS,
    GRATINGS,
    kast_sides,
    kast_spectrograph,
)
from .photometry import TemplateFilterMismatch
from .s2n import run_sides
from .structures import (
    apply_infil_to_instrument,
    apply_infil_to_observation,
    parse_binning,
    read_infil,
)
from .telescopes import lick_telescope

__all__ = ["main", "results_table", "wavelength_grid"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="obscalc-kast",
        description="Signal-to-noise for the Kast double spectrograph at Lick.",
    )
    add_grid_arguments(parser)

    inst = parser.add_argument_group("instrument")
    inst.add_argument(
        "--dichroic",
        default=DEFAULT_DICHROIC,
        choices=sorted(DICHROICS),
        help="where the beam splits (d46 = 4600 A, d55 = 5500 A)",
    )
    inst.add_argument(
        "--grism",
        default=DEFAULT_GRISM,
        choices=sorted(BLUE_SENS_FILES),
        help="blue-side disperser",
    )
    inst.add_argument(
        "--grating",
        default=DEFAULT_GRATING,
        choices=sorted(GRATINGS),
        help="red-side disperser",
    )
    inst.add_argument(
        "--slitwidth",
        type=float,
        default=DEFAULT_SLIT,
        help="slit width in arcsec (a width here, not a decker name)",
    )
    inst.add_argument(
        "--binning", default="1x1", help="spatial x dispersion, e.g. 2x1"
    )

    add_observation_arguments(parser, seeing=1.5, exptime=3600.0, mag=19.0)
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
    """Turn parsed arguments into telescope, instrument pair, observation, grid."""
    tel = lick_telescope()
    bins, bind = parse_binning(args.binning)
    blue, red = kast_spectrograph(
        grism=args.grism,
        grating=args.grating,
        dichroic=args.dichroic,
        slit=args.slitwidth,
        bins=bins,
        bind=bind,
        str_tel=tel,
    )
    obs = observation_from_args(args)

    if args.infil:
        cards = read_infil(args.infil)
        apply_infil_to_observation(obs, cards)
        for instr in (blue, red):
            apply_infil_to_instrument(instr, cards)

    wvmn = args.wvmn if args.wvmn is not None else DEFAULT_RANGE[0]
    wvmx = args.wvmx if args.wvmx is not None else DEFAULT_RANGE[1]
    return tel, (blue, red), obs, wavelength_grid(wvmn, wvmx, args.dwv)


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

    tel, (blue, red), obs, wave = _configure(args)

    try:
        result = run_sides(wave, tel, kast_sides(wave, blue, red), obs)
    except (TemplateFilterMismatch, ParameterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        summarise(result, obs, instr=blue)
    write_outputs(args, result, obs, instr=blue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
