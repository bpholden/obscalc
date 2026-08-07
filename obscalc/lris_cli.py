"""Command line driver for LRIS, replacing ``lris_calcs2n_wrapper.pro``.

LRIS is a double spectrograph: a dichroic splits the beam and the two sides have
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
from .instruments.lris import (
    DEFAULT_DICHROIC,
    DEFAULT_GRATING,
    DEFAULT_GRISM,
    DEFAULT_RANGE,
    DEFAULT_SEEING,
    DEFAULT_SLIT,
    DICHROICS,
    GRATINGS,
    GRISMS,
    lris_spectrograph,
    lris_thruput,
    sens_files,
    sensitivity_range,
    split_wavelengths,
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
        prog="obscalc-lris",
        description="Signal-to-noise for LRIS on Keck I.",
    )
    add_grid_arguments(parser)

    inst = parser.add_argument_group("instrument")
    inst.add_argument(
        "--dichroic",
        default=DEFAULT_DICHROIC,
        choices=sorted(DICHROICS),
        help="where the beam splits (D560 = 5600 A)",
    )
    inst.add_argument(
        "--grism",
        default=DEFAULT_GRISM,
        choices=sorted(GRISMS),
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

    add_observation_arguments(parser, seeing=DEFAULT_SEEING, exptime=3600.0, mag=20.0)
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
    tel = keck_telescope("KeckI")
    bins, bind = parse_binning(args.binning)
    blue, red = lris_spectrograph(
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


def _report_configuration(blue, red, stream=None):
    """Say which measurements are in use and where each one stops.

    Wavelengths outside them carry no throughput at all, which ``summarise``
    already reports as a dead range; these two lines say which measurement ran
    out and where.
    """
    stream = sys.stdout if stream is None else stream
    blue_file, red_file = sens_files(blue, red)
    lines = [""]
    for label, disperser, sens_file in (
        ("Grism", blue.grating, blue_file),
        ("Grating", red.grating, red_file),
    ):
        low, high = sensitivity_range(sens_file)
        lines.append(
            f"{label} {disperser} with dichroic {blue.dichroic}: throughput from "
            f"{sens_file}, measured over {low:.0f}-{high:.0f} A"
        )
    print("\n".join(lines), file=stream)


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
        thru = lris_thruput(wave, blue, red)
        blue_index, red_index = split_wavelengths(wave, args.dichroic)
        sides = [
            Side("blue", blue, blue_index, thru[blue_index]),
            Side("red", red, red_index, thru[red_index]),
        ]
        result = run_sides(wave, tel, sides, obs)
    except (TemplateFilterMismatch, ParameterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        summarise(result, obs, instr=blue)
        _report_configuration(blue, red)
    write_outputs(args, result, obs, instr=blue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
