"""Command line driver, replacing ``apf_calcs2n_wrapper.pro``."""

import argparse
import sys

import numpy as np
from astropy.table import Table

from . import config
from .apf_extras import apf_extras
from .instruments.apf import DECKERS, apf_spectrograph, apf_thruput
from .photometry import TemplateFilterMismatch
from .s2n import spec_calcs2n
from .structures import (
    Observation,
    apply_infil_to_instrument,
    apply_infil_to_observation,
    parse_binning,
    read_infil,
)
from .telescopes import apf_telescope


def wavelength_grid(wvmn, wvmx, dwv):
    """The grid ``apf_calcs2n_wrapper.pro`` built: inclusive of both endpoints."""
    if dwv <= 0:
        raise ValueError(f"dwv must be positive, got {dwv}")
    if wvmx <= wvmn:
        raise ValueError(f"wvmx ({wvmx}) must exceed wvmn ({wvmn})")
    count = int((wvmx - wvmn) / dwv) + 1
    return wvmn + np.arange(count) * dwv


def results_table(result):
    """Per-wavelength results as an astropy Table."""
    table = Table()
    table["wave"] = result.wave
    table["s2n"] = result.sn
    table["s2n_resel"] = result.sn_per_resolution_element
    table["obj"] = result.star
    table["sky"] = result.sky
    table["tnoise"] = result.tnoise
    table["thru"] = result.thru
    table["extinct"] = result.extinct
    table["magsky"] = result.magsky
    table["pixel"] = result.pixel

    table["wave"].unit = "Angstrom"
    table["pixel"].unit = "Angstrom"
    table["extinct"].unit = "mag"
    table["magsky"].unit = "mag"
    for name in ("obj", "sky", "tnoise"):
        table[name].unit = "count"

    # Keys are kept to 8 characters so they survive a FITS header without
    # becoming HIERARCH cards.
    table.meta.update(
        {
            "slit0": result.slit0,
            "rows": result.rows,
            "columns": result.columns,
            "nsky": result.nsky,
            "readno": result.noise,
            "ndark": result.ndark,
            "mstar": result.mstar,
            "mtype": result.mtype,
            "binc": result.binc,
            "binr": result.binr,
        }
    )
    return table


def summarise(result, instr, obs, extras=None, stream=None):
    """Print the configuration and a few summary numbers."""
    stream = sys.stdout if stream is None else stream
    system = "Vega/Johnson" if result.mtype == 1 else "AB"
    lines = [
        f"DECKER   = {instr.swidth:g}\" x {instr.sheight:g}\"",
        f"BINNING  = {result.binr}x{result.binc} (spatial x dispersion)",
        f"SEEING   = {obs.seeing:g}\"",
        f"AIRMASS  = {obs.airmass:g}",
        f"EXPTIME  = {obs.exptime:g} s",
        f"MAG      = {result.mstar:g} ({system})",
    ]
    if obs.template:
        lines.append(f"TEMPLATE = {obs.template} at z = {obs.redshift:g}")
        lines.append(f"FILTER   = {obs.filter}")
    lines += [
        "",
        f"Slit width projects to {result.columns:.3f} pixels",
        f"Extraction is {result.rows} rows, {result.nsky:.3f} sky rows per object row",
        f"Slit transmission {result.slit0:.4f}",
        f"Read noise {result.noise:.2f} e-, dark {result.ndark:.2f} e-",
        "",
        f"Median S/N {np.nanmedian(result.sn):.2f} per binned pixel, "
        f"{np.nanmedian(result.sn_per_resolution_element):.2f} per resolution element",
        f"Peak S/N   {np.nanmax(result.sn):.2f} at "
        f"{result.wave[np.nanargmax(result.sn)]:.0f} A",
    ]
    if result.nsky <= 0:
        lines.append(
            "NOTE: the extraction window is taller than the slit, so there are no "
            "sky rows and no sky-subtraction noise penalty is applied."
        )
    if extras is not None:
        lines += [
            "",
            f"Iodine region counts {extras.i2counts:.1f} e-",
        ]
        if extras.bmv is not None:
            lines += [
                f"Template B-V {extras.bmv:.3f}",
                f"Exposure meter {extras.expmeter:.4g}",
                f"RV precision {extras.precision:.2f} m/s",
            ]
    print("\n".join(lines), file=stream)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="obscalc",
        description="Signal-to-noise for the Levy spectrograph on the APF.",
    )
    grid = parser.add_argument_group("wavelength grid")
    grid.add_argument("--wvmn", type=float, help="blue limit (default: instrument's)")
    grid.add_argument("--wvmx", type=float, help="red limit (default: instrument's)")
    grid.add_argument("--dwv", type=float, default=10.0, help="step, Angstroms")

    inst = parser.add_argument_group("instrument")
    inst.add_argument(
        "--decker", default="W", choices=sorted(DECKERS), help="slit decker"
    )
    inst.add_argument(
        "--binning", default="1x1", help="spatial x dispersion, e.g. 2x1"
    )

    obs = parser.add_argument_group("observation")
    obs.add_argument("--mag", type=float, default=17.0, help="object magnitude")
    obs.add_argument(
        "--mtype",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = Vega/Johnson, 2 = AB",
    )
    obs.add_argument("--seeing", type=float, default=1.2, help="FWHM, arcsec")
    obs.add_argument("--airmass", type=float, default=1.1)
    obs.add_argument("--exptime", type=float, default=3600.0, help="seconds")
    obs.add_argument("--redshift", type=float, default=0.0)
    obs.add_argument("--template", default="", help="spectral template FITS file")
    obs.add_argument("--filter", default="", help="filter used to normalise it")
    obs.add_argument(
        "--vega-template",
        default=Observation.vega_template,
        help="Vega spectrum, for the Vega system",
    )
    obs.add_argument("--infil", help="legacy CARD/value parameter file")

    out = parser.add_argument_group("output")
    out.add_argument("--output", help="write the results table here (.ecsv/.csv/.fits)")
    out.add_argument("--plot", help="write a QA figure here (.png/.pdf)")
    out.add_argument("--table", action="store_true", help="print the full table")
    out.add_argument("--quiet", action="store_true", help="suppress the summary")

    listing = parser.add_argument_group("data")
    listing.add_argument(
        "--list-filters", action="store_true", help="list available filters and exit"
    )
    listing.add_argument(
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

    obs = Observation(
        seeing=args.seeing,
        airmass=args.airmass,
        mstar=args.mag,
        mtype=args.mtype,
        exptime=args.exptime,
        redshift=args.redshift,
        template=args.template,
        filter=args.filter,
        vega_template=args.vega_template,
    )

    if args.infil:
        cards = read_infil(args.infil)
        apply_infil_to_observation(obs, cards)
        decker = apply_infil_to_instrument(instr, cards)
        if decker:
            from .instruments.apf import set_decker

            set_decker(instr, decker)

    wvmn = args.wvmn if args.wvmn is not None else instr.wvmnx[0]
    wvmx = args.wvmx if args.wvmx is not None else instr.wvmnx[1]
    return tel, instr, obs, wavelength_grid(wvmn, wvmx, args.dwv)


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

    try:
        result = spec_calcs2n(wave, apf_thruput(wave), tel, instr, obs)
        extras = apf_extras(result, obs)
    except TemplateFilterMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    table = results_table(result)

    if not args.quiet:
        summarise(result, instr, obs, extras)
    if args.table:
        table.pprint_all()
    if args.output:
        table.write(args.output, overwrite=True)
        print(f"wrote {args.output}", file=sys.stderr)
    if args.plot:
        from .plots import save_qa_figure

        save_qa_figure(args.plot, result, instr, obs)
        print(f"wrote {args.plot}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
