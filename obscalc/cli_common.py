"""Argument groups and output handling shared by the per-instrument drivers.

Everything here is instrument independent.  ``apf_cli`` and ``kast_cli`` add
their own dispersers, deckers and dichroics on top.
"""

import sys

import numpy as np
from astropy.table import Table

from .structures import Observation

#: Meta keys that describe one detector.  Kept to six characters so a side
#: prefix still fits in a FITS keyword.
_SIDE_META = ("slit0", "rows", "cols", "nsky", "readno", "ndark", "binc", "binr", "R")


def wavelength_grid(wvmn, wvmx, dwv):
    """The grid the IDL wrappers built: inclusive of both endpoints."""
    if dwv <= 0:
        raise ValueError(f"dwv must be positive, got {dwv}")
    if wvmx <= wvmn:
        raise ValueError(f"wvmx ({wvmx}) must exceed wvmn ({wvmn})")
    count = int((wvmx - wvmn) / dwv) + 1
    return wvmn + np.arange(count) * dwv


def add_observation_arguments(parser, seeing=1.2, exptime=3600.0, mag=17.0):
    """Add the parameters every instrument shares.

    The defaults differ per instrument because each IDL wrapper set its own --
    APF used 1.2 arcsec seeing, Kast 1.5.
    """
    group = parser.add_argument_group("observation")
    group.add_argument("--mag", type=float, default=mag, help="object magnitude")
    group.add_argument(
        "--mtype",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = Vega/Johnson, 2 = AB",
    )
    group.add_argument("--seeing", type=float, default=seeing, help="FWHM, arcsec")
    group.add_argument("--airmass", type=float, default=1.1)
    group.add_argument("--exptime", type=float, default=exptime, help="seconds")
    group.add_argument("--redshift", type=float, default=0.0)
    group.add_argument("--template", default="", help="spectral template FITS file")
    group.add_argument("--filter", default="", help="filter used to normalise it")
    group.add_argument(
        "--vega-template",
        default=Observation.vega_template,
        help="Vega spectrum, for the Vega system",
    )
    group.add_argument("--infil", help="legacy CARD/value parameter file")
    return group


def add_grid_arguments(parser):
    group = parser.add_argument_group("wavelength grid")
    group.add_argument("--wvmn", type=float, help="blue limit (default: instrument's)")
    group.add_argument("--wvmx", type=float, help="red limit (default: instrument's)")
    group.add_argument("--dwv", type=float, default=10.0, help="step, Angstroms")
    return group


def add_output_arguments(parser):
    group = parser.add_argument_group("output")
    group.add_argument(
        "--output", help="write the results table here (.ecsv/.csv/.fits)"
    )
    group.add_argument("--plot", help="write a QA figure here (.png/.pdf)")
    group.add_argument("--table", action="store_true", help="print the full table")
    group.add_argument("--quiet", action="store_true", help="suppress the summary")
    return group


def observation_from_args(args):
    return Observation(
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


def results_table(result):
    """Per-wavelength results from a :class:`~obscalc.s2n.StackedResult`."""
    table = Table()
    table["wave"] = result.wave
    table["s2n"] = result.sn
    table["s2n_resel"] = result.sn_per_resolution_element
    table["obj"] = result.star
    table["sky"] = result.sky
    table["tnoise"] = result.tnoise
    table["noise"] = result.noise
    table["thru"] = result.thru
    table["extinct"] = result.extinct
    table["magsky"] = result.magsky
    table["pixel"] = result.pixel

    table["wave"].unit = "Angstrom"
    table["pixel"].unit = "Angstrom"
    table["extinct"].unit = "mag"
    table["magsky"].unit = "mag"
    for name in ("obj", "sky", "tnoise", "noise"):
        table[name].unit = "count"

    multiple = len(result.sides) > 1
    for name, _, side in result.sides:
        # One detector: plain keys, so a single-channel table reads naturally.
        # Two or more: prefix with the side's initial to stay inside the eight
        # characters a FITS keyword allows.
        prefix = f"{name[0]}_" if multiple and name else ""
        values = {
            "slit0": side.slit0,
            "rows": side.rows,
            "cols": side.columns,
            "nsky": side.nsky,
            "readno": side.noise,
            "ndark": side.ndark,
            "binc": side.binc,
            "binr": side.binr,
            "R": side.R,
        }
        for key in _SIDE_META:
            table.meta[f"{prefix}{key}"] = values[key]

    first = result.sides[0][2]
    table.meta["mstar"] = first.mstar
    table.meta["mtype"] = first.mtype
    return table


#: Throughput at or below this is effectively dead: 0.01 per cent.
DEAD_THRUPUT = 1e-4


#: Ignore dead runs shorter than this many samples.  An echelle blaze function
#: has a true zero at each order edge, one or two grid points wide, which is
#: physics rather than a gap in the calibration; a configuration that has outrun
#: its throughput measurement goes dead over far more than that.
MIN_DEAD_SAMPLES = 3


def dead_ranges(result, floor=DEAD_THRUPUT, min_samples=MIN_DEAD_SAMPLES):
    """Contiguous wavelength ranges where the throughput is effectively zero.

    These arise where a configuration reaches past its throughput measurement.
    The IDL extrapolated there and then clamped the result to a small positive
    floor, so the counts are not meaningful rather than merely small.
    """
    dead = np.asarray(result.thru) <= floor
    if not dead.any():
        return []
    index = np.flatnonzero(dead)
    breaks = np.flatnonzero(np.diff(index) > 1)
    starts = np.r_[index[0], index[breaks + 1]]
    ends = np.r_[index[breaks], index[-1]]
    return [
        (float(result.wave[s]), float(result.wave[e]))
        for s, e in zip(starts, ends)
        if e - s + 1 >= min_samples
    ]


def summarise(result, obs, instr=None, extras=None, stream=None):
    """Print the configuration and a few summary numbers."""
    stream = sys.stdout if stream is None else stream
    first = result.sides[0][2]
    system = "Vega/Johnson" if first.mtype == 1 else "AB"

    lines = [
        f"SEEING   = {obs.seeing:g}\"",
        f"AIRMASS  = {obs.airmass:g}",
        f"EXPTIME  = {obs.exptime:g} s",
        f"MAG      = {first.mstar:g} ({system})",
        f"BINNING  = {first.binr}x{first.binc} (spatial x dispersion)",
    ]
    if instr is not None:
        lines.insert(0, f"SLIT     = {instr.swidth:g}\" x {instr.sheight:g}\"")
    if obs.template:
        lines.append(f"TEMPLATE = {obs.template} at z = {obs.redshift:g}")
        lines.append(f"FILTER   = {obs.filter}")

    for name, index, side in result.sides:
        label = f" [{name}]" if name else ""
        wave = result.wave[index]
        lines += [
            "",
            f"{wave.min():.0f}-{wave.max():.0f} A{label}: R = {side.R:.0f}, "
            f"{side.columns:.2f} pixels across the slit, "
            f"{side.rows} rows extracted, {side.nsky:.2f} sky rows per object row",
            f"  slit transmission {side.slit0:.4f}, read noise {side.noise:.2f} e-, "
            f"dark {side.ndark:.2f} e-",
            f"  median S/N {np.nanmedian(side.sn):.2f} per binned pixel, "
            f"{np.nanmedian(side.sn_per_resolution_element):.2f} per resolution element",
        ]
        if side.nsky <= 0:
            lines.append(
                "  NOTE: the extraction window is taller than the slit, so there "
                "are no sky rows and no sky-subtraction penalty is applied."
            )

    lines += [
        "",
        f"Overall median S/N {np.nanmedian(result.sn):.2f} per binned pixel",
        f"Peak S/N {np.nanmax(result.sn):.2f} at "
        f"{result.wave[np.nanargmax(result.sn)]:.0f} A",
    ]

    dead = dead_ranges(result)
    if dead:
        spans = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in dead)
        fraction = sum(hi - lo for lo, hi in dead) / (
            result.wave.max() - result.wave.min()
        )
        lines += [
            "",
            f"WARNING: no usable throughput over {spans} A ({fraction:.0%} of the "
            "range). This configuration reaches beyond its throughput "
            "measurement; narrow the wavelength range or change disperser.",
        ]

    if extras is not None:
        lines += ["", f"Iodine region counts {extras.i2counts:.1f} e-"]
        if extras.bmv is not None:
            lines += [
                f"Template B-V {extras.bmv:.3f}",
                f"Exposure meter {extras.expmeter:.4g}",
                f"RV precision {extras.precision:.2f} m/s",
            ]

    print("\n".join(lines), file=stream)


def write_outputs(args, result, obs, instr=None):
    """Handle ``--table``, ``--output`` and ``--plot``."""
    table = results_table(result)
    if args.table:
        table.pprint_all()
    if args.output:
        table.write(args.output, overwrite=True)
        print(f"wrote {args.output}", file=sys.stderr)
    if args.plot:
        from .plots import save_qa_figure

        save_qa_figure(args.plot, result, instr, obs)
        print(f"wrote {args.plot}", file=sys.stderr)
    return table
