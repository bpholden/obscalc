#!/usr/bin/env python3
"""Generate a set of QA figures to look at.

    python scripts/make_plots.py [outdir]      # default: ./plots

Writes one figure per configuration below, plus the Mauna Kea sky-model
diagnostic.  Nothing here is needed by the package; it exists so the plots are
easy to regenerate and eyeball.
"""

import sys
from pathlib import Path

import numpy as np

from obscalc.cli_common import wavelength_grid
from obscalc.instruments.apf import DEFAULT_RANGE as APF_RANGE
from obscalc.instruments.apf import apf_spectrograph, apf_thruput
from obscalc.instruments.kast import DEFAULT_RANGE as KAST_RANGE
from obscalc.instruments.kast import (
    kast_spectrograph,
    kast_thruput,
    split_wavelengths,
)
from obscalc.instruments.hires import DEFAULT_RANGE as HIRES_RANGE
from obscalc.instruments.hires import hires_spectrograph, hires_thruput
from obscalc.plots import save_qa_figure, save_sky_models_figure
from obscalc.s2n import Side, run_sides
from obscalc.structures import Observation
from obscalc.telescopes import apf_telescope, keck_telescope, lick_telescope

#: APF cases: (stem, decker, binning, blaze, magnitude, system, seconds).  Like
#: HIRES the Levy is an echelle, so the blaze pair is worth comparing: with the
#: blaze off you see each order's peak throughput, which is what apf_thruput.pro
#: reported at every wavelength.
APF_CASES = [
    ("apf-W-mag13-AB", "W", (1, 1), True, 13.0, 2, 1200.0),
    ("apf-W-mag13-AB-peak", "W", (1, 1), False, 13.0, 2, 1200.0),
    ("apf-N-mag9-Vega", "N", (1, 1), True, 9.0, 1, 600.0),
    ("apf-M-mag15-AB-2x1", "M", (2, 1), True, 15.0, 2, 3600.0),
]

#: Kast cases: (stem, grism, dichroic, slit, magnitude, system, seconds).
KAST_CASES = [
    ("kast-G2-d55-mag18", "G2", "d55", 1.5, 18.0, 2, 1800.0),
    ("kast-G2-d46-mag18", "G2", "d46", 1.5, 18.0, 2, 1800.0),
    # The poor pairing: G3 is measured only to 4429 A, so d55 leaves a wide
    # dead zone.  Worth seeing next to the one above.
    ("kast-G3-d55-mag18", "G3", "d55", 1.5, 18.0, 2, 1800.0),
]

#: HIRES cases: (stem, decker, epoch, blaze, magnitude, system, seconds).
HIRES_CASES = [
    ("hires-C5-new-mag15", "C5", "new", True, 15.0, 2, 1800.0),
    ("hires-C5-new-mag15-peak", "C5", "new", False, 15.0, 2, 1800.0),
    ("hires-C1-old-mag12", "C1", "old", True, 12.0, 2, 1800.0),
]


def apf_figure(outdir, stem, decker, binning, blaze, mag, mtype, exptime):
    wave = wavelength_grid(*APF_RANGE, 10.0)
    tel = apf_telescope()
    instr = apf_spectrograph(decker=decker, bins=binning[0], bind=binning[1])
    obs = Observation(seeing=1.2, mstar=mag, mtype=mtype, exptime=exptime)
    thru = apf_thruput(wave, instr, blaze=blaze)
    sides = [Side("", instr, np.arange(wave.size), thru)]
    result = run_sides(wave, tel, sides, obs)
    return save_qa_figure(outdir / f"{stem}.png", result, instr, obs)


def kast_figure(outdir, stem, grism, dichroic, slit, mag, mtype, exptime):
    wave = wavelength_grid(*KAST_RANGE, 10.0)
    tel = lick_telescope()
    blue, red = kast_spectrograph(grism=grism, dichroic=dichroic, slit=slit)
    thru = kast_thruput(wave, blue, red)
    blue_index, red_index = split_wavelengths(wave, dichroic)
    obs = Observation(seeing=1.5, mstar=mag, mtype=mtype, exptime=exptime)
    result = run_sides(
        wave,
        tel,
        [
            Side("blue", blue, blue_index, thru[blue_index]),
            Side("red", red, red_index, thru[red_index]),
        ],
        obs,
    )
    return save_qa_figure(outdir / f"{stem}.png", result, blue, obs)


def hires_figure(outdir, stem, decker, epoch, blaze, mag, mtype, exptime):
    wave = wavelength_grid(*HIRES_RANGE, 10.0)
    tel = keck_telescope("KeckI")
    instr = hires_spectrograph(decker=decker, epoch=epoch, str_tel=tel)
    obs = Observation(seeing=0.7, mstar=mag, mtype=mtype, exptime=exptime)
    thru = hires_thruput(wave, instr, blaze=blaze)
    result = run_sides(
        wave, tel, [Side("", instr, np.arange(wave.size), thru)], obs
    )
    return save_qa_figure(outdir / f"{stem}.png", result, instr, obs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0] if argv else "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    written = [save_sky_models_figure(outdir / "mauna-kea-sky-models.png")]
    for case in APF_CASES:
        written.append(apf_figure(outdir, *case))
    for case in KAST_CASES:
        written.append(kast_figure(outdir, *case))
    for case in HIRES_CASES:
        written.append(hires_figure(outdir, *case))

    for path in written:
        print(path)
    print(f"\n{len(written)} figures in {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
