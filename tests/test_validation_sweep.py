"""Sweeps over the whole parameter space the web ETC exposes.

There is no IDL available to diff against, so these check that every reachable
configuration produces finite, physically ordered results, and that the only
rejections are genuine template/filter coverage failures.
"""

import itertools

import numpy as np
import pytest

from obscalc import (
    Observation,
    apf_spectrograph,
    apf_telescope,
    apf_thruput,
    spec_calcs2n,
)
from obscalc import config
from obscalc.apf_extras import apf_extras
from obscalc.instruments.apf import DECKERS
from obscalc.photometry import TemplateFilterMismatch, read_filter, read_template

BINNINGS = ["1x1", "2x1", "2x2", "4x4"]
SEEINGS = [0.7, 1.2, 2.5]


@pytest.fixture(scope="module")
def grid():
    wave = np.arange(3742.0, 7700.0, 10.0)
    return wave, apf_telescope(), apf_thruput(wave)


def test_every_configuration_is_finite_and_positive(grid):
    wave, tel, thru = grid
    checked = 0
    for decker, binning, seeing, mtype in itertools.product(
        sorted(DECKERS), BINNINGS, SEEINGS, (1, 2)
    ):
        bins, bind = (int(x) for x in binning.split("x"))
        instr = apf_spectrograph(decker=decker, bins=bins, bind=bind)
        obs = Observation(seeing=seeing, mstar=12.0, mtype=mtype, exptime=1200.0)
        result = spec_calcs2n(wave, thru, tel, instr, obs)

        label = f"{decker} {binning} {seeing}\" mtype={mtype}"
        assert np.all(np.isfinite(result.sn)), label
        assert np.all(result.sn > 0), label
        assert np.all(result.star > 0), label
        assert np.all(result.sky >= 0), label
        assert 0.0 < result.slit0 <= 1.0, label
        checked += 1
    assert checked == len(DECKERS) * len(BINNINGS) * len(SEEINGS) * 2


def test_short_deckers_are_the_only_ones_without_sky_rows(grid):
    """The 3 arcsec deckers, T and W, are where the IDL produced NaN S/N."""
    wave, tel, thru = grid
    without_sky = set()
    for decker, seeing in itertools.product(sorted(DECKERS), SEEINGS):
        instr = apf_spectrograph(decker=decker)
        result = spec_calcs2n(
            wave, thru, tel, instr, Observation(seeing=seeing, mstar=12.0)
        )
        if result.nsky <= 0:
            without_sky.add((decker, seeing))

    assert without_sky == {("T", 1.2), ("T", 2.5), ("W", 1.2), ("W", 2.5)}
    assert all(DECKERS[decker][1] == 3.0 for decker, _ in without_sky)


def test_all_templates_and_filters_either_work_or_report_no_coverage(grid):
    """The only acceptable failure is a filter redder than the template."""
    wave, tel, thru = grid
    instr = apf_spectrograph(decker="N")
    templates = [t for t in config.available_templates() if "alpha_lyr" not in t]
    ran = 0

    for template, filter_name, mtype in itertools.product(
        templates, config.available_filters(), (1, 2)
    ):
        obs = Observation(
            seeing=1.2,
            mstar=12.0,
            mtype=mtype,
            exptime=1200.0,
            template=template,
            filter=filter_name,
        )
        try:
            result = spec_calcs2n(wave, thru, tel, instr, obs)
            extras = apf_extras(result, obs)
        except TemplateFilterMismatch:
            template_wave, _ = read_template(template)
            filter_wave, _ = read_filter(filter_name)
            # Justify the rejection: the filter is not fully inside the template.
            assert (filter_wave.max() > template_wave.max() * (1 + obs.redshift)) or (
                filter_wave.min() < template_wave.min() * (1 + obs.redshift)
            ), f"{template} + {filter_name} rejected without a coverage gap"
            continue

        label = f"{template} + {filter_name} mtype={mtype}"
        assert np.all(np.isfinite(result.sn)), label
        assert extras.i2counts > 0, label
        assert extras.precision > 0, label
        ran += 1

    assert ran > 150


def test_signal_to_noise_falls_monotonically_with_magnitude(grid):
    wave, tel, thru = grid
    instr = apf_spectrograph(decker="N")
    medians = [
        np.median(
            spec_calcs2n(
                wave, thru, tel, instr, Observation(seeing=1.2, mstar=mag)
            ).sn
        )
        for mag in (8.0, 10.0, 12.0, 14.0, 16.0)
    ]
    assert medians == sorted(medians, reverse=True)


def test_rv_precision_degrades_monotonically_with_magnitude(grid):
    wave, tel, thru = grid
    instr = apf_spectrograph(decker="N")
    precisions = []
    for mag in (8.0, 10.0, 12.0, 14.0):
        obs = Observation(
            seeing=1.2,
            mstar=mag,
            mtype=1,
            exptime=1200.0,
            template="G5V_pickles_27.fits",
            filter="Buser_V.dat",
        )
        result = spec_calcs2n(wave, thru, tel, instr, obs)
        precisions.append(apf_extras(result, obs).precision)
    assert precisions == sorted(precisions)
    # A V=8 G star in 20 minutes should be at the m/s level, not hundreds.
    assert precisions[0] < 5.0
