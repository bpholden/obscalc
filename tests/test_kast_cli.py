import numpy as np
import pytest
from astropy.table import Table

from obscalc.instruments.kast import DICHROICS
from obscalc.kast_cli import main


def test_default_run_reports_both_sides(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "[blue]" in out
    assert "[red]" in out
    assert "Overall median S/N" in out
    # Kast has no iodine cell, so no APF-only quantities.
    assert "RV precision" not in out
    assert "Iodine" not in out


def test_default_grid_is_the_wrapper_range(capsys, tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table["wave"].min() == 3150.0
    assert table["wave"].max() == 8000.0


def test_metadata_is_prefixed_per_side(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    # Two detectors, so keys carry a side initial; each stays within the eight
    # characters a FITS keyword allows.
    assert "b_readno" in table.meta
    assert "r_readno" in table.meta
    assert table.meta["r_readno"] / table.meta["b_readno"] == pytest.approx(3.8 / 3.7)
    assert table.meta["b_Rpix"] == 4254.0
    assert table.meta["r_Rpix"] == 3164.0
    # R is delivered: the per-pixel figure over the slit projection in pixels.
    assert table.meta["b_R"] == pytest.approx(4254.0 / table.meta["b_cols"])
    assert table.meta["r_R"] == pytest.approx(3164.0 / table.meta["r_cols"])
    assert all(len(key) <= 8 for key in table.meta)


def test_fits_output_has_no_long_keywords(tmp_path):
    path = tmp_path / "out.fits"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert len(table) > 100


@pytest.mark.parametrize("dichroic", sorted(DICHROICS))
def test_each_dichroic_runs(dichroic, capsys):
    assert main(["--dichroic", dichroic, "--quiet"]) == 0


@pytest.mark.parametrize("grism", ["G2", "G3"])
def test_each_measured_grism_runs(grism, capsys):
    assert main(["--grism", grism, "--quiet"]) == 0


def test_g1_is_rejected_by_argparse(capsys):
    # G1 has a resolving power in x_initkast but no throughput measurement, so
    # it is not offered.
    with pytest.raises(SystemExit):
        main(["--grism", "G1"])
    assert "invalid choice" in capsys.readouterr().err


def test_slitwidth_is_a_width_in_arcsec(capsys):
    assert main(["--slitwidth", "2.0", "--quiet"]) == 0
    assert main(["--slitwidth", "0", "--quiet"]) == 2
    assert "positive" in capsys.readouterr().err


def test_wider_slit_passes_more_light(capsys):
    outputs = []
    for width in ("1.0", "2.0"):
        main(["--slitwidth", width])
        outputs.append(capsys.readouterr().out)
    first = float(outputs[0].split("slit transmission ")[1].split(",")[0])
    second = float(outputs[1].split("slit transmission ")[1].split(",")[0])
    assert second > first


def test_template_requires_a_filter(capsys):
    assert main(["--template", "G5V_pickles_27.fits"]) == 2
    assert "must be given together" in capsys.readouterr().err


def test_template_run_succeeds(capsys):
    code = main(
        [
            "--template",
            "G5V_pickles_27.fits",
            "--filter",
            "Buser_V.dat",
            "--mtype",
            "1",
            "--quiet",
        ]
    )
    assert code == 0


def test_mismatched_template_exits_nonzero(capsys):
    code = main(
        [
            "--template",
            "G5V_pickles_27.fits",
            "--filter",
            "Buser_V.dat",
            "--redshift",
            "5.0",
        ]
    )
    assert code == 1
    assert "covers only" in capsys.readouterr().err


def test_writes_a_qa_figure_marking_the_dichroic(tmp_path):
    path = tmp_path / "qa.png"
    assert main(["--quiet", "--plot", str(path)]) == 0
    assert path.stat().st_size > 10_000


def test_signal_to_noise_is_finite_across_the_whole_grid(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert np.all(np.isfinite(table["s2n"]))
    assert np.all(table["s2n"] > 0)
    assert np.all(table["thru"] > 0)


def test_warns_where_the_configuration_outruns_its_throughput_measurement(capsys):
    """G3 is measured only to 4429 A, so d55 leaves a wide dead zone.

    The IDL extrapolated there and clamped to a small positive floor, giving
    counts that look plausible but mean nothing.
    """
    main(["--grism", "G3", "--dichroic", "d55"])
    out = capsys.readouterr().out
    assert "WARNING: no usable throughput" in out
    assert "4560-5490" in out

    # Pairing G3 with the dichroic it was measured against is far better.
    main(["--grism", "G3", "--dichroic", "d46"])
    matched = capsys.readouterr().out
    assert "4560-4590" in matched


def test_every_configuration_warns_about_the_red_end(capsys):
    # The red grating measurement stops at 7632 A but the default grid runs to
    # 8000, so this affects all of them.
    main([])
    assert "7690-8000" in capsys.readouterr().out


def test_narrowing_the_range_removes_the_red_warning(capsys):
    main(["--grism", "G2", "--dichroic", "d55", "--wvmx", "7630"])
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_dead_ranges_returns_nothing_for_a_clean_configuration():
    import numpy as np

    from obscalc.cli_common import dead_ranges
    from obscalc.instruments.kast import kast_spectrograph, kast_thruput
    from obscalc.s2n import Side, run_sides
    from obscalc.structures import Observation
    from obscalc.telescopes import lick_telescope

    wave = np.arange(5200.0, 7600.0, 10.0)
    blue, red = kast_spectrograph(dichroic="d46")
    thru = kast_thruput(wave, blue, red)
    result = run_sides(
        wave,
        lick_telescope(),
        [Side("red", red, np.arange(wave.size), thru)],
        Observation(seeing=1.5, mstar=19.0),
    )
    assert dead_ranges(result) == []


def test_infil_overrides_defaults(capsys, tmp_path):
    infil = tmp_path / "params.dat"
    infil.write_text("SEEING 0.9\nEXPTIME 1800\nBINC 2\nBINR 2\n")
    assert main(["--infil", str(infil)]) == 0
    out = capsys.readouterr().out
    assert 'SEEING   = 0.9"' in out
    assert "EXPTIME  = 1800 s" in out
    assert "BINNING  = 2x2" in out
