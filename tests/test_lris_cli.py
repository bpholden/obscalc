import numpy as np
import pytest
from astropy.table import Table

from obscalc.instruments.lris import GRATINGS, GRISMS
from obscalc.lris_cli import main


def test_default_run_reports_both_sides(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "[blue]" in out
    assert "[red]" in out
    assert "Overall median S/N" in out
    # LRIS has no iodine cell, so no APF-only quantities.
    assert "RV precision" not in out
    assert "Iodine" not in out


def test_default_grid_is_the_wrapper_range(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table["wave"].min() == 3500.0
    assert table["wave"].max() == 10000.0


def test_metadata_is_prefixed_per_side(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table.meta["r_readno"] / table.meta["b_readno"] == pytest.approx(4.5 / 3.7)
    assert table.meta["b_Rpix"] == 7500.0
    assert table.meta["r_Rpix"] == 11820.0
    # R is delivered: a 1 arcsec slit spans 7.44 pixels, so about 1000 and 1600.
    assert table.meta["b_R"] == pytest.approx(1008.0, abs=1.0)
    assert table.meta["r_R"] == pytest.approx(1589.0, abs=1.0)
    assert all(len(key) <= 8 for key in table.meta)


def test_configuration_report_names_both_measurements(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "sens_LRISb_600_4000_D560.fits" in out
    assert "sens_LRISr_600_7500_D560.fits" in out
    assert "measured over 3101-5596 A" in out
    assert "measured over 5628-8191 A" in out


def test_the_default_configuration_warns_that_the_red_end_is_dead(capsys):
    """600/7500 stops at 8191 A but the default grid runs to 10000.

    Outside the measurement the throughput is dead, not held, so this is the same
    warning Kast gets rather than an LRIS-specific one.
    """
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "WARNING: no usable throughput over 5600-5620, 8200-10000 A" in out
    assert "28% of the range" in out


def test_a_fully_measured_configuration_does_not_warn(capsys):
    """B300 covers 2148-7652 and 400/8500 reaches 10354, so nothing is held."""
    assert main(["--grism", "B300", "--grating", "400/8500"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_narrowing_the_grid_leaves_only_the_handover_gap(capsys):
    """Trimming to 8100 A drops the red end, but 5600-5620 A cannot be trimmed.

    The dichroic hands over at 5600 A and the 600/7500 measurement starts at
    5628, so three grid points at the join are always held whatever the range.
    """
    assert main(["--wvmx", "8100"]) == 0
    out = capsys.readouterr().out
    assert "8200-10000" not in out
    assert "no usable throughput over 5600-5620 A" in out


@pytest.mark.parametrize("grism", sorted(GRISMS))
def test_each_grism_runs(grism):
    assert main(["--grism", grism, "--quiet"]) == 0


@pytest.mark.parametrize("grating", sorted(GRATINGS))
def test_each_grating_runs(grating):
    assert main(["--grating", grating, "--quiet"]) == 0


def test_unknown_dichroic_is_rejected_by_argparse():
    """lris_thruput.pro stopped on anything but D560; argparse says so first."""
    with pytest.raises(SystemExit):
        main(["--dichroic", "d46"])


def test_slitwidth_is_a_width_in_arcsec():
    assert main(["--slitwidth", "1.5", "--quiet"]) == 0
    assert main(["--slitwidth", "-1"]) == 2


def test_template_needs_a_filter():
    assert main(["--template", "G5V_pickles_27.fits"]) == 2


def test_template_and_filter_run_together(tmp_path):
    path = tmp_path / "out.ecsv"
    assert (
        main(
            [
                "--quiet",
                "--template",
                "G5V_pickles_27.fits",
                "--filter",
                "Buser_V.dat",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    table = Table.read(path)
    assert np.all(np.isfinite(table["s2n"]))
    assert np.all(table["s2n"] > 0)


def test_binning_reaches_both_sides(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--binning", "2x2", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table.meta["b_binr"] == table.meta["r_binr"] == 2
    assert table.meta["b_binc"] == table.meta["r_binc"] == 2
