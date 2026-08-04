import numpy as np
import pytest
from astropy.table import Table

from obscalc.cli import main, results_table, wavelength_grid
from obscalc.instruments.apf import DECKERS


def test_wavelength_grid_matches_the_idl_wrapper():
    # nwv = long((wvmx-wvmn)/dwv) + 1, inclusive of both ends.
    grid = wavelength_grid(3750.0, 7700.0, 10.0)
    assert grid.size == 396
    assert grid[0] == 3750.0
    assert grid[-1] == 7700.0


def test_wavelength_grid_rejects_bad_ranges():
    with pytest.raises(ValueError, match="dwv"):
        wavelength_grid(4000.0, 5000.0, 0.0)
    with pytest.raises(ValueError, match="wvmx"):
        wavelength_grid(5000.0, 4000.0, 10.0)


def test_default_run_succeeds(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Median S/N" in out
    assert "Slit transmission" in out


def test_template_requires_a_filter(capsys):
    assert main(["--template", "G5V_pickles_27.fits"]) == 2
    assert "must be given together" in capsys.readouterr().err


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


def test_listing_data_files(capsys):
    assert main(["--list-filters"]) == 0
    assert "Buser_V.dat" in capsys.readouterr().out
    assert main(["--list-templates"]) == 0
    assert "alpha_lyr_stis_005.fits" in capsys.readouterr().out


def test_every_decker_runs_from_the_command_line(capsys):
    for decker in sorted(DECKERS):
        assert main(["--decker", decker, "--quiet"]) == 0


def test_writes_an_ecsv_table_that_reads_back(tmp_path, capsys):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert {"wave", "s2n", "obj", "sky", "thru"} <= set(table.colnames)
    assert table["wave"].unit == "Angstrom"
    assert table.meta["rows"] == 9
    assert np.all(np.isfinite(table["s2n"]))


def test_writes_a_fits_table(tmp_path):
    path = tmp_path / "out.fits"
    assert main(["--quiet", "--output", str(path)]) == 0
    assert len(Table.read(path)) > 100


def test_writes_a_qa_figure(tmp_path):
    path = tmp_path / "qa.png"
    assert main(["--quiet", "--plot", str(path)]) == 0
    assert path.stat().st_size > 10_000


def test_infil_overrides_command_line_defaults(tmp_path, capsys):
    infil = tmp_path / "params.dat"
    infil.write_text("SEEING 0.8\nEXPTIME 300\nDECKER M\nBINC 2\nBINR 2\n")
    assert main(["--infil", str(infil)]) == 0
    out = capsys.readouterr().out
    assert 'SEEING   = 0.8"' in out
    assert "EXPTIME  = 300 s" in out
    assert "BINNING  = 2x2" in out
    assert 'DECKER   = 1" x 8"' in out


def test_results_table_carries_the_scalars_in_metadata():
    from obscalc import Observation, apf_spectrograph, apf_telescope, apf_thruput
    from obscalc.s2n import spec_calcs2n

    wave = np.arange(4000.0, 7000.0, 50.0)
    result = spec_calcs2n(
        wave,
        apf_thruput(wave),
        apf_telescope(),
        apf_spectrograph(),
        Observation(seeing=1.2),
    )
    table = results_table(result)
    assert table.meta["nsky"] == pytest.approx(result.nsky)
    assert table.meta["readno"] == pytest.approx(result.noise)
    assert len(table) == wave.size
