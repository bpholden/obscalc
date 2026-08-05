import itertools

import numpy as np
import pytest
from astropy.table import Table

from obscalc.deimos_cli import main
from obscalc.instruments.deimos import CENTRAL_WAVES, GRATINGS


def test_default_run_reports_the_configuration(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Grating 1200G tilted to 7000 A" in out
    assert "sens_DEIMOS_1200G.fits" in out
    assert "measured over 4014-9348 A" in out
    # DEIMOS has no iodine cell.
    assert "RV precision" not in out


def test_default_grid_is_the_wrapper_range(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table["wave"].min() == 4000.0
    assert table["wave"].max() == 10000.0


@pytest.mark.parametrize(
    "grating,cwave", list(itertools.product(sorted(GRATINGS), CENTRAL_WAVES))
)
def test_every_configuration_runs(grating, cwave):
    assert main(["--grating", grating, "--cwave", str(cwave), "--quiet"]) == 0


def test_the_invalid_idl_default_grating_is_rejected_by_argparse(capsys):
    # x_initdeimos.pro defaulted to '1200', which its own case list rejects.
    with pytest.raises(SystemExit):
        main(["--grating", "1200"])
    assert "invalid choice" in capsys.readouterr().err


def test_an_unmeasured_tilt_is_rejected_by_argparse(capsys):
    with pytest.raises(SystemExit):
        main(["--cwave", "6500"])
    assert "invalid choice" in capsys.readouterr().err


def test_resolving_power_differs_between_gratings(capsys):
    main(["--grating", "600Z"])
    assert "R = 11538" in capsys.readouterr().out
    main(["--grating", "1200G"])
    assert "R = 22727" in capsys.readouterr().out


def test_the_tilt_changes_which_file_is_used(capsys):
    main(["--grating", "900Z", "--cwave", "5000"])
    assert "sens_DEIMOS_900_500nm.fits" in capsys.readouterr().out
    main(["--grating", "900Z", "--cwave", "8000"])
    assert "sens_DEIMOS_900_800nm.fits" in capsys.readouterr().out


def test_the_default_grid_warns_that_it_outruns_the_measurement(capsys):
    """Every configuration's measurement is narrower than 4000-10000 A."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "lie outside that measurement" in out
    assert "9350-10000" in out


def test_narrowing_the_range_removes_the_warning(capsys):
    assert main(["--wvmn", "4020", "--wvmx", "9340"]) == 0
    assert "lie outside that measurement" not in capsys.readouterr().out


def test_slitwidth_is_a_width_in_arcsec(capsys):
    assert main(["--slitwidth", "0.75", "--quiet"]) == 0
    assert main(["--slitwidth", "0", "--quiet"]) == 2
    assert "positive" in capsys.readouterr().err


def test_wider_slit_passes_more_light(capsys):
    def transmission(width):
        main(["--slitwidth", width])
        out = capsys.readouterr().out
        return float(out.split("slit transmission ")[1].split(",")[0])

    assert transmission("1.5") > transmission("0.75")


def test_metadata_is_unprefixed_for_a_single_detector(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    meta = Table.read(path).meta
    assert meta["R"] == pytest.approx(22727.3)
    assert "b_R" not in meta


def test_signal_to_noise_is_finite_and_positive(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert np.all(np.isfinite(table["s2n"]))
    assert np.all(table["s2n"] > 0)
    assert np.all(table["thru"] >= 0)


def test_template_requires_a_filter(capsys):
    assert main(["--template", "sn1a10d_template.fits"]) == 2
    assert "must be given together" in capsys.readouterr().err


def test_template_run_succeeds():
    assert (
        main(
            [
                "--template",
                "sn1a10d_template.fits",
                "--filter",
                "sdss_r.dat",
                "--quiet",
            ]
        )
        == 0
    )


def test_writes_a_qa_figure(tmp_path):
    path = tmp_path / "qa.png"
    assert main(["--quiet", "--plot", str(path)]) == 0
    assert path.stat().st_size > 10_000


def test_infil_overrides_defaults(capsys, tmp_path):
    infil = tmp_path / "params.dat"
    infil.write_text("SEEING 1.0\nEXPTIME 3600\nBINC 2\nBINR 2\n")
    assert main(["--infil", str(infil)]) == 0
    out = capsys.readouterr().out
    assert 'SEEING   = 1"' in out
    assert "EXPTIME  = 3600 s" in out
    assert "BINNING  = 2x2" in out
