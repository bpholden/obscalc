import numpy as np
import pytest
from astropy.table import Table

from obscalc.hires_cli import main
from obscalc.instruments.hires import DECKERS, EPOCHS


def test_default_run_reports_the_echelle_format(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Echelle orders 37-118" in out
    assert "free spectral range" in out
    assert "Cross-disperser" in out
    # HIRES has no iodine cell, so no APF-only quantities.
    assert "RV precision" not in out


def test_default_grid_is_the_wrapper_range(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    table = Table.read(path)
    assert table["wave"].min() == 3000.0
    assert table["wave"].max() == 9500.0


def test_default_binning_is_two_by_one(capsys):
    # x_inithires.pro sets bins = 2.
    assert main([]) == 0
    assert "BINNING  = 2x1" in capsys.readouterr().out


@pytest.mark.parametrize("decker", sorted(DECKERS))
def test_every_decker_runs(decker):
    assert main(["--decker", decker, "--quiet"]) == 0


@pytest.mark.parametrize("epoch", sorted(EPOCHS))
def test_every_epoch_runs(epoch):
    assert main(["--epoch", epoch, "--quiet"]) == 0


def test_decker_is_a_name_not_a_width(capsys):
    with pytest.raises(SystemExit):
        main(["--decker", "1.0"])
    assert "invalid choice" in capsys.readouterr().err


def test_the_delivered_resolving_power_is_the_same_for_both_epochs(capsys):
    """Changing the detector changes sampling, not resolution.

    x_inithires.pro scales R with 24/pixel_size, so the 15 micron detector has
    216000 against 135000 per native pixel -- a factor 1.6.  But its smaller
    pixels also make the same slit project onto 1.6x as many of them, 6.40
    against 4.00, and the two cancel exactly.  What sets the delivered resolving
    power is the slit and the grating, which neither epoch changes.
    """
    main(["--epoch", "old"])
    old = capsys.readouterr().out
    main(["--epoch", "new"])
    new = capsys.readouterr().out

    assert "R = 33748" in old and "4.00 pixels across the slit" in old
    assert "R = 33748" in new and "6.40 pixels across the slit" in new


def test_the_blaze_is_on_by_default_and_no_blaze_is_stated(capsys):
    assert main([]) == 0
    assert "each order's peak value" not in capsys.readouterr().out

    assert main(["--no-blaze"]) == 0
    assert "each order's peak value" in capsys.readouterr().out


def test_applying_the_blaze_lowers_the_median(capsys):
    def median(argv):
        main(argv)
        out = capsys.readouterr().out
        return float(out.split("Overall median S/N ")[1].split(" ")[0])

    assert median([]) < median(["--no-blaze"])


def test_the_blaze_nulls_do_not_trigger_the_dead_throughput_warning(capsys):
    """A sinc^2 zero at each order edge is physics, not a calibration gap.

    Those nulls are one or two grid points wide; a configuration that has outrun
    its throughput measurement, as Kast G3+d55 has, goes dead over hundreds of
    Angstroms and must still be reported.
    """
    main([])
    assert "WARNING: no usable throughput" not in capsys.readouterr().out

    from obscalc.kast_cli import main as kast_main

    kast_main(["--grism", "G3", "--dichroic", "d55"])
    assert "WARNING: no usable throughput" in capsys.readouterr().out


def test_short_decker_is_flagged_but_finite(capsys, tmp_path):
    # E5 is 1 arcsec tall, so there are no sky rows.
    path = tmp_path / "out.ecsv"
    assert main(["--decker", "E5", "--output", str(path)]) == 0
    assert "no sky rows" in capsys.readouterr().out
    assert np.all(np.isfinite(Table.read(path)["s2n"]))


def test_metadata_is_unprefixed_for_a_single_detector(tmp_path):
    path = tmp_path / "out.ecsv"
    assert main(["--quiet", "--output", str(path)]) == 0
    meta = Table.read(path).meta
    assert meta["Rpix"] == 216000.0
    assert meta["R"] == pytest.approx(216000.0 / meta["cols"])
    assert meta["binr"] == 2 and meta["binc"] == 1
    assert "b_R" not in meta


def test_template_requires_a_filter(capsys):
    assert main(["--template", "G5V_pickles_27.fits"]) == 2
    assert "must be given together" in capsys.readouterr().err


def test_template_run_succeeds():
    assert (
        main(
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
        == 0
    )


def test_writes_a_qa_figure(tmp_path):
    path = tmp_path / "qa.png"
    assert main(["--quiet", "--plot", str(path)]) == 0
    assert path.stat().st_size > 10_000


def test_infil_overrides_defaults(capsys, tmp_path):
    infil = tmp_path / "params.dat"
    infil.write_text("SEEING 1.1\nEXPTIME 600\nDECKER D1\nBINC 2\nBINR 3\n")
    assert main(["--infil", str(infil)]) == 0
    out = capsys.readouterr().out
    assert 'SEEING   = 1.1"' in out
    assert "EXPTIME  = 600 s" in out
    assert "BINNING  = 3x2" in out
    assert 'SLIT     = 1.148" x 14"' in out
