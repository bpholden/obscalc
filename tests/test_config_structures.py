import pytest

from obscalc import config
from obscalc.structures import (
    Observation,
    apply_infil_to_instrument,
    apply_infil_to_observation,
    parse_binning,
    read_infil,
)
from obscalc.instruments.apf import apf_spectrograph


def test_bundled_data_is_present():
    assert "Buser_V.dat" in config.available_filters()
    assert "alpha_lyr_stis_005.fits" in config.available_templates()
    assert config.EXTINCTION_DIR.joinpath("mthamextinct.dat").exists()


def test_resolve_finds_gzipped_file_when_asked_for_plain():
    # The xidl sources name this file without the .gz that is on disk.
    found = config.resolve("sens_APF_aug2022.fits", config.THRUPUT_DIR)
    assert found.name == "sens_APF_aug2022.fits.gz"


def test_resolve_reports_the_directory_it_tried():
    with pytest.raises(FileNotFoundError, match="nope.dat"):
        config.filter_file("nope.dat")


def test_filter_dir_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSCALC_FILTER_DIR", str(tmp_path))
    assert config.filter_dir() == tmp_path


def test_filter_dir_rejects_nonexistent_override(monkeypatch):
    monkeypatch.setenv("OBSCALC_FILTER_DIR", "/no/such/place")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        config.filter_dir()


def test_parse_binning_spatial_then_dispersion():
    assert parse_binning("1x1") == (1, 1)
    assert parse_binning("2x4") == (2, 4)
    with pytest.raises(ValueError):
        parse_binning("2")


def test_observation_defaults_match_x_obsinit():
    obs = Observation()
    assert (obs.seeing, obs.airmass, obs.mphase) == (0.7, 1.1, 0)
    assert (obs.exptime, obs.mstar, obs.mtype) == (3600.0, 17.0, 1)


def test_infil_round_trip(tmp_path):
    path = tmp_path / "params.dat"
    path.write_text(
        "SEEING 1.4\nAIRMASS 1.25\nEXPTIME 600\nMAGNITUDE 12.5\n"
        "MAGTYPE 2\nMPHASE 7\nDECKER W\nBINC 2\nBINR 1\nSWIDTH 1.5\n"
    )
    cards = read_infil(path)

    obs = apply_infil_to_observation(Observation(), cards)
    assert obs.seeing == 1.4
    assert obs.airmass == 1.25
    assert obs.exptime == 600.0
    assert obs.mstar == 12.5
    assert obs.mtype == 2
    assert obs.mphase == 7

    instr = apf_spectrograph()
    decker = apply_infil_to_instrument(instr, cards)
    assert decker == "W"
    assert instr.bind == 2
    assert instr.bins == 1
    assert instr.swidth == 1.5
