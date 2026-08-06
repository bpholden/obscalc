import numpy as np
import pytest

from obscalc.webapi import (
    MAX_NEGATIVE_COUNTS,
    NO_OVERLAP_MESSAGE,
    calculate,
    empty_payload,
)

# The keys s2n_param.parse_return() left in its output dictionary.  The web UI
# and csv_gen.py read these, so the set must not shrink.
PARSE_RETURN_KEYS = {
    "wave",
    "s2n",
    "obj",
    "objperwavesec",
    "noise",
    "sky",
    "cts",
    "js2n",
    "jobj",
    "jnoise",
    "jsky",
    "com",
    "dich",
    "msg",
    "errormsg",
    "i2counts",
    "exp",
    "precision",
}

GOOD_REQUEST = {
    "inst": "apf",
    "mag": "9.0",
    "mtype": "1",
    "seeing": "1.2",
    "airmass": "1.1",
    "exptime": "600",
    "binning": "1x1",
    "slitwidth": "N",
    "template": "G5V_pickles_27.fits",
    "ffilter": "Buser_V.dat",
    "redshift": "0.0",
}


def test_empty_payload_has_the_expected_keys():
    assert set(empty_payload()) == PARSE_RETURN_KEYS


def test_a_good_request_fills_every_series():
    payload = calculate(GOOD_REQUEST)
    assert set(payload) == PARSE_RETURN_KEYS
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""

    n = len(payload["wave"])
    assert n > 100
    for key in ("s2n", "obj", "sky", "noise"):
        assert len(payload[key]) == n
        assert all(len(pair) == 2 for pair in payload[key])
        assert len(payload["j" + key]) == n
        # The pair lists carry [wavelength, value]; the j-lists just the value.
        assert payload[key][0][1] == payload["j" + key][0]
    assert all(len(row) == 5 for row in payload["cts"])
    assert len(payload["cts"]) == n


def test_noise_is_broadcast_as_a_constant():
    payload = calculate(GOOD_REQUEST)
    assert len(set(payload["jnoise"])) == 1


def test_extras_are_reported():
    payload = calculate(GOOD_REQUEST)
    assert payload["i2counts"] > 0
    assert payload["exp"] > 0
    assert payload["precision"] > 0


def test_values_are_plain_floats_for_json():
    payload = calculate(GOOD_REQUEST)
    assert isinstance(payload["wave"][0], float)
    assert isinstance(payload["js2n"][0], float)
    assert not isinstance(payload["js2n"][0], np.floating)


def test_string_defaults_apply_when_parameters_are_missing():
    payload = calculate({})
    assert payload["msg"] == ""
    assert len(payload["wave"]) > 100
    # No template, so no colour-dependent quantities.
    assert payload["precision"] == 0.0


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"mag": "bright"}, "Mag"),
        ({"seeing": "-1"}, "Seeing"),
        ({"exptime": "0"}, "Exp. time"),
        ({"mtype": "7"}, "Mag. Type"),
        ({"binning": "nonsense"}, "CCD Binning"),
        ({"slitwidth": "Z"}, "Slitwidth"),
    ],
)
def test_bad_parameters_are_reported_without_calculating(override, expected):
    payload = calculate({**GOOD_REQUEST, **override})
    assert expected in payload["msg"]
    assert payload["wave"] == []


def test_template_without_a_filter_is_rejected():
    payload = calculate({**GOOD_REQUEST, "ffilter": ""})
    assert "filter" in payload["msg"]


def test_a_template_redshifted_off_the_filter_sets_errormsg():
    payload = calculate({**GOOD_REQUEST, "redshift": "5.0"})
    assert payload["errormsg"] == NO_OVERLAP_MESSAGE
    assert payload["msg"] == ""
    assert payload["wave"] == []


def test_negative_count_threshold_is_the_one_bad_obj_used():
    assert MAX_NEGATIVE_COUNTS == 5


# --- instrument dispatch -------------------------------------------------


def test_unknown_instruments_are_refused_not_answered_with_apf():
    """The web forms post `inst`; returning APF numbers for another would be wrong.

    Every instrument the ETC serves is now ported, so this is the guard against a
    typo or a retired name rather than a gap in coverage.
    """
    inst = "nirspec"
    payload = calculate({**GOOD_REQUEST, "inst": inst})
    assert inst in payload["msg"]
    assert "apf" in payload["msg"]
    assert payload["wave"] == []
    assert payload["js2n"] == []


def test_instrument_defaults_to_apf_when_absent():
    payload = calculate({key: v for key, v in GOOD_REQUEST.items() if key != "inst"})
    assert payload["msg"] == ""
    assert payload["i2counts"] > 0


def test_instrument_name_is_case_and_space_insensitive():
    for spelling in ("APF", " apf ", "Apf"):
        assert calculate({**GOOD_REQUEST, "inst": spelling})["msg"] == ""


def test_registered_instruments_are_what_the_package_supports():
    from obscalc.instruments import available_instruments

    assert available_instruments() == ["apf", "deimos", "hires", "kast", "lris"]


def test_backend_owns_the_slitwidth_semantics():
    # For APF slitwidth is a decker letter, so a numeric width is a bad request.
    payload = calculate({**GOOD_REQUEST, "slitwidth": "1.0"})
    assert "Slitwidth" in payload["msg"]
    assert "decker" in payload["msg"]


def test_unknown_parameters_are_passed_through_without_complaint():
    # kast and lris post a dichroic; an APF request carrying stray keys should
    # still work rather than fail validation.
    payload = calculate({**GOOD_REQUEST, "dichroic": "d55", "grating": "600/7500"})
    assert payload["msg"] == ""
    assert len(payload["wave"]) > 100


def test_get_backend_raises_for_an_unknown_instrument():
    from obscalc.instruments import get_backend

    with pytest.raises(NotImplementedError, match="nirspec"):
        get_backend("nirspec")


# --- kast through the web adapter ---------------------------------------


KAST_REQUEST = {
    "inst": "kast",
    "mag": "19.0",
    "mtype": "2",
    "seeing": "1.5",
    "airmass": "1.1",
    "exptime": "900",
    "binning": "1x1",
    "slitwidth": "1.5",
    "dichroic": "d46",
    "grism": "G2",
    "grating": "600/7500",
    "redshift": "0.0",
}


def test_kast_request_fills_the_same_payload_shape():
    payload = calculate(KAST_REQUEST)
    assert set(payload) == PARSE_RETURN_KEYS
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""
    n = len(payload["wave"])
    assert n == 486  # 3150-8000 A at 10 A
    for key in ("s2n", "obj", "sky", "noise"):
        assert len(payload[key]) == n
        assert len(payload["j" + key]) == n
    assert np.all(np.isfinite(payload["js2n"]))


def test_kast_read_noise_is_piecewise_across_the_dichroic():
    """Two detectors, so `noise` is no longer a single broadcast value."""
    payload = calculate(KAST_REQUEST)
    assert len(set(np.round(payload["jnoise"], 6))) == 2


def test_kast_has_no_apf_only_extras():
    payload = calculate(KAST_REQUEST)
    assert payload["i2counts"] is None
    assert payload["exp"] is None
    assert payload["precision"] is None


def test_kast_dichroic_changes_where_the_read_noise_steps():
    def step_wavelength(dichroic):
        payload = calculate({**KAST_REQUEST, "dichroic": dichroic})
        noise = np.array(payload["jnoise"])
        wave = np.array(payload["wave"])
        return wave[np.flatnonzero(np.diff(noise) != 0)[0] + 1]

    assert step_wavelength("d46") == pytest.approx(4600.0)
    assert step_wavelength("d55") == pytest.approx(5500.0)


@pytest.mark.parametrize(
    "override,label",
    [
        ({"grism": "G9"}, "Grism"),
        ({"grating": "1200/5000"}, "Grating"),
        ({"dichroic": "d99"}, "Dichroic"),
        ({"slitwidth": "wide"}, "Slitwidth"),
    ],
)
def test_kast_bad_instrument_parameters_are_reported(override, label):
    payload = calculate({**KAST_REQUEST, **override})
    assert label in payload["msg"]
    assert payload["wave"] == []


def test_kast_accepts_a_numeric_slitwidth_that_apf_would_reject():
    # The same value means different things per instrument.
    assert calculate({**KAST_REQUEST, "slitwidth": "1.0"})["msg"] == ""
    assert "Slitwidth" in calculate({**GOOD_REQUEST, "slitwidth": "1.0"})["msg"]


# --- lris through the web adapter ---------------------------------------


LRIS_REQUEST = {
    "inst": "lris",
    "mag": "20.0",
    "mtype": "2",
    "seeing": "1.0",
    "airmass": "1.1",
    "exptime": "3600",
    "binning": "1x1",
    "slitwidth": "1.0",
    "dichroic": "D560",
    "grism": "B600",
    "grating": "600/7500",
    "redshift": "0.0",
}


def test_lris_request_fills_the_same_payload_shape():
    payload = calculate(LRIS_REQUEST)
    assert set(payload) == PARSE_RETURN_KEYS
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""
    n = len(payload["wave"])
    assert n == 651  # 3500-10000 A at 10 A
    for key in ("s2n", "obj", "sky", "noise"):
        assert len(payload[key]) == n
        assert len(payload["j" + key]) == n
    assert np.all(np.isfinite(payload["js2n"]))


def test_lris_read_noise_steps_at_the_dichroic():
    payload = calculate(LRIS_REQUEST)
    noise = np.array(payload["jnoise"])
    wave = np.array(payload["wave"])
    assert len(set(np.round(noise, 6))) == 2
    step = wave[np.flatnonzero(np.diff(noise) != 0)[0] + 1]
    assert step == pytest.approx(5600.0)


def test_lris_has_no_apf_only_extras():
    payload = calculate(LRIS_REQUEST)
    assert payload["i2counts"] is None
    assert payload["exp"] is None
    assert payload["precision"] is None


@pytest.mark.parametrize(
    "override,label",
    [
        ({"grism": "B1200"}, "Grism"),
        ({"grating": "150/7500"}, "Grating"),
        ({"dichroic": "d46"}, "Dichroic"),
        ({"slitwidth": "wide"}, "Slitwidth"),
    ],
)
def test_lris_bad_instrument_parameters_are_reported(override, label):
    payload = calculate({**LRIS_REQUEST, **override})
    assert label in payload["msg"]
    assert payload["wave"] == []


HIRES_REQUEST = {
    "inst": "hires",
    "mag": "15.0",
    "mtype": "2",
    "seeing": "0.7",
    "airmass": "1.1",
    "exptime": "1800",
    "binning": "2x1",
    "slitwidth": "C5",
    "redshift": "0.0",
}


def test_hires_request_fills_the_same_payload_shape():
    payload = calculate(HIRES_REQUEST)
    assert set(payload) == PARSE_RETURN_KEYS
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""
    n = len(payload["wave"])
    assert n == 651  # 3000-9500 A at 10 A
    assert np.all(np.isfinite(payload["js2n"]))
    assert payload["i2counts"] is None  # no iodine cell


def test_hires_slitwidth_is_a_decker_like_apf_not_a_width_like_kast():
    assert calculate({**HIRES_REQUEST, "slitwidth": "D1"})["msg"] == ""
    payload = calculate({**HIRES_REQUEST, "slitwidth": "1.0"})
    assert "Decker" in payload["msg"]
    assert payload["wave"] == []


def test_hires_epoch_can_be_selected():
    assert calculate({**HIRES_REQUEST, "epoch": "old"})["msg"] == ""
    assert "Epoch" in calculate({**HIRES_REQUEST, "epoch": "ancient"})["msg"]


def test_hires_blaze_flag_lowers_the_counts():
    plain = np.array(calculate(HIRES_REQUEST)["jobj"])
    blazed = np.array(calculate({**HIRES_REQUEST, "blaze": "true"})["jobj"])
    assert np.median(blazed) < np.median(plain)


def test_hires_read_noise_is_a_single_value():
    # One detector, unlike Kast.
    assert len(set(np.round(calculate(HIRES_REQUEST)["jnoise"], 6))) == 1


DEIMOS_REQUEST = {
    "inst": "deimos",
    "mag": "22.0",
    "mtype": "2",
    "seeing": "0.7",
    "airmass": "1.1",
    "exptime": "1200",
    "binning": "1x1",
    "slitwidth": "1.0",
    "grating": "1200G",
    "cwave": "7000",
    "redshift": "0.0",
}


def test_deimos_request_fills_the_same_payload_shape():
    payload = calculate(DEIMOS_REQUEST)
    assert set(payload) == PARSE_RETURN_KEYS
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""
    assert len(payload["wave"]) == 601  # 4000-10000 A at 10 A
    assert np.all(np.isfinite(payload["js2n"]))
    assert payload["i2counts"] is None  # no iodine cell


def test_deimos_needs_both_a_grating_and_a_tilt():
    """cwave is the second half of the configuration, unique to DEIMOS."""
    assert calculate({**DEIMOS_REQUEST, "cwave": "5000"})["msg"] == ""
    payload = calculate({**DEIMOS_REQUEST, "cwave": "6500"})
    assert "Central Wavelength" in payload["msg"]
    assert payload["wave"] == []


def test_deimos_rejects_the_invalid_idl_default_grating():
    payload = calculate({**DEIMOS_REQUEST, "grating": "1200"})
    assert "Grating" in payload["msg"]


def test_deimos_tilt_changes_the_throughput():
    blue = np.array(calculate({**DEIMOS_REQUEST, "grating": "900Z", "cwave": "5000"})["jobj"])
    red = np.array(calculate({**DEIMOS_REQUEST, "grating": "900Z", "cwave": "8000"})["jobj"])
    assert not np.allclose(blue, red)


def test_deimos_read_noise_is_a_single_value():
    assert len(set(np.round(calculate(DEIMOS_REQUEST)["jnoise"], 6))) == 1


def test_deimos_sky_model_follows_the_grating():
    """spec_calcs2n.pro's flg_sky = 1 selected the matching sky measurement.

    It tested `str_instr.grating EQ '1200'`, and deimos_calcs2n_wrapper.pro
    defaults the grating to '1200', which x_initdeimos.pro writes into the
    structure before its case statement rejects it.  So the branch was meant for
    the 1200 line gratings, and they get the sky measured with that grating.
    """
    from obscalc.instruments.deimos import deimos_spectrograph
    from obscalc.sky import sky_for

    for grating, model in (
        ("1200G", "deimos1200"),
        ("1200B", "deimos1200"),
        ("600Z", "deimos600"),
        ("900Z", "deimos600"),
    ):
        instr = deimos_spectrograph(grating=grating)
        assert sky_for("KeckII", instr).model == model, grating


def test_the_grating_changes_the_sky_a_deimos_request_sees():
    sky_1200 = np.array(calculate({**DEIMOS_REQUEST, "grating": "1200G"})["jsky"])
    sky_900 = np.array(calculate({**DEIMOS_REQUEST, "grating": "900Z"})["jsky"])
    assert not np.allclose(sky_1200, sky_900)


def test_kast_template_run():
    payload = calculate(
        {
            **KAST_REQUEST,
            "template": "G5V_pickles_27.fits",
            "ffilter": "Buser_V.dat",
        }
    )
    assert payload["msg"] == ""
    assert payload["errormsg"] == ""
    assert np.all(np.isfinite(payload["js2n"]))
