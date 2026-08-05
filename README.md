# obscalc

Signal-to-noise calculator for slit spectrographs. A Python port of the exposure
time calculator in [xidl](https://www.ucolick.org/~xavier/IDL/), whose entry
point was `Obs/S2N/spec_calcs2n.pro`.

Supported instruments:

| Instrument | Telescope | Channels | Command |
| --- | --- | --- | --- |
| Levy | APF 2.4 m | one | `obscalc-apf` (also `obscalc`) |
| Kast | Shane 3 m, Lick | two, split by a dichroic | `obscalc-kast` |

The engine is generic; everything named `apf_*` or `kast_*` is not. `s2n.py`,
`slit.py`, `photometry.py`, `idl_compat.py`, `structures.py`, `cli_common.py`,
`plots.py` and `webapi.py` know nothing about any particular instrument. Three
registries hold what is instrument specific: `atmosphere.EXTINCTION` and
`sky.SKY`, keyed on telescope name, and `instruments.BACKENDS`, keyed on the
`inst` parameter the web forms post.

Adding an instrument means: a definition and throughput curve under
`instruments/`, a `Backend` subclass (see `instruments/base.py`) registered in
`BACKENDS`, an extinction curve and sky model in the other two registries, and
its own command line module using `cli_common`. No edit to `webapi.py` or
`s2n.py` should be required.

A backend describes *detectors* rather than assuming there is one. Each returns a
list of `Side` objects — which wavelengths that detector records, its instrument
parameters and its throughput there — and `s2n.run_sides` runs the engine once
per side and stacks the results onto a shared grid. Read noise, dark counts and
resolving power then vary across the dichroic split, exactly as
`kast_calcs2n.pro` arranged by hand.

## Install

```sh
pip install -e .
pip install -e '.[test]'   # to run the tests
python -m pytest
```

The calibration data, ten filter curves and twelve spectral templates ship with
the package, so nothing external is required.

## Command line

```sh
obscalc --mag 9 --mtype 1 --exptime 600 --decker N \
        --template G5V_pickles_27.fits --filter Buser_V.dat \
        --output s2n.ecsv --plot qa.png
```

```
DECKER   = 0.5" x 8"
BINNING  = 1x1 (spatial x dispersion)
SEEING   = 1.2"
AIRMASS  = 1.1
EXPTIME  = 600 s
MAG      = 9 (Vega/Johnson)
TEMPLATE = G5V_pickles_27.fits at z = 0
FILTER   = Buser_V.dat

Slit width projects to 2.000 pixels
Extraction is 9 rows, 1.111 sky rows per object row
Slit transmission 0.3171
Read noise 7.95 e-, dark 10.95 e-

Median S/N 77.94 per binned pixel, 110.22 per resolution element
Peak S/N   83.88 at 6622 A

Iodine region counts 6276.6 e-
Template B-V 0.688
Exposure meter 2.027e+08
RV precision 2.66 m/s
```

`--list-filters` and `--list-templates` show what is bundled. `--infil` reads the
legacy `CARD value` parameter files. Results tables are written through
`astropy.io.ascii` or `astropy.io.fits`, chosen by the output extension.

### Kast

```sh
obscalc-kast --mag 18 --mtype 2 --exptime 1800 \
             --dichroic d55 --grism G2 --grating 600/7500 --slitwidth 1.5
```

Both sides are reported separately, then together:

```
3150-5490 A [blue]: R = 4254, 3.47 pixels across the slit, 11 rows extracted, ...
  slit transmission 0.6447, read noise 12.27 e-, dark 0.01 e-
  median S/N 23.34 per binned pixel, 43.47 per resolution element

5500-8000 A [red]: R = 3164, 3.47 pixels across the slit, 11 rows extracted, ...
  slit transmission 0.6447, read noise 12.60 e-, dark 0.01 e-
  median S/N 27.70 per binned pixel, 51.60 per resolution element
```

Note that `--slitwidth` is a **width in arcsec** for Kast but a **decker name**
for APF, following the two IDL wrappers.

**Watch the disperser/dichroic pairing.** Each grism was measured against one
dichroic, and the throughput measurements do not span the default 3150-8000 Å
grid:

| Disperser | Measured over |
| --- | --- |
| G2 (blue) | 3221-5241 Å |
| G3 (blue) | 3169-4429 Å |
| 600/7500 (red) | 5136-7632 Å |

`kast_thruput.pro` extrapolated past the red end of a measurement and then
clamped the result to a small positive floor, so the counts there look plausible
but mean nothing. That behaviour is preserved, and the drivers now say where it
bites:

```
WARNING: no usable throughput over 4560-5490, 7690-8000 A (26% of the range).
```

That example is `--grism G3 --dichroic d55`, the worst pairing: G3 is measured
only to 4429 Å but d55 asks the blue side to work up to 5500 Å. `--grism G3
--dichroic d46` drops it to 7%, and every configuration flags 7690-8000 Å
because the red grating measurement stops at 7632 Å. Narrow the range with
`--wvmx` to work only where there are measurements.

G1 has a resolving power in `x_initkast.pro` but no throughput measurement at
all, so `kast_thruput.pro` hit an `else: stop` for the very grism
`x_initkast.pro` defaulted to. It is not offered here.

## Library

```python
import numpy as np
from obscalc import Observation, apf_spectrograph, apf_telescope, apf_thruput, spec_calcs2n

wave = np.arange(3742.0, 7700.0, 10.0)
result = spec_calcs2n(
    wave,
    apf_thruput(wave),
    apf_telescope(),
    apf_spectrograph(decker="W", bins=1, bind=1),
    Observation(seeing=1.2, mstar=13.0, mtype=2, exptime=1200.0),
)
result.sn, result.star, result.sky, result.sn_per_resolution_element
```

`spec_calcs2n` keeps the IDL's signature and `S2NResult` carries the ten fields
of the IDL `fstrct` plus the intermediates its callers recomputed.

## Replacing the IDL in the expcalc web ETC

`expcalc` currently builds an IDL command string, `Popen`s `idl_wrapper`, and
recovers the numbers by regexp-matching the wrapper's printed output
(`s2n_param.py`). `obscalc.webapi.calculate` returns the same dictionary
`parse_return` produced, so that path becomes a direct call:

```python
from obscalc.webapi import calculate

@route('/gen_inst_s2n', method='ANY')
def gen_inst_s2n():
    return calculate(request.params)
```

No subprocess, no `env-for-xidl`, no screen scraping. Parameter validation moves
into `webapi._coerce` for the parameters every spectrograph shares, and into the
backend for the rest; both report problems in the same `msg` field the existing
forms already display.

`calculate` dispatches on `inst`, defaulting to `apf`. Because only APF is
ported, a request naming any of the other instruments the ETC serves
(`kast`, `lris`, `esi`, `deimos`, `hires`) is **refused** with a message rather
than silently answered with APF numbers:

```python
calculate({"inst": "lris", ...})["msg"]
# "Unknown instrument 'lris'. This calculator serves apf."
```

Parameters a backend does not recognise are passed through untouched, so a form
that posts a `dichroic` or `grating` will not fail validation. Note that
`slitwidth` means different things per instrument — a decker letter for APF and
HIRES, a width in arcsec for kast, lris, esi and deimos — which is why it is the
backend's business and not `webapi`'s.

## Differences from the IDL

The port fixes four defects rather than reproducing them, so results will not
match the IDL numerically. Each is commented at the site of the change.

**Negative `nsky` made the S/N NaN.** `spec_calcs2n.pro` always applied the
sky-subtraction penalty `(1 + 1/nsky)`, but `nsky` — sky rows per object row —
goes negative when the extraction window is taller than the slit. For the APF
`W` decker at 1.2" seeing the extraction is 9 rows against a 7-row slit, giving
`nsky = -0.222`, `(1 + 1/nsky) = -3.5`, and the square root of a negative
number at **383 of 396 wavelengths**. This is why the IDL wrappers filtered
their output through `where(finite(s2n))`. As in `apf_calcs2n.pro`, the penalty
is now dropped when there are no sky rows. Affects the 3" deckers, `T` and `W`,
at seeing of 1.2" or worse.

**`mtham_sky.pro:56` compared mismatched arrays.** `where(wave LT mtham_swv)`
compares the wavelength grid against the whole sky table, which IDL silently
truncates to the shorter of the two. Now compares against the table minimum.

**`precision_value` took the logarithm twice.** `i2counts_value` returned
`alog10(median(counts)) * binc * binc`. `expmeter_value` then treated that as a
logarithm (consistent), while `precision_value` applied `alog10` to it again.
The published coefficients `A = 4.47`, `B = -1.58` only make sense against
`log10(counts)`, which is 3–5 for APF; feeding them `log10(log10(counts))`
inflated the reported RV precision by about two orders of magnitude — 290 m/s
instead of 2.7 m/s for a V=9 G star in 600 s. Multiplying a logarithm by
`binc**2` is not a count scaling under any reading either. So
`apf_extras.i2counts` returns plain median counts and each consumer takes its
own logarithm once. **This changes a number the web UI displays**; the exposure
meter reading is unaffected.

**Mauna Kea's analytic sky fallback is gone.** `maunakea_sky.pro` interpolated an
empirical measurement where one existed and otherwise fell back to a table of
Vega-system sky brightness against wavelength and moon phase, bilinearly
interpolated. That fallback is removed by decision: empirical measurements only.
Two consequences, both documented in `sky.py`:

- The `/NOEMPIR` switch no longer exists. It never worked anyway — the keyword
  was declared `NOEMPIRI=noempiri` but tested as `keyword_set(NOEMPIRIC)`, so
  every caller asking to skip the empirical model silently got it. Removing the
  fallback also disposes of the second bug in that routine, where the moon-phase
  index reused `ngd` from the preceding wavelength search.
- **There is no moon-phase dependence for Mauna Kea at all**, and nothing
  measured blueward of 5000 Å. `spec_calcs2n.pro` already forced
  `phase = 0L ;; Only New Moon so far` for DEIMOS, ESI and LRIS, so this changes
  nothing for those; it is a change for Keck I and HIRES, which passed a real
  phase to a table that only mattered outside the empirical range.

**Kast's per-detector read noise was clobbered.** `x_initkast.pro` says 3.7
electrons for the blue detector and 3.8 for the red, but wrote
`kastinstr.readno = 3.7` followed by `kastinstr.readno = 3.8` on a two-element
array, so the second assignment overwrote both and every Kast calculation used
3.8 on each side. The per-detector values are used here. (The same pattern
appears in `x_initapflowspec.pro`, which is not ported.)

**A `-99` from `single_spec2mag` propagated silently.** A template that does not
cover its normalising filter now raises `TemplateFilterMismatch` instead of
carrying the sentinel into the count rates. The web adapter turns that into the
same "filter and template do not overlap" message the app shows today. The IDL
also never checked the Vega template's magnitude; that check was added.

### IDL semantics deliberately preserved

Reproduced in `idl_compat.py`, each with a test, because the obvious numpy call
behaves differently:

- `interpol` extrapolates linearly beyond its table; `np.interp` clamps.
- `interpol` accepts a descending abscissa — the APF sensitivity file is stored
  in descending wavelength order.
- `long()` truncates toward zero, so the `long(x + 0.999)` idiom is a ceiling.
- `linterp` substitutes a fill value rather than extrapolating.
- `tsum` is trapezoidal integration over an inclusive index range.

`x_gaussslit`'s 199×199 summation is vectorised but otherwise unchanged,
including the `radius >= 99` cutoff, the use of the un-offset radius, and the
half-in/half-out treatment of cells exactly on the slit edge. That discretisation
is why slit loss versus seeing is a staircase rather than a smooth curve.

`spec_calcs2n` reads `bind` as the dispersion factor and `bins` as the spatial
one, while `apf_calcs2n_wrapper.pro` filled `bins` from the first character of
`"1x1"` and `bind` from the second. Both conventions are preserved;
`structures.parse_binning` is the single place that maps the string.

### Data problem found, not fixed here

`mkea_sky_LRIS_both.fits` — `flg_sky = 2`, the model `maunakea_sky.pro` selects
for LRIS — **is not in the same units as the other sky models** and is therefore
not offered. Its fluxes have a median of 0.23 where the two DEIMOS files sit near
5e-18, and there is no `BUNIT` to say what they are. Put through the f_lambda
conversion the IDL applies to every model, it yields sky brightnesses near −19 AB
mag/arcsec², which is unphysical; `maunakea_sky.pro` produces the same nonsense.
The file and that routine were both last modified 2025-03-12, so this looks
unfinished rather than intended. Asking for it raises with that explanation, and
`sky.median_sky_magnitude` exists to catch the same mistake in a future file.

This leaves the usable Mauna Kea coverage at:

| Model | `flg_sky` | Covers |
| --- | --- | --- |
| `deimos600` (default) | 0 | 5001–9999 Å |
| `deimos1200` | 1 | 6281–9329 Å |
| `lris` | 2 | unusable, see above |

So a Keck instrument working blueward of 5000 Å currently has no measured sky to
interpolate; outside a model's range the nearest measured value is held.

### Known bug in expcalc, not fixed here

`calc_exp.py:binning_opts` rewrites the APF `binning` parameter into
`spatialbinning`/`spectralbinning` and deletes `binning`, but `insts.py` defines
a regexp only for `binning`. `build_exec_str` iterates over the regexp keys, so
both new keys are dropped and **APF binning never reaches the IDL** — every APF
run through the web ETC uses the 1×1 default. `webapi.calculate` takes
`binning` directly and honours it, which means it will not reproduce the current
web output for any binning other than 1×1.

## Validation

There is no IDL or GDL on the development machine, so **no end-to-end comparison
against the original was possible**, and the bug fixes above mean the two would
not agree anyway. Validation is therefore:

- Leaf functions checked against the literal tables in the IDL source
  (`x_fluxjohnson`, `mtham_trans`, the PSF profile, the AB zero point).
- Vega through `Buser_V.dat` gives AB −0.0102, and a Vega-system round trip
  returns 0 to 1e-9 — an independent check on `single_spec2mag`.
- Template B−V colours order correctly from O5V through G5V to M5V.
- 168 configurations (7 deckers × 4 binnings × 3 seeings × 2 magnitude systems)
  and 192 template/filter pairs produce finite, positive, correctly ordered
  results; the only rejections are genuine coverage failures.
- Scaling checks: S/N as √t when source dominated, counts as 10^(0.4Δm), signal
  falling with airmass and with seeing.

What this does **not** establish is that any IDL expression was read correctly.
The highest-risk area is `slit.py`, a 199×199 sum with three interacting
flux-accounting branches and no closed form to check against. If IDL becomes
available, dumping `fstrct` over 3742–7700 Å at 10 Å for a few configurations
would turn this into a real regression test.

## Data provenance

| Path | Source |
| --- | --- |
| `data/thruput/sens_APF_nov2016.fits.gz` | xidl `Obs/S2N/THRU_PUT_DATA`; a standard star observed by S. Vogt |
| `data/thruput/sens_Kast*.fits.gz` | xidl `Obs/S2N/THRU_PUT_DATA`; blue from 2011 Aug 29, red from 2009 Mar 18 |
| `data/sky/lick_sky_d55_2011aug29.fits.gz` | xidl `Obs/Sky/Empirical`; one dark-sky measurement at Mt Hamilton |
| `data/sky/mkea_sky_*.fits.gz` | xidl `Obs/Sky/Empirical`; new-moon Mauna Kea sky from DEIMOS data |
| `data/extinction/mthamextinct.dat` | xidl `Spec/Longslit/calib/extinction` |
| `data/filters/` | expcalc `Data/filters` (Buser, Cousins, SDSS, Gaia) |
| `data/templates/` | expcalc `Data/templates` (Pickles library, SN Ia, QSO, CALSPEC Vega) |

The sky model is a single measurement with no moon-phase dependence: `mtham_sky`
accepts a `phase` argument and ignores it, exactly as the IDL did.

## Scope

Two instruments: APF and Kast, both at Mt Hamilton using `mtham_trans` and
`mtham_sky`.

Four sites are supported for extinction and sky — APF, Lick-3m, Keck I and
Keck II — so the Mauna Kea groundwork for the remaining Keck instruments (LRIS,
ESI, DEIMOS, HIRES) is in place and tested. `atmosphere.extinction_for` and
`sky.sky_for` raise `NotImplementedError` for any other telescope rather than
quietly substituting the wrong site.

`sky_for` takes an instrument name as well as a telescope, mirroring the nested
`case str_instr.name` inside `spec_calcs2n.pro`'s Keck II branch, because Keck
chose its sky model per instrument.

What each Keck instrument still needs: its `x_init*` definition, its `*_thruput`
curve and sensitivity files, a `Backend`, and a CLI module. LRIS is the most
involved — `lris_thruput.pro` branches on both dichroic and grating.
