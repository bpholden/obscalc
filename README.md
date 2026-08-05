# obscalc

Signal-to-noise calculator for slit spectrographs. A Python port of the exposure
time calculator in [xidl](https://www.ucolick.org/~xavier/IDL/), whose entry
point was `Obs/S2N/spec_calcs2n.pro`.

Supported instruments:

| Instrument | Telescope | Channels | Command |
| --- | --- | --- | --- |
| Levy | APF 2.4 m | one, cross-dispersed echelle | `obscalc-apf` (also `obscalc`) |
| Kast | Shane 3 m, Lick | two, split by a dichroic | `obscalc-kast` |
| HIRES | Keck I 10 m | one, cross-dispersed echelle | `obscalc-hires` |
| DEIMOS | Keck II 10 m | one | `obscalc-deimos` |

The two echelles share `echelle.py`: order geometry, the blaze function, and
per-order throughput lookup. Both apply the blaze by default; `--no-blaze`
reproduces the IDL on either.

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

### HIRES

```sh
obscalc-hires --mag 15 --mtype 2 --exptime 1800 --decker C5 --epoch new
```

```
3000-9500 A: R = 216000, 6.40 pixels across the slit, 18 rows extracted, ...
  slit transmission 0.8421, read noise 6.60 e-, dark 9.00 e-
  median S/N 40.50 per binned pixel, 102.47 per resolution element

Echelle orders 37-118 across the range; free spectral range 25.6-260.3 A
Cross-disperser: blue setting below 3790 A, red above
```

HIRES is where the throughput stops being a measured curve and becomes a model,
which is what `hires_thruput.pro` does and what makes it unlike the others:

1. The echelle order containing a wavelength is `m = long(MLAMBDA / wave)`, with
   blaze centre `MLAMBDA / m` and free spectral range `centre / m` — 25 Å wide at
   3000 Å, 260 Å at 9500 Å.
2. Throughput is read from a table **at the order centre, not at the
   wavelength**, so it is constant across each order. Since the centre is the
   blaze peak, the result is a best-case value within the order.
3. Which of two tabulated cross-disperser curves applies depends on the order
   centre: the blue setting below 3800 Å, the red above.
4. The blaze function `(sin γ / γ)²` is **applied by default**, so the reported
   throughput is what each wavelength actually reaches. `hires_calcs2n.pro` never
   applied it — it passed `BLAZE=blaze` with `blaze` undefined, leaving
   `keyword_set(BLAZE)` false, and for the newer detector did not pass it at all —
   so it quoted every order's peak. `--no-blaze` restores that, raising the median
   throughput about 2.5×.

`--decker` is a name (`B2 B5 C1 C5 D1 D3 E4 E5`), as for APF, not a width in
arcsec as for Kast. `--epoch` selects the detector:

| Epoch | Pixels | R | Read noise | Dark |
| --- | --- | --- | --- | --- |
| `old` | 24 µm | 135000 | 4.3 e⁻ | 2.0 e⁻/px/hr |
| `new` (default) | 15 µm | 216000 | 2.2 e⁻ | 1.0 e⁻/px/hr |

Comparing epochs by per-pixel S/N is misleading. The newer detector has 1.29×
the throughput, but its smaller pixels raise R and so cover 0.625× the wavelength
per pixel — a net 11% loss per pixel and a 13% gain per resolution element. The
default binning is `2x1`, matching `x_inithires.pro`.

`x_inithires.pro` defines a third configuration, `flg = 3`, for MTHR on the TMT.
That is a different instrument on a different telescope with its own throughput
routine (`mthr_thruput.pro`) and is not ported.

### DEIMOS

```sh
obscalc-deimos --mag 22 --mtype 2 --exptime 1200 \
               --grating 1200G --cwave 7000 --slitwidth 1.0
```

```
4000-10000 A: R = 22727, 6.02 pixels across the slit, 13 rows extracted, ...
  slit transmission 0.7945, read noise 9.37 e-, dark 17.33 e-
  median S/N 1.82 per binned pixel, 4.46 per resolution element

Grating 1200G tilted to 7000 A: throughput from sens_DEIMOS_1200G.fits,
  measured over 4014-9348 A
```

DEIMOS is a single channel and needs no special S/N treatment — it reached
`spec_calcs2n` through `keck_calcs2n.pro` with `flg = 4`. What is particular to it
is that the throughput depends on **two** settings, and between them they pick one
of nine measured files:

| Grating | R | Tilts with their own measurement |
| --- | --- | --- |
| `600Z` | 11538 | 5000, 6000, 7000 Å (8000 reuses the 7000 Å file) |
| `900Z` | 17308 | 5000, 6000, 7000, 8000 Å — all four distinct |
| `1200G` | 22727 | one file for all four tilts |
| `1200B` | 22727 | one file for all four tilts |

`--cwave` is the grating tilt in Ångströms and must be one of 5000, 6000, 7000 or
8000 — `deimos_thruput.pro` compared `fix(str_instr.cwave)` against those exact
integers, so nothing in between exists. It selects the sensitivity measurement and
nothing else: it does not narrow the wavelength grid, so you can ask for
wavelengths a configuration does not record.

**Every configuration's measurement is narrower than the default 4000–10000 Å
grid**, so the driver says where the throughput is held rather than measured:

```
WARNING: 4000-4010, 9350-10000 A lie outside that measurement. Throughput there
is held at the nearest measured value, so those counts are an extrapolation.
```

`--slitwidth` is a width in arcsec here, as for Kast.

### Looking at plots

Either add `--plot out.png` to any run, or generate a representative set:

```sh
python scripts/make_plots.py            # writes ./plots (gitignored)
python scripts/make_plots.py /tmp/figs  # or somewhere else
open plots/                             # macOS
```

That writes fifteen figures: four APF configurations and three HIRES ones, each
set including a blaze on/off pair where the echelle order structure shows; three
Kast ones (including `G3 + d55`, where the throughput dead zone is obvious); four
DEIMOS ones (including a `900Z` tilt pair, where the measured range visibly
moves); and the Mauna Kea sky-model diagnostic from `plots.sky_models_figure`,
which shows the recovered LRIS red channel lying on top of the independent DEIMOS
measurement.

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
- **There is no moon-phase dependence for Mauna Kea at all.**
  `spec_calcs2n.pro` already forced `phase = 0L ;; Only New Moon so far` for
  DEIMOS, ESI and LRIS, so this changes nothing for those; it is a change for
  Keck I and HIRES, which passed a real phase to a table that only mattered
  outside the empirical range.

**The LRIS sky frames have their throughput divided back out.**
`bsky.eps_pang_parcsec.fits` and `rsky.eps_pang_parcsec_onemicron.fits` are in
electrons/s/Å/arcsec², so they already contain the throughput of the telescope
and instrument and are strictly valid only for the configuration they were taken
in. `maunakea_sky.pro` fed them to an f_lambda conversion anyway — its
`flg_sky = 2` model, `mkea_sky_LRIS_both.fits`, is bit-identical to the two files
summed across their overlap — which is why it returned about −19 AB mag/arcsec².

Rather than rescale detected counts per configuration, the throughput is undone
to recover a true surface brightness, which is what the engine wants and what any
instrument can then use. See below for how well that works.

**The APF throughput was treated as a smooth curve when it is per-order.** The
63 wavelengths in `sens_APF_nov2016.fits` are the 63 consecutive echelle order
centres, orders 62 to 124, matching `MLAMBDA/m` to better than 0.006 Å. It is one
measurement per order, taken at the blaze peak — the same structure as the HIRES
table. `apf_thruput.pro` interpolated it *at the wavelength*, which mixes adjacent
orders, and applied no blaze, so it reported every order's peak everywhere.
`apf_calcs2n.pro` computed the order number, blaze centre and free spectral range
(lines 113–121) and then discarded all three. Two corrections follow:

- **Per-order lookup**, unconditional. Each wavelength takes its own order's
  value. Median effect under 1%, but up to **20%** around 3800–4300 Å where the
  sensitivity curve rises steeply — 3822 Å belongs to order 121, centred at
  3851 Å, giving 0.059 rather than the interpolated 0.049.
- **The blaze**, on by default as for HIRES. Median throughput ×0.41, median S/N
  ×0.50 for a V=9 G star.

There is no option to restore the old interpolation; it was simply wrong.
`--no-blaze` gives the per-order peak, which is the closest thing to the IDL.

**This moves the APF's RV numbers, and in the right direction.**
`apf_extras.i2counts` is the median object count over 5000–6200 Å, so it falls
with the blaze applied — for a V=9 G star in 600 s: 6250 → 2762 counts, exposure
meter 2.02e8 → 8.92e7, and **RV precision 2.67 → 4.48 m/s**.

That is the better estimate. `A = 4.47`, `B = −1.58` come from empirical
measurements on real APF spectra, which carry the blaze, so blaze-inclusive counts
are the input the relation was calibrated against. The old un-blazed ETC fed it
peak-of-order counts and so reported a precision better than the instrument
achieves. Anything comparing against historical ETC output should expect the
newer, larger — and more honest — figure.

**DEIMOS's default grating did not exist, and its throughput was extrapolated.**
Two problems in one instrument:

- `x_initdeimos.pro` and `deimos_calcs2n_wrapper.pro` both default the grating to
  `'1200'`, which is not one of the four names (`600Z`, `900Z`, `1200G`, `1200B`)
  the routine then switches on — so the default configuration ran straight into
  `else: stop`. `1200G` is the default here.
- `deimos_thruput.pro` extrapolated its sensitivity curve and clipped only at
  zero with `> 0.`, which catches a curve falling negative but not one rising
  absurdly. Extrapolating the `600Z` measurement at its 5000 Å tilt, which stops
  at 8035 Å, reaches **0.848 by 10000 Å — more than twice the best efficiency ever
  measured for any DEIMOS configuration (0.358)**. The nearest measured value is
  held instead. The zero clip is kept: `sens_DEIMOS_900_500nm` holds 133 slightly
  negative efficiencies inside its own range.

**`spec_calcs2n.pro`'s DEIMOS sky branch was dead code.** It selected
`flg_sky = 1` when `str_instr.grating EQ '1200'`, but `x_initdeimos.pro` only ever
set `600Z`, `900Z`, `1200G` or `1200B`, so the 1200-line sky model was never
reached and DEIMOS always used `flg_sky = 0`. That is what it gets here too.

**The HIRES detector boost was extrapolated off the end of its table.**
`hires_thru_newccd` passed straight to `interpol`, whose table starts at 3153.9 Å
while HIRES is used from 3000 Å. Continuing the first interval trebles the boost
to 32.8 by 3000 Å, turning a 0.3% throughput into 9.8%. A quantum efficiency
ratio cannot be extrapolated that way, so the end values are held instead.

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

### Recovering the LRIS sky, and how far to trust it

The LRIS frames were taken with blue grism 400/3400, red grating 600/5000 and
dichroic 500, and **no sensitivity measurement for that configuration exists in
the xidl tree**. The nearest same-ruling D560 curves are used instead, and each
channel is restricted to where it dominates.

Choosing the curve matters, and blaze wavelength matters more than coverage:

| Channel | Curve used | Rejected alternative |
| --- | --- | --- |
| blue (400/3400, blaze 3400 Å) | `600/4000 D560` | `300/5000 D560` — blazed 1600 Å away; gives a sky that *darkens* by 1.4 mag from 4000→5000 Å where the real sky brightens |
| red (600/5000) | `600/7500 D560` | `400/8500 D560` — wider, but off by +0.59 mag |

**The red channel is the validation.** It adds no coverage the DEIMOS models
lack, so it is kept only to check the method: recovering it with `600/7500 D560`
reproduces the independent DEIMOS 600 sky — a different night, a different
instrument — to **+0.07 mag median with 0.31 scatter** over 5700–8190 Å.
`sky.validate_against_deimos()` computes it and a test asserts it. That is the
main evidence the undo is sound.

The blue channel is the useful one: it is the only Mauna Kea measurement here
blueward of 5000 Å, giving a median 22.67 AB mag/arcsec² over 3102–4999 Å.

Usable Mauna Kea models:

| Model | Covers | Notes |
| --- | --- | --- |
| `combined` (default) | 3102–9999 Å | `lris_blue` below 5000, `deimos600` above 5200 |
| `lris_blue` | 3102–4999 Å | recovered; stops at the dichroic 500 handover |
| `deimos600` | 5001–9999 Å | unreliable below ~5200 Å, still ramping off its blue edge |
| `deimos1200` | 6281–9329 Å | |
| `lris_red` | 5628–8190 Å | validation only |

Two caveats on `combined`. Nothing measures the **5000–5200 Å bridge** — the blue
channel has fallen off the dichroic and DEIMOS has not come up off its blue edge
(24.1 mag at 5001 Å against 22.5 at 5200) — so it is interpolated; the join is
smooth to better than 0.6 mag between adjacent points. And the blue half carries
whatever error the mismatched grism curve leaves, which is *not* bounded by the
red channel's +0.07 mag: that number validates the red curve, not the blue one.
Supplying the throughput for the as-observed configuration would remove the
guesswork on both sides.

Outside a model's range the nearest measured value is held.
`sky.median_sky_magnitude` exists to catch a future file whose units are wrong,
which is how `mkea_sky_LRIS_both.fits` was caught.

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
| `data/sky/bsky.*`, `data/sky/rsky.*` | LRIS sky frames, Keck I 2017-05-27/28, in e⁻/s/Å/arcsec²; throughput undone at read time |
| `data/thruput/sens_LRIS*.fits.gz` | xidl `Obs/S2N/THRU_PUT_DATA`; used to undo the LRIS sky throughput |
| `data/thruput/sens_DEIMOS_*.fits.gz` | xidl `Obs/S2N/THRU_PUT_DATA`; nine grating/tilt combinations |
| `data/extinction/mthamextinct.dat` | xidl `Spec/Longslit/calib/extinction` |
| `data/filters/` | expcalc `Data/filters` (Buser, Cousins, SDSS, Gaia) |
| `data/templates/` | expcalc `Data/templates` (Pickles library, SN Ia, QSO, CALSPEC Vega) |

The sky model is a single measurement with no moon-phase dependence: `mtham_sky`
accepts a `phase` argument and ignores it, exactly as the IDL did.

## Scope

Four instruments: APF and Kast at Mt Hamilton, HIRES on Keck I, DEIMOS on
Keck II. All four supported sites have extinction and sky models, with usable
Mauna Kea sky coverage from 3102 to 9999 Å. `atmosphere.extinction_for` and
`sky.sky_for` raise `NotImplementedError` for any other telescope rather than
quietly substituting the wrong site.

`sky_for` takes an instrument name as well as a telescope, mirroring the nested
`case str_instr.name` inside `spec_calcs2n.pro`'s Keck II branch, because Keck
chose its sky model per instrument.

Still unported, both on Keck II and so needing no new site work:

- **ESI**, single channel with one throughput curve — the most straightforward
  remaining piece.
- **LRIS**, the involved one: two channels like Kast, and `lris_thruput.pro`
  branches on both dichroic and grating.

MTHR (`flg = 3` in `x_inithires.pro`) is on the TMT and would need a new
telescope and site.
