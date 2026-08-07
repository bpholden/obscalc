"""Quality-assurance plots for a signal-to-noise calculation.

Takes a :class:`~obscalc.s2n.StackedResult`, so a dichroic instrument's two sides
are drawn on one set of axes with the crossover marked.
"""

import numpy as np

from .slit import gauss_slit


def _mark_sides(ax, result):
    """Draw a line at each dichroic crossover."""
    if len(result.sides) < 2:
        return
    for _, index, _ in result.sides[1:]:
        if index.size:
            ax.axvline(result.wave[index].min(), color="0.6", lw=0.8, ls=":")


def _panel_signal_to_noise(ax, result):
    ax.plot(result.wave, result.sn, label="per binned pixel")
    ax.plot(
        result.wave,
        result.sn_per_resolution_element,
        label="per resolution element",
        linestyle="--",
    )
    ax.set_ylabel("S/N")
    ax.set_title("Signal to noise")
    ax.legend(fontsize="small")


def _panel_counts(ax, result):
    ax.plot(result.wave, result.star, label="object")
    ax.plot(result.wave, result.sky, label="sky")
    # Read noise and dark are per detector, so these are steps rather than
    # horizontal lines once a dichroic is involved.
    ax.plot(
        result.wave,
        result.noise**2,
        color="0.4",
        linestyle=":",
        label="read noise$^2$",
    )
    ax.plot(result.wave, result.ndark, color="0.7", linestyle="-.", label="dark")
    ax.set_yscale("log")
    ax.set_ylabel("counts / binned pixel")
    ax.set_title("Noise budget")
    ax.legend(fontsize="small", loc="lower right")


def _panel_throughput(ax, result):
    ax.plot(result.wave, 100.0 * result.thru)
    ax.set_ylabel("throughput (%)")
    ax.set_title("End-to-end throughput")


def _panel_extinction(ax, result):
    ax.plot(result.wave, result.extinct)
    ax.set_ylabel("mag / airmass")
    ax.set_title("Atmospheric extinction")


def _panel_sky_brightness(ax, result):
    ax.plot(result.wave, result.magsky)
    ax.invert_yaxis()
    ax.set_ylabel("AB mag / arcsec$^2$")
    ax.set_title("Sky brightness")


def _panel_slit_loss(ax, instr, obs):
    seeing = np.linspace(0.4, 3.0, 40)
    fraction = [
        gauss_slit(instr.swidth / s, instr.sheight / s, 0.0, 0.0) for s in seeing
    ]
    ax.plot(seeing, fraction)
    ax.axvline(
        obs.seeing, color="0.5", linestyle=":", label=f"seeing = {obs.seeing:.2f}\""
    )
    ax.set_xlabel("seeing FWHM (arcsec)")
    ax.set_ylabel("fraction through slit")
    ax.set_title(f"Slit losses ({instr.swidth:g}\" x {instr.sheight:g}\" slit)")
    ax.legend(fontsize="small")


def _default_title(result, instr, obs):
    first = result.sides[0][2]
    system = "Vega" if first.mtype == 1 else "AB"
    parts = []
    if instr is not None:
        parts.append(f"{instr.name}  slit {instr.swidth:g}\"x{instr.sheight:g}\"")
    parts.append(f"{first.binr}x{first.binc} binning")
    parts.append(f"mag {first.mstar:g} {system}")
    parts.append(f"{obs.exptime:g} s")
    parts.append(f"airmass {obs.airmass:g}")
    parts.append(f"seeing {obs.seeing:g}\"")
    # Delivered resolving power, R/columns -- see S2NResult.resolving_power.
    # Shown for a single detector too, not just a dichroic pair.
    parts.append(
        " + ".join(
            f"{name} R={side.resolving_power:.0f}".lstrip()
            for name, _, side in result.sides
        )
    )
    return "  ".join(parts)


def qa_figure(result, instr, obs, title=None):
    """Six-panel QA figure for a :class:`~obscalc.s2n.StackedResult`.

    Returns the matplotlib ``Figure``; the caller decides whether to save or
    show it.  ``instr`` supplies the slit geometry for the last panel; for a
    double spectrograph either side will do, since they share a slit.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 11))

    _panel_signal_to_noise(axes[0, 0], result)
    _panel_counts(axes[0, 1], result)
    _panel_throughput(axes[1, 0], result)
    _panel_extinction(axes[1, 1], result)
    _panel_sky_brightness(axes[2, 0], result)
    if instr is not None:
        _panel_slit_loss(axes[2, 1], instr, obs)
    else:
        axes[2, 1].set_axis_off()

    for ax in list(axes[:2, :].ravel()) + [axes[2, 0]]:
        ax.set_xlabel("wavelength (Angstroms)")
        _mark_sides(ax, result)

    fig.suptitle(title or _default_title(result, instr, obs), fontsize="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def save_qa_figure(path, result, instr, obs, title=None, dpi=110):
    """Write :func:`qa_figure` to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    fig = qa_figure(result, instr, obs, title=title)
    fig.savefig(path, dpi=dpi)
    return path


def _thick_legend(ax, **kwargs):
    """Legend whose colour swatches are readable when the traces are hairlines."""
    legend = ax.legend(**kwargs)
    for line in legend.get_lines():
        line.set_linewidth(2.0)
    return legend


def sky_models_figure():
    """The Mauna Kea sky models, per source and combined.

    Documents the LRIS recovery described in :mod:`obscalc.sky`: the top panel
    shows each measurement separately, so the recovered red channel can be seen
    lying on top of the independent DEIMOS one, and the bottom shows the
    combined default against Mt Hamilton for scale.
    """
    import matplotlib.pyplot as plt

    from .sky import (
        _DEIMOS600_MIN,
        _LRIS_BLUE_MAX,
        RED_VALIDATION,
        coverage,
        maunakea_sky,
        mtham_sky,
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    sources = [
        ("lris_blue", "tab:blue", "recovered from LRIS blue"),
        ("deimos600", "tab:orange", "DEIMOS 600"),
        ("lris_red", "tab:green", "recovered from LRIS red"),
    ]
    for name, colour, label in sources:
        low, high = coverage(name)
        wave = np.arange(low, high, 2.0)
        axes[0].plot(
            wave,
            maunakea_sky(wave, model=name),
            colour,
            lw=0.7,
            label=f"{label}  ({low:.0f}-{high:.0f} A)",
        )
    axes[0].set_title(
        "Mauna Kea sky per source -- red overlays DEIMOS to "
        f"{RED_VALIDATION[0]:+.2f} mag, which validates undoing the throughput"
    )
    _thick_legend(axes[0], fontsize=8, ncol=2, loc="lower right")

    wave = np.arange(*coverage("combined"), 2.0)
    axes[1].plot(wave, maunakea_sky(wave), "k", lw=0.7, label="combined (default)")
    axes[1].plot(
        wave, mtham_sky(wave), "tab:red", lw=0.6, alpha=0.7, label="Mt Hamilton"
    )
    axes[1].set_title("Combined model, with Mt Hamilton for scale")
    _thick_legend(axes[1], fontsize=8, loc="lower right")
    axes[1].set_xlabel("wavelength (Angstroms)")

    for ax in axes:
        ax.axvspan(
            _LRIS_BLUE_MAX,
            _DEIMOS600_MIN,
            color="0.85",
            zorder=0,
            label="_nolegend_",
        )
        ax.set_ylim(25.0, 16.0)  # brighter upward
        ax.set_ylabel("AB mag / arcsec$^2$")

    fig.text(
        0.01,
        0.005,
        f"grey band {_LRIS_BLUE_MAX:.0f}-{_DEIMOS600_MIN:.0f} A is measured by "
        "neither model and is interpolated",
        fontsize=7,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def save_sky_models_figure(path, dpi=110):
    """Write :func:`sky_models_figure` to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    fig = sky_models_figure()
    fig.savefig(path, dpi=dpi)
    return path
