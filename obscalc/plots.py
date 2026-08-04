"""Quality-assurance plots for a signal-to-noise calculation."""

import numpy as np

from .slit import gauss_slit


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
    ax.axhline(result.noise**2, color="0.4", linestyle=":", label="read noise$^2$")
    ax.axhline(result.ndark, color="0.7", linestyle="-.", label="dark")
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
    ax.set_title(
        f"Slit losses ({instr.swidth:g}\" x {instr.sheight:g}\" slit)"
    )
    ax.legend(fontsize="small")


def qa_figure(result, instr, obs, title=None):
    """Six-panel QA figure for a :class:`~obscalc.s2n.S2NResult`.

    Returns the matplotlib ``Figure``; the caller decides whether to save or
    show it.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=False)

    _panel_signal_to_noise(axes[0, 0], result)
    _panel_counts(axes[0, 1], result)
    _panel_throughput(axes[1, 0], result)
    _panel_extinction(axes[1, 1], result)
    _panel_sky_brightness(axes[2, 0], result)
    _panel_slit_loss(axes[2, 1], instr, obs)

    for ax in axes[:2, :].ravel():
        ax.set_xlabel("wavelength (Angstroms)")
    axes[2, 0].set_xlabel("wavelength (Angstroms)")

    if title is None:
        system = "Vega" if result.mtype == 1 else "AB"
        title = (
            f"{instr.name}  slit {instr.swidth:g}\"x{instr.sheight:g}\"  "
            f"{result.binr}x{result.binc} binning  "
            f"mag {result.mstar:g} {system}  "
            f"{obs.exptime:g} s  airmass {obs.airmass:g}  "
            f"seeing {obs.seeing:g}\""
        )
    fig.suptitle(title, fontsize="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def save_qa_figure(path, result, instr, obs, title=None, dpi=110):
    """Write :func:`qa_figure` to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    fig = qa_figure(result, instr, obs, title=title)
    fig.savefig(path, dpi=dpi)
    return path
