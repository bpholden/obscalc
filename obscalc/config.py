"""Location of the calibration, filter and template data files.

The IDL this package replaces built paths by string concatenation on unset
environment variables, e.g. ``getenv('FILTER_DIR') + filter``, which silently
produced a relative filename when ``FILTER_DIR`` was unset.  Here every lookup
either returns a file that exists or raises with the path it tried.

Filters and templates ship with the package.  To use a larger external
collection instead, point ``OBSCALC_FILTER_DIR`` / ``OBSCALC_TEMPLATE_DIR`` (or
the legacy ``FILTER_DIR`` / ``TEMPLATE_DIR`` used by the xidl scripts) at it.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

EXTINCTION_DIR = DATA_DIR / "extinction"
SKY_DIR = DATA_DIR / "sky"
THRUPUT_DIR = DATA_DIR / "thruput"


def _env_dir(*names):
    """First of ``names`` that is set in the environment, as a directory."""
    for name in names:
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(
                    f"${name} is set to {value!r}, which is not a directory"
                )
            return path
    return None


def filter_dir():
    return _env_dir("OBSCALC_FILTER_DIR", "FILTER_DIR") or DATA_DIR / "filters"


def template_dir():
    return _env_dir("OBSCALC_TEMPLATE_DIR", "TEMPLATE_DIR") or DATA_DIR / "templates"


def _with_gz_fallback(path):
    """Accept .fits for a .fits.gz on disk, and vice versa.

    IDL's ``xmrdfits`` did this, so the xidl sources name files without the
    ``.gz`` that is actually present.
    """
    if path.exists():
        return path
    alternate = (
        path.with_suffix("") if path.suffix == ".gz" else Path(str(path) + ".gz")
    )
    if alternate.exists():
        return alternate
    return None


def resolve(name, directory):
    """Resolve ``name`` to an existing file.

    ``name`` may be a bare filename to look up in ``directory``, or a path that
    is used as given.  Web callers pass bare names such as ``"Buser_V.dat"``.
    """
    candidate = Path(name).expanduser()
    if candidate.parent != Path("."):
        found = _with_gz_fallback(candidate)
        if found is None:
            raise FileNotFoundError(f"no such file: {candidate}")
        return found

    found = _with_gz_fallback(Path(directory) / candidate)
    if found is None:
        raise FileNotFoundError(f"{candidate.name!r} not found in {directory}")
    return found


def filter_file(name):
    return resolve(name, filter_dir())


def template_file(name):
    return resolve(name, template_dir())


def available_filters():
    return sorted(p.name for p in filter_dir().glob("*.dat"))


def available_templates():
    return sorted(p.name for p in template_dir().glob("*.fits*"))
