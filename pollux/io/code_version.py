"""Version of the code and environment of runs, recorded with their products."""
from __future__ import annotations

import datetime
import logging
import os
import shutil
import socket
import subprocess
import sys
import warnings
from functools import cache
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)


def _git(*args: str) -> str | None:
    """Return the output of a git command in the repository of this package, or None if it fails."""
    try:
        return subprocess.run(['git', '-c', 'safe.directory=*', *args],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True,
                              text=True,
                              check=True,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


@cache
def code_version() -> dict[str, str | bool | None]:
    """Return the git version of the running code.

    Returns
    -------
    dict
        ``commit`` (None outside a git repository), ``branch``, and ``dirty``: whether
        tracked files have uncommitted changes.
    """
    commit, branch = _git('rev-parse', 'HEAD'), _git('rev-parse', '--abbrev-ref', 'HEAD')
    status = _git('status', '--porcelain', '--untracked-files=no')
    return dict(commit=commit and commit.strip(), branch=branch and branch.strip(), dirty=bool(status))


def save_code_version(out_dir: str, files: Iterable[str] = (), require_clean: bool = False) -> str:
    """Record the code and environment of a run in a new ``{out_dir}/code_version/{date}`` folder.

    Each call has its own folder, so that resuming a run keeps the record of the earlier
    code. Call it from the main process, before launching tasks.

    Parameters
    ----------
    out_dir : str
        Output directory of the run.
    files : iterable of str, optional
        Files to copy in the record, e.g. the run script and its configuration.
    require_clean : bool, optional
        If True, refuse to run with uncommitted changes (or outside a git repository),
        so that outputs always come from a known commit. Default is False.

    Returns
    -------
    str
        The folder written, with:

        - ``code_version.yaml``: git commit, branch and state, date, host, command, Python;
        - ``uncommitted.diff``: the uncommitted changes of tracked files, if any;
        - ``pip_freeze.txt``: the installed Python packages;
        - ``conda_packages.txt``: the conda packages of the environment, if any;
        - the copied ``files``.
    """
    version = code_version()
    if require_clean and (version['dirty'] or version['commit'] is None):
        raise RuntimeError(f"pollux has uncommitted changes or no git version: {version}")

    now = datetime.datetime.now()
    folder = os.path.join(out_dir, 'code_version', now.strftime('%Y-%m-%dT%H-%M-%S'))
    os.makedirs(folder, exist_ok=True)

    record = dict(version, date=now.isoformat(timespec='seconds'), host=socket.gethostname(), command=sys.argv, python=sys.version, prefix=sys.prefix)
    with open(os.path.join(folder, 'code_version.yaml'), 'w') as f:
        yaml.safe_dump(record, f, sort_keys=False)

    if version['dirty']:
        with open(os.path.join(folder, 'uncommitted.diff'), 'w') as f:
            f.write(_git('diff', 'HEAD') or '')
        warnings.warn(f"Running pollux with uncommitted changes, saved in {folder}.")

    # Python packages, and conda ones (which include compiled libraries): the record of each is named name-version-build.json
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=300)
    with open(os.path.join(folder, 'pip_freeze.txt'), 'w') as f:
        f.write(freeze.stdout)
    conda_meta = os.path.join(sys.prefix, 'conda-meta')
    if os.path.isdir(conda_meta):
        with open(os.path.join(folder, 'conda_packages.txt'), 'w') as f:
            f.writelines(name[:-len('.json')] + '\n' for name in sorted(os.listdir(conda_meta)) if name.endswith('.json'))

    for fname in files:
        shutil.copy(fname, folder)
    logger.info("Saved code version to %s.", folder)
    return folder
