"""Layout of the output products: directories, file names and intermediate files.

All products of one realization (simulation, nside, seed and footprint padding) live under
``{root_dir}/nside{nside}/seed{seed}[/pad{padding}]``::

    halos/                          halo catalogs, shared by all baryon models
    dmo/                            dark-matter-only products
        shells/                     shell_{iii}.fits, cls.fits, shells.fits (table of the shells)
        smoothed/{s}Mpc/            density_{iii}, t11_{iii}, t22_{iii}, t12_{iii}
        kappa/snapshots/            kappa_{iii}, gamma1_{iii}, gamma2_{iii} (one source plane per shell)
        kappa/{nz}/                 kappa_bin{b}, gamma1_bin{b}, gamma2_bin{b}, nzs.txt
        catalogs/{nz}/{tag}/        cat_bin{b}.fits, cat_2pt[_{shapes}].fits, cat_cls[_{shapes}].fits (shapes: ia,
                                    shear_plus_ia, observed)
    dmb/{tag}/                      baryonified products, with the same subtree as dmo/
    intermediate/                   files needed only to make or resume the products (sub-volume maps,
                                    a_lm, partial catalogs, workspaces, statistics of each bin pair), at the
                                    same relative paths
    code_version/{date}/            code and environment of each run

Each product directory (shells, halos, kappa, catalogs) holds a ``config.yaml`` with its parameters,
inputs and code version (:mod:`pollux.io.config`), so that it can be shared (e.g. registered) on its own. Footprint products
(NaMaster workspaces, padded masks) are shared by all realizations at ``{root_dir}/nside{nside}``.
"""
from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

INTERMEDIATE = 'intermediate'


def realization_dir(root_dir: str, nside: int, seed: int, padding_deg: float | None = None) -> str:
    """Return the directory of the products of one realization.

    Parameters
    ----------
    root_dir : str
        Root directory of the simulation's products.
    nside : int
        Healpix resolution of the maps.
    seed : int
        Seed of the random isometries.
    padding_deg : float or None, optional
        Padding of the footprint (masked footprints only).

    Returns
    -------
    str
        ``{root_dir}/nside{nside}/seed{seed}[/pad{padding_deg}]``.
    """
    path = os.path.join(root_dir, f'nside{nside}', f'seed{seed}')
    return os.path.join(path, f'pad{padding_deg:g}') if padding_deg else path


def matter_dir(realization: str, baryon_tag: str | None = None) -> str:
    """Return the directory of the dark-matter-only (``dmo``) or baryonified (``dmb/{baryon_tag}``) products.

    Parameters
    ----------
    realization : str
        Directory of the realization (:func:`realization_dir`).
    baryon_tag : str or None, optional
        Tag of the baryon model, None for dark matter only.

    Returns
    -------
    str
    """
    return os.path.join(realization, 'dmo') if baryon_tag is None else os.path.join(realization, 'dmb', baryon_tag)


def shells_dir(matter: str) -> str:
    """Return the directory of the shells of a matter model (:func:`matter_dir`)."""
    return os.path.join(matter, 'shells')


def halos_dir(realization: str) -> str:
    """Return the directory of the halo catalogs of a realization, shared by all baryon models."""
    return os.path.join(realization, 'halos')


def smoothed_dir(smoothed_root: str, smooth_Mpc: float) -> str:
    """Return the directory of the maps smoothed at ``smooth_Mpc`` (``{smoothed_root}/{s}Mpc``).

    Parameters
    ----------
    smoothed_root : str
        Directory of the smoothed maps of a product (``{matter}/smoothed`` for shells,
        ``{kappa}/smoothed`` for convergence maps).
    smooth_Mpc : float
        Comoving Gaussian smoothing scale in Mpc.

    Returns
    -------
    str
    """
    return os.path.join(smoothed_root, f'{smooth_Mpc:g}Mpc')


def kappa_dir(matter: str, nz: str) -> str:
    """Return the directory of the convergence and shear maps of the redshift distributions ``nz`` (``'snapshots'``: one source plane per shell)."""
    return os.path.join(matter, 'kappa', nz)


def catalog_dir(matter: str, nz: str, tag: str | None = None) -> str:
    """Return the directory of the catalogs of tag ``tag`` (``default`` if None) for the redshift distributions ``nz``."""
    return os.path.join(matter, 'catalogs', nz, tag or 'default')


def footprint_dir(root_dir: str, nside: int) -> str:
    """Return the directory of the footprint products (NaMaster workspaces, padded masks), shared by all realizations."""
    return os.path.join(root_dir, f'nside{nside}')


# file names (relative to their directory)

SHELL_INDEX = '{:03d}'  # shells, and maps of one source plane per shell
BIN_INDEX = 'bin{}'  # tomographic bins


def map_fname(what: str, index: str | None = None, sub: int | str | None = None, extra: str | None = None, format: str = 'fits') -> str:
    """Return the name of a map file, ``{what}[_{index}][_sub{sub}][_{extra}].{format}``.

    Parameters
    ----------
    what : str
        Field or product (e.g. ``'shell'``, ``'kappa'``, ``'gamma1'``, ``'alms'``).
    index : str or None, optional
        Formatted index of the map (:data:`SHELL_INDEX` or :data:`BIN_INDEX`).
    sub : int or str or None, optional
        Sub-volume of the map (intermediate files).
    extra : str or None, optional
        Additional string at the end of the name.
    format : str, optional
        File extension (without leading dot).

    Returns
    -------
    str
    """
    parts = [what] + [p for p in (index, None if sub is None else f'sub{sub}', extra) if p is not None]
    return '_'.join(parts) + f'.{format}'


def alms_what(replicate_octant: bool) -> str:
    """Return the name of the a_lm files, of the octant replicated over the full sky (``alms_octrep``) or not (``alms``)."""
    return 'alms_octrep' if replicate_octant else 'alms'


CLS_FNAME = 'cls.fits'  # power spectra of the maps of a product
SHELLS_TABLE_FNAME = 'shells.fits'  # table of the shells


def catalog_fname(ibin: int, ishell: int | None = None, format: str = 'fits') -> str:
    """Return the name of the catalog of a tomographic bin, ``cat_bin{b}.{format}``, or of its part from one shell (``cat_bin{b}_{iii}``)."""
    return f'cat_bin{ibin}.{format}' if ishell is None else f'cat_bin{ibin}_{ishell:03d}.{format}'


def catalog_stat_fname(stat: str, shapes: str = 'shear') -> str:
    """Return the name of a statistic of the catalogs, ``cat_{stat}.fits`` for the lensing shear, else ``cat_{stat}_{shapes}.fits``.

    Parameters
    ----------
    stat : {'cls', '2pt'}
        Statistic.
    shapes : {'shear', 'ia', 'shear_plus_ia', 'observed'}, optional
        Shapes of the shear legs.

    Returns
    -------
    str
    """
    return f'cat_{stat}.fits' if shapes == 'shear' else f'cat_{stat}_{shapes}.fits'


def intermediate_dir(product_dir: str, realization: str | None) -> str:
    """Return the directory of the intermediate files of a product, at its path relative to the realization.

    Parameters
    ----------
    product_dir : str
        Directory of the product.
    realization : str or None
        Directory of the realization. If None (product outside the layout), the intermediate
        files go to ``{product_dir}/intermediate``.

    Returns
    -------
    str
    """
    if realization is None:
        return os.path.join(product_dir, INTERMEDIATE)
    return os.path.join(realization, INTERMEDIATE, os.path.relpath(product_dir, realization))


@contextmanager
def atomic_write(fname: str) -> Iterator[str]:
    """Yield a temporary path to write ``fname`` to, renamed to ``fname`` once written.

    Readers (and resumed runs, which skip existing files) never see a partial file, and
    several workers may write the same file at once. The temporary file keeps the extension
    of ``fname`` (for writers that use it) and is removed if the writing fails.

    Parameters
    ----------
    fname : str
        Path of the file.

    Yields
    ------
    str
        Temporary path, in the same directory.
    """
    root, ext = os.path.splitext(fname)
    # unique among processes of all nodes (PIDs may coincide between nodes)
    tmp = f'{root}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{ext}'
    try:
        yield tmp
        os.replace(tmp, fname)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
