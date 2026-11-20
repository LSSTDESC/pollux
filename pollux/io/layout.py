"""Layout of the output products (:class:`PolluxIO`): where every file goes, and reading them back.

All products of one realization (simulation, nside, seed and footprint padding) live under
``{root_dir}/nside{nside}/seed{seed}[/pad{padding}]``::

    halos/                          halo catalogs, shared by all baryon models
    dmo/                            dark-matter-only products
        shells/                     shell_{iii}.fits, cls.fits, shells.fits (table of the shells)
        smoothed/{s}Mpc/            density_{iii}, t11_{iii}, t22_{iii}, t12_{iii}
        kappa/snapshots/            kappa_{iii}, gamma1_{iii}, gamma2_{iii} (one source plane per shell)
        kappa/{nz}/                 kappa_bin{b}, gamma1_bin{b}, gamma2_bin{b}, nzs.txt, smoothed/{s}Mpc/
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

import copy
import logging
import os
import pickle
import socket
import uuid
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import healpy as hp
import healsparse as hsp
import numpy as np
import sacc
from astropy.io import fits

logger = logging.getLogger(__name__)

INTERMEDIATE = 'intermediate'


class PolluxIO:
    """Files of the products of one realization: where builders and catalogs write them, and reading them back.

    Every file follows one rule: ``{product directory}[/{folder}]/{name}.{ext}``, with the name the parts given,
    joined by ``_`` (e.g. ``('kappa', 'bin3')`` for ``kappa_bin3.fits``); intermediate files go to the same
    relative path under ``intermediate/``, and smoothed maps to ``smoothed/{s}Mpc/``. The directory of each
    product is a template of :attr:`PRODUCTS`, filled with the options of the object (matter model, n(z),
    catalog tag): builders and catalogs get an object with their own options (:meth:`with_options`).

    Files are written and read by format (``write_map`` and ``read_map``, ``_alm``, ``_table``, ``_sacc``,
    ``_npy``, ``_wsp``, ``_pickle``), writes being atomic (:func:`atomic_write`): readers and resumed runs never see a
    partial file.

    Parameters
    ----------
    realization_dir : str
        Directory of the realization.
    footprint_dir : str or None, optional
        Directory of the products shared by the realizations of a footprint (NaMaster workspaces, padded
        masks). Default is ``realization_dir``.
    baryon_tag : str or None, optional
        Tag of the baryon model (``dmb/{baryon_tag}``), None for dark matter only (``dmo``).
    nz : str or None, optional
        Name of the redshift distributions of the convergence maps and catalogs (``'snapshots'``: one
        source plane per shell).
    tag : str or None, optional
        Tag of the catalogs (``default`` if None).
    """

    # directory of each product in the realization, from the options (footprint products are apart)
    PRODUCTS = {
        'halos': 'halos',
        'shells': '{matter}/shells',
        'smoothed': '{matter}/smoothed',
        'kappa': '{matter}/kappa/{nz}',
        'catalogs': '{matter}/catalogs/{nz}/{tag}',
    }
    # name of the maps of each product in their files (e.g. shell_012.fits)
    MAP_NAMES = {'halos': 'halo', 'shells': 'shell', 'kappa': 'kappa'}
    # products whose smoothed maps are in another product (the smoothed shells, shared by their fields)
    SMOOTHED_IN = {'shells': 'smoothed'}
    # indices of the maps in file names: shells (and per-shell maps), tomographic bins
    SHELL_INDEX = '{:03d}'
    BIN_INDEX = 'bin{}'

    def __init__(
            self,
            realization_dir: str,
            *,
            footprint_dir: str | None = None,
            baryon_tag: str | None = None,
            nz: str | None = None,
            tag: str | None = None) -> None:
        self.realization_dir = realization_dir
        self.footprint_dir = footprint_dir or realization_dir
        self.baryon_tag = baryon_tag
        self.nz = nz
        self.tag = tag

    @classmethod
    def from_root(cls, root_dir: str, nside: int, seed: int, padding_deg: float | None = None, **options: Any) -> PolluxIO:
        """Return the files of the realization ``{root_dir}/nside{nside}/seed{seed}[/pad{padding_deg}]``.

        Footprint products are shared by the realizations, at ``{root_dir}/nside{nside}``. ``options`` are
        those of :class:`PolluxIO`.
        """
        footprint = os.path.join(root_dir, f'nside{nside}')
        realization = os.path.join(footprint, f'seed{seed}')
        if padding_deg:
            realization = os.path.join(realization, f'pad{padding_deg:g}')
        return cls(realization, footprint_dir=footprint, **options)

    def with_options(self, **options: Any) -> PolluxIO:
        """Return a copy with other options (``baryon_tag``, ``nz``, ``tag``), e.g. for the products of a builder."""
        new = copy.copy(self)
        for k, v in options.items():
            if k not in ('baryon_tag', 'nz', 'tag'):
                raise TypeError(f"unknown option {k!r}")
            setattr(new, k, v)
        return new

    def dir(self, product: str, *, intermediate: bool = False, smooth_Mpc: float | None = None, folder: str | None = None) -> str:
        """Return the directory of a product (``'footprint'`` for the products shared by the realizations).

        Parameters
        ----------
        product : str
            Key of :attr:`PRODUCTS`, or ``'footprint'``.
        intermediate : bool, optional
            The directory of its intermediate files instead.
        smooth_Mpc : float or None, optional
            The directory of its maps smoothed at this scale (Mpc).
        folder : str or None, optional
            Subdirectory (e.g. ``'plots'``, ``'subs'``).

        Returns
        -------
        str
        """
        if product == 'footprint':
            directory = self.footprint_dir
        else:
            if smooth_Mpc is not None and product in self.SMOOTHED_IN:
                product, smooth_folder = self.SMOOTHED_IN[product], f'{smooth_Mpc:g}Mpc'
            else:
                smooth_folder = None if smooth_Mpc is None else os.path.join('smoothed', f'{smooth_Mpc:g}Mpc')
            matter = 'dmo' if self.baryon_tag is None else os.path.join('dmb', self.baryon_tag)
            relative = self.PRODUCTS[product].format(matter=matter, nz=self.nz, tag=self.tag or 'default')
            directory = os.path.join(self.realization_dir, INTERMEDIATE, relative) if intermediate else os.path.join(self.realization_dir, relative)
            if smooth_folder is not None:
                directory = os.path.join(directory, smooth_folder)
        return directory if folder is None else os.path.join(directory, folder)

    def path(self, product: str, *name: Any, ext: str = 'fits', **dir_kwargs: Any) -> str:
        """Return the path of a file of a product, ``{dir}/{name}.{ext}``.

        Parameters
        ----------
        product : str
            Product (see :meth:`dir`).
        *name
            Parts of the name, joined by ``_``; None parts are left out (e.g. ``('cat_2pt', None)`` for
            ``cat_2pt.fits``).
        ext : str, optional
            Extension, without the leading dot.
        **dir_kwargs
            ``intermediate``, ``smooth_Mpc`` and ``folder`` of :meth:`dir`.

        Returns
        -------
        str
        """
        return os.path.join(self.dir(product, **dir_kwargs), '_'.join(str(p) for p in name if p is not None) + f'.{ext}')

    @staticmethod
    def cached(fname: str, compute: Callable[[], Any], read: Callable[[str], Any], write: Callable[[str, Any], None]) -> Any:
        """Return an object saved at ``fname``, or compute it and save it there.

        For objects that computations need and that are expensive to make again (e.g. NaMaster workspaces):
        the first process computes and saves them, the others (and later calls) read them back. The products
        themselves are made by the pipeline tasks, which skip those already saved.

        Parameters
        ----------
        fname : str
            File of the object.
        compute : callable
            Returns the object.
        read : callable
            Reads the object from ``fname``.
        write : callable
            Writes the object to ``fname`` (atomically, e.g. a ``write_*`` method).

        Returns
        -------
        object
            The object read from ``fname`` if it is there and readable (a warning if not readable), otherwise
            computed and written.
        """
        if os.path.exists(fname):
            try:
                obj = read(fname)
                logger.debug("Found and loaded %s.", fname)
                return obj
            except Exception as e:
                warnings.warn(f"Cannot read {fname} ({e}): computing it again.")
        obj = compute()
        write(fname, obj)
        return obj

    # files, by format: written atomically (see :func:`atomic_write`), read back from their path (see :meth:`path`)

    @staticmethod
    def read_map(fname: str, nside_coverage: int | None = None) -> hsp.HealSparseMap:
        """Read a HealSparse map."""
        return hsp.HealSparseMap.read(fname, nside_coverage=nside_coverage)

    @staticmethod
    def write_map(fname: str, hspmap: hsp.HealSparseMap, **kwargs: Any) -> None:
        """Write a HealSparse map; ``kwargs`` go to :meth:`healsparse.HealSparseMap.write` (e.g. ``format``, ``nocompress``)."""
        with atomic_write(fname) as tmp:
            hspmap.write(tmp, clobber=True, **kwargs)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_alm(fname: str) -> np.ndarray:
        """Read spherical harmonic coefficients."""
        return hp.read_alm(fname)

    @staticmethod
    def write_alm(fname: str, alm: np.ndarray) -> None:
        """Write spherical harmonic coefficients."""
        with atomic_write(fname) as tmp:
            hp.write_alm(tmp, alm, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_table(fname: str) -> fits.HDUList:
        """Open a FITS file of tables, e.g. a catalog (to close, e.g. ``with io.read_table(fname) as hdul:``)."""
        return fits.open(fname)

    @staticmethod
    def write_table(fname: str, hdus: list[fits.hdu.base.ExtensionHDU]) -> None:
        """Write FITS tables, after an empty primary HDU if they don't start with one."""
        if not isinstance(hdus[0], fits.PrimaryHDU):
            hdus = [fits.PrimaryHDU()] + list(hdus)
        with atomic_write(fname) as tmp:
            fits.HDUList(hdus).writeto(tmp, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_sacc(fname: str) -> sacc.Sacc:
        """Read a SACC file of statistics."""
        return sacc.Sacc.load_fits(fname)

    @staticmethod
    def write_sacc(fname: str, s: sacc.Sacc) -> None:
        """Write a SACC file of statistics."""
        with atomic_write(fname) as tmp:
            s.save_fits(tmp, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_npy(fname: str, allow_pickle: bool = False) -> np.ndarray:
        """Read a NumPy array (``allow_pickle`` for arrays of objects)."""
        return np.load(fname, allow_pickle=allow_pickle)

    @staticmethod
    def write_npy(fname: str, array: Any) -> None:
        """Write a NumPy array."""
        with atomic_write(fname) as tmp:
            np.save(tmp, array)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_wsp(fname: str, read_unbinned_MCM: bool = True) -> Any:
        """Read a NaMaster workspace (``read_unbinned_MCM``: also the unbinned mode-coupling matrix, needed for bandpower windows)."""
        import pymaster as nmt  # here: only the statistics need NaMaster
        wsp = nmt.NmtWorkspace()
        wsp.read_from(fname, read_unbinned_MCM=read_unbinned_MCM)
        return wsp

    @staticmethod
    def write_wsp(fname: str, wsp: Any) -> None:
        """Write a NaMaster workspace."""
        with atomic_write(fname) as tmp:
            wsp.write_to(tmp)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_pickle(fname: str) -> Any:
        """Read a pickled object (intermediate files only: pickles depend on the code that wrote them)."""
        with open(fname, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def write_pickle(fname: str, obj: Any) -> None:
        """Pickle an object (intermediate files only)."""
        with atomic_write(fname) as tmp, open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Saved to file %s.", fname)

    def read_config(self, product: str) -> dict:
        """Return the configuration of a product (its ``config.yaml``, see :func:`pollux.io.config.read_config`)."""
        from .config import read_config  # here: pollux.io.config imports this module
        return read_config(self.dir(product))

    def write_config(self, product: str, config: dict) -> None:
        """Write the configuration of a product, or check it against the existing one (:func:`pollux.io.config.write_config`)."""
        from .config import write_config  # here: pollux.io.config imports this module
        write_config(self.dir(product), config, realization=self.realization_dir)


@contextmanager
def atomic_write(fname: str) -> Iterator[str]:
    """Yield a temporary path to write ``fname`` to, renamed to ``fname`` once written.

    Readers (and resumed runs, which skip existing files) never see a partial file, and
    several workers may write the same file at once. The temporary file keeps the extension
    of ``fname`` (for writers that use it) and is removed if the writing fails. The directory
    of ``fname`` is created if needed.

    Parameters
    ----------
    fname : str
        Path of the file.

    Yields
    ------
    str
        Temporary path, in the same directory.
    """
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    root, ext = os.path.splitext(fname)
    # unique among processes of all nodes (PIDs may coincide between nodes)
    tmp = f'{root}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{ext}'
    try:
        yield tmp
        os.replace(tmp, fname)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
