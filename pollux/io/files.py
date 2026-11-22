"""Layout of the output products (:class:`PolluxIO`): where every file goes, and reading them back.

A root directory holds the products of one simulation on one footprint (:class:`~pollux.footprint.Footprint`,
whose ``config.yaml`` is checked like those of the other products), for any number of seeds::

    footprint/                      config.yaml, padded_mask.fits, alm_weights.fits, NaMaster workspaces of the
                                    maps (wsp_b{binning}.wsp), shared by all seeds
    seed{seed}/                     products of one realization
        halos/                      halo catalogs, shared by all baryon models
        dmo/                        dark-matter-only products
            shells/                 shell_{iii}.fits, cls.fits, shells.fits (table of the shells), smoothed/{s}Mpc/
                                    (density_{iii}, t11_{iii}, t22_{iii}, t12_{iii})
            kappa/snapshots/        kappa_{iii}, gamma1_{iii}, gamma2_{iii} (one source plane per shell)
            kappa/{nz}/             kappa_bin{b}, gamma1_bin{b}, gamma2_bin{b}, nzs.txt, smoothed/{s}Mpc/
            catalogs/{nz}/{tag}/    cat_bin{b}.fits, cat_2pt[_{shapes}].fits, cat_cls[_{shapes}].fits (shapes:
                                    ia, shear_plus_ia, observed)
        dmb/{tag}/                  baryonified products, with the same subtree as dmo/
        intermediate/               files needed only to make or resume the products (sub-volume maps, a_lm,
                                    partial catalogs, workspaces, statistics of each bin pair), at the same
                                    relative paths
        code_version/{date}/        code and environment of each run

Each product directory (footprint, shells, halos, kappa, catalogs) holds a ``config.yaml`` with its parameters,
inputs and code version (:mod:`pollux.io.config`), so that it can be shared (e.g. registered) on its own, and that a
directory never mixes products of different parameters: the root directory of a run is up to the user, e.g. one per
simulation and footprint.
"""
from __future__ import annotations

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
    """Files of the products under one root directory: where builders and catalogs write them, and reading them back.

    Every file follows one rule: ``{product directory}[/{folder}]/{name}.{ext}``, with the name the parts given,
    joined by ``_`` (e.g. ``('kappa', 'bin3')`` for ``kappa_bin3.fits``); intermediate files go to the same
    relative path under ``intermediate/`` of their realization, and smoothed maps to ``smoothed/{s}Mpc/``.

    The object holds only the root directory and options: the directory of each product is a template of
    :attr:`PRODUCTS`, filled with the options that each object adds for its products (:meth:`with_options`).

    Files are written and read by format (``write_map`` and ``read_map``, ``_alm``, ``_table``, ``_sacc``,
    ``_npy``, ``_wsp``, ``_pickle``), writes being atomic (:func:`atomic_write`): readers and resumed runs never see a
    partial file.

    Parameters
    ----------
    root_dir : str
        Root directory of the products.
    **options
        Options of the templates of :attr:`PRODUCTS` (see :meth:`with_options`).
    """

    # directory of each product: (its realization, the product in it), templates filled with the options that the
    # objects add (with_options): the shell builder its seed, the baryon builder its baryon_tag (matter: dmo, or
    # dmb/{baryon_tag}), the lensing builder its n(z), the catalogs their tag
    PRODUCTS = {
        'footprint': ('', 'footprint'),
        'realization': ('seed{seed}', ''),
        'halos': ('seed{seed}', 'halos'),
        'shells': ('seed{seed}', '{matter}/shells'),
        'kappa': ('seed{seed}', '{matter}/kappa/{nz}'),
        'catalogs': ('seed{seed}', '{matter}/catalogs/{nz}/{tag}'),
    }
    # name of the maps of each product in their files (e.g. shell_012.fits)
    MAP_NAMES = {'halos': 'halo', 'shells': 'shell', 'kappa': 'kappa'}
    # indices of the maps in file names: shells (and per-shell maps), tomographic bins
    SHELL_INDEX = '{:03d}'
    BIN_INDEX = 'bin{}'

    def __init__(self, root_dir: str, **options: Any) -> None:
        self.root_dir = root_dir
        self.options = options

    def with_options(self, **options: Any) -> PolluxIO:
        """Return a copy with more options, those of an object's products (e.g. ``seed``, ``nz``).

        Parameters
        ----------
        **options
            Options of the templates of :attr:`PRODUCTS`.

        Returns
        -------
        PolluxIO
        """
        return PolluxIO(self.root_dir, **{**self.options, **options})

    def dir(self, product: str, *, intermediate: bool = False, smooth_Mpc: float | None = None, folder: str | None = None) -> str:
        """Return the directory of a product.

        Parameters
        ----------
        product : str
            Key of :attr:`PRODUCTS`.
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
        baryon_tag = self.options.get('baryon_tag')
        fields = {**self.options, 'matter': 'dmo' if baryon_tag is None else os.path.join('dmb', baryon_tag)}
        try:
            realization, relative = (template.format(**fields) for template in self.PRODUCTS[product])
        except KeyError as e:
            raise ValueError(f"The directory of {product!r} needs the option {e}: add it with with_options.") from None
        parts = (realization, INTERMEDIATE, relative) if intermediate else (realization, relative)
        directory = os.path.normpath(os.path.join(self.root_dir, *parts))
        if smooth_Mpc is not None:
            directory = os.path.join(directory, 'smoothed', f'{smooth_Mpc:g}Mpc')
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
    def catalog_stat_name(stat: str, shapes: str = 'shear') -> tuple[str, str | None]:
        """Return the parts of the name of the statistics of the catalogs (:meth:`path`), by the shapes of their shear legs.

        Parameters
        ----------
        stat : {'2pt', 'cls'}
            Statistics.
        shapes : str, optional
            Shapes of the shear legs (``'shear'``, ``'ia'``, ``'shear_plus_ia'``, ``'observed'``). Default is ``'shear'``.

        Returns
        -------
        tuple
            E.g. ``('cat_2pt', None)`` for the shear (``cat_2pt.fits``), ``('cat_2pt', 'ia')`` (``cat_2pt_ia.fits``).
        """
        return f'cat_{stat}', None if shapes == 'shear' else shapes

    # ==================================================================================================================
    # region Configs

    def read_config(self, product: str) -> dict:
        """Return the configuration of a product (its ``config.yaml``, see :func:`pollux.io.config.read_config`).

        Parameters
        ----------
        product : str
            Key of :attr:`PRODUCTS`.

        Returns
        -------
        dict
        """
        from .config import read_config  # here: pollux.io.config imports this module
        return read_config(self.dir(product))

    def write_config(self, product: str, config: dict) -> None:
        """Write the configuration of a product, or check it against the existing one (:func:`pollux.io.config.write_config`).

        Parameters
        ----------
        product : str
            Key of :attr:`PRODUCTS`.
        config : dict
            As given by the ``config()`` method of the objects.
        """
        from .config import write_config  # here: pollux.io.config imports this module
        write_config(self.dir(product), config, root_dir=self.root_dir)

    # endregion

    # ==================================================================================================================
    # region Cache

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

    # endregion

    # ==================================================================================================================
    # region Formats

    @staticmethod
    def read_map(fname: str, nside_coverage: int | None = None) -> hsp.HealSparseMap:
        """Read a HealSparse map.

        Parameters
        ----------
        fname : str
            File.
        nside_coverage : int or None, optional
            Coverage resolution (default: that of the file).

        Returns
        -------
        hsp.HealSparseMap
        """
        return hsp.HealSparseMap.read(fname, nside_coverage=nside_coverage)

    @staticmethod
    def write_map(fname: str, hspmap: hsp.HealSparseMap, **kwargs: Any) -> None:
        """Write a HealSparse map.

        Parameters
        ----------
        fname : str
            File.
        hspmap : hsp.HealSparseMap
        **kwargs
            Forwarded to :meth:`healsparse.HealSparseMap.write` (e.g. ``format``, ``nocompress``).
        """
        with atomic_write(fname) as tmp:
            hspmap.write(tmp, clobber=True, **kwargs)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_alm(fname: str) -> np.ndarray:
        """Read spherical harmonic coefficients (healpy ordering), from a ``.npz`` file or a FITS file of healpy.

        Parameters
        ----------
        fname : str
            File.

        Returns
        -------
        ndarray
        """
        if fname.endswith('.npz'):
            with np.load(fname) as f:
                return f['alm']
        return hp.read_alm(fname)

    @staticmethod
    def read_alm_mean(fname: str) -> float:
        """Read the mean removed from a field before its harmonic transform, saved with its a_lm (``.npz``).

        Parameters
        ----------
        fname : str
            File of :meth:`write_alm`.

        Returns
        -------
        float
        """
        with np.load(fname) as f:
            return float(f['mean'])

    @staticmethod
    def write_alm(fname: str, alm: np.ndarray, mean: float | None = None) -> None:
        """Write spherical harmonic coefficients (healpy ordering).

        Parameters
        ----------
        fname : str
            File: ``.npz`` (uncompressed, with ``alm`` and ``mean``: about 10 times faster to write and read than
            healpy's FITS files, and 20% smaller), or a FITS file of healpy otherwise.
        alm : ndarray
        mean : float or None, optional
            Mean removed from the field before its transform (:meth:`pollux.map_builder.MapBuilder._alm_field`),
            saved in ``.npz`` files (default 0).
        """
        with atomic_write(fname) as tmp:
            if fname.endswith('.npz'):
                with open(tmp, 'wb') as f:
                    np.savez(f, alm=alm, mean=np.float64(mean or 0.))
            else:
                hp.write_alm(tmp, alm, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_table(fname: str) -> fits.HDUList:
        """Open a FITS file of tables, e.g. a catalog.

        Parameters
        ----------
        fname : str
            File.

        Returns
        -------
        astropy.io.fits.HDUList
            To close, e.g. ``with io.read_table(fname) as hdul:``.
        """
        return fits.open(fname)

    @staticmethod
    def write_table(fname: str, hdus: list[fits.hdu.base.ExtensionHDU]) -> None:
        """Write FITS tables, after an empty primary HDU if they don't start with one.

        Parameters
        ----------
        fname : str
            File.
        hdus : list of HDU
        """
        if not isinstance(hdus[0], fits.PrimaryHDU):
            hdus = [fits.PrimaryHDU()] + list(hdus)
        with atomic_write(fname) as tmp:
            fits.HDUList(hdus).writeto(tmp, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_sacc(fname: str) -> sacc.Sacc:
        """Read a SACC file of statistics.

        Parameters
        ----------
        fname : str
            File.

        Returns
        -------
        sacc.Sacc
        """
        return sacc.Sacc.load_fits(fname)

    @staticmethod
    def write_sacc(fname: str, s: sacc.Sacc) -> None:
        """Write a SACC file of statistics.

        Parameters
        ----------
        fname : str
            File.
        s : sacc.Sacc
        """
        with atomic_write(fname) as tmp:
            s.save_fits(tmp, overwrite=True)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_npy(fname: str, allow_pickle: bool = False) -> np.ndarray:
        """Read a NumPy array.

        Parameters
        ----------
        fname : str
            File.
        allow_pickle : bool, optional
            For arrays of objects. Default is False.

        Returns
        -------
        ndarray
        """
        return np.load(fname, allow_pickle=allow_pickle)

    @staticmethod
    def write_npy(fname: str, array: Any) -> None:
        """Write a NumPy array.

        Parameters
        ----------
        fname : str
            File.
        array : array_like
        """
        with atomic_write(fname) as tmp:
            np.save(tmp, array)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_wsp(fname: str, read_unbinned_MCM: bool = True) -> Any:
        """Read a NaMaster workspace.

        Parameters
        ----------
        fname : str
            File.
        read_unbinned_MCM : bool, optional
            Also read the unbinned mode-coupling matrix, needed for bandpower windows. Default is True.

        Returns
        -------
        pymaster.NmtWorkspace
        """
        import pymaster as nmt  # here: only the statistics need NaMaster
        wsp = nmt.NmtWorkspace()
        wsp.read_from(fname, read_unbinned_MCM=read_unbinned_MCM)
        return wsp

    @staticmethod
    def write_wsp(fname: str, wsp: Any) -> None:
        """Write a NaMaster workspace.

        Parameters
        ----------
        fname : str
            File.
        wsp : pymaster.NmtWorkspace
        """
        with atomic_write(fname) as tmp:
            wsp.write_to(tmp)
        logger.debug("Saved to file %s.", fname)

    @staticmethod
    def read_pickle(fname: str) -> Any:
        """Read a pickled object (intermediate files only: pickles depend on the code that wrote them).

        Parameters
        ----------
        fname : str
            File.

        Returns
        -------
        object
        """
        with open(fname, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def write_pickle(fname: str, obj: Any) -> None:
        """Pickle an object (intermediate files only).

        Parameters
        ----------
        fname : str
            File.
        obj : object
        """
        with atomic_write(fname) as tmp, open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Saved to file %s.", fname)

    # endregion


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
