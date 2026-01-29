import os
from collections.abc import Iterable

import healpy as hp
import healsparse
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
import sacc
from itertools import product
from tqdm.auto import tqdm, trange
from itertools import product

from .utils import *


class MapBuilder:
    """Base class to build, store, and analyze Healpix/HealSparse maps.

    This class centralizes:
    - file naming and I/O for HealSparse maps,
    - mask/footprint creation,
    - alm computation (Healpy),
    - pseudo-C_ell and decoupled bandpowers (NaMaster),
    - SACC export and plotting.

    Parameters
    ----------
    nside : int
        Healpix resolution parameter of the maps.
    dtype : data-type
        Data type used to store map values (dtype for arrays written to
        HealSparse/Healpix maps and intermediate map buffers).
    save_dir : str
        Directory where outputs are written.
    base_fname : str
        Base filename prefix used by :meth:`get_fname`.
    nmaps : int
        Number of maps handled/produced by this builder.
    nside_c : int, optional
        HealSparse coverage nside. Default is 32.
    octant_only : bool, optional
        If True, restrict the footprint to one octant. Default is True.
    save_format : str, optional
        HealSparse file format (e.g. 'fits'). Default is 'fits'.
    wsp_dir : str or None, optional
        Directory used to cache NaMaster workspaces. Defaults to ``save_dir``.
    sacc_type : str or None, optional
        SACC data type name used when saving spectra.
    b : pymaster.NmtBin, optional
        NaMaster binning scheme. If None, it is constructed lazily.
    nlb : int, optional
        Number of bandpowers when constructing bins lazily. Default is 20.
    remove_monopole : bool, optional
        If True, subtract the mean within the mask before harmonic transforms.
    nocompress : bool, optional
        If True, disable HealSparse compression. Default is True.
    f_ell : {'pixwin'} or ndarray or None, optional
        Multiplicative factor applied to theory spectra before binning (e.g.
        pixel window). If 'pixwin', uses the Healpix pixel window. Default is
        'pixwin'.
    
    Notes
    -----
    Healpix ordering: this codebase uses NESTED pixel ordering by default for
    internal Healpix/HealSparse operations and IO unless otherwise stated.
    """

    def __init__(
            self,
            nside,
            dtype,
            save_dir,
            base_fname,
            nmaps,
            nside_c=32,
            octant_only=True,
            save_format='fits',
            wsp_dir=None,
            sacc_type=None,
            b=None,
            nlb=20,
            remove_monopole=True,
            nocompress=True,
            f_ell='pixwin'):
        """Initialize a MapBuilder.

        Parameters
        ----------
        nside : int
            Healpix resolution parameter of the maps.
        dtype : data-type
            Data type used for stored map values.
        save_dir : str or None
            Output directory. If None, file I/O is disabled.
        base_fname : str
            Filename prefix used by :meth:`get_fname`.
        nmaps : int
            Number of maps handled/produced by this builder.
        nside_c : int, optional
            HealSparse coverage nside (controls sparse tiling). Default is 32.
        octant_only : bool, optional
            If True, the sky footprint is restricted to one octant.
        save_format : str, optional
            File format for HealSparse I/O, e.g. ``'fits'``.
        wsp_dir : str or None, optional
            Directory for NaMaster workspace caching. If None, uses ``save_dir``.
        sacc_type : str or None, optional
            SACC data type string used when writing spectra.
        b : pymaster.NmtBin, optional
            Pre-built NaMaster binning scheme.
        nlb : int, optional
            Number of bandpowers used when constructing bins lazily.
        remove_monopole : bool, optional
            If True, subtract the mean on the mask before computing transforms.
        nocompress : bool, optional
            If True, disable HealSparse compression when writing maps.
        f_ell : {'pixwin'} or ndarray or None, optional
            Multiplicative response applied to theory spectra. If ``'pixwin'``,
            uses Healpix pixel window. If ndarray, it is interpreted as a
            per-ell factor starting at ``b.get_ell_list(0)[0]`` for the current
            binning scheme.
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.dtype = dtype
        self.nmaps = nmaps
        self.base_fname = base_fname
        self.sacc_type = sacc_type
        self.save_dir = save_dir
        self.nside_c = nside_c
        self.octant_only = octant_only
        self.save_format = save_format
        if wsp_dir is None:
            wsp_dir = save_dir
        self.wsp_dir = wsp_dir
        self.remove_monopole = remove_monopole
        self.nocompress = nocompress
        self.f_ell = f_ell

        if b is not None:
            self.b = b
        else:
            self._nlb = nlb

        # make sure folder exists
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"*** Map builder ({self.base_fname}) ***")
        print(f"  - Number of maps = {self.nmaps}")
        print(f"  - Maps saved to", self.save_dir)

    def tqdm(self, it=None, **tqdm_kwargs):
        """Wrap an iterable (or default range) with tqdm progress bars.

        Parameters
        ----------
        it : iterable or int or None, optional
            - If None: iterate over ``range(self.nmaps)`` using :func:`trange`.
            - If iterable: wrap with :func:`tqdm`.
            - If int: return a singleton list ``[it]``.

        Returns
        -------
        iterable
            Iterable with tqdm progress bar behavior.

        Raises
        ------
        ValueError
            If ``it`` is neither None, an int, nor an iterable.
        """
        if it is None:
            return trange(self.nmaps, **tqdm_kwargs)
        elif isinstance(it, Iterable):
            return tqdm(it, **tqdm_kwargs)
        elif type(it) is int:
            return [it]
        else:
            raise ValueError

    def get_fname(self, imap=None, sub=None, what=None, extra=None, format=None):
        """Build an output filename for a given data product.

        Parameters
        ----------
        imap : int or None, optional
            Map index to include in the filename.
        sub : int or str or None, optional
            Sub-volume identifier. If provided, the file is placed under
            ``{save_dir}/subs/``.
        what : str or None, optional
            Product tag (e.g. ``'alms'``, ``'cls'``, ``'gamma1'``).
        extra : str or None, optional
            Extra string appended to the filename.
        format : str or None, optional
            File extension (without leading dot). If None, uses ``self.save_format``.

        Returns
        -------
        str
            Full path to the requested product file.

        Notes
        -----
        The base filename is of the form::

            {base_fname}_nside{nside}[_{what}][_imap{imap}][_sub{sub}][_{extra}].{format}
        """
        fname = self.base_fname + f'_nside{self.nside}'

        if what is not None:
            fname += f'_{what}'

        if imap is not None:
            fname += f'_imap{imap}'

        if sub is not None:
            fname += f'_sub{sub}'

        if extra is not None:
            fname += f'_{extra}'

        if format is not None:
            fname += f'.{format}'
        else:
            fname += f'.{self.save_format}'

        if sub is None:
            return os.path.join(self.save_dir, fname)
        else:
            subs_save_dir = os.path.join(self.save_dir, 'subs')
            os.makedirs(subs_save_dir, exist_ok=True)
            return os.path.join(subs_save_dir, fname)

    def _create_tracers_ccl(self):
        """Create CCL tracer objects.

        Returns
        -------
        list
            List of tracer objects with length ``self.nmaps``.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_tracers_ccl(self):
        """Return (and cache) CCL tracer objects used for theory predictions.

        Returns
        -------
        list
            List of tracer objects with length ``self.nmaps``.
        """
        if not hasattr(self, 'tracers'):
            self.tracers = self._create_tracers_ccl()
        return self.tracers

    def _create_tracers_sacc(self):
        """Create SACC tracer objects.

        Returns
        -------
        list
            List of SACC tracer objects with length ``self.nmaps``.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_tracers_sacc(self):
        """Return (and cache) SACC tracers used to store measured spectra.

        Returns
        -------
        list
            List of SACC tracer objects with length ``self.nmaps``.
        """
        if not hasattr(self, 'tracers_sacc'):
            self.tracers_sacc = self._create_tracers_sacc()
        return self.tracers_sacc

    def construct_maps(self, imaps=None, **kwargs):
        """Construct maps and return them as HealSparse objects.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to construct. If None, constructs all maps.
        **kwargs
            Forwarded to subclass implementation.

        Returns
        -------
        list
            List of :class:`healsparse.HealSparseMap`.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def check_maps(self, imaps=None, read=False, fname=None, check_finite=False):
        """Check existence/readability of map files.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to check (ignored if ``fname`` is provided).
        read : bool, optional
            If True, also try reading the file with HealSparse.
        fname : str or None, optional
            If provided, check this single filename instead of using indices.
        check_finite : bool, optional
            If True, validates that values on valid pixels are finite and not
            identically zero.

        Returns
        -------
        bool or list of bool
            Check result(s). If a single map is checked, returns a single bool.

        Notes
        -----
        If ``read=False`` this only checks file existence.
        """
        assert (imaps is None) or (fname is None)

        def check_one(filename):
            if os.path.exists(filename):
                if read:
                    try:
                        m = healsparse.HealSparseMap.read(filename, nside_coverage=self.nside_c)
                        if check_finite:
                            a = m[m.valid_pixels]
                            return np.all(np.isfinite(a)) and np.any(a)
                        else:
                            return True
                    except:
                        print(f"Cannot read map at {fname}.")
                        return False
                else:
                    return True
            else:
                print(f"Cannot find map at {fname}.")
                return False

        if fname is not None:
            return check_one(fname)
        else:
            out = []
            for imap in self.tqdm(imaps, desc="[check_maps]"):
                fname = self.get_fname(imap)
                out.append(check_one(fname))

        if len(out) == 1:
            return out[0]
        else:
            return out

    def load_maps(self, imaps=None, **fname_kwargs):
        """Load one or multiple HealSparse maps from disk.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to load. If an int (or a single-element iterable), a
            single map is returned.
        **fname_kwargs
            Extra keywords forwarded to :meth:`get_fname` (e.g. ``what='kappa'``).

        Returns
        -------
        healsparse.HealSparseMap or list of healsparse.HealSparseMap
            Loaded map(s).
        """
        maps = []
        for imap in self.tqdm(imaps, desc="[load_maps]"):
            fname = self.get_fname(imap, **fname_kwargs)
            maps.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.nside_c))

        if len(maps) == 1:
            return maps[0]
        else:
            return maps

    def plot_maps(self, maps_or_imaps=None, nx=None, log=False, title_prefix=None, cbar=True, save=True, **hpview_kwargs):
        """Plot one or multiple maps.

        Parameters
        ----------
        maps_or_imaps : int or iterable or None, optional
            - If None: loads and plots all maps.
            - If int: loads and plots that map index.
            - If iterable of int: loads and plots those indices.
            - If iterable of maps: plots provided maps directly.
        nx : int or None, optional
            Number of columns in the subplot grid. Defaults to ``nmaps``.
        log : bool, optional
            If True and the map has positive values, plots ``log10(map)``.
        title_prefix : str or None, optional
            Prefix used for subplot titles.
        cbar : bool, optional
            If True, display a colorbar.
        save : bool, optional
            If True, save the figure to ``get_fname(format='png')``.
        **hpview_kwargs
            Forwarded to :func:`healpy.orthview` (octant mode) or
            :func:`healpy.mollview` (full sky).

        Returns
        -------
        None
        """
        # assert (imaps is None) or (maps is None)

        if maps_or_imaps is None:
            nmaps = self.nmaps
            imaps = range(nmaps)
            maps = None
        elif type(maps_or_imaps) is int:
            nmaps = 1
            imaps = [maps_or_imaps]
            maps = None
        elif isinstance(maps_or_imaps, Iterable):
            nmaps = len(maps_or_imaps)
            if type(maps_or_imaps[0]) is int:
                imaps = maps_or_imaps
                maps = None
            else:
                imaps = range(nmaps)
                maps = maps_or_imaps
        else:
            raise ValueError

        if nx is None:
            nx = nmaps
        ny = int(np.ceil(nmaps / nx))

        plt.figure(figsize=(4 * nx, 4 * ny))

        sub = 1
        for imap in self.tqdm(imaps):
            if maps is None:
                hp_map = self.load_maps(imap)
            else:
                hp_map = maps[imap]
            if isinstance(hp_map, healsparse.HealSparseMap):
                hp_map = hp_map.generate_healpix_map()
            if log and np.any(hp_map > 0.):
                hp_map = np.log10(hp_map)
            title = title_prefix + f" {imap}" if title_prefix is not None else None
            if self.octant_only:
                hp.orthview(
                    hp_map,
                    rot=(35, 20),
                    half_sky=True,
                    sub=(ny, nx, sub),
                    title=title,
                    cbar=cbar,
                    min=0. if log else None,
                    margins=(0., 0., 0., 0.),
                    nest=True,
                    **hpview_kwargs)
            else:
                hp.mollview(hp_map, sub=(ny, nx, sub), title=title, cbar=cbar, min=0. if log else None, margins=(0., 0., 0., 0.), nest=True, **hpview_kwargs)
            sub += 1

        hp.graticule()

        if save:
            fname = self.get_fname(format='png')
            plt.savefig(fname, bbox_inches='tight', dpi=150, transparent=False)

    def _create_mask(self, nside=None):
        """Create a footprint mask and list of valid pixels.

        Parameters
        ----------
        nside : int or None, optional
            If provided, constructs the mask at this nside. Otherwise uses
            ``self.nside``.

        Returns
        -------
        mask : ndarray
            Dense mask array (float or bool-like, values 0/1).
        ipix : ndarray of int
            Pixel indices where mask is non-zero, in NESTED ordering.
        """
        if nside is not None:
            npix = hp.nside2npix(nside)
        else:
            nside = self.nside
            npix = self.npix

        if self.octant_only:
            mask = make_mask_radec_cuts(nside, 0, 90, 0, 90, eps=1e-10, nest=True)  # mask fully within the octant
            ipix = np.where(mask)[0]
        else:
            mask = np.ones(npix)
            ipix = np.arange(npix)

        return mask, ipix

    def get_mask(self, nest=True, return_ipix=False, nside=None):
        """Return (and cache) the analysis mask.

        Parameters
        ----------
        nest : bool, optional
            If True, return mask in NESTED ordering. If False, return RING.
        return_ipix : bool, optional
            If True, also return the valid pixel indices.
        nside : int or None, optional
            Currently unused in cached mode; kept for API compatibility.

        Returns
        -------
        mask : ndarray
            Dense mask array at ``self.nside``.
        ipix : ndarray of int, optional
            Valid pixel indices (only if ``return_ipix=True``).

        Notes
        -----
        The mask is cached on first use as:
        ``self.mask`` / ``self.ipix`` (NESTED) and
        ``self.mask_ring`` / ``self.ipix_ring`` (RING).
        """
        if not hasattr(self, 'mask'):
            mask, ipix = self._create_mask()
            self.mask = mask
            self.ipix = ipix
            self.mask_ring = hp.reorder(self.mask, n2r=True)
            self.ipix_ring = hp.nest2ring(self.nside, self.ipix)
        if nest:
            if return_ipix:
                return self.mask, self.ipix
            else:
                return self.mask
        else:
            if return_ipix:
                return self.mask_ring, self.ipix_ring
            else:
                return self.mask_ring

    def get_octant_to_fullsky_ipix(self, nest=True):
        """Return a mapping from octant pixels to their full-sky replicas.

        Parameters
        ----------
        nest : bool, optional
            Pixel ordering used for the mapping.

        Returns
        -------
        ndarray of int
            Array of length ``hp.nside2npix(self.nside)`` giving, for each pixel
            index on the full sky, the corresponding pixel index in the octant
            (by reflecting theta and wrapping phi).

        Raises
        ------
        AssertionError
            If ``self.octant_only`` is False.
        """
        assert self.octant_only
        if not hasattr(self, 'ipix_oct_fullsky'):
            theta, phi = hp.pix2ang(self.nside, np.arange(self.npix), nest=nest)
            self.ipix_oct_fullsky = hp.ang2pix(self.nside, np.where(theta <= np.pi / 2., theta, np.pi - theta), phi % np.pi / 2., nest=nest)

        return self.ipix_oct_fullsky

    def get_cls_b(self):
        """Return (and lazily create) the NaMaster binning scheme.

        Returns
        -------
        pymaster.NmtBin
            Bandpower binning used for :meth:`compute_cls` and :meth:`predict_cls`.

        Notes
        -----
        This is implemented as a lazy getter to keep the builder pickle/serialize
        friendly.
        """
        if not hasattr(self, 'b'):
            self.b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self._nlb, f_ell=self.f_ell)
        return self.b

    def get_cls_wsp(self):
        """Return (and lazily create) the NaMaster workspace.

        The workspace stores the mode-coupling matrix and bandpower windows for
        the current mask and binning, and is cached on disk under ``self.wsp_dir``.

        Returns
        -------
        pymaster.NmtWorkspace
            Workspace containing the coupling matrix and bandpower windows.
        """

        if hasattr(self, 'wsp'):
            return self.wsp
        else:
            fname = os.path.join(self.wsp_dir, f'wsp_nside{self.nside}.wsp')
            self.wsp = nmt.NmtWorkspace()
            if os.path.exists(fname):
                print("Loading workspace in", fname)
                self.wsp.read_from(fname)
            else:
                print("Computing and saving workspace in", fname)
                mask = self.get_mask(nest=False)
                b = self.get_cls_b()
                dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)], lite=True)
                self.wsp.compute_coupling_matrix(dummy_f, dummy_f, b)
                self.wsp.write_to(fname)
            return self.wsp

    def _field_map(self, imap, nest=True):
        """Load a map and return a dense Healpix array.

        Parameters
        ----------
        imap : int
            Map index to load.
        nest : bool, optional
            Output ordering.

        Returns
        -------
        ndarray
            Dense Healpix map with UNSEEN outside the HealSparse valid region.
        """
        return self.load_maps(imap).generate_healpix_map(nest=nest)

    def get_field_map(self, imap, nest=True, replicate_octant=False):
        """Return a dense Healpix map suitable for harmonic transforms.

        Parameters
        ----------
        imap : int
            Map index.
        nest : bool, optional
            Ordering of the returned map.
        replicate_octant : bool, optional
            If True and ``octant_only`` is enabled, replicate the octant to a
            full-sky map before returning (useful for transforms assuming full
            sky).

        Returns
        -------
        ndarray
            Dense Healpix map of length ``hp.nside2npix(self.nside)``.

        Notes
        -----
        If ``self.remove_monopole`` is True, subtracts the mean of the map over
        the masked pixels prior to replication.
        """
        _map = self._field_map(imap, nest=nest)

        if self.remove_monopole:
            _, ipix = self.get_mask(nest=nest, return_ipix=True)
            _map[ipix] -= np.mean(_map[ipix])

        # Replicate if needed and apply mask
        if replicate_octant:
            ipix_o2f = self.get_octant_to_fullsky_ipix(nest=nest)
            _map = _map[ipix_o2f]

        return _map

    def compute_alms(self, imaps=None, save=True, replicate_octant=False, **map2am_kwargs):
        """Compute spherical harmonic coefficients a_lm from maps.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        save : bool, optional
            If True, saves a_lm to FITS files named by ``what='alms'``.
        replicate_octant : bool, optional
            If True, compute a_lm from the octant replicated to full sky. Cannot
            be combined with ``save=True`` in the current implementation.
        **map2am_kwargs
            Forwarded to :func:`healpy.map2alm`.

        Returns
        -------
        dict
            Mapping ``imap -> alm`` (complex ndarray).
        """
        assert not (save and replicate_octant)

        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            if save:
                fname = self.get_fname(imap, what='alms', format='fits')
                if os.path.exists(fname):
                    try:
                        alms[imap] = hp.read_alm(fname)
                        print('They already exist!')
                        continue
                    except:
                        pass

            mask = self.get_mask(nest=False)
            _map = self.get_field_map(imap, nest=False, replicate_octant=replicate_octant)

            alms[imap] = hp.map2alm(_map * mask, **map2am_kwargs)

            if save:
                hp.write_alm(fname, alms[imap], overwrite=True)

        return alms

    def compute_maps_from_alms(self, imaps=None, save=True):
        """Reconstruct HealSparse maps from saved a_lm coefficients.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to reconstruct. If None, uses all maps.
        save : bool, optional
            If True, saves reconstructed maps to the default map filenames.

        Returns
        -------
        list of healsparse.HealSparseMap
            Reconstructed maps, one per requested ``imap``.

        Notes
        -----
        Reconstruction is done in RING ordering and then written into a HealSparse
        map on the mask pixels.
        """
        _, ipix = self.get_mask(nest=False, return_ipix=True)

        maps = []
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            alm = hp.read_alm(self.get_fname(imap, what='alms', format='fits'))
            values = hp.alm2map(alm, nside=self.nside, pol=False)
            tmp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            tmp.update_values_pix(ipix, values[ipix].astype(self.dtype), nest=False)
            if save:
                fname = self.get_fname(imap)
                tmp.write(fname, clobber=True)
            maps.append(tmp)

        return maps

    def _save_cls(self, ell, cl):
        """Save angular power spectra to a SACC FITS file.

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (typically bandpower effective ells).
        cl : dict
            Dictionary keyed by (i, j) containing spectra arrays evaluated at
            ``ell``.

        Returns
        -------
        None
        """
        s = sacc.Sacc()
        tracers = self.get_tracers_sacc()
        for t in tracers:
            s.add_tracer_object(t)
        for i, j in cl.keys():
            for _ell, _cl in zip(ell, cl[i, j]):
                s.add_data_point(self.sacc_type, [tracers[i].name, tracers[j].name], _cl, ell=_ell)
        fname = self.get_fname(what='cls', format='fits')
        s.save_fits(fname, overwrite=True)

    def compute_cls(self, imaps=None, auto_only=False, save=True, use_precomputed_alms=True, pair_job=None):
        """Compute binned pseudo-C_ell bandpowers for requested map pairs.

        Parameters
        ----------
        imaps : array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        save : bool, optional
            If True, save spectra to SACC FITS (``what='cls'``).
        use_precomputed_alms : bool, optional
            If True, reads a_lm from disk and uses Healpy :func:`healpy.alm2cl`
            followed by NaMaster decoupling.
            If False, computes NaMaster fields from masked maps and runs full
            MASTER with :func:`pymaster.compute_full_master`.
        pair_job : int or None, optional
            If provided, compute only the pair at this index in the internal
            list of pairs (for embarrassingly-parallel workflows).

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers.
        cl : dict
            Dictionary keyed by (i, j) containing decoupled bandpowers.
        """
        if imaps is None:
            imaps = range(self.nmaps)

        if auto_only:
            pairs = [(i, i) for i in imaps]
        else:
            # pairs = [(i1, i2) for i, i1 in enumerate(imaps) for i2 in imaps[:i + 1]]
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]  # with this order, i2 is preserved

        cl = {}
        b = self.get_cls_b()
        wsp = self.get_cls_wsp()

        if use_precomputed_alms:
            previous_pair = (-1, -1)
            for pair_count, (i, j) in enumerate(tqdm(pairs, desc="[compute_cls]")):
                if pair_job is not None:
                    if not pair_count == pair_job:
                        continue
                if not i == previous_pair[0]:
                    almi = hp.read_alm(self.get_fname(i, what='alms', format='fits'))
                if i == j:
                    almj = almi
                else:
                    if not j == previous_pair[1]:
                        almj = hp.read_alm(self.get_fname(j, what='alms', format='fits'))
                pcl = hp.alm2cl(almi, almj)
                cl[i, j] = wsp.decouple_cell([pcl])[0]
                previous_pair = (i, j)
        else:
            mask = self.get_mask(nest=False)
            fields = {i: None for i in imaps}
            for i, j in tqdm(pairs, desc="[compute_cls]"):
                if fields[i] is None:
                    fields[i] = nmt.NmtField(mask, [self.get_field_map(i, nest=False)])
                if fields[j] is None:
                    fields[j] = nmt.NmtField(mask, [self.get_field_map(j, nest=False)])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def load_cls(self, return_dict=False):
        """Load saved angular power spectra.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return a dictionary keyed by (i, j). Otherwise, return the
            SACC object.

        Returns
        -------
        sacc.Sacc or (ell, dict)
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cls_dict)`` where ``ell`` is
            the ell array for each spectrum and ``cls_dict`` is keyed by (i, j).

        Raises
        ------
        FileNotFoundError
            If the SACC file does not exist.
        """
        fname = self.get_fname(what='cls', format='fits')
        s = sacc.Sacc.load_fits(fname)
        if return_dict:
            tracers = list(s.tracers.keys())
            tc = s.get_tracer_combinations(self.sacc_type)
            cls_out = {}
            for i in range(self.nmaps):
                for j in range(self.nmaps):
                    if (tracers[i], tracers[j]) in tc:
                        ell, cls_out[i, j] = s.get_ell_cl(self.sacc_type, tracers[i], tracers[j])
            return ell, cls_out
        else:
            return s

    def _noise_cls(self, imap):
        """Return the noise angular power spectrum for one map.

        Parameters
        ----------
        imap : int
            Map index.

        Returns
        -------
        float or ndarray
            Noise power spectrum. If float, interpreted as white noise.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses if noise is needed.
        """
        raise NotImplementedError

    def predict_cls(self, imaps=None, auto_only=False, approximate=False, add_noise=False, error_bars=False, keep_cls=False, stacked=False, ell=None):
        """Predict theory angular power spectra using CCL.

        Parameters
        ----------
        imaps : array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        approximate : bool, optional
            If True, return unbinned spectra evaluated at ``ell`` using CCL.
            If False, compute binned bandpowers using NaMaster bandpower windows.
        add_noise : bool, optional
            If True, add the map noise model for auto-spectra.
        error_bars : bool, optional
            If True, compute Gaussian (diagonal-in-ell) error bars.
        keep_cls : bool, optional
            If True, cache the results on ``self.predicted_cls``.
        stacked : bool, optional
            If True, also return spectra stacked into a single vector (and, when
            ``error_bars=True``, a diagonal covariance matrix).
        ell : ndarray or None, optional
            Multipoles at which to evaluate CCL when ``approximate=True``.
            If None, uses the effective bandpower ells from the binning scheme.

        Returns
        -------
        ell : ndarray
            Ell values (effective bandpower ells or explicit ``ell`` input).
        cl : dict
            Dictionary keyed by (i, j) with spectra arrays.

        Other Parameters
        ----------------
        error_bars : bool
            If True, also returns ``dcl`` (dict keyed by (i, j)).
        stacked : bool
            If True, also returns ``cl_stacked`` (1D array). If both
            ``stacked=True`` and ``error_bars=True``, also returns ``cov`` (2D
            array, diagonal by construction in this implementation).

        Notes
        -----
        When ``approximate=False``, this method evaluates raw C_ell up to
        ``b.lmax`` and bins them using NaMaster bandpower windows.

        The sky fraction is approximated as:
        - ``fsky = 1/8`` for octant-only,
        - ``fsky = 1`` for full-sky.
        """
        if hasattr(self, 'predicted_cls'):
            return self.predicted_cls

        if imaps is None:
            imaps = range(self.nmaps)

        if auto_only:
            pairs = [(i, i) for i in imaps]
        else:
            # pairs = [(i1, i2) for i, i1 in enumerate(imaps) for i2 in imaps[:i + 1]]
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]

        tracers = self.get_tracers_ccl()
        tracers_ = {imap: tracers[imap] for imap in imaps}

        b = self.get_cls_b()

        if ell is None:
            ell = b.get_effective_ells()
            nell = b.get_nell_list()
        else:
            assert approximate
            assert not error_bars
        fsky = 1 / 8. if self.octant_only else 1.
        order = lambda i, j: (max(i, j), min(i, j))

        if not approximate:
            wsp = self.get_cls_wsp()
            bpws = wsp.get_bandpower_windows()[0, :, 0, :]

        def angular_cl(*args):
            # return pyccl.angular_cl(*args)
            # c, t1, t2, ells = args
            # for ell in ells:
            #     print(ell, t1.chi_min, t1.chi_max, t2.chi_min, t2.chi_max)
            #     pyccl.angular_cl(c, t1, t2, ell)
            #     pass
            try:
                return pyccl.angular_cl(*args)
            except pyccl.CCLError:
                return pyccl.angular_cl(*args, limber_integration_method='spline')

        self.cosmo.compute_growth()

        cl = {}
        for i, j in tqdm(pairs, desc="[predict_cls]"):
            if approximate:
                cl[i, j] = angular_cl(self.cosmo, tracers_[i], tracers_[j], ell)
                if add_noise and i == j:
                    cl[i, j] += self._noise_cls(i)
            else:
                cl_raw = angular_cl(self.cosmo, tracers_[i], tracers_[j], range(b.lmax + 1))
                if self.f_ell is not None:
                    if self.f_ell == 'pixwin':
                        _f_ell = hp.pixwin(self.nside)[:b.lmax + 1]
                    else:
                        _f_ell = np.ones(b.lmax + 1)
                        _f_ell[b.get_ell_list(0)[0]:] = self.f_ell
                    cl_raw *= _f_ell**2
                cl[i, j] = np.dot(bpws, cl_raw)

                if add_noise and i == j:
                    nl_raw = self._noise_cls(i)
                    if self.f_ell is not None:
                        nl_raw *= _f_ell**2
                    cl[i, j] += np.dot(bpws, nl_raw)

        if error_bars:
            dcl = {}
            # Diagonal error bars
            for i, j in pairs:
                if i == j:
                    dcl2_num = 2 * (cl[i, i] + self._noise_cls(i))**2
                else:
                    dcl2_num = (cl[i, i] + self._noise_cls(i)) * (cl[j, j] + self._noise_cls(j)) + cl[order(i, j)]**2
                dcl[i, j] = np.sqrt(dcl2_num / ((2 * ell + 1.) * fsky * nell))

            if stacked:
                dcl_full = {}
                Dl = {}

                # Compute covariance blocks
                for i, j in pairs:
                    Dl[i, j] = np.copy(cl[i, j])
                    Dl[j, i] = np.copy(cl[i, j])
                    if i == j:
                        Dl[i, j] += self._noise_cls(i)

                # Stack Cl's and covariance blocks
                if auto_only:
                    for i, j in pairs:
                        dcl_full[i, j, i, j] = (Dl[i, j] * Dl[j, i] + Dl[i, i] * Dl[j, j]) / ((2 * ell + 1.) * fsky * nell)

                    cl_stacked = np.zeros(len(pairs) * len(ell))
                    cov = np.zeros([len(pairs) * len(ell), len(pairs) * len(ell)])
                    for nrow, (i, j) in enumerate(pairs):
                        cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                        cov[nrow * len(ell):(nrow + 1) * len(ell), nrow * len(ell):(nrow + 1) * len(ell)] = np.diag(dcl_full[i, j, i, j])
                else:
                    for i, j in pairs:
                        for k, l in pairs:
                            dcl_full[i, j, k, l] = (Dl[i, l] * Dl[j, k] + Dl[i, k] * Dl[j, l]) / ((2 * ell + 1.) * fsky * nell)

                    cl_stacked = np.zeros(len(pairs) * len(ell))
                    cov = np.zeros([len(pairs) * len(ell), len(pairs) * len(ell)])
                    for nrow, (i, j) in enumerate(pairs):
                        cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                        for ncol, (k, l) in enumerate(pairs):
                            cov[nrow * len(ell):(nrow + 1) * len(ell), ncol * len(ell):(ncol + 1) * len(ell)] = np.diag(dcl_full[i, j, k, l])

                if keep_cls:
                    self.predicted_cls = ell, cl, dcl, cl_stacked, cov
                return ell, cl, dcl, cl_stacked, cov
            else:
                if keep_cls:
                    self.predicted_cls = ell, cl, dcl
                return ell, cl, dcl
        else:
            if stacked:
                cl_stacked = np.zeros(len(pairs) * len(ell))
                for nrow, (i, j) in enumerate(pairs):
                    cl_stacked[nrow * len(ell):(nrow + 1) * len(ell)] = np.copy(cl[i, j])
                if keep_cls:
                    self.predicted_cls = ell, cl, cl_stacked
                return ell, cl, cl_stacked
            else:
                if keep_cls:
                    self.predicted_cls = ell, cl
                return ell, cl

    def triangle_plot_cls(
            self,
            auto_only=False,
            logy=False,
            theory_only=False,
            data_only=False,
            ratio_data_theory=False,
            diff_data_theory_over_error=False,
            save=True,
            sharey=None,
            figaxes=None,
            colortheory='orange',
            colordata='dodgerblue',
            approximate=False,
            ell_factor=0,
            error_bars=False):
        """Plot a triangle matrix of theory and/or measured angular spectra.

        Parameters
        ----------
        auto_only : bool, optional
            If True, plot only auto-spectra (single row). Default is False.
        logy : bool, optional
            If True, use a log y-scale for standard spectra panels. Default is False.
        theory_only : bool, optional
            If True, plot theory curves only. Default is False.
        data_only : bool, optional
            If True, plot measured points only. Default is False.
        ratio_data_theory : bool, optional
            If True, plot ``C_ell(obs)/C_ell(th)``. Mutually exclusive with
            ``logy/theory_only/data_only``. Default is False.
        diff_data_theory_over_error : bool, optional
            If True, plot ``(C_ell(obs)-C_ell(th))/sigma(C_ell)``. Default is False.
        save : bool, optional
            If True, save figure as ``what='cls', format='png'``. Default is True.
        sharey : bool or {'row'} or None, optional
            Matplotlib sharey option passed to :func:`plt.subplots`.
        figaxes : tuple(fig, axes) or None, optional
            If provided, draw into the given (fig, axes).
        colortheory : str, optional
            Color for theory curves. Default is 'orange'.
        colordata : str, optional
            Color for measured points. Default is 'dodgerblue'.
        approximate : bool, optional
            Forwarded to :meth:`predict_cls`. Default is False.
        ell_factor : {0, 1, 2}, optional
            Vertical scaling:
            - 0: plot ``C_ell``
            - 1: plot ``ell*C_ell``
            - 2: plot ``ell(ell+1)C_ell/2pi``
        error_bars : bool, optional
            If True, draw theory uncertainty bands from Gaussian errors.

        Returns
        -------
        None
        """
        n = self.nmaps
        m = 1 if auto_only else n
        sharey = sharey or False  #False if auto_only else 'row'
        if figaxes is None:
            # fig, axes = plt.subplots(m, n, figsize=(4 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)
            fig, axes = plt.subplots(m, n, figsize=(3 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)
        else:
            fig, axes = figaxes

        if ratio_data_theory:
            assert not (logy or theory_only or data_only)

        ell, cl, dcl = self.predict_cls(auto_only=auto_only, approximate=approximate, error_bars=True)
        s = self.load_cls()

        tracers = list(s.tracers.keys())
        tc = s.get_tracer_combinations(self.sacc_type)

        if ell_factor == 0:
            ell_factor = 1.
            ylabel = r'$C_\ell$'
        elif ell_factor == 1:
            ell_factor = ell
            ylabel = r'$\ell\,C_\ell$'
        elif ell_factor == 2:
            ell_factor = ell * (ell + 1) / (2 * np.pi)
            ylabel = r'$\ell(\ell+1)C_\ell/2\pi$'
        else:
            raise ValueError

        for i in range(m):
            for j in range(n):
                if auto_only:
                    ax = axes[0, j]
                    i = j
                    title = f'{self.base_fname} {i}'
                else:
                    ax = axes[i, j]
                    title = f'{self.base_fname} {i} {j}'
                ax_used = False
                cl_obs = s.get_ell_cl(self.sacc_type, tracers[i], tracers[j])[1] if (tracers[i], tracers[j]) in tc else None
                cl_th = cl[i, j] if (i, j) in cl.keys() else None
                if ratio_data_theory:
                    if cl_obs is not None and cl_th is not None:
                        ax.plot(ell, cl_obs / cl_th, c=colordata)
                        if error_bars:
                            ax.fill_between(ell, 1. - dcl[i, j] / cl_th, 1. + dcl[i, j] / cl_th, color=colortheory, alpha=0.2, zorder=-99)
                        ax.axhline(y=1., c='.8', ls='--')
                        ax.set_ylabel(r'$C_\ell^{\rm obs} / C_\ell^{\rm th}$')
                        ax_used = True
                elif diff_data_theory_over_error:
                    if cl_obs is not None and cl_th is not None:
                        ax.plot(ell, (cl_obs - cl_th) / dcl[i, j], c=colordata)
                        if error_bars:
                            ax.axhspan(-1., +1., color=colortheory, alpha=0.2, zorder=-99)
                        ax.axhline(y=0., c='.8', ls='--')
                        ax.set_ylabel(r'$\Delta C_\ell / \sigma(C_\ell)$')
                        ax_used = True
                else:
                    if cl_th is not None and not data_only:
                        ax.plot(ell, ell_factor * cl_th, c=colortheory)
                        if error_bars:
                            ax.fill_between(ell, ell_factor * (cl_th - dcl[i, j]), ell_factor * (cl_th + dcl[i, j]), color=colortheory, alpha=0.2, zorder=-99)
                        ax_used = True
                    if cl_obs is not None and not theory_only:
                        ax.plot(ell, ell_factor * cl_obs, c=colordata, ls=' ', marker='o', markersize=5)
                        ax_used = True
                if ax_used:
                    ax.set_xlabel('Multipole $\ell$')
                    ax.set_ylabel(ylabel)
                    ax.set_xscale('log')
                    ax.set_title(title)
                    if logy:
                        ax.set_yscale('log')
                else:
                    ax.axis('off')

        fig.tight_layout()

        if save:
            fname = self.get_fname(what='cls', format='png')
            fig.savefig(fname, bbox_inches='tight')

        # plt.show()

    def compute_map_moments(self, theta_smooth=np.geomspace(2.5, 250, 8), nside_small=512):
        """Compute 2nd/3rd/4th moments of smoothed maps.

        Parameters
        ----------
        theta_smooth : array_like, optional
            Gaussian smoothing scales in arcmin.
        nside_small : int, optional
            Nside used for the smoothed maps and moment evaluation.

        Returns
        -------
        theta_smooth : ndarray
            Smoothing scales (arcmin).
        second_moment : dict
            Covariance matrices keyed by smoothing scale (theta -> (nmaps,nmaps)).
        third_moment : dict
            Third-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps)).
        fourth_moment : dict
            Fourth-order moments keyed by smoothing scale
            (theta -> (nmaps,nmaps,nmaps,nmaps)).

        Notes
        -----
        This method:
        1) loads a_lm for each map,
        2) resizes them to ``lmax_out = 2*nside_small``,
        3) makes smoothed maps for each ``theta``,
        4) computes raw moments on the masked pixels,
        5) caches results to ``what='2nd_3rd_4th_moments', format='npy'``.
        """
        fname = self.get_fname(what='2nd_3rd_4th_moments', format='npy')

        if os.path.exists(fname):
            print('Loading pre-existing file: ' + fname)
            moments = np.load(fname, allow_pickle=True).item()
            return moments['theta_smooth'], moments['second_moment'], moments['third_moment'], moments['fourth_moment']

        kappa = {}
        lmax_out = 2 * nside_small

        # Mask
        mask, ipix = self._create_mask(nside=nside_small)

        # First, let's compute kappa maps for all bins and all theta (smoothing scales)
        for imap in range(self.nmaps):
            kappa[imap] = {}

            # read the alm's
            alm = self.compute_alms(imaps=imap)[imap]
            lmax = hp.Alm.getlmax(len(alm))
            alm = hp.sphtfunc.resize_alm(alm, lmax=lmax, mmax=lmax, lmax_out=lmax_out, mmax_out=lmax_out)

            # smooth maps for all theta
            for theta in theta_smooth:
                kappa_map = hp.alm2map(alm, nside_small, sigma=theta * np.pi / (180. * 60), lmax=lmax_out)
                kappa_map = hp.pixelfunc.reorder(kappa_map, r2n=True)
                kappa_map[~mask] = hp.UNSEEN
                kappa_map[mask] -= np.mean(kappa_map[mask])  # Remove mean for moments calculation
                kappa[imap][theta] = kappa_map

        moments = {}
        moments['theta_smooth'] = theta_smooth
        moments['second_moment'] = {}
        moments['third_moment'] = {}
        moments['fourth_moment'] = {}

        # Now compute moments
        for theta in theta_smooth:
            # Stack data for all maps at given theta
            data = np.stack([kappa[imap][theta][ipix] for imap in range(self.nmaps)], axis=0)

            # 2nd moment
            moments['second_moment'][theta] = np.cov(data)

            # 3rd moment
            moments['third_moment'][theta] = np.zeros((self.nmaps, self.nmaps, self.nmaps))
            for i, j, k in product(range(self.nmaps), repeat=3):
                moments['third_moment'][theta][i, j, k] = np.mean(data[i] * data[j] * data[k])

            # 4th moment
            moments['fourth_moment'][theta] = np.zeros((self.nmaps, self.nmaps, self.nmaps, self.nmaps))
            for i, j, k, l in product(range(self.nmaps), repeat=4):
                moments['fourth_moment'][theta][i, j, k, l] = np.mean(data[i] * data[j] * data[k] * data[l])

        # Let's save them all
        print('Saving output as:' + fname)
        np.save(fname, moments)

        return theta_smooth, moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def generate_gaussian_maps(self, imaps=None, auto_only=False, seed=-1):
        """Generate Gaussian random maps consistent with predicted spectra.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices to generate. If None, generates all maps.
        auto_only : bool, optional
            If True, generate each map independently using its auto-spectrum.
            If False, generate a correlated set of maps from all cross-spectra.
        seed : int, optional
            Random seed passed to NaMaster synfast routines. Default is -1.

        Returns
        -------
        None

        Notes
        -----
        Maps are generated full-sky and then restricted to the analysis mask
        pixels before writing as HealSparse maps.
        """
        if imaps is None:
            imaps = range(self.nmaps)

        # Compute theory Cl's
        _, cls = self.predict_cls(approximate=True, auto_only=auto_only, ell=np.arange(3 * self.nside))

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        if auto_only:
            for imap in self.tqdm(imaps, desc="[generate_gaussian_maps]"):
                # Generate full-sky Gaussian map
                out = nmt.utils.synfast_spherical(self.nside, [cls[imap, imap]], spin_arr=[0], seed=seed)[0]

                # Store in healsparse map
                hsp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsp.update_values_pix(ipix, out[ipix].astype(self.dtype), nest=False)

                # Save map
                fname = self.get_fname(imap)
                hsp.write(fname, clobber=True, format=self.save_format)

        else:
            # Generate full-sky Gaussian map
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]
            outs = nmt.utils.synfast_spherical(self.nside, [cls[p] for p in pairs], spin_arr=[0] * self.nmaps, seed=seed)

            for imap, out in zip(imaps, outs):
                # Store in healsparse map
                hsp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsp.update_values_pix(ipix, out[ipix].astype(self.dtype), nest=False)

                # Save map
                fname = self.get_fname(imap)
                hsp.write(fname, clobber=True, format=self.save_format)
