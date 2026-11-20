"""Statistics of the maps (:class:`MapStatistics`, inherited by :class:`~pollux.map_builder.MapBuilder`).

Tracers and theory of the maps, power spectra with NaMaster and their predictions, moments, Gaussian maps, and plots.
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from functools import cached_property
from itertools import product
from typing import Any, Iterable

import healpy as hp
import numpy as np
import pyccl
import pymaster as nmt
import sacc

from ..io import layout
from ..cosmo import Theory, Tracer
from ..utils.binning import binning_id, make_nmtbin_logspaced
from .base import TwoPointStatistics

logger = logging.getLogger(__name__)


class MapStatistics(TwoPointStatistics):
    """Statistics of the maps of a builder: tracers and predictions, power spectra, moments and plots.

    Mixin of :class:`~pollux.map_builder.MapBuilder`, using its maps, a_lm, footprint and binning options.
    Subclasses implement :meth:`_create_tracers`, and :meth:`_noise_cls` for noise-aware predictions.
    """

    @property
    def tracers_ccl(self) -> list:
        """Return the CCL tracers of the maps.

        Returns
        -------
        list of pyccl.Tracer
        """
        return [t.ccl(self.cosmo) for t in self.tracers]

    @cached_property
    def tracers(self) -> list:
        """Return (and lazily create) the tracers describing the maps.

        Returns
        -------
        list of pollux.cosmo.Tracer
            One tracer per map, from which CCL (:attr:`tracers_ccl`) and SACC
            (:attr:`tracers_sacc`) tracers are derived.
        """
        return self._create_tracers()

    @abstractmethod
    def _create_tracers(self) -> list:
        """Create the tracers describing the maps.

        Returns
        -------
        list of pollux.cosmo.Tracer
            One tracer per map.
        """

    def _sacc_tracers(self) -> dict[str, list]:
        """Return the SACC tracers of the maps (role ``'map'``), see :meth:`~pollux.statistics.base.TwoPointStatistics._sacc_tracers`."""
        return {'map': self.tracers_sacc}

    @cached_property
    def tracers_sacc(self) -> list:
        """Return the SACC tracers used to store measured spectra.

        Returns
        -------
        list of sacc.tracers.BaseTracer
        """
        return [t.sacc() for t in self.tracers]

    def matter_power_spectrum(self) -> pyccl.Pk2D | None:
        """Return the matter power spectrum consistent with the maps, for predictions.

        Returns
        -------
        pyccl.Pk2D or None
            None (CCL's nonlinear matter power spectrum) by default; overridden
            by builders whose maps follow another one (e.g.
            :class:`~pollux.baryons.BaryonifiedShellBuilder`, or lensing builders
            using their shell builder's).
        """
        return None

    def compute_cls(
            self,
            imaps: int | Iterable[int] | None = None,
            *,
            auto_only: bool = False,
            pairs: list | None = None,
            save: bool = True,
            use_precomputed_alms: bool = True) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
        """Compute binned pseudo-C_ell bandpowers for requested map pairs.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` map pairs (overrides ``imaps`` and ``auto_only``).
        save : bool, optional
            If True, save spectra to SACC FITS (``what='cls'``).
        use_precomputed_alms : bool, optional
            If True, reads a_lm from disk and uses Healpy :func:`healpy.alm2cl`
            followed by NaMaster decoupling.
            If False, computes NaMaster fields from masked maps and runs full
            MASTER with :func:`pymaster.compute_full_master`.

        Returns
        -------
        ell : ndarray
            Effective multipoles of the bandpowers.
        cl : dict
            Dictionary keyed by (i, j) containing decoupled bandpowers.
        """
        imaps, pairs = self._pairs(imaps, auto_only, pairs)

        cl = {}
        b = self.b
        wsp = self.wsp

        if use_precomputed_alms:
            # pseudo-C_ell from the saved alms (of the weighted maps), decoupled with the shared workspace;
            # pairs come with j fixed and i increasing, so each alm is read at most once per pair change
            previous_pair = (-1, -1)
            for i, j in self.tqdm(pairs, desc="[compute_cls]"):
                if not i == previous_pair[0]:
                    almi = self.compute_alms(i)
                if i == j:
                    almj = almi
                else:
                    if not j == previous_pair[1]:
                        almj = self.compute_alms(j)
                pcl = hp.alm2cl(almi, almj)
                cl[i, j] = wsp.decouple_cell([pcl])[0]
                previous_pair = (i, j)
        else:
            # NaMaster fields of the weighted maps, built when first needed
            fields = {i: None for i in imaps}
            for i, j in self.tqdm(pairs, desc="[compute_cls]"):
                for k in (i, j):
                    if fields[k] is None:
                        weights, _map = self._alm_field(k)
                        fields[k] = nmt.NmtField(weights, [_map])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def _pairs(self, imaps: int | Iterable[int] | None = None, auto_only: bool = False, pairs: list | None = None) -> tuple[list[int], list[tuple[int, int]]]:
        """Return the map indices and ``(i, j)`` pairs selected by ``imaps``, ``auto_only`` or ``pairs``.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, all maps.
        auto_only : bool, optional
            If True, only auto-spectra.
        pairs : list of tuple of int or None, optional
            Explicit pairs, overriding ``imaps`` and ``auto_only``.

        Returns
        -------
        imaps : list of int
            Maps involved.
        pairs : list of tuple of int
            ``(i, j)`` with ``i >= j`` (unless given explicitly).
        """
        if pairs is not None:
            pairs = [tuple(pair) for pair in pairs]
            return sorted({k for pair in pairs for k in pair}), pairs
        imaps = list(self.iter(imaps))
        if auto_only:
            return imaps, [(i, i) for i in imaps]
        return imaps, [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]  # with this order, i2 is preserved

    @cached_property
    def b(self) -> nmt.NmtBin:
        """Return (and lazily create) the NaMaster binning scheme.

        Returns
        -------
        pymaster.NmtBin
            Bandpower binning used for :meth:`compute_cls` and :meth:`predict_cls`.

        Notes
        -----
        Lazy, so that the builder stays light to pickle: the ``NmtBin`` is not
        constructed until first accessed (unless given at construction).
        """
        return make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self.nlb, f_ell=self.f_ell)

    @cached_property
    def wsp(self) -> nmt.NmtWorkspace:
        """Return (and lazily create) the NaMaster workspace.

        The workspace stores the mode-coupling matrix and bandpower windows for
        the current mask and binning, and is cached on disk with the footprint products (``io.dir('footprint')``).

        Returns
        -------
        pymaster.NmtWorkspace
            Workspace containing the coupling matrix and bandpower windows.

        Notes
        -----
        Lazy, so that the builder stays light to pickle: the ``NmtWorkspace`` is
        not constructed until first accessed.
        """
        fname = self._wsp_path
        wsp = nmt.NmtWorkspace()
        if os.path.exists(fname):
            wsp.read_from(fname)
            logger.info("Found and loaded workspace at %s.", fname)
        else:
            mask = self._alm_weights
            b = self.b
            dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)], lite=True)
            wsp.compute_coupling_matrix(dummy_f, dummy_f, b)
            with layout.atomic_write(fname) as tmp:
                wsp.write_to(tmp)
            logger.info("Computed and saved workspace to file %s.", fname)
            self._saved_bpws(fname, lambda: wsp)
        return wsp

    @property
    def _wsp_path(self) -> str:
        """File of the NaMaster workspace, shared by the realizations with the same footprint and binning."""
        padding = f'_pad{self.padding_deg:g}{self.apotype or ""}' if self.padding_deg is not None else ''
        if self.replicate_octant:
            padding = '_octrep'
        return self.io.path('footprint', f'wsp_nside{self.nside}_{self.footprint_id}{padding}_b{binning_id(self.b, self.f_ell)}', ext='wsp')

    def _save_cls(self, ell: np.ndarray, cl: dict[tuple[int, int], np.ndarray], metadata: dict | None = None) -> str:
        """Save angular power spectra to a SACC FITS file (``cls.fits`` of the product).

        Parameters
        ----------
        ell : ndarray
            Effective multipoles (typically bandpower effective ells).
        cl : dict
            Dictionary keyed by (i, j) containing spectra arrays evaluated at
            ``ell``.
        metadata : dict or None, optional
            Metadata of the SACC file (e.g. the identifier of the options of the measurement, see
            :mod:`pollux.pipeline.tasks`).

        Returns
        -------
        str
            Path to the saved SACC FITS file.
        """
        return self._save_sacc(self.io.path(self.product, 'cls'), {self.sacc_type: cl}, ell, {self.sacc_type: ('map', 'map')}, metadata=metadata)

    def load_cls(self, return_dict: bool = False, return_errors: bool = False) -> sacc.Sacc | tuple:
        """Load saved angular power spectra.

        Parameters
        ----------
        return_dict : bool, optional
            If True, return a dictionary keyed by (i, j). Otherwise, return the
            SACC object.
        return_errors : bool, optional
            With ``return_dict``, also return the error bars of the saved covariance, keyed by (i, j)
            (empty without covariance).

        Returns
        -------
        sacc.Sacc or (ell, dict[, dict])
            If ``return_dict=False``, returns the loaded SACC object.
            If ``return_dict=True``, returns ``(ell, cls_dict)`` where ``ell`` is
            the ell array for each spectrum and ``cls_dict`` is keyed by (i, j).

        Raises
        ------
        FileNotFoundError
            If the SACC file does not exist.
        """
        res = self._load_sacc(self.io.path(self.product, 'cls'), return_dict=return_dict, return_errors=return_errors)
        if not return_dict:
            return res
        ell, cl, *errors = (r.get(self.sacc_type, {}) for r in res)
        return (next(iter(ell.values())), cl, *errors)

    def predict_cls(
            self,
            imaps: int | Iterable[int] | None = None,
            *,
            auto_only: bool = False,
            pairs: list | None = None,
            binning: str = 'wsp',
            error_bars: bool = False,
            stacked: bool = False,
            ell: np.ndarray | None = None) -> tuple:
        """Predict theory angular power spectra using CCL.

        Parameters
        ----------
        imaps : int or array_like of int or None, optional
            Map indices. If None, uses all maps.
        auto_only : bool, optional
            If True, compute only auto-spectra.
        pairs : list of tuple of int or None, optional
            If given, only these ``(i, j)`` map pairs (overrides ``imaps`` and ``auto_only``).
        binning : {'wsp', 'average', 'unbinned'}, optional
            ``'wsp'`` (default): bandpowers binned with the bandpower windows of
            :attr:`wsp`, like the measured spectra. ``'average'``: bandpowers
            averaged over each bandpower. ``'unbinned'``: spectra at ``ell``.
        error_bars : bool, optional
            If True, compute Gaussian (diagonal-in-ell) error bars, with
            ``C_ell + N_ell`` for auto-spectra (noise from :meth:`_noise_cls`).
        stacked : bool, optional
            If True, also return spectra stacked into a single vector (and, when
            ``error_bars=True``, a diagonal covariance matrix).
        ell : ndarray or None, optional
            Multipoles at which to evaluate CCL when ``binning='unbinned'``.
            If None, uses the effective bandpower ells from the binning scheme.

        Returns
        -------
        ell : ndarray
            Ell values (effective bandpower ells or explicit ``ell`` input).
        cl : dict
            Dictionary keyed by (i, j) with spectra arrays.
        dcl : dict
            Only with ``error_bars``: Gaussian error bars, keyed like ``cl``.
        cl_stacked : ndarray
            Only with ``stacked``: spectra stacked into one vector.
        cov : ndarray
            Only with ``stacked`` and ``error_bars``: covariance of ``cl_stacked``
            (diagonal in ell).

        Notes
        -----
        Wraps :meth:`pollux.cosmo.Theory.predict_cls` with the builder's tracers,
        matter power spectrum, workspace bandpower windows, pixel window
        (``f_ell``), noise model (:meth:`_noise_cls`) and sky fraction.
        """

        imaps, pairs = self._pairs(imaps, auto_only, pairs)

        if ell is not None:
            if binning != 'unbinned':
                raise ValueError("ell can only be given with binning='unbinned'")
            if error_bars:
                raise ValueError("ell cannot be given with error_bars")

        # bandpower windows of the workspace, and the field window (pixel window, or a given one above the first bandpower)
        bpws, f_ell = None, None
        if binning == 'wsp':
            bpws = self.bpws
        if binning != 'unbinned':
            if self.f_ell is not None:
                if self.f_ell == 'pixwin':
                    f_ell = hp.pixwin(self.nside)[:self.b.lmax + 1]
                else:
                    f_ell = np.ones(self.b.lmax + 1)
                    f_ell[self.b.get_ell_list(0)[0]:] = self.f_ell

        # map tracers, with their noise for the error bars
        tracers = self.tracers
        noise = {tracers[i].name: self._noise_cls(i) for i in imaps} if error_bars else None
        res = self.theory.predict_cls({(i, j): (tracers[i], tracers[j]) for i, j in pairs},
                                      b=self.b,
                                      ell=ell,
                                      binning=binning,
                                      bpws=bpws,
                                      f_ell=f_ell,
                                      noise=noise,
                                      error_bars=error_bars,
                                      stacked=stacked,
                                      fsky=self.fsky,
                                      cross_covariance=not auto_only,
                                      progress=self.tqdm)
        return res

    @cached_property
    def bpws(self) -> np.ndarray:
        """Return the bandpower windows of the workspace (see :attr:`wsp`).

        They are saved apart, next to the workspace, so that predictions do not load the
        coupling matrix (several GB at high resolution), only once to save them.

        Returns
        -------
        ndarray
            Shape ``(n_cl, n_bands, n_cl, lmax + 1)``.
        """
        return self._saved_bpws(self._wsp_path, lambda: self.wsp)

    def _noise_cls(self, imap: int) -> float | np.ndarray:
        """Return the noise angular power spectrum for one map.

        Optional: only needed when calling :meth:`predict_cls` with
        ``error_bars=True``. Subclasses that don't model
        noise can leave this unimplemented; it raises ``NotImplementedError``
        if called.

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
            If this builder does not implement a noise model.
        """
        raise NotImplementedError

    @cached_property
    def theory(self) -> Theory:
        """Return the theory predictor of the maps.

        Returns
        -------
        pollux.cosmo.Theory
            With the builder's cosmology and :meth:`matter_power_spectrum`.
        """
        return Theory(self.cosmo, p_of_k_a=self.matter_power_spectrum())

    def reference_scales(self, smooth_Mpc: float | None = None) -> dict[str, np.ndarray]:
        """Return the angular scales of the simulation box, of the map resolution, and of a smoothing, per map.

        Parameters
        ----------
        smooth_Mpc : float or None, optional
            Comoving Gaussian smoothing scale in Mpc.

        Returns
        -------
        dict
            ``theta_box`` (arcmin) and ``ell_box``, and ``theta_smooth``, ``ell_smooth``
            with ``smooth_Mpc`` (at least the pixel size, see
            :meth:`pollux.cosmo.Tracer.effective_smooth_Mpc`), one value per map, at the redshift where its signal comes
            from (:meth:`pollux.cosmo.Tracer.mean_z_kernel`, e.g. the lenses for lensing maps);
            ``theta_res`` (arcmin) and ``ell_res``, the pixel size of ``nside``, the same for all maps.
        """
        z_kernel = np.array([t.mean_z_kernel(self.cosmo) for t in self.tracers])
        d_a = pyccl.comoving_angular_distance(self.cosmo, 1. / (1. + z_kernel))
        rad2arcmin = 180. * 60. / np.pi
        ell_box = 2. * np.pi / self.reader.boxsize * d_a
        theta_res = np.full_like(d_a, hp.nside2resol(self.nside))
        scales = dict(theta_box=rad2arcmin * np.pi / ell_box, ell_box=ell_box, theta_res=rad2arcmin * theta_res, ell_res=np.pi / theta_res)
        if smooth_Mpc is not None:
            theta_smooth = Tracer.effective_smooth_Mpc(smooth_Mpc, d_a, self.nside) / d_a
            scales.update(theta_smooth=rad2arcmin * theta_smooth, ell_smooth=np.pi / theta_smooth)
        return scales

    def plot_cls(
            self,
            plot_dir: str | None = None,
            *,
            modes: tuple[str, ...] = ('values', 'ratio'),
            ell_factor: int = 1,
            logy: bool = False,
            error_bars: bool = False,
            **predict_kwargs: Any) -> None:
        """Plot the saved angular power spectra against their prediction (:meth:`predict_cls`), one panel per map pair.

        Parameters
        ----------
        plot_dir : str or None, optional
            Output directory. Default is ``{save_dir}/plots``.
        modes : tuple of str, optional
            Plots to make, see :func:`pollux.statistics.plotting.plot_pairs` (``'values'``, ``'ratio'``,
            ``'diff_over_error'``). Default is ``('values', 'ratio')``.
        ell_factor : {0, 1, 2}, optional
            In ``'values'`` mode, plot ``C_ell`` (0), ``ell C_ell`` (1, default) or
            ``ell (ell + 1) C_ell / 2 pi`` (2).
        logy : bool, optional
            If True, use a log y-scale in ``'values'`` mode.
        error_bars : bool, optional
            If True, draw Gaussian errors of the prediction as bands (always used by
            ``'diff_over_error'``).
        **predict_kwargs
            Options of :meth:`predict_cls`, e.g. ``auto_only``, ``pairs`` or ``binning``: the
            predicted pairs are plotted.

        Notes
        -----
        Saves ``{map_name}_cls``, with suffix ``_{mode}`` for modes other than ``'values'``.
        """
        from .plotting import plot_pairs
        plot_dir = plot_dir or os.path.join(self.save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        need_errors = error_bars or 'diff_over_error' in modes
        ell, cl_th, *dcl = self.predict_cls(error_bars=need_errors, **predict_kwargs)
        dcl = dcl[0] if need_errors else None
        cl = self.load_cls(return_dict=True)[1]
        for mode in modes:
            plot_pairs(
                ell,
                data={ij: cl[ij] for ij in cl_th if ij in cl},
                theory=cl_th,
                theory_errors=dcl if (error_bars or mode == 'diff_over_error') else None,
                mode=mode,
                auto_only=all(i == j for i, j in cl_th),
                ell_factor=ell_factor,
                scales=self.reference_scales(),
                color_theory='orange',
                logy=logy,
                xlim=(self.b.get_ell_min(0), self.b.get_ell_max(self.b.get_n_bands() - 1)),
                title=f'{self.map_name} {{i}} {{j}}',
                sharey=False,
                fname=os.path.join(plot_dir, f'{self.map_name}_cls' + ('' if mode == 'values' else f'_{mode}') + '.png'))

    def compute_map_moments(self, theta_smooth: np.ndarray | Iterable[float], nside_small: int = 512, save: bool = True) -> tuple[np.ndarray, dict, dict, dict]:
        """Compute 2nd/3rd/4th moments of smoothed maps.

        Parameters
        ----------
        theta_smooth : array_like of float
            Gaussian smoothing scales in arcmin.
        nside_small : int, optional
            Nside used for the smoothed maps and moment evaluation.
        save : bool, optional
            If True, saves results to ``what='2nd_3rd_4th_moments', format='npy'``.

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
        5) saves results to ``what='2nd_3rd_4th_moments', format='npy'`` if ``save=True``.
        """
        if min(theta_smooth) <= hp.nside2resol(nside_small, arcmin=True):
            raise ValueError('Smoothing scale should be larger than pixel size to avoid pixel effects.')

        kappa = {}
        lmax_out = 2 * nside_small

        # Mask (downgraded from self.nside to nside_small for moment computation)
        mask_full = self._create_mask()
        mask = hp.ud_grade(mask_full, nside_small, order_in='NESTED', order_out='NESTED') > 0.5
        ipix = np.where(mask)[0]

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

        if save:
            self.io.write_npy(self.path(what='2nd_3rd_4th_moments', ext='npy'), moments)

        return theta_smooth, moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def load_map_moments(self) -> tuple[np.ndarray, dict, dict, dict]:
        """Load previously saved 2nd/3rd/4th moments of smoothed maps.

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

        Raises
        ------
        FileNotFoundError
            If the moments file does not exist.
        """
        fname = self.path(what='2nd_3rd_4th_moments', ext='npy')
        moments = self.io.read_npy(fname, allow_pickle=True).item()
        logger.debug("Found and loaded file at %s.", fname)
        return moments['theta_smooth'], moments['second_moment'], moments['third_moment'], moments['fourth_moment']

    def generate_gaussian_maps(self, imaps: int | Iterable[int] | None = None, *, auto_only: bool = False, seed: int = -1) -> None:
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

        Notes
        -----
        Maps are generated full-sky and then restricted to the padded footprint
        (like the maps they stand for) before writing as HealSparse maps.
        """
        imaps = list(self.iter(imaps))

        # Compute theory Cl's
        _, cls = self.predict_cls(binning='unbinned', auto_only=auto_only, ell=np.arange(3 * self.nside))

        _, ipix = self.get_mask(nest=False, return_ipix=True, padded=True)

        if auto_only:
            for imap in self.tqdm(imaps, desc="[generate_gaussian_maps]"):
                # Generate full-sky Gaussian map
                out = nmt.utils.synfast_spherical(self.nside, [cls[imap, imap]], spin_arr=[0], seed=seed)[0]

                # Store in HealSparse map
                hspmap = self._make_hspmap(ipix, out[ipix])

                # Save map
                fname = self.path(imap)
                self._save_map(hspmap, fname)

        else:
            # Generate full-sky Gaussian map
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]
            outs = nmt.utils.synfast_spherical(self.nside, [cls[p] for p in pairs], spin_arr=[0] * len(imaps), seed=seed)

            for imap, out in zip(imaps, outs, strict=True):
                # Store in HealSparse map
                hspmap = self._make_hspmap(ipix, out[ipix])

                # Save map
                fname = self.path(imap)
                self._save_map(hspmap, fname)
