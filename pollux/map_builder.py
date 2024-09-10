import os
from collections.abc import Iterable

import healpy as hp
import healsparse
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
import sacc
from tqdm.auto import tqdm, trange

from .utils import *


class MapBuilder:

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
            remove_monopole=True):
        """`MapBuilder` is a generic class to build maps that groups functions for I/O, Cl's computations, etc.

        Parameters
        ----------
        nside
            `nside` parameter for healpix
        dtype
            dtype of maps
        save_dir
            Folder where to save maps
        base_fname
            Basis of file names
        nmaps
            Number of maps
        nside_c, optional
            `nside_c` parameter for healsparse, by default 32
        octant_only, optional
            Whether to limit maps to the octant, by default True
        save_format, optional
            File format, by default 'fits'
        wsp_dir, optional
            Folder where to find the NaMaster workspace object, by default None
        sacc_type, optional
            Tracer type in SACC, by default None
        b, optional
            NaMaster binning object, by default None
        nlb, optional
            Number of multipole bins, by default 20
        remove_monopole, optional
            Whether the monopole should be removed when computing alm's and Cl's
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

        if b is not None:
            self.b = b
        else:
            self._nlb = nlb

        # make sure folder exists
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                # os.mkdir(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)

        print(f"*** Map builder ({self.base_fname}) ***")
        print(f"  - Number of maps = {self.nmaps}")
        print(f"  - Maps saved to", self.save_dir)

    def tqdm(self, it=None, **kwargs):
        """
        Helper function for progress bars
        """
        if it is None:
            return trange(self.nmaps, **kwargs)
        elif isinstance(it, Iterable):
            return tqdm(it, **kwargs)
        elif type(it) is int:
            return [it]
        else:
            raise ValueError

    def get_fname(self, imap=None, sub=None, what=None, format=None):
        """
        Helper function to generate file names
        """
        fname = self.base_fname + f'_nside{self.nside}'

        if what is not None:
            fname += f'_{what}'

        if imap is not None:
            fname += f'_imap{imap}'

        if sub is not None:
            fname += f'_sub{sub}'

        if format is not None:
            fname += f'.{format}' 
        else:
            fname += f'.{self.save_format}'

        if sub is None:
            return os.path.join(self.save_dir, fname)
        else:
            subs_save_dir = os.path.join(self.save_dir, 'subs')
            if not os.path.exists(subs_save_dir):
                os.makedirs(subs_save_dir, exist_ok=True)
            return os.path.join(subs_save_dir, fname)

    def _create_tracers(self):
        raise NotImplementedError

    def get_tracers(self):
        """Create CCL Tracer object for theory computations

        Parameters
        ----------
        Nz, optional
            Number of points to define splines, by default 1000
        """
        if not hasattr(self, 'tracers'):
            self.tracers = self._create_tracers()
        return self.tracers

    def _create_tracers_sacc(self):
        raise NotImplementedError

    def get_tracers_sacc(self):
        """Create SACC tracer object for measured angular power spectra

        Parameters
        ----------
        Nz, optional
            Number of points to define splines, by default 1000
        """
        if not hasattr(self, 'tracers_sacc'):
            self.tracers_sacc = self._create_tracers_sacc()
        return self.tracers_sacc

    def construct_maps(self, imaps=None, **kwargs):
        """Construct maps

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
            
        Returns
        -------
            List of maps in healsparse format
        """
        raise NotImplementedError

    def check_maps(self, imaps=None, read=False, fname=None, check_finite=False):
        """Check map files

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
        read, optional
            Whether to try reading the files, by default False
        fname, optional
            Provide file names to test other files, by default None
        check_finite, optional
            Whether to check if the map values are finite within the mask, by default False

        Returns
        -------
            Result(s) of the tests
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

    def load_maps(self, imaps=None, **kwargs):
        """Loads maps.

        Parameters
        ----------
        imaps, optional
            Map indices, by default None

        Returns
        -------
            List of maps as HealSparseMap objects.
        """
        maps = []
        for imap in self.tqdm(imaps, desc="[load_maps]"):
            fname = self.get_fname(imap, **kwargs)
            maps.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.nside_c))

        if len(maps) == 1:
            return maps[0]
        else:
            return maps

    def plot_maps(self, maps_or_imaps=None, nx=None, log=False, title_prefix=None, cbar=True, save=True, **kwargs):
        """Plots particle maps.

        Parameters
        ----------
        imaps, optional
            List of map indices, by default None
        maps, optional
            List of maps (if `imaps` is not provided), by default None
        nx, optional
            Number of maps to plot per row, by default None
        log, optional
            Plots log10(maps), by default False
        title_prefix, optional
            Title prefix, by default None
        cbar, optional
            Whether to show color bars, by default False
        save, optional
            Whether to save the plot, by default True
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
                    **kwargs)
            else:
                hp.mollview(hp_map, sub=(ny, nx, sub), title=title, cbar=cbar, min=0. if log else None, margins=(0., 0., 0., 0.), nest=True, **kwargs)
            sub += 1

        hp.graticule()

        if save:
            fname = self.get_fname(format='png')
            plt.savefig(fname, bbox_inches='tight', dpi=150, transparent=False)

    def _create_mask(self):
        if self.octant_only:
            self.mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90, nest=True)
            self.ipix = np.where(self.mask)[0]
        else:
            self.mask = np.ones(self.npix)
            self.ipix = np.arange(self.npix)

    def get_mask(self, return_ipix=False):
        """
        Helper function to compute the mask
        """
        if not hasattr(self, 'mask'):
            self._create_mask()
        if return_ipix:
            return self.mask, self.ipix
        else:
            return self.mask

    def get_octant_to_fullsky_ipix(self):
        assert self.octant_only
        if not hasattr(self, 'ipix_oct_fullsky'):
            theta, phi = hp.pix2ang(self.nside, np.arange(self.npix), nest=True)
            self.ipix_oct_fullsky = hp.ang2pix(self.nside, np.where(theta <= np.pi / 2., theta, np.pi - theta), phi % np.pi / 2., nest=True)

        return self.ipix_oct_fullsky

    def get_cls_b(self):
        """
        Workaround to keep MapBuilder serializable...
        """
        if not hasattr(self, 'b'):
            self.b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self._nlb, f_ell='pixwin')
        return self.b

    def get_cls_wsp(self):
        """Get the Namaster workspace object for the shell (same for all shells here).

        Returns
        -------
            Workspace, mask and binning scheme.
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
                mask = self.get_mask()
                dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)])
                self.wsp.compute_coupling_matrix(dummy_f, dummy_f, self.get_cls_b())
                self.wsp.write_to(fname)
            return self.wsp

    def _field_map(self, imap):
        return self.load_maps(imap).generate_healpix_map()

    def get_field_map(self, imap, reorder_n2r=True, replicate_octant=False):
        """"
        Create map ready for a NaMaster field object or healpy map2alm calculations.
        """

        mask, ipix = self.get_mask(return_ipix=True)

        _map = self._field_map(imap) * mask

        if self.remove_monopole:
            _map[ipix] -= np.mean(_map[ipix])

        # Replicate if needed and apply mask
        if replicate_octant:
            ipix_o2f = self.get_octant_to_fullsky_ipix()
            _map = _map[ipix_o2f]

        if reorder_n2r:
            _map = hp.reorder(_map, n2r=True)

        return _map

    def compute_alms(self, imaps=None, save=True, replicate_octant=False, **kwargs):
        """Compute spherical harmonic coefficients

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
        save, optional
            Whether to save the alm's, by default True

        Returns
        -------
            Dictionary of alm's
        """
        assert not (save and replicate_octant)

        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            if save:
                fname = self.get_fname(imap, what='alms', format='fits')
                if os.path.exists(fname):
                    try:
                        alms[imap] = hp.read_alm(fname)
                        continue
                    except:
                        pass

            _map = self.get_field_map(imap, reorder_n2r=True, replicate_octant=replicate_octant)

            alms[imap] = hp.map2alm(_map, **kwargs)

            if save:
                hp.write_alm(fname, alms[imap], overwrite=True)

        return alms

    def compute_maps_from_alms(self, imaps=None, save=True):
        """Compute maps from their alm's

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
        save, optional
            Whether to save the maps, by default True

        Returns
        -------
            List of maps
        """
        _, ipix = self.get_mask(return_ipix=True)

        maps = []
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            alm = hp.read_alm(self.get_fname(imap, what='alms', format='fits'))
            values = hp.reorder(hp.alm2map(alm, nside=self.nside, pol=False), r2n=True)
            tmp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            tmp.update_values_pix(ipix, values[ipix].astype(self.dtype))
            if save:
                fname = self.get_fname(imap)
                tmp.write(fname, clobber=True)
            maps.append(tmp)

        return maps

    def _save_cls(self, ell, cl):
        """
        Helper function to save angular power spectra with SACC
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
        """Compute the (auto- and cross-) angular power spectra of convergence maps

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
        auto_only, optional
            Whether to only compute auto-correlations, by default False
        save, optional
            Whether to save the angular power spectra, by default True
        use_precomputed_alms, optional
            Whether to use precomputed alm files, by default False
        pair_job, optional
            Index in the pair list to compute only one correlation (useful for parallelization), by default None

        Returns
        -------
            Multipole bin centers and dictionary of spectra.
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
            mask = self.get_mask()
            fields = {i: None for i in imaps}
            for i, j in tqdm(pairs, desc="[compute_cls]"):
                if fields[i] is None:
                    fields[i] = nmt.NmtField(mask, [self.get_field_map(i, reorder_n2r=True)])
                if fields[j] is None:
                    fields[j] = nmt.NmtField(mask, [self.get_field_map(j, reorder_n2r=True)])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def load_cls(self, return_dict=False):
        """Load angular power spectra from files

        Parameters
        ----------
        return_dict, optional
            Whether to return as a dictionary or SACC object, by default False

        Returns
        -------
            Angular power spectra in dictionary or SACC format
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
        raise NotImplementedError

    def predict_cls(self, imaps=None, auto_only=False, approximate=False, add_noise=False, error_bars=False, keep_cls=False, stacked=False):
        """Compute theory angular power spectra of convergence maps with CCL

        Parameters
        ----------
        imaps, optional
            Map indices, by default None
        auto_only, optional
            Whether to only compute auto-correlations, by default False
        approximate, optional
            Use approximate binning instead of using bandpower window matrices, by default False

        Returns
        -------
            Multipole bin centers and dictionary of spectra
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

        tracers = self.get_tracers()
        tracers_ = {imap: tracers[imap] for imap in imaps}

        b = self.get_cls_b()

        ell = b.get_effective_ells()
        nell = b.get_nell_list()
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
                cl_raw = angular_cl(self.cosmo, tracers_[i], tracers_[j], range(3 * self.nside))
                cl[i, j] = np.dot(bpws, hp.pixwin(self.nside)**2 * cl_raw)
                if add_noise and i == j:
                    cl[i, j] += np.dot(bpws, hp.pixwin(self.nside)**2 * self._noise_cls(i))

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
                for i, j in pairs:
                    for k, l in pairs:
                        dcl_full[i, j, k, l] = (Dl[i, l] * Dl[j, k] + Dl[i, k] * Dl[j, l]) / ((2 * ell + 1.) * fsky * nell)

                # Stack Cl's and covariance blocks
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
        """Plot predicted and/or measured angular power spectra of maps

        Parameters
        ----------
        auto_only, optional
            Whether to only plot auto-correlations, by default False
        logy, optional
            Use log scale on y-axis, by default False
        theory_only, optional
            Plot predicted spectra only, by default False
        data_only, optional
            Plot measured spectra only, by default False
        ratio_data_theory, optional
            Plot ratio of measured and predicted spectra, by default False
        save, optional
            Whether to save the plot, by default True
        sharey, optional
            Share y-axis scaling (can be "row" or True), by default None
        """
        n = self.nmaps
        m = 1 if auto_only else n
        sharey = sharey or False  #False if auto_only else 'row'
        if figaxes is None:
            fig, axes = plt.subplots(m, n, figsize=(4 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)
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
