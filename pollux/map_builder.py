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
            nlb=20
        ):
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

        if b is not None:
            self.b = b
        self._nlb = nlb

        # if b is None:
        #     self.b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, nlb, f_ell='pixwin')

        # make sure folder exists
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                # os.mkdir(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)

    def tqdm(self, it, **kwargs):
        if it is None:
            return trange(self.nmaps, **kwargs)
        elif isinstance(it, Iterable):
            return tqdm(it, **kwargs)
        elif type(it) is int:
            return [it]
        else:
            raise ValueError

    def get_fname(self, imap=None, sub=None, what=None, force_fits=False):
        fname = self.base_fname + f'_nside{self.nside}'

        if what is not None:
            fname += f'_{what}'

        if imap is not None:
            fname += f'_imap{imap}'

        if sub is not None:
            fname += f'_sub{sub}'

        if force_fits:
            fname += '.fits'
        else:
            fname += f'.{self.save_format}'

        return os.path.join(self.save_dir, fname)

    def get_tracers(self):
        raise NotImplementedError

    def get_tracers_sacc(self):
        raise NotImplementedError

    def construct_maps(self, imaps=None, skip_existing=False, check_read_existing=True, **kwargs):
        raise NotImplementedError

    def check_maps(self, imaps=None, read=False, fname=None, check_finite=False):
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

    def load_maps(self, imaps=None):
        """Loads maps.

        Parameters
        ----------
        imaps, optional
            Indices of maps, by default None

        Returns
        -------
            List of maps as HealSparseMap objects.
        """
        maps = []
        for imap in self.tqdm(imaps, desc="[load_maps]"):
            fname = self.get_fname(imap)
            maps.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.nside_c))

        if len(maps) == 1:
            return maps[0]
        else:
            return maps

    def plot_maps(self, maps, nx=None, log=False, title_prefix=None, cbar=False, **kwargs):
        """Plots particle maps.

        Parameters
        ----------
        maps
            List of maps.
        nx, optional
            Number of maps to plot per row, by default None
        log, optional
            Plots log10(maps), by default True
        title_prefix, optional
            Title prefix, by default None
        cbar, optional
            Whether to show color bars, by default False
        """
        nmaps = len(maps)
        if nx is None:
            nx = nmaps
        ny = int(np.ceil(nmaps / nx))

        plt.figure(figsize=(4 * nx, 4 * ny))

        for imap in range(nmaps):
            hp_map = maps[imap]
            if isinstance(hp_map, healsparse.HealSparseMap):
                hp_map = hp_map.generate_healpix_map()
            if log and np.any(hp_map > 0.):
                hp_map = np.log10(hp_map)
            title = title_prefix + f" {imap}" if title_prefix is not None else None
            if self.octant_only:
                hp.orthview(
                    hp_map, rot=(35, 20), half_sky=True, sub=(ny, nx, imap + 1), title=title, cbar=cbar, **kwargs
                    )
            else:
                hp.mollview(hp_map, sub=(ny, nx, imap + 1), title=title, cbar=cbar, **kwargs)
        hp.graticule(verbose=False)

    def get_mask(self, return_ipix=False):
        if not hasattr(self, 'mask'):
            if self.octant_only:
                self.mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90)
                self.ipix = np.where(self.mask)[0]
            else:
                self.mask = np.ones(hp.nside2npix(self.nside))
                self.ipix = np.arange(len(self.mask))
        if return_ipix:
            return self.mask, self.ipix
        else:
            return self.mask

    def get_cls_b(self):
        # Workaround to keep MapBuilder serializable...
        if not hasattr(self, 'b'):
            self.b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, self._nlb, f_ell='pixwin')
        return self.b

    def get_cls_wsp(self):
        """Gets the Namaster workspace object for the shell (same for all shells here).

        Parameters
        ----------
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

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

    def get_field_map(self, imap):
        raise NotImplementedError

    def compute_alms(self, imaps=None, save=True, skip_existing=False, check_read_existing=True):
        mask = self.get_mask()

        alms = {}
        for imap in self.tqdm(imaps, desc="[compute_alms]"):
            if save:
                fname = self.get_fname(imap, what='alms', force_fits=True)
                if skip_existing and os.path.exists(fname):
                    if check_read_existing:
                        try:
                            hp.read_alm(fname)
                            continue
                        except:
                            pass
                    else:
                        continue
            alms[imap] = hp.map2alm(self.get_field_map(imap) * mask, pol=False)
            if save:
                hp.write_alm(fname, alms[imap], overwrite=True)

        return alms

    def compute_maps_from_alms(self, imaps=None, save=True):
        _, ipix = self.get_mask(return_ipix=True)

        maps = []
        for imap in self.tqdm(imaps, desc="[compute_maps_from_alms]"):
            alm = hp.read_alm(self.get_fname(imap, what='alms', force_fits=True))
            values = hp.alm2map(alm, nside=self.nside, pol=False)
            tmp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            tmp.update_values_pix(ipix, values[ipix].astype(self.dtype))
            if save:
                fname = self.get_fname(imap)
                tmp.write(fname, clobber=True)
            maps.append(tmp)

        return maps

    def _save_cls(self, ell, cl):
        s = sacc.Sacc()
        tracers = self.get_tracers_sacc()
        for t in tracers:
            s.add_tracer_object(t)
        for i, j in cl.keys():
            for _ell, _cl in zip(ell, cl[i, j]):
                s.add_data_point(self.sacc_type, [tracers[i].name, tracers[j].name], _cl, ell=_ell)
        fname = self.get_fname(what='cls', force_fits=True)
        s.save_fits(fname, overwrite=True)

    def compute_cls(self, imaps=None, auto_only=False, save=True, use_precomputed_alms=False, pair_job=None):
        """Computes the (auto- and cross-) angular power spectra of convergence maps.

        Parameters
        ----------
        maps, optional
            List of convergence maps, by default None
        imaps, optional
            Indices of convergence maps, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

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
                    almi = hp.read_alm(self.get_fname(i, what='alms', force_fits=True))
                if i == j:
                    almj = almi
                else:
                    if not j == previous_pair[1]:
                        almj = hp.read_alm(self.get_fname(j, what='alms', force_fits=True))
                pcl = hp.alm2cl(almi, almj)
                cl[i, j] = wsp.decouple_cell([pcl])[0]
                previous_pair = (i, j)
        else:
            mask = self.get_mask()
            fields = {i: None for i in imaps}
            for i, j in tqdm(pairs, desc="[compute_cls]"):
                if fields[i] is None:
                    fields[i] = nmt.NmtField(mask, [self.get_field_map(i)])
                if fields[j] is None:
                    fields[j] = nmt.NmtField(mask, [self.get_field_map(j)])
                cl[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        ell = b.get_effective_ells()

        if save:
            self._save_cls(ell, cl)

        return ell, cl

    def load_cls(self):
        fname = self.get_fname(what='cls', force_fits=True)
        return sacc.Sacc.load_fits(fname)

    def predict_cls(self, imaps=None, auto_only=False, approximate=False):
        """Computes theory angular power spectra of convergence maps with CCL.

        Parameters
        ----------
        imaps, optional
            Indices of convergence maps, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

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
            pairs = [(i1, i2) for i, i2 in enumerate(imaps) for i1 in imaps[i:]]

        tracers = self.get_tracers()
        tracers_ = {i: t for i, t in zip(imaps, tracers)}

        b = self.get_cls_b()

        if not approximate:
            wsp = self.get_cls_wsp()
            bpws = wsp.get_bandpower_windows()[0, :, 0, :]

        cl = {}
        for i, j in tqdm(pairs, desc="[predict_cls]"):
            if approximate:
                cl[i, j] = pyccl.angular_cl(self.cosmo, tracers_[i], tracers_[j], b.get_effective_ells())
            else:
                cl_raw = pyccl.angular_cl(self.cosmo, tracers_[i], tracers_[j], range(3 * self.nside))
                cl[i, j] = np.dot(bpws, hp.pixwin(self.nside)**2 * cl_raw)

        return b.get_effective_ells(), cl

    def triangle_plot_cls(
            self, auto_only=False, logy=False, theory_only=False, data_only=False, save=True, sharey=None
        ):
        n = self.nmaps
        m = 1 if auto_only else n
        sharey = sharey or False  #False if auto_only else 'row'
        fig, axes = plt.subplots(m, n, figsize=(4 * n, 3 * m), sharex=True, sharey=sharey, squeeze=False)

        ell, cl_th = self.predict_cls(auto_only=auto_only)
        s = self.load_cls()

        tracers = list(s.tracers.keys())
        tc = s.get_tracer_combinations(self.sacc_type)

        for i in range(m):
            for j in range(n):
                if auto_only:
                    ax = axes[0, j]
                    i = j
                else:
                    ax = axes[i, j]
                ax_used = False
                if (i, j) in cl_th.keys() and not data_only:
                    ax.plot(ell, cl_th[i, j], c='orange')
                    ax_used = True
                if (tracers[i], tracers[j]) in tc and not theory_only:
                    ax.plot(
                        ell,
                        s.get_ell_cl(self.sacc_type, tracers[i], tracers[j])[1],
                        c='dodgerblue',
                        ls=' ',
                        marker='o',
                        markersize=5
                        )
                    ax_used = True
                if ax_used:
                    ax.set_xlabel('Multipole $\ell$')
                    ax.set_ylabel('$C_\ell$')
                    ax.set_xscale('log')
                    if logy:
                        ax.set_yscale('log')
                else:
                    ax.axis('off')

        fig.tight_layout()

        if save:
            fname = self.get_fname(what='cls', force_fits=True).replace('.fits', '.png')
            fig.savefig(fname, bbox_inches='tight')

        plt.show()
