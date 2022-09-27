import os

import healpy as hp
import healsparse
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
from scipy.integrate import quad
from tqdm.auto import tqdm, trange


class LensingBuilder:

    def __init__(self, shell_builder, nzs, save_dir=None):
        """LensingBuilder is a generic class to build convergence maps from particle shells.

        Parameters
        ----------
        shell_builder
            ShellBuilder object.
        nzs
            List of callables representing redshift distributions.
        """
        self.shell_builder = shell_builder
        if save_dir is None:
            self.save_dir = shell_builder.save_dir
        else:
            self.save_dir = save_dir
        self.reader = shell_builder.reader
        self.cosmo = shell_builder.cosmo
        self.nshells = shell_builder.nshells
        self.nzs = nzs
        self.nkappas = len(nzs)

    def get_tracers(self):
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, com_shell_bins) - 1.

        zz = np.linspace(0, z_shell_bins[-1], 1000)

        return [pyccl.WeakLensingTracer(self.cosmo, (zz, nz(zz))) for nz in self.nzs]

    def construct_kappas(self):
        raise NotImplementedError

    def get_fname(self, ikappa, what='kappa'):
        return os.path.join(
            self.save_dir,
            f'{what}_{self.reader.kind}_nz{ikappa}_nside{self.shell_builder.nside}_seed{self.shell_builder.seed}.' +
            self.shell_builder.save_format)

    def plot_kappas(self, kappas, nx=None, title_prefix=None, cbar=False, **kwargs):
        """Plots convergence maps.

        Parameters
        ----------
        kappas
            List of convergence maps.
        nx, optional
            Number of maps to plot per row, by default None
        title_prefix, optional
            Title prefix, by default None
        cbar, optional
            Whether to show color bars, by default False
        """
        nkappas = len(kappas)
        if nx is None:
            nx = nkappas
        ny = int(np.ceil(nkappas / nx))

        plt.figure(figsize=(4 * nx, 4 * ny))

        for ikappa in range(nkappas):
            hp_map = kappas[ikappa].generate_healpix_map()
            title = title_prefix + f" {ikappa}" if title_prefix is not None else None
            if self.shell_builder.octant_only:
                hp.orthview(hp_map,
                            rot=(35, 20),
                            half_sky=True,
                            sub=(ny, nx, ikappa + 1),
                            title=title,
                            cbar=cbar,
                            **kwargs)
            else:
                hp.mollview(hp_map, sub=(ny, nx, ikappa + 1), title=title, cbar=cbar, **kwargs)
        hp.graticule(verbose=False)

    def load_kappas(self, ikappas=None):
        """Loads convergence maps.

        Parameters
        ----------
        ikappas, optional
            Indices of maps to load, by default None

        Returns
        -------
            List of convergence maps
        """
        if ikappas is None:
            ikappas = range(self.nkappas)

        kappas = []
        for ikappa in tqdm(ikappas, desc='[load_kappas]'):
            fname = self.get_fname(ikappa)
            kappas.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.shell_builder.nside_c))
        return kappas

    def compute_kappas_cls(self, kappas=None, ikappas=None, b=None, nlb=20):
        """Computes the (auto- and cross-) angular power spectra of convergence maps.

        Parameters
        ----------
        kappas, optional
            List of convergence maps, by default None
        ikappas, optional
            Indices of convergence maps, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

        Returns
        -------
            Multipole bin centers and dictionary of spectra.
        """
        wsp, mask, b = self.shell_builder.get_cls_wsp(b=b, nlb=nlb)

        if kappas is None:
            kappas = self.load_kappas(ikappas)
        if ikappas is None:
            ikappas = range(self.nkappas)
        kappas_ = {i: k for i, k in zip(ikappas, kappas)}

        pairs = [(i1, i2) for i, i1 in enumerate(ikappas) for i2 in ikappas[:i + 1]]
        fields = {i: None for i in ikappas}
        cls_out = {}

        for i, j in tqdm(pairs, desc="[compute_kappas_cls]"):
            if fields[i] is None:
                fields[i] = nmt.NmtField(mask, [kappas_[i].generate_healpix_map()])
            if fields[j] is None:
                fields[j] = nmt.NmtField(mask, [kappas_[j].generate_healpix_map()])

            cls_out[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]

        return b.get_effective_ells(), cls_out

    def predict_kappas_cls(self, ikappas=None, b=None, nlb=20):
        """Computes theory angular power spectra of convergence maps with CCL.

        Parameters
        ----------
        ikappas, optional
            Indices of convergence maps, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

        Returns
        -------
            Multipole bin centers and dictionary of spectra.
        """
        if ikappas is None:
            ikappas = range(self.nkappas)

        _, _, b = self.shell_builder.get_cls_wsp(b=b, nlb=nlb)

        pairs = [(i1, i2) for i, i1 in enumerate(ikappas) for i2 in ikappas[:i + 1]]

        tracers = self.get_tracers()
        tracers_ = {i: t for i, t in zip(ikappas, tracers)}

        # TODO : use wsp to get exact convolution, if actually useful for validation

        cls_out = {}
        for i, j in tqdm(pairs, desc="[compute_kappas_cls]"):
            cls_out[i, j] = pyccl.angular_cl(self.cosmo, tracers_[i], tracers_[j], b.get_effective_ells())

        return b.get_effective_ells(), cls_out


class BornLensingBuilder(LensingBuilder):

    def __init__(self, shell_builder, nzs, save_dir=None):
        super().__init__(shell_builder, nzs, save_dir=save_dir)

    def construct_kappas(self, dtype=np.float32):
        """Constructs and saves convergence maps from particle shells in the Born approximation, ie by summing the particle overdensity maps with appropriate weights computed with CCL.

        Parameters
        ----------
        dtype, optional
            Dtype of the convergence map, by default np.float32

        Returns
        -------
            List of convergence maps and weights used in the summation.
        """
        # prefactor = 3.0 * pyccl.omega_x(self.cosmo, 1., 'matter') / 2.0
        npix = hp.nside2npix(self.shell_builder.nside)
        box_density = [self.shell_builder.get_box_comoving_density(ishell) for ishell in range(self.nshells)]

        kappas = []
        weights = []

        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, com_shell_bins) - 1.
        zz = np.linspace(0, z_shell_bins[-1], 1000)

        shells = self.shell_builder.load_shells()

        for iz, nz in enumerate(tqdm(self.nzs, desc="[construct_kappas]")):
            tracer = pyccl.WeakLensingTracer(self.cosmo, (zz, nz(zz)))

            kappa = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c,
                                                        self.shell_builder.nside,
                                                        dtype=dtype)

            weight = []
            for i, shell in enumerate(range(self.nshells)):
                shell = shells[i]
                chi_max = com_shell_bins[i + 1]
                chi_min = com_shell_bins[i]

                w = quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)[0]

                pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

                tmp = shell.astype(dtype) / dtype(box_density[i] * pixel_volume) - 1.
                tmp *= dtype(w)

                kappa = healsparse.operations.sum_union([tmp, kappa])

                weight.append(w)

            kappas.append(kappa)
            weights.append(weight)

            fname = self.get_fname(iz)
            kappa.write(fname, clobber=True)

        return kappas, np.array(weights)
    
    
    def get_shear_maps(self, kappas, return_alms=False, dtype=np.float32):
        ell = np.arange(3*self.shell_builder.nside)
        fl = np.zeros(len(ell))
        fl[1:] = - np.sqrt((ell[1:]+2.)*(ell[1:]-1.)/(ell[1:]*(ell[1:]+1.)))
        
        pixels = np.where(self.shell_builder.get_mask())[0]
        
        gammas = []
        alms = []
        for iz, kappa in enumerate(tqdm(kappas)):
            # Get harmonic coefficients of convergence map
            klm = hp.map2alm(kappa.generate_healpix_map(), pol=False)
            
            # Convert to shear
            glm_E = hp.almxfl(klm, fl)
            z = np.zeros_like(glm_E)
            
            # Generate shear map
            _, g1, g2 = hp.alm2map([z, glm_E, z], nside=self.shell_builder.nside, pol=True)
            
            gamma1 = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype)
            gamma2 = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype)
            
            gamma1.update_values_pix(pixels, g1[pixels].astype(dtype))
            gamma2.update_values_pix(pixels, g2[pixels].astype(dtype))
            
            gamma1.write(self.get_fname(iz, what='gamma1'), clobber=True)
            gamma2.write(self.get_fname(iz, what='gamma2'), clobber=True)

            gammas.append([gamma1, gamma2])
            if return_alms:
                alms.append(klm)
        
        if return_alms:
            return gammas, alms
        else:
            return gammas
