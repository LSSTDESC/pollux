import os

import healpy as hp
import healsparse
import numpy as np
import pyccl
import sacc
from tqdm.auto import trange

from .map_builder import MapBuilder
from .utils import quad


class MyCMBLensingTracer(pyccl.Tracer):

    def __init__(self, cosmo, z_source, n_samples=100):
        super().__init__()

        # we need the distance functions at the C layer
        cosmo.compute_distances()
        kernel = pyccl.get_kappa_kernel(cosmo, z_source=z_source, n_samples=n_samples)
        if (cosmo['sigma_0'] == 0):
            self.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
        else:
            self._MG_add_tracer(cosmo, kernel, z_source, der_bessel=-1, der_angles=1)

        self.z_source = z_source


class LensingBuilder(MapBuilder):

    def __init__(self, shell_builder, nzs, kappa_dir=None, dtype=np.float32, sigma_e=0.3, nbar=5., **kwargs):
        """`LensingBuilder` is a generic class to build convergence maps from particle shells.

        Parameters
        ----------
        shell_builder
            Shell builder object
        nzs
            List of callables or arrays (format (z, n(z))) representing redshift distributions.
        kappa_dir, optional
            Folder where to save maps, by default None
        dtype, optional
            dtype of maps, by default np.float32
        sigma_e, optional
            Ellipticity dispersion (for covariance calculation)
        """
        self.shell_builder = shell_builder
        if kappa_dir is None:
            save_dir = shell_builder.save_dir
        else:
            save_dir = os.path.join(shell_builder.save_dir, kappa_dir)
        self.reader = shell_builder.reader
        self.cosmo = shell_builder.cosmo
        self.nshells = shell_builder.nshells
        self.nzs = nzs

        self.sigma_e = sigma_e
        self.nbar = nbar * (60 * 180 / np.pi)**2  # gal/arcmin2 to gal/str

        if nzs == 'zsnapshots':
            self.nkappas = self.shell_builder.nshells
        else:
            self.nkappas = len(nzs)
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, 'nzs.npy')
            if not os.path.exists(fname):
                np.save(fname, nzs)

        base_fname = f'kappa_{shell_builder.reader.kind}_seed{shell_builder.seed}'

        super().__init__(
            nside=shell_builder.nside, dtype=dtype, save_dir=save_dir, base_fname=base_fname, nmaps=self.nkappas, sacc_type="galaxy_convergence_cl", **kwargs)

    def _create_tracers_ccl(self, Nz=1000):
        z_shell_bins = self.shell_builder.shell_edges_redshift()

        tracers = []
        if self.nzs == 'zsnapshots':
            for z_source in z_shell_bins[1:]:
                tracers.append(MyCMBLensingTracer(self.cosmo, z_source))
        else:
            for nz in self.nzs:
                if callable(nz):
                    zz = np.linspace(0, z_shell_bins[-1], Nz)
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, dndz=(zz, nz(zz))))
                else:
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, dndz=(nz[0], nz[1])))

        return tracers

    def _create_tracers_sacc(self, Nz=1000):
        z_shell_bins = self.shell_builder.shell_edges_redshift()

        tracers = []
        if self.nzs == 'zsnapshots':
            for i, _ in enumerate(z_shell_bins[1:]):
                tracers.append(sacc.tracers.MiscTracer(f"kappa_{i}"))
        else:
            for i, nz in enumerate(self.nzs):
                if callable(nz):
                    zz = np.linspace(0, z_shell_bins[-1], Nz)
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", zz, nz(zz)))
                else:
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", nz[0], nz[1]))

        return tracers

    def compute_shear_maps(self, ikappas=None, lmax=None, replicate_octant=False):
        """Compute shear maps from convergence maps

        Parameters
        ----------
        ikappas, optional
            Map indices, by default None
        lmax, optional
            Maximum multipole, by default None
        replicate_octant, optional
            If True, alm's are recomputed from octant maps replicated to cover the full sky (irrelevant if `octant_only` is False), by default False  

        Returns
        -------
            Shear maps and alm's if requested
        """
        if lmax is None:
            lmax = 3 * self.shell_builder.nside - 1

        ell = np.arange(3 * self.shell_builder.nside)
        fl = np.zeros(len(ell))
        fl[1:] = -np.sqrt((ell[1:] + 2.) * (ell[1:] - 1.) / (ell[1:] * (ell[1:] + 1.)))

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        gammas = []
        alms = []
        for ikappa in self.tqdm(ikappas, desc="[compute_shear_maps]"):
            if self.check_maps(read=True, fname=self.get_fname(ikappa, what='gamma1')) and self.check_maps(read=True, fname=self.get_fname(ikappa,
                                                                                                                                           what='gamma2')):
                continue

            # Get harmonic coefficients of convergence map
            klm = self.compute_alms(ikappa, save=not (replicate_octant), replicate_octant=replicate_octant, pol=False, lmax=lmax)[ikappa]

            # Convert to shear
            glm_E = hp.almxfl(klm, fl)

            # Generate shear map (ring-ordered)
            g1, g2 = hp.sphtfunc.alm2map_spin([glm_E, np.zeros_like(glm_E)], nside=self.nside, spin=2, lmax=lmax)

            gamma1 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            gamma2 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            gamma1.update_values_pix(ipix, g1[ipix].astype(self.dtype), nest=False)
            gamma2.update_values_pix(ipix, g2[ipix].astype(self.dtype), nest=False)

            gamma1.write(self.get_fname(ikappa, what='gamma1'), clobber=True)
            gamma2.write(self.get_fname(ikappa, what='gamma2'), clobber=True)

            gammas.append([gamma1, gamma2])

        return gammas

    def _noise_cls(self, imap):
        nbar = self.nbar
        return self.sigma_e**2 / nbar


class BornLensingBuilder(LensingBuilder):

    def __init__(self, shell_builder, nzs, kappa_dir=None, dtype=np.float32, multiple_planes=False, nbar_from_maps=False, **kwargs):
        """Lensing builder object in the Born approximation

        Parameters
        ----------
        shell_builder
            Light-cone builder object
        nzs
            List of callables representing redshift distributions.
        kappa_dir, optional
            Folder where to save maps, by default None
        dtype, optional
            dtype of maps, by default np.float32
        multiple_planes, optional
            Compute weights in the multiple lens planes approximation, instead of using Limber integrals, by default False
        nbar_from_maps, optional
            Compute the mean density from the maps themselves (less accurate as it removes the monopole), by default False
        """
        super().__init__(shell_builder, nzs, kappa_dir=kappa_dir, dtype=dtype, **kwargs)

        self.multiple_planes = multiple_planes
        self.nbar_from_maps = nbar_from_maps

        if multiple_planes:
            mpart = self.reader.mass_particles

            area = hp.nside2pixarea(self.nside)

            self.mpart_over_area = mpart / area

            zmax = self.shell_builder.shell_edges_redshift()[-1]
            self.dndz_norm = [quad(lambda z: t.get_dndz(z), 0., zmax) if isinstance(t, pyccl.tracers.NzTracer) else np.nan for t in self.get_tracers_ccl()]
            self.zlens = [self.shell_builder.mean_z(ishell) for ishell in range(self.nshells)]

    def compute_weight(self, ishell, ikappa):
        """Compute lensing weights to form convergence maps from particle shells

        Parameters
        ----------
        ishell
            Shell index
        ikappa
            Convergence map index
        """
        tracer = self.get_tracers_ccl()[ikappa]

        if self.multiple_planes:
            zmax = self.shell_builder.shell_edges_redshift()[-1]
            zlens = self.zlens[ishell]

            if zlens is None:
                zlens = self.shell_builder.mean_z(ishell)

            d_A_lens = pyccl.angular_diameter_distance(self.cosmo, 1. / (1. + zlens))  # needed to obtain surface mass density in 1/Mpc^2

            if isinstance(tracer, pyccl.tracers.NzTracer):

                def integrand(zsource, zlens, tracer):
                    sigma_critical = pyccl.background.sigma_critical(self.cosmo, a_lens=1. / (1. + zlens), a_source=1. / (1. + zsource))  # in Msun / Mpc^2
                    return tracer.get_dndz(zsource) / sigma_critical

                w = quad(integrand, zlens, zmax, args=(zlens, tracer)) / self.dndz_norm[ikappa]

            elif isinstance(tracer, pyccl.tracers.CMBLensingTracer):
                zsource = tracer.z_source
                if zsource > zlens:
                    w = 1. / pyccl.background.sigma_critical(self.cosmo, 1. / (1. + zlens), 1. / (1. + zsource))  # in Msun / Mpc^2
                else:
                    w = 0.
            else:
                raise ValueError

            return w * self.mpart_over_area / d_A_lens**2
        else:
            chi_min, chi_max = self.shell_builder.shell_edges_comoving_distance()[ishell:ishell + 2]
            return quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)

    def construct_maps(self, ikappas=None):
        if ikappas is None:
            ikappas = range(self.nkappas)
        elif type(ikappas) is int:
            ikappas = [ikappas]
        else:
            pass

        _ikappas = []
        for ikappa in ikappas:
            fname = self.get_fname(ikappa, what='alms', format='fits')
            if not self.check_maps(ikappa, read=True):
                _ikappas.append(ikappa)
        if len(_ikappas) == 0:
            return
        else:
            _ikappas = ikappas

        kappas = {ikappa: healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype) for ikappa in _ikappas}

        for ishell in trange(self.nshells, desc="[construct_maps]"):
            shell = (self.shell_builder.load_maps(ishell)).astype(self.dtype)
            if self.shell_builder.save_density:
                density = shell
            else:
                if self.nbar_from_maps:
                    s = shell[:]
                    nbar = np.mean(s[s > 0.])
                else:
                    nbar = self.shell_builder.pixel_density(ishell)
                nbar = self.dtype(nbar)
                if self.multiple_planes:
                     density = shell - nbar 
                else:
                    density = shell/nbar -1. 

            for ikappa, kappa in kappas.items():
                w = self.compute_weight(ishell, ikappa)
                # # only one multiplication and one addition
                # if self.multiple_planes:
                #     tmp = (shell - self.dtype(nbar)) * self.dtype(w)
                # else:
                #     tmp = (shell * self.dtype(w / nbar)) - self.dtype(w)
                # kappas[ikappa] = healsparse.operations.sum_union([tmp, kappa])
                kappas[ikappa] = healsparse.operations.sum_union([density*w, kappa])

        for ikappa, kappa in kappas.items():
            fname = self.get_fname(ikappa)
            kappa.write(fname, clobber=True)

        return kappas
