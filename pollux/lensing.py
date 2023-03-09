import os

import healpy as hp
import healsparse
import numpy as np
import pyccl
import sacc
from tqdm.auto import trange

from .map_builder import MapBuilder
from .utils import quad


class MyCMBLensingTracer(pyccl.CMBLensingTracer):

    def __init__(self, cosmo, z_source, n_samples=100):
        super().__init__(cosmo, z_source, n_samples=100)
        self.z_source = z_source


class LensingBuilder(MapBuilder):

    def __init__(self, shell_builder, nzs, kappa_dir=None, dtype=np.float32, **kwargs):
        """LensingBuilder is a generic class to build convergence maps from particle shells.

        Parameters
        ----------
        shell_builder
            ShellBuilder object.
        nzs
            List of callables representing redshift distributions.
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

        # self.tracers = self.get_tracers()
        # self.nkappas = len(self.tracers)
        if nzs == 'zsnapshots':
            self.nkappas = self.shell_builder.nshells
        else:
            self.nkappas = len(nzs)

        base_fname = f'kappa_{shell_builder.reader.kind}_seed{shell_builder.seed}'

        super().__init__(
            nside=shell_builder.nside,
            dtype=dtype,
            save_dir=save_dir,
            base_fname=base_fname,
            nmaps=self.nkappas,
            sacc_type="galaxy_convergence_cl",
            **kwargs)

    def get_tracers(self, Nz=1000):
        z_shell_bins = self.reader.get_z_shell_bins()

        tracers = []
        if self.nzs == 'zsnapshots':
            for z_source in z_shell_bins[1:]:
                tracers.append(MyCMBLensingTracer(self.cosmo, z_source))
        else:
            for nz in self.nzs:
                if callable(nz):
                    zz = np.linspace(0, z_shell_bins[-1], Nz)
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, (zz, nz(zz))))
                else:
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, (nz[0], nz[1])))

        return tracers

    def get_tracers_sacc(self, Nz=1000):
        z_shell_bins = self.reader.get_z_shell_bins()

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

    def get_field_map(self, imap):
        return self.load_maps(imap).generate_healpix_map()

    def compute_shear_maps(
            self,
            ikappas=None,
            return_alms=False,
            lmax=None,
            skip_existing=False,
            check_read_existing=True,
            use_precomputed_alms=False):
        if lmax is None:
            lmax = 3 * self.shell_builder.nside - 1

        ell = np.arange(3 * self.shell_builder.nside)
        fl = np.zeros(len(ell))
        fl[1:] = -np.sqrt((ell[1:] + 2.) * (ell[1:] - 1.) / (ell[1:] * (ell[1:] + 1.)))

        pixels = np.where(self.shell_builder.get_mask())[0]

        gammas = []
        alms = []
        for ikappa in self.tqdm(ikappas, desc="[compute_shear_maps]"):
            if skip_existing:
                if self.check_maps(read=check_read_existing, fname=self.get_fname(
                        ikappa, what='gamma1')) and self.check_maps(read=check_read_existing, fname=self.get_fname(
                            ikappa, what='gamma2')):
                    continue

            # Get harmonic coefficients of convergence map
            if use_precomputed_alms:
                klm = hp.read_alm(self.get_fname(ikappa, what='alms', force_fits=True))
            else:
                kappa = self.get_field_map(ikappa)
                klm = hp.map2alm(kappa, pol=False, lmax=lmax)

            # Convert to shear
            glm_E = hp.almxfl(klm, fl)

            # Generate shear map
            g1, g2 = hp.sphtfunc.alm2map_spin([glm_E, np.zeros_like(glm_E)], nside=self.nside, spin=2, lmax=lmax)

            gamma1 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            gamma2 = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            gamma1.update_values_pix(pixels, g1[pixels].astype(self.dtype))
            gamma2.update_values_pix(pixels, g2[pixels].astype(self.dtype))

            gamma1.write(self.get_fname(ikappa, what='gamma1'), clobber=True)
            gamma2.write(self.get_fname(ikappa, what='gamma2'), clobber=True)

            gammas.append([gamma1, gamma2])
            if return_alms:
                alms.append(klm)

        if return_alms:
            return gammas, alms
        else:
            return gammas


class BornLensingBuilder(LensingBuilder):

    def __init__(
            self,
            shell_builder,
            nzs,
            kappa_dir=None,
            dtype=np.float32,
            multiple_planes=False,
            nbar_from_maps=False,
            **kwargs):
        super().__init__(shell_builder, nzs, kappa_dir=kappa_dir, dtype=dtype, **kwargs)

        self.multiple_planes = multiple_planes
        self.nbar_from_maps = nbar_from_maps

        if multiple_planes:
            rho_m = pyccl.background.rho_x(self.cosmo, 1., 'matter', is_comoving=True)  #  in units of Msun / Mpc^3
            vol = self.reader.get_boxsize()**3  # in Mpc^3
            mpart = rho_m * vol / self.reader.nparticles  # in Msun

            area = hp.nside2pixarea(self.nside)

            self.mpart_over_area = mpart / area

            zmax = self.reader.zmax
            self.dndz_norm = [
                quad(lambda z: t.get_dndz(z), 0., zmax) if isinstance(t, pyccl.tracers.WeakLensingTracer) else np.nan
                for t in self.get_tracers()
            ]

    def compute_weight(self, ishell, tracer, dndz_norm=None, zlens=None):
        if self.multiple_planes:
            zmax = self.reader.zmax

            if zlens is None:
                zlens = self.shell_builder.get_mean_z(ishell)

            d_A_lens = pyccl.angular_diameter_distance(
                self.cosmo, 1. / (1. + zlens))  # needed to obtain surface mass density in 1/Mpc^2

            if isinstance(tracer, pyccl.tracers.WeakLensingTracer):
                if dndz_norm is None:
                    dndz_norm = quad(lambda z: tracer.get_dndz(z), 0., zmax)

                def integrand(zsource, zlens, tracer):
                    sigma_critical = pyccl.background.sigma_critical(
                        self.cosmo, 1. / (1. + zlens), 1. / (1. + zsource))  # in Msun / Mpc^2
                    return tracer.get_dndz(zsource) / sigma_critical

                w = quad(integrand, zlens, zmax, args=(zlens, tracer)) / dndz_norm

            elif isinstance(tracer, pyccl.tracers.CMBLensingTracer):
                zsource = tracer.z_source
                if zsource > zlens:
                    w = 1. / pyccl.background.sigma_critical(
                        self.cosmo, 1. / (1. + zlens), 1. / (1. + zsource))  # in Msun / Mpc^2
                else:
                    w = 0.
            else:
                raise ValueError

            return w * self.mpart_over_area / d_A_lens**2
        else:
            chi_min, chi_max = self.reader.get_com_shell_bins()[ishell:ishell + 2]
            return quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)

    def construct_maps(self, ikappas=None, skip_existing=False, check_read_existing=True):
        """Constructs and saves convergence maps from particle shells in the Born approximation, ie by summing the particle overdensity maps with appropriate weights computed with CCL.

        Parameters
        ----------
        dtype, optional
            Dtype of the convergence map, by default np.float32

        Returns
        -------
            List of convergence maps and weights used in the summation.
        """

        if ikappas is None:
            ikappas = range(self.nkappas)
        elif type(ikappas) is int:
            ikappas = [ikappas]
        else:
            pass

        if skip_existing:
            _ikappas = []
            for ikappa in ikappas:
                fname = self.get_fname(ikappa, what='alms', force_fits=True)
                if not self.check_maps(ikappa, read=check_read_existing):
                    _ikappas.append(ikappa)
            if len(_ikappas) == 0:
                return
        else:
            _ikappas = ikappas

        kappas = {
            ikappa: healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            for ikappa in _ikappas
        }

        tracers = self.get_tracers()

        for ishell in trange(self.nshells, desc="[construct_maps]"):
            shell = (self.shell_builder.load_maps(ishell)).astype(self.dtype)
            if self.nbar_from_maps:
                s = shell[:]
                nbar = np.mean(s[s > 0.])
            else:
                nbar = self.shell_builder.get_pixel_density(ishell)

            zlens = self.shell_builder.get_mean_z(ishell)

            for ikappa, kappa in kappas.items():
                w = self.compute_weight(ishell, tracers[ikappa], dndz_norm=self.dndz_norm[ikappa], zlens=zlens)
                # only one multiplication and one addition
                if self.multiple_planes:
                    tmp = (shell - self.dtype(nbar)) * self.dtype(w)
                else:
                    tmp = (shell * self.dtype(w / nbar)) - self.dtype(w)
                kappas[ikappa] = healsparse.operations.sum_union([tmp, kappa])

        for ikappa, kappa in kappas.items():
            fname = self.get_fname(ikappa)
            kappa.write(fname, clobber=True)

        return kappas
