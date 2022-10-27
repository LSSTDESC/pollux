import os

import healpy as hp
import healsparse
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
from scipy.integrate import quad
from tqdm.auto import tqdm, trange

import sacc


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
        
        self.tracers = self.get_tracers()
        self.nkappas = len(self.tracers)
        
        # make sure folder exists
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

    def get_fname(self, ikappa, what='kappa'):
        return os.path.join(
            self.save_dir,
            f'{what}_{self.reader.kind}_nz{ikappa}_nside{self.shell_builder.nside}_seed{self.shell_builder.seed}.' +
            self.shell_builder.save_format)
        
    def get_tracers(self):
        self.com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, self.com_shell_bins) - 1.

        zz = np.linspace(0, z_shell_bins[-1], 1000)

        tracers = []
        if self.nzs=='zsnapshots':
            for z_source in z_shell_bins[1:]:
                tracers.append(pyccl.CMBLensingTracer(self.cosmo, z_source))
        else:
            for nz in self.nzs:
                if callable(nz):
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, (zz, nz(zz))))
                else:
                    tracers.append(pyccl.WeakLensingTracer(self.cosmo, (nz[0], nz[1])))
                
        return tracers
    
    def get_tracers_sacc(self):
        self.com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, self.com_shell_bins) - 1.

        zz = np.linspace(0, z_shell_bins[-1], 1000)

        tracers = []
        if self.nzs=='zsnapshots':
            for i,_ in enumerate(z_shell_bins[1:]):
                tracers.append(sacc.tracers.MiscTracer(f"kappa_{i}"))
        else:
            for i,nz in enumerate(self.nzs):
                if callable(nz):
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", zz, nz(zz)))
                else:
                    tracers.append(sacc.tracers.NZTracer(f"kappa_{i}", nz[0], nz[1]))
                
        return tracers

    def construct_kappas(self):
        raise NotImplementedError
    
    def load_kappas(self, ikappas=None, check_files_exist_only=False):
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
            if check_files_exist_only:
                if not os.path.exists(fname):
                    print(f"Kappa {ikappa} not found in {fname}")
            else:
                kappas.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.shell_builder.nside_c))
        return kappas

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

    def compute_kappas_alms(self, kappas=None, ikappas=None, b=None, nlb=20, save=False, skip_existing=False, check_read_existing=True):
        if kappas is None:
            kappas = self.load_kappas(ikappas)
        if ikappas is None:
            ikappas = range(self.nkappas)
        kappas_ = {i: k for i, k in zip(ikappas, kappas)}
        
        mask = self.shell_builder.get_mask()
        
        alms = {}
        for ikappa in tqdm(ikappas, desc="[compute_kappas_alms]"):
            if save:
                fname = self.get_fname(ikappa, what='alms')
                if skip_existing and os.path.exists(fname):
                    if check_read_existing:
                        try:
                            hp.read_alm(fname)
                            continue
                        except:
                            pass
                    else:
                        continue
                    
            alms[ikappa] = hp.map2alm(kappas_[ikappa].generate_healpix_map()*mask, pol=False)
            
            if save:
                hp.write_alm(fname, alms[ikappa], overwrite=True)
        
        return alms
    
    def save_cls(self, ikappas, ell, cls_out):
        if ikappas is None:
            ikappas = range(self.nkappas)
        s = sacc.Sacc() 
        tracers = self.get_tracers_sacc()
        for i,t in zip(ikappas,tracers):
            if i in ikappas:
                s.add_tracer_object(t)
        for i, j in cls_out.keys():
            for _ell,_cl in zip(ell, cls_out[i, j]):
                s.add_data_point("galaxy_convergence_cl", [tracers[i].name, tracers[j].name], _cl, ell=_ell)
        fname = os.path.join(self.save_dir, f'cls_{self.reader.kind}_nside{self.shell_builder.nside}_seed{self.shell_builder.seed}.fits')
        s.save_fits(fname, overwrite=True)
        
    def load_cls(self):
        fname = os.path.join(self.save_dir, f'cls_{self.reader.kind}_nside{self.shell_builder.nside}_seed{self.shell_builder.seed}.fits')
        return sacc.Sacc.load_fits(fname)
        
    def compute_kappas_cls(self, kappas=None, ikappas=None, b=None, nlb=20, save=False, use_precomputed_alms=False, pair_job=None):
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

        if ikappas is None:
            ikappas = range(self.nkappas)

        pairs = [(i1, i2) for i, i1 in enumerate(ikappas) for i2 in ikappas[:i + 1]]
        cls_out = {}

        if use_precomputed_alms:
            for pair_count, (i, j) in enumerate(tqdm(pairs, desc="[compute_kappas_cls]")):
                if pair_job is not None:
                    if not pair_count==pair_job:
                        continue
                alm1 = hp.read_alm(self.get_fname(i, what='alms'))
                alm2 = hp.read_alm(self.get_fname(j, what='alms'))
                pcl = hp.alm2cl(alm1, alm2)
                cls_out[i, j] = wsp.decouple_cell([pcl])[0]
        else:
            if kappas is None:
                kappas = self.load_kappas(ikappas)
            kappas_ = {i: k for i, k in zip(ikappas, kappas)}
            fields = {i: None for i in ikappas}
            for i, j in tqdm(pairs, desc="[compute_kappas_cls]"):
                if fields[i] is None:
                    fields[i] = nmt.NmtField(mask, [kappas_[i].generate_healpix_map()])
                if fields[j] is None:
                    fields[j] = nmt.NmtField(mask, [kappas_[j].generate_healpix_map()])

                cls_out[i, j] = nmt.compute_full_master(fields[i], fields[j], b, workspace=wsp)[0]
            
        ell = b.get_effective_ells()
        
        if save:
            self.save_cls(kappas, ikappas, ell, cls_out)
            
        return ell, cls_out
    

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
    
    def compute_shear_maps(self, kappas=None, ikappas=None, return_alms=False, dtype=np.float32, lmax=None, skip_existing=True, check_read_existing=True, use_precomputed_alms=False):
        if kappas is None:
            kappas = self.load_kappas(ikappas=ikappas)
        if ikappas is None:
            ikappas = range(self.nkappas)
        if lmax is None:
            lmax = 3*self.shell_builder.nside-1
            
        ell = np.arange(3*self.shell_builder.nside)
        fl = np.zeros(len(ell))
        fl[1:] = - np.sqrt((ell[1:]+2.)*(ell[1:]-1.)/(ell[1:]*(ell[1:]+1.)))
        
        pixels = np.where(self.shell_builder.get_mask())[0]
        
        gammas = []
        alms = []
        for ikappa, kappa in tqdm(zip(ikappas, kappas), desc="[compute_shear_maps]"):
            if skip_existing:
                existing=0
                for i in [1,2]:
                    fname = self.get_fname(ikappa, what=f'gamma{i}')
                    if os.path.exists(fname):
                        if check_read_existing:
                            try:
                                healsparse.HealSparseMap.read(fname, nside_coverage=self.shell_builder.nside_c)
                                existing += 1
                            except:
                                print(f"Cannot read gamma{i} {ikappa} at {fname}.")
                        else:
                            existing += 1
                if existing==2:
                    continue
            
            # Get harmonic coefficients of convergence map
            if use_precomputed_alms:
                klm = hp.read_alm(self.get_fname(ikappa, what='alms'))
            else:
                klm = hp.map2alm(kappa.generate_healpix_map(), pol=False, lmax=lmax)
            
            # Convert to shear
            glm_E = hp.almxfl(klm, fl)
            
            # Generate shear map
            g1, g2 = hp.sphtfunc.alm2map_spin([glm_E, np.zeros_like(glm_E)], nside=self.shell_builder.nside, spin=2, lmax=lmax)
            
            gamma1 = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype)
            gamma2 = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype)
            
            gamma1.update_values_pix(pixels, g1[pixels].astype(dtype))
            gamma2.update_values_pix(pixels, g2[pixels].astype(dtype))
            
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

    def __init__(self, shell_builder, nzs, save_dir=None):
        super().__init__(shell_builder, nzs, save_dir=save_dir)
        
    def compute_weight(self, ishell, ikappa):
        tracer = self.tracers[ikappa]
            
        chi_max = self.com_shell_bins[ishell + 1]
        chi_min = self.com_shell_bins[ishell]

        return quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)[0]
        
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
        # z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, com_shell_bins) - 1.

        shells = self.shell_builder.load_shells()
        
        tracers = self.get_tracers()

        for ikappa in trange(self.nkappas, desc="[construct_kappas]"):
            # tracer = pyccl.WeakLensingTracer(self.cosmo, (zz, nz(zz)))
            tracer = tracers[ikappa]

            kappa = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c,
                                                        self.shell_builder.nside,
                                                        dtype=dtype)

            weight = []
            for ishell, shell in enumerate(tqdm(shells, leave=False)):
                shell = shells[i]
                chi_max = com_shell_bins[ishell + 1]
                chi_min = com_shell_bins[ishell]

                w = quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)[0]

                pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

                tmp = shell.astype(dtype) / dtype(box_density[ishell] * pixel_volume) - 1.
                tmp *= dtype(w)

                kappa = healsparse.operations.sum_union([tmp, kappa])

                weight.append(w)

            kappas.append(kappa)
            weights.append(weight)

            fname = self.get_fname(ikappa)
            kappa.write(fname, clobber=True)

        return kappas, np.array(weights)
    
    
    def construct_kappas_lowmem(self, ikappas=None, dtype=np.float32, skip_existing=False, check_read_existing=True):
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
                    
        _ikappas = []
        for ikappa in ikappas:
            fname = self.get_fname(ikappa)
            if skip_existing and os.path.exists(fname):
                if check_read_existing:
                    try:
                        healsparse.HealSparseMap.read(fname, nside_coverage=self.shell_builder.nside_c)
                    except:
                        print(f"Cannot read kappa {ikappa} at {fname}.")
                        _ikappas.append(ikappa)
                else:
                    _ikappas.append(ikappa)
            else:
                _ikappas.append(ikappa)
                
        if len(_ikappas)==0:
            return
        
        kappas = {ikappa:healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype) for ikappa in _ikappas}

        for ishell in trange(self.nshells, desc="[construct_kappas_lowmem]"):
            shell = (self.shell_builder.load_shells([ishell])[0]).astype(dtype)
            nbar = self.shell_builder.get_pixel_density(ishell)
            for ikappa, kappa in kappas.items():
                w = self.compute_weight(ishell, ikappa)
                tmp = (shell * dtype(w / nbar)) - dtype(w) # only one multiplication and one addition
                kappas[ikappa] = healsparse.operations.sum_union([tmp, kappa])

        for ikappa, kappa in kappas.items():
            fname = self.get_fname(ikappa)
            kappa.write(fname, clobber=True)


    # def construct_kappas_sub(self, ishell, dtype=np.float32):
    #     npix = hp.nside2npix(self.shell_builder.nside)

    #     kappas = [healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype) for _ in self.nzs]
    #     tracers = self.get_tracers()

    #     box_density = [self.shell_builder.get_box_comoving_density(ishell) for ishell in range(self.nshells)]        
    #     com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        
    #     shell = self.shell_builder.load_shells([ishell])[0]
            
    #     for ikappa in range(self.nkappas):
    #         tracer = tracers[ikappa]
        
    #         chi_max = com_shell_bins[ishell + 1]
    #         chi_min = com_shell_bins[ishell]

    #         w = quad(lambda chi: tracer.get_kernel(chi), chi_min, chi_max)[0]

    #         pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

    #         tmp = shell.astype(dtype) / dtype(box_density[ishell] * pixel_volume) - 1.
    #         tmp *= dtype(w)

    #         kappas[ikappa] = healsparse.operations.sum_union([tmp, kappas[ikappa]])

    #     for ikappa, kappa in enumerate(kappas):
    #         fname = self.get_fname_sub(ikappa, ishell)
    #         kappa.write(fname, clobber=True)

    #     return kappas
    
    
    # def combine_kappas(self, ikappas=None, dtype=np.float32):
    #     if ikappas is None:
    #         ikappas = range(self.nkappas)
            
    #     for ikappa in tqdm(ikappas, desc="[combine_kappas]"):
    #         kappa = healsparse.HealSparseMap.make_empty(self.shell_builder.nside_c, self.shell_builder.nside, dtype=dtype)
    #         for ishell in trange(self.nshells, leave=False):
    #             fname = self.get_fname_sub(ikappa, ishell)
    #             tmp = healsparse.HealSparseMap.read(fname, nside_coverage=self.shell_builder.nside_c)
    #             kappa = healsparse.operations.sum_union([tmp, kappa])
    #         fname = self.get_fname(ikappa)
    #         kappa.write(fname, clobber=True)

        

