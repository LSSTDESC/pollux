import itertools
import warnings
import os

import healpy as hp
import healsparse
import numba
import numpy as np
import pyccl
import sacc
import yaml
from astropy.io import fits
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm, trange
from parallel_sort import argsort as pargsort
from collections.abc import Iterable

from .map_builder import MapBuilder
from .utils import quad, split_bins, arange_endpoint, inter_diff
from .reader import DummyReader
from .lensing import BornLensingBuilder


class ShellBuilder(MapBuilder):

    def __init__(
            self,
            nshells,
            reader,
            nside,
            save_dir,
            kind='snapshot',
            seed=12345,
            dtype=np.uint64,
            save_density=False,
            apply_grid_isometry=False,
            group_isometries=None,
            apply_replicate_rotflip=False,
            subsample_box=None,
            **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        seed, optional
            Random number generator seed for isometries (ensures results are reproducible), by default 12345
        save_density, optional
            Whether the density (instead of count map) is saved, by default False
        dtype, optional
            Dtype of the particle shells, by default np.int64
        apply_grid_isometry, optional
            Apply random isometry (rotations and flips A, and translation by shift b), such that particle positions are transformed as xyz'=A(xyz+b), by default False
        group_isometries, optional
            _description_, by default None
        subsample_box, optional
            Three available methods, subsample by float fraction, fixed pixel density (need `'densitymax_z'` in keys) or fixed Cl/Nl ratio (need `'clmax_r'` in keys), by default None
        """

        self.nshells = nshells
        self.reader = reader
        self.cosmo = reader.cosmo
        self.kind = kind
        self.seed = seed
        self.save_density = save_density
        self.apply_grid_isometry = apply_grid_isometry
        self.apply_replicate_rotflip = apply_replicate_rotflip
        assert (not apply_grid_isometry) or (
            not apply_replicate_rotflip
        )  # they do not commute and grid isometry is slow and can be applied once, whereas replicate_rotflip is fast but must be applied to each replicate

        base_fname = f'shell_{self.reader.kind}_seed{self.seed}'

        if dtype == np.int32:
            warnings.warn(f"The max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        super().__init__(nside=nside, dtype=dtype, save_dir=save_dir, base_fname=base_fname, nmaps=self.nshells, sacc_type="galaxy_density_cl", **kwargs)

        # Particle subsampling
        assert type(subsample_box) is dict or type(subsample_box) is float or subsample_box is None
        if type(subsample_box) is dict:
            keys = subsample_box.keys()
            assert 'densitymax_z' in keys or 'clmax_r' in keys
        self.subsample_box = subsample_box

        # Draw random grid isometries
        if isinstance(self.apply_grid_isometry, Iterable):
            assert len(self.apply_grid_isometry) == self.nshells
            for i in range(self.nshells):
                A, b = self.apply_grid_isometry[i]
                assert A.shape == (3, 3)
                assert b.shape == (3,)
            self.Abs = self.apply_grid_isometry
            self.apply_grid_isometry = True
        elif self.apply_grid_isometry:
            self.Abs = self.get_random_isometries(group_isometries)
            self.apply_grid_isometry = True
        else:
            # assert self.apply_grid_isometry is None
            self.Abs = [(np.zeros((3, 3), dtype=self.dtype), np.zeros(3, dtype=self.dtype)) for _ in range(self.nshells)]
            self.apply_grid_isometry = False

        # Draw random replicate rotation/flips
        if type(self.apply_replicate_rotflip) is dict:
            warnings.warn("Shell builder using user-provided replicate rot/flip: make sure not to miss any!")
            self.replicate_rotflip = self.apply_replicate_rotflip
            self.apply_replicate_rotflip = True
        elif self.apply_replicate_rotflip:
            # get number of replicates per direction
            rmax = self.shell_edges_comoving_distance()[-1]
            boxsize = self.reader.boxsize()
            nreplicate = int(np.ceil(rmax / boxsize)) + 2  # not too sure about that 2, but it's fast to check...

            # draw random rotation/flips (6 rotations, 8 flips, so max 48)
            rng = np.random.default_rng(self.seed)
            self.replicate_rotflip = {}
            for ix in range(-nreplicate, nreplicate + 1):
                for iy in range(-nreplicate, nreplicate + 1):
                    for iz in range(-nreplicate, nreplicate + 1):
                        # self.replicate_rotflip[ix, iy, iz] = rng.randint(48)
                        self.replicate_rotflip[ix, iy, iz] = rng.integers(low=0, high=48)
            self.apply_replicate_rotflip = True
        else:
            # assert self.apply_replicate_rotflip is None
            # self.replicate_rotflip = numba.typed.Dict.empty(key_type=numba.types.UniTuple(numba.types.int32,3),value_type=numba.types.float32[:])
            self.replicate_rotflip = {}
            # self.replicate_rotflip[0, 0, 0] = 0
            self.apply_replicate_rotflip = False

        if len(self.replicate_rotflip) > 0:
            replicate_rotflip = np.empty((len(self.replicate_rotflip), 4), dtype=np.int64)
            for i, (k, v) in enumerate(self.replicate_rotflip.items()):
                replicate_rotflip[i, :3] = k
                replicate_rotflip[i, 3] = v
            self.replicate_rotflip = replicate_rotflip
        else:
            self.replicate_rotflip = np.zeros((1, 4), dtype=np.int64)

        # Try getting shell information from reader, if not dummy.
        try:
            # Make sure snapshots are larger than the shells
            z_shell_bins = self.shell_edges_redshift()
            chi_shell_bins = self.shell_edges_comoving_distance()
            boxsize = self.reader.boxsize()
            diff = np.diff(chi_shell_bins)
            print(f"  - Box size = {boxsize:.2f} Mpc")
            if not np.all(diff < boxsize):
                message = "Some shells are deeper than the simulation box!\n"
                for ishell in range(self.nshells):
                    if diff[ishell] > boxsize:
                        message += f"Shell {ishell} ({z_shell_bins[ishell]:.2f}<z<{z_shell_bins[ishell+1]:.2f}) has width {diff[ishell]:.2f} Mpc\n"
                warnings.warn(message)
        except NotImplementedError:
            warnings.warn("Reader does not implement `shell_edges_redshift` or `shell_edges_comoving_distance` or `boxsize.")
            assert isinstance(reader, DummyReader)

        # Dump useful metadata
        if self.save_dir is not None:
            self.save_info()

    def shell_edges_comoving_distance(self):
        """
        Compute comoving distance to shell edges
        """
        return pyccl.comoving_radial_distance(self.cosmo, 1. / (1 + self.shell_edges_redshift()))

    def shell_edges_redshift(self):
        """Compute redshift to shell edges

        Returns
        -------
            Redshifts of shell edges
        """
        if self.kind == 'snapshot':
            if not hasattr(self, 'z_shell_bins'):
                if self.shell_edges_match == 'center_redshift':
                    func = lambda z: z
                    ifunc = func
                elif self.shell_edges_match == 'center_scale_factor':
                    func = lambda z: 1. / (1. + z)
                    ifunc = lambda a: 1. / a - 1.
                elif self.shell_edges_match == 'center_comoving_distance':
                    func = lambda z: 1e-3 * pyccl.comoving_radial_distance(self.cosmo, 1. / (1. + z))
                    ifunc = None
                elif self.shell_edges_match == 'center_comoving_volume':
                    func = lambda z: (1e-3 * pyccl.comoving_radial_distance(self.cosmo, 1. / (1. + z)))**3
                    ifunc = None
                else:
                    raise ValueError

                self.z_shell_bins = split_bins(self.reader.zsnapshots, func=func, ifunc=ifunc, zmax=self.reader.zmax)

            return self.z_shell_bins

        elif self.kind == 'lightcone':
            return self.reader.zsnapshots

        else:
            raise ValueError

    def shell_info(self, fname=None):
        """
        Compute summary shell information
        """
        z_shell_bins = self.shell_edges_redshift()
        names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id']
        dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int']
        out = np.array(
            list(
                zip(
                    np.arange(self.nshells), self.reader.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:],
                    self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[:-1])),
                    self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[1:])), self.reader.snapshots)),
            dtype=list(zip(names, dtypes)))
        if fname is not None:
            if '.txt' in fname:
                np.savetxt(fname, out, header=' '.join(names), fmt='%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i')
            else:
                np.save(fname, out)
        return out

    def save_info(self, ext='npy'):
        """
        Save summary shell information and cosmology
        """
        # Shell info for later reference and compare to existing as a safe check
        assert ext in ['npy', 'txt']
        fname = os.path.join(self.save_dir, self.base_fname + '_shell_info.' + ext)

        z_shell_bins = self.shell_edges_redshift()
        pixel_density = [self.get_pixel_density(ishell) for ishell in range(self.nshells)]
        sky_density = [self.get_sky_density(ishell) for ishell in range(self.nshells)]
        if self.kind == 'snapshot':
            names = ['ishell', 'zsnapshot', 'zmin', 'zmax', 'chisnapshot', 'chimin', 'chimax', 'snapshot_id', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'int', 'float', 'float']
            shell_info = np.array(
                list(
                    zip(
                        np.arange(self.nshells), self.reader.zsnapshots, z_shell_bins[:-1], z_shell_bins[1:],
                        self.cosmo.comoving_radial_distance(1 / (1 + self.reader.zsnapshots)), self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[:-1])),
                        self.cosmo.comoving_radial_distance(1 / (1 + z_shell_bins[1:])), self.reader.snapshots, pixel_density, sky_density)),
                dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %8i %16.8f %16.8f'
        elif self.kind == 'lightcone':
            names = ['ishell', 'zmin', 'zmax', 'pixel_density', 'sky_density']
            dtypes = ['int', 'float', 'float', 'float', 'float']
            shell_info = np.array(
                list(zip(np.arange(self.nshells), z_shell_bins[:-1], z_shell_bins[1:], pixel_density, sky_density)), dtype=list(zip(names, dtypes)))
            fmt = '%8i %16.8f %16.8f %16.8f %16.8f'
        else:
            raise ValueError

        if ext == 'npy':
            if os.path.exists(fname):
                shell_info_existing = np.load(fname)
                try:
                    np.testing.assert_array_equal(shell_info, shell_info_existing)
                except TypeError:
                    warnings.warn('Shell info file exists, different type!')
            else:
                np.save(fname, shell_info)
        else:
            np.savetxt(fname, shell_info, header=' '.join(names), fmt=fmt)

        # Save cosmology
        try:
            self.reader.cosmo.write_yaml(os.path.join(self.save_dir, "cosmo_ccl.yaml"))
        except ValueError:
            warnings.warn("Cannot write cosmo yaml file, likely due to using non-serializable options (eg P(k)).")

        # Save simulation and maps info
        data = {
            "nparticles": self.reader.nparticles,
            "boxsize": self.reader.boxsize(),
            "mass_particles": self.reader.mass_particles,
            "box_comoving_density": self.reader.box_comoving_density(),
            "kind": self.reader.kind,
            "snapshots": self.reader.snapshots.tolist(),
            "nsnapshots": self.reader.nsnapshots,
            "zsnapshots": self.reader.zsnapshots.tolist(),
            "nshells": self.nshells,
            "nside": self.nside,
            "nside_c": self.nside_c,
            "npix": self.npix,
        }
        fname = os.path.join(self.save_dir, 'sim_info.yaml')
        yaml.dump(data, open(fname, "w"))

    def get_snapshot_lensing_builder(self, kappa_dir='kappa_zsnapshots'):
        return BornLensingBuilder(self, nzs='zsnapshots', kappa_dir=kappa_dir, wsp_dir=self.wsp_dir)

    def _get_dndz(self, zz):
        """
        Helper function to get redshift distribution proportional to volume, ie $dn/dz \propto dV/dz$
        """
        aa = 1. / (1. + zz)
        dVdz = pyccl.comoving_angular_distance(self.cosmo, aa)**2 / pyccl.background.h_over_h0(self.cosmo, aa)
        return dVdz / np.trapz(dVdz, zz)  # rough normalization

    def _create_tracers(self, Nz=1000):
        z_shell_bins = self.shell_edges_redshift()

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            dndz = self._get_dndz(zz)
            tracers.append(pyccl.NumberCountsTracer(self.cosmo, has_rsd=False, dndz=(zz, dndz), bias=(zz, np.ones_like(zz))))

        return tracers

    def _create_tracers_sacc(self, Nz=1000):
        z_shell_bins = self.shell_edges_redshift()

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            dndz = self._get_dndz(zz)
            tracers.append(sacc.tracers.NZTracer(f"density_{ishell}", zz, dndz))

        return tracers

    def get_mean_z(self, ishell):
        """
        Compute the mean redshift of a shell
        """
        z_shell_bins = self.shell_edges_redshift()
        zmin = z_shell_bins[ishell]
        zmax = z_shell_bins[ishell + 1]

        dVdz = lambda z: pyccl.comoving_angular_distance(self.cosmo, 1. / (1. + z))**2 / pyccl.background.h_over_h0(self.cosmo, 1. / (1. + z))

        return quad(lambda z: z * dVdz(z), zmin, zmax) / quad(dVdz, zmin, zmax)

    def get_random_isometries(self, group_isometries=None, get_shell_identifier=False):
        """Generate random isometries for each shell

        Returns
        -------
            A list of (A,b) with an isometry A (rotations and flips) and translation by shift b.
        """
        rng = np.random.default_rng(self.seed)
        boxsize = self.reader.boxsize()
        chi_shell_bins = self.shell_edges_comoving_distance()

        # maximum width a given isometry can be applied to
        if group_isometries is not None:
            # if grouped, fix as function of box size
            if type(group_isometries) is bool:
                assert group_isometries
                dchi_iso_max = boxsize
            if type(group_isometries) is float:
                dchi_iso_max = boxsize * group_isometries
        else:
            # if not grouped, set to 0 to force drawing one isometry per shell
            dchi_iso_max = 0.

        Abs = []
        n_isometries = 0
        dchi_iso = np.inf
        shell_identifier = {}

        for ishell in range(self.nshells):
            dchi_shell = chi_shell_bins[ishell + 1] - chi_shell_bins[ishell]
            if dchi_iso + dchi_shell > dchi_iso_max:
                # if grouped shell width is larger than maximum width for isometry, draw a new one
                A, b = random_isometry(rng, boxsize)
                n_isometries += 1
                dchi_iso = 0.
            dchi_iso += dchi_shell
            Abs.append([A.astype(self.reader.dtype), b.astype(self.reader.dtype)])
            shell_identifier[ishell] = n_isometries

        if group_isometries is not None:
            print(f"  - Using {n_isometries} isometries grouped within {dchi_iso_max:.2f} Mpc.")
        else:
            print(f"  - Using {n_isometries} isometries.")

        if get_shell_identifier:
            return Abs, shell_identifier
        else:
            return Abs

    def estimate_replicates(self, ishell):
        """
        Estimate the number of simulation replicates needed with a simple volume ratio
        """
        chi_shell_bins = self.shell_edges_comoving_distance()
        box_volume = self.reader.boxsize()**3

        shell_volume = 4. / 3. * np.pi * (chi_shell_bins[ishell + 1]**3 - chi_shell_bins[ishell]**3)
        if self.octant_only:
            shell_volume /= 8.

        return shell_volume / box_volume

    def get_replicates(self, **kwargs):
        """
        Compute the list of replicate indices (ix,iy,iz) for all the sub-volumes and snapshots
        """
        replicates = []
        for isub in trange(self.reader.nsubs, desc='[get_replicates]'):
            replicates.append((isub, self.construct_maps_sub(isub, get_replicates_only=True, **kwargs)))
        return replicates

    def get_subsample_ratio(self, ishell):
        """
        Compute the sub-sampling ratio
        """
        if type(self.subsample_box) is dict:
            if not hasattr(self, 'subsample_ratios'):
                if 'densitymax_z' in self.subsample_box.keys():
                    # estimated number of particles per pixel
                    ngal_pix = np.array([self.estimate_replicates(ishell) for ishell in range(self.nshells)]) * self.reader.nparticles / self.npix
                    if self.octant_only:
                        ngal_pix *= 8.
                    snr_pix = 1. / np.sqrt(ngal_pix)  # S/N per pixel
                    # S/N at redshift subsample_box_auto_z
                    snr_pix_cut = np.interp(self.subsample_box['densitymax_z'], [self.get_mean_z(ishell) for ishell in range(self.nshells)], snr_pix)
                    # fraction of particles to maintain S/N of z>subsample_box_auto_z
                    self.subsample_ratios = np.minimum(1., (snr_pix / snr_pix_cut)**2)
                elif 'clmax_r' in self.subsample_box.keys():
                    # estimated number of particles per pixel
                    ngal_pix = np.array([self.estimate_replicates(ishell) for ishell in range(self.nshells)]) * self.reader.nparticles / self.npix
                    if self.octant_only:
                        ngal_pix *= 8.
                    # compute theory Cl's
                    cl = self.predict_cls(auto_only=True, approximate=True)[1]
                    # fix particle density such that shot-noise is 100 times lower than Cl at lmax
                    nbar_th = self.subsample_box['clmax_r'] / np.array([cl[i, i][-1] for i in range(self.nshells)])
                    # convert to desired number of particles per pixel
                    ngal_th = nbar_th * hp.nside2pixarea(self.nside, degrees=False)
                    self.subsample_ratios = np.minimum(1., ngal_th / ngal_pix)
                else:
                    raise ValueError
            return self.subsample_ratios[ishell]
        elif type(self.subsample_box) is float:
            return self.subsample_box
        elif self.subsample_box is None:
            return 1.
        else:
            raise ValueError

    def box_comoving_density(self, ishell):
        return self.reader.box_comoving_density() * self.get_subsample_ratio(ishell)

    def get_pixel_density(self, ishell):
        """Computes the expected mean number of particle per pixels from the simulation density.

        Parameters
        ----------
        ishell
            Index of the shell.
        nside
            Resolution parameter nside of the shell.

        Returns
        -------
            Mean number of particles per pixel.
        """
        chi_shell_bins = self.shell_edges_comoving_distance()

        chi_max = chi_shell_bins[ishell + 1]
        chi_min = chi_shell_bins[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / self.npix

        return pixel_volume * self.box_comoving_density(ishell)

    def get_sky_density(self, ishell, degrees=False):
        area = hp.nside2pixarea(self.nside, degrees=degrees)
        return self.get_pixel_density(ishell) / area

    def combine_maps_subs(self, ishells=None, add_orphans=False):
        """
        Sums the particle shells computed independently for each sub-volume/snapshot with construct_maps_sub
        """
        shells = []

        mask, ipix = self.get_mask(nest=True, return_ipix=True)
        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            if self.check_maps(read=True, fname=fname):
                continue
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            shell.update_values_pix(ipix, self.dtype(0), nest=True)
            isubs = self.reader.get_isubs_for_snapshot(ishell)
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                tmp_shell = self.load_maps(ishell, sub=isub)
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            if add_orphans:
                print('Adding the orphan as well!')
                tmp_shell = healsparse.HealSparseMap.read(self.get_fname(ishell, sub='orphan'))
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)

            shells.append(shell)

        return shells

    def construct_maps(self, imaps=None, **kwargs):
        # first construct shells on simulation subs
        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            isubs = self.reader.get_isubs_for_snapshot(ishell)
            for isub in tqdm(isubs, desc=f"[construct_maps: shell {ishell}]", leave=False):
                self.construct_maps_sub(isub, **kwargs)

        # then combine into final shells
        return self.combine_maps_subs(imaps)

    def _field_map(self, ishell, nest=True):
        """
        Overrides the _field_map method to generate density maps from count maps.
        """
        shell = self.load_maps(ishell)
        
        if self.save_density:
            return shell.generate_healpix_map(nest=nest)
        else:
            spix = shell.valid_pixels
            s = shell[spix]
            
            if not nest:
                spix = hp.nest2ring(self.nside, spix)
                
            _, ipix = self.get_mask(nest=nest, return_ipix=True)
            
            density = np.zeros(self.npix)
            density[ipix] = -1.
            density[spix] = s / self.get_pixel_density(ishell) - 1.
            
            return density

    def _noise_cls(self, ishell):
        return 1. / self.get_sky_density(ishell)

    def compute_tidal_tensor_maps(self, ishells=None, smooth_Mpc=None, lmax=None, replicate_octant=False):
        """Compute the tidal fields of density maps

        Parameters
        ----------
        ishells, optional
            Shell indices, by default None
        smooth_Mpc, optional
            Smoothing scale of the density field in Mpc, by default None (no smoothing)
        lmax, optional
            Maximum multipole to consider, by default None
        replicate_octant, optional
            If True, alm's are recomputed from octant maps replicated to cover the full sky (irrelevant if `octant_only` is False), by default False  
        """
        if lmax is None:
            lmax = 3 * self.nside - 1

        _, ipix = self.get_mask(nest=False, return_ipix=True)

        for ishell in self.tqdm(ishells, desc="[compute_tidal_tensor]"):
            alm_E = self.compute_alms(ishell, save=not (replicate_octant), replicate_octant=replicate_octant, pol=False, lmax=lmax)[ishell]

            if smooth_Mpc is not None:
                assert smooth_Mpc > 0.
                sigma = smooth_Mpc / self.shell_edges_comoving_distance()[ishell]

                hp.sphtfunc.smoothalm(
                    alm_E,
                    sigma=sigma,
                    pol=False,
                    inplace=True,
                )

            Q, U = hp.alm2map_spin((alm_E, np.zeros_like(alm_E)), self.nside, spin=2, lmax=lmax)

            # Q = hp.reorder(Q, r2n=True)[ipix]
            # U = hp.reorder(U, r2n=True)[ipix]

            density = self.get_field_map(ishell, nest=False)[ipix]

            # In 2D cartesian, s11(k)+s22(k)=delta(k)
            # In the spherical polarization framework, we have:
            # Q is 'g1' = (t11 - t22)
            # delta is t11 + t22
            # U is 'g2' = t12
            # t11 is (Q +delta)/2
            # t22 is (delta-Q)/2
            # Need to subtract off the trace, affecting s_11 and s_22... (-delta/3)
            # t11 = (delta+Q)/2 - density/3
            # t22 = (delta-Q)/2 - density/3
            # t12 = U

            for what, val in zip(['t11', 't22', 't12'], [density / 6. + Q / 2., density / 6. - Q / 2., U]):
                hsp = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                hsp.update_values_pix(ipix, val.astype(self.dtype), nest=False)
                hsp.write(self.get_fname(ishell, what=what), clobber=True, format=self.save_format)


class SnapshotShellBuilder(ShellBuilder):

    def __init__(self, reader, nside, save_dir, shell_edges_match='center_scale_factor', **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        shell_edges_match
            Method to define shells for snapshots redshifts, must be 'center_redshift', 'center_scale_factor', 'center_comoving_distance' or 'center_comoving_volume'
        """

        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir, kind='snapshot', **kwargs)

    def construct_maps_sub(self, isub, verbose=False, replicate_ixyz=None, get_replicates_only=False, fast=True):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and replicating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Prints feedback, by default False
        replicate_ixyz, optional
            Tuple (ix,iy,iz) of the specific replicate to compute, by default None
        get_replicates_only, optional
            If True, returns the list of replicate tuples (ix,iy,iz) where the sub-volume and the shell intersect, by default False
        """

        ishell = self.reader.get_isnapshot(isub)

        # check if shell exists and/or can be saved
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        if self.check_maps(read=True, fname=fname):
            if verbose:
                print(f"Found and checked map at {fname}.")
            return

        # get shell radii
        chi_shell_bins = self.shell_edges_comoving_distance()
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

        # get number of replicates per direction
        boxsize = self.reader.boxsize()

        # read box
        xyz, box_edges_original = self.reader.read_sub(isub, read_box_edges_only=get_replicates_only)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        # particle sub-sampling
        if self.subsample_box is not None:
            ratio = self.get_subsample_ratio(ishell)
            nparticles_kept = int(len(xyz) * ratio)
            if verbose:
                print(f"Keeping {nparticles_kept} particles, ie a fraction {ratio}.")
            if nparticles_kept < len(xyz):
                rng = np.random.default_rng(self.seed + ishell)
                xyz = xyz[rng.choice(len(xyz), size=nparticles_kept, replace=False), :]

        # initialize shell
        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

        nside = self.nside
        order = hp.nside2order(self.nside)
        ncap = (nside * nside - nside) << 1
        npix = self.npix

        # print(self.apply_grid_isometry)
        # print(self.Abs[ishell][0], self.Abs[ishell][1])
        # print(self.apply_replicate_rotflip)
        # print(self.replicate_rotflip)
        # print(self.octant_only)

        uipix, c = _construct_maps_sub_helper(
            True, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
            self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
        if len(uipix) > 0:
            if verbose:
                print(f"Found {np.sum(c)} particles in the shell.")
            shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
        else:
            if verbose:
                print("Found no particle.")

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            if verbose:
                print(f"Saved to file {fname}")

        return shell

    def construct_catalog_sub(self, isub, verbose=False, **kwargs):
        """Construct a particle catalog for a given sub-volume/snapshot

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        Abs, optional
            Random isometries, if pre-computed, by default None
        verbose, optional
            Prints feedback, by default False

        Returns
        -------
            Indices in particle files, RA, DEC, and radial comoving distance
        """
        ishell = self.reader.get_isnapshot(isub)

        # get shell radii
        chi_shell_bins = self.shell_edges_comoving_distance()
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

        # get number of replicates per direction
        boxsize = self.reader.boxsize()

        # read box
        xyz, box_edges_original = self.reader.read_sub(isub, **kwargs)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        idx, ra, dec, r = _construct_catalog_sub_helper(
            True, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1], self.apply_replicate_rotflip,
            self.replicate_rotflip, self.octant_only)
        if len(idx) > 0:
            return np.concatenate(idx), np.concatenate(ra), np.concatenate(dec), np.concatenate(r)
        else:
            return [np.array([])] * 4

    def construct_catalog(self, ishell, extra_columns=[], **kwargs):
        """Construct and save a particle catalog for a given shell

        Parameters
        ----------
        ishell
            Shell index
        """
        isubs = self.reader.get_isubs_for_snapshot(ishell)

        fname = self.get_fname(ishell, what='haloes')

        if os.path.exists(fname):
            try:
                return fits.open(fname)
            except:
                pass

        hdul = [fits.PrimaryHDU()]

        for isub in self.tqdm(isubs, desc=f"[construct_halo_catalog: shell {ishell}]"):
            # Construct catalog for simulation sub-volume/snapshot
            res = self.construct_catalog_sub(isub, **kwargs)

            # Build columns for output files
            idx = fits.Column(name='ihaloes', format='J', array=res[0])
            ra = fits.Column(name='ra', format='E', array=res[1])
            dec = fits.Column(name='dec', format='E', array=res[2])
            r = fits.Column(name='chi', format='E', array=res[3])
            a = pyccl.scale_factor_of_chi(self.cosmo, res[3])
            z = fits.Column(name='z', format='E', array=1. / a - 1.)
            cols = [idx, ra, dec, r, z]

            # Add extra columns
            if len(extra_columns) > 0:
                tmp = self.reader.read_props(isub, extra_columns, ihaloes=res[0])
                for name, array in tmp.items():
                    cols.append(fits.Column(name=name, format='E', array=array))

            header = fits.Header()
            header[f'ISUB'] = isub
            header[f'ZSNAP'] = self.reader.zsnapshots[ishell]

            hdul.append(fits.BinTableHDU.from_columns(cols, header))

        fits.HDUList(hdul).writeto(fname, overwrite=True)

        return hdul

    def read_catalogs_props(self, ishell, variables):
        """Get particle (eg haloes) properties from a precomputed catalog

        Parameters
        ----------
        ishell
            Shell index
        variables
            Variables to recover from the particle data and the computed catalog (adds RA, DEC, redshift `z` and radial comoving distance `chi`)

        Returns
        -------
            Dictionary of properties
        """
        fname = self.get_fname(ishell, what='haloes')
        hdul = fits.open(fname)

        out = {k: [] for k in variables}

        catalog_variables = [v for v in variables if v in ['ra', 'dec', 'chi', 'z']]
        props_variables = [v for v in variables if v not in ['ra', 'dec', 'chi', 'z']]

        for hdu in self.tqdm(hdul[1:], desc="[read_catalogs_props]"):
            if len(props_variables) > 0:
                isub = hdu.header['ISUB']
                tmp = self.reader.read_props(isub, props_variables, ihaloes=hdu.data['ihaloes'])
                for k in props_variables:
                    out[k].append(tmp[k])
            if len(catalog_variables) > 0:
                for k in catalog_variables:
                    out[k].append(hdu.data[k])

        for k in variables:
            out[k] = np.concatenate(out[k])

        return out


class SnapshotShellBuilderLarsenTEST(ShellBuilder):

    # this one's for processing Patricia's file using the same shell edges as Cyrille's lightcone interpolation implementation

    def __init__(self, reader, nside, save_dir, **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        shell_edges_match
            Method to define shells for snapshots redshifts, must be 'center_redshift', 'center_scale_factor', 'center_comoving_distance' or 'center_comoving_volume'
        """

        super().__init__(reader.nsnapshots, reader, nside, save_dir, kind='lightcone', **kwargs)

    # def _create_tracers_sacc(self, Nz=1000):
    #     z_shell_bins = self.shell_edges_redshift()

    #     tracers = []
    #     for ishell in range(self.nshells - 1):
    #         zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
    #         dndz = self._get_dndz(zz)
    #         tracers.append(sacc.tracers.NZTracer(f"density_{ishell}", zz, dndz))

    #     return tracers

    def construct_maps_sub(self, isnapshot, isub, verbose=False):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and replicating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Prints feedback, by default False
        """

        if not os.path.exists(os.path.join(self.save_dir, 'subs')):
            os.makedirs(os.path.join(self.save_dir, 'subs'))

        fname = os.path.join(self.save_dir, 'subs/shell_hacc_nside' + str(self.nside) + '_imap' + str(isnapshot) + '_sub' + str(isub) + '.fits')

        if self.check_maps(read=True, fname=fname):
            print(fname + ' already exists!')
            shell = healsparse.HealSparseMap.read(fname)
        else:
            # get lightcone line
            z_shell_bins = self.shell_edges_redshift()
            a_i = 1. / (1. + z_shell_bins[isnapshot + 1])
            a_f = 1. / (1. + z_shell_bins[isnapshot])
            a_lc = arange_endpoint(a_i, a_f, 0.005)
            chi2_lc = self.cosmo.comoving_radial_distance(a_lc)**2
            rmin2, rmax2 = chi2_lc[-1], chi2_lc[0]

            Abs = [(np.zeros((3, 3), dtype=self.dtype), np.zeros(3, dtype=self.dtype)) for _ in range(self.nshells)]

            boxsize = self.reader.boxsize()

            try:
                xyz = self.reader.read_sub(isnapshot + 1, isub)
            except:
                print(f"Sub {isub} not available for snapshot {isnapshot}.")

            try:
                box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
                cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
                box_edges_original = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes.astype(self.dtype)
            except:
                box_edges_original = np.zeros((8, 3)).astype(self.dtype)

            if verbose:
                print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

            # initialize shell
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            nside = self.nside
            order = hp.nside2order(self.nside)
            ncap = (nside * nside - nside) << 1
            npix = self.npix

            uipix, c = _construct_maps_sub_helper(
                False, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
                self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
            if len(uipix) > 0:
                if verbose:
                    print(f"Found {np.sum(c)} particles in the shell.")
                shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
            else:
                if verbose:
                    print("Found no particle.")

            if self.save_dir is not None:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                if verbose:
                    print(f"Saved to file {fname}")

        return shell

    def combine_maps_subs(self, ishells=None):
        """
        Sums the particle shells computed independently for each sub-volume/snapshot with construct_maps_sub
        """
        shells = []

        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            if self.check_maps(read=True, fname=fname):
                continue
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            isubs = range(1000)
            n_subs_added = 0
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                try:
                    tmp_shell = self.load_maps(ishell, sub=isub)
                    shell = healsparse.operations.sum_union([shell, tmp_shell])
                    n_subs_added += 1
                except:
                    continue
            if n_subs_added != 0:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                shells.append(shell)
            else:
                print('Zero sub files found for shell number ' + str(ishell) + '!')

        return shells


class SnapshotShellBuilderLarsen(ShellBuilder):

    # this one's for processing Patricia's files using the same shell edges as Cyrille's snapshot-to-lightcone implementation

    def __init__(self, reader, nside, save_dir, shell_edges_match='center_scale_factor', **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        shell_edges_match
            Method to define shells for snapshots redshifts, must be 'center_redshift', 'center_scale_factor', 'center_comoving_distance' or 'center_comoving_volume'
        """

        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir, **kwargs)

    def construct_maps_sub(self, isnapshot, isub, verbose=False):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and replicating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Prints feedback, by default False
        """

        if not os.path.exists(os.path.join(self.save_dir, 'subs')):
            os.makedirs(os.path.join(self.save_dir, 'subs'))

        fname = os.path.join(self.save_dir, 'subs/shell_hacc_nside' + str(self.nside) + '_imap' + str(isnapshot) + '_sub' + str(isub) + '.fits')

        if self.check_maps(read=True, fname=fname):
            print(fname + ' already exists!')
            shell = healsparse.HealSparseMap.read(fname)
        else:

            # get shell radii
            chi_shell_bins = self.shell_edges_comoving_distance()
            rmin2, rmax2 = chi_shell_bins[isnapshot]**2, chi_shell_bins[isnapshot + 1]**2

            Abs = [(np.zeros((3, 3), dtype=self.dtype), np.zeros(3, dtype=self.dtype)) for _ in range(self.nshells)]

            boxsize = self.reader.boxsize()

            # read box
            if isnapshot == 0:
                xyz = self.reader.read_sub(1, isub)
            else:
                try:
                    xyz_i = self.reader.read_sub(isnapshot, isub)
                except:
                    xyz_i = None

                try:
                    xyz_f = self.reader.read_sub(isnapshot + 1, isub)
                except:
                    xyz_f = None

                if xyz_i is None and xyz_f is None:
                    print(f"Sub {isub} not available for neither snapshot {isnapshot} or {isnapshot+1}. Exiting.")
                elif xyz_i is None:
                    print(f"Sub {isub} only loaded for snapshot {isnapshot+1}.")
                    xyz = xyz_f
                elif xyz_f is None:
                    print(f"Sub {isub} only loaded for snapshot {isnapshot}.")
                    xyz = xyz_i
                else:
                    print(f"Sub {isub} loaded successfully for both snapshots {isnapshot} and {isnapshot+1}. Merging files and processing them.")
                    xyz = np.vstack((xyz_i, xyz_f))

            try:
                box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
                cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
                box_edges_original = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes.astype(self.dtype)
            except:
                box_edges_original = np.zeros((8, 3)).astype(self.dtype)

            if verbose:
                print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

            # initialize shell
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            nside = self.nside
            order = hp.nside2order(self.nside)
            ncap = (nside * nside - nside) << 1
            npix = self.npix

            uipix, c = _construct_maps_sub_helper(
                False, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
                self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
            if len(uipix) > 0:
                if verbose:
                    print(f"Found {np.sum(c)} particles in the shell.")
                shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
            else:
                if verbose:
                    print("Found no particle.")

            if self.save_dir is not None:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                if verbose:
                    print(f"Saved to file {fname}")

        return shell

    def combine_maps_subs(self, ishells=None):
        """
        Sums the particle shells computed independently for each sub-volume/snapshot with construct_maps_sub
        """
        shells = []

        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            if self.check_maps(read=True, fname=fname):
                continue
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            isubs = range(1000)
            n_subs_added = 0
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                try:
                    tmp_shell = self.load_maps(ishell, sub=isub)
                    shell = healsparse.operations.sum_union([shell, tmp_shell])
                    n_subs_added += 1
                except:
                    continue
            if n_subs_added != 0:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                shells.append(shell)
            else:
                print('Zero sub files found for shell number ' + str(ishell) + '!')

        return shells


class SnapshotShellBuilderHeitmann(ShellBuilder):

    # this one's for processing Katrin's files

    def __init__(self, reader, nside, save_dir, shell_edges_match='center_scale_factor', **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        shell_edges_match
            Method to define shells for snapshots redshifts, must be 'center_redshift', 'center_scale_factor', 'center_comoving_distance' or 'center_comoving_volume'
        """

        self.shell_edges_match = shell_edges_match

        super().__init__(reader.nsnapshots, reader, nside, save_dir, kind='snapshot', **kwargs)

    def construct_maps_sub(self, ishell, isub, verbose=False):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and replicating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Prints feedback, by default False
        """

        if not os.path.exists(os.path.join(self.save_dir, 'subs')):
            os.makedirs(os.path.join(self.save_dir, 'subs'))

        # fname = os.path.join(self.save_dir, 'subs/shell_hacc_nside' + str(self.nside) + '_imap' + str(ishell) + '_sub' + str(isub) + '.fits')
        fname = self.get_fname(ishell, sub=isub)

        if self.check_maps(read=True, fname=fname):
            print(fname + ' already exists!')
            shell = healsparse.HealSparseMap.read(fname)
        else:

            chi_shell_bins = self.shell_edges_comoving_distance()
            rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

            z_shell_bins = self.shell_edges_redshift()
            zmin, zmax = z_shell_bins[ishell], z_shell_bins[ishell + 1]

            Abs = [(np.zeros((3, 3), dtype=self.dtype), np.zeros(3, dtype=self.dtype)) for _ in range(self.nshells)]

            boxsize = self.dtype(self.reader.boxsize())

            # 1

            zsnapshot_fulllist = 1 / self.reader.get_asnapshots(self.reader.snapshots_fulllist) - 1

            xyz = []

            try:

                for isnapshot in range(len(zsnapshot_fulllist) - 1):
                    if zmin > zsnapshot_fulllist[isnapshot + 1]:
                        continue
                    if zmax < zsnapshot_fulllist[isnapshot]:
                        continue
                    xyz.append(self.reader.read_sub(isnapshot, isub))
                xyz = np.vstack(xyz)

                if ishell == 0:
                    print(['Edges', rmin2, (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2).min(), rmax2, (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2).max()])
                else:
                    print(['Edges', (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2).min(), rmin2, rmax2, (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2).max()])
            except:
                pass

            # 2

            # if self.reader.snap_sub_every_n is None and self.reader.snap_sub_every_n_downsample is None:
            #     if ishell == 0:
            #         xyz = self.reader.read_sub(0, isub)
            #     else:
            #         xyz_i = self.reader.read_sub(ishell - 1, isub)
            #         xyz_f = self.reader.read_sub(ishell, isub)
            #         xyz = np.vstack((xyz_i, xyz_f))
            # else:
            #     # read box
            #     if ishell == 0:
            #         i = np.argwhere(self.reader.snapshots[ishell] == self.reader.snapshots_fulllist)[0][0]
            #         f = np.argwhere(self.reader.snapshots[ishell + 1] == self.reader.snapshots_fulllist)[0][0]
            #         f = int(f / 2 + 1)+1
            #         for k in np.arange(i, f):
            #             try:
            #                 xyz = np.vstack((xyz, self.reader.read_sub(k, isub)))
            #             except:
            #                 xyz = self.reader.read_sub(k, isub)
            #         try:
            #             print(['Edges',rmin2,(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2).min(),rmax2,(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2).max()])
            #         except:
            #             pass
            #     else:
            #         i = np.argwhere(self.reader.snapshots[ishell - 1] == self.reader.snapshots_fulllist)[0][0]
            #         f = np.argwhere(self.reader.snapshots[ishell] == self.reader.snapshots_fulllist)[0][0]
            #         delta = int((f - i) / 2)
            #         i += delta-1
            #         f += delta+1
            #         for k in np.arange(i, f):
            #             try:
            #                 xyz = np.vstack((xyz, self.reader.read_sub(k, isub)))
            #             except:
            #                 xyz = self.reader.read_sub(k, isub)
            #         try:
            #             print(['Edges',(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2).min(),rmin2,rmax2,(xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2).max()])
            #         except:
            #             pass

            try:
                box_lims = np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])
                cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
                box_edges_original = box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes.astype(self.dtype)
            except:
                box_edges_original = np.zeros((8, 3)).astype(self.dtype)

            if verbose:
                print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

            # initialize shell
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

            nside = self.nside
            order = hp.nside2order(self.nside)
            ncap = (nside * nside - nside) << 1
            npix = self.npix

            # print(xyz)
            # print(box_edges_original)
            # print(boxsize)
            # print(rmin2)
            # print(rmax2)
            # print(self.apply_grid_isometry)
            # print(self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)

            uipix, c = _construct_maps_sub_helper(
                False, xyz, box_edges_original, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.Abs[ishell][0], self.Abs[ishell][1],
                self.apply_replicate_rotflip, self.replicate_rotflip, self.octant_only, nside, order, ncap, npix)
            if len(uipix) > 0:
                if verbose:
                    print(f"Found {np.sum(c)} particles in the shell.")
                shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)
            else:
                if verbose:
                    print("Found no particle.")

            if self.save_dir is not None:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                if verbose:
                    print(f"Saved to file {fname}")

        return shell

    def combine_maps_subs(self, ishells=None):
        """
        Sums the particle shells computed independently for each sub-volume/snapshot with construct_maps_sub
        """
        shells = []
        
        mask, ipix = self.get_mask(nest=True, return_ipix=True)
        for ishell in self.tqdm(ishells, desc="[combine_maps_subs]"):
            fname = self.get_fname(ishell)
            if self.check_maps(read=True, fname=fname):
                continue
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            shell.update_values_pix(ipix, self.dtype(0), nest=True)
            isubs = range(1000)
            n_subs_added = 0
            for isub in tqdm(isubs, desc=f"[combine_maps_subs: shell {ishell}]", leave=False):
                try:
                    tmp_shell = self.load_maps(ishell, sub=isub)
                    shell = healsparse.operations.sum_union([shell, tmp_shell])
                    n_subs_added += 1
                except:
                    continue
            if n_subs_added != 0:
                shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
                shells.append(shell)
            else:
                print('Zero sub files found for shell number ' + str(ishell) + '!')

        return shells


class LightconeShellBuilder(ShellBuilder):

    def __init__(self, reader, nside, save_dir, **kwargs):
        """`ShellBuilder` is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir
            Path of folder to save shells and other products.
        """

        super().__init__(reader.nsnapshots - 1, reader, nside, save_dir, kind='lightcone', **kwargs)

    def construct_maps_sub(self, isub, verbose=False, replicate_ixyz=None, get_replicates_only=False, fast=True, do_orphan_ishell=None):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and replicating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        verbose, optional
            Prints feedback, by default False
        replicate_ixyz, optional
            Tuple (ix,iy,iz) of the specific replicate to compute, by default None
        get_replicates_only, optional
            If True, returns the list of replicate tuples (ix,iy,iz) where the sub-volume and the shell intersect, by default False
        """

        if do_orphan_ishell is not None:
            isnap = do_orphan_ishell
        else:
            isnap = self.reader.get_isnapshot(isub)

        # check if shell exists and/or can be saved
        if do_orphan_ishell is not None:
            fname = self.get_fname(isnap, sub='orphan')
        else:
            fname = self.get_fname(isnap, sub=isub)

        # check if shell has already been created
        if self.check_maps(read=True, fname=fname):
            if verbose:
                print(f"Found and checked map at {fname}.")
            return

        # get lightcone line
        z_shell_bins = self.shell_edges_redshift()
        a_i = 1. / (1. + z_shell_bins[isnap + 1])
        a_f = 1. / (1. + z_shell_bins[isnap])
        a_lc = arange_endpoint(a_i, a_f, 0.005)
        chi2_lc = self.cosmo.comoving_radial_distance(a_lc)**2
        rmin2, rmax2 = chi2_lc[-1], chi2_lc[0]

        # get number of replicates per direction
        boxsize = self.reader.boxsize()
        # not too sure about that 2, but it's fast to check...
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2

        # read box
        if do_orphan_ishell is not None:
            keys = ['x', 'y', 'z']

            # load orphan particles and sort to match them
            xyzi = np.concatenate(
                [np.load(self.get_fname(isnap, sub=isub, what='orphan_i', format='npy')) for isub in self.reader.get_isubs_for_snapshot(isnap)])
            xyzf = np.concatenate(
                [np.load(self.get_fname(isnap, sub=isub, what='orphan_f', format='npy')) for isub in self.reader.get_isubs_for_snapshot(isnap)])
            if verbose:
                print(f"Loaded {len(xyzi)} orphan particles.")

            # sort id's and match particles
            sidxi = np.argsort(xyzi['id'])
            sidxf = np.argsort(xyzf['id'])
            xyz_i = np.vstack([xyzi[sidxi][k] for k in keys]).T
            xyz_f = np.vstack([xyzf[sidxf][k] for k in keys]).T
            if verbose:
                np.testing.assert_array_equal(xyzi['id'][sidxi], xyzf['id'][sidxf])
                print("Matched particles.")

            # move particles that went through box faces
            part_boundary_fix(xyz_i, xyz_f, boxsize, fixed='i')

            # recompute box edges accordingly
            box_lims = np.stack(
                [np.min([np.min(xyz_i, axis=0), np.min(xyz_f, axis=0)], axis=0),
                 np.max([np.max(xyz_i, axis=0), np.max(xyz_f, axis=0)], axis=0)])
            cube_nodes = ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0).astype(int)  #[(0,0,0),(0,0,1),...,(1,1,1)]
            box_edges_original = (box_lims[0] + (box_lims[1] - box_lims[0]) * cube_nodes).astype(self.dtype)

        else:
            xyz_i, box_edges_original, idx_i = self.reader.read_sub(isub, extra_variables=['id'])
            xyz_f, box_edges_original, idx_f = self.reader.read_sub(isub + self.reader.nsubs, extra_variables=['id'])

            if verbose:
                print(f"Loaded box {isub} (snapshot {isnap}) with {len(xyz_i)} particles.")
                print(f"Loaded box {isub + self.reader.nsubs} (snapshot {isnap + 1}) with {len(xyz_f)} particles.")

            # match the indices of particles
            idx_i_in, idx_f_in, idx_i_out, idx_f_out = inter_diff(idx_i['id'], idx_f['id'])
            assert len(idx_i_in) == len(idx_f_in)

            # dump orphan particles
            np.save(
                self.get_fname(isnap, sub=isub, what='orphan_i', format='npy'),
                np.rec.array((*xyz_i[idx_i_out].T, idx_i['id'][idx_i_out]), names=['x', 'y', 'z', 'id']))
            np.save(
                self.get_fname(isnap, sub=isub, what='orphan_f', format='npy'),
                np.rec.array((*xyz_f[idx_f_out].T, idx_f['id'][idx_f_out]), names=['x', 'y', 'z', 'id']))

            # keep matched particles
            xyz_i = xyz_i[idx_i_in]
            xyz_f = xyz_f[idx_f_in]

            if verbose:
                print(f"Found {len(idx_i_in)} particles in both boxes, {len(idx_i_out)}/{len(idx_f_out)} orphans in initial/final boxes saved.")

        # particle sub-sampling
        if self.subsample_box is not None:
            ratio = self.get_subsample_ratio(isnap)
            nparticles_kept = int(len(xyz_i) * ratio)
            if verbose:
                print(f"Keeping {nparticles_kept} particles, ie a fraction {ratio}.")
            if nparticles_kept < len(xyz_i):
                rng = np.random.default_rng(self.seed + isnap)
                s = rng.choice(len(xyz_i), size=nparticles_kept, replace=False)
                xyz_i = xyz_i[s, :]
                xyz_f = xyz_f[s, :]

        # initialize shell
        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

        nside = self.nside
        order = hp.nside2order(self.nside)
        ncap = (nside * nside - nside) << 1
        npix = self.npix

        if False:
            uipix, c = _construct_maps_sub_helper(
                xyz, box_edges_original, self.Abs[isnap][0], self.Abs[isnap][1], nreplicate, boxsize, rmin2, rmax2, self.apply_grid_isometry, self.octant_only,
                nside, order, ncap, npix)
            if len(uipix) > 0:
                shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)

        else:
            iso_applied = False  # to xyz

            # precompute isometry on particles (since replicates are then only a shift away)
            if self.apply_grid_isometry:
                box_edges = Axpb(*self.Abs[isnap], box_edges_original)
            else:
                box_edges = box_edges_original

            replicates = []
            # replicate the box
            for ix in range(-nreplicate, nreplicate + 1):
                for iy in range(-nreplicate, nreplicate + 1):
                    for iz in range(-nreplicate, nreplicate + 1):
                        # check if we want to do only one replicate
                        if replicate_ixyz is not None:
                            if not np.array_equal(replicate_ixyz, [ix, iy, iz]):
                                continue

                        # compute shift
                        shift_original = (np.array([ix, iy, iz], dtype=self.reader.dtype) * self.reader.boxsize())[None, :]
                        if self.apply_grid_isometry:
                            shift = Ax(self.Abs[isnap][0], shift_original)
                        else:
                            shift = shift_original

                        # apply isometry to box edges
                        box_edges_shifted = shift + box_edges

                        # check if box intersects octant, otherwise skip
                        # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                        if self.octant_only and np.any(np.all(box_edges_shifted <= 0., axis=0)):
                            continue

                        # check if box intersects with shell, otherwise skip
                        # first, check if box fully inside inner sphere
                        if np.all(xyz2r2(box_edges_shifted) < rmin2):
                            continue

                        # second, check if box intersects with outer sphere
                        sphere_center = -self.Abs[isnap][1] if self.apply_grid_isometry else np.zeros(3)
                        if not AABB_sphere_intersect(box_edges_original + shift_original, sphere_center, rmax2):
                            continue

                        # if we got this far, then this replicate intersects with the shell
                        replicates.append([ix, iy, iz])
                        if verbose:
                            print(f"Found overlap for box {isub} (snapshot {isnap}) for replicate {(ix,iy,iz)}.")
                        if get_replicates_only:
                            continue

                        # now, add particles to shell

                        # apply random isometries (once) if necessary
                        if self.apply_grid_isometry and not iso_applied:
                            xyz_i = Axpb(*self.Abs[isnap], xyz_i)
                            xyz_f = Axpb(*self.Abs[isnap], xyz_f)
                            iso_applied = True

                        # compute unique pixels into which particles fall and their counts, all done in vec2pix_unique (using numba) as follows:
                        # 1 - remove particles out of shell based on radius
                        # 2 - remove particles out of octant if octant_only
                        # 3 - compute corresponding healpy pixels of all particles (removed particles fast set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                        # 4 - get unique pixels and counts (ignoring -1's)
                        ipix = _worldline2pix(nside, order, ncap, npix, a_lc, chi2_lc, *(xyz_i.T), *(xyz_f.T), *(shift[0].T), rmin2, rmax2, self.octant_only)
                        uipix, c = _unique_pix_pos(ipix, npix)

                        if len(uipix) > 0:
                            if verbose:
                                print(f"Found {np.sum(c)} particles in the shell.")

                            # add particles to shell
                            tmp_shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                            tmp_shell.update_values_pix(uipix, c.astype(self.dtype), nest=False)

                            shell = healsparse.operations.sum_union([shell, tmp_shell])
                        else:
                            if verbose:
                                print("Found no particle.")

            if verbose:
                print(f"Found overlap for box {isub} (shell {isnap}) for {len(replicates)} replicates.")

            if get_replicates_only:
                return replicates

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)
            if verbose:
                print(f"Saved to file {fname}")

            # xyz_i[idx_i_out]
            # xyz_f[idx_f_out]

        return shell


# @numba.njit(parallel=False)
# def _add_at_one(a, index):
#     n = len(index)
#     for i in numba.prange(n):
#         a[index[i]] += 1


@numba.njit
def xyz2r2(xyz):
    """
    x,y,z to radius squared
    """
    return xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2


# @numba.njit(parallel=True)
# def xyz2r(xyz):
#     return np.sqrt(xyz2r2(xyz))

# @numba.njit(parallel=True)
# def xyzr2tp(xyz, r):
#     assert len(xyz) == len(r)
#     n = len(xyz)
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     theta = np.arccos(z / r)
#     phi = np.arctan2(x, y)
#     return theta, phi


@numba.njit(parallel=True)
def Axpb(A, b, xyz):
    """
    Affine transformation A(xyz+b)
    """
    return np.dot(xyz + b, A)


@numba.njit
def Ax(A, xyz):
    """
    Linear transformation A(xyz)
    """
    return np.dot(xyz, A)


# @numba.njit(parallel=True)
# def Axpb_inv(A, b, xyz):
#     return np.dot(xyz, A.T) - b


@numba.njit
def rotflip(rot, xyz, box_size):
    """
    Apply 90deg rotations and flips to replicate as performed in HACC
    """
    x = xyz[:, 0] - box_size / 2.
    y = xyz[:, 1] - box_size / 2.
    z = xyz[:, 2] - box_size / 2.

    rot_type = rot >> 3
    # rotations by 90 deg are swapping coordinates, so 3!=6 possibilities
    if rot_type == 0:
        xtmp = x
        ytmp = y
        ztmp = z
    elif rot_type == 1:
        xtmp = x
        ytmp = z
        ztmp = y
    elif rot_type == 2:
        xtmp = y
        ytmp = x
        ztmp = z
    elif rot_type == 3:
        xtmp = y
        ytmp = z
        ztmp = x
    elif rot_type == 4:
        xtmp = z
        ytmp = x
        ztmp = y
    elif rot_type == 5:
        xtmp = z
        ytmp = y
        ztmp = x
    else:
        raise ValueError

    # sign flips are after (2^3=8 possibilities)
    sign_type = rot & ((1 << 3) - 1)
    if ((sign_type & (1 << 0)) != 0):
        xtmp = -xtmp
    if ((sign_type & (1 << 1)) != 0):
        ytmp = -ytmp
    if ((sign_type & (1 << 2)) != 0):
        ztmp = -ztmp

    xtmp += box_size / 2.
    ytmp += box_size / 2.
    ztmp += box_size / 2.

    # note depending on replication value you will still need to translate this box to the appropriate place in the lightcone after performing this rotation, and if velocities are required you'll need to apply a similar shift to those
    return np.column_stack((xtmp, ytmp, ztmp))


def random_isometry(rng, boxsize):
    """
    Generate a random isometry defined by a matrix representing rotations/permutations/flips and a vector translation
    """
    pcomb = list(itertools.product([1, -1], repeat=3))  # all permuted orders
    r = Rotation.random(random_state=rng).as_matrix()
    p = pcomb[rng.choice(len(pcomb))]
    A = np.dot(r, np.diag(np.array(p)))
    b = rng.random(3) * boxsize
    return A, b


def AABB_sphere_intersect(box_edges, s, r2):
    """Test overlap between a box defined by its edges and a sphere (center s, radius squared r2) see https://stackoverflow.com/a/4579069"""
    c1, c2 = box_edges.min(axis=0), box_edges.max(axis=0)
    d2 = r2
    for i in range(3):
        if s[i] < c1[i]:
            d2 -= (s[i] - c1[i])**2
        elif s[i] > c2[i]:
            d2 -= (s[i] - c2[i])**2
    return d2 > 0.


@numba.njit
def _AABB_sphere_intersect(box_edges, s, r2):
    """Test overlap between a box defined by its edges and a sphere (center s, radius squared r2) see https://stackoverflow.com/a/4579069"""
    # c1, c2 = box_edges.min(axis=0), box_edges.max(axis=0)
    d2 = r2
    for i in range(3):
        c1 = np.min(box_edges[:, i])
        c2 = np.max(box_edges[:, i])
        if s[i] < c1:
            d2 -= (s[i] - c1)**2
        elif s[i] > c2:
            d2 -= (s[i] - c2)**2
    return d2 > 0.


inv_halfpi = 2. / np.pi


@numba.njit
def _loc2pix(nside, order, ncap, npix, z, phi, sth):
    """"
    Healpix function loc2pix
    """
    za = np.abs(z)
    tt = phi * inv_halfpi % 4.
    if za <= 2. / 3.:
        nl4 = 4 * nside
        temp1 = nside * (0.5 + tt)
        temp2 = nside * z * 0.75
        jp = int(temp1 - temp2)  # index of  ascending edge line
        jm = int(temp1 + temp2)  # index of descending edge line

        # ring number counted from z=2/3
        ir = nside + 1 + jp - jm  # in {1,2n+1}
        kshift = 1 - (ir % 2)  # kshift=1 if ir even, 0 otherwise

        t1 = jp + jm - nside + kshift + 1 + nl4 + nl4
        if order > 0:
            ip = (t1 >> 1) & (nl4 - 1)
        else:
            ip = ((t1 >> 1) % nl4)  # in {0,4n-1}

        return ncap + (ir - 1) * nl4 + ip

    else:  # Equatorial region
        tp = tt - np.floor(tt)
        if za < 0.99:
            tmp = nside * np.sqrt(3 * (1 - za))
        else:
            tmp = nside * sth / np.sqrt((1. + za) / 3.)

        jp = int(tp * tmp)  # increasing edge line index
        jm = int((1. - tp) * tmp)  # decreasing edge line index

        ir = jp + jm + 1  # ring number counted from the closest pole
        ip = int(tt * ir)  # in {0,4*ir-1}

        if z > 0:
            return 2 * ir * (ir - 1) + ip
        else:
            return npix - 2 * ir * (ir + 1) + ip


@numba.njit
def _vec2pix_sub(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """
    Helper function for _vec2pix
    """
    # apply shift
    tmpx = x + shiftx
    tmpy = y + shifty
    tmpz = z + shiftz

    # get radius and remove particles out of shell
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return -1

    # get longitude (phi), altitude (nz) and distance to z-axis (sth) of particle projected on the unit sphere
    phi = np.arctan2(tmpy, tmpx)
    xl = 1. / np.sqrt(r2)
    nz = tmpz * xl
    sth = np.sqrt(tmpx**2 + tmpy**2) * xl

    return _loc2pix(nside, order, ncap, npix, nz, phi, sth)


@numba.njit(parallel=True)
def _vec2pix(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """"
    Parallelized healpix function vec2pix to go from x,y,z to healpix pixel with extra checks (radii and octant)
    """
    n = len(x)
    ipix = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        ipix[i] = _vec2pix_sub(nside, order, ncap, npix, x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return ipix


@numba.njit
def _vec2radec_sub(x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """
    Helper function for _vec2radec
    """
    # apply shift
    tmpx = x + shiftx
    tmpy = y + shifty
    tmpz = z + shiftz

    # get radius and remove particles out of shel
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return (False, 0., 0., 0.)

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return (False, 0., 0, 0.)

    r = np.sqrt(r2)
    theta = np.arccos(tmpz / r)
    phi = np.arctan2(tmpy, tmpx)

    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)

    return (True, ra, dec, r)


@numba.njit(parallel=True)
def _vec2radec(x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """
    Parallelized function to go from x,y,z to RA, DEC and radius with extra checks (radii and octant)
    """
    n = len(x)
    ra = np.zeros(n, dtype=x.dtype)
    dec = np.zeros(n, dtype=x.dtype)
    r = np.zeros(n, dtype=x.dtype)
    idx = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        good, tmpra, tmpdec, tmpr = _vec2radec_sub(x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
        if good:
            idx[i] = i
            ra[i] = tmpra
            dec[i] = tmpdec
            r[i] = tmpr
        else:
            idx[i] = -1

    return idx, ra, dec, r


@numba.njit
def _unique_pix_pos(ipix, npix):
    """
    Count the unique (positive) indices
    """
    n = len(ipix)
    c = np.zeros(npix, dtype=np.int_)
    for i in range(n):
        pos = ipix[i]
        if pos >= 0:
            c[pos] += 1
    u = np.argwhere(c)[:, 0]
    return u, c[u]


@numba.njit
def _construct_maps_sub_helper(
        do_replicate, xyz_original, box_edges_original, boxsize, rmin2, rmax2, apply_grid_isometry, A, b, apply_replicate_rotflip, rep2rot, octant_only, nside,
        order, ncap, npix):
    """
    Parallelized helper function for ShellBuilder.construct_shell_sub
    """

    rot_dict = numba.typed.Dict()
    for i in range(len(rep2rot)):
        ix, iy, iz = rep2rot[i, :3]
        rot_dict[ix, iy, iz] = rep2rot[i, 3]

    # precompute isometry on particles (since replicates are then only a shift away)
    if apply_grid_isometry:
        xyz = Axpb(A, b, xyz_original)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        if not apply_replicate_rotflip:
            xyz = xyz_original
            box_edges = box_edges_original

    n = len(xyz_original)
    dtype = xyz_original.dtype
    count = np.zeros(npix, dtype=np.uint)
    if do_replicate:
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2
    else:
        nreplicate = 0

    # replicate the box
    for ix in range(-nreplicate, nreplicate + 1):
        for iy in range(-nreplicate, nreplicate + 1):
            for iz in range(-nreplicate, nreplicate + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_grid_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                if apply_replicate_rotflip:
                    if (ix, iy, iz) not in rot_dict:
                        continue
                    else:
                        box_edges = rotflip(rot_dict[ix, iy, iz], box_edges_original, boxsize).astype(dtype)

                # apply shift to box edges
                box_edges_shifted = shift + box_edges

                ## check if box intersects octant, otherwise skip
                # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                if octant_only:
                    if np.any(np.array([np.all(box_edges_shifted[:, i] <= 0.) for i in range(3)])):
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(box_edges_shifted) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                if apply_grid_isometry:
                    if not _AABB_sphere_intersect(box_edges_original + shift_original, -b, rmax2):
                        continue
                else:
                    if not _AABB_sphere_intersect(box_edges_shifted, np.zeros(3, dtype=dtype), rmax2):
                        continue

                ## add particles to shell
                # rotate/flip replicate
                if apply_replicate_rotflip:
                    xyz = rotflip(rot_dict[ix, iy, iz], xyz_original, boxsize).astype(dtype)

                # compute corresponding healpy pixels of all particles (particles out of shell/octant quickly set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                ipix = _vec2pix(nside, order, ncap, npix, xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, octant_only)

                for i in range(n):
                    pos = ipix[i]
                    if pos >= 0:
                        count[pos] += 1

    u = np.argwhere(count)[:, 0]
    return u, count[u]


@numba.njit
def _construct_catalog_sub_helper(
        do_replicate, xyz_original, box_edges_original, boxsize, rmin2, rmax2, apply_grid_isometry, A, b, apply_replicate_rotflip, rep2rot, octant_only):
    """
    Parallelized helper function for ShellBuilder.construct_catalog_sub
    """

    rot_dict = numba.typed.Dict()
    for i in range(len(rep2rot)):
        ix, iy, iz = rep2rot[i,:3]
        rot_dict[ix, iy, iz] = rep2rot[i,3]

    # precompute isometry on particles (since replicates are then only a shift away)
    if apply_grid_isometry:
        xyz = Axpb(A, b, xyz_original)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        if not apply_replicate_rotflip:
            xyz = xyz_original
            box_edges = box_edges_original

    idx = []
    ra = []
    dec = []
    r = []

    dtype = xyz.dtype
    if do_replicate:
        nreplicate = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2
    else:
        nreplicate = 0

    # replicate the box
    for ix in range(-nreplicate, nreplicate + 1):
        for iy in range(-nreplicate, nreplicate + 1):
            for iz in range(-nreplicate, nreplicate + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_grid_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                # apply shift to box edges
                box_edges_shifted = shift + box_edges

                ## check if box intersects octant, otherwise skip
                # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                if octant_only:
                    if np.any(np.array([np.all(box_edges_shifted[:, i] <= 0.) for i in range(3)])):
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(box_edges_shifted) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                if apply_grid_isometry:
                    if not _AABB_sphere_intersect(box_edges_original + shift_original, -b, rmax2):
                        continue
                else:
                    if not _AABB_sphere_intersect(box_edges_shifted, np.zeros(3, dtype=dtype), rmax2):
                        continue

                ## add particles to catalog
                # rotate/flip replicate
                if apply_replicate_rotflip:
                    xyz = rotflip(rot_dict[ix, iy, iz], xyz_original, boxsize).astype(dtype)

                # compute ra, dec and radial comoving distance (idx is -1 for particles to reject)
                tmpidx, tmpra, tmpdec, tmpr = _vec2radec(xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, octant_only)

                # add to output
                w = tmpidx >= 0
                if np.any(w):
                    idx.append(tmpidx[w])
                    ra.append(tmpra[w])
                    dec.append(tmpdec[w])
                    r.append(tmpr[w])

    return idx, ra, dec, r


@numba.njit
def intersect_lightcone(chi2_i, chi2_f, a, chi2):
    a_i = a[0]
    a_f = a[-1]
    # y = chi2 - (chi2_i + (chi2_f - chi2_i) * (a - a_i) / (a_f - a_i))
    y_last = 0.
    dchida = (chi2_f - chi2_i) / (a_f - a_i)
    m = len(a)
    for j in range(1, m):
        #     if y[j] < 0.:
        #         return a[j - 1] - y[j - 1] * (a[j - 1] - a[j]) / (y[j - 1] - y[j])
        y_new = chi2[j] - (chi2_i + dchida * (a[j] - a_i))

        if y_new < 0.:
            return a[j - 1] - y_last * (a[j - 1] - a[j]) / (y_last - y_new)
        else:
            y_last = y_new


@numba.njit
def _worldline2pix_sub(nside, order, ncap, npix, a_lc, chi2_lc, x_i, y_i, z_i, x_f, y_f, z_f, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """
    Helper function for _vec2pix
    """
    # apply shift
    tmpx_i = x_i + shiftx
    tmpy_i = y_i + shifty
    tmpz_i = z_i + shiftz

    tmpx_f = x_f + shiftx
    tmpy_f = y_f + shifty
    tmpz_f = z_f + shiftz

    # get radius and remove particles out of shell
    r2_i = tmpx_i**2 + tmpy_i**2 + tmpz_i**2
    r2_f = tmpx_f**2 + tmpy_f**2 + tmpz_f**2
    if r2_f < rmin2 or r2_i >= rmax2:  # it is enough to test initial/final radii as it cannot evolve faster than light
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx_i < 0. and tmpx_f < 0.) or (tmpy_i < 0. and tmpy_f < 0.) or (tmpz_i < 0. and tmpz_f < 0.):
            return -1

    # find intersection of particle wordline and lightcone (ie find scale factor when particle comoving radius intersects the comoving radial distance function)
    a_int = intersect_lightcone(r2_i, r2_f, a_lc, chi2_lc)

    a_i = a_lc[0]
    a_f = a_lc[-1]

    assert a_i <= a_int and a_int < a_f

    a_ratio = (a_int - a_i) / (a_f - a_i)
    tmpx = tmpx_i + (tmpx_f - tmpx_i) * a_ratio
    tmpy = tmpy_i + (tmpy_f - tmpy_i) * a_ratio
    tmpz = tmpz_i + (tmpz_f - tmpz_i) * a_ratio

    # remove particles that intersect lightcone out of shell
    r2 = tmpx**2 + tmpy**2 + tmpz**2
    if r2 < rmin2 or r2 >= rmax2:
        return -1

    # remove particles out of octant if octant_only
    if octant_only:  # check if particle out of octant
        if (tmpx < 0.) or (tmpy < 0.) or (tmpz < 0.):
            return -1

    # get longitude (phi), altitude (nz) and distance to z-axis (sth) of particle projected on the unit sphere
    phi = np.arctan2(tmpy, tmpx)
    xl = 1. / np.sqrt(r2)
    nz = tmpz * xl
    sth = np.sqrt(tmpx**2 + tmpy**2) * xl

    return _loc2pix(nside, order, ncap, npix, nz, phi, sth)


@numba.njit(parallel=True)
def _worldline2pix(nside, order, ncap, npix, a_lc, chi_lc, x_i, y_i, z_i, x_f, y_f, z_f, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
    """"
    Parallelized healpix function vec2pix to go from x,y,z to healpix pixel with extra checks (radii and octant)
    """
    n = len(x_i)
    ipix = np.zeros(n, dtype=np.int_)
    for i in numba.prange(n):
        ipix[i] = _worldline2pix_sub(
            nside, order, ncap, npix, a_lc, chi_lc, x_i[i], y_i[i], z_i[i], x_f[i], y_f[i], z_f[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
    return ipix


@numba.njit
def _inter_diff_helper(ar, sar_idx, ar1_size):
    sar = ar[sar_idx]

    mask = sar[1:] == sar[:-1]

    ar1_indices_in = sar_idx[:-1][mask]
    ar2_indices_in = sar_idx[1:][mask] - ar1_size

    flag = np.concatenate((np.array([True]), ~mask, np.array([True])))
    z = sar_idx[flag[1:] & flag[:-1]]
    w = z >= ar1_size

    ar1_indices_out = z[~w]
    ar2_indices_out = z[w] - ar1_size

    return ar1_indices_in, ar2_indices_in, ar1_indices_out, ar2_indices_out


def inter_diff(ar1, ar2):
    ar = np.concatenate((ar1, ar2))
    sar_idx = pargsort(ar)

    return _inter_diff_helper(ar, sar_idx, ar1.size)


@numba.njit
def part_boundary_fix_helper(xi, xf, boxsize, fixed='i'):
    if np.abs(xf - xi) < boxsize / 2.:
        return (xi, xf)
    else:
        if fixed == 'i':
            if xi < xf:
                return (xi, xf - boxsize)
            else:
                return (xi, xf + boxsize)
        elif fixed == 'f':
            if xi < xf:
                return (xi + boxsize, xf)
            else:
                return (xi - boxsize, xf)
        else:
            raise ValueError


@numba.njit(parallel=True)
def part_boundary_fix(xyzi, xyzf, boxsize, fixed='i'):
    n = len(xyzi)
    for i in numba.prange(n):
        for k in range(3):
            xyzi[i, k], xyzf[i, k] = part_boundary_fix_helper(xyzi[i, k], xyzf[i, k], boxsize=boxsize, fixed=fixed)
