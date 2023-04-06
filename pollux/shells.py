import itertools
import warnings
import os

import healpy as hp
import healsparse
import numba
import numpy as np
import pyccl
import sacc
from astropy.io import fits
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm, trange

from .map_builder import MapBuilder
from .utils import quad


class ShellBuilder(MapBuilder):

    def __init__(
            self,
            reader,
            nside,
            save_dir,
            apply_random_isometry=False,
            seed=12345,
            dtype=np.uint64,
            subsample_box=None,
            z_subsample_auto=2.,
            group_isometries=None,
            **kwargs):
        """ShellBuilder is a class to handle the computation of particle shells from snapshots of an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        nside
            Resolution parameter nside of the shells.
        save_dir, optional
            Path of folder to save shells and other products, by default None
        apply_random_isometry, optional
            Apply random isometry (rotations and flips A, and translation by shift b), such that particle positions are transformed as xyz'=A(xyz+b), by default False
        seed, optional
            Random number generator seed for isometries (ensures results are reproducible), by default 12345
        octant_only, optional
            Use only an octant of the sphere instead of the full sphere, by default True
        nside_c, optional
            Resolution of the coverage map of shells as defined by healpsarse, by default 32
        dtype, optional
            Dtype of the particle shells, by default np.int64
        save_format, optional
            File format of shells (fits or h5), by default 'fits'
        """

        self.reader = reader
        self.cosmo = reader.cosmo
        self.apply_random_isometry = apply_random_isometry
        self.seed = seed
        self.subsample_box = subsample_box
        self.z_subsample_auto = z_subsample_auto
        self.group_isometries = group_isometries

        self.nshells = self.reader.nshells

        base_fname = f'shell_{self.reader.kind}'
        if apply_random_isometry:
            base_fname += f'_seed{self.seed}'

        super().__init__(
            nside=nside,
            dtype=dtype,
            save_dir=save_dir,
            base_fname=base_fname,
            nmaps=self.nshells,
            sacc_type="galaxy_density_cl",
            **kwargs)

        if dtype == np.int32:
            print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        # Make sure snapshots are larger than the shells
        z_shell_bins = self.reader.shell_edges_redshift()
        chi_shell_bins = self.reader.shell_edges_comoving_distance()
        boxsize = self.reader.boxsize()
        diff = np.diff(chi_shell_bins)
        print(f"Box size = {boxsize:.2f} Mpc")
        print(f"Number of shells = {self.nmaps}")
        if not np.all(diff < boxsize):
            message = "Some shells are deeper than the simulation box!\n"
            for ishell in range(self.nmaps):
                if diff[ishell] > boxsize:
                    message += f"Shell {ishell} ({z_shell_bins[ishell]:.2f}<z<{z_shell_bins[ishell+1]:.2f}) has width {diff[ishell]:.2f} Mpc\n"
            warnings.warn(message)

        # Draw random isometries
        if apply_random_isometry:
            self.Abs = self.get_random_isometries()

        # Particle subsampling
        assert subsample_box == 'auto' or type(subsample_box) is float or subsample_box is None

    def _get_dndz(self, zz):
        """
        Helper function to get redshift distribution proportional to volume, ie $dn/dz \propto dV/dz$
        """
        aa = 1. / (1. + zz)
        dVdz = pyccl.comoving_angular_distance(self.cosmo, aa)**2 / pyccl.background.h_over_h0(self.cosmo, aa)
        return dVdz / np.trapz(dVdz, zz)  # rough normalization

    def _create_tracers(self, Nz=1000):
        z_shell_bins = self.reader.shell_edges_redshift()

        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            dndz = self._get_dndz(zz)
            tracers.append(pyccl.NumberCountsTracer(self.cosmo, False, dndz=(zz, dndz), bias=(zz, np.ones_like(zz))))

        return tracers

    def _create_tracers_sacc(self, Nz=1000):
        z_shell_bins = self.reader.shell_edges_redshift()

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
        z_shell_bins = self.reader.shell_edges_redshift()
        amin = 1. / (1. + z_shell_bins[ishell + 1])
        amax = 1. / (1. + z_shell_bins[ishell])
        dVdz = lambda a: pyccl.comoving_angular_distance(self.cosmo, a)**2 / pyccl.background.h_over_h0(self.cosmo, a)

        amean = quad(lambda a: a * dVdz(a), amin, amax) / quad(dVdz, amin, amax)

        return 1. / amean - 1.

    def get_random_isometries(self):
        """Generate random isometries for each shell

        Returns
        -------
            A list of (A,b) with an isometry A (rotations and flips) and translation by shift b.
        """
        rng = np.random.default_rng(self.seed)
        boxsize = self.reader.boxsize()

        # find number of isometries (can be grouped for adjacent shells)
        n_isometries = self.nshells
        if self.group_isometries is not None:
            chi_shell_bins = self.reader.shell_edges_comoving_distance()
            if type(self.group_isometries) is bool:
                assert self.group_isometries
                com_iso_depth = boxsize
            if type(self.group_isometries) is float:
                com_iso_depth = boxsize * self.group_isometries
            n_isometries = int(np.ceil(chi_shell_bins[-1] / com_iso_depth))
            com_iso_bins = np.arange(n_isometries + 1) * com_iso_depth
            print(f"Grouping {n_isometries} isometries for shells within {com_iso_depth:.2f} Mpc.")

        # draw random isometries
        _Abs = []
        for _ in range(n_isometries):
            A, b = random_isometry(rng, boxsize)
            _Abs.append([A.astype(self.reader.dtype), b.astype(self.reader.dtype)])

        # assign isometries to shells
        if self.group_isometries is not None:

            def get_max_overlap(a, b):
                return np.argmax(np.fmin(a[1:, None], b[None, 1:]) - np.fmax(a[:-1, None], b[None, :-1]), axis=1)

            # check overlap of volume of shells
            idx = get_max_overlap(chi_shell_bins**3, com_iso_bins**3)

            self.Abs = []
            for ishell in range(self.nshells):
                self.Abs.append(_Abs[idx[ishell]])

        else:
            self.Abs = _Abs

        return self.Abs

    def estimate_repeats(self, ishell):
        """
        Estimate the number of simulation repeats needed with a simple volume ratio
        """
        chi_shell_bins = self.reader.shell_edges_comoving_distance()
        box_volume = self.reader.boxsize()**3

        shell_volume = 4. / 3. * np.pi * (chi_shell_bins[ishell + 1]**3 - chi_shell_bins[ishell]**3)
        if self.octant_only:
            shell_volume /= 8.

        return shell_volume / box_volume

    def get_repeats(self, **kwargs):
        """
        Compute the list of repeat indices (ix,iy,iz) for all the sub-volumes and snapshots
        """
        repeats = []
        for isub in trange(self.reader.nsubs, desc='[get_repeats]'):
            repeats.append((isub, self.construct_shell_sub(isub, get_repeats_only=True, **kwargs)))
        return repeats

    def get_subsample_ratio(self, ishell):
        """
        Compute the sub-sampling ratio
        """
        if self.subsample_box == 'auto':
            if not hasattr(self, 'subsample_ratios'):
                # estimated number of galaxies per pixel
                ngal_pix = np.array([self.estimate_repeats(ishell) for ishell in range(self.nshells)
                                    ]) * self.reader.nparticles / self.npix
                if self.octant_only:
                    ngal_pix *= 8.
                snr_pix = 1. / np.sqrt(ngal_pix)  # S/N per pixel
                # S/N at redshift z_subsample_auto
                snr_pix_cut = np.interp(self.z_subsample_auto, self.reader.zsnapshots, snr_pix)
                # fraction of particles to maintain S/N of z>z_subsample_auto
                self.subsample_ratios = np.minimum(1., (snr_pix / snr_pix_cut)**2)
            ratio = self.subsample_ratios[ishell]
        elif type(self.subsample_box) is float:
            ratio = self.subsample_box
        elif self.subsample_box is None:
            ratio = 1.
        return ratio

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
        npix = hp.nside2npix(self.nside)
        chi_shell_bins = self.reader.shell_edges_comoving_distance()

        chi_max = chi_shell_bins[ishell + 1]
        chi_min = chi_shell_bins[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

        return pixel_volume * self.box_comoving_density(ishell)

    def construct_shell_sub(
            self,
            isub,
            Abs=None,
            chi_shell_bins=None,
            skip_existing=False,
            check_read_existing=True,
            verbose=False,
            repeat_ixyz=None,
            get_repeats_only=False,
            fast=True):
        """Construct a single shell from a simulation sub-volume/snapshot, applying isometries and repeating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        Abs, optional
            Random isometries, if pre-computed, by default None
        chi_shell_bins, optional
            Comoving distance of shell edges, if pre-computed, by default None
        skip_existing, optional
            Whether to skip if map file exsists, by default False
        check_read_existing, optional
            Whether to try reading the file before skipping, by default True
        verbose, optional
            Prints feedback, by default False
        repeat_ixyz, optional
            Tuple (ix,iy,iz) of the specific repeat to compute, by default None
        get_repeats_only, optional
            If True, returns the list of repeat tuples (ix,iy,iz) where the sub-volume and the shell intersect, by default False
        """

        ishell = self.reader.get_ishell(isub)

        # check if shell exists and/or can be saved
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        if skip_existing:
            if self.check_maps(read=check_read_existing, fname=fname):
                return

        # get shell radii
        if chi_shell_bins is None:
            chi_shell_bins = self.reader.shell_edges_comoving_distance()
        assert self.nshells == len(chi_shell_bins) - 1
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

        # get random isometries
        if self.apply_random_isometry:
            if Abs is None:
                Abs = self.Abs
            else:
                Abs = Abs

        # get number of repeats per direction
        boxsize = self.reader.boxsize()
        # not too sure about that 2, but it's fast to check...
        nrepeat = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2

        # read box
        xyz, box_edges_original = self.reader.read_sub(isub, read_box_edges_only=get_repeats_only)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        # particle sub-sampling
        if self.subsample_box is not None:
            if self.subsample_box == 'auto':
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
        order = hp.nside2order(nside)
        ncap = (nside * nside - nside) << 1
        npix = hp.nside2npix(nside)

        if fast:

            u, c = construct_shell_sub(
                xyz, box_edges_original, Abs[ishell][0], Abs[ishell][1], nrepeat, boxsize, rmin2, rmax2,
                self.apply_random_isometry, self.octant_only, nside, order, ncap, npix)
            if len(u) > 0:
                shell.update_values_pix(u, c.astype(self.dtype))

        else:
            # precompute isometry on particles (since repeats are then only a shift away)
            if self.apply_random_isometry:
                xyz = Axpb(*Abs[ishell], xyz)
                box_edges = Axpb(*Abs[ishell], box_edges_original)
            else:
                box_edges = box_edges_original

            repeats = []
            # repeat the box
            for ix in range(-nrepeat, nrepeat + 1):
                for iy in range(-nrepeat, nrepeat + 1):
                    for iz in range(-nrepeat, nrepeat + 1):
                        # check if we want to do only one repeat
                        if repeat_ixyz is not None:
                            if not np.array_equal(repeat_ixyz, [ix, iy, iz]):
                                continue

                        # compute shift
                        shift_original = (np.array([ix, iy, iz], dtype=self.reader.dtype) *
                                          self.reader.boxsize())[None, :]
                        if self.apply_random_isometry:
                            shift = Ax(Abs[ishell][0], shift_original)
                        else:
                            shift = shift_original

                        # apply isometry to box edges
                        tmp_box_edges = shift + box_edges

                        # check if box intersects octant, otherwise skip
                        # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                        if self.octant_only and np.any(np.all(tmp_box_edges <= 0., axis=0)):
                            continue

                        # check if box intersects with shell, otherwise skip
                        # first, check if box fully inside inner sphere
                        if np.all(xyz2r2(tmp_box_edges) < rmin2):
                            continue

                        # second, check if box intersects with outer sphere
                        sphere_center = -Abs[ishell][1] if self.apply_random_isometry else np.zeros(3)
                        if not AABB_sphere_intersect(box_edges_original + shift_original, sphere_center, rmax2):
                            continue

                        # if we got this far, then this repeat intersects with the shell
                        repeats.append([ix, iy, iz])
                        if verbose:
                            print(f"Found overlap for box {isub} (shell {ishell}) for repeat {(ix,iy,iz)}")
                        if get_repeats_only:
                            continue

                        # now, add particles to shell
                        # if self.apply_random_isometry:
                        #     shift = Ax(Abs[ishell][0], shift) # NOW DONE BEFORE

                        # compute unique pixels into which particles fall and their counts, all done in vec2pix_unique (using numba) as follows:
                        # 1 - remove particles out of shell based on radius
                        # 2 - remove particles out of octant if octant_only
                        # 3 - compute corresponding healpy pixels of all particles (removed particles fast set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                        # 4 - get unique pixels and counts (ignoring -1's)
                        ipix = _vec2pix(
                            nside, order, ncap, npix, *(xyz.T), *(shift[0].T), rmin2, rmax2, self.octant_only)
                        u, c = _unique_pix_pos(ipix, npix)
                        # u, c = _vec2pix_unique(nside, order, ncap, npix, *(xyz.T), *(shift[0].T), rmin2, rmax2, self.octant_only)

                        if len(u) > 0:
                            if verbose:
                                print(f"Found {len(u)} particles.")

                            # add particles to shell
                            tmp_shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                            tmp_shell.update_values_pix(u, c.astype(self.dtype))

                            shell = healsparse.operations.sum_union([shell, tmp_shell])
                        else:
                            if verbose:
                                print("Found no particle.")

            if verbose:
                print(f"Found overlap for box {isub} (shell {ishell}) for {len(repeats)} repeats")

            if get_repeats_only:
                return repeats

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format)
            if verbose:
                print(f"Saved to file {fname}")

        return shell

    def construct_shells(self, isubs=None, **kwargs):
        """
        Construct all shells
        """
        if isubs is None:
            # Run all files
            isubs = range(self.reader.njobs)

        for isub in tqdm(isubs, desc='[construct_shells]'):
            self.construct_shell_sub(isub, **kwargs)

    def combine_shells(self, ishells=None, skip_existing=False, check_read_existing=True):
        """
        Sums the particle shells computed independently for each sub-volume/snapshot with construct_shell_sub
        """
        shells = []

        for ishell in self.tqdm(ishells, desc="[combine_shells]"):
            fname = self.get_fname(ishell)
            if skip_existing:
                if self.check_maps(read=check_read_existing, fname=fname):
                    continue
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            isubs = self.reader.get_isubs_for_shell(ishell)
            for isub in tqdm(isubs, desc=f"[combine_shells: shell {ishell}]", leave=False):
                tmp_fname = self.get_fname(ishell, sub=isub)
                tmp_shell = healsparse.HealSparseMap.read(tmp_fname)
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            shell.write(fname, clobber=True, format=self.save_format)

            shells.append(shell)

        return shells

    def construct_maps(self, imaps=None, skip_existing=False, check_read_existing=True, **kwargs):
        # first construct shells on simulation subs
        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            isubs = self.reader.get_isubs_for_shell(ishell)
            self.construct_shells(isubs, skip_existing=skip_existing, check_read_existing=check_read_existing, **kwargs)

        # then combine into final shells
        return self.combine_shells(imaps, skip_existing=skip_existing, check_read_existing=check_read_existing)

    def get_field_map(self, ishell):
        shell = self.load_maps(ishell)
        spix = shell.valid_pixels
        density = -np.ones(self.npix)
        density[spix] = shell[spix] / self.get_pixel_density(ishell) - 1.
        return density

    def construct_shell_sub_catalog(self, isub, Abs=None, chi_shell_bins=None, verbose=False, **kwargs):
        """Construct a particle catalog for a given sub-volume/snapshot

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        Abs, optional
            Random isometries, if pre-computed, by default None
        chi_shell_bins, optional
            Comoving distance of shell edges, if pre-computed, by default None
        verbose, optional
            Prints feedback, by default False

        Returns
        -------
            Indices in particle files, RA, DEC, and radial comoving distance
        """
        ishell = self.reader.get_ishell(isub)

        # get shell radii
        if chi_shell_bins is None:
            chi_shell_bins = self.reader.shell_edges_comoving_distance()
        assert self.nshells == len(chi_shell_bins) - 1
        rmin2, rmax2 = chi_shell_bins[ishell]**2, chi_shell_bins[ishell + 1]**2

        # get random isometries
        if self.apply_random_isometry:
            if Abs is None:
                Abs = self.Abs
            else:
                Abs = Abs

        # get number of repeats per direction
        boxsize = self.reader.boxsize()
        # not too sure about that 2, but it's fast to check...
        nrepeat = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2

        # read box
        xyz, box_edges_original = self.reader.read_sub(isub, **kwargs)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        idx, ra, dec, r = construct_shell_sub_catalog(
            xyz, box_edges_original, Abs[ishell][0], Abs[ishell][1], nrepeat, boxsize, rmin2, rmax2,
            self.apply_random_isometry, self.octant_only)

        if len(idx) > 0:
            return np.concatenate(idx), np.concatenate(ra), np.concatenate(dec), np.concatenate(r)
        else:
            return [np.array([])] * 4

    def construct_shell_catalog(self, ishell, skip_existing=False, check_read_existing=True, **kwargs):
        """Construct and save a particle catalog for a given shell

        Parameters
        ----------
        ishell
            Shell index
        """
        isubs = self.reader.get_isubs_for_shell(ishell)

        fname = self.get_fname(ishell, what='haloes')
        
        if skip_existing and os.path.exists(fname):
            if check_read_existing:
                try:
                    return fits.open(fname)
                except:
                    pass
            else:
                return

        hdul = [fits.PrimaryHDU()]

        for isub in self.tqdm(isubs, desc=f"[construct_shell_catalog: shell {ishell}]"):
            res = self.construct_shell_sub_catalog(isub, **kwargs)
            idx = fits.Column(name='ihaloes', format='I', array=res[0])
            ra = fits.Column(name='ra', format='E', array=res[1])
            dec = fits.Column(name='dec', format='E', array=res[2])
            r = fits.Column(name='chi', format='E', array=res[3])
            a = pyccl.scale_factor_of_chi(self.cosmo, res[3])
            z = fits.Column(name='z', format='E', array=1. / a - 1.)
            header = fits.Header()
            header[f'ISUB'] = isub
            header[f'ZSNAP'] = self.reader.zsnapshots[ishell]
            hdul.append(fits.BinTableHDU.from_columns([idx, ra, dec, r, z], header))

        fits.HDUList(hdul).writeto(fname, overwrite=True)
        
        return hdul

    def read_shell_catalog_props(self, ishell, variables):
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

        for hdu in self.tqdm(hdul[1:], desc="[get_shell_catalog_props]"):
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


# @numba.njit(parallel=False)
# def _add_at_one(a, index):
#     n = len(index)
#     for i in numba.prange(n):
#         a[index[i]] += 1


@numba.njit(parallel=True)
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
    Affine transformation A(xyx+b)
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

    # get radius and remove particles out of shel
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
        ipix[i] = _vec2pix_sub(
            nside, order, ncap, npix, x[i], y[i], z[i], shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
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


# @numba.njit
# def _vec2pix_unique(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
#     ipix = _vec2pix(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only)
#     return unique_pix_pos(ipix, npix)

# def vec2pix_unique(nside, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only):
#     order = hp.nside2order(nside)
#     ncap = (nside * nside - nside) << 1
#     npix = hp.nside2npix(nside)
#     return _vec2pix_unique(nside, order, ncap, npix, x, y, z, shiftx, shifty, shiftz, rmin2, rmax2, octant_only)


@numba.njit
def construct_shell_sub(
        xyz, box_edges_original, A, b, nrepeat, boxsize, rmin2, rmax2, apply_random_isometry, octant_only, nside, order,
        ncap, npix):
    """
    Parallelized helper function for ShellBuilder.construct_shell_sub
    """
    n = len(xyz)

    # precompute isometry on particles (since repeats are then only a shift away)
    if apply_random_isometry:
        xyz = Axpb(A, b, xyz)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        box_edges = box_edges_original

    count = np.zeros(npix, dtype=np.uint)

    # repeats = []

    dtype = xyz.dtype

    # repeat the box
    for ix in range(-nrepeat, nrepeat + 1):
        for iy in range(-nrepeat, nrepeat + 1):
            for iz in range(-nrepeat, nrepeat + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_random_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                # apply isometry to box edges
                tmp_box_edges = shift + box_edges

                # check if box intersects octant, otherwise skip
                # (to do so, we check if the box is fully contained in the x<0 or y<0 or z<0 half spaces)
                if octant_only:
                    for i in range(3):
                        if np.all(tmp_box_edges[:, i] <= 0.):
                            continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(tmp_box_edges) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                sphere_center = -b if apply_random_isometry else np.zeros(3, dtype=dtype)
                if not _AABB_sphere_intersect(box_edges_original + shift_original, sphere_center, rmax2):
                    continue

                ## add particles to shell
                # compute corresponding healpy pixels of all particles (particles out of shell/octant quickly set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                ipix = _vec2pix(
                    nside, order, ncap, npix, xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2],
                    rmin2, rmax2, octant_only)

                for i in range(n):
                    pos = ipix[i]
                    if pos >= 0:
                        count[pos] += 1

    u = np.argwhere(count)[:, 0]
    return u, count[u]


@numba.njit
def construct_shell_sub_catalog(
        xyz, box_edges_original, A, b, nrepeat, boxsize, rmin2, rmax2, apply_random_isometry, octant_only):
    """
    Parallelized helper function for ShellBuilder.construct_shell_sub_catalog
    """
    n = len(xyz)

    # precompute isometry on particles (since repeats are then only a shift away)
    if apply_random_isometry:
        xyz = Axpb(A, b, xyz)
        box_edges = Axpb(A, b, box_edges_original)
    else:
        box_edges = box_edges_original

    idx = []
    ra = []
    dec = []
    r = []

    dtype = xyz.dtype

    # repeat the box
    for ix in range(-nrepeat, nrepeat + 1):
        for iy in range(-nrepeat, nrepeat + 1):
            for iz in range(-nrepeat, nrepeat + 1):
                # compute shift
                shift_original = np.expand_dims(np.array([ix, iy, iz]) * boxsize, 0).astype(dtype)
                if apply_random_isometry:
                    shift = Ax(A, shift_original)
                else:
                    shift = shift_original

                # apply isometry to box edges
                tmp_box_edges = shift + box_edges

                # check if intersects octant, otherwise skip
                if octant_only:
                    found_edge_in_octant = False
                    for e in tmp_box_edges:
                        if np.all(e > 0.):
                            found_edge_in_octant = True
                    if not found_edge_in_octant:
                        continue

                ## check if box intersects with shell, otherwise skip
                # first, check if box fully inside inner sphere
                if np.all(xyz2r2(tmp_box_edges) < rmin2):
                    continue

                # second, check if box intersects with outer sphere
                sphere_center = -b if apply_random_isometry else np.zeros(3, dtype=dtype)
                if not _AABB_sphere_intersect(box_edges_original + shift_original, sphere_center, rmax2):
                    continue

                # if catalog_mode:
                tmpidx, tmpra, tmpdec, tmpr = _vec2radec(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2], shift[0, 0], shift[0, 1], shift[0, 2], rmin2, rmax2, octant_only)

                w = tmpidx >= 0
                if np.any(w):
                    idx.append(tmpidx[w])
                    ra.append(tmpra[w])
                    dec.append(tmpdec[w])
                    r.append(tmpr[w])

    return idx, ra, dec, r
