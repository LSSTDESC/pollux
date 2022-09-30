import gc
import os
from selectors import EpollSelector

import healpy as hp
import healsparse
import matplotlib.pyplot as plt
import numpy as np
import pyccl
import pymaster as nmt
import scipy.stats
from tqdm.auto import tqdm, trange

from .utils import *


class ShellBuilder:

    def __init__(self,
                 reader,
                 cosmo,
                 nside,
                 save_dir=None,
                 apply_random_isometry=False,
                 seed=12345,
                 octant_only=True,
                 nside_c=32,
                 dtype=np.int64,
                 save_format='fits',
                 subsample_box=None,
                 z_subsample_auto=2.):
        """An object to handle the computation of particle shells from an N-body simulation.

        Parameters
        ----------
        reader
            N-body simulation reader.
        cosmo
            Cosmology object from CCL.
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
        self.cosmo = cosmo
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.save_dir = save_dir
        self.apply_random_isometry = apply_random_isometry
        self.seed = seed
        self.octant_only = octant_only
        self.nside_c = nside_c
        self.dtype = dtype
        self.save_format = save_format
        self.subsample_box = subsample_box
        self.z_subsample_auto = z_subsample_auto

        self.nshells = self.reader.nshells

        if apply_random_isometry:
            self.Abs = self.get_random_isometries(self.reader.get_boxsize(self.cosmo))

        # self.Abs = None
        self.wsp = None

        if dtype != np.int64:
            print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        assert subsample_box == 'auto' or type(subsample_box) is float or subsample_box is None
        if self.subsample_box=='auto':
            ngal_pix = np.array([self.estimate_repeats(ishell) for ishell in range(self.nshells)]) * self.reader.nparticles / (hp.nside2npix(
                self.nside))  # estimated number of galaxies per pixel
            if self.octant_only:
                ngal_pix *= 8.
            snr_pix = 1. / np.sqrt(ngal_pix)  # S/N per pixel
            snr_pix_cut = np.interp(self.z_subsample_auto, self.reader.zsnapshots,
                                    snr_pix)  # S/N at redshift z_subsample_auto
            self.subsample_ratios = np.minimum(1., (snr_pix / snr_pix_cut)**2) # fraction of particles to maintain S/N of z>z_subsample_auto

    def get_fname_sub(self, isub):
        return os.path.join(
            self.save_dir,
            f'shell_{self.reader.kind}_z{self.reader.get_ishell(isub)}_sub{isub}_nside{self.nside}_seed{self.seed}.' +
            self.save_format)

    def get_fname(self, ishell):
        return os.path.join(self.save_dir,
                            f'shell_{self.reader.kind}_z{ishell}_nside{self.nside}_seed{self.seed}.' + self.save_format)

    def get_random_isometries(self, size):
        """Generates random isometries for each shell.

        Returns
        -------
            A list of (A,b) with an isometry A (rotations and flips) and translation by shift b.
        """
        # s = np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Abs = []
        for _ in range(self.nshells):
            # Abs.append(random_isometry(s, size))
            Abs.append(random_isometry(rng, size))
        self.Abs = Abs
        return Abs

    def estimate_repeats(self, ishell):
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        box_volume = self.reader.get_boxsize(self.cosmo)**3

        shell_volume = 4. / 3. * np.pi * (com_shell_bins[ishell + 1]**3 - com_shell_bins[ishell]**3)
        if self.octant_only:
            shell_volume /= 8.

        return shell_volume / box_volume

    def get_subsample_ratio(self, ishell):
        if self.subsample_box == 'auto':
            ratio = self.subsample_ratios[ishell]
        elif type(self.subsample_box) is float:
            ratio = self.subsample_box
        elif self.subsample_box is None:
            ratio = 1.
        return ratio

    def construct_shell_sub(self,
                            isub,
                            Abs=None,
                            com_shell_bins=None,
                            skip_existing=False,
                            verbose=False,
                            debug=False,
                            repeat_ixyz=None,
                            get_repeats_only=False):
        """Constructs a single shell from a simulation sub-volume/snapshot, applying isometries and repeating it as necessary.

        Parameters
        ----------
        isub
            Index of the sub-volume element and the snapshot. This should be computed automatically.
        Abs, optional
            Random isometries, if pre-computed, by default None
        com_shell_bins, optional
            Comoving distance of shell edges, if pre-computed, by default None
        skip_existing, optional
            Skips the computation if a saved file was found, by default False
        verbose, optional
            Prints feedback, by default False
        debug, optional
            Enables debugging features, by default False
        repeat_ixyz, optional
            Tuple (ix,iy,iz) of the specific repeat to compute, by default None
        get_repeats_only, optional
            If True, returns the list of repeat tuples (ix,iy,iz) where the sub-volume and the shell intersect, by default False

        Returns
        -------
            Filenames and shell index.
        """

        ishell = self.reader.get_ishell(isub)

        # check if shell exists and/or can be saved
        if self.save_dir is not None:
            fname = self.get_fname_sub(isub)

            # make sure folder exists
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            # check if shell has already been created
            if skip_existing and os.path.exists(fname):
                if verbose:
                    print(f"Found shell for box {isub} (shell {ishell}) at {fname}.")
                return fname, ishell

        # get shell radii
        if com_shell_bins is None:
            com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        assert self.nshells == len(com_shell_bins) - 1
        rmin2, rmax2 = com_shell_bins[ishell]**2, com_shell_bins[ishell + 1]**2

        # get random isometries
        if self.apply_random_isometry:
            if Abs is None:
                Abs = self.Abs
            else:
                Abs = Abs

        # nrepeat = int(np.ceil(com_shell_bins[-1]/reader.boxsize)) + 2 # not too sure about that 2, but it's fast to check...
        nrepeat = int(np.ceil(np.sqrt(rmax2) / self.reader.get_boxsize(
            self.cosmo))) + 2  # not too sure about that 2, but it's fast to check...

        # read box
        xyz, box_edges = self.reader.read_sub(isub,
                                              read_box_edges_only=get_repeats_only,
                                              h_to_divide_by=None if self.reader.unit == 'Mpc' else self.cosmo["h"])
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

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

        random_isometry_applied_to_xyz = False
        
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
                    shift = (np.array([ix, iy, iz]) * self.reader.get_boxsize(self.cosmo))[None, :]

                    # apply isometry to box edges
                    if self.apply_random_isometry:
                        tmp_box_edges = Axpb(*Abs[ishell], shift + box_edges)
                    else:
                        tmp_box_edges = shift + box_edges

                    # check if intersects octant, otherwise skip
                    if self.octant_only and np.any(np.all(tmp_box_edges < 0., axis=0)):  #np.all(tmp_box_edges < 0.):
                        continue

                    ## check if box intersects with shell, otherwise skip
                    # first, check if box fully inside inner sphere
                    if np.all(xyz2r2(tmp_box_edges) < rmin2):
                        continue

                    # second, check if box intersects with outer sphere
                    if self.apply_random_isometry:
                        s = -shift - Abs[ishell][1] # sphere center, with box aligned wrt xyz axis
                    else:
                        s = -shift
                    if not AABB_sphere_intersect(box_edges, s[0], rmax2):
                        continue

                    # if we got this far, then this repeat intersects with the shell
                    repeats.append([ix, iy, iz])
                    if verbose:
                        print(f"Found overlap for box {isub} (shell {ishell}) for repeat {(ix,iy,iz)}")
                    if get_repeats_only:
                        continue

                    ## now, add particles to shell
                    # precompute isometry on particles (since repeats are then only a shift away)
                    if (not random_isometry_applied_to_xyz) and self.apply_random_isometry:
                        xyz = Axpb(*Abs[ishell], xyz)
                        random_isometry_applied_to_xyz = True
                    
                    if self.apply_random_isometry:
                        shift = Ax(Abs[ishell][0], shift)
                    
                    # compute unique pixels into which particles fall and their counts, all done in vec2pix_unique (using numba) as follows:
                    # 1 - remove particles out of shell based on radius
                    # 2 - remove particles out of octant if octant_only
                    # 3 - compute corresponding healpy pixels of all particles (removed particles fast set to -1) with vec2pix and loc2pix (translated into python from Healpix code)
                    # 4 - get unique pixels and counts (ignoring -1's)
                    u, c = vec2pix_unique(self.nside, *(xyz.T), *(shift[0].T), rmin2, rmax2, self.octant_only)

                    if len(u)>0:
                        if verbose:
                            print(f"Found {len(u)} particles.")
                    
                        # add particles to shell
                        tmp_shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                        tmp_shell.update_values_pix(u, c)

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

        if debug:
            return tmp_box_edges, tmp_xyz, shell

        return fname, ishell

    def construct_shells(self, job_id=None, **kwargs):
        """Computes all shells.

        Parameters
        ----------
        job_id, optional
            Index of sub-volume/snapshot list to compute, by default None
        """
        if job_id is None:
            # Run all files
            it = range(self.reader.nsubs)
        else:
            it = self.reader.jobs[job_id]

        for isub in tqdm(it, desc='[construct_shells]'):
            _ = self.construct_shell_sub(isub, **kwargs)

    def get_repeats(self, **kwargs):
        """Computes the list of repeat indices (ix,iy,iz) for all the sub-volumes and snapshots. 

        Returns
        -------
            List of repeat indices.
        """

        repeats = []
        for isub in trange(self.reader.nsubs, desc='[get_repeats]'):
            repeats.append((
                isub,
                self.construct_shell_sub(
                    isub,
                    get_repeats_only=True,
                    **kwargs)))

        return repeats

    def combine_shells(self, skip_existing=False):
        """Sums the particles shells computed independently for each sub-volume/snapshot with construct_shell_sub
        """
        for ishell in trange(self.nshells, desc="[combine_shells]"):
            fname = self.get_fname(ishell)
            if skip_existing and os.path.exists(fname):
                continue
            
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            
            isubs = self.reader.get_isubs_for_shell(ishell)
            for isub in tqdm(isubs, desc=f"[combine_shells: shell {ishell}]", leave=False):
                tmp_fname = self.get_fname_sub(isub)
                tmp_shell = healsparse.HealSparseMap.read(tmp_fname)
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            shell.write(fname, clobber=True, format=self.save_format)

    def load_sub_shells(self, ishell):
        """Loads shell maps of sub-volumes

        Parameters
        ----------
        ishell
            Index of shell.

        Returns
        -------
            List of shells as HealSparseMap objects.
        """
        isubs = self.reader.get_isubs_for_shell(ishell)

        shells = []
        for isub in tqdm(isubs, desc="[load_sub_shells]"):
            fname = self.get_fname_sub(isub)
            shells.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.nside_c))
        return shells

    def load_shells(self, ishells=None):
        """Loads shell maps.

        Parameters
        ----------
        ishells, optional
            Indices of shells, by default None

        Returns
        -------
            List of shells as HealSparseMap objects.
        """
        if ishells is None:
            ishells = range(self.nshells)

        shells = []
        for ishell in tqdm(ishells, desc="[load_shells]"):
            fname = self.get_fname(ishell)
            shells.append(healsparse.HealSparseMap.read(fname, nside_coverage=self.nside_c))
        return shells

    def plot_shells(self, shells, nx=None, log=True, title_prefix=None, cbar=False, **kwargs):
        """Plots particle shells.

        Parameters
        ----------
        shells
            List of shells.
        nx, optional
            Number of maps to plot per row, by default None
        log, optional
            Plots log10(shells), by default True
        title_prefix, optional
            Title prefix, by default None
        cbar, optional
            Whether to show color bars, by default False
        """
        nshells = len(shells)
        if nx is None:
            nx = nshells
        ny = int(np.ceil(nshells / nx))

        plt.figure(figsize=(4 * nx, 4 * ny))

        for ishell in range(nshells):
            hp_map = shells[ishell].generate_healpix_map()
            if log and np.any(hp_map > 0.):
                hp_map = np.log10(hp_map)
            title = title_prefix + f" {ishell}" if title_prefix is not None else None
            if self.octant_only:
                hp.orthview(hp_map,
                            rot=(35, 20),
                            half_sky=True,
                            sub=(ny, nx, ishell + 1),
                            title=title,
                            cbar=cbar,
                            **kwargs)
            else:
                hp.mollview(hp_map, sub=(ny, nx, ishell + 1), title=title, cbar=cbar, **kwargs)
        hp.graticule(verbose=False)


    def get_box_comoving_density(self,ishell):
        return self.reader.get_box_comoving_density(self.cosmo) * self.get_subsample_ratio(ishell)
        
    def get_pixel_density(self, ishell):
        """Computes the expected mean number of particle per pixels from the simulation density.

        Parameters
        ----------
        ishell
            Index of the shell.
        cosmo
            Cosmology object from CCL.
        nside
            Resolution parameter nside of the shell.

        Returns
        -------
            Mean number of particles per pixel.
        """
        npix = hp.nside2npix(self.nside)
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)

        chi_max = com_shell_bins[ishell + 1]
        chi_min = com_shell_bins[ishell]
        pixel_volume = 4. * np.pi / 3. * (chi_max**3 - chi_min**3) / npix

        return pixel_volume * self.get_box_comoving_density(ishell)


    def get_mask(self):
        if self.octant_only:
            mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90)
        else:
            mask = np.ones(hp.nside2npix(self.nside))
        return mask    
        
    def get_cls_wsp(self, b=None, nlb=20):
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
        mask = self.get_mask()

        if b is None:
            b = make_nmtbin_logspaced(self.nside, 8, 2 * self.nside, nlb, f_ell='pixwin')

        fname = os.path.join(self.save_dir, f'wsp_{self.reader.kind}_nside{self.nside}.wsp')

        wsp = nmt.NmtWorkspace()
        if os.path.exists(fname):
            print("Loading workspace in", fname)
            wsp.read_from(fname)
        else:
            print("Computing and saving workspace in", fname)
            dummy_f = nmt.NmtField(mask, [np.zeros_like(mask)])
            wsp.compute_coupling_matrix(dummy_f, dummy_f, b)
            wsp.write_to(fname)

        return wsp, mask, b

    def compute_shells_cls(self, shells=None, ishells=None, b=None, nlb=20):
        """Computes the angular power spectra of overdensity maps defined from particle shells.

        Parameters
        ----------
        shells, optional
            List of particle shells, by default None
        ishells, optional
            Indices of particle shells, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50


        Returns
        -------
            Multipole bin centers and list of spectra.
        """
        wsp, mask, b = self.get_cls_wsp(b=b, nlb=nlb)

        if shells is None:
            shells = self.load_shells(ishells)

        cls_out = []
        for ishell, shell in enumerate(tqdm(shells, desc="[compute_shells_cls]")):
            s = np.clip(shell.generate_healpix_map(), 0, np.inf)  # remove potential hp.UNSEEN for empty pixels in mask
            field = nmt.NmtField(mask, [s / self.get_pixel_density(ishell) - 1.])
            cls_out.append(nmt.compute_full_master(field, field, b, workspace=wsp)[0])

        return b.get_effective_ells(), cls_out

    def predict_shells_cls(self, ishells=None, b=None, nlb=20, **kwargs):
        """Computes theory angular power spectra of overdensity maps with CCL.

        Parameters
        ----------
        ishells, optional
            Indices of particle shells, by default None
        b, optional
            Namaster binning scheme, by default None
        nlb, optional
            Number of multipole bins, by default 50

        Returns
        -------
            Multipole bin centers and list of spectra.
        """
        if ishells is None:
            ishells = range(self.nshells)

        _, _, b = self.get_cls_wsp(b=b, nlb=nlb)

        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        z_shell_bins = 1. / pyccl.scale_factor_of_chi(self.cosmo, com_shell_bins) - 1.

        zz = np.linspace(0, z_shell_bins[-1], 10000)
        dls = [
            pyccl.NumberCountsTracer(
                self.cosmo, False,
                dndz=(zz,
                 scipy.stats.uniform.pdf(
                     zz, loc=z_shell_bins[ishell], scale=z_shell_bins[ishell + 1] - z_shell_bins[ishell])),
                bias=(zz, np.ones_like(zz))) for ishell in range(self.nshells)
        ]

        # TODO : use wsp to get exact convolution, if actually useful for validation
        
        # return dls
        ell = b.get_effective_ells()
        Cls = []
        for dl in tqdm(dls):
            cl = pyccl.angular_cl(self.cosmo, dl, dl, ell, **kwargs)
            Cls.append(cl)

        return ell, Cls
