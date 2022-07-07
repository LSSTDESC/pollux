import os

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
                 save_format='fits'):
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
        self.save_dir = save_dir
        self.apply_random_isometry = apply_random_isometry
        self.seed = seed
        self.octant_only = octant_only
        self.nside_c = nside_c
        self.dtype = dtype
        self.save_format = save_format

        self.nshells = self.reader.nshells

        self.Abs = None
        self.wsp = None

        if dtype != np.int64:
            print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

    def get_fname_sub(self, isub):
        return os.path.join(
            self.save_dir,
            f'shell_{self.reader.kind}_z{self.reader.get_ishell(isub)}_sub{isub}_nside{self.nside}_seed{self.seed}.' +
            self.save_format)

    def get_fname(self, ishell):
        return os.path.join(self.save_dir,
                            f'shell_{self.reader.kind}_z{ishell}_nside{self.nside}_seed{self.seed}.' + self.save_format)

    def get_random_isometries(self):
        """Generates random isometries for each shell.

        Returns
        -------
            A list of (A,b) with an isometry A (rotations and flips) and translation by shift b.
        """        
        s = np.random.seed(self.seed)
        Abs = []
        for _ in range(self.nshells):
            Abs.append(random_isometry(s, self.reader.boxsize))
        self.Abs = Abs
        return Abs

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
            if self.Abs is None and Abs is None:
                Abs = self.get_random_isometries()
            else:
                Abs = self.Abs

        # nrepeat = int(np.ceil(com_shell_bins[-1]/reader.boxsize)) + 2 # not too sure about that 2, but it's fast to check...
        nrepeat = int(np.ceil(
            np.sqrt(rmax2) / self.reader.boxsize)) + 2  # not too sure about that 2, but it's fast to check...

        # read box
        xyz, box_edges = self.reader.read_sub(isub, read_box_edges_only=debug or get_repeats_only)
        if verbose:
            print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

        shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)

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
                    shift = (np.array([ix, iy, iz]) * self.reader.boxsize)[None, :]

                    # apply isometry to box edges
                    if self.apply_random_isometry:
                        tmp_box_edges = Axpb(*Abs[ishell], shift + box_edges)
                    else:
                        tmp_box_edges = shift + box_edges

                    # check if intersects octant, otherwise skip
                    if self.octant_only and np.all(tmp_box_edges < 0.):
                        continue

                    ## check if box intersects with shell, otherwise skip
                    # first, check if box fully inside inner sphere
                    if np.all(xyz2r2(tmp_box_edges) < rmin2):
                        continue

                    # second, check if box intersects with outer sphere
                    s = -shift - Abs[ishell][1] if self.apply_random_isometry else -shift  # sphere center in box frame
                    if not AABB_sphere_intersect(box_edges, s[0], rmax2):
                        continue

                    # if we got this far, then this repeat intersects with the shell
                    repeats.append([ix, iy, iz])
                    if verbose:
                        print(f"Found overlap for box {isub} (shell {ishell}) for repeat {(ix,iy,iz)}")
                    if get_repeats_only:
                        continue

                    ## now, add particles to shell
                    # apply isometry to particles
                    if self.apply_random_isometry:
                        temp_xyz = Axpb(*Abs[ishell], xyz + shift)
                    else:
                        temp_xyz = xyz + shift

                    # filter particles out of shell
                    r2 = xyz2r2(temp_xyz)
                    w = (r2 >= rmin2) & (r2 < rmax2)

                    if self.octant_only:
                        w &= np.all(temp_xyz >= 0., axis=1)

                    if np.any(w):
                        # get corresponding healpy pixels
                        ipix = hp.vec2pix(self.nside, *(temp_xyz[w].T))
                        u, c = np.unique(ipix, return_counts=True)

                        # add particles to shell
                        tmp_shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
                        tmp_shell.update_values_pix(u, c)

                        shell = healsparse.operations.sum_union([shell, tmp_shell])

        if verbose:
            print(f"Found overlap for box {isub} (shell {ishell}) for {len(repeats)} repeats")

        if get_repeats_only:
            return repeats

        if self.save_dir is not None:
            shell.write(fname, clobber=True, format=self.save_format)

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

        Abs = self.get_random_isometries() if self.apply_random_isometry else None
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)

        for isub in tqdm(it, desc='[construct_shells]'):
            _ = self.construct_shell_sub(isub, Abs=Abs, com_shell_bins=com_shell_bins, **kwargs)

    def get_repeats(self, **kwargs):
        """Computes the list of repeat indices (ix,iy,iz) for all the sub-volumes and snapshots. 

        Returns
        -------
            List of repeat indices.
        """        
        Abs = self.get_random_isometries() if self.apply_random_isometry else None
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)

        repeats = []
        for isub in trange(self.reader.nsubs, desc='[get_repeats]'):
            repeats.append((isub,
                            self.construct_shell_sub(isub,
                                                     Abs=Abs,
                                                     com_shell_bins=com_shell_bins,
                                                     get_repeats_only=True,
                                                     **kwargs)))

        return repeats

    def combine_shells(self):
        """Sums the particles shells computed independently for each sub-volume/snapshot with construct_shell_sub
        """        
        for ishell in trange(self.nshells, desc="[combine_shells]"):
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            isubs = self.reader.get_isubs_for_shell(ishell)
            for isub in isubs:
                fname = self.get_fname_sub(isub)
                tmp_shell = healsparse.HealSparseMap.read(fname)
                shell = healsparse.operations.sum_union([shell, tmp_shell])

            shell.write(self.get_fname(ishell), clobber=True, format=self.save_format)

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

    def get_cls_wsp(self, b=None, nlb=50):
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
        if self.octant_only:
            mask = make_mask_radec_cuts(self.nside, 0, 90, 0, 90)
        else:
            mask = np.ones(hp.nside2npix(self.nside))

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

    def compute_shells_cls(self, shells=None, ishells=None, b=None, nlb=50):
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

        # shells_hp = [shell.generate_healpix_map() for shell in shells]

        # fields = [nmt.NmtField(mask, [s/np.mean(s[mask.astype(bool)])-1.]) for s in tqdm(shells_hp)] # TODO: get mean from number of particles and shell volume
        # fields = [nmt.NmtField(mask, [s/self.reader.get_pixel_density(ishell, self.cosmo, self.nside) - 1.]) for ishell, s in enumerate(tqdm(shells_hp))]

        # return b.get_effective_ells(), [nmt.compute_full_master(f, f, b, workspace=wsp)[0] for f in fields]

        cls_out = []
        for ishell, shell in enumerate(tqdm(shells, desc="[compute_shells_cls]")):
            s = np.clip(shell.generate_healpix_map(), 0, np.inf)  # remove potential hp.UNSEEN for empty pixels in mask
            field = nmt.NmtField(mask, [s / self.reader.get_pixel_density(ishell, self.cosmo, self.nside) - 1.])
            cls_out.append(nmt.compute_full_master(field, field, b, workspace=wsp)[0])

        return b.get_effective_ells(), cls_out

    def predict_shells_cls(self, ishells=None, b=None, nlb=50):
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

        zz = np.linspace(0, z_shell_bins[-1], 1000)
        dls = [
            pyccl.NumberCountsTracer(
                self.cosmo, False,
                (zz,
                 scipy.stats.uniform.pdf(
                     zz, loc=z_shell_bins[ishell], scale=z_shell_bins[ishell + 1] - z_shell_bins[ishell])),
                (zz, np.ones_like(zz))) for ishell in range(self.nshells)
        ]

        # TODO : use wsp to get exact convolution, if actually useful for validation

        return b.get_effective_ells(), [pyccl.angular_cl(self.cosmo, dl, dl, b.get_effective_ells()) for dl in dls]


# def get_fname_sub(save_dir, reader, isub, nside, seed, save_format):
#     return os.path.join(save_dir, f'shell_{reader.kind}_z{reader.get_ishell(isub)}_sub{isub}_nside{nside}_seed{seed}.'+save_format)

# def get_fname(save_dir, reader, ishell, nside, seed, save_format):
#     return os.path.join(save_dir, f'shell_{reader.kind}_z{ishell}_nside{nside}_seed{seed}.'+save_format)

# def get_random_isometries(seed, nshells, reader):
#     s = np.random.seed(seed)
#     Abs = []
#     for _ in range(nshells):
#         Abs.append(random_isometry(s, reader.boxsize))
#     return Abs

# def construct_shell_sub(reader, isub, cosmo, nside,
#                         save_dir=None,
#                         apply_random_isometry=False,
#                         seed=12345,
#                         Abs=None,
#                         com_shell_bins=None,
#                         octant_only=True,
#                         nside_c=32,
#                         dtype=np.int64,
#                         save_format='fits',
#                         skip_existing=False,
#                         verbose=False,
#                         debug=False,
#                         repeat_ixyz=None,
#                         get_repeats_only=False):

#     if dtype != np.int64:
#         print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

#     ishell = reader.get_ishell(isub)

#     # check if shell exists and/or can be saved
#     if save_dir is not None:
#         fname = get_fname_sub(save_dir, reader, isub, nside, seed, save_format) #os.path.join(save_dir, f'shell_z{ishell}_sub{isub}_nside{nside}_seed{seed}.'+save_format)

#         # make sure folder exists
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)

#         # check if shell has already been created
#         if skip_existing and os.path.exists(fname):
#             if verbose:
#                 print(f"Found shell for box {isub} (shell {ishell}) at {fname}.")
#             return fname, ishell

#     nshells = len(reader.zsnapshots)

#     # get shell radii
#     if com_shell_bins is None:
#         com_shell_bins = reader.get_com_shell_bins(cosmo)
#     assert nshells == len(com_shell_bins)-1
#     rmin2, rmax2 = com_shell_bins[ishell]**2, com_shell_bins[ishell+1]**2

#     # get random isometries
#     if apply_random_isometry:
#         if Abs is None:
#             Abs = get_random_isometries(seed, nshells, reader)

#     # nrepeat = int(np.ceil(com_shell_bins[-1]/reader.boxsize)) + 2 # not too sure about that 2, but it's fast to check...
#     nrepeat = int(np.ceil(np.sqrt(rmax2)/reader.boxsize)) + 2 # not too sure about that 2, but it's fast to check...

#     # read box
#     xyz, box_edges = reader.read_sub(isub, read_box_edges_only=debug or get_repeats_only)
#     if verbose:
#         print(f"Loaded box {isub} (shell {ishell}) with {len(xyz)} particles.")

#     shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)

#     repeats = []
#     # repeat the box
#     for ix in range(-nrepeat, nrepeat+1):
#         for iy in range(-nrepeat, nrepeat+1):
#             for iz in range(-nrepeat, nrepeat+1):
#                 # check if we want to do only one repeat
#                 if repeat_ixyz is not None:
#                     if not np.array_equal(repeat_ixyz, [ix, iy, iz]):
#                         continue

#                 # compute shift
#                 shift = (np.array([ix, iy, iz])*reader.boxsize)[None,:]

#                 # apply isometry to box edges
#                 if apply_random_isometry:
#                     tmp_box_edges = Axpb(*Abs[ishell], shift + box_edges)
#                 else:
#                     tmp_box_edges = shift + box_edges

#                 # check if intersects octant, otherwise skip
#                 if octant_only and np.all(tmp_box_edges<0.):
#                     continue

#                 ## check if box intersects with shell, otherwise skip
#                 # first, check if box fully inside inner sphere
#                 if np.all(xyz2r2(tmp_box_edges)<rmin2):
#                     continue

#                 # second, check if box intersects with outer sphere
#                 s = -shift-Abs[ishell][1] if apply_random_isometry else -shift # sphere center in box frame
#                 if not AABB_sphere_intersect(box_edges, s[0], rmax2):
#                     continue

#                 # if we got this far, then this repeat intersects with the shell
#                 repeats.append([ix,iy,iz])
#                 if verbose:
#                     print(f"Found overlap for box {isub} (shell {ishell}) for repeat {(ix,iy,iz)}")
#                 if get_repeats_only:
#                     continue

#                 ## now, add particles to shell
#                 # apply isometry to particles
#                 if apply_random_isometry:
#                     temp_xyz = Axpb(*Abs[ishell], xyz + shift)
#                 else:
#                     temp_xyz = xyz + shift

#                 # filter particles out of shell
#                 r2 = xyz2r2(temp_xyz)
#                 w = (r2 >= rmin2) & (r2 < rmax2)

#                 if octant_only:
#                     w &= np.all(temp_xyz>=0., axis=1)

#                 if np.any(w):
#                     # get corresponding healpy pixels
#                     ipix = hp.vec2pix(nside, *(temp_xyz[w].T))
#                     u, c = np.unique(ipix, return_counts=True)

#                     # add particles to shell
#                     tmp_shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#                     tmp_shell.update_values_pix(u, c)

#                     shell = healsparse.operations.sum_union([shell, tmp_shell])

#     if verbose:
#         print(f"Found overlap for box {isub} (shell {ishell}) for {len(repeats)} repeats")

#     if get_repeats_only:
#         return repeats

#     if save_dir is not None:
#         shell.write(fname, clobber=True, format=save_format)

#     return fname, ishell

# def get_repeats(reader, cosmo, nside, seed=12345, apply_random_isometry=False, **kwargs):
#     Abs = get_random_isometries(seed, len(reader.zsnapshots), reader) if apply_random_isometry else None
#     com_shell_bins = reader.get_com_shell_bins(cosmo)

#     repeats = []
#     for isub in trange(reader.nsubs, desc='Constructing shells'):
#         repeats.append((isub,construct_shell_sub(reader, isub, cosmo, nside, Abs=Abs, com_shell_bins=com_shell_bins, get_repeats_only=True, **kwargs)))

#     return repeats

# def construct_shells(reader, cosmo, nside, job_id=None, seed=12345, apply_random_isometry=False, **kwargs):
#     if job_id is None:
#         # Run all files
#         it = range(reader.nsubs)
#     else:
#         it = reader.jobs[job_id]

#     Abs = get_random_isometries(seed, len(reader.zsnapshots), reader) if apply_random_isometry else None
#     com_shell_bins = reader.get_com_shell_bins(cosmo)

#     for isub in tqdm(it, desc='Constructing shells'):
#         _ = construct_shell_sub(reader, isub, cosmo, nside, Abs=Abs, com_shell_bins=com_shell_bins, **kwargs)

# def combine_shells(reader, nside, save_dir, seed=12345, nside_c=32, dtype=np.int64, save_format='fits'):
#     for ishell in trange(reader.nshells, desc="Combining shells"):
#         shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#         isubs = reader.get_isubs_for_shell(ishell)
#         for isub in isubs:
#             fname = get_fname_sub(save_dir, reader, isub, nside, seed, save_format)
#             tmp_shell = healsparse.HealSparseMap.read(fname)
#             shell = healsparse.operations.sum_union([shell, tmp_shell])

#         shell.write(os.path.join(save_dir, f'shell_z{isub}'+save_format), clobber=True, format=save_format)

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

# def construct_shell_job(reader, ijob, cosmo, nside, save_dir, apply_random_isometry=False, seed=12345, octant_only=True, nside_c=32, dtype=np.int64, save_format='fits', skip_existing=False, verbose=False):
#     for isub in tqdm(reader.jobs[ijob], desc='Constructing shells'):
#         _ = construct_shell_sub(reader, isub, cosmo, nside, save_dir=save_dir, apply_random_isometry=apply_random_isometry, seed=seed, octant_only=octant_only, nside_c=nside_c, dtype=dtype, save_format=save_format, skip_existing=skip_existing, verbose=verbose)

# def construct_shell_mpi(reader, cosmo, nside, save_dir, apply_random_isometry=False, seed=12345, octant_only=True, nside_c=32, dtype=np.int64, save_format='fits', skip_existing=False, verbose=False):
#     from mpi4py import MPI
#     # from mpipool import MPIPool

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     print(rank, size)

#     def func(isub):
#         # return reader.kind
#         return construct_shell_sub(reader, isub, cosmo, nside, save_dir=save_dir, apply_random_isometry=apply_random_isometry, seed=seed, octant_only=octant_only, nside_c=nside_c, dtype=dtype, save_format=save_format, skip_existing=skip_existing, verbose=verbose)

#     if rank == 0
#     while True:

#     # with pool:
#     #     pool.workers_exit() # workers exit here, only master goes on

#     #     # Build sub shells
#     #     res = pool.map_async(func, range(reader.nsubs)).get()
#     #     print(res)

#     # comm.Barrier()

#         # # Combine sub shells
#         # for ishell in trange(reader.nshells, desc="Combining shells"):
#         #     shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#         #     isubs = reader.get_isubs_for_shell(ishell)
#         #     for isub in isubs:
#         #         fname = get_fname_sub(save_dir, reader, isub, nside, seed, save_format)
#         #         tmp_shell = healsparse.HealSparseMap.read(fname)
#         #         shell = healsparse.operations.sum_union([shell, tmp_shell])

#         #     shell.write(os.path.join(save_dir, f'shell_z{isub}'+save_format), clobber=True, format=save_format)

#     # TODO : find a better way to split jobs
#     # if rank == 0:
#     #     print("test", rank)
#     #     data = comm.scatter(range(reader.nsubs), root=0)
#     # else:
#     #     data = None

#     # fname = f"test_{data}"

#     # fnames = comm.gather(fname, root=0)

#     # if rank == 0:
#     #     print(fnames)

#     # print(rank, reader.jobs[rank])

#     # fnames_ = []
#     # for isub in tqdm(reader.jobs[rank], desc='Constructing shells'):
#     #     # fname, _ = construct_shell_sub(reader, isub, cosmo, nside, save_dir=save_dir, apply_random_isometry=apply_random_isometry, seed=seed, octant_only=octant_only, nside_c=nside_c, dtype=dtype, save_format=save_format, skip_existing=skip_existing, verbose=verbose)
#     #     fname = f"test_blablah_{isub}"
#     #     fnames_.append(fname)

#     # fnames = list(chain(comm.gather(fnames_, root=0)))

#     # if rank == 0:
#     #     print(fnames)
#     #     # Combine sub shells
#     #     for ishell in trange(reader.nshells, desc="Combining shells"):
#     #         shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#     #         isubs = reader.get_isubs_for_shell(ishell)
#     #         for isub in isubs:
#     #             # fname = fnames[isub]
#     #             fname = get_fname_sub(save_dir, reader, isub, nside, seed, save_format)
#     #             tmp_shell = healsparse.HealSparseMap.read(fname)
#     #             shell = healsparse.operations.sum_union([shell, tmp_shell])

#     #         shell.write(get_fname(save_dir, reader, ishell, nside, seed, save_format), clobber=True, format=save_format)

#     # from mpi4py.futures import MPIPoolExecutor

#     # executor = MPIPoolExecutor()

#     # def func(i):
#     #     print("task", i)
#     #     return f"test_blablah_{i}"

#     # # res = executor.map(func, range(reader.nsubs))

#     # # if rank == 0:
#     # #     print(res)

#     # from mpipool import MPIPool

#     # with MPIPool() as pool:
#     #     pool.workers_exit()
#     #     res = pool.map_async(func, range(10)).get() # reader.nsubs))
#     #     print(res)

#     # if job_id is None:
#     #     # Run all files
#     #     it = range(reader.nsubs)
#     # else:
#     #     it = reader.jobs[job_id]

#     # fnames = []
#     # for isub in trange(reader.nsubs, desc='Constructing shells'):
#     #     fname, _ = construct_shell_sub(reader, isub, cosmo, nside, save_dir=save_dir, apply_random_isometry=apply_random_isometry, seed=seed, octant_only=octant_only, nside_c=nside_c, dtype=dtype, save_format=save_format, skip_existing=skip_existing, verbose=verbose)
#     #     fnames.append(fname)

#     # if job_id is None:
#     #     # Combine sub shells
#     #     for ishell in trange(reader.nshells, desc="Combining shells"):
#     #         shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#     #         isubs = reader.get_isubs_for_shell(ishell)
#     #         for isub in isubs:
#     #             fname = fnames[isub]
#     #             tmp_shell = healsparse.HealSparseMap.read(fname)
#     #             shell = healsparse.operations.sum_union([shell, tmp_shell])

#     #         shell.write(os.path.join(save_dir, f'shell_z{isub}'+save_format), clobber=True, format=save_format)

# def construct_shell(reader, cosmo, nside, save_dir, apply_random_isometry=False, seed=12345, octant_only=True, nside_c=32, dtype=np.int64, save_format='fits'):#, z_shells, nsub, cosmo, nside, boxsize, debug=False, apply_random_isometry=False, seed=12345, z_start=0, **kwargs):

#     if dtype != np.int64:
#         print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

#     com_shell_bins = reader.get_com_shell_bins(cosmo)
#     nshells = len(com_shell_bins)-1

#     # Get random isometries
#     if apply_random_isometry:
#         s = np.random.seed(seed)
#         Abs = []
#         for _ in range(nshells):
#             Abs.append(random_isometry(s, reader.boxsize))

#     nrepeat = int(np.ceil(com_shell_bins[-1]/reader.boxsize)) + 1

#     for isub in trange(reader.nsubs, desc='Constructing shells'):
#         # read box
#         xyz, box_edges = reader.read_sub(isub)

#         for ishell in trange(nshells):
#             shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)

#             rmin2, rmax2 = com_shell_bins[ishell]**2, com_shell_bins[ishell+1]**2

#             # repeat the box
#             for ix in range(-nrepeat, nrepeat):
#                 for iy in range(-nrepeat, nrepeat):
#                     for iz in range(-nrepeat, nrepeat):
#                         # compute shift
#                         shift = (np.array([ix, iy, iz])*reader.boxsize)[None,:]

#                         # get transformed box edges
#                         if apply_random_isometry:
#                             tmp_box_edges = Axpb(*Abs[ishell], shift + box_edges)
#                         else:
#                             tmp_box_edges = shift + box_edges

#                         # check if intersects octant, otherwise skip
#                         if octant_only and np.all(tmp_box_edges<0.):
#                             continue

#                         ## check if box intersects with shell, otherwise skip
#                         # first, check if box fully inside inner sphere
#                         if np.all(xyz2r2(tmp_box_edges)<rmin2):
#                             continue

#                         # second, check if box intersects with outer sphere
#                         s = -shift-Abs[ishell][1] if apply_random_isometry else -shift # sphere center in box frame
#                         if AABB_sphere_intersect(box_edges, s, rmax2):
#                             continue

#                         ## otherwise add particles to shell
#                         # apply shift
#                         if apply_random_isometry:
#                             temp_xyz = Axpb(*Abs[ishell], xyz + shift)
#                         else:
#                             temp_xyz = xyz + shift

#                         # filter particles out of shell
#                         r2 = xyz2r2(temp_xyz)
#                         w = (r2 >= rmin2) & (r2 < rmax2)

#                         if octant_only:
#                             w &= np.all(temp_xyz>=0., axis=1)

#                         # get corresponding healpy pixels
#                         ipix = hp.vec2pix(nside, *(temp_xyz[w].T))
#                         u, c = np.unique(ipix, return_counts=True)

#                         # add particles to shell
#                         tmp_shell = healsparse.HealSparseMap.make_empty(nside_c, nside, dtype=dtype)
#                         tmp_shell.update_values_pix(u, c)

#                         shell += tmp_shell

#             shell.write(os.path.join(save_dir, f'shell_{ishell}_{isub}'+save_format), clobber=True, format=save_format)
