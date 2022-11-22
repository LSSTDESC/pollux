import healpy as hp
import healsparse
import numpy as np
import pyccl
import sacc
import scipy.stats
from tqdm.auto import tqdm, trange

from .map_builder import MapBuilder
from .utils import *
from .utils import _construct_shell_sub


class ShellBuilder(MapBuilder):

    def __init__(
            self,
            reader,
            cosmo,
            nside,
            save_dir,
            apply_random_isometry=False,
            seed=12345,
            dtype=np.uint64,
            subsample_box=None,
            z_subsample_auto=2.,
            group_isometries=None,
            **kwargs
        ):
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
        self.apply_random_isometry = apply_random_isometry
        self.seed = seed
        self.subsample_box = subsample_box
        self.z_subsample_auto = z_subsample_auto
        self.group_isometries = group_isometries

        self.nshells = self.reader.nshells

        base_fname = f'shell_{self.reader.kind}_seed{self.seed}'

        super().__init__(
            nside=nside,
            dtype=dtype,
            save_dir=save_dir,
            base_fname=base_fname,
            nmaps=self.nshells,
            sacc_type="galaxy_density_cl",
            **kwargs
            )

        if dtype == np.int32:
            print(f"Warning: the max value for {dtype} might not be enough : {np.iinfo(dtype)}")

        # Draw random isometries
        if apply_random_isometry:
            self.Abs = self.get_random_isometries()

        # Particle subsampling
        assert subsample_box == 'auto' or type(subsample_box) is float or subsample_box is None

    def get_tracers(self, Nz=1000):
        z_shell_bins = self.reader.get_z_shell_bins(self.cosmo)

        # zz = np.linspace(0, z_shell_bins[-1], Nz)

        # return [
        #     pyccl.NumberCountsTracer(
        #         self.cosmo,
        #         False,
        #         dndz=(
        #             zz,
        #             scipy.stats.uniform.pdf(
        #                 zz, loc=z_shell_bins[ishell], scale=z_shell_bins[ishell + 1] - z_shell_bins[ishell]
        #                 )
        #             ),
        #         bias=(zz, np.ones_like(zz))
        #         )
        #     ]

        ones = np.ones(Nz)
        tracers = []
        for ishell in range(self.nshells):
            zz = np.linspace(z_shell_bins[ishell], z_shell_bins[ishell + 1], Nz)
            tracers.append(pyccl.NumberCountsTracer(self.cosmo, False, dndz=(zz, ones), bias=(zz, ones)))

        return tracers

    def get_tracers_sacc(self, Nz=1000):
        z_shell_bins = self.reader.get_z_shell_bins(self.cosmo)

        zz = np.linspace(0, z_shell_bins[-1], Nz)

        return [
            sacc.tracers.NZTracer(
                f"density_{ishell}", zz,
                scipy.stats.uniform.pdf(
                    zz, loc=z_shell_bins[ishell], scale=z_shell_bins[ishell + 1] - z_shell_bins[ishell]
                    )
                ) for ishell in range(self.nshells)
            ]

    def get_random_isometries(self):
        """Generates random isometries for each shell.

        Returns
        -------
            A list of (A,b) with an isometry A (rotations and flips) and translation by shift b.
        """
        rng = np.random.default_rng(self.seed)
        boxsize = self.reader.get_boxsize(self.cosmo)

        # find number of isometries (can be grouped for adjacent shells)
        n_isometries = self.nshells
        if self.group_isometries is not None:
            com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
            if type(self.group_isometries) is bool:
                assert self.group_isometries
                com_iso_depth = boxsize
            if type(self.group_isometries) is float:
                com_iso_depth = boxsize * self.group_isometries
            n_isometries = int(np.ceil(com_shell_bins[-1] / com_iso_depth))
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
            idx = get_max_overlap(com_shell_bins**3, com_iso_bins**3)

            self.Abs = []
            for ishell in range(self.nshells):
                self.Abs.append(_Abs[idx[ishell]])

        else:
            self.Abs = _Abs

        return self.Abs

    def estimate_repeats(self, ishell):
        com_shell_bins = self.reader.get_com_shell_bins(self.cosmo)
        box_volume = self.reader.get_boxsize(self.cosmo)**3

        shell_volume = 4. / 3. * np.pi * (com_shell_bins[ishell + 1]**3 - com_shell_bins[ishell]**3)
        if self.octant_only:
            shell_volume /= 8.

        return shell_volume / box_volume

    def get_repeats(self, **kwargs):
        """Computes the list of repeat indices (ix,iy,iz) for all the sub-volumes and snapshots. 

        Returns
        -------
            List of repeat indices.
        """
        repeats = []
        for isub in trange(self.reader.nsubs, desc='[get_repeats]'):
            repeats.append((isub, self.construct_shell_sub(isub, get_repeats_only=True, **kwargs)))
        return repeats

    def get_subsample_ratio(self, ishell):
        if self.subsample_box == 'auto':
            if not hasattr(self, 'subsample_ratios'):
                ngal_pix = np.array(
                    [self.estimate_repeats(ishell) for ishell in range(self.nshells)]
                    ) * self.reader.nparticles / (hp.nside2npix(self.nside))  # estimated number of galaxies per pixel
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

    def get_box_comoving_density(self, ishell):
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

    def construct_shell_sub(
            self,
            isub,
            Abs=None,
            com_shell_bins=None,
            skip_existing=False,
            check_read_existing=True,
            verbose=False,
            debug=False,
            repeat_ixyz=None,
            get_repeats_only=False,
            fast=True
        ):
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
        # if self.save_dir is not None:
        fname = self.get_fname(ishell, sub=isub)

        # check if shell has already been created
        if skip_existing:
            if self.check_maps(read=check_read_existing, fname=fname):
                return

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

        # get number of repeats per direction
        boxsize = self.reader.get_boxsize(self.cosmo)
        # not too sure about that 2, but it's fast to check...
        nrepeat = int(np.ceil(np.sqrt(rmax2) / boxsize)) + 2

        # read box
        xyz, box_edges_original = self.reader.read_sub(
            isub,
            read_box_edges_only=get_repeats_only,
            h_to_divide_by=None if self.reader.unit == 'Mpc' else np.float64(self.cosmo["h"])
            )
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

        if fast:
            nside = self.nside
            order = hp.nside2order(nside)
            ncap = (nside * nside - nside) << 1
            npix = hp.nside2npix(nside)
            u, c = _construct_shell_sub(
                xyz, box_edges_original, Abs[ishell][0], Abs[ishell][1], nrepeat, boxsize, rmin2, rmax2,
                self.apply_random_isometry, self.octant_only, nside, order, ncap, npix
                )
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
                        shift_original = (
                            np.array([ix, iy, iz], dtype=self.reader.dtype) * self.reader.get_boxsize(self.cosmo)
                            )[None, :]
                        if self.apply_random_isometry:
                            shift = Ax(Abs[ishell][0], shift_original)
                        else:
                            shift = shift_original

                        # apply isometry to box edges
                        tmp_box_edges = shift + box_edges

                        # check if intersects octant, otherwise skip
                        # np.all(tmp_box_edges < 0.):
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
                        u, c = vec2pix_unique(self.nside, *(xyz.T), *(shift[0].T), rmin2, rmax2, self.octant_only)

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

    def construct_shells(self, isubs=None, **kwargs):
        """Computes all shells.

        Parameters
        ----------
        isubs, optional
            Index of sub-volume/snapshot list to compute, by default None
        """
        if isubs is None:
            # Run all files
            isubs = range(self.reader.njobs)

        for isub in tqdm(isubs, desc='[construct_shells]'):
            self.construct_shell_sub(isub, **kwargs)

    def combine_shells(self, ishells=None, skip_existing=False, check_read_existing=True):
        """Sums the particles shells computed independently for each sub-volume/snapshot with construct_shell_sub
        """
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

    def construct_maps(self, imaps=None, skip_existing=False, check_read_existing=True, **kwargs):
        # first construct shells on simulation subs
        for ishell in self.tqdm(imaps, desc="[construct_maps]"):
            isubs = self.reader.get_isubs_for_shell(ishell)
            self.construct_shell(isubs, skip_existing=skip_existing, check_read_existing=check_read_existing, **kwargs)

        # then combine into final shells
        self.combine_shells(imaps, skip_existing=skip_existing, check_read_existing=check_read_existing)

    def get_field_map(self, ishell):
        shell = self.load_maps(ishell)
        spix = shell.valid_pixels
        density = -np.ones(self.npix)
        density[spix] = shell[spix] / self.get_pixel_density(ishell) - 1.
        return density
