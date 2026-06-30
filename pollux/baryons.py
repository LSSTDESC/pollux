import os
from typing import Any, Iterable

from astropy.io import fits
import pyccl
import healpy as hp
import healsparse
import BaryonForge as bfg
from .map_builder import MapBuilder
from .shells import ShellBuilder, SnapshotShellBuilder


class BaryonifiedShellBuilder(ShellBuilder):
    """Apply a baryonification model to shell maps using halo catalogs.

    This builder reads per-shell halo catalogs and shell count maps from the
    provided builders, applies the BaryonForge baryonification model, and
    writes baryonified shell maps to disk.
    
    Inherits all ShellBuilder methods and attributes from the input shell_builder.
    """

    def __init__(
        self,
        shell_builder: ShellBuilder,
        halo_builder: SnapshotShellBuilder,
        dmo: bfg.Profiles,
        dmb: bfg.Profiles,
        mass_name: str,
        mass_def: tuple,
        mass_unit: str,
        read_saved_haloes_props: bool = False,
        tag: str | None = None,
        M_min: float | None = None,
        **interpolator_kwargs: Any,
    ) -> None:
        """Initialize a baryonified shell builder.

        Parameters
        ----------
        shell_builder : ShellBuilder
            Builder providing shell count maps. All shell construction parameters
            and methods are inherited from this builder.
        halo_builder : SnapshotShellBuilder
            Builder providing halo catalogs per shell.
        dmo, dmb : object
            Dark-matter-only and baryonic profiles compatible with BaryonForge.
        read_saved_haloes_props : bool, optional
            If True, read halo properties using ``read_catalogs_props``.
        mass_name : str
            Halo mass field name in the catalogs (e.g. ``'fof_halo_mass'``).
        mass_def : tuple
            Mass definition passed to ``pyccl.halos.massdef.MassDef(*mass_def)``
            (e.g. ``('fof', 'matter')``).
        mass_unit : str
            Units of the halo mass column. Must be ``'Msun'`` or ``'Msun_over_h'``.
        tag : str or None, optional
            Tag used to name the output subdirectory: ``baryons_{tag}/``.  When
            *None* uses ``baryons/``.
        M_min : float or None, optional
            Minimum halo mass used in baryonification. Defaults to
            ``10 * reader.mass_particles`` when None.
        **interpolator_kwargs
            Forwarded to ``model.setup_interpolator``.
        """
        self.shell_builder_ref = shell_builder
        self.halo_builder = halo_builder
        self.mass_name = mass_name
        self.mass_def = mass_def
        if mass_unit not in ('Msun', 'Msun_over_h'):
            raise ValueError(f"mass_unit must be 'Msun' or 'Msun_over_h', got {mass_unit!r}")
        self.mass_unit = mass_unit
        self.read_saved_haloes_props = read_saved_haloes_props
        self.tag = tag
        self.M_min = 10 * shell_builder.reader.mass_particles if M_min is None else M_min

        baryon_dir = f'baryons_{tag}' if tag is not None else 'baryons'

        # Initialize parent ShellBuilder with the same configuration
        if self.shell_builder_ref.kind == 'snapshot':
            self.shell_edges_match = self.shell_builder_ref.shell_edges_match
        super().__init__(
            reader=shell_builder.reader,
            nshells=shell_builder.nshells,
            nside=shell_builder.nside,
            save_dir=os.path.join(shell_builder.save_dir, baryon_dir),
            seed=shell_builder.seed,
            dtype=shell_builder.dtype,
            apply_grid_isometry=shell_builder.apply_grid_isometry,
            apply_replicate_rotflip=shell_builder.apply_replicate_rotflip,
            subsample_box=shell_builder.subsample_box,
            nside_c=shell_builder.nside_c,
            octant_only=shell_builder.octant_only,
            save_format=shell_builder.save_format,
            wsp_dir=shell_builder.wsp_dir,
            b=shell_builder.b,
            nlb=shell_builder.nlb,
            remove_monopole=shell_builder.remove_monopole,
            nocompress=shell_builder.nocompress,
            f_ell=shell_builder.f_ell,
            base_fname='shell')

        # Set up baryonification model
        self.cosmo_dict = self.cosmo.to_dict()
        self.cosmo_dict.update({'matter_power_spectrum': 'linear'})  # Use linear power spectrum for baryonification
        self.cosmo_linear = pyccl.Cosmology(**self.cosmo_dict)
        self.cosmo_dict.update({'Omega_m': self.cosmo['Omega_m']})

        pix = bfg.HealPixel(self.nside)
        dmo = bfg.ConvolvedProfile(dmo, pix)
        dmb = bfg.ConvolvedProfile(dmb, pix)
        self.model = bfg.Profiles.Baryonification2D(dmo, dmb, self.cosmo_linear, mass_def=pyccl.halos.massdef.MassDef(*mass_def))

        self._interpolator_setup = False
        self._interpolator_kwargs = interpolator_kwargs
        self._interpolator_kwargs['M_min'] = self.M_min
        if 'N_samples_z' not in self._interpolator_kwargs:
            self._interpolator_kwargs['N_samples_z'] = 2 * self.nshells + 1

    def construct_maps(
        self,
        ishells: int | Iterable[int] | None = None,
    ) -> healsparse.HealSparseMap | dict[int, healsparse.HealSparseMap]:
        """Baryonify shell count maps using the halo catalog.

        Parameters
        ----------
        ishells : int or array_like of int or None, optional
            Shell indices to process. If None, processes all shells.

        Returns
        -------
        healsparse.HealSparseMap or dict
            - If ``ishells`` is a single int: returns the baryonified shell map.
            - If ``ishells`` is ``None`` or an iterable: returns a dict keyed by
              shell index with baryonified shell maps.
        """
        # Set up interpolator on first call
        if not self._interpolator_setup:
            z_shell_bins = self.shell_edges_redshift()
            self.model.setup_interpolator(
                z_min=max(1e-3, z_shell_bins[0]),  # Avoid z=0 to prevent numerical issues
                z_max=z_shell_bins[self.nshells],
                **self._interpolator_kwargs,
            )
            self._interpolator_setup = True

        shells = {}

        _, ipix = self.get_mask(nest=False, return_ipix=True)
        for ishell in self.tqdm(ishells, desc="[construct_maps]"):
            fname = self.get_fname(ishell)
            shell = self.check_maps(read=True, fname=fname)
            if shell is not None:
                shells[ishell] = shell
                continue

            # Load the shell, turn into count map (in RING ordering), and set unseen/empty pixels to 0.
            count = self.shell_builder_ref.load_maps(ishell).generate_healpix_map(nest=False)
            count[count == hp.UNSEEN] = 0.
            s = bfg.utils.LightconeShell(map=count, cosmo=self.cosmo_dict)

            # Load the halo catalog for this shell and apply mass cut
            if self.read_saved_haloes_props:
                cat_halo = fits.open(self.halo_builder.get_fname(ishell, what='haloes_props'))[1].data
            else:
                cat_halo = self.halo_builder.read_catalogs_props(ishell, variables=['ra', 'dec', 'z', self.mass_name])
                
            # Convert halo mass to Msun if necessary
            M = cat_halo[self.mass_name]
            if self.mass_unit == 'Msun_over_h':
                M *= self.cosmo['h']
                
            w = M > self.M_min
            h = bfg.utils.HaloLightConeCatalog(cat_halo['ra'][w], cat_halo['dec'][w], M[w], cat_halo['z'][w], self.cosmo_dict)

            # Run baryonification
            runner = bfg.Runners.BaryonifyShell(h, s, epsilon_max=20., model=self.model)
            baryonified_count = runner.process()

            # Save the baryonified map
            shell = healsparse.HealSparseMap.make_empty(self.nside_c, self.nside, dtype=self.dtype)
            shell.update_values_pix(ipix, baryonified_count[ipix].astype(self.dtype), nest=False)
            shell.write(fname, clobber=True, format=self.save_format, nocompress=self.nocompress)

            shells[ishell] = shell

        return shells[ishells] if type(ishells) is int else shells
