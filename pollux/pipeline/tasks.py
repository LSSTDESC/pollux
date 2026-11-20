"""Parsl tasks of the pipeline, one product per task, chained by run scripts (see the pipeline documentation)."""

import os

from parsl import python_app

##############################
# Shells (dark-matter density maps from sub-volumes)


@python_app(executors=['default'])
def construct_shell_sub(ijob, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_maps_sub(None, None, ijob=ijob)


@python_app(executors=['default'])
def combine_shell_subs(ishell, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.combine_maps_subs(ishell)


@python_app(executors=['highmem'])
def plot_shell_maps(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # all shell maps in one figure ({save_dir}/plots), on a log scale, drawn at a lower nside (plot_maps nside_plot)
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).plot_maps(log=True)
    _free_memory()


##############################
# Footprint (padded mask, NaMaster workspace), shared by all builders


@python_app(executors=['highmem'])
def compute_padding(reader_dict, shell_dict, inputs=[]):
    # padded footprint and transform weights (shell_dict['padding_deg']), saved once for all builders
    _make_shell_builder(reader_dict, shell_dict, None).compute_padding()


@python_app(executors=['heavy'])
def compute_wsp(reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.wsp
    _free_memory()


##############################
# Halo catalogs and baryonification


@python_app(executors=['highmem'])
def construct_halo_catalog(ishell, reader_dict, shell_dict, baryon_dict, inputs=[]):
    _make_halo_builder(reader_dict, shell_dict).construct_catalogs(ishell, extra_columns=[baryon_dict['mass_name']])
    _free_memory()


@python_app(executors=['highmem'])
def construct_baryonified_shell(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).construct_maps(ishell)
    _free_memory()


##############################
# Convergence and shear (generic across dark-matter/baryonified shells and
# tomographic-SRD/per-snapshot lensing, selected via baryon_dict/lensing_dict)


@python_app(executors=['default'])
def construct_kappa_map(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).construct_maps(ikappa)


@python_app(executors=['heavy'])
def compute_kappa_alms(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_alms(ikappa)
    _free_memory()


@python_app(executors=['highmem'])
def plot_kappa_maps(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # all convergence maps in one figure ({save_dir}/plots), drawn at a lower nside (plot_maps nside_plot)
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).plot_maps()
    _free_memory()


##############################
# Fields of the catalogs: spherical harmonics of the shells, and shear, tidal and smoothed maps


@python_app(executors=['heavy'])
def compute_shell_alms(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_alms(ishell)
    _free_memory()


@python_app(executors=['heavy'])
def compute_shear_maps(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc=None, save_field_map=True, baryon_dict=None, inputs=[]):
    # shear maps, smoothed with smooth_Mpc (then also the smoothed convergence map with save_field_map)
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_shear_maps(
        ikappa, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map)
    _free_memory()


@python_app(executors=['heavy'])
def compute_tidal_maps(ishell, reader_dict, shell_dict, smooth_Mpc, save_field_map=True, baryon_dict=None, inputs=[]):
    # tidal field maps, smoothed with smooth_Mpc (then also the smoothed density map with save_field_map)
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_tidal_tensor_maps(ishell, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map)
    _free_memory()


@python_app(executors=['heavy'])
def compute_smoothed_density_map(ishell, reader_dict, shell_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    # smoothed density map, when its smoothing differs from the tidal field's
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_maps_from_alms(ishell, smooth_Mpc=smooth_Mpc)
    _free_memory()


@python_app(executors=['heavy'])
def compute_smoothed_kappa_map(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    # smoothed convergence map, when its smoothing differs from the shear's
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_maps_from_alms(ikappa, smooth_Mpc=smooth_Mpc)
    _free_memory()


def smoothing_plan(catalog_dict, nshells):
    """Return, per shell, the smoothed maps needed by a catalog (see CatalogGenerator smooth_fields/smooth_Mpc).

    Each is a dict with the smoothing of the tidal and shear maps ('tidal', 'shear', None if not smoothed),
    whether these tasks also save the smoothed density/convergence ('density_with_tidal', 'kappa_with_shear'),
    and the other smoothing scales of density/convergence maps, made by separate tasks ('density_extra', 'kappa_extra').
    """
    import inspect
    import pollux
    cls = pollux.catalog.CatalogGenerator
    default_fields = inspect.signature(cls.__init__).parameters['smooth_fields'].default
    scales = cls.smoothing_scales(catalog_dict.get('smooth_fields', default_fields), catalog_dict.get('smooth_Mpc'), nshells)
    plans = []
    for ishell in range(nshells):
        tidal, shear = scales['t11'][ishell], scales['gamma1'][ishell]
        density = {scales['positions'][ishell], scales['density'][ishell]} - {None}
        kappa = {scales['kappa'][ishell]} - {None}
        plans.append(
            dict(
                tidal=tidal,
                shear=shear,
                density_with_tidal=tidal in density,
                kappa_with_shear=shear in kappa,
                density_extra=sorted(density - {tidal}),
                kappa_extra=sorted(kappa - {shear})))
    return plans


def catalog_map_tasks(nshells, reader_dict, shell_dict, catalog_dict, kappa_alms, shell_inputs, baryon_dict=None):
    """Launch the per-shell map tasks needed by catalogs: shear, tidal and smoothed density/convergence maps.

    kappa_alms are the per-shell (snapshot) convergence a_lm futures, and shell_inputs(ishell) the futures
    the shell maps depend on. Returns, per shell, the list of futures of its maps.
    """
    per_shell = []
    for ishell, plan in enumerate(smoothing_plan(catalog_dict, nshells)):
        futures = [
            compute_shear_maps(
                ishell,
                reader_dict,
                shell_dict,
                None,
                smooth_Mpc=plan['shear'],
                save_field_map=plan['kappa_with_shear'],
                baryon_dict=baryon_dict,
                inputs=[kappa_alms[ishell]]),
            compute_tidal_maps(
                ishell, reader_dict, shell_dict, plan['tidal'], save_field_map=plan['density_with_tidal'], baryon_dict=baryon_dict,
                inputs=shell_inputs(ishell)),
        ]
        futures += [
            compute_smoothed_density_map(ishell, reader_dict, shell_dict, s, baryon_dict=baryon_dict, inputs=shell_inputs(ishell))
            for s in plan['density_extra']
        ]
        futures += [
            compute_smoothed_kappa_map(ishell, reader_dict, shell_dict, None, s, baryon_dict=baryon_dict, inputs=[kappa_alms[ishell]])
            for s in plan['kappa_extra']
        ]
        per_shell.append(futures)
    return per_shell


##############################
# Source catalogs


@python_app(executors=['heavy'])
def construct_catalog(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # catalog of one tomographic bin, unless saved
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog.path(ibin)):
        catalog.generate(ibin)
    _free_memory()


@python_app(executors=['highmem'])
def construct_catalog_shell(ishell, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # galaxies of one shell, for all tomographic bins (partial catalogs), unless saved or already merged
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    bins = range(catalog.lensing.nmaps)
    if not (all(os.path.exists(catalog.path(i, ishell=ishell)) for i in bins) or all(os.path.exists(catalog.path(i)) for i in bins)):
        catalog.generate(ishell=ishell)
    _free_memory()


@python_app(executors=['highmem'])
def merge_catalog(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains the construct_catalog_shell futures of all shells; skipped if the catalog is saved
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog.path(ibin)):
        catalog.combine_shells(ibin)
    _free_memory()


##############################
# Statistics of the maps (Cls of the shells and convergence maps) and comparison to predictions


@python_app(executors=['highmem'])
def compute_shell_cls_sub(pair, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # Cls of one (i, j) pair of shells, saved to an intermediate file (its name is returned; None if the Cls are saved)
    from pollux.utils.binning import binning_id
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    params_id = _params_id(binning_id(builder.b))
    if _statistic_saved(builder.path(what='cls', ext='fits'), params_id):
        return None
    fname = builder.io.path(builder.product, *pair, ext='pkl', intermediate=True, folder=f'cls_{params_id}')
    fname = _compute_pair(fname, lambda: builder.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_shell_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_shell_cls_sub, saved unless the Cls already are
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _statistic_saved(builder.path(what='cls', ext='fits'), params_id):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    builder._save_cls(ell, cls_out, metadata={'options_id': params_id})


@python_app(executors=['highmem'])
def plot_shell_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # auto-spectra of the shells against their predictions ({save_dir}/plots)
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).plot_cls(auto_only=True, logy=True)
    _free_memory()


@python_app(executors=['highmem'])
def compute_kappa_cls_sub(pair, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # Cls of one (i, j) pair of convergence maps, saved to an intermediate file (its name is returned; None if the Cls
    # are saved)
    from pollux.utils.binning import binning_id
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    params_id = _params_id(binning_id(lensing.b))
    if _statistic_saved(lensing.path(what='cls', ext='fits'), params_id):
        return None
    fname = lensing.io.path(lensing.product, *pair, ext='pkl', intermediate=True, folder=f'cls_{params_id}')
    fname = _compute_pair(fname, lambda: lensing.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_kappa_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_kappa_cls_sub, saved unless the Cls already are
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    params_id = _params_id_of_pairs(inputs)
    if _statistic_saved(lensing.path(what='cls', ext='fits'), params_id):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._save_cls(ell, cls_out, metadata={'options_id': params_id})


@python_app(executors=['highmem'])
def plot_kappa_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # spectra of the convergence maps against their predictions ({save_dir}/plots): auto-spectra only for the
    # per-shell snapshot maps (lensing_dict None; all pairs would be too many)
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    lensing.plot_cls(auto_only=lensing_dict is None, logy=True)
    _free_memory()


##############################
# Statistics of the catalogs (2pt functions, Cls) and comparison to predictions


@python_app(executors=['heavy'])
def compute_catalog_2pt_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    # 2pt function of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair, with the
    # shapes of the shear legs ('shear', 'ia', 'shear_plus_ia' or 'observed', see CatalogGenerator.compute_2pt),
    # saved to an intermediate file (its name is returned; None if the 2pt functions of these shapes are saved)
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id([stats_dict['common'], stats_dict['2pt']])
    if _statistic_saved(catalog._stat_path('2pt', shapes), params_id):
        return None
    fname = catalog.io.path('catalogs', stat, *pair, ext='pkl', intermediate=True, folder=f'2pt_{shapes}_{params_id}')
    fname = _compute_pair(
        fname, lambda: catalog.compute_2pt(
            what=stat, pairs=[pair], shapes=shapes, save=False, error_bars=True, theta_per_pair=True, **stats_dict['common'], **stats_dict['2pt']))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_catalog_2pt(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, shapes='shear', inputs=[]):
    # inputs contains the files of the (theta, xi, dxi) of compute_catalog_2pt_sub, saved to the file of the shapes
    # unless it exists
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _statistic_saved(catalog._stat_path('2pt', shapes), params_id):
        return
    theta_out, xi_out, dxi_out = {}, {}, {}
    for theta, xi, dxi in _load_pairs(inputs):
        for out, res in ((theta_out, theta), (xi_out, xi), (dxi_out, dxi)):
            for data_type, pairs in res.items():
                out.setdefault(data_type, {}).update(pairs)
    catalog._save_2pt(theta_out, xi_out, dxi=dxi_out, shapes=shapes, metadata={'options_id': params_id})


@python_app(executors=['default'])
def plot_catalog_2pt(
        reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, with_dmo_reference=False, xlim=None, shapes='shear', inputs=[]):
    # 2pt functions against their predictions ({catalog_dir}/plots), with the matter power spectrum of the catalog's shell
    # builder (S19-corrected with baryon_dict); with_dmo_reference adds the ratio to the dark-matter-only catalog's 2pt
    # functions, xlim are the x limits (the edges of the binning, theta_min and theta_max), and shapes those of the
    # shear legs (with 'ia', also GI, once the 'shear' and 'shear_plus_ia' ones are measured)
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    reference = None
    if baryon_dict is not None and with_dmo_reference:
        reference = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, None)
    catalog.plot_2pt(reference=reference, xlim=xlim, shapes=shapes)
    _free_memory()


@python_app(executors=['heavy'])
def compute_catalog_cls_wsp(stat, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # workspace shared by all bin pairs of one statistic, from the footprint (compute_cls and predict_cls with shared_wsp=True)
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog._wsp_path(stat)):
        catalog.wsp(stat)
    _free_memory()


@python_app(executors=['heavy'])
def compute_catalog_alms(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    # catalog fields (a_lm) of one tomographic bin for the Cls with the shapes of the shear legs, saved as intermediate
    # files (skipped if saved, or if the Cls are saved): the Cls of all its pairs read them instead of loading the catalog again
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog._stat_path('cls', shapes)):
        options = {k: v for k, v in {**stats_dict['common'], **stats_dict['cls']}.items() if k not in ('shared_wsp', 'auto_only')}
        catalog.compute_alms(ibin, shapes=shapes, **options)
    _free_memory()


@python_app(executors=['highmem'])
def compute_catalog_cls_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    # Cls of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair, with the shapes of the shear
    # legs (see CatalogGenerator.compute_cls), saved to an intermediate file (its name is returned; None if the Cls of
    # these shapes are saved), from the fields of its bins (inputs: compute_catalog_alms)
    from pollux.utils.binning import binning_id
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id([binning_id(catalog.b), stats_dict['common'], stats_dict['cls']])
    if _statistic_saved(catalog._stat_path('cls', shapes), params_id):
        return None
    fname = catalog.io.path('catalogs', stat, *pair, ext='pkl', intermediate=True, folder='cls_' + ('' if shapes == 'shear' else f'{shapes}_') + params_id)
    fname = _compute_pair(fname, lambda: catalog.compute_cls(what=stat, pairs=[pair], shapes=shapes, save=False, **stats_dict['common'], **stats_dict['cls']))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, shapes='shear', inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_catalog_cls_sub, saved to the file of the shapes unless it exists
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _statistic_saved(catalog._stat_path('cls', shapes), params_id):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        for data_type, pairs in cl.items():
            cls_out.setdefault(data_type, {}).update(pairs)
    catalog._save_cls(ell, cls_out, shapes=shapes, metadata={'options_id': params_id})


@python_app(executors=['default'])
def plot_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, cls_kwargs=None, shapes='shear', inputs=[]):
    # Cls against their predictions ({catalog_dir}/plots); cls_kwargs are the Cls options (shared_wsp, randoms), and
    # shapes those of the shear legs
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    catalog.plot_cls(shapes=shapes, **{k: v for k, v in (cls_kwargs or {}).items() if k in ('shared_wsp', 'randoms')})
    _free_memory()


def _free_memory():
    # at the end of memory-heavy tasks: pyplot keeps figures alive until closed, and freed memory
    # is not always returned to the system, so in a long-lived worker a task's peak (and its plots)
    # would otherwise stay for the next tasks
    import ctypes
    import gc
    import matplotlib.pyplot as plt
    plt.close('all')
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def _make_shell_builder(reader_dict, shell_dict, baryon_dict):
    if baryon_dict is not None:
        return _make_baryon_builder(reader_dict, shell_dict, baryon_dict)
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    return pollux.shells.SnapshotShellBuilder(reader, **shell_dict)


def _make_halo_builder(reader_dict, shell_dict):
    # halo catalogs: all halos (no subsampling of the boxes)
    import pollux
    return pollux.shells.SnapshotShellBuilder(pollux.reader.HACCHaloReader(**reader_dict), **{**shell_dict, 'subsample_box': None})


def _make_baryon_builder(reader_dict, shell_dict, baryon_dict):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder_halo = _make_halo_builder(reader_dict, shell_dict)
    builder_params = dict(baryon_dict)
    tag = builder_params.pop('tag', None)
    bfg_params = builder_params.pop('bfg_params')
    # the same profiles baryonify the shells and correct the predicted P(k)
    dmo, dmb = pollux.baryons.s19_profiles(bfg_params, builder_params['mass_def'])
    return pollux.baryons.BaryonifiedShellBuilder(builder, builder_halo, dmo, dmb, profile_parameters=bfg_params, tag=tag, **builder_params)


def _make_lensing(builder, lensing_dict):
    import pollux
    if lensing_dict is None:
        return builder.snapshot_lensing_builder()
    return pollux.lensing.BornLensingBuilder(builder, **lensing_dict)


def _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict):
    import pollux
    from pollux.catalog import CatalogAugmentorShear
    from pollux.ia import CatalogAugmentorIA
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    ia = CatalogAugmentorIA(**ia_dict)
    shear_aug = CatalogAugmentorShear(
        compute_reduced_shear=True,
        generate_intrinsic=True,
        compute_observed=True,
    )
    catalog = pollux.catalog.CatalogClusteredPoisson(
        lensing,
        augmentors=[ia, shear_aug],
        **catalog_dict,
    )  # checks its parameters against the config.yaml of existing catalogs (or writes it), before any task skips them
    return catalog


def _params_id(params):
    # identifier (8 hexadecimal characters) of the parameters a statistic depends on (binning, statistics options)
    # the files of the bin pairs go to a folder named with it, so that results with other parameters are not mixed
    import hashlib
    import json
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _params_id_of_pairs(fnames):
    # identifier of the parameters of the bin pairs combined, from the folder of their files (None if all skipped)
    ids = {os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1] for f in fnames if f is not None}
    if len(ids) > 1:
        raise ValueError(f"Bin pairs measured with different options: {sorted(ids)}")
    return ids.pop() if ids else None


def _statistic_saved(fname, params_id):
    """Return whether the statistic of ``fname`` is saved, with the options of identifier ``params_id``.

    The combining tasks record the identifier in the SACC metadata (``options_id``): a statistic saved with other
    options raises, instead of being kept as if it were up to date. Statistics saved before the identifier was
    recorded are kept, with a warning.
    """
    import warnings
    from pollux.io import PolluxIO
    if not os.path.exists(fname):
        return False
    saved = PolluxIO.read_sacc(fname).metadata.get('options_id')
    if saved is None:
        warnings.warn(f"{fname} has no identifier of the options it was measured with: kept as is.")
    elif params_id is not None and saved != params_id:
        raise ValueError(f"{fname} was measured with other options (identifier {saved}, now {params_id}): move it away to measure it again.")
    return True


def _compute_pair(fname, compute):
    # statistic of one bin pair, computed and saved to fname unless already saved (resumed runs); returns fname, read
    # by the task combining the pairs (so results go through the disk, not through the main process)
    from pollux.io import PolluxIO
    if not os.path.exists(fname):
        PolluxIO.write_pickle(fname, compute())
    return fname


def _load_pairs(fnames):
    # results of the bin pairs saved by _compute_pair (None for tasks skipped because their product exists)
    from pollux.io import PolluxIO
    for fname in fnames:
        if fname is not None:
            yield PolluxIO.read_pickle(fname)
