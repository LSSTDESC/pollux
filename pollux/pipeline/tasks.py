"""Parsl tasks of the pipeline, one product per task, chained by run scripts (see the pipeline documentation)."""

import os

from parsl import python_app


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
    return pollux.catalog.CatalogClusteredPoisson(
        lensing,
        augmentors=[ia, shear_aug],
        **catalog_dict,
    )


def _pair_fname(intermediate_dir, name, pair, params, stat=None):
    # intermediate file of the statistic of one bin pair: {intermediate_dir}/{name}_{id}/[{stat}_]{i}_{j}.pkl, the id
    # (8 hexadecimal characters) identifying the parameters the result depends on (binning, statistics options), so
    # that results with other parameters are not mixed
    import hashlib
    import json
    params_id = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    i, j = pair
    return os.path.join(intermediate_dir, f'{name}_{params_id}', f"{f'{stat}_' if stat else ''}{i}_{j}.pkl")


def _compute_pair(fname, compute):
    # statistic of one bin pair, computed and saved to fname unless already saved (resumed runs); returns fname, read
    # by the task combining the pairs (so results go through the disk, not through the main process)
    import pickle
    from pollux.io import layout
    if not os.path.exists(fname):
        result = compute()
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with layout.atomic_write(fname) as tmp, open(tmp, 'wb') as f:
            pickle.dump(result, f)
    return fname


def _load_pairs(fnames):
    # results of the bin pairs saved by _compute_pair (None for tasks skipped because their product exists)
    import pickle
    for fname in fnames:
        if fname is not None:
            with open(fname, 'rb') as f:
                yield pickle.load(f)


##############################
# Shells (dark-matter density maps from sub-volumes)


@python_app(executors=['default'])
def construct_shell_sub(ijob, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_maps_sub(None, None, ijob=ijob)


@python_app(executors=['default'])
def construct_shell_orphans(ishell, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_maps_sub(None, None, do_orphan_ishell=ishell)


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
def construct_catalog(ikappa, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # catalog of one tomographic bin, unless saved
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog.get_fname(ikappa)):
        catalog.generate(ikappa)
    _free_memory()


@python_app(executors=['highmem'])
def construct_catalog_shell(ishell, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # galaxies of one shell, for all tomographic bins (partial catalogs), unless saved or already merged
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    bins = range(catalog.lensing.nmaps)
    if not (all(os.path.exists(catalog.get_fname(i, ishell=ishell)) for i in bins) or all(os.path.exists(catalog.get_fname(i)) for i in bins)):
        catalog.generate(ishell=ishell)
    _free_memory()


@python_app(executors=['highmem'])
def merge_catalog(ikappa, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains the construct_catalog_shell futures of all shells; skipped if the catalog is saved
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog.get_fname(ikappa)):
        catalog.combine_shells(ikappa)
    _free_memory()


##############################
# Statistics of the maps (Cls of the shells and convergence maps) and comparison to predictions


@python_app(executors=['highmem'])
def compute_shell_cls_sub(pair, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # Cls of one (i, j) pair of shells, saved to an intermediate file (its name is returned; None if the Cls are saved)
    from pollux.utils.binning import binning_id
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    if os.path.exists(builder.get_fname(what='cls', format='fits')):
        return None
    fname = _pair_fname(builder.intermediate_dir, 'cls', pair, binning_id(builder.b))
    fname = _compute_pair(fname, lambda: builder.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_shell_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_shell_cls_sub, saved unless the Cls already are
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    if os.path.exists(builder.get_fname(what='cls', format='fits')):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    builder._save_cls(ell, cls_out)


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
    if os.path.exists(lensing.get_fname(what='cls', format='fits')):
        return None
    fname = _pair_fname(lensing.intermediate_dir, 'cls', pair, binning_id(lensing.b))
    fname = _compute_pair(fname, lambda: lensing.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_kappa_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_kappa_cls_sub, saved unless the Cls already are
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    if os.path.exists(lensing.get_fname(what='cls', format='fits')):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._save_cls(ell, cls_out)


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
    if os.path.exists(catalog._2pt_fname(shapes)):
        return None
    fname = _pair_fname(catalog._intermediate_dir, f'2pt_{shapes}', pair, [stats_dict['common'], stats_dict['2pt']], stat=stat)
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
    if os.path.exists(catalog._2pt_fname(shapes)):
        return
    theta_out, xi_out, dxi_out = {}, {}, {}
    for theta, xi, dxi in _load_pairs(inputs):
        for out, res in ((theta_out, theta), (xi_out, xi), (dxi_out, dxi)):
            for data_type, pairs in res.items():
                out.setdefault(data_type, {}).update(pairs)
    catalog._save_2pt(theta_out, xi_out, dxi=dxi_out, shapes=shapes)


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
    if not os.path.exists(catalog._wsp_fname(stat)):
        catalog.get_wsp(stat)
    _free_memory()


@python_app(executors=['heavy'])
def compute_catalog_alms(ikappa, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # catalog fields (a_lm) of one tomographic bin for the Cls, saved as intermediate files (skipped if saved, or if the
    # Cls are saved): the Cls of all its pairs read them instead of loading the catalog again
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog._cls_fname()):
        options = {k: v for k, v in {**stats_dict['common'], **stats_dict['cls']}.items() if k not in ('shared_wsp', 'auto_only')}
        catalog.compute_alms(ikappa, **options)
    _free_memory()


@python_app(executors=['highmem'])
def compute_catalog_cls_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # Cls of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair, saved to an intermediate file
    # (its name is returned; None if the Cls are saved), from the fields of its bins (inputs: compute_catalog_alms)
    from pollux.utils.binning import binning_id
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if os.path.exists(catalog._cls_fname()):
        return None
    params = [binning_id(catalog.b), stats_dict['common'], stats_dict['cls']]
    fname = _pair_fname(catalog._intermediate_dir, 'cls', pair, params, stat=stat)
    fname = _compute_pair(fname, lambda: catalog.compute_cls(what=stat, pairs=[pair], save=False, **stats_dict['common'], **stats_dict['cls']))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains the files of the (ell, cl) of compute_catalog_cls_sub, saved unless the Cls already are
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if os.path.exists(catalog._cls_fname()):
        return
    cls_out = {}
    for ell, cl in _load_pairs(inputs):
        for data_type, pairs in cl.items():
            cls_out.setdefault(data_type, {}).update(pairs)
    catalog._save_cls(ell, cls_out)


@python_app(executors=['default'])
def plot_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, cls_kwargs=None, inputs=[]):
    # Cls against their predictions ({catalog_dir}/plots); cls_kwargs are the Cls options (shared_wsp, randoms)
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    catalog.plot_cls(**{k: v for k, v in (cls_kwargs or {}).items() if k in ('shared_wsp', 'randoms')})
    _free_memory()
