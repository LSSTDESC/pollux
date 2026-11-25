"""Parsl tasks of the pipeline, one product per task, chained by run scripts (see the pipeline documentation).

Tasks decide what to make: each skips its product if saved (:func:`_run`), so that runs resume where they stopped;
the builders and catalogs only compute and write.

The tasks share their arguments, the options of the objects they rebuild in their worker (see the run scripts):
``reader_dict`` (:class:`~pollux.reader.HACCReader`), ``shell_dict`` (:class:`~pollux.shells.SnapshotShellBuilder`,
with its footprint), ``lensing_dict`` (:class:`~pollux.lensing.BornLensingBuilder`, None for per-shell lensing),
``baryon_dict`` (:class:`~pollux.baryons.BaryonifiedShellBuilder`, None for dark matter only), ``ia_dict`` and
``catalog_dict`` (:class:`~pollux.catalog.CatalogClusteredPoisson` and its augmentors), ``stats_dict`` (options of the
statistics: ``common``, ``2pt``, ``cls``), and ``inputs``, the futures of the tasks they depend on (Parsl).
"""

import os

from parsl import python_app

# ======================================================================================================================
# region Shells


@python_app(executors=['default'])
def construct_shell_sub(ijob, reader_dict, shell_dict, inputs=[]):
    """Map of the particles of one sub-volume of a shell (job ``ijob`` of the reader)."""
    builder = _make_shell_builder(reader_dict, shell_dict, None)
    ishell, isub = builder.reader.job_args(ijob)
    _run([builder.path(ishell, sub=isub)], lambda: builder.construct_maps_sub(ishell, isub))


@python_app(executors=['default'])
def combine_shell_subs(ishell, reader_dict, shell_dict, inputs=[]):
    """Map of one shell, the sum of the maps of its sub-volumes (:func:`construct_shell_sub`)."""
    builder = _make_shell_builder(reader_dict, shell_dict, None)
    _run([builder.path(ishell)], lambda: builder.combine_maps_subs(ishell))


@python_app(executors=['highmem'])
def plot_shell_maps(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """All shell maps in one figure ({save_dir}/plots), on a log scale, drawn at a lower nside (plot_maps
    nside_plot).
    """
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).plot_maps(log=True)
    _free_memory()


# endregion

# ======================================================================================================================
# region Footprint


@python_app(executors=['highmem'])
def compute_padding(reader_dict, shell_dict, inputs=[]):
    """Padded footprint and weights of the harmonic transforms (footprint padding_deg), saved once for all builders."""
    footprint = _make_shell_builder(reader_dict, shell_dict, None).footprint
    _run([footprint.path('padded_mask'), footprint.path('alm_weights')], lambda: footprint.alm_weights)


@python_app(executors=['heavy'])
def compute_wsp(reader_dict, shell_dict, inputs=[]):
    """Workspace of the maps and its bandpower windows (saved apart, see
    :meth:`~pollux.statistics.base.TwoPointStatistics._saved_bpws`), shared by all their Cls.
    """
    builder = _make_shell_builder(reader_dict, shell_dict, None)
    _run([builder._path_wsp, builder._path_bpws(builder._path_wsp)], lambda: builder.bpws)
    _free_memory()


# endregion

# ======================================================================================================================
# region Halos


@python_app(executors=['highmem'])
def construct_halo_catalog(ishell, reader_dict, shell_dict, baryon_dict, inputs=[]):
    """Halo catalog of one shell, with the halo masses of ``baryon_dict`` (``mass_name``)."""
    builder = _make_halo_builder(reader_dict, shell_dict)
    _run([builder.path(ishell, what='catalog')], lambda: builder.construct_catalogs(ishell, extra_columns=[baryon_dict['mass_name']]))
    _free_memory()


@python_app(executors=['highmem'])
def construct_baryonified_shell(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """Baryonified map of one shell (``baryon_dict``), from its dark-matter-only map and halo catalog."""
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    _run([builder.path(ishell)], lambda: builder.construct_maps(ishell))
    _free_memory()


# endregion

# ======================================================================================================================
# region Lensing


@python_app(executors=['default'])
def construct_kappa_map(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """Convergence map of one tomographic bin (or one shell, for snapshot lensing: ``lensing_dict`` None)."""
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    _run([lensing.path(ikappa)], lambda: lensing.construct_maps(ikappa))


@python_app(executors=['heavy'])
def compute_kappa_alms(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """A_lm of one convergence map, read by its shear and smoothed maps and by the Cls."""
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    _run([lensing._path_alms(ikappa)], lambda: lensing.compute_alms(ikappa))
    _free_memory()


@python_app(executors=['highmem'])
def plot_kappa_maps(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """All convergence maps in one figure ({save_dir}/plots), drawn at a lower nside (plot_maps nside_plot)."""
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).plot_maps()
    _free_memory()


# endregion

# ======================================================================================================================
# region Catalog maps


@python_app(executors=['heavy'])
def compute_shell_alms(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """A_lm of one shell, read by its tidal and smoothed maps and by the Cls."""
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    _run([builder._path_alms(ishell)], lambda: builder.compute_alms(ishell))
    _free_memory()


@python_app(executors=['heavy'])
def compute_shear_maps(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc=None, save_field_map=True, baryon_dict=None, inputs=[]):
    """Shear maps, smoothed with smooth_Mpc (then also the smoothed convergence map with save_field_map)."""
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    outputs = _smoothed_outputs(lensing, ikappa, ('gamma1', 'gamma2'), smooth_Mpc, save_field_map)
    _run(outputs, lambda: lensing.compute_shear_maps(ikappa, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map))
    _free_memory()


@python_app(executors=['heavy'])
def compute_tidal_maps(ishell, reader_dict, shell_dict, smooth_Mpc, save_field_map=True, baryon_dict=None, inputs=[]):
    """Tidal field maps, smoothed with smooth_Mpc (then also the smoothed density map with save_field_map)."""
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    outputs = _smoothed_outputs(builder, ishell, ('t11', 't22', 't12'), smooth_Mpc, save_field_map)
    _run(outputs, lambda: builder.compute_tidal_tensor_maps(ishell, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map))
    _free_memory()


@python_app(executors=['heavy'])
def compute_smoothed_density_map(ishell, reader_dict, shell_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    """Smoothed density map, when its smoothing differs from the tidal field's."""
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    _run(_smoothed_outputs(builder, ishell, (), smooth_Mpc, True), lambda: builder.compute_maps_from_alms(ishell, smooth_Mpc=smooth_Mpc))
    _free_memory()


@python_app(executors=['heavy'])
def compute_smoothed_kappa_map(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    """Smoothed convergence map, when its smoothing differs from the shear's."""
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    _run(_smoothed_outputs(lensing, ikappa, (), smooth_Mpc, True), lambda: lensing.compute_maps_from_alms(ikappa, smooth_Mpc=smooth_Mpc))
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


# endregion

# ======================================================================================================================
# region Catalogs


@python_app(executors=['heavy'])
def construct_catalog(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    """Catalog of one tomographic bin, unless saved."""
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    _run([catalog.path(ibin)], lambda: catalog.generate(ibin))
    _free_memory()


@python_app(executors=['highmem'])
def construct_catalog_shell(ishell, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    """Galaxies of one shell, for all tomographic bins (catalogs of the shell), unless saved or already merged."""
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    bins = range(catalog.lensing.nmaps)
    if not _done([catalog.path(i) for i in bins]):
        _run([catalog.path(i, ishell=ishell) for i in bins], lambda: catalog.generate(ishell=ishell))
    _free_memory()


@python_app(executors=['highmem'])
def merge_catalog(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    """Inputs contains the construct_catalog_shell futures of all shells; skipped if the catalog is saved."""
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    _run([catalog.path(ibin)], lambda: catalog.merge_shell_catalogs(ibin))
    _free_memory()


# endregion

# ======================================================================================================================
# region Map statistics


@python_app(executors=['highmem'])
def compute_shell_cls_sub(pair, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """Cls of one (i, j) pair of shells, saved to an intermediate file (its name is returned; None if the Cls are
    saved).
    """
    from pollux.utils.binning import binning_id
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    params_id = _params_id(binning_id(builder.b))
    if _done([builder.path(what='cls', ext='fits')], options_id=params_id):
        return None
    fname = builder.io.path(builder.product, *pair, ext='pkl', intermediate=True, folder=f'cls_{params_id}')
    fname = _compute_pair(fname, lambda: builder.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_shell_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """Inputs contains the files of the (ell, cl) of compute_shell_cls_sub, saved unless the Cls already are."""
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _done([builder.path(what='cls', ext='fits')], options_id=params_id):
        return
    cls_out = {}
    for ell, cl in _read_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    builder._write_cls(ell, cls_out, metadata={'options_id': params_id})


@python_app(executors=['highmem'])
def plot_shell_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    """Auto-spectra of the shells against their predictions ({save_dir}/plots)."""
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).plot_cls(auto_only=True, logy=True)
    _free_memory()


@python_app(executors=['highmem'])
def compute_kappa_cls_sub(pair, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """Cls of one (i, j) pair of convergence maps, saved to an intermediate file (its name is returned; None if the
    Cls are saved).
    """
    from pollux.utils.binning import binning_id
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    params_id = _params_id(binning_id(lensing.b))
    if _done([lensing.path(what='cls', ext='fits')], options_id=params_id):
        return None
    fname = lensing.io.path(lensing.product, *pair, ext='pkl', intermediate=True, folder=f'cls_{params_id}')
    fname = _compute_pair(fname, lambda: lensing.compute_cls(pairs=[pair], save=False, use_precomputed_alms=True))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_kappa_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """Inputs contains the files of the (ell, cl) of compute_kappa_cls_sub, saved unless the Cls already are."""
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    params_id = _params_id_of_pairs(inputs)
    if _done([lensing.path(what='cls', ext='fits')], options_id=params_id):
        return
    cls_out = {}
    for ell, cl in _read_pairs(inputs):
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._write_cls(ell, cls_out, metadata={'options_id': params_id})


@python_app(executors=['highmem'])
def plot_kappa_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    """Spectra of the convergence maps against their predictions ({save_dir}/plots): auto-spectra only for the per-
    shell snapshot maps (lensing_dict None; all pairs would be too many).
    """
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    lensing.plot_cls(auto_only=lensing_dict is None, logy=True)
    _free_memory()


# endregion

# ======================================================================================================================
# region Catalog statistics


def pair_groups_2pt(stat, pairs):
    """Bin pairs of a statistic grouped into the tasks of compute_catalog_2pt_sub, so that each task builds the
    TreeCorr tree of the randoms, which takes most of its time, for several pairs, and counts the randoms once for all
    of them (CatalogGenerator.compute_2pt, with randoms_shared): clustering pairs by first bin, shear x clustering pairs
    by source bin (the second), shear pairs alone.

    Parameters
    ----------
    stat : str
        'clustering', 'shear' or 'shear_clustering'.
    pairs : list of tuple of int
        Bin pairs (i, j) of the statistic.

    Returns
    -------
    list of list of tuple of int
        Groups of pairs, one per task.
    """
    if stat == 'shear':
        return [[tuple(pair)] for pair in pairs]
    groups = {}
    for pair in pairs:
        groups.setdefault(pair[1] if stat == 'shear_clustering' else pair[0], []).append(tuple(pair))
    return list(groups.values())


@python_app(executors=['twopt'])
def compute_catalog_2pt_sub(stat, pairs, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    """2pt function of one statistic ('clustering', 'shear' or 'shear_clustering') for a bin pair, or a list of bin
    pairs computed together (see pair_groups_2pt), with the shapes of the shear legs ('shear', 'ia', 'shear_plus_ia'
    or 'observed', see CatalogGenerator.compute_2pt), saved to an intermediate file (its name is returned; None if the
    2pt functions of these shapes are saved).
    """
    pairs = [tuple(pairs)] if not hasattr(pairs[0], '__len__') else [tuple(pair) for pair in pairs]
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id([stats_dict['common'], stats_dict['2pt']])
    if _done([catalog._path_stat('2pt', shapes)], options_id=params_id):
        return None
    fname = catalog.io.path('catalogs', stat, *sum(pairs, ()), ext='pkl', intermediate=True, folder=f'2pt_{shapes}_{params_id}')
    fname = _compute_pair(
        fname, lambda: catalog.compute_2pt(
            what=stat, pairs=pairs, shapes=shapes, save=False, error_bars=True, theta_per_pair=True, **stats_dict['common'], **stats_dict['2pt']))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_catalog_2pt(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, shapes='shear', inputs=[]):
    """Inputs contains the files of the (theta, xi, dxi) of compute_catalog_2pt_sub, saved to the file of the shapes
    unless it exists.
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _done([catalog._path_stat('2pt', shapes)], options_id=params_id):
        return
    theta_out, xi_out, dxi_out = {}, {}, {}
    for theta, xi, dxi in _read_pairs(inputs):
        for out, res in ((theta_out, theta), (xi_out, xi), (dxi_out, dxi)):
            for data_type, pairs in res.items():
                out.setdefault(data_type, {}).update(pairs)
    catalog._write_2pt(theta_out, xi_out, dxi=dxi_out, shapes=shapes, metadata={'options_id': params_id})


@python_app(executors=['default'])
def plot_catalog_2pt(
        reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, with_dmo_reference=False, xlim=None, shapes='shear', inputs=[]):
    """2pt functions against their predictions ({catalog_dir}/plots), with the matter power spectrum of the
    catalog's shell builder (S19-corrected with baryon_dict); with_dmo_reference adds the ratio to the dark-matter-
    only catalog's 2pt functions, xlim are the x limits (the edges of the binning, theta_min and theta_max), and
    shapes those of the shear legs (with 'ia', also GI, once the 'shear' and 'shear_plus_ia' ones are measured).
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    reference = None
    if baryon_dict is not None and with_dmo_reference:
        reference = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, None)
    catalog.plot_2pt(reference=reference, xlim=xlim, shapes=shapes)
    _free_memory()


@python_app(executors=['heavy'])
def compute_catalog_cls_wsp(stat, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    """Workspace shared by all bin pairs of one statistic, from the footprint (compute_cls and predict_cls with
    shared_wsp=True).
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    fname = catalog._path_wsp(stat)
    _run([fname, catalog._path_bpws(fname)], lambda: catalog.bpws(stat))
    _free_memory()


@python_app(executors=['heavy'])
def compute_catalog_alms(ibin, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    """Catalog fields (a_lm) of one tomographic bin for the Cls with the shapes of the shear legs, saved as
    intermediate files (skipped if saved, or if the Cls are saved): the Cls of all its pairs read them instead of
    loading the catalog again.
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not _done([catalog._path_stat('cls', shapes)]):
        options = {k: v for k, v in {**stats_dict['common'], **stats_dict['cls']}.items() if k not in ('shared_wsp', 'auto_only')}
        catalog.compute_alms(ibin, shapes=shapes, **options)
    _free_memory()


@python_app(executors=['highmem'])
def compute_catalog_cls_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, shapes='shear', inputs=[]):
    """Cls of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair, with the shapes of the
    shear legs (see CatalogGenerator.compute_cls), saved to an intermediate file (its name is returned; None if the
    Cls of these shapes are saved), from the fields of its bins (inputs: compute_catalog_alms).
    """
    from pollux.utils.binning import binning_id
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id([binning_id(catalog.b), stats_dict['common'], stats_dict['cls']])
    if _done([catalog._path_stat('cls', shapes)], options_id=params_id):
        return None
    fname = catalog.io.path('catalogs', stat, *pair, ext='pkl', intermediate=True, folder='cls_' + ('' if shapes == 'shear' else f'{shapes}_') + params_id)
    fname = _compute_pair(fname, lambda: catalog.compute_cls(what=stat, pairs=[pair], shapes=shapes, save=False, **stats_dict['common'], **stats_dict['cls']))
    _free_memory()
    return fname


@python_app(executors=['default'])
def compute_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, shapes='shear', inputs=[]):
    """Inputs contains the files of the (ell, cl) of compute_catalog_cls_sub, saved to the file of the shapes unless
    it exists.
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    params_id = _params_id_of_pairs(inputs)
    if _done([catalog._path_stat('cls', shapes)], options_id=params_id):
        return
    cls_out = {}
    for ell, cl in _read_pairs(inputs):
        for data_type, pairs in cl.items():
            cls_out.setdefault(data_type, {}).update(pairs)
    catalog._write_cls(ell, cls_out, shapes=shapes, metadata={'options_id': params_id})


@python_app(executors=['default'])
def plot_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, cls_kwargs=None, shapes='shear', inputs=[]):
    """Cls against their predictions ({catalog_dir}/plots); cls_kwargs are the Cls options (shared_wsp, randoms),
    and shapes those of the shear legs.
    """
    catalog = _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    catalog.plot_cls(shapes=shapes, **{k: v for k, v in (cls_kwargs or {}).items() if k in ('shared_wsp', 'randoms')})
    _free_memory()


# endregion

# ======================================================================================================================
# region Builders


def _make_shell_builder(reader_dict, shell_dict, baryon_dict):
    """Return the shell builder of the options: dark matter only, or baryonified with ``baryon_dict``."""
    if baryon_dict is not None:
        return _make_baryon_builder(reader_dict, shell_dict, baryon_dict)
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    return pollux.shells.SnapshotShellBuilder(reader, **shell_dict)


def _make_halo_builder(reader_dict, shell_dict):
    """Halo catalogs: all halos (no subsampling of the boxes)."""
    import pollux
    return pollux.shells.SnapshotShellBuilder(pollux.reader.HACCHaloReader(**reader_dict), **{**shell_dict, 'subsample_box': None})


def _make_baryon_builder(reader_dict, shell_dict, baryon_dict):
    """Return the baryonified shell builder of the options (:class:`pollux.baryons.BaryonifiedShellBuilder`)."""
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
    """Return the lensing builder of a shell builder: of n(z) bins (``lensing_dict``), or per shell if None."""
    import pollux
    if lensing_dict is None:
        return builder.snapshot_lensing_builder()
    return pollux.lensing.BornLensingBuilder(builder, **lensing_dict)


def _make_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict):
    """Return the catalog of the options, with its IA and shear augmentors."""
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


def _free_memory():
    """At the end of memory-heavy tasks: pyplot keeps figures alive until closed, and freed memory is not always
    returned to the system, so in a long-lived worker a task's peak (and its plots) would otherwise stay for the
    next tasks.
    """
    import ctypes
    import gc
    import matplotlib.pyplot as plt
    plt.close('all')
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


# endregion

# ======================================================================================================================
# region Skipping


def _done(outputs, options_id=False):
    """Return whether the outputs of a task are saved, so that the task is skipped.

    Products are written atomically, so a file that exists is complete. For a statistic (SACC file), ``options_id``
    is the identifier of the options it is measured with (None if unknown), recorded in its metadata by the combining
    tasks: a statistic saved with other options raises, instead of being kept as if it were up to date. Statistics
    saved before the identifier was recorded are kept, with a warning.
    """
    import warnings
    from pollux.io import PolluxIO
    if os.environ.get('POLLUX_CANARY'):  # a canary remakes its products (see pollux.pipeline.safeguards)
        return False
    if not all(os.path.exists(fname) for fname in outputs):
        return False
    if options_id is not False:
        for fname in outputs:
            saved = PolluxIO.read_sacc(fname).metadata.get('options_id')
            if saved is None:
                warnings.warn(f"{fname} has no identifier of the options it was measured with: kept as is.")
            elif options_id is not None and saved != options_id:
                raise ValueError(f"{fname} was measured with other options (identifier {saved}, now {options_id}): move it away to measure it again.")
    return True


def _run(outputs, compute):
    """Run compute (which writes the outputs) unless they are all saved (see _done); returns whether it ran."""
    if _done(outputs):
        return False
    compute()
    return True


def _params_id(params):
    """Identifier (8 hexadecimal characters) of the parameters a statistic depends on (binning, statistics options)
    the files of the bin pairs go to a folder named with it, so that results with other parameters are not mixed.
    """
    import hashlib
    import json
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _params_id_of_pairs(fnames):
    """Identifier of the parameters of the bin pairs combined, from the folder of their files (None if all skipped)."""
    ids = {os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1] for f in fnames if f is not None}
    if len(ids) > 1:
        raise ValueError(f"Bin pairs measured with different options: {sorted(ids)}")
    return ids.pop() if ids else None


def _compute_pair(fname, compute):
    """Statistic of one bin pair, computed and saved to fname unless already saved (resumed runs); returns fname,
    read by the task combining the pairs (so results go through the disk, not through the main process).
    """
    from pollux.io import PolluxIO
    _run([fname], lambda: PolluxIO.write_pickle(fname, compute()))
    return fname


def _read_pairs(fnames):
    """Results of the bin pairs saved by _compute_pair (None for tasks skipped because their product exists)."""
    from pollux.io import PolluxIO
    for fname in fnames:
        if fname is not None:
            yield PolluxIO.read_pickle(fname)


def _smoothed_outputs(builder, imap, whats, smooth_Mpc, save_field_map):
    """Maps whats of one map at smooth_Mpc, and its smoothed field with save_field_map (and smooth_Mpc)."""
    if save_field_map and smooth_Mpc is not None:
        whats = (*whats, builder.smoothed_field_what)
    return [builder.path(imap, what=what, smooth_Mpc=smooth_Mpc) for what in whats]


# endregion
