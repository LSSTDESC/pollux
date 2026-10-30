from parsl import python_app


def _make_baryon_builder(reader_dict, shell_dict, baryon_dict):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    reader_halo = pollux.reader.HACCHaloReader(**reader_dict)
    builder_halo = pollux.shells.SnapshotShellBuilder(reader_halo, **shell_dict)
    builder_params = dict(baryon_dict)
    tag = builder_params.pop('tag', None)
    bfg_params = builder_params.pop('bfg_params')
    # the same profiles baryonify the shells and correct the predicted P(k)
    dmo, dmb = pollux.baryons.s19_profiles(bfg_params, builder_params['mass_def'])
    return pollux.baryons.BaryonifiedShellBuilder(builder, builder_halo, dmo, dmb, tag=tag, **builder_params)


def _make_shell_builder(reader_dict, shell_dict, baryon_dict):
    if baryon_dict is not None:
        return _make_baryon_builder(reader_dict, shell_dict, baryon_dict)
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    return pollux.shells.SnapshotShellBuilder(reader, **shell_dict)


def _close_figures():
    # pyplot keeps figures alive until closed; in a long-lived worker, full-sky plots
    # would otherwise keep their memory (~170 GB at nside 8192) for later tasks
    import gc
    import matplotlib.pyplot as plt
    plt.close('all')
    gc.collect()


def _make_lensing(builder, lensing_dict):
    import pollux
    if lensing_dict is None:
        return builder.snapshot_lensing_builder()
    return pollux.lensing.BornLensingBuilder(builder, **lensing_dict)


##############################
# Shells (dark-matter density maps from sub-volumes)


@python_app(executors=['default'])
def construct_shell(ijob, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_maps_sub(None, None, ijob=ijob)


@python_app(executors=['default'])
def construct_full_shells(ishell, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.combine_maps_subs(ishell)


@python_app(executors=['default'])
def construct_orphans(ishell, reader_dict, shell_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_maps_sub(None, None, do_orphan_ishell=ishell)


@python_app(executors=['heavy'])
def compute_shells_alms(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_alms(ishell)


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


@python_app(executors=['highmem'])
def compute_shells_cls_sub(pair, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # Cls of one (i, j) pair of shells
    return _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_cls(pairs=[pair], save=False, use_precomputed_alms=True)


@python_app(executors=['highmem'])
def compute_shells_cls(reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    # inputs contains (ell, cl) results from compute_shells_cls_sub
    builder = _make_shell_builder(reader_dict, shell_dict, baryon_dict)
    cls_out = {}
    for ell, cl in inputs:
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    builder._save_cls(ell, cls_out)
    builder.plot_cls(auto_only=True, logy=True)
    _close_figures()


##############################
# Halo catalogs and baryonification


@python_app(executors=['highmem'])
def construct_halo_catalog(ishell, reader_dict, shell_dict, baryon_dict, inputs=[]):
    import pollux
    reader = pollux.reader.HACCHaloReader(**reader_dict)
    builder = pollux.shells.SnapshotShellBuilder(reader, **shell_dict)
    builder.construct_catalogs(ishell, extra_columns=[baryon_dict['mass_name']])


@python_app(executors=['highmem'])
def construct_maps(ishell, reader_dict, shell_dict, baryon_dict=None, inputs=[]):
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).construct_maps(ishell)


##############################
# Kappa/shear (generic across dark-matter/baryonified shells and
# tomographic-SRD/per-snapshot lensing, selected via baryon_dict/lensing_dict)


@python_app(executors=['default'])
def construct_kappas(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).construct_maps(ikappa)


@python_app(executors=['heavy'])
def construct_kappas_alms(ikappa, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_alms(ikappa)


@python_app(executors=['heavy'])
def compute_shear_maps(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc=None, save_field_map=True, baryon_dict=None, inputs=[]):
    # shear maps, smoothed with smooth_Mpc (then also the smoothed convergence map with save_field_map)
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_shear_maps(
        ikappa, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map)


@python_app(executors=['heavy'])
def compute_smoothed_kappa_maps(ikappa, reader_dict, shell_dict, lensing_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    # smoothed convergence map, when its smoothing differs from the shear's
    _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_maps_from_alms(ikappa, smooth_Mpc=smooth_Mpc)


@python_app(executors=['highmem'])
def compute_kappas_cls_sub(pair, reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # Cls of one (i, j) pair of convergence maps
    return _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict).compute_cls(
        pairs=[pair], save=False, use_precomputed_alms=True)


@python_app(executors=['highmem'])
def compute_kappas_cls(reader_dict, shell_dict, lensing_dict, baryon_dict=None, inputs=[]):
    # inputs contains (ell, cl) results from compute_kappas_cls_sub
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    cls_out = {}
    for ell, cl in inputs:
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._save_cls(ell, cls_out)
    # auto-spectra only for the per-shell snapshot maps (all pairs would be too many)
    lensing.plot_cls(auto_only=all(i == j for i, j in cls_out), logy=True)
    lensing.plot_maps()
    _close_figures()


@python_app(executors=['heavy'])
def compute_tidal_maps(ishell, reader_dict, shell_dict, smooth_Mpc, save_field_map=True, baryon_dict=None, inputs=[]):
    # tidal field maps, smoothed with smooth_Mpc (then also the smoothed density map with save_field_map)
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_tidal_tensor_maps(ishell, smooth_Mpc=smooth_Mpc, save_field_map=save_field_map)


@python_app(executors=['heavy'])
def compute_smoothed_density_maps(ishell, reader_dict, shell_dict, smooth_Mpc, baryon_dict=None, inputs=[]):
    # smoothed density map, when its smoothing differs from the tidal field's
    _make_shell_builder(reader_dict, shell_dict, baryon_dict).compute_maps_from_alms(ishell, smooth_Mpc=smooth_Mpc)


def smoothing_plan(catalog_dict):
    """Return the smoothed maps needed by a catalog (see CatalogSourceGenerator smooth_fields/smooth_Mpc).

    Returns a dict with the smoothing of the tidal and shear maps ('tidal', 'shear'), whether these
    tasks also save the smoothed density/convergence ('density_with_tidal', 'kappa_with_shear'),
    and the other smoothing scales of density/convergence maps, made by separate tasks
    ('density_extra', 'kappa_extra').
    """
    import inspect
    import pollux
    cls = pollux.catalog.CatalogSourceGenerator
    default_fields = inspect.signature(cls.__init__).parameters['smooth_fields'].default
    scales = cls.smoothing_scales(catalog_dict.get('smooth_fields', default_fields), catalog_dict.get('smooth_Mpc'))
    tidal, shear = scales['t11'], scales['gamma1']
    density = {scales['positions'], scales['density']} - {None}
    kappa = {scales['kappa']} - {None}
    return dict(
        tidal=tidal,
        shear=shear,
        density_with_tidal=tidal in density,
        kappa_with_shear=shear in kappa,
        density_extra=sorted(density - {tidal}),
        kappa_extra=sorted(kappa - {shear}))


def catalog_map_tasks(nshells, reader_dict, shell_dict, catalog_dict, kappa_alms, shell_inputs, baryon_dict=None):
    """Launch the per-shell map tasks needed by catalogs: shear, tidal and smoothed density/convergence maps.

    kappa_alms are the per-shell (snapshot) convergence a_lm futures, and shell_inputs(ishell) the futures
    the shell maps depend on. Returns, per shell, the list of futures of its maps.
    """
    plan = smoothing_plan(catalog_dict)
    per_shell = []
    for ishell in range(nshells):
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
            compute_smoothed_density_maps(ishell, reader_dict, shell_dict, s, baryon_dict=baryon_dict, inputs=shell_inputs(ishell))
            for s in plan['density_extra']
        ]
        futures += [
            compute_smoothed_kappa_maps(ishell, reader_dict, shell_dict, None, s, baryon_dict=baryon_dict, inputs=[kappa_alms[ishell]])
            for s in plan['kappa_extra']
        ]
        per_shell.append(futures)
    return per_shell


##############################
# Source catalogs


def _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict):
    import pollux
    from pollux.ia import IACalculator, CatalogAugmentorShear
    lensing = _make_lensing(_make_shell_builder(reader_dict, shell_dict, baryon_dict), lensing_dict)
    ia = IACalculator(**ia_dict)
    shear_aug = CatalogAugmentorShear(
        compute_reduced_shear=True,
        generate_intrinsic=True,
        compute_observed=True,
    )
    return pollux.catalog.CatalogSourceClusteredPoisson(
        lensing,
        augmentors=[ia, shear_aug],
        **catalog_dict,
    )


@python_app(executors=['heavy'])
def construct_source_catalog(ikappa, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict).generate(ikappa)


@python_app(executors=['highmem'])
def construct_source_catalog_shell(ishell, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # galaxies of one shell, for all tomographic bins (partial catalogs)
    _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict).generate(ishell=ishell)


@python_app(executors=['highmem'])
def merge_source_catalog(ikappa, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains the construct_source_catalog_shell futures of all shells
    _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict).combine_shells(ikappa)


##############################
# Catalog statistics (2pt functions, Cls) and comparison to predictions


@python_app(executors=['heavy'])
def compute_catalog_cls_wsp(stat, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # workspace shared by all bin pairs of one statistic, from the footprint (compute_cls and predict_cls with shared_wsp=True)
    import os
    catalog = _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    if not os.path.exists(catalog._wsp_fname(stat)):
        catalog.get_wsp(stat)


@python_app(executors=['heavy'])
def compute_catalog_cls_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # Cls of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair
    catalog = _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    return catalog.compute_cls(what=stat, pairs=[pair], save=False, **stats_dict['common'], **stats_dict['cls'])


@python_app(executors=['default'])
def compute_catalog_cls(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains (ell, cl) results from compute_catalog_cls_sub
    cls_out = {}
    for ell, cl in inputs:
        for data_type, pairs in cl.items():
            cls_out.setdefault(data_type, {}).update(pairs)
    _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)._save_cls(ell, cls_out)


@python_app(executors=['heavy'])
def compute_catalog_2pt_sub(stat, pair, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, stats_dict, baryon_dict=None, inputs=[]):
    # 2pt function of one statistic ('clustering', 'shear' or 'shear_clustering') for one bin pair
    catalog = _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    return catalog.compute_2pt(what=stat, pairs=[pair], save=False, error_bars=True, theta_per_pair=True, **stats_dict['common'], **stats_dict['2pt'])


@python_app(executors=['default'])
def compute_catalog_2pt(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, inputs=[]):
    # inputs contains (theta, xi, dxi) results from compute_catalog_2pt_sub
    theta_out, xi_out, dxi_out = {}, {}, {}
    for theta, xi, dxi in inputs:
        for out, res in ((theta_out, theta), (xi_out, xi), (dxi_out, dxi)):
            for data_type, pairs in res.items():
                out.setdefault(data_type, {}).update(pairs)
    _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)._save_2pt(theta_out, xi_out, dxi=dxi_out)


@python_app(executors=['highmem'])
def plot_catalog_stats(
        kinds, reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict=None, with_dmo_reference=False, cls_kwargs=None, inputs=[]):
    # predictions and comparison plots ({catalog_dir}/plots) for kinds in ('2pt', 'cls'); predictions use the matter
    # power spectrum of the catalog's shell builder (S19-corrected with baryon_dict), and with_dmo_reference adds the
    # ratio to the dark-matter-only catalog's 2pt functions; cls_kwargs are the Cls options (shared_wsp, randoms)
    catalog = _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, baryon_dict)
    reference = None
    if baryon_dict is not None and with_dmo_reference:
        reference = _make_source_catalog(reader_dict, shell_dict, lensing_dict, ia_dict, catalog_dict, None)
    if '2pt' in kinds:
        catalog.plot_2pt(reference=reference)
    if 'cls' in kinds:
        catalog.plot_cls(**{k: v for k, v in (cls_kwargs or {}).items() if k in ('shared_wsp', 'randoms')})
    _close_figures()
