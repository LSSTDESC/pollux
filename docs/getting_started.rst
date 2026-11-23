Getting started
===============

Installation
------------

pollux needs Python 3.10 or later and CCL, healpy, healsparse, NaMaster, TreeCorr, sacc, numba, fitsio, h5py, skyproj,
parallel_sort and pynverse, plus `Parsl <https://parsl.readthedocs.io/>`_ to run the pipeline,
`BaryonForge <https://github.com/DhayaaAnbajagane/BaryonForge>`_ for baryonification, and the ``pygio`` Python
bindings of `GenericIO <https://git.cels.anl.gov/hacc/genericio>`_ to read HACC files.

Then install pollux from a clone of the repository::

    pip install -e .

Overview
--------

pollux turns the particles of an N-body simulation into maps and mock catalogs, in a chain of objects, each building
on the previous one:

1. a **reader** (:mod:`pollux.reader`) gives access to the particles (and halos) of the simulation, its cosmology
   and its snapshots;
2. a **shell builder** (:mod:`pollux.shells`) bins the particles into HEALPix maps of shells of comoving distance:
   the matter lightcone (:doc:`shells`);
3. optionally, a **baryonified shell builder** (:mod:`pollux.baryons`) displaces the matter of these shells around
   halos (:doc:`baryons`);
4. a **lensing builder** (:mod:`pollux.lensing`) sums weighted shells into convergence maps, from which come shear
   maps (:doc:`lensing`);
5. a **catalog generator** (:mod:`pollux.catalog`) draws galaxies from the shells, interpolates the maps at their
   positions, and adds intrinsic alignments and observed shapes (:doc:`catalogs`, :doc:`intrinsic_alignments`).

All builders write their products to disk, in a common layout (:doc:`outputs`), measure their power spectra or
correlation functions, and predict them with CCL (:doc:`statistics`). Large runs chain the same steps as Parsl
tasks on a cluster (:doc:`pipeline`).

A first example
---------------

The following goes from a HACC simulation to a tomographic catalog with intrinsic alignments, and compares its 2pt
functions to theory. At low resolution it fits on one node; :doc:`pipeline` distributes the same steps.

.. code-block:: python

    import numpy as np
    import pollux
    from pollux.catalog import CatalogAugmentorShear
    from pollux.ia import CatalogAugmentorIA

    # simulation: one snapshot out of two (each one makes a shell)
    reader = pollux.reader.HACCReader(path='/path/to/simulation', run_params='run/indat.params', snap_sub_every_n=2)

    # products of the simulation on the full sky at nside 512 (one root directory per footprint, see the outputs)
    footprint = pollux.Footprint(512, pollux.PolluxIO('/path/to/outputs/simulation/nside512'))

    # shells of the realization seed 0: replicated boxes with random isometries
    shells = pollux.shells.SnapshotShellBuilder(reader, footprint, seed=0, apply_grid_isometry=True, group_isometries=True)
    shells.construct_maps()
    shells.compute_alms()
    shells.compute_tidal_tensor_maps()  # tidal field of each shell, from its a_lm, for intrinsic alignments

    # convergence and shear maps of tomographic bins, from their redshift distributions
    z, *nzs = np.loadtxt('nz.txt', unpack=True)
    lensing = pollux.lensing.BornLensingBuilder(shells, nzs=[(z, nz) for nz in nzs], tag='my_nz')
    lensing.construct_maps()
    lensing.compute_shear_maps()

    # catalogs interpolate the convergence and shear of a source plane at each shell
    snapshots = shells.snapshot_lensing_builder()
    snapshots.construct_maps()
    snapshots.compute_shear_maps()

    # galaxies placed with 1 + 2 delta, with tidal alignments and observed shapes
    catalog = pollux.catalog.CatalogClusteredPoisson(
        lensing, tag='b2', nbar_input=1., bias=2.,
        augmentors=[CatalogAugmentorIA('TA', {'A_TA': 1.}), CatalogAugmentorShear(generate_intrinsic=True, compute_observed=True)])
    catalog.generate()

    # w(theta), gamma_t and xi_pm, against their predictions (plots in the catalog directory)
    catalog.compute_2pt(theta_min=1., theta_max=200., subsample=0.1)
    catalog.plot_2pt()

The notebooks (:doc:`notebooks`) follow the first steps on one shell, then read and analyse the products of a
complete run.
