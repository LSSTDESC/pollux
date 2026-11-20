Statistics
==========

Statistics of the products, as mixins of the builders and catalogs: :class:`~pollux.map_builder.MapBuilder` inherits
:class:`~pollux.statistics.maps.MapStatistics` (tracers and predictions, power spectra, moments, Gaussian maps and
plots of the maps), and :class:`~pollux.catalog.CatalogGenerator` inherits
:class:`~pollux.statistics.catalogs.CatalogStatistics` (Cls and two-point functions of the catalogs, their
predictions and plots). Their methods are called on the builders and catalogs, e.g. ``shell_builder.compute_cls()``.

.. automodule:: pollux.statistics

Maps
----

.. automodule:: pollux.statistics.maps
   :members:
   :undoc-members:
   :show-inheritance:

Catalogs
--------

.. automodule:: pollux.statistics.catalogs
   :members:
   :undoc-members:
   :show-inheritance:

Plots
-----

.. automodule:: pollux.statistics.plotting
   :members:
   :undoc-members:
