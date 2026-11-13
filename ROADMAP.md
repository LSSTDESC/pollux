# Roadmap

Planned developments of pollux, with their GitHub issue when there is one. Remove an item once it is done (the git history keeps the details).

- Simulations
  - Ingest the Gower Street simulations ([#30](https://github.com/LSSTDESC/pollux/issues/30)).
- Galaxies
  - Sample galaxies from halos with a halo occupation distribution (HOD), as an alternative to the biased sampling of the density field (`HOD.py`, placeholder for now) ([#9](https://github.com/LSSTDESC/pollux/issues/9)).
  - Make the pixel-scale floor of the smoothing (`Tracer.effective_smooth_Mpc`) a parameter.
- Systematics
  - Variable depth on the sky: galaxy density and shape noise varying with position.
  - Variable redshift distributions on the sky.
  - Switch to the SRD redshift distributions of [CCLX](https://github.com/LSSTDESC/CCLX/blob/master/LSST_SRD_Redshift_Distributions_and_Binning.ipynb) ([#29](https://github.com/LSSTDESC/pollux/issues/29)).
- Lensing
  - Full ray tracing on the sphere (multiple-plane lensing), beyond the Born approximation, possibly with [Dorian](https://gitlab.mpcdf.mpg.de/fferlito/dorian).
  - Magnification ([#14](https://github.com/LSSTDESC/pollux/issues/14)).
  - CMB lensing ([#15](https://github.com/LSSTDESC/pollux/issues/15)).
- Statistics
  - Jackknife errors of the catalog 2pt functions: TreeCorr patches (`npatch`) with `var_method='jackknife'`, and the full covariance (`treecorr.estimate_multi_cov`). Only shot-noise errors are measured for now.
  - Theory covariances from [CosmoCov](https://github.com/CosmoLike/CosmoCov) or [TJPCov](https://github.com/LSSTDESC/TJPCov) ([#16](https://github.com/LSSTDESC/pollux/issues/16)).
  - Understand the deficit of a few percent in the galaxy clustering of tomographic bins 2–4 at large scales, which does not come from clipping.
- Performance
  - Column-oriented catalog format (Parquet or HDF5), so that reading a few columns or a subsample does not read whole FITS rows.
  - Move what can be to GPUs (spherical harmonic transforms, painting particles on maps, baryonification, catalog sampling).
- Tests
  - Unit tests (pytest) on small synthetic inputs (`DummyReader`, low nside), and tests on the simulations run at NERSC only.
  - Continuous integration with GitHub Actions: unit tests and a strict build of the documentation on each pull request.
- Data products
  - Register the products in the [DESC data registry](https://github.com/LSSTDESC/dataregistry): a script registering each product directory, with its `config.yaml` (parameters and inputs) and the code version of its run. The products are made in scratch and copied to the registry once validated ([#12](https://github.com/LSSTDESC/pollux/issues/12)).
