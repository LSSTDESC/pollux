# Roadmap

Planned developments of pollux. Remove an item once it is done (the git history keeps the details).

- Galaxies
  - Sample galaxies from halos with a halo occupation distribution (HOD), as an alternative to the biased sampling of the density field (`HOD.py`, placeholder for now).
  - Make the pixel-scale floor of the smoothing (`Tracer.effective_smooth_Mpc`) a parameter.
- Systematics
  - Variable depth on the sky: galaxy density and shape noise varying with position.
  - Variable redshift distributions on the sky.
- Lensing
  - Full ray tracing on the sphere (multiple-plane lensing), beyond the Born approximation, possibly with [Dorian](https://gitlab.mpcdf.mpg.de/fferlito/dorian).
- Statistics
  - Jackknife errors of the catalog 2pt functions: TreeCorr patches (`npatch`) with `var_method='jackknife'`, and the full covariance (`treecorr.estimate_multi_cov`). Only shot-noise errors are measured for now.
  - Theory covariances from [CosmoCov](https://github.com/CosmoLike/CosmoCov).
  - Understand the deficit of a few percent in the galaxy clustering of tomographic bins 2–4 at large scales, which does not come from clipping.
- Performance
  - Move what can be to GPUs (spherical harmonic transforms, painting particles on maps, baryonification, catalog sampling).
- Data products
  - Register the products in the [DESC data registry](https://github.com/LSSTDESC/dataregistry): a script registering each product directory, with its `config.yaml` (parameters and inputs) and the code version of its run. The products are made in scratch and copied to the registry once validated.
