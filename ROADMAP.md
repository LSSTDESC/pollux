# Roadmap

Planned technical developments of pollux, with their GitHub issue when there is one. The validation of the products and their known issues are in the report of the HACC-HOS-Y1 runs. Remove an item once it is done (the git history keeps the details).

- Simulations
  - Ingest the Gower Street simulations ([#30](https://github.com/LSSTDESC/pollux/issues/30)).
- Galaxies
  - Separate lens and source catalogs (e.g. maglim-like lenses and metadetect-like sources), each with its own n(z), density and interpolated fields (no shear for lenses): an option for catalogs without the shear and IA augmentors in the pipeline, and a catalog block per sample in the run scripts.
  - Sample galaxies from halos with a halo occupation distribution (HOD), as an alternative to the biased sampling of the density field (`HOD.py`, placeholder for now) ([#9](https://github.com/LSSTDESC/pollux/issues/9)).
  - Make the pixel-scale floor of the smoothing (`Tracer.effective_smooth_Mpc`) a parameter.
  - Remove the pixel effects from the catalogs, so that they compare to data at all scales above a recorded resolution:
    - evaluate the fields exactly at the galaxy positions from their a_lm (e.g. `ducc0.sht.experimental.synthesis_general`), instead of the bilinear interpolation of maps, which adds a window of its own;
    - deconvolve the pixel window of the shear and convergence (as done for the smoothed fields), with a band limit or mild smoothing against the particle shot noise near ℓmax;
    - draw the galaxy positions from the density interpolated within each pixel, instead of uniformly;
    - record the transfer function of each field in the catalog's `config.yaml` (smoothing per shell, band limit, pixel window), read by the predictions (`pixel_window` of `predict_cls` and `predict_2pt`) instead of set by hand, and documented as the scale cuts of the catalogs.
- Systematics
  - Variable depth on the sky: galaxy density and shape noise varying with position.
  - Variable redshift distributions on the sky.
  - Switch to the SRD redshift distributions of [CCLX](https://github.com/LSSTDESC/CCLX/blob/master/LSST_SRD_Redshift_Distributions_and_Binning.ipynb) ([#29](https://github.com/LSSTDESC/pollux/issues/29)).
- Lensing
  - Full ray tracing on the sphere (multiple-plane lensing), beyond the Born approximation, possibly with [Dorian](https://gitlab.mpcdf.mpg.de/fferlito/dorian).
  - Magnification ([#14](https://github.com/LSSTDESC/pollux/issues/14)).
  - CMB lensing ([#15](https://github.com/LSSTDESC/pollux/issues/15)).
- Statistics
  - Measure the 2pt functions and Cls of the catalogs with [TXPipe](https://github.com/LSSTDESC/TXPipe), the DESC pipeline: export of the catalogs, tomography and n(z) to its inputs, including 3x2pt between separate lens and source catalogs. pollux's own measurements stay as a quick validation of the products.
  - Covariance module (`pollux.covariance`): the Gaussian covariance of the Cls of maps (pollux's predictions) saved with them in `cls.fits`, and that of the catalog statistics from [TJPCov](https://github.com/LSSTDESC/TJPCov) ([#16](https://github.com/LSSTDESC/pollux/issues/16)), as a pipeline task. Prototype with TJPCov master in the benchmarks of the HACC-HOS-Y1 tests; contribute its speed-ups (binning, projection of diagonal covariances) and the fix of the assembly of the xi- blocks upstream.
  - Covariance improvements: bin-averaged Wigner transforms and exact pair counts for the noise of the 2pt functions (TJPCov branch `real_nmt_ryo`), multipoles beyond 3 nside with the pixel window on the signal only, non-Gaussian terms (super-sample and connected, halo model), intrinsic alignments.
  - Survey configurations (n_eff per bin, shape noise, sky fraction; e.g. LSST Y1 and Y10) to compute the covariance of a survey on the predictions of the products, and judge their deviations in units of its errors; and optionally generate catalogs matching a survey.
  - Jackknife errors of the catalog 2pt functions: TreeCorr patches (`npatch`) with `var_method='jackknife'`, and the full covariance (`treecorr.estimate_multi_cov`). Only shot-noise errors are measured for now.
- Performance
  - Column-oriented catalog format (Parquet or HDF5), so that reading a few columns or a subsample does not read whole FITS rows.
  - Move what can be to GPUs (spherical harmonic transforms, painting particles on maps, baryonification, catalog sampling).
- Tests
  - Unit tests (pytest) on small synthetic inputs (`DummyReader`, low nside), and tests on the simulations run at NERSC only.
  - Continuous integration with GitHub Actions: unit tests and a strict build of the documentation on each pull request.
- Data products
  - Register the products in the [DESC data registry](https://github.com/LSSTDESC/dataregistry): a script registering each product directory, with its `config.yaml` (parameters and inputs) and the code version of its run. The products are made in scratch and copied to the registry once validated ([#12](https://github.com/LSSTDESC/pollux/issues/12)).
