# pollux

DESC package to create weak lensing and large-scale structure simulations from N-body simulations for the Higher-Order Statistics Topical Team.

## Features
- Interfaces with HACC and SLICS N-body simulations.
- Construction of particle and halo lightcones with replicated boxes and random isometries (translations, rotations and flips), typically identical for groups of shells of width given by the box size.
- Lensing in the Born approximation (`BornLensingBuilder`), including tomography from user-defined redshift tomographic distributions.
- Generation of tomographic galaxy catalogs using uniform sampling (`CatalogSourceUniform`) or clustered sampling via (linearly biased) sampling of the density field with Multinomial (`CatalogSourceClusteredMultinomial`, fixed total number of galaxies) or Poisson (`CatalogSourceClusteredPoisson`, fixed density) samplings.
- Efficient parallel computation using the [Parsl](https://parsl.readthedocs.io/en/stable/index.html) framework to distribute tasks over computing resources.
- Use of DESC tools: [CCL](https://github.com/LSSTDESC/CCL), [healsparse](https://github.com/LSSTDESC/healsparse) and [NaMaster](https://github.com/LSSTDESC/NaMaster).

## Organisation of the code
- `reader.py` implements `Reader` objects to read particle data from N-body simulation snapshots, and provides basic lightcone information (particle density, shell binning, etc.)
- `map_builder.py` implements `MapBuilder` abstract objects with their basic functionalities: I/O, spherical harmonic transforms, visualization.
- `shells.py` implements `ShellBuilder` objects (subclass of `MapBuilder`) to generate particle and halo lightcone shells.
- `lensing.py` implements `LensingBuilder` objects (subclass of `MapBuilder`) to generate lensing maps from lightcone shells. 
- `catalog.py` implements `CatalogGenerator` objects to generate tomographic galaxy catalogs from the lightcone shells and lensing maps.
- `utils.py` contains a nuber of helper functions for maps, Cl's computations and array manipulations.
- `cosmo.py` contains fixed cosmologies.

![Example pollux map images](pollux_maps.png)
