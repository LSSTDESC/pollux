from setuptools import find_packages, setup

setup(
    name='pollux',
    version='0.2.1',
    description='Package to create weak lensing simulations from N-body simulations for the Higher-Order Statistics Topical Team.',
    author='Cyrille Doux',
    author_email='doux@lpsc.in2p3.fr',
    url='https://github.com/LSSTDESC/pollux',
    license='BSD-3-Clause',
    license_files=['LICENSE'],
    classifiers=['License :: OSI Approved :: BSD License'],
    packages=find_packages(include=['pollux', 'pollux.*']),
    python_requires='>=3.10',
    # enough to read the products (pollux.PolluxLoader); conda-forge is the simplest way to get them all
    install_requires=['numpy', 'scipy', 'astropy', 'matplotlib', 'pyyaml', 'healpy', 'healsparse', 'pyccl', 'sacc', 'h5py', 'fitsio'],
    # to make products and statistics, and to run the pipeline (also BaryonForge, and the pygio bindings of GenericIO
    # to read HACC files, which are not on PyPI: see the documentation)
    extras_require={'all': ['numba', 'ducc0', 'pymaster', 'treecorr', 'skyproj', 'pynverse', 'tqdm', 'parsl']},
)
