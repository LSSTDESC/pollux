from setuptools import find_packages, setup

setup(
    name='pollux',
    version='0.1',
    description='Package to create weak lensing simulations from N-body simulations for the Higher-Order Statistics Topical Team.',
    author='Cyrille Doux',
    author_email='doux@lpsc.in2p3.fr',
    url='https://github.com/LSSTDESC/pollux',
    license='BSD-3-Clause',
    license_files=['LICENSE'],
    classifiers=['License :: OSI Approved :: BSD License'],
    packages=find_packages(include=['pollux', 'pollux.*']),
)
