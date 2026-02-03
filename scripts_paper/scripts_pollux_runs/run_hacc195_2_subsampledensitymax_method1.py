import os
import sys

import numpy as np
import parsl

import pollux

print(pollux.__file__)

from config import get_config
from parsl import python_app
from tqdm.auto import tqdm

##############################
# Get input arguments

densitymax_z=1.5

sim_relative_path = sys.argv[1]
snap_sub_every_n = int(sys.argv[2])
seed = int(sys.argv[3])
nside = 8192
sim_root = '/pscratch/sd/d/desc/HOS_Sims/'
save_dir_root = os.path.join(os.environ["PSCRATCH"], "HOS_Sims")
run_suffix = 'subsampledensitymax'+str(densitymax_z)+'_nside8192'

do_shells = False
do_haloes = False
do_haloes_props = False
do_shell_cls = False  #(snap_sub_every_n > 2)
do_lensing = False
do_IA = False
do_IA_smooth = True
do_catalogs_poisson = True
do_catalogs_uniform = False

debug = False

##############################
# Define reader, cosmo and builder objects

reader_dict = dict(
    path=os.path.join(sim_root, sim_relative_path),
    run_params='run/indat.params',
    snap_sub_every_n=snap_sub_every_n,
    group_along_x=True,
)
reader_dict_halo = dict(
    path=os.path.join(sim_root, sim_relative_path),
    run_params='run/indat.params',
    snap_sub_every_n=snap_sub_every_n,
    group_along_x=False,
)
reader = pollux.reader.HACCReader(**reader_dict)
cosmo = reader.cosmo

builder_dict = dict(
    nside=nside,
    save_dir=os.path.join(save_dir_root, sim_relative_path, f'shells_z{reader.nsnapshots}_{run_suffix}'),
    apply_grid_isometry=True,
    seed=seed,
    subsample_box={"densitymax_z": densitymax_z},
    group_isometries=True,
    dtype=np.float32,
    wsp_dir=save_dir_root,
)
builder = pollux.shells.SnapshotShellBuilder(reader, **builder_dict)

dndz = np.loadtxt('../data/SRDv1_nz.txt')
lensing_dict_srd = dict(
    nzs=[(dndz[:, 0], dndz[:, i]) for i in range(1, 6)], kappa_dir='SRDv1_nz', wsp_dir=save_dir_root)
lensing_srd = pollux.lensing.BornLensingBuilder(builder, **lensing_dict_srd)

lensing_dict_zs = dict(nzs='zsnapshots', kappa_dir='kappa_zsnapshots', wsp_dir=save_dir_root)
lensing_zs = pollux.lensing.BornLensingBuilder(builder, **lensing_dict_zs)

##############################
# Configure parsl

script_name = sys.argv[0]
args_str = "_".join([str(argv) for argv in sys.argv[2:]])
workflow_name = script_name + "_" + sim_relative_path + "_" + args_str

run_dir = os.path.join(save_dir_root, "runinfo", script_name, sim_relative_path, args_str)

@python_app(executors=['default'])
def make_objs(rd, bd, ld):
    import pollux
    reader = pollux.reader.HACCReader(**rd)
    builder = pollux.shells.SnapshotShellBuilder(reader, **bd)
    lensing = pollux.lensing.BornLensingBuilder(builder, **ld)
    return reader, builder, lensing

@python_app(executors=['default'])
def construct_shell(isub, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.construct_maps_sub(isub)

@python_app(executors=['default'])
def combine_maps_subs(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.combine_maps_subs(ishell)

@python_app(executors=['default'])
def check_shells(inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.check_maps(read=True, check_finite=True)
    builder.plot_maps(log=True)
    
@python_app(executors=['shear'])
def compute_shells_alms(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.compute_alms(ishell)

@python_app(executors=['default'])
def compute_wsp(inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.get_cls_wsp()

@python_app(executors=['default'])
def compute_shells_cls_sub(ipair, inputs=[]):
    reader, builder, lensing = inputs[0]
    return builder.compute_cls(save=False, use_precomputed_alms=True, pair_job=ipair, auto_only=True)

@python_app(executors=['default'])
def compute_shells_cls(inputs=[]):
    reader, builder, lensing = inputs[0]
    cls_out = {}
    for ell, cl in inputs[1:]:
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    builder._save_cls(ell, cls_out)
    builder.triangle_plot_cls(auto_only=True, logy=True)
    
@python_app(executors=['default'])
def compute_kappashells_cls_sub(ipair, inputs=[]):
    reader, builder, lensing = inputs[0]
    return lensing.compute_cls(save=False, use_precomputed_alms=True, pair_job=ipair, auto_only=True)

@python_app(executors=['default'])
def compute_kappashells_cls(inputs=[]):
    reader, builder, lensing = inputs[0]
    cls_out = {}
    for ell, cl in inputs[1:]:
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._save_cls(ell, cls_out)
    lensing.triangle_plot_cls(auto_only=True, logy=True)

@python_app(executors=['default'])
def construct_kappas(ikappa, inputs=[]):
    reader, builder, lensing = inputs[0]
    lensing.construct_maps(ikappa)

@python_app(executors=['shear'])
def compute_kappas_alms(ikappa, inputs=[]):
    reader, builder, lensing = inputs[0]
    lensing.compute_alms(ikappa)

@python_app(executors=['default'])
def compute_kappas_cls_sub(ipair, inputs=[]):
    reader, builder, lensing = inputs[0]
    return lensing.compute_cls(save=False, use_precomputed_alms=True, pair_job=ipair)

@python_app(executors=['default'])
def compute_kappas_cls(inputs=[]):
    reader, builder, lensing = inputs[0]
    cls_out = {}
    for ell, cl in inputs[1:]:
        k = list(cl.keys())[0]
        cls_out[k] = cl[k]
    lensing._save_cls(ell, cls_out)
    if lensing.nmaps < 10:
        lensing.triangle_plot_cls(logy=True)
    lensing.plot_maps()

@python_app(executors=['shear'])
def compute_shear_maps(ikappa, inputs=[]):
    reader, builder, lensing = inputs[0]
    lensing.compute_shear_maps(ikappa)

@python_app(executors=['default'])
def make_objs_halo(rd, bd, ld):
    import pollux
    reader = pollux.reader.HACCHaloReader(halo_path='analysis/b0168/Props',**rd)
    builder = pollux.shells.SnapshotShellBuilder(reader, **bd)
    lensing = pollux.lensing.BornLensingBuilder(builder, **ld)
    return reader, builder, lensing

@python_app(executors=['shear'])
def construct_catalog(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.construct_catalog(ishell)
    
@python_app(executors=['shear'])
def construct_catalog_props(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    
    import numpy as np
    from colossus.halo import mass_defs
    from astropy.io import fits
    from colossus.cosmology import cosmology
    from pollux.cosmo import cosmo_hacc as cosmo
    from pollux.utils import save_dict_fits_file

    cosmo_c = cosmology.setCosmology(
        cosmo_name='MyCosmo',
        flat=True,
        Om0=cosmo['Omega_m'],
        Ode0=None,
        Ob0=cosmo['Omega_b'],
        H0=cosmo['H0'],
        sigma8=cosmo['sigma8'],
        ns=cosmo['n_s'],
        de_model='lambda',
        w0=cosmo['w0'],
        wa=cosmo['wa'],
        wz_function=None,
        relspecies=True,
        Tcmb0=2.7255,
        Neff=3.046,
        power_law=False,
        power_law_n=0.0,
        interpolation=True,
        persistence='rw',
        print_info=False,
        print_warnings=True,
    )

    catalog=builder.read_catalogs_props(ishell,['ra','dec','chi','z',
                                                'sod_halo_com_vx','sod_halo_com_vy','sod_halo_com_vz',
                                                'sod_halo_1D_vel_disp',
                                                'fof_halo_mass',
                                                'sod_halo_mass','sod_halo_radius','sod_halo_cdelta'])

    catalog={key: np.where(value==-101.,np.nan,value) for key,value in catalog.items()}

    catalog['Mvir']=np.full(len(catalog['ra']),np.nan)
    catalog['Rvir']=np.full(len(catalog['ra']),np.nan)
    catalog['cvir']=np.full(len(catalog['ra']),np.nan)

    pos=np.argwhere(~np.isnan(catalog['sod_halo_mass']))
    
    if any(pos):
        catalog['Mvir'][pos],catalog['Rvir'][pos],catalog['cvir'][pos]=mass_defs.changeMassDefinition(
            catalog['sod_halo_mass'][pos],
            catalog['sod_halo_cdelta'][pos],
            reader.zsnapshots[ishell], #zsnapshot
            mdef_in='200c',
            mdef_out='vir',
        )
    
    save_dict_fits_file(builder.get_fname(ishell,what='haloes_props'),catalog)
    
@python_app(executors=['default'])
def construct_mock_catalog_poisson(ikappa, inputs=[]):
    import numpy as np
    import pollux
    reader, builder, lensing = inputs[0]
    bias = 1
    interp_pixels = True
    if interp_pixels:
        name = 'catpoisson_interppixels_b'+str(bias)
    else:
        name = 'catpoisson_nointerppixels_b'+str(bias)
    catalog = pollux.catalog.CatalogSourceClusteredPoisson(lensing, biases=np.full(5, bias), nbar=np.full(5, 6.0), interp_list=['kappa', 'gamma1', 'gamma2', 't11', 't12', 't22', 't11_0.5Mpc', 't12_0.5Mpc', 't22_0.5Mpc', 'delta'], name=name, interp_pixels=interp_pixels)
    catalog.generate(ikappa)

@python_app(executors=['default'])
def construct_mock_catalog_uniform(ikappa, inputs=[]):
    import numpy as np
    import pollux
    reader, builder, lensing = inputs[0]
    interp_pixels = True
    if interp_pixels:
        name = 'catuniform_interppixels'
    else:
        name = 'catuniform_nointerppixels'
    catalog = pollux.catalog.CatalogSourceUniform(lensing, nbar=np.full(5, 6.0), interp_list=['kappa', 'gamma1', 'gamma2', 't11', 't12', 't22', 't11_0.5Mpc', 't12_0.5Mpc', 't22_0.5Mpc', 'delta'], name=name, interp_pixels=interp_pixels)
    catalog.generate(ikappa)
    
@python_app(executors=['default'])
def compute_IA_maps(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.compute_tidal_tensor_maps(ishell)

@python_app(executors=['default'])
def compute_IA_maps_smooth(ishell, inputs=[]):
    reader, builder, lensing = inputs[0]
    builder.compute_tidal_tensor_maps(ishell, smooth_Mpc=0.5)

#########################################
# Define tasks

args1 = reader_dict, builder_dict, lensing_dict_srd
args2 = reader_dict, builder_dict, lensing_dict_zs

args_halo = reader_dict_halo, builder_dict, lensing_dict_srd

futures = []

if __name__ == '__main__':

    parsl.load(get_config(run_dir, debug, workflow_name))

    # Construction of shells from sub-volumes
    if do_shells:
        job_list = range(reader.njobs)
        if False:
            job_list = job_list[::-1]
        if True:  # randomize jobs to distribute the load across workers
            job_list = np.random.permutation(reader.njobs)
        shell_sub = [construct_shell(isub, inputs=[make_objs(*args1)]) for isub in job_list]

        # Combination of full shells
        shell_full = [combine_maps_subs(ishell, inputs=[make_objs(*args1)] + shell_sub) for ishell in range(builder.nmaps)]

        # Check full shells
        shell_full_checked = [check_shells(inputs=[make_objs(*args1)] + shell_full)]

        futures += shell_sub + shell_full + shell_full_checked
    else:
        shell_full_checked = []

    # Compute halo catalogs
    if do_haloes:
        halos = [construct_catalog(ishell, inputs=[make_objs_halo(*args_halo)]) for ishell in range(builder.nmaps)]
        futures += halos
    else:
        halos = []
        
    # Compute halo catalogs with extra properties
    if do_haloes_props:
        halos_props = [construct_catalog_props(ishell, inputs=[make_objs_halo(*args_halo)] + halos) for ishell in range(builder.nmaps)]
        futures += halos_props

    # Compute full shells alms and cls Cls
    if do_shell_cls:
        shell_alms = [
            compute_shells_alms(ishell, inputs=[make_objs(*args1)] + shell_full_checked)
            for ishell in range(builder.nmaps)
        ]

        shell_cls_sub = [
            compute_shells_cls_sub(ipair, inputs=[make_objs(*args1)] + shell_alms)
            for ipair in range(builder.nmaps)
        ]
        shell_cls = compute_shells_cls(inputs=[make_objs(*args1)] + shell_cls_sub)

        futures += shell_alms + shell_cls_sub + [shell_cls]
    else:
        shell_alms = []
        
    # Compute IA maps
    if do_IA:
        IA_maps = [
            compute_IA_maps(ishell, inputs=[make_objs(*args1)] + shell_alms)
            for ishell in range(builder.nmaps)
        ]

        futures += IA_maps
    else:
        IA_maps = []

    # Compute IA maps (smoothed)
    if do_IA_smooth:
        IA_maps_smooth = [
            compute_IA_maps_smooth(ishell, inputs=[make_objs(*args1)] + shell_alms)
            for ishell in range(builder.nmaps)
        ]

        futures += IA_maps_smooth
    else:
        IA_maps_smooth = []

    # Compute kappa alms and maps
    if do_lensing:
        for lensing, args in zip((lensing_srd, lensing_zs), (args1, args2)):
            n = lensing.nmaps
            kappa_maps = [
                construct_kappas(ikappa, inputs=[make_objs(*args)] + shell_full_checked) for ikappa in range(n)
            ]

            kappa_alms = [
                compute_kappas_alms(ikappa, inputs=[make_objs(*args), kappa_maps[ikappa]]) for ikappa in range(n)
            ]

            shear_maps = [
                compute_shear_maps(ikappa, inputs=[make_objs(*args), kappa_alms[ikappa]]) for ikappa in range(n)
            ]

            futures += kappa_maps + kappa_alms + shear_maps

            if n < builder.nmaps:
                # kappa_cls = compute_kappas_cls(None, inputs=[make_objs(*args)] + kappa_alms)
                kappa_cls_sub = [
                    compute_kappas_cls_sub(ipair, inputs=[make_objs(*args)] + kappa_alms)
                    for ipair in range(int(n * (n + 1) / 2))
                ]
                kappa_cls = compute_kappas_cls(inputs=[make_objs(*args)] + kappa_cls_sub)

                futures += kappa_cls_sub + [kappa_cls]
            else:
                kappashell_cls_sub = [
                    compute_kappashells_cls_sub(ipair, inputs=[make_objs(*args)] + kappa_alms)
                    for ipair in range(n)
                ]
                kappashell_cls = compute_kappashells_cls(inputs=[make_objs(*args)] + kappashell_cls_sub)
                
                futures += kappashell_cls_sub + [kappashell_cls]
    else:
        shear_maps = []
        
    # Create the mock catalogs (poisson)
    if do_catalogs_poisson:
        lensing, args = lensing_srd, args1
        n = lensing.nmaps
        mock_catalog_poisson = [
            construct_mock_catalog_poisson(ikappa, inputs=[make_objs(*args)] + shell_full_checked + shear_maps + IA_maps + IA_maps_smooth) for ikappa in range(n)
        ]

        futures += mock_catalog_poisson
            
    # Create the mock catalogs (uniform)
    if do_catalogs_uniform:
        lensing, args = lensing_srd, args1
        n = lensing.nmaps
        mock_catalog_uniform = [
            construct_mock_catalog_uniform(ikappa, inputs=[make_objs(*args)] + shell_full_checked + shear_maps + IA_maps + IA_maps_smooth) for ikappa in range(n)
        ]

        futures += mock_catalog_uniform
        
    ##############################
    # Run tasks!
    for future in tqdm(futures):
        future.result()
