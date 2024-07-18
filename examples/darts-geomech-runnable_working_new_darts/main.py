"""
This script runs DARTS simulations for CO2 injection scenarios including geomechanics response

It sets up and executes multiple realizations of a DARTS simulation model
for CO2 injection, using permeability data from a geological dataset.
The script handles:

- Setting up simulation parameters
- Loading permeability data from a NetCDF file
- Running multiple realizations with different injection rate profiles
- Extracting and saving observations at specified monitoring points
- Printing summary information for each realization

The script uses functions from runnable_darts_geomech.py to set up and run
the simulations. It demonstrates how to use the DARTS framework for
coupled flow and geomechanical simulations in a CO2 storage context.

Usage:
    Run this script directly to execute the simulations.

"""

from runnable_darts_geomech import run_model, convert_mt_per_year_to_kmol_per_day
import numpy as np
import xarray as xr
import os

np.random.seed(0)
physics_case = 'ccus'
case = '64x64x1'
geomech_mode = 'surface'
dt = 30
n_time_steps_ = 12 * 1  # 10 year

ds = xr.open_dataset('/samoa/data/smrserraoseabr/GenerateModels/geological_datasets/FINAL_realizations_with_POR_PERM_WELL_CONSTRAINED_64x64.nc')

rate_inj_ccus = convert_mt_per_year_to_kmol_per_day(0.5)
bhp_inj_ccus = 113.08 * 2

realizations = range(2)

monitoring_points = {
    'U_z': [
        {'time': -1, 'X': slice(None), 'Y': 0, 'Z': 0},
        {'time': slice(None), 'X': 0, 'Y': 0, 'Z': 0}
    ],
    'BHP': [
        {'time': slice(None)},
    ]
}

for realization in realizations:
    rate_inj_ccus_array_multipliers = [1] + [np.random.uniform(0, 3) for _ in range(10)]
    print('Rate multipliers:')
    print(rate_inj_ccus_array_multipliers)
    
    output_dir = f'runs/{physics_case}_{case}/'
    os.makedirs(output_dir, exist_ok=True)
    case_name = f'DARTS_simulation_realization_{realization}'
    
    log_perm = ds['log_permeability'].values[realization, :, :, :]
    perm = np.exp(log_perm).flatten()
    
    m, observations = run_model(case=case,
                                case_name=case_name,
                                physics_case=physics_case,
                                geomech_mode=geomech_mode,
                                geomech_period=1,
                                permx=perm,
                                dir_out=output_dir,
                                update_rates=True,
                                rate_inj_ccus=rate_inj_ccus,
                                rate_inj_ccus_array_multipliers=rate_inj_ccus_array_multipliers,
                                bhp_inj_ccus=bhp_inj_ccus, #this is only the BHP constraint for the injection well - controls are actually but rate
                                dt=dt,
                                n_time_steps_=n_time_steps_,
                                monitoring_points=monitoring_points,
                                save_observations=True)
    
    print(f"Observations for realization {realization}:")
    for var_name, var_data in observations.items():
        print(f"  {var_name}:")
        print("    Dictionary storage:")
        for point_key, point_data in var_data['dict'].items():
            print(f"      {point_key}:")
            print(f"        Shape: {point_data['data'].shape}")
            print(f"        Coordinates: {point_data['coords']}")
        print("    List of numpy arrays:")
        for i, arr in enumerate(var_data['numpy']):
            print(f"      Point {i} shape: {arr.shape}")
    
    del m
