import os
import sys
import inspect
import pickle
import numpy as np
import xarray as xr
from typing import Dict, List, Optional

from darts.reservoirs.cpg_reservoir import read_arrays

from model_co2 import Model as Model_CCS
from geomechanics import calc_geomech

# Add parent directory to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'python'))


from tools import print_range
def get_case_files(case: str, physics_case: str) -> tuple:
    """
    Get the file paths for the simulation case.

    Args:
        case (str): The case identifier.
        physics_case (str): The physics case identifier.

    Returns:
        tuple: A tuple containing the paths to the grid file, property file, and schedule file.
    """
    prefix = os.path.join('meshes', case)
    gridfile = os.path.join(prefix, 'grid.grdecl')
    propfile = os.path.join(prefix, 'reservoir.in')
    sch_file = os.path.join(prefix, f'sch_{physics_case.split("_")[0]}.inc')
    
    for file in [gridfile, propfile, sch_file]:
        assert os.path.exists(file), f"File not found: {file}"
    
    return gridfile, propfile, sch_file

def run_model(case_name, case, dir_out=None, permx=None,
              write_grdecl=False,
              export_vtk=False,
              geomech_mode='surface',
              bhp_prod=None,#30, # well BHP constrain, bar            
              bhp_inj_ccus=None, # bar
              rate_prod=7500, # m3/day
              rate_inj_ccus= 5000*1.245*12.5, #10000*1.245*12.5, # kmol/day or 2.5 Mt/year (10000*365(days to year)*0.044(mols to tons)*1e-6 (to Mega)*12.5 (???)), for brugge is different(see below)
              update_rates = False,
              rate_inj_ccus_array_multipliers=None, # array with innjection rates multipliers for each time step
              temp_inj=None,  # degrees
              delta_p_prod=80,
              delta_p_inj_ccus=200,
              delta_t_ccus=40,
              stop_inj=False,
              plot=False,
              save_xls_pkl=False,
              compute_stress_flag=False,
              dt=30,
              n_time_steps_=12*10, # 10 years
              geomech_period=-1,
              physics_case='ccus',
              obl_cache=False,
              xarray_period=1,
              monitoring_points=None,
              save_observations=False):

    geomech_proxy = True
    gridfile, propfile, sch_fname = get_case_files(case, physics_case)
    

    #redirect_darts_output(os.path.join('logs', str(i)+'.log')

    arrays = read_arrays(gridfile, propfile)

    if permx is not None:
        arrays['PERMX'] = np.array(permx, dtype=np.float64).flatten()
        arrays['PERMY'] = arrays['PERMX']
        arrays['PERMZ'] = arrays['PERMX'] * 0.1
    else:  # perm from reservoir.in (non-esmda mode)
        perm_mult = 1

        for a in ['PERMX', 'PERMY', 'PERMZ']:
            arrays[a] *= perm_mult
            #arrays[a][:] = arrays[a].mean()  # replace by uniform


    print(physics_case, case)
    print('run_model: permx range:', arrays['PERMX'].min(), arrays['PERMX'].max(), 'mean:', arrays['PERMX'].mean())


    m = Model_CCS(arrays=arrays, geomech_mode=geomech_mode, obl_cache=obl_cache)
    n_time_steps = 4*10 if n_time_steps_ is None else n_time_steps_  # 10 years
    m.case = case

    if dir_out is None:
        dir_out = os.path.join(os.path.join(os.getcwd(), 'results'), physics_case + '_' + case)
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)  # clean previous output
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)



    rate_inj_1 = rate_inj_ccus
    bhp_inj_1 = bhp_inj_ccus  # bar
    delta_p_inj1 = delta_p_inj_ccus
    delta_t_inj1 = delta_t_ccus

    m.read_and_add_perforations(sch_fname=sch_fname, verbose=True)

    if obl_cache:
        cache_folder = 'obl_cache_' + physics_case + '_' + case
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        m.physics.cache_dir = cache_folder

    m.init()
    m.params.max_ts = dt

    ts = []#0.1, 1., 5., 10.]  # solver warming up timesteps; also to visualize the pressure changes
    n_warmup_timesteps = len(ts)
    t = dt
    for ti in range(n_time_steps):
        ts.append(t)
        t += dt
    #ts += [0.1, 1., 5., 10.] # to monitor the pressure after switching off the wells
    
    ts = np.array(ts)
    n_time_steps += n_warmup_timesteps
    #print all the variables related to time steps 


    m.store_initial_p_t()

    arr_initial = {}

    # save arrays at t=0
    if geomech_mode != 'none':
        if compute_stress_flag:
            compute_stress = True
        else:
            compute_stress = False
        arr_geomech = calc_geomech(m, geomech_mode, compute_stress=compute_stress, compute_displs=True)
        arr_initial.update(arr_geomech)

    # save initial arrays
    arrays_input = {}
    for k in ['SPECGRID', 'COORD', 'ZCORN', 'PORO', 'PERMX']:
        arrays_input[k] = arrays[k]
    m.save_cubes(dir_out, case_name, ti=0, arrays=arrays_input, write_grdecl=write_grdecl)

    m.save_cubes(dir_out, case_name, ti=0, arrays=arr_initial, write_grdecl=write_grdecl, mode_='a')

    #m.run_python(1)  # run 1 day for equilibrating (without wells)
    m.set_well_controls_custom(bhp_prod, bhp_inj_1, temp_inj, rate_prod, rate_inj_1, delta_p_prod, delta_p_inj1, delta_t_inj1)
    t = 0
    print_range(m, t)

    # timesteps when we save xarray (.nc)
    ts_xarray_ind = []
    for ti in range(n_time_steps):
        if ti == 0 or ((ti + 1) % xarray_period == 0) or (ti == n_time_steps - 1):
            ts_xarray_ind.append(ti)
    ts_xarray = np.array(ts[ts_xarray_ind])
    ti_x = 0

    ts_tmp = np.append(np.array([0.]), ts)  # add zero time for the next line, as we don't save it
    dt_list = ts_tmp[1:] - ts_tmp[:-1]
    print(f'Time steps (ts): {ts}')
    print(f'Total number of main time steps (n_time_steps): {n_time_steps}')
    print(f'Number of warm-up time steps (n_warmup_timesteps): {n_warmup_timesteps}')
    print(f'Duration of each time step in days (dt): {dt}')
    print(f'Current time in days (t): {t}')
    print(f'Indices of time steps for saving xarray datasets (ts_xarray_ind): {ts_xarray_ind}')
    print(f'Time points for saving xarray datasets (ts_xarray): {ts_xarray}')
    print(f'Current index for saving xarray datasets (ti_x): {ti_x}')
    print(f'List of time intervals between consecutive time steps (dt_list): {dt_list}')
    print(f'Modified time steps list with initial zero (ts_tmp): {ts_tmp}')
    print(f'Modified time steps from the second element onward (ts_tmp[1:]): {ts_tmp[1:]}')
    print(f'Modified time steps up to the second last element (ts_tmp[:-1]): {ts_tmp[:-1]}')

    for ti in range(len(dt_list)):
        dt = dt_list[ti]
      
        print(f'Current time (t): {t}')
        print(f'Current time step duration (dt): {dt}')
        print(f'Current time step index (ti) - Simulation step: {ti}')
        


        # stop the well after 10 years
        if stop_inj and t == 365*10 and physics_case == 'ccus_subsidence':
            print('stop well t=', t)
            m.close_wells()
            #for dt_a in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]:
            #    m.run(days=dt_a)
            
        if update_rates and t % 360 == 0:
            year_index = int(t // 360)            
            multiplier = rate_inj_ccus_array_multipliers[year_index]
            new_rate_inj_1 = multiplier * rate_inj_1
            print('-------------------------------------------------------------------------------------')
            print(f'New rate inj 1: {new_rate_inj_1} at time {t} - time step {ti}')
            print('-------------------------------------------------------------------------------------')
            m.set_well_controls_custom(bhp_prod, bhp_inj_1, temp_inj, rate_prod, new_rate_inj_1, delta_p_prod, delta_p_inj1, delta_t_inj1)

        m.run(days=dt)        

        t += dt
        print_range(m, t)

        arr_geomech = {}
        if geomech_proxy:
            if type(geomech_period) == list:
                compute_displs = (ti in geomech_period)
            else:
                compute_displs = ((ti+1) % geomech_period == 0) or (ti == 0) or (ti == n_time_steps - 1)  # calc geomech each geomech_period-th tstep
                if geomech_period == -1:
                    compute_displs = (ti == n_time_steps - 1)  # calc geomech only on last tstep

            if compute_displs and compute_stress_flag:
                compute_stress = True
            else:
                compute_stress = False

            if geomech_mode != 'none':
                arr_geomech = calc_geomech(m, geomech_mode, compute_stress=compute_stress, compute_displs=compute_displs)
        m.physics.engine.report()

        # append array xarray dataset on each timestep
        #write_grdecl_t = (ti % n_time_steps == 0) or (ti == n_time_steps - 1)
        write_grdecl_t = compute_displs and write_grdecl
        m.save_cubes(dir_out, case_name, ti=ti + 1, arrays=arr_geomech, write_grdecl=write_grdecl_t)
        if geomech_mode == 'surface':
               
            print(f"Debug: Processing geomech_mode 'surface' at time step {ti}")
            

            # Get the output properties
            output_data = m.output_properties()
            tot_props = m.physics.vars #+ m.physics.property_operators[0].props_name
            
            print(f"Debug: Total properties to process: {len(tot_props)}")
            print(f"Debug: Output data type: {type(output_data)}")
            print(f"Debug: Output data length: {len(output_data)}")

            # Assuming output_data is a tuple with two elements:
            # First element might be timestep information, second element contains property data
            #if True: #len(output_data) == 2 and isinstance(output_data[1], dict):
            #    property_data = output_data[1]
            #else:
               # print("Error: Unexpected structure of output_data")
                # Add appropriate error handling here
            property_data = output_data[1]
            # Plot each property
            for prop in tot_props:
                if prop in property_data:
                    data = property_data[prop]
                    
                    print(f"Debug: Processing property '{prop}'")
                    print(f"  Shape: {data.shape}")
                    print(f"  Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")
                    


            # Additional debugging information
            print("\nDebug: m.physics information:")
            print(f"  vars: {m.physics.vars}")
            print(f"  property_operators: {m.physics.property_operators}")
            
            print("\nDebug: Output data information:")
            for prop in tot_props:
                if prop in property_data:
                    print(f"  {prop}: shape {property_data[prop].shape}, dtype {property_data[prop].dtype}")
                else:
                    print(f"  {prop}: Not available in output data")

            # Create output_dict
            output_dict = {}
            for prop in tot_props:
                if prop in property_data:
                    if property_data[prop].ndim > 1:
                        output_dict[prop] = property_data[prop][0, :]  # Store the first time step if 2D
                    else:
                        output_dict[prop] = property_data[prop]
                else:
                    print(f"Warning: Could not add property '{prop}' to output_dict")

            print(f"Debug: Finished processing {len(output_dict)} properties")
            print(f"Debug: Keys in output_dict: {list(output_dict.keys())}")

            output_dict.update(arr_geomech)
            if ti in ts_xarray_ind:
                m.save_xarray(dir_out, case_name, ti=ti_x + 1, ts=ts_xarray, geomech_mode=geomech_mode, arrays=output_dict)
                ti_x += 1

    m.print_timers()
    m.print_stat()

    if geomech_mode == 'surface':
        m.save_xarray(dir_out, case_name, ti=ti + 1, ts=ts_xarray, geomech_mode=geomech_mode, arrays=output_dict, write_xarray=True)
        
        #print('ts_xarray_ind', ts_xarray_ind, 'ts_xarray', ts_xarray)
    
    if monitoring_points:
        observations = {}
        ds = xr.open_dataset(os.path.join(dir_out, f'{case_name}.nc'))
        
        for var_name, points in monitoring_points.items():
            if var_name in ds.data_vars:
                var_data = ds[var_name]
                obs_dict = {}
                obs_list = []
                for i, point in enumerate(points):
                    # Create a dictionary of valid dimension selections
                    selection = {}
                    for dim in ['time', 'origin', 'X', 'Y', 'Z']:
                        if dim in var_data.dims and dim in point:
                            selection[dim] = point[dim]
                    
                    # Use only the valid dimensions for selection
                    selected_data = var_data.isel(**selection)
                    
                    # Store the data in the dictionary
                    point_key = f"point_{i}"
                    obs_dict[point_key] = {
                        'data': selected_data.values,
                        'coords': {dim: point.get(dim, 'all') for dim in ['time', 'origin', 'X', 'Y', 'Z']}
                    }
                    
                    # Append to the list for numpy storage
                    obs_list.append(selected_data.values)
                
                # Store both dictionary and list of numpy arrays
                observations[var_name] = {
                    'dict': obs_dict,
                    'numpy': obs_list  # Store as a list of numpy arrays
                }
        
        # Save observations as pkl file if requested
        if save_observations:
            obs_file = os.path.join(dir_out, f'{case_name}_observations.pkl')
            with open(obs_file, 'wb') as f:
                pickle.dump(observations, f)
                
    return m, observations if monitoring_points else m

#function to read permeability from xarray
#%%
def read_var_from_xarray(nc_path: str, realization: int, variable_name: str):
    ds = xr.open_dataset(nc_path)
    var = ds[variable_name].values[realization, :, :, :]
    return ds, var


def convert_mt_per_year_to_kmol_per_day(mt_per_year):
    """
    Convert Mton/year of CO2 to kmol/day.
    
    :param mt_per_year: Mass of CO2 in Mton/year
    :return: Equivalent value in kmol/day
    """
    molar_density_co2 = 0.044  # CO2 molar density: 44 g/mol = 44 kg/kmol = 0.044 t/kmol
    days_in_year = 365.25  # Using 365.25 for precision
    mt_per_year_to_kmol_per_day = 1000000 / (days_in_year * molar_density_co2)  # Conversion factor

    kmol_per_day = mt_per_year * mt_per_year_to_kmol_per_day
    return kmol_per_day



def main():
    """
    Main function to run the DARTS simulation.
    """
    np.random.seed(0)
    physics_case = 'ccus'
    case = '64x64x1'
    dt = 30
    n_time_steps = 12 * 1  # 1 year

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
        rate_inj_ccus_array_multipliers = [1] + [np.random.uniform(0, 3) for _ in range(9)]
        
        output_dir = f'runs/{physics_case}_{case}/'
        os.makedirs(output_dir, exist_ok=True)
        case_name = f'DARTS_simulation_realization_{realization}'
        
        log_perm = ds['log_permeability'].values[realization, :, :, :]
        perm = np.exp(log_perm).flatten()
        
        m, observations = run_model(
            case_name=case_name,
            case=case,
            physics_case=physics_case,
            permx=perm,
            dir_out=output_dir,
            update_rates=True,
            rate_inj_ccus=rate_inj_ccus,
            rate_inj_ccus_array_multipliers=rate_inj_ccus_array_multipliers,
            bhp_inj_ccus=bhp_inj_ccus,
            dt=dt,
            n_time_steps=n_time_steps,
            monitoring_points=monitoring_points,
            save_observations=True
        )
        
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

if __name__ == '__main__':
    main()