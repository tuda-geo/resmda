#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import resmda
from runnable_darts_geomech import run_model, convert_mt_per_year_to_kmol_per_day
import os

# For reproducibility, we instantiate a random number generator with a fixed seed.
rng = np.random.default_rng(1848)

def flatten_all_observations(observations):
    all_data = []
    for obs_type in observations.values():
        for point_data in obs_type['dict'].values():
            all_data.append(point_data['data'].flatten())
    return np.concatenate(all_data)

print("####################################")
print("# Loading permeability ensemble")
print("####################################")

# Load permeability ensemble
perm_ensemble = xr.open_dataset('/samoa/data/smrserraoseabr/GenerateModels/geological_datasets/FINAL_realizations_with_POR_PERM_WELL_CONSTRAINED_64x64.nc') 
ne = 3  # number of prior models

perm_true_dataset = perm_ensemble.isel(Realization=0)
perm_prior_dataset = perm_ensemble.isel(Realization=slice(1, ne+1))

print(f"Loaded {ne} prior models")
print(f"Shape of true permeability: {perm_true_dataset['log_permeability'].shape}")
print(f"Shape of prior permeabilities: {perm_prior_dataset['log_permeability'].shape}")

# Assign bounds for the permeability
perm_min, perm_max = 0.1, 10

perm_true = perm_true_dataset['log_permeability'].values
perm_prior = perm_prior_dataset['log_permeability'].values

print("####################################")
print("# Setting up DARTS simulation parameters")
print("####################################")

# DARTS simulation parameters
physics_case = 'ccus'
case = '64x64x1'
geomech_mode = 'surface'
dt = 30
n_time_steps_ = 12 * 1  # 1 year

rate_inj_ccus = convert_mt_per_year_to_kmol_per_day(0.5)
bhp_inj_ccus = 113.08 * 2
rate_inj_ccus_array_multipliers = np.ones(n_time_steps_)

print(f"Physics case: {physics_case}")
print(f"Case: {case}")
print(f"Geomech mode: {geomech_mode}")
print(f"Time step: {dt} days")
print(f"Number of time steps: {n_time_steps_}")
print(f"Injection rate: {rate_inj_ccus} kmol/day")
print(f"BHP injection: {bhp_inj_ccus} bar")

monitoring_points = {
    'U_z': [
        {'time': -1, 'X': slice(None), 'Y': 0, 'Z': 0},
        {'time': slice(None), 'X': 0, 'Y': 0, 'Z': 0}
    ],
    'BHP': [
        {'time': slice(None)},
    ]
}

def sim(x):
    """Custom function to run DARTS simulation for given permeabilities."""
    ensemble_observations = []
    for i, perm in enumerate(x):
        print(f"####################################")
        print(f"# Running simulation {i+1}/{len(x)}")
        print(f"####################################")
        output_dir = f'runs/{physics_case}_{case}/'
        os.makedirs(output_dir, exist_ok=True)
        case_name = f'DARTS_simulation_realization_{i}'   
        perm = np.exp(perm)
        m, observations = run_model(
            case=case,
            case_name=case_name,
            physics_case=physics_case,
            geomech_mode=geomech_mode,
            geomech_period=1,
            permx=perm,
            dir_out=output_dir,
            update_rates=True,
            rate_inj_ccus=rate_inj_ccus,
            rate_inj_ccus_array_multipliers=rate_inj_ccus_array_multipliers,
            bhp_inj_ccus=bhp_inj_ccus,
            dt=dt,
            n_time_steps_=n_time_steps_,
            monitoring_points=monitoring_points,
            save_observations=True
        )
        print(f'Simulation {i} complete')
        observations_array = flatten_all_observations(observations)
        ensemble_observations.append(observations_array)
    return np.array(ensemble_observations)

print("####################################")
print("# Running prior models and reference case")
print("####################################")

# Run the prior models and the reference case
data_prior = sim(perm_prior)
data_true = sim(perm_true[np.newaxis, ...])  # Add a new axis to match dimensions

print(f"Shape of prior data: {data_prior.shape}")
print(f"Shape of true data: {data_true.shape}")

# Assumed standard deviation of our data
dstd = 0.000005

# Generate synthetic observations
data_obs = rng.normal(data_true, dstd)
data_obs[0, :3] = data_true[0, :3]  # Keep the first 3 observations exact

print("####################################")
print("# Setting up ES-MDA parameters")
print("####################################")

# ES-MDA parameters
def restrict_permeability(x):
    """Restrict possible permeabilities."""
    np.clip(x, perm_min, perm_max, out=x)

inp = {
    'model_prior': perm_prior,
    'forward': sim,
    'data_obs': data_obs,
    'sigma': dstd,
    'alphas': 2,
    'data_prior': data_prior,
    'callback_post': restrict_permeability,
    'random': rng,
}

print("ES-MDA input parameters:")
for key, value in inp.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: array of shape {value.shape}")
    else:
        print(f"{key}: {value}")

print("####################################")
print("# Running ES-MDA without localization")
print("####################################")

# Run ES-MDA without localization
nl_perm_post, nl_data_post = resmda.esmda(**inp)

print("Shape of perm_prior:", perm_prior.shape)
print("Shape of perm_true:", perm_true.shape)
print("Shape of nl_perm_post:", nl_perm_post.shape)
print("Shape of data_prior:", data_prior.shape)
print("Shape of data_obs:", data_obs.shape)
print("Shape of nl_data_post:", nl_data_post.shape)
#%%
# Run ES-MDA with localization (assuming you have a localization matrix)
# Uncomment and modify the following lines if you want to use localization
# loc_mat = ... # Define your localization matrix here
# wl_perm_post, wl_data_post = resmda.esmda(**inp, localization_matrix=loc_mat)

# Plotting functions (modify as needed for your specific output)
def plot_permeabilities(perm_prior, perm_true, nl_perm_post, wl_perm_post=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    
    axs[0, 0].imshow(perm_prior.mean(axis=(0, 1)))
    axs[0, 0].set_title('Prior Mean')
    
    axs[0, 1].imshow(perm_true.squeeze())
    axs[0, 1].set_title('True')
    
    axs[1, 0].imshow(nl_perm_post.mean(axis=(0, 1)))
    axs[1, 0].set_title('Posterior Mean (No Localization)')
    
    if wl_perm_post is not None:
        axs[1, 1].imshow(wl_perm_post.mean(axis=(0, 1)))
        axs[1, 1].set_title('Posterior Mean (With Localization)')
    
    for ax in axs.flat:
        ax.label_outer()
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('permeability_comparison.png')
    plt.show()


# Also update the plot_data function to handle potential shape mismatches and normalize data
def plot_data(data_prior, data_obs, nl_data_post, wl_data_post=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Normalize data
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)
    
    norm_data_prior = normalize(data_prior.reshape(data_prior.shape[0], -1))
    norm_nl_data_post = normalize(nl_data_post.reshape(nl_data_post.shape[0], -1))
    norm_data_obs = normalize(data_obs.squeeze())
    
    axs[0].plot(norm_data_prior.T, color='gray', alpha=0.5)
    axs[0].plot(norm_nl_data_post.T, color='blue', alpha=0.5)
    axs[0].plot(norm_data_obs, 'ro')
    axs[0].set_title('Normalized Data (No Localization)')
    
    if wl_data_post is not None:
        norm_wl_data_post = normalize(wl_data_post.reshape(wl_data_post.shape[0], -1))
        axs[1].plot(norm_data_prior.T, color='gray', alpha=0.5)
        axs[1].plot(norm_wl_data_post.T, color='blue', alpha=0.5)
        axs[1].plot(norm_data_obs, 'ro')
        axs[1].set_title('Normalized Data (With Localization)')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('normalized_data_comparison.png')
    plt.show()

print("####################################")
print("# Plotting results")
print("####################################")

# Plot results
plot_permeabilities(perm_prior, perm_true, nl_perm_post)
plot_data(data_prior, data_obs, nl_data_post)

# Print summary statistics
print("####################################")
print("# Summary statistics")
print("####################################")
print("Prior mean:", perm_prior.mean())
print("Prior std:", perm_prior.std())
print("Posterior mean:", nl_perm_post.mean())
print("Posterior std:", nl_perm_post.std())

resmda.Report()