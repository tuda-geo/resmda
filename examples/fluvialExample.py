#%%
r"""
2D Fluvial Reservoir ESMDA example
==========================

Ensemble Smoother Multiple Data Assimilation (ES-MDA) in Reservoir Simulation.

"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import resmda

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(1848)

# sphinx_gallery_thumbnail_number = 4

###############################################################################
# Model parameters
# ----------------
# Permeabilities
#read xarray from file
'''
xarray with 10000 realizations of facies for channelized reservoirs
all the models are 64x64x1 with 100x100x1 spacing (dx,dy,dz) which means they are pseudo 3D models

We will read the first 1001 realizations and generate the permeability maps for the first 1000 realizations and 
the first will be the reference case

The property of interest is called facies code and we will assign a fixed permeability to each facies code

Later, we can distribute the permeability across the reservoir using the facies code and the variance approach as in the basis case

'''
facies_ensemble = xr.open_dataset('/samoa/data/smrserraoseabr/GenerateModels/realizations.nc') 
ne =100 #number of prior models
reference_case = facies_ensemble.isel(Realization=0)
prior_ensemble = facies_ensemble.isel(Realization=slice(1,ne+1))

fac_array = reference_case['facies code'].values.astype(float)
fac_prior_array = prior_ensemble['facies code'].values.astype(float)
fac_prior_array = np.squeeze(fac_prior_array) #remove an extra dimension

#assign bounds for the permeability
perm_min = 0.1
perm_max =5
# Grid extension
nx = fac_array.shape[1]
ny = fac_array.shape[2]
nc = nx * ny
#assigning permeability for the facies code

perm_means = np.array([0.1, 5, 3.0])

def assign_permeability(facies_array: np.array, perm_means: np.array, nx: int, ny: int, rng: np.random.Generator) -> np.array:
    '''
    This function receives an array of any shape and converts the facies code to permeability using different
    means for each facies provided in the perm_means vector. The final permeability map is a combination of the permeabilities assigned to each facies.
    '''
    # Initialize the final permeability map with zeros, ensure it is float
    permeability_map = np.zeros_like(facies_array, dtype=float)
    
    # Iterate over each facies code and assign permeability
    for facies_code, mean in enumerate(perm_means):
        # Create a RandomPermeability instance for the current facies
        RP = resmda.RandomPermeability(nx, ny, mean, perm_min, perm_max, dtype='float64')

        
        # Generate permeability for the current facies
        facies_perm = RP(n=int(facies_array.shape[0]), random=rng)
        
        # Overlay the permeability of the current facies onto the final permeability map
        mask = facies_array == facies_code
        permeability_map[mask] = facies_perm[mask]
    
    return permeability_map

# Now generate the permeability maps for the reference case and the prior ensemble
perm_true = assign_permeability(fac_array, perm_means, nx, ny, rng)
perm_prior = assign_permeability(fac_prior_array, perm_means, nx, ny, rng)

#%%
#plot perm_true
plt.imshow(perm_true.T)
plt.colorbar(label='Log of Permeability (mD)')
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.title('True Permeability Map')
plt.savefig(f'True_Permeability_Map_{ne}_Ensembles.png')
plt.show()

#%%


# ESMDA parameters
ne = perm_prior.shape[0]                  # Number of ensembles
#  dt = np.logspace(-5, -3, 10)
dt = np.zeros(10)+0.0001  # Time steps (could be irregular, e.g., increasing!)
time = np.r_[0, np.cumsum(dt)]
nt = time.size
#%%
# Assumed sandard deviation of our data
dstd = 0.5

ox = (5, 15, 24)
oy = (5, 10, 24)

# Number of data points
nd = time.size * len(ox)

# Wells
wells = np.array([
    [5, 5, 180], [5, 12, 120],
    [15, 10, 180], [20, 5, 120],
    [24, 24, 180], [24, 17, 120]
])


# TODO: change scale in imshow to represent meters

# QC reference model and first two random models
fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].set_title('Model')
im = axs[0].imshow(perm_true.T, vmin=perm_min, vmax=perm_max)
axs[1].set_title('Random Model 1')
axs[1].imshow(perm_prior[0, ...].T, vmin=perm_min, vmax=perm_max)
axs[2].set_title('Random Model 2')
axs[2].imshow(perm_prior[1, ...].T, vmin=perm_min, vmax=perm_max)
fig.colorbar(im, ax=axs, orientation='horizontal',
             label='Log of Permeability (mD)')
for ax in axs:
    ax.set_xlabel('x-direction')
axs[0].set_ylabel('y-direction')
fig.savefig(f'QC_Models_{ne}_Ensembles.png')
fig.show()
#%%


###############################################################################
# Run the prior models and the reference case
# -------------------------------------------

# Instantiate reservoir simulator
RS = resmda.Simulator(nx, ny, wells=wells)


def sim(x):
    """Custom fct to use exp(x), and specific dt & location."""
    return RS(np.exp(x), dt=dt, data=(ox, oy)).reshape((x.shape[0], -1))


# Simulate data for the prior and true fields
data_prior = sim(perm_prior)
data_true = sim(perm_true)
data_obs = rng.normal(data_true, dstd)
data_obs[0, :3] = data_true[0, :3]


###############################################################################
# Localization Matrix
# -------------------

# Vector of nd length with the well x and y position for each nd data point
nd_positions = np.tile(np.array([ox, oy]), time.size).T

# Create matrix
RP = resmda.RandomPermeability(nx, ny, 1, perm_min, perm_max) #had to include it here just for the localization
loc_mat = resmda.build_localization_matrix(RP.cov, nd_positions, (nx, ny))

# QC localization matrix
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, constrained_layout=True)
ax.imshow(loc_mat.sum(axis=2).T)
ax.contour(loc_mat.sum(axis=2).T, levels=[2.0, ], colors='w')
for well in wells:
    ax.plot(well[0], well[1], ['C3v', 'C1^'][int(well[2] == 120)])
fig.savefig(f'Localization_Matrix_{ne}_Ensembles.png')

###############################################################################
# ES-MDA
# ------


def restrict_permeability(x):
    """Restrict possible permeabilities."""
    np.clip(x, perm_min, perm_max, out=x)


inp = {
    'model_prior': perm_prior,
    'forward': sim,
    'data_obs': data_obs,
    'sigma': dstd,
    'alphas': 4,
    'data_prior': data_prior,
    'callback_post': restrict_permeability,
    'random': rng,
}


###############################################################################
# Without localization
# ''''''''''''''''''''

nl_perm_post, nl_data_post = resmda.esmda(**inp)


###############################################################################
# With localization
# '''''''''''''''''

wl_perm_post, wl_data_post = resmda.esmda(**inp, localization_matrix=loc_mat)


###############################################################################
# Compare permeabilities
# ----------------------

# Plot posterior
fig, axs = plt.subplots(
    2, 2, figsize=(6, 6), sharex=True, sharey=True, constrained_layout=True)
axs[0, 0].set_title('Prior Mean')
axs[0, 0].imshow(perm_prior.mean(axis=0).T)
axs[0, 1].set_title('"Truth"')
axs[0, 1].imshow(perm_true.T)


axs[1, 0].set_title('Post Mean without localization')
axs[1, 0].imshow(nl_perm_post.mean(axis=0).T)
axs[1, 1].set_title('Post Mean with localization')
axs[1, 1].imshow(wl_perm_post.mean(axis=0).T)
axs[1, 1].contour(loc_mat.sum(axis=2).T, levels=[2.0, ], colors='w')

for ax in axs.ravel():
    for well in wells:
        ax.plot(well[0], well[1], ['C3v', 'C1^'][int(well[2] == 120)])
fig.savefig(f'Compare_Permeabilities_{ne}_Ensembles.png')

###############################################################################
# Compare data
# ------------

# QC data and priors
fig, axs = plt.subplots(
    2, 3, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle('Prior and posterior data')
for i, ax in enumerate(axs[0, :]):
    ax.plot(time, data_prior[..., i::3].T, color='.6', alpha=0.5)
    ax.plot(time, nl_data_post[..., i::3].T, color='C0', alpha=0.5)
    ax.plot(time, data_obs[0, i::3], 'C3o')
    ax.set_ylabel('Pressure')
for i, ax in enumerate(axs[1, :]):
    ax.plot(time, data_prior[..., i::3].T, color='.6', alpha=0.5)
    ax.plot(time, wl_data_post[..., i::3].T, color='C0', alpha=0.5)
    ax.plot(time, data_obs[0, i::3], 'C3o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure')
fig.savefig(f'Compare_Data_{ne}_Ensembles.png')
plt.show()

###############################################################################

# %%
#Plot posterior models individually

# Create separate figures for with and without localization
fig_nl, axs_nl = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
fig_nl.suptitle('Without Localization')
fig_wl, axs_wl = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
fig_wl.suptitle('With Localization')
model_indices = [0, 1, 2, 3]  # Indices of the models to display

for i, model_idx in enumerate(model_indices):
    # Without localization
    im_nl_prior = axs_nl[0, i].imshow(perm_prior[model_idx, ...].T)
    axs_nl[0, i].set_title(f'Prior Model {model_idx+1}')

    im_nl_post = axs_nl[1, i].imshow(nl_perm_post[model_idx, ...].T)
    axs_nl[1, i].set_title(f'Post Model {model_idx+1}')

    diff_nl = nl_perm_post[model_idx, ...] - perm_prior[model_idx, ...]
    im_nl_diff = axs_nl[2, i].imshow(diff_nl.T)
    axs_nl[2, i].set_title(f'Difference Model {model_idx+1}')

    # With localization
    im_wl_prior = axs_wl[0, i].imshow(perm_prior[model_idx, ...].T)
    axs_wl[0, i].set_title(f'Prior Model {model_idx+1}')

    im_wl_post = axs_wl[1, i].imshow(wl_perm_post[model_idx, ...].T)
    axs_wl[1, i].set_title(f'Post Model {model_idx+1}')

    diff_wl = wl_perm_post[model_idx, ...] - perm_prior[model_idx, ...]
    im_wl_diff = axs_wl[2, i].imshow(diff_wl.T)
    axs_wl[2, i].set_title(f'Difference Model {model_idx+1}')

# Add colorbars to the last column of each row for both figures
for row in range(3):
    fig_nl.colorbar(axs_nl[row, -1].get_images()[0], ax=axs_nl[row, :], location='right', aspect=20)
    fig_wl.colorbar(axs_wl[row, -1].get_images()[0], ax=axs_wl[row, :], location='right', aspect=20)
fig_nl.savefig(f'Posterior_Models_Without_Localization_{ne}_Ensembles.png')
fig_wl.savefig(f'Posterior_Models_With_Localization_{ne}_Ensembles.png')
# %%
resmda.Report()