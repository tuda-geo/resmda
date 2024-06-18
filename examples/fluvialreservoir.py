r"""
2D Fluvial Reservoir ESMDA example
==================================

In contrast to the basic reservoir example
:ref:`sphx_glr_gallery_basicreservoir.py`, where a single facies was used, this
example uses fluvial models. It also makes use of localization as explained in
:ref:`sphx_glr_gallery_localization.py`.

The fluvial models were generated with ``FLUVSIM`` through ``geomodpy``, for
more information see towards the end of the example where the code is shown to
reproduce the facies.


"""
import os
import json

# To retrieve the data, you need to have pooch installed:
# ``pip install pooch`` or ``conda install -c conda-forge pooch``
import pooch
import numpy as np
import matplotlib.pyplot as plt

import resmda

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(1848)

# Adjust this path to a folder of your choice.
data_path = os.path.join('..', 'download', '')

# sphinx_gallery_thumbnail_number = 3

###############################################################################
# Load and plot the facies
# ------------------------

fname = 'facies.npy'
pooch.retrieve(
    'https://raw.github.com/tuda-geo/data/2024-06-18/resmda/'+fname,
    '4bfe56c836bf17ca63453c37e5da91cb97bbef8cc6c08d605f70bd64fe7488b2',
    fname=fname,
    path=data_path,
)
facies = np.load(data_path + fname)
ne, nx, ny = facies.shape

# Define mean permeability per facies
perm_means = [0.1, 5.0, 3.0]

# Plot the facies
fig, axs = plt.subplots(2, 5, figsize=(8, 3), sharex=True, sharey=True,
                        constrained_layout=True)
axs = axs.ravel()
code_to_perm = [str(i)+" = "+str(p) for i, p in enumerate(perm_means)]
fig.suptitle(f'Facies: {code_to_perm}')
for i in range(ne):
    im = axs[i].imshow(
        facies[i, ...], cmap=plt.get_cmap("Accent", 3),
        clim=[-0.5, 2.5], origin='lower'
    )
fig.colorbar(im, ax=axs, ticks=[0, 1, 2], label='Facies code')


###############################################################################
# Assign random permeabilities to the facies
# ------------------------------------------

perm_min = 0.05
perm_max = 10.0

# Instantiate a random permeability instance
RP = resmda.RandomPermeability(
    nx, ny, perm_mean=None, perm_min=perm_min, perm_max=perm_max
)

# Fill the different facies with their permeabilities
permeabilities = np.empty(facies.shape)
for code, mean in enumerate(perm_means):
    mask = facies == code
    permeabilities[mask] = RP(ne, perm_mean=mean)[mask]

fig, axs = plt.subplots(2, 5, figsize=(8, 3), sharex=True, sharey=True,
                        constrained_layout=True)
axs = axs.ravel()
fig.suptitle('Permeabilities')
for i in range(ne):
    im = axs[i].imshow(permeabilities[i, ...], origin='lower')
fig.colorbar(im, ax=axs)


###############################################################################
# Model parameters
# ----------------

# We take the first model as "true/reference", and the other for ES-MDA.
perm_true = permeabilities[0, ...][None, ...]
perm_prior = permeabilities[1:, ...]

# Time steps
dt = np.zeros(10)+0.0001
time = np.r_[0, np.cumsum(dt)]
nt = time.size

# Assumed standard deviation of our data
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
loc_mat = resmda.build_localization_matrix(RP.cov, nd_positions, (nx, ny))


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
fig.show()

###############################################################################
# Reproduce the facies
# --------------------
#
# .. note::
#
#     The following cell is about how to reproduce the facies data loaded in
#     ``facies.npy``. This was created using ``geomodpy``.
#
#     ``geomodpy`` (Guillaume Rongier, 2023) is not open-source yet. The
#     functionality of geomodpy that we use here is a python wrapper for the
#     open-source Fortran library ``FLUVSIM``:
#
#         **Deutsch, C. V., and T. T. Tran**, 2002, FLUVSIM: a program for
#         object-based stochastic modeling of fluvial depositional systems:
#         Computers & Geosciences, 28(4), 525--535.
#
#         DOI: `10.1016/S0098-3004(01)00075-9
#         <https://doi.org/10.1016/S0098-3004(01)00075-9>`_.
#
#
# .. code-block:: python
#
#     # ==== Required imports ====
#     import json
#     import numpy as np
#
#     # FLUVSIM Version used: 2.900
#     from geomodpy.wrapper.fluvsim import FLUVSIM
#
#     # For reproducibility, we instantiate a random number generator with a
#     # fixed seed. For production, remove the seed!
#     rng = np.random.default_rng(1848)
#
#
#     # ==== Define the geological parameters ====
#
#     # Here we define the geological parameters by means of their normal
#     # distribution parameters
#
#     # Each tuple stands for (mean, std); lists contain several of them.
#     geol_distributions = {
#         'channel_orientation': (60, 20),
#         'channel_amplitude': [(100, 1), (250, 1), (400, 1)],
#         'channel_wavelength': [(1000, 5), (2000, 5), (3000, 5)],
#         'channel_thickness': [(4, 0.1), (8, 0.1), (11, 0.1)],
#         'channel_thickness_undulation': (1, 0.02),
#         'channel_thickness_undulation_wavelength': [
#             (250, 1), (400, 1), (450, 1)
#         ],
#         'channel_width_thickness_ratio': [(40, 0.5), (50, 0.5), (60, 0.5)],
#         'channel_width_undulation': (1, 0.02),
#         'channel_width_undulation_wavelength': (250, 1),
#         'channel_prop': (0.4, 0.005),
#     }
#
#
#     def generate_geol_params(geol_dists):
#         """Generate geological parameters from normal distributions.
#
#         Expects for each parameter a tuple of two values, or a list
#         containing tuples of two values each.
#         """
#         geol_params = {}
#         for param, dist in geol_dists.items():
#             if isinstance(dist, list):
#                 geol_params[param] = tuple(
#                     rng.normal(mean, std) for mean, std in dist
#                 )
#             else:
#                 geol_params[param] = rng.normal(*dist)
#         return geol_params
#
#
#     # ==== Create the facies ====
#
#     # Number of sets and realizations
#     nsets = 2
#     nreal = 5
#
#     # Model dimension
#     nx, ny, nz = 64, 64, 1
#
#     # Pre-allocate containers to store all realizations and their
#     # corresponding parameters
#     all_params = {}
#     facies = np.zeros((nsets * nreal, nz, nx, ny), dtype='i4')
#
#     for i in range(nsets):  # We create two sets
#         print(f"Generating realization {i+1} of {nsets}")
#
#         params = generate_geol_params(geol_distributions)
#         all_params[f'set-{i}'] = params
#
#         fluvsim = FLUVSIM(
#             shape=(nx, ny, nz),
#             spacing=(50, 50, 1),
#             origin=(25, 25, 0.5),
#             n_realizations=nreal,
#             **params
#         )
#
#         realizations = fluvsim.run().data_vars['facies code'].values
#         facies[i*nreal:(i+1)*nreal, ...] = realizations.astype('i4')
#
#
#     # ==== Save the output ====
#
#     # Save the input parameters to FLUVSIM as a json.
#     with open('facies.json', "w") as f:
#         json.dump(all_params, f, indent=2)
#     # Save the facies values as a compressed npy-file.
#     np.save('facies', facies.squeeze(), allow_pickle=False)


###############################################################################
# Input parameters to ``FLUVSIM``
# -------------------------------
# For reproducibility purposes we print here the used input values to FLUVSIM.
# These are, just as the data themselves, online at
# https://github.com/tuda-geo/data/resmda.

fname = 'facies.json'
pooch.retrieve(
    'https://raw.github.com/tuda-geo/data/2024-06-18/resmda/'+fname,
    'db2cb8a620775c68374c24a4fa811f6350381c7fc98a823b9571136d307540b4',
    fname=fname,
    path=data_path,
)
with open(data_path + fname, 'r') as f:
    print(json.dumps(json.load(f), indent=2))

###############################################################################
resmda.Report()
