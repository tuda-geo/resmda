#%%
r"""
Geothermal Case Study
=====================

This example demonstrates the application of the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) using the resmda library to predict temperature at a production well in a geothermal reservoir.
The script integrates with the DARTS simulator to model the impact of permeability variations on temperature over a 30-year period.
The example uses a channelized permeability field, which provides an interesting case study of ES-MDA's behavior with non-Gaussian geological features. 
Since ES-MDA operates under Gaussian assumptions, it tends to create "blurry" updates to the permeability field rather than maintaining sharp channel boundaries. 
This limitation becomes particularly visible when the algorithm identifies the need for connectivity between wells - instead of creating or modifying channels, it increases permeability in a more diffuse manner. This behavior highlights both the power of ES-MDA in matching production data and its limitations in preserving complex geological features.

**TODO Gabriel:**
Some introduction about what this example is and what it shows. DONE


.. tip::

    This example shows how you can use ``resmda`` with any external modelling
    code, here with ``darts``.

    **Thanks to Mona Devos, who provided the initial version of this example!**


Pre-computed data
-----------------

Darts computation
'''''''''''''''''

The `Darts <https://pypi.org/project/open-darts/>`_-computation in this example
takes too long for it to be run automatically on GitHub with every commit for
the documentation (CI). Therefore, the results are precomputed, and we use that
output here to show the results. Change the flag ``compute_darts`` from
``False`` to ``True`` to actually compute the results. The pre-computed results
were generated with the following versions (taking roughly 4h on a single
thread):

- ``open-darts==1.1.4``
- ``Python==3.10.15``
- ``numpy==2.0.2``
- ``scipy==1.14.1``


Fluvial models
''''''''''''''

The fluvial models were generated with ``FLUVSIM`` through ``geomodpy``, for
more information see towards the end of the example where the code is shown to
reproduce the facies.


.. important::

    To retrieve the pre-computed data, you need to have ``pooch`` installed:

    .. code-block:: bash

        pip install pooch

    or

    .. code-block:: bash

        conda install -c conda-forge pooch


"""
import pooch
import numpy as np
import matplotlib.pyplot as plt

import resmda

compute_darts = False
if compute_darts:
    from darts.engines import redirect_darts_output
    from darts.models.darts_model import DartsModel
    from darts.physics.geothermal.physics import Geothermal
    from darts.reservoirs.struct_reservoir import StructReservoir
    from darts.physics.geothermal.property_container import PropertyContainer

    redirect_darts_output("run_geothermal.log")

    # For reproducibility, we instantiate a random number generator with a
    # fixed seed. For production, remove the seed!
    rng = np.random.default_rng(42)

# sphinx_gallery_thumbnail_number = 4

###############################################################################
# Load pre-computed data
# ----------------------

folder = "data"
ffacies = "facies_geothermal.npy"
fdarts = "darts_output_geothermal.npz"

# Load Facies: Not needed if you compute the facies yourself,
#              as described at the end of the notebook.
fpfacies = pooch.retrieve(
    "https://raw.github.com/tuda-geo/data/2024-11-30/resmda/"+ffacies,
    "9b18f1c80aea93d7973effafde001aa7e72a21ac91edf08e3899d5486998ad2e",
    fname=ffacies,
    path=folder,
)
facies = np.load(fpfacies)

# Load pre-computed Darts result; only needed if `compute_darts` is `False`.
if not compute_darts:
    fpdarts = pooch.retrieve(
        "https://raw.github.com/tuda-geo/data/2024-11-30/resmda/"+fdarts,
        "5622729cd5dc7214de8a199512ace39bda48bff113b2eddb0c48593a57c020d1",
        fname=fdarts,
        path=folder,
    )
    pc_darts = np.load(fpdarts)


###############################################################################
# Convert facies to permeability
# ------------------------------
#
# Here we define some model parameters. Important: These have obviously to
# agree with the facies you computed!
#
# **TODO Gabriel:**
# nz is set to 3 - is it one layer above and below our actual layer?
'''REPLY:
The model has  actual 3 layers, all of them are 30m thick and all are perforated. It happens in the
code bellow:
# Add well
self.reservoir.add_well("PRD")
for k in range(1, self.reservoir.nz):
    self.reservoir.add_perforation(
        "PRD",
        cell_index=(self.iw[1], self.jw[1], k + 1),
        well_radius=0.16,
        multi_segment=True,
    )

'''
# Model parameters
nx, ny = 60, 60
nz = 3
dx, dy, dz = 30, 30, 30
ne = 100
years = np.arange(31)  # Time: we are modelling 30 years:

# Well locations
iw = [30, 30]
jw = [14, 46]

# Minimum and maximum values for permeability
perm_min = 100.
perm_max = 200.

# Get permeability fields by populating the facies
perm = np.zeros(facies.shape)
perm[facies == 0] = perm_min  # outside channels minimum
perm[facies > 0] = perm_max   # inside channels maximum
perm = np.stack([perm]*nz, axis=-1)  # 3x the same

# Assign true permeability (first) and prior permeability
perm_true = perm[:1, :, :, :]
perm_prior = perm[1:, :, :, :]


###############################################################################
# Prior permeabilities
# --------------------
# Plot the first 12 prior permeability models; grey shows the channels with
# higher permeability.
#
# **TODO Gabriel:**
# It probably would be good to plot the wells in the permeability plots, right?
# In the darts model, are those cell indices for the wells? Can you add some
# symbols here for the wells?
'''REPLY:
Yes, I will plot it. As in the code from previous reply you can check that
it is index - we have well perforated from the first layer to the third one.

However, I will plot after the declare the wells in the darts model.
'''

fopts = {"sharex": True, "sharey": True, "constrained_layout": True, "figsize": (12, 8), "dpi": 100}
popts = {"vmin": perm_min, "vmax": perm_max, "origin": "lower"}

fig, axs = plt.subplots(3, 4, figsize=fopts['figsize'], dpi=fopts['dpi'], 
                        sharex=True, sharey=True)
axs = axs.ravel()

fig.suptitle("Prior Permeabilities", fontsize=14, y=0.95)

for i in range(min(ne, 12)):
    im = axs[i].imshow(perm_prior[i, :, :, 0], **popts)
    
    if i == 0:
        axs[i].plot(jw[0], iw[0], '^', color='blue', markersize=10, 
                    markeredgecolor='white', label='Injection well')
        axs[i].plot(jw[1], iw[1], '^', color='red', markersize=10, 
                    markeredgecolor='white', label='Production well')
    else:
        axs[i].plot(jw[0], iw[0], '^', color='blue', markersize=10, 
                    markeredgecolor='white')
        axs[i].plot(jw[1], iw[1], '^', color='red', markersize=10, 
                    markeredgecolor='white')
    
    axs[i].set_title(f'Realization {i+1}', fontsize=10)
    
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].grid(False)
    
    if i % 4 == 0:
        axs[i].set_ylabel('Y Grid Cell', fontsize=10)
    
    if i >= 8:
        axs[i].set_xlabel('X Grid Cell', fontsize=10)

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                   label='Log Permeability (mD)')
cbar.ax.tick_params(labelsize=9)

legend_ax = fig.add_axes([0.02, 0.95, 0.2, 0.05])
legend_ax.axis('off')

handles, labels = axs[0].get_legend_handles_labels()
legend = legend_ax.legend(handles, labels, 
                         loc='center',
                         ncol=2,
                         frameon=True,
                         fontsize=10,
                         borderaxespad=0)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)

plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.05)

#%%

###############################################################################
# Darts-related functionalities
# -----------------------------
#
# Custom Darts Model class
# ''''''''''''''''''''''''
#
# In the custom ``Model`` class for Darts we define some fixed parameters
# specific to this example.



if compute_darts:

    class Model(DartsModel):
        """Custom DartsModel Class."""

        def __init__(self, perm, n_points=128, dx=dx, dy=dy, dz=dz, iw=iw, jw=jw):
            """Initialize a new DartsModel instance."""
            super().__init__()

            self.timer.node["initialization"].start()

            # Parameters for the reservoir
            nx, ny, nz = perm.shape
            nb = perm.size

            self.dx = dx
            self.dy = dy
            dz = np.ones(nb) * dz

            perm = perm.flatten("F")
            poro = np.ones(nb) * 0.2

            # Discretize structured reservoir
            self.reservoir = StructReservoir(
                timer=self.timer,
                nx=nx,
                ny=ny,
                nz=nz,
                dx=self.dx,
                dy=self.dy,
                dz=dz,
                permx=perm,
                permy=perm,
                permz=0.1*perm,
                poro=poro,
                depth=2000,
                hcap=2200,
                rcond=500,
            )

            # Add open boundaries
            self.reservoir.boundary_volumes["yz_minus"] = 1e8
            self.reservoir.boundary_volumes["yz_plus"] = 1e8
            self.reservoir.boundary_volumes["xz_minus"] = 1e8
            self.reservoir.boundary_volumes["xz_plus"] = 1e8

            # Add well locations
            self.iw = iw
            self.jw = jw

            # Create pre-defined physics for geothermal
            property_container = PropertyContainer()
            self.physics = Geothermal(
                timer=self.timer,
                n_points=n_points,
                min_p=1,
                max_p=351,
                min_e=1000,
                max_e=10000,
                cache=False,
            )
            self.physics.add_property_region(property_container)
            self.physics.init_physics()

            # Timestep parameters
            self.params.first_ts = 1e-3
            self.params.mult_ts = 2
            self.params.max_ts = 365

            # Nonlinear and linear solver tolerance
            self.params.tolerance_newton = 1e-2

            self.timer.node["initialization"].stop()

        def set_wells(self):
            """Set well parameters."""
            self.reservoir.add_well("INJ")
            for k in range(1, self.reservoir.nz):
                self.reservoir.add_perforation(
                    "INJ",
                    cell_index=(self.iw[0], self.jw[0], k + 1),
                    well_radius=0.16,
                    multi_segment=True,
                )

            # Add well
            self.reservoir.add_well("PRD")
            for k in range(1, self.reservoir.nz):
                self.reservoir.add_perforation(
                    "PRD",
                    cell_index=(self.iw[1], self.jw[1], k + 1),
                    well_radius=0.16,
                    multi_segment=True,
                )

        def set_initial_conditions(self):
            """Initialization with constant pressure and temperature."""
            self.physics.set_uniform_initial_conditions(
                self.reservoir.mesh,
                uniform_pressure=200,
                uniform_temperature=350,
            )

        def set_boundary_conditions(self):
            """Activate wells with rate control for injector and producer."""
            for i, w in enumerate(self.reservoir.wells):
                if "INJ" in w.name:
                    w.control = self.physics.new_rate_water_inj(4000, 300)
                else:
                    w.control = self.physics.new_rate_water_prod(4000)


###############################################################################
# DARTS simulation function
# '''''''''''''''''''''''''
#
# A ``simulation`` function (a wrapper for your model) is what you have to
# build to use ``resmda``. This example for Darts can serve as a blueprint for
# your own project. The function has to
#
# - take an ndarray of the shape of the prior models ``(ne, ...)``, and
# - return an ndarray of the shape of the prior data ``(ne, nd)``;
#
# ``ne`` is the number of ensembles, ``nd`` the number of data.
#
# In this case using Darts, the function ``temperature_at_production_well``
# takes permeability fields, and returns temperature (K) at production well.

if compute_darts:

    def temperature_at_production_well(permeabilities, years=years):
        """Darts function prediction temperature at production well."""

        # Pre-allocate output
        temperature = np.zeros((permeabilities.shape[0], years.size))

        # Loop over permeability fields
        for i, perm in enumerate(permeabilities):

            # Initialize a Darts model for this permeability field
            m = Model(perm=perm)
            m.init()
            m.run(1e-3)

            # Run and store every year;
            # (in future Darts versions the loop won't be needed).
            for y in years[1:]:
                m.run(365, restart_dt=365)

            # Store temperature
            # (Sometimes the first time step is cut and then repeated in the
            #  time_data; if you can change the time_data to not report cut
            #  timesteps, then this may not be necessary.)
            temperature[i, :] = np.array(
                m.physics.engine.time_data["PRD : temperature (K)"]
            )[-years.size:]

        return temperature


###############################################################################
# Simulate prior data
# -------------------
#
# Here we simulate *true* and prior data, and create *observed* data from
# adding random noise to the true data.

if compute_darts:
    data_true = temperature_at_production_well(perm_true)
    data_prior = temperature_at_production_well(perm_prior)
    # Add noise to the true data and create observed data
    dstd = 0.2
    data_obs = rng.normal(data_true, dstd)
else:
    data_true = pc_darts["data_true"].astype("d")
    data_prior = pc_darts["data_prior"].astype("d")
    data_obs = pc_darts["data_obs"].astype("d")


###############################################################################
# Plot prior data
# '''''''''''''''
#
# **TODO Gabriel:**
# How to interpret that there are, sort of, two different paths with slightly
# different temperatures?
'''REPLY:
Replied bellow, this is realated to the connectivity between the injection and
production well.
'''

fig, ax = plt.subplots()
ax.scatter(years, data_obs, label="Observed", color="red", zorder=10)
ax.plot(years, data_prior.T, color="grey", alpha=0.4,
        label=["Prior"] + [None] * (ne - 1))
ax.legend()
ax.set_xlabel("Time (years)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature at Production Well")


###############################################################################
# Perform Data Assimilation (ES-MDA)
# ----------------------------------

# Define a function to restrict the permeability values
def restrict_permeability(x):
    """Restrict possible permeabilities."""
    np.clip(x, perm_min, perm_max, out=x)


# Run ES-MDA
if compute_darts:
    perm_post, data_post = resmda.esmda(
        model_prior=perm_prior,
        forward=temperature_at_production_well,
        data_obs=data_obs,
        sigma=dstd,
        alphas=4,
        data_prior=data_prior,
        callback_post=restrict_permeability,
        random=rng,
    )

    # Store result
    np.savez_compressed(
        file="darts_output_geothermal.npz",
        data_true=data_true.astype("f"),
        data_obs=data_obs.astype("f"),
        data_prior=data_prior.astype("f"),
        data_post=data_post.astype("f"),
        perm_post=perm_post.astype("f"),
        allow_pickle=False,
    )
else:
    data_post = pc_darts["data_post"].astype("d")
    perm_post = pc_darts["perm_post"].astype("d")


###############################################################################
# Plot posterior data
# '''''''''''''''''''
#
# **TODO Gabriel:**
# How to interpret that "the colder path" won?
'''REPLY:
We can interpret that models that connect the injection and production well with 
higher permeability are the ones which match data better.

Even when we originally didn't have channels between the injection and production
well, the model found that the best way to match data is to increase a lot the permeability, 
actually this is really nice. .
'''

fig, ax = plt.subplots()
ax.scatter(years, data_obs, label="Observed", color="red", zorder=10)
ax.plot(years, data_prior.T, color="grey", alpha=0.4,
        label=["Prior"] + [None] * (ne - 1))
ax.plot(years, data_post.T, color="blue",
        label=["Posterior"] + [None] * (ne - 1))
ax.legend()
ax.set_xlabel("Time (years)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature at Production Well")

###############################################################################
# Plot posterior permeabilities
# '''''''''''''''''''''''''''''
# Plot the first 12 posterior permeability models.
#
# **TODO Gabriel:**
# Is it insightful to show the posterior permeability?
'''REPLY:
Definitely, it is insightful to show the posterior permeability
because it shows that besides we match data we still have different realizations
this means we still have uncertainty in the model. Besides, it shows that 
the ensemble is not colapsed to a single model.
'''


fig, axs = plt.subplots(3, 4, **fopts)
axs = axs.ravel()
fig.suptitle("Posterior Permeabilities")
for i in range(min(ne, 12)):
    im = axs[i].imshow(perm_post[i, :, :, 0], **popts)
    axs[i].plot(jw[0], iw[0], '^', color='blue', markersize=10, markeredgecolor='white')
    axs[i].plot(jw[1], iw[1], '^', color='red', markersize=10, markeredgecolor='white')
fig.colorbar(im, ax=axs, label="Log Permeability (mD)")


###############################################################################
# Reproduce the facies
# --------------------
#
# .. note::
#
#     The following cell is about how to reproduce the facies data loaded in
#     ``facies_geothermal.nc``. This was created using ``geomodpy``.
#
#     ``geomodpy`` (Guillaume Rongier, 2023) is not open-source yet. The
#     functionality of geomodpy that we use here is a python wrapper for the
#     Fortran library ``FLUVSIM`` published in:
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
#     # FLUVSIM Version used: 2.900
#     from geomodpy.wrapper.fluvsim import FLUVSIM
#
#     # Dimensions
#     nx, ny = 60, 60
#     dx, dy = 30, 30
#     ne = 100
#
#     # Create a fluvsim instance with the geological parameters
#     fluvsim = FLUVSIM(
#         channel_orientation=(60., 90., 120.),
#         # Proportions for each facies
#         channel_prop=0.5,
#         crevasse_prop=0.0,
#         levee_prop=0.0,
#         # Parameters defining the grid
#         shape=(nx, ny),
#         spacing=(dx, dy),
#         origin=(0, 0),
#         # Number of realizations and random seed
#         n_realizations=ne+1,  # +1 for "true"
#         seed=42,
#     )
#
#     # Run fluvsim to create the facies
#     facies = fluvsim.run()
#     facies = facies.data_vars["facies code"].values.astype("i4")
#
#     # Save the facies values as a compressed npy-file.
#     np.save("facies_geothermal.npy", facies.squeeze(), allow_pickle=False)


###############################################################################
resmda.Report()
