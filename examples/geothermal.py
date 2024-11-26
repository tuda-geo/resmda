r"""
Geothermal Case Study
=====================

**This example was contributed by Mona Devos.**

**Example how to use `resmda` with an external modeller, here with `darts`.**

.. note::

    To retrieve the data, you need to have ``pooch`` installed:

    .. code-block:: bash

        pip install pooch

    or

    .. code-block:: bash

        conda install -c conda-forge pooch


"""
import os

import h5py
import pooch
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import resmda

from darts.engines import redirect_darts_output
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.physics.geothermal.property_container import PropertyContainer

redirect_darts_output("run_geothermal.log")

# Adjust this path to a folder of your choice.
data_path = os.path.join("..", "download", "")

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(42)

###############################################################################
# Load facies models
# ------------------

fname = "facies_geothermal.nc"
pooch.retrieve(
    "https://github.com/tuda-geo/data/raw/refs/heads/main/resmda/"+fname,
    "814956d887f6e688eaae9a2c144aa0e6120361a8fb6ebe00f17d27038d877df3",
    fname=fname,
    path=data_path,
)
facies = xr.load_dataset(data_path + fname)


###############################################################################
# Just for testing; load output
# -----------------------------
#
# fname = 'output.h5'
# with h5py.File(fname, 'r') as h5file:
#     time = h5file['time'][()]
#     data_true = h5file['data_true'][()]
#     data_obs = h5file['data_obs'][()]
#     data_prior = h5file['data_prior'][()]
#     data_post = h5file['data_post'][()]
#     perm_true = h5file['perm_true'][()]
#     perm_prior = h5file['perm_prior'][()]
#     perm_post = h5file['perm_post'][()]


###############################################################################
# Create a custom Darts Model class
# ---------------------------------

class Model(DartsModel):
    """Custom DartsModel Class."""

    def __init__(self, perm, n_points=128):
        """Initialize a new DartsModel instance."""
        super().__init__()

        self.timer.node["initialization"].start()

        # Parameters for the reservoir
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz

        self.dx = 30
        self.dy = 30
        dz = np.ones(nb) * 30

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
        self.iw = [30, 30]
        self.jw = [14, 46]

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
# Convert facies to permeability
# ------------------------------

# Define the number of facies models
n = facies["facies code"].shape[0]

# Create a 3D array for facies and permeability
facies_array = np.zeros((n, 60, 60))
permeabilities = np.zeros((n, 60, 60))
permeabilities_3D = np.zeros((n, 60, 60, 3))

# Assign facies and permeability values
for i in range(n):
    facies_array[i] = facies["facies code"][i, 0].values

    permeabilities[i][facies_array[i] == 0] = 100
    permeabilities[i][facies_array[i] == 1] = 200
    permeabilities[i][facies_array[i] == 2] = 200
    permeabilities[i][facies_array[i] == 3] = 200

    permeabilities_3D[i] = np.stack([permeabilities[i]]*3, axis=-1)

# Assign maximum and minimum values for permeability
perm_max = 200
perm_min = 100


###############################################################################
# Plot example permeability
# -------------------------

# Plot 10 first permeability models
fig, axs = plt.subplots(
    2, 5, figsize=(8, 3), sharex=True, sharey=True, constrained_layout=True)
axs = axs.ravel()
fig.suptitle("Permeabilities")
for i in range(10):
    im = axs[i].imshow(
        permeabilities[i, ...], vmin=perm_min, vmax=perm_max, origin="lower"
    )
fig.colorbar(im, ax=axs, label="Log Permeability (mD)")


###############################################################################
# Create DARTS simulation functions
# ---------------------------------

def simulation(permeabilities):
    """Simulation function."""
    reduce = False
    if permeabilities.ndim == 3:
        reduce = True
        permeabilities = permeabilities[None, ...]
    temp_data = np.zeros((permeabilities.shape[0], 31))
    for i, perm in enumerate(permeabilities):
        m = Model(perm=perm)
        m.init()

        # Initial conditions
        m.run(1e-3)

        for t in range(3):
            # run and output every 10 years (30 in total)
            m.run(10*365, restart_dt=365)

        td = pd.DataFrame.from_dict(m.physics.engine.time_data)

        time = td["time"].values / 365
        temp = td["PRD : temperature (K)"].values

        # I've done this because the first time step is sometimes cut and then
        # repeated in the time_data if you can change the time_data to not
        # report cut timesteps, then this may not be necessary
        if len(time) > 31:
            time = time[1:]
            temp = temp[1:]
        temp_data[i, :] = temp

    if reduce:
        temp_data = temp_data[0, :]
    return temp_data


###############################################################################
# Prepare permeability maps, run simulation for prior and create observations
# ---------------------------------------------------------------------------

# Assign true permeability (first) and prior permeability
perm_true = permeabilities_3D[0]
perm_prior = permeabilities_3D[1:]

# Simulate true data and prior data
data_true = simulation(perm_true)
data_prior = simulation(perm_prior)

time = np.arange(31.)

# Add noise to the true data and create observed data
dstd = 0.2
data_obs = rng.normal(data_true, dstd)

# Plot the true, prior and observed data
fig, ax = plt.subplots(1, 1)
ax.scatter(time, data_obs, label="Observed", color="red", zorder=10)
for i in range(np.shape(data_prior)[0]):
    if i == 0:
        ax.plot(time, data_prior[i], label="Prior", color="grey", alpha=0.4)
    else:
        ax.plot(time, data_prior[i], color="grey", alpha=0.4)
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


# Create input dictionary for ES-MDA
inp = {
    "model_prior": perm_prior,
    "forward": simulation,
    "data_obs": data_obs,
    "sigma": dstd,
    "alphas": 4,
    "data_prior": data_prior,
    "callback_post": restrict_permeability,
    "random": rng,
}


###############################################################################

# Run ES-MDA
perm_post, data_post = resmda.esmda(**inp)


###############################################################################

# Plot the prior, observed data and posterior data
fig, ax = plt.subplots(1, 1)

ax.scatter(time, data_obs, label="Observed", color="red", zorder=10)

for i in range(np.shape(data_post)[0]):
    if i == 0:
        ax.plot(time, data_post[i], label="Posterior", color="blue")
    else:
        ax.plot(time, data_post[i], color="blue")

for i in range(np.shape(data_prior)[0]):
    if i == 0:
        ax.plot(time, data_prior[i], label="Prior", color="grey", alpha=0.4)
    else:
        ax.plot(time, data_prior[i], color="grey", alpha=0.4)

ax.legend()
ax.set_xlabel("Time (years)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature at Production Well")


###############################################################################

# Plot 10 first posterior permeability models
fig, axs = plt.subplots(
    2, 5, figsize=(8, 3), sharex=True, sharey=True, constrained_layout=True)
axs = axs.ravel()
fig.suptitle("Permeabilities")
for i in range(10):
    im = axs[i].imshow(
        perm_post[i, :, :, 0], vmin=perm_min, vmax=perm_max, origin="lower"
    )
fig.colorbar(im, ax=axs, label="Log Permeability (mD)")


###############################################################################
fname = 'output.h5'
compr = 'gzip'
with h5py.File(fname, 'w') as h5file:
    h5file.create_dataset('time', data=time, compression=compr)
    h5file.create_dataset('data_true', data=data_true, compression=compr)
    h5file.create_dataset('data_obs', data=data_obs, compression=compr)
    h5file.create_dataset('data_prior', data=data_prior, compression=compr)
    h5file.create_dataset('data_post', data=data_post, compression=compr)
    h5file.create_dataset('perm_true', data=perm_true, compression=compr)
    h5file.create_dataset('perm_prior', data=perm_prior, compression=compr)
    h5file.create_dataset('perm_post', data=perm_post, compression=compr)


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
#     # ==== Create a fluvsim instance with the geological parameters ====
#
#     # FLUVSIM Version used: 2.900
#     from geomodpy.wrapper.fluvsim import FLUVSIM
#
#     fluvsim = FLUVSIM(
#         channel_orientation=(60., 90., 120.),
#         # Proportions for each facies
#         channel_prop=0.5,
#         crevasse_prop=0.0,
#         levee_prop=0.0,
#         # Parameters defining the grid
#         shape=(60, 60),
#         spacing=(60., 60.),
#         origin=(0.5, 0.5),
#         # Number of realizations and random seed
#         n_realizations=101,
#         seed=42,
#     )
#
#     # ==== Create the facies ====
#     facies = fluvsim.run()
#
#     # Convert spacing dictionary values to a tuple
#     facies.attrs["spacing"] = tuple(facies.attrs["spacing"].values())
#     # Convert origin dictionary values to a tuple
#     facies.attrs["origin"] = tuple(facies.attrs["origin"].values())
#
#     # ==== Save the facies dataset to a NetCDF file ====
#     facies.to_netcdf("facies_geothermal.nc")

###############################################################################
resmda.Report()
