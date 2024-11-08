
r"""
Geothermal Case Study
=====================

**This example was contributed by Mona Devos.**


.. note::

    To retrieve the data, you need to have ``pooch`` installed:

    .. code-block:: bash

        pip install pooch

    or

    .. code-block:: bash

        conda install -c conda-forge pooch


"""
import os

import pooch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import resmda

from darts.engines import redirect_darts_output
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.physics.properties.iapws import iapws_property_vec
from darts.reservoirs.struct_reservoir import StructReservoir
redirect_darts_output("run_geothermal.log")

# Adjust this path to a folder of your choice.
data_path = os.path.join("..", "download", "")

###############################################################################
# Load facies models
# ------------------
#
# .. todo:
#
#       Recreate the fluvial model, store the input, and add all code at the
#       end, just as in the fluvial example; create tag and give a link with a
#       date.
#

fname = "facies_geothermal.nc"
pooch.retrieve(
    "https://github.com/tuda-geo/data/raw/refs/heads/main/resmda/"+fname,
    "814956d887f6e688eaae9a2c144aa0e6120361a8fb6ebe00f17d27038d877df3",
    fname=fname,
    path=data_path,
)
facies = xr.load_dataset(data_path + fname)


###############################################################################
class Model(DartsModel):
    def __init__(self, perm, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        # parameters for the reservoir
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz

        self.dx = 30
        self.dy = 30
        dz = np.ones(nb) * 30

        perm = perm.flatten('F')
        poro = np.ones(nb) * 0.2

        # discretize structured reservoir
        self.reservoir = StructReservoir(
            self.timer, nx=nx, ny=ny, nz=nz, dx=self.dx, dy=self.dy, dz=dz,
            permx=perm, permy=perm, permz=perm*0.1, poro=poro, depth=2000,
            hcap=2200, rcond=500,
        )

        # add open boundaries
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        # add well's locations
        self.iw = [30, 30]
        self.jw = [14, 46]

        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(
            self.timer, n_points, 1, 351, 1000, 10000, cache=False,
        )
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        # timestep parameters
        self.params.first_ts = 1e-3
        self.params.mult_ts = 2
        self.params.max_ts = 365

        # nonlinear and linear solver tolerance
        self.params.tolerance_newton = 1e-2

        self.timer.node["initialization"].stop()

    def set_wells(self):
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation(
                "INJ",
                cell_index=(self.iw[0], self.jw[0], k + 1),
                well_radius=0.16,
                multi_segment=True,
            )

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation(
                "PRD",
                cell_index=(self.iw[1], self.jw[1], k + 1),
                well_radius=0.16,
                multi_segment=True,
            )

    def set_initial_conditions(self):
        # initialization with constant pressure and temperature
        self.physics.set_uniform_initial_conditions(
            self.reservoir.mesh,
            uniform_pressure=200,
            uniform_temperature=350,
        )

    def set_boundary_conditions(self):
        # activate wells with rate control for injector and producer
        for i, w in enumerate(self.reservoir.wells):
            if 'INJ' in w.name:
                w.control = self.physics.new_rate_water_inj(4000, 300)
            else:
                w.control = self.physics.new_rate_water_prod(4000)


###############################################################################
# Convert facies to permeability
# ------------------------------

# Define the number of facies models
n = facies['facies code'].shape[0]

# Create a 3D array for facies and permeability
facies_array = np.zeros((n, 60, 60))
perm_array = np.zeros((n, 60, 60))
perm_array_3D = np.zeros((n, 60, 60, 3))

# Assign facies and permeability values
for i in range(n):
    facies_array[i] = facies['facies code'][i, 0].values

    perm_array[i][facies_array[i] == 0] = 100
    perm_array[i][facies_array[i] == 1] = 200
    perm_array[i][facies_array[i] == 2] = 200
    perm_array[i][facies_array[i] == 3] = 200

    perm_array_3D[i] = np.stack([perm_array[i]]*3, axis=-1)

# Assign maximum and minimum values for permeability
perm_max = 200
perm_min = 100


###############################################################################
# Plot example permeability
# -------------------------

# Plot 5 first facies models
for i in range(5):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title('Permeability')
    im = ax.imshow(perm_array[i], origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Permeability (mD)')
    plt.tight_layout()


###############################################################################
# Create DARTS simulation functions
# ---------------------------------

def sim_true(perm_array):
    m = Model(perm=perm_array)
    m.init()

    # initial conditions
    m.run(1e-3)

    for t in range(3):
        # run and output every 10 years (30 in total)
        m.run(10*365, restart_dt=365)

    td = pd.DataFrame.from_dict(m.physics.engine.time_data)

    time = td['time'].values / 365
    temp = td['PRD : temperature (K)'].values

    if len(time) > 31:
        time = time[1:]
        temp = temp[1:]

    return temp


###############################################################################

def sim(perm_array):
    temp_data = []
    for perm in perm_array:
        m = Model(perm=perm)
        m.init()

        # initial conditions
        m.run(1e-3)

        for t in range(3):
            # run and output every 10 years (30 in total)
            m.run(10*365, restart_dt=365)

        td = pd.DataFrame.from_dict(m.physics.engine.time_data)

        time = td['time'].values / 365
        temp = td['PRD : temperature (K)'].values

        # I've done this because the first time step is sometimes cut and then
        # repeated in the time_data if you can change the time_data to not
        # report cut timesteps, then this may not be necessary
        if len(time) > 31:
            time = time[1:]
            temp = temp[1:]
        temp_data.append(temp)

    return np.array(temp_data)


###############################################################################
# Prepare permeability maps, run simulation for prior and create observations
# ---------------------------------------------------------------------------

# Assign true permeability (first) and prior permeability
perm_true = perm_array_3D[0]
perm_prior = perm_array_3D[1:]

# Simulate true data and prior data
data_true = sim_true(perm_true)
data_prior = sim(perm_prior)

time = np.arange(0, 31, 1)

# Add noise to the true data and create observed data
dstd = 0.2
rng = np.random.default_rng(42)
data_obs = rng.normal(data_true, dstd)

# Plot the true, prior and observed data
plt.figure(figsize=(10, 6))
plt.scatter(time, data_obs, label='Observed', color='red', zorder=10)
for i in range(np.shape(data_prior)[0]):
    if i == 0:
        plt.plot(time, data_prior[i], label='Prior', color='grey', alpha=0.4)
    else:
        plt.plot(time, data_prior[i], color='grey', alpha=0.4)
plt.legend()
plt.xlabel('Time (years)')
plt.ylabel('Temperature (K)')
plt.title('Temperature at Production Well')
plt.show()


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
    "forward": sim,
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
plt.scatter(time, data_obs, label='Observed', color='red', zorder=10)
for i in range(np.shape(data_post)[0]):
    if i == 0:
        plt.plot(time, data_post[i], label='Posterior', color='blue')
    else:
        plt.plot(time, data_post[i], color='blue')
for i in range(np.shape(data_prior)[0]):
    if i == 0:
        plt.plot(time, data_prior[i], label='Prior', color='grey', alpha=0.4)
    else:
        plt.plot(time, data_prior[i], color='grey', alpha=0.4)

plt.legend()
plt.xlabel('Time (years)')
plt.ylabel('Temperature (K)')
plt.title('Temperature at Production Well')
plt.show()


###############################################################################

# Plot the first 5 posterior permeability  models
for i in range(5):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title('Permeability')
    im = ax.imshow(perm_post[i, :, :, 0], origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Permeability (mD)')
    plt.tight_layout()

###############################################################################
resmda.Report()
