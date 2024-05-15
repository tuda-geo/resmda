# Copyright 2024 Dieter Werthmüller, Gabriel Serrao Seabra
#
# This file is part of resmda.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import scipy as sp

from resmda.utils import rng

__all__ = ['Simulator', 'RandomPermeability', 'covariance']


def __dir__():
    return __all__


class Simulator:
    """A small Reservoir Simulator.


    """

    def __init__(self, nx, ny, phi=0.2, c_f=1e-5, p0=1, rho0=1, mu_w=1,
                 rw=0.15, pres_ini=150, wells=None, dx=50, dz=10):
        """Initialize a Simulation instance.

        Parameters
        ----------
        nx, ny : int
            Dimension of field
        phi : float
            Porosity (-)
        c_f : float
            Formation compressibility (1/kPa)
        p0 : float
            Initial pressure (bar or kPa?)
        rho0 : float
            Fixed density (kg/m3)
        mu_w : float
            Viscosity (cP - Pa s)
        rw : float
            Well radius (m)
        pres_ini : initial pressure [?]
        wells : location and pressure of wells [?]
        dx, dz : floats
            Cell dimensions in horizontal (dx) and vertical (dz)
            directions (m).

        """

        self.size = nx*ny
        self.shape = (nx, ny)
        self.nx = nx
        self.ny = ny

        self.phi = phi
        self.c_f = c_f
        self.p0 = p0
        self.rho0 = rho0
        self.mu_w = mu_w
        self.rw = rw
        self.dx = dx
        self.dz = dz
        self.pres_ini = pres_ini

        # Store volumes (needs adjustment for arbitrary cell volumes)
        self.volumes = np.ones(self.size) * self.dx * self.dx * self.dz
        self._vol_phi_cf = self.volumes * self.phi * self.c_f

        if wells is None:
            # Default wells setup if none provided. Each well is specified by its grid coordinates followed by its pressure.
            # The first well ([0, 0, 180]) is placed at the top-left corner with a pressure of 180 units,
            # representing an injection pressure. The second well ([self.nx-1, self.ny-1, 120]),
            # located at the bottom-right corner, has a pressure of 120 units, possibly a lower pressure or production scenario.
            self.wells = np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])  
        else:
            self.wells = np.array(wells)

        # Get well locations and set terms
        self.locs = self.wells[:, 1]*self.nx + self.wells[:, 0]

    @property
    def _set_well_terms(self):
        """
        Calculate well terms based on current permeability field, to be used in the simulation.
        Adjust well impacts using calculated terms.
        """
        wi = 2 * np.pi * self.perm_field[self.locs] * self.dz
        wi /= self.mu_w * np.log(0.208 * self.dx / self.rw)

        # Add wells
        self._add_wells_f = self.wells[:, 2] * wi
        self._add_wells_d = wi

    def solve(self, pressure, dt):
        """
        Construct and solve the K-matrix for the simulation of pressure over time.

        Parameters:
        - pressure: Current pressure state of the reservoir
        - dt: Time step for the simulation

        Returns:
        - Pressure state after applying the time step.
        """

        # Mobility ratio without permeability
        phi = self.rho0 * (1 + self.c_f * (pressure - self.p0)) / self.mu_w

        # Compr. and right-hand side f
        compr = self._vol_phi_cf / dt
        f = compr * pressure

        # Pre-allocate diagonals.
        mn = np.zeros(self.size)
        m1 = np.zeros(self.size)
        d = compr
        p1 = np.zeros(self.size)
        pn = np.zeros(self.size)

        t1 = self.dx * self.perm_field[:-1] * self.perm_field[1:]
        t1 /= self.perm_field[:-1] + self.perm_field[1:]
        t1 *= (phi[:-1] + phi[1:]) / 2
        t1[self.nx-1::self.nx] = 0.0
        d[:-1] += t1
        d[1:] += t1
        m1[:-1] -= t1
        p1[1:] -= t1

        t2 = self.dx * self.perm_field[:-self.nx] * self.perm_field[self.nx:]
        t2 /= self.perm_field[:-self.nx] + self.perm_field[self.nx:]
        t2 *= (phi[:-self.nx] + phi[self.nx:]) / 2
        d[:-self.nx] += t2
        d[self.nx:] += t2
        mn[:-self.nx] -= t2
        pn[self.nx:] -= t2

        # Add wells.
        f[self.locs] += self._add_wells_f
        d[self.locs] += self._add_wells_d

        # Bring to sparse matrix
        offsets = np.array([-self.nx, -1, 0, 1, self.nx])
        data = np.array([mn, m1, d, p1, pn])
        K = sp.sparse.dia_array((data, offsets), shape=(self.size, self.size))

        # Solve the system
        return sp.sparse.linalg.spsolve(K.tocsc(), f, use_umfpack=False)

    def __call__(self, perm_fields, dt=np.ones(10)*0.0001, data=False):
        """
        Run the simulation across multiple time steps and possibly multiple permeability scenarios.

        Parameters:
        - perm_fields: Array of permeability fields to simulate
        - dt: Array of time steps to use for simulation
        - data: Specific indices to output data for, if False, return all data

        Returns:
        - Simulation results over time for given permeability fields.
        """
        if perm_fields.ndim == 2:
            ne = 1
            perm_fields = [perm_fields, ]
        else:
            ne = perm_fields.shape[0]
        nt = dt.size+1

        out = np.zeros((ne, nt, self.nx, self.ny))
        for n, perm_field in enumerate(perm_fields):

            self.perm_field = perm_field.ravel('F')
            self._set_well_terms

            pressure = np.ones((dt.size+1, self.size)) * self.pres_ini
            for i, d in enumerate(dt):
                pressure[i+1, :] = self.solve(pressure[i, :], d)
            out[n, ...] = pressure.reshape((dt.size+1, *self.shape), order='F')
        if ne == 1:
            out = out[0, ...]

        if data:
            return out[..., data[0], data[1]]
        else:
            return out


class RandomPermeability:
    """
    Class to generate random permeability fields based on specified statistical properties.
    """

    def __init__(self, nx, ny, perm_mean, perm_min, perm_max, length=(10, 10),
                 theta=45, sigma_pr2=1.0, dtype='float32'):
        """
        Initialize parameters for generating random permeability fields.

        Parameters:
        - nx, ny: Dimensions of the grid
        - perm_mean: Mean permeability
        - perm_min, perm_max: Minimum and maximum values for permeability
        - length: Length scales for the correlation of permeability
        - theta: Rotation angle for the anisotropy in the permeability field
        - sigma_pr2: Variance scale for the permeability
        - dtype: Data type for computations, for precision and performance tuning
        """
        self.nx, self.ny = nx, ny  # Grid dimensions
        self.nc = nx * ny  # Total number of cells
        self.perm_mean, self.perm_min, self.perm_max = perm_mean, perm_min, perm_max  # Permeability statistics
        self.length, self.theta = length, theta  # Anisotropy parameters
        self.sigma_pr2, self.dtype = sigma_pr2, dtype  # Variance and data type

    @property
    def cov(self):
        """
        Lazy-loaded covariance matrix, calculated based on anisotropy and statistical parameters.
        """
        if not hasattr(self, '_cov'):
            self._cov = covariance(
                nx=self.nx, ny=self.ny, length=self.length,
                theta=self.theta, sigma_pr2=self.sigma_pr2, dtype=self.dtype
            )
        return self._cov

    @property
    def lcho(self):
        """
        Lower Cholesky decomposition of the covariance matrix, used for generating random fields.
        """
        if not hasattr(self, '_lcho'):
            self._lcho = sp.linalg.cholesky(self.cov, lower=True)
        return self._lcho

    def __call__(self, n, perm_mean=None, perm_min=None, perm_max=None,
                 random=None):
        """
        Generate n random permeability fields using the specified statistical parameters and random seed.

        Parameters:
        - n: Number of fields to generate
        - perm_mean: Mean permeability to override the initialized value
        - perm_min, perm_max: Min and max permeability values to clip the fields
        - random: Random seed for reproducibility

        Returns:
        - An array of generated permeability fields.
        """
        if perm_mean is None:
            perm_mean = self.perm_mean
        if perm_min is None:
            perm_min = self.perm_min
        if perm_max is None:
            perm_max = self.perm_max

        out = np.full((n, self.nx, self.ny), perm_mean, order='F')  # Initialize fields with mean permeability
        for i in range(n):
            z = rng(random).normal(size=self.nc)  # Generate random numbers
            out[i, ...] += (self.lcho @ z).reshape((self.nx, self.ny), order='F')  # Apply the Cholesky transform

        return out.clip(perm_min, perm_max)  # Clip the results to stay within specified bounds



def covariance(nx, ny, length, theta, sigma_pr2, dtype='float32'):
    """
    Generate a covariance matrix based on grid size, anisotropy, and statistical parameters.

    Parameters:
    - nx, ny: Dimensions of the grid
    - length: Length scales for the correlation of permeability
    - theta: Rotation angle for the anisotropy in the permeability field
    - sigma_pr2: Variance scale for the permeability
    - dtype: Data type for computations

    Returns:
    - A covariance matrix for the permeability field.
    """
    nc = nx * ny  # Total number of cells
    cost, sint = np.cos(theta), np.sin(theta)  # Precompute cosine and sine of the rotation angle

    # 1. Fill the first row of the covariance matrix
    tmp1 = np.zeros([nx, nc], dtype=dtype)
    for i in range(nx):
        tmp1[i, 0] = 1.0  # Set diagonal
        for j in range(i+1, nc):
            d0, d1 = (j % nx) - i, (j // nx)  # Distance in the x and y directions
            rot0, rot1 = cost*d0 - sint*d1, sint*d0 + cost*d1  # Rotate coordinates
            hl = np.sqrt((rot0/length[0])**2 + (rot1/length[1])**2)  # Calculate the scaled distance

            # Sphere formula for covariance, modified for anisotropy
            if sigma_pr2:  # Non-zero variance scale
                if hl <= 1:
                    tmp1[i, j-i] = sigma_pr2 * (1 - 1.5*hl + hl**3/2)

            else:  # Gaspari-Cohn function for smoothness
                if hl < 1:
                    tmp1[i, j-i] = (-(hl**5)/4 + (hl**4)/2 + (hl**3)*5/8 - (hl**2)*5/3 + 1)
                elif hl >= 1 and hl < 2:
                    tmp1[i, j-i] = ((hl**5)/12 - (hl**4)/2 + (hl**3)*5/8 + (hl**2)*5/3 - hl*5 + 4 - (1/hl)*2/3)


    # 2. Get the indices of the non-zero columns
    ind = np.where(tmp1.sum(axis=0))[0]

    # 3. Expand the non-zero colums ny-times
    tmp2 = np.zeros([nc, ind.size], dtype=dtype)
    for i, j in enumerate(ind):
        n = j//nx
        tmp2[:nc-n*nx, i] = np.tile(tmp1[:, j], ny-n)

    # 4. Construct array through sparse diagonal array
    cov = sp.sparse.dia_array((tmp2.T, -ind), shape=(nc, nc))
    return cov.toarray()
