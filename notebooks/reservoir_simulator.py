import numpy as np
import scipy as sp


class ReservoirSim:
    """A small Reservoir Simulator.

    `nx`: numbers of cells in x [-]
    `ny`: numbers of cells in y [-]
    `perm_field`: permeabilities (?), [dimensionless?]
    `phi`: porosity (?) [dimensionless?]
    `c_f`: ?
    `p0`: ?
    `rho0`: ?
    `mu_w`: ?
    `rw`: ?
    `dx`: thickness of cells in x [m]
    `dy`: thickness of cells in y [m]
    `dz`: thickness of cells in z [m]
    `pres_ini`: initial pressure [?]
    `pres_prd`: production pressure [?]
    `pres_inj`: injection pressure [?]

    """

    def __init__(self, perm_field, **kwargs):
        """Initialize a Simulation instance."""
        # TODO: MOVE TO MORE EXPLICIT INPUTS, NOT ONLY KWARGS

        # Set parameters from input or defaults
        self.nx, self.ny = perm_field.shape
        if self.nx != self.ny:
            print("Warning, if nx!=ny result is probably wrong.")
        self.size = perm_field.size
        self.shape = perm_field.shape
        self.np = (self.nx - 1) * self.ny + self.nx * (self.ny - 1)
        self.perm_field = perm_field.ravel()
        self.phi = kwargs.pop('phi', 0.2)
        self.c_f = kwargs.pop('c_f', 1e-5)
        self.p0 = kwargs.pop('p0', 1)
        self.rho0 = kwargs.pop('rho0', 1)
        self.mu_w = kwargs.pop('mu_w', 1)
        self.rw = kwargs.pop('rw', 0.15)
        self.dx = kwargs.pop('dx', 50)
        self.dy = kwargs.pop('dy', 50)
        self.dz = kwargs.pop('dz', 10)
        self.pres_ini = kwargs.pop('pres_ini', 150)

        # TODO : make a flexible list of wells
        self.pres_prd = kwargs.pop('pres_prd', 120)
        self.pres_inj = kwargs.pop('pres_inj', 180)

        # Store volumes : TODO : generalize to arb. volumes
        self.volumes = np.ones(self.size) * self.dx * self.dy * self.dz

        # TODO : only necessary for well locations
        # Only depends on dx and dz, WHY?
        self.wi = 2 * np.pi * self.perm_field * self.dz
        self.wi /= self.mu_w * np.log(0.208 * self.dx / self.rw)

    def get_matrix(self, Phi, compr, p):
        """Construct K-matrix."""

        # Pre-allocate diagonals.
        mn = np.zeros(self.size)
        m1 = np.zeros(self.size)
        d = compr.copy()
        p1 = np.zeros(self.size)
        pn = np.zeros(self.size)

        # v v TODO Adjust for nx!=ny TODO
        t1 = self.dy * self.perm_field[:-1] * self.perm_field[1:]
        t1 /= self.perm_field[:-1] + self.perm_field[1:]
        t1 *= (Phi[:-1] + Phi[1:]) / 2
        t1[self.nx-1::self.nx] = 0.0
        d[:-1] += t1
        d[1:] += t1
        m1[:-1] -= t1
        p1[1:] -= t1

        t2 = self.dx * self.perm_field[:-self.nx] * self.perm_field[self.nx:]
        t2 /= self.perm_field[:-self.nx] + self.perm_field[self.nx:]
        t2 *= (Phi[:-self.nx] + Phi[self.nx:]) / 2
        d[:-self.nx] += t2
        d[self.nx:] += t2
        mn[:-self.nx] -= t2
        pn[self.nx:] -= t2

        # Bring to sparse matrix
        # offsets = np.array([-self.ny, -1, 0, 1, self.nx])  # TODO
        offsets = np.array([-self.nx, -1, 0, 1, self.nx])  # Gabriel's version
        data = np.array([mn, m1, d, p1, pn])
        K = sp.sparse.dia_array((data, offsets), shape=(self.size, self.size))

        return K.tocsc()

    def simulate(self, realizations=10, dt=0.0001):
        """Run simulator."""
        compr = self.volumes * self.phi * self.c_f / dt

        P = np.empty((realizations+1, self.size))
        P[0, :] = np.ones(self.size) * self.pres_ini

        for i in range(realizations):

            dens = self.rho0 * (1 + self.c_f * (P[i, :] - self.p0))
            beta = dens / self.mu_w

            K = self.get_matrix(beta, compr, P[i, :])
            f = compr * P[i, :]

            # TODO : make a flexible list of wells
            f[0] += self.pres_inj * self.wi[0]
            K[0, 0] += self.wi[0]
            f[-1] += self.pres_prd * self.wi[-1]
            K[-1, -1] += self.wi[-1]

            # Solve the system
            P[i+1, :] = sp.sparse.linalg.spsolve(K, f, use_umfpack=False)

        return P.reshape((realizations+1, *self.shape))
