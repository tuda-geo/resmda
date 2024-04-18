import numpy as np
import scipy as sp


class Simulator:
    """A small Reservoir Simulator.


    """

    def __init__(self, perm_field, phi=0.2, c_f=1e-5, p0=1, rho0=1, mu_w=1,
                 rw=0.15, pres_ini=150, wells=None, dx=50, dz=10):
        """Initialize a Simulation instance.

        Parameters
        ----------
        perm_field : 2D array
            Permeabilities ny-by-nx (?)
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

        self.size = perm_field.size
        self.shape = perm_field.shape
        self.nx, self.ny = perm_field.shape
        self.perm_field = perm_field.ravel('F')

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
            self.wells = np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])
        else:
            self.wells = np.array(wells)

        # Get well locations and set terms
        self.locs = self.wells[:, 1]*self.nx + self.wells[:, 0]
        self._set_well_terms

    @property
    def _set_well_terms(self):
        # Get well terms
        wi = 2 * np.pi * self.perm_field[self.locs] * self.dz
        wi /= self.mu_w * np.log(0.208 * self.dx / self.rw)

        # Add wells
        self._add_wells_f = self.wells[:, 2] * wi
        self._add_wells_d = wi

    def solve(self, pressure, dt):
        """Construct K-matrix."""

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

    def __call__(self, dt=np.ones(10)*0.0001, perm_field=None):
        """Run simulator.

        Parameters
        ----------
        dt : array
            Time steps.

        """
        # Update permeability field if provided
        if perm_field is not None:
            self.perm_field = perm_field.ravel('F')
            self._set_well_terms

        pressure = np.ones((dt.size+1, self.size)) * self.pres_ini
        for i, d in enumerate(dt):
            pressure[i+1, :] = self.solve(pressure[i, :], d)

        return pressure.reshape((dt.size+1, *self.shape), order='F')


def covariance(nx, ny, length, theta, sigma_pr2, dtype='float32'):
    """Build co-variance matrix for permeability.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y
    theta : float
        Angle (in degrees)
    """
    nc = nx*ny
    cost = np.cos(theta)
    sint = np.sin(theta)

    # 1. Fill the first row nx * nc, but vertically
    tmp1 = np.zeros([nx, nc], dtype=dtype)
    for i in range(nx):
        tmp1[i, 0] = 1.0  # diagonal
        for j in range(i+1, nc):
            d0 = (j % nx) - i
            d1 = (j // nx)
            rot0 = cost*d0 - sint*d1
            rot1 = sint*d0 + cost*d1
            hl = np.sqrt((rot0/length[0])**2 + (rot1/length[1])**2)
            if hl <= 1:
                tmp1[i, j-i] = sigma_pr2 * (1 - 1.5*hl + hl**3/2)

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
