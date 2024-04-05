import numpy as np
import scipy as sp


class ReservoirSim:
    """A small Reservoir Simulator.


    """

    def __init__(self, perm_field, phi=0.2, c_f=1e-5, p0=1, rho0=1, mu_w=1,
                 rw=0.15, pres_ini=150, wells=None, dx=50, dy=50, dz=10):
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
        dx, dy, dz : floats or 1D array
            Cell dimensions (m).

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
        self.dy = dy
        self.dz = dz
        self.pres_ini = pres_ini

        # Store volumes : TODO : generalize to arb. volumes
        self.volumes = np.ones(self.size) * self.dx * self.dy * self.dz

        if wells is None:
            self.wells = np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])
        else:
            self.wells = np.array(wells)

        # Get well terms.
        # TODO : Only depends on dx and dz, WHY?
        # well index
        wi = 2 * np.pi * self.perm_field * self.dz
        wi /= self.mu_w * np.log(0.208 * self.dx / self.rw)

        # Add wells
        locs = self.wells[:, 1]*self.nx + self.wells[:, 0]
        self._add_wells_f = np.zeros(self.size)
        self._add_wells_d = np.zeros(self.size)
        self._add_wells_f[locs] += self.wells[:, 2] * wi[locs]
        self._add_wells_d[locs] += wi[locs]

    def solve(self, compr, p):
        """Construct K-matrix."""

        # Mobility ratio without permeability
        phi = self.rho0 * (1 + self.c_f * (p - self.p0)) / self.mu_w

        # Pre-allocate diagonals.
        mn = np.zeros(self.size)
        m1 = np.zeros(self.size)
        d = compr.copy()
        p1 = np.zeros(self.size)
        pn = np.zeros(self.size)

        t1 = self.dy * self.perm_field[:-1] * self.perm_field[1:]
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
        f = compr * p + self._add_wells_f
        d += self._add_wells_d

        # Bring to sparse matrix
        offsets = np.array([-self.nx, -1, 0, 1, self.nx])
        data = np.array([mn, m1, d, p1, pn])
        K = sp.sparse.dia_array((data, offsets), shape=(self.size, self.size))

        # Solve the system
        return sp.sparse.linalg.spsolve(K.tocsc(), f, use_umfpack=False)

    def __call__(self, dt=np.ones(10)*0.0001):
        """Run simulator.

        Parameters
        ----------
        dt : array
            Time steps.

        """

        pressure = np.ones((dt.size+1, self.size)) * self.pres_ini
        for i, d in enumerate(dt):
            compr = self.volumes * self.phi * self.c_f / d
            pressure[i+1, :] = self.solve(compr, pressure[i, :])

        return pressure.reshape((dt.size+1, *self.shape), order='F')


def index2ij(index, nx, ny):
    """Convert index numeration to ij-index."""
    return ((index % nx) + 1, (index // nx) + 1)


def ij2index(i, j, nx, ny):
    """Convert ij numeration to index."""
    return (i-1) + (j-1)*nx


# TODO 0: Also implement Sphere function
# TODO 1: Ensure it is the same as before
# TODO 2: It could be further speedup:
#         the first loop is only necessary for i=1
def build_perm_cov_matrix(nx, ny, length, theta, sigma_pr2):
    cost = np.cos(theta)
    sint = np.sin(theta)
    cov = np.zeros([nx*ny, nx*ny])
    xx = [((i % nx) + 1, (i // nx) + 1) for i in range(nx*ny)]
    for i in range(nx):
        x0 = xx[i]
        for j in range(nx*ny):
            x1 = xx[j]
            d0 = x1[0]-x0[0]
            d1 = x1[1]-x0[1]
            rot0 = cost*d0 - sint*d1
            rot1 = sint*d0 + cost*d1

            # Gaspari Cohn TODO get powers of, w\o sqrt
            hl = np.sqrt((rot0/length[0])**2 +
                         (rot1/length[1])**2)

            if hl < 1:
                cov[i, j] = (-(hl**5)/4 + (hl**4)/2 + (hl**3)*5/8 -
                             (hl**2)*5/3 + 1)
            elif hl >= 1 and hl < 2:
                cov[i, j] = ((hl**5)/12 - (hl**4)/2 + (hl**3)*5/8 +
                             (hl**2)*5/3 - hl*5 + 4 - (1/hl)*2/3)
    for j in range(1, ny):
        cov[nx*j:nx*(j+1), nx*j:] = cov[:nx, :-nx*j]
        for i in range(j):
            cov[nx*j:nx*(j+1), nx*(j-i-1):nx*(j-i)] = cov[:nx, nx*i:nx*(i+1)]

    return cov
