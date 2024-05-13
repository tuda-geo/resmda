import numpy as np
import scipy as sp

# Instantiate a random number generator
# Currently with a fixed seed for development/reproducibility
rng = np.random.default_rng(1848)


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
            self.wells = np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])
        else:
            self.wells = np.array(wells)

        # Get well locations and set terms
        self.locs = self.wells[:, 1]*self.nx + self.wells[:, 0]

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

    def __call__(self, perm_fields, dt=np.ones(10)*0.0001, data=False):
        """Run simulator.

        Parameters
        ----------
        dt : array
            Time steps.

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

    def __init__(self, nx, ny, perm_mean, perm_min, perm_max, length=(10, 10),
                 theta=45, sigma_pr2=1.0, dtype='float32'):
        self.nx = nx
        self.ny = ny
        self.nc = nx*ny
        self.perm_mean = perm_mean
        self.perm_min = perm_min
        self.perm_max = perm_max
        self.length = length
        self.theta = theta
        self.sigma_pr2 = sigma_pr2
        self.dtype = dtype

    @property
    def cov(self):
        if not hasattr(self, '_cov'):
            self._cov = covariance(
                nx=self.nx, ny=self.ny, length=self.length,
                theta=self.theta, sigma_pr2=self.sigma_pr2, dtype=self.dtype
            )
        return self._cov

    @property
    def lcho(self):
        if not hasattr(self, '_lcho'):
            self._lcho = sp.linalg.cholesky(self.cov, lower=True)
        return self._lcho

    def __call__(self, n, perm_mean=None, perm_min=None, perm_max=None):
        if perm_mean is None:
            perm_mean = self.perm_mean
        if perm_min is None:
            perm_min = self.perm_min
        if perm_max is None:
            perm_max = self.perm_max
        out = np.full((n, self.nx, self.ny), perm_mean, order='F')
        for i in range(n):
            z = rng.normal(size=self.nc)
            out[i, ...] += (self.lcho @ z).reshape(
                    (self.nx, self.ny), order='F')

        return out.clip(perm_min, perm_max)


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

            # Calculate value.
            if sigma_pr2:  # Sphere formula
                if hl <= 1:
                    tmp1[i, j-i] = sigma_pr2 * (1 - 1.5*hl + hl**3/2)

            else:  # Gaspari Cohn
                if hl < 1:
                    tmp1[i, j-i] = (-(hl**5)/4 + (hl**4)/2 + (hl**3)*5/8 -
                                    (hl**2)*5/3 + 1)
                elif hl >= 1 and hl < 2:
                    tmp1[i, j-i] = ((hl**5)/12 - (hl**4)/2 + (hl**3)*5/8 +
                                    (hl**2)*5/3 - hl*5 + 4 - (1/hl)*2/3)

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


def esmda(model_prior, forward, data_obs, sigma,
          alphas=4, data_prior=None, vmin=None, vmax=None):
    """ESMDA algorithm (Emerick and Reynolds, 2013).

    Parameters
    ---------
    model_post : (Ne, Nx, Ny)
    data_prior : (Ne, Nt)

    # TODO - adjust
    Do : [Nd] observed data
    D : [Nd, Ne] simulated data by (Ne) ensemble members
    M : [Nm, Ne] matrix of model parameters to be estimated
    alpha : es-mda inflation factor
    Ce : [Nd] variance of observed data


    Returns
    -------
    # TODO - adjust
    M2 : [Nm, Ne] matrix of model parameters posterior in each alpha iteration



    """
    # Get number of ensembles and time steps
    ne = model_prior.shape[0]
    nt = data_obs.size

    # Get alphas
    if isinstance(alphas, int):
        alphas = np.zeros(alphas) + alphas
    else:
        alphas = np.asarray(alphas)

    # Copy prior as start of post (output)
    model_post = model_prior.copy()

    # Loop over alphas
    for i, alpha in enumerate(alphas):

        # == Step (a) of Emerick & Reynolds, 2013 ==
        # Run the ensemble from time zero.

        # Get data
        if i > 0 or data_prior is None:
            data_prior = forward(model_post)

        # == Step (b) of Emerick & Reynolds, 2013 ==
        # For each ensemble member, perturb the observation vector using
        # d_uc = d_obs + sqrt(α_i) * C_D^0.5 z_d; z_d ~ N(0, I_N_d)

        zd = rng.normal(size=(ne, nt))
        data_pert = data_obs + np.sqrt(alpha) * sigma * zd

        # == Step (c) of Emerick & Reynolds, 2013 ==
        # Update the ensemble using Eq. (3) with C_D replaced by α_i * C_D

        # Center ensemble parameters and data around their means
        cmodel = model_post - model_post.mean(axis=0)
        cdata = data_prior - data_prior.mean(axis=0)

        # Assemble the matrices
        CMD = np.transpose(cmodel, [1, 2, 0]) @ cdata
        CDD = cdata.T @ cdata
        CD = np.diag(alpha * (ne - 1) * sigma**2)
        # TODO: WHY « * (ne - 1) » ? => Co-Variance normalizing!!!!

        # Compute inverse of C
        # Notes:
        # - C is a real-symmetric positive-definite matrix.
        # - Maybe use subspace inversions with Woodbury matrix identity.
        # - Or potentially use Moore-Penrose via:
        #   np.linalg.pinv, sp.linalg.pinv, spp.linalg.pinvh
        Cinv = np.linalg.inv(CDD + CD)

        # Calculate the Kalman gain
        K = CMD@Cinv

        # Update the ensemble parameters
        model_post += np.transpose(K @ (data_pert - data_prior).T, [2, 0, 1])

        # Apply model parameter bounds.
        if vmin or vmax:
            model_post = np.clip(model_post, vmin, vmax)

    # Return posterior model and corresponding data
    return model_post, forward(model_post)
