import numpy as np


class ReservoirSim:
    def __init__(self, perm_field):
        # Initialize simulation parameters
        self.nx, self.ny = perm_field.shape
        self.nb = perm_field.size
        self.shape = perm_field.shape
        self.perm_field = perm_field.ravel()

        self.phi = 0.2
        self.c_f = 1e-5
        self.p0 = 1
        self.rho0 = 1
        self.mu_w = 1
        self.rw = 0.15
        self.dx = 50
        self.dy = 50
        self.dz = 10
        self.V = self.dx * self.dy * self.dz

        self.pres_ini = 150
        self.pres_prd = 120
        self.pres_inj = 180

    def compute_beta(self, dens):
        Phi = dens / self.mu_w
        return Phi

    def compute_wi(self, k, mu, dens):
        r0 = 0.208 * self.dx
        wi = 2 * np.pi * k * self.dz / mu / np.log(r0 / self.rw)
        return wi

    def compute_density(self, p):
        dens = self.rho0 * (1 + self.c_f * (p - self.p0))
        return dens

    def get_matrix(self, n, Phi, compr, p):
        nconn = (self.nx - 1) * self.ny + self.nx * (self.ny - 1)
        block_m = np.zeros(nconn, dtype=np.int32)
        block_p = np.zeros(nconn, dtype=np.int32)
        Trans = np.ones(nconn)
        A = self.dx * self.dy

        perm_1d = self.perm_field.flatten()

        for i in range(self.nx - 1):
            for j in range(self.ny):
                k = i + j * (self.nx - 1)
                block_m[k] = i + j * self.nx
                block_p[k] = block_m[k] + 1
                gl = perm_1d[block_m[k]] * A / self.dx
                gr = perm_1d[block_p[k]] * A / self.dx
                Trans[k] = gl * gr / (gl + gr)

        for i in range(self.nx):
            for j in range(self.ny - 1):
                k = (self.nx-1) * self.ny + i + j * self.nx
                block_m[k] = i + j * self.nx
                block_p[k] = block_m[k] + self.nx
                gl = perm_1d[block_m[k]] * A / self.dy
                gr = perm_1d[block_p[k]] * A / self.dy
                Trans[k] = gl * gr / (gl + gr)

        K = np.zeros((n, n))
        f = np.ones(n) * compr * p
        for i in range(n):
            K[i, i] = compr[i]

        for k in range(nconn):
            im = block_m[k]
            ip = block_p[k]
            K[im, im] += Trans[k] * (Phi[im] + Phi[ip]) / 2
            K[im, ip] -= Trans[k] * (Phi[im] + Phi[ip]) / 2
            K[ip, im] -= Trans[k] * (Phi[im] + Phi[ip]) / 2
            K[ip, ip] += Trans[k] * (Phi[im] + Phi[ip]) / 2

        return K, f

    def __call__(self, nt=10, dt=0.0001):
        P = np.ones(self.nb) * self.pres_ini
        pressure_history = [P.copy(), ]
        for t in range(nt):
            dens = self.compute_density(P)
            beta = self.compute_beta(dens)
            wi = self.compute_wi(self.perm_field, self.mu_w, dens)
            compr = np.ones(self.nb) * self.V * self.phi * self.c_f / dt

            K, f = self.get_matrix(self.nb, beta, compr, P)

            f[0] += self.pres_inj * wi[0]
            K[0, 0] += wi[0]

            f[-1] += self.pres_prd * wi[-1]
            K[-1, -1] += wi[-1]

            P = np.linalg.solve(K, f)
            pressure_history.append(P.copy())

        return np.array(pressure_history).reshape((nt+1, *self.shape))
