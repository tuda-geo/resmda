#%%
import numpy as np
import matplotlib.pyplot as plt

def progress_bar(iteration, total, bar_length=50):
    percent = float(iteration) / float(total)
    arrow = '=' * int(round(percent * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f'\r[{arrow + spaces}] {percent * 100:.2f}%')
    sys.stdout.flush()

class ReservoirSim:
    def __init__(self, nx, ny, perm_field=None):
        # Initialize simulation parameters
        self.nx = nx
        self.ny = ny
        self.nb = nx * ny
        self.perm_field = perm_field if perm_field is not None else np.ones(self.nb) * 1000
        
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
        r0 =  0.208 * self.dx
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
    

        K = np.zeros((n,n))
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
    
    def simulate(self):
        P = np.ones(self.nb) * self.pres_ini
        dt = 0.0001
        pressure_history = []
        for t in range(10):
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
        
        z = np.reshape(P, [self.ny, self.nx])
        
        #Visualization
        # plt.imshow(z)
        # plt.colorbar()
        # plt.show()
        return np.array(pressure_history)
    
    # def plot_pressure_grid(self, pressure_history, steps_per_row=3):
    #     num_steps = len(pressure_history)
    #     num_rows = (num_steps + steps_per_row - 1) // steps_per_row

    #     # Find global minimum and maximum pressure for color scaling
    #     global_min = np.min([np.min(p) for p in pressure_history])
    #     global_max = np.max([np.max(p) for p in pressure_history])

    #     fig, axes = plt.subplots(num_rows, steps_per_row, figsize=(15, num_rows*5))
        
    #     # Flatten the axes array for easier indexing
    #     axes = axes.flatten()

    #     for i in range(num_steps):
    #         ax = axes[i]
    #         z = np.reshape(pressure_history[i], [self.ny, self.nx])
    #         im = ax.imshow(z, cmap='viridis', vmin=global_min, vmax=global_max)  # Fixed color limits
    #         ax.set_title(f"Time step {i+1}")

    #     # Hide any remaining empty subplots
    #     for i in range(num_steps, num_rows * steps_per_row):
    #         axes[i].axis('off')

    #     # Adjust layout to make room for colorbar
    #     plt.subplots_adjust(right=0.8)

    #     # Add a colorbar to the figure at the right side
    #     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig.colorbar(im, cax=cbar_ax)

    #     plt.show()

    
 
    
# perm_field = np.ones(25 * 25) * 1000  # Replace with your actual perm field
# reservoir = ReservoirSim(perm_field=perm_field)
# pressure_history = reservoir.simulate()
# reservoir.plot_middle_pressure(pressure_history)