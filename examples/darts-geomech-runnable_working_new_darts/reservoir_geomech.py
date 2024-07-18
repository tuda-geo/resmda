import numpy as np
from geomechanics import geomech, fault
from darts.engines import timer_node
from darts.reservoirs.cpg_reservoir import CPG_Reservoir, make_full_cube

class CPG_Reservoir_proxy (CPG_Reservoir):
    def __init__(self, timer: timer_node, arrays):
        # call base class constructor
        super().__init__(timer, arrays)
        # geomechanics
        self.geomech = geomech()
        self.fault = None

        # if cpg_interp used
        self.struct_grid_refinement_factor_x = 5
        self.struct_grid_refinement_factor_y = 5
        self.struct_grid_refinement_factor_z = 5

        #self.cpg_interp = True
        self.cpg_interp = False

        #self.add_boundary_points = True
        self.add_boundary_points = False

    def geomech_init_geometry(self, mode='surface'):
        self.mode = mode
        eps = 1e-3  # # [m.] to avoid deterioration of analytic solution
        if hasattr(self, 'prisms'):  # do only once
            return
        prisms_cpp = self.discr_mesh.get_prisms()
        prisms_1d = np.array(prisms_cpp, copy=True)
        self.prisms = prisms_1d.reshape(self.discr_mesh.n_cells, 6)

        # prisms and centers defined only for active cells
        centers_cpp = self.discr_mesh.get_centers()
        centers_1d = np.array(centers_cpp, copy=True) + eps
        self.centers_3d = centers_1d.reshape(self.discr_mesh.n_cells, 3).transpose()

        self.init_struct_grid()
        self.add_boundary_prisms()

    def interp_to_struct(self, arr_orig):
        # interpolate 3D array arr_orig defined on the original grid to structured grid
        from scipy.interpolate import griddata as gd
        if self.add_boundary_points:
            arr_orig = np.append(arr_orig, np.zeros(self.zero_delta_x.size))
        return gd((self.orig_x, self.orig_y, self.orig_z), arr_orig, (self.struct_x, self.struct_y, self.struct_z), method='nearest')

    def update_delta_pressure(self, P):
        '''
        should be called each time before computing displacements
        P current pressure array
        '''
        # initial_pressure = 200.
        self.delta_pressure = P - self.pressure_initial[:P.shape[0]]  # bar - bar
        self.delta_pressure *= 0.1  # convert units bar->MPa
        if self.cpg_interp:
            #print('before interp:', self.delta_pressure.min(), self.delta_pressure.max())
            self.delta_pressure = self.interp_to_struct(self.delta_pressure)
            #print('after  interp:', self.delta_pressure.min(), self.delta_pressure.max())

            #from matplotlib import pyplot as plt
            #for zi in range(self.nz_struct):
            #    dp = self.delta_pressure.reshape(self.nz_struct, self.ny_struct, self.nx_struct)[zi, :, :]
            #    plt.imshow(dp)
            #    plt.show()

    def update_delta_temperature(self, T):
        '''
        should be called each time before computing displacements
        T current temperature array
        '''
        if T is None: # isothermal
            self.delta_temperature = np.zeros(self.delta_pressure.shape)
        else:
            self.delta_temperature = T - self.temperature_initial[:T.shape[0]]
        if self.cpg_interp:
            self.delta_temperature = self.interp_to_struct(self.delta_temperature)

    def get_eval_points(self):
        if self.mode[:6] == 'layer_' or self.mode == 'surface':  # calc displs only for the 1-st layer
            nx = self.discr_mesh.nx
            ny = self.discr_mesh.ny
            actnum = np.array(self.discr_mesh.actnum, copy=False)
            if self.mode[:6] == 'layer_':
                layer = int(self.mode.split('_')[1]) # get 'k' from layer_k
            else:
                layer = 1
            self.n_act_cells_start = actnum[:(layer - 1) * nx * ny].sum()
            self.n_act_cells_end = actnum[:layer * nx * ny].sum()
            eval_points = self.centers_3d[:, self.n_act_cells_start:self.n_act_cells_end]
            if self.mode == 'surface':  # at depth = 0
                eval_points[2,:] = 0

            # surface (structured grid for xarray)
            eps = 1e-1
            eval_points = np.zeros((3, self.nx_struct_surface * self.ny_struct_surface))
            eval_points[0,:] = self.eval_y + eps
            eval_points[1,:] = self.eval_x + eps
            eval_points[2,:] = 0  # at depth = 0

        elif self.mode == 'cell_centers':  # at each cell center
            eval_points = self.centers_3d
            self.fault = fault(self.centers_3d, surface=self.centers_3d) # dummy fault_surface at each cell
            fault_angle_degrees = 20
            fault_angle = np.radians(fault_angle_degrees)
            self.fault.normal = np.array(np.array([np.cos(fault_angle), 0, np.sin(fault_angle)]))
            self.fault.normal /= np.linalg.norm(self.fault.normal)
            self.fault.normal_T = np.transpose(self.fault.normal)
            #print('fault_angle=', fault_angle_degrees, 'fault.normal', self.fault.normal)
            #stress_n_ini, stress_t_ini = self.geomech.get_stress_on_fault(self.geomech.stress_init, self.fault)
            #print('stree initial normal=', stress_n_ini, 'tangential=', stress_t_ini)
        elif self.mode == 'fault_nnc':  # at each cell center
            fault_yxz = np.array(self.discr.get_fault_xyz(), copy=True)
            if fault_yxz.size == 0:  # fault not found
                return None
            fault_yxz_pairs = fault_yxz.reshape(fault_yxz.size // 6, 6)
            # cells "right" side (first 3) and "left" side (rest 3)  to the fault
            fault_yxz_middle = (fault_yxz_pairs[:, 3:] + fault_yxz_pairs[:, :3]) * 0.5
            with open('fault_yxz', 'wb') as f:
                np.save(f, fault_yxz_middle, allow_pickle=True)
            fault_surface = fault_yxz_middle.transpose()
            self.fault = fault(self.centers_3d, fault_surface)
            eval_points = self.fault.surface
        elif self.mode == 'fault':  # at each cell center
            c = self.centers_3d.reshape((3, self.discr_mesh.nz, self.discr_mesh.nx, self.discr_mesh.ny))
            if False: # 2 layers at left and right side of the fault
                fault_surface = c[:,:,:,4:6] # layers by Xaxis: 4,5 for case43
                fault_surface = fault_surface.reshape(3, 2 * self.discr_mesh.ny * self.discr_mesh.nz)
            else:
                fault_surface = c[:, :, :, self.discr_mesh.nx//2]  # layers by Xaxis: 5 for case43
                fault_surface = fault_surface.reshape(3, self.discr_mesh.ny * self.discr_mesh.nz)
            #fy = fault_surface[0, :]
            #fx = fault_surface[1, :]
            #fz = fault_surface[2, :]
            #fz = np.unique(np.round(fz, decimals=3))
            self.fault = fault(self.centers_3d, fault_surface)

            #fault_yxz_middle = (fault_surface[:, 0] + fault_surface[:, 1]) * 0.5
            #fault_surface = fault_yxz_middle.transpose()
            #self.fault = fault(self.centers_3d, fault_yxz_middle)
            #self.fault.cell_correspondence = self.fault_2.cell_correspondence
            #self.fault.actnum = self.fault_2.actnum

            eval_points = self.fault.surface
        else:
            print('Error: unknown mode specified!', self.mode)
            eval_points = None
        print('number of evaluation points:', eval_points.size//3)
        return eval_points

    def extend_eval_points_to_cube(self, arrays, mode='surface'):
        if mode == 'surface':
            return arrays
        n_cells_all = self.nx * self.ny * self.nz
        local_to_global = np.array(self.discr_mesh.local_to_global, copy=False)
        global_to_local = np.array(self.discr_mesh.global_to_local, copy=False)
        g2l = global_to_local[:self.nx * self.ny]
        n_active_cells_in_1st_layer = g2l[g2l>=0].size
        l2g = local_to_global[:n_active_cells_in_1st_layer]
        arrays_cube = {}
        if mode[:6] == 'layer_' or mode == 'surface':
            for name in arrays.keys():
                if arrays[name] is None:
                    continue
                # fill the rest displ values with zeros to make full array
                if arrays[name].size == 0:
                    continue
                array_full = make_full_cube(arrays[name], l2g, g2l)
                n_act_cells_rest_layers = n_cells_all - self.n_act_cells_start - array_full.size
                min_val = arrays[name].min()
                arrays_cube[name] = np.hstack([np.zeros(self.n_act_cells_start)+min_val,
                                          array_full,
                                          np.zeros(n_act_cells_rest_layers)+min_val])
            return arrays_cube
        else:
            return arrays

    def calc_displs(self, eval_points, thermal=True):
        '''
        eval_points - where to calc displs
        return map of 9 displ vectors (3 port and 3 thermal and 3 total), values are  in meters
        displs computed at eval_points
        '''

        # print('centers_ptr', centers_ptr.shape)
        # first call for the first 10 cells - just to make numba compilation and speedup the main call
        if not self.geomech.compaction_cpp:
            ux1, uy1, uz1 = self.geomech.calc_displacements(eval_points[:, :10], self.prisms[:10],
                                                            self.delta_pressure[:10],
                                                            np.array([]))
            ux1, uy1, uz1 = self.geomech.calc_displacements(eval_points[:, :10], self.prisms[:10],
                                                            np.array([]),
                                                            self.delta_temperature[:10])

        # save to pickle
        #import pickle
        #pkl_file = open('geomech.pkl', 'wb')
        #dpkl_data = {'center': centers_ptr, 'prism': self.prisms, 'dp': self.delta_pressure}
        #pickle.dump(dpkl_data, pkl_file)
        #pkl_file.close()

        if not self.geomech.compaction_cpp:
            # poroelastic displacements
            upx, upy, upz = self.geomech.calc_displacements(eval_points, self.prisms,
                                                            self.delta_pressure,
                                                            np.array([]))
            # thermoelastic displacements
            if thermal:
                utx, uty, utz = self.geomech.calc_displacements(eval_points, self.prisms,
                                                            np.array([]),
                                                            self.delta_temperature)
            else:
                utx, uty, utz = 0
        else:
            upx, upy, upz, utx, uty, utz = self.geomech.calc_displacements_cpp(eval_points, self.prisms,
                                                            self.delta_pressure,
                                                            self.delta_temperature)
        # compute thermo- poro- elastic displacement
        ux = upx + utx
        uy = upy + uty
        uz = upz + utz

        arrays = {'Up_x': upx, 'Up_y': upy, 'Up_z': upz,
                  'Ut_x': utx, 'Ut_y': uty, 'Ut_z': utz,
                  'U_x': ux, 'U_y': uy, 'U_z': uz}

        return arrays

    def init_struct_grid(self):
        # 1. generate structured grid at surface - for xarray in surface mode and for prisms
        self.nx_struct = self.discr_mesh.nx * self.struct_grid_refinement_factor_x
        self.ny_struct = self.discr_mesh.ny * self.struct_grid_refinement_factor_y
        self.nz_struct = self.discr_mesh.nz * self.struct_grid_refinement_factor_z
        x1 = np.linspace(self.x_min, self.x_max, self.nx_struct + 1)
        y1 = np.linspace(self.y_min, self.y_max, self.ny_struct + 1)
        z1 = np.linspace(self.z_min, self.z_max, self.nz_struct + 1)

        if self.mode == 'surface':
            # use the same resolution as in the original grid
            self.nx_struct_surface = self.discr_mesh.nx
            self.ny_struct_surface = self.discr_mesh.ny
            self.nz_struct_surface = 1

            d = 0 #(self.x_max - self.x_min) * 0.4

            x1_s = np.linspace(self.x_min - d, self.x_max + d, self.nx_struct_surface + 1)
            y1_s = np.linspace(self.y_min - d, self.y_max + d, self.ny_struct_surface + 1)
            z1_s = np.linspace(self.z_min, self.z_max, 2)  # struct grid with one layer
            # for xarray coordinates
            self.xarray_x = (x1_s[1:] + x1_s[:-1]) * 0.5
            self.xarray_y = (y1_s[1:] + y1_s[:-1]) * 0.5
            self.xarray_z = (z1_s[1:] + z1_s[:-1]) * 0.5
            self.xarray_z[:] = 0  # on the surface
            # eval points
            x_, y_, z_ = np.meshgrid(self.xarray_x, self.xarray_y, self.xarray_z)
            self.eval_x = x_.flatten()
            self.eval_y = y_.flatten()
            self.eval_z = z_.flatten()

        if not self.cpg_interp:
            return

        # 2. generate structured grid for sources

        # generate prisms (structured grid)
        prisms = np.zeros((self.nx_struct * self.ny_struct * self.nz_struct, 6))
        for k in range(self.nz_struct):
            start_k = k * self.nx_struct * self.ny_struct
            for j in range(self.ny_struct):
                start = j * self.nx_struct + start_k
                end = (j + 1) * self.nx_struct + start_k
                prisms[start:end, 0] = y1[j]
                prisms[start:end, 1] = y1[j + 1]
                prisms[start:end, 2] = x1[:-1]
                prisms[start:end, 3] = x1[1:]
                prisms[start:end, 5] = z1[k]
                prisms[start:end, 4] = z1[k + 1]



        struct_centers = np.zeros((self.nx_struct * self.ny_struct * self.nz_struct, 3))
        for k in range(self.nz_struct):
            start_k = k * self.nx_struct * self.ny_struct
            for j in range(self.ny_struct):
                start = j * self.nx_struct + start_k
                end = (j + 1) * self.nx_struct + start_k
                struct_centers[start:end, 0] = (prisms[start:end, 0] + prisms[start:end, 1]) * 0.5  # y
                struct_centers[start:end, 1] = (prisms[start:end, 2] + prisms[start:end, 3]) * 0.5  # x
                struct_centers[start:end, 2] = (prisms[start:end, 4] + prisms[start:end, 5]) * 0.5  # z

        self.prisms = prisms
        # struct grid centers - point at which delta_p and delta_t will be computed by interpolation
        self.struct_y, self.struct_x, self.struct_z = struct_centers[:, 0], struct_centers[:, 1], struct_centers[:, 2]
        # points with known values which will be used in interpolation
        self.orig_y, self.orig_x, self.orig_z = self.centers_3d[0,:], self.centers_3d[1,:], self.centers_3d[2,:]

    def add_boundary_prisms(self):
        if not self.cpg_interp:
            return
        if not self.add_boundary_points:
            return
        # add additional virtual points around the original grid shape defined by ACTNUM and also vertical inclination
        # assume there are no cells with ACTNUM=0 inside the reservoir (which is normally is, as model is thermal)
        zero_delta_points = []
        zero_delta_prisms = []
        def add_prism_point(i, j, k, mx, my, mz):
            local_block = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i, j, k)]
            dx, dy, dz = self.discr_mesh.calc_cell_sizes(i, j, k)
            zero_delta_points.append((self.centers_3d[1, local_block] + dx * mx,
                                      self.centers_3d[0, local_block] + dy * my,
                                      self.centers_3d[2, local_block] + dz * mz))
            zero_delta_prisms.append((self.centers_3d[0, local_block] - dy / 2.,
                                              self.centers_3d[0, local_block] + dy / 2.,
                                              self.centers_3d[1, local_block] - dx / 2.,
                                              self.centers_3d[1, local_block] + dx / 2.,
                                              self.centers_3d[2, local_block] + dz / 2.,
                                              self.centers_3d[2, local_block] - dz / 2.,
                                              ))

        actnum3d = self.actnum.reshape(self.nx, self.ny, self.nz, order='F')
        # z-
        for i in range(self.nx):
            for j in range(self.ny):
                k = 0
                while k < self.nz and actnum3d[i, j, k] == 0:  # search first active cell
                    k += 1
                if k < self.nz:
                    add_prism_point(i, j, k, 0, 0, -1)
        # z+
        for i in range(self.nx):
            for j in range(self.ny):
                k = self.nz - 1
                while k >= 0 and actnum3d[i, j, k] == 0:
                    k -= 1
                if k >= 0:
                    add_prism_point(i, j, k, 0, 0, 1)
        # X-
        for k in range(self.nz):
            for j in range(self.ny):
                i = 0
                while i < self.nx and actnum3d[i, j, k] == 0:
                    i += 1
                if i < self.nx:
                    add_prism_point(i, j, k, -1, 0, 0)
        # X+
        for k in range(self.nz):
            for j in range(self.ny):
                i = self.nx - 1
                while i >= 0 and actnum3d[i, j, k] == 0:
                    i -= 1
                if i >= 0:
                    add_prism_point(i, j, k, 1, 0, 0)
        # Y-
        for k in range(self.nz):
            for i in range(self.nx):
                j = 0
                while j < self.ny and actnum3d[i, j, k] == 0:
                    j += 1
                if j < self.ny:
                    add_prism_point(i, j, k, 0, -1, 0)
        # Y+
        for k in range(self.nz):
            for i in range(self.nx):
                j = self.ny - 1
                while j >= 0 and actnum3d[i, j, k] == 0:
                    j -= 1
                if j >= 0:
                    add_prism_point(i, j, k, 0, 1, 0)

        zero_delta_points = np.array(zero_delta_points)
        self.zero_delta_x = zero_delta_points[:, 0]
        self.zero_delta_y = zero_delta_points[:, 1]
        self.zero_delta_z = zero_delta_points[:, 2]

        print('prisms:', self.prisms.shape[0], 'boundary points:', len(zero_delta_prisms))

        # add points with zero p,T difference around the grid to properly handle  grids with a shape defined by ACTNUM
        self.orig_x = np.append(self.orig_x, self.zero_delta_x)
        self.orig_y = np.append(self.orig_y, self.zero_delta_y)
        self.orig_z = np.append(self.orig_z, self.zero_delta_z)