import numpy as np
from darts.engines import timer_node
#from cpg_tools import make_full_cube
from tools import print_range_array
from datetime import datetime

# arg: t - 1D array of 6 values in Voight notation
# returns (3x3) tensor
def get_tensor_from_voight(t):
    res = np.zeros((3,3))
    res[0,0] = t[0]
    res[1,1] = t[1]
    res[2,2] = t[2]
    res[1,2] = res[2,1] = t[3] #symmetric
    res[0,2] = res[2,0] = t[4]
    res[0,1] = res[1,0] = t[5]
    return res

# compute derivative in point x using finite difference - central schema
# u_minus = u(x - step)
# u_plus  = u(x + step)
def deriv(u_minus, u_plus, step):
    return 0.5 * (u_plus - u_minus) / step

class geomech():
    def __init__(self):
        # elastic constants
        self.poisson = 0.25
        self.young = 3300.
        self.biot = 1.

        # thermal expansion coefficient
        self.thermal_expansion = 1.3e-5  # 1/°C
        #self.thermal_exp_coeff =   # 1/K
        # water 21.0e-5# 1/°C

        # Mohr-Coulomb
        self.cohesion = 0  # assume no cohesion due to healing
        self.friction = 0.7

        # set initial stress
        self.sigma_v = 45  # MPa  # z
        self.sigma_H = 35  # MPa  # y
        self.sigma_h = 35  # MPa  # x

        # to get fault slip
        #self.thermal_expansion = 1.3e-3  # 1/°C
        #self.sigma_v = 175  # MPa  # z
        #self.friction = 0.25

        # diagonal, (Compressive stresses are positives)
        self.stress_init = np.zeros((3,3))
        self.stress_init[0,0] = self.sigma_h
        self.stress_init[1,1] = self.sigma_H
        self.stress_init[2,2] = self.sigma_v

        self.slip_points = []

        self.compaction_cpp = True  # use c++ library to compute displacements
        #self.compaction_cpp = False

    def calc_displacements(self, points, prisms, delta_pressure, delta_temperature):
        '''
        arg: points: points where to compute
        arg: prisms, cells geometry
        arg: delta_pressure, in MPa
        return: displacements at points, in meters
        '''
        import compaction as cpt
        #import compaction_obl as cpt
        # import compaction_par as cpt  #parallel version
        ux = cpt.displacement_x_component(points, prisms, delta_pressure, self.poisson, self.young, delta_temperature, self.thermal_expansion)
        uy = cpt.displacement_y_component(points, prisms, delta_pressure, self.poisson, self.young, delta_temperature, self.thermal_expansion)
        uz = cpt.displacement_z_component(points, prisms, delta_pressure, self.poisson, self.young, delta_temperature, self.thermal_expansion)
        return ux, uy, uz

    def calc_displacements_cpp(self, points, prisms, delta_pressure, delta_temperature):
        from _proxygeomech import compute_geomech
        from _proxygeomech import value_vector as value_vector_geomech
        from _proxygeomech import index_vector as index_vector_geomech
        v_points = value_vector_geomech(points.transpose().flatten())
        v_prisms = value_vector_geomech(prisms.flatten())
        v_delta_pressure = value_vector_geomech(delta_pressure)
        v_delta_temperature = value_vector_geomech(delta_temperature)
        print('   in calc_displacements_cpp', points.size/3, 'eval points', prisms.size/6, 'cells')
        res = compute_geomech(v_points, v_prisms, v_delta_pressure, self.poisson, self.young, v_delta_temperature, self.thermal_expansion)
        print('   compute_geomech done!')
        ux_p = np.array(res['ux_p'], copy=True)
        uy_p = np.array(res['uy_p'], copy=True)
        uz_p = np.array(res['uz_p'], copy=True)
        ux_t = np.array(res['ux_t'], copy=True)
        uy_t = np.array(res['uy_t'], copy=True)
        uz_t = np.array(res['uz_t'], copy=True)
        return ux_p, uy_p, uz_p, ux_t, uy_t, uz_t


    # calculate strain and stress tensors from displacements on fault_surface
    def calc_strain_stress(self, fault_surface, prisms, delta_pressure, delta_temperature):
        # compute displacement derivatives
        step_x = step_y = step_z = 10  # step for derivatives, m.

        fault_surface_y_plus = fault_surface.copy()
        fault_surface_y_plus[0, :] += step_y
        fault_surface_y_minus = fault_surface.copy()
        fault_surface_y_minus[0, :] -= step_y
        fault_surface_x_plus = fault_surface.copy()
        fault_surface_x_plus[1, :] += step_x
        fault_surface_x_minus = fault_surface.copy()
        fault_surface_x_minus[1, :] -= step_x
        fault_surface_z_plus = fault_surface.copy()
        fault_surface_z_plus[2, :] += step_z
        fault_surface_z_minus = fault_surface.copy()
        fault_surface_z_minus[2, :] -= step_z

        # ux, uy, uz = get_displacements_on_surface(fault_surface)

        ux_y_plus,  uy_y_plus,  uz_y_plus  = self.calc_displacements(fault_surface_y_plus,  prisms, delta_pressure, delta_temperature)
        ux_y_minus, uy_y_minus, uz_y_minus = self.calc_displacements(fault_surface_y_minus, prisms, delta_pressure, delta_temperature)

        ux_x_plus,  uy_x_plus,  uz_x_plus  = self.calc_displacements(fault_surface_x_plus,  prisms, delta_pressure, delta_temperature)
        ux_x_minus, uy_x_minus, uz_x_minus = self.calc_displacements(fault_surface_x_minus, prisms, delta_pressure, delta_temperature)

        ux_z_plus,  uy_z_plus,  uz_z_plus  = self.calc_displacements(fault_surface_z_plus,  prisms, delta_pressure, delta_temperature)
        ux_z_minus, uy_z_minus, uz_z_minus = self.calc_displacements(fault_surface_z_minus, prisms, delta_pressure, delta_temperature)

        dux_dx = deriv(ux_x_minus, ux_x_plus, step_x)
        dux_dy = deriv(ux_y_minus, ux_y_plus, step_y)
        dux_dz = deriv(ux_z_minus, ux_z_plus, step_z)

        duy_dx = deriv(uy_x_minus, uy_x_plus, step_x)
        duy_dy = deriv(uy_y_minus, uy_y_plus, step_y)
        duy_dz = deriv(uy_z_minus, uy_z_plus, step_z)

        duz_dx = deriv(uz_x_minus, uz_x_plus, step_x)
        duz_dy = deriv(uz_y_minus, uz_y_plus, step_y)
        duz_dz = deriv(uz_z_minus, uz_z_plus, step_z)

        # Voight notation: 6 values for each fault point: 3 diagonal (xx, yy, zz) + 3 off-diagonal values (yz, xz, xy)
        strain = np.vstack(
            [dux_dx, duy_dy, duz_dz,
             0.5 * (duy_dz + duz_dy),
             0.5 * (dux_dz + duz_dx),
             0.5 * (dux_dy + duy_dx)])

        # volumetric_strain=div(displ)
        volumetric_strain = dux_dx + duy_dy + duz_dz

        # Voight notation: 3 diagonal and 3 off-diagonal values
        n_points = fault_surface.shape[1]
        kronecker = np.array(n_points * [1, 1, 1, 0, 0, 0]).reshape(n_points, 6).transpose()

        #print('strain', strain.shape)
        #print('kronecker', kronecker.shape)

        stress = self.young * (strain + self.poisson / (1 - 2 * self.poisson) *
                               volumetric_strain * kronecker) / (1 + self.poisson)
        return stress, strain

    def calc_strain_stress_cpp(self, fault_surface, prisms, delta_pressure, delta_temperature):
        # compute displacement derivatives
        step_x = step_y = step_z = 1  # step for derivatives, m.

        fault_surface_y_plus = fault_surface.copy()
        fault_surface_y_plus[0, :] += step_y
        fault_surface_y_minus = fault_surface.copy()
        fault_surface_y_minus[0, :] -= step_y
        fault_surface_x_plus = fault_surface.copy()
        fault_surface_x_plus[1, :] += step_x
        fault_surface_x_minus = fault_surface.copy()
        fault_surface_x_minus[1, :] -= step_x
        fault_surface_z_plus = fault_surface.copy()
        fault_surface_z_plus[2, :] += step_z
        fault_surface_z_minus = fault_surface.copy()
        fault_surface_z_minus[2, :] -= step_z

        # ux, uy, uz = get_displacements_on_surface(fault_surface)
        m = 3 # 0 - poro, 1 - thermo, 2 - total
        ux_y_plus  = [0]*m; uy_y_plus  = [0]*m; uz_y_plus  = [0]*m;
        ux_y_minus = [0]*m; uy_y_minus = [0]*m; uz_y_minus = [0]*m;
        ux_x_plus  = [0]*m; uy_x_plus  = [0]*m; uz_x_plus  = [0]*m;
        ux_x_minus = [0]*m; uy_x_minus = [0]*m; uz_x_minus = [0]*m;
        ux_z_plus  = [0]*m; uy_z_plus  = [0]*m; uz_z_plus  = [0]*m;
        ux_z_minus = [0]*m; uy_z_minus = [0]*m; uz_z_minus = [0]*m;

        # 0 - poro, 1 - thermo, 2 - total
        t1 = datetime.now()
        print('calc dU/dy..', t1)
        ux_y_plus[0],  uy_y_plus[0],  uz_y_plus[0], ux_y_plus[1], uy_y_plus[1],  uz_y_plus[1] = \
            self.calc_displacements_cpp(fault_surface_y_plus,  prisms, delta_pressure, delta_temperature)
        ux_y_minus[0], uy_y_minus[0], uz_y_minus[0], ux_y_minus[1], uy_y_minus[1], uz_y_minus[1] = \
            self.calc_displacements_cpp(fault_surface_y_minus, prisms, delta_pressure, delta_temperature)

        t1 = datetime.now()
        print('calc dU/dx..', t1)
        ux_x_plus[0],  uy_x_plus[0],  uz_x_plus[0], ux_x_plus[1],  uy_x_plus[1],  uz_x_plus[1]  = \
            self.calc_displacements_cpp(fault_surface_x_plus,  prisms, delta_pressure, delta_temperature)
        ux_x_minus[0], uy_x_minus[0], uz_x_minus[0], ux_x_minus[1], uy_x_minus[1], uz_x_minus[1] = \
            self.calc_displacements_cpp(fault_surface_x_minus, prisms, delta_pressure, delta_temperature)

        t1 = datetime.now()
        print('calc dU/dz..', t1)
        ux_z_plus[0],  uy_z_plus[0],  uz_z_plus[0], ux_z_plus[1],  uy_z_plus[1],  uz_z_plus[1]  = \
            self.calc_displacements_cpp(fault_surface_z_plus,  prisms, delta_pressure, delta_temperature)
        ux_z_minus[0], uy_z_minus[0], uz_z_minus[0], ux_z_minus[1], uy_z_minus[1], uz_z_minus[1] = \
            self.calc_displacements_cpp(fault_surface_z_minus, prisms, delta_pressure, delta_temperature)

        # total displacement (poro + thermo)
        ux_x_plus[2] = ux_x_plus[0] + ux_x_plus[1]
        uy_x_plus[2] = uy_x_plus[0] + uy_x_plus[1]
        uz_x_plus[2] = uz_x_plus[0] + uz_x_plus[1]
        ux_x_minus[2] = ux_x_minus[0] + ux_x_minus[1]
        uy_x_minus[2] = uy_x_minus[0] + uy_x_minus[1]
        uz_x_minus[2] = uz_x_minus[0] + uz_x_minus[1]

        ux_y_plus[2] = ux_y_plus[0] + ux_y_plus[1]
        uy_y_plus[2] = uy_y_plus[0] + uy_y_plus[1]
        uz_y_plus[2] = uz_y_plus[0] + uz_y_plus[1]
        ux_y_minus[2] = ux_y_minus[0] + ux_y_minus[1]
        uy_y_minus[2] = uy_y_minus[0] + uy_y_minus[1]
        uz_y_minus[2] = uz_y_minus[0] + uz_y_minus[1]

        ux_z_plus[2] = ux_z_plus[0] + ux_z_plus[1]
        uy_z_plus[2] = uy_z_plus[0] + uy_z_plus[1]
        uz_z_plus[2] = uz_z_plus[0] + uz_z_plus[1]
        ux_z_minus[2] = ux_z_minus[0] + ux_z_minus[1]
        uy_z_minus[2] = uy_z_minus[0] + uy_z_minus[1]
        uz_z_minus[2] = uz_z_minus[0] + uz_z_minus[1]

        t1 = datetime.now()
        print('calc deriv...', t1)
        for ui in [0,1,2]:
            ux_x_minus_ = ux_x_minus[ui]; ux_y_minus_ = ux_y_minus[ui]; ux_z_minus_ = ux_z_minus[ui];
            ux_x_plus_  = ux_x_plus[ui];  ux_y_plus_  = ux_y_plus[ui];  ux_z_plus_  = ux_z_plus[ui];
            uy_x_minus_ = uy_x_minus[ui]; uy_y_minus_ = uy_y_minus[ui]; uy_z_minus_ = uy_z_minus[ui];
            uy_x_plus_  = uy_x_plus[ui];  uy_y_plus_  = uy_y_plus[ui];  uy_z_plus_  = uy_z_plus[ui];
            uz_x_minus_ = uz_x_minus[ui]; uz_y_minus_ = uz_y_minus[ui]; uz_z_minus_ = uz_z_minus[ui];
            uz_x_plus_  = uz_x_plus[ui];  uz_y_plus_  = uz_y_plus[ui];  uz_z_plus_  = uz_z_plus[ui];

            dux_dx = deriv(ux_x_minus_, ux_x_plus_, step_x)
            dux_dy = deriv(ux_y_minus_, ux_y_plus_, step_y)
            dux_dz = deriv(ux_z_minus_, ux_z_plus_, step_z)

            duy_dx = deriv(uy_x_minus_, uy_x_plus_, step_x)
            duy_dy = deriv(uy_y_minus_, uy_y_plus_, step_y)
            duy_dz = deriv(uy_z_minus_, uy_z_plus_, step_z)

            duz_dx = deriv(uz_x_minus_, uz_x_plus_, step_x)
            duz_dy = deriv(uz_y_minus_, uz_y_plus_, step_y)
            duz_dz = deriv(uz_z_minus_, uz_z_plus_, step_z)

            # Voight notation: 6 values for each fault point: 3 diagonal (xx, yy, zz) + 3 off-diagonal values (yz, xz, xy)
            strain = np.vstack(
                [dux_dx, duy_dy, duz_dz,
                 0.5 * (duy_dz + duz_dy),
                 0.5 * (dux_dz + duz_dx),
                 0.5 * (dux_dy + duy_dx)])

            # volumetric_strain=div(displ)
            volumetric_strain = dux_dx + duy_dy + duz_dz

            # Voight notation: 3 diagonal and 3 off-diagonal values
            n_points = fault_surface.shape[1]
            kronecker = np.array(n_points * [1, 1, 1, 0, 0, 0]).reshape(n_points, 6).transpose()

            #print('strain', strain.shape)
            #print('kronecker', kronecker.shape)

            stress = self.young * (strain + self.poisson / (1 - 2 * self.poisson) *
                                   volumetric_strain * kronecker) / (1 + self.poisson)
            if ui == 0:
                stress_p, strain_p = stress.copy(), strain.copy()
            elif ui == 1:
                stress_t, strain_t = stress.copy(), strain.copy()
            else:
                stress_total, strain_total = stress.copy(), strain.copy()

        return stress_p, strain_p, stress_t, strain_t, stress_total, strain_total

    def get_stress_on_fault(self, stress_tensor, fault):
        stress_n = stress_tensor @ fault.normal @ fault.normal
        stress_t = np.linalg.norm(
            (np.eye(3, 3) - np.tensordot(fault.normal, fault.normal_T, axes=0))
            @ stress_tensor @ fault.normal)
        return [stress_n, stress_t]

    def mohr_coulomb(self, fault, stress_total_fault, pore_pressure_fault):
        stress_t = np.zeros(fault.n_points)  # tangential vector for each fault point
        stress_n = np.zeros(fault.n_points)  # normal vector for each fault point
        for i in range(fault.n_points):
            stress_total_fault_tensor = get_tensor_from_voight(stress_total_fault[:, i])
            stress_n[i], stress_t[i] = self.get_stress_on_fault(stress_total_fault_tensor, fault)

        stress_n_ini, stress_t_ini = self.get_stress_on_fault(self.stress_init, fault)
        stress_n[:] += stress_n_ini
        stress_t[:] += stress_t_ini

        #stress_n is positive
        total_eff_stress = stress_n - pore_pressure_fault * self.biot
        mcc_n_part = self.cohesion + self.friction * total_eff_stress
        mcc = np.abs(stress_t) - mcc_n_part

        if False:
            print ('Values on fault:')
            print ('mcc min =', mcc.min(), 'max =', mcc.max())
            print ('shear stress max =', np.abs(stress_t).max())
            print ('stress_n max =', stress_n.max())
            print ('pore_pressure max =', pore_pressure_fault.max())
            print ('C+m(sigma_n-p) max =', mcc_n_part.max())
        return stress_n, stress_t, total_eff_stress, mcc_n_part, mcc

    def stat(self):
        self.slip_points = np.array(self.slip_points)
        import pickle
        pkl_file = open('geomech_stat.pkl', 'wb')
        dpkl_data = {'slip_points': self.slip_points}
        pickle.dump(dpkl_data, pkl_file)
        pkl_file.close()

class fault():
    def __init__(self, cell_centers, surface):
        '''
        initialize fault by points in surface
        normal vector
        cell_correspondence
        '''
        self.surface = surface.copy()
        self.n_points = self.surface.shape[1]

        # use first 2 points and 1 last of 'surface' points to calc normal vector
        y0, y1, y2 = list(surface[:3,0])
        x0, x1, x2 = list(surface[:3,1])
        z0, z1, z2 = list(surface[:3,-1])
        # normal vector to vertical fault
        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]  # first vector
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]  # sec vector
        cross = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]  # cross product
        self.normal = np.array(cross)
        self.normal /= np.linalg.norm(self.normal)
        self.normal_T = np.transpose(self.normal)
        # fill  fault_and_array_correspondence - closest cell indices in 1d global array (like pressure), corresponding to fault points
        self.cell_correspondence = np.zeros(self.n_points, dtype=np.int32)
        for i in range(self.n_points):
            distance = np.sqrt((cell_centers[0,:] - self.surface[0, i]) ** 2 +
                               (cell_centers[1,:] - self.surface[1, i]) ** 2 +
                               (cell_centers[2,:] - self.surface[2, i]) ** 2)
            # print(i, distance.argmin(), distance.min())
            self.cell_correspondence[i] = distance.argmin()
        self.actnum = np.zeros(cell_centers.shape[1]) #TODO use n_total_cells here
        self.actnum[self.cell_correspondence] = 1

    def init_YZ_plane(self, centers):
        '''
        define fault geometry (vertical plane x=middle_x)
        '''
        eps = 1e-2 # center of cell has instability of analytic solution, so add the epsilon

        self.dims = (nz, ny)
        self.fault_axis_labels = ['Y', 'Z']

        y_min = centers[0,:].min()
        y_max = centers[0,:].max()
        x_min = centers[1,:].min()
        x_max = centers[1,:].max()
        z_min = centers[2,:].min()
        z_max = centers[2,:].max()

        fault_z = np.linspace(z_min, z_max, self.dims[0]) #TODO avoid center of cell
        fault_y = np.linspace(y_min, y_max, self.dims[1]) #TODO avoid center of cell
        fault_x = 0.5*(x_max + x_min) + eps  # middle point

        #fault_z[:] = centers[0,2] # for testing distance

        fault_y2, fault_z2 = np.meshgrid(fault_y, fault_z)

        fault_z1 = fault_z2.ravel()
        fault_y1 = fault_y2.ravel()
        fault_x1 = np.zeros(fault_y1.shape) + fault_x
        self.surface = np.vstack([fault_y1, fault_x1, fault_z1])


def calc_geomech(m, mode='surface', compute_stress=False, compute_displs=True, thermal=True):
    if not compute_displs:  # just return array names for tsteps where geomech is not calculated
        u = None  # dummy
        arrays = {'Up_x': u, 'Up_y': u, 'Up_z': u,
                  'Ut_x': u, 'Ut_y': u, 'Ut_z': u,
                  'U_x' : u, 'U_y' : u, 'U_z' : u}
        arrays_stress = {'Sp_xx': u, 'Sp_yy': u, 'Sp_zz': u, 'Sp_yz': u, 'Sp_xz': u, 'Sp_xy': u,
                         'St_xx': u, 'St_yy': u, 'St_zz': u, 'St_yz': u, 'St_xz': u, 'St_xy': u,
                         'S_xx' : u, 'S_yy' : u, 'S_zz' : u, 'S_yz' : u, 'S_xz' : u, 'S_xy' : u}
        arrays_delta = {'dp': u, 'dt': u}
        arrays_fault = {'F_p': u, 'F_p_bar': u, 'F_t': u,  'F_y': u,  'F_z': u,
                        'F_stress_normal': u, 'F_stress_tangen': u,
                        'F_total_eff_stress': u, 'F_mcc_n_part': u, 'F_mcc': u}

        arrays_fault.update({'F_stress_normal_p': u, 'F_stress_tangen_p': u,
                        'F_total_eff_stress_p': u, 'F_mcc_n_part_p': u, 'F_mcc_p': u})
        arrays_fault.update({'F_stress_normal_t': u, 'F_stress_tangen_t': u,
                        'F_total_eff_stress_t': u, 'F_mcc_n_part_t': u, 'F_mcc_t': u})
        arrays.update(arrays_stress)
        arrays.update(arrays_delta)
        arrays.update(arrays_fault)
        return arrays

    m.timer.node["geomech_init"] = timer_node()
    m.timer.node["geomech_init"].start()
    t1 = datetime.now()
    print('geomech init..', t1)

    m.reservoir.geomech_init_geometry(mode=mode)

    P = m.get_pressure()
    m.reservoir.update_delta_pressure(P)

    T = m.get_temperature()
    m.reservoir.update_delta_temperature(T)

    eval_points = m.reservoir.get_eval_points()

    print('ok! geomech_init Time=', (datetime.now() - t1).total_seconds()/60, 'min.')
    m.timer.node["geomech_init"].stop()

    t1 = datetime.now()
    print('calc_displs..', t1)
    m.timer.node["displs"] = timer_node()
    m.timer.node["displs"].start()
    arrays = m.reservoir.calc_displs(eval_points, thermal)
    print('ok! displs Time=', (datetime.now() - t1).total_seconds()/60, 'min.')
    m.timer.node["displs"].stop()

    if compute_stress:
        m.timer.node["stress"] = timer_node()
        m.timer.node["stress"].start()

        if not m.reservoir.geomech.compaction_cpp:
            t1 = datetime.now()
            print('calc poroelastic delta stress..', t1)
            stress_p, strain_p = m.reservoir.geomech.calc_strain_stress(eval_points,
                                                                    m.reservoir.prisms,
                                                                    m.reservoir.delta_pressure,
                                                                    np.array([]))
            [Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
            print('ok! stress_p Time=', (datetime.now() - t1).total_seconds()/60, 'min.')
            t1 = datetime.now()
            print('calc thermoelastic delta stress..', t1)
            stress_t, strain_t = m.reservoir.geomech.calc_strain_stress(eval_points,
                                                                    m.reservoir.prisms,
                                                                    np.array([]),
                                                                    m.reservoir.delta_temperature)
            [St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
            [S_xx, S_yy, S_zz, S_yz, S_xz, S_xy] = [None]*6  #TODO
            print('ok! stress_t Time=', (datetime.now() - t1).total_seconds()/60, 'min.')
        else:
            t1 = datetime.now()
            print('calc delta stress..', t1)
            res = m.reservoir.geomech.calc_strain_stress_cpp(eval_points,
                                                                    m.reservoir.prisms,
                                                                    m.reservoir.delta_pressure,
                                                                    m.reservoir.delta_temperature)
            stress_p, strain_p, stress_t, strain_t, stress, strain = res
            [Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
            [St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
            [S_xx, S_yy, S_zz, S_yz, S_xz, S_xy] = stress
            print('ok! stress_s_t_cpp Time=', (datetime.now() - t1).total_seconds()/60, 'min.')
        m.timer.node["stress"].stop()

        arrays_stress = {'Sp_xx': Sp_xx, 'Sp_yy': Sp_yy, 'Sp_zz': Sp_zz, 'Sp_yz': Sp_yz, 'Sp_xz': Sp_xz, 'Sp_xy': Sp_xy,
                         'St_xx': St_xx, 'St_yy': St_yy, 'St_zz': St_zz, 'St_yz': St_yz, 'St_xz': St_xz, 'St_xy': St_xy,
                         'S_xx': S_xx, 'S_yy': S_yy, 'S_zz': S_zz, 'S_yz': S_yz, 'S_xz': S_xz, 'S_xy': S_xy}
        arrays.update(arrays_stress)

        print_range_array(arrays, 'U_x', 'm')
        print_range_array(arrays, 'U_y', 'm')
        print_range_array(arrays, 'U_z', 'm')

        print_range_array(arrays_stress, 'Sp_xx', 'MPa')
        print_range_array(arrays_stress, 'St_xx', 'MPa')
        print_range_array(arrays_stress, 'S_xx', 'MPa')

        print_range_array(arrays_stress, 'Sp_xz', 'MPa')
        print_range_array(arrays_stress, 'St_xz', 'MPa')
        print_range_array(arrays_stress, 'S_xz', 'MPa')

        fault = m.reservoir.fault
        if m.reservoir.fault is not None:
            #if 'coulomb' in m.timer.node
            m.timer.node["coulomb"] = timer_node() #TODO to not create each time
            m.timer.node["coulomb"].start()
            pore_pressure_fault = P[fault.cell_correspondence[:]] * 0.1 # to MPa
            pore_pressure_initial_fault = m.reservoir.pressure_initial[fault.cell_correspondence[:]] * 0.1 # to MPa
            temperature_fault = T[fault.cell_correspondence[:]]

            mc_res_p = m.reservoir.geomech.mohr_coulomb(fault, stress_p, pore_pressure_fault)
            mc_res_t = m.reservoir.geomech.mohr_coulomb(fault, stress_t, pore_pressure_initial_fault)
            mc_res = m.reservoir.geomech.mohr_coulomb(fault, stress, pore_pressure_fault)

            stress_n_p, stress_t_p, total_eff_stress_p, mcc_n_part_p, mcc_p = mc_res_p
            stress_n_t, stress_t_t, total_eff_stress_t, mcc_n_part_t, mcc_t = mc_res_t
            stress_n, stress_t, total_eff_stress, mcc_n_part, mcc = mc_res

            arrays_fault_p = {'F_stress_normal_p': stress_n_p, 'F_stress_tangen_p':stress_t_p,
                              'F_total_eff_stress_p':total_eff_stress_p,
                              'F_mcc_n_part_p':mcc_n_part_p, 'F_mcc_p':mcc_p}
            arrays_fault_t = {'F_stress_normal_t': stress_n_t, 'F_stress_tangen_t':stress_t_t,
                              'F_total_eff_stress_t':total_eff_stress_t,
                              'F_mcc_n_part_t':mcc_n_part_t, 'F_mcc_t':mcc_t}
            arrays_fault_total = {'F_stress_normal': stress_n, 'F_stress_tangen':stress_t,
                              'F_total_eff_stress':total_eff_stress,
                              'F_mcc_n_part':mcc_n_part, 'F_mcc':mcc}

            arrays.update(arrays_fault_p)
            arrays.update(arrays_fault_t)
            arrays.update(arrays_fault_total)

            F_y = m.reservoir.centers_3d[0, fault.cell_correspondence[:]]
            F_z = m.reservoir.centers_3d[2, fault.cell_correspondence[:]]

            arrays_fault = {'F_y': F_y, 'F_z': F_z,
                            'F_p': pore_pressure_fault,   # MPa
                            'F_p_bar': pore_pressure_fault * 10,  # to bar
                            'F_t': temperature_fault,
                            'F_stress_normal': stress_n, 'F_stress_tangen':stress_t,
                            'F_total_eff_stress':total_eff_stress, 'F_mcc_n_part':mcc_n_part, 'F_mcc':mcc}

            arrays.update(arrays_fault)

            print_range_array({'pore_pressure_fault':pore_pressure_fault}, 'pore_pressure_fault', 'MPa')
            print_range_array({'temperature_fault':temperature_fault}, 'temperature_fault', 'degrees')

            print_range_array(arrays_fault, 'F_stress_normal', 'MPa')
            print_range_array(arrays_fault, 'F_mcc_n_part', 'MPa')
            print_range_array(arrays_fault, 'F_stress_tangen', 'MPa')
            print_range_array(arrays_fault, 'F_total_eff_stress', 'MPa')
            print_range_array(arrays_fault, 'F_mcc', 'MPa')

            mcc = arrays_fault['F_mcc']
            if mcc.max() > 0:
                print("Slip", len(mcc[mcc > 0]), 'of', len(mcc), 'points')

            m.reservoir.geomech.slip_points.append(len(mcc[mcc > 0]))

            m.timer.node["coulomb"].stop()

            if False:#mode != 'fault':
                for name in arrays.keys():
                    if arrays[name] is None:
                        continue
                    arrays[name] = make_full_cube(arrays[name], fault.actnum, val=arrays[name].min())

        else:
            u = None
            arrays_fault = {'F_p': u, 'F_p_bar': u, 'F_t': u,  'F_y': u,  'F_z': u,
                            'F_stress_normal': u, 'F_stress_tangen': u,
                            'F_total_eff_stress': u, 'F_mcc_n_part': u, 'F_mcc': u}
            arrays.update(arrays_fault)

    arrays_delta = {'dp': m.reservoir.delta_pressure * 10,  # to MPa
                    'dt': m.reservoir.delta_temperature}
    arrays.update(arrays_delta)

    print_range_array(arrays_delta, 'dp', 'bar')
    print_range_array(arrays_delta, 'dt', 'degrees')

    arrays_cube = m.reservoir.extend_eval_points_to_cube(arrays, mode=mode)

    #TODO make_full_cube
    #arrays.update(arrays_delta)

    return arrays_cube #arrays
