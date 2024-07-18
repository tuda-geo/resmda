import numpy as np
import os
import time

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, make_full_cube
from reservoir_geomech import CPG_Reservoir_proxy
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
from darts.discretizer import elem_loc

from darts.models.darts_model import DartsModel
from darts.tools.keyword_file_tools import save_few_keywords
from darts.engines import sim_params, timer_node

from darts.tools.keyword_file_tools import save_few_keywords

class BaseModel(DartsModel):
    def __init__(self, arrays, geomech_mode='none', n_points=1000):

        # call base class constructor
        super().__init__()

        self.n_points = n_points

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        start = time.perf_counter()

        if geomech_mode != 'none':
            self.reservoir = CPG_Reservoir_proxy(self.timer, arrays)
        else:
            self.reservoir = CPG_Reservoir(self.timer, arrays)

        self.reservoir.discretize()

        #self.reservoir.optimized_vtk_export = False  # turn off experimental feature
        end = time.perf_counter()
        print('cpg grid initialization time :', end - start, 'sec.')

        # simulator makes cells with low porosity inactive. For thermal simulation we need to prevent that
        self.poro = np.array(self.reservoir.mesh.poro, copy=False)
        self.poro[self.poro < 1e-5] = 1e-5

        #self.reservoir.set_initial_pressure_from_file(gridfile)

        # add open boundaries
        self.reservoir.set_boundary_volume(xz_minus=1e18, xz_plus=1e18, yz_minus=1e18, yz_plus=1e18)
        self.reservoir.apply_volume_depth()
        self.xwriter = None
        self.timer.node["initialization"].stop()

    def save_cubes(self, dir: str, fname: str, ti: int, arrays={}, arrays_full={}, write_grdecl=False, mode_ = 'w'):
        '''
        arrays - dictionary with numpy arrays to save, size = number of active cells
        '''
        #
        actnum = np.array(self.reservoir.actnum, copy=False)
        local_to_global = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        if write_grdecl:  # write cube to text grdecl
            fname_suf = os.path.join(dir, fname + '_step_' + str(ti) + '.grdecl')
            mode = mode_

            for name in arrays_full.keys():
                if arrays_full[name] is None:
                    continue
                save_array(arrays_full[name], fname_suf, name, local_to_global, global_to_local, mode=mode, make_full=False)
                mode = 'a'

            for name in arrays.keys():
                if arrays[name] is None:
                    continue
                try:
                    save_array(arrays[name], fname_suf, name, local_to_global, global_to_local, mode=mode)
                    mode = 'a'
                except:
                    pass
            save_array(actnum, fname_suf, 'ACTNUM', local_to_global, global_to_local, mode=mode)

    def save_xarray(self, dir_out, case_name, ti, ts, geomech_mode, arrays={}, write_xarray=False):
        if geomech_mode == 'none':
            return
        from tools import xarray_writer
        if self.xwriter is None:
            centers_cpp = self.reservoir.discr_mesh.get_centers()
            centers = np.array(centers_cpp, copy=False)  # YXZ order, n_cactive_cells
            self.xwriter = xarray_writer(verbose=False)

            if geomech_mode == 'surface':
                self.xwriter.init_dims_coords(self.reservoir.nx_struct_surface, self.reservoir.ny_struct_surface, 1, ts,
                                              self.reservoir.xarray_x, self.reservoir.xarray_y, self.reservoir.xarray_z)

        arrays_to_save = {}
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.physics.engine.X, copy=False)
        for v in range(nv):
            arrays_to_save[self.physics.vars[v]] = X[v:nb * nv:nv].astype(np.float32)

        # write porosity and permeability (for DA; "geomodels")
        arrays_to_save['Perm'] = self.reservoir.permx.astype(np.float32)
        arrays_to_save['Por'] = self.reservoir.poro.astype(np.float32)

        arrays_to_save_2d = {}
        if geomech_mode == 'fault':
            arrays_to_save['TEMPERATURE'] = arrays['TEMPERATURE']
            arrays_to_save_2d.update(arrays)
            for n in 'Por', 'Perm', 'perm', 'PRESSURE', 'TEMPERATURE', 'dp', 'dt':
                if n in arrays_to_save_2d.keys():
                    arrays_to_save_2d.pop(n)
        else:
            # merge dicts
            arrays_to_save.update(arrays)

        # reshape
        local_to_global = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        for name in arrays_to_save.keys():
            if arrays_to_save[name] is not None:
                #if arrays_to_save[name].size == 0:
                #    continue
                try:
                    arrays_to_save[name] = arrays_to_save[name].reshape(self.reservoir.nz, self.reservoir.ny, self.reservoir.nx)
                except:
                    arr_full = make_full_cube(arrays_to_save[name], local_to_global, global_to_local)
                    arrays_to_save[name] = arr_full.reshape(self.reservoir.nz, self.reservoir.ny,
                                                                        self.reservoir.nx)

        if geomech_mode=='fault':
            for name in arrays_to_save_2d.keys():
                if arrays_to_save_2d[name] is not None:
                    #print('2d', name,arrays_to_save_2d[name].shape)
                    # if eval_points are 1-layer for fault
                    arrays_to_save_2d[name] = arrays_to_save_2d[name].reshape(self.reservoir.nz, self.reservoir.ny)
                    # see get_eval_points

        fname_x = os.path.join(dir_out, case_name)

        # make 2d from 3d as we need to store only 2d at surface points
        for arr_name in arrays_to_save.keys():
            if arrays_to_save[arr_name] is not None and arrays_to_save[arr_name].shape[0] > 1:
                arrays_to_save[arr_name] = arrays_to_save[arr_name][0,:,:].reshape((1, arrays_to_save[arr_name].shape[1], arrays_to_save[arr_name].shape[2]))

        self.xwriter.write(fname_x, self.physics.engine.time_data_report, arrays=arrays_to_save,
                           arrays_2d=arrays_to_save_2d, write_x=write_xarray)

    def read_and_add_perforations(self, sch_fname, number_of_burden_layers=0, well_index=None, verbose=False):
        '''
        read COMPDAT from SCH file in Eclipse format, add wells and perforations
        note: uses only I,J,K1,K2 parameters from COMPDAT
        '''
        if sch_fname is None:
            return
        print('reading wells (COMPDAT) from', sch_fname)
        well_dia = 0.152
        well_rad = well_dia / 2

        keep_reading = True
        prev_well_name = ''
        with open(sch_fname) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if buff.startswith('--'):
                            continue
                        if len(buff) != 0:
                            CompDat = buff.split()
                            wname = CompDat[0].strip('"').strip("'") #remove quotas (" and ')
                            if len(CompDat) != 0 and '/' != wname:  # skip the empty line and '/' line
                                # define well
                                if wname == prev_well_name:
                                    pass
                                else:
                                    self.reservoir.add_well(wname)
                                    prev_well_name = wname
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3]) + number_of_burden_layers
                                k2 = int(CompDat[4]) + number_of_burden_layers
                                for k in range(k1, k2 + 1):
                                    self.reservoir.add_perforation(self.reservoir.wells[-1].name,
                                                                   cell_index=(i1, j1, k),
                                                                   well_radius=well_rad, well_index=well_index,
                                                                   well_indexD=0,
                                                                   multi_segment=False, verbose=verbose)

                                # store well XY (i1 j1 k1 should be an active cell)
                                res_block = self.reservoir.discr_mesh.get_global_index(i1-1, j1-1, k1-1)
                                local_block = self.reservoir.discr_mesh.global_to_local[res_block]
                                centers_cpp = self.reservoir.discr_mesh.get_centers()
                                centers_1d = np.array(centers_cpp, copy=True)
                                centers_3d = centers_1d.reshape((self.reservoir.discr_mesh.n_cells, 3))
                                if not hasattr(self.reservoir, 'well_coords'):
                                    self.reservoir.well_coords = []
                                self.reservoir.well_coords.append([i1-1, j1-1, k1-1])

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break

        print('WELLS read from SCH file:', len(self.reservoir.wells))



        return super().set_wells()

    def store_initial_p_t(self):
        self.reservoir.pressure_initial = self.get_pressure()
        self.reservoir.temperature_initial = self.get_temperature()

    def update_perm(self, perm):
        # make 1D, also convert the type to be able to convert to value_vector_discr
        # store to self to save in vtk
        assert self.reservoir.permx.size == perm.flatten().size, f'Grid and perm shapes are not consistent: {self.reservoir.permx.size}, {perm.size}'
        self.reservoir.permx = np.array(perm, dtype=np.float64).flatten()
        self.reservoir.permy = self.reservoir.permx
        self.reservoir.permz = self.reservoir.permx * 0.1
        permx = value_vector_discr(self.reservoir.permx)
        permy = value_vector_discr(self.reservoir.permy)
        permz = value_vector_discr(self.reservoir.permz)
        self.reservoir.discretizer.set_permeability(permx, permy, permz)
        # calculate transmissibilities
        displaced_tags = dict()
        displaced_tags[elem_loc.MATRIX] = set()
        displaced_tags[elem_loc.FRACTURE] = set()
        displaced_tags[elem_loc.BOUNDARY] = set()
        displaced_tags[elem_loc.FRACTURE_BOUNDARY] = set()
        self.reservoir.discretizer.calc_tpfa_transmissibilities(displaced_tags)


    def update_poro(self, poro):
        self.reservoir.poro = np.array(poro, dtype=np.float64).flatten()
        poro_cpp = value_vector_discr(self.reservoir.poro)
        self.reservoir.discretizer.set_porosity(poro_cpp)


        