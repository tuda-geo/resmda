import numpy as np
import pandas as pd
import xarray as xr
import re

class xarray_writer():
    def __init__(self, verbose=False):
        self.inited = False
        self.verbose = verbose

    def init_dims_coords(self,
                  nx: int, ny: int, nz: int,  # grid dimensions
                  ts: np.array,  # timesteps array, days
                  X: np.array, Y: np.array, Z: np.array  # cell center coords
                  ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = len(ts)  # number of time steps
        self.ts = ts/365.  # in years
        self.X = X
        self.Y = Y
        self.Z = Z

    def create_xarray(self,
                      arrays,  #: Dict[np.array],
                      arrays_2d
                      ):
        if self.inited:
            return
        self.inited = True

        # Initialize DataArray to store results
        data_vars = {}
        for name in arrays.keys():
            data_vars[name] = (['time', 'Z', 'Y', 'X'],
                               np.zeros((self.nt, self.nz, self.ny, self.nx)))

        attrs = {'title': 'Simulation Results'}
        self.data = xr.Dataset(data_vars=data_vars,
                          coords={'time': self.ts, 'X': self.X, 'Y': self.Y, 'Z': self.Z},
                          attrs=attrs)
        for name in arrays.keys():
            self.data[name] = xr.DataArray(np.zeros((self.nt, self.nz, self.ny, self.nx)),
                                      dims=['time', 'Z', 'Y', 'X'])

        # Initialize DataArray to store results
        data_vars = {}
        for name in arrays_2d.keys():
            data_vars[name] = (['time', 'Z', 'Y'],
                               np.zeros((self.nt, self.nz, self.ny)))

        attrs = {'title': 'Data on Fault'}
        self.data_2d = xr.Dataset(data_vars=data_vars,
                          coords={'time': self.ts, 'Z': self.Z, 'Y': self.Y},
                          attrs=attrs)
        for name in arrays_2d.keys():
            self.data_2d[name] = xr.DataArray(np.zeros((self.nt, self.nz, self.ny)),
                                      dims=['time', 'Z', 'Y'])

    def time_data_to_xarray(self, time_data):
        # parse header of time_data ("origin", "name", "unit")
        re_time_data = re.compile('(?P<origin>\w*?)[\s:]*(?P<name>[\w\s]+) \(?(?P<unit>[\w\/]+)\)?')

        if time_data == {}:
            time = np.array([0.])
        else:
            time = np.array(time_data['time'])/365.
        ds = xr.Dataset()
        for k, v in time_data.items():
            if re_time_data.match(k):
                origin, name, unit = re_time_data.match(k).groups()
                # substitute spaces with underscores in all names
                name = name.replace(' ', '_')
                origin = origin.replace(' ', '_')
                #TODO add comment
                ds = ds.merge({name: xr.DataArray(
                    data=np.array(time_data[k]).reshape(1, -1) if origin else np.array(time_data[k]),
                    coords={'origin': [origin], 'time': time} if origin else {'time': time},
                    dims=('origin', 'time') if origin else ('time'), attrs={'unit': unit})})
        return ds

    def append_xarray(self,
                      time_data,  # engine.time_data
                      arrays,  #: List[np.array],
                      arrays_2d,  #: List[np.array],
                      ):
        if 'time' not in time_data.keys():
            t = 0
        else:
            t = len(time_data['time']) - 1  # number of currently computed timesteps
        for name in arrays.keys():
            arr = arrays[name]
            if arr is None:
                arr = np.zeros((self.nz, self.ny, self.nx))
            if name not in self.data:
                self.data[name] = xr.DataArray(np.zeros((self.nt, self.nz, self.ny, self.nx)),
                                                   dims=['time', 'Z', 'Y', 'X'])
            self.data[name][t] = xr.DataArray(arr, dims=['Z', 'Y', 'X'],
                                              coords={'X': self.X, 'Y': self.Y, 'Z': self.Z})
            if self.verbose:
                print('array', name, 'time,years', t/365., 'range:', np.array(self.data[name][t]).min(), '-', np.array(self.data[name][t]).max())

        for name in arrays_2d.keys():
            arr = arrays_2d[name]
            if arr is None:
                arr = np.zeros((self.nz, self.ny))
            #print(name, arr.shape)
            if name not in self.data_2d:
                self.data_2d[name] = xr.DataArray(np.zeros((self.nt, self.nz, self.ny)),
                                          dims=['time', 'Z', 'Y'])
            self.data_2d[name][t] = xr.DataArray(arr, dims=['Z', 'Y'],
                                              coords={'Z': self.Z, 'Y': self.Y})
            if self.verbose:
                print('array_2d', name, 'time,years', int(t/365.), 'range:', np.array(self.data_2d[name][t]).min(), '-', np.array(self.data_2d[name][t]).max())

    def write(self, filename, time_data, arrays, arrays_2d, write_x=False):
        self.create_xarray(arrays, arrays_2d)

        if write_x:
            ds_time_data = self.time_data_to_xarray(time_data)
            self.data.update(ds_time_data)
            #self.data.update(arrays_2d)

            print('writing xarray data to ', filename + '.nc')
            self.data.to_netcdf(filename + '.nc', 'w')     # subsidence (surface data) or 3D data
            #self.data_2d.to_netcdf(filename + '_2d.nc', 'w')  # induced seismicity (fault data)

            #self.data_2d.to_netcdf(filename + '_fault.nc', 'w')
        else:
            self.append_xarray(time_data, arrays, arrays_2d)
        # debug
        #for array_name in arrays.keys():
        #    p = arrays[array_name]
        #    if p is not None:
        #        print('save x', array_name, p.min(), p.max())
        #print(r.keys())
        #print(r.dims())



def save_array(arr: np.array, fname: str, keyword: str, actnum: np.array, mode='w', verbose=False):
    '''
    writes numpy array of n_active_cell size to text file in GRDECL format with n_cells_total
    :param arr: numpy array to write
    :param fname: filename
    :param keyword: keyword for array
    :param actnum: actnum array
    :param mode: 'w' to rewrite the file or 'a' to append
    :return: None
    '''
    arr_full = make_full_cube(arr, actnum)
    with open(fname, mode) as f:
        f.write(keyword + '\n')
        s = ''
        for i in range(arr_full.size):
            s += str(arr_full[i]) + ' '
            if (i+1) % 6 == 0:  # write only 6 values per row
                f.write(s + '\n')
                s = ''
        f.write(s + '\n')
        f.write('/\n')
        if verbose:
            print('Array saved to file', fname, ' (keyword ' + keyword + ')')


def make_full_cube(cube: np.array, actnum: np.array, val=0):
    '''
    returns 1d-array of size nx*ny*nz, filled with zeros where actnum is zero
    :param cube: 1d-array of size n_active_cells
    :param actnum: 1d-array of size nx*ny*nz
    :return:
    '''
    if actnum.size == cube.size:
        return cube
    cube_full = np.zeros(actnum.size)+val
    #j = 0
    #for i in range(actnum.size):
    #    if actnum[i] > 0:
    #        cube_full[i] = cube[j]
    #        j += 1
    cube_full[actnum > 0] = cube
    return cube_full
    
   



def fmt(x):
    return '{:.3}'.format(x)

def fmt2(x):
    return '{:.2}'.format(x)

def print_range(model, time):
    P = model.get_pressure()
    T = model.get_temperature()

    print('Time', fmt(time/365), ' years; ',
          'P_range:', fmt(P.min()), '-', fmt(P.max()), 'bars; ',
          'T_range: ' + fmt(T.min()) + ' - ' + fmt(T.max()) + ' degrees' if T is not None else '')

def print_range_array(array, name, units=''):
    if not array[name] is None:
        print(name, ':', fmt(array[name].min()), '-', fmt(array[name].max()), units)
