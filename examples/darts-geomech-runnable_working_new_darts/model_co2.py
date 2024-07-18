import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.operators_base import PropertyOperators

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData# , EnthalpyIdeal
#from dartsflash.eos_properties import EoSDensity, EoSEnthalpy # - old
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from model_base import BaseModel
class Model(BaseModel):
    def __init__(self, arrays=None, geomech_mode='none', obl_cache=False, n_points=1000):
        '''
        :param discr_type:
        :param gridfile:
        :param propfile:
        :param sch_fname:
        :param n_points: number of OBL points for DARTS engine
        '''
        self.obl_cache = obl_cache
        # call base class constructor
        super().__init__(arrays=arrays, geomech_mode=geomech_mode)

        # set rock thermal properties: heat capacity and rock conduction
        self.hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.reservoir.mesh.rock_cond, copy=False)
        self.hcap.fill(2200)  # [kJ/m3/K]
        self.conduction.fill(181.44)  # [kJ/m/day/K]

        zero = 1e-10
        self.set_physics(zero, n_points=1001, temperature=None)
        self.temperature_initial_ = 273.15 + 76.85 #K
        self.initial_values = {"pressure": 100.,
                            "H2O": 0.99995,
                            "temperature": self.temperature_initial_
                            }
        self.inj_stream = [zero*100]
        self.inj_stream += [self.temperature_initial_] if self.physics.thermal else []

        self.set_sim_params(first_ts=1e-3, mult_ts=1.5, max_ts=5, tol_newton=1e-3, tol_linear=1e-5, it_newton=10,
                         it_linear=50)


    def set_physics(self,  zero, n_points, temperature=None, temp_inj=50.):
        """Physical properties"""
        # Fluid components, ions and solid
        components = ["H2O", "CO2"]

        phases = ["Aq", "V"]

        nc = len(components)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, min_z=zero/10)

        property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        property_container.density_ev = dict([('V', EoSDensity(pr, comp_data.Mw)),
                                              ('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                               ('Aq', PhaseRelPerm("oil"))])

        #h_ideal = EnthalpyIdeal(components) #- old
        #property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr, h_ideal)),
                                            #   ('Aq', EoSEnthalpy(aq, h_ideal))])
        
        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                            ('Aq', EoSEnthalpy(eos=aq)), ])

        
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('Aq', ConstFunc(180.)), ])

        property_container.output_props = {"satA": lambda: property_container.sat[0],
                                           "satV": lambda: property_container.sat[1],
                                           "xCO2": lambda: property_container.x[0, 1],
                                           "yH2O": lambda: property_container.x[1, 0]
                                           }

        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=1000, min_z=zero/10,
                                     max_z=1-zero/10, min_t=273.15, max_t=373.15, thermal=thermal, cache=self.obl_cache)
        self.physics.add_property_region(property_container)

        return

    def update_inj_rate(self):
        #inj_well = self.reservoir.wells[0]
        #import pandas as pd
        #time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        #col_name = time_data.filter(like=inj_well.name + ' :c 1 rate').columns.to_list()
        #inj_rate = np.array(time_data[col_name])[-1][0]  # pick the last timestep value
        #print('well', inj_well.name, 'inj_rate=', inj_rate)
        pass


    def set_well_controls(self):
        self.set_well_controls_custom(self, delta_p_inj=0, delta_t=0, rate_inj=0)

    def close_wells(self):
        for i, w in enumerate(self.reservoir.wells):
            w.control = self.physics.new_rate_prod(0, 1)

    def set_well_controls_custom(self, bhp_prod1=None, bhp_inj1=None, temp_inj=None, rate_prod=None, rate_inj=None,
                  delta_p_prod=None, delta_p_inj=None, delta_t=None):
        self.init_temperature = self.initial_values["temperature"] - 273.15
        self.init_pressure = self.initial_values["pressure"]

        if temp_inj is not None:
            temp_inj_K = temp_inj + 273.15  # degrees to K
        elif delta_t is not None:
            temp_inj_K = self.init_temperature - delta_t + 273.15  # degrees to K
        else:
            print('error in set wells')
            exit(1)
        self.inj_stream[1] = temp_inj_K

        if delta_p_inj is not None:
            bhp_inj = self.init_pressure + delta_p_inj
        else:
            print('error in set wells')
            exit(1)

        for i, w in enumerate(self.reservoir.wells):
            #w.control = self.physics.new_bhp_inj(bhp_inj, self.inj_stream)
            w.control = self.physics.new_rate_inj(rate_inj, self.inj_stream, 1)
            w.constraint = self.physics.new_bhp_inj(bhp_inj, self.inj_stream)
            print('model_ccus: well', w.name, 'control', w.control, 'bhp_inj', bhp_inj, 'temp_inj', temp_inj_K - 273.15 , 'rate_inj', rate_inj)

    def export_pro_vtk(self, file_name='co2'):
        nb = self.reservoir.mesh.n_res_blocks
        X = np.array(self.physics.engine.X)
        self.export_vtk(file_name=file_name)

    def save_cubes(self, dir: str, fname: str, ti: int, arrays = {}, arrays_full={}, write_grdecl=False, mode_='w'):
        # add temperature
        arrays_to_save = {}
        arrays_to_save['TEMPERATURE'] = self.get_temperature()
        arrays_to_save['PRESSURE'] = self.get_pressure()

        arrays.update(arrays_to_save)

        # call base_model.save_cubes()
        super().save_cubes(dir=dir, fname=fname, ti=ti, arrays=arrays, arrays_full=arrays_full, write_grdecl=write_grdecl, mode_=mode_)

    def get_pressure(self):
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        Xn = np.array(self.physics.engine.X, copy=True)
        P = Xn[:nb * nv:nv]
        return P

    def get_temperature(self):
        if self.physics.thermal is False:
            return None
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        Xn = np.array(self.physics.engine.X, copy=True)
        T = Xn[nv-1:nb * nv:nv]
        return T - 273.15

    def get_co2_saturation(self):
        return

    def set_initial_conditions(self):
        super().set_initial_conditions(initial_values=self.initial_values)
        pressure_grad = 100
        mesh = self.reservoir.mesh
        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth[:pressure.size] / 1000 * pressure_grad + 1
        return pressure