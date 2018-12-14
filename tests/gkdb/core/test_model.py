import os
import copy

from unittest import TestCase, skip
from IPython import embed

from gkdb.core.model import *
from base import DatabaseTestCase, ModelDatabaseTestCase, ModelTestCase, requires_models
from base import db

db.execute_sql('SET ROLE testuser')

def gen_default_dicts():
    default_dicts = {
        'ids_properties': dict(
            provider='testuser',
            creation_date='2018-09-25 10:05:20',
            comment='This is a test',
        ),
        'code': dict(
            name='test',
            version='newish',
            parameters={'some': 'test',
                        'values': 'in',
                        'a': 'json'},
        ),
        'model': dict(
            include_centrifugal_effects=False,
            include_a_field_parallel=True,
            include_b_field_parallel=True,
            collisions_pitch_only=False,
            collisions_momentum_conservation=True,
            collisions_energy_conservation=True,
            collisions_finite_larmor_radius=False,
            initial_value_run=False,
            non_linear_run=False,
            inconsistent_curvature_drift=False,
        ),
        'flux_surface': dict(
            r_minor_norm=0.16,
            q=1.5,
            magnetic_shear_r_minor=0.7,
            pressure_gradient_norm=0,
            ip_sign=-1,
            b_field_tor_sign=-1,
            shape_coefficients_c=[0.16],
            shape_coefficients_s=[0],
            dc_dr_minor_norm=[1],
            ds_dr_minor_norm=[0],
        ),
        'species_all': dict(
            beta_reference=0,
            velocity_tor_norm=0,
            debye_length_reference=0,
        ),
        'wavevector': dict(
            radial_component_norm=0,
            binormal_component_norm=0.36878445,
        ),
        'eigenmode': dict(
            growth_rate_norm=1,
            frequency_norm=2,
            growth_rate_tolerance=1e-2,
            phi_potential_perturbed_norm_real=[2,3,4,5],
            phi_potential_perturbed_norm_imaginary=[-8.2898922, -0.0012916868, -0.011514476, -8.0035435],
            a_field_parallel_perturbed_norm_real=[4,5,6,7],
            a_field_parallel_perturbed_norm_imaginary=[5,6,7,8],
            b_field_parallel_perturbed_norm_real=[6,7,8,9],
            b_field_parallel_perturbed_norm_imaginary=[7,8,9,10],
            poloidal_angle=[-4.1666193, -0.12558764, 0.12579195, 4.196568],
        ),
        'species_0': dict(
            charge_norm=1,
            mass_norm=1,
            density_norm=1,
            temperature_norm=1.1764706,
            density_log_gradient_norm=4,
            temperature_log_gradient_norm=8,
            velocity_tor_gradient_norm=0,
        ),
        'species_1': dict(
            charge_norm=-1,
            mass_norm=0.00027237,
            density_norm=1,
            temperature_norm=1,
            density_log_gradient_norm=4,
            temperature_log_gradient_norm=6,
            velocity_tor_gradient_norm=0,
        ),
        'fluxes_norm': dict(
            energy_phi_potential=1,
            energy_a_field_parallel=2,
            energy_b_field_parallel=3,
            particles_phi_potential=1,
            particles_a_field_parallel=2,
            particles_b_field_parallel=3,
            momentum_tor_parallel_phi_potential=1,
            momentum_tor_parallel_a_field_parallel=2,
            momentum_tor_parallel_b_field_parallel=3,
            momentum_tor_perpendicular_phi_potential=1,
            momentum_tor_perpendicular_a_field_parallel=2,
            momentum_tor_perpendicular_b_field_parallel=3,
        ),
        'fluxes_integrated_norm': dict(
            energy_phi_potential=1,
            energy_a_field_parallel=2,
            energy_b_field_parallel=3,
            particles_phi_potential=1,
            particles_a_field_parallel=2,
            particles_b_field_parallel=3,
            momentum_tor_parallel_phi_potential=1,
            momentum_tor_parallel_a_field_parallel=2,
            momentum_tor_parallel_b_field_parallel=3,
            momentum_tor_perpendicular_phi_potential=1,
            momentum_tor_perpendicular_a_field_parallel=2,
            momentum_tor_perpendicular_b_field_parallel=3,
        ),
        'moments_norm_rotating_frame': dict(
            density_real=[1,2,3,4],
            density_imaginary=[2,3,4,5],
            velocity_parallel_real=[3,4,5,6],
            velocity_parallel_imaginary=[4,5,6,7],
            temperature_parallel_real=[5,6,7,8],
            temperature_parallel_imaginary=[6,7,8,9],
            temperature_perpendicular_real=[7,8,9,10],
            temperature_perpendicular_imaginary=[8,9,10,11],
            density_gyroaveraged_real=[9,10,11,12],
            density_gyroaveraged_imaginary=[10,11,12,13],
            velocity_parallel_gyroaveraged_real=[11,12,13,14],
            velocity_parallel_gyroaveraged_imaginary=[12,13,14,15],
            temperature_parallel_gyroaveraged_real=[13,14,15,16],
            temperature_parallel_gyroaveraged_imaginary=[14,15,16,17],
            temperature_perpendicular_gyroaveraged_real=[15,16,17,18],
            temperature_perpendicular_gyroaveraged_imaginary=[16,17,18,19],
        ),
        'collisions': [[1, 2], [3, 4]]
    }
    return default_dicts

class TestIds_properties(ModelTestCase):
    requires = [Ids_properties]

    def test_creation(self):
        default_dicts = gen_default_dicts()
        Ids_properties.create(**default_dicts['ids_properties'])

class TestCode(ModelTestCase):
    requires = [Ids_properties, Code]

    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Code.create(ids_properties=ids,
                    **default_dicts['code']
        )

class TestModel(ModelTestCase):
    requires = [Ids_properties, Model]

    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Model.create(
            ids_properties=ids,
            **default_dicts['model']
        )

class TestFlux_surface(ModelTestCase):
    requires = [Ids_properties, Flux_surface]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Flux_surface.create(
            ids_properties=ids,
            **default_dicts['flux_surface']
        )

class TestSpecies_all(ModelTestCase):
    requires = [Ids_properties, Species_all]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Species_all.create(
            ids_properties=ids,
            **default_dicts['species_all']
        )

class TestWavevector(ModelTestCase):
    requires = [Ids_properties, Wavevector]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Wavevector.create(
            ids_properties=ids,
            **default_dicts['wavevector']
        )

class TestEigenmode(ModelTestCase):
    requires = [Ids_properties, Wavevector, Eigenmode]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        wv = Wavevector.create(
            ids_properties=ids,
            **default_dicts['wavevector']
        )
        Eigenmode.create(
            wavevector=wv,
            **default_dicts['eigenmode']
        )

class TestSpecies(ModelTestCase):
    requires = [Ids_properties, Species]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        Species.create(
            ids_properties=ids,
            **default_dicts['species_0']
        )


class TestMoments_norm_rotating_frame(ModelTestCase):
    requires = [Ids_properties, Species, Wavevector, Eigenmode, Moments_norm_rotating_frame]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        wv = Wavevector.create(
            ids_properties=ids,
            **default_dicts['wavevector']
        )
        em = Eigenmode.create(
            wavevector=wv,
            **default_dicts['eigenmode']
        )
        sp = Species.create(
            ids_properties=ids,
            **default_dicts['species_0']
        )
        Moments_norm_rotating_frame.create(
            species=sp,
            eigenmode=em,
            **default_dicts['moments_norm_rotating_frame']
        )

class TestCollisions(ModelTestCase):
    requires = [Ids_properties, Species, Collisions]
    def test_creation(self):
        default_dicts = gen_default_dicts()
        ids = Ids_properties.create(**default_dicts['ids_properties'])
        sp0 = Species.create(
            ids_properties=ids,
            **default_dicts['species_0']
        )
        sp1 = Species.create(
            ids_properties=ids,
            **default_dicts['species_1']
        )
        for ii, sp_first in enumerate([sp0, sp1]):
            for jj, sp_second in enumerate([sp0, sp1]):
                Collisions.create(species1_id=sp_first,
                                  species2_id=sp_second,
                                  collisionality_norm=default_dicts['collisions'][ii][jj])

def gen_wavevectors(ids, default_dicts):
    wv_dict = default_dicts['wavevector']
    wv0 = Wavevector.create(
        ids_properties=ids,
        **wv_dict
    )
    wv_dict['binormal_component_norm'] *= 2
    wv1 = Wavevector.create(
        ids_properties=ids,
        **wv_dict
    )
    wv_dict['binormal_component_norm'] *= 1.5
    wv2 = Wavevector.create(
        ids_properties=ids,
        **wv_dict
    )
    return [wv0, wv1, wv2]

def ids_properties_linear(default_dicts):
    ids = Ids_properties.create(**default_dicts['ids_properties'])
    Flux_surface.create(
        ids_properties=ids,
        **default_dicts['flux_surface']
    )
    Code.create(
        ids_properties=ids,
        **default_dicts['code']
    )
    Model.create(
        ids_properties=ids,
        **default_dicts['model']
    )
    Species_all.create(
        ids_properties=ids,
        **default_dicts['species_all']
    )
    default_dicts['wavevector']['poloidal_turns'] = 11
    wvs = gen_wavevectors(ids, default_dicts)
    sp0 = Species.create(
        ids_properties=ids,
        **default_dicts['species_0']
    )
    sp1 = Species.create(
        ids_properties=ids,
        **default_dicts['species_1']
    )
    for ii, sp_first in enumerate([sp0, sp1]):
        for jj, sp_second in enumerate([sp0, sp1]):
            Collisions.create(species1_id=sp_first,
                              species2_id=sp_second,
                              collisionality_norm=default_dicts['collisions'][ii][jj])
    gen_ems = 1
    for ii, wv in enumerate(wvs):
        if ii == len(wvs) - 1: # Add an extra eigenmode for the last wavevector
            gen_ems = 2
        ems = []
        for ii in range(gen_ems):
            em = Eigenmode.create(
                wavevector=wv,
                **default_dicts['eigenmode']
            )
            ems.append(em)
        for sp in [sp0, sp1]:
            for em in ems:
                for cls in [Fluxes_norm, Moments_norm_rotating_frame]:
                    cls.create(
                        species=sp,
                        eigenmode=em,
                        **default_dicts[cls.__name__.lower()]
                    )
    return ids

def ids_properties_nonlinear(default_dicts, eigenmodes=True):
    default_dicts['species_all']['shearing_rate_norm'] = 1
    default_dicts['eigenmode']['growth_rate_norm'] = None
    default_dicts['eigenmode']['frequency_norm'] = None
    default_dicts['eigenmode']['growth_rate_tolerance'] = None
    ids = Ids_properties.create(**default_dicts['ids_properties'])
    Flux_surface.create(
        ids_properties=ids,
        **default_dicts['flux_surface']
    )
    Code.create(
        ids_properties=ids,
        **default_dicts['code']
    )
    Model.create(
        ids_properties=ids,
        **default_dicts['model']
    )
    Species_all.create(
        ids_properties=ids,
        **default_dicts['species_all']
    )
    wvs = gen_wavevectors(ids, default_dicts)
    sp0 = Species.create(
        ids_properties=ids,
        **default_dicts['species_0']
    )
    sp1 = Species.create(
        ids_properties=ids,
        **default_dicts['species_1']
    )
    for ii, sp_first in enumerate([sp0, sp1]):
        for jj, sp_second in enumerate([sp0, sp1]):
            Collisions.create(species1_id=sp_first,
                              species2_id=sp_second,
                              collisionality_norm=default_dicts['collisions'][ii][jj])
    for sp in [sp0, sp1]:
        Fluxes_integrated_norm.create(
            species=sp,
            ids_properties=ids,
            **default_dicts['fluxes_integrated_norm']
        )
        if eigenmodes:
            for wv in wvs:
                em = Eigenmode.create(
                    wavevector=wv,
                    **default_dicts['eigenmode']
                )
                for cls in [Fluxes_norm, Moments_norm_rotating_frame]:
                    cls.create(
                        species=sp,
                        eigenmode=em,
                        **default_dicts[cls.__name__.lower()]
                    )
    return ids

def compare_nested_dict(dict1, dict2):
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        in1_not2 = set(dict1.keys()) - set(dict2.keys())
        in2_not1 = set(dict2.keys()) - set(dict1.keys())
        equality_dict = {}
        non_equality_dict = {}
        for key in set(dict2.keys()).intersection(set(dict1.keys())):
            if isinstance(dict1, dict) and isinstance(dict2, dict):
                res = compare_nested_dict(dict1[key], dict2[key])
                if not (res == True or res == (set(), set(), {})):
                    equality_dict[key] = res
            else:
                if dict1[key] == dict2[key]:
                    equality_dict[key] = True
                else:
                    equality_dict[key] = False
        return in1_not2, in2_not1, equality_dict
    else:
        return True


class TestFullIds_properties(ModelTestCase):
    #requires = [Ids_properties, Species, Wavevector, Eigenmode, Flux_surface, Code, Model, Moments_rotating, Particle_fluxes_rotating, Energy_fluxes_laboratory, Momentum_fluxes_laboratory, Energy_fluxes_rotating, Momentum_fluxes_rotating, Moments_rotating]
    requires = [Ids_properties, Species, Collisions, Wavevector, Eigenmode, Flux_surface, Code, Model, Species_all, Moments_norm_rotating_frame, Fluxes_norm, Fluxes_integrated_norm]
    @skip
    def test_from_json(self):
        Ids_properties.from_json('../../../json_files/tiny_test.json')

    def test_to_dict(self):
        default_dicts = gen_default_dicts()
        ids = ids_properties_linear(default_dicts)
        ids_dict = ids.to_dict()

    def test_to_json(self):
        default_dicts = gen_default_dicts()
        ids = ids_properties_linear(default_dicts)
        ids.to_json('test_linear.json')

    #def test_from_json(self):
    @skip
    def test_from_to_dict_roundtrip(self):
        with open('../../../json_files/p1real_kth_1.json') as file:
        #with open('./test.json') as file:
            model_dict = json.load(file)
        orig_model_dict = copy.deepcopy(model_dict)
        ids = Ids_properties.from_dict(model_dict)
        read_model_dict = ids.to_dict()
        res = compare_nested_dict(orig_model_dict, read_model_dict)

    def test_to_dict_nonlinear(self):
        default_dicts = gen_default_dicts()
        ids = ids_properties_nonlinear(default_dicts)
        ids_dict = ids.to_dict()

    def test_to_json_nonlinear(self):
        default_dicts = gen_default_dicts()
        ids = ids_properties_nonlinear(default_dicts)
        ids.to_json('test_nonlinear.json')
