import sys
from IPython import embed
if sys.version_info < (3, 0):
    print('Python 2')
    input = raw_input
from peewee import *
from peewee import FloatField, FloatField, ProgrammingError
from peewee import JOIN
import peewee
import numpy as np
import inspect
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField
from playhouse.shortcuts import model_to_dict, dict_to_model
from IPython import embed
import scipy as sc
from scipy import io
import json
import datetime
import pandas as pd
from os import environ
from playhouse.db_url import connect

try:
    HOST = environ['PGHOST'] or 'gkdb.org'
except KeyError:
    HOST = 'gkdb.org'
try:
    DATABASE = environ['PGDATABASE']
except KeyError:
    DATABASE = 'gkdb'

db = PostgresqlExtDatabase(database=DATABASE, host=HOST)
try:
    db.connect()
except OperationalError:
    u = input("username? ")
    db = PostgresqlExtDatabase(database=DATABASE, host=HOST, user=u)
    try:
        db.connect()
    except OperationalError:
        import getpass
        p = getpass.getpass()
        db = PostgresqlExtDatabase(database=DATABASE, host=HOST, user=u, password=p)
    else:
        db.close()
else:
    db.close()

try:
    db.connect()
except OperationalError:
    raise Exception('Could not connect to database')
else:
    db.close()

class BaseModel(peewee.Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class Tag(BaseModel):
    name =    TextField(null=True)
    comment = TextField(null=True)

    @classmethod
    def from_pointlist(cls, pointlist, name=None, comment=None):
        pointset = set(pointlist)
        lst = []
        tag = Tag(name=name, comment=comment)
        tag.save()
        for pp in Ids_properties.select().where(Ids_properties.id << pointlist):
            pt = Ids_properties_Tag(point=pp.id, tag=tag.id)
            pt.save()
            pointset.remove(pp.id)
            print(pointset)

        if len(pointset) != 0:
            print('Failed to add points {!s}'.format(pointset))

class Ids_properties(BaseModel):
    provider = TextField(help_text='Name of the provider of this entry')
    creation_date = DateTimeField(help_text='Creation date of this entry')
    comment = TextField(help_text='Any comment describing this entry')

    def to_dict(self):
        model_dict = {}
        model_dict['point'] = model_to_dict(self, exclude=[Ids_properties.id, Ids_properties.creation_date])
        model_dict['code'] = model_to_dict(self.code.get(), exclude=[Code.id, Code.ids_properties_id])
        model_dict['model'] = model_to_dict(self.model.get(), exclude=[Model.id, Model.ids_properties_id])
        model_dict['species'] = []
        for species in self.species:
            model_dict['species'].append(
                model_to_dict(species,
                              recurse=False,
                              exclude=[Species.id, Species.ids_properties_id]))
        model_dict['wavevectors'] = []
        model_dict['flux_surface'] = model_to_dict(self.flux_surface.get(),
                                                   recurse=False,
                                                   exclude=[
                                                       Flux_surface.id,
                                                       Flux_surface.elongation,
                                                       Flux_surface.triangularity_upper,
                                                       Flux_surface.triangularity_lower])
        for wavevector in self.wavevector.select():
            model_dict['wavevectors'].append(
                model_to_dict(wavevector,
                              recurse=False,
                              exclude=[Wavevector.id, Wavevector.ids_properties_id]))
            eigenmode_list = model_dict['wavevectors'][-1]['eigenvalues'] = []
            for eigenmode in wavevector.eigenmode.select():
                eigenmode_list.append(
                    model_to_dict(eigenmode,
                                  recurse=False,
                                  exclude=[Eigenmode.id,
                                           Eigenmode.wavevector_id,
                                           Eigenmode.phi_potential_perturbed_weight,
                                           Eigenmode.phi_potential_perturbed_parity,
                                           Eigenmode.a_field_parallel_perturbed_weight,
                                           Eigenmode.a_field_parallel_perturbed_parity,
                                           Eigenmode.b_field_parallel_perturbed_weight,
                                           Eigenmode.b_field_parallel_perturbed_parity,
                                           ]))

        for flux_table in [Particles_rotating, Heat_fluxes_rotating, Heat_fluxes_laboratory,
                           Momentum_fluxes_rotating, Momentum_fluxes_laboratory]:
            name = flux_table.__name__.lower()
            sel = (self.select(Wavevector.id, flux_table)
                       .where(Ids_properties.id == self.id)
                       .join(Wavevector, JOIN.LEFT_OUTER)
                       .join(Eigenmode, JOIN.LEFT_OUTER)
                       .join(Species, JOIN.LEFT_OUTER, (Species.ids_properties_id == Ids_properties.id))
                       .join(flux_table).tuples())
            if sel.count() > 0:
                model_dict[name] = {}
                df = pd.DataFrame.from_records(list(sel),
                                               columns=['wavevector_id', 'species_id', 'eigenvalue_id',
                                                        'phi_potential', 'a_parallel', 'b_field_parallel'],
                                               index=['wavevector_id', 'species_id', 'eigenvalue_id'])
                xr = df.to_xarray()
                for k, v in xr.data_vars.items():
                    axes = [dim[:-3] for dim in v.dims]
                    model_dict[name]['axes'] = axes
                    model_dict[name][k] =  v.data.tolist()
            else:
                model_dict[name] = None
        return model_dict


    @classmethod
    @db.atomic()
    def from_dict(cls, model_dict):
        import xarray as xr
        dict_ = model_dict.pop('point')
        dict_['date'] = datetime.datetime.now()
        ids_properties = dict_to_model(Ids_properties, dict_)
        point.save()

        specieses = []
        for species_dict in model_dict.pop('species'):
            species = dict_to_model(Species, species_dict)
            species.ids_properties = point
            species.save()
            specieses.append(species)

        for simple in [Code, Species_Global, Flux_Surface]:
            name = simple.__name__.lower()
            entry = dict_to_model(simple, model_dict.pop(name))
            entry.ids_properties = point
            entry.save(force_insert=True)

        eigenvalues = []
        for ii, wavevector_dict in enumerate(model_dict.pop('wavevectors')):
            eigenvalues.append([])
            eigenvalues_dict = wavevector_dict.pop('eigenvalues')
            wavevector = dict_to_model(Wavevector, wavevector_dict)
            wavevector.ids_properties = point
            wavevector.save()
            for jj, eigenvalue_dict in enumerate(eigenvalues_dict):
                eigenvector = dict_to_model(Eigenvector, eigenvalue_dict.pop('eigenvector'))
                eigenvalue = dict_to_model(Eigenvalue, eigenvalue_dict)
                eigenvalue.wavevector = wavevector
                eigenvalue.save()
                eigenvector.eigenvalue = eigenvalue
                eigenvector.save(force_insert=True)

                eigenvalues[ii].append(eigenvalue)


        for flux_table in [Particle_Fluxes,
                           Heat_Fluxes_Lab, Heat_Fluxes_Rotating,
                           Momentum_Fluxes_Lab, Momentum_Fluxes_Rotating,
                           Moments_Rotating]:
            name = flux_table.__name__.lower()
            flux_dict = model_dict.pop(name)
            axes = flux_dict.pop('axes')
            ds = xr.Dataset()
            for varname, data in flux_dict.items():
                ds = ds.merge(xr.Dataset({varname: (axes, data)}))
            df = ds.to_dataframe()
            if "poloidal_angle" in axes:
                df = df.unstack('poloidal_angle')
            for index, row in df.iterrows():
                ind = dict(zip(df.index.names,index))
                if "poloidal_angle" in axes:
                    row = row.unstack()
                    entry = dict_to_model(flux_table, {name: val for name, val in zip(row.index, row.as_matrix().tolist())})
                else:
                    entry = dict_to_model(flux_table, row)
                entry.species = specieses[ind['species']]
                entry.eigenvalue = eigenvalues[ind['wavevector']][ind['eigenvalue']]
                entry.save(force_insert=True)
        return point

    def to_json(self, path):
        with open(path, 'w') as file_:
            json.dump(self.to_dict(), file_, indent=4, sort_keys=True)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as file_:
            dict_ = json.load(file_)
            ids_properties = Ids_properties.from_dict(dict_)
        return point

class Ids_properties_tag(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties)
    tag = ForeignKeyField(Tag)

class Code(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='code')
    name = TextField(help_text='Name of the code used for this entry')
    version = TextField(help_text='Version of the code used for this entry')
    parameters = BinaryJSONField(help_text='Key/value store containing the code dependent inputs')

class Model(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='model')
    include_centrifugal_effects = BooleanField(help_text='True if centrifugal effects were included, false otherwise')
    include_a_field_parallel = BooleanField(help_text='True if fluctuations of the parallel vector potential (magnetic flutter) were retained, false otherwise')
    include_b_field_parallel = BooleanField(help_text='True if fluctuations of the parallel magnetic field (magnetic compression) were retained, false otherwise')

    collisions_pitch_only = BooleanField(help_text='True if pitch-angle scattering only was retained in the collision operator, false otherwise')
    collisions_ei_only = BooleanField(help_text='True if electron to main ion collisions only were retained in the collision operator. False if all species collisions were retained')
    collisions_momentum_conservation = BooleanField(help_text='True if the collision operator conserves momentum, false otherwise.')
    collisions_energy_conservation = BooleanField(help_text='True if the collision operator conserves energy, false otherwise.')
    collisions_finite_larmor_radius = BooleanField(help_text='True if the collision operator includes finite Larmor radius effects, false otherwise.')

    initial_value_run = BooleanField(help_text='True if the run was an initial value run. False if it was an eigenvalue run.')
    collision_enhancement_factor = FloatField(help_text='Enhancement factor for the collisions of electrons on main ions (to mimic the impact of impurity ions not present in the run)')

class Flux_surface(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='flux_surface')
    r_minor_norm = FloatField(help_text='Minor radius of the flux surface of interest')
    # Derived from Shape
    elongation =    FloatField(null=True, help_text='Elongation of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    triangularity_upper = FloatField(null=True, help_text='Upper triangularity of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    triangularity_lower = FloatField(null=True, help_text='Lower triangularity of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    #squareness =    FloatField(null=True, help_text='Squareness of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    # Non-derived
    q = FloatField(help_text='Safety factor')
    magnetic_shear_r_minor = FloatField(help_text='Magnetic shear')
    pressure_gradient_norm = FloatField(help_text='Total pressure gradient (with respect to r_minor) used to characterise the local magnetic equilibrium')
    ip_sign = SmallIntegerField(help_text='Direction of the toroidal plasma current, positive when anticlockwise from above')
    b_field_tor_sign = SmallIntegerField(help_text='Direction of the toroidal magnetic field, positive when anticlockwise from above')
    # Original shape
    shape_coefficients_c = ArrayField(FloatField, help_text='Array containing the c_n coefficients parametrising the flux surface of interest. ')
    shape_coefficients_s = ArrayField(FloatField, help_text='Array containing the s_n coefficients parametrising the flux surface of interest. The first element is always zero.')
    dc_dr_minor_norm = ArrayField(FloatField, help_text='Radial derivative (with respect to r_minor) of the c_n coefficients')
    ds_dr_minor_norm = ArrayField(FloatField, help_text='Radial derivative (with respect to r_minor) of the s_n coefficients. The first element is always zero.')

class Species_all(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='species_all')
    beta_reference = FloatField(help_text='Plasma beta')
    collisionality_reference = FloatField(help_text='Plasma collision frequency')
    velocity_tor_norm = FloatField(help_text='Toroidal velocity (common to all species)')
    debye_length = FloatField(help_text='Debye length')
    # Derived from Species
    zeff = FloatField(null=True)

class Wavevector(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='wavevector')
    radial_wavevector = FloatField(help_text='Radial component of the wavevector')
    binormal_wavevector = FloatField(help_text='Binormal component of the wavevector')
    poloidal_turns = IntegerField(help_text='Number of poloidal turns covered by the flux-tube domain (i.e. number of coupled radial modes included in the simulation)')

class Eigenmode(BaseModel):
    wavevector                   = ForeignKeyField(Wavevector, related_name='eigenmode')
    growth_rate_norm             = FloatField(help_text='Mode growth rate')
    frequency_norm               = FloatField(help_text='Mode frequency')
    growth_rate_tolerance        = FloatField(help_text='Tolerance used for to determine the mode growth rate convergence')

    phi_potential_perturbed_norm_real = ArrayField(FloatField, help_text='Parallel structure of the electrostatic potential perturbations (real part)', dimensions=1)
    phi_potential_perturbed_norm_imaginary = ArrayField(FloatField, help_text='Parallel structure of the electrostatic potential perturbations (imaginary part)')
    a_field_parallel_perturbed_norm_real = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel vector potential perturbations (real part)')
    a_field_parallel_perturbed_norm_imaginary = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel vector potential perturbations (imaginary part)')
    b_field_parallel_perturbed_norm_real = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel magnetic field perturbations (real part)')
    b_field_parallel_perturbed_norm_imaginary = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel magnetic field perturbations (imaginary part)')
    poloidal_angle = ArrayField(FloatField, help_text='Poloidal angle grid used to specify the parallel structure of the fields (eigenvectors)')

    # Derived quantities
    phi_potential_perturbed_weight = FloatField(null=True, help_text='Relative amplitude of the electrostatic potential perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')
    phi_potential_perturbed_parity =    FloatField(null=True, help_text='Parity of the electrostatic potential perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')
    a_field_parallel_perturbed_weight =   FloatField(null=True, help_text='Relative amplitude of the parallel vector potential perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')
    a_field_parallel_perturbed_parity =      FloatField(null=True, help_text='Parity of the parallel vector potential perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')
    b_field_parallel_perturbed_weight =   FloatField(null=True, help_text='Relative amplitude of the parallel magnetic field perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')
    b_field_parallel_perturbed_parity =      FloatField(null=True, help_text='Parity of the parallel magnetic field perturbations. Computed internally from the parallel structure of the fields (eigenvectors)')


class Species(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='species')
    charge_norm = FloatField(help_text='Species charge')
    mass_norm = FloatField(help_text='Species mass')
    density_norm = FloatField(help_text='Species density')
    temperature_norm = FloatField(help_text='Species temperature')
    density_log_gradient_norm = FloatField(help_text='Species logarithmic density gradient (with respect to r_minor)')
    temperature_log_gradient_norm = FloatField(help_text='Species logarithmic temperature gradient (with respect to r_minor)')
    toroidal_velocity_gradient_norm = FloatField(help_text='Species toroidal velocity gradient (with respect to r_minor)')

class Particles_rotating(BaseModel):
    species = ForeignKeyField(Species, related_name='particles_rotating')
    eigenmode = ForeignKeyField(Eigenmode, related_name='particles_rotating')
    phi_potential = FloatField(help_text='Gyrocenter particle flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    a_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    b_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

class Heat_fluxes_laboratory(BaseModel):
    species = ForeignKeyField(Species, related_name='heat_fluxes_laboratory')
    eigenmode = ForeignKeyField(Eigenmode, related_name='heat_fluxes_laboratory')
    phi_potential = FloatField(help_text='Gyrocenter heat flux due to the electrostatic potential fluctuations in the Laboratory frame')
    a_parallel = FloatField(null=True, help_text='Gyrocenter heat flux due to the parallel vector potential fluctuations (magnetic flutter) in the Laboratory frame')
    b_field_parallel = FloatField(null=True, help_text='Gyrocenter heat flux due to the parallel magnetic field fluctuations (magnetic compression) in the Laboratory frame')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

class Momentum_fluxes_laboratory(BaseModel):
    species = ForeignKeyField(Species, related_name='momentum_fluxes_laboratory')
    eigenmode = ForeignKeyField(Eigenmode, related_name='momentum_fluxes_laboratory')
    phi_potential = FloatField(help_text='Gyrocenter momentum flux (toroidal projection of the parallel contribution only) due to the electrostatic potential fluctuations in the Laboratory frame')
    a_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux (toroidal projection of the parallel contribution only) due to the parallel vector potential fluctuations (magnetic flutter) in the Laboratory frame')
    b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux (toroidal projection of the parallel contribution only) due to the parallel magnetic field fluctuations (magnetic compression) in the Laboratory frame')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

class Heat_fluxes_rotating(BaseModel):
    species = ForeignKeyField(Species, related_name='heat_fluxes_rotating')
    eigenmode = ForeignKeyField(Eigenmode, related_name='heat_fluxes_rotating')
    phi_potential = FloatField(help_text='Gyrocenter heat flux due to the electrostatic potential fluctuations in the rotating frame')
    a_parallel = FloatField(null=True, help_text='Gyrocenter heat flux due to the parallel vector potential fluctuations (magnetic flutter) in the rotating frame')
    b_field_parallel = FloatField(null=True, help_text='Gyrocenter heat flux due to the parallel magnetic field fluctuations (magnetic compression) in the rotating frame')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

class Momentum_fluxes_rotating(BaseModel):
    species = ForeignKeyField(Species, related_name='momentum_fluxes_rotating')
    eigenmode = ForeignKeyField(Eigenmode, related_name='momentum_fluxes_rotating')
    phi_potential = FloatField(help_text='Gyrocenter momentum flux (toroidal projection of the parallel contribution only) due to the electrostatic potential fluctuations in the rotating frame')
    a_parallel = FloatField(null=True, help_text='Gyrocenter momentm flux (toroidal projection of the parallel contribution only) due to the parallel vector potential fluctuations (magnetic flutter) in the rotating frame')
    b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux (toroidal projection of the parallel contribution only) due to the parallel magnetic field fluctuations (magnetic compression) in the rotating frame')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

class Moments_rotating(BaseModel):
    species = ForeignKeyField(Species, related_name='moments_rotating')
    eigenmode = ForeignKeyField(Eigenmode, related_name='moments_rotating')
    density_real = ArrayField(FloatField,help_text='Real part of the density moment of the gyrocenter distribution function in the rotating frame')
    density_imaginary = ArrayField(FloatField,help_text='Imaginary part of the density moment of the gyrocenter distribution function in the rotating frame')
    parallel_velocity_real = ArrayField(FloatField,help_text='Real part of the parallel velocity  moment of the gyrocenter distribution function in the rotating frame')
    parallel_velocity_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel velocity  moment of the gyrocenter distribution function in the rotating frame')
    parallel_temperature_real = ArrayField(FloatField,help_text='Real part of the parallel temperature  moment of the gyrocenter distribution function in the rotating frame')
    parallel_temperature_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel temperature  moment of the gyrocenter distribution function in the rotating frame')
    perpendicular_temperature_real = ArrayField(FloatField,help_text='Real part of the perpendicular temperature moment of the gyrocenter distribution function in the rotating frame')
    perpendicular_temperature_imaginary = ArrayField(FloatField,help_text='Imaginary part of the perpendicular temperature moment of the gyrocenter distribution function in the rotating frame')
    density_gyroaveraged_norm_real = ArrayField(FloatField,help_text='Real part of the density moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    density_gyroaveraged_norm_imaginary = ArrayField(FloatField,help_text='Imaginary part of the density moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    velocity_parallel_gyroaveraged_norm_real = ArrayField(FloatField,help_text='Real part of the parallel velocity moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    velocity_parallel_gyroaveraged_norm_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel velocity moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_parallel_gyroaveraged_norm_real = ArrayField(FloatField,help_text='Real part of the parallel temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_parallel_gyroaveraged_norm_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_perpendicular_gyroaveraged_norm_real = ArrayField(FloatField,help_text='Real part of the perpendicular temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_perpendicular_gyroaveraged_norm_imaginary = ArrayField(FloatField,help_text='Imaginary part of the perpendicular temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_tables(cls, cascade=True)
            except ProgrammingError:
                db.rollback()
    db.execute_sql('SET ROLE developer')
    db.create_tables([Tag, Ids_properties, Code, Model, Flux_surface, Wavevector, Eigenmode, Species, Heat_fluxes_laboratory, Momentum_fluxes_laboratory, Heat_fluxes_rotating, Momentum_fluxes_rotating, Particles_rotating, Moments_rotating, Species_all])

if __name__ == '__main__':
    embed()
