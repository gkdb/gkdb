import sys
from collections import OrderedDict
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

from gkdb.core.ids_checks import check_ids_entry

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
    creation_date = DateTimeField(help_text='Date this entry was last modified')
    comment = TextField(help_text='Any comment describing this entry')

    def to_dict(self, include_gkdb_calculated=False):
        model_dict = {}
        model_dict['ids_properties'] = model_to_dict(self, exclude=[Ids_properties.id, Ids_properties.creation_date])
        model_dict['code'] = model_to_dict(self.code.get(), exclude=[Code.id, Code.ids_properties_id])
        model = self.model.get()
        non_linear_run = model.non_linear_run
        model_dict['model'] = model_to_dict(model, exclude=[Model.id, Model.ids_properties_id])
        model_dict['species_all'] = model_to_dict(self.species_all.get(), exclude=[Species_all.id, Species_all.ids_properties_id, Species_all.zeff])
        model_dict['species'] = []
        model_dict['total_fluxes_norm'] = []
        for species in self.species.order_by(Species.id):
            model_dict['species'].append(
                model_to_dict(species,
                              recurse=False,
                              exclude=[Species.id, Species.ids_properties_id]))
            sel = (Total_fluxes_norm.select()
                   .where(Total_fluxes_norm.ids_properties_id == self.id)
                   .where(Total_fluxes_norm.species_id == species.id)
                   )
            if sel.count() == 1:
                totflux = model_to_dict(sel.get(),
                              recurse=False,
                              exclude=[Total_fluxes_norm.ids_properties_id,
                                       Total_fluxes_norm.species_id])
                model_dict['total_fluxes_norm'].append(totflux)

        sel = (Ids_properties.select(Collisions)
               .join(Species)
               .join(Collisions, JOIN.RIGHT_OUTER, on=(Species.id == Collisions.species1_id) & (Species.id == Collisions.species2_id))
               .order_by(Collisions.species1_id, Collisions.species2_id)
               )
        n_spec = len(model_dict['species'])
        collisions = np.full([n_spec * n_spec, 3], np.NaN)
        for ii, (species1, species2, coll) in enumerate(sel.tuples()):
            collisions[ii, :] = (species1, species2, coll)
        collisions = collisions.reshape(n_spec, n_spec, 3)
        if np.any(np.isnan(collisions)):
            raise Exception('Could not read collisions table')
        coll_list = collisions[:,:,-1].tolist()
        model_dict['collisions'] = coll_list

        model_dict['flux_surface'] = model_to_dict(self.flux_surface.get(),
                                                   recurse=False,
                                                   exclude=[
                                                       Flux_surface.id,
                                                       Flux_surface.ids_properties_id,
                                                       Flux_surface.elongation,
                                                       Flux_surface.triangularity_upper,
                                                       Flux_surface.triangularity_lower])
        model_dict['wavevector'] = []
        for wavevector in self.wavevector.select().order_by(Wavevector.id):
            model_dict['wavevector'].append(
                model_to_dict(wavevector,
                              recurse=False,
                              exclude=[Wavevector.id, Wavevector.ids_properties_id]))
            eigenmode_list = model_dict['wavevector'][-1]['eigenmode'] = []
            for eigenmode in wavevector.eigenmode.select().order_by(Eigenmode.id):
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

            #sel = (self.select(Wavevector.id, *[fn.array_agg(getattr(Fluxes, col), coerce=False).alias(col) for col in Fluxes._meta.columns if col not in ['species_id', 'eigenmode_id']])
        for table in [Fluxes_norm, Moments_norm_rotating_frame]:
            sel = (self.select(Wavevector.id.alias('wavevector'), Species.id, table)
                       .where(Ids_properties.id == self.id)
                       .join(Wavevector)
                       .join(Eigenmode)
                       .switch(Ids_properties)
                       .join(Species)
                       .join(table, on=(table.species == Species.id) & (table.eigenmode == Eigenmode.id))
                       .order_by(Wavevector.id, Eigenmode.id, Species.id)
                       #.group_by(Species.id, Wavevector.id)
                       .dicts())
            wavevectors = OrderedDict()
            for fluxlike in sel:
                wv = fluxlike.pop('wavevector')
                eig = fluxlike.pop('eigenmode')
                sp = fluxlike.pop('species')
                if wv not in wavevectors:
                    wavevectors[wv] = OrderedDict()
                if eig not in wavevectors[wv]:
                    wavevectors[wv][eig] = OrderedDict()
                wavevectors[wv][eig][sp] = fluxlike
                del fluxlike['id']

            wavevectors_list = []
            for ii, (wv_id, wavevector) in enumerate(sorted(wavevectors.items(), key=lambda x: x[0])):
                eigenmodes = []
                for jj, (eig_id, eigenmode) in enumerate(sorted(wavevector.items(), key=lambda x: x[0])):
                    model_dict['wavevector'][ii]['eigenmode'][jj][table.__name__.lower()] = []
                    for sp_id, species in sorted(eigenmode.items(), key=lambda x: x[0]):
                        model_dict['wavevector'][ii]['eigenmode'][jj][table.__name__.lower()].append(species)
        return model_dict


    @classmethod
    @db.atomic()
    def from_dict(cls, model_dict):
        if not check_ids_entry(model_dict):
            raise Exception('Point rejected')
        ids_prop = model_dict.pop('ids_properties')
        ids_prop['date'] = datetime.datetime.now()
        ids_properties = dict_to_model(Ids_properties, ids_prop)
        ids_properties.save()

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
        return ids_properties

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
    collisions_momentum_conservation = BooleanField(help_text='True if the collision operator conserves momentum, false otherwise.')
    collisions_energy_conservation = BooleanField(help_text='True if the collision operator conserves energy, false otherwise.')
    collisions_finite_larmor_radius = BooleanField(help_text='True if the collision operator includes finite Larmor radius effects, false otherwise.')

    initial_value_run = BooleanField(help_text='True if the run was an initial value run. False if it was an eigenvalue run. Always 1 for non-linear run.')
    non_linear_run = BooleanField()

    # Optional
    time_interval_norm = ArrayField(FloatField, null=True)

class Flux_surface(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='flux_surface')
    r_minor_norm = FloatField(help_text='Minor radius of the flux surface of interest')
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
    # Derived from original shape
    elongation =    FloatField(null=True, help_text='Elongation of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    triangularity_upper = FloatField(null=True, help_text='Upper triangularity of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')
    triangularity_lower = FloatField(null=True, help_text='Lower triangularity of the flux surface of interest. Computed internally from the shape parameters (c_n,s_n)')

class Species_all(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='species_all')
    beta_reference = FloatField(help_text='Plasma beta')
    debye_length_reference = FloatField(help_text='Debye length')
    velocity_tor_norm = FloatField(help_text='Toroidal velocity (common to all species)')
    # Only for non-linear runs
    shearing_rate_norm = FloatField(null=True)
    # Derived from Species
    zeff = FloatField(null=True)

class Wavevector(BaseModel):
    ids_properties = ForeignKeyField(Ids_properties, related_name='wavevector')
    radial_component_norm = FloatField(help_text='Radial component of the wavevector')
    binormal_component_norm = FloatField(help_text='Binormal component of the wavevector')
    # Only for linear
    poloidal_turns = IntegerField(null=True, help_text='Number of poloidal turns covered by the flux-tube domain (i.e. number of coupled radial modes included in the simulation)')

class Eigenmode(BaseModel):
    # Eigenmode is optional for non-linear
    wavevector                   = ForeignKeyField(Wavevector, related_name='eigenmode')
    # Only defined for linear
    growth_rate_norm             = FloatField(null=True, help_text='Mode growth rate')
    frequency_norm               = FloatField(null=True, help_text='Mode frequency')
    growth_rate_tolerance        = FloatField(null=True, help_text='Tolerance used for to determine the mode growth rate convergence')

    phi_potential_perturbed_norm_real = ArrayField(FloatField, help_text='Parallel structure of the electrostatic potential perturbations (real part)', dimensions=1)
    phi_potential_perturbed_norm_imaginary = ArrayField(FloatField, help_text='Parallel structure of the electrostatic potential perturbations (imaginary part)')
    # Always optional
    a_field_parallel_perturbed_norm_real = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel vector potential perturbations (real part)')
    a_field_parallel_perturbed_norm_imaginary = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel vector potential perturbations (imaginary part)')
    b_field_parallel_perturbed_norm_real = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel magnetic field perturbations (real part)')
    b_field_parallel_perturbed_norm_imaginary = ArrayField(FloatField, null=True, help_text='Parallel structure of the parallel magnetic field perturbations (imaginary part)')
    poloidal_angle_grid = ArrayField(FloatField, help_text='Poloidal angle grid used to specify the parallel structure of the fields (eigenvectors)')

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
    velocity_toroidal_gradient_norm = FloatField(help_text='Species toroidal velocity gradient (with respect to r_minor)')


class Collisions(BaseModel):
    species1_id = ForeignKeyField(Species, related_name='collisions')
    species2_id = ForeignKeyField(Species, related_name='collisions')
    collisionality_norm = FloatField(help_text='Normalized collisionality')


class Fluxes_norm(BaseModel):
    # Fluxes optional for non-linear runs
    # *_a_parallel is Null if include_a_field_parallel is False
    # *_b_parallel is Null if include_b_field_parallel is False
    species = ForeignKeyField(Species, related_name='fluxes')
    eigenmode = ForeignKeyField(Eigenmode, related_name='fluxes')
    particle_flux_phi_potential = FloatField(help_text='Gyrocenter particle flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    particle_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    particle_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_flux_phi_potential = FloatField(help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    energy_flux_phi_potential = FloatField(help_text='Gyrocenter energy flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    energy_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    energy_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')


class Total_fluxes_norm(BaseModel):
    # Total fluxes not defined for linear runs
    # *_a_parallel is Null if include_a_field_parallel is False
    # *_b_parallel is Null if include_b_field_parallel is False
    species = ForeignKeyField(Species, related_name='total_fluxes')
    ids_properties = ForeignKeyField(Ids_properties, related_name='total_fluxes')
    particle_flux_phi_potential = FloatField(help_text='Gyrocenter particle flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    particle_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    particle_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_flux_phi_potential = FloatField(help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    energy_flux_phi_potential = FloatField(help_text='Gyrocenter energy flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    energy_flux_a_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    energy_flux_b_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    class Meta:
        primary_key = CompositeKey('species', 'ids_properties')


class Moments_norm_rotating_frame(BaseModel):
    # Always optional
    species = ForeignKeyField(Species, related_name='moments_rotating')
    eigenmode = ForeignKeyField(Eigenmode, related_name='moments_rotating')
    density_gyroaveraged_real = ArrayField(FloatField,help_text='Real part of the density moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    density_gyroaveraged_imaginary = ArrayField(FloatField,help_text='Imaginary part of the density moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    velocity_parallel_gyroaveraged_real = ArrayField(FloatField,help_text='Real part of the parallel velocity moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    velocity_parallel_gyroaveraged_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel velocity moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_parallel_gyroaveraged_real = ArrayField(FloatField,help_text='Real part of the parallel temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_parallel_gyroaveraged_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_perpendicular_gyroaveraged_real = ArrayField(FloatField,help_text='Real part of the perpendicular temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
    temperature_perpendicular_gyroaveraged_imaginary = ArrayField(FloatField,help_text='Imaginary part of the perpendicular temperature moment of the gyrocenter distribution function times the Bessel function J_0 in the rotating frame')
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
    db.create_tables([Tag, Ids_properties, Ids_properties_tag, Code, Model, Collisions, Flux_surface, Wavevector, Eigenmode, Species, Fluxes_norm, Total_fluxes_norm, Moments_norm_rotating_frame, Species_all])

if __name__ == '__main__':
    embed()
