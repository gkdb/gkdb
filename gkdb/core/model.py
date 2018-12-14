import sys
from collections import OrderedDict
from warnings import warn
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
        model_dict['fluxes_integrated_norm'] = []
        for species in self.species.order_by(Species.id):
            model_dict['species'].append(
                model_to_dict(species,
                              recurse=False,
                              exclude=[Species.id, Species.ids_properties_id]))
            sel = (Fluxes_integrated_norm.select()
                   .where(Fluxes_integrated_norm.ids_properties_id == self.id)
                   .where(Fluxes_integrated_norm.species_id == species.id)
                   )
            if sel.count() == 1:
                totflux = model_to_dict(sel.get(),
                              recurse=False,
                              exclude=[Fluxes_integrated_norm.ids_properties_id,
                                       Fluxes_integrated_norm.species_id])
                model_dict['fluxes_integrated_norm'].append(totflux)

        sel = (Ids_properties.select(Collisions)
               .join(Species)
               .join(Collisions, JOIN.RIGHT_OUTER, on=(Species.id == Collisions.species1_id) & (Species.id == Collisions.species2_id))
               .order_by(Collisions.species1_id, Collisions.species2_id)
               )
        n_spec = len(model_dict['species'])
        collisions_norm = np.full([n_spec * n_spec, 3], np.NaN)
        for ii, (species1, species2, coll) in enumerate(sel.tuples()):
            collisions_norm[ii, :] = (species1, species2, coll)
        collisions_norm = collisions_norm.reshape(n_spec, n_spec, 3)
        if np.any(np.isnan(collisions_norm)):
            raise Exception('Could not read collisions table')
        coll_list = collisions_norm[:,:,-1].tolist()
        model_dict['collisions'] = {}
        model_dict['collisions']['collisionality_norm'] = coll_list

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
    def from_dict(cls, model_dict, raise_on_disallowance=True):
        with cls._meta.database.atomic():
            if raise_on_disallowance:
                on_disallowance = 'raise_at_end'
            else:
                on_disallowance = 'print_at_end'
            allow_entry = check_ids_entry(model_dict, on_disallowance=on_disallowance)
            if not allow_entry:
                raise Exception('Point rejected')
            ids_prop = model_dict.pop('ids_properties')
            ids_prop['creation_date'] = datetime.datetime.now()
            ids_properties = dict_to_model(Ids_properties, ids_prop)
            ids_properties.save()

            for simple in [Code, Model, Species_all, Flux_surface]:
                name = simple.__name__.lower()
                entry = dict_to_model(simple, model_dict.pop(name))
                entry.ids_properties = ids_properties
                entry.save(force_insert=True)

            specieses = []
            for species_dict in model_dict.pop('species'):
                species = dict_to_model(Species, species_dict)
                species.ids_properties = ids_properties
                species.save()
                specieses.append(species)

            n_sp = len(specieses)
            collisions = model_dict.pop('collisions')
            for ii in range(n_sp):
                for jj in range(n_sp):
                    entry_dict = {}
                    for field, array in collisions.items():
                        entry_dict[field] = array[ii][jj]
                    Collisions.create(species1_id=specieses[ii],
                                      species2_id=specieses[jj],
                                      **entry_dict)

            for wv_idx, wavevector_dict in enumerate(model_dict.pop('wavevector')):
                eigenmodes = wavevector_dict.pop('eigenmode')
                wv = dict_to_model(Wavevector, wavevector_dict)
                wv.ids_properties = ids_properties
                wv.save()
                for eig_idx, eigenmode_dict in enumerate(eigenmodes):
                    fluxlike = {}
                    for table in [Fluxes_norm, Moments_norm_rotating_frame]:
                        name = table.__name__.lower()
                        if len(eigenmode_dict[name]) > 0:
                            fluxlike[table] = eigenmode_dict.pop(name)
                        else:
                            del eigenmode_dict[name]
                    eig = dict_to_model(Eigenmode, eigenmode_dict)
                    eig.wavevector = wv
                    eig.save()
                    for table, entry_dict in fluxlike.items():
                        for sp_idx, sp_entry in enumerate(entry_dict):
                            entry = dict_to_model(table, sp_entry)
                            entry.species = specieses[sp_idx]
                            entry.eigenmode = eig
                            entry.save(force_insert=True)
            if len(model_dict['fluxes_integrated_norm']) != 0:
                # Add check! Only if non-linear!
                for sp_idx, sp_entry in enumerate(model_dict.pop('fluxes_integrated_norm')):
                    # Check if all NaNs, don't create entry. Check if true for all species!
                    entry = dict_to_model(table, sp_entry)
                    entry.species = specieses[sp_idx]
                    entry.ids_properties = ids_properties
                    entry.save(force_insert=True)
            else:
                del model_dict['fluxes_integrated_norm']

        if len(model_dict) != 0:
            warn('Could not read full model_dict! Ignoring {!s}'.format(model_dict.keys()))

        return ids_properties

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

    # Must be False if include_b_field_parallel = False
    inconsistent_curvature_drift = BooleanField()

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
    velocity_tor_gradient_norm = FloatField(help_text='Species toroidal velocity gradient (with respect to r_minor)')


class Collisions(BaseModel):
    species1_id = ForeignKeyField(Species, related_name='collisions')
    species2_id = ForeignKeyField(Species, related_name='collisions')
    collisionality_norm = FloatField(help_text='Normalized collisionality')

    class Meta:
        primary_key = CompositeKey('species1_id', 'species2_id')


class Fluxes_norm(BaseModel):
    # Fluxes optional for non-linear runs
    # *_a_parallel is Null if include_a_field_parallel is False
    # *_b_parallel is Null if include_b_field_parallel is False
    species = ForeignKeyField(Species, related_name='fluxes')
    eigenmode = ForeignKeyField(Eigenmode, related_name='fluxes')
    particles_phi_potential = FloatField(help_text='Gyrocenter particle flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    particles_a_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    particles_b_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_phi_potential = FloatField(help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_phi_potential = FloatField(null=True, help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    energy_phi_potential = FloatField(help_text='Gyrocenter energy flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    energy_a_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    energy_b_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    class Meta:
        primary_key = CompositeKey('species', 'eigenmode')


class Fluxes_integrated_norm(BaseModel):
    # Total fluxes not defined for linear runs
    # *_a_parallel is Null if include_a_field_parallel is False
    # *_b_parallel is Null if include_b_field_parallel is False
    species = ForeignKeyField(Species, related_name='fluxes_integrated_norm')
    ids_properties = ForeignKeyField(Ids_properties, related_name='fluxes_integrated_norm')
    particles_phi_potential = FloatField(help_text='Gyrocenter particle flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    particles_a_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    particles_b_field_parallel = FloatField(null=True, help_text='Gyrocenter particle flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_phi_potential = FloatField(help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_tor_parallel_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_phi_potential = FloatField(help_text='Gyrocenter momentum flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_a_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    momentum_tor_perpendicular_b_field_parallel = FloatField(null=True, help_text='Gyrocenter momentum flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    energy_phi_potential = FloatField(help_text='Gyrocenter energy flux due to the electrostatic potential fluctuations. Identical in the Laboratory and rotating frames')
    energy_a_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel vector potential fluctuations (magnetic flutter). Identical in the Laboratory and rotating frames')
    energy_b_field_parallel = FloatField(null=True, help_text='Gyrocenter energy flux due to the parallel magnetic field fluctuations (magnetic compression). Identical in the Laboratory and rotating frames')
    class Meta:
        primary_key = CompositeKey('species', 'ids_properties')


class Moments_norm_rotating_frame(BaseModel):
    # Always optional
    species = ForeignKeyField(Species, related_name='moments_norm_rotating_frame')
    eigenmode = ForeignKeyField(Eigenmode, related_name='moments_norm_rotating_frame')
    density_real = ArrayField(FloatField,help_text='Real part of the density moment of the gyrocenter distribution function in the rotating frame')
    density_imaginary = ArrayField(FloatField,help_text='Imaginary part of the density moment of the gyrocenter distribution function in the rotating frame')
    velocity_parallel_real = ArrayField(FloatField,help_text='Real part of the parallel velocity  moment of the gyrocenter distribution function in the rotating frame')
    velocity_parallel_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel velocity  moment of the gyrocenter distribution function in the rotating frame')
    temperature_parallel_real = ArrayField(FloatField,help_text='Real part of the parallel temperature  moment of the gyrocenter distribution function in the rotating frame')
    temperature_parallel_imaginary = ArrayField(FloatField,help_text='Imaginary part of the parallel temperature  moment of the gyrocenter distribution function in the rotating frame')
    temperature_perpendicular_real = ArrayField(FloatField,help_text='Real part of the perpendicular temperature moment of the gyrocenter distribution function in the rotating frame')
    temperature_perpendicular_imaginary = ArrayField(FloatField,help_text='Imaginary part of the perpendicular temperature moment of the gyrocenter distribution function in the rotating frame')
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
    db.create_tables([Tag, Ids_properties, Ids_properties_tag, Code, Model, Collisions, Flux_surface, Wavevector, Eigenmode, Species, Fluxes_norm, Fluxes_integrated_norm, Moments_norm_rotating_frame, Species_all])

if __name__ == '__main__':
    embed()
